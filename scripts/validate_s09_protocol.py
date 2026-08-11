#!/usr/bin/env python3
"""Validate the committed S09-A protocol without loading a model or running S09-B."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "s09_baseline_eval.json"
EXPECTED_MODE_IDS = (
    "full_precision_bf16_teacher",
    "static_packed_4bit",
    "static_packed_8bit",
    "hard_routed_resident_packed",
    "hard_routed_synchronous_on_demand_packed",
)
EXPECTED_MODEL_REPOSITORY = "Qwen/Qwen3-4B"
EXPECTED_MODEL_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
EXPECTED_ANY_PRECISION_REVISION = "a3257d02740cc5757c78673da534b0630ff3a4ea"
EXPECTED_DATASET = {
    "dataset": "Salesforce/wikitext",
    "config": "wikitext-2-raw-v1",
    "revision": "b08601e04326c79dfdd32d625aee71d232d685c3",
    "split": "test",
}
EXPECTED_PACKED_SHA256 = "29d9bc526b3da0bd39daf2f82afd141f82d005ca1232cabc75cfe9d9ecc1cfee"
EXPECTED_ROUTER_SHA256 = "08bf646f19759c0d7949e159bdbe4f96bbea737204b96f8760d205c8d6fd1949"
EXPECTED_SAMPLE_COUNT = 32
EXPECTED_EVALUATED_TOKENS = 4096
EXPECTED_PROMPT_IDS = (
    "s03-quality-0",
    "s03-quality-1",
    "s03-quality-2",
    "s03-quality-3",
    "s03-quality-4",
    "validation-3",
    "validation-1000",
)
EXPECTED_DATASET_SELECTION = (
    "concatenate non-empty rows in source order; take the first fixed valid windows; "
    "no random sampling"
)
EXPECTED_LABELS = "window[1:] aligned with logits from window[:-1]"
EXPECTED_LOSS = (
    "float32 token-weighted summed causal cross-entropy divided by exact "
    "evaluated target-token count"
)
EXPECTED_ROUTED_RECORDING = {
    "layers": 36,
    "unit_types": ["attention", "ffn"],
    "units_per_request": 72,
    "fraction_fields": ["4_bit", "8_bit", "overall"],
    "route_map_digest": "canonical sorted-key JSON SHA-256",
    "prompt_to_prompt_diversity": (
        "record unique maps, changed units, changed fraction, and pairwise route "
        "distance; observational only"
    ),
    "adaptivity_limitation": "retain S07 classification OTHER; no new diversity threshold",
}
EXPECTED_TRANSFER_MODE = "hard-routed_synchronous_on_demand_only"
EXPECTED_TRANSFER_RULE = (
    "D029 physical packed buffer rule: selected qweight planes plus the selected "
    "precision LUT, with actual destination numel*element_size bytes"
)
EXPECTED_TRANSFER_INPUTS = [
    "actual hard route map",
    "actual S08 packed buffer layout",
    "D029 transfer rule",
]
EXPECTED_HARDWARE_POLICY = "one fixed physical CUDA device for every mode; PAUSE rather than mix GPU models"
EXPECTED_GPU_SUBSTITUTION = "allowed only with pre-recorded identity and comparability"
EXPECTED_HARDWARE_RECORD_VERSIONS = [
    "device_index",
    "gpu_model",
    "driver",
    "cuda_runtime",
    "pytorch",
    "transformers",
    "python",
]
EXPECTED_DEFERRED_MECHANISMS = [
    "asynchronous loading",
    "prefetching",
    "transfer prediction",
    "bit-width cost penalties",
    "cross-request caching",
    "multi-query batching",
    "schedulers",
    "post-baseline optimization",
    "soft-routing final mode",
    "alternate router or checkpoint",
]


class ProtocolValidationError(ValueError):
    """Raised when a frozen protocol is incomplete, ambiguous, or inconsistent."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolValidationError(f"cannot parse JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ProtocolValidationError(f"JSON root must be an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _yaml_scalar(path: Path, key: str) -> str | None:
    pattern = re.compile(rf"^{re.escape(key)}:\s*([^#\s]+)", re.MULTILINE)
    match = pattern.search(path.read_text())
    return None if match is None else match.group(1).strip("\"'")


def _git_revision(path: Path) -> str | None:
    """Return the checked-out revision for a submodule path, if available."""

    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    revision = result.stdout.strip()
    return revision or None


def _get(mapping: Any, *keys: str) -> Any:
    value = mapping
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _add(errors: list[str], condition: bool, message: str) -> None:
    if not condition:
        errors.append(message)


def _check_equal(errors: list[str], actual: Any, expected: Any, name: str) -> None:
    _add(errors, actual == expected, f"{name}: expected {expected!r}, got {actual!r}")


def _fixed_input_digest(ids: list[int]) -> str:
    return hashlib.sha256(struct.pack("<" + "q" * len(ids), *ids)).hexdigest()


def quality_gate(static_8: float, static_4: float, margin: float = 1.10) -> bool:
    """Return the frozen arithmetic quality gate used by S09 results."""

    return static_8 <= margin * static_4


def validate_latency_records(records: list[dict[str, Any]], repeats: int = 5) -> bool:
    """Validate raw synchronized latency observations without filtering runs."""

    if len(records) != repeats:
        return False
    required = ("prefill_seconds", "decode_seconds", "end_to_end_seconds")
    for record in records:
        for name in required:
            value = record.get(name)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                return False
    return True


def _validate_modes(config: dict[str, Any], errors: list[str]) -> None:
    modes = config.get("modes")
    _add(errors, isinstance(modes, list), "modes must be a list")
    if not isinstance(modes, list):
        return
    ids = [mode.get("id") if isinstance(mode, dict) else None for mode in modes]
    _check_equal(errors, len(modes), 5, "mode count")
    _check_equal(errors, len(set(ids)), len(ids), "mode IDs must be unique")
    _check_equal(errors, tuple(ids), EXPECTED_MODE_IDS, "mode IDs and order")
    forbidden = ("soft", "alternate", "async", "prefetch", "batching", "scheduler", "cost")
    for mode in modes:
        if not isinstance(mode, dict):
            errors.append("each mode must be an object")
            continue
        mode_text = json.dumps(mode, sort_keys=True).lower()
        for term in forbidden:
            _add(errors, term not in mode_text, f"forbidden mode mechanism {term!r}")
    expected_fields = {
        EXPECTED_MODE_IDS[0]: {
            "model_kind": "full_precision",
            "routing": "none",
            "loader": "resident",
        },
        EXPECTED_MODE_IDS[1]: {
            "model_kind": "packed_static",
            "precision": 4,
            "routing": "static",
            "loader": "resident",
        },
        EXPECTED_MODE_IDS[2]: {
            "model_kind": "packed_static",
            "precision": 8,
            "routing": "static",
            "loader": "resident",
        },
        EXPECTED_MODE_IDS[3]: {
            "model_kind": "packed_routed",
            "routing": "hard_query_level",
            "loader": "resident",
        },
        EXPECTED_MODE_IDS[4]: {
            "model_kind": "packed_routed",
            "routing": "hard_query_level",
            "loader": "synchronous_on_demand",
        },
    }
    for mode in modes:
        if not isinstance(mode, dict) or mode.get("id") not in expected_fields:
            continue
        for field, expected in expected_fields[mode["id"]].items():
            _check_equal(errors, mode.get(field), expected, f"mode {mode['id']} field {field}")
    _check_equal(
        errors,
        config.get("comparison_contract", {}).get("soft_routing_final_mode"),
        False,
        "soft final mode",
    )
    _check_equal(
        errors,
        config.get("comparison_contract", {}).get("alternate_router"),
        False,
        "alternate router",
    )
    _check_equal(
        errors,
        config.get("comparison_contract", {}).get("alternate_checkpoint"),
        False,
        "alternate checkpoint",
    )
    _add(
        errors,
        "bitwise equal"
        in str(config.get("comparison_contract", {}).get("execution_equivalence_criterion", "")),
        "resident/on-demand execution-equivalence criterion is incomplete",
    )


def _validate_identities(
    config: dict[str, Any],
    root: Path,
    manifest: dict[str, Any],
    errors: list[str],
) -> None:
    identities = config.get("identities", {})
    model = identities.get("model", {})
    _check_equal(errors, model.get("repository"), EXPECTED_MODEL_REPOSITORY, "model repository")
    _check_equal(errors, model.get("revision"), EXPECTED_MODEL_REVISION, "model revision")
    _check_equal(
        errors, model.get("tokenizer_repository"), EXPECTED_MODEL_REPOSITORY, "tokenizer repository"
    )
    _check_equal(
        errors, model.get("tokenizer_revision"), EXPECTED_MODEL_REVISION, "tokenizer revision"
    )

    manifest_model = manifest.get("source_model", {})
    for key, expected in (
        ("repository", EXPECTED_MODEL_REPOSITORY),
        ("revision", EXPECTED_MODEL_REVISION),
        ("tokenizer_repository", EXPECTED_MODEL_REPOSITORY),
        ("tokenizer_revision", EXPECTED_MODEL_REVISION),
    ):
        _check_equal(errors, manifest_model.get(key), expected, f"manifest source_model.{key}")
    model_yaml = root / "configs" / "model.yaml"
    if model_yaml.is_file():
        _check_equal(
            errors,
            _yaml_scalar(model_yaml, "repository"),
            EXPECTED_MODEL_REPOSITORY,
            "model.yaml repository",
        )
        _check_equal(
            errors,
            _yaml_scalar(model_yaml, "revision"),
            EXPECTED_MODEL_REVISION,
            "model.yaml revision",
        )

    any_precision = identities.get("any_precision", {})
    manifest_backend = manifest.get("any_precision", {})
    _check_equal(
        errors,
        any_precision.get("manifest_commit"),
        EXPECTED_ANY_PRECISION_REVISION,
        "Any-Precision protocol revision",
    )
    _check_equal(
        errors,
        manifest_backend.get("commit"),
        EXPECTED_ANY_PRECISION_REVISION,
        "manifest Any-Precision revision",
    )
    submodule_path_value = any_precision.get("submodule_path")
    _check_equal(
        errors,
        submodule_path_value,
        "third_party/any-precision-llm",
        "Any-Precision submodule path",
    )
    if isinstance(submodule_path_value, str) and not Path(submodule_path_value).is_absolute():
        submodule_path = root / submodule_path_value
        _add(errors, submodule_path.is_dir(), f"Any-Precision submodule is unavailable: {submodule_path}")
        if submodule_path.is_dir():
            _check_equal(
                errors,
                _git_revision(submodule_path),
                EXPECTED_ANY_PRECISION_REVISION,
                "Any-Precision checked-out revision",
            )

    artifact = identities.get("packed_artifact", {})
    artifact_meta = manifest.get("artifact", {})
    _check_equal(
        errors,
        artifact.get("relative_path"),
        artifact_meta.get("local_path"),
        "packed artifact path",
    )
    _check_equal(
        errors, artifact.get("checkpoint_file"), "pytorch_model.bin", "packed checkpoint filename"
    )
    checkpoint_hash = next(
        (
            item.get("sha256")
            for item in artifact_meta.get("artifact_file_list", [])
            if item.get("path") == "pytorch_model.bin"
        ),
        None,
    )
    _check_equal(errors, checkpoint_hash, EXPECTED_PACKED_SHA256, "manifest packed checkpoint hash")
    _check_equal(errors, artifact.get("sha256"), checkpoint_hash, "protocol packed checkpoint hash")
    _check_equal(
        errors,
        manifest.get("artifact", {}).get("checkpoint_hashes", {}).get("pytorch_model.bin"),
        checkpoint_hash,
        "manifest checkpoint hash agreement",
    )

    router = identities.get("router", {})
    _check_equal(
        errors, router.get("sha256"), EXPECTED_ROUTER_SHA256, "protocol router checkpoint hash"
    )

    for result_name in (
        "s03_static_quality.json",
        "s07_router_training.json",
        "s08_on_demand.json",
    ):
        result_path = root / "docs" / "results" / result_name
        _add(errors, result_path.is_file(), f"required recorded result exists: {result_path}")
    if (root / "docs" / "results" / "s03_static_quality.json").is_file():
        result = _load_json(root / "docs" / "results" / "s03_static_quality.json")
        source = result.get("source_model", {})
        _check_equal(
            errors,
            source.get("repository"),
            EXPECTED_MODEL_REPOSITORY,
            "S03 result model repository",
        )
        _check_equal(
            errors, source.get("revision"), EXPECTED_MODEL_REVISION, "S03 result model revision"
        )
        _check_equal(
            errors,
            source.get("tokenizer_revision"),
            EXPECTED_MODEL_REVISION,
            "S03 result tokenizer revision",
        )
        _check_equal(
            errors,
            source.get("any_precision_commit"),
            EXPECTED_ANY_PRECISION_REVISION,
            "S03 result Any-Precision revision",
        )
        _check_equal(
            errors,
            source.get("quantized_checkpoint_sha256"),
            EXPECTED_PACKED_SHA256,
            "S03 result packed hash",
        )
    if (root / "docs" / "results" / "s07_router_training.json").is_file():
        result = _load_json(root / "docs" / "results" / "s07_router_training.json")
        source = result.get("source_model", {})
        _check_equal(
            errors, source.get("revision"), EXPECTED_MODEL_REVISION, "S07 result model revision"
        )
        _check_equal(
            errors,
            source.get("packed_student_checkpoint_sha256"),
            EXPECTED_PACKED_SHA256,
            "S07 result packed hash",
        )
        _check_equal(
            errors,
            source.get("any_precision_revision"),
            EXPECTED_ANY_PRECISION_REVISION,
            "S07 result Any-Precision revision",
        )
    if (root / "docs" / "results" / "s08_on_demand.json").is_file():
        result = _load_json(root / "docs" / "results" / "s08_on_demand.json")
        _check_equal(
            errors,
            result.get("model_revision"),
            EXPECTED_MODEL_REVISION,
            "S08 result model revision",
        )
        _check_equal(
            errors,
            result.get("packed_checkpoint_sha256"),
            EXPECTED_PACKED_SHA256,
            "S08 result packed hash",
        )
        _check_equal(
            errors,
            result.get("any_precision_revision"),
            EXPECTED_ANY_PRECISION_REVISION,
            "S08 result Any-Precision revision",
        )
        _check_equal(
            errors,
            result.get("router_checkpoint_sha256"),
            EXPECTED_ROUTER_SHA256,
            "S08 result router hash",
        )


def _validate_dataset(config: dict[str, Any], errors: list[str]) -> None:
    perplexity = config.get("perplexity", {})
    for key, expected in EXPECTED_DATASET.items():
        _check_equal(errors, perplexity.get(key), expected, f"perplexity {key}")
    for key, expected in (
        ("sequence_length", 128),
        ("source_window_length", 129),
        ("stride", 128),
        ("sample_count", EXPECTED_SAMPLE_COUNT),
        ("evaluated_token_count", EXPECTED_EVALUATED_TOKENS),
        ("padding", "none"),
        ("generated_tokens", False),
        ("random_seed", None),
    ):
        _check_equal(errors, perplexity.get(key), expected, f"perplexity {key}")
    _check_equal(
        errors,
        perplexity.get("source_window_length"),
        perplexity.get("sequence_length", 0) + 1,
        "source window relationship",
    )
    _check_equal(
        errors,
        perplexity.get("stride"),
        perplexity.get("sequence_length"),
        "non-overlapping target stride",
    )
    _check_equal(
        errors,
        perplexity.get("sample_count", 0) * perplexity.get("sequence_length", 0),
        EXPECTED_EVALUATED_TOKENS,
        "evaluated token arithmetic",
    )
    _check_equal(
        errors,
        perplexity.get("selection"),
        EXPECTED_DATASET_SELECTION,
        "dataset selection policy",
    )
    _check_equal(errors, perplexity.get("padding"), "none", "padding exclusion")
    _check_equal(errors, perplexity.get("generated_tokens"), False, "generated-token exclusion")
    _check_equal(errors, perplexity.get("labels"), EXPECTED_LABELS, "next-token labels")
    _check_equal(errors, perplexity.get("loss"), EXPECTED_LOSS, "token-weighted loss")
    _check_equal(
        errors,
        perplexity.get("tokenizer_revision"),
        EXPECTED_MODEL_REVISION,
        "perplexity tokenizer revision",
    )


def _validate_fixed_input_contract(config: dict[str, Any], errors: list[str]) -> None:
    fixed = config.get("fixed_inputs", {})
    _check_equal(errors, fixed.get("path"), "configs/s09_baseline_prompts.json", "fixed input source")
    _check_equal(
        errors,
        fixed.get("runtime_prompt_generation"),
        False,
        "fixed input runtime prompt generation",
    )
    _check_equal(
        errors,
        fixed.get("applicable_inputs_identical_across_modes"),
        True,
        "fixed inputs identical across modes",
    )

    routed = fixed.get("routed_recording")
    _add(errors, isinstance(routed, dict), "routed prompt recording contract is missing")
    if isinstance(routed, dict):
        for field, expected in EXPECTED_ROUTED_RECORDING.items():
            _check_equal(errors, routed.get(field), expected, f"routed recording {field}")
        _check_equal(
            errors,
            routed.get("units_per_request"),
            routed.get("layers", 0) * len(routed.get("unit_types", []))
            if isinstance(routed.get("layers"), int) and isinstance(routed.get("unit_types"), list)
            else None,
            "routed recording unit arithmetic",
        )

    comparison = config.get("comparison_contract", {})
    for key in (
        "same_model_and_tokenizer_revision",
        "identical_applicable_input_token_ids",
        "resident_and_on_demand_generation_settings_identical",
    ):
        _check_equal(errors, comparison.get(key), True, f"comparison contract {key}")
    deterministic = config.get("deterministic_criteria", {})
    for key in (
        "route_maps_complete",
        "resident_on_demand_route_maps_equal",
        "resident_on_demand_outputs_match_existing_correctness_criterion",
    ):
        _check_equal(errors, deterministic.get(key), True, f"deterministic criteria {key}")


def _validate_generation_and_seeds(config: dict[str, Any], errors: list[str]) -> None:
    generation = config.get("generation", {})
    for key, expected in (
        ("input_source", "fixed_inputs"),
        ("batch_size", 1),
        ("decoding", "greedy"),
        ("do_sample", False),
        ("num_beams", 1),
        ("temperature", None),
        ("max_new_tokens", 8),
        ("same_settings_and_limits_across_modes", True),
    ):
        _check_equal(errors, generation.get(key), expected, f"generation {key}")

    seeds = config.get("seeds")
    _add(errors, isinstance(seeds, dict), "deterministic seed policy is missing")
    if isinstance(seeds, dict):
        for key, expected in (
            ("global_reproducibility_seed", 1729),
            ("perplexity_random_sampling", False),
            ("generation_sampling", False),
            ("runtime_prompt_generation", False),
        ):
            _check_equal(errors, seeds.get(key), expected, f"seed policy {key}")


def _validate_prompts(
    config: dict[str, Any], root: Path, errors: list[str], prompt_payload: dict[str, Any] | None
) -> None:
    fixed = config.get("fixed_inputs", {})
    path_value = fixed.get("path")
    _add(
        errors,
        isinstance(path_value, str) and not Path(path_value).is_absolute(),
        "fixed input path must be repository-relative",
    )
    prompt_path = (
        root / path_value if isinstance(path_value, str) else root / "missing-prompts.json"
    )
    if prompt_payload is None:
        if not prompt_path.is_file():
            errors.append(f"fixed prompt file is missing: {prompt_path}")
            return
        prompt_payload = _load_json(prompt_path)
    requests = prompt_payload.get("requests")
    _add(errors, isinstance(requests, list), "fixed prompt requests must be a list")
    if not isinstance(requests, list):
        return
    ids = [item.get("id") if isinstance(item, dict) else None for item in requests]
    _check_equal(errors, len(requests), fixed.get("request_count"), "fixed request count")
    _check_equal(errors, tuple(ids), tuple(fixed.get("request_ids", [])), "fixed request IDs")
    _check_equal(errors, tuple(ids), EXPECTED_PROMPT_IDS, "expected fixed request IDs")
    _check_equal(errors, len(set(ids)), len(ids), "fixed request IDs must be unique")
    for item in requests:
        if not isinstance(item, dict):
            errors.append("each fixed request must be an object")
            continue
        input_ids = item.get("input_ids")
        _add(
            errors,
            isinstance(input_ids, list)
            and all(isinstance(value, int) and not isinstance(value, bool) for value in input_ids),
            f"{item.get('id')} input IDs must be integers",
        )
        if not isinstance(input_ids, list):
            continue
        _check_equal(
            errors,
            len(input_ids),
            item.get("input_token_count"),
            f"{item.get('id')} input token count",
        )
        _check_equal(
            errors, len(input_ids), item.get("prompt_length"), f"{item.get('id')} prompt length"
        )
        _add(
            errors,
            bool(str(item.get("prompt_text", "")).strip()),
            f"{item.get('id')} prompt text is non-empty",
        )
        if item.get("kind") == "s07_s08_validation_request":
            full_ids = item.get("full_input_ids")
            _add(
                errors,
                isinstance(full_ids, list) and len(full_ids) == item.get("full_input_token_count"),
                f"{item.get('id')} full input IDs are complete",
            )
            _add(
                errors,
                isinstance(full_ids, list) and full_ids[: len(input_ids)] == input_ids,
                f"{item.get('id')} prompt IDs match full input prefix",
            )
            if isinstance(full_ids, list) and all(isinstance(value, int) for value in full_ids):
                _check_equal(
                    errors,
                    _fixed_input_digest(full_ids),
                    item.get("full_input_ids_sha256"),
                    f"{item.get('id')} full input digest",
                )
            _check_equal(
                errors,
                item.get("prompt_token_range"),
                [0, item.get("prompt_length")],
                f"{item.get('id')} prompt range",
            )
    _check_equal(
        errors,
        prompt_payload.get("tokenizer", {}).get("revision"),
        EXPECTED_MODEL_REVISION,
        "fixed prompt tokenizer revision",
    )
    _check_equal(
        errors,
        prompt_payload.get("tokenizer", {}).get("runtime_prompt_generation"),
        False,
        "runtime prompt generation",
    )

    s03_file = root / "configs" / "s03_static_quality_prompts.txt"
    if s03_file.is_file():
        source_lines = [line for line in s03_file.read_text().splitlines() if line.strip()]
        for item in requests:
            if item.get("kind") != "s03_quality_prompt":
                continue
            line_number = item.get("source_line")
            _add(
                errors,
                isinstance(line_number, int) and 1 <= line_number <= len(source_lines),
                f"{item.get('id')} source line exists",
            )
            if isinstance(line_number, int) and 1 <= line_number <= len(source_lines):
                _check_equal(
                    errors,
                    item.get("prompt_text"),
                    source_lines[line_number - 1],
                    f"{item.get('id')} prompt text agrees with S03 source",
                )

    s07 = root / "docs" / "results" / "s07_router_training.json"
    if s07.is_file():
        manifest = _load_json(s07).get("dataset_manifest", {}).get("validation", [])
        by_id = {item.get("example_id"): item for item in manifest}
        for item in requests:
            if item.get("kind") != "s07_s08_validation_request":
                continue
            recorded = by_id.get(item.get("id"))
            _add(
                errors, recorded is not None, f"{item.get('id')} exists in S07 validation manifest"
            )
            if recorded is not None:
                for key in (
                    "source_offset",
                    "source_row",
                    "prompt_text",
                    "prompt_token_range",
                    "full_input_ids_sha256",
                ):
                    source_key = "input_ids_sha256" if key == "full_input_ids_sha256" else key
                    _check_equal(
                        errors,
                        item.get(key),
                        recorded.get(source_key),
                        f"{item.get('id')} {key} agrees with S07 manifest",
                    )


def _validate_measurement_contract(config: dict[str, Any], errors: list[str]) -> None:
    generation = config.get("generation", {})
    _add(
        errors,
        "generated_token_ids" in generation.get("records", []),
        "generation records must include generated token IDs",
    )
    _add(
        errors,
        "output_digest" in generation.get("records", []),
        "generation records must include output digest",
    )

    memory = config.get("memory", {})
    required_memory = {
        "allocated_before",
        "reserved_before",
        "peak_allocated",
        "peak_reserved",
        "allocated_after_cleanup",
        "reserved_after_cleanup",
    }
    _add(
        errors,
        required_memory.issubset(set(memory.get("record_for_all_modes", []))),
        "memory contract is incomplete",
    )
    _add(
        errors,
        memory.get("process_policy") == "fresh process for each mode",
        "memory must use fresh processes",
    )
    _check_equal(
        errors, memory.get("empty_cache_inside_interval"), False, "empty_cache inside interval"
    )
    _check_equal(
        errors,
        memory.get("reserved_is_not_live_residency"),
        True,
        "reserved-memory residency distinction",
    )
    _check_equal(
        errors, memory.get("physical_buffers_only"), True, "physical buffer memory accounting"
    )

    latency = config.get("latency", {})
    for key, expected in (
        ("warmup_requests", 1),
        ("repeats_per_fixed_latency_request", 5),
        ("retain_every_raw_value", True),
        ("outlier_removal", False),
        ("subtract_transfer_time", False),
        ("on_demand_end_to_end_includes_transfer", True),
    ):
        _check_equal(errors, latency.get(key), expected, f"latency {key}")
    _add(
        errors,
        "prefill" in latency.get("record_phases", [])
        and "decode" in latency.get("record_phases", [])
        and "end_to_end" in latency.get("record_phases", []),
        "latency phases are incomplete",
    )
    _check_equal(
        errors,
        latency.get("end_on_demand_warmup_before_measurement"),
        True,
        "on-demand warm-up cleanup",
    )

    transfer = config.get("transfer", {})
    _check_equal(errors, transfer.get("mode"), EXPECTED_TRANSFER_MODE, "transfer mode")
    _check_equal(errors, transfer.get("rule"), EXPECTED_TRANSFER_RULE, "D029 transfer rule")
    _check_equal(
        errors,
        transfer.get("expected_bytes_inputs"),
        EXPECTED_TRANSFER_INPUTS,
        "transfer expected-byte inputs",
    )
    required_transfer = {
        "first_use_bytes",
        "reuse_bytes",
        "prefill_bytes",
        "decode_bytes",
        "attention_bytes",
        "ffn_bytes",
        "total_bytes",
        "first_use_events",
        "reuse_events",
        "independently_expected_physical_bytes",
    }
    _add(
        errors,
        required_transfer.issubset(set(transfer.get("record", []))),
        "transfer accounting contract is incomplete",
    )
    _check_equal(
        errors, transfer.get("require_actual_equals_expected"), True, "transfer exact equality gate"
    )
    _check_equal(
        errors, transfer.get("transfer_unpacked_weights"), False, "unpacked transfer prohibition"
    )

    hardware = config.get("hardware", {})
    _check_equal(errors, hardware.get("policy"), EXPECTED_HARDWARE_POLICY, "hardware policy")
    _check_equal(
        errors,
        hardware.get("identical_rtx3090_substitution"),
        EXPECTED_GPU_SUBSTITUTION,
        "GPU comparability policy",
    )
    _check_equal(errors, hardware.get("preferred_device_index"), 3, "preferred device index")
    _check_equal(
        errors,
        hardware.get("required_gpu_model"),
        "NVIDIA GeForce RTX 3090",
        "required GPU model",
    )
    _add(
        errors,
        isinstance(hardware.get("preferred_device_index"), int),
        "hardware device index is required",
    )
    _add(errors, bool(hardware.get("required_gpu_model")), "hardware GPU model is required")
    _add(
        errors,
        hardware.get("record_versions") == EXPECTED_HARDWARE_RECORD_VERSIONS,
        "hardware version record is incomplete",
    )


def _validate_release_criteria(config: dict[str, Any], errors: list[str]) -> None:
    release = config.get("release_criteria", {})
    structural = release.get("structural_reproducibility_failures", [])
    required_structural = (
        "all five modes execute",
        "required outputs are finite",
        "artifact identities and hashes are exact",
        "actual transfer bytes exactly equal independently expected physical bytes",
        "cleanup releases request-owned loader references",
        "on-demand has no hidden complete packed GPU copy",
        "relevant regressions pass",
    )
    for criterion in required_structural:
        _add(errors, criterion in structural, f"missing structural release criterion: {criterion}")
    quality = release.get("quality_gates", {})
    for key in ("static_8_perplexity", "routed_resident_perplexity", "routed_on_demand_perplexity"):
        _add(errors, key in quality, f"missing quality gate: {key}")
    _check_equal(
        errors,
        quality.get("static_8_perplexity", {}).get("margin"),
        1.10,
        "static 8 quality margin",
    )
    _check_equal(
        errors,
        quality.get("routed_resident_perplexity", {}).get("margin"),
        1.10,
        "routed resident quality margin",
    )
    _add(
        errors,
        "REVISE" == release.get("failure_outcomes", {}).get("structural_or_quality_failure"),
        "structural/quality failure must be REVISE",
    )
    _add(
        errors,
        "PAUSE" == release.get("failure_outcomes", {}).get("missing_hardware_or_external_resource"),
        "missing resource must be PAUSE",
    )
    _check_equal(
        errors,
        release.get("failure_outcomes", {}).get("all_gates_pass"),
        "CONTINUE_TO_S09B",
        "all-gates-pass outcome",
    )
    _add(
        errors,
        release.get("post_result_protocol_change")
        == "Any genuine defect requires REVISE and invalidation of affected results; "
        "never silently edit frozen inputs or gates",
        "post-result policy must require REVISE and invalidation",
    )
    deferred = config.get("deferred_mechanisms")
    _check_equal(
        errors,
        deferred,
        EXPECTED_DEFERRED_MECHANISMS,
        "complete deferred mechanisms",
    )


def _validate_external_artifacts(
    config: dict[str, Any],
    root: Path,
    manifest: dict[str, Any],
    errors: list[str],
    verify_hashes: bool,
) -> None:
    artifact_config = config["identities"]["packed_artifact"]
    artifact_override = os.environ.get("QAQ_S03_ARTIFACT")
    artifact_path = (
        Path(artifact_override).expanduser()
        if artifact_override
        else root / artifact_config["relative_path"]
    )
    _add(
        errors, artifact_path.is_dir(), f"packed artifact directory is unavailable: {artifact_path}"
    )
    if artifact_path.is_dir() and verify_hashes:
        for item in manifest["artifact"]["artifact_file_list"]:
            path = artifact_path / item["path"]
            _add(errors, path.is_file(), f"packed artifact file is unavailable: {path}")
            if path.is_file():
                _check_equal(
                    errors, path.stat().st_size, item["bytes"], f"artifact size {item['path']}"
                )
                _check_equal(errors, _sha256(path), item["sha256"], f"artifact hash {item['path']}")
    router_override = os.environ.get("QAQ_S07_ROUTER_CHECKPOINT")
    router_path = (
        Path(router_override).expanduser()
        if router_override
        else Path("~/.cache/qaq/s07b/final_router.pt").expanduser()
    )
    _add(errors, router_path.is_file(), f"router checkpoint is unavailable: {router_path}")
    if router_path.is_file() and verify_hashes:
        _check_equal(errors, _sha256(router_path), EXPECTED_ROUTER_SHA256, "router checkpoint hash")
    snapshot_override = os.environ.get("QAQ_MODEL_SNAPSHOT")
    snapshot = (
        Path(snapshot_override).expanduser()
        if snapshot_override
        else Path(
            "~/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/" + EXPECTED_MODEL_REVISION
        ).expanduser()
    )
    _add(errors, snapshot.is_dir(), f"pinned model snapshot is unavailable: {snapshot}")
    for filename in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
        _add(
            errors,
            (snapshot / filename).is_file(),
            f"pinned tokenizer file is unavailable: {snapshot / filename}",
        )


def validate_protocol_payload(
    config: dict[str, Any],
    root: Path = ROOT,
    *,
    prompt_payload: dict[str, Any] | None = None,
    manifest: dict[str, Any] | None = None,
    check_external: bool = False,
    verify_hashes: bool = False,
) -> dict[str, Any]:
    """Validate a protocol payload; tests can disable external artifact checks."""

    errors: list[str] = []
    if manifest is None:
        manifest_path = root / config.get("identities", {}).get("packed_artifact", {}).get(
            "manifest_path", ""
        )
        if manifest_path.is_file():
            manifest = _load_json(manifest_path)
        else:
            errors.append(f"manifest is unavailable: {manifest_path}")
            manifest = {}
    _check_equal(errors, config.get("schema"), "qaq-s09-baseline-eval-v1", "schema")
    _check_equal(
        errors, config.get("protocol_frozen_before_final_results"), True, "protocol freeze marker"
    )
    _check_equal(errors, config.get("stage"), "S09-A", "stage")
    _validate_modes(config, errors)
    _validate_identities(config, root, manifest, errors)
    _validate_dataset(config, errors)
    _validate_fixed_input_contract(config, errors)
    _validate_generation_and_seeds(config, errors)
    _validate_prompts(config, root, errors, prompt_payload)
    _validate_measurement_contract(config, errors)
    _validate_release_criteria(config, errors)
    if check_external:
        _validate_external_artifacts(config, root, manifest, errors, verify_hashes)
    if errors:
        raise ProtocolValidationError("S09 protocol validation failed:\n- " + "\n- ".join(errors))
    return {
        "schema": config["schema"],
        "mode_count": len(config["modes"]),
        "request_count": config["fixed_inputs"]["request_count"],
        "sample_count": config["perplexity"]["sample_count"],
        "evaluated_token_count": config["perplexity"]["evaluated_token_count"],
        "external_artifacts_checked": check_external,
        "artifact_hashes_checked": bool(check_external and verify_hashes),
    }


def validate_protocol(
    config_path: Path = DEFAULT_CONFIG, *, check_external: bool = True, verify_hashes: bool = True
) -> dict[str, Any]:
    config = _load_json(config_path)
    return validate_protocol_payload(
        config, config_path.parents[1], check_external=check_external, verify_hashes=verify_hashes
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--skip-external",
        action="store_true",
        help="only for unit tests; skip artifact/snapshot resolution",
    )
    parser.add_argument(
        "--skip-hashes",
        action="store_true",
        help="only for unit tests; skip external SHA-256 checks",
    )
    args = parser.parse_args(argv)
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    try:
        result = validate_protocol(
            config_path, check_external=not args.skip_external, verify_hashes=not args.skip_hashes
        )
    except ProtocolValidationError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
