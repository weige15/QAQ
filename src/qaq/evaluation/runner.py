"""S09-B frozen-protocol orchestration, per-mode results, and aggregation.

The parent process only validates and launches one fresh child per mode.  A
child owns exactly one model and writes one structured result when explicitly
asked to execute.  ``--plan`` never imports model loaders or runs CUDA work.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable
from itertools import combinations
from pathlib import Path
from statistics import median
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "configs" / "s09_baseline_eval.json"
DEFAULT_PROMPTS = ROOT / "configs" / "s09_baseline_prompts.json"
DEFAULT_RESULTS = ROOT / "docs" / "results" / "s09b"
RESULT_SCHEMA = "qaq-s09b-per-mode-result-v1"
EXPECTED_MODE_IDS = (
    "full_precision_bf16_teacher",
    "static_packed_4bit",
    "static_packed_8bit",
    "hard_routed_resident_packed",
    "hard_routed_synchronous_on_demand_packed",
)
ROUTED_MODE_IDS = EXPECTED_MODE_IDS[-2:]
ON_DEMAND_MODE_ID = EXPECTED_MODE_IDS[-1]
MODEL_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
ANY_PRECISION_REVISION = "a3257d02740cc5757c78673da534b0630ff3a4ea"


class EvaluationRunnerError(ValueError):
    """A frozen protocol, result, or orchestration contract is invalid."""


def _json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationRunnerError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise EvaluationRunnerError(f"JSON root must be an object: {path}")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fixed_input_digest(values: list[int]) -> str:
    import struct

    return hashlib.sha256(struct.pack("<" + "q" * len(values), *values)).hexdigest()


def _tensor_digest(values: list[Any]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.detach().float().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def load_protocol(config_path: Path = DEFAULT_CONFIG) -> tuple[dict[str, Any], dict[str, Any], str]:
    config_path = config_path.resolve()
    config = _json(config_path)
    prompt_path = (ROOT / config["fixed_inputs"]["path"]).resolve()
    prompts = _json(prompt_path)
    return config, prompts, sha256_file(config_path)


def resolve_modes(config: dict[str, Any]) -> list[dict[str, Any]]:
    modes = config.get("modes")
    if not isinstance(modes, list):
        raise EvaluationRunnerError("frozen modes must be a list")
    ids = [mode.get("id") if isinstance(mode, dict) else None for mode in modes]
    if tuple(ids) != EXPECTED_MODE_IDS:
        raise EvaluationRunnerError(f"frozen mode IDs must be exactly {EXPECTED_MODE_IDS}; got {ids}")
    if len(set(ids)) != len(ids):
        raise EvaluationRunnerError("frozen mode IDs must be unique")
    return [dict(mode) for mode in modes]


def frozen_perplexity_arguments(config: dict[str, Any]) -> dict[str, Any]:
    section = config["perplexity"]
    expected = {"sample_count": 32, "sequence_length": 128, "source_window_length": 129, "stride": 128, "evaluated_token_count": 4096, "labels": "window[1:] aligned with logits from window[:-1]", "evaluator": section["evaluator"]}
    for key, value in expected.items():
        if section.get(key) != value:
            raise EvaluationRunnerError(f"S09 perplexity adapter drift: {key}")
    return expected


def frozen_generation_arguments(config: dict[str, Any]) -> dict[str, Any]:
    section = config["generation"]
    expected = {"batch_size": 1, "do_sample": False, "num_beams": 1, "max_new_tokens": 8}
    for key, value in expected.items():
        if section.get(key) != value:
            raise EvaluationRunnerError(f"S09 generation adapter drift: {key}")
    return expected


def frozen_latency_repeats(config: dict[str, Any]) -> int:
    repeats = config["latency"].get("repeats_per_fixed_latency_request")
    if repeats != 5:
        raise EvaluationRunnerError(f"S09 latency adapter drift: expected five repeats, got {repeats}")
    return repeats


def fixed_requests(prompts: dict[str, Any]) -> list[dict[str, Any]]:
    requests = prompts.get("requests")
    if not isinstance(requests, list) or not requests:
        raise EvaluationRunnerError("fixed prompt requests are missing")
    result = []
    seen: set[str] = set()
    for request in requests:
        if not isinstance(request, dict) or not isinstance(request.get("id"), str):
            raise EvaluationRunnerError("fixed prompt request is malformed")
        request_id = request["id"]
        if request_id in seen:
            raise EvaluationRunnerError(f"duplicate fixed request: {request_id}")
        seen.add(request_id)
        ids = request.get("input_ids")
        if not isinstance(ids, list) or not all(isinstance(value, int) for value in ids):
            raise EvaluationRunnerError(f"fixed input IDs are malformed: {request_id}")
        result.append({**request, "input_ids_sha256": fixed_input_digest(ids)})
    return result


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise EvaluationRunnerError(f"git provenance unavailable: {args}") from exc


def provenance() -> dict[str, Any]:
    return {"git_commit": _git("rev-parse", "HEAD"), "worktree_status": _git("status", "--short")}


def child_command(config_path: Path, mode_id: str, output_path: Path, device: str) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "scripts" / "run_s09b.py"),
        "--execute-mode",
        mode_id,
        "--config",
        str(config_path),
        "--output",
        str(output_path),
        "--device",
        device,
    ]


def plan(config_path: Path, output_dir: Path, device: str) -> dict[str, Any]:
    config, prompts, config_hash = load_protocol(config_path)
    # Importing the validator here keeps the plan's model path inert while
    # still making the committed validator the protocol authority.
    from qaq.evaluation.protocol import validate_protocol

    validation = validate_protocol(config_path, check_external=True, verify_hashes=True)
    modes = resolve_modes(config)
    requests = fixed_requests(prompts)
    outputs = [output_dir / f"{mode['id']}.json" for mode in modes]
    commands = [child_command(config_path.resolve(), mode["id"], output, device) for mode, output in zip(modes, outputs, strict=True)]
    aggregation = [
        sys.executable,
        str(ROOT / "scripts" / "run_s09b.py"),
        "--aggregate",
        "--config",
        str(config_path.resolve()),
        "--results-dir",
        str(output_dir),
    ]
    return {
        "safe": True,
        "model_loading": False,
        "cuda_inference": False,
        "benchmark": False,
        "writes_final_result": False,
        "config": str(config_path.resolve()),
        "config_sha256": config_hash,
        "protocol_validation": validation,
        "mode_ids": [mode["id"] for mode in modes],
        "request_ids": [request["id"] for request in requests],
        "output_paths": [str(path) for path in outputs],
        "child_commands": commands,
        "aggregation_command": aggregation,
        "aggregation_result_path": str(output_dir / "aggregation.json"),
    }


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _require(mapping: dict[str, Any], path: str) -> Any:
    value: Any = mapping
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise EvaluationRunnerError(f"result missing required field: {path}")
        value = value[part]
    return value


def _require_finite(mapping: dict[str, Any], path: str) -> None:
    value = _require(mapping, path)
    if not _finite(value):
        raise EvaluationRunnerError(f"result field is not finite: {path}")


def _validate_memory(result: dict[str, Any]) -> None:
    records = _require(result, "memory.records")
    if not isinstance(records, list) or not records:
        raise EvaluationRunnerError("memory.records must be non-empty")
    fields = (
        "allocated_before",
        "reserved_before",
        "peak_allocated",
        "peak_reserved",
        "allocated_after_cleanup",
        "reserved_after_cleanup",
    )
    for record in records:
        if not isinstance(record, dict):
            raise EvaluationRunnerError("memory record is malformed")
        for field in fields:
            value = record.get(field)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise EvaluationRunnerError(f"memory field is invalid: {field}")
    for field in ("physically_resident_packed_weight_bytes", "request_owned_on_demand_bytes"):
        value = _require(result, f"memory.{field}")
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise EvaluationRunnerError(f"memory physical evidence is invalid: {field}")
    method = _require(result, "memory.method")
    if (
        method.get("synchronize_before") != "torch.cuda.synchronize()"
        or method.get("reset_peak") != "torch.cuda.reset_peak_memory_stats()"
        or method.get("synchronize_after") != "torch.cuda.synchronize()"
    ):
        raise EvaluationRunnerError("memory measurement boundaries are incomplete")
    if method.get("empty_cache_inside_interval") is not False:
        raise EvaluationRunnerError("empty_cache() is forbidden inside the memory interval")


def _validate_latency(result: dict[str, Any], repeats: int) -> None:
    latency = _require(result, "latency")
    if latency.get("warmup_requests") != 1 or latency.get("repeats_per_request") != repeats:
        raise EvaluationRunnerError("latency warm-up or repeat count does not match the frozen protocol")
    raw = latency.get("raw_records")
    request_ids = result.get("fixed_inputs", {}).get("request_ids")
    if not isinstance(raw, list) or not raw or not isinstance(request_ids, list):
        raise EvaluationRunnerError("latency raw records are missing")
    counts = Counter(record.get("request_id") for record in raw if isinstance(record, dict))
    if set(counts) != set(request_ids) or any(count != repeats for count in counts.values()):
        raise EvaluationRunnerError("latency raw records do not retain exactly five repeats per request")
    for record in raw:
        if not isinstance(record, dict):
            raise EvaluationRunnerError("latency raw record is malformed")
        for field in ("prefill_seconds", "decode_seconds", "end_to_end_seconds"):
            value = record.get(field)
            if not _finite(value) or value < 0:
                raise EvaluationRunnerError(f"latency field is invalid: {field}")
    headlines = latency.get("median_seconds")
    if not isinstance(headlines, dict) or set(headlines) != set(request_ids):
        raise EvaluationRunnerError("latency median headlines are missing")
    for request_id in request_ids:
        records = [record for record in raw if record.get("request_id") == request_id]
        headline = headlines.get(request_id)
        if not isinstance(headline, dict):
            raise EvaluationRunnerError(f"latency median headline is missing: {request_id}")
        for field in ("prefill", "decode", "end_to_end"):
            value = headline.get(field)
            if not _finite(value) or value < 0:
                raise EvaluationRunnerError(f"latency median is invalid: {request_id}.{field}")
            expected = median(record[f"{field}_seconds"] for record in records)
            if value != expected:
                raise EvaluationRunnerError(f"latency median does not match raw repeats: {request_id}.{field}")
    if latency.get("outlier_removal") is not False or latency.get("subtract_transfer_time") is not False:
        raise EvaluationRunnerError("latency filtering or transfer subtraction is not allowed")


def _validate_generation(result: dict[str, Any], requests: list[dict[str, Any]], config: dict[str, Any]) -> None:
    generation = _require(result, "generation")
    expected = config["generation"]
    for field in ("batch_size", "do_sample", "num_beams", "max_new_tokens"):
        if generation.get(field) != expected[field]:
            raise EvaluationRunnerError(f"generation setting mismatch: {field}")
    records = generation.get("records")
    if not isinstance(records, list) or {item.get("request_id") for item in records if isinstance(item, dict)} != {item["id"] for item in requests}:
        raise EvaluationRunnerError("generation records do not cover the fixed requests")
    expected_inputs = {item["id"]: item["input_ids_sha256"] for item in requests}
    for record in records:
        if not isinstance(record, dict) or record.get("input_ids_sha256") != expected_inputs.get(record.get("request_id")):
            raise EvaluationRunnerError("generation record does not use the committed input IDs")
        if not isinstance(record.get("generated_token_ids"), list) or len(record["generated_token_ids"]) > expected["max_new_tokens"]:
            raise EvaluationRunnerError("generation token record is invalid")
        if not isinstance(record.get("output_digest"), str) or not record["output_digest"]:
            raise EvaluationRunnerError("generation output digest is missing")
        if not isinstance(record.get("logits_digest"), str) or not record["logits_digest"]:
            raise EvaluationRunnerError("generation logits digest is missing")
        if not isinstance(record.get("finite_value_check"), bool) or not isinstance(record.get("normal_termination"), bool):
            raise EvaluationRunnerError("generation deterministic checks are incomplete")


def _validate_routed(result: dict[str, Any], requests: list[dict[str, Any]]) -> None:
    routed = _require(result, "routed")
    records = routed.get("requests")
    if not isinstance(records, list) or {item.get("request_id") for item in records if isinstance(item, dict)} != {item["id"] for item in requests}:
        raise EvaluationRunnerError("routed records do not cover the fixed requests")
    for record in records:
        route_map = record.get("route_map") if isinstance(record, dict) else None
        if not isinstance(route_map, list) or len(route_map) != 72:
            raise EvaluationRunnerError("routed result must contain complete 72-unit route maps")
        keys = {(item.get("layer"), item.get("unit_type")) for item in route_map if isinstance(item, dict)}
        expected = {(layer, unit) for layer in range(36) for unit in ("attention", "ffn")}
        if keys != expected or any(item.get("selected_bits") not in (4, 8) for item in route_map):
            raise EvaluationRunnerError("routed route map is incomplete")
        for field in ("route_map_digest", "attention_fractions", "ffn_fractions", "overall_fractions"):
            if field not in record:
                raise EvaluationRunnerError(f"routed result missing {field}")
        if record["route_map_digest"] != _route_digest(route_map):
            raise EvaluationRunnerError("routed route-map digest does not match the measured map")
        for field, unit_type in (("attention_fractions", "attention"), ("ffn_fractions", "ffn"), ("overall_fractions", None)):
            values = record[field]
            selected = route_map if unit_type is None else [item for item in route_map if item["unit_type"] == unit_type]
            expected_fractions = {"4_bit": sum(item["selected_bits"] == 4 for item in selected) / len(selected), "8_bit": sum(item["selected_bits"] == 8 for item in selected) / len(selected)}
            if not isinstance(values, dict) or values != expected_fractions:
                raise EvaluationRunnerError(f"routed fractions are not measured: {field}")
    if not isinstance(routed.get("route_diversity"), dict):
        raise EvaluationRunnerError("route diversity summary is missing")


def _validate_on_demand(result: dict[str, Any]) -> None:
    payload = _require(result, "on_demand")
    for field in (
        "first_use_bytes",
        "reuse_bytes",
        "prefill_bytes",
        "decode_bytes",
        "attention_bytes",
        "ffn_bytes",
        "total_transfer_bytes",
        "first_use_events",
        "reuse_events",
        "independently_expected_physical_bytes",
    ):
        value = payload.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise EvaluationRunnerError(f"on-demand transfer field is invalid: {field}")
    if payload.get("actual_equals_expected") is not True:
        raise EvaluationRunnerError("on-demand transfer equality evidence is missing")
    cleanup = payload.get("cleanup_records")
    if not isinstance(cleanup, list) or not cleanup:
        raise EvaluationRunnerError("on-demand cleanup audit records are missing")
    cleanup_fields = (
        "retained_entries_before_cleanup",
        "retained_buffers_before_cleanup",
        "retained_entries_after_cleanup",
        "retained_buffers_after_cleanup",
        "retained_bytes_after_cleanup",
    )
    for record in cleanup:
        if not isinstance(record, dict) or not isinstance(record.get("request_id"), str):
            raise EvaluationRunnerError("on-demand cleanup audit record is malformed")
        for field in cleanup_fields:
            value = record.get(field)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise EvaluationRunnerError(f"on-demand cleanup field is invalid: {field}")
    for field in cleanup_fields:
        value = payload.get(field)
        expected = max(record[field] for record in cleanup)
        if value != expected:
            raise EvaluationRunnerError(f"on-demand cleanup summary is not measured: {field}")
    if any(payload[field] for field in cleanup_fields[2:]):
        raise EvaluationRunnerError("on-demand request resources were not released")
    audit = payload.get("hidden_copy_audit")
    if not isinstance(audit, dict):
        raise EvaluationRunnerError("on-demand hidden-copy audit evidence is missing")
    for field in ("any_precision_module_count", "source_count"):
        if not isinstance(audit.get(field), int) or audit[field] < 0:
            raise EvaluationRunnerError(f"on-demand hidden-copy audit field is invalid: {field}")
    for field in ("all_source_qweights_cpu", "all_source_luts_cpu", "no_complete_packed_gpu_copy", "all_repeats_passed"):
        if not isinstance(audit.get(field), bool):
            raise EvaluationRunnerError(f"on-demand hidden-copy audit field is invalid: {field}")
    if audit["any_precision_module_count"] != 0 or not audit["all_source_qweights_cpu"] or not audit["all_source_luts_cpu"] or not audit["no_complete_packed_gpu_copy"] or not audit["all_repeats_passed"]:
        raise EvaluationRunnerError("on-demand hidden-copy audit failed")
        raise EvaluationRunnerError("on-demand hidden-copy audit failed")
    if payload.get("no_complete_packed_parent_on_gpu") is not audit["no_complete_packed_gpu_copy"]:
        raise EvaluationRunnerError("on-demand hidden-copy summary is not measured")


def _validate_hardware(result: dict[str, Any], config: dict[str, Any]) -> None:
    hardware = _require(result, "hardware")
    expected_index = config["hardware"]["preferred_device_index"]
    expected_model = config["hardware"]["required_gpu_model"]
    if hardware.get("device_index") != expected_index or hardware.get("gpu_model") != expected_model:
        raise EvaluationRunnerError("hardware does not match the frozen CUDA device and GPU model")
    comparability = hardware.get("comparability")
    if comparability != {
        "reference_device_index": expected_index,
        "reference_gpu_model": expected_model,
        "identity_recorded": True,
        "compatible": True,
    }:
        raise EvaluationRunnerError("hardware comparability evidence is missing or incompatible")


def _validate_perplexity(result: dict[str, Any], config: dict[str, Any]) -> None:
    perplexity = _require(result, "perplexity")
    setup = perplexity.get("setup")
    if not isinstance(setup, dict):
        raise EvaluationRunnerError("perplexity setup evidence is missing")
    expected = frozen_perplexity_arguments(config)
    for field, value in expected.items():
        if setup.get(field) != value:
            raise EvaluationRunnerError(f"perplexity evidence mismatch: {field}")
    for field in ("dataset", "config", "revision", "split", "tokenizer_revision"):
        if setup.get(field) != config["perplexity"].get(field):
            raise EvaluationRunnerError(f"perplexity identity mismatch: {field}")
    if perplexity.get("evaluated_token_count") != expected["evaluated_token_count"]:
        raise EvaluationRunnerError("perplexity must evaluate exactly 4096 target tokens")


def _validate_deterministic_evidence(result: dict[str, Any]) -> None:
    checks = result.get("deterministic_checks")
    if not isinstance(checks, dict):
        raise EvaluationRunnerError("deterministic evidence is incomplete")
    if checks.get("fixed_inputs_identical") is not True or checks.get("all_required_outputs_finite") is not True:
        raise EvaluationRunnerError("deterministic evidence is incomplete")
    evidence = checks.get("repeat_evidence")
    request_ids = result.get("fixed_inputs", {}).get("request_ids")
    if not isinstance(evidence, list) or not isinstance(request_ids, list):
        raise EvaluationRunnerError("deterministic repeat evidence is missing")
    if {record.get("request_id") for record in evidence if isinstance(record, dict)} != set(request_ids):
        raise EvaluationRunnerError("deterministic repeat evidence does not cover fixed requests")
    for record in evidence:
        if not isinstance(record, dict) or record.get("repeat_count") != 5:
            raise EvaluationRunnerError("deterministic repeat evidence must retain five measured repeats")
        if record.get("input_ids_identical") is not True or record.get("all_outputs_finite") is not True or record.get("generated_outputs_agree") is not True:
            raise EvaluationRunnerError("deterministic repeat evidence is incomplete")
        if record.get("routed_hard_routes_agree") is not True:
            raise EvaluationRunnerError("deterministic routed-repeat evidence is incomplete")
        if not isinstance(record.get("generated_token_ids"), list) or len(record["generated_token_ids"]) != 5:
            raise EvaluationRunnerError("deterministic generated-repeat evidence is missing")
        if not isinstance(record.get("route_map_digests"), list) or len(record["route_map_digests"]) not in (0, 5):
            raise EvaluationRunnerError("deterministic route-repeat evidence is missing")


def validate_result(result: dict[str, Any], config: dict[str, Any], prompts: dict[str, Any], config_hash: str) -> None:
    if result.get("schema") != RESULT_SCHEMA:
        raise EvaluationRunnerError("unsupported per-mode result schema")
    modes = resolve_modes(config)
    mode_id = result.get("mode_id")
    mode = next((item for item in modes if item["id"] == mode_id), None)
    if mode is None:
        raise EvaluationRunnerError(f"unknown result mode: {mode_id}")
    if result.get("protocol", {}).get("config_sha256") != config_hash:
        raise EvaluationRunnerError("result protocol/config SHA-256 mismatch")
    if result.get("protocol", {}).get("frozen") is not True:
        raise EvaluationRunnerError("result does not identify the frozen protocol")
    provenance_value = result.get("provenance")
    if not isinstance(provenance_value, dict) or not isinstance(provenance_value.get("git_commit"), str) or "worktree_status" not in provenance_value:
        raise EvaluationRunnerError("result provenance is incomplete")
    hardware = result.get("hardware")
    if not isinstance(hardware, dict) or any(key not in hardware for key in ("device_index", "gpu_model", "driver", "cuda_runtime", "pytorch", "transformers", "python")):
        raise EvaluationRunnerError("hardware identity is incomplete")
    _validate_hardware(result, config)
    if not isinstance(result.get("seed"), int) or result["seed"] != config["seeds"]["global_reproducibility_seed"]:
        raise EvaluationRunnerError("seed identity is incomplete")
    fixed = result.get("fixed_inputs")
    expected_input_digests = {item["id"]: item["input_ids_sha256"] for item in fixed_requests(prompts)}
    if not isinstance(fixed, dict) or fixed.get("input_digests") != expected_input_digests:
        raise EvaluationRunnerError("fixed input digests are missing or changed")
    identities = result.get("identities")
    if not isinstance(identities, dict) or identities.get("model_repository") != config["identities"]["model"]["repository"] or identities.get("model_revision") != MODEL_REVISION or identities.get("tokenizer_revision") != MODEL_REVISION:
        raise EvaluationRunnerError("model or tokenizer revision mismatch")
    if mode["packed_artifact"]:
        if identities.get("packed_checkpoint_sha256") != config["identities"]["packed_artifact"]["sha256"]:
            raise EvaluationRunnerError("packed checkpoint SHA-256 mismatch")
        if identities.get("any_precision_revision") != ANY_PRECISION_REVISION:
            raise EvaluationRunnerError("Any-Precision revision mismatch")
    if mode_id in ROUTED_MODE_IDS and identities.get("router_checkpoint_sha256") != config["identities"]["router"]["sha256"]:
        raise EvaluationRunnerError("router checkpoint SHA-256 mismatch")
    if result.get("fixed_inputs", {}).get("request_ids") != [item["id"] for item in fixed_requests(prompts)]:
        raise EvaluationRunnerError("fixed request identity mismatch")
    _validate_finite_result(result)
    _validate_perplexity(result, config)
    _validate_generation(result, fixed_requests(prompts), config)
    _validate_memory(result)
    _validate_latency(result, int(config["latency"]["repeats_per_fixed_latency_request"]))
    _validate_deterministic_evidence(result)
    if mode_id in ROUTED_MODE_IDS:
        _validate_routed(result, fixed_requests(prompts))
    if mode_id == ON_DEMAND_MODE_ID:
        _validate_on_demand(result)


def _validate_finite_result(result: dict[str, Any]) -> None:
    for path in ("perplexity.mean_negative_log_likelihood", "perplexity.perplexity"):
        _require_finite(result, path)
    if _require(result, "perplexity.evaluated_token_count") != 4096:
        raise EvaluationRunnerError("perplexity token count must be exactly 4096")


def _route_digest(route_map: list[dict[str, Any]]) -> str:
    payload = json.dumps(sorted(route_map, key=lambda item: (item["layer"], item["unit_type"])), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _route_record(state: Any, request_id: str) -> dict[str, Any]:
    route_map = [
        *[{"layer": layer, "unit_type": "attention", "selected_bits": int(bits)} for layer, bits in enumerate(state.attention_routes)],
        *[{"layer": layer, "unit_type": "ffn", "selected_bits": int(bits)} for layer, bits in enumerate(state.ffn_routes)],
    ]
    if len(route_map) != 72 or any(item["selected_bits"] not in (4, 8) for item in route_map):
        raise EvaluationRunnerError("hard routing did not produce a complete 72-unit map")
    def fractions(items: list[dict[str, Any]]) -> dict[str, float]:
        count = len(items)
        return {"4_bit": sum(item["selected_bits"] == 4 for item in items) / count, "8_bit": sum(item["selected_bits"] == 8 for item in items) / count}
    return {
        "request_id": request_id,
        "route_map": route_map,
        "route_map_digest": _route_digest(route_map),
        "attention_fractions": fractions([item for item in route_map if item["unit_type"] == "attention"]),
        "ffn_fractions": fractions([item for item in route_map if item["unit_type"] == "ffn"]),
        "overall_fractions": fractions(route_map),
    }


def _route_diversity(records: list[dict[str, Any]]) -> dict[str, Any]:
    maps = [record["route_map"] for record in records]
    # Compare the same sorted unit across request maps; this remains descriptive only.
    ordered = [sorted(route, key=lambda item: (item["layer"], item["unit_type"])) for route in maps]
    changed_units = sum(len({route[index]["selected_bits"] for route in ordered}) > 1 for index in range(72)) if ordered else 0
    distances = [sum(left["selected_bits"] != right["selected_bits"] for left, right in zip(a, b, strict=True)) / 72 for a, b in combinations(ordered, 2)]
    return {
        "unique_route_map_count": len({record["route_map_digest"] for record in records}),
        "changed_unit_count": changed_units,
        "changed_fraction": changed_units / 72,
        "pairwise_route_distance_mean": sum(distances) / len(distances) if distances else 0.0,
        "adaptivity_classification": "OTHER",
    }


def _memory_method() -> dict[str, Any]:
    return {
        "synchronize_before": "torch.cuda.synchronize()",
        "reset_peak": "torch.cuda.reset_peak_memory_stats()",
        "synchronize_after": "torch.cuda.synchronize()",
        "empty_cache_inside_interval": False,
        "quantities": ["allocated_memory", "reserved_allocator_memory", "physically_resident_packed_weight_bytes", "request_owned_on_demand_bytes"],
    }


def _identity_record(config: dict[str, Any], manifest: dict[str, Any], mode: dict[str, Any], router_hash: str | None) -> dict[str, Any]:
    artifact = config["identities"]["packed_artifact"]
    return {
        "model_repository": config["identities"]["model"]["repository"],
        "model_revision": MODEL_REVISION,
        "tokenizer_revision": MODEL_REVISION,
        "any_precision_revision": ANY_PRECISION_REVISION if mode["packed_artifact"] else None,
        "packed_checkpoint_sha256": artifact["sha256"] if mode["packed_artifact"] else None,
        "router_checkpoint_sha256": router_hash,
    }


def _environment(torch: Any, transformers: Any, device: str) -> dict[str, Any]:
    target = torch.device(device)
    gpu_model = torch.cuda.get_device_name(target)
    return {
        "device_index": target.index,
        "gpu_model": gpu_model,
        "driver": getattr(torch.cuda, "driver_version", None) or "unknown",
        "cuda_runtime": torch.version.cuda,
        "pytorch": torch.__version__,
        "transformers": transformers.__version__,
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "comparability": {
            "reference_device_index": 3,
            "reference_gpu_model": "NVIDIA GeForce RTX 3090",
            "identity_recorded": True,
            "compatible": target.index == 3 and gpu_model == "NVIDIA GeForce RTX 3090",
        },
    }


def _physical_residency_bytes(model: Any) -> int:
    """Count actual packed qweight and LUT buffers resident in the model."""

    total = 0
    for module in model.modules():
        if module.__class__.__name__ != "AnyPrecisionLinear":
            continue
        for name in ("qweight", "lut4", "lut8"):
            tensor = getattr(module, name, None)
            if tensor is None or tensor.device.type != "cuda":
                raise EvaluationRunnerError(f"packed residency evidence is not physically measurable: {name}")
            total += int(tensor.numel() * tensor.element_size())
    return total


def _hidden_copy_audit(model: Any, context: Any) -> dict[str, Any]:
    """Reuse S08's physical source/module audit for on-demand evidence."""

    modules = sum(module.__class__.__name__ == "AnyPrecisionLinear" for module in model.modules())
    sources = {} if context is None else context.sources
    all_qweights_cpu = bool(sources) and all(source.qweight.device.type == "cpu" for source in sources.values())
    all_luts_cpu = bool(sources) and all(source.lut4.device.type == "cpu" and source.lut8.device.type == "cpu" for source in sources.values())
    no_complete_copy = modules == 0 and all_qweights_cpu and all_luts_cpu
    return {
        "any_precision_module_count": modules,
        "source_count": len(sources),
        "all_source_qweights_cpu": all_qweights_cpu,
        "all_source_luts_cpu": all_luts_cpu,
        "no_complete_packed_gpu_copy": no_complete_copy,
    }


def _seed(torch: Any, seed: int) -> None:
    import random

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _fixed_tensor(torch: Any, request: dict[str, Any], device: str) -> Any:
    return torch.tensor(request["input_ids"], dtype=torch.long, device=device).unsqueeze(0)


def _routed_policy(student: Any) -> Callable[[int, str, Any], int]:
    from qaq.router.distillation import hard_route

    def policy(layer: int, unit_type: str, feature: Any) -> int:
        return int(hard_route(student.route(layer, unit_type, feature)))

    return policy


def _routed_forward(student: Any, request_id: str, input_ids: Any, prompt_length: int, device: str, *, use_cache: bool = False, past_key_values: Any = None, phase: str = "prefill", context: Any = None, state: Any = None) -> tuple[Any, Any, Any]:
    from qaq.model.manual import PrecisionTrace
    from qaq.model.request_state import QaqRequestState

    if state is None:
        state = QaqRequestState(request_id, prompt_length=prompt_length, layer_count=36)
    if context is None and any(module.__class__.__name__ == "_OnDemandRoutedPackedLinear" for module in student.base.modules()):
        context = student.base.create_on_demand_request(state)
    trace = PrecisionTrace()
    kwargs = {
        "input_ids": input_ids,
        "attention_mask": __import__("torch").ones_like(input_ids, dtype=__import__("torch").bool),
        "use_cache": use_cache,
        "request_state": state,
        "phase": phase,
        "routing_policy": _routed_policy(student),
        "trace": trace,
        "on_demand_context": context,
    }
    if past_key_values is not None:
        kwargs["past_key_values"] = past_key_values
    if phase == "prefill":
        kwargs["prompt_attention_mask"] = __import__("torch").ones((1, prompt_length), dtype=__import__("torch").bool, device=device)
    return student.base(**kwargs), state, context


def _generate_record(model: Any, mode_id: str, request: dict[str, Any], device: str, torch: Any, *, student: Any = None) -> tuple[dict[str, Any], dict[str, Any] | None, Any]:
    input_ids = _fixed_tensor(torch, request, device)
    if student is None:
        with torch.inference_mode():
            output = model.generate(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), do_sample=False, num_beams=1, max_new_tokens=8, temperature=None, use_cache=False, return_dict_in_generate=True, output_scores=True)
        sequence = output.sequences[0].detach().cpu().tolist()
        scores_finite = all(bool(torch.isfinite(score).all().item()) for score in output.scores)
        return ({"request_id": request["id"], "input_ids_sha256": request["input_ids_sha256"], "generated_token_ids": sequence[len(request["input_ids"]):], "output_digest": hashlib.sha256(json.dumps(sequence, separators=(",", ":")).encode()).hexdigest(), "logits_digest": _tensor_digest(list(output.scores)), "finite_value_check": scores_finite and bool(torch.isfinite(output.sequences).all().item()), "normal_termination": True}, None, None)
    output, state, context = _routed_forward(student, request["id"], input_ids, len(request["input_ids"]), device, use_cache=True)
    generated = []
    logits = []
    past = output.past_key_values
    route = _route_record(state, request["id"])
    for _ in range(8):
        logits.append(output.logits[:, -1, :].detach())
        token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(int(token.item()))
        output, _, _ = _routed_forward(student, request["id"], token, len(request["input_ids"]), device, use_cache=True, past_key_values=past, phase="decode", context=context, state=state)
        past = output.past_key_values
    finite = bool(torch.isfinite(output.logits).all().item())
    state.end_request()
    return ({"request_id": request["id"], "input_ids_sha256": request["input_ids_sha256"], "generated_token_ids": generated, "output_digest": hashlib.sha256(json.dumps(generated, separators=(",", ":")).encode()).hexdigest(), "logits_digest": _tensor_digest(logits), "finite_value_check": finite and all(bool(torch.isfinite(value).all().item()) for value in logits), "normal_termination": True}, route, context)


def _perplexity_adapter(model: Any, mode_id: str, device: str) -> Any:
    if mode_id not in ROUTED_MODE_IDS:
        return model
    class RoutedAdapter:
        def __call__(self, *, input_ids: Any, use_cache: bool = False) -> Any:
            output, state, _ = _routed_forward(model, f"s09-ppl-{id(input_ids)}", input_ids, int(input_ids.shape[1]), device, use_cache=use_cache)
            state.end_request()
            return output
    return RoutedAdapter()


def _load_mode(mode: dict[str, Any], config: dict[str, Any], device: str) -> tuple[Any, Any | None, dict[str, Any]]:
    from transformers import AutoTokenizer

    from qaq.evaluation.quality import load_full_precision_model
    from qaq.model.static import load_manifest, load_static_model

    manifest = load_manifest(ROOT / "docs/quantized_model_manifest.json")
    artifact = ROOT / config["identities"]["packed_artifact"]["relative_path"]
    snapshot = Path(os.environ.get("QAQ_MODEL_SNAPSHOT", f"~/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/{MODEL_REVISION}")).expanduser()
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot if mode["model_kind"] == "full_precision" else artifact), revision=MODEL_REVISION, local_files_only=True)
    if mode["id"] == EXPECTED_MODE_IDS[0]:
        return load_full_precision_model(snapshot, device), tokenizer, manifest
    if mode["id"] in (EXPECTED_MODE_IDS[1], EXPECTED_MODE_IDS[2]):
        model = load_static_model(artifact, device)
        from qaq.model.static import set_static_precision
        set_static_precision(model, int(mode["precision"][0]) if isinstance(mode["precision"], str) else int(mode["precision"]))
        return model, tokenizer, manifest
    from qaq.model.manual import load_on_demand_model
    from qaq.router.distillation import RouterCheckpointMetadata, load_router_checkpoint
    from qaq.router.soft_model import SoftRoutedQwen3ForCausalLM, load_soft_model
    if mode["loader"] == "resident":
        student = load_soft_model(artifact, device)
    else:
        student = SoftRoutedQwen3ForCausalLM(load_on_demand_model(artifact, device))
    checkpoint = Path(os.environ.get("QAQ_S07_ROUTER_CHECKPOINT", "~/.cache/qaq/s07b/final_router.pt")).expanduser()
    metadata = RouterCheckpointMetadata(model_repository=config["identities"]["model"]["repository"], model_revision=MODEL_REVISION, quantized_checkpoint_id=manifest["artifact"]["local_path"], quantized_checkpoint_hash=f"sha256:{config['identities']['packed_artifact']['sha256']}", any_precision_revision=ANY_PRECISION_REVISION, router_architecture={"feature_dim": int(student.feature_dim), "hidden_width": 128, "activation": "GELU", "normalization": "parameter-free RMS", "normalization_epsilon": 1e-6, "temperature": 1.0, "router_count": int(student.router_count)}, candidate_ordering=(4, 8), training_step=4, training_step_metadata={"seed": 1729, "format": "qaq-s07b-router-training-v1"})
    load_router_checkpoint(checkpoint, student.routers, metadata)
    student.to(device).eval()
    return student, tokenizer, manifest


def _expected_physical_bytes(context: Any, route_map: list[dict[str, Any]]) -> dict[str, int]:
    selected = {(item["layer"], item["unit_type"]): item["selected_bits"] for item in route_map}
    total = 0
    attention = 0
    ffn = 0
    for module_id, source in context.sources.items():
        layer = int(module_id.split(".")[2])
        unit = "attention" if ".self_attn." in module_id else "ffn"
        bits = selected[(layer, unit)]
        value = int(source.qweight[:bits].numel() * source.qweight.element_size() + getattr(source, f"lut{bits}").numel() * getattr(source, f"lut{bits}").element_size())
        total += value
        if unit == "attention":
            attention += value
        else:
            ffn += value
    return {"total": total, "attention": attention, "ffn": ffn}


def _measure_request(model: Any, student: Any | None, mode_id: str, request: dict[str, Any], device: str, torch: Any) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any]]:
    input_ids = _fixed_tensor(torch, request, device)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    before = {"allocated_before": int(torch.cuda.memory_allocated()), "reserved_before": int(torch.cuda.memory_reserved())}
    start = time.perf_counter()
    route = None
    context = None
    if student is None:
        with torch.inference_mode():
            output = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), use_cache=True)
        torch.cuda.synchronize()
        prefill = time.perf_counter() - start
        past = output.past_key_values
        generated = []
        decode_start = time.perf_counter()
        for _ in range(8):
            token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated.append(int(token.item()))
            with torch.inference_mode():
                output = model(input_ids=token, attention_mask=torch.ones_like(token), past_key_values=past, use_cache=True)
            past = output.past_key_values
        torch.cuda.synchronize()
        decode = time.perf_counter() - decode_start
    else:
        output, state, context = _routed_forward(student, request["id"], input_ids, len(request["input_ids"]), device, use_cache=True)
        route = _route_record(state, request["id"])
        torch.cuda.synchronize()
        prefill = time.perf_counter() - start
        past = output.past_key_values
        generated = []
        decode_start = time.perf_counter()
        for _ in range(8):
            token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated.append(int(token.item()))
            output, _, _ = _routed_forward(student, request["id"], token, len(request["input_ids"]), device, use_cache=True, past_key_values=past, phase="decode", context=context, state=state)
            past = output.past_key_values
        torch.cuda.synchronize()
        decode = time.perf_counter() - decode_start
    end_to_end = time.perf_counter() - start
    peak = {"peak_allocated": int(torch.cuda.max_memory_allocated()), "peak_reserved": int(torch.cuda.max_memory_reserved())}
    if student is not None and context is not None:
        retained_before = {"entries": context.retained_entry_count, "buffers": context.retained_gpu_buffer_count, "bytes": context.retained_packed_bytes}
        state.end_request()
        retained_after = {"entries": context.retained_entry_count, "buffers": context.retained_gpu_buffer_count, "bytes": context.retained_packed_bytes}
    else:
        retained_before = {"entries": 0, "buffers": 0, "bytes": 0}
        retained_after = retained_before
    after = {"allocated_after_cleanup": int(torch.cuda.memory_allocated()), "reserved_after_cleanup": int(torch.cuda.memory_reserved())}
    return (
        {
            "request_id": request["id"],
            "input_ids_sha256": request["input_ids_sha256"],
            "prefill_seconds": prefill,
            "decode_seconds": decode,
            "end_to_end_seconds": end_to_end,
            "generated_token_ids": generated,
            "finite_outputs": bool(torch.isfinite(output.logits).all().item()),
            **before,
            **peak,
            **after,
            "retained_before_cleanup": retained_before,
            "retained_after_cleanup": retained_after,
        },
        route,
        {
            "context": context,
            "actual_transfer_bytes": sum(record.transferred_bytes for record in context.records) if context is not None else 0,
            "expected_transfer_bytes": _expected_physical_bytes(context, route["route_map"])["total"] if context is not None and route is not None else 0,
            "actual_attention_bytes": sum(record.transferred_bytes for record in context.records if ".self_attn." in record.module_id) if context is not None else 0,
            "actual_ffn_bytes": sum(record.transferred_bytes for record in context.records if ".mlp." in record.module_id) if context is not None else 0,
            "hidden_copy_audit": _hidden_copy_audit(model, context) if mode_id == ON_DEMAND_MODE_ID else None,
        },
    )


def execute_mode(config_path: Path, mode_id: str, output_path: Path, device: str) -> dict[str, Any]:
    config, prompts, config_hash = load_protocol(config_path)
    modes = resolve_modes(config)
    mode = next((item for item in modes if item["id"] == mode_id), None)
    if mode is None:
        raise EvaluationRunnerError(f"unknown mode: {mode_id}")
    if not os.environ.get("VIRTUAL_ENV", "").startswith(str(Path.home() / ".venv")):
        raise EvaluationRunnerError("PAUSE: ~/.venv is not active")
    import torch
    import transformers

    from qaq.evaluation.quality import build_perplexity_windows, evaluate_perplexity
    from qaq.model.static import file_sha256, source_commit

    if not torch.cuda.is_available():
        raise EvaluationRunnerError("PAUSE: CUDA is unavailable")
    torch.cuda.set_device(torch.device(device))
    _seed(torch, int(config["seeds"]["global_reproducibility_seed"]))
    model, tokenizer, manifest = _load_mode(mode, config, device)
    routed = mode_id in ROUTED_MODE_IDS
    artifact_path = ROOT / config["identities"]["packed_artifact"]["relative_path"] / config["identities"]["packed_artifact"]["checkpoint_file"]
    if mode["packed_artifact"]:
        if source_commit() != config["identities"]["any_precision"]["manifest_commit"]:
            raise EvaluationRunnerError("REVISE: Any-Precision revision changed")
        if file_sha256(artifact_path) != config["identities"]["packed_artifact"]["sha256"]:
            raise EvaluationRunnerError("REVISE: packed checkpoint SHA-256 changed")
    router_path = Path(os.environ.get("QAQ_S07_ROUTER_CHECKPOINT", "~/.cache/qaq/s07b/final_router.pt")).expanduser()
    router_hash = file_sha256(router_path) if routed and router_path.is_file() else None
    if routed and router_hash != config["identities"]["router"]["sha256"]:
        raise EvaluationRunnerError("REVISE: router checkpoint SHA-256 changed")
    requests = fixed_requests(prompts)
    perplexity_args = frozen_perplexity_arguments(config)
    generation_args = frozen_generation_arguments(config)
    repeats = frozen_latency_repeats(config)
    windows, setup = build_perplexity_windows(tokenizer, sample_count=perplexity_args["sample_count"], stride=perplexity_args["stride"])
    perplexity = evaluate_perplexity(_perplexity_adapter(model, mode_id, device), windows, device)
    generation_records = []
    route_records = []
    for request in requests:
        record, route, _ = _generate_record(model, mode_id, request, device, torch, student=model if routed else None)
        generation_records.append(record)
        if route is not None:
            route_records.append(route)
    # One warm-up request is completely finished before the five measured repeats.
    _measure_request(model, model if routed else None, mode_id, requests[0], device, torch)
    raw_latency = []
    memory_records = []
    repeat_evidence: list[dict[str, Any]] = []
    transfer_totals = {key: 0 for key in ("first_use_bytes", "reuse_bytes", "prefill_bytes", "decode_bytes", "attention_bytes", "ffn_bytes", "total_transfer_bytes", "first_use_events", "reuse_events", "independently_expected_physical_bytes")}
    hidden_copy_audits = []
    for request in requests:
        measured_repeats = []
        routes = []
        for repeat in range(repeats):
            measured, route, details = _measure_request(model, model if routed else None, mode_id, request, device, torch)
            measured_repeats.append(measured)
            routes.append(route)
            raw_latency.append({"request_id": request["id"], "repeat": repeat, **{key: measured[key] for key in ("prefill_seconds", "decode_seconds", "end_to_end_seconds")}})
            memory_records.append({
                key: measured[key]
                for key in ("request_id", "allocated_before", "reserved_before", "peak_allocated", "peak_reserved", "allocated_after_cleanup", "reserved_after_cleanup")
            } | {
                "retained_before_cleanup": dict(measured["retained_before_cleanup"]),
                "retained_after_cleanup": dict(measured["retained_after_cleanup"]),
                "finite_outputs": measured["finite_outputs"],
            })
            if mode_id == ON_DEMAND_MODE_ID and details["context"] is not None:
                context = details["context"]
                records = context.records
                total = sum(record.transferred_bytes for record in records)
                expected = _expected_physical_bytes(context, route["route_map"])
                transfer_totals["total_transfer_bytes"] += total
                transfer_totals["first_use_bytes"] += sum(record.transferred_bytes for record in records if record.event == "first_use")
                transfer_totals["reuse_bytes"] += sum(record.transferred_bytes for record in records if record.event == "reuse")
                transfer_totals["prefill_bytes"] += total
                transfer_totals["attention_bytes"] += sum(record.transferred_bytes for record in records if ".self_attn." in record.module_id)
                transfer_totals["ffn_bytes"] += sum(record.transferred_bytes for record in records if ".mlp." in record.module_id)
                transfer_totals["first_use_events"] += sum(record.event == "first_use" for record in records)
                transfer_totals["reuse_events"] += sum(record.event == "reuse" for record in records)
                transfer_totals["independently_expected_physical_bytes"] += expected["total"]
                hidden_copy_audits.append(details["hidden_copy_audit"])
        input_matches = all(item.get("input_ids_sha256") == request["input_ids_sha256"] for item in measured_repeats)
        outputs_finite = all(item["finite_outputs"] for item in measured_repeats)
        generated = [item["generated_token_ids"] for item in measured_repeats]
        route_digests = [] if not routed else [runner["route_map_digest"] for runner in routes if runner is not None]
        repeat_evidence.append({
            "request_id": request["id"],
            "input_ids_sha256": request["input_ids_sha256"],
            "input_ids_identical": input_matches,
            "repeat_count": len(measured_repeats),
            "all_outputs_finite": outputs_finite,
            "generated_token_ids": generated,
            "generated_outputs_agree": len({json.dumps(value, separators=(",", ":")) for value in generated}) == 1,
            "route_map_digests": route_digests,
            "routed_hard_routes_agree": not routed or len(set(route_digests)) == 1,
        })
    route_payload = None
    if routed:
        route_payload = {"requests": route_records, "route_diversity": _route_diversity(route_records)}
    memory_physical_bytes = 0 if mode_id == ON_DEMAND_MODE_ID else _physical_residency_bytes(model)
    cleanup_records = [
        {
            "request_id": record["request_id"],
            "retained_entries_before_cleanup": record["retained_before_cleanup"]["entries"],
            "retained_buffers_before_cleanup": record["retained_before_cleanup"]["buffers"],
            "retained_bytes_before_cleanup": record["retained_before_cleanup"]["bytes"],
            "retained_entries_after_cleanup": record["retained_after_cleanup"]["entries"],
            "retained_buffers_after_cleanup": record["retained_after_cleanup"]["buffers"],
            "retained_bytes_after_cleanup": record["retained_after_cleanup"]["bytes"],
        }
        for record in memory_records
    ]
    median_seconds = {
        request["id"]: {
            field: median(record[f"{field}_seconds"] for record in raw_latency if record["request_id"] == request["id"])
            for field in ("prefill", "decode", "end_to_end")
        }
        for request in requests
    }
    physical_cleanup_bytes = max((record["retained_before_cleanup"]["bytes"] for record in memory_records), default=0)
    output = {
        "schema": RESULT_SCHEMA,
        "mode_id": mode_id,
        "protocol": {"config_sha256": config_hash, "frozen": True, "config": str(config_path.resolve())},
        "provenance": provenance(),
        "identities": _identity_record(config, manifest, mode, router_hash),
        "hardware": _environment(torch, transformers, device),
        "seed": config["seeds"]["global_reproducibility_seed"],
        "fixed_inputs": {"path": config["fixed_inputs"]["path"], "request_ids": [item["id"] for item in requests], "input_digests": {item["id"]: item["input_ids_sha256"] for item in requests}},
        "perplexity": {"setup": {**setup, "evaluator": config["perplexity"]["evaluator"], "sequence_length": config["perplexity"]["sequence_length"], "source_window_length": config["perplexity"]["source_window_length"], "stride": config["perplexity"]["stride"], "sample_count": config["perplexity"]["sample_count"], "evaluated_token_count": config["perplexity"]["evaluated_token_count"], "labels": config["perplexity"]["labels"], "dataset": config["perplexity"]["dataset"], "config": config["perplexity"]["config"], "revision": config["perplexity"]["revision"], "split": config["perplexity"]["split"], "tokenizer_revision": config["perplexity"]["tokenizer_revision"]}, **perplexity},
        "generation": {**generation_args, "records": generation_records},
        "memory": {"method": _memory_method(), "records": memory_records, "physically_resident_packed_weight_bytes": memory_physical_bytes, "request_owned_on_demand_bytes": physical_cleanup_bytes if mode_id == ON_DEMAND_MODE_ID else 0},
        "latency": {"warmup_requests": 1, "repeats_per_request": repeats, "outlier_removal": False, "subtract_transfer_time": False, "raw_records": raw_latency, "median_seconds": median_seconds},
        "deterministic_checks": {"all_required_outputs_finite": all(item["finite_value_check"] for item in generation_records) and all(item["all_outputs_finite"] for item in repeat_evidence), "fixed_inputs_identical": all(item["input_ids_identical"] for item in repeat_evidence), "repeat_evidence": repeat_evidence},
    }
    if route_payload is not None:
        output["routed"] = route_payload
    if mode_id == ON_DEMAND_MODE_ID:
        hidden_copy_audit = {
            **(hidden_copy_audits[-1] if hidden_copy_audits else {}),
            "all_repeats_passed": bool(hidden_copy_audits) and all(
                audit["no_complete_packed_gpu_copy"]
                and audit["all_source_qweights_cpu"]
                and audit["all_source_luts_cpu"]
                for audit in hidden_copy_audits
            ),
        }
        output["on_demand"] = {
            **transfer_totals,
            "actual_equals_expected": transfer_totals["total_transfer_bytes"] == transfer_totals["independently_expected_physical_bytes"],
            "cleanup_records": cleanup_records,
            "retained_entries_before_cleanup": max((item["retained_entries_before_cleanup"] for item in cleanup_records), default=0),
            "retained_buffers_before_cleanup": max((item["retained_buffers_before_cleanup"] for item in cleanup_records), default=0),
            "retained_entries_after_cleanup": max((item["retained_entries_after_cleanup"] for item in cleanup_records), default=0),
            "retained_buffers_after_cleanup": max((item["retained_buffers_after_cleanup"] for item in cleanup_records), default=0),
            "retained_bytes_after_cleanup": max((item["retained_bytes_after_cleanup"] for item in cleanup_records), default=0),
            "no_complete_packed_parent_on_gpu": hidden_copy_audit.get("no_complete_packed_gpu_copy") is True,
            "hidden_copy_audit": hidden_copy_audit,
        }
    # Cleanup happens only after all mode evidence is collected; the process then exits.
    model.cpu()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    return output


def _aggregate_pairwise(results: dict[str, dict[str, Any]], config: dict[str, Any]) -> list[str]:
    errors = []
    hardware_fields = ("device_index", "gpu_model", "driver", "cuda_runtime", "pytorch", "transformers", "python")
    hardware_records = [results[mode_id]["hardware"] for mode_id in EXPECTED_MODE_IDS]
    reference = tuple(hardware_records[0].get(field) for field in hardware_fields)
    if any(tuple(record.get(field) for field in hardware_fields) != reference for record in hardware_records[1:]):
        errors.append("hardware identities are not comparable across modes")
    if any(record.get("gpu_model") != config["hardware"]["required_gpu_model"] for record in hardware_records):
        errors.append("mixed or unsupported GPU models")
    static4 = results[EXPECTED_MODE_IDS[1]]["perplexity"]["perplexity"]
    static8 = results[EXPECTED_MODE_IDS[2]]["perplexity"]["perplexity"]
    resident = results[EXPECTED_MODE_IDS[3]]
    ondemand = results[ON_DEMAND_MODE_ID]
    if static8 > 1.10 * static4:
        errors.append("static 8-bit perplexity quality gate failed")
    if resident["perplexity"]["perplexity"] > 1.10 * static4:
        errors.append("routed resident perplexity quality gate failed")
    resident_routes = resident["routed"]["requests"]
    ondemand_routes = ondemand["routed"]["requests"]
    if [(item["request_id"], item["route_map"]) for item in resident_routes] != [(item["request_id"], item["route_map"]) for item in ondemand_routes]:
        errors.append("resident/on-demand route maps mismatch")
    resident_gen = {item["request_id"]: (item["generated_token_ids"], item["logits_digest"]) for item in resident["generation"]["records"]}
    ondemand_gen = {item["request_id"]: (item["generated_token_ids"], item["logits_digest"]) for item in ondemand["generation"]["records"]}
    if resident_gen != ondemand_gen:
        errors.append("resident/on-demand logits or generated outputs mismatch")
    payload = ondemand["on_demand"]
    if payload["total_transfer_bytes"] != payload["independently_expected_physical_bytes"] or not payload["actual_equals_expected"]:
        errors.append("on-demand transfer accounting mismatch")
    if payload["retained_entries_after_cleanup"] or payload["retained_buffers_after_cleanup"] or payload["retained_bytes_after_cleanup"] or not payload["no_complete_packed_parent_on_gpu"]:
        errors.append("on-demand cleanup or hidden-copy audit failed")
    return errors


def _persist_aggregation(results_dir: Path, payload: dict[str, Any]) -> dict[str, Any]:
    try:
        results_dir.mkdir(parents=True, exist_ok=True)
        (results_dir / "aggregation.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    except OSError as exc:
        return {"classification": "REVISE", "errors": [f"cannot persist aggregation.json: {exc}"], "results_dir": str(results_dir)}
    return payload


def aggregate(config_path: Path, results_dir: Path) -> dict[str, Any]:
    config, prompts, config_hash = load_protocol(config_path)
    try:
        from qaq.evaluation.protocol import ProtocolValidationError, validate_protocol

        validate_protocol(config_path, check_external=False, verify_hashes=False)
    except (KeyError, OSError, ProtocolValidationError, TypeError, ValueError) as exc:
        return _persist_aggregation(results_dir, {"classification": "REVISE", "errors": [f"frozen protocol validation failed: {exc}"], "results_dir": str(results_dir)})
    modes = resolve_modes(config)
    results: dict[str, dict[str, Any]] = {}
    missing = []
    for mode in modes:
        path = results_dir / f"{mode['id']}.json"
        if not path.is_file():
            missing.append(str(path))
            continue
        try:
            result = _json(path)
            validate_result(result, config, prompts, config_hash)
        except EvaluationRunnerError as exc:
            return _persist_aggregation(results_dir, {"classification": "REVISE", "errors": [str(exc)], "results_dir": str(results_dir)})
        results[mode["id"]] = result
    if missing:
        return _persist_aggregation(results_dir, {"classification": "PAUSE", "missing_results": missing, "results_dir": str(results_dir)})
    errors = _aggregate_pairwise(results, config)
    return _persist_aggregation(results_dir, {"classification": "REVISE" if errors else "CONTINUE", "errors": errors, "mode_ids": list(results), "results_dir": str(results_dir), "protocol_config_sha256": config_hash})


__all__ = [
    "DEFAULT_CONFIG",
    "DEFAULT_RESULTS",
    "EXPECTED_MODE_IDS",
    "RESULT_SCHEMA",
    "EvaluationRunnerError",
    "aggregate",
    "child_command",
    "execute_mode",
    "fixed_input_digest",
    "fixed_requests",
    "frozen_generation_arguments",
    "frozen_latency_repeats",
    "frozen_perplexity_arguments",
    "load_protocol",
    "plan",
    "resolve_modes",
    "validate_result",
]
