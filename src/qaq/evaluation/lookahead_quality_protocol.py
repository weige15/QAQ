"""Read-only structural validator for the frozen S11-B1 quality-pilot protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import sys
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "configs" / "lookahead_quality_pilot.json"
EXPECTED_CONFIG_SHA256 = "21a664424debe4892c3577c490158228dd5399bb4b425611db728070d23a5051"
EXPECTED_TOP_LEVEL_KEYS = (
    "schema",
    "stage",
    "protocol_status",
    "protocol_frozen_before_results",
    "scope",
    "modes",
    "identities",
    "fixed_inputs",
    "execution_contract",
    "quality_metrics",
    "route_comparison",
    "treatment_provenance",
    "determinism_audit",
    "freeze_audit",
    "interpretation",
    "planned_results",
    "result_schema_contracts",
    "prohibited_work",
    "validation",
)
EXPECTED_MODE_IDS = (
    "same_unit_control",
    "lookahead_attention_one_unit_treatment",
)
EXPECTED_TIMINGS = ("same_unit", "lookahead_attention_one_unit")
EXPECTED_REQUEST_IDS = ("validation-3", "validation-1000")
EXPECTED_OUTPUTS = (
    "docs/results/s11b_quality_pilot/same_unit_control.json",
    "docs/results/s11b_quality_pilot/lookahead_attention_one_unit_treatment.json",
    "docs/results/s11b_quality_pilot/aggregation.json",
)
FORBIDDEN_CONFIG_KEYS = {
    "training",
    "training_steps",
    "optimizer",
    "learning_rate",
    "scheduler",
    "checkpoint_output",
    "prefetch",
    "performance",
    "latency",
    "memory",
    "transfer",
    "throughput",
}


class ProtocolValidationError(ValueError):
    """The frozen protocol or one of its authoritative references is invalid."""


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ProtocolValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ProtocolValidationError(f"non-finite JSON constant is forbidden: {value}")


def _read_json_bytes(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_object_without_duplicates,
            parse_constant=_reject_constant,
        )
    except ProtocolValidationError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProtocolValidationError(f"cannot parse JSON {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ProtocolValidationError(f"JSON root must be an object: {path}")
    return payload, raw


def _read_json(path: Path) -> dict[str, Any]:
    return _read_json_bytes(path)[0]


def _get(payload: dict[str, Any], path: str) -> Any:
    value: Any = payload
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise ProtocolValidationError(f"missing required field: {path}")
        value = value[part]
    return value


def _expect(payload: dict[str, Any], path: str, expected: Any) -> None:
    actual = _get(payload, path)
    if actual != expected:
        raise ProtocolValidationError(f"field {path} must be {expected!r}; got {actual!r}")


def _assert_finite(value: Any, path: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ProtocolValidationError(f"non-finite number at {path}")
    if isinstance(value, dict):
        for key, child in value.items():
            _assert_finite(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_finite(child, f"{path}[{index}]")


def _reject_forbidden_fields(value: Any, path: str = "$") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in FORBIDDEN_CONFIG_KEYS:
                raise ProtocolValidationError(f"forbidden protocol field: {path}.{key}")
            _reject_forbidden_fields(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_forbidden_fields(child, f"{path}[{index}]")


def _token_digest(token_ids: list[int]) -> str:
    encoded = struct.pack("<" + "q" * len(token_ids), *token_ids)
    return hashlib.sha256(encoded).hexdigest()


def _route_digest(request_id: str, route_logs: list[dict[str, Any]]) -> str:
    by_key: dict[tuple[int, str], int] = {}
    for record in route_logs:
        if not isinstance(record, dict) or record.get("request_id") != request_id:
            continue
        key = (record.get("layer"), record.get("unit_type"))
        bit = record.get("hard_bit")
        if key in by_key:
            raise ProtocolValidationError(f"duplicate historical route key: {request_id} {key}")
        if (
            isinstance(key[0], bool)
            or not isinstance(key[0], int)
            or key[1] not in ("attention", "ffn")
            or isinstance(bit, bool)
            or bit not in (4, 8)
        ):
            raise ProtocolValidationError(f"malformed historical route: {request_id} {key}")
        by_key[key] = bit
    expected_keys = {(layer, unit) for layer in range(36) for unit in ("attention", "ffn")}
    if set(by_key) != expected_keys:
        raise ProtocolValidationError(
            f"historical route coverage is not 72 unique units: {request_id}"
        )
    records = [
        {
            "request_id": request_id,
            "target_layer": layer,
            "unit_type": unit,
            "selected_bits": by_key[(layer, unit)],
        }
        for layer in range(36)
        for unit in ("attention", "ffn")
    ]
    encoded = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _validate_modes(config: dict[str, Any]) -> None:
    modes = config.get("modes")
    if (
        not isinstance(modes, list)
        or len(modes) != 2
        or not all(isinstance(mode, dict) for mode in modes)
    ):
        raise ProtocolValidationError("modes must contain exactly two objects")
    if tuple(mode.get("id") for mode in modes) != EXPECTED_MODE_IDS:
        raise ProtocolValidationError(f"mode order must be exactly {EXPECTED_MODE_IDS}")
    if tuple(mode.get("routing_timing") for mode in modes) != EXPECTED_TIMINGS:
        raise ProtocolValidationError(f"routing timings must be exactly {EXPECTED_TIMINGS}")
    shared = [
        {key: value for key, value in mode.items() if key not in {"id", "routing_timing"}}
        for mode in modes
    ]
    if shared[0] != shared[1]:
        raise ProtocolValidationError("modes may differ only in id and routing_timing")
    expected_shared = {
        "execution": "resident_physically_packed_hard_4_8",
        "candidate_order": [4, 8],
        "packed_weight_state": "resident",
        "hard_query_level_routes": True,
        "training_or_retraining": False,
        "on_demand_loading": False,
        "generation_or_decode": False,
        "perplexity": False,
        "performance_or_resource_measurement": False,
    }
    if shared[0] != expected_shared:
        raise ProtocolValidationError("shared mode contract drifted")


def _validate_authoritative_sources(config: dict[str, Any], root: Path) -> None:
    baseline_protocol = _read_json(root / "configs" / "baseline_evaluation.json")
    router_training_config = _read_json(root / "configs" / "baseline_router_training.json")
    manifest = _read_json(root / "docs" / "quantized_model_manifest.json")
    router_training_result = _read_json(root / "docs" / "results" / "s07_router_training.json")
    prompts = _read_json(root / "configs" / "baseline_evaluation_prompts.json")

    baseline_model = _get(baseline_protocol, "identities.model")
    manifest_model = _get(manifest, "source_model")
    result_model = _get(router_training_result, "source_model")
    model = _get(config, "identities.model")
    tokenizer = _get(config, "identities.tokenizer")
    if model != {
        "repository": baseline_model.get("repository"),
        "revision": baseline_model.get("revision"),
    }:
        raise ProtocolValidationError("model identity disagrees with S09")
    if model != {
        "repository": manifest_model.get("repository"),
        "revision": manifest_model.get("revision"),
    } or model != {
        "repository": result_model.get("repository"),
        "revision": result_model.get("revision"),
    }:
        raise ProtocolValidationError("model identity disagrees with manifest or S07 result")
    if tokenizer != {
        "repository": baseline_model.get("tokenizer_repository"),
        "revision": baseline_model.get("tokenizer_revision"),
    } or tokenizer != {
        "repository": manifest_model.get("tokenizer_repository"),
        "revision": manifest_model.get("tokenizer_revision"),
    }:
        raise ProtocolValidationError("tokenizer identity disagrees with S09 or manifest")
    if (
        prompts.get("tokenizer", {}).get("repository") != tokenizer["repository"]
        or prompts.get("tokenizer", {}).get("revision") != tokenizer["revision"]
    ):
        raise ProtocolValidationError("fixed-input tokenizer identity drifted")

    packed = _get(config, "identities.packed_artifact")
    baseline_packed = _get(baseline_protocol, "identities.packed_artifact")
    expected_packed = {
        "manifest_path": baseline_packed.get("manifest_path"),
        "relative_path": baseline_packed.get("relative_path"),
        "checkpoint_file": baseline_packed.get("checkpoint_file"),
        "sha256": baseline_packed.get("sha256"),
    }
    if packed != expected_packed:
        raise ProtocolValidationError("packed identity disagrees with S09")
    if (
        manifest.get("artifact", {}).get("local_path") != packed["relative_path"]
        or manifest.get("artifact", {}).get("checkpoint_hashes", {}).get("pytorch_model.bin")
        != packed["sha256"]
        or result_model.get("packed_student_checkpoint") != packed["relative_path"]
        or result_model.get("packed_student_checkpoint_sha256") != packed["sha256"]
    ):
        raise ProtocolValidationError("packed identity disagrees with manifest or S07 result")

    any_precision = _get(config, "identities.any_precision")
    baseline_backend = _get(baseline_protocol, "identities.any_precision")
    if any_precision != {
        "submodule_path": baseline_backend.get("submodule_path"),
        "commit": baseline_backend.get("manifest_commit"),
    }:
        raise ProtocolValidationError("Any-Precision identity disagrees with S09")
    if (
        manifest.get("any_precision", {}).get("commit") != any_precision["commit"]
        or result_model.get("any_precision_revision") != any_precision["commit"]
    ):
        raise ProtocolValidationError("Any-Precision identity disagrees with manifest or S07")

    router = _get(config, "identities.router_checkpoint")
    baseline_router = _get(baseline_protocol, "identities.router")
    result_checkpoint = _get(router_training_result, "checkpoint")
    if router != {
        "path_env_override": baseline_router.get("checkpoint_path_env_override"),
        "recorded_external_path": result_checkpoint.get("external_path"),
        "sha256": baseline_router.get("sha256"),
        "historical_result_path": "docs/results/s07_router_training.json",
        "historical_result_format": router_training_result.get("format"),
        "candidate_order": result_checkpoint.get("metadata", {}).get("candidate_ordering"),
    }:
        raise ProtocolValidationError("router checkpoint identity disagrees with S09/S07")
    if (
        result_checkpoint.get("sha256") != router["sha256"]
        or router_training_result.get("checkpoint_roundtrip", {}).get("expected_checkpoint_sha256")
        != router["sha256"]
        or router_training_result.get("checkpoint_roundtrip", {}).get(
            "hard_routes_match_recorded_result"
        )
        is not True
    ):
        raise ProtocolValidationError("S07 checkpoint or historical hard-route evidence drifted")
    if _get(config, "identities.fixed_inputs_path") != _get(baseline_protocol, "fixed_inputs.path"):
        raise ProtocolValidationError("fixed-input path disagrees with S09")
    if _get(config, "quality_metrics.teacher_student_kl.temperature") != _get(
        router_training_config, "training.distillation_temperature"
    ):
        raise ProtocolValidationError("KL temperature disagrees with S07")

    requests = prompts.get("requests")
    if not isinstance(requests, list):
        raise ProtocolValidationError("S09 fixed inputs have no requests")
    by_id = {record.get("id"): record for record in requests if isinstance(record, dict)}
    frozen_requests = _get(config, "fixed_inputs.requests")
    if not isinstance(frozen_requests, list) or len(frozen_requests) != 2:
        raise ProtocolValidationError("protocol must freeze exactly two input records")
    router_training_validation = {
        record.get("example_id"): record
        for record in router_training_result.get("dataset_manifest", {}).get("validation", [])
        if isinstance(record, dict)
    }
    for request_id, frozen in zip(EXPECTED_REQUEST_IDS, frozen_requests, strict=True):
        source = by_id.get(request_id)
        historical = router_training_validation.get(request_id)
        if not isinstance(source, dict) or not isinstance(historical, dict):
            raise ProtocolValidationError(f"authoritative fixed input is missing: {request_id}")
        token_ids = source.get("full_input_ids")
        if (
            not isinstance(token_ids, list)
            or len(token_ids) != 64
            or any(isinstance(token, bool) or not isinstance(token, int) for token in token_ids)
        ):
            raise ProtocolValidationError(f"full_input_ids must contain 64 integers: {request_id}")
        digest = _token_digest(token_ids)
        expected = {
            "source_record_id": request_id,
            "token_count": 64,
            "token_digest_sha256": digest,
            "prompt_token_range": [0, 32],
            "completion_token_range": [32, 64],
            "causal_completion_loss_logit_range": [31, 63],
        }
        if frozen != expected:
            raise ProtocolValidationError(f"frozen full-input record drifted: {request_id}")
        if (
            source.get("full_input_ids_sha256") != digest
            or source.get("full_input_token_count") != 64
            or source.get("prompt_token_range") != [0, 32]
            or historical.get("input_ids_sha256") != digest
            or historical.get("prompt_token_range") != [0, 32]
            or historical.get("completion_token_range") != [32, 64]
        ):
            raise ProtocolValidationError(f"authoritative input digest/range drifted: {request_id}")

    route_logs = router_training_result.get("evaluation", {}).get("hard", {}).get("route_logs")
    if not isinstance(route_logs, list):
        raise ProtocolValidationError("historical S07 hard route logs are missing")
    frozen_digests = _get(
        config, "route_comparison.historical_same_unit_control.request_route_digests"
    )
    expected_digests = {
        request_id: _route_digest(request_id, route_logs) for request_id in EXPECTED_REQUEST_IDS
    }
    if frozen_digests != expected_digests:
        raise ProtocolValidationError("historical same-unit route digests drifted")


def _validate_exact_contract(config: dict[str, Any]) -> None:
    if tuple(config) != EXPECTED_TOP_LEVEL_KEYS:
        raise ProtocolValidationError("top-level fields/order drifted")
    _expect(config, "schema", "qaq-s11b-quality-pilot-v1")
    _expect(config, "stage", "S11-B1")
    _expect(config, "protocol_status", "FROZEN_BEFORE_RESULTS")
    _expect(config, "protocol_frozen_before_results", True)
    _validate_modes(config)

    assertions = {
        "fixed_inputs.source_field": "full_input_ids",
        "fixed_inputs.request_order": list(EXPECTED_REQUEST_IDS),
        "fixed_inputs.sequence_length": 64,
        "fixed_inputs.prompt_tokens": 32,
        "fixed_inputs.completion_tokens": 32,
        "fixed_inputs.batch_size": 1,
        "fixed_inputs.stable_order": True,
        "fixed_inputs.padding": "none",
        "fixed_inputs.runtime_tokenization": False,
        "fixed_inputs.dataset_access": False,
        "fixed_inputs.input_replacement": False,
        "execution_contract.seed": 1729,
        "execution_contract.fresh_child_processes_per_mode": 1,
        "execution_contract.repeats_within_fresh_child": 2,
        "execution_contract.identical_fresh_process_setup_across_modes": True,
        "execution_contract.explicit_cuda_device_required": True,
        "execution_contract.same_physical_gpu_identity_across_children": True,
        "execution_contract.full_teacher_forced_forward": True,
        "execution_contract.use_cache": False,
        "execution_contract.generation": False,
        "execution_contract.decode": False,
        "execution_contract.perplexity": False,
        "execution_contract.dataset_loading": False,
        "execution_contract.runtime_tokenization": False,
        "execution_contract.resident_hard_packed_execution": True,
        "execution_contract.on_demand_loader": False,
        "quality_metrics.teacher_student_kl.authoritative_operation": "qaq.router.distillation.masked_kl_distillation_loss",
        "quality_metrics.teacher_student_kl.temperature": 2.0,
        "quality_metrics.teacher_student_kl.completion_only": True,
        "quality_metrics.width_combined_quality_scalar": False,
        "quality_metrics.route_distance_is_quality_metric": False,
        "route_comparison.route_key": ["request_id", "target_layer", "unit_type"],
        "route_comparison.serialization_order": "layer-major: target_layer 0..35, attention then ffn",
        "route_comparison.routes_per_request": 72,
        "route_comparison.attention_routes_per_request": 36,
        "route_comparison.ffn_routes_per_request": 36,
        "route_comparison.allowed_selected_bits": [4, 8],
        "route_comparison.cross_mode_equality.ffn_equality_beyond_layer_0_required": False,
        "treatment_provenance.layer_0_attention": "same_unit",
        "treatment_provenance.lookahead_target_attention_layers": [1, 35],
        "treatment_provenance.lookahead_source_attention_layers": [0, 34],
        "treatment_provenance.source_point": "post_attention_pre_ffn",
        "treatment_provenance.ffn_timing": "same_layer",
        "treatment_provenance.candidate_order_preserved": [4, 8],
        "determinism_audit.repeats_per_mode": 2,
        "determinism_audit.identical_input_digests": True,
        "determinism_audit.bitwise_equal_logits": True,
        "determinism_audit.identical_72_unit_routes": True,
        "determinism_audit.identical_provenance": True,
        "determinism_audit.finite_logits_and_metrics": True,
        "freeze_audit.before_after_hash_equality_required": True,
        "freeze_audit.optimizer_absent": True,
        "freeze_audit.gradients_absent": True,
        "freeze_audit.state_changes_allowed": False,
        "interpretation.precedence": [
            "INVALID_EVIDENCE",
            "PAUSE",
            "ADVANCE_TO_BROADER_QUALITY_CHECK",
            "CHECKPOINT_REUSE_DEGRADES",
        ],
        "interpretation.quality_thresholds.treatment_aggregate_kl_max_control_factor": 1.10,
        "interpretation.quality_thresholds.treatment_each_request_kl_max_paired_control_factor": 1.25,
        "interpretation.quality_thresholds.treatment_aggregate_mean_absolute_logit_error_max_control_factor": 1.10,
        "interpretation.quality_thresholds.implementation_choices_not_paper_facts": True,
        "planned_results.per_mode_schema": "qaq-s11b-quality-pilot-mode-result-v1",
        "planned_results.aggregation_schema": "qaq-s11b-quality-pilot-aggregation-v1",
        "planned_results.create_in_s11b1": False,
        "planned_results.overwrite_allowed": False,
        "validation.read_only": True,
        "validation.standard_library_only": True,
        "validation.loads_model_checkpoint_or_artifact_bytes": False,
        "validation.imports_ml_runtime": False,
        "validation.refuses_existing_planned_results": True,
    }
    for path, expected in assertions.items():
        _expect(config, path, expected)

    required_layer_zero = _get(
        config, "route_comparison.cross_mode_equality.required_keys_per_request"
    )
    if required_layer_zero != [
        {"target_layer": 0, "unit_type": "attention"},
        {"target_layer": 0, "unit_type": "ffn"},
    ]:
        raise ProtocolValidationError("cross-mode equality must cover only layer-0 attention/FFN")
    if _get(config, "result_schema_contracts.per_mode.schema") != _get(
        config, "planned_results.per_mode_schema"
    ) or _get(config, "result_schema_contracts.aggregation.schema") != _get(
        config, "planned_results.aggregation_schema"
    ):
        raise ProtocolValidationError("planned result schema versions disagree")
    classifications = _get(config, "result_schema_contracts.aggregation.classification_values")
    if classifications != _get(config, "interpretation.precedence"):
        raise ProtocolValidationError("classification names/order disagree")


def _validate_output_paths(config: dict[str, Any], root: Path) -> None:
    mode_outputs = _get(config, "planned_results.mode_outputs")
    if not isinstance(mode_outputs, list):
        raise ProtocolValidationError("planned mode outputs must be a list")
    paths = [record.get("path") for record in mode_outputs if isinstance(record, dict)]
    paths.append(_get(config, "planned_results.aggregation_output"))
    if tuple(paths) != EXPECTED_OUTPUTS or len(set(paths)) != 3:
        raise ProtocolValidationError("planned result paths must be exact and unique")
    if [record.get("mode_id") for record in mode_outputs] != list(EXPECTED_MODE_IDS):
        raise ProtocolValidationError("planned mode outputs are not in frozen mode order")
    for value in paths:
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or str(path) != value:
            raise ProtocolValidationError(f"planned result path is not project-relative: {value}")
        if os.path.lexists(root / path):
            raise ProtocolValidationError(f"planned result already exists: {value}")


def validate_protocol(config_path: Path = DEFAULT_CONFIG, *, root: Path = ROOT) -> dict[str, Any]:
    """Validate the frozen description without loading any runtime or result bytes."""

    config_path = Path(config_path)
    root = Path(root)
    config, raw = _read_json_bytes(config_path)
    _assert_finite(config)
    digest = hashlib.sha256(raw).hexdigest()
    if digest != EXPECTED_CONFIG_SHA256:
        raise ProtocolValidationError(
            f"frozen config SHA-256 mismatch: expected {EXPECTED_CONFIG_SHA256}, got {digest}"
        )
    _reject_forbidden_fields(config)
    _validate_exact_contract(config)
    _validate_authoritative_sources(config, root)
    _validate_output_paths(config, root)
    return {
        "schema": config["schema"],
        "config_sha256": digest,
        "mode_ids": list(EXPECTED_MODE_IDS),
        "request_ids": list(EXPECTED_REQUEST_IDS),
        "planned_result_paths": list(EXPECTED_OUTPUTS),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    try:
        summary = validate_protocol(args.config)
    except (ProtocolValidationError, KeyError, TypeError, ValueError) as error:
        print(f"S11-B1 protocol invalid: {error}", file=sys.stderr)
        return 1
    print(
        "S11-B1 protocol valid: "
        f"{summary['schema']} (2 modes, 2 inputs; no planned results present)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
