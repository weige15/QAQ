"""Deterministic, standard-library-only S11-D block-sensitivity contract.

The module builds and validates the already-defined 42-unit same-unit study,
validates complete future intervention evidence, enforces sequential 4-first /
6-fallback dispatch, persists validated JSON atomically without overwrite, and
aggregates only a complete compatible result set.  It never imports or invokes
model, CUDA, training, router, or lookahead runtime code.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from qaq.evaluation import s11d_route_diagnostic as diagnostic

ROOT = Path(__file__).resolve().parents[3]
DEFINITION_PATH = ROOT / "docs/results/s11d_route_policy_diagnostic.json"
EXPECTED_DEFINITION_SHA256 = "cbc8965e48e4b751d0497190d3ecbcbb996611a5dc6b38d8766e7a747763b064"
EXPECTED_S11D_PROTOCOL_SHA256 = diagnostic.EXPECTED_PROTOCOL_SHA256
RESULT_PARENT = ROOT / "docs/results/s11d_same_unit_block_sensitivity"
AGGREGATION_OUTPUT = RESULT_PARENT / "aggregation.json"
PLAN_SCHEMA = "qaq-s11d-same-unit-block-sensitivity-plan-v1"
RESULT_SCHEMA = "qaq-s11d-same-unit-block-sensitivity-unit-result-v1"
AGGREGATION_SCHEMA = "qaq-s11d-same-unit-block-sensitivity-aggregation-v1"
RESUME_SCHEMA = "qaq-s11d-same-unit-block-sensitivity-resume-v1"
SEEDS = diagnostic.SEEDS
REQUEST_IDS = (
    "validation-3",
    "validation-270",
    "validation-500",
    "validation-761",
    "validation-1000",
    "validation-1252",
    "validation-1500",
    "validation-1759",
    "validation-2000",
    "validation-2250",
    "validation-2500",
    "validation-2755",
)
PRECISION_ORDER = (4, 6)
QUALITY_FACTORS = {
    "per_seed_aggregate_kl_max_control_factor": 1.10,
    "per_seed_aggregate_mean_absolute_error_max_control_factor": 1.10,
    "per_request_kl_max_paired_control_factor": 1.25,
}
METRIC_KEYS = (
    "completion_only_temperature_2_masked_teacher_relative_kl",
    "full_logit_mean_absolute_teacher_error",
    "full_logit_maximum_absolute_teacher_error",
)
_HEX64 = re.compile(r"[0-9a-f]{64}")
_CUDA_DEVICE = re.compile(r"cuda:(0|[1-9][0-9]*)")
HARDWARE_KEYS = (
    "cuda_device",
    "device_index",
    "gpu_model",
    "driver_version",
    "cuda_runtime_version",
    "pytorch_version",
    "transformers_version",
    "python_version",
)
REQUIRED_GPU_MODEL = "NVIDIA GeForce RTX 3090"


class SensitivityError(ValueError):
    """A study definition, evidence, dispatch, persistence, or pairing defect."""


class MissingEvidence(SensitivityError):
    """Required complete external evidence is not yet available."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SensitivityError(message)


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SensitivityError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise SensitivityError(f"non-finite JSON value: {value}")


def _load_json(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(
            raw,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SensitivityError(f"cannot load strict JSON {path}: {exc}") from exc
    _require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value, raw


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def serialize(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=False, allow_nan=False) + "\n").encode()


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _unit_id(layer: int, unit_type: str) -> str:
    return f"layer-{layer:02d}__{unit_type}"


def _result_relpath(unit_id: str, precision: int) -> str:
    return f"docs/results/s11d_same_unit_block_sensitivity/{unit_id}/precision-{precision}.json"


def expected_result_path(unit_id: str, precision: int) -> Path:
    return RESULT_PARENT / unit_id / f"precision-{precision}.json"


def _forced_route_map(
    source: Sequence[Mapping[str, Any]], layer: int, unit_type: str, bits: int
) -> list[dict[str, Any]]:
    changed: list[dict[str, Any]] = []
    found = 0
    for route in source:
        item = dict(route)
        if item.get("target_layer") == layer and item.get("unit_type") == unit_type:
            item["selected_bits"] = bits
            found += 1
        changed.append(item)
    _require(found == 1, "source route map does not contain exactly one target unit")
    return changed


def _definition() -> tuple[dict[str, Any], str]:
    value, raw = _load_json(DEFINITION_PATH)
    digest = _sha256(raw)
    _require(digest == EXPECTED_DEFINITION_SHA256, "study definition SHA-256 drifted")
    _require(
        value.get("schema") == "qaq-s11d-route-policy-diagnostic-v1", "definition schema drifted"
    )
    study = value.get("proposed_same_unit_block_sensitivity")
    _require(isinstance(study, dict), "study definition is missing")
    _require(study.get("status") == "defined_not_executed", "study definition status drifted")
    candidates = study.get("candidate_units")
    _require(isinstance(candidates, list) and len(candidates) == 42, "candidate count drifted")
    rows, _ = diagnostic.load_canonical_rows()
    _require(
        candidates == diagnostic._future_study_candidates(rows),
        "candidate membership/order drifted",
    )
    return value, digest


def _source_contexts() -> list[dict[str, Any]]:
    contexts: list[dict[str, Any]] = []
    expected_units = [(layer, unit) for layer in range(36) for unit in diagnostic.UNITS]
    for seed in SEEDS:
        trial_id = f"seed-{seed}__same_unit_468_control__lambda-0"
        path = diagnostic.CANONICAL_PARENT / f"{trial_id}.json"
        trial, raw = _load_json(path)
        _require(
            _sha256(raw) == diagnostic.EXPECTED_TRIAL_SHA256[trial_id],
            f"source trial drifted: {trial_id}",
        )
        _require(trial.get("seed") == seed, f"source seed drifted: {trial_id}")
        _require(trial.get("routing_timing") == "same_unit", f"source timing drifted: {trial_id}")
        _require(trial.get("lambda_bit") == 0.0, f"source lambda drifted: {trial_id}")
        requests = trial.get("hard_validation")
        _require(isinstance(requests, list), f"source requests missing: {trial_id}")
        _require(
            tuple(item.get("request_id") for item in requests) == REQUEST_IDS,
            "request order drifted",
        )
        for request in requests:
            routes = request.get("route_map")
            _require(
                isinstance(routes, list) and len(routes) == 72, "source route coverage drifted"
            )
            _require(
                [(item.get("target_layer"), item.get("unit_type")) for item in routes]
                == expected_units,
                "source route order or uniqueness drifted",
            )
            _require(
                all(item.get("selected_bits") in (4, 6, 8) for item in routes),
                "source route precision drifted",
            )
            contexts.append(
                {
                    "context_id": f"seed-{seed}__{request['request_id']}",
                    "seed": seed,
                    "request_id": request["request_id"],
                    "source_trial_id": trial_id,
                    "source_trial_sha256": diagnostic.EXPECTED_TRIAL_SHA256[trial_id],
                    "input_digest": request["input_digest"],
                    "teacher_digest": request["teacher_digest"],
                    "source_route_map_sha256": _digest(routes),
                }
            )
    _require(
        len(contexts) == 36 and len({item["context_id"] for item in contexts}) == 36,
        "context coverage drifted",
    )
    return contexts


def _source_route_index() -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for seed in SEEDS:
        trial_id = f"seed-{seed}__same_unit_468_control__lambda-0"
        trial, _ = _load_json(diagnostic.CANONICAL_PARENT / f"{trial_id}.json")
        for request in trial["hard_validation"]:
            result[f"seed-{seed}__{request['request_id']}"] = request["route_map"]
    return result


def _plan_core() -> dict[str, Any]:
    definition, definition_digest = _definition()
    contexts = _source_contexts()
    candidates = definition["proposed_same_unit_block_sensitivity"]["candidate_units"]
    units = []
    for index, candidate in enumerate(candidates, start=1):
        unit_id = _unit_id(candidate["layer"], candidate["unit_type"])
        units.append(
            {
                "unit_index": index,
                "unit_id": unit_id,
                "layer": candidate["layer"],
                "unit_type": candidate["unit_type"],
                "region": candidate["region"],
                "observed_same_unit_cost_downgrades": candidate[
                    "observed_same_unit_cost_downgrades"
                ],
                "context_ids": [item["context_id"] for item in contexts],
                "precision_sequence": [4, 6],
                "precision_4_output": _result_relpath(unit_id, 4),
                "precision_6_output": _result_relpath(unit_id, 6),
                "precision_6_condition": "only_after_complete_valid_precision_4_failure",
                "control_bits": 8,
            }
        )
    study_identity = {
        "definition_sha256": definition_digest,
        "s11d_protocol_sha256": EXPECTED_S11D_PROTOCOL_SHA256,
        "source_trial_sha256": [item["source_trial_sha256"] for item in contexts[::12]],
        "candidate_units": candidates,
        "seed_order": list(SEEDS),
        "request_order": list(REQUEST_IDS),
        "precision_sequence": list(PRECISION_ORDER),
        "quality_factors": QUALITY_FACTORS,
    }
    return {
        "schema": PLAN_SCHEMA,
        "study_id": _digest(study_identity),
        "definition_path": "docs/results/s11d_route_policy_diagnostic.json",
        "definition_sha256": definition_digest,
        "source_protocol_sha256": EXPECTED_S11D_PROTOCOL_SHA256,
        "design": {
            "unit_count": 42,
            "one_block_at_a_time": True,
            "routing_timing": "same_unit",
            "route_context": "canonical_same_unit_lambda_0",
            "seed_contexts": list(SEEDS),
            "requests": list(REQUEST_IDS),
            "paired_contexts_per_intervention": 36,
            "control_bits": 8,
            "precision_sequence": [4, 6],
            "fallback_rule": "run precision 6 only when complete valid precision 4 evidence fails",
            "initial_intervention_count": 42,
            "maximum_intervention_count": 84,
            "immediate_repeat_count": 1,
        },
        "quality_contract": {
            "metrics": list(METRIC_KEYS),
            "factors": dict(QUALITY_FACTORS),
            "pass_scope": "all three seed contexts must independently pass every factor",
            "maximum_absolute_error_diagnostic_only": True,
            "finite_values_required": True,
            "repeat_identical_required": True,
            "route_maps_may_differ_only_at_target": True,
        },
        "source_contexts": contexts,
        "units": units,
        "evidence_contract": {
            "schema": RESULT_SCHEMA,
            "one_atomic_file_per_unit_precision": True,
            "paired_control_and_treatment_in_same_file": True,
            "required_context_count": 36,
            "required_seed_summaries": 3,
            "source_input_teacher_and_route_identities_required": True,
            "primary_and_immediate_repeat_digests_and_metrics_required": True,
            "complete_identity_hardware_route_finiteness_repeat_and_prohibited_work_audits_required": True,
            "hardware_software_fields": list(HARDWARE_KEYS),
            "required_gpu_model": REQUIRED_GPU_MODEL,
        },
        "persistence_contract": {
            "same_directory_temporary_file": True,
            "fsync_before_promotion": True,
            "atomic_no_overwrite_hard_link_promotion": True,
            "post_promotion_byte_and_sha256_verification": True,
            "existing_or_linked_destination_rejected": True,
            "interrupted_temporary_file_never_counts_as_complete": True,
        },
        "aggregation_contract": {
            "all_42_units_required": True,
            "precision_4_required_for_every_unit": True,
            "precision_6_required_exactly_when_precision_4_fails": True,
            "precision_6_forbidden_when_precision_4_passes": True,
            "mixed_study_identity_rejected": True,
            "cross_result_hardware_software_identity_required": True,
            "canonical_result_paths_required": True,
            "missing_duplicate_or_unexpected_evidence_rejected": True,
            "lowest_safe_precision_rule": "4 if precision 4 passes; otherwise 6 if precision 6 passes; otherwise 8",
        },
        "resumption_contract": {
            "non_mutating_scan": True,
            "absent_parent_means_all_precision_4_pending": True,
            "authoritative_unit_order": True,
            "one_next_action_per_unit": True,
            "complete_valid_existing_results_are_reused": True,
            "temporary_symlinked_malformed_or_unexpected_evidence_rejected": True,
            "mixed_study_or_execution_provenance_rejected": True,
        },
        "commands": {
            "resume_plan": [
                "python",
                "scripts/run_s11d_block_sensitivity.py",
                "--resume-plan",
            ],
            "precision_4": [
                "python",
                "scripts/run_s11d_block_sensitivity.py",
                "--validate-dispatch",
                "--unit",
                "<unit-id>",
                "--precision",
                "4",
                "--device",
                "<explicit-cuda-device>",
                "--output",
                "<exact-unit-precision-4-output>",
            ],
            "precision_6_fallback": [
                "python",
                "scripts/run_s11d_block_sensitivity.py",
                "--validate-dispatch",
                "--unit",
                "<unit-id>",
                "--precision",
                "6",
                "--device",
                "<explicit-cuda-device>",
                "--output",
                "<exact-unit-precision-6-output>",
            ],
            "aggregation": [
                "python",
                "scripts/run_s11d_block_sensitivity.py",
                "--aggregate",
                "--output",
                "docs/results/s11d_same_unit_block_sensitivity/aggregation.json",
            ],
        },
        "non_execution_audit": {
            "model_loading": False,
            "cuda_activity": False,
            "sensitivity_execution": False,
            "router_training": False,
            "lambda_retuning": False,
            "lookahead_work": False,
            "result_write_activity": False,
        },
    }


def build_plan() -> dict[str, Any]:
    plan = _plan_core()
    validate_plan(plan)
    return plan


def validate_plan(plan: Mapping[str, Any]) -> None:
    expected = _plan_core()
    _require(plan == expected, "plan differs from the authoritative deterministic study")
    units = plan["units"]
    _require(
        len(units) == 42 and len({item["unit_id"] for item in units}) == 42,
        "plan units are missing or duplicated",
    )
    _require([item["unit_index"] for item in units] == list(range(1, 43)), "unit ordering drifted")
    _require(len(plan["source_contexts"]) == 36, "plan context coverage drifted")
    _require(plan["design"]["seed_contexts"] == list(SEEDS), "seed contexts drifted")
    _require(plan["design"]["requests"] == list(REQUEST_IDS), "requests drifted")
    _require(plan["design"]["precision_sequence"] == [4, 6], "4-first/6-fallback design drifted")
    _require(plan["quality_contract"]["factors"] == QUALITY_FACTORS, "S11 quality factors drifted")


def _unit(plan: Mapping[str, Any], unit_id: str) -> Mapping[str, Any]:
    matches = [item for item in plan["units"] if item["unit_id"] == unit_id]
    _require(len(matches) == 1, f"unknown or duplicated unit: {unit_id!r}")
    return matches[0]


def _metrics(value: Any, label: str) -> dict[str, float]:
    _require(
        isinstance(value, dict) and tuple(value) == METRIC_KEYS, f"{label} metric fields drifted"
    )
    for key in METRIC_KEYS:
        item = value[key]
        _require(
            isinstance(item, (int, float)) and not isinstance(item, bool),
            f"{label} metric is not numeric: {key}",
        )
        _require(math.isfinite(item) and item >= 0.0, f"{label} metric is invalid: {key}")
    return value


def _expected_arm(
    route_map: Sequence[Mapping[str, Any]], layer: int, unit_type: str, bits: int
) -> str:
    return _digest(_forced_route_map(route_map, layer, unit_type, bits))


def _summaries(contexts: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    summaries = []
    for seed in SEEDS:
        selected = [item for item in contexts if item["seed"] == seed]
        _require(len(selected) == 12, f"seed context coverage incomplete: {seed}")
        control_kl = (
            sum(item["control"]["primary_metrics"][METRIC_KEYS[0]] for item in selected) / 12
        )
        treatment_kl = (
            sum(item["treatment"]["primary_metrics"][METRIC_KEYS[0]] for item in selected) / 12
        )
        control_mae = (
            sum(item["control"]["primary_metrics"][METRIC_KEYS[1]] for item in selected) / 12
        )
        treatment_mae = (
            sum(item["treatment"]["primary_metrics"][METRIC_KEYS[1]] for item in selected) / 12
        )
        per_request = [
            {
                "request_id": item["request_id"],
                "control_kl": item["control"]["primary_metrics"][METRIC_KEYS[0]],
                "treatment_kl": item["treatment"]["primary_metrics"][METRIC_KEYS[0]],
                "limit": QUALITY_FACTORS["per_request_kl_max_paired_control_factor"]
                * item["control"]["primary_metrics"][METRIC_KEYS[0]],
                "passed": item["treatment"]["primary_metrics"][METRIC_KEYS[0]]
                <= QUALITY_FACTORS["per_request_kl_max_paired_control_factor"]
                * item["control"]["primary_metrics"][METRIC_KEYS[0]],
            }
            for item in selected
        ]
        checks = {
            "aggregate_kl_passed": treatment_kl
            <= QUALITY_FACTORS["per_seed_aggregate_kl_max_control_factor"] * control_kl,
            "aggregate_mean_absolute_error_passed": treatment_mae
            <= QUALITY_FACTORS["per_seed_aggregate_mean_absolute_error_max_control_factor"]
            * control_mae,
            "each_request_kl_passed": all(item["passed"] for item in per_request),
            "all_values_finite": all(
                item[arm]["finite"] for item in selected for arm in ("control", "treatment")
            ),
            "all_repeats_identical": all(
                item[arm]["repeat_identical"]
                for item in selected
                for arm in ("control", "treatment")
            ),
            "all_route_pairs_isolate_target": all(
                item["route_pair_isolates_target"] for item in selected
            ),
        }
        summaries.append(
            {
                "seed": seed,
                "control_aggregate_kl": control_kl,
                "treatment_aggregate_kl": treatment_kl,
                "control_aggregate_mean_absolute_error": control_mae,
                "treatment_aggregate_mean_absolute_error": treatment_mae,
                "paired_requests": per_request,
                "checks": checks,
                "passed": all(checks.values()),
            }
        )
    return summaries


def validate_unit_result(result: Mapping[str, Any], plan: Mapping[str, Any] | None = None) -> None:
    plan = build_plan() if plan is None else plan
    validate_plan(plan)
    expected_keys = (
        "schema",
        "study_id",
        "unit_id",
        "target",
        "precision",
        "identities",
        "hardware",
        "paired_contexts",
        "seed_summaries",
        "passed",
        "audit",
    )
    _require(tuple(result) == expected_keys, "unit result fields/order drifted")
    _require(result["schema"] == RESULT_SCHEMA, "unit result schema drifted")
    _require(result["study_id"] == plan["study_id"], "unit result study identity drifted")
    spec = _unit(plan, result["unit_id"])
    _require(
        result["target"] == {"layer": spec["layer"], "unit_type": spec["unit_type"]},
        "unit target drifted",
    )
    precision = result["precision"]
    _require(
        precision in PRECISION_ORDER and not isinstance(precision, bool),
        "unit precision is not 4 or 6",
    )
    config, config_raw = _load_json(ROOT / "configs/lookahead_468_training.json")
    _require(_sha256(config_raw) == EXPECTED_S11D_PROTOCOL_SHA256, "source protocol bytes drifted")
    _require(
        result["identities"] == config["identities"], "model/tokenizer/artifact identities drifted"
    )
    hardware = result["hardware"]
    _require(
        isinstance(hardware, dict) and tuple(hardware) == HARDWARE_KEYS,
        "hardware/software identity fields drifted",
    )
    _require(
        isinstance(hardware["cuda_device"], str)
        and _CUDA_DEVICE.fullmatch(hardware["cuda_device"]) is not None,
        "hardware CUDA device invalid",
    )
    _require(
        isinstance(hardware["device_index"], int)
        and not isinstance(hardware["device_index"], bool)
        and hardware["device_index"] == int(hardware["cuda_device"].split(":")[1]),
        "hardware CUDA device index drifted",
    )
    for field in HARDWARE_KEYS[2:]:
        _require(
            isinstance(hardware[field], str) and bool(hardware[field]),
            f"hardware/software identity is missing: {field}",
        )
    _require(hardware["gpu_model"] == REQUIRED_GPU_MODEL, "comparable GPU identity drifted")
    contexts = result["paired_contexts"]
    _require(
        isinstance(contexts, list) and len(contexts) == 36, "paired context coverage must be 36"
    )
    _require(
        [item.get("context_id") for item in contexts] == spec["context_ids"],
        "paired context order/membership drifted",
    )
    plan_contexts = {item["context_id"]: item for item in plan["source_contexts"]}
    routes = _source_route_index()
    for item in contexts:
        expected = plan_contexts[item["context_id"]]
        _require(
            tuple(item)
            == (
                "context_id",
                "seed",
                "request_id",
                "source",
                "control",
                "treatment",
                "route_pair_isolates_target",
            ),
            "paired context fields drifted",
        )
        _require(
            (item["seed"], item["request_id"]) == (expected["seed"], expected["request_id"]),
            "paired context identity drifted",
        )
        _require(
            item["source"]
            == {
                key: expected[key]
                for key in (
                    "source_trial_id",
                    "source_trial_sha256",
                    "input_digest",
                    "teacher_digest",
                    "source_route_map_sha256",
                )
            },
            "paired source evidence drifted",
        )
        for arm, bits in (("control", 8), ("treatment", precision)):
            evidence = item[arm]
            _require(
                tuple(evidence)
                == (
                    "forced_bits",
                    "route_map_sha256",
                    "primary_logits_sha256",
                    "repeat_logits_sha256",
                    "primary_metrics",
                    "repeat_metrics",
                    "finite",
                    "repeat_identical",
                ),
                f"{arm} evidence fields drifted",
            )
            _require(evidence["forced_bits"] == bits, f"{arm} forced precision drifted")
            _require(
                evidence["route_map_sha256"]
                == _expected_arm(
                    routes[item["context_id"]], spec["layer"], spec["unit_type"], bits
                ),
                f"{arm} route map drifted",
            )
            _require(
                _HEX64.fullmatch(evidence["primary_logits_sha256"] or "") is not None,
                f"{arm} logits digest invalid",
            )
            _require(
                evidence["repeat_logits_sha256"] == evidence["primary_logits_sha256"],
                f"{arm} repeat logits differ",
            )
            _metrics(evidence["primary_metrics"], f"{arm} primary")
            _metrics(evidence["repeat_metrics"], f"{arm} repeat")
            _require(
                evidence["repeat_metrics"] == evidence["primary_metrics"],
                f"{arm} repeat metrics differ",
            )
            _require(
                evidence["finite"] is True and evidence["repeat_identical"] is True,
                f"{arm} finite/repeat audit failed",
            )
        _require(
            item["route_pair_isolates_target"] is True,
            "control/treatment routes do not isolate target",
        )
    summaries = _summaries(contexts)
    _require(result["seed_summaries"] == summaries, "seed summaries differ from recomputation")
    passed = all(item["passed"] for item in summaries)
    _require(result["passed"] is passed, "unit pass classification drifted")
    _require(
        result["audit"]
        == {
            "complete_context_count": 36,
            "same_unit_only": True,
            "lambda_retuning_observed": False,
            "router_training_observed": False,
            "lookahead_execution_observed": False,
            "one_target_block_only": True,
            "paired_forced_8_controls_complete": True,
            "immediate_repeats_complete": True,
            "all_values_finite": True,
            "complete": True,
        },
        "unit prohibited-work/completeness audit drifted",
    )


def validate_execution_request(
    *,
    unit_id: str,
    precision: int,
    device: str,
    output: Path,
    plan: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    plan = build_plan() if plan is None else plan
    validate_plan(plan)
    spec = _unit(plan, unit_id)
    _require(
        precision in PRECISION_ORDER and not isinstance(precision, bool), "precision must be 4 or 6"
    )
    _require(
        _CUDA_DEVICE.fullmatch(device or "") is not None,
        "execution requires an explicit cuda:<index> device",
    )
    expected = expected_result_path(unit_id, precision)
    _require(
        Path(output).resolve() == expected.resolve(),
        f"output must be the exact result path: {expected}",
    )
    parent = expected.parent
    if not parent.exists():
        raise MissingEvidence(f"result unit directory is absent: {parent}")
    _require(
        parent.is_dir() and not parent.is_symlink(),
        "result unit directory must be a real directory",
    )
    _require(not os.path.lexists(expected), "result already exists; overwrite is forbidden")
    if precision == 6:
        first = expected_result_path(unit_id, 4)
        if not first.is_file() or first.is_symlink():
            raise MissingEvidence("precision 6 requires complete valid precision 4 evidence")
        first_result, _ = _load_json(first)
        validate_unit_result(first_result, plan)
        _require(
            first_result["passed"] is False, "precision 6 is forbidden after precision 4 passes"
        )
    return spec


def persist_unit_result(
    result: dict[str, Any], destination: Path, plan: Mapping[str, Any] | None = None
) -> str:
    plan = build_plan() if plan is None else plan
    validate_unit_result(result, plan)
    expected = expected_result_path(result["unit_id"], result["precision"])
    if result["precision"] == 6:
        first = expected_result_path(result["unit_id"], 4)
        if not first.is_file() or first.is_symlink():
            raise MissingEvidence(
                "precision 6 persistence requires complete valid precision 4 evidence"
            )
        first_result, _ = _load_json(first)
        validate_unit_result(first_result, plan)
        _require(
            first_result["passed"] is False,
            "precision 6 persistence is forbidden after precision 4 passes",
        )
        for first_context, fallback_context in zip(
            first_result["paired_contexts"], result["paired_contexts"], strict=True
        ):
            _require(
                first_context["source"] == fallback_context["source"]
                and first_context["control"] == fallback_context["control"],
                "precision 6 persistence control pairing drifted",
            )
    _require(Path(destination).resolve() == expected.resolve(), f"destination must be {expected}")
    _require(
        expected.parent.is_dir() and not expected.parent.is_symlink(),
        "destination parent must be an existing real directory",
    )
    _require(not os.path.lexists(expected), "destination exists; overwrite is forbidden")
    payload = serialize(result)
    temporary: Path | None = None
    try:
        descriptor, name = tempfile.mkstemp(
            prefix=f".{expected.name}.", suffix=".tmp", dir=expected.parent
        )
        temporary = Path(name)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        reloaded, raw = _load_json(temporary)
        _require(raw == payload, "temporary result bytes changed")
        validate_unit_result(reloaded, plan)
        _require(not os.path.lexists(expected), "destination appeared before promotion")
        os.link(temporary, expected)
        promoted = expected.read_bytes()
        _require(promoted == payload, "promoted result bytes changed")
        digest = _sha256(promoted)
        _require(digest == _sha256(payload), "promoted result digest changed")
        temporary.unlink()
        temporary = None
        directory = os.open(expected.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return digest
    except FileExistsError as exc:
        raise SensitivityError("destination appeared during atomic no-overwrite promotion") from exc
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def persist_aggregation(
    aggregation: dict[str, Any],
    destination: Path,
    results: Sequence[Mapping[str, Any]],
    plan: Mapping[str, Any] | None = None,
) -> str:
    """Atomically persist only an independently recomputed complete aggregation."""

    plan = build_plan() if plan is None else plan
    expected_value = build_aggregation(results, plan)
    _require(aggregation == expected_value, "aggregation differs from independent recomputation")
    expected = AGGREGATION_OUTPUT
    _require(Path(destination).resolve() == expected.resolve(), f"destination must be {expected}")
    _require(
        expected.parent.is_dir() and not expected.parent.is_symlink(),
        "aggregation parent must be an existing real directory",
    )
    _require(
        not os.path.lexists(expected), "aggregation destination exists; overwrite is forbidden"
    )
    payload = serialize(aggregation)
    temporary: Path | None = None
    try:
        descriptor, name = tempfile.mkstemp(
            prefix=f".{expected.name}.", suffix=".tmp", dir=expected.parent
        )
        temporary = Path(name)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        reloaded, raw = _load_json(temporary)
        _require(raw == payload and reloaded == expected_value, "temporary aggregation changed")
        _require(not os.path.lexists(expected), "aggregation destination appeared before promotion")
        os.link(temporary, expected)
        promoted = expected.read_bytes()
        _require(promoted == payload, "promoted aggregation bytes changed")
        digest = _sha256(promoted)
        _require(digest == _sha256(payload), "promoted aggregation digest changed")
        temporary.unlink()
        temporary = None
        directory = os.open(expected.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return digest
    except FileExistsError as exc:
        raise SensitivityError("aggregation appeared during atomic no-overwrite promotion") from exc
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def _validate_shared_execution_provenance(results: Sequence[Mapping[str, Any]]) -> None:
    if not results:
        return
    reference = results[0]["hardware"]
    _require(
        all(result["hardware"] == reference for result in results[1:]),
        "cross-result hardware/software execution provenance drifted",
    )


def _scan_result_files(
    plan: Mapping[str, Any], *, require_all_unit_directories: bool
) -> tuple[dict[tuple[str, int], dict[str, Any]], dict[str, Any] | None]:
    """Strictly load canonical result paths without mutating result state."""

    if not os.path.lexists(RESULT_PARENT):
        if require_all_unit_directories:
            raise MissingEvidence(f"result parent is absent: {RESULT_PARENT}")
        return {}, None
    _require(
        RESULT_PARENT.is_dir() and not RESULT_PARENT.is_symlink(),
        f"result parent is not a real directory: {RESULT_PARENT}",
    )
    unit_ids = {item["unit_id"] for item in plan["units"]}
    aggregation_name = AGGREGATION_OUTPUT.name
    entries = {entry.name: entry for entry in RESULT_PARENT.iterdir()}
    unexpected = set(entries) - unit_ids - {aggregation_name}
    _require(not unexpected, f"unexpected result parent entries: {sorted(unexpected)}")
    present_units = set(entries) & unit_ids
    if require_all_unit_directories and present_units != unit_ids:
        missing = sorted(unit_ids - present_units)
        raise MissingEvidence(f"result unit directories are incomplete: {missing}")

    index: dict[tuple[str, int], dict[str, Any]] = {}
    for spec in plan["units"]:
        unit_id = spec["unit_id"]
        parent = entries.get(unit_id)
        if parent is None:
            continue
        _require(
            parent.is_dir() and not parent.is_symlink(), f"unsafe result unit directory: {parent}"
        )
        files = {entry.name: entry for entry in parent.iterdir()}
        allowed_files = {"precision-4.json": 4, "precision-6.json": 6}
        unexpected_files = set(files) - set(allowed_files)
        _require(
            not unexpected_files,
            f"unexpected or temporary result evidence in {parent}: {sorted(unexpected_files)}",
        )
        for filename, precision in allowed_files.items():
            path = files.get(filename)
            if path is None:
                continue
            _require(
                path.is_file() and not path.is_symlink(), f"unsafe result evidence path: {path}"
            )
            result, _ = _load_json(path)
            validate_unit_result(result, plan)
            _require(
                result["unit_id"] == unit_id, f"result unit does not match canonical path: {path}"
            )
            _require(
                result["precision"] == precision,
                f"result precision does not match canonical path: {path}",
            )
            key = (unit_id, precision)
            _require(key not in index, f"duplicate result evidence: {key}")
            index[key] = result

    _validate_shared_execution_provenance(list(index.values()))
    aggregation: dict[str, Any] | None = None
    aggregation_path = entries.get(aggregation_name)
    if aggregation_path is not None:
        _require(
            aggregation_path.resolve() == AGGREGATION_OUTPUT.resolve()
            and aggregation_path.is_file()
            and not aggregation_path.is_symlink(),
            f"unsafe aggregation evidence path: {aggregation_path}",
        )
        aggregation, _ = _load_json(aggregation_path)
    return index, aggregation


def build_resume_state(plan: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return the deterministic next action for every unit without writing files."""

    plan = build_plan() if plan is None else plan
    validate_plan(plan)
    index, existing_aggregation = _scan_result_files(plan, require_all_unit_directories=False)
    actions: list[dict[str, Any]] = []
    results = list(index.values())
    for spec in plan["units"]:
        unit_id = spec["unit_id"]
        first = index.get((unit_id, 4))
        fallback = index.get((unit_id, 6))
        if first is None:
            _require(fallback is None, f"precision 6 exists without precision 4: {unit_id}")
            actions.append(
                {
                    "unit_id": unit_id,
                    "next_action": "run_precision_4",
                    "existing_precisions": [],
                    "next_output": spec["precision_4_output"],
                    "lowest_safe_precision": None,
                }
            )
            continue
        if first["passed"]:
            _require(fallback is None, f"precision 6 exists after precision 4 pass: {unit_id}")
            actions.append(
                {
                    "unit_id": unit_id,
                    "next_action": "complete",
                    "existing_precisions": [4],
                    "next_output": None,
                    "lowest_safe_precision": 4,
                }
            )
            continue
        if fallback is None:
            actions.append(
                {
                    "unit_id": unit_id,
                    "next_action": "run_precision_6",
                    "existing_precisions": [4],
                    "next_output": spec["precision_6_output"],
                    "lowest_safe_precision": None,
                }
            )
            continue
        for left, right in zip(first["paired_contexts"], fallback["paired_contexts"], strict=True):
            _require(
                left["source"] == right["source"] and left["control"] == right["control"],
                f"fallback control pairing drifted: {unit_id}",
            )
        actions.append(
            {
                "unit_id": unit_id,
                "next_action": "complete",
                "existing_precisions": [4, 6],
                "next_output": None,
                "lowest_safe_precision": 6 if fallback["passed"] else 8,
            }
        )

    counts = {
        action: sum(item["next_action"] == action for item in actions)
        for action in ("run_precision_4", "run_precision_6", "complete")
    }
    aggregation_ready = counts["complete"] == 42
    if existing_aggregation is not None:
        _require(aggregation_ready, "aggregation exists before all units are complete")
        _require(
            existing_aggregation == build_aggregation(results, plan),
            "persisted aggregation differs from complete validated evidence",
        )
    return {
        "schema": RESUME_SCHEMA,
        "study_id": plan["study_id"],
        "result_parent": "docs/results/s11d_same_unit_block_sensitivity",
        "result_parent_present": os.path.lexists(RESULT_PARENT),
        "unit_actions": actions,
        "next_action_counts": counts,
        "complete_result_files": len(index),
        "aggregation_ready": aggregation_ready,
        "aggregation_present": existing_aggregation is not None,
        "non_mutating": True,
        "errors": [],
    }


def build_aggregation(
    results: Sequence[Mapping[str, Any]], plan: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    plan = build_plan() if plan is None else plan
    validate_plan(plan)
    index: dict[tuple[str, int], Mapping[str, Any]] = {}
    for result in results:
        validate_unit_result(result, plan)
        key = (result["unit_id"], result["precision"])
        _require(key not in index, f"duplicate result evidence: {key}")
        index[key] = result
    _validate_shared_execution_provenance(list(index.values()))
    classifications = []
    expected_keys: set[tuple[str, int]] = set()
    for spec in plan["units"]:
        unit_id = spec["unit_id"]
        first = index.get((unit_id, 4))
        if first is None:
            raise MissingEvidence(f"missing precision 4 result: {unit_id}")
        expected_keys.add((unit_id, 4))
        if first["passed"]:
            _require(
                (unit_id, 6) not in index,
                f"unexpected precision 6 result after precision 4 pass: {unit_id}",
            )
            selected = 4
        else:
            fallback = index.get((unit_id, 6))
            if fallback is None:
                raise MissingEvidence(f"missing required precision 6 fallback: {unit_id}")
            expected_keys.add((unit_id, 6))
            # The paired forced-8 control must be byte-equivalent across the two precision attempts.
            for left, right in zip(
                first["paired_contexts"], fallback["paired_contexts"], strict=True
            ):
                _require(
                    left["source"] == right["source"] and left["control"] == right["control"],
                    f"fallback control pairing drifted: {unit_id}",
                )
            selected = 6 if fallback["passed"] else 8
        classifications.append(
            {
                "unit_id": unit_id,
                "layer": spec["layer"],
                "unit_type": spec["unit_type"],
                "precision_4_passed": first["passed"],
                "precision_6_run": not first["passed"],
                "precision_6_passed": None if first["passed"] else index[(unit_id, 6)]["passed"],
                "lowest_safe_precision": selected,
            }
        )
    _require(
        set(index) == expected_keys,
        "aggregation contains unexpected, orphaned, or out-of-protocol evidence",
    )
    counts = {
        str(bits): sum(item["lowest_safe_precision"] == bits for item in classifications)
        for bits in (4, 6, 8)
    }
    return {
        "schema": AGGREGATION_SCHEMA,
        "study_id": plan["study_id"],
        "definition_sha256": plan["definition_sha256"],
        "unit_order": [item["unit_id"] for item in plan["units"]],
        "unit_classifications": classifications,
        "lowest_safe_precision_counts": counts,
        "evidence_files_consumed": len(index),
        "complete_unit_count": 42,
        "safeguards": {
            "all_results_independently_validated": True,
            "all_required_4_first_results_present": True,
            "6_fallback_boundary_enforced": True,
            "paired_forced_8_controls_equal_across_fallbacks": True,
            "single_study_identity": True,
            "single_hardware_software_execution_provenance": True,
            "canonical_result_paths_required_by_loader": True,
            "no_partial_duplicate_or_unexpected_evidence": True,
            "quality_factors": dict(QUALITY_FACTORS),
        },
        "complete": True,
        "errors": [],
    }


def load_results_for_aggregation(plan: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    plan = build_plan() if plan is None else plan
    validate_plan(plan)
    index, _ = _scan_result_files(plan, require_all_unit_directories=True)
    results = list(index.values())
    # This independently enforces every required/forbidden fallback before a caller writes.
    build_aggregation(results, plan)
    return results


__all__ = [
    "AGGREGATION_OUTPUT",
    "AGGREGATION_SCHEMA",
    "EXPECTED_DEFINITION_SHA256",
    "HARDWARE_KEYS",
    "METRIC_KEYS",
    "PLAN_SCHEMA",
    "PRECISION_ORDER",
    "QUALITY_FACTORS",
    "REQUEST_IDS",
    "REQUIRED_GPU_MODEL",
    "RESULT_PARENT",
    "RESULT_SCHEMA",
    "RESUME_SCHEMA",
    "SEEDS",
    "MissingEvidence",
    "SensitivityError",
    "build_aggregation",
    "build_plan",
    "build_resume_state",
    "expected_result_path",
    "load_results_for_aggregation",
    "persist_aggregation",
    "persist_unit_result",
    "serialize",
    "validate_execution_request",
    "validate_plan",
    "validate_unit_result",
]
