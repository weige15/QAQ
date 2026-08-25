"""Production runtime for the frozen paired lookahead 4/6/8 trial contract.

The standard-library dispatcher imports this module only after it has validated
an exact frozen trial request.  The scheduler is object-based so structural
tests can exercise every audit with tiny deterministic Torch objects without
loading model, dataset, Any-Precision, or CUDA resources.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch

from qaq.evaluation import lookahead_468_executor as contract

REQUEST_METRICS = (
    "completion_only_temperature_2_masked_teacher_relative_kl",
    "full_logit_mean_absolute_teacher_error",
    "full_logit_maximum_absolute_teacher_error",
    "soft_expected_width",
    "hard_mean_selected_width",
    "hard_4_6_8_counts_and_fractions",
    "complete_72_unit_layer_major_route_map_attention_then_ffn",
    "attention_ffn_overall_selected_width",
    "paired_route_transitions",
)


class RuntimeFailure(contract.ProtocolError):
    """A fail-closed runtime defect classified by the frozen outcome boundary."""


@dataclass(frozen=True)
class TrialOutcome:
    classification: str
    errors: tuple[str, ...]
    result: dict[str, Any] | None
    output_path: str | None
    written: bool


@dataclass(frozen=True)
class AggregationOutcome:
    classification: str
    errors: tuple[str, ...]
    result: dict[str, Any] | None
    output_path: str | None
    written: bool


class PairedLookaheadRuntime(Protocol):
    """Minimal model/data boundary owned by the protocol-locked scheduler."""

    enforce_frozen_model_contract: bool
    train_examples: Sequence[Any]

    def prepare(self, config: Mapping[str, Any], device: str) -> None: ...

    def build_seed_model(self, seed: int, device: str) -> Any: ...

    def router_state(self, model: Any) -> Mapping[str, torch.Tensor]: ...

    def restore_router_state(self, model: Any, state: Mapping[str, torch.Tensor]) -> None: ...

    def initial_identity_matches(self, seed: int, digest: str) -> bool: ...

    def frozen_snapshot(self, model: Any) -> Mapping[str, Any]: ...

    def frozen_audit(self, model: Any, before: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def loss(
        self,
        model: Any,
        example: Any,
        spec: Mapping[str, Any],
        step: int,
        device: str,
    ) -> Mapping[str, Any]: ...

    def finalize_training_request(self, evidence: Mapping[str, Any]) -> Mapping[str, bool]: ...

    def validate(
        self, model: Any, spec: Mapping[str, Any], mode: str, device: str
    ) -> Sequence[Mapping[str, Any]]: ...

    def close_model(self, model: Any) -> None: ...


def _require(condition: bool, message: str, *, outcome: str = "REVISE") -> None:
    if not condition:
        raise RuntimeFailure(outcome, message)


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _state_bytes(state: Mapping[str, torch.Tensor]) -> bytes:
    payload = bytearray()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        payload.extend(name.encode())
        payload.extend(b"\0")
        payload.extend(str(value.dtype).encode())
        payload.extend(b"\0")
        payload.extend(repr(tuple(value.shape)).encode())
        payload.extend(b"\0")
        payload.extend(value.view(torch.uint8).numpy().tobytes())
        payload.extend(b"\n")
    return bytes(payload)


def _state_hash(state: Mapping[str, torch.Tensor]) -> str:
    return hashlib.sha256(_state_bytes(state)).hexdigest()


def _clone_state(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().contiguous().clone() for name, value in state.items()}


def _router_items(model: Any) -> list[tuple[str, torch.nn.Parameter]]:
    return [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if name.startswith("routers.")
    ]


def _build_optimizer(model: Any, config: Mapping[str, Any]) -> Any:
    items = _router_items(model)
    _require(bool(items), "model has no router parameters")
    return torch.optim.AdamW(
        [parameter for _, parameter in items],
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
        betas=tuple(float(value) for value in config["betas"]),
        eps=float(config["eps"]),
        amsgrad=bool(config["amsgrad"]),
    )


def _audit_optimizer(model: Any, optimizer: Any, serial: int) -> dict[str, Any]:
    items = _router_items(model)
    expected_ids = {id(parameter) for _, parameter in items}
    _require(isinstance(optimizer, torch.optim.AdamW), "optimizer is not AdamW")
    actual = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    actual_ids = [id(parameter) for parameter in actual]
    audit = {
        "construction_serial": serial,
        "class": type(optimizer).__name__,
        "fresh_state_entries": len(optimizer.state),
        "router_parameter_names": [name for name, _ in items],
        "router_parameter_count": len(items),
        "router_scalar_count": sum(parameter.numel() for _, parameter in items),
        "identity_membership": len(actual_ids) == len(set(actual_ids))
        and set(actual_ids) == expected_ids,
        "router_only": all(
            name.startswith("routers.")
            for name, parameter in model.named_parameters()
            if id(parameter) in set(actual_ids)
        ),
    }
    _require(not optimizer.state, "fresh AdamW has inherited state")
    _require(audit["identity_membership"] is True, "optimizer membership is not exact")
    _require(audit["router_only"] is True, "optimizer contains non-router parameters")
    return audit


def _optimizer_step(optimizer: Any, step: int, completed: list[int]) -> None:
    optimizer.step()
    completed.append(step)


def _perform_optimizer_update(
    runtime: PairedLookaheadRuntime,
    optimizer: Any,
    evidence: Mapping[str, Any],
    router_parameters: Sequence[torch.nn.Parameter],
    *,
    request_id: str,
    routing_timing: str,
    step: int,
    completed: list[int],
) -> tuple[torch.Tensor, list[torch.Tensor], list[dict[str, Any]], dict[str, bool]]:
    result: tuple[torch.Tensor, list[torch.Tensor], list[dict[str, Any]]] | None = None
    try:
        total = evidence.get("total_loss")
        _require(isinstance(total, torch.Tensor) and total.ndim == 0, "runtime loss is missing")
        _require(bool(torch.isfinite(total).item()), "training loss is non-finite")
        total.backward()
        gradients = [parameter.grad for parameter in router_parameters]
        _require(
            all(value is not None for value in gradients), "router gradient coverage is incomplete"
        )
        typed_gradients = [value for value in gradients if value is not None]
        _require(
            all(bool(torch.isfinite(value).all().item()) for value in typed_gradients),
            "router gradient is non-finite",
        )
        _require(
            any(bool(torch.count_nonzero(value).item()) for value in typed_gradients),
            "router soft-gradient path is zero",
        )
        provenance = _validate_provenance(request_id, routing_timing, evidence.get("provenance"))
        _require(
            evidence.get("request_state_complete") is True,
            "request-state coverage or lookahead consumption failed",
        )
        _optimizer_step(optimizer, step, completed)
        result = total, typed_gradients, provenance
    finally:
        request_state_audit = dict(runtime.finalize_training_request(evidence))
        _require(
            request_state_audit == {"complete": True, "cleanup": True},
            "request-state coverage, lookahead consumption, or cleanup failed",
        )
    if result is None:  # pragma: no cover - exceptions propagate through finally
        raise RuntimeError("optimizer update did not produce evidence")
    return (*result, request_state_audit)


def _example_ids(examples: Sequence[Any]) -> tuple[str, ...]:
    ids: list[str] = []
    for example in examples:
        value = getattr(example, "example_id", None)
        _require(isinstance(value, str) and bool(value), "runtime example identity is missing")
        ids.append(value)
    return tuple(ids)


def _expected_provenance(
    request_id: str, routing_timing: str, layer: int, unit_type: str
) -> dict[str, Any]:
    if routing_timing == "lookahead_attention_one_unit" and unit_type == "attention" and layer > 0:
        source_layer = layer - 1
        source_point = "post_attention_pre_ffn"
    else:
        source_layer = layer
        source_point = (
            "same_unit_pre_attention" if unit_type == "attention" else "post_attention_pre_ffn"
        )
    return {
        "request_id": request_id,
        "source_layer": source_layer,
        "target_layer": layer,
        "target_unit_type": unit_type,
        "source_point": source_point,
        "routing_timing": routing_timing,
        "candidate_order": [4, 6, 8],
    }


def _validate_provenance(
    request_id: str, routing_timing: str, records: Any
) -> list[dict[str, Any]]:
    expected = [
        _expected_provenance(request_id, routing_timing, layer, unit)
        for layer in range(36)
        for unit in ("attention", "ffn")
    ]
    _require(records == expected, "target ownership or routing provenance drifted")
    return [dict(item) for item in records]


def _validate_route_map(request_id: str, value: Any) -> list[dict[str, Any]]:
    _require(isinstance(value, list) and len(value) == 72, "route map must contain 72 units")
    expected_keys = [(layer, unit) for layer in range(36) for unit in ("attention", "ffn")]
    actual_keys = [
        (item.get("target_layer"), item.get("unit_type"))
        for item in value
        if isinstance(item, Mapping)
    ]
    _require(actual_keys == expected_keys, "route map order, ownership, or coverage drifted")
    _require(
        all(item.get("selected_bits") in (4, 6, 8) for item in value),
        "route map contains a non-frozen precision",
    )
    _require(all(item.get("request_id") == request_id for item in value), "route request drifted")
    return [dict(item) for item in value]


def _validate_request(
    item: Mapping[str, Any], *, request_id: str, routing_timing: str, mode: str
) -> dict[str, Any]:
    _require(item.get("request_id") == request_id, "validation request order drifted")
    _require(
        item.get("request_state_audit") == {"complete": True, "cleanup": True},
        "request-state coverage, lookahead consumption, or cleanup failed",
    )
    for name in (
        "completion_only_temperature_2_masked_teacher_relative_kl",
        "full_logit_mean_absolute_teacher_error",
        "full_logit_maximum_absolute_teacher_error",
    ):
        _require(_finite(item.get(name)), f"required metric is missing or non-finite: {name}")
    for name in ("input_digest", "teacher_digest", "logits_digest"):
        _require(
            isinstance(item.get(name), str) and len(item[name]) == 64,
            f"required digest is invalid: {name}",
        )
    normalized = dict(item)
    if mode == "soft":
        _require(_finite(item.get("soft_expected_width")), "soft expected width is invalid")
        _require(4.0 <= float(item["soft_expected_width"]) <= 8.0, "soft width is outside 4/6/8")
    else:
        routes = _validate_route_map(request_id, item.get("route_map"))
        provenance = _validate_provenance(request_id, routing_timing, item.get("provenance"))
        bits = [int(route["selected_bits"]) for route in routes]
        counts = {str(bit): bits.count(bit) for bit in (4, 6, 8)}
        fractions = {key: count / 72 for key, count in counts.items()}
        mean_width = sum(bits) / 72
        _require(item.get("hard_counts") == counts, "hard 4/6/8 counts drifted")
        _require(item.get("hard_fractions") == fractions, "hard 4/6/8 fractions drifted")
        _require(item.get("hard_mean_selected_width") == mean_width, "hard width drifted")
        _require(
            item.get("attention_mean_selected_width")
            == sum(bits[index] for index in range(0, 72, 2)) / 36,
            "attention width summary drifted",
        )
        _require(
            item.get("ffn_mean_selected_width")
            == sum(bits[index] for index in range(1, 72, 2)) / 36,
            "FFN width summary drifted",
        )
        _require(item.get("overall_mean_selected_width") == mean_width, "overall width drifted")
        normalized["route_map"] = routes
        normalized["provenance"] = provenance
    return normalized


def _validate_mode(
    records: Sequence[Mapping[str, Any]], *, routing_timing: str, mode: str
) -> list[dict[str, Any]]:
    _require(
        [item.get("request_id") for item in records] == list(contract.VALIDATION_IDS),
        "validation requests are incomplete or reordered",
    )
    return [
        _validate_request(item, request_id=request_id, routing_timing=routing_timing, mode=mode)
        for request_id, item in zip(contract.VALIDATION_IDS, records, strict=True)
    ]


def _aggregate(records: Sequence[Mapping[str, Any]], mode: str) -> dict[str, Any]:
    result = {
        "completion_only_temperature_2_masked_teacher_relative_kl": sum(
            float(item["completion_only_temperature_2_masked_teacher_relative_kl"])
            for item in records
        )
        / 12,
        "full_logit_mean_absolute_teacher_error": sum(
            float(item["full_logit_mean_absolute_teacher_error"]) for item in records
        )
        / 12,
        "full_logit_maximum_absolute_teacher_error": max(
            float(item["full_logit_maximum_absolute_teacher_error"]) for item in records
        ),
    }
    width = "soft_expected_width" if mode == "soft" else "hard_mean_selected_width"
    result[width] = sum(float(item[width]) for item in records) / 12
    return result


def _validate_trial_result(result: Mapping[str, Any], spec: Mapping[str, Any]) -> None:
    _require(result.get("trial_id") == spec["trial_id"], "persisted trial identity drifted")
    _require(result.get("trial_index") == spec["trial_index"], "persisted trial order drifted")
    _require(result.get("optimizer_steps_completed") == 24, "persisted update count drifted")
    history = result.get("training_history")
    _require(
        isinstance(history, list)
        and [item.get("step") for item in history] == list(range(1, 25))
        and [item.get("example_id") for item in history] == list(contract.TRAIN_IDS)
        and all(
            item.get("request_state_audit") == {"complete": True, "cleanup": True}
            for item in history
        ),
        "persisted training order or request-state audit drifted",
    )
    _require(
        result.get("training_examples_seen") == list(contract.TRAIN_IDS),
        "persisted data order drifted",
    )
    _require(result.get("route_decisions") == 864, "persisted route coverage drifted")
    _require(
        tuple(result.get("required_metrics", ())) == REQUEST_METRICS, "metric contract drifted"
    )
    references = result.get("paired_route_transition_references")
    _require(
        isinstance(references, Mapping)
        and references.get("zero_cost_reference_trial_id") == spec["zero_cost_reference_trial_id"]
        and references.get("same_cost_timing_pair_trial_id")
        == spec["same_cost_timing_pair_trial_id"],
        "paired transition references drifted",
    )
    _require(result.get("audits", {}).get("passed") is True, "persisted audits did not pass")
    _require(len(result.get("soft_validation", [])) == 12, "persisted soft evidence is incomplete")
    _require(len(result.get("hard_validation", [])) == 12, "persisted hard evidence is incomplete")
    _require(
        result.get("immediate_hard_repeat", {}).get("identical") is True, "persisted repeat failed"
    )
    soft = _validate_mode(
        result.get("soft_validation", ()), routing_timing=spec["routing_timing"], mode="soft"
    )
    hard = _validate_mode(
        result.get("hard_validation", ()), routing_timing=spec["routing_timing"], mode="hard"
    )
    _require(result.get("soft_aggregate") == _aggregate(soft, "soft"), "soft aggregate drifted")
    _require(result.get("hard_aggregate") == _aggregate(hard, "hard"), "hard aggregate drifted")


def _route_transition(reference: Mapping[str, Any], current: Mapping[str, Any]) -> dict[str, Any]:
    left = reference["route_map"]
    right = current["route_map"]
    _require(
        [(item["target_layer"], item["unit_type"]) for item in left]
        == [(item["target_layer"], item["unit_type"]) for item in right],
        "paired route ownership/order drifted",
    )
    transitions = {
        f"{source}_to_{target}": sum(
            a["selected_bits"] == source and b["selected_bits"] == target
            for a, b in zip(left, right, strict=True)
        )
        for source in (4, 6, 8)
        for target in (4, 6, 8)
    }
    changed = sum(
        a["selected_bits"] != b["selected_bits"] for a, b in zip(left, right, strict=True)
    )
    return {
        "transition_counts": transitions,
        "changed_count": changed,
        "hamming_distance": changed / 72,
        "mean_selected_width_delta": current["hard_mean_selected_width"]
        - reference["hard_mean_selected_width"],
    }


def _trial_by_cell(
    trials: Sequence[Mapping[str, Any]], seed: int, timing: str, lambda_bit: float
) -> Mapping[str, Any]:
    matches = [
        item
        for item in trials
        if item["seed"] == seed
        and item["routing_timing"] == timing
        and item["lambda_bit"] == lambda_bit
    ]
    _require(len(matches) == 1, "paired trial cell is missing or duplicated")
    return matches[0]


def _hard_by_request(trial: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {item["request_id"]: item for item in trial["hard_validation"]}


def build_aggregation(
    trials: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> dict[str, Any]:
    """Build the frozen paired scientific evaluation from twelve validated trials."""

    specs = contract.trial_specs()
    _require(len(trials) == 12, "aggregation requires exactly twelve trials", outcome="PAUSE")
    _require(
        [item.get("trial_id") for item in trials] == [item["trial_id"] for item in specs],
        "aggregation trial identity/order drifted",
    )
    for trial, spec in zip(trials, specs, strict=True):
        _require(trial.get("trial_spec") == spec, "aggregation trial specification drifted")
        _validate_trial_result(trial, spec)

    by_id = {item["trial_id"]: item for item in trials}
    transitions = []
    for trial in trials:
        timing_reference = by_id[trial["trial_spec"]["same_cost_timing_pair_trial_id"]]
        cost_reference = by_id[trial["trial_spec"]["zero_cost_reference_trial_id"]]
        current_requests = _hard_by_request(trial)
        timing_requests = _hard_by_request(timing_reference)
        cost_requests = _hard_by_request(cost_reference)
        transitions.append(
            {
                "trial_id": trial["trial_id"],
                "same_cost_timing_reference_trial_id": timing_reference["trial_id"],
                "zero_cost_reference_trial_id": cost_reference["trial_id"],
                "requests": [
                    {
                        "request_id": request_id,
                        "treatment_control": _route_transition(
                            timing_requests[request_id], current_requests[request_id]
                        ),
                        "positive_zero_cost": _route_transition(
                            cost_requests[request_id], current_requests[request_id]
                        ),
                    }
                    for request_id in contract.VALIDATION_IDS
                ],
            }
        )

    kl_name = "completion_only_temperature_2_masked_teacher_relative_kl"
    error_name = "full_logit_mean_absolute_teacher_error"
    quality_thresholds = config["thresholds"]["quality"]
    precision_thresholds = config["thresholds"]["precision"]
    refine_thresholds = config["thresholds"]["refine"]
    per_seed = []
    request_factor_zero = []
    request_factor_control = []
    aggregate_error_zero = []
    aggregate_kl_control = []
    aggregate_error_control = []
    for seed in contract.SEEDS:
        positive = _trial_by_cell(trials, seed, "lookahead_attention_one_unit", 0.03)
        zero = _trial_by_cell(trials, seed, "lookahead_attention_one_unit", 0.0)
        control = _trial_by_cell(trials, seed, "same_unit", 0.03)
        positive_requests = _hard_by_request(positive)
        zero_requests = _hard_by_request(zero)
        control_requests = _hard_by_request(control)
        request_factor_zero.append(
            all(
                positive_requests[item][kl_name]
                <= quality_thresholds["lookahead_positive_each_request_hard_kl_max_zero_factor"]
                * zero_requests[item][kl_name]
                for item in contract.VALIDATION_IDS
            )
        )
        request_factor_control.append(
            all(
                positive_requests[item][kl_name]
                <= quality_thresholds[
                    "lookahead_positive_each_request_hard_kl_max_same_unit_positive_factor"
                ]
                * control_requests[item][kl_name]
                for item in contract.VALIDATION_IDS
            )
        )
        aggregate_error_zero.append(
            positive["hard_aggregate"][error_name]
            <= quality_thresholds["lookahead_positive_aggregate_hard_mean_error_max_zero_factor"]
            * zero["hard_aggregate"][error_name]
        )
        aggregate_kl_control.append(
            positive["hard_aggregate"][kl_name]
            <= quality_thresholds[
                "lookahead_positive_aggregate_hard_kl_max_same_unit_positive_factor"
            ]
            * control["hard_aggregate"][kl_name]
        )
        aggregate_error_control.append(
            positive["hard_aggregate"][error_name]
            <= quality_thresholds[
                "lookahead_positive_aggregate_hard_mean_error_max_same_unit_positive_factor"
            ]
            * control["hard_aggregate"][error_name]
        )
        per_seed.append(
            {
                "seed": seed,
                "lookahead_positive_minus_zero_hard_kl": positive["hard_aggregate"][kl_name]
                - zero["hard_aggregate"][kl_name],
                "lookahead_positive_minus_zero_hard_width": positive["hard_aggregate"][
                    "hard_mean_selected_width"
                ]
                - zero["hard_aggregate"]["hard_mean_selected_width"],
                "lookahead_positive_minus_same_unit_positive_hard_width": positive[
                    "hard_aggregate"
                ]["hard_mean_selected_width"]
                - control["hard_aggregate"]["hard_mean_selected_width"],
            }
        )

    median_kl = statistics.median(
        item["lookahead_positive_minus_zero_hard_kl"] for item in per_seed
    )
    median_width = statistics.median(
        item["lookahead_positive_minus_zero_hard_width"] for item in per_seed
    )
    quality_factors = all(
        request_factor_zero
        + request_factor_control
        + aggregate_error_zero
        + aggregate_kl_control
        + aggregate_error_control
    )
    quality_passed = (
        median_kl <= quality_thresholds["lookahead_positive_minus_zero_median_hard_kl_max"]
        and quality_factors
    )
    negative_width_count = sum(
        item["lookahead_positive_minus_zero_hard_width"] < 0 for item in per_seed
    )
    precision_passed = (
        median_width <= precision_thresholds["lookahead_positive_minus_zero_median_hard_width_max"]
        and negative_width_count >= precision_thresholds["strictly_negative_seed_delta_minimum"]
    )
    width_reduction = -median_width
    refine_quality = (
        precision_passed
        and quality_factors
        and quality_thresholds["lookahead_positive_minus_zero_median_hard_kl_max"]
        < median_kl
        < refine_thresholds["quality_near_miss_hard_kl_delta_strict_max"]
    )
    refine_precision = (
        quality_passed
        and negative_width_count == 3
        and refine_thresholds["precision_near_miss_median_reduction_minimum"]
        <= width_reduction
        < refine_thresholds["precision_pass_reduction_minimum"]
    )
    if quality_passed and precision_passed:
        classification = "CONTINUE"
    elif refine_quality or refine_precision:
        classification = "REFINE"
    else:
        classification = "STOP"
    return {
        "schema": "qaq-s11d-paired-lookahead-468-aggregation-v1",
        "protocol_sha256": contract.EXPECTED_CONFIG_SHA256,
        "trial_order": [item["trial_id"] for item in specs],
        "request_order": list(contract.VALIDATION_IDS),
        "paired_route_transitions": transitions,
        "per_seed": per_seed,
        "median_paired_seed_deltas": {
            "lookahead_positive_minus_zero_hard_kl": median_kl,
            "lookahead_positive_minus_zero_hard_width": median_width,
        },
        "quality": {
            "median_kl_passed": median_kl
            <= quality_thresholds["lookahead_positive_minus_zero_median_hard_kl_max"],
            "all_factor_safeguards_passed": quality_factors,
            "passed": quality_passed,
        },
        "precision": {
            "median_width_passed": median_width
            <= precision_thresholds["lookahead_positive_minus_zero_median_hard_width_max"],
            "strictly_negative_seed_count": negative_width_count,
            "passed": precision_passed,
        },
        "refine_regions": {
            "quality_near_miss": refine_quality,
            "precision_near_miss": refine_precision,
        },
        "classification": classification,
        "quality_and_precision_separate": True,
        "production_lambda_selected": False,
        "audits": {"complete": True, "finite": True, "paired": True, "passed": True},
    }


def _validate_aggregation_result(result: Mapping[str, Any]) -> None:
    _require(
        result.get("trial_order") == [item["trial_id"] for item in contract.trial_specs()],
        "persisted aggregation trial order drifted",
    )
    _require(
        result.get("request_order") == list(contract.VALIDATION_IDS),
        "persisted aggregation request order drifted",
    )
    _require(
        result.get("classification") in {"CONTINUE", "REFINE", "STOP"},
        "persisted aggregation classification is invalid",
    )
    _require(result.get("audits", {}).get("passed") is True, "aggregation audits failed")
    _require(
        len(result.get("paired_route_transitions", ())) == 12,
        "persisted paired transitions are incomplete",
    )


def _write_result(result: dict[str, Any], destination: Path) -> None:
    """Publish one complete validated result atomically and never leave a partial parent."""

    _validate_trial_result(result, result["trial_spec"])
    parent = destination.parent
    created_parent = False
    temporary: Path | None = None
    try:
        if not parent.exists():
            parent.mkdir(parents=False)
            created_parent = True
        _require(
            parent.is_dir() and not parent.is_symlink(),
            f"result parent is unavailable or redirected: {parent}",
            outcome="PAUSE",
        )
        _require(
            not os.path.lexists(destination),
            f"trial result already exists: {destination}",
            outcome="PAUSE",
        )
        descriptor, name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=parent
        )
        temporary = Path(name)
        payload = json.dumps(result, indent=2, sort_keys=False, allow_nan=False) + "\n"
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        reloaded = json.loads(temporary.read_text(encoding="utf-8"))
        _validate_trial_result(reloaded, result["trial_spec"])
        os.link(temporary, destination)
        temporary.unlink()
        temporary = None
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        if created_parent:
            try:
                parent.rmdir()
            except OSError:
                pass
        raise


def _write_aggregation(result: dict[str, Any], destination: Path) -> None:
    _validate_aggregation_result(result)
    temporary: Path | None = None
    try:
        _require(
            destination.parent.is_dir() and not destination.parent.is_symlink(),
            "aggregation parent is unavailable or redirected",
            outcome="PAUSE",
        )
        _require(
            not os.path.lexists(destination),
            f"aggregation result already exists: {destination}",
            outcome="PAUSE",
        )
        descriptor, name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
        temporary = Path(name)
        payload = json.dumps(result, indent=2, sort_keys=False, allow_nan=False) + "\n"
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        reloaded = json.loads(temporary.read_text(encoding="utf-8"))
        _validate_aggregation_result(reloaded)
        os.link(temporary, destination)
        temporary.unlink()
        temporary = None
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def aggregate_trials(
    trials: Sequence[Mapping[str, Any]],
    *,
    config: Mapping[str, Any],
    output: Path | None = None,
) -> AggregationOutcome:
    try:
        result = build_aggregation(trials, config)
        _validate_aggregation_result(result)
        if output is not None:
            _write_aggregation(result, Path(output))
        return AggregationOutcome(
            result["classification"],
            (),
            result,
            str(output) if output else None,
            output is not None,
        )
    except contract.ProtocolError as exc:
        return AggregationOutcome(exc.outcome, (str(exc),), None, None, False)
    except (KeyError, TypeError, ValueError, RuntimeError, OSError) as exc:
        return AggregationOutcome("REVISE", (str(exc),), None, None, False)


def execute_aggregation(
    *, paths: Sequence[Path], config: Mapping[str, Any], output: Path
) -> AggregationOutcome:
    try:
        trials = [json.loads(Path(path).read_text(encoding="utf-8")) for path in paths]
    except (OSError, json.JSONDecodeError) as exc:
        return AggregationOutcome("PAUSE", (str(exc),), None, None, False)
    return aggregate_trials(trials, config=config, output=output)


def run_trial(
    runtime: PairedLookaheadRuntime,
    *,
    config: Mapping[str, Any],
    spec: Mapping[str, Any],
    device: str,
    output: Path | None = None,
) -> TrialOutcome:
    """Execute one already-dispatched trial against an injected or production runtime."""

    model: Any | None = None
    try:
        frozen_spec = next(
            (item for item in contract.trial_specs() if item["trial_id"] == spec.get("trial_id")),
            None,
        )
        _require(frozen_spec == dict(spec), "runtime received a non-frozen trial specification")
        _require(
            isinstance(device, str) and device.startswith("cuda:"),
            "runtime requires CUDA",
            outcome="PAUSE",
        )
        runtime.prepare(config, device)
        _require(
            _example_ids(runtime.train_examples) == contract.TRAIN_IDS,
            "training data/order drifted",
        )
        model = runtime.build_seed_model(int(spec["seed"]), device)
        router_items = _router_items(model)
        _require(len(getattr(model, "routers", {})) == 72, "runtime must expose 72 routers")
        _require(len(router_items) == 288, "runtime router tensor count drifted")
        if runtime.enforce_frozen_model_contract:
            _require(
                sum(parameter.numel() for _, parameter in router_items) == 23630040,
                "production router scalar count drifted",
            )
        canonical = _clone_state(runtime.router_state(model))
        initial_bytes = _state_bytes(canonical)
        initial_hash = hashlib.sha256(initial_bytes).hexdigest()
        _require(
            runtime.initial_identity_matches(int(spec["seed"]), initial_hash),
            "seed initialization identity mismatch",
        )
        if runtime.enforce_frozen_model_contract:
            _require(
                initial_hash == spec["initial_router_state_sha256"],
                "frozen seed initialization hash mismatch",
            )
        runtime.restore_router_state(model, canonical)
        _require(
            _state_bytes(runtime.router_state(model)) == initial_bytes,
            "paired initialization is not byte-identical",
        )
        optimizer = _build_optimizer(model, config["training"])
        optimizer_audit = _audit_optimizer(model, optimizer, int(spec["trial_index"]))
        frozen_before = dict(runtime.frozen_snapshot(model))
        history: list[dict[str, Any]] = []
        completed_optimizer_steps: list[int] = []
        router_parameters = [parameter for _, parameter in router_items]
        for step, example in enumerate(runtime.train_examples, start=1):
            optimizer.zero_grad(set_to_none=True)
            evidence = runtime.loss(model, example, spec, step, device)
            total, gradients, provenance, request_state_audit = _perform_optimizer_update(
                runtime,
                optimizer,
                evidence,
                router_parameters,
                request_id=example.example_id,
                routing_timing=spec["routing_timing"],
                step=step,
                completed=completed_optimizer_steps,
            )
            history.append(
                {
                    "step": step,
                    "example_id": example.example_id,
                    "kd_loss": float(evidence["kd_loss"]),
                    "bit_loss": float(evidence["bit_loss"]),
                    "total_loss": float(total.detach().item()),
                    "router_gradient_norm": float(
                        torch.sqrt(
                            sum(value.detach().float().square().sum() for value in gradients)
                        ).item()
                    ),
                    "provenance_count": len(provenance),
                    "request_state_audit": request_state_audit,
                    "optimizer_state_entries": len(optimizer.state),
                }
            )
            _require(
                all(
                    _finite(history[-1][name])
                    for name in ("kd_loss", "bit_loss", "total_loss", "router_gradient_norm")
                ),
                "training evidence is non-finite",
            )
        _require(
            completed_optimizer_steps == list(range(1, 25)),
            "optimizer update count or order drifted",
        )
        _require(
            [item["step"] for item in history] == completed_optimizer_steps,
            "training history does not match optimizer updates",
        )
        frozen_after_training = dict(runtime.frozen_audit(model, frozen_before))
        _require(
            frozen_after_training.get("passed") is True,
            "teacher or packed base changed during training",
        )
        soft = _validate_mode(
            runtime.validate(model, spec, "soft", device),
            routing_timing=spec["routing_timing"],
            mode="soft",
        )
        hard = _validate_mode(
            runtime.validate(model, spec, "hard", device),
            routing_timing=spec["routing_timing"],
            mode="hard",
        )
        repeat = _validate_mode(
            runtime.validate(model, spec, "hard", device),
            routing_timing=spec["routing_timing"],
            mode="hard",
        )
        hard_bytes = json.dumps(
            hard, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        repeat_bytes = json.dumps(
            repeat, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        _require(hard_bytes == repeat_bytes, "immediate hard repeat is not byte-identical")
        frozen_after_evaluation = dict(runtime.frozen_audit(model, frozen_before))
        _require(
            frozen_after_evaluation.get("passed") is True,
            "teacher or packed base changed during evaluation",
        )
        freeze_audit = {
            "after_training": frozen_after_training,
            "after_evaluation": frozen_after_evaluation,
            "passed": True,
        }
        result = {
            "schema": "qaq-s11d-paired-lookahead-468-trial-v1",
            "trial_id": spec["trial_id"],
            "trial_index": spec["trial_index"],
            "trial_spec": dict(spec),
            "protocol_sha256": contract.EXPECTED_CONFIG_SHA256,
            "seed": spec["seed"],
            "arm_id": spec["arm_id"],
            "routing_timing": spec["routing_timing"],
            "lambda_bit": spec["lambda_bit"],
            "initial_router_state_sha256": initial_hash,
            "final_router_state_sha256": _state_hash(runtime.router_state(model)),
            "training_examples_seen": list(contract.TRAIN_IDS),
            "optimizer_steps_completed": len(completed_optimizer_steps),
            "training_history": history,
            "optimizer_audit": optimizer_audit,
            "freeze_audit": freeze_audit,
            "soft_validation": soft,
            "hard_validation": hard,
            "soft_aggregate": _aggregate(soft, "soft"),
            "hard_aggregate": _aggregate(hard, "hard"),
            "route_decisions": sum(len(item["route_map"]) for item in hard),
            "required_metrics": list(config["evaluation"]["request_metrics"]),
            "immediate_hard_repeat": {
                "count": 1,
                "identical": True,
                "sha256": hashlib.sha256(hard_bytes).hexdigest(),
            },
            "paired_route_transition_references": {
                "zero_cost_reference_trial_id": spec["zero_cost_reference_trial_id"],
                "same_cost_timing_pair_trial_id": spec["same_cost_timing_pair_trial_id"],
                "computed_by_frozen_aggregation_after_all_trials": True,
            },
            "production_checkpoint_created": False,
            "audits": {
                "identity": True,
                "data_order": True,
                "paired_initialization": True,
                "optimizer": True,
                "update_count": True,
                "freeze": True,
                "gradients": True,
                "route_and_provenance": True,
                "repeat": True,
                "prohibited_work": True,
                "passed": True,
            },
        }
        _validate_trial_result(result, spec)
        if output is not None:
            _write_result(result, Path(output))
        return TrialOutcome(
            "TRIAL_COMPLETE", (), result, str(output) if output else None, output is not None
        )
    except contract.ProtocolError as exc:
        return TrialOutcome(exc.outcome, (str(exc),), None, None, False)
    except (KeyError, TypeError, ValueError, RuntimeError, OSError) as exc:
        outcome = getattr(exc, "outcome", "REVISE")
        if outcome not in {"PAUSE", "REVISE"}:
            outcome = "REVISE"
        return TrialOutcome(outcome, (str(exc),), None, None, False)
    finally:
        if model is not None:
            try:
                runtime.close_model(model)
            except (OSError, RuntimeError):
                pass


def execute_production(
    *, config: Mapping[str, Any], spec: Mapping[str, Any], device: str, output: Path
) -> TrialOutcome:
    """Construct the heavy pinned runtime only after dispatcher validation."""

    return run_trial(
        ProductionRuntime(config), config=config, spec=spec, device=device, output=output
    )


class ProductionRuntime:
    """Pinned Qwen/Any-Precision/dataset implementation of the runtime boundary."""

    enforce_frozen_model_contract = True

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = config
        self.delegate: Any | None = None
        self.train_examples: Sequence[Any] = ()
        self.validation_examples: Sequence[Any] = ()
        self.teacher_targets: dict[str, Any] = {}
        self.teacher: Any | None = None

    def prepare(self, config: Mapping[str, Any], device: str) -> None:
        _require(torch.cuda.is_available(), "CUDA is unavailable", outcome="PAUSE")
        _require(
            torch.device(device).type == "cuda", "explicit CUDA device is required", outcome="PAUSE"
        )
        # Reuse the established synchronous, resident packed loader and exact
        # S10-H data selection; no prefetch/cache/scheduler path is introduced.
        from qaq.router import broader_validation_executor as established
        from qaq.router import broader_validation_protocol as broader_contract

        broader_config = broader_contract._load_frozen_config()
        preflight = {
            "identities": {
                **dict(config["identities"]),
                "manifest_sha256": contract.EXPECTED_S10H_SHA256,
                "packed_artifact": "docs/quantized_model_manifest.json",
            }
        }
        self.delegate = established.QwenRuntime(broader_config, preflight=preflight)
        self.delegate.prepare(broader_config, device)
        canonical_dataset = json.loads(
            (contract.ROOT / config["data"]["canonical_manifest_path"]).read_text()
        )["dataset"]
        _require(
            self.delegate.train_manifest == canonical_dataset["train_manifest"],
            "runtime training manifest differs from canonical evidence",
        )
        _require(
            self.delegate.validation_manifest == canonical_dataset["validation_manifest"],
            "runtime validation manifest differs from canonical evidence",
        )
        self.train_examples = tuple(self.delegate.train_examples)
        self.validation_examples = tuple(self.delegate.validation_examples)
        self.teacher_targets = self.delegate.teacher_targets
        self.teacher = self.delegate.teacher

    def build_seed_model(self, seed: int, device: str) -> Any:
        return self.delegate.build_seed_model(seed, device)

    def router_state(self, model: Any) -> Mapping[str, torch.Tensor]:
        return self.delegate.router_state(model)

    def restore_router_state(self, model: Any, state: Mapping[str, torch.Tensor]) -> None:
        self.delegate.restore_router_state(model, dict(state))

    def initial_identity_matches(self, seed: int, digest: str) -> bool:
        return digest == contract.INITIAL_HASHES[seed]

    def frozen_snapshot(self, model: Any) -> Mapping[str, Any]:
        return self.delegate.frozen_snapshot(model)

    def frozen_audit(self, model: Any, before: Mapping[str, Any]) -> Mapping[str, Any]:
        return self.delegate.frozen_audit(model, dict(before))

    @staticmethod
    def _provenance(state: Any, request_id: str, routing_timing: str) -> list[dict[str, Any]]:
        records = []
        for layer in range(36):
            for unit, values in (
                ("attention", state.attention_provenance),
                ("ffn", state.ffn_provenance),
            ):
                expected = _expected_provenance(request_id, routing_timing, layer, unit)
                value = values[layer]
                if routing_timing == "lookahead_attention_one_unit" and unit == "attention":
                    if layer == 0:
                        _require(
                            value is None, "treatment layer-0 provenance representation drifted"
                        )
                    else:
                        _require(
                            value is not None
                            and {
                                "request_id": request_id,
                                **value.to_dict(),
                                "candidate_order": [4, 6, 8],
                            }
                            == expected,
                            f"lookahead provenance is missing or invalid at attention layer {layer}",
                        )
                records.append(expected)
        return records

    @staticmethod
    def _cleanup_request_state(state: Any) -> bool:
        state.end_request()
        return bool(
            state.ended
            and all(
                item is None
                for values in (
                    state.attention_routes,
                    state.ffn_routes,
                    state.attention_features,
                    state.ffn_features,
                    state.attention_probabilities,
                    state.ffn_probabilities,
                    state.attention_provenance,
                    state.ffn_provenance,
                )
                for item in values
            )
        )

    def loss(
        self, model: Any, example: Any, spec: Mapping[str, Any], step: int, device: str
    ) -> Mapping[str, Any]:
        del step
        from qaq.model.request_state import QaqRequestState
        from qaq.router.baseline_training import _model_kwargs
        from qaq.router.distillation import (
            DistillationBatch,
            cost_aware_distillation_loss,
            masked_kl_distillation_loss,
            request_state_expected_bit_cost,
        )

        batch = DistillationBatch.from_examples([example])
        state = QaqRequestState(
            example.example_id,
            int(example.prompt_mask().sum()),
            layer_count=36,
            candidate_bits=(4, 6, 8),
            routing_timing=spec["routing_timing"],
        )
        try:
            logits = model(
                **_model_kwargs(example),
                request_state=state,
                phase="prefill",
                prompt_attention_mask=batch.prompt_attention_mask,
            ).logits
            state.assert_soft_complete()
            teacher = self.teacher_targets[example.example_id].to(device)
            kd = masked_kl_distillation_loss(
                teacher, logits, batch.completion_loss_mask, temperature=2.0
            )
            bit = request_state_expected_bit_cost(state)
            total = cost_aware_distillation_loss(kd, bit, float(spec["lambda_bit"]))
            return {
                "total_loss": total,
                "kd_loss": float(kd.detach()),
                "bit_loss": float(bit.detach()),
                "provenance": self._provenance(state, example.example_id, spec["routing_timing"]),
                "request_state_complete": True,
                "_request_state": state,
            }
        except Exception:
            self._cleanup_request_state(state)
            raise

    def finalize_training_request(self, evidence: Mapping[str, Any]) -> Mapping[str, bool]:
        state = evidence.get("_request_state")
        _require(state is not None, "training request state is unavailable for cleanup")
        return {
            "complete": evidence.get("request_state_complete") is True,
            "cleanup": self._cleanup_request_state(state),
        }

    @staticmethod
    def _tensor_digest(value: torch.Tensor) -> str:
        return hashlib.sha256(
            value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        ).hexdigest()

    def validate(
        self, model: Any, spec: Mapping[str, Any], mode: str, device: str
    ) -> Sequence[Mapping[str, Any]]:
        from qaq.model.request_state import QaqRequestState
        from qaq.router.baseline_training import _model_kwargs
        from qaq.router.distillation import hard_route, masked_kl_distillation_loss

        model.eval()
        records = []
        for example in self.validation_examples:
            state = QaqRequestState(
                example.example_id,
                int(example.prompt_mask().sum()),
                layer_count=36,
                candidate_bits=(4, 6, 8),
                routing_timing=spec["routing_timing"],
            )
            with torch.inference_mode():
                if mode == "soft":
                    logits = model(
                        **_model_kwargs(example),
                        request_state=state,
                        phase="prefill",
                        prompt_attention_mask=example.prompt_mask().unsqueeze(0),
                    ).logits
                else:

                    def policy(layer: int, unit_type: str, feature: Any) -> int:
                        return int(
                            hard_route(
                                model.route(layer, unit_type, feature), candidate_bits=(4, 6, 8)
                            )
                        )

                    logits = model.base(
                        **_model_kwargs(example),
                        request_state=state,
                        phase="prefill",
                        prompt_attention_mask=example.prompt_mask().unsqueeze(0),
                        routing_policy=policy,
                    ).logits
            if mode == "soft":
                state.assert_soft_complete()
            else:
                state.assert_complete()
            teacher = self.teacher_targets[example.example_id].to(device)
            mask = example.completion_loss_mask.unsqueeze(0)
            kd = masked_kl_distillation_loss(teacher, logits, mask, temperature=2.0)
            error = (logits.float() - teacher.float()).abs()
            item: dict[str, Any] = {
                "request_id": example.example_id,
                "input_digest": self._tensor_digest(example.input_ids),
                "teacher_digest": self._tensor_digest(teacher),
                "logits_digest": self._tensor_digest(logits),
                "completion_only_temperature_2_masked_teacher_relative_kl": float(kd),
                "full_logit_mean_absolute_teacher_error": float(error.mean()),
                "full_logit_maximum_absolute_teacher_error": float(error.max()),
            }
            probabilities = state.attention_probabilities + state.ffn_probabilities
            if mode == "soft":
                expected = [
                    sum(float(probability[index]) * bit for index, bit in enumerate((4, 6, 8)))
                    for probability in probabilities
                    if probability is not None
                ]
                item["soft_expected_width"] = sum(expected) / 72
            else:
                route_map = []
                for layer in range(36):
                    for unit, values in (
                        ("attention", state.attention_routes),
                        ("ffn", state.ffn_routes),
                    ):
                        route_map.append(
                            {
                                "request_id": example.example_id,
                                "target_layer": layer,
                                "unit_type": unit,
                                "selected_bits": int(values[layer]),
                            }
                        )
                bits = [value["selected_bits"] for value in route_map]
                item.update(
                    {
                        "route_map": route_map,
                        "provenance": self._provenance(
                            state, example.example_id, spec["routing_timing"]
                        ),
                        "hard_counts": {str(bit): bits.count(bit) for bit in (4, 6, 8)},
                        "hard_fractions": {str(bit): bits.count(bit) / 72 for bit in (4, 6, 8)},
                        "hard_mean_selected_width": sum(bits) / 72,
                        "attention_mean_selected_width": sum(bits[0::2]) / 36,
                        "ffn_mean_selected_width": sum(bits[1::2]) / 36,
                        "overall_mean_selected_width": sum(bits) / 72,
                    }
                )
            cleanup = self._cleanup_request_state(state)
            _require(cleanup, "request-state cleanup failed")
            item["request_state_audit"] = {"complete": True, "cleanup": True}
            records.append(item)
        return records

    def close_model(self, model: Any) -> None:
        self.delegate.close_model(model)


__all__ = [
    "REQUEST_METRICS",
    "AggregationOutcome",
    "PairedLookaheadRuntime",
    "ProductionRuntime",
    "RuntimeFailure",
    "TrialOutcome",
    "aggregate_trials",
    "build_aggregation",
    "execute_aggregation",
    "execute_production",
    "run_trial",
]
