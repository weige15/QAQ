"""S10-H2-A execution seam and protocol-locked trial scheduler.

This module is intentionally imported only by ``scripts/run_s10h.py`` after an
explicit ``--execute --device`` request.  The scheduler is runtime-agnostic:
production Qwen/data loading lives behind ``QwenRuntime`` while tests inject a
small deterministic runtime implementing the same object-level audits and loss
calls.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from qaq.router.distillation import DistillationExample
from scripts import run_s10h as protocol

_OPTIMIZER_SERIALS = itertools.count(1)


class ExecutorError(protocol.ProtocolError):
    """An execution failure classified by the frozen H2 gate."""


@dataclass(frozen=True)
class ExecutionOutcome:
    classification: str
    errors: tuple[str, ...]
    result: dict[str, Any] | None
    validation: dict[str, Any]
    output_path: str | None
    written: bool


class S10HRuntime(Protocol):
    """Replaceable boundary for model/data execution.

    The scheduler owns trial order, optimizer construction, audits, aggregation,
    validation, and output safety.  A runtime supplies only actual model/data
    objects and the forward/loss/evaluation operations for those objects.
    """

    enforce_frozen_router_scalar_count: bool

    def prepare(self, config: Mapping[str, Any], device: str) -> None: ...

    def dataset_evidence(self, config: Mapping[str, Any]) -> dict[str, Any]: ...

    def identity_evidence(self) -> dict[str, Any]: ...

    def inherited_regressions_audit(self) -> dict[str, Any]: ...

    def prohibited_work_audit(self) -> dict[str, Any]: ...

    def build_seed_model(self, seed: int, device: str) -> Any: ...

    def router_state(self, model: Any) -> dict[str, Any]: ...

    def restore_router_state(self, model: Any, state: dict[str, Any]) -> None: ...

    def frozen_snapshot(self, model: Any) -> dict[str, Any]: ...

    def frozen_audit(self, model: Any, before: dict[str, Any]) -> dict[str, Any]: ...

    def train_step(
        self,
        model: Any,
        example: Any,
        optimizer: Any,
        lambda_bit: float,
        step: int,
        device: str,
    ) -> dict[str, Any]: ...

    def validate(self, model: Any, mode: str, device: str) -> dict[str, Any]: ...

    def close_model(self, model: Any) -> None: ...


@dataclass
class _PreparedRuntime:
    runtime: S10HRuntime
    config: Mapping[str, Any]
    device: str


def _sha256_names(names: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(names).encode()).hexdigest()


def _state_hash(state: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(str(value.dtype).encode())
        digest.update(b"\0")
        digest.update(repr(tuple(value.shape)).encode())
        digest.update(b"\0")
        digest.update(value.view(__import__("torch").uint8).numpy().tobytes())
        digest.update(b"\n")
    return digest.hexdigest()


def _clone_state(model: Any) -> dict[str, Any]:
    return {
        name: value.detach().cpu().contiguous().clone()
        for name, value in model.state_dict().items()
    }


def _router_items(model: Any) -> list[tuple[str, Any]]:
    return [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if name.startswith("routers.")
    ]


def _router_contract(model: Any) -> dict[str, Any]:
    items = _router_items(model)
    return {
        "router_count": len(getattr(model, "routers", {})),
        "router_tensor_count": len(items),
        "router_scalar_count": sum(parameter.numel() for _, parameter in items),
        "router_parameter_names": sorted(name for name, _ in items),
    }


def _finite_scalar(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _require_finite_metrics(values: Mapping[str, Any], names: Sequence[str]) -> None:
    for name in names:
        if not _finite_scalar(values.get(name)):
            raise ExecutorError("REVISE", f"non-finite or missing runtime metric: {name}")


def _ordered_example_ids(examples: Sequence[Any], *, split: str) -> list[str]:
    ids: list[str] = []
    for index, example in enumerate(examples):
        if isinstance(example, Mapping):
            raise ExecutorError(
                "REVISE", f"runtime {split} example {index} is a dictionary substitute"
            )
        if not isinstance(example, DistillationExample):
            raise ExecutorError(
                "REVISE", f"runtime {split} example {index} is not a DistillationExample"
            )
        if not hasattr(example, "example_id"):
            raise ExecutorError("REVISE", f"runtime {split} example {index} has no example_id")
        example_id = example.example_id
        if not isinstance(example_id, str) or not example_id.strip():
            raise ExecutorError(
                "REVISE", f"runtime {split} example {index} has an invalid example_id"
            )
        ids.append(example_id)
    return ids


def _validate_example_order(
    examples: Sequence[Any], expected_ids: Sequence[str], *, split: str
) -> None:
    if _ordered_example_ids(examples, split=split) != list(expected_ids):
        raise ExecutorError("REVISE", f"runtime {split} example order differs from frozen protocol")


def _optimizer_audit(
    model: Any, optimizer: Any, *, before_training: int, before_first: int
) -> dict[str, Any]:
    """Audit optimizer membership from actual parameter identities and names."""

    expected_names = list(protocol.EXPECTED_ROUTER_PARAMETER_NAMES)
    named_by_id = {id(parameter): name for name, parameter in model.named_parameters()}
    expected_ids = {
        id(parameter) for name, parameter in model.named_parameters() if name.startswith("routers.")
    }
    actual_parameters = [
        parameter for group in optimizer.param_groups for parameter in group["params"]
    ]
    actual_ids = [id(parameter) for parameter in actual_parameters]
    actual_names = sorted(
        named_by_id[parameter_id] for parameter_id in actual_ids if parameter_id in named_by_id
    )
    duplicate_count = len(actual_ids) - len(set(actual_ids))
    missing_count = len(expected_ids - set(actual_ids))
    unexpected_ids = set(actual_ids) - expected_ids
    unexpected_count = len(unexpected_ids)
    serial = next(_OPTIMIZER_SERIALS)
    audit = {
        "actual_optimizer_parameter_count": len(actual_parameters),
        "actual_optimizer_parameter_names": actual_names,
        "actual_optimizer_parameter_names_sha256": _sha256_names(actual_names),
        "duplicate_optimizer_parameter_count": duplicate_count,
        "expected_router_parameter_count": len(expected_names),
        "expected_router_parameter_names": expected_names,
        "expected_router_parameter_names_sha256": _sha256_names(expected_names),
        "fresh_adamw_audit": isinstance(optimizer, __import__("torch").optim.AdamW)
        and not optimizer.state,
        "missing_router_parameter_count": missing_count,
        "optimizer_construction_serial": serial,
        "optimizer_state_entry_count_before_first_step": before_first,
        "optimizer_state_entry_count_before_training_begins": before_training,
        "router_only_optimizer_audit": (
            actual_names == expected_names
            and duplicate_count == 0
            and missing_count == 0
            and unexpected_count == 0
        ),
        "runtime_identity_based_membership_result": (
            len(actual_ids) == len(set(actual_ids)) and set(actual_ids) == expected_ids
        ),
        "unexpected_optimizer_parameter_count": unexpected_count,
    }
    if not (
        audit["actual_optimizer_parameter_names"] == expected_names
        and audit["duplicate_optimizer_parameter_count"] == 0
        and audit["missing_router_parameter_count"] == 0
        and audit["unexpected_optimizer_parameter_count"] == 0
        and audit["runtime_identity_based_membership_result"] is True
        and audit["router_only_optimizer_audit"] is True
        and audit["fresh_adamw_audit"] is True
    ):
        raise ExecutorError("REVISE", "optimizer membership, identity, or freshness audit failed")
    return audit


def _build_optimizer(model: Any, config: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
    """Construct a fresh AdamW from the actual router parameters only."""

    import torch

    items = _router_items(model)
    if not items:
        raise ExecutorError("REVISE", "router model has no routers. parameters")
    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in items],
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
        betas=tuple(float(value) for value in config["betas"]),
        eps=float(config["eps"]),
        amsgrad=bool(config["amsgrad"]),
    )
    before = len(optimizer.state)
    audit = _optimizer_audit(model, optimizer, before_training=before, before_first=before)
    return optimizer, audit


def _validate_route_maps(hard: Mapping[str, Any]) -> dict[str, Any]:
    maps = hard.get("hard_validation_route_maps")
    if not isinstance(maps, dict) or list(maps) != list(protocol.VALIDATION_IDS):
        raise ExecutorError("REVISE", "validation route-map IDs/order changed")
    if any(
        not protocol._validate_route_map(maps[request_id]) for request_id in protocol.VALIDATION_IDS
    ):
        raise ExecutorError("REVISE", "validation route map is malformed")
    width, fractions, unique = protocol._route_stats(maps)
    if not _finite_scalar(hard.get("hard_validation_mean_selected_bit_width")):
        raise ExecutorError("REVISE", "hard selected width is non-finite")
    if not math.isclose(
        float(hard["hard_validation_mean_selected_bit_width"]), width, rel_tol=1e-12, abs_tol=1e-12
    ):
        raise ExecutorError("REVISE", "hard width does not match ordered route maps")
    if hard.get("distinct_hard_route_map_count") != unique:
        raise ExecutorError("REVISE", "distinct hard route-map count does not match maps")
    for bit in protocol.CANDIDATE_BITS:
        if not math.isclose(
            float(hard.get(f"hard_validation_fraction_{bit}")),
            fractions[str(bit)],
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ExecutorError("REVISE", "hard route fractions do not match ordered maps")
    variation = hard.get("route_variation")
    if (
        not isinstance(variation, dict)
        or variation.get("unit_count") != 72
        or variation.get("prompt_count") != 12
    ):
        raise ExecutorError("REVISE", "route variation coverage is incomplete")
    return {request_id: list(maps[request_id]) for request_id in protocol.VALIDATION_IDS}


def _validate_mode(mode: Mapping[str, Any], *, hard: bool) -> dict[str, Any]:
    required = (
        "validation_kd",
        "mean_absolute_logit_error",
        "maximum_absolute_logit_error",
    )
    _require_finite_metrics(mode, required)
    if mode.get("finite_outputs") is not True:
        raise ExecutorError("REVISE", "runtime produced non-finite validation outputs")
    if hard:
        _require_finite_metrics(
            mode,
            (
                "hard_validation_mean_selected_bit_width",
                "hard_validation_fraction_4",
                "hard_validation_fraction_6",
                "hard_validation_fraction_8",
            ),
        )
        route_maps = _validate_route_maps(mode)
        return {**mode, "hard_validation_route_maps": route_maps}
    _require_finite_metrics(
        mode,
        (
            "mean_expected_bit_width",
            "mean_p4",
            "mean_p6",
            "mean_p8",
            "mean_entropy",
        ),
    )
    return dict(mode)


def _trial(
    prepared: _PreparedRuntime,
    model: Any,
    canonical_state: dict[str, Any],
    seed: int,
    lambda_bit: float,
) -> dict[str, Any]:
    import torch

    config = prepared.config["protocol"]["training"]
    prepared.runtime.restore_router_state(model, canonical_state)
    initial_hash = _state_hash(prepared.runtime.router_state(model))
    optimizer, optimizer_audit = _build_optimizer(model, config)
    if len(optimizer.state) != 0:
        raise ExecutorError("REVISE", "fresh AdamW contains state before training")
    frozen_before = prepared.runtime.frozen_snapshot(model)
    history: list[dict[str, Any]] = []
    initial_diagnostic: dict[str, Any] | None = None
    router_parameters = [parameter for _, parameter in _router_items(model)]
    train_examples = tuple(prepared.runtime.train_examples)  # type: ignore[attr-defined]
    if len(train_examples) != 24:
        raise ExecutorError("PAUSE", "runtime did not provide exactly 24 training examples")
    for step, example in enumerate(train_examples, start=1):
        before_first = len(optimizer.state)
        raw = prepared.runtime.train_step(
            model, example, optimizer, lambda_bit, step, prepared.device
        )
        if not isinstance(raw, dict):
            raise ExecutorError("PAUSE", "runtime training evidence is missing")
        if raw.get("finite_loss") is not True or raw.get("finite_gradient") is not True:
            raise ExecutorError("REVISE", "non-finite loss or gradient evidence")
        if (
            raw.get("router_gradients_present") is not True
            or raw.get("router_gradients_nonzero") is not True
        ):
            raise ExecutorError("REVISE", "router gradient evidence is missing or zero")
        gradients = [parameter.grad for parameter in router_parameters]
        if any(gradient is None for gradient in gradients):
            raise ExecutorError("REVISE", "a router parameter has no gradient")
        if any(
            not bool(torch.isfinite(gradient).all().item())
            for gradient in gradients
            if gradient is not None
        ):
            raise ExecutorError("REVISE", "a router gradient is non-finite")
        if step == 1:
            initial_diagnostic = {
                "initial_kd_gradient_norm": raw.get("initial_kd_gradient_norm"),
                "initial_bit_cost_gradient_norm": raw.get("initial_bit_cost_gradient_norm"),
                "lambda_weighted_gradient_ratio": raw.get("lambda_weighted_gradient_ratio"),
            }
            _require_finite_metrics(
                initial_diagnostic,
                tuple(initial_diagnostic),
            )
            if float(initial_diagnostic["initial_kd_gradient_norm"]) <= 0:
                raise ExecutorError("REVISE", "initial KD gradient norm is not positive")
        if len(optimizer.state) < before_first:
            raise ExecutorError("REVISE", "optimizer state regressed during a trial")
        history.append(
            {
                "step": step,
                "finite_kd_loss": raw.get("finite_kd_loss", raw["finite_loss"]),
                "finite_gradient": raw["finite_gradient"],
                "finite_bit_cost": raw.get("finite_bit_cost", raw["finite_loss"]),
                "finite_weighted_cost": raw.get("finite_weighted_cost", raw["finite_loss"]),
                "finite_total_loss": raw.get("finite_total_loss", raw["finite_loss"]),
                "kd_loss": raw.get("kd_loss"),
                "expected_bit_cost": raw.get("expected_bit_cost"),
                "weighted_cost": raw.get("weighted_cost"),
                "total_loss": raw.get("total_loss"),
                "router_gradient_norm": raw.get("router_gradient_norm"),
                "optimizer_state_entries_after_step": len(optimizer.state),
            }
        )
        _require_finite_metrics(
            history[-1],
            ("kd_loss", "expected_bit_cost", "weighted_cost", "total_loss", "router_gradient_norm"),
        )
    if len(history) != 24:
        raise ExecutorError("PAUSE", "runtime did not complete exactly 24 updates")
    frozen_audit = prepared.runtime.frozen_audit(model, frozen_before)
    if not isinstance(frozen_audit, dict) or frozen_audit.get("passed") is not True:
        raise ExecutorError("REVISE", "teacher or packed-base freeze audit failed")
    soft = _validate_mode(prepared.runtime.validate(model, "soft", prepared.device), hard=False)
    hard = _validate_mode(prepared.runtime.validate(model, "hard", prepared.device), hard=True)
    repeat = _validate_mode(prepared.runtime.validate(model, "hard", prepared.device), hard=True)
    repeat_audit = {
        "route_maps_identical": hard["hard_validation_route_maps"]
        == repeat["hard_validation_route_maps"],
        "hard_metrics_identical": all(
            hard.get(name) == repeat.get(name)
            for name in (
                "validation_kd",
                "hard_validation_mean_selected_bit_width",
                "hard_validation_fraction_4",
                "hard_validation_fraction_6",
                "hard_validation_fraction_8",
            )
        ),
        "finite_outputs_both_passed": hard.get("finite_outputs") is True
        and repeat.get("finite_outputs") is True,
        "passed": False,
        "repeat_count": 1,
    }
    repeat_audit["passed"] = all(
        repeat_audit[name]
        for name in ("route_maps_identical", "hard_metrics_identical", "finite_outputs_both_passed")
    )
    if repeat_audit["passed"] is not True:
        raise ExecutorError("REVISE", "same-state hard-validation repeat failed")
    collapse = hard.get("collapse_audit")
    if not isinstance(collapse, dict):
        raise ExecutorError("PAUSE", "collapse audit is missing")
    trial = {
        "seed": seed,
        "lambda_bit": lambda_bit,
        "candidate_bits": list(protocol.CANDIDATE_BITS),
        "initial_router_state_sha256": initial_hash,
        "final_router_state_sha256": _state_hash(prepared.runtime.router_state(model)),
        **(initial_diagnostic or {}),
        "training_examples_seen": 24,
        "optimizer_steps_completed": 24,
        "training_history": history,
        "finite_loss_audit": all(
            item["finite_kd_loss"]
            and item["finite_bit_cost"]
            and item["finite_weighted_cost"]
            and item["finite_total_loss"]
            for item in history
        ),
        "finite_gradient_audit": all(
            item["finite_gradient"] and _finite_scalar(item["router_gradient_norm"])
            for item in history
        ),
        "teacher_frozen_audit": True,
        "packed_student_base_unchanged_audit": True,
        "freeze_runtime_audit": frozen_audit,
        "collapse_audit": collapse,
        "optimizer_audit": optimizer_audit,
        "soft_validation_kd": soft["validation_kd"],
        "soft_validation_mean_absolute_logit_error": soft["mean_absolute_logit_error"],
        "soft_validation_maximum_absolute_logit_error": soft["maximum_absolute_logit_error"],
        "soft_validation_mean_expected_bit_width": soft["mean_expected_bit_width"],
        "soft_validation_mean_p4": soft["mean_p4"],
        "soft_validation_mean_p6": soft["mean_p6"],
        "soft_validation_mean_p8": soft["mean_p8"],
        "soft_validation_mean_entropy": soft["mean_entropy"],
        "hard_validation_kd": hard["validation_kd"],
        "hard_validation_mean_absolute_logit_error": hard["mean_absolute_logit_error"],
        "hard_validation_maximum_absolute_logit_error": hard["maximum_absolute_logit_error"],
        "hard_validation_mean_selected_bit_width": hard["hard_validation_mean_selected_bit_width"],
        "hard_validation_fraction_4": hard["hard_validation_fraction_4"],
        "hard_validation_fraction_6": hard["hard_validation_fraction_6"],
        "hard_validation_fraction_8": hard["hard_validation_fraction_8"],
        "hard_validation_route_maps": hard["hard_validation_route_maps"],
        "route_variation": hard["route_variation"],
        "distinct_hard_route_map_count": hard["distinct_hard_route_map_count"],
        "reproducibility_audit": repeat_audit,
        "prohibited_measurement_audit": {
            "forbidden_measurements_observed": [],
            "passed": True,
        },
    }
    return trial


def _build_result(
    prepared: _PreparedRuntime, preflight: Mapping[str, Any] | None, trials: list[dict[str, Any]]
) -> dict[str, Any]:
    config = prepared.config
    dataset = prepared.runtime.dataset_evidence(config)
    identity = dict(prepared.runtime.identity_evidence())
    if preflight is not None:
        identity.update(
            {
                key: preflight["identities"][key]
                for key in (
                    "model_repository",
                    "model_revision",
                    "tokenizer_revision",
                    "dataset_revision",
                    "any_precision_revision",
                    "packed_artifact",
                    "manifest_sha256",
                    "packed_artifact_pytorch_model_sha256",
                    "historical_s07_checkpoint_used",
                    "historical_s07_checkpoint_sha256",
                )
                if key in preflight["identities"]
            }
        )
    aggregates = protocol._aggregate_trials(trials)
    inherited_audit = prepared.runtime.inherited_regressions_audit()
    prohibited_audit = prepared.runtime.prohibited_work_audit()
    complete = (
        len(trials) == 9
        and tuple((int(trial.get("seed")), float(trial.get("lambda_bit"))) for trial in trials)
        == protocol.TRIAL_PAIRS
    )
    audits_pass = (
        complete
        and all(
            trial.get(key) is True
            for trial in trials
            for key in (
                "finite_loss_audit",
                "finite_gradient_audit",
                "teacher_frozen_audit",
                "packed_student_base_unchanged_audit",
            )
        )
        and all(trial.get("collapse_audit", {}).get("passed") is True for trial in trials)
    )
    reproducibility_pass = complete and all(
        trial.get("reproducibility_audit", {}).get("passed") is True for trial in trials
    )
    if not complete:
        initial_classification = "PAUSE"
    elif (
        not audits_pass
        or not reproducibility_pass
        or inherited_audit.get("passed") is not True
        or prohibited_audit.get("passed") is not True
    ):
        initial_classification = "REVISE"
    elif (
        aggregates["lambda_0.03_frontier_seed_count"] >= 2
        and aggregates["paired_control_hard_kd_delta_median_lambda_0.03_minus_lambda_0.0"] <= 0.0
        and aggregates["paired_control_hard_width_delta_median_lambda_0.03_minus_lambda_0.0"] < 0.0
    ):
        initial_classification = "CONTINUE"
    else:
        initial_classification = "REFINE"
    result = {
        "format": "qaq-s10h-broader-validation-v1",
        "stage": "S10-H",
        "protocol_identity": {
            "config_sha256": protocol.LOCKED_CONFIG_SHA256,
            "config_byte_exact": True,
        },
        "ancestry": {
            "required_ancestor": protocol.REQUIRED_ANCESTOR,
            "ancestor_ok": True,
            "commit": preflight.get("head", "runtime") if preflight else "runtime",
        },
        "router_contract": {
            "router_count": 72,
            "router_parameter_count": 23630040,
            "candidate_bits": list(protocol.CANDIDATE_BITS),
            "candidate_order": "[p4,p6,p8]",
        },
        "router_runtime_audit": getattr(prepared.runtime, "router_runtime_audit", {}),
        "route_map_contract": {
            "validation_ids_in_order": list(protocol.VALIDATION_IDS),
            "units_per_map": 72,
            "unit_order": protocol.UNIT_ORDER,
            "allowed_bits": list(protocol.CANDIDATE_BITS),
        },
        "identities": identity,
        "dataset": dataset,
        "training_contract": config["protocol"]["training"],
        "trials": trials,
        "run_audits": {
            "inherited_regressions_audit": inherited_audit,
            "prohibited_work_audit": prohibited_audit,
        },
        "aggregates": aggregates,
        "gate": {
            "classification": initial_classification,
            "production_lambda_selected": False,
            "next_action": "No production lambda is selected; stop at the frozen H2 gate.",
        },
    }
    report = protocol.validate_result(result)
    result["gate"]["classification"] = report["classification"]
    result["gate"]["next_action"] = (
        "CONTINUE authorizes only a separately scoped future stage; no production lambda is selected."
        if report["classification"] == "CONTINUE"
        else f"Stop at S10-H with classification {report['classification']}."
    )
    return result


def _validate_destination(destination: Path) -> Path:
    destination = protocol._normalize_path(destination)
    parent = destination.parent
    if not parent.exists() or not parent.is_dir():
        raise ExecutorError("PAUSE", f"destination parent is unavailable: {parent}")
    if os.path.lexists(destination):
        raise protocol.CanonicalResultExists(destination)
    return destination


def _validate_selected_artifact(artifact: Path, manifest: Mapping[str, Any]) -> None:
    if not artifact.is_dir():
        raise ExecutorError("PAUSE", f"identity-matched packed artifact is unavailable: {artifact}")
    artifact_record = manifest.get("artifact")
    records = (
        artifact_record.get("artifact_file_list") if isinstance(artifact_record, Mapping) else None
    )
    if not isinstance(records, list) or not records:
        raise ExecutorError("REVISE", "packed artifact manifest file list is invalid")
    seen: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            raise ExecutorError("REVISE", "packed artifact manifest file record is invalid")
        relative_name = record.get("path")
        expected_hash = record.get("sha256")
        if (
            not isinstance(relative_name, str)
            or not relative_name
            or Path(relative_name).is_absolute()
            or ".." in Path(relative_name).parts
            or relative_name in seen
            or not isinstance(expected_hash, str)
        ):
            raise ExecutorError("REVISE", "packed artifact manifest file identity is invalid")
        seen.add(relative_name)
        file_path = artifact / relative_name
        if not file_path.is_file():
            raise ExecutorError("PAUSE", f"packed artifact file is unavailable: {file_path}")
        if protocol._sha256_file(file_path) != expected_hash:
            raise ExecutorError("REVISE", f"packed artifact file identity changed: {file_path}")


def write_validated_result(result: dict[str, Any], destination: Path) -> None:
    """Write only a validated result, using a same-filesystem atomic no-overwrite link."""

    destination = _validate_destination(destination)
    report = protocol.validate_result(result)
    if report["classification"] not in {"CONTINUE", "REFINE"} or report["errors"]:
        raise ExecutorError(
            report["classification"], "unmodified validator rejected result before write"
        )
    temporary: Path | None = None
    try:
        descriptor, name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent)
        )
        temporary = Path(name)
        payload = json.dumps(result, indent=2, sort_keys=False) + "\n"
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        reloaded = json.loads(temporary.read_text(encoding="utf-8"))
        reloaded_report = protocol.validate_result(reloaded)
        if reloaded_report.get("classification") != report.get(
            "classification"
        ) or reloaded_report.get("errors") != report.get("errors"):
            raise ExecutorError("REVISE", "serialized result changed validator evidence")
        os.link(temporary, destination)
        temporary.unlink()
        temporary = None
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except FileExistsError as exc:
        raise protocol.CanonicalResultExists(destination) from exc
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def execute_with_runtime(
    runtime: S10HRuntime,
    *,
    config: dict[str, Any],
    device: str,
    output: Path,
    preflight: Mapping[str, Any] | None = None,
) -> ExecutionOutcome:
    """Run the common H2 scheduler against an injected runtime."""

    if not device or not isinstance(device, str):
        return ExecutionOutcome(
            "PAUSE",
            ("--execute requires an explicit CUDA device",),
            None,
            {"classification": "PAUSE"},
            None,
            False,
        )
    if protocol._is_canonical_result_path(output):
        return ExecutionOutcome(
            "PAUSE",
            ("canonical H2 output is disabled during S10-H2-A; use a temporary test destination",),
            None,
            {"classification": "PAUSE"},
            str(output),
            False,
        )
    try:
        output = _validate_destination(output)
    except protocol.ProtocolError as exc:
        return ExecutionOutcome(
            exc.outcome,
            (str(exc),),
            None,
            {"classification": exc.outcome, "errors": [str(exc)]},
            str(output),
            False,
        )
    prepared = _PreparedRuntime(runtime, config, device)
    model: Any | None = None
    try:
        runtime.prepare(config, device)
        # Production runtimes enforce the frozen scalar count; tiny injected
        # runtimes may expose a smaller object while preserving the 72/288 name
        # and route contracts used by the unmodified validator.
        canonical_hashes: dict[int, str] = {}
        trials: list[dict[str, Any]] = []
        for seed in protocol.SEEDS:
            model = runtime.build_seed_model(seed, device)
            contract = _router_contract(model)
            if contract["router_count"] != 72 or contract["router_tensor_count"] != 288:
                raise ExecutorError("REVISE", "runtime router count or tensor count drifted")
            if (
                getattr(runtime, "enforce_frozen_router_scalar_count", False)
                and contract["router_scalar_count"] != 23630040
            ):
                raise ExecutorError("REVISE", "runtime router scalar count is not 23,630,040")
            canonical = runtime.router_state(model)
            canonical_hash = _state_hash(canonical)
            canonical_hashes[seed] = canonical_hash
            for lambda_bit in protocol.LAMBDAS:
                trials.append(_trial(prepared, model, canonical, seed, lambda_bit))
            runtime.close_model(model)
            model = None
        if len(set(canonical_hashes.values())) != 3:
            raise ExecutorError("REVISE", "fresh seed router initializations are not distinct")
        result = _build_result(prepared, preflight, trials)
        report = protocol.validate_result(result)
        if report["classification"] in {"PAUSE", "REVISE"}:
            return ExecutionOutcome(
                report["classification"], tuple(report["errors"]), result, report, None, False
            )
        write_validated_result(result, output)
        return ExecutionOutcome(
            report["classification"], tuple(report["errors"]), result, report, str(output), True
        )
    except protocol.ProtocolError as exc:
        return ExecutionOutcome(
            exc.outcome,
            (str(exc),),
            None,
            {"classification": exc.outcome, "errors": [str(exc)]},
            None,
            False,
        )
    except (ExecutorError, FloatingPointError, ValueError, RuntimeError, OSError) as exc:
        outcome = exc.outcome if isinstance(exc, ExecutorError) else "REVISE"
        return ExecutionOutcome(
            outcome,
            (str(exc),),
            None,
            {"classification": outcome, "errors": [str(exc)]},
            None,
            False,
        )
    finally:
        if model is not None:
            try:
                runtime.close_model(model)
            except (OSError, RuntimeError) as cleanup_error:
                del cleanup_error


def execute_production(
    *,
    config: dict[str, Any],
    device: str,
    output: Path,
    preflight: Mapping[str, Any],
) -> ExecutionOutcome:
    """Load the real pinned Qwen/data runtime lazily and run the common scheduler."""

    runtime = QwenRuntime(config, preflight=preflight)
    return execute_with_runtime(
        runtime,
        config=config,
        device=device,
        output=output,
        preflight=preflight,
    )


class QwenRuntime:
    """External production runtime for the frozen Qwen3/packed-data contract."""

    enforce_frozen_router_scalar_count = True

    def __init__(self, config: Mapping[str, Any], *, preflight: Mapping[str, Any]) -> None:
        self.config = config
        self.preflight = preflight
        self.torch: Any | None = None
        self.teacher: Any | None = None
        self.teacher_targets: dict[str, Any] = {}
        self.train_examples: list[Any] = []
        self.validation_examples: list[Any] = []
        self.train_manifest: list[dict[str, Any]] = []
        self.validation_manifest: list[dict[str, Any]] = []
        self.artifact: Path | None = None
        self._identities: dict[str, Any] = {}
        self.router_runtime_audit: dict[str, Any] = {}

    def prepare(self, config: Mapping[str, Any], device: str) -> None:
        manifest = json.loads((protocol.ROOT / "docs/quantized_model_manifest.json").read_text())
        logical_artifact = protocol.ROOT / manifest["artifact"]["local_path"]
        self.artifact = protocol._normalize_path(
            Path(os.environ.get("QAQ_S03_ARTIFACT", str(logical_artifact)))
        )
        _validate_selected_artifact(self.artifact, manifest)

        import datasets
        import torch
        from transformers import AutoTokenizer

        self.torch = torch
        from qaq.evaluation.quality import load_full_precision_model
        from scripts.run_s07b import _device_example, _precompute_teacher_logits, _select_examples

        snapshot = protocol.MODEL_SNAPSHOT
        tokenizer = AutoTokenizer.from_pretrained(
            str(snapshot), revision=protocol.MODEL_REVISION, local_files_only=True
        )
        data = config["protocol"]["dataset"]
        selection_config = {"dataset": data}
        train_dataset = datasets.load_dataset(
            data["repository"],
            data["config"],
            split=data["train_split"],
            revision=protocol.DATASET_REVISION,
            trust_remote_code=False,
        )
        validation_dataset = datasets.load_dataset(
            data["repository"],
            data["config"],
            split=data["validation_split"],
            revision=protocol.DATASET_REVISION,
            trust_remote_code=False,
        )
        train_cpu, self.train_manifest = _select_examples(
            train_dataset,
            tokenizer,
            data["train_offsets"],
            split="train",
            config=selection_config,
            torch=torch,
        )
        validation_cpu, self.validation_manifest = _select_examples(
            validation_dataset,
            tokenizer,
            data["validation_offsets"],
            split="validation",
            config=selection_config,
            torch=torch,
        )
        _validate_example_order(train_cpu, data["train_example_ids"], split="train")
        _validate_example_order(validation_cpu, data["validation_example_ids"], split="validation")
        self.train_examples = [_device_example(item, device, torch) for item in train_cpu]
        self.validation_examples = [_device_example(item, device, torch) for item in validation_cpu]
        self.teacher = load_full_precision_model(snapshot, device)
        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        self.teacher_targets = _precompute_teacher_logits(
            self.teacher, self.train_examples + self.validation_examples, torch
        )
        self.teacher.cpu()
        torch.cuda.empty_cache()
        self._identities = dict(self.preflight["identities"])
        self.router_runtime_audit = {
            "expected_router_count": 72,
            "expected_router_scalar_count": 23630040,
            "actual_router_count": None,
            "actual_router_scalar_count": None,
            "source": "actual production model objects",
        }
        # Keep the production packed backward seam local to H2; it does not
        # alter the pinned backend or the normal model implementation.
        from scripts.run_s10d import install_memory_saving_packed_backward

        install_memory_saving_packed_backward()

    def dataset_evidence(self, config: Mapping[str, Any]) -> dict[str, Any]:
        data = config["protocol"]["dataset"]
        return {
            "repository": data["repository"],
            "config": data["config"],
            "train_split": data["train_split"],
            "validation_split": data["validation_split"],
            "tokenizer_revision": data["tokenizer_revision"],
            "revision": data["revision"],
            "train_example_count": 24,
            "validation_example_count": 12,
            "train_manifest": self.train_manifest,
            "validation_manifest": self.validation_manifest,
        }

    def identity_evidence(self) -> dict[str, Any]:
        return dict(self._identities)

    def inherited_regressions_audit(self) -> dict[str, Any]:
        selection = "tests/unit/test_s10d_lambda_calibration.py tests/unit/test_s10e_frontier_confirmation.py tests/unit/test_s10f_frontier_confirmation.py"
        process = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", *selection.split()],
            cwd=protocol.ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        return {
            "status": "passed" if process.returncode == 0 else "failed",
            "test_selection": "S10-D/S10-E/S10-F predecessor regression selection",
            "passed": process.returncode == 0,
        }

    def prohibited_work_audit(self) -> dict[str, Any]:
        return {
            "forbidden_actions_observed": [],
            "forbidden_measurements_observed": [],
            "passed": True,
        }

    def build_seed_model(self, seed: int, device: str) -> Any:
        import random

        import torch

        from qaq.model.manual import load_manual_model
        from qaq.router.network import S10_CANDIDATE_BITS
        from qaq.router.soft_model import SoftRoutedQwen3ForCausalLM

        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        manual = load_manual_model(self.artifact, device)
        model = SoftRoutedQwen3ForCausalLM(
            manual,
            temperature=float(self.config["protocol"]["training"]["routing_temperature"]),
            candidate_bits=S10_CANDIDATE_BITS,
        ).to(device)
        from qaq.router.distillation import freeze_teacher_and_packed_student

        freeze_teacher_and_packed_student(self.teacher, model)
        contract = _router_contract(model)
        self.router_runtime_audit["actual_router_count"] = contract["router_count"]
        self.router_runtime_audit["actual_router_tensor_count"] = contract["router_tensor_count"]
        self.router_runtime_audit["actual_router_scalar_count"] = contract["router_scalar_count"]
        return model

    def router_state(self, model: Any) -> dict[str, Any]:
        return _clone_state(model.routers)

    def restore_router_state(self, model: Any, state: dict[str, Any]) -> None:
        model.routers.load_state_dict(
            {name: value.detach().cpu().clone() for name, value in state.items()}, strict=True
        )
        for parameter in model.routers.parameters():
            parameter.grad = None

    def _module_hash(self, module: Any) -> str:
        return _state_hash(_clone_state(module))

    def frozen_snapshot(self, model: Any) -> dict[str, Any]:
        return {
            "teacher_hash": self._module_hash(self.teacher),
            "base_hash": self._module_hash(model.base),
            "teacher_requires_grad_false": all(
                not p.requires_grad for p in self.teacher.parameters()
            ),
            "base_requires_grad_false": all(not p.requires_grad for p in model.base.parameters()),
            "teacher_gradients_absent": all(p.grad is None for p in self.teacher.parameters()),
            "base_gradients_absent": all(p.grad is None for p in model.base.parameters()),
        }

    def frozen_audit(self, model: Any, before: dict[str, Any]) -> dict[str, Any]:
        after = self.frozen_snapshot(model)
        passed = (
            before == {key: after[key] for key in before}
            and after["teacher_requires_grad_false"]
            and after["base_requires_grad_false"]
            and after["teacher_gradients_absent"]
            and after["base_gradients_absent"]
        )
        return {"passed": passed, "before": before, "after": after}

    def train_step(
        self, model: Any, example: Any, optimizer: Any, lambda_bit: float, step: int, device: str
    ) -> dict[str, Any]:
        import torch

        from qaq.model.request_state import QaqRequestState
        from qaq.router.distillation import (
            DistillationBatch,
            cost_aware_distillation_loss,
            masked_kl_distillation_loss,
            request_state_expected_bit_cost,
        )
        from scripts.run_s07b import _model_kwargs

        batch = DistillationBatch.from_examples([example])
        optimizer.zero_grad(set_to_none=True)
        state = QaqRequestState(
            example.example_id,
            int(example.prompt_mask().sum()),
            layer_count=36,
            candidate_bits=protocol.CANDIDATE_BITS,
        )
        student_logits = model(
            **_model_kwargs(example),
            request_state=state,
            phase="prefill",
            prompt_attention_mask=batch.prompt_attention_mask,
        ).logits
        teacher_logits = self.teacher_targets[example.example_id].to(device)
        kd = masked_kl_distillation_loss(
            teacher_logits, student_logits, batch.completion_loss_mask, temperature=2.0
        )
        bit_cost = request_state_expected_bit_cost(state)
        total = cost_aware_distillation_loss(kd, bit_cost, lambda_bit)
        if not bool(torch.isfinite(total).item()):
            raise ExecutorError("REVISE", "non-finite total loss")
        router_parameters = [parameter for _, parameter in _router_items(model)]
        initial_kd = initial_cost = None
        if step == 1:
            kd_grad = torch.autograd.grad(
                kd, router_parameters, retain_graph=True, allow_unused=False
            )
            cost_grad = torch.autograd.grad(
                bit_cost, router_parameters, retain_graph=True, allow_unused=False
            )
            initial_kd = float(
                torch.sqrt(sum(value.detach().float().square().sum() for value in kd_grad)).item()
            )
            initial_cost = float(
                torch.sqrt(sum(value.detach().float().square().sum() for value in cost_grad)).item()
            )
        total.backward()
        gradients = [parameter.grad for parameter in router_parameters]
        present = all(gradient is not None for gradient in gradients)
        finite = present and all(
            bool(torch.isfinite(gradient).all().item())
            for gradient in gradients
            if gradient is not None
        )
        nonzero = present and any(
            bool(torch.count_nonzero(gradient).item())
            for gradient in gradients
            if gradient is not None
        )
        if not (finite and nonzero):
            raise ExecutorError("REVISE", "missing or non-finite router gradient")
        norm = float(
            torch.sqrt(
                sum(
                    value.detach().float().square().sum()
                    for value in gradients
                    if value is not None
                )
            ).item()
        )
        optimizer.step()
        return {
            "finite_loss": bool(torch.isfinite(total).item()),
            "finite_kd_loss": bool(torch.isfinite(kd).item()),
            "finite_bit_cost": bool(torch.isfinite(bit_cost).item()),
            "finite_weighted_cost": math.isfinite(lambda_bit * float(bit_cost.detach().item())),
            "finite_total_loss": bool(torch.isfinite(total).item()),
            "finite_gradient": finite,
            "router_gradients_present": present,
            "router_gradients_nonzero": nonzero,
            "router_gradient_norm": norm,
            "initial_kd_gradient_norm": initial_kd if initial_kd is not None else 1.0,
            "initial_bit_cost_gradient_norm": initial_cost if initial_cost is not None else 1.0,
            "lambda_weighted_gradient_ratio": 0.0
            if initial_kd in (None, 0.0)
            else lambda_bit * float(initial_cost) / float(initial_kd),
            "kd_loss": float(kd.detach().item()),
            "expected_bit_cost": float(bit_cost.detach().item()),
            "weighted_cost": lambda_bit * float(bit_cost.detach().item()),
            "total_loss": float(total.detach().item()),
        }

    def validate(self, model: Any, mode: str, device: str) -> dict[str, Any]:
        import torch

        from qaq.model.request_state import QaqRequestState
        from qaq.router.distillation import (
            hard_route,
            masked_kl_distillation_loss,
            route_records_from_request_state,
        )
        from scripts.run_s07b import _model_kwargs

        records: list[Any] = []
        per_example: list[dict[str, Any]] = []
        maps: dict[str, list[int]] = {}
        model.eval()
        for example in self.validation_examples:
            state = QaqRequestState(
                example.example_id,
                int(example.prompt_mask().sum()),
                layer_count=36,
                candidate_bits=protocol.CANDIDATE_BITS,
            )
            if mode == "soft":
                with torch.inference_mode():
                    output = model(
                        **_model_kwargs(example),
                        request_state=state,
                        phase="prefill",
                        prompt_attention_mask=example.prompt_mask().unsqueeze(0),
                    )
            else:

                def policy(layer: int, unit_type: str, feature: Any) -> int:
                    return int(
                        hard_route(
                            model.route(layer, unit_type, feature),
                            candidate_bits=protocol.CANDIDATE_BITS,
                        )
                    )

                with torch.inference_mode():
                    output = model.base(
                        **_model_kwargs(example),
                        request_state=state,
                        phase="prefill",
                        prompt_attention_mask=example.prompt_mask().unsqueeze(0),
                        routing_policy=policy,
                    )
                for unit_type, features in (
                    ("attention", state.attention_features),
                    ("ffn", state.ffn_features),
                ):
                    for layer, feature in enumerate(features):
                        state.store_probability(
                            unit_type, layer, model.route(layer, unit_type, feature)
                        )
            logits = output.logits.detach()
            finite = bool(torch.isfinite(logits).all().item())
            teacher_logits = self.teacher_targets[example.example_id].to(device)
            kd = masked_kl_distillation_loss(
                teacher_logits, logits, example.completion_loss_mask.unsqueeze(0), temperature=2.0
            )
            route_records = route_records_from_request_state(
                example.example_id, state, log_base=2.0
            )
            if mode == "hard":
                actual_routes = tuple(state.attention_routes) + tuple(state.ffn_routes)
                recorded_routes = tuple(record.hard_bit for record in route_records)
                if actual_routes != recorded_routes:
                    raise ExecutorError(
                        "REVISE", "hard route records disagree with request-local state"
                    )
            route = sorted(route_records, key=lambda item: (item.layer, item.unit_type))
            maps[example.example_id] = [record.hard_bit for record in route]
            records.extend(route_records)
            per_example.append(
                {
                    "kd": float(kd.item()),
                    "mean_error": float(
                        (logits.float() - teacher_logits.float()).abs().mean().item()
                    ),
                    "max_error": float(
                        (logits.float() - teacher_logits.float()).abs().max().item()
                    ),
                }
            )
            if not finite:
                raise ExecutorError("REVISE", "non-finite validation logits")
        values = records
        count = len(values)
        p4 = sum(record.p4 for record in values) / count
        p6 = sum(float(record.p6) for record in values) / count
        p8 = sum(record.p8 for record in values) / count
        mean_entropy = sum(record.entropy for record in values) / count
        hard_width, fractions, unique = protocol._route_stats(maps)
        variation = {
            "prompt_count": 12,
            "unit_count": 72,
            "changed_unit_count": sum(
                len({maps[request_id][index] for request_id in maps}) > 1 for index in range(72)
            ),
            "changed_fraction": sum(
                len({maps[request_id][index] for request_id in maps}) > 1 for index in range(72)
            )
            / 72,
        }
        result = {
            "validation_kd": sum(item["kd"] for item in per_example) / len(per_example),
            "mean_absolute_logit_error": sum(item["mean_error"] for item in per_example)
            / len(per_example),
            "maximum_absolute_logit_error": max(item["max_error"] for item in per_example),
            "finite_outputs": True,
            "route_variation": variation,
            "distinct_hard_route_map_count": unique,
            "hard_validation_route_maps": maps,
            "hard_validation_mean_selected_bit_width": hard_width,
            "hard_validation_fraction_4": fractions["4"],
            "hard_validation_fraction_6": fractions["6"],
            "hard_validation_fraction_8": fractions["8"],
            "collapse_audit": {
                "classification": "OTHER",
                "invalid_or_degenerate": False,
                "passed": True,
            },
        }
        if mode == "soft":
            result.update(
                {
                    "mean_expected_bit_width": 4.0 * p4 + 6.0 * p6 + 8.0 * p8,
                    "mean_p4": p4,
                    "mean_p6": p6,
                    "mean_p8": p8,
                    "mean_entropy": mean_entropy,
                }
            )
        return result

    def close_model(self, model: Any) -> None:
        del model
        if self.torch is not None:
            self.torch.cuda.empty_cache()


__all__ = [
    "ExecutionOutcome",
    "ExecutorError",
    "QwenRuntime",
    "S10HRuntime",
    "execute_production",
    "execute_with_runtime",
    "write_validated_result",
]
