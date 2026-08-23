#!/usr/bin/env python3
"""Calibrate the S10-D normalized bit-cost coefficient on the locked S07 run."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import subprocess
import sys
from collections.abc import Iterable
from itertools import combinations
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from qaq.model.request_state import QaqRequestState
from qaq.router.baseline_training import (
    _device_example,
    _model_kwargs,
    _precompute_teacher_logits,
    _seed_everything,
    _select_examples,
)
from qaq.router.distillation import (
    DistillationBatch,
    build_router_optimizer,
    cost_aware_distillation_loss,
    freeze_teacher_and_packed_student,
    hard_route,
    masked_kl_distillation_loss,
    request_state_expected_bit_cost,
    route_records_from_request_state,
    route_statistics,
)
from qaq.router.network import THREE_WAY_CANDIDATE_BITS

CONFIG_PATH = ROOT / "configs/s10d_lambda_calibration.json"
MANIFEST_PATH = ROOT / "docs/quantized_model_manifest.json"
MODEL_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
MODEL_SNAPSHOT = Path(
    os.environ.get(
        "QAQ_MODEL_SNAPSHOT",
        "~/.cache/huggingface/hub/models--Qwen--Qwen3-4B/"
        "snapshots/1cfa9a7208912126459214e8b04321603b3df60c",
    )
).expanduser()
EXPECTED_MODEL_SNAPSHOT = (
    Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-4B" / "snapshots" / MODEL_REVISION
)
ANY_PRECISION_REVISION = "a3257d02740cc5757c78673da534b0630ff3a4ea"
DATASET_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
CANDIDATE_BITS = THREE_WAY_CANDIDATE_BITS
LAYER_COUNT = 36
MIN_FREE_GPU_BYTES = 20 * 1024**3
_OPTIMIZER_CONSTRUCTION_SERIALS = itertools.count(1)
LOCKED_CONFIG_SHA256 = "22649ec4cdafa7a8ff669f72c159c7fbfbaa33ecea50888a953301a8225bb5c1"


def _load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    config_bytes = path.read_bytes()
    if hashlib.sha256(config_bytes).hexdigest() != LOCKED_CONFIG_SHA256:
        raise RuntimeError("REVISE: S10-D config differs from the locked protocol")
    config = json.loads(config_bytes)
    if tuple(config["candidate_bits"]) != CANDIDATE_BITS:
        raise RuntimeError("REVISE: S10-D candidate ordering is not (4,6,8)")
    if config["dataset"]["revision"] != DATASET_REVISION:
        raise RuntimeError("REVISE: locked dataset revision differs from S10-D")
    if config["model"]["revision"] != MODEL_REVISION:
        raise RuntimeError("REVISE: locked model revision differs from S10-D")
    if config["model"]["any_precision_revision"] != ANY_PRECISION_REVISION:
        raise RuntimeError("REVISE: locked Any-Precision revision differs from S10-D")
    if (
        config["model"]["router_count"] != 72
        or config["model"]["router_parameter_count"] != 23_630_040
    ):
        raise RuntimeError("REVISE: locked three-way router count differs from S10-D")
    if config["lambda_grid"] != [0.0, 0.003, 0.01, 0.03, 0.1]:
        raise RuntimeError("REVISE: S10-D lambda grid is not the locked grid")
    return config


def _tensor_hash(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    return hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest()


def _state_snapshot(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().contiguous().clone() for name, value in state.items()}


def router_only_state(router: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Clone only router state; teacher and packed base state never enters it."""

    return _state_snapshot(router.state_dict())


def router_state_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(str(value.dtype).encode())
        digest.update(b"\0")
        digest.update(repr(tuple(value.shape)).encode())
        digest.update(b"\0")
        digest.update(value.view(torch.uint8).numpy().tobytes())
        digest.update(b"\n")
    return digest.hexdigest()


def restore_router_state(router: torch.nn.Module, state: dict[str, torch.Tensor]) -> str:
    """Reload an exact canonical router-only state and return its hash."""

    router.load_state_dict(_state_snapshot(state), strict=True)
    for parameter in router.parameters():
        parameter.grad = None
    restored = router_only_state(router)
    actual_hash = router_state_hash(restored)
    expected_hash = router_state_hash(state)
    if actual_hash != expected_hash:
        raise RuntimeError("REVISE: canonical router state did not reload exactly")
    return actual_hash


def validate_canonical_initialization(model: torch.nn.Module, config: dict[str, Any]) -> None:
    if getattr(model, "router_count", None) != config["model"]["router_count"]:
        raise RuntimeError("REVISE: three-way router count is not 72")
    if getattr(model, "router_parameter_count", None) != config["model"]["router_parameter_count"]:
        raise RuntimeError("REVISE: three-way router scalar count is not 23,630,040")
    if tuple(model.candidate_bits) != CANDIDATE_BITS:
        raise RuntimeError("REVISE: three-way candidate ordering is not explicit")


def fresh_router_optimizer(model: torch.nn.Module, config: dict[str, Any]):
    """Build a new AdamW with no state inherited from another lambda trial."""

    optimizer, audit = build_router_optimizer(
        model,
        lr=float(config["training"]["learning_rate"]),
        optimizer_cls=torch.optim.AdamW,
        weight_decay=float(config["training"]["weight_decay"]),
    )
    if optimizer.state:
        raise RuntimeError("REVISE: fresh lambda optimizer unexpectedly has state")
    if audit.scalar_count != config["model"]["router_parameter_count"]:
        raise RuntimeError("REVISE: lambda optimizer scalar count is not router-only")
    return optimizer, audit


def _module_state_hash(module: torch.nn.Module) -> str:
    return router_state_hash(_state_snapshot(module.state_dict()))


def _optimizer_runtime_evidence(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    audit: Any,
    construction_serial: int,
) -> dict[str, Any]:
    expected = [(name, parameter) for name, parameter in model.named_parameters() if name.startswith("routers.")]
    expected_ids = {id(parameter) for _, parameter in expected}
    actual_parameters = [
        parameter for group in optimizer.param_groups for parameter in group["params"]
    ]
    actual_ids = [id(parameter) for parameter in actual_parameters]
    expected_names = tuple(sorted(name for name, _ in expected))
    actual_names = tuple(
        sorted(name for name, parameter in expected if id(parameter) in set(actual_ids))
    )
    return {
        "expected_router_parameter_count": len(expected),
        "actual_optimizer_parameter_count": len(actual_parameters),
        "expected_router_parameter_names_sha256": hashlib.sha256(
            "\\n".join(expected_names).encode()
        ).hexdigest(),
        "actual_optimizer_parameter_names_sha256": hashlib.sha256(
            "\\n".join(actual_names).encode()
        ).hexdigest(),
        "unexpected_optimizer_parameter_count": len(set(actual_ids) - expected_ids),
        "missing_router_parameter_count": len(expected_ids - set(actual_ids)),
        "duplicate_optimizer_parameter_count": len(actual_ids) - len(set(actual_ids)),
        "runtime_identity_based_membership_result": (
            len(actual_ids) == len(set(actual_ids)) and set(actual_ids) == expected_ids
        ),
        "expected_router_parameter_names": list(expected_names),
        "actual_optimizer_parameter_names": list(actual_names),
        "optimizer_construction_serial": construction_serial,
        "optimizer_state_entry_count_before_first_step": len(optimizer.state),
        "optimizer_state_entry_count_before_training_begins": len(optimizer.state),
        "router_only_optimizer_audit": (
            audit.included_name_prefixes == ("routers.",)
            and len(actual_ids) == len(set(actual_ids))
            and set(actual_ids) == expected_ids
        ),
        "fresh_adamw_audit": isinstance(optimizer, torch.optim.AdamW) and not optimizer.state,
    }


def _finite(value: torch.Tensor) -> bool:
    return bool(torch.isfinite(value.detach()).all().item())


class _RecomputedPackedPath(torch.autograd.Function):
    """Recompute frozen packed weights in backward instead of saving dense weights."""

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, qweight: torch.Tensor, lut: torch.Tensor, bits: int):
        from qaq.loading.loader import execute_packed_linear, pinned_backend

        dequant_kbit, matmul_kbit = pinned_backend()
        with torch.no_grad():
            output = execute_packed_linear(
                inputs,
                qweight,
                lut,
                int(bits),
                dequant_kbit=dequant_kbit,
                matmul_kbit=matmul_kbit,
            )
        ctx.qweight = qweight
        ctx.lut = lut
        ctx.bits = int(bits)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        from qaq.loading.loader import pinned_backend

        dequant_kbit, _ = pinned_backend()
        with torch.no_grad():
            weight = dequant_kbit(ctx.qweight, ctx.lut, ctx.bits)
            grad_inputs = torch.matmul(grad_output, weight)
        return grad_inputs, None, None, None


def _memory_saving_mix(
    packed: torch.nn.Module,
    inputs: torch.Tensor,
    probabilities: torch.Tensor,
    *,
    candidate_bits: tuple[int, ...],
) -> torch.Tensor:
    """Use the existing packed execution helper with a recomputed backward path."""

    from qaq.router.network import validate_probabilities
    from qaq.router.soft_linear import mix_packed_outputs

    validate_probabilities(probabilities, candidate_bits, context="soft routing probabilities")
    if not hasattr(packed, "qweight"):
        return mix_packed_outputs(packed, inputs, probabilities, candidate_bits=candidate_bits)
    outputs = []
    for bits in candidate_bits:
        output = _RecomputedPackedPath.apply(
            inputs, packed.qweight, packed._buffers[f"lut{bits}"], int(bits)
        )
        if packed.bias is not None:
            output = output + packed.bias
        outputs.append(output)
    probabilities = probabilities.to(dtype=outputs[0].dtype)
    output = torch.zeros_like(outputs[0])
    for index, packed_output in enumerate(outputs):
        weight = probabilities[..., index]
        while weight.ndim < packed_output.ndim:
            weight = weight.unsqueeze(-1)
        output = output + weight * packed_output
    if not _finite(output):
        raise FloatingPointError("S10-D memory-saving packed mixture produced NaN or Inf")
    return output


def install_memory_saving_packed_backward() -> None:
    """Install only for this runner; production routing code remains untouched."""

    import qaq.model.manual as manual_module

    manual_module.mix_packed_outputs = _memory_saving_mix


def _gradient_norm(gradients: Iterable[torch.Tensor | None]) -> float:
    values = tuple(gradients)
    if not values or any(value is None for value in values):
        raise FloatingPointError("REVISE: missing or non-finite router gradient")
    if any(not _finite(value) for value in values):
        raise FloatingPointError("REVISE: missing or non-finite router gradient")
    total = sum(value.detach().float().square().sum() for value in values)
    norm = torch.sqrt(total)
    if not _finite(norm):
        raise FloatingPointError("REVISE: non-finite router gradient norm")
    return float(norm.item())


def _route_map(records: Iterable[Any]) -> tuple[int, ...]:
    ordered = sorted(records, key=lambda record: (record.layer, record.unit_type))
    return tuple(int(record.hard_bit) for record in ordered)


def _probability_means(records: Iterable[Any]) -> dict[str, float]:
    values = tuple(records)
    count = len(values)
    return {
        "p4": sum(record.p4 for record in values) / count,
        "p6": sum(float(record.p6) for record in values) / count,
        "p8": sum(record.p8 for record in values) / count,
    }


def _route_distance(maps: Iterable[tuple[int, ...]]) -> float:
    values = tuple(maps)
    if len(values) < 2:
        return 0.0
    distances = [
        sum(left != right for left, right in zip(first, second, strict=True)) / len(first)
        for first, second in combinations(values, 2)
    ]
    return sum(distances) / len(distances)


def summarize_route_records(
    records: Iterable[Any],
    *,
    validation_ids: tuple[str, ...],
    logits_finite: bool,
    entropy_log_base: float = 2.0,
) -> dict[str, Any]:
    values = tuple(records)
    stats = route_statistics(values, entropy_log_base=entropy_log_base)
    maps = {
        request_id: list(_route_map(record for record in values if record.request_id == request_id))
        for request_id in validation_ids
    }
    probabilities = _probability_means(values)
    stats.update(
        {
            "finite_logits": bool(logits_finite),
            "prompt_to_prompt_route_distance": _route_distance(
                tuple(tuple(route) for route in maps.values())
            ),
            "mean_p4": probabilities["p4"],
            "mean_p6": probabilities["p6"],
            "mean_p8": probabilities["p8"],
            "mean_expected_bit_width": 4.0 * probabilities["p4"]
            + 6.0 * probabilities["p6"]
            + 8.0 * probabilities["p8"],
            "mean_hard_selected_bit_width": sum(record.hard_bit for record in values) / len(values),
            "hard_fraction_sum": sum(stats[f"hard_fraction_{bit}"] for bit in CANDIDATE_BITS),
            "unique_hard_route_map_count": len({tuple(route) for route in maps.values()}),
            "per_validation_route_maps": maps,
            "any_validation_decision_selects_6": any(6 in route for route in maps.values()),
            "route_logs": [record.to_dict() for record in values],
        }
    )
    return stats


def static_mode_name(bits: int) -> str:
    """Return the explicit static runner mode, including the S10-D six-bit mode."""

    if bits not in CANDIDATE_BITS:
        raise ValueError(f"static mode supports exactly {CANDIDATE_BITS}; got {bits}")
    return f"static{bits}"


def classify_collapse(stats: dict[str, Any], *, collapse_fraction: float = 0.95) -> str:
    """Apply the configured collapse labels, then S07 observational labels."""

    if stats["hard_fraction_4"] >= collapse_fraction:
        return "COLLAPSED_TO_4"
    if stats["hard_fraction_6"] >= collapse_fraction:
        return "COLLAPSED_TO_6"
    if stats["hard_fraction_8"] >= collapse_fraction:
        return "COLLAPSED_TO_8"
    variation = stats["route_variation_across_prompts"]
    if (
        variation["changed_fraction"] > 0
        and stats.get("prompt_to_prompt_route_distance", 0.0) >= 0.05
    ):
        return "ADAPTIVE_OBSERVED"
    if variation["changed_fraction"] == 0:
        return "PROMPT_INVARIANT"
    return "OTHER"


def _pareto_frontier(
    points: Iterable[dict[str, Any]], *, kd_key: str, width_key: str
) -> list[dict[str, Any]]:
    """Return deterministic non-dominated points on (KD, width), lower is better."""

    ordered = sorted(
        (
            {
                "lambda": float(point["lambda"]),
                "validation_kd_loss": float(point[kd_key]),
                "width": float(point[width_key]),
            }
            for point in points
        ),
        key=lambda point: (point["validation_kd_loss"], point["width"], point["lambda"]),
    )
    frontier: list[dict[str, Any]] = []
    for point in ordered:
        dominated = any(
            other["validation_kd_loss"] <= point["validation_kd_loss"]
            and other["width"] <= point["width"]
            and (
                other["validation_kd_loss"] < point["validation_kd_loss"]
                or other["width"] < point["width"]
            )
            for other in ordered
        )
        if not dominated:
            frontier.append(point)
    return frontier


def pareto_frontiers(trials: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    values = tuple(trials)
    soft_points = [
        {
            "lambda": trial["lambda"],
            "validation_kd_loss": trial["soft"]["validation_kd_loss"],
            "mean_expected_bit_width": trial["soft"]["mean_expected_bit_width"],
        }
        for trial in values
    ]
    hard_points = [
        {
            "lambda": trial["lambda"],
            "validation_kd_loss": trial["hard"]["validation_kd_loss"],
            "mean_hard_selected_bit_width": trial["hard"]["mean_hard_selected_bit_width"],
        }
        for trial in values
    ]
    return {
        "soft": _pareto_frontier(
            soft_points,
            kd_key="validation_kd_loss",
            width_key="mean_expected_bit_width",
        ),
        "hard": _pareto_frontier(
            hard_points,
            kd_key="validation_kd_loss",
            width_key="mean_hard_selected_bit_width",
        ),
    }


def _mean_absolute_error(logits: torch.Tensor, teacher_logits: torch.Tensor) -> tuple[float, float]:
    values = (logits.float() - teacher_logits.float()).abs()
    return float(values.mean().item()), float(values.max().item())


def _evaluate_static(
    static_model: torch.nn.Module,
    examples: list[Any],
    teacher_targets: dict[str, torch.Tensor],
    device: str,
    config: dict[str, Any],
    torch_module: Any,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for bits in CANDIDATE_BITS:
        mode = static_mode_name(bits)
        from qaq.model.static import set_static_precision

        set_static_precision(static_model, bits)
        per_example = []
        for example in examples:
            with torch_module.inference_mode():
                logits = static_model(**_model_kwargs(example)).logits.detach()
            teacher_logits = teacher_targets[example.example_id].to(device)
            finite = bool(torch_module.isfinite(logits).all().item())
            if not finite:
                raise FloatingPointError(f"REVISE: non-finite {mode} logits")
            mean_error, max_error = _mean_absolute_error(logits, teacher_logits)
            kd = masked_kl_distillation_loss(
                teacher_logits,
                logits,
                example.completion_loss_mask.unsqueeze(0),
                temperature=float(config["training"]["distillation_temperature"]),
            )
            if not _finite(kd):
                raise FloatingPointError(f"REVISE: non-finite {mode} masked KD")
            per_example.append(
                {
                    "example_id": example.example_id,
                    "masked_kd_loss": float(kd.item()),
                    "mean_absolute_logit_error": mean_error,
                    "maximum_absolute_logit_error": max_error,
                    "finite_logits": finite,
                }
            )
        result[mode] = {
            "precision": bits,
            "count": len(per_example),
            "masked_kd_loss": sum(item["masked_kd_loss"] for item in per_example)
            / len(per_example),
            "mean_absolute_logit_error": sum(
                item["mean_absolute_logit_error"] for item in per_example
            )
            / len(per_example),
            "maximum_absolute_logit_error": max(
                item["maximum_absolute_logit_error"] for item in per_example
            ),
            "finite_logits": all(item["finite_logits"] for item in per_example),
            "per_example": per_example,
        }
    return result


def _fill_hard_probabilities(student: torch.nn.Module, state: QaqRequestState) -> None:
    for unit_type, features in (
        ("attention", state.attention_features),
        ("ffn", state.ffn_features),
    ):
        for layer, feature in enumerate(features):
            if feature is None:
                raise RuntimeError(f"REVISE: missing hard-route {unit_type} feature {layer}")
            state.store_probability(unit_type, layer, student.route(layer, unit_type, feature))


def _evaluate_learned(
    student: torch.nn.Module,
    examples: list[Any],
    teacher_targets: dict[str, torch.Tensor],
    device: str,
    mode: str,
    config: dict[str, Any],
    torch_module: Any,
) -> dict[str, Any]:
    if mode not in ("soft", "hard"):
        raise ValueError(mode)
    records = []
    per_example = []
    validation_ids = tuple(example.example_id for example in examples)
    student.eval()
    for example in examples:
        kwargs = _model_kwargs(example)
        state = QaqRequestState(
            example.example_id,
            int(example.prompt_mask().sum()),
            layer_count=LAYER_COUNT,
            candidate_bits=CANDIDATE_BITS,
        )
        with torch_module.inference_mode():
            if mode == "soft":
                output = student(
                    **kwargs,
                    request_state=state,
                    phase="prefill",
                    prompt_attention_mask=example.prompt_mask().unsqueeze(0),
                )
            else:

                def policy(layer: int, unit_type: str, feature: torch.Tensor) -> int:
                    probabilities = student.route(layer, unit_type, feature)
                    return int(hard_route(probabilities, candidate_bits=CANDIDATE_BITS))

                output = student.base(
                    **kwargs,
                    request_state=state,
                    phase="prefill",
                    prompt_attention_mask=example.prompt_mask().unsqueeze(0),
                    routing_policy=policy,
                )
        logits = output.logits.detach()
        teacher_logits = teacher_targets[example.example_id].to(device)
        finite = bool(torch_module.isfinite(logits).all().item())
        if not finite:
            raise FloatingPointError(f"REVISE: non-finite learned {mode} logits")
        if mode == "hard":
            _fill_hard_probabilities(student, state)
            state.assert_complete()
        else:
            state.assert_soft_complete()
        route_records = route_records_from_request_state(
            example.example_id,
            state,
            log_base=float(config["evaluation"]["entropy_log_base"]),
        )
        if mode == "hard":
            actual = tuple(state.attention_routes) + tuple(state.ffn_routes)
            recorded = tuple(record.hard_bit for record in route_records)
            if actual != recorded:
                raise RuntimeError("REVISE: hard route records disagree with actual request state")
        records.extend(route_records)
        kd = masked_kl_distillation_loss(
            teacher_logits,
            logits,
            example.completion_loss_mask.unsqueeze(0),
            temperature=float(config["training"]["distillation_temperature"]),
        )
        if not _finite(kd):
            raise FloatingPointError(f"REVISE: non-finite learned {mode} masked KD")
        mean_error, max_error = _mean_absolute_error(logits, teacher_logits)
        per_example.append(
            {
                "example_id": example.example_id,
                "masked_kd_loss": float(kd.item()),
                "mean_absolute_logit_error": mean_error,
                "maximum_absolute_logit_error": max_error,
                "finite_logits": finite,
                "route_map": list(_route_map(route_records)),
            }
        )
    summary = summarize_route_records(
        records,
        validation_ids=validation_ids,
        logits_finite=True,
        entropy_log_base=float(config["evaluation"]["entropy_log_base"]),
    )
    summary.update(
        {
            "count": len(per_example),
            "validation_kd_loss": sum(item["masked_kd_loss"] for item in per_example)
            / len(per_example),
            "mean_absolute_logit_error": sum(
                item["mean_absolute_logit_error"] for item in per_example
            )
            / len(per_example),
            "maximum_absolute_logit_error": max(
                item["maximum_absolute_logit_error"] for item in per_example
            ),
            "per_example": per_example,
            "collapse_classification": (
                classify_collapse(
                    summary,
                    collapse_fraction=float(
                        config["adaptive_extensions"]["low_lambda"]["trigger_collapse_fraction"]
                    ),
                )
                if mode == "hard"
                else None
            ),
        }
    )
    return summary


def _baseline_deltas(
    trial: dict[str, Any], baseline: dict[str, Any]
) -> dict[str, dict[str, float]]:
    metric_names = (
        "validation_kd_loss",
        "mean_absolute_logit_error",
        "maximum_absolute_logit_error",
        "mean_expected_bit_width",
        "mean_hard_selected_bit_width",
        "mean_entropy",
        "mean_p4",
        "mean_p6",
        "mean_p8",
        "hard_fraction_4",
        "hard_fraction_6",
        "hard_fraction_8",
        "route_variation_changed_fraction",
        "unique_hard_route_map_count",
    )
    result: dict[str, dict[str, float]] = {}
    for mode in ("soft", "hard"):
        values: dict[str, float] = {}
        for name in metric_names:
            if name == "route_variation_changed_fraction":
                current = trial[mode]["route_variation_across_prompts"]["changed_fraction"]
                reference = baseline[mode]["route_variation_across_prompts"]["changed_fraction"]
            elif name == "unique_hard_route_map_count":
                current = trial[mode]["unique_hard_route_map_count"]
                reference = baseline[mode]["unique_hard_route_map_count"]
            elif name in trial[mode] and name in baseline[mode]:
                current = trial[mode][name]
                reference = baseline[mode][name]
            else:
                continue
            values[name] = float(current) - float(reference)
        result[mode] = values
    return result


def _git_commit() -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _validate_model_snapshot(path: Path) -> None:
    if path.resolve(strict=False) != EXPECTED_MODEL_SNAPSHOT:
        raise SystemExit(
            "PAUSE: QAQ_MODEL_SNAPSHOT must be the exact pinned Hugging Face snapshot path"
        )
    if not path.is_dir():
        raise SystemExit(f"PAUSE: exact pinned teacher snapshot is unavailable: {path}")


def _environment(device: str, free_bytes: int) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(torch.device(device))
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "device": device,
        "gpu_name": properties.name,
        "gpu_total_bytes": int(properties.total_memory),
        "gpu_free_bytes_before_experiment": int(free_bytes),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "preflight_command": "source ~/.venv/bin/activate && which python && python --version && nvidia-smi",
    }


def _identity(manifest: dict[str, Any], artifact: Path) -> dict[str, Any]:
    from qaq.model.static import file_sha256, source_commit

    artifact_hash = file_sha256(artifact / "pytorch_model.bin")
    expected = manifest["artifact"]["checkpoint_hashes"]["pytorch_model.bin"]
    if artifact_hash != expected:
        raise RuntimeError("PAUSE: identity-matched packed artifact hash is unavailable")
    if source_commit() != ANY_PRECISION_REVISION:
        raise RuntimeError("PAUSE: pinned Any-Precision source is unavailable or dirty")
    return {
        "model_repository": manifest["source_model"]["repository"],
        "model_revision": MODEL_REVISION,
        "tokenizer_revision": MODEL_REVISION,
        "any_precision_revision": ANY_PRECISION_REVISION,
        "packed_artifact": str(artifact.relative_to(ROOT)),
        "packed_artifact_pytorch_model_sha256": artifact_hash,
        "historical_s07_checkpoint_used": False,
        "historical_s07_checkpoint_sha256": "08bf646f19759c0d7949e159bdbe4f96bbea737204b96f8760d205c8d6fd1949",
        "artifact_manifest_hash": hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest(),
    }


def _run_trial(
    student: torch.nn.Module,
    canonical_state: dict[str, torch.Tensor],
    lambda_bit: float,
    trial_phase: str,
    train_examples: list[Any],
    teacher_targets: dict[str, torch.Tensor],
    config: dict[str, Any],
    device: str,
    diagnostic_lambdas: list[float],
    torch_module: Any,
) -> dict[str, Any]:
    initial_hash = restore_router_state(student.routers, canonical_state)
    optimizer, optimizer_audit = fresh_router_optimizer(student, config)
    optimizer_construction_serial = next(_OPTIMIZER_CONSTRUCTION_SERIALS)
    runtime_optimizer_evidence = _optimizer_runtime_evidence(
        student, optimizer, optimizer_audit, optimizer_construction_serial
    )
    if initial_hash != router_state_hash(canonical_state):
        raise RuntimeError("REVISE: lambda trial did not start from canonical router state")
    base_hash_before = _module_state_hash(student.base)
    student.train()
    history = []
    gradient_diagnostic = None
    temperature = float(config["training"]["distillation_temperature"])
    router_parameters = [
        parameter for name, parameter in student.named_parameters() if name.startswith("routers.")
    ]
    for step, example in enumerate(train_examples, start=1):
        batch = DistillationBatch.from_examples([example])
        optimizer.zero_grad(set_to_none=True)
        state = QaqRequestState(
            example.example_id,
            int(example.prompt_mask().sum()),
            layer_count=LAYER_COUNT,
            candidate_bits=CANDIDATE_BITS,
        )
        teacher_logits = teacher_targets[example.example_id].to(device)
        student_logits = student(
            **_model_kwargs(example),
            request_state=state,
            phase="prefill",
            prompt_attention_mask=batch.prompt_attention_mask,
        ).logits
        kd_loss = masked_kl_distillation_loss(
            teacher_logits,
            student_logits,
            batch.completion_loss_mask,
            temperature=temperature,
        )
        cost_diagnostics = request_state_expected_bit_cost(state, return_diagnostics=True)
        bit_cost = cost_diagnostics.expected_bit_cost
        total_loss = cost_aware_distillation_loss(kd_loss, bit_cost, lambda_bit)
        if not all(_finite(value) for value in (kd_loss, bit_cost, total_loss)):
            raise FloatingPointError("REVISE: non-finite S10-D loss")
        if step == 1:
            kd_gradients = torch_module.autograd.grad(
                kd_loss, router_parameters, retain_graph=True, allow_unused=False
            )
            bit_gradients = torch_module.autograd.grad(
                bit_cost, router_parameters, retain_graph=True, allow_unused=False
            )
            kd_norm = _gradient_norm(kd_gradients)
            bit_norm = _gradient_norm(bit_gradients)
            if kd_norm <= 0:
                raise FloatingPointError("REVISE: initial KD gradient norm is zero")
            gradient_diagnostic = {
                "finite": True,
                "kd_gradient_norm": kd_norm,
                "bit_gradient_norm": bit_norm,
                "lambda_weighted_gradient_norm": lambda_bit * bit_norm,
                "lambda_weighted_ratio": lambda_bit * bit_norm / kd_norm,
                "lambda_weighted_ratio_by_lambda": {
                    str(value): float(value) * bit_norm / kd_norm for value in diagnostic_lambdas
                },
            }
        total_loss.backward()
        gradient_norm = _gradient_norm([parameter.grad for parameter in router_parameters])
        if any(
            parameter.grad is not None
            for name, parameter in student.named_parameters()
            if not name.startswith("routers.")
        ):
            raise RuntimeError("REVISE: frozen packed student received a gradient")
        records = route_records_from_request_state(
            example.example_id,
            state,
            log_base=float(config["evaluation"]["entropy_log_base"]),
        )
        probability_means = _probability_means(records)
        optimizer.step()
        history.append(
            {
                "step": step,
                "finite_kd_loss": _finite(kd_loss),
                "finite_bit_cost": _finite(bit_cost),
                "finite_weighted_cost": math.isfinite(lambda_bit * float(bit_cost.item())),
                "finite_total_loss": _finite(total_loss),
                "kd_loss": float(kd_loss.detach().item()),
                "expected_bit_cost": float(bit_cost.detach().item()),
                "weighted_cost": lambda_bit * float(bit_cost.detach().item()),
                "total_loss": float(total_loss.detach().item()),
                "expected_bit_width": None
                if cost_diagnostics.expected_bit_width is None
                else float(cost_diagnostics.expected_bit_width.detach().item()),
                "router_gradient_norm": gradient_norm,
                "mean_entropy": float(
                    route_statistics(
                        records,
                        entropy_log_base=float(config["evaluation"]["entropy_log_base"]),
                    )["mean_entropy"]
                ),
                "p4": probability_means["p4"],
                "p6": probability_means["p6"],
                "p8": probability_means["p8"],
                "optimizer_state_entries_after_step": len(optimizer.state),
            }
        )
        del state, teacher_logits, student_logits, kd_loss, bit_cost, total_loss, records, batch
        torch_module.cuda.empty_cache()
    student.eval()
    base_hash_after = _module_state_hash(student.base)
    if base_hash_before != base_hash_after:
        raise RuntimeError("REVISE: packed student base changed during lambda trial")
    if len(history) != int(config["training"]["optimizer_steps"]):
        raise RuntimeError("REVISE: S10-D did not execute exactly four optimizer steps")
    return {
        "lambda": lambda_bit,
        "phase": trial_phase,
        "initial_router_state_sha256": initial_hash,
        "final_router_state_sha256": router_state_hash(router_only_state(student.routers)),
        "optimizer": optimizer_audit.to_dict(),
        "runtime_optimizer_evidence": runtime_optimizer_evidence,
        "optimizer_state_was_fresh": True,
        "history": history,
        "gradient_diagnostic": gradient_diagnostic,
        "frozen_packed_student_state_before": base_hash_before,
        "frozen_packed_student_state_after": base_hash_after,
        "frozen_packed_student_unchanged": base_hash_before == base_hash_after,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "docs/results/s10d_lambda_calibration.json"
    )
    args = parser.parse_args()
    config = _load_config(args.config)
    device = args.device or str(config["device"])
    if not os.environ.get("VIRTUAL_ENV", "").startswith(str(Path.home() / ".venv")):
        raise SystemExit("PAUSE: ~/.venv is not active")
    _validate_model_snapshot(MODEL_SNAPSHOT)
    if not torch.cuda.is_available():
        raise SystemExit("PAUSE: CUDA is unavailable; CPU fallback is forbidden")
    torch.cuda.set_device(torch.device(device))
    free_bytes, _total_bytes = torch.cuda.mem_get_info(torch.device(device))
    if free_bytes < MIN_FREE_GPU_BYTES:
        raise SystemExit(
            f"PAUSE: intended GPU {device} has only {free_bytes} free bytes; explicit free GPU required"
        )
    manifest = json.loads(MANIFEST_PATH.read_text())
    artifact = ROOT / manifest["artifact"]["local_path"]
    if not artifact.is_dir():
        raise SystemExit(f"PAUSE: identity-matched packed artifact is unavailable: {artifact}")
    identities = _identity(manifest, artifact)

    import datasets
    from transformers import AutoTokenizer

    from qaq.evaluation.quality import load_full_precision_model
    from qaq.model.manual import load_manual_model
    from qaq.model.static import load_static_model
    from qaq.router.soft_model import SoftRoutedQwen3ForCausalLM

    _seed_everything(int(config["seed"]), torch)
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_SNAPSHOT), revision=MODEL_REVISION, local_files_only=True
    )
    train_dataset = datasets.load_dataset(
        config["dataset"]["repository"],
        config["dataset"]["config"],
        split=config["dataset"]["train_split"],
        revision=DATASET_REVISION,
        trust_remote_code=False,
    )
    validation_dataset = datasets.load_dataset(
        config["dataset"]["repository"],
        config["dataset"]["config"],
        split=config["dataset"]["validation_split"],
        revision=DATASET_REVISION,
        trust_remote_code=False,
    )
    train_examples_cpu, train_manifest = _select_examples(
        train_dataset,
        tokenizer,
        config["dataset"]["train_offsets"],
        split="train",
        config=config,
        torch=torch,
    )
    validation_examples_cpu, validation_manifest = _select_examples(
        validation_dataset,
        tokenizer,
        config["dataset"]["validation_offsets"],
        split="validation",
        config=config,
        torch=torch,
    )
    if len(train_examples_cpu) != 4 or len(validation_examples_cpu) != 2:
        raise RuntimeError("REVISE: locked example counts are not 4 train and 2 validation")
    train_examples = [_device_example(example, device, torch) for example in train_examples_cpu]
    validation_examples = [
        _device_example(example, device, torch) for example in validation_examples_cpu
    ]

    print("S10-D: loading pinned teacher", flush=True)
    teacher = load_full_precision_model(MODEL_SNAPSHOT, device)
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    teacher_frozen_before = all(not parameter.requires_grad for parameter in teacher.parameters())
    teacher_hash_before = _module_state_hash(teacher)
    teacher_targets = _precompute_teacher_logits(
        teacher, train_examples + validation_examples, torch
    )
    if not all(_finite(value) for value in teacher_targets.values()):
        raise FloatingPointError("REVISE: pinned teacher targets are non-finite")
    teacher.cpu()
    torch.cuda.empty_cache()

    print("S10-D: measuring static 4/6/8 references", flush=True)
    static_model = load_static_model(artifact, device)
    static_references = _evaluate_static(
        static_model, validation_examples, teacher_targets, device, config, torch
    )
    del static_model
    torch.cuda.empty_cache()

    print("S10-D: constructing canonical three-way router", flush=True)
    manual_model = load_manual_model(artifact, device)
    _seed_everything(int(config["seed"]), torch)
    student = SoftRoutedQwen3ForCausalLM(
        manual_model,
        temperature=float(config["training"]["routing_temperature"]),
        candidate_bits=CANDIDATE_BITS,
    )
    student.to(device)
    install_memory_saving_packed_backward()
    freeze_teacher_and_packed_student(teacher, student)
    validate_canonical_initialization(student, config)
    canonical_state = router_only_state(student.routers)
    canonical_hash = router_state_hash(canonical_state)
    lambda_grid = [float(value) for value in config["lambda_grid"]]
    initial_hashes_by_lambda = {
        str(value): restore_router_state(student.routers, canonical_state) for value in lambda_grid
    }
    if len(set(initial_hashes_by_lambda.values())) != 1:
        raise RuntimeError("REVISE: canonical initial router hashes differ across lambda grid")

    trial_results: list[dict[str, Any]] = []
    trial_by_lambda: dict[float, dict[str, Any]] = {}
    for lambda_bit in lambda_grid:
        print(f"S10-D: running grid lambda={lambda_bit}", flush=True)
        trial = _run_trial(
            student,
            canonical_state,
            lambda_bit,
            "grid",
            train_examples,
            teacher_targets,
            config,
            device,
            lambda_grid,
            torch,
        )
        trial["soft"] = _evaluate_learned(
            student, validation_examples, teacher_targets, device, "soft", config, torch
        )
        trial["hard"] = _evaluate_learned(
            student, validation_examples, teacher_targets, device, "hard", config, torch
        )
        trial["collapse_label"] = trial["hard"]["collapse_classification"]
        trial_results.append(trial)
        trial_by_lambda[lambda_bit] = trial
        torch.cuda.empty_cache()

    extensions: list[float] = []
    low_policy = config["adaptive_extensions"]["low_lambda"]
    low_lambda = float(low_policy["trigger_lambda"])
    low_trial = trial_by_lambda[low_lambda]
    if low_trial["collapse_label"] in low_policy["trigger_collapses"]:
        extensions.append(float(low_policy["point"]))
    high_policy = config["adaptive_extensions"]["high_lambda"]
    high_lambda = float(high_policy["trigger_lambda"])
    high_trial = trial_by_lambda[high_lambda]
    zero_trial = trial_by_lambda[0.0]
    exact_hard_map = (
        high_trial["hard"]["per_validation_route_maps"]
        == zero_trial["hard"]["per_validation_route_maps"]
    )
    soft_width_delta = abs(
        high_trial["soft"]["mean_expected_bit_width"]
        - zero_trial["soft"]["mean_expected_bit_width"]
    )
    if exact_hard_map == bool(
        high_policy["trigger_exact_lambda_zero_hard_map"]
    ) and soft_width_delta < float(high_policy["trigger_soft_width_delta_bits"]):
        extensions.append(float(high_policy["point"]))
    extensions = list(dict.fromkeys(extensions))[: int(config["adaptive_extensions"]["max_points"])]

    extension_decisions = {
        "low_lambda_0.003": {
            "collapse_label": low_trial["collapse_label"],
            "point_added": float(low_policy["point"]) in extensions,
        },
        "high_lambda_0.1": {
            "exact_lambda_zero_hard_map": exact_hard_map,
            "soft_width_delta_bits": soft_width_delta,
            "point_added": float(high_policy["point"]) in extensions,
        },
    }
    for lambda_bit in extensions:
        print(f"S10-D: running permitted extension lambda={lambda_bit}", flush=True)
        trial = _run_trial(
            student,
            canonical_state,
            lambda_bit,
            "adaptive_extension",
            train_examples,
            teacher_targets,
            config,
            device,
            lambda_grid + extensions,
            torch,
        )
        trial["soft"] = _evaluate_learned(
            student, validation_examples, teacher_targets, device, "soft", config, torch
        )
        trial["hard"] = _evaluate_learned(
            student, validation_examples, teacher_targets, device, "hard", config, torch
        )
        trial["collapse_label"] = trial["hard"]["collapse_classification"]
        trial_results.append(trial)
        trial_by_lambda[lambda_bit] = trial
        torch.cuda.empty_cache()

    baseline = trial_by_lambda[0.0]
    for trial in trial_results:
        trial["baseline_deltas"] = _baseline_deltas(trial, baseline)
    teacher_hash_after = _module_state_hash(teacher)
    teacher_gradients_absent = all(parameter.grad is None for parameter in teacher.parameters())
    packed_student_hash = _module_state_hash(student.base)
    packed_student_hash_before = trial_results[0]["frozen_packed_student_state_before"]
    if teacher_hash_before != teacher_hash_after or not teacher_gradients_absent:
        raise RuntimeError("REVISE: pinned teacher changed or received gradients")
    if packed_student_hash != packed_student_hash_before:
        raise RuntimeError("REVISE: packed student base changed across S10-D")

    result = {
        "format": "qaq-s10d-lambda-calibration-v1",
        "stage": "S10-D",
        "commit": _git_commit(),
        "environment": _environment(device, int(free_bytes)),
        "identities": identities,
        "dataset": {
            "configuration": config["dataset"],
            "order": {
                "train_example_ids": [item["example_id"] for item in train_manifest],
                "validation_example_ids": [item["example_id"] for item in validation_manifest],
                "train_source_offsets": config["dataset"]["train_offsets"],
                "validation_source_offsets": config["dataset"]["validation_offsets"],
            },
            "manifest": {"train": train_manifest, "validation": validation_manifest},
        },
        "canonical_initialization": {
            "seed": int(config["seed"]),
            "candidate_bits": list(CANDIDATE_BITS),
            "router_count": student.router_count,
            "router_parameter_count": student.router_parameter_count,
            "router_only_state_sha256": canonical_hash,
            "initial_hashes_by_lambda": initial_hashes_by_lambda,
            "all_initial_hashes_match": len(set(initial_hashes_by_lambda.values())) == 1,
            "historical_two_way_checkpoint_loaded": False,
            "reload_method": "router-only state_dict clone/reload before every lambda trial",
        },
        "static_references": static_references,
        "gradient_diagnostic": {
            "per_trial": {
                str(trial["lambda"]): trial["gradient_diagnostic"] for trial in trial_results
            },
            "objective_formula": "L_total = L_KD + lambda_bit * L_bit; no objective normalization",
            "kd_implementation": "unchanged masked_kl_distillation_loss",
            "bit_cost_implementation": "request_state_expected_bit_cost",
        },
        "grid": {
            "values": lambda_grid,
            "completed": [trial["lambda"] for trial in trial_results if trial["phase"] == "grid"],
            "completed_before_extensions": True,
        },
        "extensions": {
            "policy": config["adaptive_extensions"],
            "decisions": extension_decisions,
            "performed": extensions,
            "recursive_expansion": False,
        },
        "trials": trial_results,
        "pareto_frontiers": pareto_frontiers(trial_results),
        "audits": {
            "teacher_frozen_before_precompute": teacher_frozen_before,
            "teacher_frozen_after": all(
                not parameter.requires_grad for parameter in teacher.parameters()
            ),
            "teacher_gradients_absent_after_targets_and_trials": teacher_gradients_absent,
            "teacher_state_sha256_before": teacher_hash_before,
            "teacher_state_sha256_after": teacher_hash_after,
            "teacher_unchanged": teacher_hash_before == teacher_hash_after,
            "packed_student_base_state_sha256_after": packed_student_hash,
            "packed_student_base_unchanged": packed_student_hash == packed_student_hash_before,
            "router_only_optimizer": all(
                trial["optimizer"]["included_name_prefixes"] == ["routers."]
                or tuple(trial["optimizer"]["included_name_prefixes"]) == ("routers.",)
                for trial in trial_results
            ),
            "fresh_adamw_per_trial": all(
                trial["optimizer_state_was_fresh"] for trial in trial_results
            ),
            "all_initial_hashes_match": len(set(initial_hashes_by_lambda.values())) == 1,
            "finite_measurements": all(
                all(
                    item["finite_kd_loss"] and item["finite_bit_cost"] and item["finite_total_loss"]
                    for item in trial["history"]
                )
                and trial["soft"]["finite_logits"]
                and trial["hard"]["finite_logits"]
                for trial in trial_results
            ),
            "s08_on_demand_invoked": False,
            "historical_s07_script_changed": False,
        },
        "limitations": [
            "This is a fixed four-step, four-train-example calibration observation, not full router training.",
            "The normalized bit-plane-count objective is not a latency, memory, transfer, energy, or kernel-runtime measurement.",
            "No production lambda is selected by this artifact; firstmate/captain must review the observed frontier.",
            "No S08 on-demand loading, transfer, latency, memory, extra seed, extra epoch, or larger-data measurement was performed.",
        ],
        "next_action": "firstmate/captain reviews the observed frontier and decides whether to refine, confirm, or begin full training",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
