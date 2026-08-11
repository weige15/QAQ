#!/usr/bin/env python3
"""Run the single locked S07-B router-distillation baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qaq.model.request_state import QaqRequestState
from qaq.s07_distillation import (
    DistillationBatch,
    DistillationExample,
    RouteLogCollector,
    RouteLogRecord,
    TokenRange,
    causal_target_ids,
    route_statistics,
)

CONFIG_PATH = ROOT / "configs/s07_router_training.json"
MANIFEST_PATH = ROOT / "docs/quantized_model_manifest.json"
SNAPSHOT = Path(
    os.environ.get(
        "QAQ_MODEL_SNAPSHOT",
        "~/.cache/huggingface/hub/models--Qwen--Qwen3-4B/"
        "snapshots/1cfa9a7208912126459214e8b04321603b3df60c",
    )
).expanduser()
MODEL_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
ANY_PRECISION_REVISION = "a3257d02740cc5757c78673da534b0630ff3a4ea"
DATASET_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"


def _load_config() -> dict[str, Any]:
    payload = json.loads(CONFIG_PATH.read_text())
    if payload["dataset"]["revision"] != DATASET_REVISION:
        raise RuntimeError("REVISE: locked dataset revision differs from the script")
    return payload


def _seed_everything(seed: int, torch: Any) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _tensor_hash(value: Any) -> str:
    tensor = value.detach().cpu().contiguous()
    return hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest()


def _parameter_hashes(model: Any) -> dict[str, str]:
    return {name: _tensor_hash(parameter) for name, parameter in model.named_parameters()}


def _aggregate_hash(values: dict[str, str]) -> str:
    digest = hashlib.sha256()
    for name in sorted(values):
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(values[name].encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _check_environment() -> None:
    if not os.environ.get("VIRTUAL_ENV", "").startswith(str(Path.home() / ".venv")):
        raise SystemExit("PAUSE: ~/.venv is not active")
    if not SNAPSHOT.is_dir() or SNAPSHOT.name != MODEL_REVISION:
        raise SystemExit(f"PAUSE: exact pinned model snapshot is unavailable: {SNAPSHOT}")


def _select_examples(dataset: Any, tokenizer: Any, offsets: list[int], *, split: str, config: dict[str, Any], torch: Any):
    sequence_length = int(config["dataset"]["sequence_length"])
    prompt_tokens = int(config["dataset"]["prompt_tokens"])
    completion_tokens = int(config["dataset"]["completion_tokens"])
    examples = []
    manifest = []
    for offset in offsets:
        selected = None
        for row_index in range(offset, len(dataset)):
            text = str(dataset[row_index]["text"]).strip()
            if not text:
                continue
            token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            if len(token_ids) < sequence_length:
                continue
            selected = (row_index, text, token_ids[:sequence_length])
            break
        if selected is None:
            raise RuntimeError(f"PAUSE: no qualifying {split} row at or after offset {offset}")
        row_index, text, token_ids = selected
        input_ids = torch.tensor(token_ids, dtype=torch.long)
        attention_mask = torch.ones(sequence_length, dtype=torch.bool)
        completion_mask = torch.zeros(sequence_length, dtype=torch.bool)
        completion_mask[prompt_tokens - 1 : prompt_tokens + completion_tokens - 1] = True
        target_ids = causal_target_ids(input_ids, attention_mask)
        prompt_text = tokenizer.decode(token_ids[:prompt_tokens], skip_special_tokens=False)
        completion_text = tokenizer.decode(token_ids[prompt_tokens:], skip_special_tokens=False)
        example = DistillationExample(
            example_id=f"{split}-{row_index}",
            tokenizer_revision=MODEL_REVISION,
            input_ids=input_ids,
            attention_mask=attention_mask,
            completion_loss_mask=completion_mask,
            target_ids=target_ids,
            prompt_text=prompt_text,
            completion_text=completion_text,
            prompt_token_range=TokenRange(0, prompt_tokens),
            completion_token_range=TokenRange(prompt_tokens, sequence_length),
        )
        examples.append(example)
        manifest.append(
            {
                "example_id": example.example_id,
                "split": split,
                "source_row": row_index,
                "source_offset": offset,
                "source_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "input_ids_sha256": _tensor_hash(input_ids),
                "prompt_token_range": [0, prompt_tokens],
                "completion_token_range": [prompt_tokens, sequence_length],
                "prompt_text": prompt_text,
                "completion_text": completion_text,
            }
        )
    return examples, manifest


def _device_example(example: Any, device: str, torch: Any):
    return DistillationExample(
        example_id=example.example_id,
        tokenizer_revision=example.tokenizer_revision,
        input_ids=example.input_ids.to(device),
        attention_mask=example.attention_mask.to(device),
        completion_loss_mask=example.completion_loss_mask.to(device),
        target_ids=example.target_ids.to(device),
        prompt_text=example.prompt_text,
        completion_text=example.completion_text,
        prompt_token_range=example.prompt_token_range,
        completion_token_range=example.completion_token_range,
        prompt_attention_mask=(
            example.prompt_attention_mask.to(device)
            if example.prompt_attention_mask is not None
            else None
        ),
        sequence_positions=(
            example.sequence_positions.to(device)
            if example.sequence_positions is not None
            else None
        ),
    )


def _model_kwargs(example: Any) -> dict[str, Any]:
    execution = example.execution_inputs()
    return {
        "input_ids": execution.input_ids,
        "attention_mask": execution.attention_mask,
        "position_ids": execution.sequence_positions,
        "use_cache": False,
    }


def _records_for_state(request_id: str, state: Any, student: Any, *, log_base: float):
    records = []
    for unit_type, features in (
        ("attention", state.attention_features),
        ("ffn", state.ffn_features),
    ):
        for layer, feature in enumerate(features):
            if feature is None:
                raise RuntimeError(f"REVISE: missing {unit_type} feature at layer {layer}")
            probability = student.route(layer, unit_type, feature)
            records.append(
                RouteLogRecord.from_probabilities(
                    request_id, layer, unit_type, probability, log_base=log_base
                )
            )
    collector = RouteLogCollector(layer_count=36, request_ids=(request_id,))
    for record in records:
        collector.add(record)
    collector.finalize()
    return tuple(records)


def _probabilities_by_layer(records: Any) -> dict[str, dict[str, float]]:
    result = {}
    for layer in sorted({record.layer for record in records}):
        subset = [record for record in records if record.layer == layer]
        result[str(layer)] = {
            "p4": sum(record.p4 for record in subset) / len(subset),
            "p8": sum(record.p8 for record in subset) / len(subset),
        }
    return result


def _route_map(records: Any) -> tuple[int, ...]:
    ordered = sorted(records, key=lambda item: (item.layer, item.unit_type))
    return tuple(record.hard_bit for record in ordered)


def _route_distance(maps: list[tuple[int, ...]]) -> float:
    if len(maps) < 2:
        return 0.0
    distances = [sum(left != right for left, right in zip(a, b)) / len(a) for a, b in combinations(maps, 2)]
    return sum(distances) / len(distances)


def _route_summary(records: list[Any], *, distillation_loss: float | None = None) -> dict[str, Any]:
    stats = route_statistics(records, distillation_loss=distillation_loss)
    maps_by_request: dict[str, tuple[int, ...]] = {}
    for request_id in sorted({record.request_id for record in records}):
        maps_by_request[request_id] = _route_map(
            [record for record in records if record.request_id == request_id]
        )
    stats.update(
        {
            "probability_by_layer": _probabilities_by_layer(records),
            "unique_hard_route_map_count": len(set(maps_by_request.values())),
            "mean_hard_selected_bit_width": sum(record.hard_bit for record in records) / len(records),
            "parameter_weighted_mean_selected_bit_width": None,
            "prompt_to_prompt_route_distance": _route_distance(list(maps_by_request.values())),
            "fraction_prompts_routed_entirely_to_8": (
                sum(all(bit == 8 for bit in route_map) for route_map in maps_by_request.values())
                / len(maps_by_request)
            ),
            "request_route_maps": {request_id: list(route_map) for request_id, route_map in maps_by_request.items()},
        }
    )
    return stats


def _check_probabilities(state: Any, torch: Any) -> None:
    values = state.attention_probabilities + state.ffn_probabilities
    if any(value is None for value in values):
        raise RuntimeError("REVISE: route log coverage is incomplete")
    if any(not bool(torch.isfinite(value).all().item()) for value in values if value is not None):
        raise FloatingPointError("REVISE: NaN or Inf router probabilities")


def _precompute_teacher_logits(teacher: Any, examples: list[Any], torch: Any) -> dict[str, Any]:
    teacher.eval()
    targets = {}
    for example in examples:
        with torch.no_grad():
            logits = teacher(**_model_kwargs(example)).logits.detach().cpu()
        if not bool(torch.isfinite(logits).all().item()):
            raise FloatingPointError("REVISE: non-finite teacher logits")
        targets[example.example_id] = logits
    return targets


def _train(
    student: Any,
    train_examples: list[Any],
    teacher_targets: dict[str, Any],
    config: dict[str, Any],
    device: str,
    torch: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, str], dict[str, str]]:
    from qaq.router.network import trainable_parameter_audit
    from qaq.s07_distillation import build_router_optimizer, masked_kl_distillation_loss

    for parameter in student.base.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    audit = trainable_parameter_audit(student)
    if audit["trainable_parameter_count"] != student.router_parameter_count:
        raise RuntimeError("REVISE: trainable parameter audit is not router-only")
    if any(not name.startswith("routers.") for name in audit["trainable_names"]):
        raise RuntimeError("REVISE: non-router parameter is trainable")
    frozen_hashes_before = {}
    frozen_hashes_before.update(
        {f"student.{name}": digest for name, digest in _parameter_hashes(student).items() if not name.startswith("routers.")}
    )
    optimizer, optimizer_audit = build_router_optimizer(
        student,
        lr=float(config["training"]["learning_rate"]),
        optimizer_cls=torch.optim.AdamW,
        weight_decay=float(config["training"]["weight_decay"]),
    )
    if optimizer_audit.scalar_count != student.router_parameter_count:
        raise RuntimeError("REVISE: optimizer audit is not router-only")
    history = []
    student.train()
    temperature = float(config["training"]["distillation_temperature"])
    for step, example in enumerate(train_examples, start=1):
        batch = DistillationBatch.from_examples([example])
        execution = batch.execution_inputs()
        optimizer.zero_grad(set_to_none=True)
        teacher_logits = teacher_targets[example.example_id].to(device)
        state = QaqRequestState(example.example_id, int(example.prompt_mask().sum()), layer_count=36)
        student_logits = student(
            **_model_kwargs(example),
            request_state=state,
            phase="prefill",
            prompt_attention_mask=batch.prompt_attention_mask,
        ).logits
        loss = masked_kl_distillation_loss(
            teacher_logits, student_logits, batch.completion_loss_mask, temperature=temperature
        )
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError("REVISE: NaN or Inf KD loss")
        _check_probabilities(state, torch)
        loss.backward()
        router_parameters = [parameter for name, parameter in student.named_parameters() if name.startswith("routers.")]
        gradients = [parameter.grad for parameter in router_parameters]
        if any(gradient is None or not bool(torch.isfinite(gradient).all().item()) for gradient in gradients):
            raise FloatingPointError("REVISE: missing, NaN, or Inf router gradient")
        gradient_norm = float(torch.sqrt(sum(gradient.detach().float().square().sum() for gradient in gradients)).item())
        optimizer.step()
        if any(parameter.grad is not None for name, parameter in student.named_parameters() if not name.startswith("routers.")):
            raise RuntimeError("REVISE: frozen parameter received a gradient")
        records = list(_records_for_state(example.example_id, state, student, log_base=2.0))
        history.append(
            {
                "step": step,
                "training_kd_loss": float(loss.detach().item()),
                "router_gradient_norm": gradient_norm,
                "route_statistics": _route_summary(records, distillation_loss=float(loss.detach().item())),
            }
        )
        del teacher_logits, student_logits, loss, state, records, batch, execution
        torch.cuda.empty_cache()
    frozen_hashes_after = {}
    frozen_hashes_after.update(
        {f"student.{name}": digest for name, digest in _parameter_hashes(student).items() if not name.startswith("routers.")}
    )
    if frozen_hashes_before != frozen_hashes_after:
        raise RuntimeError("REVISE: frozen teacher or packed-student parameter changed")
    audit["optimizer"] = optimizer_audit.to_dict()
    return history, audit, frozen_hashes_before, frozen_hashes_after


def _eval_mode(
    teacher_targets: dict[str, Any], student: Any, examples: list[Any], mode: str, torch: Any
) -> tuple[dict[str, Any], list[Any], list[Any]]:
    from qaq.s04_manual import PrecisionPlan, PrecisionTrace
    from qaq.s07_distillation import hard_route, masked_kl_distillation_loss

    records = []
    per_example = []
    teacher_logits_all = []
    student_logits_all = []
    student.eval()
    for example in examples:
        kwargs = _model_kwargs(example)
        with torch.no_grad():
            teacher_logits = teacher_targets[example.example_id].to(kwargs["input_ids"].device)
            if mode == "soft":
                state = QaqRequestState(example.example_id, int(example.prompt_mask().sum()), layer_count=36)
                trace = PrecisionTrace()
                output = student(
                    **kwargs,
                    request_state=state,
                    phase="prefill",
                    prompt_attention_mask=example.prompt_mask().unsqueeze(0),
                    trace=trace,
                )
                logits = output.logits.detach()
                example_records = list(_records_for_state(example.example_id, state, student, log_base=2.0))
            elif mode == "hard":
                state = QaqRequestState(example.example_id, int(example.prompt_mask().sum()), layer_count=36)
                trace = PrecisionTrace()

                def policy(layer: int, unit_type: str, feature: Any) -> int:
                    return int(hard_route(student.route(layer, unit_type, feature)))

                output = student.base(
                    **kwargs,
                    request_state=state,
                    phase="prefill",
                    prompt_attention_mask=example.prompt_mask().unsqueeze(0),
                    routing_policy=policy,
                    trace=trace,
                )
                logits = output.logits.detach()
                example_records = list(_records_for_state(example.example_id, state, student, log_base=2.0))
            elif mode in ("static4", "static8"):
                bits = int(mode[-1])
                output = student.base(
                    **kwargs,
                    precision_plan=PrecisionPlan.uniform(bits),
                    trace=PrecisionTrace(),
                )
                logits = output.logits.detach()
                example_records = []
            else:  # pragma: no cover
                raise ValueError(mode)
        if not bool(torch.isfinite(logits).all().item()):
            raise FloatingPointError(f"REVISE: non-finite {mode} logits")
        teacher_float = teacher_logits.float()
        logits_float = logits.float()
        mean_error = float((logits_float - teacher_float).abs().mean().item())
        max_error = float((logits_float - teacher_float).abs().max().item())
        kd = None
        if mode in ("soft", "hard"):
            kd = float(
                masked_kl_distillation_loss(
                    teacher_logits, logits, example.completion_loss_mask.unsqueeze(0), temperature=2.0
                ).item()
            )
        per_example.append(
            {
                "example_id": example.example_id,
                "mean_absolute_logit_error": mean_error,
                "maximum_absolute_logit_error": max_error,
                "kd_loss": kd,
                "finite_logits": True,
            }
        )
        if example_records:
            records.extend(example_records)
        teacher_logits_all.append(teacher_float.cpu())
        student_logits_all.append(logits_float.cpu())
    result = {
        "count": len(per_example),
        "quality_metric": "mean absolute logit error against full-precision teacher",
        "mean_absolute_logit_error": sum(item["mean_absolute_logit_error"] for item in per_example) / len(per_example),
        "maximum_absolute_logit_error": max(item["maximum_absolute_logit_error"] for item in per_example),
        "per_example": per_example,
    }
    if records:
        result["route_statistics"] = _route_summary(records)
        result["route_logs"] = [
            {
                "request_id": record.request_id,
                "layer": record.layer,
                "unit_type": record.unit_type,
                "p4": record.p4,
                "p8": record.p8,
                "hard_bit": record.hard_bit,
                "entropy": record.entropy,
            }
            for record in records
        ]
        result["route_log_coverage"] = len(records) == 72 * len(examples)
    return result, teacher_logits_all, student_logits_all


def _classify_adaptivity(hard_result: dict[str, Any]) -> str:
    stats = hard_result["route_statistics"]
    if stats["hard_fraction_8"] >= 0.95:
        return "COLLAPSED_TO_8"
    if stats["hard_fraction_4"] >= 0.95:
        return "COLLAPSED_TO_4"
    variation = stats["route_variation_across_prompts"]
    if variation["changed_fraction"] > 0 and stats["prompt_to_prompt_route_distance"] >= 0.05:
        return "ADAPTIVE_OBSERVED"
    if variation["changed_fraction"] == 0:
        return "PROMPT_INVARIANT"
    return "OTHER"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=os.environ.get("QAQ_MODEL_DEVICE", "cuda:3"))
    parser.add_argument("--output", type=Path, default=ROOT / "docs/results/s07_router_training.json")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(os.environ.get("QAQ_S07_CHECKPOINT", "~/.cache/qaq/s07b/final_router.pt")).expanduser(),
    )
    args = parser.parse_args()
    _check_environment()

    import torch
    from datasets import load_dataset
    from transformers import AutoTokenizer

    from qaq.s03_quality import load_full_precision_model
    from qaq.s03_static import PINNED_ANY_PRECISION_COMMIT, load_manifest, source_commit
    from qaq.s06_soft import load_soft_model
    from qaq.s07_distillation import (
        RouterCheckpointMetadata,
        freeze_teacher_and_packed_student,
        save_router_checkpoint,
    )

    config = _load_config()
    if not torch.cuda.is_available():
        raise SystemExit("PAUSE: CUDA is unavailable")
    torch.cuda.set_device(torch.device(args.device))
    manifest = load_manifest(MANIFEST_PATH)
    if manifest["source_model"]["revision"] != MODEL_REVISION:
        raise SystemExit("REVISE: teacher model revision changed")
    if manifest["any_precision"]["commit"] != ANY_PRECISION_REVISION or source_commit() != PINNED_ANY_PRECISION_COMMIT:
        raise SystemExit("REVISE: Any-Precision revision changed")
    artifact = ROOT / manifest["artifact"]["local_path"]
    if not artifact.is_dir():
        raise SystemExit(f"PAUSE: packed student artifact is unavailable: {artifact}")
    artifact_hash = manifest["artifact"]["checkpoint_hashes"]["pytorch_model.bin"]
    tokenizer = AutoTokenizer.from_pretrained(str(SNAPSHOT), revision=MODEL_REVISION, local_files_only=True)
    dataset_name = config["dataset"]["repository"]
    dataset_config = config["dataset"]["config"]
    train_dataset = load_dataset(dataset_name, dataset_config, split="train", revision=DATASET_REVISION, trust_remote_code=False)
    validation_dataset = load_dataset(dataset_name, dataset_config, split="validation", revision=DATASET_REVISION, trust_remote_code=False)
    train_examples_cpu, train_manifest = _select_examples(
        train_dataset, tokenizer, config["dataset"]["train_offsets"], split="train", config=config, torch=torch
    )
    validation_examples_cpu, validation_manifest = _select_examples(
        validation_dataset, tokenizer, config["dataset"]["validation_offsets"], split="validation", config=config, torch=torch
    )
    train_examples = [_device_example(example, args.device, torch) for example in train_examples_cpu]
    validation_examples = [_device_example(example, args.device, torch) for example in validation_examples_cpu]

    _seed_everything(int(config["dataset"]["seed"]), torch)
    print("S07-B: loading full-precision teacher", flush=True)
    teacher = load_full_precision_model(SNAPSHOT, args.device)
    print("S07-B: loading packed student and router", flush=True)
    student = load_soft_model(
        artifact,
        args.device,
        temperature=float(config["training"]["routing_temperature"]),
    )
    student.to(args.device)
    # Use the audited production freeze seam before any teacher-logit work.
    freeze_teacher_and_packed_student(teacher, student)
    teacher_parameter_hash_before = _aggregate_hash(_parameter_hashes(teacher))
    teacher_frozen_before = all(not parameter.requires_grad for parameter in teacher.parameters())
    teacher_targets = _precompute_teacher_logits(teacher, train_examples + validation_examples, torch)
    teacher_gradients_absent_after_precompute = all(
        parameter.grad is None for parameter in teacher.parameters()
    )
    if not teacher_frozen_before or not teacher_gradients_absent_after_precompute:
        raise RuntimeError("REVISE: teacher freeze or gradient audit failed before training")
    teacher.cpu()
    torch.cuda.empty_cache()
    initial_router_hashes = _parameter_hashes(student.routers)
    initial_router_hash = _aggregate_hash(initial_router_hashes)
    history, freeze_audit, frozen_before, frozen_after = _train(
        student, train_examples, teacher_targets, config, args.device, torch
    )
    teacher_parameter_hash_after = _aggregate_hash(_parameter_hashes(teacher))
    teacher_frozen_after = all(not parameter.requires_grad for parameter in teacher.parameters())
    teacher_gradients_absent_after_training = all(
        parameter.grad is None for parameter in teacher.parameters()
    )
    if (
        teacher_parameter_hash_before != teacher_parameter_hash_after
        or not teacher_frozen_after
        or not teacher_gradients_absent_after_training
    ):
        raise RuntimeError("REVISE: teacher freeze, gradient, or value audit failed")
    final_router_hashes = _parameter_hashes(student.routers)
    final_router_hash = _aggregate_hash(final_router_hashes)
    router_parameters_changed = initial_router_hash != final_router_hash
    if not router_parameters_changed:
        raise RuntimeError("REVISE: router parameters did not change")
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_metadata = RouterCheckpointMetadata(
        model_repository=manifest["source_model"]["repository"],
        model_revision=MODEL_REVISION,
        quantized_checkpoint_id=manifest["artifact"]["local_path"],
        quantized_checkpoint_hash=f"sha256:{artifact_hash}",
        any_precision_revision=ANY_PRECISION_REVISION,
        router_architecture={
            "feature_dim": student.feature_dim,
            "hidden_width": 128,
            "activation": "GELU",
            "normalization": "parameter-free RMS",
            "normalization_epsilon": 1e-6,
            "temperature": config["training"]["routing_temperature"],
            "router_count": student.router_count,
        },
        training_step=len(history),
        training_step_metadata={"seed": config["dataset"]["seed"], "format": config["format"]},
    )
    optimizer_for_checkpoint, _ = __import__("qaq.s07_distillation", fromlist=["build_router_optimizer"]).build_router_optimizer(
        student, lr=float(config["training"]["learning_rate"]), optimizer_cls=torch.optim.AdamW, weight_decay=0.0
    )
    save_router_checkpoint(args.checkpoint, student.routers, checkpoint_metadata, optimizer=optimizer_for_checkpoint)
    checkpoint_hash = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()

    print("S07-B: evaluating static, soft, and deterministic hard routes", flush=True)
    static4, _, _ = _eval_mode(teacher_targets, student, validation_examples, "static4", torch)
    static8, _, _ = _eval_mode(teacher_targets, student, validation_examples, "static8", torch)
    soft, _, _ = _eval_mode(teacher_targets, student, validation_examples, "soft", torch)
    hard, _, _ = _eval_mode(teacher_targets, student, validation_examples, "hard", torch)
    adaptivity = _classify_adaptivity(hard)
    soft["final_validation_kd_loss"] = sum(item["kd_loss"] for item in soft["per_example"]) / len(soft["per_example"])
    hard["final_validation_kd_loss"] = sum(item["kd_loss"] for item in hard["per_example"]) / len(hard["per_example"])
    hard["soft_vs_hard_quality_difference"] = hard["mean_absolute_logit_error"] - soft["mean_absolute_logit_error"]
    hard["static8_vs_hard_quality_difference"] = hard["mean_absolute_logit_error"] - static8["mean_absolute_logit_error"]
    hard["static4_vs_hard_quality_difference"] = hard["mean_absolute_logit_error"] - static4["mean_absolute_logit_error"]
    hard["adaptivity_classification"] = adaptivity

    result = {
        "format": "qaq-s07b-router-training-v1",
        "scope": "S07-B only; exactly one baseline router-distillation run; no S08 loading",
        "source_model": {
            "repository": manifest["source_model"]["repository"],
            "revision": MODEL_REVISION,
            "tokenizer_revision": MODEL_REVISION,
            "any_precision_revision": ANY_PRECISION_REVISION,
            "packed_student_checkpoint": manifest["artifact"]["local_path"],
            "packed_student_checkpoint_sha256": artifact_hash,
        },
        "training_configuration": config,
        "dataset_manifest": {"train": train_manifest, "validation": validation_manifest},
        "freeze_audit": {
            "trainable_parameter_count": freeze_audit["trainable_parameter_count"],
            "trainable_names": freeze_audit["trainable_names"],
            "optimizer": freeze_audit["optimizer"],
            "frozen_parameter_hash_before": _aggregate_hash(frozen_before),
            "frozen_parameter_hash_after": _aggregate_hash(frozen_after),
            "frozen_parameter_hashes_match": frozen_before == frozen_after,
            "teacher_frozen": teacher_frozen_after,
            "teacher_requires_grad_false_before_precompute": teacher_frozen_before,
            "teacher_gradients_absent_after_precompute": teacher_gradients_absent_after_precompute,
            "teacher_gradients_absent_after_training": teacher_gradients_absent_after_training,
            "teacher_parameter_hash_before": teacher_parameter_hash_before,
            "teacher_parameter_hash_after": teacher_parameter_hash_after,
            "teacher_parameter_hashes_match": teacher_parameter_hash_before == teacher_parameter_hash_after,
            "student_non_router_frozen": all(
                not parameter.requires_grad for name, parameter in student.named_parameters() if not name.startswith("routers.")
            ),
            "student_non_router_gradients_absent": all(
                parameter.grad is None
                for name, parameter in student.named_parameters()
                if not name.startswith("routers.")
            ),
            "packed_buffers_non_trainable": all(not buffer.requires_grad for buffer in student.buffers()),
            "teacher_in_optimizer": False,
            "router_only": True,
        },
        "initial_router_parameter_sha256": initial_router_hash,
        "final_router_parameter_sha256": final_router_hash,
        "router_parameters_changed": router_parameters_changed,
        "training": {
            "optimizer_steps": len(history),
            "initial_training_kd_loss": history[0]["training_kd_loss"],
            "final_training_kd_loss": history[-1]["training_kd_loss"],
            "history": history,
            "all_losses_finite": all(math.isfinite(item["training_kd_loss"]) for item in history),
            "all_router_gradients_finite": all(math.isfinite(item["router_gradient_norm"]) for item in history),
        },
        "checkpoint": {
            "external_path": str(args.checkpoint),
            "sha256": checkpoint_hash,
            "router_only": True,
            "metadata": checkpoint_metadata.to_dict(),
        },
        "evaluation": {
            "static_4": static4,
            "static_8": static8,
            "soft": soft,
            "hard": hard,
        },
        "hard_route_determinism": {
            "same_process_repeat": "pending separate fixed-subset check",
            "checkpoint_reload": "pending separate fresh-process check",
        },
        "objective": {
            "formula": "T^2 * masked KL(teacher || student)",
            "completion_only": True,
            "extra_penalties": [],
        },
        "stage_gate": {
            "engineering_gate": "CONTINUE",
            "query_adaptivity_demonstrated": adaptivity == "ADAPTIVE_OBSERVED",
            "adaptivity_classification": adaptivity,
            "next_action": "Begin S08: implement synchronous request-owned on-demand transfer of selected packed planes.",
        },
        "commands": {
            "training": "source ~/.venv/bin/activate && which python && python --version && PYTHONPATH=src:third_party/any-precision-llm python scripts/run_s07b.py",
            "checkpoint": str(args.checkpoint),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
