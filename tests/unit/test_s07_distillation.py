from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from qaq.router.network import SoftPrecisionRouter
from qaq.s07_distillation import (
    DistillationExample,
    ExecutionInputs,
    RouteLogCollector,
    RouteLogRecord,
    RouterCheckpointMetadata,
    TokenRange,
    audit_router_optimizer,
    build_router_optimizer,
    hard_route,
    load_router_checkpoint,
    masked_kl_distillation_loss,
    route_statistics,
    save_router_checkpoint,
    validate_execution_alignment,
)


def _example(*, completion_mask: list[int] | None = None) -> DistillationExample:
    return DistillationExample(
        example_id="example-0",
        tokenizer_revision="tok-r1",
        input_ids=torch.tensor([11, 12, 13, 14, 0]),
        target_ids=torch.tensor([11, 12, 13, 14, 0]),
        attention_mask=torch.tensor([1, 1, 1, 1, 0]),
        completion_loss_mask=torch.tensor(completion_mask or [0, 0, 1, 1, 0]),
        prompt_token_range=TokenRange(0, 2),
        completion_token_range=TokenRange(2, 4),
    )


def test_kd_mask_is_explicit_and_numerically_correct():
    teacher = torch.tensor([[[4.0, 0.0], [100.0, -100.0], [1.0, 3.0], [9.0, -9.0]]])
    student = torch.tensor([[[0.0, 1.0], [0.0, 1.0], [2.0, 2.0], [1.0, 0.0]]], requires_grad=True)
    mask = torch.tensor([[0, 0, 1, 0]], dtype=torch.bool)
    temperature = 2.0
    actual = masked_kl_distillation_loss(teacher, student, mask, temperature=temperature)
    teacher_log = torch.log_softmax(teacher[0, 2] / temperature, dim=-1)
    student_log = torch.log_softmax(student[0, 2] / temperature, dim=-1)
    expected = (teacher_log.exp() * (teacher_log - student_log)).sum() * temperature**2
    assert torch.allclose(actual, expected, atol=1e-7, rtol=0)
    actual.backward()
    assert student.grad is not None and torch.isfinite(student.grad).all()


def test_prompt_and_padding_changes_do_not_affect_loss_but_completion_does():
    teacher = torch.zeros(1, 4, 3)
    student = torch.zeros(1, 4, 3)
    mask = torch.tensor([[0, 0, 1, 0]], dtype=torch.bool)
    baseline = masked_kl_distillation_loss(teacher, student, mask, temperature=1.0)
    changed_prompt_padding = teacher.clone()
    changed_prompt_padding[0, 0] = torch.tensor([100.0, -20.0, 3.0])
    changed_prompt_padding[0, 3] = torch.tensor([-100.0, 20.0, 8.0])
    assert torch.equal(
        baseline,
        masked_kl_distillation_loss(changed_prompt_padding, student, mask, temperature=1.0),
    )
    changed_completion = teacher.clone()
    changed_completion[0, 2] = torch.tensor([10.0, -10.0, 0.0])
    assert not torch.equal(
        baseline,
        masked_kl_distillation_loss(changed_completion, student, mask, temperature=1.0),
    )


def test_zero_completion_and_invalid_batch_alignment_fail_clearly():
    with pytest.raises(ValueError, match="zero valid completion"):
        masked_kl_distillation_loss(
            torch.zeros(1, 2, 3), torch.zeros(1, 2, 3), torch.zeros(1, 2), temperature=1.0
        )
    with pytest.raises(ValueError, match="zero valid completion"):
        DistillationExample(
            example_id="zero",
            tokenizer_revision="tok-r1",
            input_ids=torch.tensor([1, 2]),
            target_ids=torch.tensor([1, 2]),
            attention_mask=torch.ones(2, dtype=torch.long),
            completion_loss_mask=torch.zeros(2, dtype=torch.long),
            prompt_token_range=TokenRange(0, 1),
            completion_token_range=TokenRange(1, 2),
        )
    left = ExecutionInputs(
        "tok-r1",
        torch.tensor([[1, 2]]),
        torch.ones(1, 2),
        torch.tensor([[0, 1]]),
        torch.tensor([[0, 1]]),
    )
    right = ExecutionInputs(
        "tok-r2",
        torch.tensor([[1, 2]]),
        torch.ones(1, 2),
        torch.tensor([[0, 1]]),
        torch.tensor([[0, 1]]),
    )
    with pytest.raises(ValueError, match="tokenizer revisions"):
        validate_execution_alignment(left, right)


def test_hard_route_mapping_shape_and_tie_determinism():
    assert hard_route(torch.tensor([1.0, 0.0])) == 4
    assert hard_route(torch.tensor([0.0, 1.0])) == 8
    assert hard_route(torch.tensor([0.5, 0.5])) == 4
    mapped = hard_route(torch.tensor([[0.5, 0.5], [0.1, 0.9], [0.9, 0.1]]))
    assert torch.equal(mapped, torch.tensor([4, 8, 4]))
    with pytest.raises(ValueError, match="shape"):
        hard_route(torch.ones(3))


class _RouterOnlyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.routers = nn.ModuleDict({"attention_0": SoftPrecisionRouter(4, hidden_width=4)})
        self.base = nn.Linear(4, 4)


def test_optimizer_audit_is_router_only_and_reports_counts():
    model = _RouterOnlyModel()
    optimizer, audit = build_router_optimizer(model, lr=1e-2, optimizer_cls=torch.optim.SGD)
    assert audit.tensor_count == 4
    assert audit.scalar_count == sum(parameter.numel() for parameter in model.routers.parameters())
    assert audit.included_name_prefixes == ("routers.",)
    assert audit_router_optimizer(model, optimizer).included_names == audit.included_names
    assert all(
        id(parameter) not in {id(item) for item in model.base.parameters()}
        for group in optimizer.param_groups
        for parameter in group["params"]
    )


def test_router_checkpoint_metadata_and_probability_hard_round_trip(tmp_path):
    torch.manual_seed(1729)
    router = SoftPrecisionRouter(4, hidden_width=4)
    restored = copy.deepcopy(router)
    metadata = RouterCheckpointMetadata(
        model_repository="Qwen/Qwen3-4B",
        model_revision="model-r1",
        quantized_checkpoint_id="packed-r1",
        quantized_checkpoint_hash="sha256:packed",
        any_precision_revision="any-r1",
        router_architecture={"feature_dim": 4, "hidden_width": 4, "activation": "GELU"},
        training_step=3,
        training_step_metadata={"seed": 1729, "smoke": True},
    )
    path = tmp_path / "router.pt"
    save_router_checkpoint(path, router, metadata)
    load_router_checkpoint(path, restored, metadata)
    feature = torch.tensor([1.0, -2.0, 0.5, 3.0])
    assert torch.equal(router(feature), restored(feature))
    assert hard_route(router(feature)) == hard_route(restored(feature))
    incompatible = copy.copy(metadata)
    incompatible = RouterCheckpointMetadata(**{**metadata.to_dict(), "model_revision": "other"})
    with pytest.raises(ValueError, match="incompatible"):
        load_router_checkpoint(path, restored, incompatible)


def test_route_log_coverage_statistics_and_variation():
    collector = RouteLogCollector(layer_count=2, request_ids=("a", "b"))
    for request_id in ("a", "b"):
        for layer in range(2):
            for unit_type in ("attention", "ffn"):
                probability = (
                    torch.tensor([0.75, 0.25]) if request_id == "a" else torch.tensor([0.25, 0.75])
                )
                collector.add(
                    RouteLogRecord.from_probabilities(request_id, layer, unit_type, probability)
                )
    records = collector.finalize()
    assert len(records) == 8
    stats = route_statistics(records, distillation_loss=0.25)
    assert stats["distillation_loss"] == 0.25
    assert stats["entropy_log_base"] == 2.0
    assert stats["hard_fraction_4"] == 0.5
    assert stats["route_variation_across_prompts"]["changed_unit_count"] == 4
    duplicate = RouteLogCollector(layer_count=1, request_ids=("a",))
    record = RouteLogRecord.from_probabilities("a", 0, "attention", torch.tensor([1.0, 0.0]))
    duplicate.add(record)
    duplicate.add(record)
    with pytest.raises(AssertionError, match="duplicate"):
        duplicate.finalize()
