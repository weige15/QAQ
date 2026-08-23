from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from qaq.model.manual import (
    ATTENTION_PROJECTIONS,
    FFN_PROJECTIONS,
    ManualRoutedQwen3ForCausalLM,
    _RoutedPackedLinear,
)
from qaq.model.request_state import QaqRequestState
from qaq.router.distillation import (
    CAUSAL_TARGET_IGNORE_INDEX,
    DistillationBatch,
    DistillationExample,
    RouteLogCollector,
    RouterCheckpointMetadata,
    RouterDistillationTrainer,
    TokenRange,
    build_router_optimizer,
    hard_route,
    load_router_checkpoint,
    route_records_from_request_state,
    route_statistics,
    save_router_checkpoint,
)
from qaq.router.soft_model import SoftRoutedQwen3ForCausalLM


class _DistinctPrecisionLinear(nn.Module):
    """Tiny deterministic packed-path fixture; production S06 uses the pinned backend."""

    def __init__(self, linear: nn.Linear) -> None:
        super().__init__()
        self.linear = linear

    def forward(self, inputs: torch.Tensor, *, precision: int) -> torch.Tensor:
        return self.linear(inputs) * (1.0 if precision == 4 else 1.25)


def _tiny_teacher_and_student():
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(1729)
    config = Qwen3Config(
        vocab_size=31,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=36,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=16,
        torch_dtype="float32",
    )
    config._attn_implementation = "eager"
    teacher = Qwen3ForCausalLM(config).eval()
    static = Qwen3ForCausalLM(config).eval()
    static.load_state_dict(copy.deepcopy(teacher.state_dict()))
    for layer_index, layer in enumerate(static.model.layers):
        for unit_type, projections in (
            ("attention", ATTENTION_PROJECTIONS),
            ("ffn", FFN_PROJECTIONS),
        ):
            parent = layer.self_attn if unit_type == "attention" else layer.mlp
            for projection in projections:
                path = f"model.layers.{layer_index}.{'self_attn' if unit_type == 'attention' else 'mlp'}.{projection}"
                setattr(
                    parent,
                    projection,
                    _RoutedPackedLinear(
                        _DistinctPrecisionLinear(getattr(parent, projection)),
                        layer_index=layer_index,
                        unit_type=unit_type,
                        module_path=path,
                    ),
                )
    student = SoftRoutedQwen3ForCausalLM(
        ManualRoutedQwen3ForCausalLM(static).eval(), hidden_width=4, temperature=2.0
    )
    return teacher, student


@pytest.mark.parametrize("step_count", [2])
def test_tiny_end_to_end_router_distillation_smoke(tmp_path, step_count):
    torch.manual_seed(1729)
    teacher, student = _tiny_teacher_and_student()
    optimizer, audit = build_router_optimizer(student, lr=1e-2, optimizer_cls=torch.optim.SGD)
    states = []

    def make_state(request_id: str, prompt_length: int):
        state = QaqRequestState(request_id, prompt_length, layer_count=36)
        states.append(state)
        return state

    trainer = RouterDistillationTrainer(
        teacher,
        student,
        optimizer,
        temperature=2.0,
        request_state_factory=make_state,
    )
    example = DistillationExample(
        example_id="smoke-0",
        tokenizer_revision="tok-r1",
        input_ids=torch.tensor([1, 2, 3, 4]),
        target_ids=torch.tensor([2, 3, 4, CAUSAL_TARGET_IGNORE_INDEX]),
        attention_mask=torch.ones(4, dtype=torch.bool),
        completion_loss_mask=torch.tensor([0, 1, 1, 0]),
        prompt_token_range=TokenRange(0, 2),
        completion_token_range=TokenRange(2, 4),
    )
    batch = DistillationBatch.from_examples([example])
    results = [trainer.step(batch) for _ in range(step_count)]
    assert all(torch.isfinite(torch.tensor(result.loss)) for result in results)
    assert all(result.router_gradient_norm > 0 for result in results)
    assert all(result.router_parameter_changed for result in results)
    assert audit.scalar_count == student.router_parameter_count

    records = route_records_from_request_state(states[-1].request_id, states[-1])
    collector = RouteLogCollector(layer_count=36, request_ids=(states[-1].request_id,))
    for record in records:
        collector.add(record)
    assert len(collector.finalize()) == 72
    stats = route_statistics(records, distillation_loss=results[-1].loss)
    assert torch.isfinite(torch.tensor(stats["mean_entropy"]))
    assert stats["attention_vs_ffn_distribution"].keys() == {"attention", "ffn"}

    metadata = RouterCheckpointMetadata(
        model_repository="fixture/qwen3",
        model_revision="model-r1",
        quantized_checkpoint_id="fixture-packed-r1",
        quantized_checkpoint_hash="sha256:fixture",
        any_precision_revision="fixture-any-r1",
        router_architecture={"feature_dim": 16, "hidden_width": 4, "temperature": 2.0},
        training_step=step_count,
        training_step_metadata={"seed": 1729, "smoke_only": True},
    )
    checkpoint = tmp_path / "router-smoke.pt"
    save_router_checkpoint(checkpoint, student.routers, metadata, optimizer=optimizer)
    restored = copy.deepcopy(student.routers)
    restored_wrapper = nn.Module()
    restored_wrapper.add_module("routers", restored)
    restored_optimizer, _ = build_router_optimizer(
        restored_wrapper, lr=1e-2, optimizer_cls=torch.optim.SGD
    )
    load_router_checkpoint(checkpoint, restored, metadata, optimizer=restored_optimizer)
    feature = states[-1].attention_features[0]
    probability_before = student.routers["attention_0"](feature)
    probability_after = restored["attention_0"](feature)
    assert torch.equal(probability_before, probability_after)
    assert int(hard_route(probability_before)) == int(hard_route(probability_after))
