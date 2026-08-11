from __future__ import annotations

import torch
from torch import nn

from qaq.model.request_state import QaqRequestState
from qaq.s04_manual import (
    ATTENTION_PROJECTIONS,
    FFN_PROJECTIONS,
    ManualRoutedQwen3ForCausalLM,
    PrecisionTrace,
    _RoutedPackedLinear,
)
from qaq.s06_soft import SoftRoutedQwen3ForCausalLM


class _DistinctPrecisionLinear(nn.Module):
    def __init__(self, linear: nn.Linear) -> None:
        super().__init__()
        self.linear = linear

    def forward(self, inputs: torch.Tensor, *, precision: int) -> torch.Tensor:
        return self.linear(inputs) * (1.0 if precision == 4 else 1.25)


def _soft_tiny_model() -> SoftRoutedQwen3ForCausalLM:
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(1729)
    config = Qwen3Config(
        vocab_size=97,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=36,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        torch_dtype="float32",
    )
    config._attn_implementation = "eager"
    static = Qwen3ForCausalLM(config).eval()
    for layer_index, layer in enumerate(static.model.layers):
        for unit_type, projections in (
            ("attention", ATTENTION_PROJECTIONS),
            ("ffn", FFN_PROJECTIONS),
        ):
            parent = layer.self_attn if unit_type == "attention" else layer.mlp
            for projection in projections:
                path = (
                    f"model.layers.{layer_index}."
                    f"{'self_attn' if unit_type == 'attention' else 'mlp'}.{projection}"
                )
                packed = _DistinctPrecisionLinear(getattr(parent, projection))
                setattr(
                    parent,
                    projection,
                    _RoutedPackedLinear(
                        packed,
                        layer_index=layer_index,
                        unit_type=unit_type,
                        module_path=path,
                    ),
                )
    return SoftRoutedQwen3ForCausalLM(ManualRoutedQwen3ForCausalLM(static).eval())


def test_one_soft_probability_pair_is_shared_within_attention_and_ffn():
    model = _soft_tiny_model()
    assert model.router_count == 72
    assert model.router_parameter_count == 72 * (32 * 128 + 128 + 128 * 2 + 2)
    state = QaqRequestState("s06-sharing", prompt_length=3)
    trace = PrecisionTrace()
    model(
        input_ids=torch.tensor([[1, 2, 3]]),
        attention_mask=torch.ones(1, 3, dtype=torch.long),
        use_cache=False,
        request_state=state,
        trace=trace,
    )
    assert len(trace.records) == 0
    assert len(trace.soft_records) == 252
    for layer_index in range(36):
        attention = [
            record
            for record in trace.soft_records
            if record.layer_index == layer_index and record.unit_type == "attention"
        ]
        ffn = [
            record
            for record in trace.soft_records
            if record.layer_index == layer_index and record.unit_type == "ffn"
        ]
        assert len(attention) == 4
        assert len(ffn) == 3
        assert len({id(record.probabilities) for record in attention}) == 1
        assert len({id(record.probabilities) for record in ffn}) == 1
        assert torch.equal(attention[0].probabilities, state.attention_probabilities[layer_index])
        assert torch.equal(ffn[0].probabilities, state.ffn_probabilities[layer_index])


def test_only_router_parameters_receive_gradients_and_change_on_one_step():
    model = _soft_tiny_model().train()
    state = QaqRequestState("s06-gradients", prompt_length=3)
    frozen_before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if not name.startswith("routers.")
    }
    router_before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if name.startswith("routers.")
    }
    outputs = model(
        input_ids=torch.tensor([[1, 2, 3]]),
        attention_mask=torch.ones(1, 3, dtype=torch.long),
        use_cache=False,
        request_state=state,
    )
    outputs.logits.square().mean().backward()
    audit = model.parameter_audit()
    assert audit["trainable_parameter_count"] == model.router_parameter_count
    assert audit["frozen_parameter_count"] > 0
    assert all(name.startswith("routers.") for name in audit["trainable_names"])
    router_grads = [parameter.grad for name, parameter in model.named_parameters() if name.startswith("routers.")]
    assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in router_grads)
    assert any(torch.count_nonzero(gradient).item() > 0 for gradient in router_grads)
    assert all(parameter.grad is None for name, parameter in model.named_parameters() if not name.startswith("routers."))

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    optimizer.step()
    assert any(
        not torch.equal(router_before[name], parameter)
        for name, parameter in model.named_parameters()
        if name.startswith("routers.")
    )
    assert all(
        torch.equal(frozen_before[name], parameter)
        for name, parameter in model.named_parameters()
        if not name.startswith("routers.")
    )
