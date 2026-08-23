from __future__ import annotations

import torch
from torch import nn

from qaq.model.manual import (
    ATTENTION_PROJECTIONS,
    FFN_PROJECTIONS,
    ManualRoutedQwen3ForCausalLM,
    PrecisionTrace,
    _RoutedPackedLinear,
)
from qaq.model.request_state import (
    LOOKAHEAD_ATTENTION_ONE_UNIT,
    SAME_UNIT,
    QaqRequestState,
)
from qaq.router.distillation import request_state_expected_bit_cost
from qaq.router.network import THREE_WAY_CANDIDATE_BITS
from qaq.router.soft_model import SoftRoutedQwen3ForCausalLM


class _DistinctPrecisionLinear(nn.Module):
    def __init__(self, linear: nn.Linear) -> None:
        super().__init__()
        self.linear = linear

    def forward(self, inputs: torch.Tensor, *, precision: int) -> torch.Tensor:
        return self.linear(inputs) * (1.0 if precision == 4 else 1.25)


def _soft_tiny_model(
    candidate_bits: tuple[int, ...] = (4, 8),
) -> SoftRoutedQwen3ForCausalLM:
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
    return SoftRoutedQwen3ForCausalLM(
        ManualRoutedQwen3ForCausalLM(static).eval(), candidate_bits=candidate_bits
    )


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


def test_three_way_soft_router_propagates_explicit_ordering():
    model = _soft_tiny_model(THREE_WAY_CANDIDATE_BITS)
    state = QaqRequestState("s10b-soft", prompt_length=3, candidate_bits=THREE_WAY_CANDIDATE_BITS)
    trace = PrecisionTrace()
    model(
        input_ids=torch.tensor([[1, 2, 3]]),
        attention_mask=torch.ones(1, 3, dtype=torch.long),
        use_cache=False,
        request_state=state,
        trace=trace,
    )
    assert all(record.candidate_bits == THREE_WAY_CANDIDATE_BITS for record in trace.soft_records)
    assert all(probability.shape == (3,) for probability in state.attention_probabilities)
    assert all(probability.shape == (3,) for probability in state.ffn_probabilities)


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
    router_grads = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if name.startswith("routers.")
    ]
    assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in router_grads)
    assert any(torch.count_nonzero(gradient).item() > 0 for gradient in router_grads)
    assert all(
        parameter.grad is None
        for name, parameter in model.named_parameters()
        if not name.startswith("routers.")
    )

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


def test_explicit_same_unit_soft_mode_matches_the_default_numerics_and_trace():
    model = _soft_tiny_model().eval()
    kwargs = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.ones(1, 3, dtype=torch.long),
        "use_cache": False,
    }
    default_state = QaqRequestState("same-soft", prompt_length=3)
    explicit_state = QaqRequestState("same-soft", prompt_length=3, routing_timing=SAME_UNIT)
    default_trace = PrecisionTrace()
    explicit_trace = PrecisionTrace()
    default = model(request_state=default_state, trace=default_trace, **kwargs)
    explicit = model(request_state=explicit_state, trace=explicit_trace, **kwargs)

    assert torch.equal(default.logits, explicit.logits)
    assert default_trace.events == explicit_trace.events
    assert default_trace.route_records == explicit_trace.route_records
    assert all(
        torch.equal(left, right)
        for left, right in zip(
            default_state.attention_probabilities + default_state.ffn_probabilities,
            explicit_state.attention_probabilities + explicit_state.ffn_probabilities,
            strict=True,
        )
    )


def test_soft_lookahead_updates_only_the_target_router_and_keeps_packed_base_frozen():
    model = _soft_tiny_model().train()
    state = QaqRequestState(
        "s11-soft",
        prompt_length=3,
        routing_timing=LOOKAHEAD_ATTENTION_ONE_UNIT,
    )
    trace = PrecisionTrace()
    calls = {"attention_0": 0, "attention_1": 0}
    hooks = [
        model.routers[name].register_forward_hook(
            lambda module, inputs, output, key=name: calls.__setitem__(key, calls[key] + 1)
        )
        for name in calls
    ]
    frozen_before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if not name.startswith("routers.")
    }
    output = model(
        input_ids=torch.tensor([[1, 2, 3]]),
        attention_mask=torch.ones(1, 3, dtype=torch.long),
        use_cache=False,
        request_state=state,
        trace=trace,
    )
    for hook in hooks:
        hook.remove()

    assert torch.isfinite(output.logits).all()
    assert calls == {"attention_0": 1, "attention_1": 1}
    assert sum(value is not None for value in state.attention_probabilities) == 36
    assert sum(value is not None for value in state.ffn_probabilities) == 36
    assert state.attention_provenance[0] is None
    assert all(value is not None for value in state.attention_provenance[1:])
    assert state.early_attention_probability_consumed == (False,) + (True,) * 35
    assert torch.isfinite(request_state_expected_bit_cost(state))

    model.zero_grad(set_to_none=True)
    target_router = model.routers["attention_1"]
    target_before = {
        name: parameter.detach().clone() for name, parameter in target_router.named_parameters()
    }
    target_probability = state.attention_probabilities[1]
    target_probability[0].backward()
    target_gradients = [parameter.grad for parameter in target_router.parameters()]
    assert all(
        gradient is not None and torch.isfinite(gradient).all() for gradient in target_gradients
    )
    assert any(torch.count_nonzero(gradient).item() for gradient in target_gradients)
    assert all(parameter.grad is None for parameter in model.routers["attention_0"].parameters())
    assert all(
        parameter.grad is None
        for name, parameter in model.named_parameters()
        if not name.startswith("routers.")
    )

    optimizer = torch.optim.SGD(target_router.parameters(), lr=1e-3)
    optimizer.step()
    assert any(
        not torch.equal(target_before[name], parameter)
        for name, parameter in target_router.named_parameters()
    )
    assert all(
        torch.equal(frozen_before[name], parameter)
        for name, parameter in model.named_parameters()
        if not name.startswith("routers.")
    )
