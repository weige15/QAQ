from __future__ import annotations

import pytest
import torch
from torch import nn

from qaq.model.manual import (
    ATTENTION_PROJECTIONS,
    FFN_PROJECTIONS,
    ManualRoutedQwen3ForCausalLM,
    PrecisionPlan,
    PrecisionTrace,
    _RoutedPackedLinear,
)
from qaq.model.request_state import (
    LOOKAHEAD_ATTENTION_ONE_UNIT,
    SAME_UNIT,
    QaqRequestState,
)


class _PrecisionLinear(nn.Module):
    """Test adapter preserving the S04 explicit-precision call contract."""

    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.linear = linear

    def forward(self, inputs, *, precision):
        del precision
        return self.linear(inputs)


def _tiny_model():
    from transformers import Qwen3Config, Qwen3ForCausalLM

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
            for projection in projections:
                path = f"model.layers.{layer_index}.{'self_attn' if unit_type == 'attention' else 'mlp'}.{projection}"
                parent = layer.self_attn if unit_type == "attention" else layer.mlp
                packed = getattr(parent, projection)
                packed = _PrecisionLinear(packed)
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
    return ManualRoutedQwen3ForCausalLM(static).eval()


def test_real_qwen3_wrapper_prefill_and_decode_reuse_request_routes():
    model = _tiny_model()
    prompt = torch.tensor([[1, 2, 3]])
    mask = torch.ones_like(prompt)
    state = QaqRequestState("tiny-request", prompt_length=3)
    prefill_trace = PrecisionTrace()
    with torch.inference_mode():
        model(
            input_ids=prompt,
            attention_mask=mask,
            use_cache=False,
            precision_plan=PrecisionPlan.uniform(4),
            request_state=state,
            phase="prefill",
            trace=prefill_trace,
        )
    assert len(prefill_trace.route_records) == 72
    assert all(
        item.feature_computed and item.policy_invoked for item in prefill_trace.route_records
    )
    saved = [feature.clone() for feature in state.attention_features + state.ffn_features]
    routes = state.attention_routes[:] + state.ffn_routes[:]

    def fail_policy(*args):
        raise AssertionError("decode must not invoke a policy")

    decode_trace = PrecisionTrace()
    with torch.inference_mode():
        model(
            input_ids=torch.tensor([[42]]),
            attention_mask=torch.ones(1, 1, dtype=torch.long),
            use_cache=False,
            request_state=state,
            phase="decode",
            routing_policy=fail_policy,
            trace=decode_trace,
        )
    assert all(
        not item.feature_computed and not item.policy_invoked for item in decode_trace.route_records
    )
    assert routes == state.attention_routes[:] + state.ffn_routes[:]
    assert all(
        torch.equal(before, after)
        for before, after in zip(saved, state.attention_features + state.ffn_features)
    )


def test_explicit_same_unit_hard_mode_matches_default_query_routing_numerics_and_trace():
    torch.manual_seed(1729)
    model = _tiny_model()
    prompt = torch.tensor([[1, 2, 3]])
    mask = torch.ones_like(prompt)
    default_state = QaqRequestState("same-hard", prompt_length=3)
    explicit_state = QaqRequestState("same-hard", prompt_length=3, routing_timing=SAME_UNIT)
    default_trace = PrecisionTrace()
    explicit_trace = PrecisionTrace()
    with torch.inference_mode():
        default = model(
            input_ids=prompt,
            attention_mask=mask,
            use_cache=False,
            precision_plan=PrecisionPlan.uniform(4),
            request_state=default_state,
            phase="prefill",
            trace=default_trace,
        )
        explicit = model(
            input_ids=prompt,
            attention_mask=mask,
            use_cache=False,
            precision_plan=PrecisionPlan.uniform(4),
            request_state=explicit_state,
            phase="prefill",
            trace=explicit_trace,
        )
    assert torch.equal(default.logits, explicit.logits)
    assert default_trace.events == explicit_trace.events
    assert default_trace.route_records == explicit_trace.route_records
    assert default_state.attention_routes == explicit_state.attention_routes
    assert default_state.ffn_routes == explicit_state.ffn_routes


def test_lookahead_hard_decode_reuses_prefill_routes_without_features_or_policy_calls():
    torch.manual_seed(1729)
    model = _tiny_model()
    state = QaqRequestState(
        "lookahead-decode",
        prompt_length=3,
        routing_timing=LOOKAHEAD_ATTENTION_ONE_UNIT,
    )
    calls = []

    def policy(layer, unit_type, feature):
        del feature
        calls.append((layer, unit_type))
        return 4 if (layer + (unit_type == "ffn")) % 2 == 0 else 8

    with torch.inference_mode():
        model(
            input_ids=torch.tensor([[1, 2, 3]]),
            attention_mask=torch.ones(1, 3, dtype=torch.long),
            use_cache=False,
            request_state=state,
            phase="prefill",
            routing_policy=policy,
        )
    routes = state.attention_routes[:] + state.ffn_routes[:]
    features = [value.clone() for value in state.attention_features + state.ffn_features]
    provenance = state.attention_provenance[:]
    prefill_calls = calls[:]

    def fail_policy(*args):
        raise AssertionError("decode must not invoke a policy")

    decode_trace = PrecisionTrace()
    with torch.inference_mode():
        model(
            input_ids=torch.tensor([[42]]),
            attention_mask=torch.ones(1, 1, dtype=torch.long),
            use_cache=False,
            request_state=state,
            phase="decode",
            routing_policy=fail_policy,
            trace=decode_trace,
        )
    assert len(prefill_calls) == 72
    assert calls == prefill_calls
    assert routes == state.attention_routes[:] + state.ffn_routes[:]
    assert provenance == state.attention_provenance
    assert all(
        torch.equal(before, after)
        for before, after in zip(
            features,
            state.attention_features + state.ffn_features,
            strict=True,
        )
    )
    assert len(decode_trace.route_records) == 72
    assert all(
        not record.feature_computed and not record.policy_invoked
        for record in decode_trace.route_records
    )


def test_missing_early_attention_route_fails_before_target_packed_execution():
    torch.manual_seed(1729)
    model = _tiny_model()
    state = QaqRequestState(
        "missing-lookahead",
        prompt_length=3,
        routing_timing=LOOKAHEAD_ATTENTION_ONE_UNIT,
    )
    trace = PrecisionTrace()

    def policy(layer, unit_type, feature):
        del feature
        if layer == 0 and unit_type == "ffn":
            state.attention_routes[1] = None
        return 4

    with pytest.raises(RuntimeError, match="missing its early route"):
        model(
            input_ids=torch.tensor([[1, 2, 3]]),
            attention_mask=torch.ones(1, 3, dtype=torch.long),
            use_cache=False,
            request_state=state,
            phase="prefill",
            routing_policy=policy,
            trace=trace,
        )
    assert not any(
        record.layer_index == 1 and record.unit_type == "attention" for record in trace.records
    )
    assert not any(
        event.layer_index == 1
        and event.unit_type == "attention"
        and event.event in ("target_attention_execution", "unit_execute")
        for event in trace.events
    )


@pytest.mark.parametrize("input_kind", ["input_ids", "inputs_embeds"])
def test_query_request_wrapper_rejects_batches_larger_than_one(input_kind):
    model = _tiny_model()
    state = QaqRequestState("batched-request", prompt_length=3)
    inputs = {
        "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
        "inputs_embeds": torch.randn(2, 3, 32),
    }
    with pytest.raises(ValueError, match="batch-size-one"):
        model(
            **{input_kind: inputs[input_kind]},
            attention_mask=torch.ones(1, 3, dtype=torch.long),
            use_cache=False,
            precision_plan=PrecisionPlan.uniform(4),
            request_state=state,
            phase="prefill",
        )
