from __future__ import annotations

import torch
from torch import nn

from qaq.model.request_state import QaqRequestState
from qaq.s04_manual import (
    ATTENTION_PROJECTIONS,
    FFN_PROJECTIONS,
    ManualRoutedQwen3ForCausalLM,
    PrecisionPlan,
    PrecisionTrace,
    _RoutedPackedLinear,
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
    assert all(item.feature_computed and item.policy_invoked for item in prefill_trace.route_records)
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
    assert all(not item.feature_computed and not item.policy_invoked for item in decode_trace.route_records)
    assert routes == state.attention_routes[:] + state.ffn_routes[:]
    assert all(torch.equal(before, after) for before, after in zip(saved, state.attention_features + state.ffn_features))
