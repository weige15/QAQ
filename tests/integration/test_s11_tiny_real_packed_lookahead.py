from __future__ import annotations

from collections import Counter

import torch

from qaq.model.manual import (
    ATTENTION_PROJECTIONS,
    FFN_PROJECTIONS,
    ManualRoutedQwen3ForCausalLM,
    PrecisionTrace,
    _RoutedPackedLinear,
)
from qaq.model.request_state import (
    LOOKAHEAD_ATTENTION_ONE_UNIT,
    POST_ATTENTION_PRE_FFN,
    QaqRequestState,
)
from qaq.quantization.backend import load_pinned_backend


def _tiny_real_packed_model() -> ManualRoutedQwen3ForCausalLM:
    from transformers import Qwen3Config, Qwen3ForCausalLM

    device = torch.device("cuda:0")
    torch.manual_seed(1729)
    torch.cuda.manual_seed_all(1729)
    config = Qwen3Config(
        vocab_size=97,
        hidden_size=64,
        intermediate_size=64,
        num_hidden_layers=36,
        num_attention_heads=1,
        num_key_value_heads=1,
        max_position_embeddings=32,
        torch_dtype="float16",
    )
    config._attn_implementation = "eager"
    static = Qwen3ForCausalLM(config).to(device=device, dtype=torch.float16).eval()
    AnyPrecisionLinear, _, _ = load_pinned_backend()

    for layer_index, layer in enumerate(static.model.layers):
        for unit_type, projections in (
            ("attention", ATTENTION_PROJECTIONS),
            ("ffn", FFN_PROJECTIONS),
        ):
            parent = layer.self_attn if unit_type == "attention" else layer.mlp
            for projection in projections:
                original = getattr(parent, projection)
                packed = AnyPrecisionLinear(
                    original.in_features,
                    original.out_features,
                    [4, 8],
                    bias=False,
                    precisions=[4, 8],
                    device=device,
                    dtype=torch.float16,
                )
                with torch.no_grad():
                    packed.qweight.fill_(0x55555555)
                    for bits in (4, 8):
                        levels = torch.linspace(
                            -0.015,
                            0.015,
                            1 << bits,
                            device=device,
                            dtype=torch.float16,
                        )
                        packed._buffers[f"lut{bits}"].copy_(
                            levels.unsqueeze(0).expand(original.out_features, -1)
                        )
                module_path = (
                    f"model.layers.{layer_index}."
                    f"{'self_attn' if unit_type == 'attention' else 'mlp'}.{projection}"
                )
                setattr(
                    parent,
                    projection,
                    _RoutedPackedLinear(
                        packed,
                        layer_index=layer_index,
                        unit_type=unit_type,
                        module_path=module_path,
                    ),
                )
    return ManualRoutedQwen3ForCausalLM(static).eval()


def _execute(model: ManualRoutedQwen3ForCausalLM, request_id: str):
    state = QaqRequestState(
        request_id,
        prompt_length=3,
        routing_timing=LOOKAHEAD_ATTENTION_ONE_UNIT,
    )
    calls = []

    def policy(layer, unit_type, feature):
        del feature
        calls.append((layer, unit_type))
        return 4 if (layer + int(unit_type == "ffn")) % 2 == 0 else 8

    trace = PrecisionTrace()
    with torch.inference_mode():
        output = model(
            input_ids=torch.tensor([[1, 2, 3]], device="cuda:0"),
            attention_mask=torch.ones(1, 3, dtype=torch.long, device="cuda:0"),
            use_cache=False,
            request_state=state,
            phase="prefill",
            routing_policy=policy,
            trace=trace,
        )
    torch.cuda.synchronize()
    return output.logits.detach(), state, trace, calls


def test_tiny_real_packed_lookahead_execution_order_ownership_and_determinism():
    model = _tiny_real_packed_model()
    packed_modules = [
        module.packed for module in model.modules() if isinstance(module, _RoutedPackedLinear)
    ]
    assert len(packed_modules) == 252
    assert all(module.__class__.__name__ == "AnyPrecisionLinear" for module in packed_modules)

    first_logits, state, trace, calls = _execute(model, "s11-real-packed")
    second_logits, second_state, second_trace, second_calls = _execute(
        model, "s11-real-packed"
    )

    assert torch.isfinite(first_logits).all()
    assert torch.equal(first_logits, second_logits)
    assert state.attention_routes == second_state.attention_routes
    assert state.ffn_routes == second_state.ffn_routes
    assert calls == second_calls
    assert trace.events == second_trace.events
    assert len(trace.records) == 252
    assert len(trace.route_records) == 72
    assert sum(route is not None for route in state.attention_routes) == 36
    assert sum(route is not None for route in state.ffn_routes) == 36
    assert state.attention_provenance[0] is None
    assert state.early_attention_route_consumed == (False,) + (True,) * 35

    call_counts = Counter(calls)
    assert call_counts[(0, "attention")] == 1
    assert call_counts[(1, "attention")] == 1
    assert all(call_counts[(layer, "attention")] == 1 for layer in range(36))
    assert all(call_counts[(layer, "ffn")] == 1 for layer in range(36))

    expected_chain = [
        "source_attention_execution",
        "source_attention_residual_completion",
        "lookahead_target_feature_computed",
        "lookahead_target_route_available",
        "source_ffn_execution",
        "target_layer_entry",
        "target_route_consumed",
        "target_attention_execution",
    ]
    for source_layer in range(35):
        events = [
            event
            for event in trace.events
            if event.source_layer == source_layer and event.target_layer == source_layer + 1
        ]
        assert [event.event for event in events] == expected_chain
        assert all(event.target_unit_type == "attention" for event in events)
        assert all(event.source_point == POST_ATTENTION_PRE_FFN for event in events)
        assert all(
            event.routing_timing == LOOKAHEAD_ATTENTION_ONE_UNIT for event in events
        )

        route_available = trace.events.index(events[3])
        source_ffn = next(
            index
            for index, event in enumerate(trace.events)
            if event.layer_index == source_layer
            and event.unit_type == "ffn"
            and event.event == "unit_execute"
        )
        assert route_available < source_ffn
        assert trace.events.index(events[6]) < trace.events.index(events[7])

    provenance_events = [event for event in trace.events if event.source_layer is not None]
    assert {event.target_layer for event in provenance_events} == set(range(1, 36))
    assert not any(event.target_layer == 36 for event in provenance_events)
