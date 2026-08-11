import torch

from qaq.model.request_state import QaqRequestState
from qaq.s04_manual import PrecisionTrace, _select_request_route


def test_completion_steps_cannot_change_prompt_features_or_routes():
    state = QaqRequestState("request-a", prompt_length=3, layer_count=1)
    state.validate_for_model(layer_count=1, feature_dim=2)
    prompt_hidden = torch.tensor([[[1.0, 2.0], [3.0, 5.0], [7.0, 11.0]]])
    policy_calls = []

    def policy(layer, unit, feature):
        policy_calls.append((layer, unit, feature.clone()))
        return 4 if unit == "attention" else 8

    prefill_trace = PrecisionTrace()
    for unit in ("attention", "ffn"):
        _select_request_route(
            request_state=state,
            layer_index=0,
            unit_type=unit,
            incoming_hidden=prompt_hidden,
            prompt_attention_mask=torch.ones(1, 3, dtype=torch.long),
            phase="prefill",
            routing_policy=policy,
            trace=prefill_trace,
        )
    saved_features = [feature.clone() for feature in state.attention_features + state.ffn_features]
    saved_routes = state.attention_routes[:] + state.ffn_routes[:]

    def fail_if_called(*args):
        raise AssertionError(f"decode invoked policy: {args}")

    for completion in (torch.tensor([[[13.0, 17.0]]]), torch.tensor([[[-5.0, 101.0]]])):
        decode_trace = PrecisionTrace()
        for unit in ("attention", "ffn"):
            _select_request_route(
                request_state=state,
                layer_index=0,
                unit_type=unit,
                incoming_hidden=completion,
                prompt_attention_mask=None,
                phase="decode",
                routing_policy=fail_if_called,
                trace=decode_trace,
            )
        assert all(torch.equal(before, after) for before, after in zip(saved_features, state.attention_features + state.ffn_features))
        assert saved_routes == state.attention_routes[:] + state.ffn_routes[:]
        assert all(not record.feature_computed and not record.policy_invoked for record in decode_trace.route_records)
    assert len(policy_calls) == 2
