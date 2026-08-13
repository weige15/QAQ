import torch

from qaq.model.request_state import QaqRequestState
from qaq.model.manual import PrecisionTrace, _select_request_route


def test_interleaved_request_states_do_not_share_features_or_routes():
    state_a = QaqRequestState("request-a", prompt_length=2, layer_count=1)
    state_b = QaqRequestState("request-b", prompt_length=2, layer_count=1)
    for state in (state_a, state_b):
        state.validate_for_model(layer_count=1, feature_dim=2)
    trace_a = PrecisionTrace()
    trace_b = PrecisionTrace()
    _select_request_route(
        request_state=state_a,
        layer_index=0,
        unit_type="attention",
        incoming_hidden=torch.tensor([[[1.0, 1.0], [3.0, 3.0]]]),
        prompt_attention_mask=torch.tensor([[1, 1]]),
        phase="prefill",
        routing_policy=lambda layer, unit, feature: 4,
        trace=trace_a,
    )
    _select_request_route(
        request_state=state_b,
        layer_index=0,
        unit_type="attention",
        incoming_hidden=torch.tensor([[[10.0, 20.0], [30.0, 40.0]]]),
        prompt_attention_mask=torch.tensor([[1, 1]]),
        phase="prefill",
        routing_policy=lambda layer, unit, feature: 8,
        trace=trace_b,
    )
    assert state_a.attention_routes == [4]
    assert state_b.attention_routes == [8]
    assert not torch.equal(state_a.attention_features[0], state_b.attention_features[0])
    assert trace_a.route_records[0].request_id == "request-a"
    assert trace_b.route_records[0].request_id == "request-b"
