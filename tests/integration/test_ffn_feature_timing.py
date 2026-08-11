import torch

from qaq.model.request_state import QaqRequestState
from qaq.s04_manual import PrecisionTrace, _select_request_route


def test_ffn_timing_is_incoming_feature_route_then_unit():
    state = QaqRequestState("request-a", prompt_length=2, layer_count=1)
    state.validate_for_model(layer_count=1, feature_dim=2)
    trace = PrecisionTrace()
    _select_request_route(
        request_state=state,
        layer_index=0,
        unit_type="ffn",
        incoming_hidden=torch.tensor([[[10.0, 20.0], [30.0, 40.0]]]),
        prompt_attention_mask=torch.tensor([[1, 1]]),
        phase="prefill",
        routing_policy=lambda layer, unit, feature: 8,
        trace=trace,
    )
    trace.record_event(
        request_id="request-a",
        layer_index=0,
        unit_type="ffn",
        phase="prefill",
        event="unit_execute",
        precision=8,
    )
    assert [event.event for event in trace.events] == [
        "incoming_hidden",
        "feature_computed",
        "route_available",
        "unit_execute",
    ]
    assert state.ffn_features[0].tolist() == [20.0, 30.0]
    assert trace.route_records[0].feature_computed
    assert trace.route_records[0].policy_invoked

