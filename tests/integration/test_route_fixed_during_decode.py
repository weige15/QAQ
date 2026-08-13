import torch

from qaq.model.request_state import QaqRequestState
from qaq.model.manual import PrecisionTrace, _select_request_route


def test_decode_reuses_each_prefill_route_without_policy_calls():
    state = QaqRequestState("request-a", prompt_length=2, layer_count=2)
    state.validate_for_model(layer_count=2, feature_dim=3)
    prompt = torch.ones(1, 2, 3)
    selected = {"attention": [4, 8], "ffn": [8, 4]}
    calls = []

    def policy(layer, unit, feature):
        calls.append((layer, unit))
        return selected[unit][layer]

    prefill = PrecisionTrace()
    for layer in range(2):
        for unit in ("attention", "ffn"):
            _select_request_route(
                request_state=state,
                layer_index=layer,
                unit_type=unit,
                incoming_hidden=prompt,
                prompt_attention_mask=torch.tensor([[1, 1]]),
                phase="prefill",
                routing_policy=policy,
                trace=prefill,
            )
    assert calls == [(0, "attention"), (0, "ffn"), (1, "attention"), (1, "ffn")]

    def fail_policy(*args):
        raise AssertionError("decode must not invoke the policy")

    decode = PrecisionTrace()
    observed = []
    for layer in range(2):
        for unit in ("attention", "ffn"):
            observed.append(
                _select_request_route(
                    request_state=state,
                    layer_index=layer,
                    unit_type=unit,
                    incoming_hidden=torch.randn(1, 1, 3),
                    prompt_attention_mask=None,
                    phase="decode",
                    routing_policy=fail_policy,
                    trace=decode,
                )
            )
    assert observed == [4, 8, 8, 4]
    assert all(
        not item.feature_computed and not item.policy_invoked for item in decode.route_records
    )
