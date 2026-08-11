import pytest
import torch

from qaq.model.request_state import QaqRequestState


def test_request_state_initializes_owned_per_layer_slots():
    state = QaqRequestState("request-a", prompt_length=3, layer_count=2)
    assert state.attention_routes == [None, None]
    assert state.ffn_routes == [None, None]
    assert state.attention_features == [None, None]
    assert state.ffn_features == [None, None]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"attention_routes": [None]},
        {"ffn_routes": [None]},
        {"attention_features": [None]},
        {"ffn_features": [None]},
    ],
)
def test_request_state_rejects_invalid_layer_counts(kwargs):
    with pytest.raises(ValueError, match="exactly 2"):
        QaqRequestState("request-a", prompt_length=3, layer_count=2, **kwargs)


def test_request_state_rejects_invalid_feature_dimensions_and_routes():
    with pytest.raises(ValueError, match="feature dimension"):
        QaqRequestState(
            "request-a",
            prompt_length=3,
            layer_count=2,
            attention_features=[torch.zeros(4), torch.zeros(5)],
        )
    with pytest.raises(ValueError, match="None, 4, or 8"):
        QaqRequestState("request-a", prompt_length=3, layer_count=2, attention_routes=[6, None])


def test_request_state_ownership_and_complete_lifecycle():
    state = QaqRequestState("request-a", prompt_length=2, layer_count=1)
    owner_a = object()
    owner_b = object()
    state.bind_owner(owner_a)
    with pytest.raises(RuntimeError, match="another model"):
        state.bind_owner(owner_b)
    state.validate_for_model(layer_count=1, feature_dim=3)
    state.begin_prefill(prompt_length=2)
    feature = torch.tensor([1.0, 2.0, 3.0])
    state.store_feature("attention", 0, feature)
    state.store_route("attention", 0, 4)
    state.store_feature("ffn", 0, feature)
    state.store_route("ffn", 0, 8)
    state.assert_complete()
    assert state.route_for_decode("attention", 0) == 4
    assert state.route_for_decode("ffn", 0) == 8


def test_request_state_rejects_missing_mask_lifecycle_length_and_route_before_feature():
    state = QaqRequestState("request-a", prompt_length=2, layer_count=1)
    state.validate_for_model(layer_count=1, feature_dim=3)
    with pytest.raises(ValueError, match="prompt_length"):
        state.begin_prefill(prompt_length=1)
    state.begin_prefill(prompt_length=2)
    with pytest.raises(RuntimeError, match="before its feature"):
        state.store_route("attention", 0, 4)
