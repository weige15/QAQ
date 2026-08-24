from __future__ import annotations

import pytest
import torch

from qaq.model.manual import (
    PrecisionTrace,
    _predict_lookahead_attention_route,
    _select_request_route,
)
from qaq.model.request_state import (
    LOOKAHEAD_ATTENTION_ONE_UNIT,
    POST_ATTENTION_PRE_FFN,
    SAME_UNIT,
    QaqRequestState,
    RoutingProvenance,
)
from qaq.router.distillation import hard_route
from qaq.router.network import THREE_WAY_CANDIDATE_BITS


def _provenance(source_layer: int) -> RoutingProvenance:
    return RoutingProvenance(
        source_layer=source_layer,
        target_layer=source_layer + 1,
        target_unit_type="attention",
        source_point=POST_ATTENTION_PRE_FFN,
        routing_timing=LOOKAHEAD_ATTENTION_ONE_UNIT,
    )


def test_routing_timing_default_and_explicit_mode_validation():
    assert QaqRequestState("default", 1).routing_timing == SAME_UNIT
    assert QaqRequestState("explicit", 1, routing_timing=SAME_UNIT).routing_timing == SAME_UNIT
    assert (
        QaqRequestState("lookahead", 1, routing_timing=LOOKAHEAD_ATTENTION_ONE_UNIT).routing_timing
        == LOOKAHEAD_ATTENTION_ONE_UNIT
    )
    with pytest.raises(ValueError, match="routing_timing"):
        QaqRequestState("invalid", 1, routing_timing="two_units")
    with pytest.raises(TypeError, match="routing_timing"):
        QaqRequestState("invalid", 1, routing_timing=None)


def test_lookahead_prediction_is_target_owned_and_uses_target_router_identity():
    assert int(hard_route(torch.tensor([0.5, 0.5]))) == 4
    state = QaqRequestState(
        "mapping",
        2,
        layer_count=2,
        candidate_bits=THREE_WAY_CANDIDATE_BITS,
        routing_timing=LOOKAHEAD_ATTENTION_ONE_UNIT,
    )
    state.validate_for_model(layer_count=2, feature_dim=2)
    calls = []

    def policy(layer, unit_type, feature):
        calls.append((layer, unit_type, feature.detach().clone()))
        return int(
            hard_route(torch.tensor([0.0, 0.5, 0.5]), candidate_bits=THREE_WAY_CANDIDATE_BITS)
        )

    precision = _predict_lookahead_attention_route(
        request_state=state,
        source_layer=0,
        incoming_hidden=torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),
        prompt_attention_mask=torch.ones(1, 2, dtype=torch.bool),
        routing_policy=policy,
        trace=PrecisionTrace(),
    )

    assert precision == 6  # first maximum in explicit (4, 6, 8) order
    assert [(layer, unit) for layer, unit, _ in calls] == [(1, "attention")]
    assert state.attention_routes == [None, 6]
    assert state.attention_features[0] is None
    assert state.attention_features[1].tolist() == [2.0, 3.0]
    assert state.attention_provenance[1] == _provenance(0)


def test_layer_zero_fallback_duplicate_missing_and_final_layer_guards():
    state = QaqRequestState("guards", 2, layer_count=2, routing_timing=LOOKAHEAD_ATTENTION_ONE_UNIT)
    state.validate_for_model(layer_count=2, feature_dim=2)
    calls = []

    def policy(layer, unit_type, feature):
        del feature
        calls.append((layer, unit_type))
        return 4

    _select_request_route(
        request_state=state,
        layer_index=0,
        unit_type="attention",
        incoming_hidden=torch.ones(1, 2, 2),
        prompt_attention_mask=torch.ones(1, 2, dtype=torch.bool),
        phase="prefill",
        routing_policy=policy,
        trace=PrecisionTrace(),
    )
    assert calls == [(0, "attention")]
    assert state.attention_provenance[0] is None

    state.store_feature("attention", 1, torch.ones(2), provenance=_provenance(0))
    with pytest.raises(RuntimeError, match="already stored"):
        state.store_feature("attention", 1, torch.ones(2), provenance=_provenance(0))
    with pytest.raises(RuntimeError, match="missing its early route"):
        state.consume_early_attention_route(1)
    state.store_route("attention", 1, 8)
    with pytest.raises(RuntimeError, match="already stored"):
        state.store_route("attention", 1, 4)
    assert state.consume_early_attention_route(1) == 8
    with pytest.raises(RuntimeError, match="already consumed"):
        state.consume_early_attention_route(1)

    with pytest.raises(ValueError, match="beyond the final layer"):
        _predict_lookahead_attention_route(
            request_state=state,
            source_layer=1,
            incoming_hidden=torch.ones(1, 2, 2),
            prompt_attention_mask=torch.ones(1, 2, dtype=torch.bool),
            routing_policy=policy,
            trace=PrecisionTrace(),
        )
    assert len(state.attention_routes) == 2


def test_lookahead_probability_consumption_is_once_and_cleanup_is_request_local():
    state_a = QaqRequestState(
        "same-id", 1, layer_count=2, routing_timing=LOOKAHEAD_ATTENTION_ONE_UNIT
    )
    state_b = QaqRequestState(
        "same-id", 1, layer_count=2, routing_timing=LOOKAHEAD_ATTENTION_ONE_UNIT
    )
    for state, probability in (
        (state_a, torch.tensor([0.25, 0.75], requires_grad=True)),
        (state_b, torch.tensor([0.75, 0.25], requires_grad=True)),
    ):
        state.validate_for_model(layer_count=2, feature_dim=2)
        state.store_feature("attention", 1, torch.ones(2), provenance=_provenance(0))
        state.store_probability("attention", 1, probability)

    observed = state_a.consume_early_attention_probability(1)
    assert torch.equal(observed, torch.tensor([0.25, 0.75]))
    assert observed.grad_fn is not None
    assert not state_b.early_attention_probability_consumed[1]
    with pytest.raises(RuntimeError, match="already consumed"):
        state_a.consume_early_attention_probability(1)

    state_a.end_request()
    assert state_a.ended
    assert all(value is None for value in state_a.attention_features)
    assert all(value is None for value in state_a.attention_probabilities)
    assert all(value is None for value in state_a.attention_provenance)
    assert state_b.attention_features[1] is not None
    assert state_b.attention_probabilities[1] is not None
