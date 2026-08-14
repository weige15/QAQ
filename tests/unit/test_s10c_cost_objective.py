from __future__ import annotations

import math

import pytest
import torch

from qaq.model.request_state import QaqRequestState
from qaq.router.distillation import (
    cost_aware_distillation_loss,
    expected_bit_cost,
    request_state_expected_bit_cost,
)
from qaq.router.network import CANDIDATE_BITS, S10_CANDIDATE_BITS


def test_endpoint_and_mixed_costs_use_explicit_candidate_order():
    three_way = torch.eye(3)
    assert torch.equal(
        expected_bit_cost(three_way, S10_CANDIDATE_BITS), torch.tensor([0.0, 0.5, 1.0])
    )
    historical = torch.eye(2)
    assert torch.equal(expected_bit_cost(historical, CANDIDATE_BITS), torch.tensor([0.0, 1.0]))
    mixed = torch.tensor([0.2, 0.3, 0.5])
    assert expected_bit_cost(mixed, S10_CANDIDATE_BITS).item() == pytest.approx(0.65)
    assert expected_bit_cost(torch.full((3,), 1 / 3), S10_CANDIDATE_BITS).item() == pytest.approx(0.5)


def test_cost_validation_is_explicit_and_bounded():
    with pytest.raises(ValueError, match="exactly"):
        expected_bit_cost(torch.ones(3) / 3, (4, 8, 6))
    with pytest.raises(ValueError, match="exactly"):
        expected_bit_cost(torch.ones(3) / 3, (4, 5, 8))
    for probabilities in (
        torch.tensor([0.5, 0.4, 0.1, 0.0]),
        torch.tensor([0.5, -0.1, 0.6]),
        torch.tensor([0.5, float("nan"), 0.5]),
        torch.tensor([0.5, float("inf"), 0.5]),
        torch.tensor([0.5, 0.5, 0.1]),
    ):
        with pytest.raises(ValueError):
            expected_bit_cost(probabilities, S10_CANDIDATE_BITS)
    values = expected_bit_cost(torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]))
    assert torch.isfinite(values).all()
    assert bool(((values >= 0) & (values <= 1)).all())


def test_expected_width_relationship_and_fixed_kd_positive_lambda_ordering():
    probabilities = torch.tensor([0.2, 0.3, 0.5])
    bit_cost = expected_bit_cost(probabilities, S10_CANDIDATE_BITS)
    expected_width = 4 + 4 * bit_cost
    assert expected_width.item() == pytest.approx(6.6)
    kd = torch.tensor(2.0)
    losses = [
        cost_aware_distillation_loss(kd, expected_bit_cost(torch.eye(3)[index]), 0.2).item()
        for index in range(3)
    ]
    assert losses[0] < losses[1] < losses[2]


def test_request_state_diagnostics_expose_three_way_expected_width():
    state = QaqRequestState(
        "diagnostic-request", prompt_length=1, layer_count=1, candidate_bits=S10_CANDIDATE_BITS
    )
    for unit_type in ("attention", "ffn"):
        state.store_feature(unit_type, 0, torch.ones(2))
        state.store_probability(unit_type, 0, torch.tensor([0.2, 0.3, 0.5]))
    diagnostics = request_state_expected_bit_cost(state, return_diagnostics=True)
    assert diagnostics.expected_bit_cost.item() == pytest.approx(0.65)
    assert diagnostics.expected_bit_width is not None
    assert diagnostics.expected_bit_width.item() == pytest.approx(
        4 + 4 * diagnostics.expected_bit_cost.item()
    )


def test_request_state_rejects_leading_dimensions_and_empty_slots():
    state = QaqRequestState(
        "shape-request", prompt_length=1, layer_count=1, candidate_bits=S10_CANDIDATE_BITS
    )
    state.store_feature("attention", 0, torch.ones(2))
    for probabilities in (torch.full((2, 3), 1 / 3), torch.empty((0, 3))):
        with pytest.raises(ValueError, match="shape"):
            state.store_probability("attention", 0, probabilities)

    state.store_probability("attention", 0, torch.tensor([1.0, 0.0, 0.0]))
    state.store_feature("ffn", 0, torch.ones(2))
    state.store_probability("ffn", 0, torch.tensor([1.0, 0.0, 0.0]))
    state.attention_probabilities[0] = torch.empty((0, 3))
    with pytest.raises(ValueError, match="shape"):
        request_state_expected_bit_cost(state)


def test_cost_weight_validation_and_lambda_zero_scalar_gradient_compatibility():
    logits = torch.tensor([0.2, -0.3, 0.7], requires_grad=True)
    probabilities = torch.softmax(logits, dim=-1)
    kd = (logits.square()).sum()
    cost = expected_bit_cost(probabilities)
    zero_loss = cost_aware_distillation_loss(kd, cost, 0.0)
    assert zero_loss is kd
    zero_gradient = torch.autograd.grad(zero_loss, logits, retain_graph=True)[0]
    kd_gradient = torch.autograd.grad(kd, logits, retain_graph=True)[0]
    assert torch.equal(zero_gradient, kd_gradient)
    for invalid in (-1.0, float("nan"), float("inf"), True, "0.1"):
        with pytest.raises((TypeError, ValueError)):
            cost_aware_distillation_loss(kd, cost, invalid)


def test_eight_dominant_softmax_gradient_pushes_toward_lower_cost():
    logits = torch.tensor([0.0, 0.0, 5.0], requires_grad=True)
    cost = expected_bit_cost(torch.softmax(logits, dim=-1))
    gradient = torch.autograd.grad(cost, logits)[0]
    assert gradient[2] > 0
    assert gradient[0] < 0 and gradient[1] < 0
    assert torch.isfinite(gradient).all()


def test_request_state_aggregates_each_attention_and_ffn_unit_once_with_gradients():
    state = QaqRequestState(
        "cost-request", prompt_length=2, layer_count=36, candidate_bits=S10_CANDIDATE_BITS
    )
    logits = [
        torch.tensor(
            [float((index % 3) == 0), float((index % 3) == 1), float((index % 3) == 2)],
            requires_grad=True,
        )
        for index in range(72)
    ]
    for index, (unit_type, layer) in enumerate(
        [(unit_type, layer) for unit_type in ("attention", "ffn") for layer in range(36)]
    ):
        state.store_feature(unit_type, layer, torch.ones(2))
        state.store_probability(unit_type, layer, torch.softmax(logits[index], dim=-1))
    aggregated = request_state_expected_bit_cost(state)
    manual = torch.stack([expected_bit_cost(torch.softmax(value, dim=-1)) for value in logits]).mean()
    assert torch.allclose(aggregated, manual)
    assert aggregated.item() == pytest.approx(manual.item())
    combined = cost_aware_distillation_loss(torch.tensor(1.0, requires_grad=True), aggregated, 0.3)
    combined.backward()
    assert torch.isfinite(combined)
    assert all(value.grad is not None and torch.isfinite(value.grad).all() for value in logits)
    assert all(value.grad.abs().sum() > 0 for value in logits)


def test_request_state_cost_preserves_frozen_state():
    frozen = torch.nn.Parameter(torch.tensor([3.0, 4.0]), requires_grad=False)
    before = frozen.detach().clone()
    state = QaqRequestState(
        "frozen-request", prompt_length=1, layer_count=1, candidate_bits=S10_CANDIDATE_BITS
    )
    state.store_feature("attention", 0, frozen)
    state.store_feature("ffn", 0, frozen)
    probability = torch.tensor([0.2, 0.3, 0.5], requires_grad=True)
    state.store_probability("attention", 0, probability)
    state.store_probability("ffn", 0, probability)
    loss = request_state_expected_bit_cost(state)
    loss.backward()
    assert torch.equal(frozen, before)
    assert frozen.grad is None
    assert probability.grad is not None and torch.isfinite(probability.grad).all()
    assert math.isfinite(loss.item())
