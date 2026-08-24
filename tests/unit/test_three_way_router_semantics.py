from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from qaq.model.manual import PrecisionPlan, PrecisionTrace, _RoutedPackedLinear
from qaq.model.request_state import QaqRequestState
from qaq.router.distillation import (
    RouteLogRecord,
    RouterCheckpointMetadata,
    hard_route,
    load_router_checkpoint,
    route_statistics,
    save_router_checkpoint,
)
from qaq.router.network import (
    CANDIDATE_BITS,
    THREE_WAY_CANDIDATE_BITS,
    SoftPrecisionRouter,
    validate_candidate_bits,
)
from qaq.router.soft_linear import SoftPackedLinear


class _ThreeWayPackedLinear(nn.Module):
    def forward(self, inputs: torch.Tensor, *, precision: int) -> torch.Tensor:
        return inputs * float(precision)


def test_candidate_validation_and_router_shapes_counts():
    assert validate_candidate_bits((4, 8)) == (4, 8)
    assert validate_candidate_bits((4, 6, 8)) == (4, 6, 8)
    for invalid in ((), (4,), (6, 8), (4, 8, 6), (8, 6, 4), (4, 4, 8), (4, 5, 8), (4, 6, 7, 8)):
        with pytest.raises(ValueError):
            validate_candidate_bits(invalid)

    historical = SoftPrecisionRouter(8, hidden_width=4)
    three_way = SoftPrecisionRouter(8, hidden_width=4, candidate_bits=THREE_WAY_CANDIDATE_BITS)
    assert historical.candidate_bits == CANDIDATE_BITS
    assert historical(torch.ones(8)).shape == (2,)
    assert three_way(torch.ones(8)).shape == (3,)
    assert sum(parameter.numel() for parameter in historical.parameters()) == 8 * 4 + 4 + 4 * 2 + 2
    assert sum(parameter.numel() for parameter in three_way.parameters()) == 8 * 4 + 4 + 4 * 3 + 3


def test_three_way_soft_mixture_endpoints_and_gradients():
    packed = _ThreeWayPackedLinear()
    soft = SoftPackedLinear(packed, candidate_bits=THREE_WAY_CANDIDATE_BITS)
    inputs = torch.tensor([[1.0, 2.0]], requires_grad=False)
    outputs = [packed(inputs, precision=bits) for bits in THREE_WAY_CANDIDATE_BITS]
    for index, expected in enumerate(outputs):
        probabilities = torch.nn.functional.one_hot(torch.tensor(index), 3).float().requires_grad_()
        assert torch.equal(soft(inputs, probabilities), expected)
    probabilities = torch.tensor([0.2, 0.3, 0.5], requires_grad=True)
    output = soft(inputs, probabilities)
    assert torch.allclose(output, 4 * 0.2 * inputs + 6 * 0.3 * inputs + 8 * 0.5 * inputs)
    output.square().mean().backward()
    assert probabilities.grad is not None
    assert torch.isfinite(probabilities.grad).all()
    assert torch.count_nonzero(probabilities.grad).item() > 0


def test_request_state_explicit_three_way_routes_and_probability_order():
    state = QaqRequestState(
        "s10b",
        prompt_length=2,
        layer_count=1,
        candidate_bits=THREE_WAY_CANDIDATE_BITS,
    )
    state.validate_for_model(layer_count=1, feature_dim=2)
    state.begin_prefill(prompt_length=2)
    state.store_feature("attention", 0, torch.ones(2))
    state.store_probability("attention", 0, torch.tensor([0.1, 0.2, 0.7]))
    state.store_route("attention", 0, 6)
    assert state.attention_probabilities[0].tolist() == pytest.approx([0.1, 0.2, 0.7])
    assert state.route_for_decode("attention", 0) == 6
    with pytest.raises(ValueError, match="final dimension"):
        state.store_probability("ffn", 0, torch.tensor([0.5, 0.5]))


def test_resident_hard_route_executes_explicit_six_bit_candidate():
    state = QaqRequestState(
        "resident-s10b", prompt_length=1, layer_count=1, candidate_bits=THREE_WAY_CANDIDATE_BITS
    )
    state.validate_for_model(layer_count=1, feature_dim=2)
    state.begin_prefill(prompt_length=1)
    state.store_feature("attention", 0, torch.ones(2))
    state.store_route("attention", 0, 6)
    routed = _RoutedPackedLinear(
        _ThreeWayPackedLinear(), layer_index=0, unit_type="attention", module_path="fixture"
    )
    output = routed(
        torch.ones(1, 2),
        precision=state.route_for_decode("attention", 0),
        request_state=state,
        trace=PrecisionTrace(),
    )
    assert torch.equal(output, torch.full((1, 2), 6.0))


def test_hard_route_mapping_and_historical_precision_plan():
    assert hard_route(torch.tensor([0.5, 0.5, 0.0]), candidate_bits=THREE_WAY_CANDIDATE_BITS) == 4
    assert hard_route(torch.tensor([0.0, 0.5, 0.5]), candidate_bits=THREE_WAY_CANDIDATE_BITS) == 6
    assert (
        hard_route(torch.tensor([1 / 3, 1 / 3, 1 / 3]), candidate_bits=THREE_WAY_CANDIDATE_BITS)
        == 4
    )
    assert hard_route(torch.tensor([0.5, 0.5])) == 4
    with pytest.raises(ValueError, match="4 or 8"):
        PrecisionPlan.uniform(6)


def test_three_way_route_logs_and_statistics_are_explicit():
    records = [
        RouteLogRecord.from_probabilities(
            "s10b", 0, unit, torch.tensor([0.1, 0.2, 0.7]), candidate_bits=THREE_WAY_CANDIDATE_BITS
        )
        for unit in ("attention", "ffn")
    ]
    assert all(record.candidate_bits == THREE_WAY_CANDIDATE_BITS for record in records)
    assert records[0].p6 == pytest.approx(0.2)
    assert records[0].soft_average_width == pytest.approx(7.2)
    stats = route_statistics(records)
    assert stats["hard_fraction_6"] == 0.0
    assert stats["hard_fraction_8"] == 1.0
    assert stats["attention_vs_ffn_distribution"]["attention"] == {"4": 0.0, "6": 0.0, "8": 1.0}
    historical = RouteLogRecord.from_probabilities("old", 0, "attention", torch.tensor([0.5, 0.5]))
    assert historical.candidate_bits == CANDIDATE_BITS
    assert historical.p6 is None


def _metadata(candidate_bits: tuple[int, ...]) -> RouterCheckpointMetadata:
    return RouterCheckpointMetadata(
        model_repository="fixture/qaq",
        model_revision="model-r1",
        quantized_checkpoint_id="packed-r1",
        quantized_checkpoint_hash="sha256:packed",
        any_precision_revision="any-r1",
        router_architecture={"feature_dim": 4, "hidden_width": 4},
        candidate_ordering=candidate_bits,
    )


def test_three_way_checkpoint_roundtrip_and_mismatch_rejection(tmp_path):
    router = SoftPrecisionRouter(4, hidden_width=4, candidate_bits=THREE_WAY_CANDIDATE_BITS)
    restored = copy.deepcopy(router)
    metadata = _metadata(THREE_WAY_CANDIDATE_BITS)
    assert metadata.to_dict()["candidate_ordering"] == [4, 6, 8]
    path = tmp_path / "three-way.pt"
    save_router_checkpoint(path, router, metadata)
    load_router_checkpoint(path, restored, metadata)
    feature = torch.tensor([1.0, -2.0, 0.5, 3.0])
    assert torch.equal(router(feature), restored(feature))
    assert int(hard_route(router(feature), candidate_bits=THREE_WAY_CANDIDATE_BITS)) == int(
        hard_route(restored(feature), candidate_bits=THREE_WAY_CANDIDATE_BITS)
    )

    old_router = SoftPrecisionRouter(4, hidden_width=4)
    old_metadata = _metadata(CANDIDATE_BITS)
    old_path = tmp_path / "old.pt"
    save_router_checkpoint(old_path, old_router, old_metadata)
    with pytest.raises(ValueError, match="incompatible"):
        load_router_checkpoint(old_path, restored, metadata)
    with pytest.raises(ValueError, match="incompatible"):
        load_router_checkpoint(path, old_router, old_metadata)
