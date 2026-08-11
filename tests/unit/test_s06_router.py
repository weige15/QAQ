from __future__ import annotations

import pytest
import torch
from torch import nn

from qaq.router.network import (
    FeatureRMSNorm,
    SoftPrecisionRouter,
    probabilities_from_logits,
    trainable_parameter_audit,
)
from qaq.router.soft_linear import SoftPackedLinear


class _ToyPackedLinear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("weight4", torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
        self.register_buffer("weight8", torch.tensor([[2.0, 0.0], [0.0, 3.0]]))

    def forward(self, inputs: torch.Tensor, *, precision: int) -> torch.Tensor:
        weight = self.weight4 if precision == 4 else self.weight8
        return inputs @ weight.t()


def test_router_probability_sum_shape_and_finiteness():
    torch.manual_seed(1729)
    router = SoftPrecisionRouter(8)
    one = router(torch.ones(8))
    batch = router(torch.ones(3, 8))
    assert one.shape == (2,)
    assert batch.shape == (3, 2)
    assert torch.isfinite(batch).all()
    assert torch.all(batch >= 0)
    assert torch.allclose(batch.sum(dim=-1), torch.ones(3), atol=1e-6, rtol=0)


def test_feature_normalization_is_parameter_free_and_handles_zero():
    normalizer = FeatureRMSNorm()
    assert list(normalizer.parameters()) == []
    zero = normalizer(torch.zeros(4))
    ordinary = normalizer(torch.tensor([3.0, 4.0]))
    assert torch.isfinite(zero).all() and torch.equal(zero, torch.zeros(4))
    assert torch.allclose(ordinary.square().mean(), torch.ones(()), atol=1e-6)


@pytest.mark.parametrize("temperature", [0.0, -1.0, float("inf"), float("nan")])
def test_invalid_temperature_is_rejected(temperature):
    with pytest.raises(ValueError, match="temperature"):
        SoftPrecisionRouter(4, temperature=temperature)


def test_temperature_sharpens_and_flattens_fixed_logits():
    logits = torch.tensor([2.0, 0.0])
    cold = probabilities_from_logits(logits, temperature=0.5)
    hot = probabilities_from_logits(logits, temperature=2.0)
    assert cold[0] > hot[0] > 0.5
    assert cold[1] < hot[1]


def test_soft_packed_linear_endpoints_and_gradient():
    packed = _ToyPackedLinear()
    soft = SoftPackedLinear(packed)
    inputs = torch.tensor([[1.0, 2.0]])
    hard4 = packed(inputs, precision=4)
    hard8 = packed(inputs, precision=8)
    assert torch.equal(soft(inputs, torch.tensor([1.0, 0.0])), hard4)
    assert torch.equal(soft(inputs, torch.tensor([0.0, 1.0])), hard8)

    probabilities = torch.tensor([0.25, 0.75], requires_grad=True)
    loss = soft(inputs, probabilities).sum()
    loss.backward()
    assert probabilities.grad is not None
    assert torch.isfinite(probabilities.grad).all()
    assert torch.count_nonzero(probabilities.grad).item() > 0


def test_trainable_parameter_audit_rejects_non_router_parameters():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.routers = nn.ModuleDict({"attention_0": SoftPrecisionRouter(4)})
            self.frozen = nn.Linear(2, 2)
            for parameter in self.frozen.parameters():
                parameter.requires_grad_(False)

    model = Model()
    audit = trainable_parameter_audit(model)
    assert audit["trainable_parameter_count"] > 0
    assert all(name.startswith("routers.") for name in audit["trainable_names"])
    model.frozen.weight.requires_grad_(True)
    with pytest.raises(AssertionError, match="non-router"):
        trainable_parameter_audit(model)
