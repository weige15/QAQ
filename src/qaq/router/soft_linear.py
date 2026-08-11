"""Differentiable mixtures over the pinned packed 4-bit and 8-bit calls."""

from __future__ import annotations

import torch
from torch import nn

from .network import CANDIDATE_BITS


def _validate_probabilities(probabilities: torch.Tensor) -> torch.Tensor:
    if not isinstance(probabilities, torch.Tensor) or probabilities.shape[-1] != 2:
        raise ValueError("soft routing probabilities must have a final dimension of size 2")
    if not torch.isfinite(probabilities).all():
        raise ValueError("soft routing probabilities must be finite")
    if not torch.all(probabilities >= 0):
        raise ValueError("soft routing probabilities must be non-negative")
    if not torch.allclose(
        probabilities.sum(dim=-1), torch.ones_like(probabilities[..., 0]), atol=1e-5, rtol=0
    ):
        raise ValueError("soft routing probabilities must sum to one")
    return probabilities


def mix_packed_outputs(
    packed: nn.Module, inputs: torch.Tensor, probabilities: torch.Tensor
) -> torch.Tensor:
    """Execute both real packed paths and mix their outputs without detaching ``p``."""

    probabilities = _validate_probabilities(probabilities)
    y4 = packed(inputs, precision=CANDIDATE_BITS[0])
    y8 = packed(inputs, precision=CANDIDATE_BITS[1])
    if y4.shape != y8.shape:
        raise RuntimeError(f"packed precision paths returned different shapes: {y4.shape} != {y8.shape}")
    if not torch.isfinite(y4).all() or not torch.isfinite(y8).all():
        raise FloatingPointError("packed precision path produced NaN or Inf")
    probabilities = probabilities.to(dtype=y4.dtype)
    if probabilities.ndim == 1:
        p4, p8 = probabilities[0], probabilities[1]
    else:
        p4 = probabilities[..., 0]
        p8 = probabilities[..., 1]
        while p4.ndim < y4.ndim:
            p4 = p4.unsqueeze(-1)
            p8 = p8.unsqueeze(-1)
    output = p4 * y4 + p8 * y8
    if not torch.isfinite(output).all():
        raise FloatingPointError("soft packed mixture produced NaN or Inf")
    return output


class SoftPackedLinear(nn.Module):
    """A frozen packed linear exposing a differentiable 4/8 mixture."""

    def __init__(self, packed: nn.Module) -> None:
        super().__init__()
        self.packed = packed
        for parameter in self.packed.parameters():
            parameter.requires_grad_(False)

    def forward(self, inputs: torch.Tensor, probabilities: torch.Tensor) -> torch.Tensor:
        return mix_packed_outputs(self.packed, inputs, probabilities)


__all__ = ["SoftPackedLinear", "mix_packed_outputs"]
