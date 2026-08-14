"""Differentiable mixtures over the pinned packed 4-bit and 8-bit calls."""

from __future__ import annotations

import torch
from torch import nn

from .network import CANDIDATE_BITS, validate_candidate_bits, validate_probabilities


def _validate_probabilities(
    probabilities: torch.Tensor, candidate_bits: tuple[int, ...]
) -> torch.Tensor:
    return validate_probabilities(probabilities, candidate_bits, context="soft routing probabilities")


def mix_packed_outputs(
    packed: nn.Module,
    inputs: torch.Tensor,
    probabilities: torch.Tensor,
    *,
    candidate_bits: tuple[int, ...] = CANDIDATE_BITS,
) -> torch.Tensor:
    """Execute each configured real packed path and mix without detaching probabilities."""

    candidate_bits = validate_candidate_bits(candidate_bits)
    probabilities = _validate_probabilities(probabilities, candidate_bits)
    outputs = [packed(inputs, precision=precision) for precision in candidate_bits]
    shapes = {output.shape for output in outputs}
    if len(shapes) != 1:
        raise RuntimeError(f"packed precision paths returned different shapes: {sorted(shapes)}")
    if any(not torch.isfinite(output).all() for output in outputs):
        raise FloatingPointError("packed precision path produced NaN or Inf")
    probabilities = probabilities.to(dtype=outputs[0].dtype)
    output = torch.zeros_like(outputs[0])
    for index, packed_output in enumerate(outputs):
        weight = probabilities[..., index]
        while weight.ndim < packed_output.ndim:
            weight = weight.unsqueeze(-1)
        output = output + weight * packed_output
    if not torch.isfinite(output).all():
        raise FloatingPointError("soft packed mixture produced NaN or Inf")
    return output


class SoftPackedLinear(nn.Module):
    """A frozen packed linear exposing a differentiable 4/8 mixture."""

    def __init__(
        self, packed: nn.Module, *, candidate_bits: tuple[int, ...] = CANDIDATE_BITS
    ) -> None:
        super().__init__()
        self.candidate_bits = validate_candidate_bits(candidate_bits)
        self.packed = packed
        for parameter in self.packed.parameters():
            parameter.requires_grad_(False)

    def forward(self, inputs: torch.Tensor, probabilities: torch.Tensor) -> torch.Tensor:
        return mix_packed_outputs(
            self.packed, inputs, probabilities, candidate_bits=self.candidate_bits
        )


__all__ = ["SoftPackedLinear", "mix_packed_outputs"]
