"""Trainable S06 routers over the frozen S05 prompt features."""

from __future__ import annotations

import torch
from torch import nn

CANDIDATE_BITS = (4, 8)
S10_CANDIDATE_BITS = (4, 6, 8)
_ALLOWED_CANDIDATE_BITS = (CANDIDATE_BITS, S10_CANDIDATE_BITS)
ROUTER_HIDDEN_WIDTH = 128
ROUTER_TEMPERATURE = 1.0
NORMALIZATION_EPSILON = 1e-6


def validate_candidate_bits(candidate_bits: object) -> tuple[int, ...]:
    """Validate the only learned-router orderings supported by QAQ."""

    if not isinstance(candidate_bits, tuple):
        raise TypeError("candidate_bits must be a tuple")
    if candidate_bits not in _ALLOWED_CANDIDATE_BITS:
        raise ValueError(
            "candidate_bits must be exactly (4, 8) or (4, 6, 8); "
            f"got {candidate_bits!r}"
        )
    return candidate_bits


def validate_probabilities(
    probabilities: torch.Tensor,
    candidate_bits: tuple[int, ...] = CANDIDATE_BITS,
    *,
    context: str = "routing probabilities",
    require_vector: bool = False,
) -> torch.Tensor:
    """Validate a probability vector against its explicit candidate order."""

    candidate_bits = validate_candidate_bits(candidate_bits)
    if not isinstance(probabilities, torch.Tensor) or probabilities.ndim < 1:
        raise ValueError(
            f"{context} must have shape [..., {len(candidate_bits)}] "
            f"(final dimension for candidate_bits={candidate_bits})"
        )
    if probabilities.shape[-1] != len(candidate_bits):
        raise ValueError(
            f"{context} must have shape [..., {len(candidate_bits)}] "
            f"(final dimension for candidate_bits={candidate_bits})"
        )
    if require_vector and probabilities.ndim != 1:
        raise ValueError(
            f"{context} must have one-dimensional shape [{len(candidate_bits)}] "
            f"for candidate_bits={candidate_bits}"
        )
    if not torch.isfinite(probabilities).all():
        raise ValueError(f"{context} must be finite")
    if not torch.all(probabilities >= 0):
        raise ValueError(f"{context} must be non-negative")
    if not torch.allclose(
        probabilities.sum(dim=-1), torch.ones_like(probabilities[..., 0]), atol=1e-5, rtol=0
    ):
        raise ValueError(f"{context} must sum to one")
    return probabilities


class FeatureRMSNorm(nn.Module):
    """Deterministic, parameter-free RMS normalization for one feature vector."""

    def __init__(self, *, eps: float = NORMALIZATION_EPSILON) -> None:
        super().__init__()
        if not 0 < eps < float("inf"):
            raise ValueError(f"normalization epsilon must be positive and finite; got {eps}")
        self.eps = float(eps)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        if feature.ndim < 1:
            raise ValueError("router features must have at least one dimension")
        if not torch.isfinite(feature).all():
            raise ValueError("router features must be finite")
        values = feature.float()
        rms = values.square().mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        normalized = values / rms
        if not torch.isfinite(normalized).all():
            raise ValueError("feature normalization produced non-finite values")
        return normalized


def probabilities_from_logits(
    logits: torch.Tensor,
    *,
    temperature: float,
    candidate_bits: tuple[int, ...] = CANDIDATE_BITS,
) -> torch.Tensor:
    """Convert candidate-aware logits into probabilities in canonical bit order."""

    candidate_bits = validate_candidate_bits(candidate_bits)
    if logits.shape[-1] != len(candidate_bits):
        raise ValueError(
            f"router logits must have {len(candidate_bits)} outputs for "
            f"candidate_bits={candidate_bits}; got {logits.shape[-1]}"
        )
    if not torch.isfinite(logits).all():
        raise ValueError("router logits must be finite")
    if not isinstance(temperature, (float, int)) or isinstance(temperature, bool):
        raise TypeError("routing temperature must be a finite positive number")
    if not torch.isfinite(torch.tensor(float(temperature))) or temperature <= 0:
        raise ValueError(f"routing temperature must be greater than zero; got {temperature}")
    probabilities = torch.softmax(logits / float(temperature), dim=-1)
    if not torch.isfinite(probabilities).all():
        raise ValueError("router probabilities must be finite")
    return probabilities


class SoftPrecisionRouter(nn.Module):
    """One lightweight MLP router with canonical outputs ``[p4, p8]``."""

    candidate_bits = CANDIDATE_BITS
    hidden_width = ROUTER_HIDDEN_WIDTH
    activation_name = "GELU"

    def __init__(
        self,
        feature_dim: int,
        *,
        hidden_width: int = ROUTER_HIDDEN_WIDTH,
        temperature: float = ROUTER_TEMPERATURE,
        candidate_bits: tuple[int, ...] = CANDIDATE_BITS,
    ) -> None:
        super().__init__()
        if isinstance(feature_dim, bool) or not isinstance(feature_dim, int) or feature_dim <= 0:
            raise ValueError(f"feature_dim must be a positive integer; got {feature_dim}")
        if isinstance(hidden_width, bool) or not isinstance(hidden_width, int) or hidden_width <= 0:
            raise ValueError(f"router hidden width must be a positive integer; got {hidden_width}")
        self.feature_dim = feature_dim
        self.hidden_width = hidden_width
        self.candidate_bits = validate_candidate_bits(candidate_bits)
        self.temperature = self._validate_temperature(temperature)
        self.normalization = FeatureRMSNorm()
        self.input_projection = nn.Linear(feature_dim, hidden_width)
        self.activation = nn.GELU()
        self.output_projection = nn.Linear(hidden_width, len(self.candidate_bits))

    @staticmethod
    def _validate_temperature(value: float) -> float:
        if isinstance(value, bool) or not isinstance(value, (float, int)):
            raise TypeError("routing temperature must be a finite positive number")
        temperature = float(value)
        if not torch.isfinite(torch.tensor(temperature)) or temperature <= 0:
            raise ValueError(f"routing temperature must be greater than zero; got {value}")
        return temperature

    def logits(self, feature: torch.Tensor) -> torch.Tensor:
        if feature.shape[-1] != self.feature_dim:
            raise ValueError(
                f"router feature dimension must be {self.feature_dim}; got {feature.shape[-1]}"
            )
        values = feature.detach()
        values = self.normalization(values).to(dtype=self.input_projection.weight.dtype)
        return self.output_projection(self.activation(self.input_projection(values)))

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return probabilities_from_logits(
            self.logits(feature), temperature=self.temperature, candidate_bits=self.candidate_bits
        )


def router_parameter_count(router: nn.Module) -> int:
    return sum(parameter.numel() for parameter in router.parameters())


def trainable_parameter_audit(
    model: nn.Module, *, router_prefix: str = "routers."
) -> dict[str, object]:
    """Verify that only the named router parameter namespace is trainable."""

    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    invalid = [name for name, _ in trainable if not name.startswith(router_prefix)]
    if invalid:
        raise AssertionError(f"non-router parameters are trainable: {invalid}")
    return {
        "trainable_names": [name for name, _ in trainable],
        "trainable_parameter_count": sum(parameter.numel() for _, parameter in trainable),
        "frozen_parameter_count": sum(
            parameter.numel() for _, parameter in model.named_parameters() if not parameter.requires_grad
        ),
    }


__all__ = [
    "CANDIDATE_BITS",
    "S10_CANDIDATE_BITS",
    "FeatureRMSNorm",
    "SoftPrecisionRouter",
    "probabilities_from_logits",
    "router_parameter_count",
    "trainable_parameter_audit",
    "validate_candidate_bits",
    "validate_probabilities",
]
