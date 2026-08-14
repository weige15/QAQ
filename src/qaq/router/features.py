"""Prompt-only feature construction and deterministic S05 route policies."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from .network import CANDIDATE_BITS, validate_candidate_bits

SUPPORTED_BITS = CANDIDATE_BITS


def validate_prompt_mask(
    attention_mask: torch.Tensor | None, *, sequence_length: int
) -> torch.Tensor:
    """Validate and normalize the batch-size-one prompt mask to boolean values."""

    if attention_mask is None:
        raise ValueError("S05 prefill requires an explicit attention_mask")
    if attention_mask.ndim != 2 or attention_mask.shape[0] != 1:
        raise ValueError(
            "S05 supports only a batch-size-one attention_mask with shape [1, sequence_length]"
        )
    if attention_mask.shape[1] != sequence_length:
        raise ValueError(
            "attention_mask sequence length must match prompt hidden states: "
            f"{attention_mask.shape[1]} != {sequence_length}"
        )
    if attention_mask.dtype == torch.bool:
        mask = attention_mask
    elif attention_mask.is_floating_point():
        if not torch.isfinite(attention_mask).all():
            raise ValueError("attention_mask must contain only finite 0/1 values")
        if not torch.all((attention_mask == 0) | (attention_mask == 1)):
            raise ValueError("attention_mask must contain only 0/1 values")
        mask = attention_mask != 0
    elif attention_mask.dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        if not torch.all((attention_mask == 0) | (attention_mask == 1)):
            raise ValueError("attention_mask must contain only 0/1 values")
        mask = attention_mask != 0
    else:
        raise TypeError(f"unsupported attention_mask dtype: {attention_mask.dtype}")
    if not bool(mask.any()):
        raise ValueError("S05 prompt must contain at least one non-padding token")
    return mask


def masked_mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool valid prompt positions and return one detached ``[hidden]`` feature.

    Accumulation is performed in float32 for stable, explicit
    ``sum(valid hidden states) / valid token count`` semantics.  The returned
    feature is detached because S05 has no trainable router.
    """

    if hidden_states.ndim != 3 or hidden_states.shape[0] != 1:
        raise ValueError(
            "S05 feature pooling requires batch-size-one hidden states with shape [1, sequence, hidden]"
        )
    mask = validate_prompt_mask(attention_mask, sequence_length=hidden_states.shape[1])
    valid_count = int(mask.sum().item())
    if valid_count <= 0:  # defensive; validate_prompt_mask handles this case
        raise ValueError("S05 prompt must contain at least one non-padding token")
    values = hidden_states.float()
    weights = mask.to(device=values.device, dtype=values.dtype).unsqueeze(-1)
    feature = (values * weights).sum(dim=1)[0] / valid_count
    return feature.detach().clone()


@dataclass(frozen=True, slots=True)
class ManualPrecisionPolicy:
    """Adapter from an already-supplied S04 ``PrecisionPlan`` to S05 callbacks."""

    precision_plan: Any

    def __post_init__(self) -> None:
        if not hasattr(self.precision_plan, "attention_bits") or not hasattr(
            self.precision_plan, "ffn_bits"
        ):
            raise TypeError("ManualPrecisionPolicy requires an S04 PrecisionPlan")
        self.precision_plan.validate()

    def __call__(self, layer_index: int, unit_type: str, feature: torch.Tensor) -> int:
        del feature  # The deterministic S04 policy intentionally is not adaptive.
        if unit_type == "attention":
            precision = self.precision_plan.attention_bits[layer_index]
        elif unit_type == "ffn":
            precision = self.precision_plan.ffn_bits[layer_index]
        else:
            raise ValueError(f"unsupported S05 routing unit: {unit_type}")
        if isinstance(precision, bool) or precision not in SUPPORTED_BITS:
            raise ValueError(f"manual policy returned unsupported precision: {precision!r}")
        return precision


RoutePolicy = Callable[[int, str, torch.Tensor], int]


def coerce_manual_policy(policy: Any) -> RoutePolicy:
    """Return a validated deterministic callback without creating router state."""

    if isinstance(policy, ManualPrecisionPolicy):
        return policy
    if hasattr(policy, "attention_bits") and hasattr(policy, "ffn_bits"):
        return ManualPrecisionPolicy(policy)
    if not callable(policy):
        raise TypeError("S05 prefill requires an S04 PrecisionPlan or deterministic callback")
    return policy


def validate_policy_result(
    precision: object, *, candidate_bits: tuple[int, ...] = CANDIDATE_BITS
) -> int:
    candidate_bits = validate_candidate_bits(candidate_bits)
    if isinstance(precision, bool) or not isinstance(precision, int):
        raise TypeError(f"routing policy must return an integer from {candidate_bits}")
    if precision not in candidate_bits:
        allowed = "4 or 8" if candidate_bits == (4, 8) else f"one of {candidate_bits}"
        raise ValueError(f"routing policy must return {allowed}; got {precision}")
    return precision
