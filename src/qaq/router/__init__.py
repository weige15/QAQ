"""Prompt-feature and trainable soft-routing primitives."""

from .features import ManualPrecisionPolicy, masked_mean_pool, validate_prompt_mask
from .network import SoftPrecisionRouter
from .soft_linear import SoftPackedLinear, mix_packed_outputs

__all__ = [
    "ManualPrecisionPolicy",
    "SoftPackedLinear",
    "SoftPrecisionRouter",
    "masked_mean_pool",
    "mix_packed_outputs",
    "validate_prompt_mask",
]
