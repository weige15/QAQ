"""Deterministic prompt-feature routing primitives for S05."""

from .features import ManualPrecisionPolicy, masked_mean_pool, validate_prompt_mask

__all__ = ["ManualPrecisionPolicy", "masked_mean_pool", "validate_prompt_mask"]
