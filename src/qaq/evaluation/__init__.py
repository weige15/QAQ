"""Evaluation and frozen-baseline reporting helpers.

Exports stay lazy so standard-library-only planning modules can be imported
without importing Torch through the historical quality helpers.
"""

from __future__ import annotations

from typing import Any

__all__ = ["aggregate", "build_perplexity_windows", "evaluate_perplexity", "execute_mode", "plan"]


def __getattr__(name: str) -> Any:
    if name in {"build_perplexity_windows", "evaluate_perplexity"}:
        from . import quality

        return getattr(quality, name)
    if name in {"aggregate", "execute_mode", "plan"}:
        from . import runner

        return getattr(runner, name)
    raise AttributeError(name)
