"""Evaluation and frozen-baseline reporting helpers."""

from .quality import build_perplexity_windows, evaluate_perplexity
from .runner import aggregate, execute_mode, plan

__all__ = ["aggregate", "build_perplexity_windows", "evaluate_perplexity", "execute_mode", "plan"]
