"""Temporary compatibility alias for historical command records.

Active repository consumers use ``scripts/train_baseline_router.py``.  This shim preserves
commands embedded in frozen protocols and result provenance without retaining
a second implementation.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

_PATH_ALIASES = {
    "configs/s03_static_generation_prompts.txt": "configs/static_generation_prompts.txt",
    "configs/s03_static_quality_prompts.txt": "configs/static_quality_prompts.txt",
    "configs/s07_router_training.json": "configs/baseline_router_training.json",
    "configs/s09_baseline_eval.json": "configs/baseline_evaluation.json",
    "configs/s09_baseline_prompts.json": "configs/baseline_evaluation_prompts.json",
    "configs/s10d_lambda_calibration.json": "configs/router_cost_calibration.json",
    "configs/s10e_frontier_confirmation.json": "configs/router_frontier_confirmation.json",
    "configs/s10g_broader_validation.json": "configs/broader_router_validation.json",
    "configs/s11b_quality_pilot.json": "configs/lookahead_quality_pilot.json",
}


def main() -> None:
    target = Path(__file__).with_name("train_baseline_router.py")
    sys.argv = [str(target), *(_PATH_ALIASES.get(arg, arg) for arg in sys.argv[1:])]
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
