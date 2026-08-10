"""Run and print the complete S01 Any-Precision backend evidence."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from qaq.s01_backend import full_validation_report


def main() -> int:
    try:
        report = full_validation_report()
    except (AssertionError, ImportError, OSError, RuntimeError, ValueError) as exc:  # pragma: no cover
        print(f"S01 validation FAILED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
