#!/usr/bin/env python3
"""Compatibility entry point for baseline evaluation protocol validation."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qaq.evaluation import protocol as _implementation

if __name__ == "__main__":
    raise SystemExit(_implementation.main())

sys.modules[__name__] = _implementation
