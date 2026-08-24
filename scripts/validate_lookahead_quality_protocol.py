#!/usr/bin/env python3
"""Read-only adapter for the frozen lookahead quality-pilot protocol validator."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qaq.evaluation import lookahead_quality_protocol as _protocol

# Keep the historical script import surface available to focused B1 tests while
# the reusable implementation remains in a semantically named source module.
globals().update(
    {
        name: getattr(_protocol, name)
        for name in dir(_protocol)
        if not name.startswith("__")
    }
)

if __name__ == "__main__":
    raise SystemExit(_protocol.main())
