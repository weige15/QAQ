#!/usr/bin/env python3
"""Build or verify the read-only S11-D route-policy diagnostic."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.dont_write_bytecode = True

from qaq.evaluation import s11d_route_diagnostic as diagnostic


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--output", type=Path, help="write the derived diagnostic JSON")
    action.add_argument("--check", type=Path, help="verify an existing derived diagnostic JSON")
    args = parser.parse_args(argv)

    value = diagnostic.build_diagnostic()
    raw = diagnostic.serialize_diagnostic(value)
    path = args.output or args.check
    if path is not None and diagnostic.CANONICAL_PARENT.resolve() in path.resolve().parents:
        parser.error("derived diagnostics must not be written inside the canonical S11-D directory")
    if args.check is not None:
        if args.check.read_bytes() != raw:
            print(f"derived diagnostic is stale: {args.check}", file=sys.stderr)
            return 1
        print(f"verified derived diagnostic: {args.check}")
        return 0
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(raw)
        print(f"wrote derived diagnostic: {args.output}")
        return 0
    sys.stdout.buffer.write(raw)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
