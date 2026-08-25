#!/usr/bin/env python3
"""Print or fail-closed validate the frozen S11-D paired 4/6/8 plan."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qaq.evaluation import lookahead_468_executor as executor


def _print(value: dict) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--plan", action="store_true")
    action.add_argument("--execute-trial")
    action.add_argument("--aggregate", action="store_true")
    parser.add_argument("--config", type=Path, default=executor.DEFAULT_CONFIG)
    parser.add_argument("--device")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.execute_trial is not None:
            if args.device is None or args.output is None:
                parser.error("--execute-trial requires --device and --output")
            executor.validate_execution_request(
                trial_id=args.execute_trial,
                device=args.device,
                output=args.output,
                config_path=args.config,
            )
        elif args.aggregate:
            if args.device is not None or args.output is None:
                parser.error("--aggregate requires --output and does not accept --device")
            executor.validate_aggregation_request(output=args.output, config_path=args.config)
        else:
            if args.device is not None or args.output is not None:
                parser.error("plan/default mode accepts neither --device nor --output")
            _print(executor.plan(args.config))
            return 0
    except executor.ProtocolError as exc:
        _print(
            {
                "classification": exc.outcome,
                "errors": [str(exc)],
                "executed": False,
                "written": False,
            }
        )
        return 2 if exc.outcome == "PAUSE" else 1
    raise AssertionError("execution validation must stop before runtime dispatch")


if __name__ == "__main__":
    raise SystemExit(main())
