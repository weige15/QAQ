#!/usr/bin/env python3
"""Plan or explicitly execute the frozen S09-B five-mode comparison."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from qaq.s09_runner import (
    DEFAULT_CONFIG,
    DEFAULT_RESULTS,
    S09RunnerError,
    aggregate,
    execute_mode,
    plan,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--plan", action="store_true", help="validate and print the non-executing plan")
    action.add_argument("--execute", action="store_true", help="launch one fresh child for each frozen mode")
    action.add_argument("--execute-mode", metavar="MODE", help=argparse.SUPPRESS)
    action.add_argument("--aggregate", action="store_true", help="validate five existing per-mode results")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda:3")
    args = parser.parse_args(argv)

    try:
        if args.execute_mode:
            if args.output is None:
                parser.error("--execute-mode requires --output")
            result = execute_mode(args.config.resolve(), args.execute_mode, args.output.resolve(), args.device)
            print(json.dumps({"mode_id": result["mode_id"], "output": str(args.output.resolve())}, sort_keys=True))
            return 0
        if args.aggregate:
            result = aggregate(args.config.resolve(), args.results_dir.resolve())
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0 if result["classification"] == "CONTINUE" else 2 if result["classification"] == "PAUSE" else 1
        if args.execute:
            details = plan(args.config.resolve(), args.results_dir.resolve(), args.device)
            for command in details["child_commands"]:
                completed = subprocess.run(command, check=False)
                if completed.returncode:
                    return completed.returncode
            return 0
        details = plan(args.config.resolve(), args.results_dir.resolve(), args.device)
        print(json.dumps(details, indent=2, sort_keys=True))
        return 0
    except (S09RunnerError, OSError, subprocess.CalledProcessError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
