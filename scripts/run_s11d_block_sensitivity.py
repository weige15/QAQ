#!/usr/bin/env python3
"""Print or structurally validate the defined S11-D block-sensitivity study.

This command deliberately contains no model/CUDA runtime import.  Dispatch mode
only validates a future execution request; it never executes the study.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qaq.evaluation import block_sensitivity as sensitivity


def _print(value: dict) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--plan", action="store_true")
    action.add_argument("--validate-plan", type=Path)
    action.add_argument("--validate-result", type=Path)
    action.add_argument("--resume-plan", action="store_true")
    action.add_argument("--validate-dispatch", action="store_true")
    action.add_argument("--aggregate", action="store_true")
    parser.add_argument("--unit")
    parser.add_argument("--precision", type=int, choices=sensitivity.PRECISION_ORDER)
    parser.add_argument("--device")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        plan = sensitivity.build_plan()
        if args.validate_plan:
            value, _ = sensitivity._load_json(args.validate_plan)
            sensitivity.validate_plan(value)
            _print(
                {"classification": "PLAN_VALID", "errors": [], "executed": False, "written": False}
            )
            return 0
        if args.validate_result:
            value, _ = sensitivity._load_json(args.validate_result)
            sensitivity.validate_unit_result(value, plan)
            _print(
                {
                    "classification": "UNIT_RESULT_VALID",
                    "errors": [],
                    "executed": False,
                    "written": False,
                }
            )
            return 0
        if args.resume_plan:
            if any(
                value is not None for value in (args.unit, args.precision, args.device, args.output)
            ):
                parser.error("--resume-plan accepts no unit, precision, device, or output")
            _print(sensitivity.build_resume_state(plan))
            return 0
        if args.validate_dispatch:
            if None in (args.unit, args.precision, args.device, args.output):
                parser.error(
                    "--validate-dispatch requires --unit, --precision, --device, and --output"
                )
            sensitivity.validate_execution_request(
                unit_id=args.unit,
                precision=args.precision,
                device=args.device,
                output=args.output,
                plan=plan,
            )
            _print(
                {
                    "classification": "DISPATCH_VALID_NOT_EXECUTED",
                    "unit_id": args.unit,
                    "precision": args.precision,
                    "output": str(args.output),
                    "errors": [],
                    "executed": False,
                    "written": False,
                }
            )
            return 0
        if args.aggregate:
            if args.unit is not None or args.precision is not None or args.device is not None:
                parser.error("--aggregate accepts no unit, precision, or device")
            output = args.output or sensitivity.AGGREGATION_OUTPUT
            if output.resolve() != sensitivity.AGGREGATION_OUTPUT.resolve():
                parser.error(f"--aggregate output must be {sensitivity.AGGREGATION_OUTPUT}")
            results = sensitivity.load_results_for_aggregation(plan)
            aggregation = sensitivity.build_aggregation(results, plan)
            digest = sensitivity.persist_aggregation(aggregation, output, results, plan)
            _print(
                {
                    "classification": "AGGREGATION_VALID",
                    "output": str(output),
                    "sha256": digest,
                    "errors": [],
                    "executed": False,
                    "written": True,
                }
            )
            return 0
        if any(
            value is not None for value in (args.unit, args.precision, args.device, args.output)
        ):
            parser.error("plan mode accepts no unit, precision, device, or output")
        _print(plan)
        return 0
    except sensitivity.MissingEvidence as exc:
        _print(
            {"classification": "PAUSE", "errors": [str(exc)], "executed": False, "written": False}
        )
        return 2
    except (OSError, KeyError, TypeError, ValueError) as exc:
        _print(
            {"classification": "REVISE", "errors": [str(exc)], "executed": False, "written": False}
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
