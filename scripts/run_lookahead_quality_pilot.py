#!/usr/bin/env python3
"""Plan, explicitly execute, or aggregate the frozen S11-B quality pilot.

The default path is deterministic and non-executing.  Production runtime code
is imported only after exact mode/device/output dispatch validation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from qaq.evaluation.lookahead_quality_runner import (
    AGGREGATION_OUTPUT,
    DEFAULT_CONFIG,
    MODE_IDS,
    OUTPUTS,
    LookaheadQualityError,
    PersistencePolicy,
    aggregate_paths,
    execute_mode_with_runtime,
    load_protocol,
    persist_validated_result,
    plan,
    validate_dispatch,
)


def _print(payload: dict) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--plan", action="store_true", help="print the inert deterministic plan")
    action.add_argument("--execute-mode", choices=MODE_IDS, help="execute one exact frozen mode")
    action.add_argument(
        "--aggregate", action="store_true", help="validate and pair both mode results"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", help="explicit cuda:<index>, required only for --execute-mode")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    try:
        if args.execute_mode is not None:
            if args.device is None or args.output is None:
                parser.error("--execute-mode requires both --device and --output")
            config, _ = validate_dispatch(
                mode_id=args.execute_mode,
                device=args.device,
                output=args.output,
                config_path=args.config,
            )
            # The only production-heavy import boundary.
            from qaq.evaluation.lookahead_quality_runtime import ProductionRuntime

            result = execute_mode_with_runtime(
                ProductionRuntime(),
                config=config,
                mode_id=args.execute_mode,
                device=args.device,
            )
            expected = ROOT / OUTPUTS[args.execute_mode]
            digest = persist_validated_result(
                result,
                args.output,
                policy=PersistencePolicy(expected, expected.parent),
                config=config,
                kind="mode",
            )
            _print(
                {
                    "classification": "MODE_RESULT_VALID",
                    "mode_id": args.execute_mode,
                    "output": str(expected),
                    "sha256": digest,
                    "written": True,
                }
            )
            return 0

        if args.aggregate:
            expected = ROOT / AGGREGATION_OUTPUT
            if args.device is not None:
                parser.error("--aggregate does not accept --device")
            if args.output is not None and args.output.resolve() != expected.resolve():
                parser.error(f"--aggregate output must be {expected}")
            aggregate, report = aggregate_paths(config_path=args.config)
            if aggregate is None:
                _print({**report, "written": False})
                return 2 if report["classification"] == "PAUSE" else 1
            config, _ = load_protocol(args.config, require_results_absent=False)
            control = json.loads((ROOT / OUTPUTS[MODE_IDS[0]]).read_text())
            treatment = json.loads((ROOT / OUTPUTS[MODE_IDS[1]]).read_text())
            digest = persist_validated_result(
                aggregate,
                expected,
                policy=PersistencePolicy(expected, expected.parent),
                config=config,
                kind="aggregation",
                paired_results=(control, treatment),
            )
            _print(
                {
                    "classification": aggregate["classification"],
                    "errors": aggregate["errors"],
                    "output": str(expected),
                    "sha256": digest,
                    "written": True,
                }
            )
            return 0

        if args.device is not None or args.output is not None:
            parser.error("plan/default mode accepts neither --device nor --output")
        _print(plan(args.config))
        return 0
    except (OSError, LookaheadQualityError, RuntimeError, ValueError) as exc:
        text = str(exc)
        classification = "PAUSE" if text.startswith("PAUSE:") else "INVALID_EVIDENCE"
        _print({"classification": classification, "errors": [text], "written": False})
        return 2 if classification == "PAUSE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
