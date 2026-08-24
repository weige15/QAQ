#!/usr/bin/env python3
"""Plan, explicitly execute, or aggregate the frozen S11-C broader quality check."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qaq.evaluation import lookahead_broader_quality as contract


def _print(value: dict) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--plan", action="store_true")
    action.add_argument("--execute-mode", choices=contract.MODE_IDS)
    action.add_argument("--aggregate", action="store_true")
    parser.add_argument("--config", type=Path, default=contract.DEFAULT_CONFIG)
    parser.add_argument("--device")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.execute_mode:
            if args.device is None or args.output is None:
                parser.error("--execute-mode requires both --device and --output")
            config, _ = contract.validate_dispatch(
                mode_id=args.execute_mode,
                device=args.device,
                output=args.output,
                config_path=args.config,
            )
            # Reuse the existing production runtime only after fail-closed dispatch.
            from qaq.evaluation.lookahead_quality_runtime import ProductionRuntime

            result = contract.execute_mode_with_runtime(
                ProductionRuntime(),
                config=config,
                mode_id=args.execute_mode,
                device=args.device,
            )
            digest = contract.persist_validated_result(
                result,
                args.output,
                config=config,
                kind="mode",
            )
            _print(
                {
                    "classification": "MODE_RESULT_VALID",
                    "mode_id": args.execute_mode,
                    "output": str(args.output),
                    "sha256": digest,
                    "written": True,
                }
            )
            return 0
        if args.aggregate:
            if args.device is not None:
                parser.error("--aggregate does not accept --device")
            expected = ROOT / contract.AGGREGATION_OUTPUT
            if args.output is not None and args.output.resolve() != expected.resolve():
                parser.error(f"--aggregate output must be {expected}")
            aggregate, report = contract.aggregate_paths(config_path=args.config)
            if aggregate is None:
                _print({**report, "written": False})
                return 2 if report["classification"] == "PAUSE" else 1
            config, _ = contract.load_protocol(args.config)
            control = json.loads((ROOT / contract.OUTPUTS[contract.MODE_IDS[0]]).read_text())
            treatment = json.loads((ROOT / contract.OUTPUTS[contract.MODE_IDS[1]]).read_text())
            digest = contract.persist_validated_result(
                aggregate,
                expected,
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
        _print(contract.plan(args.config))
        return 0
    except (OSError, RuntimeError, ValueError) as exc:
        text = str(exc)
        classification = "PAUSE" if text.startswith("PAUSE:") else "REVISE"
        _print({"classification": classification, "errors": [text], "written": False})
        return 2 if classification == "PAUSE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
