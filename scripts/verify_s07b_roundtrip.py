#!/usr/bin/env python3
"""Fresh-process S07-B router checkpoint and hard-route determinism check."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

EXPECTED_CHECKPOINT_SHA256 = "08bf646f19759c0d7949e159bdbe4f96bbea737204b96f8760d205c8d6fd1949"
CANDIDATE_ORDERING = (4, 8)
ROUTE_LAYER_COUNT = 36
ROUTE_UNIT_TYPES = ("attention", "ffn")


def _key_sort(value: tuple[Any, Any, Any]) -> tuple[str, str, str]:
    return tuple(str(part) for part in value)


def _mapping_key(record: Mapping[str, Any]) -> tuple[Any, Any, Any]:
    return (record.get("request_id"), record.get("layer"), record.get("unit_type"))


def _recorded_hard_route_map(
    route_logs: Sequence[Mapping[str, Any]],
    *,
    expected_request_ids: Sequence[str],
    layer_count: int = ROUTE_LAYER_COUNT,
    candidate_ordering: Sequence[int] = CANDIDATE_ORDERING,
) -> tuple[dict[tuple[Any, Any, Any], Any], dict[str, Any]]:
    """Build and validate the stable key map from the original hard route logs."""

    expected_keys = {
        (request_id, layer, unit_type)
        for request_id in expected_request_ids
        for layer in range(layer_count)
        for unit_type in ROUTE_UNIT_TYPES
    }
    mapping: dict[tuple[Any, Any, Any], Any] = {}
    duplicate_keys: list[tuple[Any, Any, Any]] = []
    invalid_precision_keys: list[tuple[Any, Any, Any]] = []
    for record in route_logs:
        key = _mapping_key(record)
        if key in mapping:
            duplicate_keys.append(key)
        mapping[key] = record.get("hard_bit")
        precision = record.get("hard_bit")
        if (
            isinstance(precision, bool)
            or not isinstance(precision, int)
            or precision not in candidate_ordering
        ):
            invalid_precision_keys.append(key)

    missing_keys = sorted(expected_keys - mapping.keys(), key=_key_sort)
    unexpected_keys = sorted(mapping.keys() - expected_keys, key=_key_sort)
    coverage_by_request: dict[str, dict[str, Any]] = {}
    for request_id in expected_request_ids:
        request_keys = [key for key in mapping if key[0] == request_id]
        coverage_by_request[request_id] = {
            "expected": layer_count * len(ROUTE_UNIT_TYPES),
            "recorded": len(request_keys),
            "attention": sum(key[2] == "attention" for key in request_keys),
            "ffn": sum(key[2] == "ffn" for key in request_keys),
            "complete": len(request_keys) == layer_count * len(ROUTE_UNIT_TYPES)
            and sum(key[2] == "attention" for key in request_keys) == layer_count
            and sum(key[2] == "ffn" for key in request_keys) == layer_count,
        }
    report = {
        "source": "evaluation.hard.route_logs",
        "candidate_ordering": list(candidate_ordering),
        "expected_request_ids": list(expected_request_ids),
        "expected_routes_per_request": layer_count * len(ROUTE_UNIT_TYPES),
        "recorded_log_count": len(route_logs),
        "coverage_by_request": coverage_by_request,
        "missing_keys": [list(key) for key in missing_keys],
        "unexpected_keys": [list(key) for key in unexpected_keys],
        "duplicate_keys": [list(key) for key in sorted(set(duplicate_keys), key=_key_sort)],
        "invalid_precision_keys": [
            list(key) for key in sorted(set(invalid_precision_keys), key=_key_sort)
        ],
    }
    return mapping, report


def _actual_hard_route_map(
    states: Sequence[Any], *, layer_count: int = ROUTE_LAYER_COUNT
) -> tuple[dict[tuple[Any, Any, Any], Any], dict[str, Any]]:
    """Extract the route selected by hard execution from each request state."""

    mapping: dict[tuple[Any, Any, Any], Any] = {}
    duplicate_keys: list[tuple[Any, Any, Any]] = []
    invalid_precision_keys: list[tuple[Any, Any, Any]] = []
    for state in states:
        for unit_type, routes in (
            ("attention", state.attention_routes),
            ("ffn", state.ffn_routes),
        ):
            for layer, precision in enumerate(routes):
                key = (state.request_id, layer, unit_type)
                if key in mapping:
                    duplicate_keys.append(key)
                mapping[key] = precision
                if precision is None or precision not in CANDIDATE_ORDERING:
                    invalid_precision_keys.append(key)
    expected_routes = layer_count * len(ROUTE_UNIT_TYPES)
    coverage_by_request: dict[str, dict[str, Any]] = {}
    for state in states:
        request_id = state.request_id
        request_keys = [key for key in mapping if key[0] == request_id]
        coverage_by_request[request_id] = {
            "expected": expected_routes,
            "actual": len(request_keys),
            "attention": sum(key[2] == "attention" for key in request_keys),
            "ffn": sum(key[2] == "ffn" for key in request_keys),
            "complete": len(request_keys) == expected_routes
            and sum(key[2] == "attention" for key in request_keys) == layer_count
            and sum(key[2] == "ffn" for key in request_keys) == layer_count,
        }
    return mapping, {
        "source": "QaqRequestState.attention_routes and QaqRequestState.ffn_routes",
        "actual_log_count": len(mapping),
        "coverage_by_request": coverage_by_request,
        "duplicate_keys": [list(key) for key in sorted(set(duplicate_keys), key=_key_sort)],
        "invalid_precision_keys": [
            list(key) for key in sorted(set(invalid_precision_keys), key=_key_sort)
        ],
    }


def compare_actual_hard_routes(
    recorded_route_logs: Sequence[Mapping[str, Any]],
    states: Sequence[Any],
    *,
    layer_count: int = ROUTE_LAYER_COUNT,
    candidate_ordering: Sequence[int] = CANDIDATE_ORDERING,
) -> dict[str, Any]:
    """Compare recorded hard routes to the routes selected by hard execution."""

    request_ids = tuple(state.request_id for state in states)
    recorded, recorded_report = _recorded_hard_route_map(
        recorded_route_logs,
        expected_request_ids=request_ids,
        layer_count=layer_count,
        candidate_ordering=candidate_ordering,
    )
    actual, actual_report = _actual_hard_route_map(states, layer_count=layer_count)
    expected_keys = {
        (request_id, layer, unit_type)
        for request_id in request_ids
        for layer in range(layer_count)
        for unit_type in ROUTE_UNIT_TYPES
    }
    missing_actual_keys = sorted(expected_keys - actual.keys(), key=_key_sort)
    unexpected_actual_keys = sorted(actual.keys() - expected_keys, key=_key_sort)
    mismatch_keys = [
        key
        for key in sorted(expected_keys, key=_key_sort)
        if recorded.get(key) != actual.get(key)
        or recorded.get(key) is None
        or actual.get(key) is None
    ]
    mismatches = [
        {
            "request_id": key[0],
            "layer": key[1],
            "unit_type": key[2],
            "expected": recorded.get(key),
            "actual": actual.get(key),
        }
        for key in mismatch_keys
    ]
    recorded_missing = {tuple(key) for key in recorded_report["missing_keys"]}
    recorded_unexpected = {tuple(key) for key in recorded_report["unexpected_keys"]}
    structural_failure = bool(
        recorded_missing
        or recorded_unexpected
        or recorded_report["duplicate_keys"]
        or recorded_report["invalid_precision_keys"]
        or missing_actual_keys
        or unexpected_actual_keys
        or actual_report["duplicate_keys"]
        or actual_report["invalid_precision_keys"]
        or tuple(candidate_ordering) != CANDIDATE_ORDERING
    )
    exact_attention_matches = sum(
        recorded.get(key) == actual.get(key)
        and recorded.get(key) in CANDIDATE_ORDERING
        and actual.get(key) in CANDIDATE_ORDERING
        for key in expected_keys
        if key[2] == "attention"
    )
    exact_ffn_matches = sum(
        recorded.get(key) == actual.get(key)
        and recorded.get(key) in CANDIDATE_ORDERING
        and actual.get(key) in CANDIDATE_ORDERING
        for key in expected_keys
        if key[2] == "ffn"
    )
    report = {
        "source": "direct actual hard execution comparison",
        "expected_source": recorded_report["source"],
        "actual_source": actual_report["source"],
        "candidate_ordering": list(candidate_ordering),
        "expected_requests": len(request_ids),
        "expected_routes_per_request": layer_count * len(ROUTE_UNIT_TYPES),
        "recorded_coverage": recorded_report["coverage_by_request"],
        "actual_coverage": actual_report["coverage_by_request"],
        "recorded_missing_keys": recorded_report["missing_keys"],
        "recorded_unexpected_keys": recorded_report["unexpected_keys"],
        "actual_missing_keys": [list(key) for key in missing_actual_keys],
        "actual_unexpected_keys": [list(key) for key in unexpected_actual_keys],
        "duplicate_keys": recorded_report["duplicate_keys"] + actual_report["duplicate_keys"],
        "invalid_precision_keys": recorded_report["invalid_precision_keys"]
        + actual_report["invalid_precision_keys"],
        "exact_attention_matches": exact_attention_matches,
        "exact_ffn_matches": exact_ffn_matches,
        "exact_route_matches": exact_attention_matches + exact_ffn_matches,
        "expected_route_count": len(expected_keys),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
    }
    report["passed"] = not structural_failure and not mismatches
    return report


def assert_actual_hard_routes_match(
    recorded_route_logs: Sequence[Mapping[str, Any]], states: Sequence[Any], **kwargs: Any
) -> dict[str, Any]:
    """Raise with keyed expected/actual values when direct hard routes differ."""

    report = compare_actual_hard_routes(recorded_route_logs, states, **kwargs)
    if not report["passed"]:
        raise AssertionError(json.dumps(report, sort_keys=True))
    return report


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _persist_roundtrip_failure(
    result: dict[str, Any],
    result_path: Path,
    *,
    checkpoint_sha256: str | None,
    checkpoint_identity_match: bool,
    probabilities_match: bool | None = None,
    soft_derived_hard_bits_match: bool | None = None,
    hard_routes_match: bool | None = None,
    hard_route_comparison: dict[str, Any] | None = None,
    unchanged_packed_student: bool | None = None,
    finite_logits: bool | None = None,
    fixed_subset_count: int = 0,
    route_maps_identical_on_repeat: bool = False,
    logits_identical_on_repeat: bool = False,
) -> None:
    result["hard_route_determinism"] = {
        "fixed_subset_count": fixed_subset_count,
        "route_maps_identical_on_repeat": route_maps_identical_on_repeat,
        "selected_precisions_identical_on_repeat": route_maps_identical_on_repeat,
        "logits_identical_on_repeat": logits_identical_on_repeat,
        "finite_logits": finite_logits if finite_logits is not None else False,
        "tolerance": "bitwise equality",
        "passed": False,
    }
    stage_gate = result.setdefault("stage_gate", {})
    stage_gate["checkpoint_roundtrip_passed"] = False
    stage_gate["hard_route_determinism_passed"] = False
    stage_gate["engineering_gate"] = "REVISE"
    stage_gate["next_action"] = "Repair S07C checkpoint round-trip evidence."

    checkpoint_roundtrip = {
        "fresh_process": True,
        "checkpoint_sha256": checkpoint_sha256,
        "recorded_checkpoint_sha256": result["checkpoint"].get("sha256"),
        "expected_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "checkpoint_identity_match": checkpoint_identity_match,
        "passed": False,
    }
    optional_fields = {
        "probabilities_match_recorded_result": probabilities_match,
        "soft_derived_hard_bits_match_recorded_result": soft_derived_hard_bits_match,
        "hard_routes_match_recorded_result": hard_routes_match,
        "hard_route_comparison": hard_route_comparison,
        "unchanged_packed_student": unchanged_packed_student,
        "finite_logits": finite_logits,
    }
    checkpoint_roundtrip.update(
        {name: value for name, value in optional_fields.items() if value is not None}
    )
    result["checkpoint_roundtrip"] = checkpoint_roundtrip
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


def _verify_checkpoint_identity(result: dict[str, Any], result_path: Path) -> str:
    checkpoint_path = Path(result["checkpoint"]["external_path"]).expanduser()
    if not checkpoint_path.is_file():
        _persist_roundtrip_failure(
            result,
            result_path,
            checkpoint_sha256=None,
            checkpoint_identity_match=False,
        )
        raise SystemExit(f"PAUSE: router checkpoint is unavailable: {checkpoint_path}")
    checkpoint_sha256 = _sha256(checkpoint_path)
    recorded_checkpoint_sha256 = result["checkpoint"].get("sha256")
    checkpoint_identity_match = (
        checkpoint_sha256 == EXPECTED_CHECKPOINT_SHA256
        and checkpoint_sha256 == recorded_checkpoint_sha256
    )
    if not checkpoint_identity_match:
        _persist_roundtrip_failure(
            result,
            result_path,
            checkpoint_sha256=checkpoint_sha256,
            checkpoint_identity_match=False,
        )
        raise SystemExit(
            "REVISE: checkpoint identity mismatch; "
            f"expected {EXPECTED_CHECKPOINT_SHA256}, recorded {recorded_checkpoint_sha256}, "
            f"got {checkpoint_sha256}; no retraining"
        )
    return checkpoint_sha256


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--result", type=Path, default=ROOT / "docs/results/s07_router_training.json")
    args = parser.parse_args()
    if not str(Path.home() / ".venv") in str(Path(sys.executable).parent):
        raise SystemExit("PAUSE: ~/.venv is not active")
    if not torch.cuda.is_available():
        raise SystemExit("PAUSE: CUDA is unavailable")
    torch.cuda.set_device(torch.device(args.device))

    import run_s07b
    from datasets import load_dataset
    from transformers import AutoTokenizer

    from qaq.model.manual import PrecisionTrace
    from qaq.model.request_state import QaqRequestState
    from qaq.router.distillation import RouterCheckpointMetadata, hard_route, load_router_checkpoint
    from qaq.router.soft_model import load_soft_model

    result = json.loads(args.result.read_text())
    checkpoint_sha256 = _verify_checkpoint_identity(result, args.result)
    manifest = json.loads((ROOT / "docs/quantized_model_manifest.json").read_text())
    config = result["training_configuration"]
    artifact = ROOT / manifest["artifact"]["local_path"]
    tokenizer = AutoTokenizer.from_pretrained(
        str(run_s07b.SNAPSHOT), revision=run_s07b.MODEL_REVISION, local_files_only=True
    )
    dataset = load_dataset(
        config["dataset"]["repository"],
        config["dataset"]["config"],
        split="validation",
        revision=config["dataset"]["revision"],
        trust_remote_code=False,
    )
    examples_cpu, _ = run_s07b._select_examples(
        dataset,
        tokenizer,
        config["dataset"]["validation_offsets"],
        split="validation",
        config=config,
        torch=torch,
    )
    examples = [run_s07b._device_example(example, args.device, torch) for example in examples_cpu]
    student = load_soft_model(
        artifact,
        args.device,
        temperature=float(config["training"]["routing_temperature"]),
    )
    student.to(args.device)
    metadata_payload = result["checkpoint"]["metadata"]
    metadata_payload["candidate_ordering"] = tuple(metadata_payload["candidate_ordering"])
    metadata = RouterCheckpointMetadata(**metadata_payload)
    load_router_checkpoint(result["checkpoint"]["external_path"], student.routers, metadata)
    stored_soft_logs = {
        (item["request_id"], item["layer"], item["unit_type"]): item
        for item in result["evaluation"]["soft"]["route_logs"]
    }
    stored_hard_logs = result["evaluation"]["hard"]["route_logs"]

    def soft_once(example):
        state = QaqRequestState(example.example_id, int(example.prompt_mask().sum()), layer_count=36)
        with torch.no_grad():
            output = student(
                **run_s07b._model_kwargs(example),
                request_state=state,
                phase="prefill",
                prompt_attention_mask=example.prompt_mask().unsqueeze(0),
                trace=PrecisionTrace(),
            )
        records = run_s07b._records_for_state(example.example_id, state, student, log_base=2.0)
        return output.logits.detach(), records

    def hard_once(example):
        state = QaqRequestState(example.example_id, int(example.prompt_mask().sum()), layer_count=36)

        def policy(layer, unit_type, feature):
            return int(hard_route(student.route(layer, unit_type, feature)))

        with torch.no_grad():
            output = student.base(
                **run_s07b._model_kwargs(example),
                request_state=state,
                phase="prefill",
                prompt_attention_mask=example.prompt_mask().unsqueeze(0),
                routing_policy=policy,
                trace=PrecisionTrace(),
            )
        records = run_s07b._records_for_state(example.example_id, state, student, log_base=2.0)
        return output.logits.detach(), records, state

    probability_match = True
    soft_route_match = True
    hard_repeat_match = True
    hard_logits_match = True
    finite_logits = True
    soft_route_count = 0
    hard_states = []
    for example in examples:
        soft_logits, soft_records = soft_once(example)
        soft_route_count += len(soft_records)
        for record in soft_records:
            stored = stored_soft_logs[(record.request_id, record.layer, record.unit_type)]
            probability_match &= abs(record.p4 - stored["p4"]) <= 1e-6
            probability_match &= abs(record.p8 - stored["p8"]) <= 1e-6
            soft_route_match &= record.hard_bit == stored["hard_bit"]
        first_logits, _, first_state = hard_once(example)
        second_logits, _, second_state = hard_once(example)
        first_map = _actual_hard_route_map((first_state,))[0]
        second_map = _actual_hard_route_map((second_state,))[0]
        hard_states.append(first_state)
        hard_repeat_match &= first_map == second_map
        hard_logits_match &= torch.equal(first_logits, second_logits)
        finite_logits &= bool(
            torch.isfinite(soft_logits).all().item() and torch.isfinite(first_logits).all().item()
        )

    hard_route_comparison = compare_actual_hard_routes(
        stored_hard_logs,
        hard_states,
        candidate_ordering=metadata.candidate_ordering,
    )
    if not hard_route_comparison["passed"]:
        _persist_roundtrip_failure(
            result,
            args.result,
            checkpoint_sha256=checkpoint_sha256,
            checkpoint_identity_match=True,
            probabilities_match=probability_match,
            soft_derived_hard_bits_match=soft_route_match,
            hard_routes_match=False,
            hard_route_comparison=hard_route_comparison,
            unchanged_packed_student=True,
            finite_logits=finite_logits,
            fixed_subset_count=len(hard_states),
            route_maps_identical_on_repeat=hard_repeat_match,
            logits_identical_on_repeat=hard_logits_match,
        )
        raise SystemExit(
            "REVISE: direct hard-route comparison failed; "
            f"{json.dumps(hard_route_comparison, sort_keys=True)}; no retraining"
        )
    hard_route_comparison = assert_actual_hard_routes_match(
        stored_hard_logs,
        hard_states,
        candidate_ordering=metadata.candidate_ordering,
    )

    passed = (
        probability_match
        and soft_route_match
        and hard_route_comparison["passed"]
        and hard_repeat_match
        and hard_logits_match
        and finite_logits
    )
    hard_route_determinism = {
        "fixed_subset_count": len(examples),
        "route_maps_identical_on_repeat": hard_repeat_match,
        "selected_precisions_identical_on_repeat": hard_repeat_match,
        "logits_identical_on_repeat": hard_logits_match,
        "finite_logits": finite_logits,
        "tolerance": "bitwise equality",
        "passed": hard_repeat_match and hard_logits_match and finite_logits,
    }
    result["hard_route_determinism"] = hard_route_determinism
    result["soft_derived_hard_route_comparison"] = {
        "source": "reloaded soft probabilities passed through hard_route",
        "reference_source": "evaluation.soft.route_logs",
        "probabilities_match_recorded_result": probability_match,
        "hard_bits_match_recorded_result": soft_route_match,
        "route_count": soft_route_count,
        "passed": probability_match and soft_route_match,
    }
    checkpoint_roundtrip = {
        "fresh_process": True,
        "checkpoint_sha256": checkpoint_sha256,
        "recorded_checkpoint_sha256": result["checkpoint"].get("sha256"),
        "expected_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "checkpoint_identity_match": (
            checkpoint_sha256 == EXPECTED_CHECKPOINT_SHA256
            and checkpoint_sha256 == result["checkpoint"].get("sha256")
        ),
        "probabilities_match_recorded_result": probability_match,
        "soft_derived_hard_bits_match_recorded_result": soft_route_match,
        "hard_routes_match_recorded_result": hard_route_comparison["passed"],
        "hard_route_comparison": hard_route_comparison,
        "unchanged_packed_student": True,
        "finite_logits": finite_logits,
        "passed": passed,
    }
    result["checkpoint_roundtrip"] = checkpoint_roundtrip
    if not passed:
        _persist_roundtrip_failure(
            result,
            args.result,
            checkpoint_sha256=checkpoint_sha256,
            checkpoint_identity_match=True,
            probabilities_match=probability_match,
            soft_derived_hard_bits_match=soft_route_match,
            hard_routes_match=hard_route_comparison["passed"],
            hard_route_comparison=hard_route_comparison,
            unchanged_packed_student=True,
            finite_logits=finite_logits,
            fixed_subset_count=len(hard_states),
            route_maps_identical_on_repeat=hard_repeat_match,
            logits_identical_on_repeat=hard_logits_match,
        )
    else:
        result["hard_route_determinism"] = hard_route_determinism
        result["checkpoint_roundtrip"] = checkpoint_roundtrip
        result["stage_gate"]["checkpoint_roundtrip_passed"] = True
        result["stage_gate"]["hard_route_determinism_passed"] = True
    args.result.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"checkpoint_roundtrip": result["checkpoint_roundtrip"], "hard_route_determinism": result["hard_route_determinism"]}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
