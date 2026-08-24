"""Fail-closed S11-B2 planning, result validation, pairing, and persistence.

This module is deliberately standard-library-only.  Production model/CUDA work
is imported only after explicit execution dispatch in ``scripts/run_lookahead_quality_pilot.py``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import struct
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "configs" / "lookahead_quality_pilot.json"
EXPECTED_CONFIG_SHA256 = "21a664424debe4892c3577c490158228dd5399bb4b425611db728070d23a5051"
MODE_SCHEMA = "qaq-s11b-quality-pilot-mode-result-v1"
AGGREGATION_SCHEMA = "qaq-s11b-quality-pilot-aggregation-v1"
MODE_IDS = ("same_unit_control", "lookahead_attention_one_unit_treatment")
ROUTING_TIMINGS = ("same_unit", "lookahead_attention_one_unit")
REQUEST_IDS = ("validation-3", "validation-1000")
UNIT_TYPES = ("attention", "ffn")
CANDIDATE_BITS = (4, 8)
OUTPUTS = {
    "same_unit_control": "docs/results/s11b_quality_pilot/same_unit_control.json",
    "lookahead_attention_one_unit_treatment": (
        "docs/results/s11b_quality_pilot/lookahead_attention_one_unit_treatment.json"
    ),
}
AGGREGATION_OUTPUT = "docs/results/s11b_quality_pilot/aggregation.json"
CLASSIFICATIONS = (
    "INVALID_EVIDENCE",
    "PAUSE",
    "ADVANCE_TO_BROADER_QUALITY_CHECK",
    "CHECKPOINT_REUSE_DEGRADES",
)
PER_MODE_KEYS = (
    "schema",
    "mode_id",
    "protocol_config_sha256",
    "identities",
    "hardware",
    "seed",
    "inputs",
    "repeats",
    "quality",
    "routes",
    "provenance",
    "freeze_audit",
    "prohibited_work_audit",
)
AGGREGATION_KEYS = (
    "schema",
    "protocol_config_sha256",
    "mode_result_paths",
    "paired_inputs",
    "paired_quality",
    "route_comparison",
    "determinism",
    "freeze_audits",
    "result_checks",
    "classification",
    "errors",
)
_HEX64 = re.compile(r"[0-9a-f]{64}")
_CUDA_DEVICE = re.compile(r"cuda:(0|[1-9][0-9]*)")
_PROHIBITED_DATA_KEYS = {
    "generation",
    "generated_tokens",
    "decode",
    "perplexity",
    "latency",
    "memory",
    "transfer",
    "throughput",
    "benchmark",
    "profiling",
    "optimizer_state",
    "checkpoint_output",
}


class LookaheadQualityError(ValueError):
    """A protocol, result, dispatch, or persistence invariant failed."""


class InvalidEvidence(LookaheadQualityError):
    """Complete supplied evidence is malformed or inconsistent."""


class MissingExternalResult(LookaheadQualityError):
    """A frozen external result required for aggregation is absent."""


@dataclass(frozen=True)
class PersistencePolicy:
    """One exact destination and its already-existing allowed parent."""

    expected_destination: Path
    allowed_parent: Path


class LookaheadQualityRuntime(Protocol):
    """Injected boundary shared by production and structural scheduling tests."""

    evidence_label: str

    def prepare(
        self,
        protocol: Mapping[str, Any],
        mode: Mapping[str, Any],
        device: str,
        requests: Sequence[Mapping[str, Any]],
    ) -> None: ...

    def hardware_evidence(self) -> dict[str, Any]: ...

    def identity_evidence(self) -> dict[str, Any]: ...

    def run_request(
        self,
        *,
        mode: Mapping[str, Any],
        request: Mapping[str, Any],
        repeat_index: int,
        device: str,
    ) -> dict[str, Any]: ...

    def freeze_audit(self) -> dict[str, Any]: ...

    def close(self) -> None: ...


def _reject_constant(value: str) -> None:
    raise LookaheadQualityError(f"non-finite JSON constant is forbidden: {value}")


def _no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise LookaheadQualityError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json_bytes(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_no_duplicates,
            parse_constant=_reject_constant,
        )
    except LookaheadQualityError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LookaheadQualityError(f"cannot parse JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise LookaheadQualityError(f"JSON root must be an object: {path}")
    return payload, raw


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _token_digest(token_ids: Sequence[int]) -> str:
    return hashlib.sha256(struct.pack("<" + "q" * len(token_ids), *token_ids)).hexdigest()


def _is_hex64(value: Any) -> bool:
    return isinstance(value, str) and _HEX64.fullmatch(value) is not None


def _finite_nonnegative(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise InvalidEvidence(message)


def _normal(path: Path) -> Path:
    return Path(os.path.abspath(os.path.expanduser(path)))


def _assert_no_prohibited_data_fields(value: Any, *, path: str = "result") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in _PROHIBITED_DATA_KEYS:
                raise InvalidEvidence(f"prohibited result field: {path}.{key}")
            _assert_no_prohibited_data_fields(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_no_prohibited_data_fields(child, path=f"{path}[{index}]")


def load_protocol(
    config_path: Path = DEFAULT_CONFIG, *, require_results_absent: bool = False
) -> tuple[dict[str, Any], str]:
    """Load the byte-frozen protocol, optionally applying the B1 absence gate."""

    config_path = Path(config_path)
    config, raw = _load_json_bytes(config_path)
    digest = hashlib.sha256(raw).hexdigest()
    if digest != EXPECTED_CONFIG_SHA256:
        raise LookaheadQualityError(
            f"frozen config SHA-256 mismatch: {digest} != {EXPECTED_CONFIG_SHA256}"
        )
    from qaq.evaluation import lookahead_quality_protocol as frozen

    # Reuse every B1 semantic check.  Execution/aggregation may encounter an
    # already completed sibling mode, so only the inert plan applies B1's
    # all-results-absent check.
    frozen._assert_finite(config)
    frozen._reject_forbidden_fields(config)
    frozen._validate_exact_contract(config)
    frozen._validate_authoritative_sources(config, ROOT)
    if require_results_absent:
        frozen._validate_output_paths(config, ROOT)
    else:
        mode_outputs = config["planned_results"]["mode_outputs"]
        paths = tuple(item["path"] for item in mode_outputs) + (
            config["planned_results"]["aggregation_output"],
        )
        if paths != tuple(OUTPUTS.values()) + (AGGREGATION_OUTPUT,):
            raise LookaheadQualityError("frozen output paths drifted")
    return config, digest


def _fixed_requests(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    frozen_path = str(config["identities"]["fixed_inputs_path"])
    prompt_path = {
        "configs/s09_baseline_prompts.json": "configs/baseline_evaluation_prompts.json"
    }.get(frozen_path, frozen_path)
    prompts, _ = _load_json_bytes(ROOT / prompt_path)
    source = {
        item.get("id"): item for item in prompts.get("requests", []) if isinstance(item, dict)
    }
    requests: list[dict[str, Any]] = []
    for frozen in config["fixed_inputs"]["requests"]:
        request_id = frozen["source_record_id"]
        record = source.get(request_id)
        if not isinstance(record, dict):
            raise LookaheadQualityError(f"fixed request is unavailable: {request_id}")
        token_ids = record.get("full_input_ids")
        if (
            not isinstance(token_ids, list)
            or len(token_ids) != 64
            or any(isinstance(value, bool) or not isinstance(value, int) for value in token_ids)
            or _token_digest(token_ids) != frozen["token_digest_sha256"]
        ):
            raise LookaheadQualityError(f"fixed request token identity drifted: {request_id}")
        requests.append({**frozen, "request_id": request_id, "full_input_ids": list(token_ids)})
    if tuple(item["request_id"] for item in requests) != REQUEST_IDS:
        raise LookaheadQualityError("fixed request order drifted")
    return requests


def _mode(config: Mapping[str, Any], mode_id: str) -> dict[str, Any]:
    if mode_id not in MODE_IDS:
        raise LookaheadQualityError(f"unknown S11-B mode: {mode_id!r}")
    modes = config.get("modes")
    if not isinstance(modes, list):
        raise LookaheadQualityError("frozen modes are unavailable")
    result = next((item for item in modes if item.get("id") == mode_id), None)
    if not isinstance(result, dict):
        raise LookaheadQualityError(f"frozen mode is unavailable: {mode_id}")
    return dict(result)


def _validate_device(device: str) -> str:
    if not isinstance(device, str) or _CUDA_DEVICE.fullmatch(device) is None:
        raise LookaheadQualityError("an explicit CUDA device of the form cuda:<index> is required")
    return device


def plan(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Return the deterministic non-executing S11-B2 plan."""

    config, digest = load_protocol(config_path, require_results_absent=True)
    requests = _fixed_requests(config)
    script = ROOT / "scripts" / "run_lookahead_quality_pilot.py"
    child_commands = []
    for mode_id in MODE_IDS:
        child_commands.append(
            [
                sys.executable,
                str(script),
                "--execute-mode",
                mode_id,
                "--config",
                str(Path(config_path).resolve()),
                "--device",
                "<explicit-cuda-device>",
                "--output",
                str(ROOT / OUTPUTS[mode_id]),
            ]
        )
    aggregation_command = [
        sys.executable,
        str(script),
        "--aggregate",
        "--config",
        str(Path(config_path).resolve()),
        "--output",
        str(ROOT / AGGREGATION_OUTPUT),
    ]
    return {
        "schema": "qaq-s11b-quality-pilot-plan-v1",
        "protocol_config": str(Path(config_path).resolve()),
        "protocol_config_sha256": digest,
        "mode_order": list(MODE_IDS),
        "request_order": [item["request_id"] for item in requests],
        "fresh_child_processes_per_mode": 1,
        "repeats_within_fresh_child": 2,
        "child_commands": child_commands,
        "mode_output_paths": [str(ROOT / OUTPUTS[mode_id]) for mode_id in MODE_IDS],
        "aggregation_command": aggregation_command,
        "aggregation_output_path": str(ROOT / AGGREGATION_OUTPUT),
        "required_environment_names": ["VIRTUAL_ENV", "QAQ_S07_ROUTER_CHECKPOINT"],
        "required_resources": [
            "pinned Qwen3-4B snapshot",
            "resident physically packed S03 artifact",
            "historical S07 router checkpoint",
            "pinned Any-Precision checkout",
            "one explicit CUDA device shared by both children",
        ],
        "model_loading": False,
        "cuda_activity": False,
        "pilot_execution": False,
        "result_write_activity": False,
    }


def expected_mode_destination(mode_id: str) -> Path:
    if mode_id not in OUTPUTS:
        raise LookaheadQualityError(f"unknown S11-B mode: {mode_id!r}")
    return ROOT / OUTPUTS[mode_id]


def validate_dispatch(
    *, mode_id: str, device: str, output: Path, config_path: Path = DEFAULT_CONFIG
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fail before the heavy import on any ambiguous execution request."""

    # Destination safety is intentionally checked before even reading production
    # resources.  This makes an absent/linked parent or any existing canonical
    # destination a hard pre-import stop.
    if mode_id not in MODE_IDS:
        raise LookaheadQualityError(f"unknown S11-B mode: {mode_id!r}")
    _validate_device(device)
    expected = _normal(expected_mode_destination(mode_id))
    if _normal(Path(output)) != expected:
        raise LookaheadQualityError(f"mode output must be the frozen destination: {expected}")
    validate_destination(
        Path(output),
        PersistencePolicy(expected_destination=expected, allowed_parent=expected.parent),
    )
    config, _ = load_protocol(config_path, require_results_absent=False)
    return config, _mode(config, mode_id)


def _expected_identities(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "model": dict(config["identities"]["model"]),
        "tokenizer": dict(config["identities"]["tokenizer"]),
        "packed_artifact": dict(config["identities"]["packed_artifact"]),
        "any_precision": dict(config["identities"]["any_precision"]),
        "router_checkpoint": {
            "sha256": config["identities"]["router_checkpoint"]["sha256"],
            "candidate_order": list(CANDIDATE_BITS),
            "metadata_validated": True,
            "read_only": True,
        },
        "fixed_inputs_path": config["identities"]["fixed_inputs_path"],
    }


def _expected_inputs(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "request_id": item["source_record_id"],
            "full_input_ids_sha256": item["token_digest_sha256"],
            "token_count": item["token_count"],
            "prompt_token_range": item["prompt_token_range"],
            "completion_token_range": item["completion_token_range"],
            "causal_completion_loss_logit_range": item["causal_completion_loss_logit_range"],
        }
        for item in config["fixed_inputs"]["requests"]
    ]


def _canonical_routes(route_maps: Any) -> dict[str, list[dict[str, Any]]]:
    _require(isinstance(route_maps, list) and len(route_maps) == 2, "two route maps required")
    result: dict[str, list[dict[str, Any]]] = {}
    for expected_id, item in zip(REQUEST_IDS, route_maps, strict=True):
        _require(isinstance(item, dict), "route-map record must be an object")
        _require(item.get("request_id") == expected_id, "route-map request order drifted")
        routes = item.get("routes")
        _require(isinstance(routes, list) and len(routes) == 72, "route map must have 72 units")
        expected_keys = [(layer, unit) for layer in range(36) for unit in UNIT_TYPES]
        actual_keys: list[tuple[Any, Any]] = []
        normalized: list[dict[str, Any]] = []
        for route in routes:
            _require(isinstance(route, dict), "route record must be an object")
            _require(
                tuple(route) == ("request_id", "target_layer", "unit_type", "selected_bits"),
                "route record fields/order drifted",
            )
            _require(route["request_id"] == expected_id, "route request identity drifted")
            layer = route["target_layer"]
            unit = route["unit_type"]
            bit = route["selected_bits"]
            _require(
                isinstance(layer, int) and not isinstance(layer, bool),
                "route layer must be an integer",
            )
            _require(unit in UNIT_TYPES, "route unit type is invalid")
            _require(bit in CANDIDATE_BITS and not isinstance(bit, bool), "route bit is invalid")
            actual_keys.append((layer, unit))
            normalized.append(dict(route))
        _require(actual_keys == expected_keys, "route order/coverage/uniqueness drifted")
        result[expected_id] = normalized
    return result


def _expected_provenance(mode_id: str, request_id: str, layer: int, unit: str) -> dict[str, Any]:
    if unit == "attention" and mode_id == MODE_IDS[1] and layer > 0:
        source_layer = layer - 1
        source_point = "post_attention_pre_ffn"
        timing = ROUTING_TIMINGS[1]
    else:
        source_layer = layer
        source_point = (
            "same_unit_pre_attention" if unit == "attention" else "post_attention_pre_ffn"
        )
        timing = ROUTING_TIMINGS[0] if mode_id == MODE_IDS[0] else ROUTING_TIMINGS[1]
    return {
        "request_id": request_id,
        "source_layer": source_layer,
        "target_layer": layer,
        "target_unit_type": unit,
        "source_point": source_point,
        "routing_timing": timing,
        "candidate_order": [4, 8],
    }


def _canonical_provenance(mode_id: str, value: Any) -> dict[str, list[dict[str, Any]]]:
    _require(isinstance(value, dict), "provenance must be an object")
    _require(
        value.get("routing_timing") == ROUTING_TIMINGS[MODE_IDS.index(mode_id)],
        "provenance timing drifted",
    )
    records_by_request = value.get("records_by_request")
    _require(
        isinstance(records_by_request, list) and len(records_by_request) == 2,
        "provenance must cover two requests",
    )
    result: dict[str, list[dict[str, Any]]] = {}
    for request_id, item in zip(REQUEST_IDS, records_by_request, strict=True):
        _require(
            isinstance(item, dict) and item.get("request_id") == request_id,
            "provenance request order drifted",
        )
        records = item.get("records")
        _require(isinstance(records, list) and len(records) == 72, "provenance must cover 72 units")
        expected = [
            _expected_provenance(mode_id, request_id, layer, unit)
            for layer in range(36)
            for unit in UNIT_TYPES
        ]
        _require(records == expected, "target-owned provenance drifted")
        result[request_id] = [dict(record) for record in records]
    return result


def _route_summaries(route_maps: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    def scope(request_id: str, unit: str | None) -> list[int]:
        return [
            item["selected_bits"]
            for item in route_maps[request_id]
            if unit is None or item["unit_type"] == unit
        ]

    def fractions(values: Sequence[int]) -> dict[str, float]:
        return {
            "fraction_4": values.count(4) / len(values),
            "fraction_8": values.count(8) / len(values),
        }

    return {
        "fractions_overall": [
            {"request_id": request_id, **fractions(scope(request_id, None))}
            for request_id in REQUEST_IDS
        ],
        "fractions_attention": [
            {"request_id": request_id, **fractions(scope(request_id, "attention"))}
            for request_id in REQUEST_IDS
        ],
        "fractions_ffn": [
            {"request_id": request_id, **fractions(scope(request_id, "ffn"))}
            for request_id in REQUEST_IDS
        ],
        "mean_selected_bit_width": [
            {
                "request_id": request_id,
                "value": sum(scope(request_id, None)) / 72,
            }
            for request_id in REQUEST_IDS
        ],
    }


def _historical_routes() -> dict[str, list[dict[str, Any]]]:
    result, _ = _load_json_bytes(ROOT / "docs/results/s07_router_training.json")
    records = result.get("evaluation", {}).get("hard", {}).get("route_logs")
    if not isinstance(records, list):
        raise LookaheadQualityError("historical S07 hard routes are unavailable")
    keyed: dict[tuple[str, int, str], int] = {}
    for item in records:
        if not isinstance(item, dict) or item.get("request_id") not in REQUEST_IDS:
            continue
        key = (item["request_id"], item["layer"], item["unit_type"])
        if key in keyed:
            raise LookaheadQualityError(f"duplicate historical route key: {key}")
        keyed[key] = item["hard_bit"]
    expected = {
        (request_id, layer, unit)
        for request_id in REQUEST_IDS
        for layer in range(36)
        for unit in UNIT_TYPES
    }
    if set(keyed) != expected:
        raise LookaheadQualityError("historical S07 route coverage drifted")
    return {
        request_id: [
            {
                "request_id": request_id,
                "target_layer": layer,
                "unit_type": unit,
                "selected_bits": keyed[(request_id, layer, unit)],
            }
            for layer in range(36)
            for unit in UNIT_TYPES
        ]
        for request_id in REQUEST_IDS
    }


def _validate_state_entries(entries: Any, *, component: str) -> list[dict[str, Any]]:
    _require(isinstance(entries, list) and entries, f"{component} state entries are missing")
    _require(
        entries == sorted(entries, key=lambda item: (item.get("name", ""), item.get("kind", ""))),
        f"{component} state entries are not sorted",
    )
    names: set[tuple[str, str]] = set()
    for item in entries:
        _require(isinstance(item, dict), f"{component} state entry is malformed")
        _require(
            tuple(item)
            == (
                "name",
                "kind",
                "dtype",
                "shape",
                "requires_grad",
                "gradient_absent",
                "value_sha256",
            ),
            f"{component} state entry fields/order drifted",
        )
        _require(isinstance(item["name"], str) and item["name"], "state name is invalid")
        _require(item["kind"] in ("parameter", "buffer"), "state kind is invalid")
        _require(isinstance(item["dtype"], str) and item["dtype"], "state dtype is invalid")
        _require(
            isinstance(item["shape"], list)
            and all(isinstance(value, int) and value >= 0 for value in item["shape"]),
            "state shape is invalid",
        )
        _require(item["requires_grad"] is False, "inference state must have requires_grad=false")
        _require(item["gradient_absent"] is True, "state gradient must be absent")
        _require(_is_hex64(item["value_sha256"]), "state value hash is invalid")
        key = (item["name"], item["kind"])
        _require(key not in names, "state entry is duplicated")
        names.add(key)
    return [dict(item) for item in entries]


def _validate_freeze_audit(value: Any) -> None:
    _require(isinstance(value, dict), "freeze audit must be an object")
    _require(
        tuple(value)
        == (
            "components",
            "before_hashes",
            "after_hashes",
            "hashes_equal",
            "optimizer_absent",
            "gradients_absent",
        ),
        "freeze audit fields/order drifted",
    )
    components = value["components"]
    expected_components = (
        "teacher",
        "packed_weights_and_buffers",
        "non_router_base",
        "router",
    )
    _require(
        isinstance(components, dict) and tuple(components) == expected_components,
        "freeze components drifted",
    )
    before_hashes: dict[str, str] = {}
    after_hashes: dict[str, str] = {}
    for component in expected_components:
        audit = components[component]
        _require(isinstance(audit, dict), f"{component} freeze audit is malformed")
        _require(
            tuple(audit)
            == (
                "before_entries",
                "after_entries",
                "parameter_count",
                "buffer_count",
                "before_aggregate_sha256",
                "after_aggregate_sha256",
                "hashes_equal",
            ),
            f"{component} freeze fields/order drifted",
        )
        before = _validate_state_entries(audit["before_entries"], component=component)
        after = _validate_state_entries(audit["after_entries"], component=component)
        parameter_count = sum(item["kind"] == "parameter" for item in before)
        buffer_count = sum(item["kind"] == "buffer" for item in before)
        _require(audit["parameter_count"] == parameter_count, "state parameter count drifted")
        _require(audit["buffer_count"] == buffer_count, "state buffer count drifted")
        before_hash = _digest(before)
        after_hash = _digest(after)
        _require(audit["before_aggregate_sha256"] == before_hash, "before state aggregate drifted")
        _require(audit["after_aggregate_sha256"] == after_hash, "after state aggregate drifted")
        _require(before == after and audit["hashes_equal"] is True, f"{component} state changed")
        before_hashes[component] = before_hash
        after_hashes[component] = after_hash
    _require(value["before_hashes"] == before_hashes, "before component hashes drifted")
    _require(value["after_hashes"] == after_hashes, "after component hashes drifted")
    _require(
        value["hashes_equal"] is True and before_hashes == after_hashes, "aggregate state changed"
    )
    _require(value["optimizer_absent"] is True, "optimizer must be absent")
    _require(value["gradients_absent"] is True, "gradients must be absent")


def _validate_raw_request(
    item: Any,
    *,
    request: Mapping[str, Any],
    mode_id: str,
    expected_routes: list[dict[str, Any]],
    expected_provenance: list[dict[str, Any]],
) -> None:
    _require(isinstance(item, dict), "repeat request evidence must be an object")
    expected_keys = (
        "request_id",
        "full_input_ids_sha256",
        "teacher_logits_digest",
        "student_logits_digest",
        "teacher_logits_shape",
        "student_logits_shape",
        "finite_teacher_logits",
        "finite_student_logits",
        "kl",
        "mean_absolute_logit_error",
        "maximum_absolute_logit_error",
        "routes",
        "provenance",
        "request_cleanup",
    )
    _require(tuple(item) == expected_keys, "repeat request fields/order drifted")
    _require(item["request_id"] == request["source_record_id"], "repeat request ID drifted")
    _require(
        item["full_input_ids_sha256"] == request["token_digest_sha256"],
        "repeat input digest drifted",
    )
    for field in ("teacher_logits_digest", "student_logits_digest"):
        _require(_is_hex64(item[field]), f"{field} is invalid")
    for field in ("teacher_logits_shape", "student_logits_shape"):
        shape = item[field]
        _require(
            isinstance(shape, list)
            and len(shape) == 3
            and shape[0] == 1
            and shape[1] == 64
            and isinstance(shape[2], int)
            and shape[2] > 0,
            f"{field} is invalid",
        )
    _require(
        item["teacher_logits_shape"] == item["student_logits_shape"],
        "teacher/student logit shape mismatch",
    )
    _require(item["finite_teacher_logits"] is True, "teacher logits are non-finite")
    _require(item["finite_student_logits"] is True, "student logits are non-finite")
    for field in ("kl", "mean_absolute_logit_error", "maximum_absolute_logit_error"):
        _require(_finite_nonnegative(item[field]), f"metric is invalid: {field}")
    _require(item["routes"] == expected_routes, "repeat raw routes drifted")
    _require(item["provenance"] == expected_provenance, "repeat raw provenance drifted")
    cleanup = item["request_cleanup"]
    _require(
        cleanup
        == {
            "state_ended": True,
            "routes_released": True,
            "features_released": True,
            "probabilities_released": True,
            "provenance_released": True,
            "passed": True,
        },
        "request cleanup evidence failed",
    )


def validate_mode_result(
    result: dict[str, Any],
    config: Mapping[str, Any],
    *,
    historical_routes: Mapping[str, list[dict[str, Any]]] | None = None,
) -> None:
    """Validate one complete per-mode result without trusting summaries."""

    try:
        _assert_no_prohibited_data_fields(result)
        _require(tuple(result) == PER_MODE_KEYS, "per-mode top-level schema fields/order drifted")
        _require(result["schema"] == MODE_SCHEMA, "unsupported per-mode result schema")
        mode_id = result["mode_id"]
        mode = _mode(config, mode_id)
        _require(
            result["protocol_config_sha256"] == EXPECTED_CONFIG_SHA256, "protocol hash drifted"
        )
        _require(result["identities"] == _expected_identities(config), "frozen identities drifted")
        hardware = result["hardware"]
        _require(
            isinstance(hardware, dict)
            and tuple(hardware)
            == (
                "cuda_device",
                "device_index",
                "gpu_model",
                "driver_version",
                "cuda_runtime_version",
                "pytorch_version",
                "transformers_version",
                "python_version",
            ),
            "hardware/software identity fields drifted",
        )
        _validate_device(hardware["cuda_device"])
        _require(
            hardware["device_index"] == int(hardware["cuda_device"].split(":")[1]),
            "CUDA device index mismatch",
        )
        for field in (
            "gpu_model",
            "driver_version",
            "cuda_runtime_version",
            "pytorch_version",
            "transformers_version",
            "python_version",
        ):
            _require(
                isinstance(hardware[field], str) and hardware[field],
                f"hardware field is missing: {field}",
            )
        _require(
            hardware["gpu_model"] == "NVIDIA GeForce RTX 3090",
            "frozen comparable GPU identity drifted",
        )
        _require(result["seed"] == 1729, "seed drifted")
        _require(result["inputs"] == _expected_inputs(config), "fixed input evidence drifted")

        route_maps = _canonical_routes(result["routes"].get("target_owned_route_maps"))
        summaries = _route_summaries(route_maps)
        _require(
            result["routes"].get("fractions_overall") == summaries["fractions_overall"],
            "overall route summary drifted",
        )
        _require(
            result["routes"].get("fractions_attention") == summaries["fractions_attention"],
            "attention route summary drifted",
        )
        _require(
            result["routes"].get("fractions_ffn") == summaries["fractions_ffn"],
            "FFN route summary drifted",
        )
        _require(
            result["routes"].get("mean_selected_bit_width") == summaries["mean_selected_bit_width"],
            "mean selected width drifted",
        )
        history = dict(historical_routes) if historical_routes is not None else _historical_routes()
        control_equal = mode_id != MODE_IDS[0] or all(
            route_maps[request_id] == history[request_id] for request_id in REQUEST_IDS
        )
        _require(
            result["routes"].get("historical_control_equality") is control_equal,
            "historical control equality summary drifted",
        )
        _require(control_equal, "same-unit control differs from historical S07 keyed routes")
        provenance = _canonical_provenance(mode_id, result["provenance"])

        repeats = result["repeats"]
        _require(isinstance(repeats, list) and len(repeats) == 2, "exactly two repeats required")
        expected_input_digests = [
            item["token_digest_sha256"] for item in config["fixed_inputs"]["requests"]
        ]
        for repeat_index, repeat in enumerate(repeats):
            _require(isinstance(repeat, dict), "repeat evidence is malformed")
            _require(
                tuple(repeat)
                == (
                    "repeat_index",
                    "input_digests",
                    "logits_digest",
                    "route_map_digest",
                    "provenance_digest",
                    "finite_logits",
                    "finite_metrics",
                    "requests",
                ),
                "repeat fields/order drifted",
            )
            _require(repeat["repeat_index"] == repeat_index, "repeat order drifted")
            _require(
                repeat["input_digests"] == expected_input_digests, "repeat input digests drifted"
            )
            raw_requests = repeat["requests"]
            _require(
                isinstance(raw_requests, list) and len(raw_requests) == 2,
                "repeat requests are incomplete",
            )
            for request, raw in zip(config["fixed_inputs"]["requests"], raw_requests, strict=True):
                request_id = request["source_record_id"]
                _validate_raw_request(
                    raw,
                    request=request,
                    mode_id=mode_id,
                    expected_routes=route_maps[request_id],
                    expected_provenance=provenance[request_id],
                )
            expected_logits_digest = _digest(
                [
                    [item["teacher_logits_digest"], item["student_logits_digest"]]
                    for item in raw_requests
                ]
            )
            _require(
                repeat["logits_digest"] == expected_logits_digest, "repeat logits digest drifted"
            )
            _require(
                repeat["route_map_digest"] == _digest(result["routes"]["target_owned_route_maps"]),
                "repeat route digest drifted",
            )
            _require(
                repeat["provenance_digest"] == _digest(result["provenance"]),
                "repeat provenance digest drifted",
            )
            _require(
                repeat["finite_logits"] is True and repeat["finite_metrics"] is True,
                "repeat finite audit failed",
            )
        comparable_first = [
            {key: value for key, value in item.items() if key != "request_cleanup"}
            for item in repeats[0]["requests"]
        ]
        comparable_second = [
            {key: value for key, value in item.items() if key != "request_cleanup"}
            for item in repeats[1]["requests"]
        ]
        _require(
            comparable_first == comparable_second,
            "repeat logits/metrics/routes/provenance are not deterministic",
        )

        quality = result["quality"]
        _require(
            isinstance(quality, dict)
            and tuple(quality)
            == (
                "per_request",
                "aggregate_kl",
                "aggregate_mean_absolute_logit_error",
                "aggregate_maximum_absolute_logit_error",
                "all_finite",
            ),
            "quality fields/order drifted",
        )
        expected_per_request = [
            {
                "request_id": item["request_id"],
                "kl": item["kl"],
                "mean_absolute_logit_error": item["mean_absolute_logit_error"],
                "maximum_absolute_logit_error": item["maximum_absolute_logit_error"],
            }
            for item in repeats[0]["requests"]
        ]
        _require(
            quality["per_request"] == expected_per_request, "per-request quality summary drifted"
        )
        for field in ("kl", "mean_absolute_logit_error", "maximum_absolute_logit_error"):
            aggregate_field = "aggregate_" + field
            expected = sum(item[field] for item in expected_per_request) / 2
            _require(
                quality[aggregate_field] == expected,
                f"quality aggregate drifted: {aggregate_field}",
            )
        _require(quality["all_finite"] is True, "quality finite audit failed")
        _validate_freeze_audit(result["freeze_audit"])

        audit = result["prohibited_work_audit"]
        _require(
            audit
            == {
                "execution": {
                    "routing_timing": mode["routing_timing"],
                    "execution": "resident_physically_packed_hard_4_8",
                    "candidate_order": [4, 8],
                    "fresh_process": True,
                    "fresh_child_processes_per_mode": 1,
                    "repeats_within_fresh_child": 2,
                    "full_teacher_forced_forward": True,
                    "phase": "prefill",
                    "use_cache": False,
                    "batch_size": 1,
                    "sequence_length": 64,
                    "prompt_only_mask": True,
                    "inference_only": True,
                    "runtime_tokenization": False,
                    "dataset_access": False,
                },
                "training_or_retraining_observed": False,
                "optimizer_present": False,
                "gradient_work_observed": False,
                "checkpoint_created": False,
                "on_demand_loading_observed": False,
                "generation_observed": False,
                "decode_observed": False,
                "perplexity_observed": False,
                "performance_or_resource_measurement_observed": False,
                "async_prefetch_cache_batch_schedule_observed": False,
                "all_requests_cleaned": True,
                "evidence_label": audit.get("evidence_label"),
                "passed": True,
            },
            "execution/prohibited-work audit drifted",
        )
        _require(
            audit["evidence_label"]
            in ("production pilot evidence", "test-only structural evidence"),
            "evidence label is invalid",
        )
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, InvalidEvidence):
            raise
        raise InvalidEvidence(str(exc)) from exc


def _repeat_record(
    raw_requests: list[dict[str, Any]], route_payload: Any, provenance: Any, index: int
) -> dict[str, Any]:
    return {
        "repeat_index": index,
        "input_digests": [item["full_input_ids_sha256"] for item in raw_requests],
        "logits_digest": _digest(
            [
                [item["teacher_logits_digest"], item["student_logits_digest"]]
                for item in raw_requests
            ]
        ),
        "route_map_digest": _digest(route_payload),
        "provenance_digest": _digest(provenance),
        "finite_logits": all(
            item["finite_teacher_logits"] and item["finite_student_logits"] for item in raw_requests
        ),
        "finite_metrics": all(
            _finite_nonnegative(item[field])
            for item in raw_requests
            for field in ("kl", "mean_absolute_logit_error", "maximum_absolute_logit_error")
        ),
        "requests": raw_requests,
    }


def execute_mode_with_runtime(
    runtime: LookaheadQualityRuntime,
    *,
    config: Mapping[str, Any],
    mode_id: str,
    device: str,
) -> dict[str, Any]:
    """Run the frozen two-repeat schedule through an injected runtime."""

    mode = _mode(config, mode_id)
    _validate_device(device)
    requests = _fixed_requests(config)
    try:
        runtime.prepare(config, mode, device, requests)
        repeat_requests: list[list[dict[str, Any]]] = []
        for repeat_index in range(2):
            current: list[dict[str, Any]] = []
            for request in requests:
                raw = runtime.run_request(
                    mode=mode,
                    request=request,
                    repeat_index=repeat_index,
                    device=device,
                )
                if not isinstance(raw, dict):
                    raise LookaheadQualityError("runtime request evidence is missing")
                current.append(raw)
            repeat_requests.append(current)
        route_payload = [
            {
                "request_id": request_id,
                "routes": repeat_requests[0][index]["routes"],
            }
            for index, request_id in enumerate(REQUEST_IDS)
        ]
        provenance = {
            "routing_timing": mode["routing_timing"],
            "records_by_request": [
                {
                    "request_id": request_id,
                    "records": repeat_requests[0][index]["provenance"],
                }
                for index, request_id in enumerate(REQUEST_IDS)
            ],
        }
        route_maps = _canonical_routes(route_payload)
        summaries = _route_summaries(route_maps)
        history = _historical_routes()
        historical_equal = mode_id != MODE_IDS[0] or all(
            route_maps[request_id] == history[request_id] for request_id in REQUEST_IDS
        )
        quality_records = [
            {
                "request_id": item["request_id"],
                "kl": item["kl"],
                "mean_absolute_logit_error": item["mean_absolute_logit_error"],
                "maximum_absolute_logit_error": item["maximum_absolute_logit_error"],
            }
            for item in repeat_requests[0]
        ]
        result = {
            "schema": MODE_SCHEMA,
            "mode_id": mode_id,
            "protocol_config_sha256": EXPECTED_CONFIG_SHA256,
            "identities": runtime.identity_evidence(),
            "hardware": runtime.hardware_evidence(),
            "seed": config["execution_contract"]["seed"],
            "inputs": _expected_inputs(config),
            "repeats": [
                _repeat_record(repeat_requests[index], route_payload, provenance, index)
                for index in range(2)
            ],
            "quality": {
                "per_request": quality_records,
                "aggregate_kl": sum(item["kl"] for item in quality_records) / 2,
                "aggregate_mean_absolute_logit_error": sum(
                    item["mean_absolute_logit_error"] for item in quality_records
                )
                / 2,
                "aggregate_maximum_absolute_logit_error": sum(
                    item["maximum_absolute_logit_error"] for item in quality_records
                )
                / 2,
                "all_finite": True,
            },
            "routes": {
                "target_owned_route_maps": route_payload,
                **summaries,
                "historical_control_equality": historical_equal,
            },
            "provenance": provenance,
            "freeze_audit": runtime.freeze_audit(),
            "prohibited_work_audit": {
                "execution": {
                    "routing_timing": mode["routing_timing"],
                    "execution": "resident_physically_packed_hard_4_8",
                    "candidate_order": [4, 8],
                    "fresh_process": True,
                    "fresh_child_processes_per_mode": 1,
                    "repeats_within_fresh_child": 2,
                    "full_teacher_forced_forward": True,
                    "phase": "prefill",
                    "use_cache": False,
                    "batch_size": 1,
                    "sequence_length": 64,
                    "prompt_only_mask": True,
                    "inference_only": True,
                    "runtime_tokenization": False,
                    "dataset_access": False,
                },
                "training_or_retraining_observed": False,
                "optimizer_present": False,
                "gradient_work_observed": False,
                "checkpoint_created": False,
                "on_demand_loading_observed": False,
                "generation_observed": False,
                "decode_observed": False,
                "perplexity_observed": False,
                "performance_or_resource_measurement_observed": False,
                "async_prefetch_cache_batch_schedule_observed": False,
                "all_requests_cleaned": True,
                "evidence_label": runtime.evidence_label,
                "passed": True,
            },
        }
        validate_mode_result(result, config)
        return result
    finally:
        runtime.close()


def _distance(
    left: Sequence[dict[str, Any]], right: Sequence[dict[str, Any]], unit: str | None
) -> float:
    pairs = [
        (a, b) for a, b in zip(left, right, strict=True) if unit is None or a["unit_type"] == unit
    ]
    return sum(a["selected_bits"] != b["selected_bits"] for a, b in pairs) / len(pairs)


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0.0:
        return 1.0 if numerator == 0.0 else None
    return numerator / denominator


def build_aggregation(
    control: dict[str, Any], treatment: dict[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    """Independently validate and recompute every paired aggregate."""

    validate_mode_result(control, config)
    validate_mode_result(treatment, config)
    _require(control["mode_id"] == MODE_IDS[0], "control result is not first frozen mode")
    _require(treatment["mode_id"] == MODE_IDS[1], "treatment result is not second frozen mode")
    for field in ("protocol_config_sha256", "identities", "hardware", "seed", "inputs"):
        _require(control[field] == treatment[field], f"cross-mode identity mismatch: {field}")
    control_routes = _canonical_routes(control["routes"]["target_owned_route_maps"])
    treatment_routes = _canonical_routes(treatment["routes"]["target_owned_route_maps"])
    control_quality = {item["request_id"]: item for item in control["quality"]["per_request"]}
    treatment_quality = {item["request_id"]: item for item in treatment["quality"]["per_request"]}
    thresholds = config["interpretation"]["quality_thresholds"]
    paired_request = []
    request_checks = []
    for request_id in REQUEST_IDS:
        control_kl = control_quality[request_id]["kl"]
        treatment_kl = treatment_quality[request_id]["kl"]
        limit = thresholds["treatment_each_request_kl_max_paired_control_factor"] * control_kl
        passed = treatment_kl <= limit
        request_checks.append(passed)
        paired_request.append(
            {
                "request_id": request_id,
                "control_kl": control_kl,
                "treatment_kl": treatment_kl,
                "ratio": _safe_ratio(treatment_kl, control_kl),
                "limit": limit,
                "passed": passed,
            }
        )
    control_kl = control["quality"]["aggregate_kl"]
    treatment_kl = treatment["quality"]["aggregate_kl"]
    control_mae = control["quality"]["aggregate_mean_absolute_logit_error"]
    treatment_mae = treatment["quality"]["aggregate_mean_absolute_logit_error"]
    aggregate_kl_passed = (
        treatment_kl <= thresholds["treatment_aggregate_kl_max_control_factor"] * control_kl
    )
    aggregate_mae_passed = (
        treatment_mae
        <= thresholds["treatment_aggregate_mean_absolute_logit_error_max_control_factor"]
        * control_mae
    )
    quality_passed = aggregate_kl_passed and aggregate_mae_passed and all(request_checks)
    distances = {scope: [] for scope in ("overall", "attention", "ffn")}
    changed: list[dict[str, Any]] = []
    layer_zero_equal = True
    for request_id in REQUEST_IDS:
        left = control_routes[request_id]
        right = treatment_routes[request_id]
        distances["overall"].append(
            {"request_id": request_id, "distance": _distance(left, right, None)}
        )
        distances["attention"].append(
            {"request_id": request_id, "distance": _distance(left, right, "attention")}
        )
        distances["ffn"].append(
            {"request_id": request_id, "distance": _distance(left, right, "ffn")}
        )
        for a, b in zip(left, right, strict=True):
            if a["target_layer"] == 0 and a["unit_type"] in UNIT_TYPES:
                layer_zero_equal &= a["selected_bits"] == b["selected_bits"]
            if a["selected_bits"] != b["selected_bits"]:
                changed.append(
                    {
                        "request_id": request_id,
                        "target_layer": a["target_layer"],
                        "unit_type": a["unit_type"],
                        "control_bits": a["selected_bits"],
                        "treatment_bits": b["selected_bits"],
                    }
                )
    _require(layer_zero_equal, "paired layer-0 attention/FFN equality failed")
    teacher_equal = all(
        control["repeats"][0]["requests"][index]["teacher_logits_digest"]
        == treatment["repeats"][0]["requests"][index]["teacher_logits_digest"]
        for index in range(2)
    )
    _require(teacher_equal, "teacher logits differ across paired modes")
    classification = (
        "ADVANCE_TO_BROADER_QUALITY_CHECK" if quality_passed else "CHECKPOINT_REUSE_DEGRADES"
    )
    return {
        "schema": AGGREGATION_SCHEMA,
        "protocol_config_sha256": EXPECTED_CONFIG_SHA256,
        "mode_result_paths": [OUTPUTS[mode_id] for mode_id in MODE_IDS],
        "paired_inputs": {
            "request_order": list(REQUEST_IDS),
            "inputs_identical": True,
            "device_and_software_identical": True,
            "teacher_identity_and_logits_identical": teacher_equal,
        },
        "paired_quality": {
            "control_aggregate_kl": control_kl,
            "treatment_aggregate_kl": treatment_kl,
            "aggregate_kl_ratio": _safe_ratio(treatment_kl, control_kl),
            "paired_request_kl": paired_request,
            "control_aggregate_mean_absolute_logit_error": control_mae,
            "treatment_aggregate_mean_absolute_logit_error": treatment_mae,
            "aggregate_mean_absolute_logit_error_ratio": _safe_ratio(treatment_mae, control_mae),
            "threshold_checks": {
                "aggregate_kl_passed": aggregate_kl_passed,
                "each_request_kl_passed": all(request_checks),
                "aggregate_mean_absolute_logit_error_passed": aggregate_mae_passed,
                "all_quality_thresholds_passed": quality_passed,
            },
        },
        "route_comparison": {
            "paired_distances_overall": distances["overall"],
            "paired_distances_attention": distances["attention"],
            "paired_distances_ffn": distances["ffn"],
            "changed_target_units": changed,
            "changed_target_unit_count": len(changed),
            "layer_0_equal": layer_zero_equal,
            "later_differences_recorded": True,
        },
        "determinism": {
            "control_repeats_validated": True,
            "treatment_repeats_validated": True,
            "teacher_logits_equal_across_modes": teacher_equal,
        },
        "freeze_audits": {"control_passed": True, "treatment_passed": True},
        "result_checks": {
            "independent_per_mode_validation": True,
            "raw_summaries_recomputed": True,
            "prohibited_fields_absent": True,
            "complete_evidence": True,
        },
        "classification": classification,
        "errors": [],
    }


def validate_aggregation_result(
    aggregate: dict[str, Any],
    control: dict[str, Any],
    treatment: dict[str, Any],
    config: Mapping[str, Any],
) -> None:
    _require(tuple(aggregate) == AGGREGATION_KEYS, "aggregation top-level fields/order drifted")
    expected = build_aggregation(control, treatment, config)
    _require(aggregate == expected, "aggregation differs from independently recomputed evidence")


def aggregate_paths(
    *,
    config_path: Path = DEFAULT_CONFIG,
    control_path: Path | None = None,
    treatment_path: Path | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Load paired external results, returning PAUSE for absence and INVALID otherwise."""

    config, _ = load_protocol(config_path, require_results_absent=False)
    paths = (
        Path(control_path) if control_path is not None else ROOT / OUTPUTS[MODE_IDS[0]],
        Path(treatment_path) if treatment_path is not None else ROOT / OUTPUTS[MODE_IDS[1]],
    )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        return None, {
            "classification": "PAUSE",
            "errors": [f"missing required external result: {path}" for path in missing],
        }
    try:
        control, _ = _load_json_bytes(paths[0])
        treatment, _ = _load_json_bytes(paths[1])
        aggregate = build_aggregation(control, treatment, config)
        return aggregate, {"classification": aggregate["classification"], "errors": []}
    except (InvalidEvidence, LookaheadQualityError, KeyError, TypeError, ValueError) as exc:
        return None, {"classification": "INVALID_EVIDENCE", "errors": [str(exc)]}


def validate_destination(destination: Path, policy: PersistencePolicy) -> Path:
    """Reject mismatched, existing, linked, or unsafe destinations without mutation."""

    destination = _normal(Path(destination))
    expected = _normal(policy.expected_destination)
    parent = _normal(policy.allowed_parent)
    if destination != expected or destination.parent != parent:
        raise LookaheadQualityError(f"destination is not the exact allowed output: {destination}")
    if parent.is_symlink():
        raise LookaheadQualityError(f"allowed destination parent is a symlink: {parent}")
    if not parent.exists():
        raise LookaheadQualityError(f"allowed destination parent is absent: {parent}")
    if not parent.is_dir():
        raise LookaheadQualityError(f"allowed destination parent is not a directory: {parent}")
    if destination.is_symlink():
        raise LookaheadQualityError(f"destination is a symlink; refusing overwrite: {destination}")
    if destination.is_file():
        raise LookaheadQualityError(
            f"destination is an existing file; refusing overwrite: {destination}"
        )
    if destination.is_dir():
        raise LookaheadQualityError(
            f"destination is an existing directory; refusing overwrite: {destination}"
        )
    if os.path.lexists(destination):
        raise LookaheadQualityError(
            f"destination already exists; refusing overwrite: {destination}"
        )
    return destination


def _serialize_result(result: dict[str, Any]) -> bytes:
    return (json.dumps(result, indent=2, sort_keys=False, allow_nan=False) + "\n").encode()


def persist_atomically(
    result: dict[str, Any],
    destination: Path,
    *,
    policy: PersistencePolicy,
    validator: Callable[[dict[str, Any]], None],
) -> str:
    """Persist validated JSON through the shared same-directory no-overwrite boundary."""

    destination = validate_destination(destination, policy)
    validator(result)
    temporary: Path | None = None
    try:
        descriptor, name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent)
        )
        temporary = Path(name)
        payload = _serialize_result(result)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        reloaded, raw = _load_json_bytes(temporary)
        if raw != payload:
            raise LookaheadQualityError("serialized result bytes changed on reread")
        validator(reloaded)
        validate_destination(destination, policy)
        os.link(temporary, destination)
        promoted = destination.read_bytes()
        if promoted != payload:
            raise LookaheadQualityError("promoted result bytes differ from validated bytes")
        digest = hashlib.sha256(promoted).hexdigest()
        if digest != hashlib.sha256(payload).hexdigest():
            raise LookaheadQualityError("promoted result SHA-256 differs from validated bytes")
        temporary.unlink()
        temporary = None
        directory = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return digest
    except FileExistsError as exc:
        raise LookaheadQualityError(
            f"destination appeared during no-overwrite promotion: {destination}"
        ) from exc
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def persist_validated_result(
    result: dict[str, Any],
    destination: Path,
    *,
    policy: PersistencePolicy,
    config: Mapping[str, Any],
    kind: str,
    paired_results: tuple[dict[str, Any], dict[str, Any]] | None = None,
) -> str:
    """Persist a validated S11-B result through the shared atomic boundary."""

    if kind == "mode":
        validator = lambda value: validate_mode_result(value, config)
    elif kind == "aggregation":
        if paired_results is None:
            raise LookaheadQualityError(
                "aggregation persistence requires both validated mode results"
            )
        validator = lambda value: validate_aggregation_result(
            value, paired_results[0], paired_results[1], config
        )
    else:
        raise LookaheadQualityError(f"unknown result kind: {kind}")
    return persist_atomically(
        result,
        destination,
        policy=policy,
        validator=validator,
    )


__all__ = [
    "AGGREGATION_OUTPUT",
    "AGGREGATION_SCHEMA",
    "CLASSIFICATIONS",
    "DEFAULT_CONFIG",
    "EXPECTED_CONFIG_SHA256",
    "MODE_IDS",
    "MODE_SCHEMA",
    "OUTPUTS",
    "REQUEST_IDS",
    "InvalidEvidence",
    "LookaheadQualityError",
    "LookaheadQualityRuntime",
    "MissingExternalResult",
    "PersistencePolicy",
    "aggregate_paths",
    "build_aggregation",
    "execute_mode_with_runtime",
    "expected_mode_destination",
    "load_protocol",
    "persist_atomically",
    "persist_validated_result",
    "plan",
    "validate_aggregation_result",
    "validate_destination",
    "validate_dispatch",
    "validate_mode_result",
]
