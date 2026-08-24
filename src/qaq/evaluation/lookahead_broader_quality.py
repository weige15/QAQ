"""S11-C2 structural contract, scheduler, validation, aggregation, and persistence.

The default plan is standard-library-only and non-executing.  The production
runtime is the existing S11-B implementation and is imported only by the thin
command after exact dispatch validation.
"""

from __future__ import annotations

import hashlib
import os
import struct
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from qaq.evaluation import lookahead_quality_runner as shared

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "configs/lookahead_broader_quality.json"
FIXTURE_PATH = ROOT / "configs/lookahead_broader_quality_inputs.json"
EXPECTED_CONFIG_SHA256 = "320c42901046d26c310d97fe1d3331d8653ce7c913daf3bff0bab7df02e585b5"
EXPECTED_FIXTURE_SHA256 = "a33cb9a7373f6ed68216e31249317ee35f25dc86d1e095b6428843671e8f3a08"
EXPECTED_S10_RESULT_SHA256 = "7d9e0aff3b686570be0d1d57b5513ee921d60bd5470f275b0cd7cbb4fd63db20"
EXPECTED_S11B_CONTROL_SHA256 = "ba748dd09b8319c1ff395f65be130ecbb0bea1571c1afb76e0016a88b6e5a073"
MODE_SCHEMA = "qaq-s11c-broader-quality-mode-result-v1"
AGGREGATION_SCHEMA = "qaq-s11c-broader-quality-aggregation-v1"
PLAN_SCHEMA = "qaq-s11c-broader-quality-plan-v1"
MODE_IDS = shared.MODE_IDS
ROUTING_TIMINGS = shared.ROUTING_TIMINGS
UNIT_TYPES = shared.UNIT_TYPES
CANDIDATE_BITS = shared.CANDIDATE_BITS
REQUEST_IDS = (
    "validation-3",
    "validation-270",
    "validation-500",
    "validation-761",
    "validation-1000",
    "validation-1252",
    "validation-1500",
    "validation-1759",
    "validation-2000",
    "validation-2250",
    "validation-2500",
    "validation-2755",
)
SOURCE_OFFSETS = tuple(range(0, 3000, 250))
SOURCE_ROWS = (3, 270, 500, 761, 1000, 1252, 1500, 1759, 2000, 2250, 2500, 2755)
TOKEN_DIGESTS = (
    "bbd7a25c172570f90d29d6fff0efc65975139ab7d65bb22409e87d10094f404b",
    "23e957c1cb5713a17c5332c2fd2bcb080c8d752d29cb51a5acb436fc8842f604",
    "dfe59fb1e0c1689410f5295037850be536ab56ce60dbbde4c8b6430969004b79",
    "d816c84ccd24ae11ca9a8124dd92130699393338847f900aa6cf46c7368c871b",
    "99c0183a064c79daea4cb461de16ddeb2144dbbe2af64b375f6f2088bb6e659e",
    "84ccbabf826875b899036c663e07080b558d1c0f047268b860e96da8a1bf7d17",
    "735f2670539c002602b9e7500a4288e1393f91bd7e8cb8617d3b8f34ba625d5c",
    "e6c401f8f0dad17504e55c4a9db5c2436a213786a5824c56f54826b4f1a8febc",
    "37f62b98ee3bc4466d0cbf64866d8ae6bc27a0cf723321aa5247cfd93bf703be",
    "6f6bf29d4ec5962df94dd58cadb51ea9a8eea484e4aa9f22c069f2aa2ed26378",
    "360df7b59d764cda62bad990346db508d7bfbe059a5bb0d2593fcf3a5540d4b8",
    "4dad3d315098a80c4d31a6198a7b120ff7cec5af66e481609a69b3adf0b659a4",
)
OUTPUTS = {
    MODE_IDS[0]: "docs/results/s11c_broader_quality/same_unit_control.json",
    MODE_IDS[1]: "docs/results/s11c_broader_quality/lookahead_attention_one_unit_treatment.json",
}
AGGREGATION_OUTPUT = "docs/results/s11c_broader_quality/aggregation.json"
OVERLAP_IDS = ("validation-3", "validation-1000")


class BroaderQualityError(shared.LookaheadQualityError):
    """A frozen S11-C protocol, evidence, dispatch, or persistence defect."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BroaderQualityError(message)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _token_digest(values: Sequence[int]) -> str:
    return hashlib.sha256(struct.pack("<" + "q" * len(values), *values)).hexdigest()


def _read(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        return shared._load_json_bytes(path)
    except shared.LookaheadQualityError as exc:
        raise BroaderQualityError(str(exc)) from exc


def _validate_frozen_sources(config: Mapping[str, Any], fixture: Mapping[str, Any]) -> None:
    _require(config["schema"] == "qaq-s11c-broader-quality-v1", "protocol schema drifted")
    _require(config["stage"] == "S11-C1", "protocol stage drifted")
    _require(tuple(item["id"] for item in config["modes"]) == MODE_IDS, "mode order drifted")
    _require(
        tuple(item["routing_timing"] for item in config["modes"]) == ROUTING_TIMINGS,
        "routing timing drifted",
    )
    _require(config["execution_contract"]["seed"] == 1729, "seed drifted")
    _require(
        config["execution_contract"]["repeats_within_fresh_child"] == 2, "repeat count drifted"
    )
    _require(config["modes"][0]["candidate_order"] == [4, 8], "candidate bits drifted")
    fixed = config["fixed_inputs"]
    _require(tuple(fixed["request_order"]) == REQUEST_IDS, "request order drifted")
    _require(
        tuple(item["source_offset"] for item in fixed["requests"]) == SOURCE_OFFSETS,
        "source offsets drifted",
    )
    _require(
        tuple(item["source_row"] for item in fixed["requests"]) == SOURCE_ROWS,
        "source rows drifted",
    )
    _require(
        tuple(item["token_digest_sha256"] for item in fixed["requests"]) == TOKEN_DIGESTS,
        "token digests drifted",
    )
    thresholds = config["interpretation"]["quality_thresholds"]
    _require(
        thresholds
        == {
            "treatment_aggregate_kl_max_control_factor": 1.1,
            "treatment_each_request_kl_max_paired_control_factor": 1.25,
            "treatment_aggregate_mean_absolute_logit_error_max_control_factor": 1.1,
            "implementation_choices_not_paper_facts": True,
        },
        "quality thresholds drifted",
    )
    _require(
        config["interpretation"]["precedence"] == ["PAUSE", "REVISE", "STOP", "CONTINUE"],
        "classification precedence drifted",
    )
    planned = config["planned_results"]
    _require(
        tuple(item["path"] for item in planned["mode_outputs"]) == tuple(OUTPUTS.values()),
        "mode outputs drifted",
    )
    _require(planned["aggregation_output"] == AGGREGATION_OUTPUT, "aggregation output drifted")

    _require(fixture["schema"] == "qaq-s11c-broader-quality-inputs-v1", "fixture schema drifted")
    records = fixture.get("requests")
    _require(
        isinstance(records, list) and len(records) == 12, "fixture must contain twelve requests"
    )
    for index, (record, request_id, offset, row, digest) in enumerate(
        zip(records, REQUEST_IDS, SOURCE_OFFSETS, SOURCE_ROWS, TOKEN_DIGESTS, strict=True)
    ):
        _require(record["request_id"] == request_id, f"fixture request order drifted at {index}")
        _require(
            record["source_offset"] == offset and record["source_row"] == row,
            f"fixture source identity drifted: {request_id}",
        )
        values = record.get("full_input_ids")
        _require(
            isinstance(values, list) and len(values) == 64,
            f"fixture token count drifted: {request_id}",
        )
        _require(
            all(isinstance(value, int) and not isinstance(value, bool) for value in values),
            f"fixture token type drifted: {request_id}",
        )
        _require(
            _token_digest(values) == digest == record["full_input_ids_sha256"],
            f"fixture token digest drifted: {request_id}",
        )
        _require(record["prompt_token_range"] == [0, 32], "prompt range drifted")
        _require(record["completion_token_range"] == [32, 64], "completion range drifted")
        _require(record["causal_completion_loss_logit_range"] == [31, 63], "causal range drifted")

    s10_path = ROOT / "docs/results/s10h_broader_validation.json"
    _require(_sha256(s10_path) == EXPECTED_S10_RESULT_SHA256, "authoritative S10 result drifted")
    s10, _ = _read(s10_path)
    manifest = s10["dataset"]["validation_manifest"]
    _require(
        tuple(item["example_id"] for item in manifest) == REQUEST_IDS, "S10 request order disagrees"
    )
    _require(tuple(item["source_row"] for item in manifest) == SOURCE_ROWS, "S10 rows disagree")
    _require(
        tuple(item["source_offset"] for item in manifest) == SOURCE_OFFSETS, "S10 offsets disagree"
    )
    _require(
        tuple(item["input_ids_sha256"] for item in manifest) == TOKEN_DIGESTS,
        "S10 token digests disagree",
    )
    b_path = ROOT / "docs/results/s11b_quality_pilot/same_unit_control.json"
    _require(_sha256(b_path) == EXPECTED_S11B_CONTROL_SHA256, "canonical S11-B control drifted")


def load_protocol(
    config_path: Path = DEFAULT_CONFIG, *, require_results_absent: bool = False
) -> tuple[dict[str, Any], str]:
    config_path = Path(config_path)
    _require(
        config_path.resolve() == DEFAULT_CONFIG.resolve(),
        "only the frozen S11-C config is accepted",
    )
    config, raw = _read(config_path)
    digest = hashlib.sha256(raw).hexdigest()
    _require(digest == EXPECTED_CONFIG_SHA256, "frozen S11-C config SHA-256 mismatch")
    fixture, fixture_raw = _read(FIXTURE_PATH)
    _require(
        hashlib.sha256(fixture_raw).hexdigest() == EXPECTED_FIXTURE_SHA256,
        "frozen S11-C fixture SHA-256 mismatch",
    )
    _validate_frozen_sources(config, fixture)
    if require_results_absent:
        parent = ROOT / Path(AGGREGATION_OUTPUT).parent
        _require(
            not os.path.lexists(parent),
            f"canonical S11-C result parent must remain absent: {parent}",
        )
        for path in (*OUTPUTS.values(), AGGREGATION_OUTPUT):
            _require(not os.path.lexists(ROOT / path), f"canonical S11-C result exists: {path}")
    return config, digest


def fixed_requests(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    fixture, _ = _read(FIXTURE_PATH)
    by_id = {item["request_id"]: item for item in fixture["requests"]}
    requests = []
    for frozen in config["fixed_inputs"]["requests"]:
        record = by_id[frozen["source_record_id"]]
        requests.append(
            {
                **frozen,
                "request_id": frozen["source_record_id"],
                "full_input_ids": list(record["full_input_ids"]),
            }
        )
    _require(
        tuple(item["request_id"] for item in requests) == REQUEST_IDS,
        "runtime request order drifted",
    )
    return requests


def _mode(config: Mapping[str, Any], mode_id: str) -> dict[str, Any]:
    _require(mode_id in MODE_IDS, f"unknown S11-C mode: {mode_id!r}")
    return dict(config["modes"][MODE_IDS.index(mode_id)])


def _expected_inputs(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "request_id": item["source_record_id"],
            "source_offset": item["source_offset"],
            "source_row": item["source_row"],
            "full_input_ids_sha256": item["token_digest_sha256"],
            "token_count": item["token_count"],
            "prompt_token_range": item["prompt_token_range"],
            "completion_token_range": item["completion_token_range"],
            "causal_completion_loss_logit_range": item["causal_completion_loss_logit_range"],
        }
        for item in config["fixed_inputs"]["requests"]
    ]


def _expected_identities(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "model": dict(config["identities"]["model"]),
        "tokenizer": dict(config["identities"]["tokenizer"]),
        "packed_artifact": dict(config["identities"]["packed_artifact"]),
        "any_precision": dict(config["identities"]["any_precision"]),
        "router_checkpoint": {
            "sha256": config["identities"]["router_checkpoint"]["sha256"],
            "candidate_order": [4, 8],
            "metadata_validated": True,
            "read_only": True,
        },
        "fixed_inputs_path": "configs/lookahead_broader_quality_inputs.json",
        "fixed_inputs_sha256": EXPECTED_FIXTURE_SHA256,
    }


def _canonical_routes(value: Any) -> dict[str, list[dict[str, Any]]]:
    _require(isinstance(value, list) and len(value) == 12, "route maps must cover twelve requests")
    result = {}
    expected_keys = [(layer, unit) for layer in range(36) for unit in UNIT_TYPES]
    for request_id, item in zip(REQUEST_IDS, value, strict=True):
        _require(
            isinstance(item, dict) and item.get("request_id") == request_id,
            "route request order drifted",
        )
        routes = item.get("routes")
        _require(isinstance(routes, list) and len(routes) == 72, "route coverage must be 72")
        normalized = []
        actual_keys = []
        for route in routes:
            _require(
                tuple(route) == ("request_id", "target_layer", "unit_type", "selected_bits"),
                "route fields/order drifted",
            )
            _require(route["request_id"] == request_id, "route request identity drifted")
            _require(
                route["selected_bits"] in CANDIDATE_BITS
                and not isinstance(route["selected_bits"], bool),
                "route bit drifted",
            )
            actual_keys.append((route["target_layer"], route["unit_type"]))
            normalized.append(dict(route))
        _require(actual_keys == expected_keys, "route order/coverage/uniqueness drifted")
        result[request_id] = normalized
    return result


def _canonical_provenance(mode_id: str, value: Any) -> dict[str, list[dict[str, Any]]]:
    _require(
        isinstance(value, dict)
        and value.get("routing_timing") == ROUTING_TIMINGS[MODE_IDS.index(mode_id)],
        "provenance timing drifted",
    )
    records = value.get("records_by_request")
    _require(
        isinstance(records, list) and len(records) == 12, "provenance must cover twelve requests"
    )
    result = {}
    for request_id, item in zip(REQUEST_IDS, records, strict=True):
        expected = [
            shared._expected_provenance(mode_id, request_id, layer, unit)
            for layer in range(36)
            for unit in UNIT_TYPES
        ]
        _require(
            item == {"request_id": request_id, "records": expected},
            "target-owned provenance drifted",
        )
        result[request_id] = expected
    return result


def _scope(routes: Sequence[dict[str, Any]], unit: str | None = None) -> list[int]:
    return [item["selected_bits"] for item in routes if unit is None or item["unit_type"] == unit]


def _route_summaries(maps: Mapping[str, list[dict[str, Any]]]) -> dict[str, Any]:
    def one(request_id: str, unit: str | None) -> dict[str, Any]:
        values = _scope(maps[request_id], unit)
        return {
            "request_id": request_id,
            "count_4": values.count(4),
            "count_8": values.count(8),
            "fraction_4": values.count(4) / len(values),
            "fraction_8": values.count(8) / len(values),
            "mean_selected_bit_width": sum(values) / len(values),
        }

    per_request = {
        scope: [
            one(request_id, None if scope == "overall" else scope) for request_id in REQUEST_IDS
        ]
        for scope in ("overall", "attention", "ffn")
    }
    aggregate = {}
    for scope in ("overall", "attention", "ffn"):
        values = [
            bit
            for request_id in REQUEST_IDS
            for bit in _scope(maps[request_id], None if scope == "overall" else scope)
        ]
        aggregate[scope] = {
            "count_4": values.count(4),
            "count_8": values.count(8),
            "fraction_4": values.count(4) / len(values),
            "fraction_8": values.count(8) / len(values),
            "mean_selected_bit_width": sum(values) / len(values),
        }
    return {
        "per_request": per_request,
        "aggregate": aggregate,
        "distinct_route_map_count": len(
            {
                tuple(item["selected_bits"] for item in maps[request_id])
                for request_id in REQUEST_IDS
            }
        ),
    }


def _historical_overlap() -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    result, _ = _read(ROOT / "docs/results/s11b_quality_pilot/same_unit_control.json")
    maps = {
        item["request_id"]: item["routes"]
        for item in result["routes"]["target_owned_route_maps"]
        if item["request_id"] in OVERLAP_IDS
    }
    quality = {
        item["request_id"]: item
        for item in result["quality"]["per_request"]
        if item["request_id"] in OVERLAP_IDS
    }
    _require(
        tuple(maps) == OVERLAP_IDS and tuple(quality) == OVERLAP_IDS,
        "S11-B overlap evidence incomplete",
    )
    return maps, quality


def _validate_raw_request(
    item: Any,
    request: Mapping[str, Any],
    routes: list[dict[str, Any]],
    provenance: list[dict[str, Any]],
) -> None:
    _require(isinstance(item, dict), "raw request evidence missing")
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
    _require(tuple(item) == expected_keys, "raw request fields/order drifted")
    _require(
        item["request_id"] == request["source_record_id"]
        and item["full_input_ids_sha256"] == request["token_digest_sha256"],
        "raw request identity drifted",
    )
    for field in ("teacher_logits_digest", "student_logits_digest"):
        _require(shared._is_hex64(item[field]), f"{field} invalid")
    _require(item["teacher_logits_shape"] == item["student_logits_shape"], "logit shapes differ")
    _require(
        item["teacher_logits_shape"][:2] == [1, 64] and item["teacher_logits_shape"][2] > 0,
        "logit shape invalid",
    )
    _require(
        item["finite_teacher_logits"] is True and item["finite_student_logits"] is True,
        "non-finite logits",
    )
    for field in ("kl", "mean_absolute_logit_error", "maximum_absolute_logit_error"):
        _require(shared._finite_nonnegative(item[field]), f"invalid metric: {field}")
    _require(
        item["routes"] == routes and item["provenance"] == provenance,
        "raw route/provenance drifted",
    )
    _require(
        item["request_cleanup"]
        == {
            "state_ended": True,
            "routes_released": True,
            "features_released": True,
            "probabilities_released": True,
            "provenance_released": True,
            "passed": True,
        },
        "request cleanup failed",
    )


def validate_mode_result(result: dict[str, Any], config: Mapping[str, Any]) -> None:
    shared._assert_no_prohibited_data_fields(result)
    expected_keys = (
        "schema",
        "mode_id",
        "protocol_config_sha256",
        "fixed_inputs_sha256",
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
    _require(tuple(result) == expected_keys, "mode result fields/order drifted")
    _require(result["schema"] == MODE_SCHEMA, "mode result schema drifted")
    mode_id = result["mode_id"]
    mode = _mode(config, mode_id)
    _require(
        result["protocol_config_sha256"] == EXPECTED_CONFIG_SHA256
        and result["fixed_inputs_sha256"] == EXPECTED_FIXTURE_SHA256,
        "protocol/fixture identity drifted",
    )
    _require(result["identities"] == _expected_identities(config), "frozen identities drifted")
    hardware = result["hardware"]
    hardware_keys = (
        "cuda_device",
        "device_index",
        "gpu_model",
        "driver_version",
        "cuda_runtime_version",
        "pytorch_version",
        "transformers_version",
        "python_version",
    )
    _require(
        isinstance(hardware, dict) and tuple(hardware) == hardware_keys,
        "hardware/software identity fields drifted",
    )
    shared._validate_device(hardware["cuda_device"])
    _require(
        isinstance(hardware["device_index"], int)
        and not isinstance(hardware["device_index"], bool)
        and hardware["device_index"] == int(hardware["cuda_device"].split(":")[1]),
        "device index drifted",
    )
    for field in hardware_keys[2:]:
        _require(
            isinstance(hardware[field], str) and bool(hardware[field]),
            f"hardware/software identity is missing: {field}",
        )
    _require(
        hardware["gpu_model"] == "NVIDIA GeForce RTX 3090",
        "comparable hardware identity drifted",
    )
    _require(
        result["seed"] == 1729 and result["inputs"] == _expected_inputs(config),
        "seed or input evidence drifted",
    )
    routes = result["routes"]
    _require(
        isinstance(routes, dict)
        and tuple(routes)
        == ("target_owned_route_maps", "summaries", "historical_overlap_equality"),
        "route result fields/order drifted",
    )
    maps = _canonical_routes(routes["target_owned_route_maps"])
    provenance = _canonical_provenance(mode_id, result["provenance"])
    summaries = _route_summaries(maps)
    _require(routes["summaries"] == summaries, "route summaries drifted")
    historical_maps, historical_quality = _historical_overlap()
    overlap_required = mode_id == MODE_IDS[0]
    overlap_passed = (not overlap_required) or all(
        maps[request_id] == historical_maps[request_id] for request_id in OVERLAP_IDS
    )
    _require(
        routes["historical_overlap_equality"]
        == {
            "required": overlap_required,
            "request_ids": list(OVERLAP_IDS),
            "passed": overlap_passed,
        },
        "historical overlap route audit drifted",
    )
    _require(overlap_passed, "control overlap routes differ from S11-B3")
    repeats = result["repeats"]
    _require(isinstance(repeats, list) and len(repeats) == 2, "exactly two repeats required")
    for index, repeat in enumerate(repeats):
        _require(
            isinstance(repeat, dict)
            and tuple(repeat)
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
        _require(
            repeat["repeat_index"] == index
            and isinstance(repeat["requests"], list)
            and len(repeat["requests"]) == 12,
            "repeat coverage/order drifted",
        )
        _require(repeat["input_digests"] == list(TOKEN_DIGESTS), "repeat input digests drifted")
        for request, raw in zip(
            config["fixed_inputs"]["requests"], repeat["requests"], strict=True
        ):
            request_id = request["source_record_id"]
            _validate_raw_request(raw, request, maps[request_id], provenance[request_id])
        _require(
            repeat["logits_digest"]
            == shared._digest(
                [
                    [item["teacher_logits_digest"], item["student_logits_digest"]]
                    for item in repeat["requests"]
                ]
            ),
            "repeat logits digest drifted",
        )
        _require(
            repeat["route_map_digest"]
            == shared._digest(result["routes"]["target_owned_route_maps"]),
            "repeat route digest drifted",
        )
        _require(
            repeat["provenance_digest"] == shared._digest(result["provenance"]),
            "repeat provenance digest drifted",
        )
        _require(
            repeat["finite_logits"] is True and repeat["finite_metrics"] is True,
            "repeat finite audit failed",
        )
    clean = lambda items: [
        {k: v for k, v in item.items() if k != "request_cleanup"} for item in items
    ]
    _require(
        clean(repeats[0]["requests"]) == clean(repeats[1]["requests"]), "repeat determinism failed"
    )
    expected_quality = [
        {
            "request_id": item["request_id"],
            "kl": item["kl"],
            "mean_absolute_logit_error": item["mean_absolute_logit_error"],
            "maximum_absolute_logit_error": item["maximum_absolute_logit_error"],
        }
        for item in repeats[0]["requests"]
    ]
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
    _require(quality["per_request"] == expected_quality, "per-request quality drifted")
    for field in ("kl", "mean_absolute_logit_error", "maximum_absolute_logit_error"):
        _require(
            quality["aggregate_" + field] == sum(item[field] for item in expected_quality) / 12,
            f"aggregate {field} drifted",
        )
    _require(quality["all_finite"] is True, "quality finite audit failed")
    if overlap_required:
        by_id = {item["request_id"]: item for item in expected_quality}
        for request_id in OVERLAP_IDS:
            _require(
                by_id[request_id] == historical_quality[request_id],
                f"control overlap metrics differ from S11-B3: {request_id}",
            )
    shared._validate_freeze_audit(result["freeze_audit"])
    audit = result["prohibited_work_audit"]
    _require(isinstance(audit, dict), "prohibited-work audit must be an object")
    evidence_label = audit.get("evidence_label")
    _require(
        evidence_label in ("production pilot evidence", "test-only structural evidence"),
        "evidence label invalid",
    )
    expected_audit = {
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
        "evidence_label": evidence_label,
        "passed": True,
    }
    _require(audit == expected_audit, "execution/prohibited-work audit drifted")


def _repeat_record(
    raw: list[dict[str, Any]], route_payload: Any, provenance: Any, index: int
) -> dict[str, Any]:
    return {
        "repeat_index": index,
        "input_digests": [item["full_input_ids_sha256"] for item in raw],
        "logits_digest": shared._digest(
            [[item["teacher_logits_digest"], item["student_logits_digest"]] for item in raw]
        ),
        "route_map_digest": shared._digest(route_payload),
        "provenance_digest": shared._digest(provenance),
        "finite_logits": all(
            item["finite_teacher_logits"] and item["finite_student_logits"] for item in raw
        ),
        "finite_metrics": all(
            shared._finite_nonnegative(item[field])
            for item in raw
            for field in ("kl", "mean_absolute_logit_error", "maximum_absolute_logit_error")
        ),
        "requests": raw,
    }


def execute_mode_with_runtime(
    runtime: shared.LookaheadQualityRuntime, *, config: Mapping[str, Any], mode_id: str, device: str
) -> dict[str, Any]:
    mode = _mode(config, mode_id)
    shared._validate_device(device)
    requests = fixed_requests(config)
    try:
        runtime.prepare(config, mode, device, requests)
        repeats = []
        for repeat_index in range(2):
            repeats.append(
                [
                    runtime.run_request(
                        mode=mode, request=request, repeat_index=repeat_index, device=device
                    )
                    for request in requests
                ]
            )
        route_payload = [
            {"request_id": request_id, "routes": repeats[0][index]["routes"]}
            for index, request_id in enumerate(REQUEST_IDS)
        ]
        provenance = {
            "routing_timing": mode["routing_timing"],
            "records_by_request": [
                {"request_id": request_id, "records": repeats[0][index]["provenance"]}
                for index, request_id in enumerate(REQUEST_IDS)
            ],
        }
        maps = _canonical_routes(route_payload)
        historical_maps, historical_quality = _historical_overlap()
        overlap_required = mode_id == MODE_IDS[0]
        overlap_passed = (not overlap_required) or all(
            maps[request_id] == historical_maps[request_id] for request_id in OVERLAP_IDS
        )
        quality_records = [
            {
                "request_id": item["request_id"],
                "kl": item["kl"],
                "mean_absolute_logit_error": item["mean_absolute_logit_error"],
                "maximum_absolute_logit_error": item["maximum_absolute_logit_error"],
            }
            for item in repeats[0]
        ]
        result = {
            "schema": MODE_SCHEMA,
            "mode_id": mode_id,
            "protocol_config_sha256": EXPECTED_CONFIG_SHA256,
            "fixed_inputs_sha256": EXPECTED_FIXTURE_SHA256,
            "identities": runtime.identity_evidence(),
            "hardware": runtime.hardware_evidence(),
            "seed": 1729,
            "inputs": _expected_inputs(config),
            "repeats": [
                _repeat_record(repeats[index], route_payload, provenance, index)
                for index in range(2)
            ],
            "quality": {
                "per_request": quality_records,
                "aggregate_kl": sum(item["kl"] for item in quality_records) / 12,
                "aggregate_mean_absolute_logit_error": sum(
                    item["mean_absolute_logit_error"] for item in quality_records
                )
                / 12,
                "aggregate_maximum_absolute_logit_error": sum(
                    item["maximum_absolute_logit_error"] for item in quality_records
                )
                / 12,
                "all_finite": True,
            },
            "routes": {
                "target_owned_route_maps": route_payload,
                "summaries": _route_summaries(maps),
                "historical_overlap_equality": {
                    "required": overlap_required,
                    "request_ids": list(OVERLAP_IDS),
                    "passed": overlap_passed,
                },
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
        if overlap_required:
            by_id = {item["request_id"]: item for item in quality_records}
            _require(
                all(
                    by_id[request_id] == historical_quality[request_id]
                    for request_id in OVERLAP_IDS
                ),
                "control overlap quality differs from S11-B3",
            )
        validate_mode_result(result, config)
        return result
    finally:
        runtime.close()


def _diagnostics(
    control: Mapping[str, list[dict[str, Any]]], treatment: Mapping[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    per_request = []
    changed = []
    aggregate_counts = {
        scope: {"changed": 0, "total": 0, "4_to_8": 0, "8_to_4": 0}
        for scope in ("overall", "attention", "ffn")
    }
    for request_id in REQUEST_IDS:
        scope_records = {}
        deltas = {}
        for scope in ("overall", "attention", "ffn"):
            unit = None if scope == "overall" else scope
            left = _scope(control[request_id], unit)
            right = _scope(treatment[request_id], unit)
            changed_count = sum(a != b for a, b in zip(left, right, strict=True))
            four_eight = sum(a == 4 and b == 8 for a, b in zip(left, right, strict=True))
            eight_four = sum(a == 8 and b == 4 for a, b in zip(left, right, strict=True))
            scope_records[scope] = {
                "hamming_count": changed_count,
                "hamming_distance": changed_count / len(left),
                "transition_4_to_8": four_eight,
                "transition_8_to_4": eight_four,
            }
            deltas[scope] = sum(right) / len(right) - sum(left) / len(left)
            for key, value in (
                ("changed", changed_count),
                ("total", len(left)),
                ("4_to_8", four_eight),
                ("8_to_4", eight_four),
            ):
                aggregate_counts[scope][key] += value
        per_request.append(
            {
                "request_id": request_id,
                "scopes": scope_records,
                "treatment_minus_control_mean_selected_width": deltas,
            }
        )
        for left, right in zip(control[request_id], treatment[request_id], strict=True):
            if left["selected_bits"] != right["selected_bits"]:
                layer = left["target_layer"]
                unit = left["unit_type"]
                provenance = shared._expected_provenance(MODE_IDS[1], request_id, layer, unit)
                changed.append(
                    {
                        "request_id": request_id,
                        "target_layer": layer,
                        "unit_type": unit,
                        "control_bits": left["selected_bits"],
                        "treatment_bits": right["selected_bits"],
                        "source_layer": provenance["source_layer"],
                        "source_point": provenance["source_point"],
                    }
                )
    aggregate = {}
    for scope, values in aggregate_counts.items():
        left = [
            bit
            for request_id in REQUEST_IDS
            for bit in _scope(control[request_id], None if scope == "overall" else scope)
        ]
        right = [
            bit
            for request_id in REQUEST_IDS
            for bit in _scope(treatment[request_id], None if scope == "overall" else scope)
        ]
        aggregate[scope] = {
            "hamming_count": values["changed"],
            "hamming_distance": values["changed"] / values["total"],
            "transition_4_to_8": values["4_to_8"],
            "transition_8_to_4": values["8_to_4"],
            "treatment_minus_control_mean_selected_width": sum(right) / len(right)
            - sum(left) / len(left),
        }
    return {
        "per_request": per_request,
        "aggregate": aggregate,
        "changed_target_units": changed,
        "changed_target_unit_count": len(changed),
        "control_route_summaries": _route_summaries(control),
        "treatment_route_summaries": _route_summaries(treatment),
    }


def build_aggregation(
    control: dict[str, Any], treatment: dict[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    validate_mode_result(control, config)
    validate_mode_result(treatment, config)
    _require(
        control["mode_id"] == MODE_IDS[0] and treatment["mode_id"] == MODE_IDS[1],
        "mode pairing/order drifted",
    )
    for field in (
        "protocol_config_sha256",
        "fixed_inputs_sha256",
        "identities",
        "hardware",
        "seed",
        "inputs",
    ):
        _require(control[field] == treatment[field], f"cross-mode identity mismatch: {field}")
    control_routes = _canonical_routes(control["routes"]["target_owned_route_maps"])
    treatment_routes = _canonical_routes(treatment["routes"]["target_owned_route_maps"])
    for request_id in REQUEST_IDS:
        for unit in UNIT_TYPES:
            _require(
                control_routes[request_id][UNIT_TYPES.index(unit)]["selected_bits"]
                == treatment_routes[request_id][UNIT_TYPES.index(unit)]["selected_bits"],
                "layer-0 route equality failed",
            )
    for index in range(12):
        _require(
            control["repeats"][0]["requests"][index]["teacher_logits_digest"]
            == treatment["repeats"][0]["requests"][index]["teacher_logits_digest"],
            "teacher logits differ across modes",
        )
    cq = {item["request_id"]: item for item in control["quality"]["per_request"]}
    tq = {item["request_id"]: item for item in treatment["quality"]["per_request"]}
    thresholds = config["interpretation"]["quality_thresholds"]
    paired = []
    for request_id in REQUEST_IDS:
        limit = 1.25 * cq[request_id]["kl"]
        paired.append(
            {
                "request_id": request_id,
                "control_kl": cq[request_id]["kl"],
                "treatment_kl": tq[request_id]["kl"],
                "ratio": shared._safe_ratio(tq[request_id]["kl"], cq[request_id]["kl"]),
                "limit": limit,
                "passed": tq[request_id]["kl"] <= limit,
            }
        )
    aggregate_kl_passed = (
        treatment["quality"]["aggregate_kl"]
        <= thresholds["treatment_aggregate_kl_max_control_factor"]
        * control["quality"]["aggregate_kl"]
    )
    aggregate_mae_passed = (
        treatment["quality"]["aggregate_mean_absolute_logit_error"]
        <= thresholds["treatment_aggregate_mean_absolute_logit_error_max_control_factor"]
        * control["quality"]["aggregate_mean_absolute_logit_error"]
    )
    all_passed = (
        aggregate_kl_passed and aggregate_mae_passed and all(item["passed"] for item in paired)
    )
    return {
        "schema": AGGREGATION_SCHEMA,
        "protocol_config_sha256": EXPECTED_CONFIG_SHA256,
        "fixed_inputs_sha256": EXPECTED_FIXTURE_SHA256,
        "mode_result_paths": [OUTPUTS[item] for item in MODE_IDS],
        "paired_inputs": {
            "request_order": list(REQUEST_IDS),
            "inputs_identical": True,
            "device_and_software_identical": True,
            "teacher_identity_and_logits_identical": True,
        },
        "paired_quality": {
            "control_aggregate_kl": control["quality"]["aggregate_kl"],
            "treatment_aggregate_kl": treatment["quality"]["aggregate_kl"],
            "aggregate_kl_ratio": shared._safe_ratio(
                treatment["quality"]["aggregate_kl"], control["quality"]["aggregate_kl"]
            ),
            "paired_request_kl": paired,
            "control_aggregate_mean_absolute_logit_error": control["quality"][
                "aggregate_mean_absolute_logit_error"
            ],
            "treatment_aggregate_mean_absolute_logit_error": treatment["quality"][
                "aggregate_mean_absolute_logit_error"
            ],
            "aggregate_mean_absolute_logit_error_ratio": shared._safe_ratio(
                treatment["quality"]["aggregate_mean_absolute_logit_error"],
                control["quality"]["aggregate_mean_absolute_logit_error"],
            ),
            "control_aggregate_maximum_absolute_logit_error": control["quality"][
                "aggregate_maximum_absolute_logit_error"
            ],
            "treatment_aggregate_maximum_absolute_logit_error": treatment["quality"][
                "aggregate_maximum_absolute_logit_error"
            ],
            "threshold_checks": {
                "aggregate_kl_passed": aggregate_kl_passed,
                "each_request_kl_passed": all(item["passed"] for item in paired),
                "aggregate_mean_absolute_logit_error_passed": aggregate_mae_passed,
                "all_quality_thresholds_passed": all_passed,
            },
        },
        "route_diagnostics": _diagnostics(control_routes, treatment_routes),
        "determinism": {
            "control_repeats_validated": True,
            "treatment_repeats_validated": True,
            "teacher_logits_equal_across_modes": True,
        },
        "freeze_audits": {"control_passed": True, "treatment_passed": True},
        "result_checks": {
            "independent_per_mode_validation": True,
            "raw_summaries_recomputed": True,
            "prohibited_fields_absent": True,
            "complete_evidence": True,
        },
        "classification": "CONTINUE" if all_passed else "STOP",
        "errors": [],
    }


def validate_aggregation_result(
    value: dict[str, Any],
    control: dict[str, Any],
    treatment: dict[str, Any],
    config: Mapping[str, Any],
) -> None:
    _require(
        value == build_aggregation(control, treatment, config),
        "aggregation differs from independent recomputation",
    )


def aggregate_paths(
    *,
    config_path: Path = DEFAULT_CONFIG,
    control_path: Path | None = None,
    treatment_path: Path | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    config, _ = load_protocol(config_path)
    paths = (
        Path(control_path) if control_path else ROOT / OUTPUTS[MODE_IDS[0]],
        Path(treatment_path) if treatment_path else ROOT / OUTPUTS[MODE_IDS[1]],
    )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        return None, {
            "classification": "PAUSE",
            "errors": [f"missing required external result: {path}" for path in missing],
        }
    try:
        control, _ = _read(paths[0])
        treatment, _ = _read(paths[1])
        value = build_aggregation(control, treatment, config)
        return value, {"classification": value["classification"], "errors": []}
    except (KeyError, TypeError, ValueError, shared.LookaheadQualityError) as exc:
        return None, {"classification": "REVISE", "errors": [str(exc)]}


def expected_mode_destination(mode_id: str) -> Path:
    _require(mode_id in MODE_IDS, f"unknown S11-C mode: {mode_id!r}")
    return ROOT / OUTPUTS[mode_id]


def validate_dispatch(
    *, mode_id: str, device: str, output: Path, config_path: Path = DEFAULT_CONFIG
) -> tuple[dict[str, Any], dict[str, Any]]:
    _require(mode_id in MODE_IDS, f"unknown S11-C mode: {mode_id!r}")
    shared._validate_device(device)
    expected = expected_mode_destination(mode_id).resolve()
    _require(
        Path(output).expanduser().resolve() == expected,
        f"mode output must be the frozen destination: {expected}",
    )
    shared.validate_destination(Path(output), shared.PersistencePolicy(expected, expected.parent))
    config, _ = load_protocol(config_path)
    return config, _mode(config, mode_id)


def persist_validated_result(
    result: dict[str, Any],
    destination: Path,
    *,
    config: Mapping[str, Any],
    kind: str,
    paired_results: tuple[dict[str, Any], dict[str, Any]] | None = None,
) -> str:
    expected = ROOT / (AGGREGATION_OUTPUT if kind == "aggregation" else OUTPUTS[result["mode_id"]])
    policy = shared.PersistencePolicy(expected, expected.parent)
    if kind == "mode":
        validator = lambda value: validate_mode_result(value, config)
    elif kind == "aggregation" and paired_results is not None:
        validator = lambda value: validate_aggregation_result(
            value, paired_results[0], paired_results[1], config
        )
    else:
        raise BroaderQualityError("aggregation persistence requires both validated mode results")
    return shared.persist_atomically(result, destination, policy=policy, validator=validator)


def plan(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config, digest = load_protocol(config_path, require_results_absent=True)
    requests = fixed_requests(config)
    script = ROOT / "scripts/run_lookahead_broader_quality.py"
    commands = [
        [
            sys.executable,
            str(script),
            "--execute-mode",
            mode_id,
            "--config",
            str(DEFAULT_CONFIG),
            "--device",
            "<explicit-cuda-device>",
            "--output",
            str(ROOT / OUTPUTS[mode_id]),
        ]
        for mode_id in MODE_IDS
    ]
    return {
        "schema": PLAN_SCHEMA,
        "protocol_config": str(DEFAULT_CONFIG),
        "protocol_config_sha256": digest,
        "fixed_inputs": str(FIXTURE_PATH),
        "fixed_inputs_sha256": EXPECTED_FIXTURE_SHA256,
        "mode_order": list(MODE_IDS),
        "request_order": [item["request_id"] for item in requests],
        "seed": 1729,
        "fresh_child_processes_per_mode": 1,
        "repeats_within_fresh_child": 2,
        "child_commands": commands,
        "aggregation_command": [
            sys.executable,
            str(script),
            "--aggregate",
            "--config",
            str(DEFAULT_CONFIG),
            "--output",
            str(ROOT / AGGREGATION_OUTPUT),
        ],
        "mode_output_paths": [str(ROOT / OUTPUTS[item]) for item in MODE_IDS],
        "aggregation_output_path": str(ROOT / AGGREGATION_OUTPUT),
        "model_loading": False,
        "dataset_loading": False,
        "runtime_tokenization": False,
        "cuda_activity": False,
        "experiment_execution": False,
        "training": False,
        "benchmarking": False,
        "result_write_activity": False,
    }


__all__ = [
    "AGGREGATION_OUTPUT",
    "DEFAULT_CONFIG",
    "EXPECTED_CONFIG_SHA256",
    "EXPECTED_FIXTURE_SHA256",
    "MODE_IDS",
    "OUTPUTS",
    "REQUEST_IDS",
    "BroaderQualityError",
    "aggregate_paths",
    "build_aggregation",
    "execute_mode_with_runtime",
    "fixed_requests",
    "load_protocol",
    "persist_validated_result",
    "plan",
    "validate_aggregation_result",
    "validate_dispatch",
    "validate_mode_result",
]
