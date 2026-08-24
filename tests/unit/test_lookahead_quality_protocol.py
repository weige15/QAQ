from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path, PurePosixPath

import pytest

from scripts import validate_lookahead_quality_protocol as validator

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "lookahead_quality_pilot.json"


def _payload() -> dict:
    return json.loads(CONFIG.read_text())


def _write_payload(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def _assert_rejected(tmp_path: Path, payload: dict) -> None:
    with pytest.raises(validator.ProtocolValidationError):
        validator.validate_protocol(_write_payload(tmp_path, payload))


def test_canonical_protocol_succeeds():
    summary = validator.validate_protocol(CONFIG)
    assert summary == {
        "schema": "qaq-s11b-quality-pilot-v1",
        "config_sha256": validator.EXPECTED_CONFIG_SHA256,
        "mode_ids": [
            "same_unit_control",
            "lookahead_attention_one_unit_treatment",
        ],
        "request_ids": ["validation-3", "validation-1000"],
        "planned_result_paths": list(validator.EXPECTED_OUTPUTS),
    }


def test_cli_is_deterministic_across_repeated_read_only_runs():
    command = [
        sys.executable,
        str(ROOT / "scripts" / "validate_lookahead_quality_protocol.py"),
        "--config",
        str(CONFIG),
    ]
    environment = {**os.environ, "PYTHONPATH": "src:."}
    first = subprocess.run(
        command, cwd=ROOT, env=environment, text=True, capture_output=True, check=False
    )
    second = subprocess.run(
        command, cwd=ROOT, env=environment, text=True, capture_output=True, check=False
    )
    assert (first.returncode, first.stdout, first.stderr) == (
        second.returncode,
        second.stdout,
        second.stderr,
    )
    assert first.returncode == 0
    assert first.stdout == (
        "S11-B1 protocol valid: qaq-s11b-quality-pilot-v1 "
        "(2 modes, 2 inputs; no planned results present)\n"
    )
    assert first.stderr == ""


def test_validator_writes_nothing_and_all_planned_results_remain_absent():
    watched = [
        CONFIG,
        ROOT / "configs" / "baseline_evaluation.json",
        ROOT / "configs" / "baseline_evaluation_prompts.json",
        ROOT / "docs" / "quantized_model_manifest.json",
        ROOT / "docs" / "results" / "s07_router_training.json",
    ]
    before = {
        path: (path.stat().st_mtime_ns, hashlib.sha256(path.read_bytes()).hexdigest())
        for path in watched
    }
    assert not any(os.path.lexists(ROOT / path) for path in validator.EXPECTED_OUTPUTS)
    validator.validate_protocol(CONFIG)
    after = {
        path: (path.stat().st_mtime_ns, hashlib.sha256(path.read_bytes()).hexdigest())
        for path in watched
    }
    assert after == before
    assert not any(os.path.lexists(ROOT / path) for path in validator.EXPECTED_OUTPUTS)


def test_mode_and_input_order_are_stable_and_mode_difference_is_unambiguous():
    payload = _payload()
    modes = payload["modes"]
    assert [mode["id"] for mode in modes] == list(validator.EXPECTED_MODE_IDS)
    assert [mode["routing_timing"] for mode in modes] == list(validator.EXPECTED_TIMINGS)
    shared = [
        {key: value for key, value in mode.items() if key not in {"id", "routing_timing"}}
        for mode in modes
    ]
    assert shared[0] == shared[1]
    assert payload["fixed_inputs"]["request_order"] == ["validation-3", "validation-1000"]


def test_authoritative_identities_and_full_input_ids_agree():
    payload = _payload()
    baseline_protocol = json.loads((ROOT / "configs" / "baseline_evaluation.json").read_text())
    manifest = json.loads((ROOT / "docs" / "quantized_model_manifest.json").read_text())
    router_training_result = json.loads(
        (ROOT / "docs" / "results" / "s07_router_training.json").read_text()
    )
    prompts = json.loads((ROOT / "configs" / "baseline_evaluation_prompts.json").read_text())
    identities = payload["identities"]

    assert identities["model"] == {
        "repository": baseline_protocol["identities"]["model"]["repository"],
        "revision": manifest["source_model"]["revision"],
    }
    assert identities["tokenizer"]["revision"] == manifest["source_model"]["tokenizer_revision"]
    assert identities["packed_artifact"]["relative_path"] == manifest["artifact"]["local_path"]
    assert (
        identities["packed_artifact"]["sha256"]
        == manifest["artifact"]["checkpoint_hashes"]["pytorch_model.bin"]
    )
    assert (
        identities["router_checkpoint"]["sha256"] == router_training_result["checkpoint"]["sha256"]
    )
    assert identities["any_precision"]["commit"] == manifest["any_precision"]["commit"]

    source = {record["id"]: record for record in prompts["requests"]}
    assert payload["fixed_inputs"]["source_field"] == "full_input_ids"
    for frozen in payload["fixed_inputs"]["requests"]:
        request = source[frozen["source_record_id"]]
        assert len(request["full_input_ids"]) == 64
        assert validator._token_digest(request["full_input_ids"]) == frozen["token_digest_sha256"]
        assert frozen["token_digest_sha256"] == request["full_input_ids_sha256"]


def test_interpretation_paths_and_result_schemas_are_explicit_and_unique():
    payload = _payload()
    interpretation = payload["interpretation"]
    assert interpretation["precedence"] == [
        "INVALID_EVIDENCE",
        "PAUSE",
        "ADVANCE_TO_BROADER_QUALITY_CHECK",
        "CHECKPOINT_REUSE_DEGRADES",
    ]
    assert interpretation["quality_thresholds"] == {
        "treatment_aggregate_kl_max_control_factor": 1.10,
        "treatment_each_request_kl_max_paired_control_factor": 1.25,
        "treatment_aggregate_mean_absolute_logit_error_max_control_factor": 1.10,
        "implementation_choices_not_paper_facts": True,
    }
    assert payload["quality_metrics"]["width_combined_quality_scalar"] is False
    assert payload["quality_metrics"]["route_distance_is_quality_metric"] is False
    assert (
        payload["route_comparison"]["cross_mode_equality"]["ffn_equality_beyond_layer_0_required"]
        is False
    )

    paths = [
        *(record["path"] for record in payload["planned_results"]["mode_outputs"]),
        payload["planned_results"]["aggregation_output"],
    ]
    assert tuple(paths) == validator.EXPECTED_OUTPUTS
    assert len(paths) == len(set(paths))
    assert all(
        not PurePosixPath(path).is_absolute() and ".." not in PurePosixPath(path).parts
        for path in paths
    )
    assert (
        payload["result_schema_contracts"]["per_mode"]["schema"]
        == payload["planned_results"]["per_mode_schema"]
    )
    assert (
        payload["result_schema_contracts"]["aggregation"]["schema"]
        == payload["planned_results"]["aggregation_schema"]
    )


def test_reusable_protocol_source_is_non_executable():
    source = ROOT / "src" / "qaq" / "evaluation" / "lookahead_quality_protocol.py"
    assert source.stat().st_mode & 0o111 == 0


def test_validator_imports_without_ml_or_backend_packages():
    script = ROOT / "scripts" / "validate_lookahead_quality_protocol.py"
    code = f"""
import importlib.abc
import importlib.util
import sys
blocked = {{'torch', 'transformers', 'datasets', 'any_precision', 'any_precision_ext'}}
class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in blocked:
            raise AssertionError('forbidden import: ' + fullname)
        return None
sys.meta_path.insert(0, Blocker())
spec = importlib.util.spec_from_file_location('lookahead_validator_isolated', {str(script)!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
print(module.EXPECTED_CONFIG_SHA256)
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", code], text=True, capture_output=True, check=False
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == validator.EXPECTED_CONFIG_SHA256
    assert completed.stderr == ""


def _apply_mutation(payload: dict, name: str) -> None:
    if name == "wrong_schema":
        payload["schema"] = "wrong"
    elif name == "missing_schema":
        payload.pop("schema")
    elif name == "reversed_modes":
        payload["modes"].reverse()
    elif name == "extra_mode":
        payload["modes"].append(copy.deepcopy(payload["modes"][0]))
    elif name == "wrong_timing":
        payload["modes"][1]["routing_timing"] = "same_unit"
    elif name == "non_timing_mode_difference":
        payload["modes"][1]["packed_weight_state"] = "cpu"
    elif name == "wrong_candidates":
        payload["modes"][0]["candidate_order"] = [8, 4]
    elif name == "three_way_candidates":
        payload["modes"][0]["candidate_order"] = [4, 6, 8]
    elif name == "on_demand_loading":
        payload["modes"][0]["on_demand_loading"] = True
    elif name == "generation_decode":
        payload["execution_contract"]["generation"] = True
        payload["execution_contract"]["decode"] = True
    elif name == "runtime_tokenization":
        payload["fixed_inputs"]["runtime_tokenization"] = True
    elif name == "dataset_access":
        payload["fixed_inputs"]["dataset_access"] = True
    elif name == "request_reorder":
        payload["fixed_inputs"]["request_order"].reverse()
    elif name == "prompt_ids_instead_of_full_ids":
        payload["fixed_inputs"]["source_field"] = "input_ids"
    elif name == "digest_drift":
        payload["fixed_inputs"]["requests"][0]["token_digest_sha256"] = "0" * 64
    elif name == "range_drift":
        payload["fixed_inputs"]["requests"][0]["prompt_token_range"] = [0, 31]
    elif name == "length_drift":
        payload["fixed_inputs"]["requests"][0]["token_count"] = 63
    elif name == "checkpoint_identity_drift":
        payload["identities"]["router_checkpoint"]["sha256"] = "0" * 64
    elif name == "packed_identity_drift":
        payload["identities"]["packed_artifact"]["sha256"] = "0" * 64
    elif name == "model_identity_drift":
        payload["identities"]["model"]["revision"] = "main"
    elif name == "tokenizer_identity_drift":
        payload["identities"]["tokenizer"]["revision"] = "main"
    elif name == "any_precision_identity_drift":
        payload["identities"]["any_precision"]["commit"] = "main"
    elif name in {"optimizer", "training_steps", "learning_rate", "scheduler", "checkpoint_output"}:
        payload[name] = 1
    elif name == "kl_temperature_drift":
        payload["quality_metrics"]["teacher_student_kl"]["temperature"] = 1.0
    elif name == "aggregate_margin_drift":
        payload["interpretation"]["quality_thresholds"][
            "treatment_aggregate_kl_max_control_factor"
        ] = 1.11
    elif name == "request_margin_drift":
        payload["interpretation"]["quality_thresholds"][
            "treatment_each_request_kl_max_paired_control_factor"
        ] = 1.20
    elif name == "repeat_omission":
        payload["determinism_audit"].pop("repeats_per_mode")
    elif name == "freeze_omission":
        payload.pop("freeze_audit")
    elif name == "route_coverage_drift":
        payload["route_comparison"]["routes_per_request"] = 71
    elif name == "route_bit_drift":
        payload["route_comparison"]["allowed_selected_bits"] = [4, 6, 8]
    elif name == "performance_field":
        payload["performance"] = {}
    elif name == "prefetch_field":
        payload["prefetch"] = True
    elif name == "absolute_output":
        payload["planned_results"]["aggregation_output"] = "/tmp/aggregation.json"
    elif name == "duplicate_output":
        payload["planned_results"]["aggregation_output"] = payload["planned_results"][
            "mode_outputs"
        ][0]["path"]
    else:  # pragma: no cover - parametrization is static
        raise AssertionError(name)


@pytest.mark.parametrize(
    "name",
    [
        "wrong_schema",
        "missing_schema",
        "reversed_modes",
        "extra_mode",
        "wrong_timing",
        "non_timing_mode_difference",
        "wrong_candidates",
        "three_way_candidates",
        "on_demand_loading",
        "generation_decode",
        "runtime_tokenization",
        "dataset_access",
        "request_reorder",
        "prompt_ids_instead_of_full_ids",
        "digest_drift",
        "range_drift",
        "length_drift",
        "checkpoint_identity_drift",
        "packed_identity_drift",
        "model_identity_drift",
        "tokenizer_identity_drift",
        "any_precision_identity_drift",
        "optimizer",
        "training_steps",
        "learning_rate",
        "scheduler",
        "checkpoint_output",
        "kl_temperature_drift",
        "aggregate_margin_drift",
        "request_margin_drift",
        "repeat_omission",
        "freeze_omission",
        "route_coverage_drift",
        "route_bit_drift",
        "performance_field",
        "prefetch_field",
        "absolute_output",
        "duplicate_output",
    ],
)
def test_protocol_mutations_fail_closed(tmp_path: Path, name: str):
    payload = _payload()
    _apply_mutation(payload, name)
    _assert_rejected(tmp_path, payload)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_non_finite_json_is_rejected(tmp_path: Path, constant: str):
    path = tmp_path / "non-finite.json"
    path.write_text('{"schema": ' + constant + "}\n")
    with pytest.raises(validator.ProtocolValidationError, match="non-finite"):
        validator.validate_protocol(path)


def test_malformed_json_is_rejected(tmp_path: Path):
    path = tmp_path / "malformed.json"
    path.write_text("{\n")
    with pytest.raises(validator.ProtocolValidationError, match="cannot parse JSON"):
        validator.validate_protocol(path)


def test_pre_existing_planned_result_is_rejected_without_creating_one(monkeypatch):
    real_lexists = validator.os.path.lexists
    blocked = str(ROOT / validator.EXPECTED_OUTPUTS[1])

    def fake_lexists(path):
        return str(path) == blocked or real_lexists(path)

    monkeypatch.setattr(validator.os.path, "lexists", fake_lexists)
    with pytest.raises(validator.ProtocolValidationError, match="planned result already exists"):
        validator.validate_protocol(CONFIG)
