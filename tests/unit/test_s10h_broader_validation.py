from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import run_s10h as runner

ROOT = Path(__file__).parents[2]


def _fixture() -> dict[str, object]:
    return runner.synthetic_structural_fixture()


def test_frozen_protocol_and_pre_execution_identity_are_fail_closed():
    config = runner._load_frozen_config()
    runner._validate_protocol(config)
    context = runner._validate_pre_execution(result_path=ROOT / "tmp-s10h-result-that-does-not-exist")
    assert context["head"]
    assert context["identities"]["model_revision"] == runner.MODEL_REVISION
    assert context["historical_hashes"]["docs/results/s10f_frontier_confirmation.json"] == (
        runner.HISTORICAL_ATTEMPT_1_SHA256
    )


def test_frozen_config_rejects_sha_and_in_memory_mutation(tmp_path):
    payload = json.loads(runner.CONFIG_PATH.read_text())
    payload["protocol"]["seeds"] = [1729, 1730, 1732]
    path = tmp_path / "mutated.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    with pytest.raises(runner.ProtocolError, match="byte-for-byte"):
        runner._load_frozen_config(path)

    config = runner._load_frozen_config()
    config["protocol"]["candidate_bits"] = [4, 8, 6]
    with pytest.raises(runner.ProtocolError, match="fields differ"):
        runner._validate_protocol(config)


def test_historical_s10f_mutation_is_rejected_without_rewrite(tmp_path, monkeypatch):
    source = runner.HISTORICAL_ATTEMPT_1_PATH.read_bytes()
    mutated = tmp_path / "historical.json"
    mutated.write_bytes(source + b"\n")
    monkeypatch.setattr(runner, "HISTORICAL_ATTEMPT_1_PATH", mutated)
    with pytest.raises(runner.ProtocolError, match="identity changed"):
        runner._validate_historical_artifacts()
    assert mutated.read_bytes() == source + b"\n"


def test_synthetic_fixture_validates_exact_matrix_contract():
    report = runner.validate_result(_fixture())
    assert report["classification"] == "CONTINUE"
    assert report["errors"] == []
    assert len(runner.synthetic_structural_fixture()["trials"]) == 9


def test_missing_and_extra_trial_pairs_follow_fail_closed_precedence():
    missing = _fixture()
    missing["trials"] = missing["trials"][:-1]
    report = runner.validate_result(missing)
    assert report["classification"] == "PAUSE"

    extra = _fixture()
    extra["trials"].append(copy.deepcopy(extra["trials"][0]))
    report = runner.validate_result(extra)
    assert report["classification"] == "REVISE"

    reordered = _fixture()
    reordered["trials"] = list(reversed(reordered["trials"]))
    assert runner.validate_result(reordered)["classification"] == "REVISE"


def test_training_candidate_route_map_and_optimizer_drift_is_rejected():
    candidate = _fixture()
    candidate["trials"][0]["candidate_bits"] = [4, 8, 6]
    assert runner.validate_result(candidate)["classification"] == "REVISE"

    route = _fixture()
    route["trials"][0]["hard_validation_route_maps"][runner.VALIDATION_IDS[0]] = [4] * 71
    assert runner.validate_result(route)["classification"] == "PAUSE"

    optimizer = _fixture()
    optimizer["trials"][0]["optimizer_audit"]["router_only_optimizer_audit"] = False
    assert runner.validate_result(optimizer)["classification"] == "REVISE"

    freeze = _fixture()
    freeze["trials"][0]["teacher_frozen_audit"] = False
    assert runner.validate_result(freeze)["classification"] == "REVISE"


def test_prohibited_work_and_reproducibility_are_revise_before_refine():
    prohibited = _fixture()
    prohibited["run_audits"]["prohibited_work_audit"]["forbidden_actions_observed"] = ["warm_start"]
    assert runner.validate_result(prohibited)["classification"] == "REVISE"

    repeat = _fixture()
    repeat["trials"][0]["reproducibility_audit"]["passed"] = False
    repeat["aggregates"] = runner._aggregate_trials(repeat["trials"])
    repeat["gate"]["classification"] = "REVISE"
    assert runner.validate_result(repeat)["classification"] == "REVISE"


def test_complete_valid_matrix_that_misses_threshold_is_refine():
    fixture = _fixture()
    for trial in fixture["trials"]:
        if trial["lambda_bit"] == 0.03:
            trial["hard_validation_kd"] = 1.1
    fixture["aggregates"] = runner._aggregate_trials(fixture["trials"])
    fixture["gate"]["classification"] = "REFINE"
    report = runner.validate_result(fixture)
    assert report["classification"] == "REFINE"


def test_forbidden_measurement_fields_and_route_order_fail_closed():
    forbidden = _fixture()
    forbidden["trials"][0]["latency"] = 1.0
    assert runner.validate_result(forbidden)["classification"] == "REVISE"

    maps = _fixture()
    route_maps = maps["trials"][0]["hard_validation_route_maps"]
    maps["trials"][0]["hard_validation_route_maps"] = {
        key: route_maps[key] for key in reversed(runner.VALIDATION_IDS)
    }
    assert runner.validate_result(maps)["classification"] == "REVISE"


def test_canonical_result_refuses_overwrite(tmp_path):
    existing = tmp_path / "canonical.json"
    existing.write_text("preserve me")
    with pytest.raises(runner.CanonicalResultExists, match="refusing overwrite"):
        runner._validate_pre_execution(result_path=existing)
    assert existing.read_text() == "preserve me"


def test_cli_plan_is_deterministic_nonexecuting_and_subprocess_safe(tmp_path):
    output = tmp_path / "canonical.json"
    command = [sys.executable, str(ROOT / "scripts/run_s10h.py"), "--plan", "--output", str(output)]
    first = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True)
    second = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True)
    assert first.stdout == second.stdout
    payload = json.loads(first.stdout)
    assert payload["mode"] == "plan"
    assert payload["protocol_identity"]["sha256"] == runner.LOCKED_CONFIG_SHA256
    assert payload["trials"]["count"] == 9
    assert payload["data"]["train_rows"] == 24
    assert payload["data"]["validation_rows"] == 12
    assert payload["plan_loads_model"] is False
    assert payload["plan_trains"] is False
    assert payload["plan_evaluates_cuda"] is False
    assert not output.exists()


def test_cli_execute_requires_h2_and_never_writes_result(tmp_path):
    output = tmp_path / "canonical.json"
    process = subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_s10h.py"), "--execute", "--output", str(output)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode != 0
    assert "S10-H2 executor is intentionally unavailable" in process.stderr
    assert not output.exists()


def test_historical_hash_constants_match_preserved_artifacts():
    assert hashlib.sha256(runner.HISTORICAL_ATTEMPT_1_PATH.read_bytes()).hexdigest() == runner.HISTORICAL_ATTEMPT_1_SHA256
    assert hashlib.sha256(runner.HISTORICAL_ATTEMPT_2_PATH.read_bytes()).hexdigest() == runner.HISTORICAL_ATTEMPT_2_SHA256
