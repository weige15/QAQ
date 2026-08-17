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


def test_frozen_config_rejects_sha_and_in_memory_mutation(tmp_path, monkeypatch):
    payload = json.loads(runner.CONFIG_PATH.read_text())
    payload["protocol"]["seeds"] = [1729, 1730, 1732]
    path = tmp_path / "mutated.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    with pytest.raises(runner.ProtocolError, match="byte-for-byte"):
        runner._load_frozen_config(path)

    whitespace = tmp_path / "whitespace.json"
    whitespace.write_bytes(runner.CONFIG_PATH.read_bytes() + b"\n")
    with pytest.raises(runner.ProtocolError, match="byte-for-byte"):
        runner._load_frozen_config(whitespace)

    reordered = tmp_path / "reordered.json"
    reordered.write_text(json.dumps(json.loads(runner.CONFIG_PATH.read_text()), indent=2, sort_keys=True) + "\n")
    with pytest.raises(runner.ProtocolError, match="byte-for-byte"):
        runner._load_frozen_config(reordered)

    monkeypatch.setattr(runner, "CONFIG_PATH", whitespace)
    with pytest.raises(runner.ProtocolError, match="byte-for-byte"):
        runner._load_frozen_config()

    monkeypatch.undo()
    config = runner._load_frozen_config()
    config["protocol"]["candidate_bits"] = [4, 8, 6]
    with pytest.raises(runner.ProtocolError, match="fields differ"):
        runner._validate_protocol(config)

    reordered_config = dict(reversed(runner._load_frozen_config().items()))
    with pytest.raises(runner.ProtocolError, match="fields differ"):
        runner._validate_protocol(reordered_config)


def test_packed_artifact_bytes_are_verified_against_the_frozen_digest(monkeypatch):
    monkeypatch.setattr(runner, "_sha256_file", lambda path: "tampered")
    with pytest.raises(runner.ProtocolError, match="artifact bytes"):
        runner._validate_frozen_identity(runner._load_frozen_config())


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


def test_incomplete_trial_evidence_does_not_dereference_or_aggregate():
    missing_hash = _fixture()
    missing_hash["trials"][0].pop("initial_router_state_sha256")
    report = runner.validate_result(missing_hash)
    assert report["classification"] == "PAUSE"

    malformed_history = _fixture()
    malformed_history["trials"][0]["training_history"][0] = None
    report = runner.validate_result(malformed_history)
    assert report["classification"] == "PAUSE"


def test_non_object_result_and_unhashable_collapse_classification_fail_closed():
    assert runner.validate_result([])["classification"] == "PAUSE"

    malformed = _fixture()
    malformed["trials"][0]["collapse_audit"]["classification"] = []
    assert runner.validate_result(malformed)["classification"] == "REVISE"


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


@pytest.mark.parametrize("tamper", ["duplicate", "missing_extra", "non_router"])
def test_optimizer_membership_is_recomputed_from_canonical_actual_names(tamper):
    candidate = _fixture()
    audit = candidate["trials"][0]["optimizer_audit"]
    names = list(audit["actual_optimizer_parameter_names"])
    if tamper == "duplicate":
        names[1] = names[0]
    elif tamper == "missing_extra":
        names[1] = "routers.injected.parameter"
    else:
        names[1] = "student.base.weight"
    digest = runner._sha256_names(names)
    audit.update(
        actual_optimizer_parameter_names=names,
        actual_optimizer_parameter_names_sha256=digest,
        expected_router_parameter_names=names,
        expected_router_parameter_names_sha256=digest,
        duplicate_optimizer_parameter_count=0,
        missing_router_parameter_count=0,
        unexpected_optimizer_parameter_count=0,
    )
    assert runner.validate_result(candidate)["classification"] == "REVISE"


def test_packed_artifact_path_and_manifest_identity_are_required_in_future_result():
    candidate = _fixture()
    candidate["identities"].pop("packed_artifact")
    assert runner.validate_result(candidate)["classification"] == "REVISE"

    candidate = _fixture()
    candidate["identities"]["packed_artifact"] = "other-artifact"
    assert runner.validate_result(candidate)["classification"] == "REVISE"

    candidate = _fixture()
    candidate["identities"]["manifest_sha256"] = "wrong-manifest"
    assert runner.validate_result(candidate)["classification"] == "REVISE"


def test_prohibited_work_and_reproducibility_are_revise_before_refine():
    prohibited = _fixture()
    prohibited["run_audits"]["prohibited_work_audit"]["forbidden_actions_observed"] = ["warm_start"]
    assert runner.validate_result(prohibited)["classification"] == "REVISE"

    repeat = _fixture()
    repeat["trials"][0]["reproducibility_audit"]["passed"] = False
    repeat["aggregates"] = runner._aggregate_trials(repeat["trials"])
    repeat["gate"]["classification"] = "REVISE"
    assert runner.validate_result(repeat)["classification"] == "REVISE"


@pytest.mark.parametrize("field", ["route_maps_identical", "hard_metrics_identical", "finite_outputs_both_passed"])
def test_reproducibility_subaudits_must_all_pass(field):
    candidate = _fixture()
    candidate["trials"][0]["reproducibility_audit"][field] = False
    assert runner.validate_result(candidate)["classification"] == "REVISE"

    candidate["trials"][0]["reproducibility_audit"]["passed"] = False
    assert runner.validate_result(candidate)["classification"] == "REVISE"


def test_collapse_audit_rejects_invalid_or_inconsistent_claims():
    invalid = _fixture()
    invalid["trials"][0]["collapse_audit"]["invalid_or_degenerate"] = True
    assert runner.validate_result(invalid)["classification"] == "REVISE"

    inconsistent = _fixture()
    inconsistent["trials"][0]["collapse_audit"]["passed"] = False
    assert runner.validate_result(inconsistent)["classification"] == "REVISE"


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


@pytest.mark.parametrize(
    "field",
    ["repository", "config", "train_split", "validation_split", "tokenizer_revision", "revision"],
)
def test_dataset_source_identity_is_required_and_matches_frozen_protocol(field):
    missing = _fixture()
    missing["dataset"].pop(field)
    assert runner.validate_result(missing)["classification"] == "PAUSE"

    mismatched = _fixture()
    mismatched["dataset"][field] = "tampered"
    assert runner.validate_result(mismatched)["classification"] == "REVISE"


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
