from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from qaq.evaluation import block_sensitivity as sensitivity

ROOT = Path(__file__).parents[2]


def _hex(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _fake_result(plan: dict, unit_id: str, precision: int, *, passes: bool = True) -> dict:
    spec = next(item for item in plan["units"] if item["unit_id"] == unit_id)
    routes = sensitivity._source_route_index()
    contexts = []
    for source in plan["source_contexts"]:
        control_metrics = {
            sensitivity.METRIC_KEYS[0]: 1.0,
            sensitivity.METRIC_KEYS[1]: 1.0,
            sensitivity.METRIC_KEYS[2]: 2.0,
        }
        treatment_factor = 1.0 if passes else 1.3
        treatment_metrics = {
            sensitivity.METRIC_KEYS[0]: treatment_factor,
            sensitivity.METRIC_KEYS[1]: treatment_factor,
            sensitivity.METRIC_KEYS[2]: 2.1,
        }
        context_id = source["context_id"]
        control_digest = _hex(f"{context_id}:control")
        treatment_digest = _hex(f"{context_id}:precision-{precision}")
        contexts.append(
            {
                "context_id": context_id,
                "seed": source["seed"],
                "request_id": source["request_id"],
                "source": {
                    key: source[key]
                    for key in (
                        "source_trial_id",
                        "source_trial_sha256",
                        "input_digest",
                        "teacher_digest",
                        "source_route_map_sha256",
                    )
                },
                "control": {
                    "forced_bits": 8,
                    "route_map_sha256": sensitivity._expected_arm(
                        routes[context_id], spec["layer"], spec["unit_type"], 8
                    ),
                    "primary_logits_sha256": control_digest,
                    "repeat_logits_sha256": control_digest,
                    "primary_metrics": control_metrics,
                    "repeat_metrics": dict(control_metrics),
                    "finite": True,
                    "repeat_identical": True,
                },
                "treatment": {
                    "forced_bits": precision,
                    "route_map_sha256": sensitivity._expected_arm(
                        routes[context_id], spec["layer"], spec["unit_type"], precision
                    ),
                    "primary_logits_sha256": treatment_digest,
                    "repeat_logits_sha256": treatment_digest,
                    "primary_metrics": treatment_metrics,
                    "repeat_metrics": dict(treatment_metrics),
                    "finite": True,
                    "repeat_identical": True,
                },
                "route_pair_isolates_target": True,
            }
        )
    config = json.loads((ROOT / "configs/lookahead_468_training.json").read_text())
    summaries = sensitivity._summaries(contexts)
    return {
        "schema": sensitivity.RESULT_SCHEMA,
        "study_id": plan["study_id"],
        "unit_id": unit_id,
        "target": {"layer": spec["layer"], "unit_type": spec["unit_type"]},
        "precision": precision,
        "identities": config["identities"],
        "hardware": {
            "cuda_device": "cuda:0",
            "device_index": 0,
            "gpu_model": "NVIDIA GeForce RTX 3090",
            "driver_version": "test-driver",
            "cuda_runtime_version": "test-cuda",
            "pytorch_version": "test-torch",
            "transformers_version": "test-transformers",
            "python_version": "3.12",
        },
        "paired_contexts": contexts,
        "seed_summaries": summaries,
        "passed": all(item["passed"] for item in summaries),
        "audit": {
            "complete_context_count": 36,
            "same_unit_only": True,
            "lambda_retuning_observed": False,
            "router_training_observed": False,
            "lookahead_execution_observed": False,
            "one_target_block_only": True,
            "paired_forced_8_controls_complete": True,
            "immediate_repeats_complete": True,
            "all_values_finite": True,
            "complete": True,
        },
    }


@pytest.fixture(scope="module")
def plan() -> dict:
    return sensitivity.build_plan()


def test_plan_is_exactly_the_defined_42_unit_study(plan):
    assert plan["schema"] == sensitivity.PLAN_SCHEMA
    assert plan["definition_sha256"] == sensitivity.EXPECTED_DEFINITION_SHA256
    assert plan["design"] == {
        "unit_count": 42,
        "one_block_at_a_time": True,
        "routing_timing": "same_unit",
        "route_context": "canonical_same_unit_lambda_0",
        "seed_contexts": [1729, 1730, 1731],
        "requests": list(sensitivity.REQUEST_IDS),
        "paired_contexts_per_intervention": 36,
        "control_bits": 8,
        "precision_sequence": [4, 6],
        "fallback_rule": "run precision 6 only when complete valid precision 4 evidence fails",
        "initial_intervention_count": 42,
        "maximum_intervention_count": 84,
        "immediate_repeat_count": 1,
    }
    assert len(plan["units"]) == len({item["unit_id"] for item in plan["units"]}) == 42
    assert [item["unit_index"] for item in plan["units"]] == list(range(1, 43))
    assert len(plan["source_contexts"]) == 3 * 12 == 36
    assert plan["quality_contract"]["factors"] == {
        "per_seed_aggregate_kl_max_control_factor": 1.1,
        "per_seed_aggregate_mean_absolute_error_max_control_factor": 1.1,
        "per_request_kl_max_paired_control_factor": 1.25,
    }


def test_plan_serialization_is_deterministic_and_nonexecuting(plan):
    first = sensitivity.serialize(plan)
    second = sensitivity.serialize(sensitivity.build_plan())
    assert first == second
    assert plan["non_execution_audit"] == {
        "model_loading": False,
        "cuda_activity": False,
        "sensitivity_execution": False,
        "router_training": False,
        "lambda_retuning": False,
        "lookahead_work": False,
        "result_write_activity": False,
    }
    assert not os.path.lexists(sensitivity.RESULT_PARENT)


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda value: value["units"].pop(), "authoritative deterministic study"),
        (
            lambda value: value["units"].append(copy.deepcopy(value["units"][0])),
            "authoritative deterministic study",
        ),
        (
            lambda value: value["design"].__setitem__("seed_contexts", [1729, 1730]),
            "authoritative deterministic study",
        ),
        (lambda value: value["design"]["requests"].reverse(), "authoritative deterministic study"),
        (
            lambda value: value["design"].__setitem__("precision_sequence", [6, 4]),
            "authoritative deterministic study",
        ),
        (
            lambda value: value["quality_contract"]["factors"].__setitem__(
                "per_request_kl_max_paired_control_factor", 1.26
            ),
            "authoritative deterministic study",
        ),
    ],
)
def test_plan_rejects_missing_duplicate_reordered_or_retuned_structure(plan, mutation, match):
    changed = copy.deepcopy(plan)
    mutation(changed)
    with pytest.raises(sensitivity.SensitivityError, match=match):
        sensitivity.validate_plan(changed)


def test_complete_unit_evidence_validates_and_thresholds_are_recomputed(plan):
    unit_id = plan["units"][0]["unit_id"]
    passing = _fake_result(plan, unit_id, 4, passes=True)
    sensitivity.validate_unit_result(passing, plan)
    assert passing["passed"] is True
    assert len(passing["paired_contexts"]) == 36
    assert all(item["passed"] for item in passing["seed_summaries"])

    failing = _fake_result(plan, unit_id, 4, passes=False)
    sensitivity.validate_unit_result(failing, plan)
    assert failing["passed"] is False
    assert not any(item["passed"] for item in failing["seed_summaries"])


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda value: value["paired_contexts"].pop(), "coverage"),
        (
            lambda value: value["paired_contexts"].append(
                copy.deepcopy(value["paired_contexts"][0])
            ),
            "coverage",
        ),
        (
            lambda value: value["paired_contexts"][0]["source"].__setitem__(
                "input_digest", "0" * 64
            ),
            "source evidence",
        ),
        (
            lambda value: value["paired_contexts"][0]["control"].__setitem__(
                "route_map_sha256", "0" * 64
            ),
            "control route map",
        ),
        (
            lambda value: value["paired_contexts"][0]["treatment"].__setitem__(
                "repeat_identical", False
            ),
            "finite/repeat",
        ),
        (lambda value: value["seed_summaries"].clear(), "seed summaries"),
        (lambda value: value["audit"].__setitem__("router_training_observed", True), "audit"),
    ],
)
def test_unit_evidence_rejects_incomplete_conflicting_or_prohibited_records(plan, mutation, match):
    result = _fake_result(plan, plan["units"][0]["unit_id"], 4)
    mutation(result)
    with pytest.raises(sensitivity.SensitivityError, match=match):
        sensitivity.validate_unit_result(result, plan)


def test_dispatch_enforces_exact_destination_and_4_first_6_fallback(monkeypatch, tmp_path, plan):
    parent = tmp_path / "results"
    monkeypatch.setattr(sensitivity, "RESULT_PARENT", parent)
    monkeypatch.setattr(sensitivity, "AGGREGATION_OUTPUT", parent / "aggregation.json")
    unit_id = plan["units"][0]["unit_id"]
    unit_parent = parent / unit_id
    unit_parent.mkdir(parents=True)
    p4 = unit_parent / "precision-4.json"
    p6 = unit_parent / "precision-6.json"

    with pytest.raises(sensitivity.SensitivityError, match="explicit cuda"):
        sensitivity.validate_execution_request(
            unit_id=unit_id, precision=4, device="cpu", output=p4, plan=plan
        )
    with pytest.raises(sensitivity.SensitivityError, match="exact result path"):
        sensitivity.validate_execution_request(
            unit_id=unit_id, precision=4, device="cuda:0", output=tmp_path / "wrong.json", plan=plan
        )
    sensitivity.validate_execution_request(
        unit_id=unit_id, precision=4, device="cuda:0", output=p4, plan=plan
    )
    with pytest.raises(sensitivity.MissingEvidence, match="precision 4 evidence"):
        sensitivity.validate_execution_request(
            unit_id=unit_id, precision=6, device="cuda:0", output=p6, plan=plan
        )

    failed = _fake_result(plan, unit_id, 4, passes=False)
    digest = sensitivity.persist_unit_result(failed, p4, plan)
    assert digest == hashlib.sha256(p4.read_bytes()).hexdigest()
    assert not list(unit_parent.glob("*.tmp"))
    sensitivity.validate_execution_request(
        unit_id=unit_id, precision=6, device="cuda:0", output=p6, plan=plan
    )
    mismatched = _fake_result(plan, unit_id, 6, passes=True)
    mismatched["paired_contexts"][0]["control"]["primary_logits_sha256"] = _hex("mismatch")
    mismatched["paired_contexts"][0]["control"]["repeat_logits_sha256"] = _hex("mismatch")
    with pytest.raises(sensitivity.SensitivityError, match="control pairing drifted"):
        sensitivity.persist_unit_result(mismatched, p6, plan)
    assert not p6.exists()
    sensitivity.persist_unit_result(_fake_result(plan, unit_id, 6, passes=True), p6, plan)
    assert p6.is_file()
    with pytest.raises(sensitivity.SensitivityError, match="overwrite"):
        sensitivity.persist_unit_result(failed, p4, plan)


def test_interrupted_atomic_persistence_leaves_no_complete_or_temporary_file(
    monkeypatch, tmp_path, plan
):
    parent = tmp_path / "results"
    monkeypatch.setattr(sensitivity, "RESULT_PARENT", parent)
    monkeypatch.setattr(sensitivity, "AGGREGATION_OUTPUT", parent / "aggregation.json")
    unit_id = plan["units"][0]["unit_id"]
    unit_parent = parent / unit_id
    unit_parent.mkdir(parents=True)
    destination = unit_parent / "precision-4.json"

    def interrupt(*_args, **_kwargs):
        raise OSError("simulated interruption")

    monkeypatch.setattr(sensitivity.os, "link", interrupt)
    with pytest.raises(OSError, match="simulated interruption"):
        sensitivity.persist_unit_result(
            _fake_result(plan, unit_id, 4, passes=True), destination, plan
        )
    assert not destination.exists()
    assert list(unit_parent.iterdir()) == []


def test_dispatch_forbids_fallback_after_precision_4_pass(monkeypatch, tmp_path, plan):
    parent = tmp_path / "results"
    monkeypatch.setattr(sensitivity, "RESULT_PARENT", parent)
    monkeypatch.setattr(sensitivity, "AGGREGATION_OUTPUT", parent / "aggregation.json")
    unit_id = plan["units"][0]["unit_id"]
    unit_parent = parent / unit_id
    unit_parent.mkdir(parents=True)
    p4 = unit_parent / "precision-4.json"
    sensitivity.persist_unit_result(_fake_result(plan, unit_id, 4, passes=True), p4, plan)
    with pytest.raises(sensitivity.SensitivityError, match="forbidden after precision 4 passes"):
        sensitivity.validate_execution_request(
            unit_id=unit_id,
            precision=6,
            device="cuda:0",
            output=unit_parent / "precision-6.json",
            plan=plan,
        )


def test_aggregation_requires_exact_complete_sequential_evidence(plan):
    results = [_fake_result(plan, item["unit_id"], 4, passes=True) for item in plan["units"]]
    aggregate = sensitivity.build_aggregation(results, plan)
    assert aggregate["complete"] is True
    assert aggregate["complete_unit_count"] == 42
    assert aggregate["evidence_files_consumed"] == 42
    assert aggregate["lowest_safe_precision_counts"] == {"4": 42, "6": 0, "8": 0}

    with pytest.raises(sensitivity.MissingEvidence, match="missing precision 4"):
        sensitivity.build_aggregation(results[:-1], plan)
    with pytest.raises(sensitivity.SensitivityError, match="duplicate"):
        sensitivity.build_aggregation([*results, results[0]], plan)
    unexpected_fallback = _fake_result(plan, plan["units"][0]["unit_id"], 6, passes=True)
    with pytest.raises(sensitivity.SensitivityError, match="unexpected precision 6"):
        sensitivity.build_aggregation([*results, unexpected_fallback], plan)


def test_aggregation_accepts_only_required_fallback_and_equal_forced_8_controls(plan):
    results = []
    first_id = plan["units"][0]["unit_id"]
    for spec in plan["units"]:
        if spec["unit_id"] == first_id:
            results.append(_fake_result(plan, first_id, 4, passes=False))
            results.append(_fake_result(plan, first_id, 6, passes=True))
        else:
            results.append(_fake_result(plan, spec["unit_id"], 4, passes=True))
    aggregate = sensitivity.build_aggregation(results, plan)
    assert aggregate["evidence_files_consumed"] == 43
    assert aggregate["lowest_safe_precision_counts"] == {"4": 41, "6": 1, "8": 0}

    changed = copy.deepcopy(results)
    fallback = next(
        item for item in changed if item["unit_id"] == first_id and item["precision"] == 6
    )
    fallback["paired_contexts"][0]["control"]["primary_logits_sha256"] = _hex("conflict")
    fallback["paired_contexts"][0]["control"]["repeat_logits_sha256"] = _hex("conflict")
    sensitivity.validate_unit_result(fallback, plan)
    with pytest.raises(sensitivity.SensitivityError, match="fallback control pairing"):
        sensitivity.build_aggregation(changed, plan)


def test_aggregation_persistence_is_atomic_and_no_overwrite(monkeypatch, tmp_path, plan):
    parent = tmp_path / "results"
    parent.mkdir()
    monkeypatch.setattr(sensitivity, "RESULT_PARENT", parent)
    monkeypatch.setattr(sensitivity, "AGGREGATION_OUTPUT", parent / "aggregation.json")
    results = [_fake_result(plan, item["unit_id"], 4, passes=True) for item in plan["units"]]
    aggregation = sensitivity.build_aggregation(results, plan)
    destination = parent / "aggregation.json"
    digest = sensitivity.persist_aggregation(aggregation, destination, results, plan)
    assert digest == hashlib.sha256(destination.read_bytes()).hexdigest()
    assert not list(parent.glob("*.tmp"))
    with pytest.raises(sensitivity.SensitivityError, match="overwrite"):
        sensitivity.persist_aggregation(aggregation, destination, results, plan)


def test_aggregation_rejects_mixed_study_identity(plan):
    results = [_fake_result(plan, item["unit_id"], 4, passes=True) for item in plan["units"]]
    results[0]["study_id"] = "0" * 64
    with pytest.raises(sensitivity.SensitivityError, match="study identity"):
        sensitivity.build_aggregation(results, plan)


def test_hardware_software_provenance_is_exact_complete_and_compatible(plan):
    result = _fake_result(plan, plan["units"][0]["unit_id"], 4)
    assert tuple(result["hardware"]) == sensitivity.HARDWARE_KEYS
    sensitivity.validate_unit_result(result, plan)

    missing = copy.deepcopy(result)
    missing["hardware"].pop("python_version")
    with pytest.raises(sensitivity.SensitivityError, match="fields drifted"):
        sensitivity.validate_unit_result(missing, plan)

    empty = copy.deepcopy(result)
    empty["hardware"]["transformers_version"] = ""
    with pytest.raises(sensitivity.SensitivityError, match="identity is missing"):
        sensitivity.validate_unit_result(empty, plan)

    wrong_device = copy.deepcopy(result)
    wrong_device["hardware"]["device_index"] = 1
    with pytest.raises(sensitivity.SensitivityError, match="device index"):
        sensitivity.validate_unit_result(wrong_device, plan)

    wrong_gpu = copy.deepcopy(result)
    wrong_gpu["hardware"]["gpu_model"] = "Other GPU"
    with pytest.raises(sensitivity.SensitivityError, match="comparable GPU"):
        sensitivity.validate_unit_result(wrong_gpu, plan)


def test_resume_state_is_deterministic_nonmutating_and_starts_with_all_precision_4(
    monkeypatch, tmp_path, plan
):
    parent = tmp_path / "absent-results"
    monkeypatch.setattr(sensitivity, "RESULT_PARENT", parent)
    monkeypatch.setattr(sensitivity, "AGGREGATION_OUTPUT", parent / "aggregation.json")
    first = sensitivity.build_resume_state(plan)
    second = sensitivity.build_resume_state(plan)
    assert first == second
    assert first["result_parent_present"] is False
    assert first["next_action_counts"] == {
        "run_precision_4": 42,
        "run_precision_6": 0,
        "complete": 0,
    }
    assert [item["unit_id"] for item in first["unit_actions"]] == [
        item["unit_id"] for item in plan["units"]
    ]
    assert all(item["next_action"] == "run_precision_4" for item in first["unit_actions"])
    assert not parent.exists()


def test_resume_state_classifies_valid_progress_in_authoritative_order(monkeypatch, tmp_path, plan):
    parent = tmp_path / "results"
    monkeypatch.setattr(sensitivity, "RESULT_PARENT", parent)
    monkeypatch.setattr(sensitivity, "AGGREGATION_OUTPUT", parent / "aggregation.json")
    first_id, second_id, third_id = [item["unit_id"] for item in plan["units"][:3]]
    for unit_id in (first_id, second_id, third_id):
        (parent / unit_id).mkdir(parents=True)
    sensitivity.persist_unit_result(
        _fake_result(plan, first_id, 4, passes=True),
        parent / first_id / "precision-4.json",
        plan,
    )
    sensitivity.persist_unit_result(
        _fake_result(plan, second_id, 4, passes=False),
        parent / second_id / "precision-4.json",
        plan,
    )
    sensitivity.persist_unit_result(
        _fake_result(plan, third_id, 4, passes=False),
        parent / third_id / "precision-4.json",
        plan,
    )
    sensitivity.persist_unit_result(
        _fake_result(plan, third_id, 6, passes=True),
        parent / third_id / "precision-6.json",
        plan,
    )
    state = sensitivity.build_resume_state(plan)
    actions = {item["unit_id"]: item for item in state["unit_actions"]}
    assert actions[first_id]["next_action"] == "complete"
    assert actions[first_id]["lowest_safe_precision"] == 4
    assert actions[second_id]["next_action"] == "run_precision_6"
    assert actions[second_id]["existing_precisions"] == [4]
    assert actions[third_id]["next_action"] == "complete"
    assert actions[third_id]["lowest_safe_precision"] == 6
    assert state["next_action_counts"] == {
        "run_precision_4": 39,
        "run_precision_6": 1,
        "complete": 2,
    }


def test_resume_rejects_interrupted_temporary_and_wrongly_named_evidence(
    monkeypatch, tmp_path, plan
):
    parent = tmp_path / "results"
    monkeypatch.setattr(sensitivity, "RESULT_PARENT", parent)
    monkeypatch.setattr(sensitivity, "AGGREGATION_OUTPUT", parent / "aggregation.json")
    unit_id = plan["units"][0]["unit_id"]
    unit_parent = parent / unit_id
    unit_parent.mkdir(parents=True)
    temporary = unit_parent / ".precision-4.json.interrupted.tmp"
    temporary.write_text("partial")
    with pytest.raises(sensitivity.SensitivityError, match="temporary result evidence"):
        sensitivity.build_resume_state(plan)
    temporary.unlink()
    wrong = unit_parent / "result.json"
    wrong.write_bytes(sensitivity.serialize(_fake_result(plan, unit_id, 4)))
    with pytest.raises(sensitivity.SensitivityError, match="unexpected or temporary"):
        sensitivity.build_resume_state(plan)


def test_resume_rejects_symlinked_result_paths(monkeypatch, tmp_path, plan):
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    monkeypatch.setattr(sensitivity, "RESULT_PARENT", linked_parent)
    monkeypatch.setattr(sensitivity, "AGGREGATION_OUTPUT", linked_parent / "aggregation.json")
    with pytest.raises(sensitivity.SensitivityError, match="not a real directory"):
        sensitivity.build_resume_state(plan)


def test_resume_and_aggregation_reject_incompatible_execution_provenance(
    monkeypatch, tmp_path, plan
):
    parent = tmp_path / "results"
    monkeypatch.setattr(sensitivity, "RESULT_PARENT", parent)
    monkeypatch.setattr(sensitivity, "AGGREGATION_OUTPUT", parent / "aggregation.json")
    first_id, second_id = [item["unit_id"] for item in plan["units"][:2]]
    first = _fake_result(plan, first_id, 4)
    second = _fake_result(plan, second_id, 4)
    second["hardware"]["driver_version"] = "different-driver"
    for result in (first, second):
        destination = parent / result["unit_id"] / "precision-4.json"
        destination.parent.mkdir(parents=True)
        destination.write_bytes(sensitivity.serialize(result))
    with pytest.raises(sensitivity.SensitivityError, match="cross-result hardware/software"):
        sensitivity.build_resume_state(plan)
    remaining = [_fake_result(plan, item["unit_id"], 4) for item in plan["units"][2:]]
    with pytest.raises(sensitivity.SensitivityError, match="cross-result hardware/software"):
        sensitivity.build_aggregation([first, second, *remaining], plan)


def test_aggregation_loader_requires_every_canonical_directory_and_filename(
    monkeypatch, tmp_path, plan
):
    parent = tmp_path / "results"
    monkeypatch.setattr(sensitivity, "RESULT_PARENT", parent)
    monkeypatch.setattr(sensitivity, "AGGREGATION_OUTPUT", parent / "aggregation.json")
    for spec in plan["units"]:
        (parent / spec["unit_id"]).mkdir(parents=True)
    first_id = plan["units"][0]["unit_id"]
    (parent / first_id / "arbitrary.json").write_bytes(
        sensitivity.serialize(_fake_result(plan, first_id, 4))
    )
    with pytest.raises(sensitivity.SensitivityError, match="unexpected or temporary"):
        sensitivity.load_results_for_aggregation(plan)


@pytest.mark.parametrize(
    "payload,match",
    [
        ('{"schema":"one","schema":"two"}', "duplicate JSON key"),
        ('{"metric":NaN}', "non-finite JSON value"),
    ],
)
def test_strict_json_rejects_duplicate_and_nonfinite_values(tmp_path, payload, match):
    path = tmp_path / "invalid.json"
    path.write_text(payload)
    with pytest.raises(sensitivity.SensitivityError, match=match):
        sensitivity._load_json(path)


def test_default_cli_is_standard_library_only_deterministic_and_inert():
    code = f"""
import importlib.abc, runpy, sys
blocked={{'torch','transformers','datasets','any_precision','any_precision_ext'}}
class Blocker(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  if fullname.split('.')[0] in blocked: raise AssertionError(fullname)
  return None
sys.meta_path.insert(0, Blocker())
sys.argv={[str(ROOT / "scripts/run_s11d_block_sensitivity.py"), "--plan"]!r}
runpy.run_path(sys.argv[0], run_name='__main__')
"""
    first = subprocess.run(
        [sys.executable, "-I", "-c", code], cwd=ROOT, text=True, capture_output=True, check=False
    )
    second = subprocess.run(
        [sys.executable, "-I", "-c", code], cwd=ROOT, text=True, capture_output=True, check=False
    )
    assert (first.returncode, first.stdout, first.stderr) == (
        second.returncode,
        second.stdout,
        second.stderr,
    )
    assert first.returncode == 0
    value = json.loads(first.stdout)
    assert value["design"]["unit_count"] == 42
    assert value["non_execution_audit"]["cuda_activity"] is False
    assert not os.path.lexists(sensitivity.RESULT_PARENT)


def test_resume_cli_is_inert_and_reports_authoritative_next_actions():
    completed = subprocess.run(
        [sys.executable, "scripts/run_s11d_block_sensitivity.py", "--resume-plan"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    value = json.loads(completed.stdout)
    assert value["schema"] == sensitivity.RESUME_SCHEMA
    assert value["next_action_counts"] == {
        "complete": 0,
        "run_precision_4": 42,
        "run_precision_6": 0,
    }
    assert value["non_mutating"] is True
    assert not os.path.lexists(sensitivity.RESULT_PARENT)
