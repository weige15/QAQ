from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from qaq.evaluation import lookahead_broader_quality as broad
from qaq.evaluation import lookahead_quality_runner as shared

ROOT = Path(__file__).resolve().parents[2]
CONFIG, _ = broad.load_protocol(require_results_absent=True)


def _entry(name: str) -> dict:
    return {
        "name": name,
        "kind": "parameter",
        "dtype": "torch.float32",
        "shape": [1],
        "requires_grad": False,
        "gradient_absent": True,
        "value_sha256": hashlib.sha256(name.encode()).hexdigest(),
    }


def _freeze_audit() -> dict:
    components = {}
    for component in ("teacher", "packed_weights_and_buffers", "non_router_base", "router"):
        entries = [_entry(component)]
        digest = shared._digest(entries)
        components[component] = {
            "before_entries": copy.deepcopy(entries),
            "after_entries": copy.deepcopy(entries),
            "parameter_count": 1,
            "buffer_count": 0,
            "before_aggregate_sha256": digest,
            "after_aggregate_sha256": digest,
            "hashes_equal": True,
        }
    hashes = {name: item["before_aggregate_sha256"] for name, item in components.items()}
    return {
        "components": components,
        "before_hashes": dict(hashes),
        "after_hashes": dict(hashes),
        "hashes_equal": True,
        "optimizer_absent": True,
        "gradients_absent": True,
    }


class TinyRuntime:
    evidence_label = "test-only structural evidence"

    def __init__(self, *, treatment_scale: float = 1.0) -> None:
        self.treatment_scale = treatment_scale
        self.calls: list[tuple[str, int, str]] = []
        self.closed = False
        self.mode_id = ""
        self.historical_maps, self.historical_quality = broad._historical_overlap()

    def prepare(self, protocol, mode, device, requests):
        assert protocol is CONFIG
        assert device == "cuda:0"
        assert [item["request_id"] for item in requests] == list(broad.REQUEST_IDS)
        self.mode_id = mode["id"]

    def hardware_evidence(self):
        return {
            "cuda_device": "cuda:0",
            "device_index": 0,
            "gpu_model": "NVIDIA GeForce RTX 3090",
            "driver_version": "test",
            "cuda_runtime_version": "test",
            "pytorch_version": "test",
            "transformers_version": "test",
            "python_version": "3.12.3",
        }

    def identity_evidence(self):
        return broad._expected_identities(CONFIG)

    def run_request(self, *, mode, request, repeat_index, device):
        assert device == "cuda:0"
        request_id = request["request_id"]
        self.calls.append((mode["id"], repeat_index, request_id))
        if request_id in self.historical_maps:
            routes = copy.deepcopy(self.historical_maps[request_id])
            base_quality = self.historical_quality[request_id]
            kl = base_quality["kl"]
            mae = base_quality["mean_absolute_logit_error"]
            maximum = base_quality["maximum_absolute_logit_error"]
        else:
            routes = [
                {
                    "request_id": request_id,
                    "target_layer": layer,
                    "unit_type": unit,
                    "selected_bits": 4 if (layer + broad.UNIT_TYPES.index(unit)) % 3 == 0 else 8,
                }
                for layer in range(36)
                for unit in broad.UNIT_TYPES
            ]
            index = broad.REQUEST_IDS.index(request_id)
            kl = 0.1 + index / 100
            mae = 0.2 + index / 100
            maximum = 0.5 + index / 100
        if mode["id"] == broad.MODE_IDS[1]:
            kl *= self.treatment_scale
            mae *= self.treatment_scale
            maximum *= self.treatment_scale
            if request_id == "validation-270":
                target = next(
                    item
                    for item in routes
                    if item["target_layer"] == 1 and item["unit_type"] == "attention"
                )
                target["selected_bits"] = 12 - target["selected_bits"]
        return {
            "request_id": request_id,
            "full_input_ids_sha256": request["token_digest_sha256"],
            "teacher_logits_digest": shared._digest([request_id, "teacher"]),
            "student_logits_digest": shared._digest([mode["id"], request_id, "student"]),
            "teacher_logits_shape": [1, 64, 16],
            "student_logits_shape": [1, 64, 16],
            "finite_teacher_logits": True,
            "finite_student_logits": True,
            "kl": kl,
            "mean_absolute_logit_error": mae,
            "maximum_absolute_logit_error": maximum,
            "routes": routes,
            "provenance": [
                shared._expected_provenance(mode["id"], request_id, layer, unit)
                for layer in range(36)
                for unit in broad.UNIT_TYPES
            ],
            "request_cleanup": {
                "state_ended": True,
                "routes_released": True,
                "features_released": True,
                "probabilities_released": True,
                "provenance_released": True,
                "passed": True,
            },
        }

    def freeze_audit(self):
        return _freeze_audit()

    def close(self):
        self.closed = True


def _result(mode_id: str, *, treatment_scale: float = 1.0) -> dict:
    return broad.execute_mode_with_runtime(
        TinyRuntime(treatment_scale=treatment_scale),
        config=CONFIG,
        mode_id=mode_id,
        device="cuda:0",
    )


def _refresh_quality(result: dict) -> None:
    records = [
        {
            "request_id": item["request_id"],
            "kl": item["kl"],
            "mean_absolute_logit_error": item["mean_absolute_logit_error"],
            "maximum_absolute_logit_error": item["maximum_absolute_logit_error"],
        }
        for item in result["repeats"][0]["requests"]
    ]
    result["quality"] = {
        "per_request": records,
        "aggregate_kl": sum(item["kl"] for item in records) / 12,
        "aggregate_mean_absolute_logit_error": sum(
            item["mean_absolute_logit_error"] for item in records
        )
        / 12,
        "aggregate_maximum_absolute_logit_error": sum(
            item["maximum_absolute_logit_error"] for item in records
        )
        / 12,
        "all_finite": True,
    }


def test_frozen_contract_fixture_and_authoritative_manifest_validate():
    config, digest = broad.load_protocol(require_results_absent=True)
    assert digest == broad.EXPECTED_CONFIG_SHA256
    requests = broad.fixed_requests(config)
    assert len(requests) == 12
    assert tuple(item["request_id"] for item in requests) == broad.REQUEST_IDS
    assert tuple(item["token_digest_sha256"] for item in requests) == broad.TOKEN_DIGESTS
    assert all(len(item["full_input_ids"]) == 64 for item in requests)


def test_protocol_and_fixture_byte_mutations_fail_closed(tmp_path, monkeypatch):
    changed = tmp_path / "config.json"
    changed.write_bytes(broad.DEFAULT_CONFIG.read_bytes() + b" ")
    with pytest.raises(broad.BroaderQualityError, match="only the frozen"):
        broad.load_protocol(changed)
    monkeypatch.setattr(broad, "EXPECTED_FIXTURE_SHA256", "0" * 64)
    with pytest.raises(broad.BroaderQualityError, match="fixture SHA-256"):
        broad.load_protocol()


def test_default_plan_is_byte_deterministic_heavy_import_free_and_nonexecuting():
    code = f"""
import importlib.abc, runpy, sys
blocked={{'torch','transformers','datasets','any_precision','any_precision_ext'}}
class Blocker(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  if fullname.split('.')[0] in blocked: raise AssertionError(fullname)
  return None
sys.meta_path.insert(0, Blocker())
sys.argv=[{str(ROOT / "scripts/run_lookahead_broader_quality.py")!r}]
runpy.run_path(sys.argv[0], run_name='__main__')
"""
    command = [sys.executable, "-I", "-c", code]
    first = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    second = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    assert (first.returncode, first.stdout, first.stderr) == (
        second.returncode,
        second.stdout,
        second.stderr,
    )
    assert first.returncode == 0
    plan = json.loads(first.stdout)
    assert plan["request_order"] == list(broad.REQUEST_IDS)
    assert plan["mode_order"] == list(broad.MODE_IDS)
    assert len(plan["child_commands"]) == 2
    for field in (
        "model_loading",
        "dataset_loading",
        "runtime_tokenization",
        "cuda_activity",
        "experiment_execution",
        "training",
        "benchmarking",
        "result_write_activity",
    ):
        assert plan[field] is False
    assert not os.path.lexists(ROOT / "docs/results/s11c_broader_quality")


def test_invalid_dispatch_stops_before_production_import(tmp_path):
    code = f"""
import importlib.abc, runpy, sys
class Blocker(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  if fullname.endswith('lookahead_quality_runtime'): raise AssertionError(fullname)
  return None
sys.meta_path.insert(0, Blocker())
sys.argv=[{str(ROOT / "scripts/run_lookahead_broader_quality.py")!r},'--execute-mode','same_unit_control','--device','cpu','--output',{str(tmp_path / "wrong.json")!r}]
runpy.run_path(sys.argv[0], run_name='__main__')
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", code],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 1
    assert json.loads(completed.stdout)["classification"] == "REVISE"


def test_injected_scheduler_covers_twelve_requests_two_repeats_and_864_routes():
    runtime = TinyRuntime()
    result = broad.execute_mode_with_runtime(
        runtime,
        config=CONFIG,
        mode_id=broad.MODE_IDS[0],
        device="cuda:0",
    )
    assert runtime.closed is True
    assert len(runtime.calls) == 24
    assert len(result["inputs"]) == 12
    assert len(result["routes"]["target_owned_route_maps"]) == 12
    assert sum(len(item["routes"]) for item in result["routes"]["target_owned_route_maps"]) == 864
    broad.validate_mode_result(result, CONFIG)


def test_aggregation_recomputes_quality_and_complete_route_diagnostics():
    control = _result(broad.MODE_IDS[0])
    treatment = _result(broad.MODE_IDS[1])
    aggregate = broad.build_aggregation(control, treatment, CONFIG)
    assert aggregate["classification"] == "CONTINUE"
    assert len(aggregate["paired_quality"]["paired_request_kl"]) == 12
    diagnostics = aggregate["route_diagnostics"]
    assert len(diagnostics["per_request"]) == 12
    assert diagnostics["changed_target_unit_count"] == 1
    assert set(diagnostics["aggregate"]) == {"overall", "attention", "ffn"}
    assert diagnostics["aggregate"]["overall"]["hamming_count"] == 1
    assert diagnostics["aggregate"]["overall"]["transition_4_to_8"] in (0, 1)
    assert diagnostics["aggregate"]["overall"]["transition_8_to_4"] in (0, 1)
    broad.validate_aggregation_result(aggregate, control, treatment, CONFIG)


@pytest.mark.parametrize("failure", ["aggregate_kl", "one_request_kl", "aggregate_mae"])
def test_each_quality_margin_failure_is_stop(failure):
    control = _result(broad.MODE_IDS[0])
    treatment = _result(broad.MODE_IDS[1])
    requests = treatment["repeats"]
    if failure == "aggregate_kl":
        for repeat in requests:
            for item in repeat["requests"]:
                item["kl"] *= 2
    elif failure == "one_request_kl":
        for repeat in requests:
            repeat["requests"][1]["kl"] *= 2
    else:
        for repeat in requests:
            for item in repeat["requests"]:
                item["mean_absolute_logit_error"] *= 2
    for index, repeat in enumerate(requests):
        treatment["repeats"][index] = broad._repeat_record(
            repeat["requests"],
            treatment["routes"]["target_owned_route_maps"],
            treatment["provenance"],
            index,
        )
    _refresh_quality(treatment)
    assert broad.build_aggregation(control, treatment, CONFIG)["classification"] == "STOP"


def test_missing_results_pause_and_malformed_complete_evidence_revises(tmp_path):
    aggregate, report = broad.aggregate_paths(
        control_path=tmp_path / "missing-control.json",
        treatment_path=tmp_path / "missing-treatment.json",
    )
    assert aggregate is None and report["classification"] == "PAUSE"
    control_path = tmp_path / "control.json"
    treatment_path = tmp_path / "treatment.json"
    control_path.write_text("{}")
    treatment_path.write_text("{}")
    aggregate, report = broad.aggregate_paths(
        control_path=control_path,
        treatment_path=treatment_path,
    )
    assert aggregate is None and report["classification"] == "REVISE"
    treatment_path.unlink()
    aggregate, report = broad.aggregate_paths(
        control_path=control_path,
        treatment_path=treatment_path,
    )
    assert aggregate is None and report["classification"] == "PAUSE"


def test_repeat_route_identity_and_prohibited_work_mutations_revise():
    result = _result(broad.MODE_IDS[1])
    mutations = []
    bad = copy.deepcopy(result)
    bad["repeats"][1]["requests"][0]["student_logits_digest"] = "0" * 64
    mutations.append(bad)
    bad = copy.deepcopy(result)
    bad["routes"]["target_owned_route_maps"][0]["routes"][0]["selected_bits"] = 6
    mutations.append(bad)
    bad = copy.deepcopy(result)
    bad["identities"]["model"]["revision"] = "wrong"
    mutations.append(bad)
    bad = copy.deepcopy(result)
    bad["prohibited_work_audit"]["training_or_retraining_observed"] = True
    mutations.append(bad)
    for value in mutations:
        with pytest.raises((broad.BroaderQualityError, shared.LookaheadQualityError)):
            broad.validate_mode_result(value, CONFIG)


@pytest.mark.parametrize(
    "field",
    (
        "cuda_device",
        "device_index",
        "gpu_model",
        "driver_version",
        "cuda_runtime_version",
        "pytorch_version",
        "transformers_version",
        "python_version",
    ),
)
def test_hardware_schema_rejects_every_omitted_field(field):
    result = _result(broad.MODE_IDS[1])
    result["hardware"].pop(field)
    with pytest.raises(broad.BroaderQualityError, match="hardware/software identity fields"):
        broad.validate_mode_result(result, CONFIG)


def test_hardware_schema_rejects_extra_field():
    result = _result(broad.MODE_IDS[1])
    result["hardware"]["unexpected"] = "drift"
    with pytest.raises(broad.BroaderQualityError, match="hardware/software identity fields"):
        broad.validate_mode_result(result, CONFIG)


@pytest.mark.parametrize(
    "field",
    (
        "gpu_model",
        "driver_version",
        "cuda_runtime_version",
        "pytorch_version",
        "transformers_version",
        "python_version",
    ),
)
def test_hardware_schema_rejects_empty_identity(field):
    result = _result(broad.MODE_IDS[1])
    result["hardware"][field] = ""
    with pytest.raises(broad.BroaderQualityError, match="hardware/software identity"):
        broad.validate_mode_result(result, CONFIG)


@pytest.mark.parametrize(
    ("scope", "operation", "field"),
    (
        ("repeat", "omit", "logits_digest"),
        ("repeat", "extra", "unexpected"),
        ("quality", "omit", "all_finite"),
        ("quality", "extra", "unexpected"),
        ("routes", "omit", "summaries"),
        ("routes", "extra", "unexpected"),
        ("audit", "omit", "passed"),
        ("audit", "extra", "unexpected"),
        ("execution_audit", "omit", "dataset_access"),
        ("execution_audit", "extra", "unexpected"),
    ),
)
def test_nested_schema_rejects_omitted_and_extra_fields(scope, operation, field):
    result = _result(broad.MODE_IDS[1])
    targets = {
        "repeat": result["repeats"][0],
        "quality": result["quality"],
        "routes": result["routes"],
        "audit": result["prohibited_work_audit"],
        "execution_audit": result["prohibited_work_audit"]["execution"],
    }
    target = targets[scope]
    if operation == "omit":
        target.pop(field)
    else:
        target[field] = "drift"
    with pytest.raises(broad.BroaderQualityError):
        broad.validate_mode_result(result, CONFIG)


@pytest.mark.parametrize(
    ("scope", "field", "value"),
    (
        ("execution", "fresh_process", False),
        ("execution", "use_cache", True),
        ("execution", "inference_only", False),
        ("execution", "runtime_tokenization", True),
        ("execution", "dataset_access", True),
        ("audit", "all_requests_cleaned", False),
        ("audit", "training_or_retraining_observed", True),
        ("audit", "checkpoint_created", True),
        ("audit", "performance_or_resource_measurement_observed", True),
        ("audit", "passed", False),
        ("request_cleanup", "state_ended", False),
    ),
)
def test_execution_and_prohibited_work_claim_drift_is_rejected(scope, field, value):
    result = _result(broad.MODE_IDS[1])
    targets = {
        "execution": result["prohibited_work_audit"]["execution"],
        "audit": result["prohibited_work_audit"],
        "request_cleanup": result["repeats"][0]["requests"][0]["request_cleanup"],
    }
    targets[scope][field] = value
    with pytest.raises(broad.BroaderQualityError):
        broad.validate_mode_result(result, CONFIG)


def test_shared_atomic_boundary_validates_no_overwrite_and_cleans_temp(tmp_path):
    destination = tmp_path / "result.json"
    policy = shared.PersistencePolicy(destination, tmp_path)
    value = {"valid": True}
    digest = shared.persist_atomically(
        value,
        destination,
        policy=policy,
        validator=lambda payload: broad._require(payload == value, "invalid"),
    )
    assert digest == hashlib.sha256(destination.read_bytes()).hexdigest()
    with pytest.raises(shared.LookaheadQualityError, match="existing file"):
        shared.persist_atomically(
            value,
            destination,
            policy=policy,
            validator=lambda payload: None,
        )
    assert not list(tmp_path.glob("*.tmp"))


def test_exact_s11c_destination_parent_is_not_created_by_dispatch_or_plan():
    parent = ROOT / "docs/results/s11c_broader_quality"
    assert not os.path.lexists(parent)
    with pytest.raises(shared.LookaheadQualityError, match="parent is absent"):
        broad.validate_dispatch(
            mode_id=broad.MODE_IDS[0],
            device="cuda:0",
            output=ROOT / broad.OUTPUTS[broad.MODE_IDS[0]],
        )
    assert not os.path.lexists(parent)
