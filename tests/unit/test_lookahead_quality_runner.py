from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from qaq.evaluation import lookahead_quality_runner as runner

ROOT = Path(__file__).resolve().parents[2]
CONFIG, _ = runner.load_protocol(runner.DEFAULT_CONFIG, require_results_absent=True)


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
    for component in (
        "teacher",
        "packed_weights_and_buffers",
        "non_router_base",
        "router",
    ):
        entries = [_entry(component)]
        digest = runner._digest(entries)
        components[component] = {
            "before_entries": copy.deepcopy(entries),
            "after_entries": copy.deepcopy(entries),
            "parameter_count": 1,
            "buffer_count": 0,
            "before_aggregate_sha256": digest,
            "after_aggregate_sha256": digest,
            "hashes_equal": True,
        }
    hashes = {name: audit["before_aggregate_sha256"] for name, audit in components.items()}
    return {
        "components": components,
        "before_hashes": dict(hashes),
        "after_hashes": dict(hashes),
        "hashes_equal": True,
        "optimizer_absent": True,
        "gradients_absent": True,
    }


class TinyRuntime:
    """Injected deterministic runtime: test-only structural evidence, never Qwen evidence."""

    evidence_label = "test-only structural evidence"

    def __init__(
        self,
        *,
        kl: dict[str, float] | None = None,
        mae: dict[str, float] | None = None,
        flip_layer_zero: bool = False,
        hardware_suffix: str = "",
    ) -> None:
        self.kl = kl or {"validation-3": 0.10, "validation-1000": 0.20}
        self.mae = mae or {"validation-3": 0.30, "validation-1000": 0.40}
        self.flip_layer_zero = flip_layer_zero
        self.hardware_suffix = hardware_suffix
        self.mode = None
        self.closed = False
        self.calls = []

    def prepare(self, protocol, mode, device, requests):
        assert protocol is CONFIG
        assert device == "cuda:0"
        assert [item["request_id"] for item in requests] == list(runner.REQUEST_IDS)
        self.mode = mode

    def hardware_evidence(self):
        return {
            "cuda_device": "cuda:0",
            "device_index": 0,
            "gpu_model": "NVIDIA GeForce RTX 3090" + self.hardware_suffix,
            "driver_version": "580.159.03",
            "cuda_runtime_version": "12.1",
            "pytorch_version": "test-only",
            "transformers_version": "test-only",
            "python_version": "3.12.3",
        }

    def identity_evidence(self):
        return runner._expected_identities(CONFIG)

    def run_request(self, *, mode, request, repeat_index, device):
        del device
        request_id = request["request_id"]
        self.calls.append((mode["id"], repeat_index, request_id))
        routes = copy.deepcopy(runner._historical_routes()[request_id])
        if mode["id"] == runner.MODE_IDS[1]:
            for layer, unit in ((1, "attention"), (2, "ffn")):
                record = next(
                    item
                    for item in routes
                    if item["target_layer"] == layer and item["unit_type"] == unit
                )
                record["selected_bits"] = 12 - record["selected_bits"]
            if self.flip_layer_zero:
                routes[0]["selected_bits"] = 12 - routes[0]["selected_bits"]
        teacher_digest = runner._digest([request_id, "teacher"])
        student_digest = runner._digest([mode["id"], request_id, "student"])
        return {
            "request_id": request_id,
            "full_input_ids_sha256": request["token_digest_sha256"],
            "teacher_logits_digest": teacher_digest,
            "student_logits_digest": student_digest,
            "teacher_logits_shape": [1, 64, 16],
            "student_logits_shape": [1, 64, 16],
            "finite_teacher_logits": True,
            "finite_student_logits": True,
            "kl": self.kl[request_id],
            "mean_absolute_logit_error": self.mae[request_id],
            "maximum_absolute_logit_error": self.mae[request_id] + 0.5,
            "routes": routes,
            "provenance": [
                runner._expected_provenance(mode["id"], request_id, layer, unit)
                for layer in range(36)
                for unit in runner.UNIT_TYPES
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


def _result(mode_id: str, **runtime_kwargs) -> dict:
    return runner.execute_mode_with_runtime(
        TinyRuntime(**runtime_kwargs),
        config=CONFIG,
        mode_id=mode_id,
        device="cuda:0",
    )


def _refresh_quality(result: dict) -> None:
    per_request = [
        {
            "request_id": item["request_id"],
            "kl": item["kl"],
            "mean_absolute_logit_error": item["mean_absolute_logit_error"],
            "maximum_absolute_logit_error": item["maximum_absolute_logit_error"],
        }
        for item in result["repeats"][0]["requests"]
    ]
    result["quality"] = {
        "per_request": per_request,
        "aggregate_kl": sum(item["kl"] for item in per_request) / 2,
        "aggregate_mean_absolute_logit_error": sum(
            item["mean_absolute_logit_error"] for item in per_request
        )
        / 2,
        "aggregate_maximum_absolute_logit_error": sum(
            item["maximum_absolute_logit_error"] for item in per_request
        )
        / 2,
        "all_finite": True,
    }


def test_inert_plan_is_deterministic_and_has_no_side_effects():
    before = {
        path: os.path.lexists(ROOT / path)
        for path in (*runner.OUTPUTS.values(), runner.AGGREGATION_OUTPUT)
    }
    first = runner.plan()
    second = runner.plan()
    assert first == second
    assert first["mode_order"] == list(runner.MODE_IDS)
    assert [
        command[command.index("--execute-mode") + 1] for command in first["child_commands"]
    ] == list(runner.MODE_IDS)
    assert first["fresh_child_processes_per_mode"] == 1
    assert first["repeats_within_fresh_child"] == 2
    assert first["model_loading"] is False
    assert first["cuda_activity"] is False
    assert first["pilot_execution"] is False
    assert first["result_write_activity"] is False
    assert before == {path: os.path.lexists(ROOT / path) for path in before}


def test_runtime_module_import_itself_is_heavy_import_free():
    code = f"""
import importlib.abc, sys
sys.path.insert(0, {str(ROOT / "src")!r})
blocked = {{'torch','transformers','datasets','any_precision','any_precision_ext'}}
class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in blocked:
            raise AssertionError('forbidden import: ' + fullname)
        return None
sys.meta_path.insert(0, Blocker())
from qaq.evaluation.lookahead_quality_runtime import ProductionRuntime
assert ProductionRuntime.evidence_label == 'production pilot evidence'
assert not blocked.intersection(sys.modules)
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", code],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == ""
    assert completed.stderr == ""


def test_plan_subprocess_forbids_heavy_imports_and_is_byte_deterministic():
    blocker = ROOT / "tests" / "unit" / "_s11b_import_blocker.py"
    code = f"""
import importlib.abc, runpy, sys
blocked = {{'torch','transformers','datasets','any_precision','any_precision_ext'}}
class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in blocked:
            raise AssertionError('forbidden import: ' + fullname)
        return None
sys.meta_path.insert(0, Blocker())
sys.argv = [{str(ROOT / "scripts/run_lookahead_quality_pilot.py")!r}]
runpy.run_path(sys.argv[0], run_name='__main__')
"""
    command = [sys.executable, "-I", "-c", code]
    first = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    second = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    assert blocker.exists() is False
    assert (first.returncode, first.stdout, first.stderr) == (
        second.returncode,
        second.stdout,
        second.stderr,
    )
    assert first.returncode == 0, first.stderr
    payload = json.loads(first.stdout)
    assert payload["pilot_execution"] is False


@pytest.mark.parametrize(
    "arguments",
    [
        ["--execute-mode", "same_unit_control"],
        ["--execute-mode", "same_unit_control", "--device", "cpu", "--output", "/tmp/x"],
        ["--execute-mode", "same_unit_control", "--device", "cuda:0", "--output", "/tmp/x"],
        ["--execute-mode", "unknown", "--device", "cuda:0", "--output", "/tmp/x"],
        ["--aggregate", "--device", "cuda:0"],
    ],
)
def test_invalid_dispatch_fails_before_lazy_runtime_import(arguments):
    code = f"""
import importlib.abc, runpy, sys
class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'qaq.evaluation.lookahead_quality_runtime' or fullname.split('.')[0] in {{'torch','transformers','datasets','any_precision'}}:
            raise AssertionError('forbidden import: ' + fullname)
        return None
sys.meta_path.insert(0, Blocker())
sys.argv = [{str(ROOT / "scripts/run_lookahead_quality_pilot.py")!r}, *{arguments!r}]
runpy.run_path(sys.argv[0], run_name='__main__')
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", code],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "forbidden import" not in completed.stderr


@pytest.mark.parametrize("mode_id", runner.MODE_IDS)
@pytest.mark.parametrize(
    ("destination_state", "expected_reason"),
    [
        ("absent_parent", "allowed destination parent is absent"),
        ("existing_file", "destination is an existing file; refusing overwrite"),
        (
            "existing_directory",
            "destination is an existing directory; refusing overwrite",
        ),
        ("destination_symlink", "destination is a symlink; refusing overwrite"),
        ("linked_parent", "allowed destination parent is a symlink"),
    ],
)
def test_exact_canonical_dispatch_path_safety_precedes_runtime_import(
    tmp_path, mode_id, destination_state, expected_reason
):
    isolated_root = tmp_path / "isolated-root"
    canonical = isolated_root / runner.OUTPUTS[mode_id]
    code = f"""
import importlib.abc, pathlib, sys
sys.path.insert(0, {str(ROOT / "src")!r})
blocked = {{'torch','transformers','datasets','any_precision','any_precision_ext'}}
class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'qaq.evaluation.lookahead_quality_runtime' or fullname.split('.')[0] in blocked:
            raise AssertionError('forbidden import: ' + fullname)
        return None
sys.meta_path.insert(0, Blocker())
from qaq.evaluation import lookahead_quality_runner as runner
repository_root = pathlib.Path({str(ROOT)!r})
assert runner.ROOT == repository_root
runner.load_protocol(runner.DEFAULT_CONFIG, require_results_absent=False)
destination = pathlib.Path({str(canonical)!r})
runner.expected_mode_destination = lambda requested_mode: (
    destination if requested_mode == {mode_id!r} else repository_root / runner.OUTPUTS[requested_mode]
)
state = {destination_state!r}
if state == 'linked_parent':
    destination.parent.parent.mkdir(parents=True)
    linked_target = pathlib.Path({str(isolated_root / "actual-parent")!r})
    linked_target.mkdir(parents=True)
    destination.parent.symlink_to(linked_target, target_is_directory=True)
elif state != 'absent_parent':
    destination.parent.mkdir(parents=True)
    if state == 'existing_file':
        destination.write_text('preserve')
    elif state == 'existing_directory':
        destination.mkdir()
    elif state == 'destination_symlink':
        destination.symlink_to(destination.parent / 'missing-target')
try:
    runner.validate_dispatch(
        mode_id={mode_id!r},
        device='cuda:0',
        output=destination,
        config_path=runner.DEFAULT_CONFIG,
    )
except runner.LookaheadQualityError as error:
    assert {expected_reason!r} in str(error), str(error)
else:
    raise AssertionError('unsafe canonical dispatch was accepted')
assert runner.ROOT == repository_root
assert 'qaq.evaluation.lookahead_quality_runtime' not in sys.modules
assert not blocked.intersection(sys.modules)
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", code],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == ""
    assert completed.stderr == ""


def test_injected_runtime_schedule_builds_accepted_raw_evidence():
    runtime = TinyRuntime()
    result = runner.execute_mode_with_runtime(
        runtime,
        config=CONFIG,
        mode_id=runner.MODE_IDS[0],
        device="cuda:0",
    )
    runner.validate_mode_result(result, CONFIG)
    assert runtime.calls == [
        (runner.MODE_IDS[0], repeat, request_id)
        for repeat in range(2)
        for request_id in runner.REQUEST_IDS
    ]
    assert runtime.closed is True
    assert result["prohibited_work_audit"]["evidence_label"] == "test-only structural evidence"
    assert result["routes"]["historical_control_equality"] is True


@pytest.mark.parametrize(
    ("name", "mutate"),
    [
        ("schema", lambda value: value.__setitem__("schema", "wrong")),
        ("mode", lambda value: value.__setitem__("mode_id", runner.MODE_IDS[1])),
        ("protocol", lambda value: value.__setitem__("protocol_config_sha256", "0" * 64)),
        ("identity", lambda value: value["identities"]["model"].__setitem__("revision", "main")),
        ("hardware", lambda value: value["hardware"].__setitem__("gpu_model", "other")),
        ("device", lambda value: value["hardware"].__setitem__("cuda_device", "cpu")),
        ("seed", lambda value: value.__setitem__("seed", 1730)),
        ("input", lambda value: value["inputs"][0].__setitem__("token_count", 63)),
        ("repeat_count", lambda value: value["repeats"].pop()),
        ("repeat_index", lambda value: value["repeats"][0].__setitem__("repeat_index", 1)),
        ("logit_digest", lambda value: value["repeats"][0].__setitem__("logits_digest", "0" * 64)),
        (
            "repeat_determinism",
            lambda value: value["repeats"][1]["requests"][0].__setitem__(
                "student_logits_digest", "0" * 64
            ),
        ),
        (
            "finite",
            lambda value: value["repeats"][0]["requests"][0].__setitem__(
                "finite_student_logits", False
            ),
        ),
        (
            "negative_metric",
            lambda value: value["repeats"][0]["requests"][0].__setitem__("kl", -1.0),
        ),
        (
            "route_count",
            lambda value: value["routes"]["target_owned_route_maps"][0]["routes"].pop(),
        ),
        (
            "route_order",
            lambda value: value["routes"]["target_owned_route_maps"][0]["routes"].reverse(),
        ),
        (
            "route_bit",
            lambda value: value["routes"]["target_owned_route_maps"][0]["routes"][0].__setitem__(
                "selected_bits", 6
            ),
        ),
        (
            "route_summary",
            lambda value: value["routes"]["fractions_overall"][0].__setitem__("fraction_4", 0.0),
        ),
        (
            "historical",
            lambda value: value["routes"].__setitem__("historical_control_equality", False),
        ),
        (
            "provenance",
            lambda value: value["provenance"]["records_by_request"][0]["records"][2].__setitem__(
                "source_layer", 9
            ),
        ),
        (
            "cleanup",
            lambda value: value["repeats"][0]["requests"][0]["request_cleanup"].__setitem__(
                "passed", False
            ),
        ),
        (
            "state_hash",
            lambda value: value["freeze_audit"]["components"]["teacher"].__setitem__(
                "before_aggregate_sha256", "0" * 64
            ),
        ),
        (
            "requires_grad",
            lambda value: value["freeze_audit"]["components"]["router"]["after_entries"][
                0
            ].__setitem__("requires_grad", True),
        ),
        (
            "gradient",
            lambda value: value["freeze_audit"]["components"]["router"]["after_entries"][
                0
            ].__setitem__("gradient_absent", False),
        ),
        ("optimizer", lambda value: value["freeze_audit"].__setitem__("optimizer_absent", False)),
        (
            "execution",
            lambda value: value["prohibited_work_audit"]["execution"].__setitem__(
                "use_cache", True
            ),
        ),
        (
            "prohibited_flag",
            lambda value: value["prohibited_work_audit"].__setitem__("decode_observed", True),
        ),
        ("prohibited_field", lambda value: value.__setitem__("generation", {})),
    ],
)
def test_per_mode_evidence_rejections(name, mutate):
    result = _result(runner.MODE_IDS[0])
    mutate(result)
    with pytest.raises(runner.InvalidEvidence):
        runner.validate_mode_result(result, CONFIG)


@pytest.mark.parametrize(
    ("name", "mutate"),
    [
        (
            "tokenizer_repository",
            lambda value: value["identities"]["tokenizer"].__setitem__("repository", "other"),
        ),
        (
            "tokenizer_revision",
            lambda value: value["identities"]["tokenizer"].__setitem__("revision", "main"),
        ),
        (
            "packed_manifest",
            lambda value: value["identities"]["packed_artifact"].__setitem__(
                "manifest_path", "other.json"
            ),
        ),
        (
            "packed_path",
            lambda value: value["identities"]["packed_artifact"].__setitem__(
                "relative_path", "other"
            ),
        ),
        (
            "packed_file",
            lambda value: value["identities"]["packed_artifact"].__setitem__(
                "checkpoint_file", "other.bin"
            ),
        ),
        (
            "packed_hash",
            lambda value: value["identities"]["packed_artifact"].__setitem__("sha256", "0" * 64),
        ),
        (
            "any_precision_path",
            lambda value: value["identities"]["any_precision"].__setitem__(
                "submodule_path", "other"
            ),
        ),
        (
            "any_precision_commit",
            lambda value: value["identities"]["any_precision"].__setitem__("commit", "0" * 40),
        ),
        (
            "checkpoint_hash",
            lambda value: value["identities"]["router_checkpoint"].__setitem__("sha256", "0" * 64),
        ),
        (
            "checkpoint_candidates",
            lambda value: value["identities"]["router_checkpoint"].__setitem__(
                "candidate_order", [8, 4]
            ),
        ),
        (
            "checkpoint_metadata",
            lambda value: value["identities"]["router_checkpoint"].__setitem__(
                "metadata_validated", False
            ),
        ),
        (
            "checkpoint_mutability",
            lambda value: value["identities"]["router_checkpoint"].__setitem__("read_only", False),
        ),
        (
            "fixed_inputs_path",
            lambda value: value["identities"].__setitem__("fixed_inputs_path", "other.json"),
        ),
        (
            "fixed_input_digest",
            lambda value: value["inputs"][0].__setitem__("full_input_ids_sha256", "0" * 64),
        ),
        (
            "prompt_range",
            lambda value: value["inputs"][0].__setitem__("prompt_token_range", [0, 31]),
        ),
        (
            "completion_range",
            lambda value: value["inputs"][0].__setitem__("completion_token_range", [31, 64]),
        ),
        (
            "causal_range",
            lambda value: value["inputs"][0].__setitem__(
                "causal_completion_loss_logit_range", [32, 64]
            ),
        ),
        ("device_index", lambda value: value["hardware"].__setitem__("device_index", 1)),
        ("missing_driver", lambda value: value["hardware"].__setitem__("driver_version", "")),
        (
            "missing_cuda_runtime",
            lambda value: value["hardware"].__setitem__("cuda_runtime_version", ""),
        ),
        ("missing_pytorch", lambda value: value["hardware"].__setitem__("pytorch_version", "")),
        (
            "missing_transformers",
            lambda value: value["hardware"].__setitem__("transformers_version", ""),
        ),
        ("missing_python", lambda value: value["hardware"].__setitem__("python_version", "")),
    ],
)
def test_result_identity_and_range_mutations_are_rejected(name, mutate):
    result = _result(runner.MODE_IDS[0])
    mutate(result)
    with pytest.raises(runner.InvalidEvidence):
        runner.validate_mode_result(result, CONFIG)


@pytest.mark.parametrize(
    "field",
    [
        "driver_version",
        "cuda_runtime_version",
        "pytorch_version",
        "transformers_version",
        "python_version",
    ],
)
def test_cross_mode_hardware_and_software_mutations_are_rejected(field):
    control = _result(runner.MODE_IDS[0])
    treatment = _result(runner.MODE_IDS[1])
    treatment["hardware"][field] += "-different"
    with pytest.raises(runner.InvalidEvidence, match="cross-mode"):
        runner.build_aggregation(control, treatment, CONFIG)


def test_keyed_historical_control_rejects_missing_and_changed_keys():
    result = _result(runner.MODE_IDS[0])
    changed = copy.deepcopy(runner._historical_routes())
    changed[runner.REQUEST_IDS[0]][1]["selected_bits"] = (
        12 - changed[runner.REQUEST_IDS[0]][1]["selected_bits"]
    )
    with pytest.raises(runner.InvalidEvidence, match="historical"):
        runner.validate_mode_result(result, CONFIG, historical_routes=changed)
    missing = copy.deepcopy(runner._historical_routes())
    missing.pop(runner.REQUEST_IDS[1])
    with pytest.raises(runner.InvalidEvidence):
        runner.validate_mode_result(result, CONFIG, historical_routes=missing)


def test_aggregation_advance_and_recomputes_routes_and_quality():
    control = _result(runner.MODE_IDS[0])
    treatment = _result(
        runner.MODE_IDS[1],
        kl={"validation-3": 0.105, "validation-1000": 0.21},
        mae={"validation-3": 0.31, "validation-1000": 0.41},
    )
    aggregate = runner.build_aggregation(control, treatment, CONFIG)
    assert aggregate["classification"] == "ADVANCE_TO_BROADER_QUALITY_CHECK"
    assert aggregate["route_comparison"]["changed_target_unit_count"] == 4
    assert aggregate["route_comparison"]["layer_0_equal"] is True
    runner.validate_aggregation_result(aggregate, control, treatment, CONFIG)


@pytest.mark.parametrize(
    ("kl", "mae", "failed_check"),
    [
        (
            {"validation-3": 0.115, "validation-1000": 0.23},
            {"validation-3": 0.30, "validation-1000": 0.40},
            "aggregate_kl_passed",
        ),
        (
            {"validation-3": 0.126, "validation-1000": 0.20},
            {"validation-3": 0.30, "validation-1000": 0.40},
            "each_request_kl_passed",
        ),
        (
            {"validation-3": 0.10, "validation-1000": 0.20},
            {"validation-3": 0.34, "validation-1000": 0.44},
            "aggregate_mean_absolute_logit_error_passed",
        ),
    ],
)
def test_independent_quality_margin_failures_classify_degrades(kl, mae, failed_check):
    aggregate = runner.build_aggregation(
        _result(runner.MODE_IDS[0]),
        _result(runner.MODE_IDS[1], kl=kl, mae=mae),
        CONFIG,
    )
    assert aggregate["classification"] == "CHECKPOINT_REUSE_DEGRADES"
    assert aggregate["paired_quality"]["threshold_checks"][failed_check] is False


def test_invalid_paired_evidence_and_tampered_aggregate_are_rejected():
    control = _result(runner.MODE_IDS[0])
    treatment = _result(runner.MODE_IDS[1])
    treatment["hardware"]["driver_version"] = "different-test-driver"
    with pytest.raises(runner.InvalidEvidence, match="cross-mode"):
        runner.build_aggregation(control, treatment, CONFIG)

    treatment = _result(runner.MODE_IDS[1])
    aggregate = runner.build_aggregation(control, treatment, CONFIG)
    aggregate["route_comparison"]["changed_target_unit_count"] = 0
    with pytest.raises(runner.InvalidEvidence, match="recomputed"):
        runner.validate_aggregation_result(aggregate, control, treatment, CONFIG)


def test_layer_zero_pair_mismatch_is_invalid_evidence():
    with pytest.raises(runner.InvalidEvidence, match="layer-0"):
        runner.build_aggregation(
            _result(runner.MODE_IDS[0]),
            _result(runner.MODE_IDS[1], flip_layer_zero=True),
            CONFIG,
        )


def test_missing_result_is_pause_and_malformed_complete_evidence_is_invalid(tmp_path):
    aggregate, report = runner.aggregate_paths(
        control_path=tmp_path / "missing-control.json",
        treatment_path=tmp_path / "missing-treatment.json",
    )
    assert aggregate is None
    assert report["classification"] == "PAUSE"

    control_path = tmp_path / "control.json"
    treatment_path = tmp_path / "treatment.json"
    control_path.write_text(json.dumps(_result(runner.MODE_IDS[0])))
    treatment_path.write_text(json.dumps({"schema": runner.MODE_SCHEMA}))
    aggregate, report = runner.aggregate_paths(
        control_path=control_path,
        treatment_path=treatment_path,
    )
    assert aggregate is None
    assert report["classification"] == "INVALID_EVIDENCE"


def _runtime():
    from qaq.evaluation.lookahead_quality_runtime import ProductionRuntime

    return ProductionRuntime()


@pytest.mark.parametrize("active_environment", [None, "/tmp/wrong-venv"])
def test_production_environment_preflight_stops_before_external_or_heavy_work(
    monkeypatch, active_environment
):
    runtime = _runtime()
    external_preflight_called = False

    def unexpected_external_preflight(_protocol):
        nonlocal external_preflight_called
        external_preflight_called = True
        raise AssertionError("external preflight must not run")

    monkeypatch.setattr(runtime, "_external_preflight", unexpected_external_preflight)
    if active_environment is None:
        monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    else:
        monkeypatch.setenv("VIRTUAL_ENV", active_environment)

    with pytest.raises(RuntimeError, match=r"^PAUSE: exact ~/.venv environment is not active$"):
        runtime.prepare(CONFIG, CONFIG["modes"][0], "cuda:0", [])
    assert external_preflight_called is False
    assert runtime.torch is None
    assert runtime.student is None


class _FakeCuda:
    def __init__(
        self,
        *,
        available: bool = True,
        device_count: int = 8,
        gpu_model: str = "NVIDIA GeForce RTX 3090",
    ) -> None:
        self._available = available
        self._device_count = device_count
        self._gpu_model = gpu_model
        self.selected_device = None

    def is_available(self):
        return self._available

    def device_count(self):
        return self._device_count

    def set_device(self, target):
        self.selected_device = target

    def get_device_name(self, _target):
        return self._gpu_model


@pytest.mark.parametrize(
    ("case", "device", "expected_reason"),
    [
        ("cuda_unavailable", "cuda:0", "CUDA is unavailable"),
        ("invalid_index", "cuda:8", "explicit CUDA device is unavailable: cuda:8"),
        (
            "wrong_gpu",
            "cuda:0",
            "frozen comparable GPU is unavailable on cuda:0: Other GPU",
        ),
        ("driver_query_failure", "cuda:0", "NVIDIA driver query failed"),
    ],
)
def test_production_cuda_resource_preflights_classify_pause_with_fake_modules(
    tmp_path, monkeypatch, case, device, expected_reason
):
    runtime = _runtime()
    fake_cuda = _FakeCuda(
        available=case != "cuda_unavailable",
        gpu_model="Other GPU" if case == "wrong_gpu" else "NVIDIA GeForce RTX 3090",
    )
    fake_torch = SimpleNamespace(
        __version__="test-only",
        cuda=fake_cuda,
        device=lambda value: SimpleNamespace(index=int(value.split(":")[1])),
        version=SimpleNamespace(cuda="test-only"),
    )
    monkeypatch.setenv("VIRTUAL_ENV", str(Path.home() / ".venv"))
    monkeypatch.setattr(
        runtime,
        "_external_preflight",
        lambda _protocol: (tmp_path / "snapshot", tmp_path / "artifact", tmp_path / "router"),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(__version__="test-only"))
    if case == "driver_query_failure":
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout="", stderr="failed"),
        )

    with pytest.raises(RuntimeError, match=rf"^PAUSE: {re.escape(expected_reason)}$"):
        runtime.prepare(CONFIG, CONFIG["modes"][0], device, [])
    assert runtime.student is None


class AnyPrecisionLinear:
    def __init__(self, int32_marker) -> None:
        self.qweight = SimpleNamespace(
            dtype=int32_marker,
            ndim=3,
            shape=(8, 1, 1),
            is_cuda=True,
        )
        self._buffers = {"lut4": object(), "lut8": object()}


class _FakeStudentRepresentation:
    def __init__(self, int32_marker) -> None:
        self._packed_modules = [AnyPrecisionLinear(int32_marker) for _ in range(252)]
        self.router_count = 72
        self.router_parameter_count = 23620752

    def modules(self):
        return iter(self._packed_modules)


@pytest.mark.parametrize(
    ("drift", "expected_reason"),
    [
        (None, None),
        (
            "packed_target_count",
            "resident student does not contain exactly 252 physical packed targets",
        ),
        ("packed_representation", "resident student packed-plane/LUT representation drifted"),
        ("router_count", "router count drifted"),
        ("router_scalar_count", "historical router scalar count drifted"),
    ],
)
def test_student_representation_validation_classifies_present_drift_invalid(
    drift, expected_reason
):
    runtime = _runtime()
    int32_marker = object()
    fake_torch = SimpleNamespace(int32=int32_marker)
    student = _FakeStudentRepresentation(int32_marker)
    if drift == "packed_target_count":
        student._packed_modules.pop()
    elif drift == "packed_representation":
        student._packed_modules[0].qweight.is_cuda = False
    elif drift == "router_count":
        student.router_count = 71
    elif drift == "router_scalar_count":
        student.router_parameter_count = 23620751

    if expected_reason is None:
        runtime._validate_student_representation(student, fake_torch)
    else:
        with pytest.raises(
            RuntimeError,
            match=rf"^INVALID_EVIDENCE: {re.escape(expected_reason)}$",
        ):
            runtime._validate_student_representation(student, fake_torch)


def _write_snapshot(path: Path, *, architecture: str = "Qwen3ForCausalLM") -> None:
    path.mkdir(parents=True)
    (path / "config.json").write_text(
        json.dumps({"model_type": "qwen3", "architectures": [architecture]})
    )
    (path / "tokenizer.json").write_text("{}")
    (path / "tokenizer_config.json").write_text("{}")


def test_production_snapshot_preflight_classifies_missing_wrong_and_architecture(
    tmp_path, monkeypatch
):
    runtime = _runtime()
    revision = "pinned-revision"
    missing = tmp_path / revision
    monkeypatch.setenv("QAQ_MODEL_SNAPSHOT", str(missing))
    with pytest.raises(RuntimeError, match=r"^PAUSE: pinned Qwen3 snapshot unavailable"):
        runtime._snapshot_path(revision)

    wrong_revision = tmp_path / "wrong-revision"
    _write_snapshot(wrong_revision)
    monkeypatch.setenv("QAQ_MODEL_SNAPSHOT", str(wrong_revision))
    with pytest.raises(RuntimeError, match=r"^INVALID_EVIDENCE: model snapshot revision"):
        runtime._snapshot_path(revision)

    pinned = tmp_path / revision
    _write_snapshot(pinned, architecture="WrongArchitecture")
    monkeypatch.setenv("QAQ_MODEL_SNAPSHOT", str(pinned))
    with pytest.raises(RuntimeError, match=r"^INVALID_EVIDENCE: model snapshot architecture"):
        runtime._snapshot_path(revision)


def test_production_snapshot_preflight_missing_required_file_is_resource(tmp_path, monkeypatch):
    runtime = _runtime()
    revision = "pinned-revision"
    snapshot = tmp_path / revision
    snapshot.mkdir()
    (snapshot / "config.json").write_text(
        json.dumps({"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]})
    )
    (snapshot / "tokenizer.json").write_text("{}")
    monkeypatch.setenv("QAQ_MODEL_SNAPSHOT", str(snapshot))
    with pytest.raises(RuntimeError, match=r"^PAUSE: snapshot file unavailable"):
        runtime._snapshot_path(revision)


@pytest.mark.parametrize("label", ["packed checkpoint", "S07 router checkpoint"])
def test_production_checkpoint_preflight_classifies_missing_and_tampered(tmp_path, label):
    runtime = _runtime()
    path = tmp_path / (label.replace(" ", "-") + ".bin")
    expected = hashlib.sha256(b"expected").hexdigest()
    with pytest.raises(RuntimeError, match=rf"^PAUSE: {label} unavailable"):
        runtime._verified_file(path, expected, label)
    path.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match=rf"^INVALID_EVIDENCE: {label} SHA-256 drifted"):
        runtime._verified_file(path, expected, label)


def test_production_any_precision_preflight_classifies_missing_wrong_and_dirty(
    tmp_path, monkeypatch
):
    runtime = _runtime()
    checkout = tmp_path / "any-precision"
    with pytest.raises(RuntimeError, match=r"^PAUSE: pinned Any-Precision checkout"):
        runtime._backend_preflight(checkout, "expected")

    checkout.mkdir()
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="wrong\n", stderr=""),
    )
    with pytest.raises(RuntimeError, match=r"^INVALID_EVIDENCE: Any-Precision revision"):
        runtime._backend_preflight(checkout, "expected")

    def dirty(command, **kwargs):
        del kwargs
        stdout = "expected\n" if "rev-parse" in command else " M source.py\n"
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(subprocess, "run", dirty)
    with pytest.raises(RuntimeError, match=r"^INVALID_EVIDENCE: Any-Precision checkout is dirty"):
        runtime._backend_preflight(checkout, "expected")


def test_production_wrong_comparable_gpu_is_resource_unavailable():
    with pytest.raises(RuntimeError, match=r"^PAUSE: frozen comparable GPU is unavailable"):
        _runtime()._validate_comparable_gpu("Other GPU", "cuda:0")


def _policy(path: Path) -> runner.PersistencePolicy:
    return runner.PersistencePolicy(path, path.parent)


def test_atomic_mode_success_bytes_digest_and_existing_destination(tmp_path):
    result = _result(runner.MODE_IDS[0])
    output = tmp_path / "mode.json"
    digest = runner.persist_validated_result(
        result, output, policy=_policy(output), config=CONFIG, kind="mode"
    )
    assert hashlib.sha256(output.read_bytes()).hexdigest() == digest
    assert json.loads(output.read_text()) == result
    original = output.read_bytes()
    with pytest.raises(runner.LookaheadQualityError, match="existing file"):
        runner.persist_validated_result(
            result, output, policy=_policy(output), config=CONFIG, kind="mode"
        )
    assert output.read_bytes() == original
    assert not list(tmp_path.glob(".*.tmp"))


def test_atomic_writer_race_is_no_overwrite_and_cleans_temp(tmp_path, monkeypatch):
    result = _result(runner.MODE_IDS[0])
    output = tmp_path / "race.json"

    def race(_source, destination):
        Path(destination).write_bytes(b"competitor")
        raise FileExistsError(destination)

    monkeypatch.setattr(runner.os, "link", race)
    with pytest.raises(runner.LookaheadQualityError, match="appeared"):
        runner.persist_validated_result(
            result, output, policy=_policy(output), config=CONFIG, kind="mode"
        )
    assert output.read_bytes() == b"competitor"
    assert not list(tmp_path.glob(".*.tmp"))


def test_malformed_serialization_cleans_temp_and_preserves_unrelated_paths(tmp_path, monkeypatch):
    result = _result(runner.MODE_IDS[0])
    output = tmp_path / "malformed.json"
    unrelated = tmp_path / "unrelated.txt"
    unrelated.write_text("preserve")
    monkeypatch.setattr(runner, "_serialize_result", lambda _result: b"{malformed")
    with pytest.raises(runner.LookaheadQualityError, match="parse JSON"):
        runner.persist_validated_result(
            result, output, policy=_policy(output), config=CONFIG, kind="mode"
        )
    assert not output.exists()
    assert unrelated.read_text() == "preserve"
    assert not list(tmp_path.glob(".*.tmp"))


@pytest.mark.parametrize("kind", ["directory", "symlink", "wrong_parent"])
def test_destination_safety_rejects_non_regular_or_disallowed_paths(tmp_path, kind):
    result = _result(runner.MODE_IDS[0])
    output = tmp_path / "safe.json"
    policy = _policy(output)
    if kind == "directory":
        output.mkdir()
    elif kind == "symlink":
        target = tmp_path / "target.json"
        output.symlink_to(target)
    else:
        output = tmp_path / "other" / "safe.json"
    with pytest.raises(runner.LookaheadQualityError):
        runner.persist_validated_result(result, output, policy=policy, config=CONFIG, kind="mode")
    assert not list(tmp_path.glob(".*.tmp"))


def test_atomic_aggregation_success_uses_both_validated_modes(tmp_path):
    control = _result(runner.MODE_IDS[0])
    treatment = _result(runner.MODE_IDS[1])
    aggregate = runner.build_aggregation(control, treatment, CONFIG)
    output = tmp_path / "aggregation.json"
    digest = runner.persist_validated_result(
        aggregate,
        output,
        policy=_policy(output),
        config=CONFIG,
        kind="aggregation",
        paired_results=(control, treatment),
    )
    assert hashlib.sha256(output.read_bytes()).hexdigest() == digest
    assert json.loads(output.read_text())["classification"] == aggregate["classification"]
