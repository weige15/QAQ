from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from qaq.evaluation import lookahead_468_executor as executor

ROOT = Path(__file__).parents[2]


def test_frozen_protocol_and_manifest_validate_exactly():
    config, digest = executor.load_protocol()
    assert digest == executor.EXPECTED_CONFIG_SHA256
    assert config["data"]["canonical_manifest_sha256"] == executor.EXPECTED_S10H_SHA256
    assert tuple(config["data"]["train_ids"]) == executor.TRAIN_IDS
    assert tuple(config["data"]["validation_ids"]) == executor.VALIDATION_IDS
    assert config["training"]["optimizer_steps"] == 24
    assert config["evaluation"]["route_decisions_per_trial"] == 864
    assert config["classifications"]["precedence"] == [
        "PAUSE",
        "REVISE",
        "CONTINUE",
        "REFINE",
        "STOP",
    ]


def test_plan_has_exact_paired_twelve_trial_identity_and_order():
    plan = executor.plan()
    expected = [
        (seed, arm, value) for seed in executor.SEEDS for arm, value in executor.WITHIN_SEED_ORDER
    ]
    actual = [(item["seed"], item["arm_id"], item["lambda_bit"]) for item in plan["trials"]]
    assert actual == expected
    assert plan["trial_count"] == len(plan["trials"]) == len(plan["trial_commands"]) == 12
    assert plan["trial_order"] == [item["trial_id"] for item in plan["trials"]]
    assert len(set(plan["trial_order"])) == 12
    assert {item["seed"] for item in plan["trials"]} == set(executor.SEEDS)
    assert {item["arm_id"] for item in plan["trials"]} == {item[0] for item in executor.ARMS}
    assert {item["lambda_bit"] for item in plan["trials"]} == set(executor.LAMBDAS)
    assert all(
        item["initial_router_state_sha256"] == executor.INITIAL_HASHES[item["seed"]]
        for item in plan["trials"]
    )
    for item, command in zip(plan["trials"], plan["trial_commands"], strict=True):
        assert command[3] == item["trial_id"]
        assert item["zero_cost_reference_trial_id"] in plan["trial_order"]
        assert item["same_cost_timing_pair_trial_id"] in plan["trial_order"]


def test_plan_preserves_metrics_thresholds_and_nonexecuting_boundaries():
    plan = executor.plan()
    assert plan["candidate_bits"] == [4, 6, 8]
    assert plan["probability_order"] == ["p4", "p6", "p8"]
    assert plan["optimizer_steps_per_trial"] == 24
    assert len(plan["request_metrics"]) == 9
    assert (
        plan["thresholds"]["quality"]["lookahead_positive_each_request_hard_kl_max_zero_factor"]
        == 1.25
    )
    assert (
        plan["thresholds"]["precision"]["lookahead_positive_minus_zero_median_hard_width_max"]
        == -0.4907407407407405
    )
    assert (
        plan["thresholds"]["refine"]["quality_near_miss_hard_kl_delta_strict_max"]
        == 0.014972516723598044
    )
    config, _ = executor.load_protocol()
    assert config["classifications"]["rules"]["REFINE"]["regions"] == [
        {
            "precision_passes": True,
            "all_factor_safeguards_pass": True,
            "lookahead_paired_median_hard_kl_delta_strict_min": 0.0,
            "lookahead_paired_median_hard_kl_delta_strict_max": 0.014972516723598044,
        },
        {
            "quality_passes": True,
            "all_three_seed_width_deltas_negative": True,
            "median_width_reduction_inclusive_min": 0.24537037037037025,
            "median_width_reduction_strict_max": 0.4907407407407405,
        },
    ]
    for field in (
        "model_loading",
        "dataset_loading",
        "cuda_activity",
        "training",
        "evaluation",
        "checkpoint_creation",
        "result_write_activity",
        "execution_authorized",
    ):
        assert plan[field] is False
    assert not os.path.lexists(executor.FUTURE_RESULT_PARENT)


def _run_without_heavy_imports(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    code = f"""
import importlib.abc, runpy, sys
blocked={{'torch','transformers','datasets','any_precision','any_precision_ext'}}
class Blocker(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  if fullname.split('.')[0] in blocked: raise AssertionError(fullname)
  return None
sys.meta_path.insert(0, Blocker())
sys.argv={[str(ROOT / "scripts/run_lookahead_468_training.py"), *arguments]!r}
runpy.run_path(sys.argv[0], run_name='__main__')
"""
    return subprocess.run(
        [sys.executable, "-I", "-c", code],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_default_plan_is_byte_deterministic_standard_library_only_and_inert():
    first = _run_without_heavy_imports([])
    second = _run_without_heavy_imports(["--plan"])
    assert (first.returncode, first.stdout, first.stderr) == (
        second.returncode,
        second.stdout,
        second.stderr,
    )
    assert first.returncode == 0
    assert json.loads(first.stdout)["trial_count"] == 12
    assert not os.path.lexists(executor.FUTURE_RESULT_PARENT)


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda value: value["matrix"].__setitem__("trial_count", 13), "trial count"),
        (lambda value: value["matrix"]["seeds"].reverse(), "seed set/order"),
        (lambda value: value["matrix"]["arms"].reverse(), "arms or routing timings"),
        (lambda value: value["matrix"]["lambdas"].append(0.1), "lambda set/order"),
        (
            lambda value: value["data"]["train_ids"].__setitem__(1, value["data"]["train_ids"][0]),
            "training data/order",
        ),
        (
            lambda value: value["thresholds"]["quality"].__setitem__(
                "lookahead_positive_each_request_hard_kl_max_zero_factor", 1.26
            ),
            "thresholds",
        ),
        (
            lambda value: value["classifications"]["rules"]["REFINE"]["regions"][0].__setitem__(
                "lookahead_paired_median_hard_kl_delta_strict_max", 0.02
            ),
            "REFINE rules",
        ),
    ],
)
def test_invalid_incomplete_duplicate_reordered_or_extra_protocol_fails_closed(mutate, match):
    config, _ = executor.load_protocol()
    changed = copy.deepcopy(config)
    mutate(changed)
    with pytest.raises(executor.ProtocolError, match=match):
        executor._validate_protocol(changed)


def test_noncanonical_config_path_is_rejected_before_use(tmp_path):
    changed = tmp_path / "config.json"
    changed.write_bytes(executor.DEFAULT_CONFIG.read_bytes())
    with pytest.raises(executor.ProtocolError, match="only the frozen"):
        executor.load_protocol(changed)


def test_execution_dispatch_validates_boundaries_then_refuses_real_work():
    spec = executor.trial_specs()[0]
    expected = executor.FUTURE_RESULT_PARENT / f"{spec['trial_id']}.json"
    with pytest.raises(executor.ProtocolError, match="explicit cuda"):
        executor.validate_execution_request(
            trial_id=spec["trial_id"], device="cpu", output=expected
        )
    with pytest.raises(executor.ProtocolError, match="unknown, missing, or non-frozen"):
        executor.validate_execution_request(
            trial_id="extra-trial", device="cuda:0", output=expected
        )
    with pytest.raises(executor.ProtocolError, match="trial output must be"):
        executor.validate_execution_request(
            trial_id=spec["trial_id"], device="cuda:0", output=ROOT / "wrong.json"
        )
    with pytest.raises(executor.ProtocolError, match="real S11-D3 execution is not authorized"):
        executor.validate_execution_request(
            trial_id=spec["trial_id"], device="cuda:0", output=expected
        )
    assert not os.path.lexists(executor.FUTURE_RESULT_PARENT)


def test_exact_execute_command_imports_no_runtime_and_writes_nothing():
    spec = executor.trial_specs()[0]
    output = f"docs/results/s11d_paired_468/{spec['trial_id']}.json"
    completed = _run_without_heavy_imports(
        ["--execute-trial", spec["trial_id"], "--device", "cuda:0", "--output", output]
    )
    assert completed.returncode == 2
    report = json.loads(completed.stdout)
    assert report["classification"] == "PAUSE"
    assert report["executed"] is False
    assert report["written"] is False
    assert not os.path.lexists(executor.FUTURE_RESULT_PARENT)


def test_aggregation_is_inert_and_requires_exact_future_output():
    with pytest.raises(executor.ProtocolError, match="aggregation output must be"):
        executor.validate_aggregation_request(output=ROOT / "wrong.json")
    with pytest.raises(executor.ProtocolError, match="requires twelve"):
        executor.validate_aggregation_request(output=executor.AGGREGATION_OUTPUT)
    assert not os.path.lexists(executor.FUTURE_RESULT_PARENT)
