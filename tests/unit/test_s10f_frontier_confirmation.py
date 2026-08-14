from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path

import pytest
import torch
from torch import nn

from qaq.router.distillation import audit_router_optimizer
from qaq.router.network import SoftPrecisionRouter
from scripts import run_s10f as runner
from scripts.run_s10d import fresh_router_optimizer

ROOT = Path(__file__).parents[2]


def _trial(
    seed: int,
    lambda_bit: float,
    *,
    kd: float,
    width: float,
    frontier: bool = True,
    collapse: str = "OTHER",
    reproducible: bool = True,
) -> dict[str, object]:
    return {
        "seed": seed,
        "lambda_bit": lambda_bit,
        "initial_router_state_sha256": f"initial-{seed}",
        "final_router_state_sha256": f"final-{seed}-{lambda_bit}",
        "initial_kd_gradient_norm": 1.0,
        "initial_bit_cost_gradient_norm": 0.5,
        "lambda_weighted_gradient_ratio": lambda_bit / 2,
        "finite_loss_audit": True,
        "finite_gradient_audit": True,
        "teacher_frozen_audit": True,
        "packed_student_base_unchanged_audit": True,
        "router_only_optimizer_audit": True,
        "fresh_adamw_audit": True,
        "soft_validation_kd": kd,
        "soft_validation_mean_expected_bit_width": width,
        "soft_validation_mean_p4": 0.3,
        "soft_validation_mean_p6": 0.4,
        "soft_validation_mean_p8": 0.3,
        "soft_validation_mean_entropy": 1.5,
        "hard_validation_kd": kd,
        "hard_validation_mean_selected_bit_width": width,
        "hard_validation_fraction_4": 0.2,
        "hard_validation_fraction_6": 0.5,
        "hard_validation_fraction_8": 0.3,
        "hard_validation_route_map_validation-3": [6] * 72,
        "hard_validation_route_map_validation-1000": [8] * 72,
        "route_variation": {
            "prompt_count": 2,
            "unit_count": 72,
            "changed_unit_count": 72,
            "changed_fraction": 1.0,
        },
        "distinct_hard_route_map_count": 2,
        "reproducibility_audit": {
            "passed": reproducible,
            "route_maps_identical": reproducible,
            "hard_metrics_identical": reproducible,
            "finite_outputs_both_passed": True,
            "repeat_count": 1,
        },
        "collapse_audit": {
            "classification": collapse,
            "invalid_or_degenerate": collapse.startswith("COLLAPSED_TO_"),
            "passed": not collapse.startswith("COLLAPSED_TO_"),
        },
        "training_history": [{"step": step} for step in range(1, 5)],
        "frontier_fixture_marker": frontier,
    }


def _complete_trials() -> list[dict[str, object]]:
    trials = []
    for seed in runner.EXPECTED_SEEDS:
        for lambda_bit in runner.EXPECTED_LAMBDAS:
            if seed == 1731 and lambda_bit == 0.03:
                kd, width = 1.1, 5.0
            elif lambda_bit == 0.0:
                kd, width = 1.0, 6.0
            elif lambda_bit == 0.03:
                kd, width = 0.9, 5.0
            else:
                kd, width = (0.8, 4.0) if seed == 1731 else (0.8, 7.0)
            trials.append(_trial(seed, lambda_bit, kd=kd, width=width))
    return trials


def test_frozen_config_is_byte_exact_and_historical_base_is_not_execution_base():
    config = runner._load_frozen_config()
    assert runner._sha256_bytes(runner.CONFIG_PATH.read_bytes()) == runner.LOCKED_CONFIG_SHA256
    assert config["required_starting_commit"] == runner.HISTORICAL_PROTOCOL_BASE
    assert runner.EXPECTED_IMPLEMENTATION_BASE == "7fc136eabdba302e199354ae001cd1e1cd42199f"
    assert not (ROOT / "scripts/run_s10e.py").exists()


def test_frozen_config_rejects_any_byte_drift(tmp_path):
    payload = json.loads(runner.CONFIG_PATH.read_text())
    payload["protocol"]["seeds"] = [1729, 1730, 1732]
    path = tmp_path / "s10e.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    with pytest.raises(RuntimeError, match="byte-for-byte"):
        runner._load_frozen_config(path)


@pytest.mark.parametrize(
    "mutator",
    [
        lambda p: p["protocol"].update(seeds=[1729, 1730, 1732]),
        lambda p: p["protocol"].update(lambdas=[0.0, 0.1, 0.03]),
        lambda p: p["protocol"].update(trial_count=8),
        lambda p: p["protocol"]["pairing"].update(fresh_adamw_per_lambda=False),
        lambda p: p["inherited_s10d_contract"]["training"].update(optimizer_steps=5),
        lambda p: p["inherited_s10d_contract"]["training"].update(learning_rate=0.0001),
        lambda p: p["inherited_s10d_contract"]["objective"].update(
            cost_reduction="weighted arithmetic mean"
        ),
        lambda p: p["router_contract"].update(router_count=71),
        lambda p: p["future_measurements"]["route_map_contract"].update(units_per_map=71),
        lambda p: p["future_measurements"]["forbidden_measurements"].remove("energy"),
    ],
)
def test_protocol_rejects_all_frozen_control_and_prohibition_drift(mutator):
    protocol = json.loads(runner.CONFIG_PATH.read_text())
    mutator(protocol)
    with pytest.raises(RuntimeError):
        runner._validate_protocol(protocol)


def test_s10d_evidence_is_pinned_and_complete():
    config, result = runner._validate_s10d_evidence()
    assert config["lambda_grid"] == [0.0, 0.003, 0.01, 0.03, 0.1]
    assert result["extensions"]["performed"] == []
    assert result["audits"]["all_initial_hashes_match"]
    assert result["audits"]["packed_student_base_unchanged"]


def test_execution_config_reuses_nested_training_and_s10d_entropy_choice():
    protocol = runner._load_frozen_config()
    s10d, _ = runner._validate_s10d_evidence()
    execution = runner._execution_config(protocol, s10d)
    assert execution["training"] == protocol["inherited_s10d_contract"]["training"]
    assert execution["dataset"] == protocol["inherited_s10d_contract"]["dataset"]
    assert execution["model"]["router_count"] == 72
    assert execution["evaluation"]["entropy_log_base"] == 2.0
    assert execution["training"]["optimizer_steps"] == 4


def test_starting_base_requires_merged_s10e_ancestor_and_returns_current_head(monkeypatch):
    current_head = "17bbb2ed7f1e886bfd1f2d25c159e77eae877927"
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if command[3:5] == ["rev-parse", "HEAD"]:
            return subprocess.CompletedProcess(command, 0, current_head + "\n", "")
        assert command[3:6] == ["merge-base", "--is-ancestor", runner.EXPECTED_IMPLEMENTATION_BASE]
        assert command[6] == current_head
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    assert runner._validate_starting_base() == current_head
    assert calls[0][3:] == ["rev-parse", "HEAD"]
    assert calls[1][3:6] == ["merge-base", "--is-ancestor", runner.EXPECTED_IMPLEMENTATION_BASE]


@pytest.mark.parametrize("current_head", [
    "7fc136eabdba302e199354ae001cd1e1cd42199f",
    "17bbb2ed7f1e886bfd1f2d25c159e77eae877927",
    "delivery-test-descendant",
])
def test_starting_base_accepts_exact_base_and_legitimate_descendants(monkeypatch, current_head):
    def fake_run(command, **kwargs):
        if command[3:5] == ["rev-parse", "HEAD"]:
            return subprocess.CompletedProcess(command, 0, current_head + "\n", "")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    assert runner._validate_starting_base() == current_head


@pytest.mark.parametrize("substituted_base", [
    "e718f27fe6b02082709d65665396640e251e602c",
    "unrelated-non-ancestor",
])
def test_starting_base_rejects_wrong_or_non_ancestor_base(monkeypatch, substituted_base):
    def fake_run(command, **kwargs):
        if command[3:5] == ["rev-parse", "HEAD"]:
            return subprocess.CompletedProcess(command, 0, "current-head\n", "")
        assert command[5] == substituted_base
        return subprocess.CompletedProcess(command, 1, "", "")

    monkeypatch.setattr(runner, "EXPECTED_IMPLEMENTATION_BASE", substituted_base)
    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="merged S10-E implementation base is unavailable"):
        runner._validate_starting_base()


def test_trial_matrix_is_exact_order_count_and_pairing():
    trials = _complete_trials()
    runner._validate_trial_matrix(trials)
    with pytest.raises(RuntimeError, match="nine ordered"):
        runner._validate_trial_matrix(list(reversed(trials)))
    with pytest.raises(RuntimeError, match="nine ordered"):
        runner._validate_trial_matrix(trials[:-1])
    changed = copy.deepcopy(trials)
    changed[1]["lambda_bit"] = 0.1
    with pytest.raises(RuntimeError, match="nine ordered"):
        runner._validate_trial_matrix(changed)


def test_route_map_contract_is_exactly_72_layer_major_values():
    assert runner._validate_route_map([4] * 72) == [4] * 72
    with pytest.raises(RuntimeError, match="exactly 72"):
        runner._validate_route_map([4] * 71)
    with pytest.raises(RuntimeError, match="unconfigured"):
        runner._validate_route_map([5] * 72)


def test_state_hash_pairing_is_stable_and_seed_initializations_are_independent():
    torch.manual_seed(1729)
    first = SoftPrecisionRouter(4, hidden_width=4, candidate_bits=runner.S10_CANDIDATE_BITS)
    first_hash = runner.router_state_hash(runner.router_only_state(first))
    torch.manual_seed(1730)
    second = SoftPrecisionRouter(4, hidden_width=4, candidate_bits=runner.S10_CANDIDATE_BITS)
    second_hash = runner.router_state_hash(runner.router_only_state(second))
    assert first_hash != second_hash
    assert first_hash == runner.router_state_hash(runner.router_only_state(first))


def test_optimizer_prefix_audit_accepts_tuple_or_list_but_not_order_or_duplicates():
    assert runner._router_only_optimizer_audit({"included_name_prefixes": ("routers.",)})
    assert runner._router_only_optimizer_audit({"included_name_prefixes": ["routers."]})
    assert not runner._router_only_optimizer_audit({"included_name_prefixes": []})
    assert not runner._router_only_optimizer_audit({"included_name_prefixes": ["base."]})
    assert not runner._router_only_optimizer_audit(
        {"included_name_prefixes": ["routers.", "routers."]}
    )


class _OptimizerAuditBundle(nn.Module):
    def __init__(self):
        super().__init__()
        self.routers = nn.ParameterDict(
            {
                "first": nn.Parameter(torch.ones(2)),
                "second": nn.Parameter(torch.ones(2)),
            }
        )
        self.base = nn.Parameter(torch.ones(2))


def test_optimizer_identity_audit_rejects_missing_extra_and_duplicate_parameters():
    model = _OptimizerAuditBundle()
    first = model.routers["first"]
    second = model.routers["second"]

    reversed_audit = audit_router_optimizer(
        model, torch.optim.AdamW([second, first], lr=1e-3)
    )
    assert reversed_audit.included_names == ("routers.first", "routers.second")
    with pytest.raises(AssertionError, match="missing"):
        audit_router_optimizer(model, torch.optim.AdamW([first], lr=1e-3))
    with pytest.raises(AssertionError, match="extra"):
        audit_router_optimizer(model, torch.optim.AdamW([first, second, model.base], lr=1e-3))
    with pytest.raises(AssertionError, match="duplicate"):
        audit_router_optimizer(model, torch.optim.AdamW([first, second, first], lr=1e-3))


def test_fresh_adamw_audit_requires_empty_pre_step_state_and_rejects_reuse():
    model = _OptimizerAuditBundle()
    config = {
        "training": {"learning_rate": 1e-3, "weight_decay": 0.0},
        "model": {"router_parameter_count": sum(p.numel() for p in model.routers.parameters())},
    }
    optimizer, audit = fresh_router_optimizer(model, config)
    assert not optimizer.state
    assert audit.included_names == ("routers.first", "routers.second")
    assert runner._fresh_adamw_audit({"optimizer_state_was_fresh": True})

    for parameter in model.routers.parameters():
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    assert optimizer.state
    assert not runner._fresh_adamw_audit({"optimizer_state_was_fresh": False})


def test_collapse_audit_rejects_each_degenerate_hard_collapse():
    for label in ("COLLAPSED_TO_4", "COLLAPSED_TO_6", "COLLAPSED_TO_8"):
        audit = runner._collapse_audit(label)
        assert audit["invalid_or_degenerate"]
        assert not audit["passed"]
    for label in ("PROMPT_INVARIANT", "ADAPTIVE_OBSERVED", "OTHER"):
        assert runner._collapse_audit(label)["passed"]


def test_reproducibility_audit_requires_exact_repeat_maps_and_metrics():
    first = {
        "per_validation_route_maps": {"validation-3": [4], "validation-1000": [6]},
        "validation_kd_loss": 1.0,
        "mean_hard_selected_bit_width": 5.0,
        "hard_fraction_4": 0.5,
        "hard_fraction_6": 0.5,
        "hard_fraction_8": 0.0,
        "finite_logits": True,
    }
    repeat = copy.deepcopy(first)
    assert runner._reproducibility_audit(first, repeat)["passed"]
    repeat["per_validation_route_maps"]["validation-3"] = [8]
    assert not runner._reproducibility_audit(first, repeat)["passed"]


def test_fresh_adamw_four_step_and_freeze_audits_are_required_for_gate():
    trials = _complete_trials()
    assert all(len(trial["training_history"]) == 0 for trial in trials) is False
    trials[0]["fresh_adamw_audit"] = False
    aggregate = runner._aggregate_trials(
        trials, inherited_regressions_status="passed"
    )
    assert aggregate["classification"] == "REFINE"
    assert not aggregate["gate_checks"]["all_required_audits_pass"]


def test_aggregate_uses_within_seed_pairing_before_medians_and_frontier_membership():
    aggregate = runner._aggregate_trials(
        _complete_trials(), inherited_regressions_status="passed"
    )
    assert aggregate["classification"] == "CONTINUE"
    assert aggregate["lambda_0.03_frontier_seed_count"] == 2
    assert aggregate["per_seed_hard_frontier_membership_for_lambda_0.03"] == {
        "1729": True,
        "1730": True,
        "1731": False,
    }
    assert aggregate[
        "paired_control_hard_kd_delta_median_lambda_0.03_minus_lambda_0.0"
    ] == pytest.approx(-0.1)
    assert aggregate[
        "paired_control_hard_width_delta_median_lambda_0.03_minus_lambda_0.0"
    ] == pytest.approx(-1.0)


def test_gate_boundary_cases_are_classified_without_rule_changes():
    trials = _complete_trials()
    assert runner._aggregate_trials(trials, inherited_regressions_status="missing")[
        "classification"
    ] == "PAUSE"
    assert runner._aggregate_trials(
        trials, inherited_regressions_status="failed"
    )["classification"] == "REVISE"
    assert runner._aggregate_trials(
        trials, inherited_regressions_status="passed", invalidated_trial_count=1
    )["classification"] == "REVISE"
    collapsed = copy.deepcopy(trials)
    collapsed[0]["collapse_audit"] = {
        "classification": "COLLAPSED_TO_4",
        "invalid_or_degenerate": True,
        "passed": False,
    }
    assert runner._aggregate_trials(
        collapsed, inherited_regressions_status="passed"
    )["classification"] == "REFINE"


def test_aggregate_rejects_reproducibility_failure_as_refine():
    trials = _complete_trials()
    trials[0]["reproducibility_audit"]["passed"] = False
    aggregate = runner._aggregate_trials(
        trials, inherited_regressions_status="passed"
    )
    assert aggregate["classification"] == "REFINE"
    assert aggregate["reproducibility_failure_count"] == 1


def test_incomplete_matrix_is_external_pause_and_preserves_no_fake_aggregate():
    aggregate = runner._aggregate_trials(
        _complete_trials()[:4], inherited_regressions_status="passed"
    )
    assert aggregate["classification"] == "PAUSE"
    assert aggregate["per_lambda_median_hard_validation_kd"] == {}


def test_forbidden_measurements_are_rejected_from_result_schema():
    with pytest.raises(RuntimeError, match="forbidden measurement"):
        runner._reject_forbidden_fields({"trials": [{"latency": 1.0}]})
    with pytest.raises(RuntimeError, match="forbidden measurement"):
        runner._reject_forbidden_fields({"memory": {"value": 1}})


def test_result_schema_uses_only_required_route_map_names_and_no_production_selection():
    config = runner._load_frozen_config()
    fields = set(config["future_measurements"]["per_trial_required_fields"])
    assert "hard_validation_route_map_validation-3" in fields
    assert "hard_validation_route_map_validation-1000" in fields
    assert "latency" not in fields
    assert "production_lambda_selection" not in fields
