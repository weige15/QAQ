from __future__ import annotations

import json

import pytest
import torch
from torch import nn

from qaq.router.distillation import RouteLogRecord, cost_aware_distillation_loss, expected_bit_cost
from qaq.router.network import THREE_WAY_CANDIDATE_BITS, SoftPrecisionRouter
from scripts.calibrate_router_cost import (
    _gradient_norm,
    _load_config,
    _validate_model_snapshot,
    classify_collapse,
    fresh_router_optimizer,
    pareto_frontiers,
    restore_router_state,
    router_only_state,
    router_state_hash,
    static_mode_name,
    summarize_route_records,
)


def test_canonical_router_reset_restores_identical_hash():
    torch.manual_seed(1729)
    router = SoftPrecisionRouter(4, hidden_width=4, candidate_bits=THREE_WAY_CANDIDATE_BITS)
    canonical = router_only_state(router)
    expected = router_state_hash(canonical)
    with torch.no_grad():
        for parameter in router.parameters():
            parameter.add_(1.0)
    assert restore_router_state(router, canonical) == expected
    assert router_state_hash(router_only_state(router)) == expected


def test_fresh_optimizer_has_no_lambda_state_leakage():
    class Bundle(nn.Module):
        def __init__(self):
            super().__init__()
            self.routers = nn.ModuleDict(
                {
                    "attention_0": SoftPrecisionRouter(
                        4, hidden_width=4, candidate_bits=THREE_WAY_CANDIDATE_BITS
                    )
                }
            )
            self.base = nn.Linear(4, 4)
            for parameter in self.base.parameters():
                parameter.requires_grad_(False)

    model = Bundle()
    scalar_count = sum(parameter.numel() for parameter in model.routers.parameters())
    config = {
        "training": {"learning_rate": 0.001, "weight_decay": 0.0},
        "model": {"router_parameter_count": scalar_count},
    }
    first, _ = fresh_router_optimizer(model, config)
    first.zero_grad(set_to_none=True)
    next(model.routers.parameters()).grad = torch.ones_like(next(model.routers.parameters()))
    first.step()
    assert first.state
    second, _ = fresh_router_optimizer(model, config)
    assert second is not first
    assert not second.state


def test_explicit_order_lambda_zero_and_six_bit_mode():
    assert THREE_WAY_CANDIDATE_BITS == (4, 6, 8)
    kd = torch.tensor(2.0, requires_grad=True)
    cost = expected_bit_cost(torch.tensor([0.2, 0.3, 0.5]))
    assert cost_aware_distillation_loss(kd, cost, 0.0) is kd
    assert cost_aware_distillation_loss(kd, cost, 0.3).item() == pytest.approx(2.195)
    assert static_mode_name(6) == "static6"
    with pytest.raises(ValueError):
        static_mode_name(5)


def test_probability_metrics_hard_fractions_and_width_are_finite():
    records = [
        RouteLogRecord.from_probabilities(
            request,
            layer,
            unit,
            torch.tensor(probabilities),
            candidate_bits=THREE_WAY_CANDIDATE_BITS,
        )
        for request, probabilities in (
            ("validation-3", [1.0, 0.0, 0.0]),
            ("validation-1000", [0.0, 1.0, 0.0]),
        )
        for layer, unit in ((0, "attention"), (0, "ffn"))
    ]
    summary = summarize_route_records(
        records,
        validation_ids=("validation-3", "validation-1000"),
        logits_finite=True,
    )
    assert summary["hard_fraction_4"] + summary["hard_fraction_6"] + summary[
        "hard_fraction_8"
    ] == pytest.approx(1.0)
    assert summary["mean_p6"] == pytest.approx(0.5)
    assert summary["mean_expected_bit_width"] == pytest.approx(5.0)
    assert summary["finite_logits"]
    assert summary["any_validation_decision_selects_6"]
    assert summary["unique_hard_route_map_count"] == 2


def test_route_summary_uses_requested_entropy_log_base():
    records = [
        RouteLogRecord.from_probabilities(
            "validation-3",
            0,
            "attention",
            torch.tensor([0.25, 0.25, 0.5]),
            candidate_bits=THREE_WAY_CANDIDATE_BITS,
        )
    ]
    summary = summarize_route_records(
        records,
        validation_ids=("validation-3",),
        logits_finite=True,
        entropy_log_base=10.0,
    )
    assert summary["entropy_log_base"] == 10.0


def test_three_way_collapse_labels_follow_locked_threshold():
    def stats(fractions, changed=0.0, distance=0.0):
        return {
            "hard_fraction_4": fractions[0],
            "hard_fraction_6": fractions[1],
            "hard_fraction_8": fractions[2],
            "route_variation_across_prompts": {"changed_fraction": changed},
            "prompt_to_prompt_route_distance": distance,
        }

    assert classify_collapse(stats((0.95, 0.05, 0.0))) == "COLLAPSED_TO_4"
    assert classify_collapse(stats((0.0, 0.95, 0.05))) == "COLLAPSED_TO_6"
    assert classify_collapse(stats((0.0, 0.05, 0.95))) == "COLLAPSED_TO_8"
    assert (
        classify_collapse(stats((0.4, 0.3, 0.3), changed=0.2, distance=0.1)) == "ADAPTIVE_OBSERVED"
    )
    assert classify_collapse(stats((0.4, 0.3, 0.3))) == "PROMPT_INVARIANT"
    assert classify_collapse(stats((0.9, 0.1, 0.0)), collapse_fraction=0.9) == "COLLAPSED_TO_4"


def test_gradient_norm_rejects_missing_gradients():
    with pytest.raises(FloatingPointError, match="missing"):
        _gradient_norm((torch.ones(1), None))


def test_locked_config_rejects_field_override(tmp_path):
    config_path = tmp_path / "s10d.json"
    config = _load_config()
    config["evaluation"]["entropy_log_base"] = 10.0
    config_path.write_text(json.dumps(config))
    with pytest.raises(RuntimeError, match="locked protocol"):
        _load_config(config_path)


def test_model_snapshot_rejects_noncanonical_path(tmp_path):
    snapshot = (
        tmp_path
        / "models--Qwen--Qwen3-4B"
        / "snapshots"
        / "1cfa9a7208912126459214e8b04321603b3df60c"
    )
    snapshot.mkdir(parents=True)
    with pytest.raises(SystemExit, match="exact pinned"):
        _validate_model_snapshot(snapshot)


def test_pareto_frontier_is_deterministic_and_not_a_scalar_selection():
    def trial(lam, kd, soft_width, hard_width):
        return {
            "lambda": lam,
            "soft": {"validation_kd_loss": kd, "mean_expected_bit_width": soft_width},
            "hard": {"validation_kd_loss": kd, "mean_hard_selected_bit_width": hard_width},
        }

    trials = [trial(0.1, 1.0, 5.0, 5.0), trial(0.0, 0.8, 6.0, 6.0), trial(0.03, 1.1, 4.0, 4.0)]
    frontier = pareto_frontiers(reversed(trials))
    assert [point["lambda"] for point in frontier["soft"]] == [0.0, 0.1, 0.03]
    assert [point["lambda"] for point in frontier["hard"]] == [0.0, 0.1, 0.03]
    assert all("width" in point for point in frontier["soft"])


def test_all_locked_lambdas_serialize_without_reordering():
    config = _load_config()
    encoded = json.dumps({"lambda_grid": config["lambda_grid"]}, sort_keys=True)
    assert json.loads(encoded)["lambda_grid"] == [0.0, 0.003, 0.01, 0.03, 0.1]
