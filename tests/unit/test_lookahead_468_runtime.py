from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass

import pytest
import torch
from torch import nn

from qaq.evaluation import lookahead_468_executor as contract
from qaq.evaluation import lookahead_468_runtime as runtime
from qaq.model.request_state import QaqRequestState, RoutingProvenance


@dataclass(frozen=True)
class TinyExample:
    example_id: str


class TinyRouter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input = nn.Linear(1, 2)
        self.output = nn.Linear(2, 3)


class TinyModel(nn.Module):
    def __init__(self, seed: int) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.routers = nn.ModuleDict(
            {
                f"{unit}_{layer}": TinyRouter()
                for unit in ("attention", "ffn")
                for layer in range(36)
            }
        )
        self.base = nn.Linear(1, 1)
        self.teacher = nn.Linear(1, 1)
        for parameter in (*self.base.parameters(), *self.teacher.parameters()):
            parameter.requires_grad_(False)


class TinyRuntime:
    enforce_frozen_model_contract = False

    def __init__(self, failure: str | None = None) -> None:
        self.failure = failure
        ids = list(contract.TRAIN_IDS)
        if failure == "data":
            ids.reverse()
        self.train_examples = tuple(TinyExample(value) for value in ids)
        self.validation_calls = 0

    @staticmethod
    def _new(seed: int) -> TinyModel:
        return TinyModel(seed)

    def prepare(self, config, device) -> None:
        assert config["schema"] == "qaq-s11d-paired-lookahead-468-v1"
        assert device == "cuda:0"

    def build_seed_model(self, seed, device):
        del device
        return self._new(seed)

    def router_state(self, model):
        return {
            name: value.detach().cpu().clone() for name, value in model.routers.state_dict().items()
        }

    def restore_router_state(self, model, state) -> None:
        model.routers.load_state_dict(copy.deepcopy(state), strict=True)
        if self.failure == "initialization":
            with torch.no_grad():
                next(model.routers.parameters()).add_(1)

    def initial_identity_matches(self, seed, digest) -> bool:
        if self.failure == "identity":
            return False
        expected = runtime._state_hash(self.router_state(self._new(seed)))
        return digest == expected

    @staticmethod
    def frozen_snapshot(model):
        return {
            "base": runtime._state_hash(model.base.state_dict()),
            "teacher": runtime._state_hash(model.teacher.state_dict()),
            "base_grads": tuple(parameter.grad for parameter in model.base.parameters()),
            "teacher_grads": tuple(parameter.grad for parameter in model.teacher.parameters()),
        }

    def frozen_audit(self, model, before):
        after = self.frozen_snapshot(model)
        return {
            "passed": self.failure != "freeze" and before == after,
            "before": before,
            "after": after,
        }

    @staticmethod
    def _provenance(request_id: str, timing: str):
        return [
            runtime._expected_provenance(request_id, timing, layer, unit)
            for layer in range(36)
            for unit in ("attention", "ffn")
        ]

    def loss(self, model, example, spec, step, device):
        del device
        values = [parameter.sum() for parameter in model.routers.parameters()]
        combined = torch.stack(values).sum()
        kd = (combined - step / 1000) ** 2 / 1000
        bit = torch.sigmoid(combined / 1000)
        total = kd + float(spec["lambda_bit"]) * bit
        if self.failure == "gradient":
            total = combined * 0
        provenance = self._provenance(example.example_id, spec["routing_timing"])
        if self.failure == "provenance":
            provenance.pop()
        return {
            "total_loss": total,
            "kd_loss": float(kd.detach()),
            "bit_loss": float(bit.detach()),
            "provenance": provenance,
            "request_state_complete": self.failure != "request_state",
        }

    def finalize_training_request(self, evidence):
        return {
            "complete": evidence.get("request_state_complete") is True,
            "cleanup": self.failure != "request_state",
        }

    def validate(self, model, spec, mode, device):
        del model, device
        self.validation_calls += 1
        records = []
        for request_index, request_id in enumerate(contract.VALIDATION_IDS):
            token = f"{spec['trial_id']}:{mode}:{request_id}"
            if self.failure == "repeat" and mode == "hard" and self.validation_calls == 3:
                token += ":changed"
            digest = hashlib.sha256(token.encode()).hexdigest()
            item = {
                "request_id": request_id,
                "request_state_audit": {"complete": True, "cleanup": True},
                "input_digest": hashlib.sha256(request_id.encode()).hexdigest(),
                "teacher_digest": hashlib.sha256(f"teacher:{request_id}".encode()).hexdigest(),
                "logits_digest": digest,
                "completion_only_temperature_2_masked_teacher_relative_kl": 0.1
                + request_index / 1000,
                "full_logit_mean_absolute_teacher_error": 0.2 + request_index / 1000,
                "full_logit_maximum_absolute_teacher_error": 0.3 + request_index / 1000,
            }
            if mode == "soft":
                item["soft_expected_width"] = 6.0
            else:
                routes = []
                for layer in range(36):
                    for unit_index, unit in enumerate(("attention", "ffn")):
                        bit = (4, 6, 8)[(layer + unit_index + request_index) % 3]
                        routes.append(
                            {
                                "request_id": request_id,
                                "target_layer": layer,
                                "unit_type": unit,
                                "selected_bits": bit,
                            }
                        )
                if self.failure == "route":
                    routes.pop()
                bits = [item["selected_bits"] for item in routes]
                item.update(
                    {
                        "route_map": routes,
                        "provenance": self._provenance(request_id, spec["routing_timing"]),
                        "hard_counts": {str(bit): bits.count(bit) for bit in (4, 6, 8)},
                        "hard_fractions": {str(bit): bits.count(bit) / 72 for bit in (4, 6, 8)},
                        "hard_mean_selected_width": sum(bits) / 72,
                        "attention_mean_selected_width": sum(bits[0::2]) / 36,
                        "ffn_mean_selected_width": sum(bits[1::2]) / 36,
                        "overall_mean_selected_width": sum(bits) / 72,
                    }
                )
            records.append(item)
        return records

    def close_model(self, model) -> None:
        del model


def _run(spec=None, *, failure=None, output=None):
    config, _ = contract.load_protocol()
    return runtime.run_trial(
        TinyRuntime(failure),
        config=config,
        spec=spec or contract.trial_specs()[0],
        device="cuda:0",
        output=output,
    )


def test_all_twelve_trials_run_in_frozen_order_with_complete_structural_evidence():
    outcomes = [_run(spec) for spec in contract.trial_specs()]
    assert [item.result["trial_id"] for item in outcomes] == [
        item["trial_id"] for item in contract.trial_specs()
    ]
    assert all(item.classification == "TRIAL_COMPLETE" and not item.written for item in outcomes)
    assert all(item.result["optimizer_steps_completed"] == 24 for item in outcomes)
    assert all(
        [entry["step"] for entry in item.result["training_history"]] == list(range(1, 25))
        for item in outcomes
    )
    assert all(item.result["route_decisions"] == 864 for item in outcomes)
    assert all(len(item.result["soft_validation"]) == 12 for item in outcomes)
    assert all(len(item.result["hard_validation"]) == 12 for item in outcomes)
    assert all(item.result["immediate_hard_repeat"]["identical"] for item in outcomes)
    assert len({item.result["optimizer_audit"]["construction_serial"] for item in outcomes}) == 12
    assert all(item.result["optimizer_audit"]["class"] == "AdamW" for item in outcomes)
    assert all(item.result["optimizer_audit"]["fresh_state_entries"] == 0 for item in outcomes)
    assert all(item.result["optimizer_audit"]["router_only"] for item in outcomes)
    assert all(item.result["freeze_audit"]["passed"] for item in outcomes)
    assert all(item.result["audits"]["gradients"] for item in outcomes)
    assert all(item.result["production_checkpoint_created"] is False for item in outcomes)


def test_production_provenance_audit_preserves_layer_zero_and_rejects_missing_lookahead():
    state = QaqRequestState(
        "validation-3",
        32,
        layer_count=36,
        candidate_bits=(4, 6, 8),
        routing_timing="lookahead_attention_one_unit",
    )
    for layer in range(1, 36):
        state.attention_provenance[layer] = RoutingProvenance(
            source_layer=layer - 1,
            target_layer=layer,
            target_unit_type="attention",
            source_point="post_attention_pre_ffn",
            routing_timing="lookahead_attention_one_unit",
        )
    records = runtime.ProductionRuntime._provenance(
        state, "validation-3", "lookahead_attention_one_unit"
    )
    assert state.attention_provenance[0] is None
    assert records[0]["source_layer"] == records[0]["target_layer"] == 0
    state.attention_provenance[17] = None
    with pytest.raises(runtime.RuntimeFailure, match="missing or invalid"):
        runtime.ProductionRuntime._provenance(state, "validation-3", "lookahead_attention_one_unit")
    assert runtime.ProductionRuntime._cleanup_request_state(state)


def test_same_seed_cells_have_byte_identical_deterministic_initialization():
    outcomes = [_run(spec) for spec in contract.trial_specs() if spec["seed"] == 1729]
    assert len({item.result["initial_router_state_sha256"] for item in outcomes}) == 1
    assert all(item.result["audits"]["paired_initialization"] for item in outcomes)


def _aggregation_trials(profile: str = "continue"):
    trials = [copy.deepcopy(_run(spec).result) for spec in contract.trial_specs()]
    for trial in trials:
        positive = (
            trial["routing_timing"] == "lookahead_attention_one_unit"
            and trial["lambda_bit"] == 0.03
        )
        kl = 0.08 if positive else 0.1
        if profile == "refine" and positive:
            kl = 0.105
        width_bits = [4] * 18 + [6] * 54 if positive and profile != "stop" else [6] * 72
        for item in trial["hard_validation"]:
            item["completion_only_temperature_2_masked_teacher_relative_kl"] = kl
            item["full_logit_mean_absolute_teacher_error"] = 0.18 if positive else 0.2
            for route, bit in zip(item["route_map"], width_bits, strict=True):
                route["selected_bits"] = bit
            item["hard_counts"] = {str(bit): width_bits.count(bit) for bit in (4, 6, 8)}
            item["hard_fractions"] = {str(bit): width_bits.count(bit) / 72 for bit in (4, 6, 8)}
            item["hard_mean_selected_width"] = sum(width_bits) / 72
            item["attention_mean_selected_width"] = sum(width_bits[0::2]) / 36
            item["ffn_mean_selected_width"] = sum(width_bits[1::2]) / 36
            item["overall_mean_selected_width"] = sum(width_bits) / 72
        trial["hard_aggregate"] = runtime._aggregate(trial["hard_validation"], "hard")
    return trials


@pytest.mark.parametrize(
    "profile,expected", [("continue", "CONTINUE"), ("refine", "REFINE"), ("stop", "STOP")]
)
def test_aggregation_applies_frozen_outcome_regions_and_complete_transitions(profile, expected):
    config, _ = contract.load_protocol()
    outcome = runtime.aggregate_trials(_aggregation_trials(profile), config=config)
    assert outcome.classification == expected
    assert outcome.result["classification"] == expected
    assert outcome.result["trial_order"] == [item["trial_id"] for item in contract.trial_specs()]
    assert [item["seed"] for item in outcome.result["per_seed"]] == list(contract.SEEDS)
    assert len(outcome.result["paired_route_transitions"]) == 12
    assert all(
        [item["request_id"] for item in trial["requests"]] == list(contract.VALIDATION_IDS)
        for trial in outcome.result["paired_route_transitions"]
    )
    first = outcome.result["paired_route_transitions"][0]["requests"][0]
    assert sum(first["treatment_control"]["transition_counts"].values()) == 72
    assert sum(first["positive_zero_cost"]["transition_counts"].values()) == 72


def test_aggregation_incomplete_reordered_and_mismatched_trials_fail_closed():
    config, _ = contract.load_protocol()
    trials = _aggregation_trials()
    assert runtime.aggregate_trials(trials[:-1], config=config).classification == "PAUSE"
    reordered = copy.deepcopy(trials)
    reordered[0], reordered[1] = reordered[1], reordered[0]
    assert runtime.aggregate_trials(reordered, config=config).classification == "REVISE"
    mismatched = copy.deepcopy(trials)
    mismatched[0]["trial_spec"]["seed"] = 999
    assert runtime.aggregate_trials(mismatched, config=config).classification == "REVISE"


def test_aggregation_persistence_failure_leaves_no_partial_evidence(monkeypatch, tmp_path):
    def reject(*args, **kwargs):
        del args, kwargs
        raise OSError("injected aggregation persistence failure")

    monkeypatch.setattr(runtime.os, "link", reject)
    config, _ = contract.load_protocol()
    output = tmp_path / "aggregation.json"
    outcome = runtime.aggregate_trials(_aggregation_trials(), config=config, output=output)
    assert outcome.classification == "REVISE"
    assert outcome.result is None and not outcome.written
    assert not output.exists()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "failure,expected",
    [
        ("identity", "initialization identity"),
        ("data", "data/order"),
        ("initialization", "byte-identical"),
        ("freeze", "packed base"),
        ("gradient", "soft-gradient"),
        ("provenance", "provenance"),
        ("request_state", "request-state"),
        ("route", "72 units"),
        ("repeat", "repeat"),
    ],
)
def test_injected_runtime_defects_fail_closed_without_evidence(failure, expected):
    outcome = _run(failure=failure)
    assert outcome.classification in {"PAUSE", "REVISE"}
    assert outcome.result is None
    assert outcome.written is False
    assert expected in " ".join(outcome.errors)
    assert not contract.FUTURE_RESULT_PARENT.exists()


@pytest.mark.parametrize("defect", ["membership", "freshness"])
def test_malformed_optimizer_is_detected_and_fails_closed(monkeypatch, defect):
    def malformed(model, config):
        parameters = list(model.routers.parameters())
        selected = parameters[:1] if defect == "membership" else parameters
        optimizer = torch.optim.AdamW(selected, lr=float(config["learning_rate"]))
        if defect == "freshness":
            optimizer.state[selected[0]]["injected"] = torch.tensor(1)
        return optimizer

    monkeypatch.setattr(runtime, "_build_optimizer", malformed)
    outcome = _run()
    assert outcome.classification == "REVISE"
    assert outcome.result is None and not outcome.written
    assert ("membership" if defect == "membership" else "inherited state") in outcome.errors[0]


def test_missing_ordered_optimizer_step_is_detected_after_actual_step_calls(monkeypatch):
    calls = []

    def omit_final_count(optimizer, step, completed):
        optimizer.step()
        calls.append(step)
        if step != 24:
            completed.append(step)

    monkeypatch.setattr(runtime, "_optimizer_step", omit_final_count)
    outcome = _run()
    assert calls == list(range(1, 25))
    assert outcome.classification == "REVISE"
    assert outcome.result is None and not outcome.written
    assert "update count or order" in outcome.errors[0]


def test_persistence_failure_leaves_no_partial_canonical_evidence(monkeypatch):
    def reject(*args, **kwargs):
        del args, kwargs
        raise OSError("injected persistence failure")

    monkeypatch.setattr(runtime.os, "link", reject)
    spec = contract.trial_specs()[0]
    outcome = _run(spec, output=contract.FUTURE_RESULT_PARENT / f"{spec['trial_id']}.json")
    assert outcome.classification == "REVISE"
    assert outcome.result is None and not outcome.written
    assert "persistence failure" in outcome.errors[0]
    assert not contract.FUTURE_RESULT_PARENT.exists()


def test_non_frozen_spec_and_non_cuda_device_fail_before_runtime_work():
    config, _ = contract.load_protocol()
    changed = dict(contract.trial_specs()[0])
    changed["trial_index"] = 2
    outcome = runtime.run_trial(
        TinyRuntime(), config=config, spec=changed, device="cuda:0", output=None
    )
    assert outcome.classification == "REVISE"
    outcome = runtime.run_trial(
        TinyRuntime(), config=config, spec=contract.trial_specs()[0], device="cpu", output=None
    )
    assert outcome.classification == "PAUSE"
