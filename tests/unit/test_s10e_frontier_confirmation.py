from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

CONFIG_PATH = Path(__file__).parents[2] / "configs/s10e_frontier_confirmation.json"
REQUIRED_STARTING_COMMIT = "e718f27fe6b02082709d65665396640e251e602c"
EXPECTED_BITS = [4, 6, 8]
EXPECTED_LAMBDAS = [0.0, 0.03, 0.1]
EXPECTED_SEEDS = [1729, 1730, 1731]
EXPECTED_TRAIN_IDS = ["train-3", "train-1003", "train-2002", "train-3037"]
EXPECTED_VALIDATION_IDS = ["validation-3", "validation-1000"]
EXPECTED_UNIT_ORDER = (
    "layer-major: layer 0 attention, layer 0 ffn, then layer 1 attention, "
    "layer 1 ffn, through layer 35"
)
EXPECTED_PER_TRIAL_FIELDS = [
    "seed",
    "lambda_bit",
    "initial_router_state_sha256",
    "final_router_state_sha256",
    "initial_kd_gradient_norm",
    "initial_bit_cost_gradient_norm",
    "lambda_weighted_gradient_ratio",
    "finite_loss_audit",
    "finite_gradient_audit",
    "teacher_frozen_audit",
    "packed_student_base_unchanged_audit",
    "router_only_optimizer_audit",
    "fresh_adamw_audit",
    "soft_validation_kd",
    "soft_validation_mean_expected_bit_width",
    "soft_validation_mean_p4",
    "soft_validation_mean_p6",
    "soft_validation_mean_p8",
    "soft_validation_mean_entropy",
    "hard_validation_kd",
    "hard_validation_mean_selected_bit_width",
    "hard_validation_fraction_4",
    "hard_validation_fraction_6",
    "hard_validation_fraction_8",
    "hard_validation_route_map_validation-3",
    "hard_validation_route_map_validation-1000",
    "route_variation",
    "distinct_hard_route_map_count",
    "reproducibility_audit",
]


def load_protocol() -> dict[str, object]:
    return json.loads(CONFIG_PATH.read_text())


def validate_protocol(protocol: dict[str, object]) -> None:
    """Validate the frozen S10-E contract without importing or running QAQ."""

    assert protocol["format"] == "qaq-s10e-frontier-confirmation-v1"
    assert protocol["stage"] == "S10-E"
    assert protocol["required_starting_commit"] == REQUIRED_STARTING_COMMIT

    matrix = protocol["protocol"]
    assert matrix["candidate_bits"] == EXPECTED_BITS
    assert matrix["lambdas"] == EXPECTED_LAMBDAS
    assert matrix["seeds"] == EXPECTED_SEEDS
    assert matrix["trial_count"] == 9
    assert matrix["paired_control_lambda"] == 0.0
    assert matrix["confirmation_lambda"] == 0.03

    pairing = matrix["pairing"]
    assert pairing == {
        "one_canonical_fresh_three_way_router_initialization_per_seed": True,
        "clone_canonical_initialization_identically_across_lambdas": True,
        "fresh_adamw_per_lambda": True,
        "same_lambda_order_per_seed": True,
        "warm_start_allowed": False,
        "historical_s07_two_way_checkpoint_loading_allowed": False,
        "historical_s07_two_way_checkpoint_sha256": (
            "08bf646f19759c0d7949e159bdbe4f96bbea737204b96f8760d205c8d6fd1949"
        ),
    }

    inherited = protocol["inherited_s10d_contract"]
    dataset = inherited["dataset"]
    assert dataset["repository"] == "Salesforce/wikitext"
    assert dataset["config"] == "wikitext-2-raw-v1"
    assert dataset["revision"] == "b08601e04326c79dfdd32d625aee71d232d685c3"
    assert dataset["train_split"] == "train"
    assert dataset["validation_split"] == "validation"
    assert dataset["train_offsets"] == [0, 1000, 2000, 3000]
    assert dataset["validation_offsets"] == [0, 1000]
    assert dataset["train_example_ids"] == EXPECTED_TRAIN_IDS
    assert dataset["validation_example_ids"] == EXPECTED_VALIDATION_IDS
    assert dataset["source_order"] == {
        "train_offsets": [0, 1000, 2000, 3000],
        "validation_offsets": [0, 1000],
        "train_example_ids": EXPECTED_TRAIN_IDS,
        "validation_example_ids": EXPECTED_VALIDATION_IDS,
    }
    assert dataset["train_examples"] == 4
    assert dataset["validation_examples"] == 2
    assert dataset["sequence_length"] == 64
    assert dataset["prompt_tokens"] == 32
    assert dataset["completion_tokens"] == 32
    assert dataset["prompt_boundary"] == [0, 32]
    assert dataset["completion_boundary"] == [32, 64]
    assert dataset["tokenizer_revision"] == "1cfa9a7208912126459214e8b04321603b3df60c"

    training = inherited["training"]
    assert training == {
        "batch_size": 1,
        "gradient_accumulation_steps": 1,
        "optimizer": "AdamW",
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "epochs": 1,
        "optimizer_steps": 4,
        "scheduler": "none",
        "distillation_temperature": 2.0,
        "routing_temperature": 1.0,
        "logging_interval_steps": 1,
    }

    objective = inherited["objective"]
    assert objective["candidate_bits"] == EXPECTED_BITS
    assert objective["normalized_bit_costs"] == [0.0, 0.5, 1.0]
    assert objective["cost_order"] == EXPECTED_BITS
    assert objective["normalized_bit_cost_formula"] == "c(bit) = (bit - 4) / (8 - 4)"
    assert objective["loss_formula"] == "L_total = L_KD + lambda_bit * L_bit"
    assert objective["kd_loss"] == "unchanged completion-only T^2 masked KL teacher-student distillation"
    assert objective["hardware_cost_claim"] is False

    frozen = inherited["frozen_components"]
    assert frozen["teacher_frozen"] is True
    assert frozen["packed_student_base_frozen"] is True
    assert frozen["router_only_optimizer"] is True
    assert frozen["any_precision_revision"] == "a3257d02740cc5757c78673da534b0630ff3a4ea"
    assert frozen["packed_artifact_pytorch_model_sha256"] == (
        "29d9bc526b3da0bd39daf2f82afd141f82d005ca1232cabc75cfe9d9ecc1cfee"
    )
    assert frozen["historical_s07_checkpoint_used"] is False

    router = protocol["router_contract"]
    assert router["router_count"] == 72
    assert router["router_parameter_count"] == 23630040
    assert router["candidate_bits"] == EXPECTED_BITS
    assert router["request_owned_routing_state"] is True
    assert router["completion_only_kd_objective_unchanged"] is True

    measurements = protocol["future_measurements"]
    assert measurements["per_trial_required_fields"] == EXPECTED_PER_TRIAL_FIELDS
    assert measurements["cross_seed_aggregate_required_fields"] == [
        "per_lambda_median_hard_validation_kd",
        "per_lambda_median_hard_mean_selected_bit_width",
        "per_seed_hard_frontier_membership_for_lambda_0.03",
        "lambda_0.03_frontier_seed_count",
        "paired_control_hard_kd_delta_median_lambda_0.03_minus_lambda_0.0",
        "paired_control_hard_width_delta_median_lambda_0.03_minus_lambda_0.0",
        "reproducibility_failure_count",
    ]
    assert measurements["forbidden_measurements"] == [
        "latency",
        "memory",
        "transfer",
        "throughput",
        "energy",
    ]
    assert measurements["route_map_contract"] == {
        "validation_ids_in_order": EXPECTED_VALIDATION_IDS,
        "units_per_map": 72,
        "unit_order": EXPECTED_UNIT_ORDER,
        "allowed_bits": EXPECTED_BITS,
    }

    rules = protocol["decision_rules"]
    assert rules["all_nine_trials_complete"] is True
    assert rules["all_required_audits_pass"] is True
    assert rules["invalid_or_degenerate_collapse_allowed"] is False
    assert rules["confirmation_lambda_on_per_seed_hard_kd_width_frontier"] == {
        "lambda": 0.03,
        "minimum_seed_count": 2,
        "seed_count": 3,
        "frontier_axes": ["hard_validation_kd", "hard_validation_mean_selected_bit_width"],
        "lower_is_better": True,
    }
    assert rules["paired_control"] == {
        "candidate_lambda": 0.03,
        "control_lambda": 0.0,
        "hard_kd_delta": "median(candidate hard KD - control hard KD) <= 0.0",
        "hard_selected_width_delta": "median(candidate hard selected width - control hard selected width) < 0.0",
    }
    assert rules["no_reproducibility_failure"] is True
    assert rules["scalar_combined_score_allowed"] is False
    assert rules["arbitrary_quality_loss_threshold_allowed"] is False
    assert rules["production_lambda_selection"] is False
    assert rules["success_outcome"] == "CONTINUE; authorize only later broader validation"
    assert rules["failure_outcome"] == "REFINE"
    assert rules["incomplete_evidence_outcome"] == "PAUSE"

    prohibitions = protocol["prohibitions"]
    assert prohibitions == {
        "adaptive_extension_allowed": False,
        "production_lambda_selection_allowed": False,
        "full_training_allowed": False,
        "historical_s07_two_way_checkpoint_loading_prohibited": True,
        "s10e_confirmation_trial_execution_allowed": False,
        "latency_measurement_allowed": False,
        "memory_measurement_allowed": False,
        "transfer_measurement_allowed": False,
        "throughput_measurement_allowed": False,
        "energy_measurement_allowed": False,
    }


def test_frozen_protocol_is_exact_and_does_not_execute_any_experiment():
    protocol = load_protocol()
    validate_protocol(protocol)
    assert not (CONFIG_PATH.parents[2] / "scripts/run_s10e.py").exists()


@pytest.mark.parametrize(
    "mutator",
    [
        lambda p: p["protocol"].update(seeds=[1729, 1731]),
        lambda p: p["protocol"].update(seeds=[1729, 1730, 1731, 1732]),
        lambda p: p["protocol"].update(seeds=[1730, 1729, 1731]),
        lambda p: p["protocol"].update(lambdas=[0.0, 0.1]),
        lambda p: p["protocol"].update(lambdas=[0.0, 0.03, 0.1, 0.2]),
        lambda p: p["protocol"].update(lambdas=[0.0, 0.1, 0.03]),
        lambda p: p["protocol"].update(lambdas=[0.0, 0.003, 0.03, 0.1]),
        lambda p: p["protocol"].update(lambdas=[0.0, 0.01, 0.03, 0.1]),
        lambda p: p["prohibitions"].update(adaptive_extension_allowed=True),
        lambda p: p["prohibitions"].update(production_lambda_selection_allowed=True),
        lambda p: p["prohibitions"].update(full_training_allowed=True),
        lambda p: p["protocol"].update(candidate_bits=[4, 8, 6]),
        lambda p: p["router_contract"].update(router_count=71),
        lambda p: p["router_contract"].update(router_parameter_count=23620752),
        lambda p: p["inherited_s10d_contract"]["dataset"].update(
            revision="changed-dataset-revision"
        ),
        lambda p: p["inherited_s10d_contract"]["dataset"]["source_order"].update(
            train_offsets=[0, 2000, 1000, 3000]
        ),
        lambda p: p["inherited_s10d_contract"]["training"].update(optimizer_steps=5),
        lambda p: p["inherited_s10d_contract"]["training"].update(learning_rate=0.0001),
        lambda p: p["inherited_s10d_contract"]["training"].update(weight_decay=0.01),
        lambda p: p["protocol"]["pairing"].update(
            one_canonical_fresh_three_way_router_initialization_per_seed=False
        ),
        lambda p: p["protocol"]["pairing"].update(
            clone_canonical_initialization_identically_across_lambdas=False
        ),
        lambda p: p["protocol"]["pairing"].update(fresh_adamw_per_lambda=False),
        lambda p: p["protocol"]["pairing"].update(warm_start_allowed=True),
        lambda p: p["protocol"]["pairing"].update(
            historical_s07_two_way_checkpoint_loading_allowed=True
        ),
    ],
)
def test_protocol_rejects_required_drift(mutator):
    protocol = copy.deepcopy(load_protocol())
    mutator(protocol)
    with pytest.raises((AssertionError, KeyError)):
        validate_protocol(protocol)


def test_protocol_rejects_missing_paired_initialization_semantics():
    protocol = load_protocol()
    del protocol["protocol"]["pairing"]
    with pytest.raises(KeyError):
        validate_protocol(protocol)


def test_protocol_rejects_measurement_and_objective_drift():
    protocol = load_protocol()
    protocol["future_measurements"]["per_trial_required_fields"].remove("hard_validation_fraction_6")
    with pytest.raises(AssertionError):
        validate_protocol(protocol)

    protocol = load_protocol()
    protocol["future_measurements"]["forbidden_measurements"].remove("energy")
    with pytest.raises(AssertionError):
        validate_protocol(protocol)

    protocol = load_protocol()
    protocol["inherited_s10d_contract"]["objective"]["normalized_bit_costs"] = [0.0, 1.0, 2.0]
    with pytest.raises(AssertionError):
        validate_protocol(protocol)
