from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

CONFIG_PATH = Path(__file__).parents[2] / "configs/s10g_broader_validation.json"

EXPECTED_BITS = [4, 6, 8]
EXPECTED_LAMBDAS = [0.0, 0.03, 0.1]
EXPECTED_SEEDS = [1729, 1730, 1731]
EXPECTED_TRAIN_OFFSETS = list(range(0, 24000, 1000))
EXPECTED_TRAIN_ROWS = [
    3,
    1003,
    2002,
    3037,
    4005,
    5002,
    6001,
    7001,
    8000,
    9068,
    10001,
    11003,
    12003,
    13000,
    14000,
    15000,
    16002,
    17002,
    18000,
    19001,
    20006,
    21000,
    22000,
    23000,
]
EXPECTED_TRAIN_IDS = [f"train-{row}" for row in EXPECTED_TRAIN_ROWS]
EXPECTED_VALIDATION_OFFSETS = list(range(0, 3000, 250))
EXPECTED_VALIDATION_ROWS = [3, 270, 500, 761, 1000, 1252, 1500, 1759, 2000, 2250, 2500, 2755]
EXPECTED_VALIDATION_IDS = [f"validation-{row}" for row in EXPECTED_VALIDATION_ROWS]
EXPECTED_UNIT_ORDER = (
    "layer-major: layer 0 attention, layer 0 ffn, then layer 1 attention, layer 1 ffn, "
    "through layer 35"
)
EXPECTED_PER_TRIAL_FIELDS = {
    "seed",
    "lambda_bit",
    "initial_router_state_sha256",
    "final_router_state_sha256",
    "initial_kd_gradient_norm",
    "initial_bit_cost_gradient_norm",
    "lambda_weighted_gradient_ratio",
    "training_examples_seen",
    "optimizer_steps_completed",
    "finite_loss_audit",
    "finite_gradient_audit",
    "teacher_frozen_audit",
    "packed_student_base_unchanged_audit",
    "collapse_audit",
    "optimizer_audit",
    "soft_validation_kd",
    "soft_validation_mean_absolute_logit_error",
    "soft_validation_maximum_absolute_logit_error",
    "soft_validation_mean_expected_bit_width",
    "soft_validation_mean_p4",
    "soft_validation_mean_p6",
    "soft_validation_mean_p8",
    "soft_validation_mean_entropy",
    "hard_validation_kd",
    "hard_validation_mean_absolute_logit_error",
    "hard_validation_maximum_absolute_logit_error",
    "hard_validation_mean_selected_bit_width",
    "hard_validation_fraction_4",
    "hard_validation_fraction_6",
    "hard_validation_fraction_8",
    "hard_validation_route_maps",
    "route_variation",
    "distinct_hard_route_map_count",
    "reproducibility_audit",
    "prohibited_measurement_audit",
}
EXPECTED_CROSS_SEED_FIELDS = {
    "per_lambda_median_hard_validation_kd",
    "per_lambda_median_hard_mean_selected_bit_width",
    "per_seed_hard_frontier_membership_for_lambda_0.03",
    "lambda_0.03_frontier_seed_count",
    "paired_control_hard_kd_delta_median_lambda_0.03_minus_lambda_0.0",
    "paired_control_hard_width_delta_median_lambda_0.03_minus_lambda_0.0",
    "reproducibility_failure_count",
}
EXPECTED_RUN_LEVEL_FIELDS = {
    "inherited_regressions_audit",
    "prohibited_work_audit",
}
EXPECTED_OPTIMIZER_AUDIT_FIELDS = {
    "actual_optimizer_parameter_count",
    "actual_optimizer_parameter_names",
    "actual_optimizer_parameter_names_sha256",
    "duplicate_optimizer_parameter_count",
    "expected_router_parameter_count",
    "expected_router_parameter_names",
    "expected_router_parameter_names_sha256",
    "fresh_adamw_audit",
    "missing_router_parameter_count",
    "optimizer_construction_serial",
    "optimizer_state_entry_count_before_first_step",
    "optimizer_state_entry_count_before_training_begins",
    "router_only_optimizer_audit",
    "runtime_identity_based_membership_result",
    "unexpected_optimizer_parameter_count",
}
EXPECTED_FORBIDDEN_MEASUREMENTS = [
    "latency",
    "memory",
    "transfer",
    "throughput",
    "energy",
    "hardware_cost",
]
EXPECTED_FORBIDDEN_ACTIONS = [
    "adaptive_lambda_search",
    "adaptive_extension_allowed",
    "post_result_seed_replacement",
    "post_result_example_replacement",
    "warm_start",
    "s07_conversion",
    "historical_s07_two_way_checkpoint_loading",
    "training_teacher",
    "training_packed_base",
    "non_router_optimizer_membership",
    "candidate_bits_change",
    "normalized_cost_change",
    "s08_loader_changes",
    "six_bit_on_demand_loading",
    "production_lambda_selection",
    "s10h_execution",
    "s10g_execution",
]


def load_protocol() -> dict[str, object]:
    return json.loads(CONFIG_PATH.read_text())


def validate_frozen_protocol(protocol: dict[str, object]) -> None:
    """Validate only the frozen data contract; this function runs no model code."""

    assert protocol["format"] == "qaq-s10g-broader-validation-v1"
    assert protocol["stage"] == "S10-G"
    freeze = protocol["protocol_freeze"]
    assert freeze == {
        "classification": "CONTINUE",
        "protocol_only": True,
        "execution_allowed": False,
        "result_claimed": False,
        "runner_path": None,
        "result_path": None,
        "next_action": (
            "A separately authorized future execution may consume this frozen protocol; "
            "S10-G itself performs no training or evaluation."
        ),
    }

    facts = protocol["source_project_established_facts"]
    assert facts["completed_stages"] == ["S10-A", "S10-B", "S10-C", "S10-D", "S10-E", "S10-F"]
    assert facts["s10f_attempt_1"] == {
        "path": "docs/results/s10f_frontier_confirmation.json",
        "sha256": "d68f041e0a3dc32c465e8b8068ca3ab230d39253757e30f3019ca7e681b14233",
        "classification": "REVISE",
        "preserved": True,
    }
    assert facts["s10f_attempt_2"]["path"] == "docs/results/s10f_frontier_confirmation_rerun.json"
    assert facts["s10f_attempt_2"]["sha256"] == (
        "b3bcc0e45d45852ac5060209c4789453ed452462f528f7bffd4cb80fb1ef58cb"
    )
    assert facts["s10f_attempt_2"]["classification"] == "CONTINUE"
    assert facts["s10f_attempt_2"]["authorized_scope"] == (
        "only a separately scoped broader-validation decision"
    )
    assert facts["s10f_attempt_2"]["production_lambda_selected"] is False
    assert facts["no_broader_validation_has_run"] is True
    assert facts["pinned_model"] == {
        "repository": "Qwen/Qwen3-4B",
        "revision": "1cfa9a7208912126459214e8b04321603b3df60c",
        "tokenizer_revision": "1cfa9a7208912126459214e8b04321603b3df60c",
    }
    assert facts["pinned_backend"] == {
        "any_precision_revision": "a3257d02740cc5757c78673da534b0630ff3a4ea",
        "packed_artifact": "quantized/s03b_qwen3_4b/backend_cache/packed/anyprec-(1cfa9a7208912126459214e8b04321603b3df60c)-w8_orig4-gc1-c4_s1_blk64",
        "packed_artifact_pytorch_model_sha256": "29d9bc526b3da0bd39daf2f82afd141f82d005ca1232cabc75cfe9d9ecc1cfee",
    }

    protocol_body = protocol["protocol"]
    assert protocol_body["required_ancestor_commit"] == "7fc136eabdba302e199354ae001cd1e1cd42199f"
    assert protocol_body["candidate_bits"] == EXPECTED_BITS
    assert protocol_body["lambdas"] == EXPECTED_LAMBDAS
    assert protocol_body["seeds"] == EXPECTED_SEEDS
    assert protocol_body["trial_count"] == 9
    assert protocol_body["pairing"] == {
        "one_canonical_fresh_three_way_router_initialization_per_seed": True,
        "clone_canonical_initialization_identically_across_lambdas": True,
        "fresh_router_initialization_per_seed": True,
        "fresh_adamw_per_lambda": True,
        "same_lambda_order_per_seed": True,
        "warm_start_allowed": False,
        "historical_s07_two_way_checkpoint_loading_allowed": False,
        "historical_s07_two_way_checkpoint_sha256": (
            "08bf646f19759c0d7949e159bdbe4f96bbea737204b96f8760d205c8d6fd1949"
        ),
    }

    dataset = protocol_body["dataset"]
    assert dataset["repository"] == "Salesforce/wikitext"
    assert dataset["config"] == "wikitext-2-raw-v1"
    assert dataset["revision"] == "b08601e04326c79dfdd32d625aee71d232d685c3"
    assert dataset["train_split"] == "train"
    assert dataset["validation_split"] == "validation"
    assert dataset["tokenizer_revision"] == "1cfa9a7208912126459214e8b04321603b3df60c"
    assert dataset["sequence_length"] == 64
    assert dataset["prompt_tokens"] == 32
    assert dataset["completion_tokens"] == 32
    assert dataset["prompt_boundary"] == [0, 32]
    assert dataset["completion_boundary"] == [32, 64]
    assert dataset["preprocessing"] == (
        "tokenize each raw row with the pinned tokenizer and add_special_tokens=false; "
        "retain the first 64 tokens; prompt is tokens [0,32), completion is tokens [32,64)"
    )
    assert dataset["selection_rule"] == (
        "For each source offset, scan forward inclusively, skip blank or short rows, "
        "and select the first row with at least 64 tokens."
    )
    assert dataset["train_offsets"] == EXPECTED_TRAIN_OFFSETS
    assert dataset["train_source_rows"] == EXPECTED_TRAIN_ROWS
    assert dataset["train_example_ids"] == EXPECTED_TRAIN_IDS
    assert dataset["validation_offsets"] == EXPECTED_VALIDATION_OFFSETS
    assert dataset["validation_source_rows"] == EXPECTED_VALIDATION_ROWS
    assert dataset["validation_example_ids"] == EXPECTED_VALIDATION_IDS
    assert dataset["train_examples"] == 24
    assert dataset["validation_examples"] == 12
    assert dataset["source_order"] == {
        "train_offsets": EXPECTED_TRAIN_OFFSETS,
        "train_source_rows": EXPECTED_TRAIN_ROWS,
        "train_example_ids": EXPECTED_TRAIN_IDS,
        "validation_offsets": EXPECTED_VALIDATION_OFFSETS,
        "validation_source_rows": EXPECTED_VALIDATION_ROWS,
        "validation_example_ids": EXPECTED_VALIDATION_IDS,
    }

    training = protocol_body["training"]
    assert training == {
        "examples_seen": 24,
        "batch_size": 1,
        "gradient_accumulation_steps": 1,
        "optimizer": "AdamW",
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "betas": [0.9, 0.999],
        "eps": 1e-08,
        "amsgrad": False,
        "epochs": 1,
        "optimizer_steps": 24,
        "scheduler": "none",
        "distillation_temperature": 2.0,
        "routing_temperature": 1.0,
        "logging_interval_steps": 1,
        "one_pass_no_resampling": True,
    }
    assert protocol["implementation_choices"]["training_scope"]["examples_seen"] == 24
    assert protocol["implementation_choices"]["training_scope"]["optimizer_steps"] == 24

    assert protocol_body["objective"] == {
        "candidate_bits": EXPECTED_BITS,
        "normalized_bit_cost_formula": "c(bit) = (bit - 4) / (8 - 4)",
        "normalized_bit_costs": [0.0, 0.5, 1.0],
        "cost_order": EXPECTED_BITS,
        "loss_formula": "L_total = L_KD + lambda_bit * L_bit",
        "kd_loss": "unchanged completion-only T^2 masked KL teacher-student distillation",
        "cost_reduction": "unweighted arithmetic mean across all 36 attention and 36 FFN decisions",
        "hardware_cost_claim": False,
    }
    assert protocol_body["frozen_components"]["teacher_frozen"] is True
    assert protocol_body["frozen_components"]["packed_student_base_frozen"] is True
    assert protocol_body["frozen_components"]["router_only_optimizer"] is True
    assert protocol_body["frozen_components"]["historical_s07_checkpoint_used"] is False
    assert protocol_body["router_contract"] == {
        "router_count": 72,
        "router_parameter_count": 23630040,
        "candidate_bits": EXPECTED_BITS,
        "soft_routing": "resident three-way packed mixture in explicit [p4,p6,p8] order",
        "hard_routing": "deterministic argmax in explicit [p4,p6,p8] order",
        "request_owned_routing_state": True,
        "completion_only_kd_objective_unchanged": True,
    }

    measurements = protocol_body["future_measurements"]
    assert set(measurements["per_trial_required_fields"]) == EXPECTED_PER_TRIAL_FIELDS
    assert len(measurements["per_trial_required_fields"]) == len(EXPECTED_PER_TRIAL_FIELDS)
    assert set(measurements["run_level_required_fields"]) == EXPECTED_RUN_LEVEL_FIELDS
    assert set(measurements["cross_seed_aggregate_required_fields"]) == EXPECTED_CROSS_SEED_FIELDS
    assert measurements["entropy_log_base"] == 2.0
    assert measurements["optimizer_audit_contract"] == {
        "required": True,
        "required_fields": sorted(EXPECTED_OPTIMIZER_AUDIT_FIELDS),
        "expected_router_name_prefix": "routers.",
        "expected_router_parameter_count": 288,
        "identity_based_membership_required": True,
        "fresh_state_before_first_step_required": True,
    }
    assert measurements["inherited_regressions_audit_contract"] == {
        "required": True,
        "required_fields": ["status", "test_selection", "passed"],
        "status": "passed",
        "test_selection": "S10-D/S10-E/S10-F predecessor regression selection",
        "must_be_true": True,
    }
    assert measurements["prohibited_work_audit_contract"] == {
        "required": True,
        "required_fields": [
            "forbidden_actions_observed",
            "forbidden_measurements_observed",
            "passed",
        ],
        "must_be_true": True,
        "forbidden_actions": EXPECTED_FORBIDDEN_ACTIONS,
        "forbidden_measurements": EXPECTED_FORBIDDEN_MEASUREMENTS,
    }
    assert measurements["route_map_contract"] == {
        "validation_ids_in_order": EXPECTED_VALIDATION_IDS,
        "validation_map_count": 12,
        "units_per_map": 72,
        "unit_order": EXPECTED_UNIT_ORDER,
        "allowed_bits": EXPECTED_BITS,
        "map_shape": (
            "object keyed by validation_ids_in_order; each value is an ordered list of exactly 72 integer bits"
        ),
    }
    assert measurements["reproducibility_contract"] == {
        "repeat_count": 1,
        "method": "one immediate same-state hard validation repeat at the unchanged trained router state",
        "required_fields": [
            "route_maps_identical",
            "hard_metrics_identical",
            "finite_outputs_both_passed",
            "passed",
        ],
    }
    assert measurements["prohibited_measurement_audit"] == {
        "required": True,
        "must_be_true": True,
        "forbidden_measurements": EXPECTED_FORBIDDEN_MEASUREMENTS,
    }

    gate = protocol["future_two_axis_gate"]
    assert gate["axes"] == ["hard_validation_kd", "hard_validation_mean_selected_bit_width"]
    assert gate["direction"] == "lower_is_better"
    assert gate["scalar_combined_score_allowed"] is False
    assert gate["arbitrary_quality_loss_threshold_allowed"] is False
    assert gate["required_conditions"] == {
        "all_required_evidence_complete": True,
        "all_nine_trials_complete": True,
        "all_required_audits_pass": True,
        "inherited_regressions_pass": True,
        "no_invalid_or_degenerate_collapse": True,
        "no_prohibited_work": True,
        "lambda_0.03_frontier_seed_count_minimum": 2,
        "paired_hard_kd_delta_median_maximum": 0.0,
        "paired_hard_kd_delta_rule": "median(candidate hard KD - control hard KD) <= 0.0",
        "paired_hard_width_delta_median_strictly_less_than": 0.0,
        "paired_hard_width_delta_rule": "median(candidate hard selected width - control hard selected width) < 0.0",
        "reproducibility_failure_count_maximum": 0,
    }
    assert gate["outcome_precedence"] == ["PAUSE", "REVISE", "REFINE", "CONTINUE"]
    assert gate["outcome_rules"] == {
        "PAUSE": {
            "when": "Any required trial, prerequisite, or evidence field is missing or incomplete; evaluate this before other outcomes.",
            "failed_conditions": ["all_required_evidence_complete", "all_nine_trials_complete"],
        },
        "REVISE": {
            "when": "Evidence is complete but an inherited regression, required audit, collapse, reproducibility, or prohibited-work integrity condition fails; evaluate this after PAUSE and before REFINE.",
            "failed_conditions": [
                "all_required_audits_pass",
                "inherited_regressions_pass",
                "no_invalid_or_degenerate_collapse",
                "no_prohibited_work",
                "reproducibility_failure_count_maximum",
            ],
        },
        "REFINE": {
            "when": "PAUSE and REVISE conditions pass, but one or more two-axis frontier or paired-control thresholds fail.",
            "failed_conditions": [
                "lambda_0.03_frontier_seed_count_minimum",
                "paired_hard_kd_delta_median_maximum",
                "paired_hard_width_delta_median_strictly_less_than",
            ],
        },
        "CONTINUE": {
            "when": "Every required condition passes.",
            "failed_conditions": [],
        },
    }
    assert set(gate["outcomes"]) == {"CONTINUE", "REFINE", "REVISE", "PAUSE"}
    assert gate["production_lambda_selection"] is False

    prohibitions = protocol["prohibitions"]
    assert all(value is False for value in prohibitions.values())
    assert prohibitions == {
        "adaptive_lambda_search": False,
        "adaptive_extension_allowed": False,
        "post_result_seed_replacement": False,
        "post_result_example_replacement": False,
        "warm_start": False,
        "s07_conversion": False,
        "historical_s07_two_way_checkpoint_loading": False,
        "training_teacher": False,
        "training_packed_base": False,
        "non_router_optimizer_membership": False,
        "candidate_bits_change": False,
        "normalized_cost_change": False,
        "s08_loader_changes": False,
        "six_bit_on_demand_loading": False,
        "production_lambda_selection": False,
        "s10h_execution": False,
        "s10g_execution": False,
        "latency_measurement": False,
        "memory_measurement": False,
        "transfer_measurement": False,
        "throughput_measurement": False,
        "energy_measurement": False,
        "hardware_cost_measurement": False,
        "scalar_combined_score": False,
    }
    assert protocol["execution_artifacts"] == {
        "runner_created": False,
        "result_json_created": False,
        "execution_path_created": False,
        "allowed_runner_path": None,
        "allowed_result_path": None,
    }


def test_frozen_protocol_is_machine_readable_and_has_no_execution_artifacts():
    validate_frozen_protocol(load_protocol())
    assert not (CONFIG_PATH.parents[1] / "scripts/run_s10g.py").exists()
    assert not (CONFIG_PATH.parents[1] / "docs/results/s10g_broader_validation.json").exists()
    assert not (
        CONFIG_PATH.parents[1] / "docs/results/s10g_broader_validation_result.json"
    ).exists()


def test_route_map_schema_requires_all_twelve_ids_and_72_entries():
    protocol = load_protocol()
    contract = protocol["protocol"]["future_measurements"]["route_map_contract"]
    valid = {example_id: [4] * 72 for example_id in contract["validation_ids_in_order"]}
    assert list(valid) == EXPECTED_VALIDATION_IDS
    assert all(len(route_map) == 72 for route_map in valid.values())
    assert all(bit in EXPECTED_BITS for route_map in valid.values() for bit in route_map)

    invalid = copy.deepcopy(valid)
    invalid[EXPECTED_VALIDATION_IDS[0]] = [4] * 71
    assert any(len(route_map) != contract["units_per_map"] for route_map in invalid.values())
    invalid = copy.deepcopy(valid)
    invalid[EXPECTED_VALIDATION_IDS[0]][0] = 5
    assert any(bit not in EXPECTED_BITS for route_map in invalid.values() for bit in route_map)


@pytest.mark.parametrize(
    "mutator",
    [
        lambda p: p["protocol"].update(candidate_bits=[4, 8, 6]),
        lambda p: p["protocol"].update(candidate_bits=[4, 6]),
        lambda p: p["protocol"].update(lambdas=[0.0, 0.1, 0.03]),
        lambda p: p["protocol"].update(lambdas=[0.0, 0.03]),
        lambda p: p["protocol"].update(lambdas=[0.0, 0.03, 0.1, 0.2]),
        lambda p: p["protocol"].update(seeds=[1730, 1729, 1731]),
        lambda p: p["protocol"].update(seeds=[1729, 1730]),
        lambda p: p["protocol"].update(seeds=[1729, 1730, 1731, 1732]),
        lambda p: p["protocol"]["dataset"].update(revision="changed"),
        lambda p: p["source_project_established_facts"]["s10f_attempt_1"].update(path="changed"),
        lambda p: p["source_project_established_facts"]["s10f_attempt_1"].update(sha256="changed"),
        lambda p: p["source_project_established_facts"]["pinned_model"].update(revision="changed"),
        lambda p: p["source_project_established_facts"]["pinned_backend"].update(
            any_precision_revision="changed"
        ),
        lambda p: p["source_project_established_facts"]["pinned_backend"].update(
            packed_artifact_pytorch_model_sha256="changed"
        ),
        lambda p: p["protocol"]["dataset"].update(preprocessing="changed"),
        lambda p: p["protocol"]["dataset"].update(selection_rule="changed"),
        lambda p: p["protocol"]["dataset"].update(train_offsets=[0, 2000, 1000]),
        lambda p: p["protocol"]["dataset"].update(train_example_ids=EXPECTED_TRAIN_IDS[:4]),
        lambda p: p["protocol"]["dataset"].update(train_examples=4),
        lambda p: p["protocol"]["dataset"].update(validation_examples=2),
        lambda p: p["protocol"]["training"].update(optimizer_steps=4),
        lambda p: p["protocol"]["training"].update(learning_rate=0.0001),
        lambda p: p["protocol"]["training"].update(weight_decay=0.01),
        lambda p: p["protocol"]["training"].update(optimizer="SGD"),
        lambda p: p["protocol"]["objective"].update(normalized_bit_costs=[0.0, 1.0, 2.0]),
        lambda p: p["protocol"]["objective"].update(cost_order=[4, 8, 6]),
        lambda p: p["protocol"]["pairing"].update(warm_start_allowed=True),
        lambda p: p["protocol"]["pairing"].update(fresh_adamw_per_lambda=False),
        lambda p: p["protocol"]["pairing"].update(
            historical_s07_two_way_checkpoint_loading_allowed=True
        ),
        lambda p: p["protocol"]["frozen_components"].update(teacher_frozen=False),
        lambda p: p["protocol"]["frozen_components"].update(packed_student_base_frozen=False),
        lambda p: p["protocol"]["frozen_components"].update(router_only_optimizer=False),
        lambda p: p["protocol"]["future_measurements"]["per_trial_required_fields"].remove(
            "teacher_frozen_audit"
        ),
        lambda p: p["protocol"]["future_measurements"]["per_trial_required_fields"].remove(
            "collapse_audit"
        ),
        lambda p: p["protocol"]["future_measurements"]["per_trial_required_fields"].remove(
            "optimizer_audit"
        ),
        lambda p: p["protocol"]["future_measurements"]["per_trial_required_fields"].append(
            "latency"
        ),
        lambda p: p["protocol"]["future_measurements"].update(entropy_log_base=10.0),
        lambda p: p["protocol"]["future_measurements"]["run_level_required_fields"].remove(
            "inherited_regressions_audit"
        ),
        lambda p: p["protocol"]["future_measurements"]["prohibited_measurement_audit"][
            "forbidden_measurements"
        ].append("latency"),
        lambda p: p["future_two_axis_gate"]["required_conditions"].update(
            **{"lambda_0.03_frontier_seed_count_minimum": 1}
        ),
        lambda p: p["future_two_axis_gate"]["required_conditions"].update(no_prohibited_work=False),
        lambda p: p["future_two_axis_gate"]["outcome_precedence"].reverse(),
        lambda p: p["future_two_axis_gate"].update(scalar_combined_score_allowed=True),
        lambda p: p["future_two_axis_gate"].update(production_lambda_selection=True),
        lambda p: p["prohibitions"].update(adaptive_lambda_search=True),
        lambda p: p["prohibitions"].update(s10h_execution=True),
        lambda p: p["prohibitions"].update(six_bit_on_demand_loading=True),
        lambda p: p["execution_artifacts"].update(runner_created=True),
    ],
)
def test_protocol_rejects_drift_and_prohibited_changes(mutator):
    protocol = copy.deepcopy(load_protocol())
    mutator(protocol)
    with pytest.raises((AssertionError, KeyError)):
        validate_frozen_protocol(protocol)


def test_protocol_rejects_missing_or_reordered_validation_order():
    protocol = copy.deepcopy(load_protocol())
    protocol["protocol"]["dataset"]["source_order"]["validation_example_ids"] = list(
        reversed(EXPECTED_VALIDATION_IDS)
    )
    with pytest.raises(AssertionError):
        validate_frozen_protocol(protocol)


def test_future_gate_has_four_distinct_non_scalar_outcomes():
    gate = load_protocol()["future_two_axis_gate"]
    outcomes = gate["outcomes"]
    assert set(outcomes) == {"CONTINUE", "REFINE", "REVISE", "PAUSE"}
    assert "scalar" in outcomes["REFINE"].lower()
    assert "invalid" in outcomes["REVISE"].lower()
    assert "missing" in outcomes["PAUSE"].lower()
    assert "production lambda" in outcomes["CONTINUE"]
    assert gate["axes"] == [
        "hard_validation_kd",
        "hard_validation_mean_selected_bit_width",
    ]
    assert gate["scalar_combined_score_allowed"] is False
    assert gate["production_lambda_selection"] is False
    assert gate["outcome_precedence"] == ["PAUSE", "REVISE", "REFINE", "CONTINUE"]
    assert gate["outcome_rules"]["REVISE"]["failed_conditions"] == [
        "all_required_audits_pass",
        "inherited_regressions_pass",
        "no_invalid_or_degenerate_collapse",
        "no_prohibited_work",
        "reproducibility_failure_count_maximum",
    ]


def test_s10g_freeze_does_not_claim_a_broader_result_or_select_lambda():
    protocol = load_protocol()
    assert protocol["protocol_freeze"]["execution_allowed"] is False
    assert protocol["protocol_freeze"]["result_claimed"] is False
    assert protocol["protocol_freeze"]["classification"] in {"CONTINUE", "REVISE", "PAUSE"}
    assert protocol["future_two_axis_gate"]["production_lambda_selection"] is False
    assert protocol["prohibitions"]["production_lambda_selection"] is False
