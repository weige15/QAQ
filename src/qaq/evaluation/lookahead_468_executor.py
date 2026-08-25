"""Fail-closed dispatcher and deterministic inert paired 12-trial plan.

This module remains standard-library-only.  Execution requests cross into the
separately imported production runtime only after this dispatcher validates the
exact frozen protocol, trial, CUDA device, and destination.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "configs/lookahead_468_training.json"
EXPECTED_CONFIG_SHA256 = "4a62aeb7d8ae90a6349dc9dc8aab6dda4196b54876c4d0546c05808936fefe92"
EXPECTED_S10H_SHA256 = "7d9e0aff3b686570be0d1d57b5513ee921d60bd5470f275b0cd7cbb4fd63db20"
PLAN_SCHEMA = "qaq-s11d2-paired-lookahead-468-plan-v1"
ARMS = (
    ("same_unit_468_control", "same_unit"),
    ("lookahead_attention_one_unit_468_treatment", "lookahead_attention_one_unit"),
)
LAMBDAS = (0.0, 0.03)
SEEDS = (1729, 1730, 1731)
WITHIN_SEED_ORDER = (
    (ARMS[0][0], LAMBDAS[0]),
    (ARMS[1][0], LAMBDAS[0]),
    (ARMS[0][0], LAMBDAS[1]),
    (ARMS[1][0], LAMBDAS[1]),
)
INITIAL_HASHES = {
    1729: "7b5b5bd2a1ed89b98c0c1358e6a38f5579d0919d0ffc980e06aa7ad09a434123",
    1730: "cca1b7cf3c06679fa4b2178ee2e8dfa4100a07738d0f1d4c9e928b4a08c0d55a",
    1731: "c96ce0f8da7541ecb13594458772d8a254bcb8c378d52096554dd53257b8baf1",
}
TRAIN_IDS = (
    "train-3",
    "train-1003",
    "train-2002",
    "train-3037",
    "train-4005",
    "train-5002",
    "train-6001",
    "train-7001",
    "train-8000",
    "train-9068",
    "train-10001",
    "train-11003",
    "train-12003",
    "train-13000",
    "train-14000",
    "train-15000",
    "train-16002",
    "train-17002",
    "train-18000",
    "train-19001",
    "train-20006",
    "train-21000",
    "train-22000",
    "train-23000",
)
VALIDATION_IDS = (
    "validation-3",
    "validation-270",
    "validation-500",
    "validation-761",
    "validation-1000",
    "validation-1252",
    "validation-1500",
    "validation-1759",
    "validation-2000",
    "validation-2250",
    "validation-2500",
    "validation-2755",
)
FUTURE_RESULT_PARENT = ROOT / "docs/results/s11d_paired_468"
AGGREGATION_OUTPUT = FUTURE_RESULT_PARENT / "aggregation.json"
_HEX_64 = re.compile(r"[0-9a-f]{64}")
_CUDA_DEVICE = re.compile(r"cuda:[0-9]+")


class ProtocolError(RuntimeError):
    """A frozen-protocol or execution-boundary failure with an outcome."""

    def __init__(self, outcome: str, message: str) -> None:
        super().__init__(f"{outcome}: {message}")
        self.outcome = outcome


def _require(condition: bool, message: str, *, outcome: str = "REVISE") -> None:
    if not condition:
        raise ProtocolError(outcome, message)


def _sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise ProtocolError(
            "PAUSE", f"required frozen source is unavailable: {path}: {exc}"
        ) from exc


def _load_json(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ProtocolError(
            "PAUSE", f"required frozen source is unavailable: {path}: {exc}"
        ) from exc
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ProtocolError("REVISE", f"frozen JSON is invalid: {path}: {exc}") from exc
    _require(isinstance(value, dict), f"frozen JSON root must be an object: {path}")
    return value, raw


def _trial_id(seed: int, arm_id: str, lambda_bit: float) -> str:
    suffix = "0" if lambda_bit == 0.0 else "0p03"
    return f"seed-{seed}__{arm_id}__lambda-{suffix}"


def trial_specs() -> tuple[dict[str, Any], ...]:
    timings = dict(ARMS)
    return tuple(
        {
            "trial_index": index,
            "trial_id": _trial_id(seed, arm_id, lambda_bit),
            "seed": seed,
            "arm_id": arm_id,
            "routing_timing": timings[arm_id],
            "lambda_bit": lambda_bit,
            "initial_router_state_sha256": INITIAL_HASHES[seed],
            "paired_initialization_group": f"seed-{seed}",
            "zero_cost_reference_trial_id": _trial_id(seed, arm_id, 0.0),
            "same_cost_timing_pair_trial_id": _trial_id(
                seed,
                ARMS[1][0] if arm_id == ARMS[0][0] else ARMS[0][0],
                lambda_bit,
            ),
        }
        for index, (seed, (arm_id, lambda_bit)) in enumerate(
            ((seed, cell) for seed in SEEDS for cell in WITHIN_SEED_ORDER), start=1
        )
    )


def _validate_manifest(config: Mapping[str, Any]) -> str:
    data = config["data"]
    path = ROOT / data["canonical_manifest_path"]
    digest = _sha256(path)
    _require(digest == EXPECTED_S10H_SHA256, "canonical S10-H manifest SHA-256 mismatch")
    _require(data["canonical_manifest_sha256"] == digest, "config manifest identity drifted")
    result, _ = _load_json(path)
    dataset = result.get("dataset")
    _require(isinstance(dataset, dict), "canonical S10-H dataset evidence is missing")
    _require(
        (dataset.get("repository"), dataset.get("config"), dataset.get("revision"))
        == ("Salesforce/wikitext", "wikitext-2-raw-v1", config["identities"]["dataset_revision"]),
        "canonical dataset identity drifted",
    )
    for split, expected_ids in (("train", TRAIN_IDS), ("validation", VALIDATION_IDS)):
        records = dataset.get(f"{split}_manifest")
        _require(isinstance(records, list), f"canonical {split} manifest is missing")
        _require(
            tuple(item.get("example_id") for item in records) == expected_ids,
            f"canonical {split} manifest IDs/order drifted",
        )
        _require(
            len(records) == len({item["example_id"] for item in records}),
            f"canonical {split} manifest contains duplicate IDs",
        )
        for item in records:
            _require(item.get("split") == split, f"canonical {split} split identity drifted")
            _require(isinstance(item.get("source_row"), int), f"canonical {split} row is invalid")
            _require(
                isinstance(item.get("source_offset"), int), f"canonical {split} offset is invalid"
            )
            _require(
                isinstance(item.get("source_text_sha256"), str)
                and _HEX_64.fullmatch(item["source_text_sha256"]) is not None,
                f"canonical {split} text digest is invalid",
            )
            _require(
                isinstance(item.get("input_ids_sha256"), str)
                and _HEX_64.fullmatch(item["input_ids_sha256"]) is not None,
                f"canonical {split} token digest is invalid",
            )
            _require(item.get("prompt_token_range") == [0, 32], "prompt range drifted")
            _require(item.get("completion_token_range") == [32, 64], "completion range drifted")
    return digest


def _validate_protocol(config: Mapping[str, Any]) -> None:
    _require(config.get("schema") == "qaq-s11d-paired-lookahead-468-v1", "schema drifted")
    _require(config.get("stage") == "S11-D1", "stage identity drifted")
    matrix = config["matrix"]
    _require(
        tuple((item["id"], item["routing_timing"]) for item in matrix["arms"]) == ARMS,
        "arms or routing timings drifted",
    )
    _require(tuple(matrix["lambdas"]) == LAMBDAS, "lambda set/order drifted")
    _require(tuple(matrix["seeds"]) == SEEDS, "seed set/order drifted")
    _require(matrix["candidate_bits"] == [4, 6, 8], "candidate bits/order drifted")
    _require(matrix["probability_order"] == ["p4", "p6", "p8"], "probability order drifted")
    _require(matrix["normalized_bit_costs"] == [0.0, 0.5, 1.0], "bit costs drifted")
    _require(matrix["trial_count"] == 12, "trial count drifted")
    _require(
        tuple((item[0], item[1]) for item in matrix["within_seed_order"]) == WITHIN_SEED_ORDER,
        "within-seed trial order drifted",
    )
    _require(
        {
            int(seed): digest
            for seed, digest in matrix["initial_router_state_sha256_by_seed"].items()
        }
        == INITIAL_HASHES,
        "initial router identities drifted",
    )
    _require(tuple(config["data"]["train_ids"]) == TRAIN_IDS, "training data/order drifted")
    _require(
        tuple(config["data"]["validation_ids"]) == VALIDATION_IDS, "validation data/order drifted"
    )
    _require(len(set(config["data"]["train_ids"])) == 24, "training IDs are duplicated")
    _require(len(set(config["data"]["validation_ids"])) == 12, "validation IDs are duplicated")
    tokenization = config["data"]["tokenization"]
    _require(
        tokenization
        == {
            "add_special_tokens": False,
            "sequence_length": 64,
            "prompt_token_range": [0, 32],
            "completion_token_range": [32, 64],
            "causal_completion_loss_logit_range": [31, 63],
            "selection": "first 64 tokens; no resampling or replacement",
        },
        "tokenization or mask ranges drifted",
    )
    training = config["training"]
    _require(
        {
            key: training[key]
            for key in (
                "examples_seen",
                "batch_size",
                "gradient_accumulation_steps",
                "epochs",
                "optimizer_steps",
                "optimizer",
                "learning_rate",
                "weight_decay",
                "betas",
                "eps",
                "amsgrad",
                "scheduler",
                "routing_temperature",
                "distillation_temperature",
            )
        }
        == {
            "examples_seen": 24,
            "batch_size": 1,
            "gradient_accumulation_steps": 1,
            "epochs": 1,
            "optimizer_steps": 24,
            "optimizer": "AdamW",
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "amsgrad": False,
            "scheduler": "none",
            "routing_temperature": 1.0,
            "distillation_temperature": 2.0,
        },
        "training budget or optimizer drifted",
    )
    _require(
        training["loss_formula"] == "L_total = L_KD + lambda_bit * L_bit", "loss formula drifted"
    )
    evaluation = config["evaluation"]
    _require(evaluation["route_decisions_per_trial"] == 864, "route coverage drifted")
    _require(evaluation["immediate_hard_repeat_count"] == 1, "repeat count drifted")
    _require(
        evaluation["combined_quality_precision_score_allowed"] is False,
        "combined score prohibition drifted",
    )
    _require(
        config["thresholds"]
        == {
            "quality": {
                "lookahead_positive_minus_zero_median_hard_kl_max": 0.0,
                "lookahead_positive_each_request_hard_kl_max_zero_factor": 1.25,
                "lookahead_positive_aggregate_hard_mean_error_max_zero_factor": 1.1,
                "lookahead_positive_aggregate_hard_kl_max_same_unit_positive_factor": 1.1,
                "lookahead_positive_aggregate_hard_mean_error_max_same_unit_positive_factor": 1.1,
                "lookahead_positive_each_request_hard_kl_max_same_unit_positive_factor": 1.25,
            },
            "precision": {
                "lookahead_positive_minus_zero_median_hard_width_max": -0.4907407407407405,
                "strictly_negative_seed_delta_minimum": 2,
            },
            "refine": {
                "quality_near_miss_hard_kl_delta_strict_max": 0.014972516723598044,
                "precision_near_miss_median_reduction_minimum": 0.24537037037037025,
                "precision_pass_reduction_minimum": 0.4907407407407405,
            },
        },
        "quality, precision, or REFINE thresholds drifted",
    )
    classifications = config["classifications"]
    _require(
        classifications["precedence"] == ["PAUSE", "REVISE", "CONTINUE", "REFINE", "STOP"],
        "classification precedence drifted",
    )
    _require(
        classifications["quality_and_precision_separate"] is True,
        "quality/precision separation drifted",
    )
    rules = classifications["rules"]
    _require(
        rules["PAUSE"]
        == {
            "required_evidence_unavailable_or_incomplete": True,
            "interrupted_trial_incomplete": True,
            "material_scientific_ambiguity": True,
            "substitution_allowed": False,
        },
        "PAUSE rule drifted",
    )
    _require(
        rules["REVISE"]["defect_classes"]
        == [
            "protocol",
            "identity",
            "pairing",
            "initialization",
            "ordering",
            "optimizer",
            "update_count",
            "freeze",
            "gradient",
            "route_or_provenance",
            "repeat",
            "persistence",
            "regression",
            "prohibited_work",
        ]
        and rules["REVISE"]["complete_evidence_identifies_invalid_protocol_or_execution_evidence"]
        is True
        and rules["REVISE"]["scientific_threshold_or_arm_change_allowed"] is False,
        "REVISE rule drifted",
    )
    _require(
        rules["CONTINUE"]
        == {
            "all_evidence_and_audits_complete_and_valid": True,
            "all_quality_criteria_pass": True,
            "all_selected_precision_criteria_pass": True,
            "production_lambda_selected": False,
        },
        "CONTINUE rule drifted",
    )
    _require(
        rules["REFINE"]
        == {
            "all_other_scientific_criteria_must_pass": True,
            "regions": [
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
            ],
        },
        "REFINE rules drifted",
    )
    _require(
        rules["STOP"]
        == {
            "complete_valid_evidence": True,
            "outside_continue_and_exact_refine_regions": True,
        },
        "STOP rule drifted",
    )
    _require(
        config["execution"]["real_execution_authorized_in_s11_d2"] is False,
        "S11-D2 execution boundary drifted",
    )
    _require(
        config["execution"]["future_result_parent"] == "docs/results/s11d_paired_468",
        "future result boundary drifted",
    )
    _require(
        len(trial_specs()) == 12 and len({item["trial_id"] for item in trial_specs()}) == 12,
        "derived trial identities are incomplete or duplicated",
    )


def load_protocol(
    config_path: Path = DEFAULT_CONFIG, *, require_results_absent: bool = True
) -> tuple[dict[str, Any], str]:
    config_path = Path(config_path)
    _require(
        config_path.resolve() == DEFAULT_CONFIG.resolve(),
        "only the frozen S11-D1 config is accepted",
    )
    config, raw = _load_json(config_path)
    digest = hashlib.sha256(raw).hexdigest()
    _require(digest == EXPECTED_CONFIG_SHA256, "frozen S11-D1 config SHA-256 mismatch")
    _validate_protocol(config)
    _validate_manifest(config)
    if require_results_absent:
        _require(
            not os.path.lexists(FUTURE_RESULT_PARENT),
            f"S11-D result parent must remain absent in D2: {FUTURE_RESULT_PARENT}",
            outcome="PAUSE",
        )
    return config, digest


def plan(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config, digest = load_protocol(config_path)
    specs = trial_specs()
    commands = [
        [
            "python",
            "scripts/run_lookahead_468_training.py",
            "--execute-trial",
            item["trial_id"],
            "--device",
            "<cuda-device>",
            "--output",
            f"docs/results/s11d_paired_468/{item['trial_id']}.json",
            "--config",
            "configs/lookahead_468_training.json",
        ]
        for item in specs
    ]
    return {
        "schema": PLAN_SCHEMA,
        "protocol_sha256": digest,
        "canonical_manifest_sha256": EXPECTED_S10H_SHA256,
        "trial_count": 12,
        "trial_order": [item["trial_id"] for item in specs],
        "trials": list(specs),
        "trial_commands": commands,
        "aggregation_command": [
            "python",
            "scripts/run_lookahead_468_training.py",
            "--aggregate",
            "--output",
            "docs/results/s11d_paired_468/aggregation.json",
            "--config",
            "configs/lookahead_468_training.json",
        ],
        "candidate_bits": [4, 6, 8],
        "probability_order": ["p4", "p6", "p8"],
        "train_ids": list(TRAIN_IDS),
        "validation_ids": list(VALIDATION_IDS),
        "optimizer_steps_per_trial": 24,
        "request_metrics": list(config["evaluation"]["request_metrics"]),
        "thresholds": config["thresholds"],
        "classification_precedence": config["classifications"]["precedence"],
        "model_loading": False,
        "dataset_loading": False,
        "cuda_activity": False,
        "training": False,
        "evaluation": False,
        "checkpoint_creation": False,
        "result_write_activity": False,
        "execution_authorized": False,
    }


def validate_execution_request(
    *, trial_id: str, device: str, output: Path, config_path: Path = DEFAULT_CONFIG
) -> dict[str, Any]:
    load_protocol(config_path, require_results_absent=False)
    known = {item["trial_id"]: item for item in trial_specs()}
    _require(trial_id in known, f"unknown, missing, or non-frozen trial ID: {trial_id!r}")
    _require(
        _CUDA_DEVICE.fullmatch(device or "") is not None,
        "execution requires an explicit cuda:<index> device",
    )
    expected = FUTURE_RESULT_PARENT / f"{trial_id}.json"
    _require(Path(output).resolve() == expected.resolve(), f"trial output must be {expected}")
    if os.path.lexists(FUTURE_RESULT_PARENT):
        _require(
            FUTURE_RESULT_PARENT.is_dir() and not FUTURE_RESULT_PARENT.is_symlink(),
            f"trial result parent must be a real directory: {FUTURE_RESULT_PARENT}",
            outcome="PAUSE",
        )
    _require(
        not os.path.lexists(expected), f"trial output already exists: {expected}", outcome="PAUSE"
    )
    return dict(known[trial_id])


def validate_aggregation_request(
    *, output: Path, config_path: Path = DEFAULT_CONFIG
) -> tuple[Path, ...]:
    load_protocol(config_path, require_results_absent=False)
    _require(
        Path(output).resolve() == AGGREGATION_OUTPUT.resolve(),
        f"aggregation output must be {AGGREGATION_OUTPUT}",
    )
    _require(
        FUTURE_RESULT_PARENT.is_dir() and not FUTURE_RESULT_PARENT.is_symlink(),
        "aggregation requires the real canonical result directory",
        outcome="PAUSE",
    )
    expected = tuple(FUTURE_RESULT_PARENT / f"{item['trial_id']}.json" for item in trial_specs())
    for path in expected:
        _require(
            path.is_file() and not path.is_symlink(),
            f"aggregation requires complete trial evidence: {path}",
            outcome="PAUSE",
        )
    allowed = {path.name for path in expected}
    _require(
        {path.name for path in FUTURE_RESULT_PARENT.iterdir()} == allowed,
        "aggregation result directory is incomplete or contains unexpected evidence",
        outcome="PAUSE",
    )
    _require(
        not os.path.lexists(AGGREGATION_OUTPUT),
        f"aggregation output already exists: {AGGREGATION_OUTPUT}",
        outcome="PAUSE",
    )
    return expected


__all__ = [
    "AGGREGATION_OUTPUT",
    "ARMS",
    "DEFAULT_CONFIG",
    "EXPECTED_CONFIG_SHA256",
    "EXPECTED_S10H_SHA256",
    "FUTURE_RESULT_PARENT",
    "INITIAL_HASHES",
    "LAMBDAS",
    "PLAN_SCHEMA",
    "SEEDS",
    "TRAIN_IDS",
    "VALIDATION_IDS",
    "WITHIN_SEED_ORDER",
    "ProtocolError",
    "load_protocol",
    "plan",
    "trial_specs",
    "validate_aggregation_request",
    "validate_execution_request",
]
