#!/usr/bin/env python3
"""Validate and plan the protocol-locked S10-H broader-validation run.

The default and ``--plan`` paths retain the S10-H1 standard-library-only
boundary: they load no model, dataset, CUDA, training, or result executor.
The explicit H2-A dispatch lazily imports the real executor after validating
the frozen protocol, predecessor identities, and future-result shape. A
deterministic synthetic fixture is provided for testing that validator only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = ROOT / "configs/s10g_broader_validation.json"
RESULT_PATH = ROOT / "docs/results/s10h_broader_validation.json"
MANIFEST_PATH = ROOT / "docs/quantized_model_manifest.json"
ANY_PRECISION_PATH = ROOT / "third_party/any-precision-llm"
MODEL_SNAPSHOT = (
    Path.home()
    / ".cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/"
    / "1cfa9a7208912126459214e8b04321603b3df60c"
)

LOCKED_CONFIG_SHA256 = "fcb66902174558e5d3f9198f34a8430b685568fd4e21e1632b40f6870aa4aec7"
LOCKED_MANIFEST_SHA256 = "1e2b3515072e22d71ac35a35a3002e3a1dcd5ce44887c554b1408f735c928530"
REQUIRED_ANCESTOR = "7fc136eabdba302e199354ae001cd1e1cd42199f"
MODEL_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
DATASET_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
ANY_PRECISION_REVISION = "a3257d02740cc5757c78673da534b0630ff3a4ea"
ARTIFACT_SHA256 = "29d9bc526b3da0bd39daf2f82afd141f82d005ca1232cabc75cfe9d9ecc1cfee"
PACKED_ARTIFACT = "quantized/s03b_qwen3_4b/backend_cache/packed/anyprec-(1cfa9a7208912126459214e8b04321603b3df60c)-w8_orig4-gc1-c4_s1_blk64"
HISTORICAL_S07_CHECKPOINT_SHA256 = (
    "08bf646f19759c0d7949e159bdbe4f96bbea737204b96f8760d205c8d6fd1949"
)
HISTORICAL_ATTEMPT_1_SHA256 = "d68f041e0a3dc32c465e8b8068ca3ab230d39253757e30f3019ca7e681b14233"
HISTORICAL_ATTEMPT_2_SHA256 = "b3bcc0e45d45852ac5060209c4789453ed452462f528f7bffd4cb80fb1ef58cb"
HISTORICAL_ATTEMPT_1_PATH = ROOT / "docs/results/s10f_frontier_confirmation.json"
HISTORICAL_ATTEMPT_2_PATH = ROOT / "docs/results/s10f_frontier_confirmation_rerun.json"

CANDIDATE_BITS = (4, 6, 8)
SEEDS = (1729, 1730, 1731)
LAMBDAS = (0.0, 0.03, 0.1)
TRIAL_PAIRS = tuple((seed, value) for seed in SEEDS for value in LAMBDAS)
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
TRAIN_OFFSETS = tuple(range(0, 24000, 1000))
TRAIN_ROWS = (
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
)
VALIDATION_OFFSETS = tuple(range(0, 3000, 250))
VALIDATION_ROWS = (3, 270, 500, 761, 1000, 1252, 1500, 1759, 2000, 2250, 2500, 2755)
TRAIN_IDS = tuple(f"train-{row}" for row in TRAIN_ROWS)

UNIT_ORDER = (
    "layer-major: layer 0 attention, layer 0 ffn, then layer 1 attention, "
    "layer 1 ffn, through layer 35"
)
FORBIDDEN_MEASUREMENTS = ("latency", "memory", "transfer", "throughput", "energy", "hardware_cost")
FORBIDDEN_ACTIONS = (
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
)
REQUIRED_PER_TRIAL_FIELDS = frozenset(
    {
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
)
REQUIRED_AGGREGATE_FIELDS = (
    "per_lambda_median_hard_validation_kd",
    "per_lambda_median_hard_mean_selected_bit_width",
    "per_seed_hard_frontier_membership_for_lambda_0.03",
    "lambda_0.03_frontier_seed_count",
    "paired_control_hard_kd_delta_median_lambda_0.03_minus_lambda_0.0",
    "paired_control_hard_width_delta_median_lambda_0.03_minus_lambda_0.0",
    "reproducibility_failure_count",
)


class ProtocolError(RuntimeError):
    """A fail-closed validation error with its prescribed gate outcome."""

    def __init__(self, outcome: str, message: str) -> None:
        super().__init__(f"{outcome}: {message}")
        self.outcome = outcome


class CanonicalResultExists(ProtocolError):
    def __init__(self, path: Path) -> None:
        super().__init__("PAUSE", f"canonical result already exists; refusing overwrite: {path}")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_names(names: list[str]) -> str:
    return hashlib.sha256("\n".join(names).encode()).hexdigest()


EXPECTED_ROUTER_PARAMETER_NAMES = tuple(
    f"routers.{unit}_{layer}.{projection}.{parameter}"
    for unit in ("attention", "ffn")
    for layer in sorted(range(36), key=str)
    for projection in ("input_projection", "output_projection")
    for parameter in ("bias", "weight")
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_path(path: Path) -> Path:
    return path.expanduser().resolve()


def _is_canonical_result_path(path: Path) -> bool:
    return _normalize_path(path) == _normalize_path(RESULT_PATH)


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _require(condition: bool, outcome: str, message: str) -> None:
    if not condition:
        raise ProtocolError(outcome, message)


def _canonical_config() -> dict[str, Any]:
    try:
        raw = CONFIG_PATH.read_bytes()
    except OSError as exc:
        raise ProtocolError("PAUSE", f"frozen S10-G config is unavailable: {exc}") from exc
    if _sha256_bytes(raw) != LOCKED_CONFIG_SHA256:
        raise ProtocolError(
            "REVISE", "default S10-G config differs byte-for-byte from the frozen protocol"
        )
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ProtocolError("REVISE", f"frozen S10-G config is not valid JSON: {exc}") from exc


def _same_json_order(left: Any, right: Any) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        return list(left) == list(right) and all(
            _same_json_order(left[key], right[key]) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _same_json_order(item, other) for item, other in zip(left, right, strict=True)
        )
    return left == right


def _validate_protocol(config: dict[str, Any]) -> None:
    """Reject every in-memory mutation of the frozen S10-G protocol."""

    canonical = _canonical_config()
    if not _same_json_order(config, canonical):
        raise ProtocolError("REVISE", "S10-G protocol fields differ from the frozen config")
    _require(
        config.get("format") == "qaq-s10g-broader-validation-v1",
        "REVISE",
        "protocol format drifted",
    )
    _require(config.get("stage") == "S10-G", "REVISE", "protocol stage drifted")
    protocol = config["protocol"]
    _require(
        tuple(protocol["candidate_bits"]) == CANDIDATE_BITS, "REVISE", "candidate ordering drifted"
    )
    _require(tuple(protocol["seeds"]) == SEEDS, "REVISE", "seed ordering drifted")
    _require(
        tuple(protocol["lambdas"]) == LAMBDAS and protocol["trial_count"] == 9,
        "REVISE",
        "trial matrix drifted",
    )
    _require(
        protocol["required_ancestor_commit"] == REQUIRED_ANCESTOR,
        "REVISE",
        "required ancestor drifted",
    )
    _require(
        config["prohibitions"]["production_lambda_selection"] is False,
        "REVISE",
        "production selection is not prohibited",
    )
    _require(
        config["prohibitions"]["s10h_execution"] is False,
        "REVISE",
        "S10-H execution prohibition drifted",
    )
    route_contract = config["protocol"]["future_measurements"]["route_map_contract"]
    _require(
        route_contract["validation_ids_in_order"] == list(VALIDATION_IDS),
        "REVISE",
        "validation route-map order drifted",
    )
    _require(
        route_contract["units_per_map"] == 72
        and route_contract["allowed_bits"] == list(CANDIDATE_BITS),
        "REVISE",
        "route-map contract drifted",
    )
    _require(
        config["protocol"]["future_measurements"]["entropy_log_base"] == 2.0,
        "REVISE",
        "entropy base drifted",
    )
    _require(
        config["future_two_axis_gate"]["outcome_precedence"]
        == ["PAUSE", "REVISE", "REFINE", "CONTINUE"],
        "REVISE",
        "gate precedence drifted",
    )


def _load_frozen_config(path: Path | None = None) -> dict[str, Any]:
    path = CONFIG_PATH if path is None else path
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ProtocolError("PAUSE", f"frozen S10-G config is unavailable: {exc}") from exc
    if _sha256_bytes(raw) != LOCKED_CONFIG_SHA256:
        raise ProtocolError("REVISE", "S10-G config differs byte-for-byte from the frozen protocol")
    try:
        config = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ProtocolError("REVISE", f"frozen S10-G config is not valid JSON: {exc}") from exc
    _validate_protocol(config)
    return config


def _git(*args: str, check: bool = True) -> str:
    process = subprocess.run(
        ["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=False
    )
    if check and process.returncode != 0:
        raise ProtocolError("PAUSE", f"git prerequisite failed: git {' '.join(args)}")
    return process.stdout.strip()


def _validate_ancestry() -> str:
    head = _git("rev-parse", "HEAD")
    process = subprocess.run(
        ["git", "-C", str(ROOT), "merge-base", "--is-ancestor", REQUIRED_ANCESTOR, head],
        capture_output=True,
        text=True,
        check=False,
    )
    if process.returncode != 0:
        raise ProtocolError("PAUSE", f"required ancestor {REQUIRED_ANCESTOR} is unavailable")
    return head


def _validate_historical_artifacts() -> dict[str, str]:
    expected = (
        (HISTORICAL_ATTEMPT_1_PATH, HISTORICAL_ATTEMPT_1_SHA256),
        (HISTORICAL_ATTEMPT_2_PATH, HISTORICAL_ATTEMPT_2_SHA256),
    )
    hashes: dict[str, str] = {}
    for path, digest in expected:
        if not path.is_file():
            raise ProtocolError("PAUSE", f"historical S10-F artifact is unavailable: {path}")
        actual = _sha256_bytes(path.read_bytes())
        if actual != digest:
            raise ProtocolError("REVISE", f"historical S10-F artifact identity changed: {path}")
        hashes[str(path.relative_to(ROOT))] = actual
    return hashes


def _validate_frozen_identity(config: dict[str, Any]) -> dict[str, Any]:
    facts = config["source_project_established_facts"]
    _require(
        facts["pinned_model"]
        == {
            "repository": "Qwen/Qwen3-4B",
            "revision": MODEL_REVISION,
            "tokenizer_revision": MODEL_REVISION,
        },
        "REVISE",
        "model or tokenizer identity drifted",
    )
    _require(
        facts["pinned_backend"]["any_precision_revision"] == ANY_PRECISION_REVISION,
        "REVISE",
        "Any-Precision identity drifted",
    )
    _require(
        facts["pinned_backend"]["packed_artifact_pytorch_model_sha256"] == ARTIFACT_SHA256,
        "REVISE",
        "packed artifact identity drifted",
    )
    if not MANIFEST_PATH.is_file():
        raise ProtocolError("PAUSE", f"quantized model manifest is unavailable: {MANIFEST_PATH}")
    if _sha256_bytes(MANIFEST_PATH.read_bytes()) != LOCKED_MANIFEST_SHA256:
        raise ProtocolError("REVISE", "quantized model manifest identity changed")
    try:
        manifest = json.loads(MANIFEST_PATH.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError("REVISE", f"quantized model manifest is invalid: {exc}") from exc
    artifact = facts["pinned_backend"]["packed_artifact"]
    _require(
        manifest["source_model"]["repository"] == "Qwen/Qwen3-4B",
        "REVISE",
        "manifest model identity drifted",
    )
    _require(
        manifest["source_model"]["revision"] == MODEL_REVISION,
        "REVISE",
        "manifest model revision drifted",
    )
    _require(
        manifest["any_precision"]["commit"] == ANY_PRECISION_REVISION,
        "REVISE",
        "manifest backend revision drifted",
    )
    _require(
        manifest["artifact"]["local_path"] == artifact, "REVISE", "manifest artifact path drifted"
    )
    _require(
        manifest["artifact"]["checkpoint_hashes"]["pytorch_model.bin"] == ARTIFACT_SHA256,
        "REVISE",
        "manifest artifact hash drifted",
    )
    artifact_path = ROOT / artifact
    artifact_file = artifact_path / "pytorch_model.bin"
    if not artifact_path.is_dir() or not artifact_file.is_file():
        raise ProtocolError(
            "PAUSE", f"identity-matched packed artifact is unavailable: {artifact_path}"
        )
    if _sha256_file(artifact_file) != ARTIFACT_SHA256:
        raise ProtocolError("REVISE", "packed artifact bytes differ from the frozen identity")
    if not MODEL_SNAPSHOT.is_dir():
        raise ProtocolError("PAUSE", f"pinned model snapshot is unavailable: {MODEL_SNAPSHOT}")
    # Use a separate command because _git is intentionally rooted at this worktree.
    revision_process = subprocess.run(
        ["git", "-C", str(ANY_PRECISION_PATH), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if (
        revision_process.returncode != 0
        or revision_process.stdout.strip() != ANY_PRECISION_REVISION
    ):
        raise ProtocolError("PAUSE", "pinned Any-Precision checkout is unavailable")
    dirty = subprocess.run(
        ["git", "-C", str(ANY_PRECISION_PATH), "status", "--porcelain=v1"],
        capture_output=True,
        text=True,
        check=False,
    )
    if dirty.returncode != 0 or dirty.stdout.strip():
        raise ProtocolError("REVISE", "pinned Any-Precision checkout is dirty")
    return {
        "model_repository": "Qwen/Qwen3-4B",
        "model_revision": MODEL_REVISION,
        "tokenizer_revision": MODEL_REVISION,
        "dataset_revision": DATASET_REVISION,
        "any_precision_revision": ANY_PRECISION_REVISION,
        "packed_artifact": artifact,
        "packed_artifact_pytorch_model_sha256": ARTIFACT_SHA256,
        "artifact_path_present": artifact_path.is_dir(),
        "manifest_sha256": LOCKED_MANIFEST_SHA256,
        "historical_s07_checkpoint_used": False,
        "historical_s07_checkpoint_sha256": HISTORICAL_S07_CHECKPOINT_SHA256,
    }


def _validate_historical_facts(config: dict[str, Any]) -> dict[str, str]:
    _validate_frozen_identity(config)
    return _validate_historical_artifacts()


def _validate_pre_execution(
    *, config_path: Path = CONFIG_PATH, result_path: Path = RESULT_PATH
) -> dict[str, Any]:
    result_path = _normalize_path(result_path)
    config = _load_frozen_config(config_path)
    head = _validate_ancestry()
    identities = _validate_frozen_identity(config)
    historical = _validate_historical_artifacts()
    if result_path.exists():
        raise CanonicalResultExists(result_path)
    return {
        "config": config,
        "head": head,
        "historical_hashes": historical,
        "identities": identities,
    }


def _validate_trial_pairs(trials: Any) -> tuple[str, list[tuple[int, float]]]:
    if not isinstance(trials, list):
        return "pause", []
    pairs: list[tuple[int, float]] = []
    for trial in trials:
        if not isinstance(trial, dict) or "seed" not in trial or "lambda_bit" not in trial:
            return "pause", pairs
        try:
            pairs.append((int(trial["seed"]), float(trial["lambda_bit"])))
        except (TypeError, ValueError):
            return "revise", pairs
    actual = tuple(pairs)
    expected = TRIAL_PAIRS
    if len(actual) < len(expected) or any(pair not in actual for pair in expected):
        return "pause", pairs
    if len(actual) != len(expected) or len(set(actual)) != len(expected) or actual != expected:
        return "revise", pairs
    return "ok", pairs


def _validate_route_map(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 72
        and all(
            isinstance(bit, int) and not isinstance(bit, bool) and bit in CANDIDATE_BITS
            for bit in value
        )
    )


def _route_stats(maps: dict[str, list[int]]) -> tuple[float, dict[str, float], int]:
    values = [bit for route in maps.values() for bit in route]
    fractions = {str(bit): values.count(bit) / len(values) for bit in CANDIDATE_BITS}
    return sum(values) / len(values), fractions, len({tuple(route) for route in maps.values()})


def _is_frontier(points: list[tuple[float, float, float]], candidate: float) -> bool:
    current = next((point for point in points if point[0] == candidate), None)
    if current is None:
        return False
    return not any(
        other[0] != candidate
        and other[1] <= current[1]
        and other[2] <= current[2]
        and (other[1] < current[1] or other[2] < current[2])
        for other in points
    )


def _aggregate_trials(trials: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[int, dict[float, dict[str, Any]]] = {seed: {} for seed in SEEDS}
    for trial in trials:
        grouped[int(trial["seed"])][float(trial["lambda_bit"])] = trial
    kd = {
        str(value): statistics.median(grouped[seed][value]["hard_validation_kd"] for seed in SEEDS)
        for value in LAMBDAS
    }
    width = {
        str(value): statistics.median(
            grouped[seed][value]["hard_validation_mean_selected_bit_width"] for seed in SEEDS
        )
        for value in LAMBDAS
    }
    membership: dict[str, bool] = {}
    kd_deltas: list[float] = []
    width_deltas: list[float] = []
    for seed in SEEDS:
        points = [
            (
                value,
                float(grouped[seed][value]["hard_validation_kd"]),
                float(grouped[seed][value]["hard_validation_mean_selected_bit_width"]),
            )
            for value in LAMBDAS
        ]
        membership[str(seed)] = _is_frontier(points, 0.03)
        kd_deltas.append(points[1][1] - points[0][1])
        width_deltas.append(points[1][2] - points[0][2])
    return {
        "per_lambda_median_hard_validation_kd": kd,
        "per_lambda_median_hard_mean_selected_bit_width": width,
        "per_seed_hard_frontier_membership_for_lambda_0.03": membership,
        "lambda_0.03_frontier_seed_count": sum(membership.values()),
        "paired_control_hard_kd_delta_median_lambda_0.03_minus_lambda_0.0": statistics.median(
            kd_deltas
        ),
        "paired_control_hard_width_delta_median_lambda_0.03_minus_lambda_0.0": statistics.median(
            width_deltas
        ),
        "reproducibility_failure_count": sum(
            not trial["reproducibility_audit"]["passed"] for trial in trials
        ),
    }


def _same_number(left: Any, right: Any) -> bool:
    return (
        _finite_number(left)
        and _finite_number(right)
        and math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-12)
    )


def _reject_forbidden_result_fields(value: Any, path: str = "result") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in FORBIDDEN_MEASUREMENTS:
                raise ProtocolError("REVISE", f"prohibited measurement field {path}.{key}")
            _reject_forbidden_result_fields(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_forbidden_result_fields(child, f"{path}[{index}]")


def _validate_optimizer_audit(audit: Any) -> bool:
    if not isinstance(audit, dict):
        return False
    required = {
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
    if not required.issubset(audit):
        return False
    expected = audit["expected_router_parameter_names"]
    actual = audit["actual_optimizer_parameter_names"]
    canonical = list(EXPECTED_ROUTER_PARAMETER_NAMES)
    if not isinstance(expected, list) or not isinstance(actual, list):
        return False
    if actual != canonical or expected != canonical:
        return False
    if any(not isinstance(name, str) or not name.startswith("routers.") for name in actual):
        return False
    actual_set = set(actual)
    duplicate_count = len(actual) - len(actual_set)
    missing_count = len(set(canonical) - actual_set)
    unexpected_count = len(actual_set - set(canonical))
    return (
        audit["expected_router_parameter_count"] == len(canonical)
        and audit["actual_optimizer_parameter_count"] == len(actual)
        and audit["expected_router_parameter_names_sha256"] == _sha256_names(canonical)
        and audit["actual_optimizer_parameter_names_sha256"] == _sha256_names(actual)
        and audit["duplicate_optimizer_parameter_count"] == duplicate_count == 0
        and audit["missing_router_parameter_count"] == missing_count == 0
        and audit["unexpected_optimizer_parameter_count"] == unexpected_count == 0
        and isinstance(audit["optimizer_construction_serial"], int)
        and not isinstance(audit["optimizer_construction_serial"], bool)
        and audit["optimizer_construction_serial"] > 0
        and audit["optimizer_state_entry_count_before_first_step"] == 0
        and audit["optimizer_state_entry_count_before_training_begins"] == 0
        and audit["router_only_optimizer_audit"] is True
        and audit["runtime_identity_based_membership_result"] is True
        and audit["fresh_adamw_audit"] is True
    )


def _validate_trial(trial: dict[str, Any]) -> tuple[bool, bool, str | None]:
    if not isinstance(trial, dict):
        return False, False, "trial evidence is not an object"
    missing = REQUIRED_PER_TRIAL_FIELDS - trial.keys()
    if missing:
        return False, False, f"trial is missing fields: {sorted(missing)}"
    candidate_bits = trial.get("candidate_bits", CANDIDATE_BITS)
    if not isinstance(candidate_bits, (list, tuple)):
        return False, False, "trial candidate ordering is incomplete"
    if tuple(candidate_bits) != CANDIDATE_BITS:
        return True, False, "trial candidate ordering changed"
    if trial["training_examples_seen"] != 24 or trial["optimizer_steps_completed"] != 24:
        return True, False, "training example/update count drifted"
    if not isinstance(trial.get("training_history"), list) or len(trial["training_history"]) != 24:
        return False, False, "training history is missing or incomplete"
    if any(not isinstance(item, dict) for item in trial["training_history"]):
        return False, False, "training history is malformed"
    if [item.get("step") for item in trial["training_history"]] != list(range(1, 25)):
        return True, False, "training update order drifted"
    for key in (
        "finite_loss_audit",
        "finite_gradient_audit",
        "teacher_frozen_audit",
        "packed_student_base_unchanged_audit",
    ):
        if trial[key] is not True:
            return True, False, f"{key} failed"
    collapse = trial["collapse_audit"]
    if not isinstance(collapse, dict):
        return False, False, "collapse audit is incomplete"
    classification = collapse.get("classification")
    if not isinstance(classification, str) or classification not in {
        "PROMPT_INVARIANT",
        "ADAPTIVE_OBSERVED",
        "OTHER",
    }:
        return True, False, "invalid collapse classification"
    invalid_or_degenerate = collapse.get("invalid_or_degenerate")
    passed = collapse.get("passed")
    if not isinstance(invalid_or_degenerate, bool) or not isinstance(passed, bool):
        return False, False, "collapse audit is incomplete"
    if invalid_or_degenerate or passed is not (not invalid_or_degenerate):
        return True, False, "collapse audit failed"
    if not _validate_optimizer_audit(trial["optimizer_audit"]):
        return True, False, "optimizer audit failed"
    prohibited = trial["prohibited_measurement_audit"]
    if (
        not isinstance(prohibited, dict)
        or prohibited.get("passed") is not True
        or prohibited.get("forbidden_measurements_observed") != []
    ):
        return True, False, "prohibited-measurement audit failed"
    reproducibility = trial["reproducibility_audit"]
    if not isinstance(reproducibility, dict):
        return False, False, "reproducibility audit is incomplete"
    for key in (
        "route_maps_identical",
        "hard_metrics_identical",
        "finite_outputs_both_passed",
        "passed",
    ):
        if key not in reproducibility or not isinstance(reproducibility[key], bool):
            return False, False, "reproducibility audit is incomplete"
    subaudits_passed = all(
        reproducibility[key]
        for key in ("route_maps_identical", "hard_metrics_identical", "finite_outputs_both_passed")
    )
    if (
        reproducibility.get("repeat_count") != 1
        or not subaudits_passed
        or reproducibility.get("passed") is not True
    ):
        return True, False, "reproducibility audit failed"
    maps = trial["hard_validation_route_maps"]
    if not isinstance(maps, dict):
        return False, False, "route maps are missing"
    if list(maps) != list(VALIDATION_IDS):
        return True, False, "validation route-map IDs/order changed"
    if any(not _validate_route_map(maps[request_id]) for request_id in VALIDATION_IDS):
        return False, False, "a validation route map is incomplete or malformed"
    width, fractions, unique = _route_stats(maps)
    if trial["distinct_hard_route_map_count"] != unique or not _same_number(
        trial["hard_validation_mean_selected_bit_width"], width
    ):
        return True, False, "route-map aggregate changed"
    for bit in CANDIDATE_BITS:
        if not _same_number(trial[f"hard_validation_fraction_{bit}"], fractions[str(bit)]):
            return True, False, "hard route fractions changed"
    numeric_fields = [
        key
        for key in REQUIRED_PER_TRIAL_FIELDS
        if key.endswith(("kd", "error", "width", "p4", "p6", "p8", "entropy", "ratio", "norm"))
    ]
    if any(not _finite_number(trial.get(key)) for key in numeric_fields):
        return True, False, "a required numeric measurement is non-finite"
    variation = trial["route_variation"]
    if (
        not isinstance(variation, dict)
        or variation.get("unit_count") != 72
        or variation.get("prompt_count") != 12
    ):
        return True, False, "route variation coverage changed"
    return True, True, None


def validate_result(result: dict[str, Any]) -> dict[str, Any]:
    """Validate a future H2 result; synthetic fixtures are the only H1 caller."""

    if not isinstance(result, dict):
        return {"classification": "PAUSE", "errors": ["result is not an object"]}
    try:
        _reject_forbidden_result_fields(result)
    except ProtocolError as exc:
        return {"classification": exc.outcome, "errors": [str(exc)]}
    errors: list[str] = []
    pause_errors: list[str] = []
    revise_errors: list[str] = []
    if (
        not isinstance(result, dict)
        or result.get("format") != "qaq-s10h-broader-validation-v1"
        or result.get("stage") != "S10-H"
    ):
        pause_errors.append("result format/stage is missing or invalid")
    protocol_identity = result.get("protocol_identity") if isinstance(result, dict) else None
    if (
        not isinstance(protocol_identity, dict)
        or protocol_identity.get("config_sha256") != LOCKED_CONFIG_SHA256
        or protocol_identity.get("config_byte_exact") is not True
    ):
        revise_errors.append("result does not identify the byte-exact frozen config")
    ancestry = result.get("ancestry") if isinstance(result, dict) else None
    if (
        not isinstance(ancestry, dict)
        or ancestry.get("required_ancestor") != REQUIRED_ANCESTOR
        or ancestry.get("ancestor_ok") is not True
    ):
        revise_errors.append("result ancestry proof is missing or drifted")
    router_contract = result.get("router_contract") if isinstance(result, dict) else None
    expected_router_contract = {
        "router_count": 72,
        "router_parameter_count": 23630040,
        "candidate_bits": list(CANDIDATE_BITS),
        "candidate_order": "[p4,p6,p8]",
    }
    if router_contract != expected_router_contract:
        revise_errors.append("result candidate/router contract drifted")
    route_contract = result.get("route_map_contract") if isinstance(result, dict) else None
    expected_route_contract = {
        "validation_ids_in_order": list(VALIDATION_IDS),
        "units_per_map": 72,
        "unit_order": UNIT_ORDER,
        "allowed_bits": list(CANDIDATE_BITS),
    }
    if route_contract != expected_route_contract:
        revise_errors.append("result route-map ordering contract drifted")
    identities = result.get("identities") if isinstance(result, dict) else None
    expected_identity = {
        "model_repository": "Qwen/Qwen3-4B",
        "model_revision": MODEL_REVISION,
        "tokenizer_revision": MODEL_REVISION,
        "dataset_revision": DATASET_REVISION,
        "any_precision_revision": ANY_PRECISION_REVISION,
        "packed_artifact": PACKED_ARTIFACT,
        "manifest_sha256": LOCKED_MANIFEST_SHA256,
        "packed_artifact_pytorch_model_sha256": ARTIFACT_SHA256,
        "historical_s07_checkpoint_used": False,
        "historical_s07_checkpoint_sha256": HISTORICAL_S07_CHECKPOINT_SHA256,
    }
    if not isinstance(identities, dict) or any(
        identities.get(key) != value for key, value in expected_identity.items()
    ):
        revise_errors.append("frozen model/backend/artifact identity drifted")
    dataset = result.get("dataset") if isinstance(result, dict) else None
    if not isinstance(dataset, dict):
        pause_errors.append("dataset evidence is missing")
    else:
        expected_dataset_identity = {
            "repository": "Salesforce/wikitext",
            "config": "wikitext-2-raw-v1",
            "train_split": "train",
            "validation_split": "validation",
            "tokenizer_revision": MODEL_REVISION,
            "revision": DATASET_REVISION,
        }
        for name, expected in expected_dataset_identity.items():
            if name not in dataset:
                pause_errors.append(f"dataset source identity is missing: {name}")
            elif dataset[name] != expected:
                revise_errors.append(f"dataset source identity drifted: {name}")
        for name, expected in (("train_example_count", 24), ("validation_example_count", 12)):
            if name not in dataset:
                pause_errors.append(f"dataset field is missing: {name}")
            elif dataset[name] != expected:
                revise_errors.append(f"dataset count drifted: {name}")
        for name, ids, rows, offsets in (
            ("train_manifest", TRAIN_IDS, TRAIN_ROWS, TRAIN_OFFSETS),
            ("validation_manifest", VALIDATION_IDS, VALIDATION_ROWS, VALIDATION_OFFSETS),
        ):
            manifest = dataset.get(name)
            if not isinstance(manifest, list):
                pause_errors.append(f"dataset manifest is missing: {name}")
                continue
            if len(manifest) != len(ids):
                pause_errors.append(f"dataset manifest is incomplete: {name}")
                continue
            actual_ids = [
                item.get("example_id") if isinstance(item, dict) else None for item in manifest
            ]
            actual_rows = [
                item.get("source_row") if isinstance(item, dict) else None for item in manifest
            ]
            actual_offsets = [
                item.get("source_offset") if isinstance(item, dict) else None for item in manifest
            ]
            if (
                actual_ids != list(ids)
                or actual_rows != list(rows)
                or actual_offsets != list(offsets)
            ):
                revise_errors.append(f"dataset source order drifted: {name}")
    training = result.get("training_contract") if isinstance(result, dict) else None
    expected_training = {
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
    if training is None:
        pause_errors.append("training contract is missing")
    elif training != expected_training:
        revise_errors.append("training contract drifted")
    trial_status, pairs = _validate_trial_pairs(result.get("trials"))
    if trial_status == "pause":
        pause_errors.append("trial matrix is missing or incomplete")
    elif trial_status == "revise":
        revise_errors.append(f"trial matrix is not the exact ordered pairs {TRIAL_PAIRS!r}")
    trials = result.get("trials") if isinstance(result.get("trials"), list) else []
    trials_valid = False
    if trial_status == "ok":
        trials_valid = True
        initial_by_seed: dict[int, set[str]] = {seed: set() for seed in SEEDS}
        serials: list[int] = []
        for trial in trials:
            complete, valid, error = _validate_trial(trial)
            if not complete:
                pause_errors.append(error or "trial evidence is incomplete")
                trials_valid = False
                continue
            elif not valid:
                revise_errors.append(error or "trial evidence failed")
                trials_valid = False
                continue
            initial_by_seed[int(trial["seed"])].add(str(trial["initial_router_state_sha256"]))
            serials.append(
                int(trial["optimizer_audit"]["optimizer_construction_serial"])
                if isinstance(trial.get("optimizer_audit"), dict)
                and isinstance(trial["optimizer_audit"].get("optimizer_construction_serial"), int)
                else -1
            )
        if trials_valid and (
            any(len(values) != 1 for values in initial_by_seed.values())
            or len({next(iter(values)) for values in initial_by_seed.values()}) != 3
        ):
            revise_errors.append("paired canonical router initialization drifted")
        if trials_valid and (
            len(serials) != len(set(serials)) or any(serial <= 0 for serial in serials)
        ):
            revise_errors.append("fresh optimizer construction serials are not unique")
    run_audits = result.get("run_audits") if isinstance(result, dict) else None
    if not isinstance(run_audits, dict):
        pause_errors.append("run-level audits are missing")
    else:
        inherited = run_audits.get("inherited_regressions_audit")
        prohibited = run_audits.get("prohibited_work_audit")
        if not isinstance(inherited, dict) or not {"status", "test_selection", "passed"}.issubset(
            inherited
        ):
            pause_errors.append("inherited regression audit is incomplete")
        elif inherited != {
            "status": "passed",
            "test_selection": "S10-D/S10-E/S10-F predecessor regression selection",
            "passed": True,
        }:
            revise_errors.append("inherited regression audit failed or drifted")
        if not isinstance(prohibited, dict) or not {
            "forbidden_actions_observed",
            "forbidden_measurements_observed",
            "passed",
        }.issubset(prohibited):
            pause_errors.append("prohibited-work audit is incomplete")
        elif (
            prohibited["passed"] is not True
            or prohibited["forbidden_actions_observed"] != []
            or prohibited["forbidden_measurements_observed"] != []
        ):
            revise_errors.append("prohibited-work audit failed")
    aggregates = result.get("aggregates") if isinstance(result, dict) else None
    if not isinstance(aggregates, dict):
        pause_errors.append("cross-seed aggregates are missing")
    elif trial_status == "ok" and trials_valid:
        expected_aggregates = _aggregate_trials(trials)
        for key in REQUIRED_AGGREGATE_FIELDS:
            if key not in aggregates:
                pause_errors.append(f"aggregate field is missing: {key}")
            elif isinstance(expected_aggregates[key], dict):
                if aggregates[key] != expected_aggregates[key]:
                    revise_errors.append(f"aggregate drifted: {key}")
            elif not _same_number(aggregates[key], expected_aggregates[key]):
                revise_errors.append(f"aggregate drifted: {key}")
    gate = result.get("gate") if isinstance(result, dict) else None
    if not isinstance(gate, dict) or "classification" not in gate:
        pause_errors.append("gate evidence is missing")
    else:
        # Gate precedence is evaluated below after the structural evidence pass.
        if gate.get("production_lambda_selected") is not False:
            revise_errors.append("production lambda selection is prohibited")
    errors.extend(pause_errors)
    errors.extend(revise_errors)
    if pause_errors:
        classification = "PAUSE"
    elif revise_errors:
        classification = "REVISE"
    elif not isinstance(aggregates, dict) or not isinstance(gate, dict):
        classification = "PAUSE"
    else:
        checks = {
            "all_required_audits_pass": True,
            "inherited_regressions_pass": True,
            "no_invalid_or_degenerate_collapse": all(
                trial["collapse_audit"]["passed"] for trial in trials
            ),
            "no_prohibited_work": True,
            "reproducibility_failures_zero": aggregates["reproducibility_failure_count"] == 0,
            "frontier": aggregates["lambda_0.03_frontier_seed_count"] >= 2,
            "paired_kd": aggregates[
                "paired_control_hard_kd_delta_median_lambda_0.03_minus_lambda_0.0"
            ]
            <= 0.0,
            "paired_width": aggregates[
                "paired_control_hard_width_delta_median_lambda_0.03_minus_lambda_0.0"
            ]
            < 0.0,
        }
        # Explicitly apply REVISE before REFINE, then the two-axis thresholds.
        if not all(
            checks[key]
            for key in ("no_invalid_or_degenerate_collapse", "reproducibility_failures_zero")
        ):
            classification = "REVISE"
        elif not all(checks[key] for key in ("frontier", "paired_kd", "paired_width")):
            classification = "REFINE"
        else:
            classification = "CONTINUE"
        if gate.get("classification") != classification:
            errors.append("gate classification does not follow PAUSE > REVISE > REFINE > CONTINUE")
            classification = "REVISE"
    return {"classification": classification, "errors": errors, "pairs": pairs}


def _plan(context: dict[str, Any]) -> dict[str, Any]:
    config = context["config"]
    protocol = config["protocol"]
    data = protocol["dataset"]
    return {
        "mode": "plan",
        "protocol_identity": {
            "path": str(CONFIG_PATH.relative_to(ROOT)),
            "sha256": LOCKED_CONFIG_SHA256,
            "format": config["format"],
            "stage": config["stage"],
            "protocol_only": True,
        },
        "ancestry": {
            "required_ancestor": REQUIRED_ANCESTOR,
            "current_head": context["head"],
            "ancestor_ok": True,
        },
        "frozen_revisions": context["identities"],
        "historical_artifacts": context["historical_hashes"],
        "data": {
            "repository": data["repository"],
            "revision": data["revision"],
            "train_rows": 24,
            "validation_rows": 12,
            "train_source_rows": data["train_source_rows"],
            "validation_source_rows": data["validation_source_rows"],
            "train_offsets": data["train_offsets"],
            "validation_offsets": data["validation_offsets"],
            "train_ids_in_order": data["train_example_ids"],
            "validation_ids_in_order": data["validation_example_ids"],
        },
        "trials": {
            "seeds": list(SEEDS),
            "lambdas": list(LAMBDAS),
            "ordered_pairs": [list(pair) for pair in TRIAL_PAIRS],
            "count": 9,
        },
        "training_contract": protocol["training"],
        "future_output": {
            "format": "qaq-s10h-broader-validation-v1",
            "path": str(RESULT_PATH.relative_to(ROOT)),
            "exists": RESULT_PATH.exists(),
            "overwrite": "refuse",
        },
        "prohibitions": config["prohibitions"],
        "prohibited_actions": list(FORBIDDEN_ACTIONS),
        "prohibited_measurements": list(FORBIDDEN_MEASUREMENTS),
        "thresholds": config["future_two_axis_gate"]["required_conditions"],
        "gate_precedence": ["PAUSE", "REVISE", "REFINE", "CONTINUE"],
        "explicit_execute_command": "source ~/.venv/bin/activate && which python && python --version && nvidia-smi && PYTHONPATH=src:third_party/any-precision-llm:. python scripts/run_s10h.py --execute --device cuda:0 --config configs/s10g_broader_validation.json --output <temporary-destination-on-target-filesystem>",
        "execution_allowed_in_h1": False,
        "plan_loads_model": False,
        "plan_trains": False,
        "plan_evaluates_cuda": False,
        "plan_writes_result": False,
    }


def synthetic_structural_fixture() -> dict[str, Any]:
    """Return a deterministic H2-shaped fixture without importing ML code."""

    names = list(EXPECTED_ROUTER_PARAMETER_NAMES)
    train_manifest = [
        {"example_id": id_, "source_row": row, "source_offset": offset}
        for id_, row, offset in zip(TRAIN_IDS, TRAIN_ROWS, TRAIN_OFFSETS, strict=True)
    ]
    validation_manifest = [
        {"example_id": id_, "source_row": row, "source_offset": offset}
        for id_, row, offset in zip(
            VALIDATION_IDS, VALIDATION_ROWS, VALIDATION_OFFSETS, strict=True
        )
    ]
    trials: list[dict[str, Any]] = []
    serial = 1
    for seed in SEEDS:
        for lambda_bit in LAMBDAS:
            if lambda_bit == 0.0:
                route = [4] * 4 + [8] * 68
                kd = 1.0
            elif lambda_bit == 0.03:
                route = [6] * 36 + [8] * 36
                kd = 0.9
            else:
                route = [4] * 4 + [8] * 68
                kd = 0.8
            maps = {request_id: list(route) for request_id in VALIDATION_IDS}
            width, fractions, unique = _route_stats(maps)
            audit = {
                "actual_optimizer_parameter_count": 288,
                "actual_optimizer_parameter_names": names,
                "actual_optimizer_parameter_names_sha256": _sha256_names(names),
                "duplicate_optimizer_parameter_count": 0,
                "expected_router_parameter_count": 288,
                "expected_router_parameter_names": names,
                "expected_router_parameter_names_sha256": _sha256_names(names),
                "fresh_adamw_audit": True,
                "missing_router_parameter_count": 0,
                "optimizer_construction_serial": serial,
                "optimizer_state_entry_count_before_first_step": 0,
                "optimizer_state_entry_count_before_training_begins": 0,
                "router_only_optimizer_audit": True,
                "runtime_identity_based_membership_result": True,
                "unexpected_optimizer_parameter_count": 0,
            }
            trials.append(
                {
                    "seed": seed,
                    "lambda_bit": lambda_bit,
                    "initial_router_state_sha256": f"synthetic-initial-{seed}",
                    "final_router_state_sha256": f"synthetic-final-{seed}-{lambda_bit}",
                    "initial_kd_gradient_norm": 1.0,
                    "initial_bit_cost_gradient_norm": 0.5,
                    "lambda_weighted_gradient_ratio": lambda_bit / 2,
                    "training_examples_seen": 24,
                    "optimizer_steps_completed": 24,
                    "training_history": [{"step": step} for step in range(1, 25)],
                    "finite_loss_audit": True,
                    "finite_gradient_audit": True,
                    "teacher_frozen_audit": True,
                    "packed_student_base_unchanged_audit": True,
                    "collapse_audit": {
                        "classification": "PROMPT_INVARIANT",
                        "invalid_or_degenerate": False,
                        "passed": True,
                    },
                    "optimizer_audit": audit,
                    "soft_validation_kd": kd,
                    "soft_validation_mean_absolute_logit_error": 0.1,
                    "soft_validation_maximum_absolute_logit_error": 0.2,
                    "soft_validation_mean_expected_bit_width": width,
                    "soft_validation_mean_p4": fractions["4"],
                    "soft_validation_mean_p6": fractions["6"],
                    "soft_validation_mean_p8": fractions["8"],
                    "soft_validation_mean_entropy": 0.5,
                    "hard_validation_kd": kd,
                    "hard_validation_mean_absolute_logit_error": 0.1,
                    "hard_validation_maximum_absolute_logit_error": 0.2,
                    "hard_validation_mean_selected_bit_width": width,
                    "hard_validation_fraction_4": fractions["4"],
                    "hard_validation_fraction_6": fractions["6"],
                    "hard_validation_fraction_8": fractions["8"],
                    "hard_validation_route_maps": maps,
                    "route_variation": {
                        "prompt_count": 12,
                        "unit_count": 72,
                        "changed_unit_count": 0,
                        "changed_fraction": 0.0,
                    },
                    "distinct_hard_route_map_count": unique,
                    "reproducibility_audit": {
                        "route_maps_identical": True,
                        "hard_metrics_identical": True,
                        "finite_outputs_both_passed": True,
                        "passed": True,
                        "repeat_count": 1,
                    },
                    "prohibited_measurement_audit": {
                        "forbidden_measurements_observed": [],
                        "passed": True,
                    },
                }
            )
            serial += 1
    result = {
        "format": "qaq-s10h-broader-validation-v1",
        "stage": "S10-H",
        "protocol_identity": {"config_sha256": LOCKED_CONFIG_SHA256, "config_byte_exact": True},
        "ancestry": {
            "required_ancestor": REQUIRED_ANCESTOR,
            "ancestor_ok": True,
            "commit": "synthetic",
        },
        "router_contract": {
            "router_count": 72,
            "router_parameter_count": 23630040,
            "candidate_bits": list(CANDIDATE_BITS),
            "candidate_order": "[p4,p6,p8]",
        },
        "route_map_contract": {
            "validation_ids_in_order": list(VALIDATION_IDS),
            "units_per_map": 72,
            "unit_order": UNIT_ORDER,
            "allowed_bits": list(CANDIDATE_BITS),
        },
        "identities": {
            "model_repository": "Qwen/Qwen3-4B",
            "model_revision": MODEL_REVISION,
            "tokenizer_revision": MODEL_REVISION,
            "dataset_revision": DATASET_REVISION,
            "any_precision_revision": ANY_PRECISION_REVISION,
            "packed_artifact": PACKED_ARTIFACT,
            "manifest_sha256": LOCKED_MANIFEST_SHA256,
            "packed_artifact_pytorch_model_sha256": ARTIFACT_SHA256,
            "historical_s07_checkpoint_used": False,
            "historical_s07_checkpoint_sha256": HISTORICAL_S07_CHECKPOINT_SHA256,
        },
        "dataset": {
            "repository": "Salesforce/wikitext",
            "config": "wikitext-2-raw-v1",
            "train_split": "train",
            "validation_split": "validation",
            "tokenizer_revision": MODEL_REVISION,
            "revision": DATASET_REVISION,
            "train_example_count": 24,
            "validation_example_count": 12,
            "train_manifest": train_manifest,
            "validation_manifest": validation_manifest,
        },
        "training_contract": {
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
        },
        "trials": trials,
        "run_audits": {
            "inherited_regressions_audit": {
                "status": "passed",
                "test_selection": "S10-D/S10-E/S10-F predecessor regression selection",
                "passed": True,
            },
            "prohibited_work_audit": {
                "forbidden_actions_observed": [],
                "forbidden_measurements_observed": [],
                "passed": True,
            },
        },
        "aggregates": _aggregate_trials(trials),
        "gate": {"classification": "CONTINUE", "production_lambda_selected": False},
    }
    return result


def _dispatch_execute(*, context: dict[str, Any], device: str | None, output: Path) -> int:
    """Lazy H2 dispatch; importing the executor is the only heavy boundary."""

    if not device:
        raise ProtocolError("PAUSE", "--execute requires an explicit CUDA device")
    if _is_canonical_result_path(output):
        raise ProtocolError(
            "PAUSE",
            "canonical H2 output is disabled during S10-H2-A; use a temporary test destination",
        )
    if str(ROOT / "src") not in sys.path:
        sys.path.insert(0, str(ROOT / "src"))
    from qaq.router.broader_validation_executor import execute_production

    outcome = execute_production(
        config=context["config"],
        device=device,
        output=output,
        preflight=context,
    )
    print(
        json.dumps(
            {
                "classification": outcome.classification,
                "errors": list(outcome.errors),
                "output_path": outcome.output_path,
                "written": outcome.written,
            },
            sort_keys=True,
        )
    )
    return 0 if outcome.classification in {"CONTINUE", "REFINE"} and outcome.written else 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--plan", action="store_true", help="validate and print the non-executing plan"
    )
    mode.add_argument(
        "--execute",
        action="store_true",
        help="run the S10-H2-A executor; requires an explicit --device and temporary output",
    )
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    parser.add_argument(
        "--device",
        default=None,
        help="explicit CUDA device required by --execute; plan mode does not select a device",
    )
    args = parser.parse_args(argv)
    try:
        if args.execute and not args.device:
            raise ProtocolError("PAUSE", "--execute requires an explicit CUDA device")
        if args.execute and _is_canonical_result_path(args.output):
            raise ProtocolError(
                "PAUSE",
                "canonical H2 output is disabled during S10-H2-A; use a temporary test destination",
            )
        context = _validate_pre_execution(
            config_path=args.config,
            result_path=args.output if args.execute else Path("/dev/null/noncanonical-result"),
        )
        if args.execute:
            return _dispatch_execute(context=context, device=args.device, output=args.output)
        print(json.dumps(_plan(context), indent=2, sort_keys=True))
        return 0
    except ProtocolError as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
