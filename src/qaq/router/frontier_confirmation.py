"""Execute the frozen S10-F three-seed frontier confirmation protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from qaq.router.baseline_training import (
    _device_example,
    _precompute_teacher_logits,
    _seed_everything,
    _select_examples,
)
from qaq.router.calibration import (
    CANDIDATE_BITS,
    MIN_FREE_GPU_BYTES,
    _evaluate_learned,
    _module_state_hash,
    _run_trial,
    _validate_model_snapshot,
    freeze_teacher_and_packed_student,
    install_memory_saving_packed_backward,
    router_only_state,
    router_state_hash,
    validate_canonical_initialization,
)
from qaq.router.calibration import (
    _load_config as _load_calibration_config,
)
from qaq.router.network import THREE_WAY_CANDIDATE_BITS
from qaq.router.soft_model import SoftRoutedQwen3ForCausalLM

CONFIG_PATH = ROOT / "configs/router_frontier_confirmation.json"
CALIBRATION_CONFIG_PATH = ROOT / "configs/router_cost_calibration.json"
CALIBRATION_RESULT_PATH = ROOT / "docs/results/s10d_lambda_calibration.json"
MANIFEST_PATH = ROOT / "docs/quantized_model_manifest.json"
RESULT_PATH = ROOT / "docs/results/s10f_frontier_confirmation_rerun.json"
HISTORICAL_RESULT_PATH = ROOT / "docs/results/s10f_frontier_confirmation.json"
HISTORICAL_RESULT_SHA256 = "d68f041e0a3dc32c465e8b8068ca3ab230d39253757e30f3019ca7e681b14233"
MODEL_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
ANY_PRECISION_REVISION = "a3257d02740cc5757c78673da534b0630ff3a4ea"
DATASET_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
EXPECTED_IMPLEMENTATION_BASE = "7fc136eabdba302e199354ae001cd1e1cd42199f"
HISTORICAL_PROTOCOL_BASE = "e718f27fe6b02082709d65665396640e251e602c"
LOCKED_CONFIG_SHA256 = "fe5ff8826f17605ca8b2dc7d83555e858d3d9f5fa67d14b49bb09b7cbf66a879"
LOCKED_CALIBRATION_CONFIG_SHA256 = (
    "22649ec4cdafa7a8ff669f72c159c7fbfbaa33ecea50888a953301a8225bb5c1"
)
LOCKED_CALIBRATION_RESULT_SHA256 = (
    "6ecdf9d8e0d2899fca4650fed083b17734ef1cc2f531fdc7c42e1faf3f72a865"
)
VALIDATION_IDS = ("validation-3", "validation-1000")
EXPECTED_SEEDS = (1729, 1730, 1731)
EXPECTED_LAMBDAS = (0.0, 0.03, 0.1)
EXPECTED_TRIAL_PAIRS = tuple(
    (seed, lambda_bit) for seed in EXPECTED_SEEDS for lambda_bit in EXPECTED_LAMBDAS
)
FORBIDDEN_RESULT_FIELDS = frozenset({"latency", "memory", "transfer", "throughput", "energy"})


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _load_frozen_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Load only byte-identical S10-E protocol bytes."""

    raw = path.read_bytes()
    if _sha256_bytes(raw) != LOCKED_CONFIG_SHA256:
        raise RuntimeError("REVISE: S10-E frozen config differs byte-for-byte")
    config = json.loads(raw)
    _validate_protocol(config)
    return config


def _validate_protocol(config: dict[str, Any]) -> None:
    if config.get("format") != "qaq-s10e-frontier-confirmation-v1":
        raise RuntimeError("REVISE: unexpected S10-E protocol format")
    if config.get("stage") != "S10-E":
        raise RuntimeError("REVISE: unexpected frozen protocol stage")
    protocol = config["protocol"]
    if tuple(protocol["candidate_bits"]) != CANDIDATE_BITS:
        raise RuntimeError("REVISE: candidate bit ordering drifted")
    if tuple(protocol["lambdas"]) != EXPECTED_LAMBDAS:
        raise RuntimeError("REVISE: S10-F lambda ordering drifted")
    if tuple(protocol["seeds"]) != EXPECTED_SEEDS or protocol["trial_count"] != 9:
        raise RuntimeError("REVISE: S10-F seed matrix drifted")
    if protocol["paired_control_lambda"] != 0.0 or protocol["confirmation_lambda"] != 0.03:
        raise RuntimeError("REVISE: S10-F paired-control values drifted")
    pairing = protocol["pairing"]
    if pairing != {
        "one_canonical_fresh_three_way_router_initialization_per_seed": True,
        "clone_canonical_initialization_identically_across_lambdas": True,
        "fresh_adamw_per_lambda": True,
        "same_lambda_order_per_seed": True,
        "warm_start_allowed": False,
        "historical_s07_two_way_checkpoint_loading_allowed": False,
        "historical_s07_two_way_checkpoint_sha256": (
            "08bf646f19759c0d7949e159bdbe4f96bbea737204b96f8760d205c8d6fd1949"
        ),
    }:
        raise RuntimeError("REVISE: S10-F pairing contract drifted")
    inherited = config["inherited_s10d_contract"]
    if inherited["dataset"]["revision"] != DATASET_REVISION:
        raise RuntimeError("REVISE: pinned dataset revision drifted")
    if inherited["dataset"]["tokenizer_revision"] != MODEL_REVISION:
        raise RuntimeError("REVISE: pinned tokenizer revision drifted")
    if inherited["training"] != {
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
    }:
        raise RuntimeError("REVISE: inherited training settings drifted")
    if inherited["objective"] != {
        "candidate_bits": [4, 6, 8],
        "normalized_bit_cost_formula": "c(bit) = (bit - 4) / (8 - 4)",
        "normalized_bit_costs": [0.0, 0.5, 1.0],
        "cost_order": [4, 6, 8],
        "loss_formula": "L_total = L_KD + lambda_bit * L_bit",
        "kd_loss": "unchanged completion-only T^2 masked KL teacher-student distillation",
        "cost_reduction": "unweighted arithmetic mean across all 36 attention and 36 FFN decisions",
        "hardware_cost_claim": False,
    }:
        raise RuntimeError("REVISE: inherited objective settings drifted")
    frozen = inherited["frozen_components"]
    if (
        frozen["any_precision_revision"] != ANY_PRECISION_REVISION
        or frozen["model_revision"] != MODEL_REVISION
        or frozen["historical_s07_checkpoint_used"] is not False
    ):
        raise RuntimeError("REVISE: frozen model/backend identities drifted")
    router_contract = config["router_contract"]
    if router_contract != {
        "router_count": 72,
        "router_parameter_count": 23630040,
        "candidate_bits": [4, 6, 8],
        "soft_routing": "resident three-way packed mixture in explicit [p4,p6,p8] order",
        "hard_routing": "deterministic argmax in explicit [p4,p6,p8] order",
        "request_owned_routing_state": True,
        "completion_only_kd_objective_unchanged": True,
    }:
        raise RuntimeError("REVISE: router contract drifted")
    measurements = config["future_measurements"]
    required = {
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
    }
    if set(measurements["per_trial_required_fields"]) != required:
        raise RuntimeError("REVISE: required per-trial schema drifted")
    if measurements["route_map_contract"] != {
        "validation_ids_in_order": list(VALIDATION_IDS),
        "units_per_map": 72,
        "unit_order": (
            "layer-major: layer 0 attention, layer 0 ffn, then layer 1 attention, "
            "layer 1 ffn, through layer 35"
        ),
        "allowed_bits": [4, 6, 8],
    }:
        raise RuntimeError("REVISE: route-map contract drifted")
    if measurements["forbidden_measurements"] != [
        "latency",
        "memory",
        "transfer",
        "throughput",
        "energy",
    ]:
        raise RuntimeError("REVISE: forbidden-measurement contract drifted")


def _validate_starting_base() -> str:
    head = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    ancestor = subprocess.run(
        [
            "git",
            "-C",
            str(ROOT),
            "merge-base",
            "--is-ancestor",
            EXPECTED_IMPLEMENTATION_BASE,
            head,
        ],
        check=False,
    )
    if ancestor.returncode != 0:
        raise RuntimeError(
            "PAUSE: merged S10-E implementation base is unavailable; "
            f"expected ancestor {EXPECTED_IMPLEMENTATION_BASE}, got {head}"
        )
    return head


def _validate_calibration_evidence() -> tuple[dict[str, Any], dict[str, Any]]:
    if _sha256_bytes(CALIBRATION_CONFIG_PATH.read_bytes()) != LOCKED_CALIBRATION_CONFIG_SHA256:
        raise RuntimeError("PAUSE: canonical S10-D config identity is unavailable")
    if _sha256_bytes(CALIBRATION_RESULT_PATH.read_bytes()) != LOCKED_CALIBRATION_RESULT_SHA256:
        raise RuntimeError("PAUSE: canonical S10-D result identity is unavailable")
    config = _load_calibration_config(CALIBRATION_CONFIG_PATH)
    result = json.loads(CALIBRATION_RESULT_PATH.read_text())
    if result.get("format") != "qaq-s10d-lambda-calibration-v1" or result.get("stage") != "S10-D":
        raise RuntimeError("PAUSE: canonical S10-D result schema is unavailable")
    if result["grid"]["completed"] != [0.0, 0.003, 0.01, 0.03, 0.1]:
        raise RuntimeError("PAUSE: canonical S10-D grid evidence is incomplete")
    if result["extensions"]["performed"] != []:
        raise RuntimeError("PAUSE: canonical S10-D adaptive evidence is not frozen")
    if not all(
        result["audits"][key]
        for key in (
            "all_initial_hashes_match",
            "finite_measurements",
            "fresh_adamw_per_trial",
            "packed_student_base_unchanged",
            "router_only_optimizer",
            "teacher_frozen_after",
            "teacher_unchanged",
        )
    ):
        raise RuntimeError("PAUSE: canonical S10-D audits are incomplete")
    return config, result


def _execution_config(
    protocol: dict[str, Any], calibration_config: dict[str, Any]
) -> dict[str, Any]:
    """Adapt locked nested S10-E fields to the existing S10-D execution seam."""

    inherited = protocol["inherited_s10d_contract"]
    dataset = dict(inherited["dataset"])
    training = dict(inherited["training"])
    model = {
        "router_count": protocol["router_contract"]["router_count"],
        "router_parameter_count": protocol["router_contract"]["router_parameter_count"],
    }
    return {
        "dataset": dataset,
        "training": training,
        "model": model,
        "evaluation": {
            # S10-E does not restate this diagnostic base; S10-D's locked value
            # is reused and is recorded as an implementation choice.
            "entropy_log_base": calibration_config["evaluation"]["entropy_log_base"],
        },
        "adaptive_extensions": {
            "low_lambda": {
                "trigger_collapse_fraction": calibration_config["adaptive_extensions"][
                    "low_lambda"
                ]["trigger_collapse_fraction"]
            }
        },
    }


def _environment(device: str, *, free_gpu_check_passed: bool) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(torch.device(device))
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "device": device,
        "gpu_name": properties.name,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "free_gpu_preflight_check_passed": free_gpu_check_passed,
        "preflight_command": "source ~/.venv/bin/activate && which python && python --version && nvidia-smi",
    }


def _check_runtime(device: str) -> dict[str, Any]:
    if not os.environ.get("VIRTUAL_ENV", "").startswith(str(Path.home() / ".venv")):
        raise SystemExit("PAUSE: ~/.venv is not active")
    if not torch.cuda.is_available():
        raise SystemExit("PAUSE: CUDA is unavailable; CPU fallback is forbidden")
    torch.cuda.set_device(torch.device(device))
    free_bytes, _ = torch.cuda.mem_get_info(torch.device(device))
    if free_bytes < MIN_FREE_GPU_BYTES:
        raise SystemExit(f"PAUSE: explicit GPU {device} does not have enough free capacity")
    return _environment(device, free_gpu_check_passed=True)


def _resolve_artifact(manifest: dict[str, Any]) -> Path:
    logical_path = ROOT / manifest["artifact"]["local_path"]
    return Path(os.environ.get("QAQ_S03_ARTIFACT", str(logical_path))).expanduser()


def _configure_any_precision_root() -> str:
    logical_path = ROOT / "third_party" / "any-precision-llm"
    source_path = Path(os.environ.get("QAQ_ANY_PRECISION_ROOT", str(logical_path))).expanduser()
    if not source_path.is_dir():
        raise SystemExit(f"PAUSE: pinned Any-Precision source is unavailable: {source_path}")
    revision = subprocess.run(
        ["git", "-C", str(source_path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(source_path), "status", "--porcelain=v1"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != ANY_PRECISION_REVISION or status:
        raise SystemExit("PAUSE: pinned Any-Precision source is unavailable or dirty")
    # The disposable worktree may have an uninitialized gitlink.  Keep the
    # production seams unchanged and point their read-only source root at the
    # explicit pinned checkout for this execution only.
    import qaq.loading.loader as loader_module
    import qaq.model.static as static_module
    import qaq.quantization.backend as backend_module

    static_module.ANY_PRECISION_ROOT = source_path
    loader_module.ANY_PRECISION_ROOT = source_path
    backend_module.ANY_PRECISION_ROOT = source_path
    return (
        "QAQ_ANY_PRECISION_ROOT override" if source_path != logical_path else "repository submodule"
    )


def _identity_for_artifact(manifest: dict[str, Any], artifact: Path) -> dict[str, Any]:
    from qaq.model.static import file_sha256, source_commit

    artifact_hash = file_sha256(artifact / "pytorch_model.bin")
    expected = manifest["artifact"]["checkpoint_hashes"]["pytorch_model.bin"]
    if artifact_hash != expected:
        raise RuntimeError("PAUSE: identity-matched packed artifact hash is unavailable")
    if source_commit() != ANY_PRECISION_REVISION:
        raise RuntimeError("PAUSE: pinned Any-Precision source is unavailable or dirty")
    return {
        "model_repository": manifest["source_model"]["repository"],
        "model_revision": MODEL_REVISION,
        "tokenizer_revision": MODEL_REVISION,
        "any_precision_revision": ANY_PRECISION_REVISION,
        "any_precision_source": "pinned read-only checkout",
        "packed_artifact": manifest["artifact"]["local_path"],
        "packed_artifact_source": (
            "QAQ_S03_ARTIFACT read-only override"
            if artifact.resolve() != (ROOT / manifest["artifact"]["local_path"]).resolve()
            else "repository logical artifact path"
        ),
        "packed_artifact_pytorch_model_sha256": artifact_hash,
        "historical_s07_checkpoint_used": False,
        "historical_s07_checkpoint_sha256": (
            "08bf646f19759c0d7949e159bdbe4f96bbea737204b96f8760d205c8d6fd1949"
        ),
        "artifact_manifest_hash": _sha256_bytes(MANIFEST_PATH.read_bytes()),
    }


def _validate_route_map(route_map: Any) -> list[int]:
    if not isinstance(route_map, list) or len(route_map) != 72:
        raise RuntimeError("REVISE: validation route map must contain exactly 72 units")
    if any(bit not in CANDIDATE_BITS for bit in route_map):
        raise RuntimeError("REVISE: validation route map contains an unconfigured bit")
    return [int(bit) for bit in route_map]


def _validate_trial_matrix(trials: Iterable[dict[str, Any]]) -> None:
    pairs = tuple((int(trial["seed"]), float(trial["lambda_bit"])) for trial in trials)
    if pairs != EXPECTED_TRIAL_PAIRS:
        raise RuntimeError(
            "REVISE: S10-F trials must be the nine ordered seed/lambda pairs "
            f"{EXPECTED_TRIAL_PAIRS!r}"
        )


def _reproducibility_audit(first: dict[str, Any], repeat: dict[str, Any]) -> dict[str, Any]:
    """One same-state hard-validation repeat, following the S07 determinism precedent."""

    first_maps = first["per_validation_route_maps"]
    repeat_maps = repeat["per_validation_route_maps"]
    maps_identical = first_maps == repeat_maps
    metric_names = (
        "validation_kd_loss",
        "mean_hard_selected_bit_width",
        "hard_fraction_4",
        "hard_fraction_6",
        "hard_fraction_8",
    )
    metrics_identical = all(first[name] == repeat[name] for name in metric_names)
    finite_both = bool(first["finite_logits"] and repeat["finite_logits"])
    return {
        "method": "one immediate hard validation repeat at the unchanged trained router state",
        "repeat_count": 1,
        "route_maps_identical": maps_identical,
        "hard_metrics_identical": metrics_identical,
        "finite_outputs_both_passed": finite_both,
        "passed": bool(maps_identical and metrics_identical and finite_both),
    }


def _collapse_audit(label: str) -> dict[str, Any]:
    invalid = label in {"COLLAPSED_TO_4", "COLLAPSED_TO_6", "COLLAPSED_TO_8"}
    return {
        "classification": label,
        "invalid_or_degenerate": invalid,
        "passed": not invalid,
    }


def _router_only_optimizer_audit(optimizer_audit: dict[str, Any]) -> bool:
    """Interpret the serialized or in-memory router-prefix audit strictly."""

    prefixes = optimizer_audit.get("included_name_prefixes")
    return isinstance(prefixes, (list, tuple)) and tuple(prefixes) == ("routers.",)


def _fresh_adamw_audit(raw: dict[str, Any]) -> bool:
    """Accept only the explicit pre-first-step fresh-state observation."""

    return raw.get("optimizer_state_was_fresh") is True


def _trial_record(
    seed: int,
    lambda_bit: float,
    raw: dict[str, Any],
    soft: dict[str, Any],
    hard: dict[str, Any],
    hard_repeat: dict[str, Any],
    teacher_hash_before: str,
    teacher: torch.nn.Module,
) -> dict[str, Any]:
    history = raw["history"]
    gradient = raw["gradient_diagnostic"]
    finite_loss = (
        all(
            item["finite_kd_loss"]
            and item["finite_bit_cost"]
            and item["finite_weighted_cost"]
            and item["finite_total_loss"]
            for item in history
        )
        and len(history) == 4
    )
    finite_gradient = bool(
        gradient["finite"]
        and math.isfinite(gradient["kd_gradient_norm"])
        and math.isfinite(gradient["bit_gradient_norm"])
        and all(math.isfinite(item["router_gradient_norm"]) for item in history)
    )
    teacher_hash_after = _module_state_hash(teacher)
    teacher_frozen = all(not parameter.requires_grad for parameter in teacher.parameters())
    teacher_grads_absent = all(parameter.grad is None for parameter in teacher.parameters())
    optimizer = raw["optimizer"]
    router_only = _router_only_optimizer_audit(optimizer)
    fresh_adamw = _fresh_adamw_audit(raw) and router_only
    route_maps = hard["per_validation_route_maps"]
    route_3 = _validate_route_map(route_maps[VALIDATION_IDS[0]])
    route_1000 = _validate_route_map(route_maps[VALIDATION_IDS[1]])
    reproducibility = _reproducibility_audit(hard, hard_repeat)
    collapse = _collapse_audit(hard["collapse_classification"])
    return {
        "seed": seed,
        "lambda_bit": lambda_bit,
        "initial_router_state_sha256": raw["initial_router_state_sha256"],
        "final_router_state_sha256": raw["final_router_state_sha256"],
        "initial_kd_gradient_norm": gradient["kd_gradient_norm"],
        "initial_bit_cost_gradient_norm": gradient["bit_gradient_norm"],
        "lambda_weighted_gradient_ratio": gradient["lambda_weighted_ratio"],
        "finite_loss_audit": finite_loss,
        "finite_gradient_audit": finite_gradient,
        "teacher_frozen_audit": bool(
            teacher_frozen and teacher_grads_absent and teacher_hash_before == teacher_hash_after
        ),
        "packed_student_base_unchanged_audit": raw["frozen_packed_student_unchanged"],
        "router_only_optimizer_audit": router_only,
        "fresh_adamw_audit": fresh_adamw,
        "soft_validation_kd": soft["validation_kd_loss"],
        "soft_validation_mean_expected_bit_width": soft["mean_expected_bit_width"],
        "soft_validation_mean_p4": soft["mean_p4"],
        "soft_validation_mean_p6": soft["mean_p6"],
        "soft_validation_mean_p8": soft["mean_p8"],
        "soft_validation_mean_entropy": soft["mean_entropy"],
        "hard_validation_kd": hard["validation_kd_loss"],
        "hard_validation_mean_selected_bit_width": hard["mean_hard_selected_bit_width"],
        "hard_validation_fraction_4": hard["hard_fraction_4"],
        "hard_validation_fraction_6": hard["hard_fraction_6"],
        "hard_validation_fraction_8": hard["hard_fraction_8"],
        "hard_validation_route_map_validation-3": route_3,
        "hard_validation_route_map_validation-1000": route_1000,
        "route_variation": hard["route_variation_across_prompts"],
        "distinct_hard_route_map_count": hard["unique_hard_route_map_count"],
        "reproducibility_audit": reproducibility,
        "collapse_audit": collapse,
        "optimizer_audit": raw["runtime_optimizer_evidence"],
        "training_history": history,
    }


def _is_frontier_point(points: Iterable[dict[str, float]], candidate_lambda: float) -> bool:
    values = tuple(points)
    candidate = next(point for point in values if point["lambda"] == candidate_lambda)
    return not any(
        other["lambda"] != candidate_lambda
        and other["kd"] <= candidate["kd"]
        and other["width"] <= candidate["width"]
        and (other["kd"] < candidate["kd"] or other["width"] < candidate["width"])
        for other in values
    )


def _aggregate_trials(
    trials: list[dict[str, Any]],
    *,
    inherited_regressions_status: str,
    invalidated_trial_count: int = 0,
    prohibited_work_occurred: bool = False,
) -> dict[str, Any]:
    pairs = tuple((int(trial["seed"]), float(trial["lambda_bit"])) for trial in trials)
    complete_matrix = pairs == EXPECTED_TRIAL_PAIRS
    by_seed: dict[int, dict[float, dict[str, Any]]] = {}
    for trial in trials:
        by_seed.setdefault(int(trial["seed"]), {})[float(trial["lambda_bit"])] = trial
    if not complete_matrix:
        aggregate = {
            "per_lambda_median_hard_validation_kd": {},
            "per_lambda_median_hard_mean_selected_bit_width": {},
            "per_seed_hard_frontier_membership_for_lambda_0.03": {},
            "lambda_0.03_frontier_seed_count": 0,
            "paired_control_hard_kd_delta_median_lambda_0.03_minus_lambda_0.0": None,
            "paired_control_hard_width_delta_median_lambda_0.03_minus_lambda_0.0": None,
            "reproducibility_failure_count": sum(
                not trial["reproducibility_audit"]["passed"] for trial in trials
            ),
            "paired_control_hard_kd_deltas_by_seed": {},
            "paired_control_hard_width_deltas_by_seed": {},
        }
        gate_checks = {
            "all_nine_trials_complete": False,
            "all_required_audits_pass": False,
            "no_invalid_or_degenerate_collapse": False,
            "lambda_0.03_frontier_seed_count_at_least_2": False,
            "paired_hard_kd_delta_non_positive": False,
            "paired_hard_width_delta_strictly_negative": False,
            "reproducibility_failures_zero": aggregate["reproducibility_failure_count"] == 0,
            "inherited_regressions_pass": inherited_regressions_status == "passed",
            "no_prohibited_work": not prohibited_work_occurred,
        }
        aggregate["gate_checks"] = gate_checks
        aggregate["classification"] = "REVISE" if invalidated_trial_count else "PAUSE"
        return aggregate
    hard_kd_medians = {
        str(lambda_bit): statistics.median(
            by_seed[seed][lambda_bit]["hard_validation_kd"] for seed in sorted(by_seed)
        )
        for lambda_bit in EXPECTED_LAMBDAS
    }
    hard_width_medians = {
        str(lambda_bit): statistics.median(
            by_seed[seed][lambda_bit]["hard_validation_mean_selected_bit_width"]
            for seed in sorted(by_seed)
        )
        for lambda_bit in EXPECTED_LAMBDAS
    }
    frontier_membership: dict[str, bool] = {}
    for seed in sorted(by_seed):
        points = [
            {
                "lambda": lambda_bit,
                "kd": by_seed[seed][lambda_bit]["hard_validation_kd"],
                "width": by_seed[seed][lambda_bit]["hard_validation_mean_selected_bit_width"],
            }
            for lambda_bit in EXPECTED_LAMBDAS
        ]
        frontier_membership[str(seed)] = _is_frontier_point(points, 0.03)
    paired_kd = [
        by_seed[seed][0.03]["hard_validation_kd"] - by_seed[seed][0.0]["hard_validation_kd"]
        for seed in sorted(by_seed)
    ]
    paired_width = [
        by_seed[seed][0.03]["hard_validation_mean_selected_bit_width"]
        - by_seed[seed][0.0]["hard_validation_mean_selected_bit_width"]
        for seed in sorted(by_seed)
    ]
    aggregate = {
        "per_lambda_median_hard_validation_kd": hard_kd_medians,
        "per_lambda_median_hard_mean_selected_bit_width": hard_width_medians,
        "per_seed_hard_frontier_membership_for_lambda_0.03": frontier_membership,
        "lambda_0.03_frontier_seed_count": sum(frontier_membership.values()),
        "paired_control_hard_kd_delta_median_lambda_0.03_minus_lambda_0.0": statistics.median(
            paired_kd
        ),
        "paired_control_hard_width_delta_median_lambda_0.03_minus_lambda_0.0": statistics.median(
            paired_width
        ),
        "reproducibility_failure_count": sum(
            not trial["reproducibility_audit"]["passed"] for trial in trials
        ),
        "paired_control_hard_kd_deltas_by_seed": {
            str(seed): value for seed, value in zip(sorted(by_seed), paired_kd, strict=True)
        },
        "paired_control_hard_width_deltas_by_seed": {
            str(seed): value for seed, value in zip(sorted(by_seed), paired_width, strict=True)
        },
    }
    complete = (
        len(trials) == 9
        and set(by_seed) == set(EXPECTED_SEEDS)
        and all(set(by_seed[seed]) == set(EXPECTED_LAMBDAS) for seed in by_seed)
    )
    all_audits = complete and all(
        trial[key]
        for trial in trials
        for key in (
            "finite_loss_audit",
            "finite_gradient_audit",
            "teacher_frozen_audit",
            "packed_student_base_unchanged_audit",
            "router_only_optimizer_audit",
            "fresh_adamw_audit",
        )
    )
    no_collapse = complete and all(trial["collapse_audit"]["passed"] for trial in trials)
    gate_checks = {
        "all_nine_trials_complete": complete,
        "all_required_audits_pass": all_audits,
        "no_invalid_or_degenerate_collapse": no_collapse,
        "lambda_0.03_frontier_seed_count_at_least_2": aggregate["lambda_0.03_frontier_seed_count"]
        >= 2,
        "paired_hard_kd_delta_non_positive": aggregate[
            "paired_control_hard_kd_delta_median_lambda_0.03_minus_lambda_0.0"
        ]
        <= 0.0,
        "paired_hard_width_delta_strictly_negative": aggregate[
            "paired_control_hard_width_delta_median_lambda_0.03_minus_lambda_0.0"
        ]
        < 0.0,
        "reproducibility_failures_zero": aggregate["reproducibility_failure_count"] == 0,
        "inherited_regressions_pass": inherited_regressions_status == "passed",
        "no_prohibited_work": not prohibited_work_occurred,
    }
    if (
        invalidated_trial_count
        or inherited_regressions_status == "failed"
        or prohibited_work_occurred
    ):
        classification = "REVISE"
    elif not complete or inherited_regressions_status == "missing":
        classification = "PAUSE"
    elif all(gate_checks.values()):
        classification = "CONTINUE"
    else:
        classification = "REFINE"
    aggregate["gate_checks"] = gate_checks
    aggregate["classification"] = classification
    return aggregate


def _reject_forbidden_fields(value: Any, path: str = "result") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in FORBIDDEN_RESULT_FIELDS:
                raise RuntimeError(f"REVISE: forbidden measurement field {path}.{key}")
            _reject_forbidden_fields(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_forbidden_fields(child, f"{path}[{index}]")


def _make_result(
    protocol: dict[str, Any],
    calibration_config: dict[str, Any],
    calibration_result: dict[str, Any],
    *,
    base_head: str,
    environment: dict[str, Any],
    identities: dict[str, Any],
    dataset: dict[str, Any],
    train_manifest: list[dict[str, Any]],
    validation_manifest: list[dict[str, Any]],
    trials: list[dict[str, Any]],
    inherited_regressions_status: str,
) -> dict[str, Any]:
    aggregates = _aggregate_trials(
        trials, inherited_regressions_status=inherited_regressions_status
    )
    result = {
        "format": "qaq-s10f-frontier-confirmation-v1",
        "stage": "S10-F",
        "attempt": 2,
        "previous_attempt_status": "REVISE",
        "previous_attempt_result_sha256": HISTORICAL_RESULT_SHA256,
        "previous_attempt_result_path": str(HISTORICAL_RESULT_PATH.relative_to(ROOT)),
        "commit": base_head,
        "project_established_facts": {
            "merged_s10e_implementation_base": EXPECTED_IMPLEMENTATION_BASE,
            "historical_protocol_required_starting_commit": HISTORICAL_PROTOCOL_BASE,
            "s10a_through_s10e_complete": True,
            # These serialized keys are part of the frozen historical result schema.
            "canonical_s10d_config_sha256": LOCKED_CALIBRATION_CONFIG_SHA256,
            "canonical_s10d_result_sha256": LOCKED_CALIBRATION_RESULT_SHA256,
            "canonical_s10d_result_commit": calibration_result["commit"],
            "inherited_s10d_config_format": calibration_config["format"],
            "historical_s07_two_way_checkpoint_loaded": False,
            "s08_on_demand_path_invoked": False,
        },
        "implementation_choices": {
            "paired_initialization": "one canonical initialization per seed, cloned by exact router state before each lambda",
            "lambda_order_per_seed": list(EXPECTED_LAMBDAS),
            "independent_seed_order": list(EXPECTED_SEEDS),
            "entropy_log_base": calibration_config["evaluation"]["entropy_log_base"],
            "reproducibility_audit": "one immediate same-state hard validation repeat; exact route-map and hard-metric equality",
            "collapse_audit": "S10-D collapse labels COLLAPSED_TO_4/6/8 are invalid; prompt-invariant/adaptive/other are observational and valid",
            "frontier_membership": "lower-is-better KD and selected width; a point is on the frontier when no other lambda is no-worse on both with one strict improvement",
            "inherited_regressions_status": inherited_regressions_status,
        },
        "environment": environment,
        "identities": identities,
        "frozen_protocol": {
            "config_sha256": LOCKED_CONFIG_SHA256,
            "config_byte_exact": True,
            "seeds": list(EXPECTED_SEEDS),
            "lambdas": list(EXPECTED_LAMBDAS),
            "trial_count": 9,
            "candidate_bits": list(CANDIDATE_BITS),
            "validation_ids_in_order": list(VALIDATION_IDS),
            "route_map_units": 72,
            "optimizer_steps_per_trial": 4,
        },
        "dataset": {
            "configuration": dataset,
            "train_manifest": train_manifest,
            "validation_manifest": validation_manifest,
        },
        "trials": trials,
        "aggregates": aggregates,
        "gate": {
            "classification": aggregates["classification"],
            "checks": aggregates["gate_checks"],
            "production_lambda_selected": False,
            "next_action": (
                "CONTINUE authorizes only later broader validation; no production lambda is selected."
                if aggregates["classification"] == "CONTINUE"
                else f"Stop at S10-F with classification {aggregates['classification']}."
            ),
        },
        "limitations": [
            "The confirmation uses the frozen four-example, four-step training budget.",
            "This evidence confirms only the frozen frontier gate and does not select a production lambda or checkpoint.",
            "Serving/resource benchmarks and S08 on-demand loading remain outside this stage.",
        ],
    }
    _reject_forbidden_fields(result)
    return result


def _load_data_and_teacher(
    protocol: dict[str, Any], execution_config: dict[str, Any], device: str
) -> tuple[
    Any,
    list[Any],
    list[Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, torch.Tensor],
    str,
    bool,
]:
    import datasets
    from transformers import AutoTokenizer

    from qaq.evaluation.quality import load_full_precision_model

    snapshot = Path(
        os.environ.get(
            "QAQ_MODEL_SNAPSHOT",
            "~/.cache/huggingface/hub/models--Qwen--Qwen3-4B/"
            "snapshots/1cfa9a7208912126459214e8b04321603b3df60c",
        )
    ).expanduser()
    _validate_model_snapshot(snapshot)
    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot), revision=MODEL_REVISION, local_files_only=True
    )
    dataset_config = protocol["inherited_s10d_contract"]["dataset"]
    train_dataset = datasets.load_dataset(
        dataset_config["repository"],
        dataset_config["config"],
        split=dataset_config["train_split"],
        revision=DATASET_REVISION,
        trust_remote_code=False,
    )
    validation_dataset = datasets.load_dataset(
        dataset_config["repository"],
        dataset_config["config"],
        split=dataset_config["validation_split"],
        revision=DATASET_REVISION,
        trust_remote_code=False,
    )
    train_cpu, train_manifest = _select_examples(
        train_dataset,
        tokenizer,
        dataset_config["train_offsets"],
        split="train",
        config=execution_config,
        torch=torch,
    )
    validation_cpu, validation_manifest = _select_examples(
        validation_dataset,
        tokenizer,
        dataset_config["validation_offsets"],
        split="validation",
        config=execution_config,
        torch=torch,
    )
    if [item["example_id"] for item in train_manifest] != dataset_config["train_example_ids"]:
        raise RuntimeError("REVISE: train example order differs from frozen S10-F protocol")
    if [item["example_id"] for item in validation_manifest] != dataset_config[
        "validation_example_ids"
    ]:
        raise RuntimeError("REVISE: validation example order differs from frozen S10-F protocol")
    train_examples = [_device_example(item, device, torch) for item in train_cpu]
    validation_examples = [_device_example(item, device, torch) for item in validation_cpu]
    teacher = load_full_precision_model(snapshot, device)
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    teacher_hash = _module_state_hash(teacher)
    teacher_targets = _precompute_teacher_logits(
        teacher, train_examples + validation_examples, torch
    )
    teacher.cpu()
    torch.cuda.empty_cache()
    teacher_frozen = all(not parameter.requires_grad for parameter in teacher.parameters())
    return (
        teacher,
        train_examples,
        validation_examples,
        train_manifest,
        validation_manifest,
        teacher_targets,
        teacher_hash,
        teacher_frozen,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    parser.add_argument(
        "--inherited-regressions-passed",
        action="store_true",
        help="assert that the required pre-run inherited regression command passed",
    )
    args = parser.parse_args()
    protocol = _load_frozen_config(args.config)
    base_head = _validate_starting_base()
    calibration_config, calibration_result = _validate_calibration_evidence()
    environment = _check_runtime(args.device)
    any_precision_source = _configure_any_precision_root()
    manifest = json.loads(MANIFEST_PATH.read_text())
    artifact = _resolve_artifact(manifest)
    if not artifact.is_dir():
        raise SystemExit(f"PAUSE: identity-matched packed artifact is unavailable: {artifact}")
    identities = _identity_for_artifact(manifest, artifact)
    identities["any_precision_source"] = any_precision_source
    execution_config = _execution_config(protocol, calibration_config)
    (
        teacher,
        train_examples,
        validation_examples,
        train_manifest,
        validation_manifest,
        teacher_targets,
        teacher_hash,
        teacher_frozen,
    ) = _load_data_and_teacher(protocol, execution_config, args.device)
    if not teacher_frozen:
        raise RuntimeError("REVISE: teacher did not remain frozen before trials")
    install_memory_saving_packed_backward()
    trials: list[dict[str, Any]] = []
    canonical_hashes: dict[int, str] = {}
    for seed in EXPECTED_SEEDS:
        print(f"S10-F: constructing seed={seed} canonical three-way router", flush=True)
        _seed_everything(seed, torch)
        from qaq.model.manual import load_manual_model

        manual_model = load_manual_model(artifact, args.device)
        student = SoftRoutedQwen3ForCausalLM(
            manual_model,
            temperature=float(execution_config["training"]["routing_temperature"]),
            candidate_bits=THREE_WAY_CANDIDATE_BITS,
        )
        student.to(args.device)
        freeze_teacher_and_packed_student(teacher, student)
        validate_canonical_initialization(student, execution_config)
        canonical_state = router_only_state(student.routers)
        canonical_hash = router_state_hash(canonical_state)
        canonical_hashes[seed] = canonical_hash
        for lambda_bit in EXPECTED_LAMBDAS:
            print(f"S10-F: running seed={seed} lambda={lambda_bit}", flush=True)
            raw = _run_trial(
                student,
                canonical_state,
                lambda_bit,
                "confirmation",
                train_examples,
                teacher_targets,
                execution_config,
                args.device,
                list(EXPECTED_LAMBDAS),
                torch,
            )
            if raw["initial_router_state_sha256"] != canonical_hash:
                raise RuntimeError(
                    "REVISE: paired trial did not start from its seed canonical state"
                )
            soft = _evaluate_learned(
                student,
                validation_examples,
                teacher_targets,
                args.device,
                "soft",
                execution_config,
                torch,
            )
            hard = _evaluate_learned(
                student,
                validation_examples,
                teacher_targets,
                args.device,
                "hard",
                execution_config,
                torch,
            )
            hard_repeat = _evaluate_learned(
                student,
                validation_examples,
                teacher_targets,
                args.device,
                "hard",
                execution_config,
                torch,
            )
            trials.append(
                _trial_record(
                    seed,
                    lambda_bit,
                    raw,
                    soft,
                    hard,
                    hard_repeat,
                    teacher_hash,
                    teacher,
                )
            )
            torch.cuda.empty_cache()
        del student, manual_model
        torch.cuda.empty_cache()
    if len(canonical_hashes) != 3 or len(set(canonical_hashes.values())) != 3:
        raise RuntimeError(
            "REVISE: independent seeds did not produce distinct canonical router states"
        )
    result = _make_result(
        protocol,
        calibration_config,
        calibration_result,
        base_head=base_head,
        environment=environment,
        identities=identities,
        dataset=protocol["inherited_s10d_contract"]["dataset"],
        train_manifest=train_manifest,
        validation_manifest=validation_manifest,
        trials=trials,
        inherited_regressions_status=("passed" if args.inherited_regressions_passed else "missing"),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["gate"]["classification"] == "CONTINUE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
