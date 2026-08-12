from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts import validate_s09_protocol as validator
from scripts.validate_s09_protocol import (
    EXPECTED_MODE_IDS,
    ProtocolValidationError,
    quality_gate,
    validate_latency_records,
    validate_protocol_payload,
)

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _mock_initialized_any_precision_submodule(monkeypatch):
    monkeypatch.setattr(
        validator,
        "_gitlink_revision",
        lambda root, relative_path: validator.EXPECTED_ANY_PRECISION_REVISION,
    )
    monkeypatch.setattr(
        validator, "_git_revision", lambda path: validator.EXPECTED_ANY_PRECISION_REVISION
    )
    monkeypatch.setattr(validator, "_git_superproject_worktree", lambda path: str(ROOT.resolve()))


def _config_and_prompt():
    config = json.loads((ROOT / "configs/s09_baseline_eval.json").read_text())
    prompt_payload = json.loads((ROOT / "configs/s09_baseline_prompts.json").read_text())
    return config, prompt_payload


def test_protocol_has_exactly_five_unique_modes():
    config, prompt_payload = _config_and_prompt()
    result = validate_protocol_payload(
        config, ROOT, prompt_payload=prompt_payload, check_external=False
    )
    assert result["mode_count"] == 5
    assert tuple(mode["id"] for mode in config["modes"]) == EXPECTED_MODE_IDS
    assert len({mode["id"] for mode in config["modes"]}) == 5


def test_protocol_rejects_duplicate_or_ambiguous_modes():
    config, prompt_payload = _config_and_prompt()
    duplicate = copy.deepcopy(config)
    duplicate["modes"][4]["id"] = duplicate["modes"][3]["id"]
    with pytest.raises(ProtocolValidationError, match="mode IDs"):
        validate_protocol_payload(
            duplicate, ROOT, prompt_payload=prompt_payload, check_external=False
        )

    incomplete = copy.deepcopy(config)
    del incomplete["latency"]["repeats_per_fixed_latency_request"]
    with pytest.raises(ProtocolValidationError, match="latency repeats_per_fixed_latency_request"):
        validate_protocol_payload(
            incomplete, ROOT, prompt_payload=prompt_payload, check_external=False
        )


def test_protocol_rejects_any_frozen_mode_field_drift():
    for field, mode_index, value in (
        ("label", 0, "changed"),
        ("packed_artifact", 1, False),
        ("precision", 1, 9),
        ("router_checkpoint", 3, "other"),
        ("loader", 4, "async"),
    ):
        config, prompt_payload = _config_and_prompt()
        mode = config["modes"][mode_index]
        mode[field] = value
        with pytest.raises(ProtocolValidationError, match="mode .* fields"):
            validate_protocol_payload(
                config, ROOT, prompt_payload=prompt_payload, check_external=False
            )


def test_protocol_identity_agrees_with_recorded_manifests():
    config, prompt_payload = _config_and_prompt()
    validate_protocol_payload(config, ROOT, prompt_payload=prompt_payload, check_external=False)

    changed = copy.deepcopy(config)
    changed["identities"]["model"]["revision"] = "wrong-revision"
    with pytest.raises(ProtocolValidationError, match="model revision"):
        validate_protocol_payload(
            changed, ROOT, prompt_payload=prompt_payload, check_external=False
        )


def test_quality_gate_uses_frozen_ten_percent_margin():
    assert quality_gate(10.0, 10.0)
    assert quality_gate(11.0, 10.0)
    assert not quality_gate(11.0001, 10.0)


def test_latency_contract_requires_five_raw_finite_nonnegative_phase_records():
    records = [
        {"prefill_seconds": 1.0 + i, "decode_seconds": 0.5, "end_to_end_seconds": 1.5 + i}
        for i in range(5)
    ]
    assert validate_latency_records(records)
    assert not validate_latency_records(records[:-1])
    records[0]["decode_seconds"] = -1.0
    assert not validate_latency_records(records)
    records[0]["decode_seconds"] = float("nan")
    assert not validate_latency_records(records)


def test_protocol_rejects_incomplete_routed_prompt_recording():
    config, prompt_payload = _config_and_prompt()
    config["fixed_inputs"]["routed_recording"]["units_per_request"] = 71
    with pytest.raises(ProtocolValidationError, match="routed recording units_per_request"):
        validate_protocol_payload(config, ROOT, prompt_payload=prompt_payload, check_external=False)

    config, prompt_payload = _config_and_prompt()
    config["fixed_inputs"]["routed_recording"]["adaptivity_limitation"] = "adaptive"
    with pytest.raises(ProtocolValidationError, match="routed recording adaptivity_limitation"):
        validate_protocol_payload(config, ROOT, prompt_payload=prompt_payload, check_external=False)


def test_protocol_rejects_generation_or_seed_policy_drift():
    config, prompt_payload = _config_and_prompt()
    config["generation"]["input_source"] = "runtime_prompt"
    with pytest.raises(ProtocolValidationError, match="generation input_source"):
        validate_protocol_payload(config, ROOT, prompt_payload=prompt_payload, check_external=False)

    config, prompt_payload = _config_and_prompt()
    config["seeds"]["global_reproducibility_seed"] = 7
    with pytest.raises(ProtocolValidationError, match="seed policy global_reproducibility_seed"):
        validate_protocol_payload(config, ROOT, prompt_payload=prompt_payload, check_external=False)


def test_protocol_rejects_physical_transfer_contract_drift():
    config, prompt_payload = _config_and_prompt()
    config["transfer"]["mode"] = "nominal_bytes"
    with pytest.raises(ProtocolValidationError, match="transfer mode"):
        validate_protocol_payload(config, ROOT, prompt_payload=prompt_payload, check_external=False)

    for section, field in (
        ("memory", "cuda_boundaries"),
        ("memory", "on_demand_extra_records"),
        ("memory", "no_complete_packed_parent_on_gpu"),
        ("latency", "same_warmup_policy_for_comparable_modes"),
        ("latency", "cuda_boundaries"),
        ("latency", "cross_request_packed_planes"),
    ):
        config, prompt_payload = _config_and_prompt()
        value = config[section][field]
        config[section][field] = [] if isinstance(value, list) else not value
        with pytest.raises(
            ProtocolValidationError,
            match=(
                "(memory|latency|complete packed GPU copy).*(boundaries|records|prohibition|"
                "same_warmup|cross_request)"
            ),
        ):
            validate_protocol_payload(
                config, ROOT, prompt_payload=prompt_payload, check_external=False
            )

    config, prompt_payload = _config_and_prompt()
    config["transfer"]["expected_bytes_inputs"] = ["bit width"]
    with pytest.raises(ProtocolValidationError, match="expected-byte inputs"):
        validate_protocol_payload(config, ROOT, prompt_payload=prompt_payload, check_external=False)


def test_protocol_rejects_incomplete_fixed_gpu_identity_policy():
    config, prompt_payload = _config_and_prompt()
    config["hardware"]["record_versions"] = ["device_index", "python"]
    with pytest.raises(ProtocolValidationError, match="hardware version record"):
        validate_protocol_payload(config, ROOT, prompt_payload=prompt_payload, check_external=False)

    config, prompt_payload = _config_and_prompt()
    config["hardware"]["identical_rtx3090_substitution"] = "allowed"
    with pytest.raises(ProtocolValidationError, match="GPU comparability policy"):
        validate_protocol_payload(config, ROOT, prompt_payload=prompt_payload, check_external=False)


def test_protocol_rejects_ambiguous_dataset_and_loss_wording():
    config, prompt_payload = _config_and_prompt()
    config["perplexity"]["selection"] = "source order; random sampling disabled"
    with pytest.raises(ProtocolValidationError, match="dataset selection policy"):
        validate_protocol_payload(config, ROOT, prompt_payload=prompt_payload, check_external=False)

    config, prompt_payload = _config_and_prompt()
    config["perplexity"]["loss"] = "not token-weighted"
    with pytest.raises(ProtocolValidationError, match="token-weighted loss"):
        validate_protocol_payload(config, ROOT, prompt_payload=prompt_payload, check_external=False)


def test_protocol_rejects_stale_any_precision_manifest_or_checkout(monkeypatch):
    config, prompt_payload = _config_and_prompt()
    config["identities"]["any_precision"]["manifest_commit"] = "stale"
    with pytest.raises(ProtocolValidationError, match="protocol revision"):
        validate_protocol_payload(config, ROOT, prompt_payload=prompt_payload, check_external=False)

    config, prompt_payload = _config_and_prompt()
    monkeypatch.setattr(
        validator,
        "_gitlink_revision",
        lambda root, relative_path: "stale",
    )
    with pytest.raises(ProtocolValidationError, match="superproject gitlink revision"):
        validate_protocol_payload(config, ROOT, prompt_payload=prompt_payload, check_external=False)

    config, prompt_payload = _config_and_prompt()
    monkeypatch.setattr(validator, "_git_superproject_worktree", lambda path: None)
    with pytest.raises(ProtocolValidationError, match="superproject worktree"):
        validate_protocol_payload(config, ROOT, prompt_payload=prompt_payload, check_external=False)

    config, prompt_payload = _config_and_prompt()
    monkeypatch.setattr(
        validator,
        "_gitlink_revision",
        lambda root, relative_path: validator.EXPECTED_ANY_PRECISION_REVISION,
    )
    monkeypatch.setattr(validator, "_git_superproject_worktree", lambda path: str(ROOT.resolve()))
    monkeypatch.setattr(validator, "_git_revision", lambda path: "stale")
    with pytest.raises(ProtocolValidationError, match="checked-out revision"):
        validate_protocol_payload(config, ROOT, prompt_payload=prompt_payload, check_external=False)


def test_protocol_rejects_incomplete_post_result_policy_or_deferred_list():
    config, prompt_payload = _config_and_prompt()
    config["release_criteria"]["failure_outcomes"].pop("all_gates_pass")
    with pytest.raises(ProtocolValidationError, match="release failure outcomes"):
        validate_protocol_payload(config, ROOT, prompt_payload=prompt_payload, check_external=False)

    for section, mutation in (
        ("structural_reproducibility_failures", lambda value: value[:-1]),
        ("quality_gates", lambda value: {**value, "static_8_perplexity": {"operator": ">="}}),
        ("performance_validity", lambda value: value[:-1]),
        (
            "failure_outcomes",
            lambda value: {key: value[key] for key in value if key != "all_gates_pass"},
        ),
        ("post_result_protocol_change", lambda value: "silent"),
    ):
        config, prompt_payload = _config_and_prompt()
        config["release_criteria"][section] = mutation(config["release_criteria"][section])
        with pytest.raises(
            ProtocolValidationError,
            match="(release criteria|performance validity|failure outcomes|post-result policy)",
        ):
            validate_protocol_payload(
                config, ROOT, prompt_payload=prompt_payload, check_external=False
            )

    config, prompt_payload = _config_and_prompt()
    config["deferred_mechanisms"] = config["deferred_mechanisms"][:-1]
    with pytest.raises(ProtocolValidationError, match="complete deferred mechanisms"):
        validate_protocol_payload(config, ROOT, prompt_payload=prompt_payload, check_external=False)
