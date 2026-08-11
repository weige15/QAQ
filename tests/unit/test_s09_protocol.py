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
        validate_protocol_payload(
            config, ROOT, prompt_payload=prompt_payload, check_external=False
        )

    config, prompt_payload = _config_and_prompt()
    config["fixed_inputs"]["routed_recording"]["adaptivity_limitation"] = "adaptive"
    with pytest.raises(ProtocolValidationError, match="routed recording adaptivity_limitation"):
        validate_protocol_payload(
            config, ROOT, prompt_payload=prompt_payload, check_external=False
        )


def test_protocol_rejects_generation_or_seed_policy_drift():
    config, prompt_payload = _config_and_prompt()
    config["generation"]["input_source"] = "runtime_prompt"
    with pytest.raises(ProtocolValidationError, match="generation input_source"):
        validate_protocol_payload(
            config, ROOT, prompt_payload=prompt_payload, check_external=False
        )

    config, prompt_payload = _config_and_prompt()
    config["seeds"]["global_reproducibility_seed"] = 7
    with pytest.raises(ProtocolValidationError, match="seed policy global_reproducibility_seed"):
        validate_protocol_payload(
            config, ROOT, prompt_payload=prompt_payload, check_external=False
        )


def test_protocol_rejects_physical_transfer_contract_drift():
    config, prompt_payload = _config_and_prompt()
    config["transfer"]["mode"] = "nominal_bytes"
    with pytest.raises(ProtocolValidationError, match="transfer mode"):
        validate_protocol_payload(
            config, ROOT, prompt_payload=prompt_payload, check_external=False
        )

    config, prompt_payload = _config_and_prompt()
    config["transfer"]["expected_bytes_inputs"] = ["bit width"]
    with pytest.raises(ProtocolValidationError, match="expected-byte inputs"):
        validate_protocol_payload(
            config, ROOT, prompt_payload=prompt_payload, check_external=False
        )


def test_protocol_rejects_incomplete_fixed_gpu_identity_policy():
    config, prompt_payload = _config_and_prompt()
    config["hardware"]["record_versions"] = ["device_index", "python"]
    with pytest.raises(ProtocolValidationError, match="hardware version record"):
        validate_protocol_payload(
            config, ROOT, prompt_payload=prompt_payload, check_external=False
        )

    config, prompt_payload = _config_and_prompt()
    config["hardware"]["identical_rtx3090_substitution"] = "allowed"
    with pytest.raises(ProtocolValidationError, match="GPU comparability policy"):
        validate_protocol_payload(
            config, ROOT, prompt_payload=prompt_payload, check_external=False
        )


def test_protocol_rejects_ambiguous_dataset_and_loss_wording():
    config, prompt_payload = _config_and_prompt()
    config["perplexity"]["selection"] = "source order; random sampling disabled"
    with pytest.raises(ProtocolValidationError, match="dataset selection policy"):
        validate_protocol_payload(
            config, ROOT, prompt_payload=prompt_payload, check_external=False
        )

    config, prompt_payload = _config_and_prompt()
    config["perplexity"]["loss"] = "not token-weighted"
    with pytest.raises(ProtocolValidationError, match="token-weighted loss"):
        validate_protocol_payload(
            config, ROOT, prompt_payload=prompt_payload, check_external=False
        )


def test_protocol_rejects_stale_any_precision_manifest_or_checkout(monkeypatch):
    config, prompt_payload = _config_and_prompt()
    config["identities"]["any_precision"]["manifest_commit"] = "stale"
    with pytest.raises(ProtocolValidationError, match="protocol revision"):
        validate_protocol_payload(
            config, ROOT, prompt_payload=prompt_payload, check_external=False
        )

    config, prompt_payload = _config_and_prompt()
    monkeypatch.setattr(validator, "_git_revision", lambda path: "stale")
    with pytest.raises(ProtocolValidationError, match="checked-out revision"):
        validate_protocol_payload(
            config, ROOT, prompt_payload=prompt_payload, check_external=False
        )


def test_protocol_rejects_incomplete_post_result_policy_or_deferred_list():
    config, prompt_payload = _config_and_prompt()
    config["release_criteria"]["failure_outcomes"].pop("all_gates_pass")
    with pytest.raises(ProtocolValidationError, match="all-gates-pass outcome"):
        validate_protocol_payload(
            config, ROOT, prompt_payload=prompt_payload, check_external=False
        )

    config, prompt_payload = _config_and_prompt()
    config["deferred_mechanisms"] = config["deferred_mechanisms"][:-1]
    with pytest.raises(ProtocolValidationError, match="complete deferred mechanisms"):
        validate_protocol_payload(
            config, ROOT, prompt_payload=prompt_payload, check_external=False
        )
