from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

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
