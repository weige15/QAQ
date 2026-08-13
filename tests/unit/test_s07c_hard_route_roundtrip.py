from __future__ import annotations

import copy
import json

import pytest
import torch

from qaq.model.request_state import QaqRequestState
from qaq.router.distillation import hard_route
from scripts import verify_s07b_roundtrip
from scripts.verify_s07b_roundtrip import assert_actual_hard_routes_match


def _route_logs_and_states():
    logs = []
    states = []
    for request_index, request_id in enumerate(("validation-3", "validation-1000")):
        attention_routes = [4 if (layer + request_index) % 2 == 0 else 8 for layer in range(36)]
        ffn_routes = [8 if (layer + request_index) % 2 == 0 else 4 for layer in range(36)]
        states.append(
            QaqRequestState(
                request_id,
                prompt_length=32,
                attention_routes=attention_routes,
                ffn_routes=ffn_routes,
                layer_count=36,
            )
        )
        for unit_type, routes in (("attention", attention_routes), ("ffn", ffn_routes)):
            for layer, selected_bit in enumerate(routes):
                probabilities = (0.75, 0.25) if selected_bit == 4 else (0.25, 0.75)
                logs.append(
                    {
                        "request_id": request_id,
                        "layer": layer,
                        "unit_type": unit_type,
                        "p4": probabilities[0],
                        "p8": probabilities[1],
                        "hard_bit": selected_bit,
                    }
                )
    return logs, states


def test_direct_hard_route_comparison_catches_altered_hard_record():
    soft_logs, states = _route_logs_and_states()
    assert len(soft_logs) == 144
    assert all(
        int(hard_route(torch.tensor([record["p4"], record["p8"]]))) == record["hard_bit"]
        for record in soft_logs
    )

    altered_hard_logs = copy.deepcopy(soft_logs)
    altered_hard_logs[0]["hard_bit"] = 8 if altered_hard_logs[0]["hard_bit"] == 4 else 4

    with pytest.raises(AssertionError, match="mismatch_count"):
        assert_actual_hard_routes_match(altered_hard_logs, states)


def test_checkpoint_identity_requires_recorded_hash_match(tmp_path, monkeypatch):
    checkpoint_path = tmp_path / "router.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    result_path = tmp_path / "result.json"
    result = {
        "checkpoint": {
            "external_path": str(checkpoint_path),
            "sha256": "stale-recorded-hash",
        },
        "hard_route_determinism": {"passed": True},
        "stage_gate": {
            "checkpoint_roundtrip_passed": True,
            "hard_route_determinism_passed": True,
            "engineering_gate": "CONTINUE",
            "next_action": "Begin S08",
        },
    }
    monkeypatch.setattr(verify_s07b_roundtrip, "EXPECTED_CHECKPOINT_SHA256", "expected-hash")
    monkeypatch.setattr(verify_s07b_roundtrip, "_sha256", lambda path: "expected-hash")

    with pytest.raises(SystemExit, match="checkpoint identity mismatch"):
        verify_s07b_roundtrip._verify_checkpoint_identity(result, result_path)

    saved = json.loads(result_path.read_text())
    assert saved["checkpoint_roundtrip"]["checkpoint_identity_match"] is False
    assert saved["checkpoint_roundtrip"]["recorded_checkpoint_sha256"] == "stale-recorded-hash"
    assert saved["checkpoint_roundtrip"]["passed"] is False
    assert saved["hard_route_determinism"]["passed"] is False
    assert saved["stage_gate"]["checkpoint_roundtrip_passed"] is False
    assert saved["stage_gate"]["hard_route_determinism_passed"] is False
    assert saved["stage_gate"]["engineering_gate"] == "REVISE"


def test_roundtrip_failure_persists_failed_gate_fields(tmp_path):
    result_path = tmp_path / "result.json"
    result = {
        "checkpoint": {"sha256": "recorded-hash"},
        "hard_route_determinism": {"passed": True},
        "stage_gate": {
            "checkpoint_roundtrip_passed": True,
            "hard_route_determinism_passed": True,
            "engineering_gate": "CONTINUE",
            "next_action": "Begin S08",
        },
    }

    verify_s07b_roundtrip._persist_roundtrip_failure(
        result,
        result_path,
        checkpoint_sha256="actual-hash",
        checkpoint_identity_match=True,
        probabilities_match=True,
        soft_derived_hard_bits_match=True,
        hard_routes_match=False,
        hard_route_comparison={"passed": False},
        unchanged_packed_student=True,
        finite_logits=True,
        fixed_subset_count=2,
        route_maps_identical_on_repeat=True,
        logits_identical_on_repeat=True,
    )

    saved = json.loads(result_path.read_text())
    assert saved["checkpoint_roundtrip"]["hard_routes_match_recorded_result"] is False
    assert saved["checkpoint_roundtrip"]["passed"] is False
    assert saved["hard_route_determinism"]["passed"] is False
    assert saved["stage_gate"]["checkpoint_roundtrip_passed"] is False
    assert saved["stage_gate"]["hard_route_determinism_passed"] is False
    assert saved["stage_gate"]["engineering_gate"] == "REVISE"
