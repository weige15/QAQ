from __future__ import annotations

import copy

import pytest
import torch

from qaq.model.request_state import QaqRequestState
from qaq.router.distillation import hard_route
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
