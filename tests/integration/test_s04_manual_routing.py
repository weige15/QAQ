from __future__ import annotations

import os

import pytest

from qaq.s03_static import run_static_smoke, smoke_inputs, tensor_sha256
from qaq.s04_manual import (
    LAYER_COUNT,
    PrecisionPlan,
    PrecisionTrace,
    expected_trace,
    load_manual_model,
)

PARITY_ATOL = 1e-3
PARITY_RTOL = 1e-3


@pytest.fixture(scope="session")
def manual_case(artifact):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("S04 manual routing integration tests require CUDA")
    device = os.environ.get("QAQ_MODEL_DEVICE", "cuda:3")
    model = load_manual_model(artifact, device)
    inputs, _ = smoke_inputs(artifact, device)
    return model, inputs, torch


def _run(model, inputs, torch, plan):
    trace = PrecisionTrace()
    with torch.inference_mode():
        logits = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            use_cache=False,
            precision_plan=plan,
            trace=trace,
        ).logits.detach()
    torch.cuda.synchronize(logits.device)
    return logits, trace


def test_manual_all_four_matches_verified_s03_static_four(manual_case, static_case, manifest):
    manual_model, manual_inputs, torch = manual_case
    static_model, static_inputs, _ = static_case
    plan = PrecisionPlan.uniform(4)

    manual_logits, trace = _run(manual_model, manual_inputs, torch, plan)
    static_result = run_static_smoke(static_model, static_inputs, 4, torch)

    assert static_result["logits_sha256"] == manifest["static_smoke"]["4"]["logits_sha256"]
    assert torch.isfinite(manual_logits).all()
    assert torch.allclose(
        manual_logits.float(), static_result["logits"].float(), atol=PARITY_ATOL, rtol=PARITY_RTOL
    )
    assert trace.records == expected_trace(plan)


def test_manual_all_eight_matches_verified_s03_static_eight(manual_case, static_case, manifest):
    manual_model, manual_inputs, torch = manual_case
    static_model, static_inputs, _ = static_case
    plan = PrecisionPlan.uniform(8)

    manual_logits, trace = _run(manual_model, manual_inputs, torch, plan)
    static_result = run_static_smoke(static_model, static_inputs, 8, torch)

    assert static_result["logits_sha256"] == manifest["static_smoke"]["8"]["logits_sha256"]
    assert torch.isfinite(manual_logits).all()
    assert torch.allclose(
        manual_logits.float(), static_result["logits"].float(), atol=PARITY_ATOL, rtol=PARITY_RTOL
    )
    assert trace.records == expected_trace(plan)


def test_attention_unit_change_only_changes_that_layers_four_attention_calls(manual_case):
    model, inputs, torch = manual_case
    layer_index = 7
    all_four = PrecisionPlan.uniform(4)
    attention_changed = PrecisionPlan(
        attention_bits=all_four.attention_bits[:layer_index]
        + (8,)
        + all_four.attention_bits[layer_index + 1 :],
        ffn_bits=all_four.ffn_bits,
    )

    base_logits, base_trace = _run(model, inputs, torch, all_four)
    changed_logits, changed_trace = _run(model, inputs, torch, attention_changed)
    changed = [
        (before, after)
        for before, after in zip(base_trace.records, changed_trace.records)
        if before.selected_bits != after.selected_bits
    ]

    assert len(changed) == 4
    assert all(after.layer_index == layer_index for _, after in changed)
    assert all(after.unit_type == "attention" for _, after in changed)
    assert [after.module_path.rsplit(".", 1)[-1] for _, after in changed] == [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ]
    assert not torch.equal(base_logits, changed_logits)


def test_ffn_unit_change_only_changes_that_layers_three_ffn_calls(manual_case):
    model, inputs, torch = manual_case
    layer_index = 19
    all_four = PrecisionPlan.uniform(4)
    ffn_changed = PrecisionPlan(
        attention_bits=all_four.attention_bits,
        ffn_bits=all_four.ffn_bits[:layer_index] + (8,) + all_four.ffn_bits[layer_index + 1 :],
    )

    base_logits, base_trace = _run(model, inputs, torch, all_four)
    changed_logits, changed_trace = _run(model, inputs, torch, ffn_changed)
    changed = [
        (before, after)
        for before, after in zip(base_trace.records, changed_trace.records)
        if before.selected_bits != after.selected_bits
    ]

    assert len(changed) == 3
    assert all(after.layer_index == layer_index for _, after in changed)
    assert all(after.unit_type == "ffn" for _, after in changed)
    assert [after.module_path.rsplit(".", 1)[-1] for _, after in changed] == [
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    assert not torch.equal(base_logits, changed_logits)


@pytest.mark.parametrize(
    "plan",
    [
        PrecisionPlan(
            attention_bits=(8,) * LAYER_COUNT,
            ffn_bits=(4,) * LAYER_COUNT,
        ),
        PrecisionPlan(
            attention_bits=(4,) * LAYER_COUNT,
            ffn_bits=(8,) * LAYER_COUNT,
        ),
        PrecisionPlan(
            attention_bits=tuple(4 if layer % 2 == 0 else 8 for layer in range(LAYER_COUNT)),
            ffn_bits=tuple(8 if layer % 2 == 0 else 4 for layer in range(LAYER_COUNT)),
        ),
    ],
)
def test_mixed_manual_plans_are_finite_deterministic_and_trace_exact(manual_case, plan):
    model, inputs, torch = manual_case
    first, first_trace = _run(model, inputs, torch, plan)
    second, second_trace = _run(model, inputs, torch, plan)

    assert torch.isfinite(first).all()
    assert tensor_sha256(first.float()) == tensor_sha256(second.float())
    assert first_trace.records == expected_trace(plan)
    assert second_trace.records == expected_trace(plan)


def test_sequential_plans_do_not_leak_state(manual_case):
    model, inputs, torch = manual_case
    all_four = PrecisionPlan.uniform(4)
    all_eight = PrecisionPlan.uniform(8)
    mixed = PrecisionPlan(
        attention_bits=(8,) * LAYER_COUNT,
        ffn_bits=(4,) * LAYER_COUNT,
    )
    plans = (all_four, all_eight, all_four, mixed, all_eight)
    outputs = []
    traces = []
    for plan in plans:
        logits, trace = _run(model, inputs, torch, plan)
        outputs.append(logits)
        traces.append(trace)

    assert torch.equal(outputs[0], outputs[2])
    assert torch.equal(outputs[1], outputs[4])
    for plan, trace in zip(plans, traces):
        assert trace.records == expected_trace(plan)
