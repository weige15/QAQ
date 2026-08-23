from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import evaluate_synchronous_packed_loading

pytestmark = pytest.mark.skipif(
    os.environ.get("QAQ_RUN_S08_REAL") != "1",
    reason="set QAQ_RUN_S08_REAL=1 for the external Qwen3 artifact",
)


@pytest.fixture(scope="module")
def real_models():
    examples = evaluate_synchronous_packed_loading._load_examples()
    return (
        examples,
        evaluate_synchronous_packed_loading._load_student("resident"),
        evaluate_synchronous_packed_loading._load_student("on_demand"),
    )


def test_real_on_demand_has_cpu_authority_and_no_hidden_packed_copy(real_models):
    examples, _, on_demand = real_models
    state, context = evaluate_synchronous_packed_loading._context_for(
        on_demand, examples[0], "s08-real-audit"
    )
    assert context is not None
    assert len(context.sources) == 252
    assert all(source.qweight.device.type == "cpu" for source in context.sources.values())
    assert all(source.lut4.device.type == "cpu" for source in context.sources.values())
    assert all(source.lut8.device.type == "cpu" for source in context.sources.values())
    assert not any(
        module.__class__.__name__ == "AnyPrecisionLinear" for module in on_demand.base.modules()
    )
    assert context.retained_packed_bytes == 0
    state.end_request()


def test_real_hard_routes_and_transferred_execution_match(real_models):
    examples, resident, on_demand = real_models
    example = examples[0]
    input_ids = example.input_ids.unsqueeze(0).to(evaluate_synchronous_packed_loading.DEVICE)
    resident_output, resident_state, _, _ = evaluate_synchronous_packed_loading._hard_forward(
        resident,
        example,
        request_id="s08-real-resident",
        input_ids=input_ids,
        use_cache=False,
    )
    on_output, on_state, context, _ = evaluate_synchronous_packed_loading._hard_forward(
        on_demand,
        example,
        request_id="s08-real-on-demand",
        input_ids=input_ids,
        use_cache=False,
    )
    assert context is not None
    assert evaluate_synchronous_packed_loading._route_map(
        resident_state
    ) == evaluate_synchronous_packed_loading._route_map(on_state)
    assert torch.isfinite(resident_output.logits).all()
    assert torch.isfinite(on_output.logits).all()
    assert torch.equal(resident_output.logits, on_output.logits)
    transfer = evaluate_synchronous_packed_loading._transfer_summary(
        context,
        evaluate_synchronous_packed_loading._route_map(on_state),
        prefill_count=len(context.records),
    )
    assert transfer["physical_accounting_matches"]
    assert transfer["decode_bytes"] == 0
    assert transfer["first_use_events"] == 252
    assert transfer["reuse_bytes"] == 0
    resident_state.end_request()
    on_state.end_request()


def test_real_decode_reuses_and_later_request_transfers_again(real_models):
    examples, _, on_demand = real_models
    generation = evaluate_synchronous_packed_loading._generation(
        on_demand, examples[0], request_id="s08-real-generation"
    )
    assert generation["all_finite"]
    assert generation["route_map_unchanged_during_decode"]
    assert generation["decode_transfer_bytes"] == 0
    _, state, context, _ = evaluate_synchronous_packed_loading._hard_forward(
        on_demand,
        examples[1],
        request_id="s08-real-later",
        input_ids=examples[1].input_ids.unsqueeze(0).to(evaluate_synchronous_packed_loading.DEVICE),
        use_cache=False,
    )
    assert context is not None
    assert context.records[0].event == "first_use"
    assert sum(record.transferred_bytes for record in context.records) > 0
    state.end_request()
