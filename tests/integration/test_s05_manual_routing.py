from __future__ import annotations

import pytest

from qaq.model.request_state import QaqRequestState
from qaq.model.static import run_static_smoke
from qaq.model.manual import LAYER_COUNT, PrecisionPlan, PrecisionTrace


@pytest.mark.parametrize("bits", [4, 8])
def test_s05_prefill_manual_policy_preserves_s04_static_parity(
    manual_case, static_case, manifest, bits
):
    model, inputs, torch = manual_case
    static_model, static_inputs, _ = static_case
    mask = inputs["attention_mask"]
    state = QaqRequestState(
        request_id=f"s05-parity-{bits}",
        prompt_length=int(mask.sum().item()),
    )
    trace = PrecisionTrace()
    with torch.inference_mode():
        routed = model(
            input_ids=inputs["input_ids"],
            attention_mask=mask,
            use_cache=False,
            precision_plan=PrecisionPlan.uniform(bits),
            request_state=state,
            phase="prefill",
            trace=trace,
        ).logits.detach()
    static = run_static_smoke(static_model, static_inputs, bits, torch)
    assert torch.isfinite(routed).all()
    assert torch.allclose(routed.float(), static["logits"].float(), atol=1e-3, rtol=1e-3)
    assert len(trace.route_records) == 2 * LAYER_COUNT
    assert all(item.feature_computed and item.policy_invoked for item in trace.route_records)
    assert all(item.precision == bits for item in trace.route_records)
    assert manifest["static_smoke"][str(bits)]["logits_sha256"] == static["logits_sha256"]
