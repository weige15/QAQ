from __future__ import annotations

import torch

from qaq.model.manual import PrecisionTrace
from qaq.model.request_state import QaqRequestState
from qaq.model.static import run_static_smoke
from qaq.quantization.backend import build_case, packed_output
from qaq.router.network import THREE_WAY_CANDIDATE_BITS
from qaq.router.soft_linear import SoftPackedLinear


def test_real_packed_soft_endpoints_match_pinned_4_and_8_bit_paths():
    case = build_case()
    soft = SoftPackedLinear(case.linear)
    output4 = soft(case.inputs, torch.tensor([1.0, 0.0], device=case.device))
    output8 = soft(case.inputs, torch.tensor([0.0, 1.0], device=case.device))
    assert torch.equal(output4, packed_output(case, 4))
    assert torch.equal(output8, packed_output(case, 8))
    assert torch.isfinite(output4).all() and torch.isfinite(output8).all()


def test_real_qwen3_soft_endpoints_match_s04_and_s03(manual_case, static_case):
    manual_model, inputs, torch_module = manual_case
    static_model, static_inputs, _ = static_case
    for bits in (4, 8):
        state = QaqRequestState(f"s06-endpoint-{bits}", prompt_length=int(inputs["attention_mask"].sum()))
        trace = PrecisionTrace()

        def fixed_router(layer_index, unit_type, feature, selected_bits=bits):
            del layer_index, unit_type
            probabilities = torch.zeros(2, device=feature.device, dtype=feature.dtype)
            probabilities[0 if selected_bits == 4 else 1] = 1
            return probabilities

        with torch_module.inference_mode():
            soft_logits = manual_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                use_cache=False,
                request_state=state,
                phase="prefill",
                soft_router=fixed_router,
                trace=trace,
            ).logits
        static_logits = run_static_smoke(static_model, static_inputs, bits, torch_module)["logits"]
        assert torch_module.isfinite(soft_logits).all()
        assert torch_module.allclose(soft_logits.float(), static_logits.float(), atol=1e-3, rtol=1e-3)
        assert len(trace.soft_records) == 252


def test_real_qwen3_three_way_soft_endpoints_match_static(manual_case, static_case):
    manual_model, inputs, torch_module = manual_case
    static_model, static_inputs, _ = static_case
    for index, bits in enumerate(THREE_WAY_CANDIDATE_BITS):
        state = QaqRequestState(
            f"s10b-qwen3-endpoint-{bits}",
            prompt_length=int(inputs["attention_mask"].sum()),
            candidate_bits=THREE_WAY_CANDIDATE_BITS,
        )

        def fixed_router(layer_index, unit_type, feature, selected=index):
            del layer_index, unit_type
            probabilities = torch.zeros(3, device=feature.device, dtype=feature.dtype)
            probabilities[selected] = 1
            return probabilities

        with torch_module.inference_mode():
            soft_logits = manual_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                use_cache=False,
                request_state=state,
                phase="prefill",
                soft_router=fixed_router,
            ).logits
        static_logits = run_static_smoke(static_model, static_inputs, bits, torch_module)["logits"]
        assert torch_module.isfinite(soft_logits).all()
        assert torch_module.allclose(soft_logits.float(), static_logits.float(), atol=1e-3, rtol=1e-3)


def test_real_packed_soft_mixture_propagates_gradient_to_probabilities():
    case = build_case()
    soft = SoftPackedLinear(case.linear)
    probabilities = torch.tensor([0.3, 0.7], device=case.device, requires_grad=True)
    output = soft(case.inputs, probabilities)
    output.square().mean().backward()
    assert probabilities.grad is not None
    assert torch.isfinite(probabilities.grad).all()
    assert torch.count_nonzero(probabilities.grad).item() > 0
