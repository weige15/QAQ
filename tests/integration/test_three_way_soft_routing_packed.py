from __future__ import annotations

import torch

from qaq.quantization.backend import build_case
from qaq.router.network import THREE_WAY_CANDIDATE_BITS
from qaq.router.soft_linear import SoftPackedLinear


def test_real_packed_three_way_endpoints_and_mixture():
    case = build_case()
    # The legacy S01 fixture intentionally contains only lut4/lut8.  Build a
    # same-parent test module with real pinned backend LUT6 storage so the
    # three-way test exercises the production CUDA six-bit dispatch.
    linear = case.linear.__class__(
        1024,
        64,
        list(range(4, 9)),
        bias=False,
        precisions=list(range(4, 9)),
        device=case.device,
        dtype=case.inputs.dtype,
    )
    with torch.no_grad():
        linear.qweight.copy_(case.linear.qweight)
        linear.lut4.copy_(case.linear.lut4)
        linear.lut6.copy_(case.linear.lut8[:, ::4].contiguous())
        linear.lut8.copy_(case.linear.lut8)
        linear.lut5.copy_(case.linear.lut8[:, ::8].contiguous())
        linear.lut7.copy_(case.linear.lut8[:, ::2].contiguous())
    soft = SoftPackedLinear(linear, candidate_bits=THREE_WAY_CANDIDATE_BITS)
    expected = [linear(case.inputs, precision=bits) for bits in THREE_WAY_CANDIDATE_BITS]
    for index, output in enumerate(expected):
        probabilities = torch.nn.functional.one_hot(torch.tensor(index), 3).to(
            device=case.device, dtype=case.inputs.dtype
        )
        actual = soft(case.inputs, probabilities)
        assert torch.equal(actual, output)
        assert torch.isfinite(actual).all()

    probabilities = torch.tensor([0.2, 0.3, 0.5], device=case.device, requires_grad=True)
    output = soft(case.inputs, probabilities)
    expected_mix = sum(
        weight * value for weight, value in zip(probabilities, expected, strict=True)
    )
    assert torch.allclose(output, expected_mix)
    output.square().mean().backward()
    assert probabilities.grad is not None
    assert torch.isfinite(probabilities.grad).all()
    assert torch.count_nonzero(probabilities.grad).item() > 0
