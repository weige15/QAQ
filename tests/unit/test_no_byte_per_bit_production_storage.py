import torch

from qaq.s01_backend import QWEIGHT_DTYPE, K, N, build_case, require_cuda


def test_production_facing_qweight_uses_int32_packed_planes():
    require_cuda()
    case = build_case()
    qweight = case.linear.qweight

    assert qweight.dtype == QWEIGHT_DTYPE == torch.int32
    assert list(qweight.shape) == [8, N, K // 32]
    assert qweight.numel() * qweight.element_size() == 8 * N * (K // 32) * 4
    # A byte-per-logical-plane-bit reference would cost 8*N*K bytes here.
    assert qweight.numel() * qweight.element_size() < 8 * N * K
