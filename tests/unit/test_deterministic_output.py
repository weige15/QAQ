import torch

from qaq.s01_backend import (
    build_case,
    dequantized_reference,
    packed_output,
    require_cuda,
    tensor_digest,
)


def test_repeated_four_and_eight_bit_executions_are_bitwise_deterministic():
    require_cuda()
    case = build_case()
    for precision in (4, 8):
        first = packed_output(case, precision)
        second = packed_output(case, precision)
        assert torch.equal(first, second)
        assert tensor_digest(first) == tensor_digest(second)


def test_four_and_eight_bit_paths_are_genuinely_distinct():
    require_cuda()
    case = build_case()
    weight4, _ = dequantized_reference(case, 4)
    weight8, _ = dequantized_reference(case, 8)

    assert case.linear.precisions == [4, 8]
    assert list(case.linear.lut4.shape) == [64, 16]
    assert list(case.linear.lut8.shape) == [64, 256]
    assert torch.count_nonzero(case.linear.qweight[4:]).item() > 0
    assert not torch.equal(weight4, weight8)
    assert torch.max((weight4.float() - weight8.float()).abs()).item() > 0
