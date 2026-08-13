import torch

from qaq.quantization.backend import (
    QWEIGHT_DTYPE,
    K,
    M,
    N,
    build_case,
    packed_output,
    require_cuda,
    storage_observations,
)


def test_single_linear_executes_packed_four_bit_path():
    require_cuda()
    case = build_case()
    output = packed_output(case, 4)
    storage = storage_observations(case)

    assert list(output.shape) == [M, N]
    assert str(output.dtype) == "torch.float16"
    assert output.is_cuda
    assert case.linear.qweight.dtype == QWEIGHT_DTYPE
    assert list(case.linear.qweight.shape) == [8, N, K // 32]
    assert storage["selected_packed_bytes"]["4"] == 4 * N * (K // 32) * 4
    assert storage["lookup"]["4"]["shape"] == [N, 16]
    assert storage["lookup"]["4"]["bytes"] == N * 16 * 2
    assert case.linear.precisions == [4, 8]


def test_four_bit_path_consumes_packed_storage():
    require_cuda()
    case = build_case()
    baseline = packed_output(case, 4)
    old_word = case.linear.qweight[0, 0, 0].clone()
    with torch.no_grad():
        case.linear.qweight[0, 0, 0] = old_word ^ torch.tensor(
            1, device=case.device, dtype=QWEIGHT_DTYPE
        )
    changed = packed_output(case, 4)
    with torch.no_grad():
        case.linear.qweight[0, 0, 0] = old_word
    assert not torch.equal(baseline, changed)
