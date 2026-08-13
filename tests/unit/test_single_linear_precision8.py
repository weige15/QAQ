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


def test_single_linear_executes_packed_eight_bit_path():
    require_cuda()
    case = build_case()
    output = packed_output(case, 8)
    storage = storage_observations(case)

    assert list(output.shape) == [M, N]
    assert str(output.dtype) == "torch.float16"
    assert output.is_cuda
    assert case.linear.qweight.dtype == QWEIGHT_DTYPE
    assert list(case.linear.qweight.shape) == [8, N, K // 32]
    assert storage["selected_packed_bytes"]["8"] == 8 * N * (K // 32) * 4
    assert storage["lookup"]["8"]["shape"] == [N, 256]
    assert storage["lookup"]["8"]["bytes"] == N * 256 * 2
    assert case.linear.precisions == [4, 8]
