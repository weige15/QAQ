from qaq.s01_backend import assert_matches_reference, build_case, require_cuda


def test_packed_cuda_matches_pinned_dequantized_reference_at_both_precisions():
    require_cuda()
    case = build_case()
    for precision in (4, 8):
        result = assert_matches_reference(case, precision)
        assert result["allclose"]
        assert result["max_absolute_error"] >= 0.0
        assert result["mean_absolute_error"] >= 0.0
        assert result["meaningful_max_relative_error"] >= 0.0
