import numpy as np
import pytest

from qaq.quantization.reference_codec import pack, unpack
from qaq.s01_backend import load_pinned_backend


def test_aligned_width_is_supported_without_padding():
    codes = np.arange(4 * 32, dtype=np.uint8).reshape(4, 32)
    packed = pack(codes)
    assert packed.shape == (8, 4, 1)
    np.testing.assert_array_equal(unpack(packed), codes)


@pytest.mark.parametrize("logical_k", [31, 33, 40, 56])
def test_non_aligned_width_is_rejected_instead_of_silently_padded(logical_k):
    codes = np.zeros((4, logical_k), dtype=np.uint8)
    with pytest.raises(ValueError, match="divisible by 32"):
        pack(codes)


@pytest.mark.parametrize("logical_k", [31, 33, 40, 56])
def test_pinned_source_pack_path_rejects_non_aligned_width(logical_k):
    load_pinned_backend()
    from any_precision.quantization import pack as pinned_pack

    n_rows = 4
    codes = np.zeros((n_rows, logical_k), dtype=np.uint8)
    with pytest.raises((AssertionError, ValueError)):
        bitmaps = np.empty((8, codes.size // 8), dtype=np.uint8)
        for plane in range(8):
            bitmaps[plane] = np.packbits(
                ((codes.reshape(-1) >> (7 - plane)) & 1).astype(bool)
            )
        bitmaps = bitmaps.reshape(8, n_rows, logical_k // 8)
        pinned_pack._permute_bitmaps_int32(bitmaps)
