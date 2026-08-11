import importlib

import numpy as np

from qaq.quantization.reference_codec import pack
from qaq.s01_backend import ANY_PRECISION_ROOT, load_pinned_backend


def _pinned_pack(codes):
    load_pinned_backend()
    pinned_pack = importlib.import_module("any_precision.quantization.pack")
    parent_bits = 8
    n_rows, logical_k = codes.shape
    bitmaps = np.empty((parent_bits, codes.size // 8), dtype=np.uint8)
    flat_codes = codes.reshape(-1)
    for plane in range(parent_bits):
        bitmaps[plane] = np.packbits(
            ((flat_codes >> (parent_bits - 1 - plane)) & 1).astype(bool)
        )
    bitmaps = bitmaps.reshape(parent_bits, n_rows, logical_k // 8)
    return pinned_pack._permute_bitmaps_int32(bitmaps)


def test_reference_plane_order_matches_pinned_packer():
    assert str(ANY_PRECISION_ROOT).endswith("third_party/any-precision-llm")
    generator = np.random.default_rng(20260811)
    codes = generator.integers(0, 256, size=(3, 1024), dtype=np.uint16).astype(np.uint8)
    np.testing.assert_array_equal(pack(codes), _pinned_pack(codes))


def test_adjacent_parent_planes_have_distinct_physical_words():
    codes = np.zeros((4, 32), dtype=np.uint8)
    codes[:, ::2] = 0x80
    codes[:, 1::2] = 0x40
    packed = pack(codes)
    words = packed.view(np.uint32)
    assert words[0, 0, 0] == 0xAAAAAAAA
    assert words[1, 0, 0] == 0x55555555
    assert np.count_nonzero(words[2:]) == 0
