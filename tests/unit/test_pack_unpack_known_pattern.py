import hashlib

import numpy as np

from qaq.quantization.reference_codec import pack, reconstruct, unpack


def _u32_words(packed):
    return packed.view(np.uint32)


def test_known_patterns_reconstruct_and_expose_word_order():
    n_rows, logical_k = 4, 32

    patterns = {
        "all_zeros": np.zeros((n_rows, logical_k), dtype=np.uint8),
        "all_ones": np.full((n_rows, logical_k), 255, dtype=np.uint8),
        "positions_0_1_31": np.zeros((n_rows, logical_k), dtype=np.uint8),
        "alternating": np.tile([1, 0], (n_rows, logical_k // 2)).astype(np.uint8),
        "one_plane": np.full((n_rows, logical_k), 16, dtype=np.uint8),
    }
    patterns["positions_0_1_31"][:, [0, 1, 31]] = 1

    expected_words = {
        "all_zeros": [0] * 8,
        "all_ones": [0xFFFFFFFF] * 8,
        "positions_0_1_31": [0, 0, 0, 0, 0, 0, 0, 0xC0000001],
        "alternating": [0, 0, 0, 0, 0, 0, 0, 0xAAAAAAAA],
        "one_plane": [0, 0, 0, 0xFFFFFFFF, 0, 0, 0, 0],
    }
    expected_digests = {
        "all_zeros": "38723a2e5e8a17aa7950dc008209944e898f69a7bd10a23c839d341e935fd5ca",
        "all_ones": "e9175db65a9789096ca9cb5524d3abc2107df03e3c9ba3af1aca628f9c5d3bd2",
        "positions_0_1_31": "5bbd124bb7a8d14b52ab941119dab7af0bc4b5c8da168d5de56f5d3b28b8c6d3",
        "alternating": "7cf87f2f95dda7d35c75e2b031267703c7b169c93a4df2f29faa6e4aa75d8a16",
        "one_plane": "60b8df3bc1cdfda128c6aa5a9300682f972b5aac5ab1b0339b952331febc4ea3",
    }

    lut = np.arange(256, dtype=np.float32).reshape(1, 256).repeat(n_rows, axis=0)
    for name, codes in patterns.items():
        packed = pack(codes, parent_bits=8)
        assert packed.shape == (8, n_rows, 1)
        assert packed.dtype == np.int32
        assert _u32_words(packed[:, 0, 0]).tolist() == expected_words[name]
        assert hashlib.sha256(packed.tobytes()).hexdigest() == expected_digests[name]
        np.testing.assert_array_equal(unpack(packed), codes)
        np.testing.assert_array_equal(
            reconstruct(packed, lut, precision=8), codes.astype(np.float32)
        )
