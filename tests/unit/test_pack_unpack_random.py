import hashlib

import numpy as np

from qaq.quantization.reference_codec import pack, reconstruct, unpack


def test_random_round_trip_and_reconstruction_is_deterministic():
    generator = np.random.default_rng(20260811)
    codes = generator.integers(0, 256, size=(5, 1024), dtype=np.uint16).astype(np.uint8)
    lut = generator.normal(0.0, 0.5, size=(5, 256)).astype(np.float16)

    packed_first = pack(codes, parent_bits=8)
    packed_second = pack(codes, parent_bits=8)

    np.testing.assert_array_equal(packed_first, packed_second)
    assert hashlib.sha256(codes.tobytes()).hexdigest() == (
        "476bcfc7f78e4bf89bec37847f6f3a49370fb5d09cba0dc28b6b30af50171334"
    )
    assert hashlib.sha256(packed_first.tobytes()).hexdigest() == (
        "1981743588b29dd277ab68f7f69bbdff7bce90e3d3ce4a40bc90066691d9a648"
    )
    np.testing.assert_array_equal(unpack(packed_first), codes)
    np.testing.assert_array_equal(
        reconstruct(packed_first, lut, precision=8),
        np.take_along_axis(lut, codes, axis=1),
    )
