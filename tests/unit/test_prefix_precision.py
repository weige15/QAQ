import numpy as np

from qaq.quantization.reference_codec import pack, reconstruct, unpack


def test_four_bit_execution_consumes_the_leading_parent_plane_prefix():
    generator = np.random.default_rng(20260811)
    codes8 = generator.integers(0, 256, size=(4, 1024), dtype=np.uint16).astype(np.uint8)
    packed = pack(codes8)
    lut4 = np.linspace(-1.0, 1.0, 16, dtype=np.float16).reshape(1, 16).repeat(4, axis=0)
    lut8 = np.linspace(-2.0, 2.0, 256, dtype=np.float16).reshape(1, 256).repeat(4, axis=0)

    np.testing.assert_array_equal(unpack(packed, precision=4), codes8 >> 4)
    np.testing.assert_array_equal(
        reconstruct(packed, lut4, precision=4),
        np.take_along_axis(lut4, codes8 >> 4, axis=1),
    )
    np.testing.assert_array_equal(
        reconstruct(packed, lut8, precision=8),
        np.take_along_axis(lut8, codes8, axis=1),
    )
