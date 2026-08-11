import numpy as np
import torch

from qaq.quantization.reference_codec import pack, reconstruct
from qaq.s01_backend import load_pinned_backend, require_cuda


def test_reference_reconstruction_matches_pinned_dequantization_for_4_and_8_bits():
    require_cuda()
    _, dequant_kbit, _ = load_pinned_backend()
    generator = np.random.default_rng(20260811)
    codes = generator.integers(0, 256, size=(4, 32), dtype=np.uint16).astype(np.uint8)
    packed = pack(codes)
    packed_cuda = torch.from_numpy(packed).to(device="cuda", dtype=torch.int32)

    lut4 = np.linspace(-1.0, 1.0, 16, dtype=np.float16).reshape(1, 16).repeat(4, axis=0)
    lut8 = np.linspace(-2.0, 2.0, 256, dtype=np.float16).reshape(1, 256).repeat(4, axis=0)
    for precision, lut in ((4, lut4), (8, lut8)):
        lut_cuda = torch.from_numpy(lut).to(device="cuda", dtype=torch.float16)
        actual = dequant_kbit(packed_cuda[:precision].contiguous(), lut_cuda, precision)
        expected = reconstruct(packed, lut, precision)
        np.testing.assert_array_equal(actual.cpu().numpy(), expected)
