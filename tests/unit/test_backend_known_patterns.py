import numpy as np
import torch

from qaq.quantization.reference_codec import pack
from qaq.quantization.backend import load_pinned_backend, require_cuda


def test_pinned_backend_reconstructs_deliberate_known_patterns():
    require_cuda()
    _, dequant_kbit, _ = load_pinned_backend()
    n_rows, logical_k = 4, 32
    patterns = {
        "all_zeros": np.zeros((n_rows, logical_k), dtype=np.uint8),
        "all_ones": np.full((n_rows, logical_k), 255, dtype=np.uint8),
        "positions_0_1_31": np.zeros((n_rows, logical_k), dtype=np.uint8),
        "alternating": np.tile([1, 0], (n_rows, logical_k // 2)).astype(np.uint8),
        "one_plane": np.full((n_rows, logical_k), 16, dtype=np.uint8),
    }
    patterns["positions_0_1_31"][:, [0, 1, 31]] = 1
    lut4 = torch.arange(16, dtype=torch.float16, device="cuda").repeat(n_rows, 1)
    lut8 = torch.arange(256, dtype=torch.float16, device="cuda").repeat(n_rows, 1)

    for codes in patterns.values():
        packed = pack(codes)
        packed_cuda = torch.from_numpy(packed).to(device="cuda", dtype=torch.int32)
        actual8 = dequant_kbit(packed_cuda, lut8, 8).cpu().numpy()
        actual4 = dequant_kbit(packed_cuda[:4].contiguous(), lut4, 4).cpu().numpy()
        np.testing.assert_array_equal(actual8, codes.astype(np.float16))
        np.testing.assert_array_equal(actual4, (codes >> 4).astype(np.float16))
