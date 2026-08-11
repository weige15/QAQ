import numpy as np
import torch

from qaq.quantization.reference_codec import lut_byte_count, pack, packed_byte_count


def test_physical_packed_bytes_match_tensor_numel_times_element_size():
    n_rows, logical_k, parent_bits = 4, 1024, 8
    codes = np.zeros((n_rows, logical_k), dtype=np.uint8)
    packed = pack(codes, parent_bits=parent_bits)
    tensor = torch.from_numpy(packed)

    assert tensor.numel() * tensor.element_size() == packed.nbytes
    assert packed.nbytes == packed_byte_count(n_rows, logical_k, parent_bits)
    assert packed.nbytes == 8 * 4 * 32 * 4
    assert lut_byte_count(n_rows, 4) == 4 * 16 * 2
    assert lut_byte_count(n_rows, 8) == 4 * 256 * 2
