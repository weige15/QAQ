import hashlib

import numpy as np

from qaq.s01_backend import load_pinned_backend


def test_pinned_nested_quantizer_shares_parent_labels_but_changes_luts():
    load_pinned_backend()
    from any_precision.quantization.quantize import _seed_and_upscale_layer

    generator = np.random.default_rng(20260811)
    weights = generator.normal(0.0, 0.5, size=(4, 32)).astype(np.float32)
    gradients = np.ones_like(weights, dtype=np.float32)
    luts_by_module, parent_weights_by_module = _seed_and_upscale_layer(
        [gradients], [weights], seed_bit=4, parent_bit=8, group_count=1, random_state=1729
    )

    parent = parent_weights_by_module[0][:, 0, :].astype(np.uint8)
    lut4 = luts_by_module[0][0][:, 0, :].astype(np.float16)
    lut8 = luts_by_module[0][4][:, 0, :].astype(np.float16)
    assert parent.shape == (4, 32)
    assert lut4.shape == (4, 16)
    assert lut8.shape == (4, 256)
    assert np.array_equal(parent >> 4, (parent.astype(np.uint16) >> 4).astype(np.uint8))
    assert np.any(lut4 != lut8[:, :16])

    reconstruction4 = np.take_along_axis(lut4, parent >> 4, axis=1)
    reconstruction8 = np.take_along_axis(lut8, parent, axis=1)
    assert reconstruction4.shape == reconstruction8.shape == weights.shape
    assert np.any(reconstruction4 != reconstruction8)
    assert hashlib.sha256(parent.tobytes()).hexdigest() == (
        "5a31c268934a617b8b55d6198abeb35078a0214ff0d8ceb49fe26b55dde66010"
    )
    assert hashlib.sha256(lut4.tobytes()).hexdigest() == (
        "3e545d5c27ce357903878ddc1380e8e1c8edf776476aa4cbc9c318b39efe8204"
    )
    assert hashlib.sha256(lut8.tobytes()).hexdigest() == (
        "500c4cbe06f7f4dfecca4f588a0f90797c1deff551f71630f09e8eeffbf20791"
    )
