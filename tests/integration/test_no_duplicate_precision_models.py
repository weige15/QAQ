import torch


def test_one_nested_parent_payload_per_target(checkpoint, manifest):
    qweights = [key for key in checkpoint if key.endswith(".qweight")]
    assert len(qweights) == manifest["artifact"]["target_count"] == 252
    assert all(checkpoint[key].dtype == torch.int32 for key in qweights)
    assert all(checkpoint[key].shape[0] == 8 for key in qweights)
    assert manifest["artifact"]["nested_representation"]["separate_4bit_qweight_copy"] is False
    assert manifest["artifact"]["nested_representation"]["separate_8bit_model_copy"] is False
    assert manifest["artifact"]["parent_suffix_nonzero_elements"] > 0


def test_four_bit_is_a_prefix_of_the_eight_bit_parent(checkpoint):
    for key in ("model.layers.0.self_attn.q_proj.qweight", "model.layers.0.mlp.down_proj.qweight"):
        qweight = checkpoint[key]
        assert qweight[:4].shape[0] == 4
        assert torch.count_nonzero(qweight[4:]) > 0
        lut4 = checkpoint[key.removesuffix(".qweight") + ".lut4"]
        lut8 = checkpoint[key.removesuffix(".qweight") + ".lut8"]
        assert lut4.shape[1] == 16
        assert lut8.shape[1] == 256
