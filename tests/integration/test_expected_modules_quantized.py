from qaq.s03_static import assert_target_invariant


def test_exact_verified_quantized_target_set(checkpoint, manifest):
    expected = set(assert_target_invariant())
    observed = {key.removesuffix(".qweight") for key in checkpoint if key.endswith(".qweight")}
    assert observed == expected
    assert len(observed) == 252
    assert manifest["target_invariant"]["omitted_targets"] == []
    assert manifest["target_invariant"]["unexpected_targets"] == []


def test_excluded_modules_are_not_replaced(checkpoint):
    excluded = ("embed_tokens", "lm_head", "norm", "q_norm", "k_norm", "rotary_emb")
    assert not any(
        key.endswith(".qweight") and any(fragment in key for fragment in excluded)
        for key in checkpoint
    )
    assert "model.embed_tokens.weight" in checkpoint
    assert "lm_head.weight" in checkpoint
    assert "model.norm.weight" in checkpoint
