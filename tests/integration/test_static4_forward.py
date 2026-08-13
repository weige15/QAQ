from qaq.model.static import run_static_smoke


def test_static_four_bit_forward_is_finite_and_deterministic(static_case, manifest):
    model, inputs, torch = static_case
    first = run_static_smoke(model, inputs, 4, torch)
    second = run_static_smoke(model, inputs, 4, torch)
    assert first["finite_values"]
    assert first["logits_shape"] == [1, 8, 151936]
    assert first["logits_sha256"] == second["logits_sha256"]
    assert first["logits_sha256"] == manifest["static_smoke"]["4"]["logits_sha256"]
