from __future__ import annotations

import os

import pytest
import torch

from qaq.model.static import add_pinned_backend_to_path, assert_target_invariant, run_static_smoke


def _quantized_modules(model):
    return [module for module in model.modules() if module.__class__.__name__ == "AnyPrecisionLinear"]


def test_real_artifact_has_finite_lut6_and_shared_parent_accounting(checkpoint, manifest):
    targets = assert_target_invariant()
    qweights = [checkpoint[f"{target}.qweight"] for target in targets]
    lut6 = [checkpoint[f"{target}.lut6"] for target in targets]

    assert len(qweights) == len(lut6) == manifest["artifact"]["target_count"] == 252
    assert all(qweight.dtype == torch.int32 and qweight.shape[0] == 8 for qweight in qweights)
    assert all(
        tuple(table.shape) == (qweight.shape[1], 64)
        and table.dtype == torch.float16
        and bool(torch.isfinite(table).all().item())
        for qweight, table in zip(qweights, lut6)
    )
    assert not any(key.endswith(".qweight6") for key in checkpoint)
    assert manifest["artifact"]["nested_representation"]["one_qweight_per_target"] is True
    assert manifest["artifact"]["nested_representation"]["separate_4bit_qweight_copy"] is False
    assert manifest["artifact"]["nested_representation"]["separate_8bit_model_copy"] is False

    expected_six_plane_bytes = sum(
        6 * qweight.shape[1] * qweight.shape[2] * qweight.element_size() for qweight in qweights
    )
    actual_lut6_bytes = sum(table.numel() * table.element_size() for table in lut6)
    assert expected_six_plane_bytes == 2_724_986_880
    assert actual_lut6_bytes == 141_557_760


def test_real_pinned_precision6_execution_matches_dequantized_reference(checkpoint):
    if not torch.cuda.is_available():
        pytest.skip("S10-A real precision-6 integration requires CUDA")

    add_pinned_backend_to_path()
    from any_precision.modules.AnyPrecisionLinear import AnyPrecisionLinear
    from any_precision_ext import dequant_kbit

    target = "model.layers.0.mlp.down_proj"
    device = torch.device(os.environ.get("QAQ_MODEL_DEVICE", "cuda:3"))
    qweight = checkpoint[f"{target}.qweight"]
    lut6 = checkpoint[f"{target}.lut6"]
    linear = AnyPrecisionLinear(
        qweight.shape[2] * 32,
        qweight.shape[1],
        supported_bits=[4, 5, 6, 7, 8],
        bias=False,
        precisions=[4, 5, 6, 7, 8],
        device=device,
        dtype=torch.float16,
    )
    with torch.no_grad():
        linear.qweight.copy_(qweight.to(device))
        linear.lut6.copy_(lut6.to(device))
    generator = torch.Generator(device="cpu").manual_seed(1729)
    inputs = torch.randn(
        (4, qweight.shape[2] * 32), generator=generator, dtype=torch.float32
    ).to(device=device, dtype=torch.float16)

    with torch.no_grad():
        actual = linear(inputs, precision=6)
        repeat = linear(inputs, precision=6)
        dense_weight = dequant_kbit(linear.qweight, linear.lut6, 6)
        reference = torch.matmul(inputs, dense_weight.transpose(0, 1))
    torch.cuda.synchronize(device)

    assert tuple(actual.shape) == (4, qweight.shape[1])
    assert actual.dtype == torch.float16 and actual.device == device
    assert bool(torch.isfinite(actual).all().item())
    assert bool(torch.isfinite(reference).all().item())
    assert torch.allclose(actual, reference, atol=5e-2, rtol=1e-2)
    assert torch.equal(actual, repeat)
    assert not list(linear.named_parameters())
    assert "weight" not in linear._parameters


def test_full_qwen3_static_precision6_smoke_is_finite_and_uses_one_packed_model(static_case):
    model, inputs, torch_module = static_case

    first = run_static_smoke(model, inputs, 6, torch_module)
    second = run_static_smoke(model, inputs, 6, torch_module)

    assert first["logits_shape"] == [1, 8, 151936]
    assert first["finite_values"]
    assert first["logits_sha256"] == second["logits_sha256"]
    assert len(_quantized_modules(model)) == 252
    assert sum(name.endswith(".qweight") for name, _ in model.named_buffers()) == 252
    assert not any(name.endswith(".qweight6") for name, _ in model.named_buffers())
    assert all("weight" not in module._parameters for module in _quantized_modules(model))
