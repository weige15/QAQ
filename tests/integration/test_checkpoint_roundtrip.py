from __future__ import annotations

import hashlib

import torch

from qaq.s03_static import load_static_model, run_static_smoke, smoke_inputs, source_commit


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_checkpoint_roundtrip_and_manifest_identity(artifact, checkpoint, manifest):
    assert manifest["source_model"]["revision"] == "1cfa9a7208912126459214e8b04321603b3df60c"
    assert manifest["any_precision"]["commit"] == source_commit()
    assert len([key for key in checkpoint if key.endswith(".qweight")]) == 252
    for name, record in manifest["artifact"]["representative_tensors"].items():
        assert list(checkpoint[f"{name}.qweight"].shape) == record["qweight_shape"]
        assert list(checkpoint[f"{name}.lut4"].shape) == record["lut4_shape"]
        assert list(checkpoint[f"{name}.lut8"].shape) == record["lut8_shape"]
    for relative, digest in manifest["artifact"]["checkpoint_hashes"].items():
        assert _sha256(artifact / relative) == digest


def test_fresh_process_loader_constructs_the_quantized_graph(artifact):
    model = load_static_model(artifact, "cuda:3" if torch.cuda.is_available() else "cpu")
    quantized = [module for module in model.modules() if module.__class__.__name__ == "AnyPrecisionLinear"]
    assert len(quantized) == 252


def test_fresh_process_static_smoke_matches_recorded_digests(artifact, manifest):
    if not torch.cuda.is_available():
        return
    device = "cuda:3"
    model = load_static_model(artifact, device)
    inputs, _ = smoke_inputs(artifact, device)
    for precision in (4, 8):
        first = run_static_smoke(model, inputs, precision, torch)
        second = run_static_smoke(model, inputs, precision, torch)
        assert first["finite_values"]
        assert first["logits_sha256"] == second["logits_sha256"]
        assert first["logits_sha256"] == manifest["static_smoke"][str(precision)]["logits_sha256"]
