from __future__ import annotations

import hashlib

import torch

from qaq.s03_static import load_static_model, source_commit


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
