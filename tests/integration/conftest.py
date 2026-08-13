"""Shared S03-B integration fixtures; real artifact tests are opt-in."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _artifact_path() -> Path:
    manifest = json.loads((ROOT / "docs" / "quantized_model_manifest.json").read_text())
    configured = os.environ.get("QAQ_S03_ARTIFACT")
    path = Path(configured) if configured else ROOT / manifest["artifact"]["local_path"]
    return path.expanduser().resolve()


@pytest.fixture(scope="session")
def manifest():
    return json.loads((ROOT / "docs" / "quantized_model_manifest.json").read_text())


@pytest.fixture(scope="session")
def artifact():
    path = _artifact_path()
    if not path.is_dir():
        pytest.skip(f"S03-B artifact is unavailable: {path}")
    return path


@pytest.fixture(scope="session")
def checkpoint(artifact):
    import torch

    return torch.load(artifact / "pytorch_model.bin", map_location="cpu", weights_only=False)


@pytest.fixture(scope="session")
def static_case(artifact):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("S03-B static integration tests require CUDA")
    device = os.environ.get("QAQ_MODEL_DEVICE", "cuda:3")
    from qaq.model.static import load_static_model, smoke_inputs

    model = load_static_model(artifact, device)
    inputs, _ = smoke_inputs(artifact, device)
    return model, inputs, torch


@pytest.fixture(scope="session")
def manual_case(artifact):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("S04/S05 manual routing integration tests require CUDA")
    device = os.environ.get("QAQ_MODEL_DEVICE", "cuda:3")
    from qaq.model.static import smoke_inputs
    from qaq.model.manual import load_manual_model

    model = load_manual_model(artifact, device)
    inputs, _ = smoke_inputs(artifact, device)
    return model, inputs, torch
