"""S03 static nested-model helpers built around the pinned Any-Precision backend."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
ANY_PRECISION_ROOT = ROOT / "third_party" / "any-precision-llm"
PINNED_ANY_PRECISION_COMMIT = "a3257d02740cc5757c78673da534b0630ff3a4ea"
MODEL_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
TARGET_PRECISIONS = (4, 8)


def expected_target_names() -> list[str]:
    """Return the exact S03-A target list, in its recorded order."""

    evidence = json.loads((DOCS / "actual_model_modules.json").read_text())
    names = [
        record["full_module_path"]
        for record in evidence["target_modules"]
        if record["proposed_quantization_target"]
    ]
    if len(names) != evidence["total_target_count"]:
        raise ValueError("S03-A target evidence count does not match its target records")
    if len(names) != len(set(names)):
        raise ValueError("S03-A target evidence contains duplicate module names")
    return names


def mapping_target_names() -> list[str]:
    """Expand the explicit seven-projection Qwen mapping for all 36 layers."""

    relative = (
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    )
    return [
        f"model.layers.{layer}.{path}"
        for layer in range(36)
        for path in relative
    ]


def assert_target_invariant() -> list[str]:
    """Require exact equality between the verified list and explicit mapping."""

    actual = expected_target_names()
    mapped = mapping_target_names()
    if set(actual) != set(mapped):
        missing = sorted(set(actual) - set(mapped))
        extra = sorted(set(mapped) - set(actual))
        raise ValueError(f"S03-B target invariant failed: missing={missing}, extra={extra}")
    if len(actual) != 252 or len(actual) != len(set(actual)):
        raise ValueError("S03-B target invariant failed: expected 252 unique targets")
    return actual


def source_commit() -> str:
    import subprocess

    result = subprocess.run(
        ["git", "-C", str(ANY_PRECISION_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    commit = result.stdout.strip()
    if commit != PINNED_ANY_PRECISION_COMMIT:
        raise ValueError(f"Any-Precision commit mismatch: {commit}")
    status = subprocess.run(
        ["git", "-C", str(ANY_PRECISION_ROOT), "status", "--porcelain=v1"],
        check=True,
        capture_output=True,
        text=True,
    )
    if status.stdout.strip():
        raise ValueError("pinned Any-Precision source is dirty")
    return commit


def add_pinned_backend_to_path() -> None:
    source = str(ANY_PRECISION_ROOT)
    if source not in sys.path:
        sys.path.insert(0, source)


def tensor_sha256(tensor: Any) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path: str | os.PathLike[str]) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _replace_module(root: Any, name: str, replacement: Any) -> None:
    parts = name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part) if not part.isdigit() else parent[int(part)]
    setattr(parent, parts[-1], replacement)


def set_static_precision(model: Any, precision: int) -> None:
    if precision not in TARGET_PRECISIONS:
        raise ValueError(f"unsupported S03 precision: {precision}")
    for module in model.modules():
        if module.__class__.__name__ == "AnyPrecisionLinear":
            module.set_precision(precision)


def load_static_model(artifact: str | os.PathLike[str], device: str):
    """Load one saved nested model, then move its packed tensors to CUDA.

    The loader intentionally constructs the ordinary Qwen3 graph and replaces
    only the exact verified target paths.  This avoids the pinned backend's
    unchecked AutoArchConfig fallback for Qwen3 while using its
    AnyPrecisionLinear and CUDA kernels unchanged.
    """

    add_pinned_backend_to_path()
    import torch
    from any_precision.modules.AnyPrecisionLinear import AnyPrecisionLinear
    from transformers import AutoConfig, AutoModelForCausalLM

    artifact_path = Path(artifact).resolve()
    config = AutoConfig.from_pretrained(str(artifact_path), local_files_only=True)
    model = AutoModelForCausalLM.from_config(
        config=config, torch_dtype=torch.float16, trust_remote_code=False
    )
    supported_bits = list(range(config.anyprec["seed_precision"], config.anyprec["parent_precision"] + 1))
    target_names = assert_target_invariant()
    for name in target_names:
        module = dict(model.named_modules())[name]
        replacement = AnyPrecisionLinear(
            module.in_features,
            module.out_features,
            supported_bits=supported_bits,
            bias=module.bias is not None,
            precisions=supported_bits,
            dtype=torch.float16,
        )
        _replace_module(model, name, replacement)

    checkpoint = torch.load(artifact_path / "pytorch_model.bin", map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(checkpoint, strict=False)
    if missing or unexpected:
        raise ValueError(f"checkpoint graph mismatch: missing={missing}, unexpected={unexpected}")
    model.to(device)
    model.eval()
    return model


def smoke_inputs(artifact: str | os.PathLike[str], device: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(Path(artifact).resolve()), local_files_only=True
    )
    encoded = tokenizer("QAQ full-precision smoke test.", return_tensors="pt")
    return {key: value.to(device) for key, value in encoded.items()}, tokenizer


def run_static_smoke(model: Any, inputs: dict[str, Any], precision: int, torch: Any) -> dict[str, Any]:
    set_static_precision(model, precision)
    torch.cuda.reset_peak_memory_stats(inputs["input_ids"].device)
    with torch.inference_mode():
        outputs = model(input_ids=inputs["input_ids"], use_cache=False)
    torch.cuda.synchronize(inputs["input_ids"].device)
    logits = outputs.logits.detach()
    return {
        "precision": precision,
        "logits_shape": list(logits.shape),
        "logits_dtype": str(logits.dtype),
        "finite_values": bool(torch.isfinite(logits).all().item()),
        "logits_sha256": tensor_sha256(logits.float()),
        "selected_logits": [float(value) for value in logits[0, -1, :8].float().cpu()],
        "peak_allocated_gpu_bytes": int(torch.cuda.max_memory_allocated(inputs["input_ids"].device)),
        "logits": logits,
    }
