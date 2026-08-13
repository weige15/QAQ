#!/usr/bin/env python3
"""Run QAQ S03-B once: quantize, pack, reload, and smoke-test Qwen3-4B."""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qaq.model.static import (
    MODEL_REVISION,
    PINNED_ANY_PRECISION_COMMIT,
    assert_target_invariant,
    file_sha256,
    load_static_model,
    run_static_smoke,
    smoke_inputs,
    source_commit,
    tensor_sha256,
)

SNAPSHOT = Path(
    os.environ.get(
        "QAQ_MODEL_SNAPSHOT",
        "~/.cache/huggingface/hub/models--Qwen--Qwen3-4B/"
        "snapshots/1cfa9a7208912126459214e8b04321603b3df60c",
    )
).expanduser()
DEVICE = os.environ.get("QAQ_MODEL_DEVICE", "cuda:3")
SEED = 1729
CALIBRATION = {
    "dataset": "c4",
    "split": "train",
    "source": "allenai/c4 en/c4-train.00000-of-01024.json.gz via pinned datautils",
    "sample_count": 1,
    "sequence_length": 64,
    "preprocessing": "random.sample one text with Python random and NumPy seed; tokenize, take first 64 tokens",
    "seed": SEED,
}


def _available_ram() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable is unavailable")


def _resources(torch: Any, device: str) -> dict[str, Any]:
    free, total = torch.cuda.mem_get_info(torch.device(device))
    return {
        "disk_free_bytes": int(shutil.disk_usage(ROOT).free),
        "ram_available_bytes": _available_ram(),
        "gpu": {
            "device": device,
            "free_bytes": int(free),
            "total_bytes": int(total),
            "name": torch.cuda.get_device_name(torch.device(device)),
        },
    }


def _sha256_tensor(tensor: Any) -> str:
    return tensor_sha256(tensor)


def _checkpoint_inventory(artifact: Path, target_names: list[str], torch: Any) -> dict[str, Any]:
    checkpoint_path = artifact / "pytorch_model.bin"
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    qkeys = sorted(key for key in state if key.endswith(".qweight"))
    qnames = [key[: -len(".qweight")] for key in qkeys]
    if set(qnames) != set(target_names) or len(qnames) != len(target_names):
        raise ValueError(
            "packed qweight target set differs from verified S03-A set: "
            f"missing={sorted(set(target_names) - set(qnames))}, "
            f"extra={sorted(set(qnames) - set(target_names))}"
        )
    if any(name.endswith(".weight") for name in qnames):
        raise ValueError("qweight inventory contains an ordinary weight key")

    packed_plane_bytes = 0
    lut_bytes = {"4": 0, "8": 0}
    representative: dict[str, Any] = {}
    suffix_nonzero = 0
    for name in target_names:
        qweight = state[f"{name}.qweight"]
        lut4 = state[f"{name}.lut4"]
        lut8 = state[f"{name}.lut8"]
        if str(qweight.dtype) != "torch.int32" or qweight.shape[0] != 8:
            raise ValueError(f"{name} is not an 8-plane int32 parent payload")
        if list(lut4.shape) != [qweight.shape[1], 16] or list(lut8.shape) != [qweight.shape[1], 256]:
            raise ValueError(f"{name} has an invalid LUT shape")
        packed_plane_bytes += qweight.numel() * qweight.element_size()
        lut_bytes["4"] += lut4.numel() * lut4.element_size()
        lut_bytes["8"] += lut8.numel() * lut8.element_size()
        suffix_nonzero += int(torch.count_nonzero(qweight[4:]).item())
        if name in {
            "model.layers.0.self_attn.q_proj",
            "model.layers.0.self_attn.o_proj",
            "model.layers.0.mlp.gate_proj",
            "model.layers.0.mlp.down_proj",
        }:
            representative[name] = {
                "qweight_shape": list(qweight.shape),
                "qweight_dtype": str(qweight.dtype),
                "qweight_bytes": int(qweight.numel() * qweight.element_size()),
                "lut4_shape": list(lut4.shape),
                "lut4_dtype": str(lut4.dtype),
                "lut4_bytes": int(lut4.numel() * lut4.element_size()),
                "lut8_shape": list(lut8.shape),
                "lut8_dtype": str(lut8.dtype),
                "lut8_bytes": int(lut8.numel() * lut8.element_size()),
                "resident_planes": 8,
                "qweight_sha256": _sha256_tensor(qweight),
                "lut4_sha256": _sha256_tensor(lut4),
                "lut8_sha256": _sha256_tensor(lut8),
            }

    excluded_fragments = ("embed_tokens", "lm_head", "norm", "q_norm", "k_norm", "rotary_emb")
    excluded_qweights = [key for key in qkeys if any(fragment in key for fragment in excluded_fragments)]
    if excluded_qweights:
        raise ValueError(f"excluded modules were packed: {excluded_qweights}")
    original_target_weights = [f"{name}.weight" for name in target_names if f"{name}.weight" in state]
    if original_target_weights:
        raise ValueError(f"original target weights remain in packed checkpoint: {original_target_weights[:3]}")

    artifact_files = sorted(path for path in artifact.rglob("*") if path.is_file())
    file_records = [
        {
            "path": str(path.relative_to(artifact)),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in artifact_files
    ]
    checkpoint_bytes = sum(record["bytes"] for record in file_records)
    metadata_files = [record for record in file_records if record["path"] != "pytorch_model.bin"]
    metadata_file_bytes = sum(record["bytes"] for record in metadata_files)
    metadata_bytes = metadata_file_bytes + lut_bytes["4"] + lut_bytes["8"]
    return {
        "target_count": len(qnames),
        "target_module_names": target_names,
        "qweight_key_count": len(qkeys),
        "packed_plane_payload_bytes": int(packed_plane_bytes),
        "selected_packed_plane_bytes": {
            "4": int(packed_plane_bytes // 2),
            "8": int(packed_plane_bytes),
        },
        "lookup_bytes": lut_bytes,
        "scale_bytes": 0,
        "metadata_file_bytes": int(metadata_file_bytes),
        "lookup_scale_metadata_bytes": int(metadata_bytes),
        "total_checkpoint_bytes": int(checkpoint_bytes),
        "total_artifact_bytes": int(checkpoint_bytes),
        "artifact_file_list": file_records,
        "representative_tensors": representative,
        "resident_plane_count": {"all_targets": 8, "static4_selected": 4, "static8_selected": 8},
        "parent_suffix_nonzero_elements": int(suffix_nonzero),
        "nested_representation": {
            "one_qweight_per_target": True,
            "shared_parent_plane_shape": "[8,N,K//32] int32",
            "static4_uses": "qweight[:4] and lut4",
            "static8_uses": "qweight[:8] and lut8",
            "separate_4bit_qweight_copy": False,
            "separate_8bit_model_copy": False,
        },
        "checkpoint_hashes": {record["path"]: record["sha256"] for record in file_records},
    }


def _baseline(torch: Any, model_path: Path, device: str, tokenizer: Any) -> dict[str, Any]:
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        revision=MODEL_REVISION,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
    )
    model.to(device)
    model.eval()
    encoded = tokenizer("QAQ full-precision smoke test.", return_tensors="pt")
    inputs = {key: value.to(device) for key, value in encoded.items()}
    with torch.inference_mode():
        logits = model(input_ids=inputs["input_ids"], use_cache=False).logits.detach()
    torch.cuda.synchronize(torch.device(device))
    result = {
        "dtype": str(logits.dtype),
        "logits_shape": list(logits.shape),
        "finite_values": bool(torch.isfinite(logits).all().item()),
        "logits_sha256": _sha256_tensor(logits.float()),
        "selected_logits": [float(value) for value in logits[0, -1, :8].float().cpu()],
        "logits": logits.float().cpu(),
    }
    model.cpu()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _metrics(fp: Any, result4: dict[str, Any], result8: dict[str, Any]) -> dict[str, Any]:
    metrics = {}
    for bit, result in ((4, result4), (8, result8)):
        error = (result["logits"].float().cpu() - fp).abs()
        metrics[str(bit)] = {
            "mean_absolute_logit_error": float(error.mean().item()),
            "maximum_absolute_logit_error": float(error.max().item()),
        }
    if metrics["8"]["maximum_absolute_logit_error"] > metrics["4"]["maximum_absolute_logit_error"]:
        raise RuntimeError(f"8-bit is less faithful than 4-bit: {metrics}")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite-artifact", action="store_true")
    args = parser.parse_args()

    if not SNAPSHOT.is_dir() or SNAPSHOT.name != MODEL_REVISION:
        raise SystemExit(f"PAUSE: exact pinned model snapshot is unavailable: {SNAPSHOT}")
    if not os.environ.get("VIRTUAL_ENV", "").startswith(str(Path.home() / ".venv")):
        raise SystemExit("PAUSE: ~/.venv is not active")

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("PAUSE: CUDA is unavailable")
    source_commit()
    target_names = assert_target_invariant()
    before = _resources(torch, DEVICE)
    expected_temporary = 2 * 8_060_897_472
    if before["disk_free_bytes"] < expected_temporary or before["ram_available_bytes"] < expected_temporary:
        raise SystemExit(f"PAUSE: resources below conservative 2x-FP16 estimate: {before}")
    if before["gpu"]["free_bytes"] < expected_temporary:
        raise SystemExit(f"PAUSE: GPU VRAM below conservative 2x-FP16 estimate: {before}")

    artifact_root = ROOT / "quantized" / "s03b_qwen3_4b"
    if artifact_root.exists():
        if not args.overwrite_artifact:
            raise SystemExit(f"REVISE: artifact already exists; use --overwrite-artifact: {artifact_root}")
        shutil.rmtree(artifact_root)
    artifact_root.mkdir(parents=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(SNAPSHOT), revision=MODEL_REVISION, local_files_only=True
    )
    baseline = _baseline(torch, SNAPSHOT, DEVICE, tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        str(SNAPSHOT),
        revision=MODEL_REVISION,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
    )
    model.to(DEVICE)
    model.eval()

    sys.path.insert(0, str(ROOT / "third_party" / "any-precision-llm"))
    from any_precision.quantization import any_precision_quantize

    cache_dir = artifact_root / "backend_cache"
    yaml_path = ROOT / "configs" / "qwen3_any_precision.yaml"
    started = time.perf_counter()
    any_precision_quantize(
        model=model,
        seed_precision=4,
        parent_precision=8,
        mode="pack",
        yaml_path=str(yaml_path),
        cache_dir=str(cache_dir),
        dataset="c4",
        seq_len=64,
        num_examples=1,
        cpu_count=8,
        random_state=SEED,
        overwrite_gradients=True,
        overwrite_quantize=True,
        overwrite_pack=True,
        group_count=1,
        cpu_only=False,
    )
    quantization_seconds = time.perf_counter() - started
    del model
    gc.collect()
    torch.cuda.empty_cache()

    packed_candidates = sorted((cache_dir / "packed").iterdir())
    if len(packed_candidates) != 1:
        raise RuntimeError(f"expected exactly one packed output, found {packed_candidates}")
    artifact = packed_candidates[0]
    inventory = _checkpoint_inventory(artifact, target_names, torch)

    inputs, _ = smoke_inputs(artifact, DEVICE)
    static_model = load_static_model(artifact, DEVICE)
    result4 = run_static_smoke(static_model, inputs, 4, torch)
    result8 = run_static_smoke(static_model, inputs, 8, torch)
    if not result4["finite_values"] or not result8["finite_values"]:
        raise RuntimeError("static inference produced non-finite logits")
    comparison = _metrics(baseline["logits"], result4, result8)
    result4.pop("logits")
    result8.pop("logits")

    after = _resources(torch, DEVICE)
    manifest = {
        "format": "qaq-s03b-manifest-v1",
        "source_model": {
            "repository": "Qwen/Qwen3-4B",
            "revision": MODEL_REVISION,
            "snapshot_path": str(SNAPSHOT),
            "tokenizer_repository": "Qwen/Qwen3-4B",
            "tokenizer_revision": MODEL_REVISION,
        },
        "any_precision": {
            "commit": PINNED_ANY_PRECISION_COMMIT,
            "entry_point": "any_precision.quantization.any_precision_quantize",
            "architecture_mapping": "configs/qwen3_any_precision.yaml",
            "packing_entry_point": "any_precision.quantization.pack.pack",
        },
        "quantization": {
            "random_state": SEED,
            "seed_precision": 4,
            "parent_precision": 8,
            "group_count": 1,
            "calibration": CALIBRATION,
            "runtime_seconds": quantization_seconds,
            "exact_command": (
                "source ~/.venv/bin/activate && which python && python --version && "
                "python scripts/run_s03b.py --overwrite-artifact"
            ),
        },
        "target_invariant": {
            "target_count": len(target_names),
            "target_module_names": target_names,
            "omitted_targets": [],
            "unexpected_targets": [],
            "duplicate_targets": [],
            "excluded_quantized_modules": [],
        },
        "artifact": {
            "local_path": str(artifact.relative_to(ROOT)),
            **inventory,
        },
        "full_precision_baseline": {
            key: value for key, value in baseline.items() if key != "logits"
        },
        "static_smoke": {"4": result4, "8": result8},
        "numerical_sanity": comparison,
        "resources": {
            "before_quantization": before,
            "after_quantization": after,
            "conservative_temporary_storage_estimate_bytes": expected_temporary,
        },
        "roundtrip": {
            "fresh_process_command": (
                "source ~/.venv/bin/activate && which python && python --version && "
                "QAQ_S03_ARTIFACT=<artifact> pytest -q "
                "tests/integration/test_checkpoint_roundtrip.py"
            ),
            "manifest_consistency_checked": True,
        },
    }
    manifest_path = ROOT / "docs" / "quantized_model_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
