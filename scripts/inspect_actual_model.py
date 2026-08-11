#!/usr/bin/env python3
"""Inspect QAQ metadata or the exact pinned Qwen3 model without quantization."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import inspect_model

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
MODEL_CONFIG = ROOT / "configs" / "model.yaml"

def _model_identity() -> dict[str, str]:
    values = inspect_model._simple_yaml(MODEL_CONFIG)
    expected = {
        "repository": "Qwen/Qwen3-4B",
        "revision": "1cfa9a7208912126459214e8b04321603b3df60c",
    }
    for key, value in expected.items():
        if values.get(key) != value:
            raise ValueError(f"{MODEL_CONFIG} does not contain the locked {key}: {value}")
    return values


def _read_structure() -> dict[str, Any]:
    return json.loads((DOCS / "model_structure.json").read_text())


def _available_ram() -> int | None:
    try:
        with Path("/proc/meminfo").open() as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        return None
    return None


def _gpu_memory(torch: Any, device: str) -> dict[str, int] | None:
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return None
    index = torch.device(device).index
    if index is None:
        index = torch.cuda.current_device()
    free, total = torch.cuda.mem_get_info(index)
    return {"free_bytes": int(free), "total_bytes": int(total)}


def _resource_snapshot(torch: Any, device: str) -> dict[str, Any]:
    return {
        "ram_available_bytes": _available_ram(),
        "gpu": _gpu_memory(torch, device),
    }


def _expected_targets(structure: dict[str, Any]) -> dict[str, dict[str, Any]]:
    layers = structure["global_dimensions"]["num_decoder_layers"]
    expected: dict[str, dict[str, Any]] = {}
    for layer in range(layers):
        for unit, projections in (
            ("attention", structure["attention"]["projections"]),
            ("FFN", structure["ffn"]["projections"]),
        ):
            for spec in projections.values():
                path = f"model.layers.{layer}.{spec['path']}"
                expected[path] = {
                    "input_features": spec["input_dim"],
                    "output_features": spec["output_dim"],
                    "bias": spec["bias"],
                    "unit_type": unit,
                    "layer_index": layer,
                }
    return expected


def _tensor_shape(module: Any) -> list[int]:
    return list(module.weight.shape)


def _target_record(name: str, module: Any, expected: dict[str, Any]) -> dict[str, Any]:
    weight = module.weight
    return {
        "full_module_path": name,
        "python_module_class": f"{type(module).__module__}.{type(module).__name__}",
        "input_features": int(module.in_features),
        "output_features": int(module.out_features),
        "bias_present": module.bias is not None,
        "weight_dtype": str(weight.dtype),
        "weight_shape": _tensor_shape(module),
        "layer_index": expected["layer_index"],
        "unit_type": expected["unit_type"],
        "proposed_quantization_target": True,
    }


def _module_record(name: str, module: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": name,
        "python_module_class": f"{type(module).__module__}.{type(module).__name__}",
        "present": module is not None,
    }
    if hasattr(module, "weight") and module.weight is not None:
        record["weight_shape"] = _tensor_shape(module)
        record["weight_dtype"] = str(module.weight.dtype)
        record["bias_present"] = getattr(module, "bias", None) is not None
    return record


def _exclusions(model: Any, modules: dict[str, Any], torch: Any) -> dict[str, Any]:
    categories: dict[str, Any] = {}

    def one(category: str, path: str, policy: str) -> None:
        module = modules.get(path)
        categories[category] = {
            "policy": policy,
            "modules": [_module_record(path, module)] if module is not None else [],
            "present": module is not None,
        }

    one("token_embeddings", "model.embed_tokens", "excluded")
    one("lm_output_head", "lm_head", "excluded")
    one("final_normalization", "model.norm", "excluded")
    categories["per_layer_normalization"] = {
        "policy": "excluded",
        "modules": [
            _module_record(name, module)
            for name, module in modules.items()
            if name.endswith(("input_layernorm", "post_attention_layernorm"))
        ],
    }
    categories["qk_normalization"] = {
        "policy": "excluded",
        "modules": [
            _module_record(name, module)
            for name, module in modules.items()
            if name.endswith(("q_norm", "k_norm"))
        ],
    }
    one("rotary_position", "model.rotary_emb", "excluded")
    categories["activation_functions"] = {
        "policy": "excluded",
        "modules": [
            _module_record(name, module)
            for name, module in modules.items()
            if name.endswith("act_fn")
        ],
    }
    categories["kv_cache"] = {
        "policy": "runtime structure, not a model module",
        "modules": [],
        "present": False,
        "note": "past_key_values is created by the forward call when use_cache is enabled.",
    }
    return categories


def _smoke_forward(model: Any, tokenizer: Any, torch: Any, device: str) -> dict[str, Any]:
    prompt = "QAQ full-precision smoke test."
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        outputs = model(**inputs, use_cache=False)
    logits = outputs.logits.detach()
    digest = hashlib.sha256(logits.float().cpu().contiguous().numpy().tobytes()).hexdigest()
    return {
        "prompt": prompt,
        "input_shape": list(inputs["input_ids"].shape),
        "logits_shape": list(logits.shape),
        "logits_dtype": str(logits.dtype),
        "finite_values": bool(torch.isfinite(logits).all().item()),
        "logits_float32_sha256": digest,
        "selected_logits": [float(value) for value in logits[0, -1, :8].float().cpu()],
    }


def inspect_actual(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    values = _model_identity()
    structure = _read_structure()
    expected = _expected_targets(structure)
    model_path = Path(args.model_path).expanduser().resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"model snapshot directory does not exist: {model_path}")
    revision = values["revision"]
    if model_path.name != revision:
        raise ValueError(f"model snapshot basename must equal pinned revision {revision}")
    configured_dtype = values.get("torch_dtype", structure["identity"]["configured_dtype"])
    dtype = getattr(torch, configured_dtype)
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the intended Qwen3 loading path")

    before = _resource_snapshot(torch, device)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        revision=revision,
        local_files_only=True,
        trust_remote_code=values.get("trust_remote_code", "false").lower() == "true",
        torch_dtype=dtype,
    )
    model.eval()
    model.to(device)
    after_load = _resource_snapshot(torch, device)

    modules = dict(model.named_modules())
    actual_targets = {
        name: module for name, module in modules.items() if name in expected
    }
    target_records = [
        _target_record(name, actual_targets[name], expected[name])
        for name in sorted(actual_targets)
    ]
    missing = sorted(set(expected) - set(actual_targets))
    duplicates = sorted(name for name in actual_targets if name not in set(expected))
    mismatches: list[dict[str, Any]] = []
    for name, spec in expected.items():
        module = actual_targets.get(name)
        if module is None:
            continue
        if not isinstance(module, torch.nn.Linear):
            mismatches.append({"path": name, "issue": "unexpected_type", "actual": type(module).__name__})
            continue
        observed = {
            "input_features": int(module.in_features),
            "output_features": int(module.out_features),
            "bias": module.bias is not None,
            "shape": _tensor_shape(module),
        }
        wanted = {
            "input_features": spec["input_features"],
            "output_features": spec["output_features"],
            "bias": spec["bias"],
            "shape": [spec["output_features"], spec["input_features"]],
        }
        if observed != wanted:
            mismatches.append({"path": name, "expected": wanted, "actual": observed})

    unexpected_linears = []
    for name, module in modules.items():
        if isinstance(module, torch.nn.Linear) and name not in expected:
            classification = "excluded output head" if name == "lm_head" else "unexpected linear"
            unexpected_linears.append(
                {**_module_record(name, module), "classification": classification}
            )

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    parameter_dtypes = sorted({str(parameter.dtype) for parameter in model.parameters()})
    parameter_devices = sorted({str(parameter.device) for parameter in model.parameters()})
    embeddings = model.get_input_embeddings().weight
    output_head = model.get_output_embeddings().weight
    tied = {
        "config_tie_word_embeddings": bool(model.config.tie_word_embeddings),
        "same_parameter_object": embeddings is output_head,
        "same_storage_pointer": embeddings.data_ptr() == output_head.data_ptr(),
        "embedding_path": "model.embed_tokens.weight",
        "output_head_path": "lm_head.weight",
    }

    result: dict[str, Any] = {
        "exact_model_repository": values["repository"],
        "exact_model_revision": revision,
        "model_snapshot_path": str(model_path),
        "model_class": f"{type(model).__module__}.{type(model).__name__}",
        "dtype": str(dtype),
        "parameter_count": int(parameter_count),
        "parameter_dtypes": parameter_dtypes,
        "device_placement": parameter_devices,
        "target_modules": target_records,
        "attention_target_count": sum(item["unit_type"] == "attention" for item in target_records),
        "ffn_target_count": sum(item["unit_type"] == "FFN" for item in target_records),
        "total_target_count": len(target_records),
        "expected_target_count": len(expected),
        "target_paths_duplicate": len({item["full_module_path"] for item in target_records}) != len(target_records),
        "layer_indices": sorted({item["layer_index"] for item in target_records}),
        "excluded_module_categories": _exclusions(model, modules, torch),
        "unexpected_linear_modules": unexpected_linears,
        "tied_weight_verification": tied,
        "s00_mapping_comparison": {
            "missing_expected_targets": missing,
            "unexpected_target_paths": duplicates,
            "module_property_mismatches": mismatches,
            "all_expected_targets_present": not missing,
            "all_targets_supported_linear": not any(item.get("issue") == "unexpected_type" for item in mismatches),
            "dimensions_and_biases_match": not mismatches,
            "layer_indices_match": sorted({item["layer_index"] for item in target_records})
            == list(range(structure["global_dimensions"]["num_decoder_layers"])),
            "result": "MATCH" if not missing and not duplicates and not mismatches else "MISMATCH",
        },
        "resource_measurements": {"before_load": before, "after_load": after_load},
        "model_revision_argument": revision,
        "trust_remote_code": values.get("trust_remote_code", "false"),
        "quantization_performed": False,
    }
    if not args.skip_smoke:
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), revision=revision, local_files_only=True
        )
        result["full_precision_smoke_forward"] = _smoke_forward(model, tokenizer, torch, device)
    else:
        result["full_precision_smoke_forward"] = {"status": "SKIPPED"}
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("metadata", "actual"), default="metadata")
    parser.add_argument("--model-config", default=str(MODEL_CONFIG))
    parser.add_argument("--model-path", help="Exact local Hugging Face snapshot for --mode actual")
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.mode == "metadata":
        result = inspect_model.inspect(argparse.Namespace(
            model_config=args.model_config,
            config_json=None,
            source_file=None,
            output=None,
        ))
    else:
        if not args.model_path:
            parser.error("--model-path is required for --mode actual")
        result = inspect_actual(args)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    else:
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
