#!/usr/bin/env python3
"""Inspect the pinned Qwen3 configuration and source definitions without loading weights."""

from __future__ import annotations

import argparse
import ast
import importlib.metadata
import json
import re
import sys
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
MODEL_CONFIG = ROOT / "configs" / "model.yaml"
TRANSFORMERS_SOURCE_COMMIT = "0720e206c6ba28887e4d60ef60a6a089f6c1cc76"
TRANSFORMERS_SOURCE_VERSION = "4.51.0"


def _simple_yaml(path: Path) -> dict[str, str]:
    """Read the scalar identity fields used by configs/model.yaml without extra dependencies."""
    values: dict[str, str] = {}
    section = ""
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.endswith(":"):
            section = line[:-1]
            continue
        if ":" not in line:
            continue
        key, value = (part.strip() for part in line.split(":", 1))
        value = value.strip().strip('"').strip("'")
        if section == "tokenizer":
            values[f"tokenizer_{key}"] = value
        else:
            values[key] = value
    return values


def _fetch(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "qaq-s00-inspection/1"})
    with urlopen(request, timeout=30) as response:
        return response.read()


def _load_json(args: argparse.Namespace, repository: str, revision: str) -> tuple[dict, str]:
    if args.config_json:
        path = Path(args.config_json)
        return json.loads(path.read_text()), str(path)
    url = f"https://huggingface.co/{repository}/resolve/{revision}/config.json"
    return json.loads(_fetch(url)), url


def _load_source(args: argparse.Namespace, expected_version: str) -> tuple[str, str]:
    if args.source_file:
        path = Path(args.source_file)
        return path.read_text(), str(path)
    if expected_version != TRANSFORMERS_SOURCE_VERSION:
        raise ValueError(
            f"No pinned source reference for Transformers {expected_version}; provide --source-file"
        )
    url = (
        "https://raw.githubusercontent.com/huggingface/transformers/"
        f"{TRANSFORMERS_SOURCE_COMMIT}/src/transformers/models/qwen3/modeling_qwen3.py"
    )
    return _fetch(url).decode("utf-8"), url


def _installed_transformers() -> dict[str, object]:
    try:
        distribution = importlib.metadata.distribution("transformers")
        root = Path(distribution.locate_file("transformers"))
        version = distribution.version
    except importlib.metadata.PackageNotFoundError:
        return {"version": None, "qwen3_source_present": False, "source_root": None}
    return {
        "version": version,
        "qwen3_source_present": (root / "models" / "qwen3").is_dir(),
        "source_root": str(root),
    }


def _class_names(source: str) -> set[str]:
    tree = ast.parse(source)
    return {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}


def _make_structure(
    config: dict,
    config_source: str,
    source_source: str,
    source: str,
    configuration_source: str,
) -> dict:
    required_config = {
        "architectures",
        "model_type",
        "torch_dtype",
        "num_hidden_layers",
        "hidden_size",
        "intermediate_size",
        "vocab_size",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "max_position_embeddings",
        "attention_bias",
        "tie_word_embeddings",
        "hidden_act",
        "use_cache",
    }
    missing = sorted(required_config - config.keys())
    if missing:
        raise ValueError(f"pinned config is missing required fields: {missing}")

    layers = int(config["num_hidden_layers"])
    hidden = int(config["hidden_size"])
    intermediate = int(config["intermediate_size"])
    heads = int(config["num_attention_heads"])
    kv_heads = int(config["num_key_value_heads"])
    head_dim = int(config["head_dim"])
    vocab = int(config["vocab_size"])

    attention = {
        "q_proj": {"path": "self_attn.q_proj", "input_dim": hidden, "output_dim": heads * head_dim, "bias": bool(config["attention_bias"]), "quantization_target": True},
        "k_proj": {"path": "self_attn.k_proj", "input_dim": hidden, "output_dim": kv_heads * head_dim, "bias": bool(config["attention_bias"]), "quantization_target": True},
        "v_proj": {"path": "self_attn.v_proj", "input_dim": hidden, "output_dim": kv_heads * head_dim, "bias": bool(config["attention_bias"]), "quantization_target": True},
        "o_proj": {"path": "self_attn.o_proj", "input_dim": heads * head_dim, "output_dim": hidden, "bias": bool(config["attention_bias"]), "quantization_target": True},
    }
    ffn = {
        "gate_proj": {"path": "mlp.gate_proj", "input_dim": hidden, "output_dim": intermediate, "bias": False, "quantization_target": True},
        "up_proj": {"path": "mlp.up_proj", "input_dim": hidden, "output_dim": intermediate, "bias": False, "quantization_target": True},
        "down_proj": {"path": "mlp.down_proj", "input_dim": intermediate, "output_dim": hidden, "bias": False, "quantization_target": True},
    }
    target_modules = [
        f'model.layers.{layer}.{projection["path"]}'
        for layer in range(layers)
        for projection in (*attention.values(), *ffn.values())
    ]

    source_classes = sorted(_class_names(source))
    source_markers = {
        "causal_lm_constructs_base_model": "self.model = Qwen3Model(config)" in source,
        "base_model_constructs_layers": "self.layers = nn.ModuleList" in source,
        "base_model_constructs_embeddings": "self.embed_tokens = nn.Embedding" in source,
        "base_model_constructs_final_norm": "self.norm = Qwen3RMSNorm" in source,
        "base_model_constructs_rotary": "self.rotary_emb = Qwen3RotaryEmbedding" in source,
        "attention_has_qk_norm": "self.q_norm = Qwen3RMSNorm" in source and "self.k_norm = Qwen3RMSNorm" in source,
        "causal_lm_declares_tied_head": "_tied_weights_keys = [\"lm_head.weight\"]" in source,
        "cache_updates_past_key_value": "past_key_value.update" in source,
    }
    expected_modeling_classes = [
        "Qwen3RMSNorm", "Qwen3MLP", "Qwen3Attention", "Qwen3DecoderLayer",
        "Qwen3RotaryEmbedding", "Qwen3Model", "Qwen3PreTrainedModel", "Qwen3ForCausalLM",
    ]
    source_markers["required_modeling_classes_present"] = all(
        name in source_classes for name in expected_modeling_classes
    )
    source_markers["configuration_class_present"] = "Qwen3Config" in _class_names(configuration_source)

    module_prefix = "model.layers.<i>"
    return {
        "identity": {
            "repository": "Qwen/Qwen3-4B",
            "revision": "1cfa9a7208912126459214e8b04321603b3df60c",
            "config_source": config_source,
            "architecture_class": config["architectures"][0],
            "model_type": config["model_type"],
            "configured_dtype": config["torch_dtype"],
            "config_transformers_version": config.get("transformers_version"),
        },
        "global_dimensions": {
            "num_decoder_layers": layers,
            "hidden_size": hidden,
            "intermediate_ffn_size": intermediate,
            "vocab_size": vocab,
            "num_attention_heads": heads,
            "num_key_value_heads": kv_heads,
            "attention_head_dimension": head_dim,
            "max_position_embeddings": int(config["max_position_embeddings"]),
        },
        "class_hierarchy": {
            "causal_lm_wrapper": "Qwen3ForCausalLM",
            "base_model": "Qwen3Model",
            "decoder_layer": "Qwen3DecoderLayer",
            "attention": "Qwen3Attention",
            "ffn": "Qwen3MLP",
            "normalization": "Qwen3RMSNorm",
            "token_embedding": "Qwen3Model.embed_tokens",
            "output_head": "Qwen3ForCausalLM.lm_head",
            "configuration_source": f"https://raw.githubusercontent.com/huggingface/transformers/{TRANSFORMERS_SOURCE_COMMIT}/src/transformers/models/qwen3/configuration_qwen3.py",
            "source": source_source,
        },
        "paths": {
            "model": "model",
            "decoder_layers": "model.layers",
            "attention_unit": f"{module_prefix}.self_attn",
            "ffn_unit": f"{module_prefix}.mlp",
        },
        "attention": {
            "projections": attention,
            "operations_outside_packed_linear": [
                {"path": f"{module_prefix}.input_layernorm", "type": "Qwen3RMSNorm", "quantization_target": False},
                {"path": f"{module_prefix}.self_attn.q_norm", "type": "Qwen3RMSNorm", "shape": [head_dim], "quantization_target": False},
                {"path": f"{module_prefix}.self_attn.k_norm", "type": "Qwen3RMSNorm", "shape": [head_dim], "quantization_target": False},
                {"path": "model.rotary_emb", "type": "Qwen3RotaryEmbedding", "quantization_target": False},
                {"path": f"{module_prefix}.post_attention_layernorm", "type": "Qwen3RMSNorm", "quantization_target": False},
            ],
            "qk_normalization_order": "q_proj/k_proj -> reshape -> q_norm/k_norm -> transpose -> rotary position embedding",
            "rotary_position_processing": "Qwen3RotaryEmbedding produces cos/sin; apply_rotary_pos_emb transforms query and key only",
        },
        "ffn": {
            "projections": ffn,
            "activation": config["hidden_act"],
            "gating": "down_proj(silu(gate_proj(x)) * up_proj(x))",
        },
        "non_target_components": [
            {"path": "model.embed_tokens", "type": "nn.Embedding", "shape": [vocab, hidden], "policy": "retained BF16/FP16; not a packed linear target"},
            {"path": "model.norm", "type": "Qwen3RMSNorm", "shape": [hidden], "policy": "retained BF16/FP16; not a packed linear target"},
            {"path": f"{module_prefix}.input_layernorm", "type": "Qwen3RMSNorm", "shape": [hidden], "policy": "retained BF16/FP16; not a packed linear target"},
            {"path": f"{module_prefix}.post_attention_layernorm", "type": "Qwen3RMSNorm", "shape": [hidden], "policy": "retained BF16/FP16; not a packed linear target"},
            {"path": f"{module_prefix}.self_attn.q_norm", "type": "Qwen3RMSNorm", "shape": [head_dim], "policy": "retained BF16/FP16; not a packed linear target"},
            {"path": f"{module_prefix}.self_attn.k_norm", "type": "Qwen3RMSNorm", "shape": [head_dim], "policy": "retained BF16/FP16; not a packed linear target"},
            {"path": "model.rotary_emb", "type": "Qwen3RotaryEmbedding", "policy": "not a weight quantization target"},
            {"path": "lm_head", "type": "nn.Linear", "shape": [hidden, vocab], "bias": False, "policy": "retained BF16/FP16; excluded from packed target list"},
            {"path": "past_key_values", "type": "DynamicCache when use_cache=true", "policy": "runtime KV cache; not a packed weight target"},
        ],
        "tied_weights": {
            "config_tie_word_embeddings": bool(config["tie_word_embeddings"]),
            "implementation_marker": "Qwen3ForCausalLM._tied_weights_keys = [\"lm_head.weight\"]",
            "relationship": "lm_head.weight is tied to model.embed_tokens.weight by Transformers tie_weights handling",
        },
        "quantization_targets": {
            "per_layer_projection_paths": [item["path"] for item in (*attention.values(), *ffn.values())],
            "module_names": target_modules,
            "attention_projection_count": len(attention) * layers,
            "ffn_projection_count": len(ffn) * layers,
            "total_count": len(target_modules),
            "count_formula": f"{layers} layers * (4 attention + 3 FFN) = {len(target_modules)}",
            "no_duplicates": len(target_modules) == len(set(target_modules)),
        },
        "any_precision_analysis": {
            "pinned_revision": "a3257d02740cc5757c78673da534b0630ff3a4ea",
            "explicit_architecture_yaml_classes": ["LlamaForCausalLM", "MistralForCausalLM", "OPTForCausalLM", "PhiForCausalLM"],
            "qwen3_explicitly_supported": False,
            "discovery_behavior": "No Qwen3 YAML exists; analyzer falls back to AutoArchConfig, which scans the first decoder layer for torch.nn.Linear modules and warns that automatic detection may be incorrect.",
            "structural_linear_fit": {
                "all_seven_are_standard_linear": True,
                "all_input_dimensions_divisible_by_32": True,
                "bias_supported_and_all_false": True,
                "explicit_mapping_unambiguous": True,
            },
            "limitation": "Structural fit is not official Qwen3 support. Later work must provide an explicit Qwen3 mapping and execute a backend validation under a Transformers version that contains Qwen3.",
        },
        "source_markers": source_markers,
        "runtime_environment": _installed_transformers(),
        "weight_policy": {"weights_loaded": False, "weight_shards_downloaded": False, "full_model_instantiated": False},
    }


def inspect(args: argparse.Namespace) -> dict:
    model_values = _simple_yaml(Path(args.model_config))
    repository = model_values.get("repository")
    revision = model_values.get("revision")
    if repository != "Qwen/Qwen3-4B" or not re.fullmatch(r"[0-9a-f]{40}", revision or ""):
        raise ValueError("configs/model.yaml must contain the locked Qwen/Qwen3-4B full revision")
    config, config_source = _load_json(args, repository, revision)
    source, source_source = _load_source(args, str(config.get("transformers_version", "")))
    configuration_url = (
        "https://raw.githubusercontent.com/huggingface/transformers/"
        f"{TRANSFORMERS_SOURCE_COMMIT}/src/transformers/models/qwen3/configuration_qwen3.py"
    )
    configuration_source = _fetch(configuration_url).decode("utf-8")
    structure = _make_structure(config, config_source, source_source, source, configuration_source)
    structure["model_config_source"] = str(args.model_config)
    return structure


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", default=str(MODEL_CONFIG))
    parser.add_argument("--config-json", help="Use a previously retrieved pinned config JSON")
    parser.add_argument("--source-file", help="Use a previously retrieved Qwen3 Transformers source file")
    parser.add_argument("--output", type=Path, help="Write JSON to this path")
    args = parser.parse_args()
    result = inspect(args)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    else:
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
