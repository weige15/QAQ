#!/usr/bin/env python3
"""Run the deliberately small S03-C static-quality evaluation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qaq.evaluation.quality import (
    REFERENCE_PROMPT,
    build_perplexity_windows,
    capture_full_precision_prompt_set,
    evaluate_perplexity,
    evaluate_reference,
    evaluate_static_prompt_set,
    generate_fixed,
    load_full_precision_model,
    read_prompt_file,
    unload_model,
)
from qaq.model.static import (
    MODEL_REVISION,
    PINNED_ANY_PRECISION_COMMIT,
    assert_target_invariant,
    load_static_model,
    source_commit,
)

SNAPSHOT = Path(
    os.environ.get(
        "QAQ_MODEL_SNAPSHOT",
        "~/.cache/huggingface/hub/models--Qwen--Qwen3-4B/"
        "snapshots/1cfa9a7208912126459214e8b04321603b3df60c",
    )
).expanduser()
ARTIFACT = ROOT / "quantized/s03b_qwen3_4b/backend_cache/packed/anyprec-(1cfa9a7208912126459214e8b04321603b3df60c)-w8_orig4-gc1-c4_s1_blk64"
PROMPTS = ROOT / "configs/s03_static_quality_prompts.txt"
GENERATION_PROMPTS = ROOT / "configs/s03_static_generation_prompts.txt"
MANIFEST = ROOT / "docs/quantized_model_manifest.json"


def _resources(torch, device: str) -> dict[str, int]:
    return {
        "peak_allocated_gpu_bytes": int(torch.cuda.max_memory_allocated(torch.device(device))),
        "peak_reserved_gpu_bytes": int(torch.cuda.max_memory_reserved(torch.device(device))),
    }


def _check_environment() -> None:
    if not os.environ.get("VIRTUAL_ENV", "").startswith(str(Path.home() / ".venv")):
        raise SystemExit("PAUSE: ~/.venv is not active")
    if not SNAPSHOT.is_dir() or SNAPSHOT.name != MODEL_REVISION:
        raise SystemExit(f"PAUSE: exact pinned model snapshot is unavailable: {SNAPSHOT}")
    if not ARTIFACT.is_dir():
        raise SystemExit(f"PAUSE: S03-B artifact is unavailable: {ARTIFACT}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=os.environ.get("QAQ_MODEL_DEVICE", "cuda:3"))
    parser.add_argument("--output", type=Path, default=ROOT / "docs/results/s03_static_quality.json")
    args = parser.parse_args()
    _check_environment()

    import torch
    from transformers import AutoTokenizer

    if not torch.cuda.is_available():
        raise SystemExit("PAUSE: CUDA is unavailable")
    torch.cuda.set_device(torch.device(args.device))
    source_commit()
    target_names = assert_target_invariant()
    manifest = json.loads(MANIFEST.read_text())
    if manifest["source_model"]["revision"] != MODEL_REVISION:
        raise SystemExit("REVISE: source model revision differs from the S03-B manifest")
    if manifest["any_precision"]["commit"] != PINNED_ANY_PRECISION_COMMIT:
        raise SystemExit("REVISE: Any-Precision revision differs from the S03-B manifest")
    if manifest["artifact"]["target_count"] != len(target_names):
        raise SystemExit("REVISE: packed checkpoint target count changed")

    tokenizer = AutoTokenizer.from_pretrained(str(SNAPSHOT), revision=MODEL_REVISION, local_files_only=True)
    prompt_set = read_prompt_file(PROMPTS)
    generation_prompts = read_prompt_file(GENERATION_PROMPTS)

    print("S03-C: loading full-precision teacher", flush=True)
    torch.cuda.reset_peak_memory_stats(torch.device(args.device))
    fp_model = load_full_precision_model(SNAPSHOT, args.device)
    fp_reference = evaluate_reference(fp_model, tokenizer, args.device)
    fp_windows, perplexity_setup = build_perplexity_windows(tokenizer)
    fp_perplexity = evaluate_perplexity(fp_model, fp_windows, args.device)
    fp_generation = generate_fixed(fp_model, tokenizer, generation_prompts, None, args.device)
    fp_prompt_records = capture_full_precision_prompt_set(
        fp_model, tokenizer, prompt_set, args.device
    )
    fp_memory = _resources(torch, args.device)
    unload_model(fp_model)

    print("S03-C: loading static nested checkpoint", flush=True)
    static_model = load_static_model(ARTIFACT, args.device)
    static_reference = {}
    static_perplexity = {}
    static_generation = {}
    static_memory = {}
    for precision in (4, 8):
        from qaq.model.static import set_static_precision

        torch.cuda.reset_peak_memory_stats(torch.device(args.device))
        set_static_precision(static_model, precision)
        static_reference[str(precision)] = evaluate_reference(static_model, tokenizer, args.device)
        static_perplexity[str(precision)] = evaluate_perplexity(static_model, fp_windows, args.device)
        static_generation[str(precision)] = generate_fixed(
            static_model, tokenizer, generation_prompts, precision, args.device
        )
        static_memory[str(precision)] = _resources(torch, args.device)
    prompt_set_result = evaluate_static_prompt_set(
        static_model, tokenizer, fp_prompt_records, args.device
    )

    static_model.cpu()
    del static_model

    result = {
        "format": "qaq-s03c-static-quality-v1",
        "scope": "S03-C only; no routing, training, or on-demand loading",
        "source_model": {
            "repository": manifest["source_model"]["repository"],
            "revision": MODEL_REVISION,
            "tokenizer_revision": MODEL_REVISION,
            "any_precision_commit": PINNED_ANY_PRECISION_COMMIT,
            "quantized_checkpoint": manifest["artifact"]["local_path"],
            "quantized_checkpoint_sha256": manifest["artifact"]["checkpoint_hashes"]["pytorch_model.bin"],
        },
        "target_invariant": {
            "target_count": len(target_names),
            "omitted_targets": manifest["target_invariant"]["omitted_targets"],
            "unexpected_targets": manifest["target_invariant"]["unexpected_targets"],
            "duplicate_precision_models": manifest["artifact"]["nested_representation"][
                "separate_4bit_qweight_copy"
            ]
            or manifest["artifact"]["nested_representation"]["separate_8bit_model_copy"],
        },
        "reference_prompt": {
            "text": REFERENCE_PROMPT,
            "full_precision": fp_reference,
            "static_4": static_reference["4"],
            "static_8": static_reference["8"],
        },
        "prompt_set": {
            "file": str(PROMPTS.relative_to(ROOT)),
            "prompts": prompt_set,
            **prompt_set_result,
        },
        "perplexity": {
            **perplexity_setup,
            "implementation": (
                "same causal cross-entropy implementation for every mode; input is the first "
                "128 tokens of each 129-token window and labels are the next 128 tokens; "
                "no padding and no generated tokens"
            ),
            "full_precision": fp_perplexity,
            "static_4": static_perplexity["4"],
            "static_8": static_perplexity["8"],
            "quality_criterion": "static 8-bit perplexity must be <= static 4-bit perplexity * 1.10",
        },
        "generation": {
            "file": str(GENERATION_PROMPTS.relative_to(ROOT)),
            "prompts": generation_prompts,
            "full_precision": fp_generation,
            "static_4": static_generation["4"],
            "static_8": static_generation["8"],
        },
        "memory_observations": {
            "note": "Development-only peak measurements; no transfer-savings claim.",
            "full_precision": fp_memory,
            "static_4": static_memory["4"],
            "static_8": static_memory["8"],
        },
        "checkpoint_reload": {
            "command": (
                "source ~/.venv/bin/activate && which python && python --version && "
                "QAQ_S03_ARTIFACT=<artifact> pytest -q tests/integration/test_checkpoint_roundtrip.py"
            ),
            "status": "pending separate fresh-process command",
        },
        "commands": {
            "quality": (
                "source ~/.venv/bin/activate && which python && python --version && "
                "python scripts/run_s03c.py"
            ),
            "artifact": manifest["quantization"]["exact_command"],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
