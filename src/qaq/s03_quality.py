"""Small deterministic S03-C quality evaluations for the static QAQ models."""

from __future__ import annotations

import gc
import hashlib
import math
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .s03_static import MODEL_REVISION, set_static_precision, tensor_sha256

REFERENCE_PROMPT = "QAQ full-precision smoke test."
PROMPT_MAX_LENGTH = 128
PERPLEXITY_SEQUENCE_LENGTH = 128
PERPLEXITY_SAMPLE_COUNT = 4
PERPLEXITY_DATASET = "Salesforce/wikitext"
PERPLEXITY_CONFIG = "wikitext-2-raw-v1"
PERPLEXITY_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
PERPLEXITY_SPLIT = "test"
GENERATION_MAX_NEW_TOKENS = 8


def read_prompt_file(path: str | os.PathLike[str]) -> list[str]:
    prompts = [line.rstrip("\n") for line in Path(path).read_text().splitlines()]
    prompts = [prompt for prompt in prompts if prompt.strip()]
    if not prompts:
        raise ValueError(f"prompt file is empty: {path}")
    return prompts


def load_full_precision_model(snapshot: str | os.PathLike[str], device: str):
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        str(snapshot),
        revision=MODEL_REVISION,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
    )
    model.to(device)
    model.eval()
    return model


def tokenize_prompt(tokenizer: Any, prompt: str, device: str) -> dict[str, Any]:
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=True,
        truncation=True,
        max_length=PROMPT_MAX_LENGTH,
        padding=False,
    )
    return {key: value.to(device) for key, value in encoded.items()}


def _logit_stats(logits: Any) -> dict[str, float]:
    values = logits.float()
    return {
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
        "minimum": float(values.min().item()),
        "maximum": float(values.max().item()),
    }


def run_logits(model: Any, inputs: dict[str, Any]) -> Any:
    with __import__("torch").inference_mode():
        return model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            use_cache=False,
        ).logits.detach()


def _memory_snapshot(torch: Any, device: str) -> dict[str, int]:
    return {
        "peak_allocated_gpu_bytes": int(torch.cuda.max_memory_allocated(torch.device(device))),
        "peak_reserved_gpu_bytes": int(torch.cuda.max_memory_reserved(torch.device(device))),
    }


def evaluate_reference(model: Any, tokenizer: Any, device: str) -> dict[str, Any]:
    import torch

    inputs = tokenize_prompt(tokenizer, REFERENCE_PROMPT, device)
    torch.cuda.reset_peak_memory_stats(torch.device(device))
    first = run_logits(model, inputs)
    second = run_logits(model, inputs)
    torch.cuda.synchronize(torch.device(device))
    if not bool(torch.isfinite(first).all().item()):
        raise ValueError("reference prompt produced non-finite logits")
    return {
        "prompt": REFERENCE_PROMPT,
        "token_count": int(inputs["input_ids"].shape[-1]),
        "logits_shape": list(first.shape),
        "finite_values": True,
        "deterministic_repeat": tensor_sha256(first.float()) == tensor_sha256(second.float()),
        "logits_sha256": tensor_sha256(first.float()),
        "selected_logits": [float(value) for value in first[0, -1, :8].float().cpu()],
        "logit_statistics": _logit_stats(first),
        **_memory_snapshot(torch, device),
    }


def _prompt_result(
    fp_logits: Any,
    quantized_logits: Any,
    quantized_repeat: Any,
    precision: int,
    inputs: dict[str, Any],
) -> dict[str, Any]:
    import torch

    if list(fp_logits.shape) != list(quantized_logits.shape):
        raise ValueError(f"shape mismatch for static {precision}-bit prompt")
    if not bool(torch.isfinite(quantized_logits).all().item()):
        raise ValueError(f"static {precision}-bit prompt produced non-finite logits")
    error = (fp_logits.float() - quantized_logits.float()).abs()
    return {
        "finite_values": True,
        "token_count": int(inputs["input_ids"].shape[-1]),
        "logits_shape": list(quantized_logits.shape),
        "logits_sha256": tensor_sha256(quantized_logits.float()),
        "deterministic_repeat": tensor_sha256(quantized_logits.float())
        == tensor_sha256(quantized_repeat.float()),
        "mean_absolute_logit_error": float(error.mean().item()),
        "maximum_absolute_logit_error": float(error.max().item()),
    }


def capture_full_precision_prompt_set(
    model: Any, tokenizer: Any, prompts: Iterable[str], device: str
) -> list[dict[str, Any]]:
    records = []
    for index, prompt in enumerate(prompts):
        inputs = tokenize_prompt(tokenizer, prompt, device)
        first = run_logits(model, inputs)
        repeat = run_logits(model, inputs)
        import torch

        if not bool(torch.isfinite(first).all().item()):
            raise ValueError(f"full-precision prompt {index} produced non-finite logits")
        records.append(
            {
                "index": index,
                "prompt": prompt,
                "full_precision": {
                    "finite_values": True,
                    "logits_shape": list(first.shape),
                    "logits_sha256": tensor_sha256(first.float()),
                    "deterministic_repeat": tensor_sha256(first.float())
                    == tensor_sha256(repeat.float()),
                },
                "_logits": first.float().cpu(),
            }
        )
    return records


def evaluate_static_prompt_set(
    model: Any,
    tokenizer: Any,
    full_precision_records: list[dict[str, Any]],
    device: str,
) -> dict[str, Any]:
    records = []
    for source in full_precision_records:
        inputs = tokenize_prompt(tokenizer, source["prompt"], device)
        fp_logits = source["_logits"]
        set_static_precision(model, 4)
        logits4 = run_logits(model, inputs)
        repeat4 = run_logits(model, inputs)
        set_static_precision(model, 8)
        logits8 = run_logits(model, inputs)
        repeat8 = run_logits(model, inputs)
        record = {
            "index": source["index"],
            "prompt": source["prompt"],
            "full_precision": source["full_precision"],
            "static_4": _prompt_result(
                fp_logits, logits4.float().cpu(), repeat4.float().cpu(), 4, inputs
            ),
            "static_8": _prompt_result(
                fp_logits, logits8.float().cpu(), repeat8.float().cpu(), 8, inputs
            ),
        }
        records.append(record)
    mean4 = sum(record["static_4"]["mean_absolute_logit_error"] for record in records) / len(
        records
    )
    mean8 = sum(record["static_8"]["mean_absolute_logit_error"] for record in records) / len(
        records
    )
    max4 = max(record["static_4"]["maximum_absolute_logit_error"] for record in records)
    max8 = max(record["static_8"]["maximum_absolute_logit_error"] for record in records)
    return {
        "count": len(records),
        "tokenization": {
            "add_special_tokens": True,
            "truncation": True,
            "max_length": PROMPT_MAX_LENGTH,
            "padding": False,
        },
        "comparison_metric": "mean absolute logit error and maximum absolute logit error",
        "aggregate": {
            "mean_of_prompt_mean_absolute_logit_error": {"4": mean4, "8": mean8},
            "maximum_prompt_maximum_absolute_logit_error": {"4": max4, "8": max8},
        },
        "fidelity_criterion": "aggregate mean 8-bit error must be <= aggregate mean 4-bit error",
        "records": records,
    }


def evaluate_prompt_set(
    fp_model: Any,
    static_model: Any,
    tokenizer: Any,
    prompts: Iterable[str],
    device: str,
) -> dict[str, Any]:
    import torch

    records: list[dict[str, Any]] = []
    prompts = list(prompts)
    for index, prompt in enumerate(prompts):
        inputs = tokenize_prompt(tokenizer, prompt, device)
        fp_logits = run_logits(fp_model, inputs)
        fp_repeat = run_logits(fp_model, inputs)
        set_static_precision(static_model, 4)
        logits4 = run_logits(static_model, inputs)
        repeat4 = run_logits(static_model, inputs)
        set_static_precision(static_model, 8)
        logits8 = run_logits(static_model, inputs)
        repeat8 = run_logits(static_model, inputs)
        if not bool(torch.isfinite(fp_logits).all().item()):
            raise ValueError(f"full-precision prompt {index} produced non-finite logits")
        records.append(
            {
                "index": index,
                "prompt": prompt,
                "full_precision": {
                    "finite_values": True,
                    "logits_shape": list(fp_logits.shape),
                    "logits_sha256": tensor_sha256(fp_logits.float()),
                    "deterministic_repeat": tensor_sha256(fp_logits.float())
                    == tensor_sha256(fp_repeat.float()),
                },
                "static_4": _prompt_result(fp_logits, logits4, repeat4, 4, inputs),
                "static_8": _prompt_result(fp_logits, logits8, repeat8, 8, inputs),
            }
        )
    mean4 = sum(record["static_4"]["mean_absolute_logit_error"] for record in records) / len(
        records
    )
    mean8 = sum(record["static_8"]["mean_absolute_logit_error"] for record in records) / len(
        records
    )
    max4 = max(record["static_4"]["maximum_absolute_logit_error"] for record in records)
    max8 = max(record["static_8"]["maximum_absolute_logit_error"] for record in records)
    return {
        "count": len(records),
        "tokenization": {
            "add_special_tokens": True,
            "truncation": True,
            "max_length": PROMPT_MAX_LENGTH,
            "padding": False,
        },
        "comparison_metric": "mean absolute logit error and maximum absolute logit error",
        "aggregate": {
            "mean_of_prompt_mean_absolute_logit_error": {"4": mean4, "8": mean8},
            "maximum_prompt_maximum_absolute_logit_error": {"4": max4, "8": max8},
        },
        "fidelity_criterion": "aggregate mean 8-bit error must be <= aggregate mean 4-bit error",
        "records": records,
    }


def select_token_windows(
    token_ids: list[int], *, sequence_length: int, sample_count: int, stride: int
) -> list[list[int]]:
    """Select deterministic source-order windows with non-overlapping targets."""

    if sequence_length <= 0 or sample_count <= 0 or stride <= 0:
        raise ValueError("sequence_length, sample_count, and stride must be positive")
    window_width = sequence_length + 1
    starts = [index * stride for index in range(sample_count)]
    required = starts[-1] + window_width
    if len(token_ids) < required:
        raise ValueError(f"token sequence yielded {len(token_ids)} tokens, need {required}")
    return [token_ids[start : start + window_width] for start in starts]


def _dataset_windows(
    tokenizer: Any,
    *,
    sample_count: int = PERPLEXITY_SAMPLE_COUNT,
    stride: int = PERPLEXITY_SEQUENCE_LENGTH + 1,
) -> tuple[list[Any], dict[str, Any]]:
    from datasets import load_dataset

    dataset = load_dataset(
        PERPLEXITY_DATASET,
        PERPLEXITY_CONFIG,
        split=PERPLEXITY_SPLIT,
        revision=PERPLEXITY_REVISION,
        trust_remote_code=False,
    )
    nonempty_text = []
    window_width = PERPLEXITY_SEQUENCE_LENGTH + 1
    required = (sample_count - 1) * stride + window_width
    token_ids = []
    for row in dataset:
        if row["text"].strip():
            nonempty_text.append(row["text"])
            token_ids = tokenizer("\n\n".join(nonempty_text), add_special_tokens=False)["input_ids"]
            if len(token_ids) >= required:
                break
    if len(token_ids) < required:
        raise ValueError(f"dataset yielded {len(token_ids)} tokens, need {required}")
    windows = select_token_windows(
        token_ids,
        sequence_length=PERPLEXITY_SEQUENCE_LENGTH,
        sample_count=sample_count,
        stride=stride,
    )
    metadata = {
        "dataset": PERPLEXITY_DATASET,
        "config": PERPLEXITY_CONFIG,
        "revision": PERPLEXITY_REVISION,
        "split": PERPLEXITY_SPLIT,
        "sample_count": sample_count,
        "sample_selection": (
            "concatenate non-empty test rows in source order and take the first "
            f"{sample_count} fixed windows with target stride {stride}"
        ),
        "sequence_length": PERPLEXITY_SEQUENCE_LENGTH,
        "window_width": window_width,
        "stride": stride,
        "window_start_offsets": [index * stride for index in range(sample_count)],
        "tokenizer_revision": MODEL_REVISION,
        "random_seed": None,
        "evaluated_token_count": sample_count * PERPLEXITY_SEQUENCE_LENGTH,
    }
    return [
        __import__("torch").tensor(window, dtype=__import__("torch").long) for window in windows
    ], metadata


def evaluate_perplexity(model: Any, windows: list[Any], device: str) -> dict[str, Any]:
    import torch

    total_nll = 0.0
    token_count = 0
    for window in windows:
        input_ids = window[:-1].unsqueeze(0).to(device)
        labels = window[1:].unsqueeze(0).to(device)
        with torch.inference_mode():
            logits = model(input_ids=input_ids, use_cache=False).logits.float()
        if not bool(torch.isfinite(logits).all().item()):
            raise ValueError("perplexity evaluation produced non-finite logits")
        loss_sum = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="sum"
        )
        total_nll += float(loss_sum.item())
        token_count += int(labels.numel())
    mean_nll = total_nll / token_count
    return {
        "mean_negative_log_likelihood": mean_nll,
        "perplexity": math.exp(mean_nll),
        "evaluated_token_count": token_count,
    }


def build_perplexity_windows(
    tokenizer: Any,
    *,
    sample_count: int = PERPLEXITY_SAMPLE_COUNT,
    stride: int = PERPLEXITY_SEQUENCE_LENGTH + 1,
) -> tuple[list[Any], dict[str, Any]]:
    """Build deterministic source-order windows using the S03 evaluator path."""

    return _dataset_windows(tokenizer, sample_count=sample_count, stride=stride)


def _sequence_digest(sequence: Any) -> str:
    return hashlib.sha256(sequence.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def generate_fixed(
    model: Any, tokenizer: Any, prompts: Iterable[str], precision: int | None, device: str
) -> dict[str, Any]:
    import torch

    if precision is not None:
        set_static_precision(model, precision)
    records = []
    for index, prompt in enumerate(prompts):
        inputs = tokenize_prompt(tokenizer, prompt, device)
        outputs = []
        for _ in range(2):
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=GENERATION_MAX_NEW_TOKENS,
                    use_cache=False,
                    return_dict_in_generate=True,
                    output_scores=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
            if generated.scores and not all(
                bool(torch.isfinite(score).all().item()) for score in generated.scores
            ):
                raise ValueError(f"generation prompt {index} produced non-finite scores")
            sequence = generated.sequences[0]
            outputs.append(
                {
                    "sequence_sha256": _sequence_digest(sequence),
                    "text": tokenizer.decode(sequence, skip_special_tokens=True),
                    "generated_token_count": max(
                        0, int(sequence.numel() - inputs["input_ids"].shape[-1])
                    ),
                    "stopped_by_eos": bool(
                        tokenizer.eos_token_id is not None
                        and tokenizer.eos_token_id
                        in sequence[inputs["input_ids"].shape[-1] :].tolist()
                    ),
                }
            )
        records.append(
            {
                "index": index,
                "prompt": prompt,
                "deterministic_repeat": outputs[0]["sequence_sha256"]
                == outputs[1]["sequence_sha256"],
                "first": outputs[0],
                "repeat": outputs[1],
            }
        )
    return {
        "max_new_tokens": GENERATION_MAX_NEW_TOKENS,
        "decoding": "greedy (do_sample=false, num_beams=1)",
        "batch_size": 1,
        "records": records,
    }


def unload_model(model: Any) -> None:
    model.cpu()
    del model
    gc.collect()
    import torch

    torch.cuda.empty_cache()
