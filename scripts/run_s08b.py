#!/usr/bin/env python3
"""Run the bounded S08-B real Qwen3 hard-route comparison and measurements."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

MODEL_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
ANY_PRECISION_REVISION = "a3257d02740cc5757c78673da534b0630ff3a4ea"
ROUTER_CHECKPOINT = Path("~/.cache/qaq/s07b/final_router.pt").expanduser()
DEVICE = os.environ.get("QAQ_MODEL_DEVICE", "cuda:3")
ARTIFACT = Path(
    os.environ.get(
        "QAQ_S03_ARTIFACT",
        ROOT
        / "quantized/s03b_qwen3_4b/backend_cache/packed/"
        "anyprec-(1cfa9a7208912126459214e8b04321603b3df60c)-w8_orig4-gc1-c4_s1_blk64",
    )
).resolve()
CODE_PATHS = (
    Path("src/qaq/s04_manual.py"),
    Path("src/qaq/s08_loader.py"),
    Path("scripts/run_s08b.py"),
    Path("tests/integration/test_s08_real_hard_routed.py"),
)


def _code_provenance() -> dict[str, Any]:
    status = subprocess.check_output(
        ["git", "-C", str(ROOT), "status", "--porcelain=v1"], text=True
    )
    return {
        "git_head": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "worktree_status": status.splitlines(),
        "relevant_file_sha256": {
            str(path): _file_sha256(ROOT / path) for path in CODE_PATHS
        },
    }


def _require_environment() -> None:
    virtual_env = os.environ.get("VIRTUAL_ENV", "")
    expected = str(Path.home() / ".venv")
    if not virtual_env.startswith(expected):
        raise SystemExit("PAUSE: ~/.venv is not active")
    if not ARTIFACT.is_dir():
        raise SystemExit(f"PAUSE: packed artifact is unavailable: {ARTIFACT}")
    if not ROUTER_CHECKPOINT.is_file():
        raise SystemExit(f"PAUSE: router checkpoint is unavailable: {ROUTER_CHECKPOINT}")
    if not torch.cuda.is_available():
        raise SystemExit("PAUSE: CUDA is unavailable")


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()


def _load_examples() -> list[Any]:
    from datasets import load_dataset
    from run_s07b import _load_config, _select_examples
    from transformers import AutoTokenizer

    config = _load_config()
    tokenizer = AutoTokenizer.from_pretrained(str(ARTIFACT), local_files_only=True)
    dataset = load_dataset(
        config["dataset"]["repository"],
        config["dataset"]["config"],
        split=config["dataset"]["validation_split"],
        revision=config["dataset"]["revision"],
    )
    examples, manifest = _select_examples(
        dataset,
        tokenizer,
        config["dataset"]["validation_offsets"],
        split="validation",
        config=config,
        torch=torch,
    )
    recorded = json.loads((ROOT / "docs/results/s07_router_training.json").read_text())
    expected = {item["example_id"]: item for item in recorded["dataset_manifest"]["validation"]}
    for item in manifest:
        if item["example_id"] not in expected:
            raise RuntimeError(f"REVISE: validation request is not in the S07 record: {item['example_id']}")
        if item["input_ids_sha256"] != expected[item["example_id"]]["input_ids_sha256"]:
            raise RuntimeError(f"REVISE: token digest changed for {item['example_id']}")
    return examples


def _checkpoint_metadata(manifest: dict[str, Any], router: Any) -> Any:
    from qaq.s07_distillation import RouterCheckpointMetadata

    return RouterCheckpointMetadata(
        model_repository="Qwen/Qwen3-4B",
        model_revision=MODEL_REVISION,
        quantized_checkpoint_id=manifest["artifact"]["local_path"],
        quantized_checkpoint_hash=f"sha256:{manifest['artifact']['checkpoint_hashes']['pytorch_model.bin']}",
        any_precision_revision=ANY_PRECISION_REVISION,
        router_architecture={
            "feature_dim": int(router.feature_dim),
            "hidden_width": 128,
            "activation": "GELU",
            "normalization": "parameter-free RMS",
            "normalization_epsilon": 1e-6,
            "temperature": 1.0,
            "router_count": int(router.router_count),
        },
        candidate_ordering=(4, 8),
        training_step=4,
        training_step_metadata={"seed": 1729, "format": "qaq-s07b-router-training-v1"},
    )


def _load_student(mode: str) -> Any:
    from qaq.s04_manual import load_on_demand_model
    from qaq.s06_soft import SoftRoutedQwen3ForCausalLM, load_soft_model
    from qaq.s07_distillation import load_router_checkpoint

    manifest = json.loads((ROOT / "docs/quantized_model_manifest.json").read_text())
    if mode == "resident":
        student = load_soft_model(str(ARTIFACT), DEVICE)
    elif mode == "on_demand":
        student = SoftRoutedQwen3ForCausalLM(load_on_demand_model(str(ARTIFACT), DEVICE))
    else:
        raise ValueError(mode)
    metadata = _checkpoint_metadata(manifest, student)
    load_router_checkpoint(ROUTER_CHECKPOINT, student.routers, metadata)
    student.to(DEVICE)
    student.eval()
    return student


def _policy(student: Any):
    from qaq.s07_distillation import hard_route

    def choose(layer: int, unit_type: str, feature: torch.Tensor) -> int:
        return int(hard_route(student.route(layer, unit_type, feature)))

    return choose


def _context_for(student: Any, example: Any, request_id: str):
    from qaq.model.request_state import QaqRequestState

    state = QaqRequestState(
        request_id,
        int(example.prompt_mask().sum().item()),
        layer_count=36,
    )
    context = None
    if any(module.__class__.__name__ == "_OnDemandRoutedPackedLinear" for module in student.base.modules()):
        context = student.base.create_on_demand_request(state)
    return state, context


def _hard_forward(student: Any, example: Any, *, request_id: str, input_ids: torch.Tensor, use_cache: bool):
    from qaq.s04_manual import PrecisionTrace

    state, context = _context_for(student, example, request_id)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    prompt_mask = example.prompt_mask().unsqueeze(0).to(input_ids.device)
    trace = PrecisionTrace()
    output = student.base(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=use_cache,
        request_state=state,
        phase="prefill",
        routing_policy=_policy(student),
        prompt_attention_mask=prompt_mask,
        on_demand_context=context,
        trace=trace,
    )
    return output, state, context, trace


def _route_map(state: Any) -> list[dict[str, Any]]:
    return [
        *[
            {"layer": layer, "unit_type": "attention", "selected_bits": bits}
            for layer, bits in enumerate(state.attention_routes)
        ],
        *[
            {"layer": layer, "unit_type": "ffn", "selected_bits": bits}
            for layer, bits in enumerate(state.ffn_routes)
        ],
    ]


def _route_digest(route_map: list[dict[str, Any]]) -> str:
    payload = json.dumps(route_map, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _expected_transfers(context: Any, route_map: list[dict[str, Any]]) -> dict[str, Any]:
    by_unit = {(item["layer"], item["unit_type"]): item["selected_bits"] for item in route_map}
    expected_by_projection: dict[str, int] = {}
    for module_id, source in context.sources.items():
        layer = int(module_id.split(".")[2])
        unit_type = "attention" if ".self_attn." in module_id else "ffn"
        bits = by_unit[(layer, unit_type)]
        lut = source.lut4 if bits == 4 else source.lut8
        expected_by_projection[module_id] = int(
            source.qweight[:bits].numel() * source.qweight.element_size()
            + lut.numel() * lut.element_size()
        )
    return {
        "total_bytes": sum(expected_by_projection.values()),
        "by_projection": expected_by_projection,
        "by_layer": {
            str(layer): sum(
                value for module_id, value in expected_by_projection.items() if int(module_id.split(".")[2]) == layer
            )
            for layer in range(36)
        },
        "by_unit": {
            unit: sum(
                value for module_id, value in expected_by_projection.items()
                if (".self_attn." in module_id) == (unit == "attention")
            )
            for unit in ("attention", "ffn")
        },
    }


def _transfer_summary(context: Any, route_map: list[dict[str, Any]], *, prefill_count: int) -> dict[str, Any]:
    records = context.records
    expected = _expected_transfers(context, route_map)
    actual_by_projection: defaultdict[str, int] = defaultdict(int)
    for record in records:
        actual_by_projection[record.module_id] += record.transferred_bytes
    actual_total = sum(record.transferred_bytes for record in records)
    prefill_records = records[:prefill_count]
    decode_records = records[prefill_count:]
    return {
        "total_cpu_to_gpu_bytes": actual_total,
        "prefill_bytes": sum(record.transferred_bytes for record in prefill_records),
        "decode_bytes": sum(record.transferred_bytes for record in decode_records),
        "first_use_bytes": sum(record.transferred_bytes for record in records if record.event == "first_use"),
        "upgrade_bytes": sum(
            record.transferred_bytes
            for record in records
            if any("qweight[4:8]" == item["name"] for item in record.buffers)
        ),
        "reuse_bytes": sum(record.transferred_bytes for record in records if record.event == "reuse"),
        "first_use_events": sum(record.event == "first_use" for record in records),
        "reuse_events": sum(record.event == "reuse" for record in records),
        "by_layer": {
            str(layer): sum(value for module_id, value in actual_by_projection.items() if int(module_id.split(".")[2]) == layer)
            for layer in range(36)
        },
        "by_unit": {
            unit: sum(
                value for module_id, value in actual_by_projection.items()
                if (".self_attn." in module_id) == (unit == "attention")
            )
            for unit in ("attention", "ffn")
        },
        "by_projection": dict(sorted(actual_by_projection.items())),
        "expected_total_bytes": expected["total_bytes"],
        "expected_by_layer": expected["by_layer"],
        "expected_by_unit": expected["by_unit"],
        "expected_by_projection": dict(sorted(expected["by_projection"].items())),
        "physical_accounting_matches": actual_total == expected["total_bytes"]
        and dict(actual_by_projection) == expected["by_projection"],
        "records": [
            {
                "module_id": record.module_id,
                "precision": record.precision,
                "event": record.event,
                "transferred_bytes": record.transferred_bytes,
                "buffers": list(record.buffers),
            }
            for record in records
        ],
    }


def _generation(student: Any, example: Any, *, request_id: str, steps: int = 4) -> dict[str, Any]:
    prompt = example.input_ids[:32].unsqueeze(0).to(DEVICE)
    state, context = _context_for(student, example, request_id)
    from qaq.s04_manual import PrecisionTrace

    prefill_trace = PrecisionTrace()
    output = student.base(
        input_ids=prompt,
        attention_mask=torch.ones_like(prompt, dtype=torch.bool),
        use_cache=True,
        request_state=state,
        phase="prefill",
        routing_policy=_policy(student),
        prompt_attention_mask=torch.ones_like(prompt, dtype=torch.bool),
        on_demand_context=context,
        trace=prefill_trace,
    )
    route_map = _route_map(state)
    generated: list[int] = []
    decode_trace = PrecisionTrace()
    past = output.past_key_values
    for _ in range(steps):
        token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(int(token.item()))
        output = student.base(
            input_ids=token,
            attention_mask=torch.ones_like(token, dtype=torch.bool),
            past_key_values=past,
            use_cache=True,
            request_state=state,
            phase="decode",
            trace=decode_trace,
            on_demand_context=context,
        )
        past = output.past_key_values
    torch.cuda.synchronize(DEVICE)
    decode_bytes = 0 if context is None else sum(record.transferred_bytes for record in context.records)
    unchanged = route_map == _route_map(state)
    result = {
        "token_ids": generated,
        "all_finite": bool(torch.isfinite(output.logits).all().item()),
        "route_map": route_map,
        "route_map_digest": _route_digest(route_map),
        "route_map_unchanged_during_decode": unchanged,
        "decode_transfer_bytes": decode_bytes
        if context is None
        else sum(record.transferred_bytes for record in context.records[len(prefill_trace.records) :]),
        "decode_transfer_records": 0 if context is None else len(context.records) - len(prefill_trace.records),
        "decode_route_records": len(decode_trace.route_records),
    }
    state.end_request()
    return result


def _correctness(examples: list[Any]) -> dict[str, Any]:
    resident = _load_student("resident")
    on_demand = _load_student("on_demand")
    per_request = []
    for example in examples:
        input_ids = example.input_ids.unsqueeze(0).to(DEVICE)
        resident_output, resident_state, _, resident_trace = _hard_forward(
            resident, example, request_id=f"{example.example_id}-resident", input_ids=input_ids, use_cache=False
        )
        torch.cuda.synchronize(DEVICE)
        on_output, on_state, on_context, on_trace = _hard_forward(
            on_demand, example, request_id=f"{example.example_id}-on-demand", input_ids=input_ids, use_cache=False
        )
        torch.cuda.synchronize(DEVICE)
        resident_logits = resident_output.logits.detach()
        on_logits = on_output.logits.detach()
        resident_map = _route_map(resident_state)
        on_map = _route_map(on_state)
        prefill_records = 0 if on_context is None else len(on_context.records)
        transfer = None if on_context is None else _transfer_summary(on_context, on_map, prefill_count=prefill_records)
        resident_state.end_request()
        generation_resident = _generation(resident, example, request_id=f"{example.example_id}-generation-resident")
        generation_on_demand = _generation(on_demand, example, request_id=f"{example.example_id}-generation-on-demand")
        retained_before_end = None if on_context is None else {
            "entries": on_context.retained_entry_count,
            "buffers": on_context.retained_gpu_buffer_count,
            "packed_bytes": on_context.retained_packed_bytes,
        }
        if on_context is not None:
            on_state.end_request()
            retained_after_end = {
                "entries": on_context.retained_entry_count,
                "buffers": on_context.retained_gpu_buffer_count,
                "packed_bytes": on_context.retained_packed_bytes,
            }
        else:
            retained_after_end = None
        hidden_copy = {
            "any_precision_module_count": sum(
                module.__class__.__name__ == "AnyPrecisionLinear" for module in on_demand.base.modules()
            ),
            "source_count": 0 if on_context is None else len(on_context.sources),
            "all_source_qweights_cpu": False
            if on_context is None
            else all(source.qweight.device.type == "cpu" for source in on_context.sources.values()),
            "all_source_luts_cpu": False
            if on_context is None
            else all(source.lut4.device.type == "cpu" and source.lut8.device.type == "cpu" for source in on_context.sources.values()),
            "no_complete_packed_gpu_copy": sum(
                module.__class__.__name__ == "AnyPrecisionLinear" for module in on_demand.base.modules()
            ) == 0,
        }
        per_request.append(
            {
                "example_id": example.example_id,
                "logits_shape": list(resident_logits.shape),
                "finite_outputs": bool(torch.isfinite(resident_logits).all().item())
                and bool(torch.isfinite(on_logits).all().item()),
                "resident_digest": _tensor_sha256(resident_logits),
                "on_demand_digest": _tensor_sha256(on_logits),
                "mean_absolute_logit_difference": float((resident_logits.float() - on_logits.float()).abs().mean().item()),
                "maximum_absolute_logit_difference": float((resident_logits.float() - on_logits.float()).abs().max().item()),
                "bitwise_equal": bool(torch.equal(resident_logits, on_logits)),
                "resident_route_map": resident_map,
                "on_demand_route_map": on_map,
                "route_map_digest": _route_digest(resident_map),
                "route_maps_equal": resident_map == on_map,
                "resident_trace_records": len(resident_trace.records),
                "on_demand_trace_records": len(on_trace.records),
                "transfer": transfer,
                "hidden_copy_audit": hidden_copy,
                "retained_before_end": retained_before_end,
                "retained_after_end": retained_after_end,
                "generation_resident": generation_resident,
                "generation_on_demand": generation_on_demand,
                "generation_token_ids_equal": generation_resident["token_ids"] == generation_on_demand["token_ids"],
                "generation_finite": generation_resident["all_finite"] and generation_on_demand["all_finite"],
                "generation_routes_fixed": generation_resident["route_map_unchanged_during_decode"]
                and generation_on_demand["route_map_unchanged_during_decode"],
            }
        )
    return {"per_request": per_request}


def _measure(mode: str, examples: list[Any], repeats: int) -> dict[str, Any]:
    student = _load_student(mode)
    example = examples[0]
    prompt = example.input_ids[:32].unsqueeze(0).to(DEVICE)
    # Equal warm-up policy; the warm-up request is ended before measurement.
    warmup = _generation(student, example, request_id=f"{mode}-warmup", steps=1)
    torch.cuda.synchronize(DEVICE)
    device = torch.device(DEVICE)
    runs = []
    for repeat in range(repeats):
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        pre_allocated = int(torch.cuda.memory_allocated(device))
        pre_reserved = int(torch.cuda.memory_reserved(device))
        state, context = _context_for(student, example, f"{mode}-measured-{repeat}")
        trace = __import__("qaq.s04_manual", fromlist=["PrecisionTrace"]).PrecisionTrace()
        start = time.perf_counter()
        prefill_start = time.perf_counter()
        output = student.base(
            input_ids=prompt,
            attention_mask=torch.ones_like(prompt, dtype=torch.bool),
            use_cache=True,
            request_state=state,
            phase="prefill",
            routing_policy=_policy(student),
            prompt_attention_mask=torch.ones_like(prompt, dtype=torch.bool),
            on_demand_context=context,
            trace=trace,
        )
        torch.cuda.synchronize(device)
        prefill_seconds = time.perf_counter() - prefill_start
        past = output.past_key_values
        decode_start = time.perf_counter()
        decode_tokens = []
        for _ in range(4):
            token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            decode_tokens.append(int(token.item()))
            output = student.base(
                input_ids=token,
                attention_mask=torch.ones_like(token, dtype=torch.bool),
                past_key_values=past,
                use_cache=True,
                request_state=state,
                phase="decode",
                on_demand_context=context,
                trace=trace,
            )
            past = output.past_key_values
        torch.cuda.synchronize(device)
        decode_seconds = time.perf_counter() - decode_start
        end_to_end_seconds = time.perf_counter() - start
        retained_before_end = None if context is None else {
            "entries": context.retained_entry_count,
            "buffers": context.retained_gpu_buffer_count,
            "packed_bytes": context.retained_packed_bytes,
        }
        prefill_record_count = 0 if context is None else len(context.records)
        # Preserve the request boundary so transfer accounting separates prefill
        # from decode even when decode transfers zero bytes.
        state.end_request()
        torch.cuda.synchronize(device)
        post_allocated = int(torch.cuda.memory_allocated(device))
        post_reserved = int(torch.cuda.memory_reserved(device))
        after_cleanup = None if context is None else {
            "entries": context.retained_entry_count,
            "buffers": context.retained_gpu_buffer_count,
            "packed_bytes": context.retained_packed_bytes,
        }
        transfer = None if context is None else _transfer_summary(
            context, _route_map(state), prefill_count=prefill_record_count
        )
        runs.append(
            {
                "repeat": repeat,
                "prefill_seconds": prefill_seconds,
                "decode_seconds": decode_seconds,
                "end_to_end_seconds": end_to_end_seconds,
                "generated_token_ids": decode_tokens,
                "all_finite": bool(torch.isfinite(output.logits).all().item()),
                "pre_request_allocated": pre_allocated,
                "pre_request_reserved": pre_reserved,
                "peak_allocated": int(torch.cuda.max_memory_allocated(device)),
                "peak_reserved": int(torch.cuda.max_memory_reserved(device)),
                "post_cleanup_allocated": post_allocated,
                "post_cleanup_reserved": post_reserved,
                "retained_before_end": retained_before_end,
                "retained_after_end": after_cleanup,
                "transfer": transfer,
                "route_map": _route_map(state),
                "route_map_digest": _route_digest(_route_map(state)),
            }
        )
    def median(name: str) -> float:
        values = sorted(run[name] for run in runs)
        return values[len(values) // 2]
    return {
        "mode": mode,
        "measurement_method": (
            "Two synchronized CUDA warm-up/measurement phases per request: one 32-token "
            "hard-route prefill followed by four greedy decode tokens; median of the "
            "requested repeats. Peak allocated/reserved memory uses PyTorch CUDA allocator "
            "counters, and transfer bytes sum destination packed-buffer numel*element_size."
        ),
        "cuda_device": DEVICE,
        "cuda_name": torch.cuda.get_device_name(device),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "warmup": warmup,
        "warmup_policy": "one fixed prompt request with one greedy decode token; request ended before measurement",
        "repeats": runs,
        "median_seconds": {
            "prefill": median("prefill_seconds"),
            "decode": median("decode_seconds"),
            "end_to_end": median("end_to_end_seconds"),
        },
        "allocator_note": "reserved memory may remain cached by PyTorch; request-owned packed references are the cleanup criterion",
        "empty_cache_between_runs": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("compare", "resident", "on_demand"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    _require_environment()
    examples = _load_examples()
    if args.mode == "compare":
        payload = {
            "format": "qaq-s08b-real-hard-routed-v1",
            "scope": "S08-B only; no S09 comparison",
            "code_provenance": _code_provenance(),
            "model_revision": MODEL_REVISION,
            "packed_checkpoint_sha256": _file_sha256(ARTIFACT / "pytorch_model.bin"),
            "any_precision_revision": ANY_PRECISION_REVISION,
            "router_checkpoint_sha256": _file_sha256(ROUTER_CHECKPOINT),
            "cuda_device": DEVICE,
            "cuda_name": torch.cuda.get_device_name(torch.device(DEVICE)),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "request_count": len(examples),
            "request_ids": [example.example_id for example in examples],
            "input_token_digests": {
                example.example_id: _tensor_sha256(example.input_ids) for example in examples
            },
            "correctness": _correctness(examples),
            "limitations": [
                "Two locked S07 validation requests only.",
                "Short greedy generation uses the first 32 recorded prompt tokens and four new tokens.",
                "This is not the S09 final mode comparison or benchmark suite.",
            ],
        }
    else:
        payload = {
            "format": "qaq-s08b-real-hard-routed-measurement-v1",
            "scope": "S08-B only; one hard-routed mode",
            "code_provenance": _code_provenance(),
            "model_revision": MODEL_REVISION,
            "packed_checkpoint_sha256": _file_sha256(ARTIFACT / "pytorch_model.bin"),
            "any_precision_revision": ANY_PRECISION_REVISION,
            "router_checkpoint_sha256": _file_sha256(ROUTER_CHECKPOINT),
            "request_count": 1,
            "request_id": examples[0].example_id,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "measurement": _measure(args.mode, examples, args.repeats),
            "limitations": [
                "One locked S07 validation prompt for the bounded measurement sample.",
                "Warm-up and two measured repeats; no large benchmark was run.",
                "Memory values are process-local CUDA allocator observations.",
            ],
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
