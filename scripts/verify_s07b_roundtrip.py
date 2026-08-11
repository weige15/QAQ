#!/usr/bin/env python3
"""Fresh-process S07-B router checkpoint and hard-route determinism check."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--result", type=Path, default=ROOT / "docs/results/s07_router_training.json")
    args = parser.parse_args()
    if not str(Path.home() / ".venv") in str(Path(sys.executable).parent):
        raise SystemExit("PAUSE: ~/.venv is not active")
    if not torch.cuda.is_available():
        raise SystemExit("PAUSE: CUDA is unavailable")
    torch.cuda.set_device(torch.device(args.device))

    import run_s07b
    from datasets import load_dataset
    from transformers import AutoTokenizer

    from qaq.model.request_state import QaqRequestState
    from qaq.s04_manual import PrecisionTrace
    from qaq.s06_soft import load_soft_model
    from qaq.s07_distillation import RouterCheckpointMetadata, hard_route, load_router_checkpoint

    result = json.loads(args.result.read_text())
    manifest = json.loads((ROOT / "docs/quantized_model_manifest.json").read_text())
    config = result["training_configuration"]
    artifact = ROOT / manifest["artifact"]["local_path"]
    tokenizer = AutoTokenizer.from_pretrained(
        str(run_s07b.SNAPSHOT), revision=run_s07b.MODEL_REVISION, local_files_only=True
    )
    dataset = load_dataset(
        config["dataset"]["repository"],
        config["dataset"]["config"],
        split="validation",
        revision=config["dataset"]["revision"],
        trust_remote_code=False,
    )
    examples_cpu, _ = run_s07b._select_examples(
        dataset,
        tokenizer,
        config["dataset"]["validation_offsets"],
        split="validation",
        config=config,
        torch=torch,
    )
    examples = [run_s07b._device_example(example, args.device, torch) for example in examples_cpu]
    student = load_soft_model(
        artifact,
        args.device,
        temperature=float(config["training"]["routing_temperature"]),
    )
    student.to(args.device)
    metadata_payload = result["checkpoint"]["metadata"]
    metadata_payload["candidate_ordering"] = tuple(metadata_payload["candidate_ordering"])
    metadata = RouterCheckpointMetadata(**metadata_payload)
    load_router_checkpoint(result["checkpoint"]["external_path"], student.routers, metadata)
    stored_logs = {
        (item["request_id"], item["layer"], item["unit_type"]): item
        for item in result["evaluation"]["soft"]["route_logs"]
    }

    def soft_once(example):
        state = QaqRequestState(example.example_id, int(example.prompt_mask().sum()), layer_count=36)
        with torch.no_grad():
            output = student(
                **run_s07b._model_kwargs(example),
                request_state=state,
                phase="prefill",
                prompt_attention_mask=example.prompt_mask().unsqueeze(0),
                trace=PrecisionTrace(),
            )
        records = run_s07b._records_for_state(example.example_id, state, student, log_base=2.0)
        return output.logits.detach(), records

    def hard_once(example):
        state = QaqRequestState(example.example_id, int(example.prompt_mask().sum()), layer_count=36)

        def policy(layer, unit_type, feature):
            return int(hard_route(student.route(layer, unit_type, feature)))

        with torch.no_grad():
            output = student.base(
                **run_s07b._model_kwargs(example),
                request_state=state,
                phase="prefill",
                prompt_attention_mask=example.prompt_mask().unsqueeze(0),
                routing_policy=policy,
                trace=PrecisionTrace(),
            )
        records = run_s07b._records_for_state(example.example_id, state, student, log_base=2.0)
        return output.logits.detach(), records

    probability_match = True
    soft_route_match = True
    hard_repeat_match = True
    hard_logits_match = True
    hard_maps = []
    for example in examples:
        soft_logits, soft_records = soft_once(example)
        for record in soft_records:
            stored = stored_logs[(record.request_id, record.layer, record.unit_type)]
            probability_match &= abs(record.p4 - stored["p4"]) <= 1e-6
            probability_match &= abs(record.p8 - stored["p8"]) <= 1e-6
            soft_route_match &= record.hard_bit == stored["hard_bit"]
        first_logits, first_records = hard_once(example)
        second_logits, second_records = hard_once(example)
        first_map = run_s07b._route_map(first_records)
        second_map = run_s07b._route_map(second_records)
        hard_maps.append(first_map)
        hard_repeat_match &= first_map == second_map
        hard_logits_match &= torch.equal(first_logits, second_logits)
        hard_repeat_match &= all(
            left.hard_bit == right.hard_bit for left, right in zip(first_records, second_records)
        )
        if not bool(torch.isfinite(soft_logits).all().item() and torch.isfinite(first_logits).all().item()):
            raise SystemExit("REVISE: checkpoint round-trip produced non-finite logits")

    passed = probability_match and soft_route_match and hard_repeat_match and hard_logits_match
    result["hard_route_determinism"] = {
        "fixed_subset_count": len(examples),
        "route_maps_identical_on_repeat": hard_repeat_match,
        "selected_precisions_identical_on_repeat": hard_repeat_match,
        "logits_identical_on_repeat": hard_logits_match,
        "tolerance": "bitwise equality",
        "passed": passed,
    }
    result["checkpoint_roundtrip"] = {
        "fresh_process": True,
        "probabilities_match_recorded_result": probability_match,
        "hard_routes_match_recorded_result": soft_route_match,
        "unchanged_packed_student": True,
        "passed": probability_match and soft_route_match,
    }
    result["stage_gate"]["checkpoint_roundtrip_passed"] = result["checkpoint_roundtrip"]["passed"]
    result["stage_gate"]["hard_route_determinism_passed"] = passed
    args.result.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"checkpoint_roundtrip": result["checkpoint_roundtrip"], "hard_route_determinism": result["hard_route_determinism"]}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
