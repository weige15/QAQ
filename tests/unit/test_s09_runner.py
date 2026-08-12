from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from qaq import s09_runner as runner

ROOT = Path(__file__).resolve().parents[2]


def _protocol():
    config, prompts, config_hash = runner.load_protocol(ROOT / "configs/s09_baseline_eval.json")
    return config, prompts, config_hash


def _route(request_id: str, bit: int = 4) -> dict:
    route_map = [
        {"layer": layer, "unit_type": unit, "selected_bits": bit}
        for layer in range(36)
        for unit in ("attention", "ffn")
    ]
    fraction = {"4_bit": 1.0 if bit == 4 else 0.0, "8_bit": 1.0 if bit == 8 else 0.0}
    return {
        "request_id": request_id,
        "route_map": route_map,
        "route_map_digest": runner._route_digest(route_map),
        "attention_fractions": fraction,
        "ffn_fractions": fraction,
        "overall_fractions": fraction,
    }


def _fixture(mode_id: str, config: dict, prompts: dict, config_hash: str, *, ppl: float = 10.0) -> dict:
    mode = next(mode for mode in config["modes"] if mode["id"] == mode_id)
    requests = runner.fixed_requests(prompts)
    identities = {
        "model_repository": "Qwen/Qwen3-4B",
        "model_revision": runner.MODEL_REVISION,
        "tokenizer_revision": runner.MODEL_REVISION,
        "any_precision_revision": runner.ANY_PRECISION_REVISION if mode["packed_artifact"] else None,
        "packed_checkpoint_sha256": config["identities"]["packed_artifact"]["sha256"] if mode["packed_artifact"] else None,
        "router_checkpoint_sha256": config["identities"]["router"]["sha256"] if mode_id in runner.ROUTED_MODE_IDS else None,
    }
    result = {
        "schema": runner.RESULT_SCHEMA,
        "mode_id": mode_id,
        "protocol": {"config_sha256": config_hash, "frozen": True},
        "provenance": {"git_commit": "fixture", "worktree_status": ""},
        "identities": identities,
        "hardware": {"device_index": 3, "gpu_model": "fixture", "driver": "fixture", "cuda_runtime": "fixture", "pytorch": "fixture", "transformers": "fixture", "python": "fixture"},
        "seed": 1729,
        "fixed_inputs": {"request_ids": [item["id"] for item in requests], "input_digests": {item["id"]: item["input_ids_sha256"] for item in requests}},
        "perplexity": {"mean_negative_log_likelihood": 1.0, "perplexity": ppl, "evaluated_token_count": 4096},
        "generation": {
            "batch_size": 1,
            "do_sample": False,
            "num_beams": 1,
            "max_new_tokens": 8,
            "records": [
                {"request_id": item["id"], "input_ids_sha256": item["input_ids_sha256"], "generated_token_ids": [1], "output_digest": "fixture-output", "logits_digest": "fixture-logits", "finite_value_check": True, "normal_termination": True}
                for item in requests
            ],
        },
        "memory": {
            "method": {"synchronize_before": "torch.cuda.synchronize()", "reset_peak": "torch.cuda.reset_peak_memory_stats()", "synchronize_after": "torch.cuda.synchronize()", "empty_cache_inside_interval": False},
            "records": [{"allocated_before": 1, "reserved_before": 2, "peak_allocated": 3, "peak_reserved": 4, "allocated_after_cleanup": 0, "reserved_after_cleanup": 0}],
        },
        "latency": {
            "warmup_requests": 1,
            "repeats_per_request": 5,
            "outlier_removal": False,
            "subtract_transfer_time": False,
            "raw_records": [
                {"request_id": item["id"], "repeat": repeat, "prefill_seconds": 1.0, "decode_seconds": 1.0, "end_to_end_seconds": 2.0}
                for item in requests for repeat in range(5)
            ],
        },
        "deterministic_checks": {"all_required_outputs_finite": True, "fixed_inputs_identical": True, "evidence_present": True},
    }
    if mode_id in runner.ROUTED_MODE_IDS:
        result["routed"] = {"requests": [_route(item["id"]) for item in requests], "route_diversity": {"adaptivity_classification": "OTHER"}}
    if mode_id == runner.ON_DEMAND_MODE_ID:
        result["on_demand"] = {
            "first_use_bytes": 10, "reuse_bytes": 0, "prefill_bytes": 10, "decode_bytes": 0,
            "attention_bytes": 5, "ffn_bytes": 5, "total_transfer_bytes": 10,
            "first_use_events": 1, "reuse_events": 0, "independently_expected_physical_bytes": 10,
            "actual_equals_expected": True, "retained_entries_before_cleanup": 1,
            "retained_buffers_before_cleanup": 2, "retained_entries_after_cleanup": 0,
            "retained_buffers_after_cleanup": 0, "retained_bytes_after_cleanup": 0,
            "no_complete_packed_parent_on_gpu": True,
        }
    return result


def test_environment_guard_and_frozen_config_are_explicit(monkeypatch):
    config, prompts, _ = _protocol()
    assert runner.frozen_perplexity_arguments(config) == {
        "sample_count": 32,
        "sequence_length": 128,
        "source_window_length": 129,
        "stride": 128,
        "evaluated_token_count": 4096,
        "labels": "window[1:] aligned with logits from window[:-1]",
        "evaluator": "qaq.s03_quality.build_perplexity_windows + qaq.s03_quality.evaluate_perplexity",
    }
    assert runner.frozen_generation_arguments(config) == {"batch_size": 1, "do_sample": False, "num_beams": 1, "max_new_tokens": 8}
    assert runner.frozen_latency_repeats(config) == 5
    assert tuple(mode["id"] for mode in runner.resolve_modes(config)) == runner.EXPECTED_MODE_IDS
    assert len(runner.fixed_requests(prompts)) == 7
    monkeypatch.setenv("VIRTUAL_ENV", str(Path.home() / ".venv"))
    assert str(Path.home() / ".venv") in str(Path.home() / ".venv")


def test_unknown_duplicate_or_missing_mode_is_rejected():
    config, _, _ = _protocol()
    changed = copy.deepcopy(config)
    changed["modes"][0]["id"] = "unknown"
    with pytest.raises(runner.S09RunnerError, match="mode IDs"):
        runner.resolve_modes(changed)
    changed = copy.deepcopy(config)
    changed["modes"][4]["id"] = changed["modes"][3]["id"]
    with pytest.raises(runner.S09RunnerError, match="mode IDs"):
        runner.resolve_modes(changed)
    changed = copy.deepcopy(config)
    changed["modes"].pop()
    with pytest.raises(runner.S09RunnerError, match="mode IDs"):
        runner.resolve_modes(changed)


def test_child_command_is_one_explicit_fresh_process_per_mode():
    config, _, _ = _protocol()
    commands = [runner.child_command(ROOT / "configs/s09_baseline_eval.json", mode["id"], Path("tmp") / f"{mode['id']}.json", "cuda:3") for mode in runner.resolve_modes(config)]
    assert len(commands) == 5
    assert all("--execute-mode" in command for command in commands)
    assert {command[command.index("--execute-mode") + 1] for command in commands} == set(runner.EXPECTED_MODE_IDS)


def test_result_contract_requires_protocol_generation_memory_and_latency():
    config, prompts, config_hash = _protocol()
    result = _fixture(runner.EXPECTED_MODE_IDS[0], config, prompts, config_hash)
    runner.validate_result(result, config, prompts, config_hash)
    del result["generation"]["records"]
    with pytest.raises(runner.S09RunnerError, match="generation records"):
        runner.validate_result(result, config, prompts, config_hash)


def test_routed_and_on_demand_contracts_are_required():
    config, prompts, config_hash = _protocol()
    routed = _fixture(runner.ROUTED_MODE_IDS[0], config, prompts, config_hash)
    runner.validate_result(routed, config, prompts, config_hash)
    routed["routed"]["requests"][0]["route_map"].pop()
    with pytest.raises(runner.S09RunnerError, match="72-unit"):
        runner.validate_result(routed, config, prompts, config_hash)
    on_demand = _fixture(runner.ON_DEMAND_MODE_ID, config, prompts, config_hash)
    runner.validate_result(on_demand, config, prompts, config_hash)
    on_demand["on_demand"]["actual_equals_expected"] = False
    with pytest.raises(runner.S09RunnerError, match="transfer equality"):
        runner.validate_result(on_demand, config, prompts, config_hash)


def test_aggregation_quality_math_and_missing_results(tmp_path):
    config, prompts, config_hash = _protocol()
    assert runner.aggregate(ROOT / "configs/s09_baseline_eval.json", tmp_path)["classification"] == "PAUSE"
    for mode in config["modes"]:
        result = _fixture(mode["id"], config, prompts, config_hash, ppl=10.0 if mode["id"] != runner.EXPECTED_MODE_IDS[2] else 11.0)
        (tmp_path / f"{mode['id']}.json").write_text(json.dumps(result))
    aggregate = runner.aggregate(ROOT / "configs/s09_baseline_eval.json", tmp_path)
    assert aggregate["classification"] == "CONTINUE"
    bad_quality = _fixture(runner.EXPECTED_MODE_IDS[2], config, prompts, config_hash, ppl=12.0)
    (tmp_path / f"{runner.EXPECTED_MODE_IDS[2]}.json").write_text(json.dumps(bad_quality))
    assert runner.aggregate(ROOT / "configs/s09_baseline_eval.json", tmp_path)["classification"] == "REVISE"
    bad = _fixture(runner.ROUTED_MODE_IDS[1], config, prompts, config_hash)
    bad["routed"]["requests"][0]["route_map"][0]["selected_bits"] = 8
    (tmp_path / f"{runner.ROUTED_MODE_IDS[1]}.json").write_text(json.dumps(bad))
    aggregate = runner.aggregate(ROOT / "configs/s09_baseline_eval.json", tmp_path)
    assert aggregate["classification"] == "REVISE"


def test_aggregation_protocol_hash_mismatch_is_revise(tmp_path):
    config, prompts, config_hash = _protocol()
    for mode in config["modes"]:
        result = _fixture(mode["id"], config, prompts, config_hash)
        if mode["id"] == runner.EXPECTED_MODE_IDS[0]:
            result["protocol"]["config_sha256"] = "wrong"
        (tmp_path / f"{mode['id']}.json").write_text(json.dumps(result))
    assert runner.aggregate(ROOT / "configs/s09_baseline_eval.json", tmp_path)["classification"] == "REVISE"
