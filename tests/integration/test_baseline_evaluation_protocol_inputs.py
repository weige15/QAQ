from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from qaq.evaluation.quality import evaluate_perplexity, select_token_windows
from scripts.validate_baseline_evaluation_protocol import validate_protocol_payload

ROOT = Path(__file__).resolve().parents[2]


def test_fixed_prompt_file_is_deterministic_and_complete():
    config = json.loads((ROOT / "configs/baseline_evaluation.json").read_text())
    prompts = json.loads((ROOT / "configs/baseline_evaluation_prompts.json").read_text())
    result = validate_protocol_payload(config, ROOT, prompt_payload=prompts, check_external=False)
    assert result["request_count"] == 7
    assert prompts["tokenizer"]["runtime_prompt_generation"] is False
    assert [request["id"] for request in prompts["requests"]] == config["fixed_inputs"][
        "request_ids"
    ]
    assert all(
        request["input_token_count"] == len(request["input_ids"]) for request in prompts["requests"]
    )


def test_source_order_window_selection_has_non_overlapping_targets():
    tokens = list(range(4097))
    windows = select_token_windows(tokens, sequence_length=128, sample_count=32, stride=128)
    assert len(windows) == 32
    assert all(len(window) == 129 for window in windows)
    assert windows[0][0] == 0
    assert windows[1][0] == 128
    assert windows[-1][0] == 3968
    assert windows[0][1:] == list(range(1, 129))
    assert windows[1][1:] == list(range(129, 257))
    assert sum(len(window) - 1 for window in windows) == 4096


class ConstantTargetModel:
    def __init__(self, vocabulary_size: int, target: int, target_logit: float):
        self.vocabulary_size = vocabulary_size
        self.target = target
        self.target_logit = target_logit

    def __call__(self, input_ids, use_cache=False):
        logits = torch.zeros(
            (*input_ids.shape, self.vocabulary_size), dtype=torch.float32, device=input_ids.device
        )
        logits[..., self.target] = self.target_logit
        return SimpleNamespace(logits=logits)


def test_controlled_perplexity_accounting_is_token_weighted_and_next_token_aligned():
    windows = [torch.tensor([0, 1, 1]), torch.tensor([0, 1, 1, 1])]
    result = evaluate_perplexity(ConstantTargetModel(2, 1, 2.0), windows, "cpu")
    expected_nll = torch.nn.functional.cross_entropy(
        torch.tensor([[0.0, 2.0]]), torch.tensor([1]), reduction="none"
    ).item()
    assert result["evaluated_token_count"] == 5
    assert result["mean_negative_log_likelihood"] == pytest.approx(expected_nll)
    assert result["perplexity"] == pytest.approx(torch.exp(torch.tensor(expected_nll)).item())


def test_protocol_freezes_expected_sample_count_and_token_count():
    config = json.loads((ROOT / "configs/baseline_evaluation.json").read_text())
    assert config["perplexity"]["sample_count"] == 32
    assert config["perplexity"]["sequence_length"] == 128
    assert (
        config["perplexity"]["sample_count"] * config["perplexity"]["sequence_length"]
        == config["perplexity"]["evaluated_token_count"]
        == 4096
    )
