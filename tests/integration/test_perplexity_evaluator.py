from types import SimpleNamespace

import pytest
import torch

from qaq.s03_quality import evaluate_perplexity


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


def test_perplexity_is_token_weighted_and_uses_next_token_labels():
    windows = [torch.tensor([0, 1, 1]), torch.tensor([0, 1, 1, 1])]
    model = ConstantTargetModel(vocabulary_size=2, target=1, target_logit=2.0)
    result = evaluate_perplexity(model, windows, "cpu")
    expected_nll = torch.nn.functional.cross_entropy(
        torch.tensor([[0.0, 2.0]]), torch.tensor([1]), reduction="none"
    ).item()
    assert result["evaluated_token_count"] == 5
    assert result["mean_negative_log_likelihood"] == pytest.approx(expected_nll)
    assert result["perplexity"] == pytest.approx(torch.exp(torch.tensor(expected_nll)).item())
