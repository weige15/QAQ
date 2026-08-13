import json
from pathlib import Path

from qaq.evaluation.quality import read_prompt_file, run_logits, set_static_precision, tokenize_prompt

ROOT = Path(__file__).resolve().parents[2]


def test_recorded_prompt_set_has_complete_finite_deterministic_results():
    result = json.loads((ROOT / "docs/results/s03_static_quality.json").read_text())
    prompt_set = result["prompt_set"]
    assert prompt_set["count"] == len(prompt_set["prompts"]) >= 3
    assert prompt_set["fidelity_criterion"]
    for record in prompt_set["records"]:
        assert record["full_precision"]["finite_values"]
        assert record["full_precision"]["deterministic_repeat"]
        for mode in ("static_4", "static_8"):
            assert record[mode]["finite_values"]
            assert record[mode]["deterministic_repeat"]


def test_static_prompt_set_forward_is_finite_and_repeatable(static_case, artifact):
    model, _, _ = static_case
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(artifact), local_files_only=True)
    prompts = read_prompt_file(ROOT / "configs/s03_static_quality_prompts.txt")
    for precision in (4, 8):
        set_static_precision(model, precision)
        for prompt in prompts:
            inputs = tokenize_prompt(tokenizer, prompt, "cuda:3")
            first = run_logits(model, inputs)
            second = run_logits(model, inputs)
            assert first.shape == second.shape
            assert first.isfinite().all()
            assert torch_equal(first, second)


def torch_equal(first, second):
    import torch

    return torch.equal(first, second)
