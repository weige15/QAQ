from pathlib import Path

from qaq.evaluation.quality import generate_fixed, read_prompt_file

ROOT = Path(__file__).resolve().parents[2]


def test_static_generation_is_finite_and_deterministic(static_case, artifact):
    model, _, _ = static_case
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(artifact), local_files_only=True)
    prompts = read_prompt_file(ROOT / "configs/s03_static_generation_prompts.txt")
    for precision in (4, 8):
        result = generate_fixed(model, tokenizer, prompts, precision, "cuda:3")
        assert all(record["deterministic_repeat"] for record in result["records"])
        assert all(record["first"]["generated_token_count"] <= 8 for record in result["records"])
        assert all(record["first"]["sequence_sha256"] for record in result["records"])
