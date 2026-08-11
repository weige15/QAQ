# S03 — Static 4-bit and 8-bit model baselines

## Goal

Quantize the target model into one nested representation and validate static 4-bit and 8-bit inference.

## Tasks

- Use the S02 physical format and the pinned backend.
- Quantize the target model once into the nested representation.
- Keep embeddings, normalization, activations, KV cache, and output head in BF16/FP16 as specified by D005.
- Run static 4-bit and static 8-bit inference with deterministic inputs.
- Record quality, memory, latency, and serialization evidence with exact commands.
- Keep quantized weights frozen; no router training occurs in this stage.

## Tests

- Model conversion is deterministic and reproducible.
- Static 4-bit inference runs and meets the documented correctness gate.
- Static 8-bit inference runs and meets the documented correctness gate.
- Output and resource measurements use physically packed weights.
- Non-quantized component dtypes match the recorded decision.

## Required outputs

- Nested packed model artifact or reproducible generation command.
- Static 4-bit and 8-bit correctness and resource reports.
- Tests for representative model paths.
- Updated decisions and status.

## Known uncertainties

- Target model conversion compatibility and quality thresholds remain unverified.
- Exact evaluator and reference tolerances must come from S00 evidence.

## CONTINUE condition

Both static routes are reproducible, correct under the documented gate, and measured with real packed storage.

## PAUSE condition

Required model artifacts or evaluation resources are unavailable.

## REVISE condition

A conversion or dtype assumption needs correction while preserving the S03 objective.

## STOP condition

Static 4-bit or 8-bit inference cannot be made trustworthy, or the implementation would require fake packing or unfrozen quantized weights.

## S03-A — Actual target-model verification

**Status: IN_PROGRESS — CONTINUE to S03-B, which is not executed here.**

### Exact identity and acquisition

- Repository: `Qwen/Qwen3-4B`.
- Immutable revision loaded: `1cfa9a7208912126459214e8b04321603b3df60c`.
- The revision was obtained from `configs/model.yaml`; `main` and all model substitutions were rejected.
- Transformers `4.51.0` was installed because the pinned model configuration declares that version and the prior `4.39.3` environment has no Qwen3 implementation.
- Snapshot/cache path: `/nfs/home/s314511048/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c`.
- Downloaded files were the three safetensors shards, `model.safetensors.index.json`, `config.json`, `generation_config.json`, and the four tokenizer files (`tokenizer.json`, `tokenizer_config.json`, `vocab.json`, `merges.txt`).
- Logical cached bytes: `8,060,897,472`; disk free changed from `6,665,635,299,328` to `6,657,542,258,688` bytes (disk delta `8,093,040,640`).
- No model files are tracked; `.gitignore` excludes full-precision weights, Hugging Face cache material, quantized checkpoints, and conversion outputs.

### Resources and load

Immediately before loading, available RAM was `236,805,435,392` bytes and CUDA device `cuda:3` had `24,112,726,016` free of `25,296,044,032` bytes. After loading, available RAM was `236,667,547,648` bytes and CUDA device `cuda:3` had `16,009,330,688` free. The model was loaded in BF16 on `cuda:3` with `trust_remote_code=false`; no CPU-only fallback was used.

- Python class: `transformers.models.qwen3.modeling_qwen3.Qwen3ForCausalLM`.
- Loaded dtype: `torch.bfloat16`.
- Parameter count: `4,022,468,096`.
- Actual parameter device: `cuda:3`.
- Loaded revision argument: `1cfa9a7208912126459214e8b04321603b3df60c`.

### Actual module tree

The real instantiated tree contains 144 attention targets and 108 FFN targets, for 252 total targets across layer indices 0 through 35. Every target is `torch.nn.modules.linear.Linear`, has no bias, and has the S00 input/output shape. The generated records include full path, class, dimensions, bias, dtype, shape, layer, unit, and proposed-target fields in [`../actual_model_modules.json`](../actual_model_modules.json).

The comparison against [`../model_structure.json`](../model_structure.json) and [`../QWEN3_MAPPING.md`](../QWEN3_MAPPING.md) is `MATCH`: no missing targets, duplicate paths, unexpected target paths, dimension mismatches, bias mismatches, or layer-index gaps were found. The only other instantiated `nn.Linear` is `lm_head`, classified as the explicitly excluded output head.

The exclusion audit records token embeddings, LM/output head, final and per-layer normalization, Q/K normalization, rotary position processing, activation functions, and KV-cache runtime structures as outside the target list. No Qwen module was replaced.

### Tied weights and smoke test

The instantiated model confirms `model.embed_tokens.weight` and `lm_head.weight` are the same parameter object and share the same storage pointer, with `tie_word_embeddings=true`.

The deterministic full-precision smoke prompt was `QAQ full-precision smoke test.` with batch size 1 and input shape `[1, 8]`. The unmodified model returned finite BF16 logits of shape `[1, 8, 151936]`; float32 logits digest: `a59aa0c2a7d31a8e4a5e9687ce229f9fcaa461344d3ea68f506867355fd73a18`.

### Scope and gate

No Any-Precision quantizer, bitsandbytes, GPTQ, AWQ, fake quantization, packed Qwen weight, 4-bit model, 8-bit model, routing work, or S04 work was run. S03 remains `IN_PROGRESS`; the next action is exactly:

`S03-B: quantize the verified Qwen3 target modules into one nested 4-bit/8-bit packed representation.`

The S03-A gate is **CONTINUE**. The pinned model loaded and executed, the concrete module tree matches S00, exclusions and tied weights were verified, and no quantization was performed.
