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

**Status: COMPLETE — CONTINUE evidence recorded for S03-B.**

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

## S03-B — Nested Qwen3 4/8-bit packed baseline

**Status: COMPLETE — CONTINUE evidence recorded for S03-C.** S03-B produced and validated the static nested checkpoint.

- Pinned quantizer entry point: `any_precision.quantization.any_precision_quantize`, with its `any_precision.quantization.pack.pack` packing stage.
- Exact mapping: `configs/qwen3_any_precision.yaml`, checked against the S03-A records in `docs/actual_model_modules.json` by exact set and count equality before quantization.
- Source model: `Qwen/Qwen3-4B`, revision `1cfa9a7208912126459214e8b04321603b3df60c`; Any-Precision commit `a3257d02740cc5757c78673da534b0630ff3a4ea`.
- Exact command: `source ~/.venv/bin/activate && which python && python --version && PYTHONPATH=src:third_party/any-precision-llm python scripts/run_s03b.py --overwrite-artifact`.
- Quantization: seed precision `4`, parent precision `8`, group count `1`, deterministic random state `1729`.
- Calibration: pinned C4 loader, train split `allenai/c4` shard `en/c4-train.00000-of-01024.json.gz`, one sample, 64 tokens, tokenizer first-64-token truncation, Python/NumPy seed `1729`.
- Runtime: `197.52182836900465` seconds on CUDA device `cuda:3` / NVIDIA GeForce RTX 3090. Peak quantization RAM was not captured; pre-quantization available RAM was `242553462784` bytes and GPU free VRAM was `24112726016` bytes. Static smoke peak allocated GPU memory was `5585867264` bytes at 4 bits and `5588298240` bytes at 8 bits.
- Artifact: `quantized/s03b_qwen3_4b/backend_cache/packed/anyprec-(1cfa9a7208912126459214e8b04321603b3df60c)-w8_orig4-gc1-c4_s1_blk64`.
- Target count: `252`; omitted, unexpected, duplicate, and excluded quantized targets: none.
- Physical storage: parent `qweight` payload `3633315840` bytes as `torch.int32` `[8,N,K//32]` tensors; selected 4-bit prefix `1816657920` bytes; selected 8-bit payload `3633315840` bytes. LUT4 bytes `35389440`, LUT8 bytes `566231040`, separate scale bytes `0`, metadata-file bytes `15883586`, lookup/scale/metadata total `617504066` bytes. Total checkpoint/artifact bytes: `5525158010`.
- Nested proof: each target has exactly one 8-plane parent payload and separate row LUTs of `[N,16]` and `[N,256]`; static 4-bit selects `qweight[:4]` with `lut4`, while static 8-bit selects all eight planes with `lut8`. No independent 4-bit qweight or 8-bit model copy exists. The parent suffix had `454160316` nonzero elements.
- Full-precision baseline: BF16 logits shape `[1,8,151936]`, finite, digest `a59aa0c2a7d31a8e4a5e9687ce229f9fcaa461344d3ea68f506867355fd73a18`.
- Static 4-bit smoke: finite logits `[1,8,151936]`, digest `8b28d8ae1cf0d27462b0704d2661ebe90f67073c4435bbd8e21ad2ef19a6aa5d`.
- Static 8-bit smoke: finite logits `[1,8,151936]`, digest `9337bad41bf1f9294aca8ba7721a313ad5abfe14e279970e2cf45142946f04c3`.
- Numerical sanity: FP-vs-4 mean/max absolute logit error `0.38069865107536316` / `3.34375`; FP-vs-8 `0.04913947731256485` / `0.6796875`. The 8-bit result is at least as faithful on both recorded measures.
- Round-trip: the fresh-process integration checkpoint reload and manifest/hash checks passed.
- Checkpoint hashes and complete tensor inventory are tracked in `docs/quantized_model_manifest.json`; weight payloads remain ignored and untracked.
- Limitations: calibration is intentionally a one-sample smoke baseline, and peak quantization RAM was not captured. Routing, CPU-to-GPU on-demand loading, training, and S04 were not started.

## S03-C — Broader static-quality evaluation and closeout

**Status: COMPLETE — CONTINUE.** Exact result evidence is in [`../results/s03_static_quality.json`](../results/s03_static_quality.json), using the unchanged S03-B checkpoint and unchanged pinned source revisions.

### Fixed prompt set

The committed prompt set is [`../../configs/s03_static_quality_prompts.txt`](../../configs/s03_static_quality_prompts.txt) and contains five materially different prompts. Every mode used tokenizer revision `1cfa9a7208912126459214e8b04321603b3df60c`, `add_special_tokens=true`, truncation at 128 tokens, and no padding. The reference prompt remained `QAQ full-precision smoke test.` from S03-B.

For every prompt, FP, static 4-bit, and static 8-bit logits were finite and repeatable. The comparison metric was mean absolute logit error plus maximum absolute logit error against FP logits. The aggregate criterion was: the mean of per-prompt mean absolute errors for 8-bit must be no greater than the corresponding 4-bit aggregate. Aggregate mean errors were `0.5141653061` for 4-bit and `0.0578794084` for 8-bit; aggregate maximum-per-prompt maxima were `7.7890625` and `1.0078125`.

### Perplexity sample

The same evaluator was used for all modes on `Salesforce/wikitext`, config `wikitext-2-raw-v1`, dataset revision `b08601e04326c79dfdd32d625aee71d232d685c3`, split `test`. It concatenated non-empty rows in source order and evaluated the first four non-overlapping 129-token windows as 128-token inputs with next-token labels, for 512 weighted tokens total. No padding, generated tokens, or random sampling was used. Mean negative log likelihood and perplexity were accumulated token-weighted. Results were FP `3.2209646702` / `25.0522757118`, static 4-bit `3.3002486229` / `27.1193805814`, and static 8-bit `3.2140788436` / `24.8803626466`. The predefined development quality criterion was static 8-bit perplexity no greater than 110% of static 4-bit; it passed.

This is a development sample, not a final benchmark or paper-score reproduction.

### Generation, reload, coverage, and resources

Greedy batch-one generation used two committed prompts and `max_new_tokens=8`. All three modes executed without non-finite generation scores, stayed within the token limit, and produced deterministic repeated sequences. A fresh pytest process ran the checkpoint round-trip tests: `3 passed`; static 4-bit and 8-bit smoke digests matched the recorded S03-B digests.

Peak development memory observations were FP `8,325,107,712` allocated / `8,355,053,568` reserved bytes, static 4-bit `5,780,443,136` / `5,823,791,104`, and static 8-bit `5,780,443,136` / `5,823,791,104`. These are not final memory-savings claims and no transfer savings were measured.

The verified target set remained complete at 252 projections with no omissions, unexpected targets, or duplicate independent precision models. Full project regression was `50 passed, 1 skipped`; the resource-heavy S03-A model check was `1 passed`. No routing, training, or on-demand loading was added.

### Exact commands and limitations

- Quality evaluation: `source ~/.venv/bin/activate && which python && python --version && PYTHONPATH=src:third_party/any-precision-llm QAQ_MODEL_DEVICE=cuda:3 python scripts/run_s03c.py`.
- Fresh checkpoint reload: `source ~/.venv/bin/activate && which python && python --version && QAQ_S03_ARTIFACT=<artifact> pytest -q tests/integration/test_checkpoint_roundtrip.py`.
- Full regression: `source ~/.venv/bin/activate && which python && python --version && QAQ_S03_ARTIFACT=<artifact> pytest -q tests`.
- Resource-heavy S03-A check: `source ~/.venv/bin/activate && which python && python --version && QAQ_RUN_RESOURCE_HEAVY=1 QAQ_MODEL_DEVICE=cuda:3 pytest -q tests/system/test_actual_model_load.py`.

The small prompt and perplexity samples are only static-baseline trust evidence. They do not establish final language-model quality, adaptive routing quality, transfer behavior, or on-demand residency behavior.
