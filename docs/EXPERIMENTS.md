# Experiments

This file records experiment plans and links to completed stage evidence.
S02's physical-format evidence is authoritative in
[`docs/stages/S02_BITPACKING.md`](stages/S02_BITPACKING.md) and
[`docs/BITPLANE_FORMAT.md`](BITPLANE_FORMAT.md).

## S01 pinned backend evidence (2026-08-11)

This is the only experiment recorded by S01. It uses no model, Qwen3 weight,
dataset, or network-dependent input.

- Command: `python scripts/validate_backend.py`, after the mandatory environment activation block in `docs/stages/S01_BACKEND.md`.
- Any-Precision source: `a3257d02740cc5757c78673da534b0630ff3a4ea`.
- Hardware: CUDA device 0, NVIDIA GeForce RTX 3090.
- Seed and dimensions: seed `1729`, `M=4`, `N=64`, `K=1024`.
- Dtypes: input/LUT/output `float16`; packed qweight `int32`; bias disabled.
- 4-bit result: output digest `7a7d75ef8b5a56ff91f230f4c60ac49df46cdead833bc3cf6d8af0be9d146001`; max absolute error `0.00872802734375`; mean absolute error `0.00251007080078125`; meaningful max relative error `0.1104247123003006`; allclose `true` with `atol=0.05`, `rtol=0.01`.
- 8-bit result: output digest `7b218306e70f434aca7a7101ff57d973f9ffc120c8a1ac7b5b08ffad9f6d121c`; max absolute error `0.01171875`; mean absolute error `0.0023452043533325195`; meaningful max relative error `0.048128340393304825`; allclose `true` with the same tolerance.
- Storage: full qweight `[8,64,32]` int32, `65,536` bytes; selected packed 4-bit prefix `32,768` bytes; selected 8-bit planes `65,536` bytes; LUT4 `[64,16]` float16, `2,048` bytes; LUT8 `[64,256]` float16, `32,768` bytes.
- Determinism: repeated 4-bit and 8-bit outputs were bitwise equal with the digests above.
- Distinct paths: nonzero 8-bit qweight suffix, different LUT shapes, and different pinned-helper effective-weight digests; maximum effective-weight delta `0.04443359375`.

## Required comparison at S09

Compare:

1. full-precision teacher;
2. static 4-bit model;
3. static 8-bit model;
4. routed resident mode;
5. routed synchronous on-demand mode.

Record quality, selected routes, GPU memory, actual packed transfer bytes, and latency.
Every result must include the exact command, environment versions, model and data identifiers, deterministic seed, and relevant configuration.

## S03-B nested Qwen3 static baseline (2026-08-11)

- Command: `source ~/.venv/bin/activate && which python && python --version && PYTHONPATH=src:third_party/any-precision-llm python scripts/run_s03b.py --overwrite-artifact`.
- Model: `Qwen/Qwen3-4B`, immutable revision `1cfa9a7208912126459214e8b04321603b3df60c`; tokenizer uses the same revision. Any-Precision commit: `a3257d02740cc5757c78673da534b0630ff3a4ea`.
- Mapping and targets: explicit `configs/qwen3_any_precision.yaml`; exact S03-A target-set check passed for 252 projections, with no omitted, unexpected, duplicate, or excluded targets.
- Quantizer settings: seed `4`, parent `8`, group count `1`, random state `1729`; pinned C4 train shard, one 64-token sample, tokenizer first-64-token truncation.
- Quantization runtime: `197.52182836900465` seconds on CUDA `cuda:3`, NVIDIA GeForce RTX 3090.
- Physical storage: packed parent planes `3633315840` bytes; selected 4-plane prefix `1816657920` bytes; selected 8-plane payload `3633315840` bytes; LUT4 `35389440` bytes; LUT8 `566231040` bytes; scales `0` bytes; lookup/scale/metadata `617504066` bytes; total artifact `5525158010` bytes.
- Static 4-bit: logits `[1,8,151936]`, finite, digest `8b28d8ae1cf0d27462b0704d2661ebe90f67073c4435bbd8e21ad2ef19a6aa5d`, peak allocated GPU memory `5585867264` bytes.
- Static 8-bit: logits `[1,8,151936]`, finite, digest `9337bad41bf1f9294aca8ba7721a313ad5abfe14e279970e2cf45142946f04c3`, peak allocated GPU memory `5588298240` bytes.
- BF16 full-precision comparison digest: `a59aa0c2a7d31a8e4a5e9687ce229f9fcaa461344d3ea68f506867355fd73a18`.
- FP-vs-4 error: mean/max absolute `0.38069865107536316` / `3.34375`. FP-vs-8 error: mean/max absolute `0.04913947731256485` / `0.6796875`.
- Artifact: `quantized/s03b_qwen3_4b/backend_cache/packed/anyprec-(1cfa9a7208912126459214e8b04321603b3df60c)-w8_orig4-gc1-c4_s1_blk64`; complete hashes are in `docs/quantized_model_manifest.json`.
- Fresh-process round-trip and manifest integration tests: `9 passed`. Relevant S01/S02 regression tests: `27 passed`. Ruff: clean.

## S03-C broader static-quality evaluation (2026-08-11)

- Scope: full-precision teacher, the one nested static 4/8-bit checkpoint, and no routing, training, or on-demand loading.
- Fixed prompt file: `configs/s03_static_quality_prompts.txt`, five prompts. Tokenizer revision `1cfa9a7208912126459214e8b04321603b3df60c`; `add_special_tokens=true`, truncation `128`, no padding. Reference prompt remained `QAQ full-precision smoke test.`.
- Quality command: `source ~/.venv/bin/activate && which python && python --version && PYTHONPATH=src:third_party/any-precision-llm QAQ_MODEL_DEVICE=cuda:3 python scripts/run_s03c.py`.
- Prompt metric: mean and maximum absolute logit error against FP, with deterministic repeats. Five prompts completed with finite outputs. Aggregate mean-of-prompt mean error: 4-bit `0.5141653061`, 8-bit `0.0578794084`; maximum prompt maximum error: 4-bit `7.7890625`, 8-bit `1.0078125`. Criterion was aggregate 8-bit error `<=` aggregate 4-bit error; passed.
- Perplexity: `Salesforce/wikitext`, config `wikitext-2-raw-v1`, revision `b08601e04326c79dfdd32d625aee71d232d685c3`, split `test`; concatenate non-empty rows in source order, first four non-overlapping windows, sequence length `128`, window width/stride `129`, `512` evaluated tokens, tokenizer revision `1cfa9a7208912126459214e8b04321603b3df60c`, no random seed. FP `25.0522757118`, static 4-bit `27.1193805814`, static 8-bit `24.8803626466`; 8-bit <= 110% of 4-bit, passed.
- Generation command is included in the quality command; two fixed prompts, greedy batch size 1, `max_new_tokens=8`. All modes generated finite-score deterministic repeats within the limit.
- Round-trip command: `source ~/.venv/bin/activate && which python && python --version && QAQ_S03_ARTIFACT=<artifact> pytest -q tests/integration/test_checkpoint_roundtrip.py`; fresh process result `3 passed`, with both static smoke digests matching.
- Peak development memory: FP `8325107712` allocated / `8355053568` reserved bytes; static 4-bit `5780443136` / `5823791104`; static 8-bit `5780443136` / `5823791104`. These are not final savings claims; no transfer savings were measured.
- Regression commands: full project suite `50 passed, 1 skipped`; resource-heavy S03-A model load `1 passed`. Target coverage remained 252/252 with no duplicate independent precision models.
- Result file: `docs/results/s03_static_quality.json`. Limitations: this is a deliberately small development sample, not a final benchmark or paper-score reproduction.

## S06 trainable soft router and packed mixture (2026-08-11)

- Scope: 72 distinct prompt-feature routers, one for each attention or FFN
  unit; both real pinned 4-bit and 8-bit packed operations execute in the soft
  training path. No dataset, distillation, hard inference, or on-demand
  loading was used.
- Configuration: Qwen3-4B hidden size `2560`, router hidden width `128`, GELU,
  parameter-free RMS normalization with epsilon `1e-6`, temperature `1.0`,
  canonical outputs `[p4, p8]`, and PyTorch Linear default initialization.
- Router count and parameters: 36 attention plus 36 FFN routers, 72 total;
  `23,620,752` trainable router parameters. All packed model parameters,
  embeddings, normalizations, output head, LUTs, and quantization metadata were
  frozen.
- Focused command: `source ~/.venv/bin/activate && which python && python --version && pytest -q tests/unit/test_s06_router.py tests/integration/test_s06_soft_routing.py`; result `13 passed`.
- Pinned backend command: `source ~/.venv/bin/activate && which python && python --version && pytest -q tests/integration/test_s06_soft_packed.py -k 'not qwen3'`; result `2 passed`.
- Full artifact command: `source ~/.venv/bin/activate && which python && python --version && QAQ_S03_ARTIFACT=<S03-B artifact> QAQ_MODEL_DEVICE=cuda:3 pytest -q tests/integration/test_s06_soft_packed.py`; result `3 passed in 419.02s` on CUDA device 3, NVIDIA GeForce RTX 3090.
- Endpoint result: forced `[1,0]` and `[0,1]` mixtures matched the real hard 4-bit and hard 8-bit Qwen3 paths within `atol=1e-3`, `rtol=1e-3`; the synthetic pinned-backend endpoint comparisons were bitwise equal.
- Probability result: one and batched router outputs had shapes `[2]` and `[3,2]`; finite non-negative probabilities summed to one within `1e-6` in the router and `1e-5` at the packed boundary.
- Temperature result: fixed logits `[2,0]` were more concentrated toward 4-bit at temperature `0.5` than at `2.0`.
- Sharing result: each attention unit recorded one shared probability tensor across q/k/v/o; each FFN unit recorded one shared tensor across gate/up/down.
- Gradient result: finite nonzero router gradients reached the soft mixture; a one-step SGD smoke check changed router parameters only. No real training or quality claim was made.
- Regression command: `source ~/.venv/bin/activate && which python && python --version && pytest -q tests/unit`; result `67 passed`.
- Artifact regression command: `source ~/.venv/bin/activate && which python && python --version && QAQ_S03_ARTIFACT=<S03-B artifact> QAQ_MODEL_DEVICE=cuda:3 pytest -q tests/integration/test_s04_manual_routing.py tests/integration/test_s05_manual_routing.py tests/integration/test_static4_forward.py tests/integration/test_static8_forward.py`; result `12 passed in 431.49s`.

## Boundaries before baseline freeze

Do not introduce asynchronous transfers, prefetching, transfer prediction, bit-width cost penalties, cross-request caching, multi-query batching, or unrelated research improvements.

## S07-A router-distillation smoke (2026-08-11)

- Scope: reusable masked teacher-student KD, explicit prompt/completion
  example and alignment contracts, frozen teacher/S06 packed base, explicit
  router-only optimizer, router-only checkpoint, deterministic hard routes,
  route logs, and observational statistics. No real dataset or baseline-scale
  training was run.
- Command: `source ~/.venv/bin/activate && which python && python --version && PYTHONPATH=src pytest -q tests/unit/test_s07_distillation.py tests/integration/test_s07_distillation_smoke.py`.
- Fixture: seed `1729`, tiny 36-layer Qwen3-shaped teacher/student, sequence
  length `4`, prompt `[0,2)`, completion `[2,4)`, completion mask
  `[0,1,1,0]`, smoke temperature `2.0`, SGD `lr=1e-2`, two steps.
- Measurements: step 1 loss `0.00010941564687527716`, gradient norm
  `0.00015089756434509636`; step 2 loss `0.00010947752161882818`, gradient
  norm `0.00015089709935220638`. Both losses/gradients were finite and both
  steps changed router parameters.
- Instrumentation: 72 route records covered 36 attention and 36 FFN units
  exactly once; checkpoint probability and hard-route round trips matched;
  teacher and packed student base remained frozen and unchanged.
- Limitation: all values and settings are smoke-only evidence, not final
  router-training hyperparameters or routing-quality results.

## S07-B single real router-distillation run (2026-08-11)

- Locked configuration: `configs/s07_router_training.json`; implementation
  choices are also recorded as D026 in `docs/DECISIONS.md`.
- Dataset: `Salesforce/wikitext`, `wikitext-2-raw-v1`, revision
  `b08601e04326c79dfdd32d625aee71d232d685c3`, train split offsets
  `[0,1000,2000,3000]`, validation split offsets `[0,1000]`; four train and
  two validation examples. The pinned Qwen3-4B tokenizer revision is
  `1cfa9a7208912126459214e8b04321603b3df60c`.
- Preprocessing: no special tokens, first 64 tokens retained, explicit prompt
  `[0,32)` and completion `[32,64)` ranges, causal completion mask `[31,63)`;
  rows shorter than 64 tokens are skipped. No generated targets were used.
- Training: seed 1729, batch size 1, gradient accumulation 1, one epoch/four
  steps, AdamW `lr=1e-3`, weight decay 0, no scheduler, KD temperature 2.0,
  routing temperature 1.0. Teacher logits were precomputed under `no_grad`
  before optimization to fit the resident packed student; this does not add a
  loss term.
- Command:
  `source ~/.venv/bin/activate && which python && python --version && PYTHONPATH=src:third_party/any-precision-llm QAQ_MODEL_DEVICE=cuda:3 python scripts/run_s07b.py`.
- Training result: KD loss `0.1730574965` → `0.0317778103`; all four losses,
  router gradients, probabilities, and route logs were finite. The optimizer
  audit contained only 23,620,752 router scalars. Packed-student non-router
  hashes matched before/after. The final router-only checkpoint is external
  to Git at `~/.cache/qaq/s07b/final_router.pt`, SHA-256
  `08bf646f19759c0d7949e159bdbe4f96bbea737204b96f8760d205c8d6fd1949`.
- Evaluation: static-4/static-8 mean absolute logit errors were
  `0.7434162199`/`0.0910567641`; soft was `0.2430240735` with KD
  `0.0386699643`; hard was `0.2928081304` with KD `0.0631424394`. Hard
  fractions were 4-bit `0.2013889` and 8-bit `0.7986111`; attention was
  4/8=`0.1666667`/`0.8333333`; FFN was 4/8=`0.2361111`/`0.7638889`.
  There were two unique hard maps and complete 72-unit logs per request.
- Determinism command:
  `source ~/.venv/bin/activate && which python && python --version && PYTHONPATH=src:third_party/any-precision-llm python scripts/verify_s07b_roundtrip.py`.
  Fresh-process probability/route reload and fixed-subset hard-route/logit
  repeats passed bitwise.
- Gate: engineering is **REVISE**, because the completed run did not set
  teacher parameters to `requires_grad=False` before the freeze audit. The
  teacher was still outside the optimizer, evaluated under `no_grad`, and
  unchanged. Query-adaptivity classification is `OTHER`; query-adaptive
  behavior was not demonstrated. No S08 work was started.
- Result artifact: `docs/results/s07_router_training.json`.

## S04 explicit manual routing (2026-08-11)

- Scope: one resident S03-B nested checkpoint, explicit immutable 36-layer
  attention/FFN plans, no query features, learned router, request-specific
  route generation, or on-demand loading.
- Model and backend: `Qwen/Qwen3-4B` revision
  `1cfa9a7208912126459214e8b04321603b3df60c`; Any-Precision commit
  `a3257d02740cc5757c78673da534b0630ff3a4ea`; artifact and complete hashes are
  recorded in `docs/quantized_model_manifest.json`.
- Primary test command:
  `source ~/.venv/bin/activate && which python && python --version && QAQ_S03_ARTIFACT=<artifact> QAQ_MODEL_DEVICE=cuda:3 pytest -q tests/integration/test_s04_manual_routing.py`.
  Result: `8 passed in 425.13s`; Ruff passed for the S04 files.
- Measurement environment: CUDA `cuda:3`, NVIDIA GeForce RTX 3090, deterministic
  S03 smoke inputs, `use_cache=False`; manual and static models were loaded in
  one process. Model/checkpoint loading took `415.39432963589206` seconds.
- Plan contract: frozen `PrecisionPlan` with `attention_bits` and `ffn_bits`
  as exactly 36-entry tuples; only 4 and 8 are accepted. Canonical JSON
  serialization round-tripped an alternating plan; the unit validation and
  serialization tests passed (`7 passed`).
- Numerical parity against the underlying S03 static logits used `atol=1e-3`
  and `rtol=1e-3`. All-4 and all-8 manual outputs were bitwise equal to their
  static outputs: mean/max absolute error `0.0` / `0.0` for both. The manual
  digests were the recorded S03 digests `8b28d8ae1cf0d27462b0704d2661ebe90f67073c4435bbd8e21ad2ef19a6aa5d`
  (4-bit) and `9337bad41bf1f9294aca8ba7721a313ad5abfe14e279970e2cf45142946f04c3`
  (8-bit). Each trace contained 252 exact calls.
- Isolation: changing only layer-7 attention from 4 to 8 changed exactly its
  four `q_proj`, `k_proj`, `v_proj`, and `o_proj` calls; changing only layer-19
  FFN from 4 to 8 changed exactly its three `gate_proj`, `up_proj`, and
  `down_proj` calls. Both changed final logits and both complete traces matched
  their expected scope.
- Mixed plans: attention-8/FFN-4, attention-4/FFN-8, and the deterministic
  even-layer attention-4/FFN-8 / odd-layer attention-8/FFN-4 plan were finite,
  bitwise repeatable, and exact 252-call trace matches.
- Leakage: in one process, all-4 → all-8 → all-4 → attention-8/FFN-4 → all-8
  reproduced the earlier all-4 and all-8 outputs exactly. No sequential plan
  state leakage was observed.
- Regression: the artifact-supplied S03 static/checkpoint selection passed
  `13 tests`; the unit/S01/S02 selection passed `43 tests` with five expected
  artifact-path skips when run without `QAQ_S03_ARTIFACT`. No production packed
  representation or pinned upstream source was changed.
