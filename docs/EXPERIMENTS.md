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

## S09-A protocol freeze and closeout (2026-08-12)

S09-A freezes the final comparison protocol before any S09-B result. The
machine-readable owner is `configs/baseline_evaluation.json`, and fixed request
inputs are in `configs/baseline_evaluation_prompts.json`. The detailed procedure and
validation gate are maintained in
`docs/stages/S09_BASELINE_FREEZE.md`.

PR #5 landed the protocol and validator corrections at merge commit
`0f5802a777983c210b6f65ca26fd55368f49bf51`. The frozen configuration and fixed
inputs were unchanged during closeout. Its protocol/config SHA-256 is
`01ca65c6b3b7e16d7af66f1533140b1c9f31749c90bc91e097d096d463bf2e1c`.

The canonical full validation command was:

```text
source ~/.venv/bin/activate && which python && python --version && python scripts/validate_baseline_evaluation_protocol.py --config configs/baseline_evaluation.json
```

It exited `0` with hashes enabled. The packed artifact SHA-256 matched
`29d9bc526b3da0bd39daf2f82afd141f82d005ca1232cabc75cfe9d9ecc1cfee`; the S07
router checkpoint matched
`08bf646f19759c0d7949e159bdbe4f96bbea737204b96f8760d205c8d6fd1949`; the
Qwen3-4B model and tokenizer matched revision
`1cfa9a7208912126459214e8b04321603b3df60c`; and the Any-Precision submodule
matched `a3257d02740cc5757c78673da534b0630ff3a4ea`.

The focused S09-A command was:

```text
source ~/.venv/bin/activate && which python && python --version && PYTHONPATH=src:. pytest -q tests/unit/test_baseline_evaluation_protocol.py tests/integration/test_baseline_evaluation_protocol_inputs.py tests/integration/test_perplexity_evaluator.py
```

It passed `18 passed`. Ruff also passed with:

```text
source ~/.venv/bin/activate && which python && python --version && ruff check scripts/validate_baseline_evaluation_protocol.py tests/unit/test_baseline_evaluation_protocol.py tests/integration/test_baseline_evaluation_protocol_inputs.py tests/integration/test_perplexity_evaluator.py
```

S09-A is **COMPLETE**. No S09-B benchmark, five-mode baseline evaluation,
final result artifact, or final comparison number was produced. S09-B
execution machinery is **MISSING**: no complete executable S09-B runner exists
in the repository beyond the non-benchmark protocol validator.

Next action: Implement the minimal S09-B evaluation runner required to execute
the frozen `configs/baseline_evaluation.json` contract, without running the final
evaluation yet.

## S09-B1 runner implementation (2026-08-12)

This work unit implemented the non-benchmark S09-B runner without changing
`configs/baseline_evaluation.json` or `configs/baseline_evaluation_prompts.json`.
The frozen protocol/config SHA-256 remained
`01ca65c6b3b7e16d7af66f1533140b1c9f31749c90bc91e097d096d463bf2e1c`.

Executed commands:

```text
source ~/.venv/bin/activate && which python && python --version && python scripts/validate_baseline_evaluation_protocol.py --config configs/baseline_evaluation.json
source ~/.venv/bin/activate && which python && python --version && PYTHONPATH=src:. pytest -q tests/unit/test_baseline_evaluation_runner.py tests/integration/test_baseline_evaluation_runner_plan.py
source ~/.venv/bin/activate && which python && python --version && ruff check src/qaq/evaluation/runner.py scripts/run_baseline_evaluation.py tests/unit/test_baseline_evaluation_runner.py tests/integration/test_baseline_evaluation_runner_plan.py
source ~/.venv/bin/activate && which python && python --version && python scripts/run_baseline_evaluation.py --plan --config configs/baseline_evaluation.json --results-dir /tmp/qaq-s09b-plan
```

The validator passed with all five modes, seven fixed requests, 32 samples,
and 4096 evaluated target tokens. The runner tests passed `8 passed`, Ruff
passed, and the plan resolved five fresh-process child commands and one
aggregation command while reporting no model loading, CUDA inference,
benchmark, or final-result write.
No S09-B mode child was launched and no final S09 result artifact was created.
The next action is the frozen five-mode S09-B execution, which remains
unexecuted.

## S09-B1R runner correctness repair (2026-08-12)

S09-B1R corrected the pre-execution runner without changing either frozen JSON
file or creating any S09-B result. The on-demand result path now retains the
measured cleanup records, derives cleanup summaries from those records, and
records the S08 CPU-authority/hidden-copy audit. Every mode records actual
physical packed-buffer residency rather than `None`. All fixed requests retain
all five raw prefill/decode/end-to-end timings and median headlines, while
repeat evidence proves fixed-input identity, finite outputs, deterministic
generated outputs, and repeated hard-route agreement. Result validation now
requires exact S09 perplexity setup and 4096 target tokens, exact packed and
router identities where applicable, fixed RTX 3090 hardware comparability, and
measured deterministic evidence. Aggregation writes its structured
classification to the advertised `aggregation.json` path.

The focused repair tests include a mocked complete on-demand result
serialization path that exercises the former `retained_before_cleanup`
KeyError boundary, plus negative aggregation tests for cleanup, hidden-copy,
latency, hardware, perplexity, and deterministic evidence. No model was loaded,
no CUDA benchmark ran, no real S09-B evaluation ran, and no S09-B result
artifact exists.

## S03-B nested Qwen3 static baseline (2026-08-11)

- Command: `source ~/.venv/bin/activate && which python && python --version && PYTHONPATH=src:third_party/any-precision-llm python scripts/build_packed_model.py --overwrite-artifact`.
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
- Fixed prompt file: `configs/static_quality_prompts.txt`, five prompts. Tokenizer revision `1cfa9a7208912126459214e8b04321603b3df60c`; `add_special_tokens=true`, truncation `128`, no padding. Reference prompt remained `QAQ full-precision smoke test.`.
- Quality command: `source ~/.venv/bin/activate && which python && python --version && PYTHONPATH=src:third_party/any-precision-llm QAQ_MODEL_DEVICE=cuda:3 python scripts/evaluate_static_model_quality.py`.
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
- Focused command: `source ~/.venv/bin/activate && which python && python --version && pytest -q tests/unit/test_router_network.py tests/integration/test_soft_routing.py`; result `13 passed`.
- Pinned backend command: `source ~/.venv/bin/activate && which python && python --version && pytest -q tests/integration/test_soft_routing_packed_endpoints.py -k 'not qwen3'`; result `2 passed`.
- Full artifact command: `source ~/.venv/bin/activate && which python && python --version && QAQ_S03_ARTIFACT=<S03-B artifact> QAQ_MODEL_DEVICE=cuda:3 pytest -q tests/integration/test_soft_routing_packed_endpoints.py`; result `3 passed in 419.02s` on CUDA device 3, NVIDIA GeForce RTX 3090.
- Endpoint result: forced `[1,0]` and `[0,1]` mixtures matched the real hard 4-bit and hard 8-bit Qwen3 paths within `atol=1e-3`, `rtol=1e-3`; the synthetic pinned-backend endpoint comparisons were bitwise equal.
- Probability result: one and batched router outputs had shapes `[2]` and `[3,2]`; finite non-negative probabilities summed to one within `1e-6` in the router and `1e-5` at the packed boundary.
- Temperature result: fixed logits `[2,0]` were more concentrated toward 4-bit at temperature `0.5` than at `2.0`.
- Sharing result: each attention unit recorded one shared probability tensor across q/k/v/o; each FFN unit recorded one shared tensor across gate/up/down.
- Gradient result: finite nonzero router gradients reached the soft mixture; a one-step SGD smoke check changed router parameters only. No real training or quality claim was made.
- Regression command: `source ~/.venv/bin/activate && which python && python --version && pytest -q tests/unit`; result `67 passed`.
- Artifact regression command: `source ~/.venv/bin/activate && which python && python --version && QAQ_S03_ARTIFACT=<S03-B artifact> QAQ_MODEL_DEVICE=cuda:3 pytest -q tests/integration/test_manual_routing.py tests/integration/test_query_routing_manual_policy.py tests/integration/test_static4_forward.py tests/integration/test_static8_forward.py`; result `12 passed in 431.49s`.

## Boundaries before baseline freeze

Do not introduce asynchronous transfers, prefetching, transfer prediction, bit-width cost penalties, cross-request caching, multi-query batching, or unrelated research improvements.

## S07-A router-distillation smoke (2026-08-11)

- Scope: reusable masked teacher-student KD, explicit prompt/completion
  example and alignment contracts, frozen teacher/S06 packed base, explicit
  router-only optimizer, router-only checkpoint, deterministic hard routes,
  route logs, and observational statistics. No real dataset or baseline-scale
  training was run.
- Command: `source ~/.venv/bin/activate && which python && python --version && PYTHONPATH=src pytest -q tests/unit/test_router_distillation.py tests/integration/test_router_distillation_smoke.py`.
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

## S07-B first real router-distillation run (2026-08-11; D027 defect)

- Locked configuration: `configs/baseline_router_training.json`; implementation
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
  `source ~/.venv/bin/activate && which python && python --version && PYTHONPATH=src:third_party/any-precision-llm QAQ_MODEL_DEVICE=cuda:3 python scripts/train_baseline_router.py`.
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
  `source ~/.venv/bin/activate && which python && python --version && PYTHONPATH=src:third_party/any-precision-llm python scripts/verify_router_checkpoint_roundtrip.py`.
  Fresh-process probability/route reload and fixed-subset hard-route/logit
  repeats passed bitwise.
- Gate: engineering is **REVISE**, because the completed run did not set
  teacher parameters to `requires_grad=False` before the freeze audit. The
  teacher was still outside the optimizer, evaluated under `no_grad`, and
  unchanged. Query-adaptivity classification is `OTHER`; query-adaptive
  behavior was not demonstrated. No S08 work was started.
- Result artifact: `docs/results/s07_router_training.json`.

## S07-B corrected D008-1 router-distillation rerun (2026-08-11)

- Authorization: exactly one corrected rerun using the unchanged D026 locked configuration; no S08 work and no new objective were introduced.
- Production correction: `scripts/train_baseline_router.py` invokes the audited teacher/packed-student freeze seam before teacher-logit precomputation and records teacher before/after parameter hashes plus gradient absence. The first run's D027 defect remains documented above.
- Command: `source ~/.venv/bin/activate && which python && python --version && PYTHONPATH=src:third_party/any-precision-llm QAQ_MODEL_DEVICE=cuda:3 python scripts/train_baseline_router.py`.
- Freeze evidence: teacher `requires_grad=False`, no teacher gradients, matching teacher hashes, unchanged packed-student non-router hashes, router-only optimizer with 23,620,752 scalars, finite router gradients, and changed router parameters.
- Training: four finite KD losses, `0.1730574965` → `0.0317778103`; the corrected values exactly matched the prior run, so no material numerical difference was found.
- Evaluation: soft KD/error `0.0386699643`/`0.2430240735`; hard KD/error `0.0631424394`/`0.2928081304`; static 4/8-bit errors `0.7434162199`/`0.0910567641`; hard 4/8 fractions `20.1389%`/`79.8611%`; attention 4/8 `29.1667%`/`70.8333%`; FFN 4/8 `11.1111%`/`88.8889%`; two unique route maps; prompt distance `0.0138889`; classification `OTHER`; complete 72-unit logs for each of two validation requests.
- Objective: exactly `T^2 * masked KL(teacher || student)` over completion targets; no width, cost, latency, transfer, entropy, sparsity, balance, or auxiliary routing term.
- Fresh-process command: `source ~/.venv/bin/activate && which python && python --version && PYTHONPATH=src:third_party/any-precision-llm python scripts/verify_router_checkpoint_roundtrip.py`; checkpoint reload and deterministic hard-route repeat passed bitwise.
- Result artifact: `docs/results/s07_router_training.json`; S07 engineering gate is **CONTINUE**. The repository-defined next action is S08, but it was not executed.

## S04 explicit manual routing (2026-08-11)

- Scope: one resident S03-B nested checkpoint, explicit immutable 36-layer
  attention/FFN plans, no query features, learned router, request-specific
  route generation, or on-demand loading.
- Model and backend: `Qwen/Qwen3-4B` revision
  `1cfa9a7208912126459214e8b04321603b3df60c`; Any-Precision commit
  `a3257d02740cc5757c78673da534b0630ff3a4ea`; artifact and complete hashes are
  recorded in `docs/quantized_model_manifest.json`.
- Primary test command:
  `source ~/.venv/bin/activate && which python && python --version && QAQ_S03_ARTIFACT=<artifact> QAQ_MODEL_DEVICE=cuda:3 pytest -q tests/integration/test_manual_routing.py`.
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

## S08-A synchronous packed-plane loader fixture (2026-08-11)

- Scope: establish and test the request-scoped synchronous loader contract only.
  No Qwen3-4B on-demand evaluation, allocator/memory comparison, latency
  comparison, asynchronous transfer, prefetching, or S09 work was run.
- Fixture: the real S01 pinned Any-Precision CUDA fixture, seed `1729`,
  `M=4`, `N=64`, `K=1024`, input/LUT/output `float16`, and physically packed
  `int32` parent qweight `[8,64,32]`; the loader source was copied into CPU
  authoritative buffers and the resident module remained the reference.
- Commands:
  `source ~/.venv/bin/activate && which python && python --version &&`
  `PYTHONPATH=src:third_party/any-precision-llm pytest -q`
  `tests/unit/test_synchronous_loading_contract.py`
  `tests/integration/test_synchronous_loading_transfer.py`
  `tests/integration/test_synchronous_loading_request_lifetime.py`
  Result: `8 passed in 8.77s` on CUDA device `cuda:0`, NVIDIA GeForce RTX 3090.
  Ruff passed for the changed S08 source, request-state source, and focused
  tests.
- CPU authority: qweight `[8,64,32]` `torch.int32`, LUT4 `[64,16]` and LUT8
  `[64,256]` `torch.float16`, all contiguous and CPU-resident before first use.
- First-use accounting: 4-bit transferred `qweight[:4]` (`32,768` bytes) and
  `lut4` (`2,048` bytes), total `34,816`; fresh 8-bit transferred `qweight[:8]`
  (`65,536`) and `lut8` (`32,768`), total `98,304`. A 4-to-8 upgrade transferred
  only `qweight[4:8]` (`32,768`) and `lut8` (`32,768`), total `65,536`.
  Reuse transferred `0` bytes. Every event's count was the sum of actual
  destination tensor `numel()*element_size()` values.
- Lifecycle/isolation: explicit `QaqRequestState.end_request()` reduced the
  retained GPU entry and buffer counts to zero. Two independent state objects
  with the same textual request ID each recorded an independent first-use
  transfer and had distinct state identities.
- Correctness: on-demand 4-bit and 8-bit outputs were finite and bitwise equal
  to the existing resident pinned execution. Copies used ordinary synchronous
  `.to(device=...)` followed by `torch.cuda.synchronize`; no non-blocking copy,
  stream, future, worker, cache, or prefetch path exists.
- Decision: implementation details are recorded as D029 in
  `docs/DECISIONS.md`. This evidence supports S08-A **CONTINUE** only.

## S08-B real Qwen3 hard-routed on-demand integration (2026-08-11)

- Scope: complete S08-B only; no S09 baseline comparison or execution.
- Interruption handling: the prior worker ended with an external Codex service-overload error after three retries. This was not treated as a QAQ defect, and no completed S08-B evidence was discarded or rerun solely for reproduction.
- Inputs: the two locked S07 validation requests `validation-3` and `validation-1000`, with input-token digests recorded in `docs/results/s08_on_demand.json`.
- Model and revisions: `Qwen/Qwen3-4B` revision `1cfa9a7208912126459214e8b04321603b3df60c`, Any-Precision revision `a3257d02740cc5757c78673da534b0630ff3a4ea`, packed checkpoint SHA-256 `29d9bc526b3da0bd39daf2f82afd141f82d005ca1232cabc75cfe9d9ecc1cfee`, and router checkpoint SHA-256 `08bf646f19759c0d7949e159bdbe4f96bbea737204b96f8760d205c8d6fd1949`.
- Environment: `/nfs/home/s314511048/.venv/bin/python`, Python `3.12.3`, PyTorch `2.2.2+cu121`, CUDA runtime `12.1`, NVIDIA GeForce RTX 3090, device `cuda:3`.
- CPU authority and hidden-copy audit: 252 on-demand sources remained CPU-resident, all LUTs and packed qweights were CPU-resident, zero `AnyPrecisionLinear` modules remained, and no complete packed GPU copy remained before first use.
- Route parity and logits: resident and on-demand route maps matched for both requests; both outputs were finite and bitwise equal with zero mean and maximum absolute logit difference.
- Generation: four greedy decode tokens matched exactly for both modes and both requests; route maps remained fixed during decode; on-demand decode transfer bytes were zero.
- Transfer accounting: on-demand prefill transferred `3,817,717,760` bytes for `validation-3` and `3,835,002,880` bytes for `validation-1000`. Each total matched the independently computed per-projection expected bytes. There were 252 first-use events and zero reuse transfer bytes per request.
- Phase accounting: prefill transfer equaled total transfer and decode transfer was zero for both requests.
- Cleanup and isolation: each request retained 252 projection entries, 504 packed GPU buffers, and the measured packed bytes before cleanup, then zero entries, buffers, and bytes after `end_request()`. A later fresh request transferred its buffers again.
- Memory: two synchronized measurement repeats produced maximum resident peak allocated memory of `5,724,945,408` bytes and maximum on-demand peak allocated memory of `4,806,114,304` bytes. Full allocated/reserved observations are in the result artifact.
- Latency: resident median prefill/decode/end-to-end was `0.145354`/`0.187833`/`0.332110` seconds; on-demand was `5.815631`/`0.229669`/`6.031509` seconds.
- Commands: the focused S08-B suite passed `3 tests in 438.03s`; the S08-A focused suite passed `8 tests in 8.55s`; Ruff passed for all changed S08 files. The previously recorded S08-B regression command remained valid at `8 passed in 651.74s` and was not rerun.
- Result: `docs/results/s08_on_demand.json`, including code/worktree provenance, exact artifact identities, request identities, method, transfer records, allocator readings, cleanup evidence, and commands.
- Gate: S08 **COMPLETE**; next action is S09, which was not executed.

## S09-B2 failed five-mode evidence and diagnosis

The actual S09-B2 runner execution command was:

```text
source ~/.venv/bin/activate && which python && python --version && python scripts/run_baseline_evaluation.py --execute --config configs/baseline_evaluation.json
```

It produced the historical five-mode evidence under `docs/results/s09b/`.
The committed B2 aggregation command and result were:

```text
source ~/.venv/bin/activate && which python && python --version && python scripts/run_baseline_evaluation.py --aggregate --config configs/baseline_evaluation.json --results-dir docs/results/s09b
```

The result was `REVISE` with `deterministic repeat evidence is incomplete`.
The quality, route, transfer, cleanup, and generated-token parity observations
were otherwise sensible, but routed decode logits were not bitwise
reproducible/equivalent. B3 isolated nondeterminism in the pinned atomic
k-split `matmul_kbit` path at the real routed decode shape.

## S09-B4 repair and S09-B5 targeted rerun

B4 changed only the routed packed execution family proven by B3: the exact
non-Orin, one-row, packed `K > 4096`, precision-at-least-7 dispatch uses pinned
`dequant_kbit` plus `torch.matmul`; all other packed calls retain the pinned
kernel. The Any-Precision source, frozen protocol, fixed inputs, and B2 JSON
files were unchanged. The FP teacher and static 4/8-bit execution paths were
unchanged by the diff.

B5 invoked only the two explicit-mode runner children for
`hard_routed_resident_packed` and
`hard_routed_synchronous_on_demand_packed`, writing the targeted rerun to
`/tmp/qaq-s09b5-routed-rerun/`; the per-child command lines are not embedded in
the committed JSON and are not reconstructed here. The three unaffected
FP/static JSON results were reused without modification. The committed B5
aggregation records `CONTINUE` with `errors: []` and retains its temporary
`results_dir` as execution provenance.

Canonical final evidence is `docs/results/s09b_b5/`, committed at
`443f6994582500857afca9bad6032cc285448a86`. Its exact quality values are FP
`30.648146290315317`, static 4-bit `32.53290622283182`, static 8-bit
`30.57498909612196`, routed resident `30.678448224528175`, and routed
on-demand `30.678448224528175`. The ratios are
`static8/static4 = 0.9398173310032849` and
`routed-resident/static4 = 0.9429974689134236`.

B5 had five agreeing deterministic repeats for all seven requests in all five
modes. Resident/on-demand route maps, generated token IDs, and logits digests
matched. Route diversity was five unique maps, four changed units,
`0.05555555555555555` changed fraction, mean pairwise distance
`0.022486772486772486`, classification `OTHER`, with per-request fractions in
`docs/stages/S09_BASELINE_FREEZE.md`.

Peak allocated/reserved bytes were FP
`8125394944/8355053568`, static 4-bit
`5622764544/5811208192`, static 8-bit
`5622764544/5811208192`, routed resident
`5726520832/5918162944`, and routed on-demand
`4886706176/5167382528`. Physical packed residency was zero for FP, zero for
on-demand, and `4234936320` for each packed resident mode. On-demand peak
request-owned packed bytes were `3900211200`.

On-demand actual physical transfer was `134138675200` bytes, independently
expected bytes were `134138675200`, and decode transfer was zero. Cleanup
returned 252 entries, 504 buffers, and the request-owned bytes to zero; the
hidden-copy audit passed.

Exact final five-repeat latency medians are recorded in the S09 stage document;
they are retained directly from the B5 JSON without speed-gate interpretation.
The routed latency changed after the deterministic fallback.

## S09-C closeout validation

The frozen protocol validation command was:

```text
source ~/.venv/bin/activate && which python && python --version && python scripts/validate_baseline_evaluation_protocol.py --config configs/baseline_evaluation.json
```

It passed with the frozen protocol SHA-256
`01ca65c6b3b7e16d7af66f1533140b1c9f31749c90bc91e097d096d463bf2e1c` and fixed
input SHA-256
`da1d33f0f2330cfc341c38945fe4b205f946223f8c9069c35d44999d400fbb49`.

A new temporary copy was made at `/tmp/qaq-s09-closeout-verify`; copied JSON
hashes matched before aggregation. The read-only closeout command was:

```text
source ~/.venv/bin/activate && which python && python --version && python scripts/run_baseline_evaluation.py --aggregate --config configs/baseline_evaluation.json --results-dir /tmp/qaq-s09-closeout-verify
```

It returned `CONTINUE` with `errors: []`. Committed B2 and B5 JSON hashes were
rechecked afterward and were unchanged. The focused non-benchmark closeout
command was:

```text
source ~/.venv/bin/activate && which python && python --version && PYTHONPATH=src:. pytest -q tests/unit/test_baseline_evaluation_runner.py tests/integration/test_baseline_evaluation_runner_plan.py tests/unit/test_baseline_evaluation_protocol.py tests/integration/test_baseline_evaluation_protocol_inputs.py tests/integration/test_perplexity_evaluator.py
```

Result: `28 passed in 3.71s`. No GPU benchmark, model mode, production code,
test, config, frozen input, pinned dependency, or result JSON was changed.
S09 status is **COMPLETE**. No later stage is defined; the next action is to
stop and define an explicit post-baseline stage and decision before any
optimization or additional research mechanism.

## S10-H2-B attempt 1 and S10-H2-BR1 repair evidence

The production contract and stage gate are owned by
[`docs/stages/S10_6BIT_ROUTING.md`](stages/S10_6BIT_ROUTING.md). This section
owns the attempt and repair evidence.

Attempt 1 began from `87786fe6e549fdc279ab545be86c00745a144649` and returned
**REVISE before training**. The actual S07 selection helper returned a
`DistillationExample`, but production `QwenRuntime.prepare()` attempted
`example["example_id"]` at the frozen order check. The preserved traceback was
`TypeError: 'DistillationExample' object is not subscriptable`; the failed-run
log SHA-256 is
`a0e4e9ab696884e20e96db4e792b196428fc514658f44d82f056bde4d756c283`.
The injected H2-A runtime had masked this production-only consumer. No trial,
result, teacher or packed-model load, CUDA model execution, optimizer update,
or router training completed, and no router/lambda conclusion or Git discard
followed. The prior named Herdr lab was intentionally torn down.

BR1 reproduced the defect without model work using actual `_select_examples()`,
a one-row in-memory dataset, deterministic fake tokenizer, and CPU tensors:
the selected object type was `DistillationExample`, `.example_id` returned
`train-0`, and dictionary subscription raised the same TypeError. The first
regression failed for that exact reason. The repair now consumes IDs only via
the attribute, validates non-empty strings, and fails missing, empty,
non-string, dictionary-substitute, or reordered values with structured REVISE.

Executed repair-only checks, each after the mandatory environment/GPU
preflight:

```text
PYTHONPATH=src:. pytest -q tests/unit/test_broader_router_validation_executor.py -k 'example or prepare or selection or order'
8 passed, 5 deselected

PYTHONPATH=src:. pytest -q tests/unit/test_broader_router_validation_executor.py tests/unit/test_broader_router_validation.py
46 passed

PYTHONPATH=src:. pytest -q tests/unit/test_router_distillation.py tests/unit/test_broader_router_validation_executor.py
21 passed

PYTHONPATH=src:. pytest -q tests/unit/test_broader_router_validation_protocol.py tests/unit/test_router_frontier_confirmation.py tests/unit/test_router_frontier_protocol.py tests/unit/test_router_cost_calibration.py
134 passed, 1 existing warning

PYTHONPATH=src:. pytest -q tests/unit
309 passed, 1 existing warning

python scripts/run_broader_router_validation.py --plan --config configs/broader_router_validation.json
ruff check src/qaq/router/s10h_executor.py tests/unit/test_broader_router_validation_executor.py
git diff --check
```

The plan and static checks passed without execution. This BR1 task ran no
`--execute` command or real preparation smoke and created neither S10-H result
path. The frozen S10-G config, both historical S10-F results
(`docs/results/s10f_frontier_confirmation.json` and
`docs/results/s10f_frontier_confirmation_rerun.json`),
`docs/quantized_model_manifest.json`, `scripts/run_broader_router_validation.py`, the distillation,
router-network, soft-model, and request-state sources, the packed artifact
bytes, and the clean pinned Any-Precision checkout were preserved byte-for-byte.
S10-H quality remains unknown. No new implementation choice was introduced,
so `docs/DECISIONS.md` is unchanged. The next experiment action is separate
S10-H2-B2 authorization from the repaired, reviewed, merged commit.

## S10-H2-B2 repaired canonical broader-validation retry (2026-08-19)

This is operational attempt 2 and the one consumed repaired B2 retry. Attempt 1
at `87786fe6e549fdc279ab545be86c00745a144649` remains **REVISE before
training**; the BR1 evidence above remains unchanged. The execution commit is
`b1aca71bcc584f0e3559e5fe7caf142c2f750db3`, PR #15's merged repair.

### Reconciled preflight

The focused collection selected exactly these nine nodes and deselected five:

```text
tests/unit/test_broader_router_validation_executor.py::test_selected_examples_accept_exact_train_and_validation_order_without_subscription
tests/unit/test_broader_router_validation_executor.py::test_selected_examples_reject_reordered_frozen_ids[train]
tests/unit/test_broader_router_validation_executor.py::test_selected_examples_reject_reordered_frozen_ids[validation]
tests/unit/test_broader_router_validation_executor.py::test_ordered_example_ids_reject_invalid_attribute_contract[example0-not a DistillationExample]
tests/unit/test_broader_router_validation_executor.py::test_ordered_example_ids_reject_invalid_attribute_contract[example1-not a DistillationExample]
tests/unit/test_broader_router_validation_executor.py::test_ordered_example_ids_reject_invalid_attribute_contract[example2-not a DistillationExample]
tests/unit/test_broader_router_validation_executor.py::test_ordered_example_ids_reject_invalid_attribute_contract[example3-not a DistillationExample]
tests/unit/test_broader_router_validation_executor.py::test_ordered_example_ids_reject_invalid_attribute_contract[example4-dictionary substitute]
tests/unit/test_broader_router_validation_executor.py::test_qwen_prepare_retains_real_selection_order_and_distillation_objects
```

The captain-reconciled outcomes at exact `b1aca71...` were `9 passed, 5
deselected`; combined S10-H executor/validator `47 passed`; S07 distillation
plus executor `22 passed`; predecessor protocols `134 passed` with only the
established duplicate-optimizer warning; and full unit suite `310 passed` with
that same warning. Ruff on the four runner/executor/test files and
`git diff --check` passed. The plan command exited `0` and loaded, trained,
evaluated, and wrote nothing. These commands ran before the one-time execution
boundary; the final closeout reran none of them.

The config, manifest, packed artifact, model/tokenizer, dataset, backend, and
historical hashes were respectively
`fcb66902174558e5d3f9198f34a8430b685568fd4e21e1632b40f6870aa4aec7`,
`1e2b3515072e22d71ac35a35a3002e3a1dcd5ce44887c554b1408f735c928530`,
`29d9bc526b3da0bd39daf2f82afd141f82d005ca1232cabc75cfe9d9ecc1cfee`,
`1cfa9a7208912126459214e8b04321603b3df60c`,
`b08601e04326c79dfdd32d625aee71d232d685c3`,
`a3257d02740cc5757c78673da534b0630ff3a4ea`,
`d68f041e0a3dc32c465e8b8068ca3ab230d39253757e30f3019ca7e681b14233`, and
`b3bcc0e45d45852ac5060209c4789453ed452462f528f7bffd4cb80fb1ef58cb`.
The existing logical artifact was verified rather than reprovisioned; no
artifact override was set.

### Consumed execution

The execution started `2026-08-19T17:45:47Z` in Herdr session
`fm-lab-qaq-s10h2b2-2026-1101046-3135`, workspace `w1`, tab `w1:t1`, pane
`w1:p1`. Python was `/nfs/home/s314511048/.venv/bin/python` 3.12.3. The selected
GPU was `cuda:0`, physical index 0, NVIDIA GeForce RTX 3090,
UUID `GPU-384b6377-8f0c-e3d2-8b3a-b3408b54fd53`; mapping was direct with
`CUDA_VISIBLE_DEVICES` unset. No compute process was present. The exact command
was run once:

```text
set -o pipefail
PYTHONPATH=src:third_party/any-precision-llm:. \
python scripts/run_broader_router_validation.py \
  --execute \
  --device "${QAQ_S10H_DEVICE}" \
  --config configs/broader_router_validation.json \
  --output "${QAQ_S10H_CANDIDATE}" \
  2>&1 | tee "${QAQ_S10H_LOG}"
execution_status=$?
printf '%s\n' "${execution_status}" > /tmp/qaq-s10h2b2-execution.exitcode
```

It exited `0` and the runner reported `REFINE`, no errors, and `written=true`.
Log `/tmp/qaq-s10h2b2-execution.log` hashes to
`1f3da7860eb44dd7f710762d2be41357deb8af8fd2ecb6c4a37e12c006e04f55`.
The candidate hashed to
`7d9e0aff3b686570be0d1d57b5513ee921d60bd5470f275b0cd7cbb4fd63db20`.
No smoke, retry, resume, alternate device, monitoring loop, profiler, or
modified trial/data/training setting was used.

### Result and closeout

The exact trial order was `(1729,0.0)`, `(1729,0.03)`, `(1729,0.1)`,
`(1730,0.0)`, `(1730,0.03)`, `(1730,0.1)`, `(1731,0.0)`, `(1731,0.03)`,
`(1731,0.1)`. Every trial completed 24 examples and optimizer updates, retained
24 ordered history records, passed finite-gradient/loss, optimizer, teacher/base
freeze, twelve-map, reproducibility, collapse, inherited-regression, and
prohibited-work audits. The complete per-trial measurements and aggregate
interpretation are recorded in the S10 stage document and canonical JSON.

The execution shell's expected-candidate clean-status PAUSE and the first
closeout pane's wrong commit-path assertion PAUSE are preserved as
orchestration incidents; neither reran or invalidated the experiment. The
captain-authorized final closeout used control directory
`/tmp/qaq-s10h2b2-closeout.7vNem5` and session
`fm-lab-qaq-s10h2b2-clos-1146146-961`. It read only
`payload.get("ancestry", {}).get("commit")`, independently obtained `REFINE`
with no errors, passed a 3,306-check/zero-error full audit, then promoted with a
same-directory no-overwrite `os.link`, verified byte identity, and unlinked the
candidate. Independent canonical validation again returned `REFINE` with no
errors. Canonical result:
`docs/results/s10h_broader_validation.json`, SHA-256
`7d9e0aff3b686570be0d1d57b5513ee921d60bd5470f275b0cd7cbb4fd63db20`.

Aggregate observations are median hard KL
`{0.0: 0.01439695991575718, 0.03: 0.028918379141638677, 0.1: 0.07732601106787722}`;
median hard width
`{0.0: 7.643518518518518, 0.03: 7.1342592592592595, 0.1: 6.150462962962963}`;
`0.03` frontier membership `{1729: true, 1730: true, 1731: true}` (`3/3`);
paired median hard-KL delta `0.014972516723598044`; paired median hard-width
delta `-0.4907407407407405`; and zero reproducibility failures. The positive
paired hard-KL delta fails the frozen threshold, so the valid complete gate is
**REFINE**. No production lambda is selected or recommended.

Earlier PAUSE reports remain preserved at
`/tmp/qaq-s10h2b2-completion-report.md`,
`/tmp/qaq-s10h2b2-reauthorized-completion-report.md`,
`/tmp/qaq-s10h2b2-authorized-completion-report.md`,
`/tmp/qaq-s10h2b2-authorized-lab-retry-completion-report.md`, and
`/tmp/qaq-s10h2b2-closeout-completion-report.md`. The final additional-closeout
report is `/tmp/qaq-s10h2b2-final-closeout-completion-report.md`.
