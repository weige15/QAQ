# QAQ decision ledger

This ledger separates source-supported behavior from implementation choices.
Every entry below is an **implementation choice**, not a paper-established fact, unless a later source review adds direct evidence and cites the source explicitly.
Unspecified details must not be silently filled in.

## Initial implementation choices

### D001 — Any-Precision backend

**Choice:** Use the official Any-Precision LLM implementation as the selected baseline backend for nested quantization, bitplane packing, and CUDA-kernel execution.
**Status:** Resolved; S00 is complete.
**Evidence:** The clean checkout at `/tmp/qaq-any-precision-test` identifies the official upstream repository and was the source used to build the locally installed CUDA extension. The same revision was copied into `third_party/any-precision-llm` as a pinned submodule and passed the package-import and CUDA backend smoke checks.
**Source basis:** This is an implementation choice. The source papers do not by themselves establish QAQ's complete backend contract.
**Consequence:** Later QAQ backend work must use the pinned submodule revision and must not modify upstream source during this stage.
**Reversal path:** Replace the submodule only after a new compatibility probe and a new D002 provenance record establish a different exact revision.

### D002 — Pin upstream revision

**Choice:** Pin the exact Any-Precision commit before modifying or wrapping it.
**Status:** Resolved; S00 is complete.
**Evidence:** Upstream `https://github.com/SNU-ARC/any-precision-llm.git`, full commit `a3257d02740cc5757c78673da534b0630ff3a4ea`, commit date `2025-07-04T16:00:35+09:00`, branch `main`, and clean checkout status. The existing checkout's reflog records its clone from that upstream URL, and the installed extension's `direct_url.json` points to that checkout. No separate prior test log was present, so the identification relies on those local provenance records plus the successful rerun below.
**Why selected:** This is the revision that passed the local Python 3.12/CUDA compatibility probe; it is not a latest-commit substitution.
**Representation:** Git submodule at `third_party/any-precision-llm`, with `.gitmodules` recording the upstream URL and the gitlink recording the exact commit.
**Compatibility rerun:** With `source ~/.venv/bin/activate`, `which python` resolving to `/nfs/home/s314511048/.venv/bin/python`, and Python `3.12.3`, `import any_precision` passed and the real `any_precision_ext.dequant_kbit` plus `matmul_kbit` CUDA smoke check passed on an RTX 3090. No model quantization or full benchmark was run.
**Python support statement:** Python 3.12.3 compatibility is an empirical result from this machine, not an upstream support claim. The upstream README lists Python 3.11 as its prerequisite.
**Consequence:** Stateless workers must initialize the submodule and use commit `a3257d02740cc5757c78673da534b0630ff3a4ea`; floating branches are not sufficient.
**Reversal path:** Remove or replace the submodule only after preserving the current provenance and completing a separately recorded compatibility probe for the replacement.

### D003 — Supported routes

**Choice:** Initially support only 4-bit and 8-bit router candidates.
**Status:** Baseline scope.
**Source basis:** Implementation scope choice; not asserted as a limitation of the papers.

### D004 — Separate unit routes

**Choice:** Route attention and FFN separately. All projections inside one selected unit use the same precision.
**Status:** Baseline scope.
**Source basis:** Implementation choice unless a later source review establishes direct support.

### D005 — Non-quantized components

**Choice:** Keep embeddings, normalization, activations, KV cache, and output head in BF16/FP16.
**Status:** Baseline scope.
**Source basis:** Implementation choice; the source papers do not establish this exact component policy here.

### D006 — Route reuse

**Choice:** During prefill, route each attention or FFN unit using its incoming prompt hidden states; store the selected route and reuse it during decoding.
**Status:** Baseline scope.
**Source basis:** Implementation choice; exact route timing and reuse are not established by this scaffold.

### D007 — Prompt feature

**Choice:** Initially mean-pool only non-padding prompt positions for the router feature.
**Status:** Baseline scope.
**Source basis:** Implementation choice for an unspecified feature-construction detail.

### D008 — Router objective

**Choice:** Initially train using teacher-student logit distillation without a bit-width penalty.
**Status:** Baseline scope.
**Source basis:** Implementation choice; no cost penalty is permitted before baseline freeze.

### D009 — Hard routes

**Choice:** Convert soft routing to hard inference routing using argmax.
**Status:** Baseline scope.
**Source basis:** Implementation choice unless later source review finds direct support.

### D010 — On-demand storage and lifetime

**Choice:** In on-demand mode, CPU packed storage is authoritative; selected packed planes are synchronously transferred on first use and retained until that request ends.
**Status:** Baseline scope.
**Source basis:** Implementation choice; loading lifetime and authority are not established here.

### D011 — Batch size

**Choice:** Initial inference supports batch size one only.
**Status:** Baseline scope.
**Source basis:** Implementation scope choice.

### D012 — Baseline freeze boundary

**Choice:** Do not add asynchronous loading, prefetching, layer-dependency signals, token schedulers, or new router objectives before the baseline is frozen.
**Status:** Baseline boundary.
**Source basis:** Implementation-control choice, not a claim about the papers.

### D013 — Environment capture (S00, 2026-08-11)

**Observation:** The audited host has Python 3.12.3 in `/nfs/home/s314511048/.venv`, Ubuntu 24.04.4 LTS on kernel 6.8.0-124-generic, CUDA Toolkit 12.4 from `/usr/local/cuda-12.4/bin/nvcc`, GCC 12.4.0, eight NVIDIA GeForce RTX 3090 GPUs with 24,576 MiB each, 251.5 GiB RAM, and 6,185.27 GiB available disk space. The active PyTorch is 2.2.2+cu121 with CUDA available, and `transformers` 4.39.3 is present. The minimal PyTorch CUDA operation passed.

**Preliminary prerequisite results:** Python 3.11 — **FAIL** because the active interpreter is 3.12.3; CUDA Toolkit 12 or newer — **PASS**; GCC 9 or newer — **PASS**. This is only a documented-prerequisite comparison and does not establish Any-Precision or extension-build compatibility.

**Evidence:** `docs/environment.json`, generated by `source ~/.venv/bin/activate`, `which python`, `python --version`, and `python scripts/inspect_environment.py`. The earlier S00 snapshot recorded PyTorch 2.4.0+cu124 and `transformers` 5.12.1; this current audit supersedes that snapshot so the repository has one internally consistent environment record. No model was selected, downloaded, loaded, or inspected.

**Consequence:** Environment evidence is complete, but target-model selection and inspection remain required before S00 can close. Do not begin S01.

**Reversal path:** Re-run the inspection after an explicitly authorized environment change and preserve the new command output as the current snapshot.

### D014 — QAQ baseline target-model identity and immutable revision

**Source fact:** The local `papers/QAQ.pdf` reports evaluation on Qwen3-4B, Qwen3-8B, and Llama3.1-8B. The implementation plan selects the smaller reported Qwen3-4B model for the initial baseline. The paper does not identify an exact Hugging Face repository revision.

**Choice:** Select the official `Qwen/Qwen3-4B` repository for the initial QAQ baseline. Do not substitute `Qwen/Qwen3-4B-Base`, a later 2507 variant, a quantized derivative, or a community mirror.

**Pinned revision:** Resolve `main` to immutable commit `1cfa9a7208912126459214e8b04321603b3df60c`, dated `2025-07-26T03:46:39Z` by the Hugging Face commits API. The repository metadata identifies owner `Qwen`, license `apache-2.0`, and an approximate repository size of 8,060,926,626 bytes including the three weight shards.

**Implementation choice:** The pinned revision is our reproduction choice because the QAQ source does not state the authors' exact Hugging Face revision. It must not be described as the author-used revision.

**Tokenizer identity:** Use tokenizer files from `Qwen/Qwen3-4B` at the same immutable revision. The identity record names `tokenizer.json`, `tokenizer_config.json`, `vocab.json`, and `merges.txt` without downloading or storing model weights.

**Compatibility status:** The pinned backend's synthetic single-linear execution is validated in S01. Qwen3 runtime integration remains unproven; S01 intentionally uses no model or model weights.

**Consequence:** S00 is complete. Qwen3 integration remains a later-stage task and must not be inferred from this S01 synthetic backend result.

**Reversal path:** Replace the target only after recording contradictory source evidence or a separately justified model decision with a new immutable repository revision and tokenizer identity.

### D015 — Qwen3 architecture mapping and Transformers-version boundary

**Source fact:** The pinned `Qwen/Qwen3-4B` configuration declares `architectures: [Qwen3ForCausalLM]`, `model_type: qwen3`, and `transformers_version: 4.51.0`. The official Transformers `4.51.0` source at commit `0720e206c6ba28887e4d60ef60a6a089f6c1cc76` defines the Qwen3 class hierarchy and the seven standard linear projections used by this mapping.

**Observed environment fact:** The current project environment contains Transformers `4.39.3` and has no `transformers.models.qwen3` source package. It cannot instantiate Qwen3 until a later dependency/runtime validation addresses this version boundary.

**Mapping choice:** Treat `model.layers.<i>.self_attn` and `model.layers.<i>.mlp` as separate QAQ routing units. Replace only `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj`; retain embeddings, the tied LM head, all RMS norms, rotary processing, activation/gating, and KV-cache state outside packed linear replacement.

**Any-Precision status:** The pinned Any-Precision revision explicitly configures Llama, Mistral, OPT, and Phi architectures, but has no Qwen3 configuration. Qwen3 is structurally mappable to `AnyPrecisionLinear` because all seven modules are standard linear layers, have no bias, and have input dimensions divisible by 32. This is not an official Qwen3 support claim; later work must add an explicit mapping without modifying the pinned upstream source and validate it under a Transformers version containing Qwen3.

**Evidence:** `docs/model_structure.json`, `docs/QWEN3_MAPPING.md`, and `scripts/inspect_model.py`. No model object, random full-model tensors, or weight shard was loaded.

**Consequence:** The S00 architecture and mapping specification is resolved. S01 validates the pinned backend only on a synthetic linear; the Transformers runtime mismatch and explicit Qwen3 mapping remain later work.

**Reversal path:** If later source or runtime checks contradict the module paths, revise the mapping specification and keep S00 open rather than switching models silently.

### D016 — S01 synthetic backend contract and reference (2026-08-11)

**Choice:** Validate the pinned backend with one deterministic synthetic operation at `M=4`, `N=64`, `K=1024`, seed `1729`, `float16` input/LUT/output, `int32` packed storage, and `bias=False`. Use the pinned `dequant_kbit` helper plus `torch.matmul` as the reference and compare with `atol=0.05`, `rtol=0.01`; meaningful relative error uses a `0.01` reference floor.

**Evidence:** `src/qaq/s01_backend.py`, `tests/unit/test_cuda_vs_dequantized_reference.py`, and the measured report in `docs/stages/S01_BACKEND.md`. The pinned source requires `K` divisible by 32, supports `M` from 1 through 8 in `matmul_kbit`, and stores LUTs in `float16`; `M=4`, `N=64`, and `K=1024` satisfy the observed kernel alignment and exercise the four-row packed path.

**Alternatives considered:** A test-only reimplementation of dequantization was unnecessary because the pinned helper is available. A full-model or real-weight test was excluded by the S01 scope gate.

**Consequence:** The adapter delegates packing, dequantization, and matmul to the pinned source. S01 does not assert experimentally determined bit-plane order, padding, signed encoding, or serialization endianness; D017 records their later S02 resolution.

**Reversal path:** If a later pinned-source review or target hardware changes the constructor, supported range, reference behavior, or tolerance contract, return S01 to IN_PROGRESS and preserve the new measured evidence rather than loosening checks silently.

### D018 — S03-A actual Qwen3 model verification (2026-08-11)

**Choice:** Load the exact `Qwen/Qwen3-4B` revision `1cfa9a7208912126459214e8b04321603b3df60c` using Transformers `4.51.0`, BF16, and CUDA device `cuda:3`; inspect the instantiated tree before any quantization.

**Evidence:** The local snapshot contains the three pinned safetensors shards and required config/tokenizer files. The real `Qwen3ForCausalLM` has 252 target `torch.nn.Linear` modules: 144 attention and 108 FFN projections across 36 layers. Every S00 path, shape, and bias matches. The only other linear is the excluded tied `lm_head`. The embedding and output head are the same parameter object and storage pointer. A deterministic unmodified BF16 forward returned finite logits with shape `[1,8,151936]`.

**Environment consequence:** The prior environment had optional `flash-attn 2.8.3.post1`, `torchao 0.5.0`, and `torchvision 0.19.0+cu124` binaries incompatible with the recorded PyTorch `2.2.2`; they were removed so the standard Transformers Qwen3 import could run. No quantization library or quantization operation was used. The pinned Any-Precision submodule remains unchanged at `a3257d02740cc5757c78673da534b0630ff3a4ea`.

**Alternatives rejected:** Loading with Transformers `4.39.3` was impossible because it lacks Qwen3. CPU-only loading was not used because it would not represent the intended later CUDA path. No model revision, Qwen variant, attention projection, or excluded component was substituted.

**Consequence:** S03-A is CONTINUE and S03 remains IN_PROGRESS. S03-B may quantize only the verified seven projections per layer; routing and S04 remain out of scope.

**Reversal path:** If a future dependency or model revision changes the concrete tree, rerun this actual-model inspection and return S03 to REVISE before quantization.

### D019 — S03-B deterministic calibration procedure (2026-08-11)

**Choice:** Use the smallest pinned-backend calibration run that exercises its required gradient path: one sample from the pinned C4 train shard, sequence length 64, tokenizer first-64-token truncation, Python and NumPy sampling seed `1729`, and Any-Precision `random_state=1729`.

**Evidence:** The pinned `any_precision_quantization` entry point requires gradient-derived sensitivity data before seed quantization and incremental upscaling. Its `datautils` supports deterministic C4 sampling and the requested sequence length. The run completed without a large evaluation dataset or benchmark.

**Alternatives rejected:** The backend default of 100 C4 samples at sequence length 512 was unnecessarily large for this static smoke baseline. A synthetic or silently substituted calibration source would not exercise the authoritative backend data path and was rejected.

**Consequence:** The S03-B artifact is a deterministic baseline conversion, not a quality claim. Later static quality evaluation must use its own explicitly documented dataset and must not treat this one-sample calibration as sufficient evidence for S03 completion.

**Reversal path:** If the S03 static quality gate requires a larger or different calibration set, record a new dated decision and regenerate the artifact without changing the pinned model or Any-Precision revision.

### D020 — S03-C development-quality evaluation contract (2026-08-11)

**Choice:** Close S03 using five committed prompts, a deterministic 512-token WikiText-2 test sample, and two fixed greedy generation prompts. Compare static logits against the full-precision teacher with mean and maximum absolute error. Require aggregate mean 8-bit logit error to be no greater than aggregate 4-bit error. For the small language-model sample, define unexpected 8-bit degradation as perplexity greater than 110% of static 4-bit perplexity.

**Evidence:** `docs/results/s03_static_quality.json`; all three modes produced finite deterministic outputs, aggregate prompt errors were `0.5141653061` (4-bit) and `0.0578794084` (8-bit), and perplexities were `25.0522757118` (FP), `27.1193805814` (4-bit), and `24.8803626466` (8-bit). Fresh-process checkpoint reload and target coverage checks passed.

**Alternatives rejected:** A full benchmark suite was excluded because S03-C is a deliberately small development evaluation. Subjective generation quality was not used as a gate. No paper-score target was imposed because this sample is not a reproduction claim.

**Consequence:** S03 is complete and may continue to S04. The result does not establish routing, transfer, on-demand residency, or final model quality.

**Reversal path:** If a later implementation changes the model, tokenizer, checkpoint, evaluator accounting, or static representation, reopen S03 and rerun this evidence rather than carrying the result forward.

### D021 — S04 explicit plan propagation and trace boundary (2026-08-11)

**Choice:** Carry a frozen `PrecisionPlan` through a manual Qwen3 execution
wrapper into each layer's attention or FFN unit, and pass the selected `4` or
`8` value explicitly to every packed linear call. Capture debug records in a
`PrecisionTrace` supplied for that forward call; do not store a selected plan,
route, or trace on the model or in a module/global singleton.

**Evidence:** The verified Qwen3 forward source calls attention projections
inside `Qwen3Attention.forward` and calls `Qwen3MLP.forward` without a route
argument. The existing S03 path selects a static precision through the mutable
`AnyPrecisionLinear.set_precision()` state. S04 integration tests showed exact
all-4 and all-8 parity with the S03 path, 252 exact trace records for every
plan, four-call attention isolation, three-call FFN isolation, and exact
reproduction after the all-4 → all-8 → all-4 → mixed → all-8 sequence.

**Alternatives rejected:** A mutable module precision setter would make a
sequential plan leak possible; a process-global/context-local route would hide
the request boundary; and changing the pinned Transformers or Any-Precision
source would violate the S03 baseline. A wrapper carrying the unchanged
Qwen3 attention, FFN, normalization, rotary, and cache operations was selected
because it is the smallest explicit seam that reaches both attention and FFN
units without modifying upstream code.

**Consequence:** S04 supports resident manual plans only. The route interface
is intentionally not prompt-derived and does not imply request-specific state,
on-demand transfer, or a learned router. Trace records are diagnostic and do
not alter numerical operations.

**Reversal path:** If a future pinned Transformers release exposes an official
route argument through the model, decoder layer, and both units, replace the
wrapper only after rerunning the exact parity, scope, mixed-plan, and leakage
tests and recording the new source revision.

### D022 — S04 parity tolerance (2026-08-11)

**Choice:** Document `atol=1e-3` and `rtol=1e-3` for manual-versus-static
logit comparisons.

**Evidence:** On the verified S03-B artifact and deterministic smoke input,
all-4 and all-8 manual logits were bitwise equal to the S03 static logits, with
measured mean and maximum absolute errors of `0.0` for both. The tolerance is
therefore a documented guard for future backend/runtime variation, not a
substitute for the observed exact parity.

**Alternatives rejected:** Comparing generated text would not test the
underlying numerical logits. Reusing the S03 full-precision quality error
threshold would conflate static quantization error with routing propagation.

**Consequence:** A future manual wrapper or backend change that exceeds this
tolerance must return S04 to REVISE; static baselines must not be redefined.

**Reversal path:** Re-measure on the same pinned artifact and deterministic
inputs after a justified runtime change, then record a new tolerance decision
with mean/max error evidence.

### D023 — S05 feature storage and request ownership (2026-08-11)

**Implementation choice:** Use the model hidden size as the feature dimension
(2560 for the pinned Qwen3-4B model), accumulate masked prompt means in
float32, and store detached cloned one-dimensional tensors of shape
`[hidden_size]`. The attention feature is taken after the layer input RMS norm
and before any attention projection. The FFN feature is taken after the real
attention residual and post-attention RMS norm, before the first FFN
projection. The valid-token count is the denominator; all-padding prompts are
rejected.

**Implementation choice:** Require an explicit `phase` of `prefill` or
`decode`, and require an explicit batch-size-one 0/1 prompt mask for prefill.
Prefill invokes only an S04 `PrecisionPlan` adapter or a callback with
`(layer_index, unit_type, feature) -> 4|8`; decode ignores policy input and
reuses the stored route. `request_id` is metadata rather than a global lookup
key: duplicate IDs are allowed only across independent state objects, while a
concrete state object binds to one model owner and cannot be reused by another.

**Source basis:** These are implementation choices for S05 details left
unspecified by the reviewed sources. They do not assert a learned-router
design, routing probabilities, a loss, or a paper-defined feature shape.

**Consequence:** S05 remains batch-size one, computes no completion-token
features, stores no model-global route state, and adds no learned-router or
loading machinery.

**Reversal path:** If a later stage needs gradients through features or a
different feature representation, reopen S05 and preserve this deterministic
baseline as its own measured path.

### D024 — S06 soft-router parameterization (2026-08-11)

**Implementation choices:** Use one distinct MLP router for each of the 36
attention units and 36 FFN units. Each router uses parameter-free RMS feature
normalization `x / sqrt(mean(x**2) + 1e-6)`, `Linear(d_model, 128)`, GELU,
and `Linear(128, 2)`. Use a fixed configurable temperature with baseline
`1.0`, canonical output ordering `[p4, p8]`, and PyTorch Linear default
`reset_parameters()` initialization under the caller's seed.

**Source basis:** The reviewed QAQ source supports the general lightweight MLP,
precision probabilities, and temperature-controlled routing. It does not
specify this width, activation, normalization, temperature value, or
initialization. These are explicit smallest-baseline implementation choices,
not paper-established facts.

**Evidence:** S06 tests passed finite probability and shape checks, fixed-logit
temperature behavior, real pinned-backend 4-bit/8-bit endpoint parity, shared
attention/FFN probability identity, nonzero router gradients, and frozen-model
checks. The Qwen3-4B baseline has 72 routers and 23,620,752 router parameters.

**Consequence:** S06 executes both pinned packed paths and mixes their outputs
without a hard selection. The feature is detached before entering a router;
all non-router parameters remain frozen. No bit-width penalty, dataset
training, distillation, hard route, or on-demand loading is introduced.

**Reversal path:** Reopen S06 and preserve the current endpoint, gradient, and
freeze evidence before changing router sharing, normalization, width,
activation, temperature, or candidate ordering.

### D025 — S07-A distillation seams (2026-08-11)

**Choice:** Use explicit causal target IDs and completion-logit masks with the
`T^2` masked mean KL objective; keep the prompt-only feature mask separate from
the causal loss mask; freeze the full-precision teacher and S06 packed student
base; construct and audit the optimizer from explicit `routers.` prefixes;
map hard routes through ordinary argmax over `[p4, p8]` with index `0 -> 4`
and `1 -> 8`; keep route logs/statistics observational; and serialize only
router state, optional optimizer state, and checked metadata.

**Source basis:** The reviewed QAQ source supports teacher-student router
training and query-conditioned routing, but does not establish these exact
masking, optimizer-audit, hard-route, observation, or checkpoint details.
They are S07-A implementation choices, not paper-established facts.

**Evidence:** `src/qaq/s07_distillation.py`,
`tests/unit/test_s07_distillation.py`,
`tests/integration/test_s07_distillation_smoke.py`, and
`docs/stages/S07_DISTILLATION.md`; the deterministic tiny fixture passed the
focused S07-A unit/integration checks with finite loss and gradients, changed
router parameters, preserved frozen teacher/student-base values, and passed
route and checkpoint round trips.

**Consequence:** S07-A provides reusable machinery and smoke evidence only.
Its fixture, temperature, optimizer, learning rate, and step count remain
unresolved smoke values and must not become S07-B baseline decisions. No
bit-width, latency, transfer, or entropy penalty is part of this baseline.

**Reversal path:** Reopen S07-A if causal alignment, freeze boundaries, hard
route mapping, observational fields, or checkpoint contents change; preserve
the current regression evidence and update the focused tests and stage record.

### D026 — S07-B locked real-training baseline (2026-08-11)

**Choice:** Use the cached `Salesforce/wikitext` `wikitext-2-raw-v1` dataset at
revision `b08601e04326c79dfdd32d625aee71d232d685c3`, with four deterministic
training rows from source offsets `[0, 1000, 2000, 3000]` and two deterministic
validation rows from offsets `[0, 1000]`. For each selected row, tokenize raw
text with the pinned Qwen3-4B tokenizer and no special tokens, retain the first
64 tokens, use `[0,32)` as the explicit prompt and `[32,64)` as the explicit
completion, and reject shorter rows. Use seed `1729`, batch size `1`, gradient
accumulation `1`, one epoch/four optimizer steps, AdamW with learning rate
`1e-3`, weight decay `0`, default AdamW auxiliary parameters, no scheduler,
KD temperature `2.0`, fixed S06 routing temperature `1.0`, final-only
checkpoint/evaluation interval at step `4`, and per-step logging.

**Source basis:** The dataset and revision, tokenizer revision, sequence-length
precedent, and deterministic source-order style reuse the repository's S03
quality/calibration evidence. The row offsets, prompt/completion split, sample
counts, optimizer values, schedule, step count, and interval choices are
implementation choices, not QAQ-paper facts.

**Alternatives rejected:** Generated completions were not used as targets. No
random subset or hyperparameter search was used. The S07-A smoke optimizer,
learning rate, and step count were not silently promoted; the baseline values
are recorded here and in `configs/s07_router_training.json`.

**Execution consequence:** Teacher logits are precomputed with the frozen
teacher under `no_grad` and held on CPU before router optimization so the
full-precision teacher and resident packed student do not exceed the 24-GiB
GPU during the one run. This does not alter the KD objective or student
checkpoint contents.

**Reversal path:** Reopen S07 and record a new decision before changing the
source, split, boundaries, sample counts, optimizer, temperatures, or schedule.

### D027 — S07-B actual run freeze-audit defect (2026-08-11)

**Observation:** The single completed baseline run used cached teacher logits
under `no_grad`, excluded the teacher from the optimizer, and left the teacher
parameter values unchanged, but the run did not set the full-precision
teacher parameters to `requires_grad=False` before the freeze audit. The saved
result therefore reports `teacher_frozen: false` even though the packed student
base remained frozen and the optimizer contained only the 23,620,752 router
scalars.

**Decision:** Treat this as a S07 REVISE result rather than marking S07
complete. The script now sets the teacher parameters non-trainable before
precomputation, but the one-run rule forbids silently rerunning the baseline in
this turn. No penalty, temperature change, data change, or second training run
is authorized by this result.

**Evidence:** `docs/results/s07_router_training.json`; initial/final frozen
student parameter aggregate hashes match, finite KD losses and router
gradients were observed for all four steps, fresh-process checkpoint reload
passed, and hard-route repeats were bitwise deterministic.

**Reversal path:** A future explicitly authorized S07-B rerun may use the
corrected freeze audit and the unchanged locked configuration, or S07 may be
reopened with a new decision if the baseline contract changes.

### D028 — D008-1 corrected S07-B rerun (2026-08-11)

**Authorization:** Perform exactly one corrected S07-B baseline router-distillation rerun and re-evaluate the S07 gate. Keep the D026 dataset, model, packed artifact, optimizer, learning rate, temperatures, batch size, steps, seed, and Any-Precision revision unchanged. Do not begin S08 or add a router cost objective.

**Evidence:** The corrected production path invokes the audited teacher/packed-student freeze seam before teacher-logit precomputation. The teacher had `requires_grad=False`, no gradients, and matching before/after parameter hashes. Packed-student non-router hashes matched, the optimizer contained only the 23,620,752 router scalars, router gradients were finite, and router parameters changed. The completion-only KD objective remained `T^2 * masked KL(teacher || student)` with no extra penalties. The focused pre-run suite passed `9 tests`; the S05-S07 regression selection passed `34 tests`; fresh-process checkpoint and deterministic hard-route verification passed.

**Result:** Four finite training losses decreased from `0.1730574965` to `0.0317778103`. Soft validation KD/error were `0.0386699643`/`0.2430240735`; hard validation KD/error were `0.0631424394`/`0.2928081304`; static 4/8-bit errors were `0.7434162199`/`0.0910567641`. Hard 4/8 fractions were `20.1389%`/`79.8611%`, attention 4/8 fractions were `29.1667%`/`70.8333%`, FFN 4/8 fractions were `11.1111%`/`88.8889%`, route coverage was 72/72 per request, and prompt distance was `0.0138889`. The values exactly matched the first run; no material difference was found. Adaptivity remains `OTHER`, a non-blocking observation under the existing gate.

**Consequence:** S07 engineering gate is **CONTINUE** and `docs/STATUS.md` is updated to `COMPLETE`. The next repository-defined action is S08, but this task stops before executing it.

**Reversal path:** Reopen S07 if any freeze, optimizer isolation, finite-value, masking, determinism, checkpoint, or regression property changes, or record a new decision before changing the locked baseline contract.

### D029 — S08-A request-scoped synchronous packed loader (2026-08-11)

**Choice:** Keep one verified nested parent `qweight` tensor and both row-wise
lookup tables in a CPU-authoritative `PackedLinearSource`. A concrete
`SynchronousPackedPlaneLoader` is bound to one `QaqRequestState` object, not to
its textual `request_id`, and registers explicit cleanup with
`QaqRequestState.end_request()`.

**Transfer granularity:** The pinned Any-Precision CUDA path requires the
leading `qweight` planes and the lookup table for the selected precision. A
4-bit first use copies `qweight[:4]` and `lut4`; an 8-bit first use copies
`qweight[:8]` and `lut8`. If a request upgrades from 4 to 8, it copies only
`qweight[4:8]` and `lut8`, combines the newly copied suffix with the retained
GPU prefix, and does not repeat the CPU transfer of the first four planes.
The optional bias, when present, is copied once as another required backend
buffer.

**Evidence:** The pinned `AnyPrecisionLinear.forward` and CUDA kernels read
exactly `w_bits` leading `int32` planes and the matching `float16` LUT. The
S08-A fixture reused the real S01 physical `[8,64,32]` `int32` packed tensor
and pinned `matmul_kbit` execution. Focused tests passed for CPU authority,
first use, request-local reuse, 4-to-8 upgrade accounting, cleanup, duplicate
textual IDs, invalid precision, and resident-versus-on-demand outputs.
Measured first-use transfer bytes were `34,816` for 4-bit
(`qweight[:4]` `32,768` + `lut4` `2,048`), `98,304` for fresh 8-bit
(`qweight[:8]` `65,536` + `lut8` `32,768`), and `65,536` incremental bytes
for the 4-to-8 upgrade (`qweight[4:8]` `32,768` + `lut8` `32,768`).

**Alternatives rejected:** Copying the full parent payload for a 4-bit call
would violate the verified leading-plane contract. A process-global request
cache would violate S05 ownership. Repacking, dequantizing, asynchronous
copies, streams, futures, and prefetching are outside the S08-A baseline.

**Consequence:** S08-A establishes only the synchronous loader seam and small
fixture correctness. It makes no full-model memory, transfer, or latency
claim. The next S08 work unit may integrate this seam with real hard-routed
Qwen3 execution and controlled measurements.

**Reversal path:** If a future pinned backend changes the required planes or
LUTs, rerun the source inspection and fixture parity tests before changing
transfer granularity or byte accounting. If real integration requires a
lifetime or ownership model different from `QaqRequestState.end_request()`,
reopen this decision rather than adding a global registry.

### D030 — S08-B real Qwen3 on-demand baseline (2026-08-11)

**Choice:** Complete S08-B with the synchronous CPU-authoritative loader integrated into real hard-routed Qwen3-4B execution.
Use the two locked S07 validation requests, retain selected packed buffers for one request, and compare against the resident routed model under synchronized CUDA measurements.
Do not add asynchronous transfer, prefetching, cross-request caching, batching, schedulers, or a new routing objective.

**Evidence:** The required artifact, router checkpoint, model snapshot, Any-Precision revision, and CUDA device were present and matched the recorded identities.
The on-demand graph had 252 CPU-authoritative sources, zero remaining packed modules, and no complete packed GPU copy.
Resident and on-demand route maps matched, logits were finite and bitwise equal, and four-token greedy generation matched for both requests.
On-demand transfer totals were `3,817,717,760` and `3,835,002,880` bytes, matching independent expected-byte sums exactly.
All bytes transferred during prefill; decode transfer was zero.
Request cleanup released all retained entries, buffers, and packed bytes, and a later request transferred independently.
The real focused suite passed `3 tests`, the S08-A suite passed `8 tests`, and Ruff passed.
Synchronized two-repeat memory and latency observations are recorded in `docs/results/s08_on_demand.json`.

**Measurement integrity:** The result records the code/worktree snapshot, model revision, packed checkpoint hash, Any-Precision revision, router checkpoint hash, CUDA device, Python and Torch versions, request identifiers and input digests, measurement method, exact transfer records, and allocator readings.
The previously recorded `8 passed in 651.74s` regression evidence was preserved without rerun because no relevant execution path changed after that result.

**Consequence:** The S08 gate is **COMPLETE**.
The repository next action is S09, but S09 must not be started as part of this decision.

**Reversal path:** If a future change alters packed source authority, route parity, logits, transfer accounting, request cleanup, or measurement comparability, reopen S08-B and preserve this baseline result before changing the mechanism.

### D031 — S09-A final evaluation protocol freeze (2026-08-12)

**Choice:** Freeze the machine-readable S09-A protocol in
`configs/s09_baseline_eval.json` before any final S09-B result exists. The
comparison has exactly five modes: BF16 teacher, static packed 4-bit, static
packed 8-bit, hard-routed resident packed, and hard-routed synchronous
on-demand packed. The fixed request/input records are in
`configs/s09_baseline_prompts.json`.

The frozen protocol SHA-256 is
`01ca65c6b3b7e16d7af66f1533140b1c9f31749c90bc91e097d096d463bf2e1c`.
The complete contract is maintained in the config and the detailed procedure
in `docs/stages/S09_BASELINE_FREEZE.md`; this decision records the rationale
and gate, not a second copy of every protocol field.

**Source-supported and established behavior:** The protocol reuses the S03
token-weighted causal cross-entropy evaluator and the S08 synchronous loader
contract. S07 establishes 72 separate attention/FFN hard-route units and the
locked router checkpoint; S08 establishes physical selected-plane-plus-LUT
transfer accounting and request cleanup. These sources do not establish a
paper score, a final quality threshold, or a universal latency/memory target.

**Implementation choices:** The dataset, inputs, generation, measurement,
quality-gate, failure-outcome, and deferred-mechanism values are frozen in the
authoritative config. The 10% quality factor is a QAQ baseline implementation
gate, not a paper claim. Structural or quality failures are REVISE;
missing/incomparable hardware or external artifacts are PAUSE.

**Evidence:** The exact artifact and router provenance are recorded by the
config and prior stage evidence. The non-benchmark validator and focused
protocol checks passed at the freeze; no five-mode comparison or S09-B result
exists in this decision.

**Alternatives rejected:** A sixth or soft-routing final mode, alternate
router/checkpoint, random prompt/data sampling, a divergent perplexity
calculation, subjective text score, nominal bit-width memory estimates,
reserved-memory residency claims, asynchronous/prefetch/caching/batching
mechanisms, and changing the 10% gate after results are all outside this
freeze. A later genuine defect requires REVISE plus invalidation of affected
results, not an undocumented protocol edit.

**Consequence:** S09-A is IN_PROGRESS and may hand off only to S09-B from the
committed frozen config. The protocol is not a final quality, memory, latency,
or transfer conclusion. Deferred mechanisms remain outside the baseline.

**Reversal path:** If the S09-A validator or a later controlled check finds a
protocol defect, mark the protocol REVISE, identify affected inputs/results,
invalidate them, update this ledger with the replacement decision, and rerun
the validation gate before any new S09-B execution.

### D032 — S09-A validator review follow-up (2026-08-12)

**Choice:** Keep the D031 S09-A protocol, inputs, seeds, five-mode matrix,
measurement boundaries, and provenance unchanged while making the validator
enforce the contract it already documents. The follow-up validates complete
72-unit routed records and S07's `OTHER` limitation; fixed-input greedy
generation and the complete seed policy; D029 physical transfer mode/rule and
real expected-byte inputs; the fixed RTX 3090 comparability and identity
record; exact deterministic dataset/label/loss wording; the live
Any-Precision submodule revision; and the complete REVISE/invalidation,
all-gates-pass, and deferred-mechanisms policies.

**Evidence:** Mutation-focused S09 protocol tests pass, the lightweight
validator passes with external artifact presence checks, Ruff passes, and the
checked-out submodule revision is read from Git rather than trusted from the
manifest alone. No S09-B benchmark or result artifact was generated.

**Consequence:** S09-A remains frozen before results and may proceed only to
the separately authorized S09-B evaluation after this review-fix commit and
the no-mistakes gate. A future protocol change still requires REVISE and
invalidation of affected results.

**Reversal path:** If any focused check reveals a mismatch with D031, stop at
S09-A, preserve the evidence, and record a replacement decision before any
benchmark execution.

### D034 — S09-B1R runner correctness repair (2026-08-12)

**Choice:** Correct the frozen S09-B runner before any five-mode execution. Keep
both frozen JSON files, the five-mode matrix, the S03/S07/S08 mechanisms, and
all evaluation gates unchanged.

**Evidence:** The first implementation dropped measured on-demand cleanup
records, fabricated cleanup and hidden-copy success, omitted latency medians,
used a constant deterministic-evidence flag, lacked exact perplexity-result
validation, did not enforce cross-mode hardware comparability, validated
Any-Precision identity only for routed modes, left physical packed-byte fields
empty, and advertised but did not persist `aggregation.json`. The correction
adds measured fixture coverage for each defect and routes cleanup, latency,
repeat, hardware, perplexity, packed-identity, physical-residency, and
aggregation evidence through the existing S03/S07/S08 seams.

**Consequence:** S09-B1R is the required pre-execution correction. No model was
loaded for the repair tests, no real five-mode evaluation ran, and no S09-B
result artifact exists. The next action remains the separately authorized
frozen S09-B execution only after this correction passes its gate.

**Reversal path:** If safe validation finds another result-invalidating defect,
return REVISE and invalidate no results because none exist; never change the
frozen protocol to accommodate the runner.

### D033 — S09-B1 minimal runner structure (2026-08-12)

**Choice:** Implement S09-B as one parent entry point,
`scripts/run_s09b.py`, which launches itself as one explicit-mode child per
frozen mode; keep result validation and aggregation in `qaq.s09_runner`.
The default and explicit `--plan` paths validate the frozen protocol, consume
the committed fixed-input file for identity planning, print all five child
commands and the aggregation command, and write no result.

**Established:** S03 supplies the pinned full-precision/static loaders,
precision selection, causal perplexity evaluator, and cleanup; S07 supplies
the router checkpoint and hard-route primitive; S08 supplies the synchronous
request-owned packed source and physical transfer seams.

**Unknown:** The real five-mode S09-B comparison values remain unknown because
this decision does not execute any model mode.

**Alternatives rejected:** A sequential parent model process, a persistent
service, a worker pool, a sixth mode, regenerated prompt inputs, a second
perplexity implementation, and synthetic final result data were rejected
because they would weaken process isolation or the frozen comparison.

**Consequence:** Each later mode execution has a fresh process boundary and a
structured per-mode result; aggregation can classify missing results as
`PAUSE` and structural or quality failures as `REVISE` without fabricating a
comparison or marking S09 complete.

**Reversal path:** If a later non-benchmark validation proves that an existing
S03, S07, or S08 seam cannot satisfy the frozen contract, stop and return
`REVISE` without changing the frozen inputs, identities, gates, or mode list.
The runner structure is an implementation assumption, not a paper fact.

### D035 — S09-B3 routed decode diagnosis (2026-08-12)

**Observation:** Preserved S09-B2 evidence has equal resident/on-demand route maps and generated token IDs for all seven requests, but different logits digests. Artifact-only analysis found resident `s03-quality-3` generated-token divergence at zero-based position 6; on-demand generated tokens were stable. The narrow diagnostic found bitwise-equal resident/on-demand prefill logits, then decode-logit divergence at the first decode step while selected tokens still matched. With the exact real-model shape `[1,1,9728]`, 8-bit `matmul_kbit` repeated outputs were not bitwise stable and differed between resident and on-demand executions; the corresponding `dequant_kbit` plus `torch.matmul` path was bitwise stable and equal across both buffer paths. The pinned kernel selects its `M=1`, `K>4096`, 8-bit k-split path, which combines partial sums with `atomicAdd`.

**Decision:** S09-B2 remains **REVISE** and S09 remains **IN_PROGRESS**. The established failure mechanism is numerical nondeterminism in the pinned real-shape `matmul_kbit` k-split projection path, not route selection, request cleanup, transfer accounting, or result-file bookkeeping. CUDA determinism settings were observed but not changed. The diagnosis did not modify frozen protocol/configuration, production execution code, or preserved S09-B2 result files.

**Consequence:** A future repair must first address the affected packed projection execution path without silently changing the frozen equivalence criterion, then rerun only the invalidated routed evidence and re-evaluate S09-B gates. The proposed repair has not been tested. Remaining unknowns include the smallest acceptable baseline-preserving repair and whether every affected routed projection requires the same treatment.

**Reversal path:** If a separately authorized repair disproves the kernel-path explanation or changes the pinned baseline mechanism, preserve this diagnosis and record new evidence before revising the conclusion.

### D036 — S09-B4 deterministic routed packed execution repair (2026-08-13)

**Established:** The pinned Any-Precision dispatch uses its atomic k-split
kernel exactly when the device is not Orin, the effective row count is
`M == 1`, the packed input width is `K > 4096`, and `w_bits >= 7`. Within QAQ's
locked 4/8-bit route set, this is precisely the 8-bit, one-row, input-width-
greater-than-4096 family. The pinned kernel source allocates
`num_ksplit = ceil(K / 4096)` and combines partial sums with `atomicAdd`.

**Repair:** A shared `execute_packed_linear` helper now mirrors that dispatch.
The exact family uses the pinned `dequant_kbit` followed by
`torch.matmul` with a temporary CUDA dense weight; all other one-to-eight-row
calls retain `matmul_kbit`, and the pre-existing greater-than-eight-row
reference path remains unchanged. Both `_RoutedPackedLinear` resident calls
and `SynchronousPackedPlaneLoader` request-owned calls use the helper. The
helper creates no registered parameter or buffer and no dense weight survives
a projection call or request cleanup.

**Evidence:** The focused real-shape test used `[1,1,9728]` and 8-bit packed
inputs, passed five bitwise-identical finite executions, matched resident and
request-owned results, and separately proved an unaffected 8-bit `K=1024`
path still calls `matmul_kbit`. The real Qwen3 inventory has 252 targeted
projections: only the 36 `model.layers.<i>.mlp.down_proj` projections have
`in_features=9728` and can enter the fallback, and only when selected at
8-bit; the remaining 216 targeted projections have `in_features=2560` and
retain the packed kernel for 4/8-bit routes. Narrow CUDA validation on
`cuda:3` passed for `s03-quality-3` and `validation-3`: prefill and all eight
decode logits were finite and bitwise equal across resident/on-demand modes,
route maps and selected tokens matched, and five repeated `s03-quality-3`
generations had identical route maps, per-step logits digests, and token
sequences in both modes. On-demand transfer remained packed-only and exactly
matched expected bytes (`3,835,002,880` and `3,817,717,760`), decode transfer
was zero, cleanup returned 252 entries/504 buffers to zero, and the hidden
copy audit passed. The pinned Any-Precision submodule stayed clean at
`a3257d02740cc5757c78673da534b0630ff3a4ea`; the frozen config/input hashes
and all six failed S09-B2 artifacts remained unchanged.

**Scope:** The FP teacher and static 4/8-bit paths are untouched by the diff.
The original routed S09-B2 results are invalidated by this execution repair;
the original FP/static results remain valid pending the later rerun's final
comparison. Corrected routed quality, resource, and latency results remain
unknown.

**Proposed next step:** Rerun only the invalidated routed resident and routed
synchronous on-demand S09-B evidence under this verified deterministic repair,
then preserve and reuse unaffected FP/static S09-B2 evidence only after
confirming their execution paths remain unchanged. Do not rerun S09-B's five-
mode evaluation as part of this repair.

**Reversal path:** If a later targeted run finds resident/on-demand bitwise
divergence, nondeterminism, transfer or cleanup regression, or an execution
path outside this condition is changed, return S09-B4 to REVISE and preserve
this diagnosis without changing the frozen protocol or pinned dependency.

### D037 — S09-C frozen baseline closeout (2026-08-13)

**Established:** S09-B5 is the passing frozen baseline evidence, with canonical
artifacts in `docs/results/s09b_b5/`. The old S09-B2 artifacts remain preserved
in `docs/results/s09b/` as failed evidence. The routed deterministic fallback
from `4a0dc702178fef0f84eb9ffd9bd6d1810e5dc564` is part of the frozen baseline
implementation. FP/static reuse is justified because the execution-path diff
from `a2e31188be952f97a1439ff7df46d9f43100bae5..4a0dc702178fef0f84eb9ffd9bd6d1810e5dc564`
is empty for `scripts/run_s09b.py`, `src/qaq/s09_runner.py`,
`src/qaq/s03_quality.py`, and `src/qaq/s03_static.py`; commit
`443f6994582500857afca9bad6032cc285448a86` added only `docs/results/s09b_b5/`.
The committed and temporary read-only aggregations returned `CONTINUE` with no
errors, and all frozen release criteria passed.

**Unknown:** Behavior of any post-baseline optimization remains unmeasured.
This baseline is not an exact reproduction of QAQ paper scores, and `OTHER`
remains an observational route-diversity classification.

**Next-step rule:** No optimization begins automatically after S09. Later work
requires an explicitly defined new stage and decision. No later authoritative
stage is currently defined; the exact next action is: Baseline frozen. Stop.
Define an explicit post-baseline stage and decision before implementing any
optimization or additional research mechanism.

## Decision protocol

A worker must add a dated or commit-linked entry when a stage resolves an unknown or introduces a new assumption.
The entry must state the evidence, alternatives considered when material, consequence, and reversal path.
A stage cannot be declared complete while its required decision gate contains an unresolved blocker.

### D017 — S02 pinned physical bit-plane contract (2026-08-11)

**Choice:** Adopt the versioned v1 contract in `docs/BITPLANE_FORMAT.md` for
the pinned Any-Precision revision `a3257d02740cc5757c78673da534b0630ff3a4ea`:
contiguous `int32` qweight shape `[P,N,K//32]`, MSB-first plane order,
the source's warp-oriented byte permutation, leading-plane 4-bit selection,
row-wise `float16` direct LUTs, and a strict `K % 32 == 0` baseline boundary.

**Status:** Resolved; S02 evidence passes on 2026-08-11.

**Evidence:** The pinned source was inspected at `quantization/pack.py`,
`modules/AnyPrecisionLinear.py`, `modules/kernels/main.cu`,
`modules/kernels/dequant.cuh`, `modules/kernels/matmul.cuh`, and
`quantization/quantize.py`. Deterministic known patterns established
`0 -> 0x80000000`, `1 -> 0x40000000`, and `31 -> 0x00000001` in the `K=32`
case; all-zero, all-one, alternating, one-plane, and adjacent-plane words
are asserted with stable SHA-256 digests. The independent reference codec
matches the pinned pack helper for a seeded `[3,1024]` random fixture and
matches pinned CUDA dequantization at both 4 and 8 bits. The actual pinned
nested quantizer produced one shared parent-label tensor and distinct `[N,16]`
and `[N,256]` LUTs, with reconstruction checked at both precisions.
Serialization checks confirmed PyTorch's little-endian data payload matches
the contiguous `int32` tensor bytes. The production-facing qweight guard
observed `torch.int32` storage and rejected byte-per-logical-bit accounting.

**Alternatives considered:** A contiguous logical-bit word order was rejected
because `_permute_bitmaps_int32` and the CUDA masks demonstrate the warp
transpose plus per-word byte reversal. A byte-per-bit tensor remains a
correctness-only oracle and cannot support resource claims. A sign plane,
scale, or zero-point field was not added because the pinned quantizer stores
unsigned labels and direct floating centroids in the LUTs; negative LUT values
already provide signed reconstructed values. Implicit zero padding was rejected
because the source rejects non-aligned widths and the constructor's floor
division is not safe padding.

**Consequence:** S02 reference and future baseline code must keep packed planes
as the authoritative production representation, account for LUTs/scales (with
the pinned backend having no separate scales) separately, and reject unsupported
alignment or grouped-LUT layouts. No upstream source or production backend
implementation was modified.

**Reversal path:** If the pinned gitlink changes, if a future backend revision
changes the masks/byte permutation, or if a supported grouped/padded format is
introduced, preserve this contract and add a new dated decision with new
known-word, reconstruction, serialization, and byte-count evidence before
changing the format version.

### D038 — S10-A static six-bit gate (2026-08-13)

**Choice:** Expose exactly `(4, 6, 8)` from `qaq.model.static` while retaining
the pinned Any-Precision constructor's inclusive 4–8 internal buffers. Static
precision 6 uses the existing parent `qweight[:6]` and the artifact's existing
`lut6`; no artifact regeneration, separate six-bit qweight, dense persistent
weight, router change, or new routing behavior is introduced.

**Evidence:** The identity-matched Qwen3 artifact contains all 252 expected
parent qweights as `[8,N,K//32]` `torch.int32`, all 252 finite `torch.float16`
LUT6 tensors with `[N,64]` shape, and 141,557,760 LUT6 bytes. A real
representative precision-6 CUDA call on pinned Any-Precision commit
`a3257d02740cc5757c78673da534b0630ff3a4ea` matched the pinned dequantizer plus
FP16 matmul at the established `atol=0.05`, `rtol=0.01` tolerance and was
bitwise deterministic. Full-model static precision-6 smoke, existing 4/8
static paths, target inventory, duplicate-model, S06, and S07 structural
regressions passed.

**Alternatives considered:** Adding a six-bit router candidate was rejected as
out of scope for S10-A and would alter the frozen 4/8 router semantics.
Regenerating or requantizing the nested artifact was rejected because the
existing LUT6 state is already identity-matched and verified. Adding a dense
6-bit reconstruction as module state was rejected because it would violate the
packed-storage baseline boundary.

**Consequence:** Static model callers may select only 4, 6, or 8 and invalid
values fail before module dispatch. The next 6-bit routing stage must make a
separate decision about routing semantics and must preserve this static
execution evidence.

**Reversal path:** If a future pinned backend or artifact format changes the
inclusive LUT contract, preserve this evidence and add a new decision after a
fresh source/artifact review; do not silently broaden the public static set or
modify router candidates.

### D039 — S07C-EVIDENCE-005 direct hard-route round trip (2026-08-14)

**Decision:** Treat a hard-route checkpoint round trip as valid only when the
original `evaluation.hard.route_logs` keyed map
`(request_id, layer, unit_type) -> hard_bit` matches the route selected by
fresh hard execution in `QaqRequestState.attention_routes` and
`QaqRequestState.ffn_routes`. Keep the existing probability and soft-derived
`hard_bit` comparison as separate weaker evidence.

**Evidence:** The repaired verifier checked the required checkpoint SHA-256
`08bf646f19759c0d7949e159bdbe4f96bbea737204b96f8760d205c8d6fd1949` before
model/data execution. For `validation-3` and `validation-1000`, recorded and
actual coverage were each exactly 36 attention plus 36 FFN routes (72 per
request), with exact attention matches `72/72`, FFN matches `72/72`, total
matches `144/144`, no missing/unexpected/duplicate/invalid keys, and mismatch
count `0`. Repeated actual route maps and selected precisions matched,
repeated hard logits were bitwise equal, logits were finite, and the packed
student remained unchanged. The focused regression proves that a deliberately
altered hard record fails this direct assertion even when soft probabilities
and soft-derived bits still match.

**Historical boundary:** The old verifier established probability equality
and soft-derived hard-bit equality after reload; it did not establish this
actual-execution invariant. This entry records the repair and does not rewrite
the earlier S07-B evidence.

**Consequence:** S07C-EVIDENCE-005 is resolved with CONTINUE. No router
training, router-semantic change, six-bit routing, objective change, or S10-B
execution is part of this evidence repair. The next action is **Begin S10-B:
Three-Way Router Semantics.**

**Reversal path:** If a future route representation, candidate ordering, hard
execution path, or checkpoint format changes, preserve this S07C evidence and
reopen the gate with a new keyed actual-execution comparison before carrying
the result forward.

### D040 — S10-B explicit learned-router candidate ordering (2026-08-14)

**Implementation choice:** Centralize learned-router candidate validation in
`qaq.router.network.validate_candidate_bits` and permit exactly `(4, 8)` and
`(4, 6, 8)`. Carry the selected `candidate_bits` on each `QaqRequestState`,
size router heads and probability vectors from that tuple, and record the tuple
on soft traces, route logs, and checkpoint metadata. Keep the default
historical tuple `(4, 8)`.

**Source/project-established facts:** S06 defines the existing 4/8 router
architecture and canonical index order; S10-A establishes the packed static
6-bit path as `qweight[:6] + lut6`; S04 `PrecisionPlan` and S08 synchronous
loading remain 4/8-only.

**Evidence:** Focused candidate, state, hard-route, trace, logging, statistics,
checkpoint, and freeze tests passed. A real pinned Any-Precision CUDA fixture
executed all three configured packed calls exactly once, including precision 6,
and passed forced endpoints, finite differentiable mixing, and gradient checks.
Historical unit, S06/S07 lifecycle, and S08 loader regressions passed.

**Alternatives rejected:** Inferring `(4, 6, 8)` from vector length was rejected
because it makes stored probabilities ambiguous. Duplicating the router
architecture was rejected because only the output width changes. Changing
`PrecisionPlan`, extending the S08 loader, adding a cost penalty, or training a
new router was rejected as outside S10-B.

**Checkpoint format:** Keep format version 1. Existing metadata already stores
candidate ordering, so the serialized schema remains unambiguous for both
2-output and 3-output router states; metadata and PyTorch shape checks reject
cross-ordering loads without padding or conversion.

**Consequence:** New callers can construct an explicit three-way learned router
with canonical `[p4, p6, p8]` probabilities and resident hard route 6, while
historical callers and the S07 checkpoint remain 4/8-compatible. No S08
6-bit on-demand claim is made.

**Reversal path:** If a future candidate set, checkpoint schema, or packed
loading boundary changes, add a new stage decision with fresh compatibility and
artifact-backed evidence rather than widening this validator implicitly.

### D041 — S10-C normalized cost-aware router objective (2026-08-14)

**Implementation choice:** Compose the unchanged S07 completion-only masked KL
loss with an optional normalized bit-plane-count surrogate. For explicit
candidate ordering, use `c(bit) = (bit - 4) / (8 - 4)`, compute
`C(p) = sum_b p_b*c(b)`, and reduce `L_bit` as an unweighted arithmetic mean
across every included attention and FFN routing decision. The supported cost
vectors are `[0.0, 0.5, 1.0]` for `(4, 6, 8)` and `[0.0, 1.0]` for historical
`(4, 8)`; expected three-way width is the diagnostic `4 + 4*L_bit`.

**Source/project-established facts:** S07 establishes the KL-only objective;
S10-B establishes explicit `(4, 8)` and `(4, 6, 8)` ordering and differentiable
probability outputs for all 72 learned units. No cost coefficient was
established elsewhere. The new term is an implementation choice, not a paper
claim or measured hardware cost.

**Compatibility choice:** `lambda_bit` is validated as finite, numeric,
non-negative, and non-boolean. Its default is zero for backwards-compatible
KD scalar and gradient behavior; no nonzero production value is selected. The
request-state aggregator preserves the probability autograd graph and counts
each attention and FFN slot exactly once. Invalid candidate orderings,
probability shapes/values, and weights are rejected explicitly.

**Alternatives rejected:** Modifying or duplicating `masked_kl_distillation_loss`,
selecting a production lambda, sweeping lambda, weighting by measured
hardware, adding route quotas, entropy or exploration terms, or adding any
latency/memory/transfer/energy/kernel objective were outside S10-C.

**Evidence:** Focused S10-C unit tests passed exact endpoint, historical,
mixed/uniform, boundedness, validation, width, positive test-only lambda,
lambda-zero scalar/gradient, 8-bit gradient-pressure, request-state
aggregation, finite-gradient, and frozen-state checks. No training,
checkpoint generation, artifact-backed Qwen3 execution, S08 loader change,
Any-Precision change, or historical-result rewrite occurred.

**Reversal path:** If a future caller needs a different candidate set, cost
scale, weighting, production coefficient, or hardware objective, preserve this
normalized baseline and record a new decision before changing the composition
API or training path.

### D042 — S10-D fixed lambda calibration protocol (2026-08-14)

**Implementation choice:** Run the complete fixed grid
`[0.0, 0.003, 0.01, 0.03, 0.1]` using the locked S07 data/training values and
explicit three-way `(4,6,8)` routers. Before every trial, clone/reload one
canonical seed-1729 router-only state, verify 72 routers and 23,630,040
scalars, and construct a fresh AdamW. Permit at most the two exact adaptive
points `0.001` and `0.3` only under the conditions recorded in the authoritative
S10-D config. Do not select a production coefficient from the sweep.

**Established facts:** S07's masked KL, teacher/packed-student freeze seam,
data order, temperatures, and four-step budget are locked. S10-B supplies
resident three-way routing and `[p4,p6,p8]`; S10-C supplies the unchanged
composition `L_KD + lambda_bit*L_bit`. Static 4/6/8 references must precede
learned-route interpretation, and Pareto frontiers are reported without a
scalar selection.

**Execution choice:** The runner uses the existing `execute_packed_linear`
helper in a local autograd seam that recomputes frozen packed weights during
backward. This avoids retaining dense dequantized weights for all three paths
on the 24-GiB GPU while preserving the packed forward and router gradients;
it is not a production-model or pinned-backend change. No historical S07
checkpoint, S08 loading, asynchronous transfer, prefetching, latency/memory
benchmark, or objective redesign is used.

**Evidence:** On the required starting commit
`41e598b0e00e9b72444b498c5cd39b2f335c2257`, the identity-matched teacher,
artifact, dataset, and clean Any-Precision revision were available. Static
4/6/8 logits were finite; all five grid trials completed on free `cuda:0`;
all initial router hashes matched; every trial had finite gradients/losses and
unchanged teacher/packed state; and the result records hard 6 fractions, mean
p6, route maps, deltas, and soft/hard Pareto frontiers. No adaptive point was
triggered. Focused S10-D plus S10-C/S10-B/S07/request-state regressions passed
44 tests and Ruff passed.

**Consequence:** The result is a calibration observation only. The measured
soft frontier is `{0.0, 0.003, 0.03, 0.1}` and the hard frontier is `{0.03, 0.1}`
under the recorded validation KD/width coordinates, but neither frontier
selects a production lambda. Firstmate/captain must review the evidence and
choose whether to refine, confirm, or begin full training.

**Reversal path:** If a rerun changes the locked data, initialization,
objective, adaptive trigger, candidate order, or execution path, preserve this
artifact and record a new decision before comparing or extending the result.

### D043 — S10-D review validation repairs (2026-08-14)

**Implementation choice:** Keep the S10-D experiment unchanged while making
the runner reject noncanonical protocol bytes, pass configured KD temperature,
entropy base, and adaptive trigger values through every relevant path, require
the exact pinned Hugging Face cache snapshot path, and reject any missing router
gradient before norm calculation.

**Evidence:** The canonical result remains valid because the locked config
values and execution outputs are unchanged. The focused repair suite passed
`11` tests, including config, snapshot-path, entropy-plumbing, collapse
threshold, and missing-gradient regressions.

**Consequence:** `--config` can only select an exact copy of the locked
protocol, and `QAQ_MODEL_SNAPSHOT` cannot redirect the run to another local
snapshot. No production lambda or later-stage work is authorized by this
repair.

### D044 — S10-E frontier confirmation protocol freeze (2026-08-14)

**Source/project-established facts:** The merged starting point is
`e718f27fe6b02082709d65665396640e251e602c`, and S10-A through S10-D are
complete. The canonical S10-D evidence completed exactly
`0.0, 0.003, 0.01, 0.03, 0.1`, ran no adaptive extension, selected no
production lambda, and records the hard frontier at `0.03` and `0.1`. S07,
S10-B, and S10-C establish the inherited data/training, three-way resident
routing, frozen teacher/packed base, normalized cost, and unchanged
completion-only KD semantics.

**Implementation choice — captain-selected controls:** Freeze exactly three
seeds `[1729, 1730, 1731]`, exactly the three candidates/lambdas
`[0.0, 0.03, 0.1]`, and nine paired trials. Each seed has one fresh canonical
three-way router initialization cloned identically across lambdas; each lambda
uses a fresh AdamW in the same order, with no warm start and no historical
S07 two-way checkpoint. The seeds and three-candidate confirmation are
captain-selected controls, not source-paper facts.

The machine-readable protocol also freezes the 72-router/
23,630,040-scalar contract, resident soft and deterministic hard routing,
request-owned state, explicit validation route maps and `fraction_6`, finite
and freeze/optimizer audits, and cross-seed aggregates. It forbids latency,
memory, transfer, throughput, and energy measurements. The confirmation gate
requires all nine complete trials, passing audits, no invalid or degenerate
collapse, `0.03` on the per-seed hard KD/width frontier in at least two of
three seeds, paired-control median hard KD delta no greater than `0.0`,
strictly lower paired-control median hard selected width, and no
reproducibility failure. There is no scalar combined score or arbitrary
quality-loss threshold. Success authorizes only later broader validation;
failure is `REFINE`; incomplete evidence is `PAUSE`.

Validation route maps retain the inherited S10-D layer-major serialization:
layer 0 attention, layer 0 FFN, then layer 1 attention, layer 1 FFN, through
layer 35. This is the canonical order for S10-E route-map comparisons.

**Evidence:** The focused protocol tests validate exact fields and reject
missing, extra, reordered, or reintroduced seeds/lambdas, adaptive extension,
data/order/training/candidate/router/selection drift, missing pairing
semantics, and forbidden measurement drift. They execute no experiment.

**Consequence:** S10-E freezes only `configs/s10e_frontier_confirmation.json`,
its focused tests, and this stage/decision documentation. No S10-E
confirmation trial, full training, production selection, S10-D rewrite, or
S10-F work is authorized here.

**Reversal path:** If a later protocol review finds an identity, semantic, or
gate defect, return S10-E to REVISE, preserve all S10-D historical evidence,
and record a replacement decision before any confirmation execution.

### D045 — S10-F frozen execution interpretations (2026-08-15)

**Implementation choices made before canonical trials:** S10-F accepts only
the byte-identical S10-E config and requires the merged S10-E implementation
base `7fc136eabdba302e199354ae001cd1e1cd42199f` as an ancestor. The config's
historical `required_starting_commit` remains provenance for the frozen
protocol and is not used as the S10-F execution base.

The S10-D diagnostic entropy base `2.0` is reused because S10-E inherits the
S10-D route-statistics contract without restating that diagnostic value. The
reproducibility audit is one immediate hard-validation repeat at the unchanged
trained router state; it requires exact equality of both 72-entry validation
route maps, hard summary metrics, and finite-output status, and performs no
additional optimizer trial. This is the smallest repeat audit consistent with
the established S07 deterministic hard-route precedent.

The existing S10-D collapse labels are interpreted as follows: any
`COLLAPSED_TO_4`, `COLLAPSED_TO_6`, or `COLLAPSED_TO_8` label is an invalid or
degenerate collapse for this gate; `PROMPT_INVARIANT`, `ADAPTIVE_OBSERVED`, and
`OTHER` remain observational labels and do not invalidate a trial. A hard
frontier point is non-dominated on validation hard KD and selected width,
with both axes lower-is-better and one strict improvement required for
domination. The `0.03` versus `0.0` aggregate is computed within each seed
before taking the median.

The inherited regression requirement is an external pre-run evidence input to
the runner; absent evidence is `PAUSE`, a failed inherited selection is
`REVISE`, and a complete nine-trial matrix with a failed frozen result gate is
`REFINE`. No static S10-D references, historical S07 two-way checkpoint,
adaptive lambda, production selection, or prohibited serving/resource
measurement is part of S10-F.

### D046 — S10-F canonical execution invalidated by runner audit defect (2026-08-15)

**Observed evidence:** The frozen nine-pair matrix completed on `cuda:0`,
NVIDIA GeForce RTX 3090, from implementation base
`7fc136eabdba302e199354ae001cd1e1cd42199f`, with the pinned model/tokenizer,
dataset, packed artifact, and Any-Precision identities. The result artifact
is `docs/results/s10f_frontier_confirmation.json` with SHA-256
`d68f041e0a3dc32c465e8b8068ca3ab230d39253757e30f3019ca7e681b14233`.

The measured aggregate values in that artifact are frontier membership for
`0.03` in `2/3` seeds, paired hard KD delta median
`-0.004020056687295437`, paired hard selected-width delta median
`-0.16666666666666696`, and zero reproducibility failures. All finite,
teacher-frozen, packed-base, route-map, collapse, and repeat observations were
recorded. However, the runner compared the inherited optimizer audit's Python
tuple `("routers.",)` only to a list, so all nine serialized
`router_only_optimizer_audit` and `fresh_adamw_audit` fields were falsely
recorded as `false`, despite the raw audit recording fresh state and the
`routers.` prefix. The generated artifact therefore reports `REFINE`, but its
gate evidence is invalidated by this implementation defect.

**Decision:** Classify this completion as `REVISE`, preserve all nine raw trial
records and their hashes, do not repair the runner or rerun any trial in this
task, and do not authorize broader validation or production-lambda selection.

### D047 — S10-F audit repair and historical evidence decision (2026-08-15)

The original packed artifact preflight passed with SHA-256
`29d9bc526b3da0bd39daf2f82afd141f82d005ca1232cabc75cfe9d9ecc1cfee`. The
preserved S10-F result remained byte-identical at SHA-256
`d68f041e0a3dc32c465e8b8068ca3ab230d39253757e30f3019ca7e681b14233`.
The exact defect was `optimizer["included_name_prefixes"] == ["routers."]`:
the left operand was the tuple `("routers.",)` and the right operand was the
list `["routers."]`; they have equal contents but Python semantic equality is
false. The lower-level identity audit separately rejects missing router
parameters, extra non-router parameters, and duplicate parameter tensors.

The S10-F-only repair normalizes the accepted prefix container to a tuple
before comparing it with the single exact `("routers.",)` prefix and requires
the fresh-state observation to be the explicit boolean `True`. It does not
change S10-D or router semantics. Regression coverage proves list/tuple
representation equivalence, missing/extra/duplicate identity rejection,
fresh AdamW empty state before its first step, and rejection of reused state.

Historical revalidation is **Branch B: `PAUSE / RERUN_REQUIRED`**. The
preserved per-trial result contains only `included_name_prefixes` and the
router scalar count; it does not preserve included parameter names or
parameter identities, group membership, or duplicate checks. It also records
only a fresh-state boolean, not an independent preserved optimizer-state
snapshot. Reclassifying either audit would therefore require inferring
runtime proof from source behavior, which is not allowed. The measured
frontier values and original result artifact remain unmodified; no repaired
historical audit fields or CONTINUE gate are claimed. No canonical training,
evaluation, extra trial, or broader validation rerun is authorized by this
decision.

### D048 — S10-F canonical rerun with runtime optimizer evidence (2026-08-15)

**Choice:** Execute one fresh attempt-2 nine-trial S10-F matrix after adding
observational optimizer evidence to the runner. The frozen protocol, seeds,
lambdas, data, training budget, objective, routing, and GPU selection were
unchanged. The instrumentation records identity-based membership, expected and
actual parameter counts and name digests, missing/extra/duplicate counts,
construction serials, and empty optimizer state before training; it does not
alter optimizer behavior.

**Evidence:** Attempt 2 completed all nine ordered trials on `cuda:0` with the
pinned artifact hash `29d9bc526b3da0bd39daf2f82afd141f82d005ca1232cabc75cfe9d9ecc1cfee`,
model/tokenizer revision `1cfa9a7208912126459214e8b04321603b3df60c`, dataset
revision `b08601e04326c79dfdd32d625aee71d232d685c3`, and Any-Precision revision
`a3257d02740cc5757c78673da534b0630ff3a4ea`. Every runtime audit passed.
The new result is `docs/results/s10f_frontier_confirmation_rerun.json` with
SHA-256 `b3bcc0e45d45852ac5060209c4789453ed452462f528f7bffd4cb80fb1ef58cb`.
Attempt 1 remains byte-identical at
`d68f041e0a3dc32c465e8b8068ca3ab230d39253757e30f3019ca7e681b14233`.
The frozen aggregates give `0.03` frontier membership `2/3`, paired hard KD
median delta `-0.004020056687295437`, paired hard width median delta
`-0.16666666666666696`, and zero reproducibility failures. Focused tests passed
`65`, inherited regressions passed `46`, Ruff and `git diff --check` passed.

**Consequence:** S10-F is **CONTINUE**. This authorizes only a separately
scoped broader-validation decision; no production lambda or later stage was
started.

**Reversal path:** If any implementation or runtime audit identity changes,
reopen S10-F and invalidate attempt 2 rather than interpreting the result as a
valid confirmation.

### D049 — S10-G broader-validation protocol freeze (2026-08-15)

**Source/project-established facts:** S10-A through S10-F are complete. S10-F
attempt 1 remains preserved at
`docs/results/s10f_frontier_confirmation.json` with SHA-256
`d68f041e0a3dc32c465e8b8068ca3ab230d39253757e30f3019ca7e681b14233` and its
REVISE classification. S10-F attempt 2 remains at
`docs/results/s10f_frontier_confirmation_rerun.json` with SHA-256
`b3bcc0e45d45852ac5060209c4789453ed452462f528f7bffd4cb80fb1ef58cb` and its
CONTINUE classification. Attempt 2 selected no production lambda and
authorized only a separately scoped broader-validation decision; no broader
validation has run. The pinned Qwen3/tokenizer, Wikitext revision,
Any-Precision revision, packed artifact, 72-router contract, explicit
`[p4,p6,p8]` order, normalized costs `[0.0,0.5,1.0]`, completion-only KD,
frozen teacher/packed base, router-only optimizer, and S10-F route-map and
paired-control semantics are established by the preceding stage artifacts.

**Implementation choice — broader deterministic data:** Freeze exactly 24
training examples and 12 validation examples, six times the S10-F counts, from
`Salesforce/wikitext` `wikitext-2-raw-v1` at revision
`b08601e04326c79dfdd32d625aee71d232d685c3`. Use the pinned tokenizer revision
`1cfa9a7208912126459214e8b04321603b3df60c`, sequence length 64, prompt
`[0,32)`, completion `[32,64)`, and the inherited first-qualifying-row
selection rule. The exact ascending train offsets `[0,1000,...,23000]` and
validation offsets `[0,250,...,2750]`, selected row indices, IDs, and ordering
are in `configs/s10g_broader_validation.json`. The sixfold counts and 1000/250
offset spacing are choices: they materially broaden coverage while keeping a
bounded, deterministic, source-ordered protocol and avoiding random sampling.
No row may be substituted after results.

**Implementation choice — training budget:** Use exactly one pass over the 24
listed training examples and exactly 24 optimizer updates, batch size 1,
gradient accumulation 1, and no scheduler. Preserve S10-F's AdamW learning
rate `0.001`, weight decay `0.0`, KD temperature `2.0`, routing temperature
`1.0`, epoch count `1`, and logging interval `1`. Explicitly record AdamW's
preserved defaults `betas=[0.9,0.999]`, `eps=1e-8`, and `amsgrad=false`.
The 24-example/24-update values are choices required to pair the sixfold data
scope with one pass; the other optimizer values are carried forward rather
than tuned.

**Implementation choice — paired matrix and audits:** Retain exactly the
S10-F seeds `[1729,1730,1731]` and lambdas `[0.0,0.03,0.1]` in that order,
with no added seeds. For each seed, create one fresh canonical three-way
router initialization and clone it identically before each lambda; construct a
fresh AdamW per lambda, include only `routers.` parameters, and freeze both
teacher and packed base. Record exact training counts, finite/freeze/optimizer
audits, soft and hard KD/logit/width/probability/entropy metrics, all twelve
72-entry validation route maps in layer-major order, route variation,
reproducibility, and cross-seed paired comparisons. Retaining three seeds,
three lambdas, 72 units, and one immediate repeat is a compatibility choice
from S10-F, not a new source-paper fact. The future result schema also requires
an explicit collapse audit, a structured optimizer audit with identity-based
membership and fresh-state evidence, a run-level inherited-regression audit,
and a run-level prohibited-work audit; booleans alone are not sufficient proof.

**Implementation choice — future gate:** Use only two lower-is-better axes,
hard validation KD and hard selected width. A lambda is per-seed
non-dominated when no configured lambda is no-worse on both axes with one
strict improvement. CONTINUE requires all nine trials/audits and inherited
regressions, no invalid collapse, `0.03` frontier membership in at least 2 of
3 seeds, median paired hard KD delta `<=0.0`, median paired hard-width delta
`<0.0`, and zero reproducibility failures. A complete structurally valid
matrix failing a two-axis condition is REFINE; invalid/drifted evidence or
prohibited work is REVISE; incomplete evidence is PAUSE. The threshold 2/3,
zero KD delta, strict negative width delta, and zero failures are frozen
implementation gates carried forward from S10-F. Outcome precedence is PAUSE
for missing or incomplete evidence, REVISE for complete but failed integrity,
inherited-regression, collapse, reproducibility, or prohibited-work evidence,
REFINE for valid complete evidence that misses a two-axis threshold, and
CONTINUE only when every required condition passes. No scalar score, arbitrary
quality-loss threshold, or production-lambda selection is allowed. Soft mean
entropy uses the inherited base-2 logarithm.

**Boundary:** This decision freezes only the protocol and configuration tests.
It creates no `scripts/run_s10g.py`, result JSON, or execution path and
performs no training, evaluation, or hardware/resource measurement. Adaptive
lambda search, post-result seed/example replacement, warm starts, S07
conversion, teacher/base training, non-router optimizer membership, candidate
or normalized-cost changes, S08 changes, six-bit on-demand loading,
production-lambda selection, and S10-H execution are prohibited.

**Evidence and consequence:** The configuration-only S10-G test passed `40`;
the S10-D/S10-E/S10-F predecessor selection passed `121`; Ruff and
`git diff --check` passed. The S10-G protocol-freeze outcome is CONTINUE, not
an experiment result. A separately authorized future action is required before
any broader validation execution.

**Reversal path:** If data identity/order, tokenizer, training budget,
optimizer membership, freeze audit, route-map schema, reproducibility contract,
gate threshold, or prohibition changes, mark S10-G REVISE, preserve this
configuration and all S10-F artifacts, and record a replacement decision
before any execution.
