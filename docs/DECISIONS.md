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

**Choice:** Initially support only 4-bit and 8-bit routes.
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
