Current stage: S10-D
Status: COMPLETE

## S09-A closeout — canonical validation gate

PR #5 landed the frozen S09-A protocol and validator corrections at merge
commit `0f5802a777983c210b6f65ca26fd55368f49bf51`. The implementation and
review fixes are already merged; this closeout records the completed
validation gate rather than treating them as pending changes.

S09-A is **COMPLETE**. The frozen configuration and fixed inputs were not
changed. The canonical full validator passed with hashes enabled:

```text
source ~/.venv/bin/activate
which python
python --version
python scripts/validate_s09_protocol.py --config configs/s09_baseline_eval.json
```

The validator exited `0`, checked all five modes, seven fixed requests, and
32 quality windows over 4096 target tokens. Packed artifact SHA-256 matched
`29d9bc526b3da0bd39daf2f82afd141f82d005ca1232cabc75cfe9d9ecc1cfee`; the S07
router checkpoint matched
`08bf646f19759c0d7949e159bdbe4f96bbea737204b96f8760d205c8d6fd1949`; the
Qwen3-4B model and tokenizer matched revision
`1cfa9a7208912126459214e8b04321603b3df60c`; and the Any-Precision submodule
matched `a3257d02740cc5757c78673da534b0630ff3a4ea` in both the gitlink and
checkout. The frozen protocol/config SHA-256 is
`01ca65c6b3b7e16d7af66f1533140b1c9f31749c90bc91e097d096d463bf2e1c`.

The focused S09-A command passed `18 passed`:

```text
PYTHONPATH=src:. pytest -q tests/unit/test_s09_protocol.py tests/integration/test_s09_protocol_inputs.py tests/integration/test_perplexity_evaluator.py
```

Ruff passed for the validator and focused S09-A test files. At S09-A closeout, no S09-B benchmark, five-mode baseline evaluation, or
final result artifact existed. At that point S09-B execution machinery was
**MISSING**; the committed S09 script was only the non-benchmark protocol
validator.

## S09-B1 runner implementation — CONTINUE

S09-B1 adds `scripts/run_s09b.py` and `qaq.s09_runner`. The parent resolves the
five frozen mode IDs and launches one explicit `--execute-mode` child per mode,
so no process can retain models for a second mode. The default path is the
non-executing plan, which invokes the canonical validator, prints child and
aggregation commands, and writes no result.

The runner consumes `configs/s09_baseline_prompts.json`, passes S09's explicit
32-window, stride-128, 4096-target perplexity arguments to the S03 evaluator,
records fixed-input generation, routed 72-unit maps, S08 physical transfer
accounting, request cleanup, allocator boundaries, and five raw latency repeats.
The per-mode schema and aggregation path validate identities, deterministic
evidence, release gates, route/output agreement, transfer equality, cleanup,
and hidden-copy audits. Missing real results classify as PAUSE; structural or
quality failures classify as REVISE; complete validated results classify as
CONTINUE.

Non-benchmark evidence for S09-B1:

- Canonical S09-A validator passed with hashes enabled.
- `python scripts/run_s09b.py --plan --config configs/s09_baseline_eval.json`
  passed and resolved all five child commands plus the aggregation command.
- Focused runner tests passed: `8 passed`.
- No mode child was launched, no model evaluation ran, and no final S09 result
  artifact was created.

Current stage: S09
Status: IN_PROGRESS
S09-B1R: runner correctness repair required before execution. The correction
keeps the frozen protocol and fixed inputs unchanged, preserves measured S08
cleanup and physical residency evidence, computes and validates five-repeat
latency medians, records deterministic repeat evidence, enforces exact
hardware and perplexity identities, validates packed identities for every
packed mode, and persists the aggregation classification to `aggregation.json`.
No S09-B mode was executed and no S09-B result artifact exists.
Next action: Execute S09-B: run the frozen five-mode baseline evaluation using
the corrected and verified S09-B runner and configs/s09_baseline_eval.json, then
evaluate the frozen release gates.

## S09-A protocol owner

The authoritative machine-readable protocol is
`configs/s09_baseline_eval.json`; its fixed inputs are
`configs/s09_baseline_prompts.json`. The detailed human-readable procedure and
validation gate are owned by `docs/stages/S09_BASELINE_FREEZE.md`. D031 records
the freeze decision and D032 records the validator review follow-up; this
status page records only the current state and evidence. No S09-B benchmark or
final quality, memory, latency, or transfer conclusion exists.

S00 through S06 are COMPLETE. S07-A is complete with reusable teacher-student
distillation machinery, explicit completion masking, frozen teacher/packed
student evidence, router-only optimization, deterministic hard routes, compact
route logs/statistics, and router-only checkpoint round trips.

S06 evidence:
- 72 distinct routers: one per attention or FFN unit across 36 layers.
- Qwen3-4B router configuration: hidden width 128, GELU, parameter-free RMS
  normalization with epsilon 1e-6, temperature 1.0, and canonical output
  ordering `[p4, p8]`.
- Full router parameter count: 23,620,752.
- Every soft unit executes both real pinned packed paths and mixes them without
  hard selection.
- Forced 4-bit and 8-bit endpoints match the verified S03/S04 executions within
  the documented `atol=1e-3`, `rtol=1e-3`; synthetic pinned-backend endpoints
  are bitwise equal.
- Focused real packed S06 soft-routing regression: 2 passed against
  Any-Precision commit `a3257d02740cc5757c78673da534b0630ff3a4ea`; the
  artifact-dependent Qwen3 endpoint test remains skipped because the S03-B
  artifact is absent in this worktree.
- Probability, shape, finite-value, temperature, attention-sharing, FFN-sharing,
  gradient, optimizer-step, and frozen-model checks pass.
- S06 focused suite: 14 passed.
- Unit regression suite: 67 passed.
- Artifact-backed S04/S05/static regression selection: 12 passed.
- No real dataset training, distillation, hard argmax inference, or on-demand
  loading was performed.

Passing S06 implementation commit: `8f59215`.

S07-A evidence:
- 9 focused S07 unit/integration tests passed on the deterministic tiny
  fixture; both smoke steps had finite KD loss and finite router gradients,
  and router parameters changed.
- Teacher parameters and packed S06 student parameters remained frozen and
  unchanged; the optimizer audit included only `routers.` parameters.
- Explicit completion-mask tests proved prompt and padding changes do not
  affect loss, completion changes do affect loss, and zero-completion inputs
  fail. Alignment, hard argmax, route-log coverage, statistics, and checkpoint
  probability/hard-route round trips passed.
- Relevant S04-S06/S05 regression selection: 40 passed, 11 artifact-dependent
  tests skipped because the disposable worktree has no S03-B artifact. No real
  baseline training was run.

S07-B evidence:
- The first run remains recorded as D027 **REVISE** because its teacher-freeze
  audit did not explicitly set teacher parameters to `requires_grad=False`.
  It used `no_grad`, excluded the teacher from the optimizer, and left teacher
  values unchanged.
- D008-1 authorized exactly one corrected rerun with the unchanged locked
  configuration. The locked configuration remains in
  `configs/s07_router_training.json`: four deterministic Wikitext training
  examples, two validation examples, 32-token prompt/completion boundaries,
  sequence length 64, batch size 1, AdamW, four steps, KD temperature 2.0,
  routing temperature 1.0, and seed 1729.
- The corrected production path explicitly froze the teacher before logit
  precomputation. Teacher `requires_grad=False`, no gradients, matching
  before/after hashes, unchanged packed-student non-router hashes, router-only
  optimizer membership, and the 23,620,752 router scalar count all passed.
- KD loss decreased from `0.1730574965` to `0.0317778103`; all losses and
  router gradients were finite and router parameters changed. The objective
  remained completion-only teacher-student distillation with no extra penalty.
- Soft validation KD/error were `0.0386699643`/`0.2430240735`; hard
  validation KD/error were `0.0631424394`/`0.2928081304`. Static 4/8-bit errors
  were `0.7434162199`/`0.0910567641`. Hard 4/8 fractions were `20.1389%`/
  `79.8611%`; attention 4/8 fractions were `29.1667%`/`70.8333%`; FFN 4/8
  fractions were `11.1111%`/`88.8889%`. There were two route maps, prompt
  distance `0.0138889`, and complete 72-unit logs for each validation request.
- The corrected values exactly matched the first run, with no material
  numerical difference. Fresh-process checkpoint reload and fixed-subset
  deterministic hard-route repeats passed bitwise. Adaptivity remains
  `OTHER`, a non-blocking observation under the existing S07 gate. No S08
  work was started.

Result artifact: `docs/results/s07_router_training.json`.

Passing corrected D008-1 evidence commit: `33631f5`.

## S07C-EVIDENCE-005 — hard-route checkpoint round-trip evidence repair

Status: RESOLVED — CONTINUE.

The previous fresh-process verifier proved checkpoint reload, probability
equality, and equality of a `hard_bit` recomputed from reloaded soft
probabilities against the recorded soft route logs. It did not prove that the
route actually selected by hard execution matched the original S07-B hard
route record. This repair adds that missing invariant without changing router
semantics, the `[4, 8]` candidate ordering, training data, objective,
checkpoint, or any router parameters.

The new comparison builds the expected keyed map
`(request_id, layer, unit_type) -> hard_bit` from
`evaluation.hard.route_logs` and compares it with the routes selected during
fresh-process `hard_once()` execution from
`QaqRequestState.attention_routes[layer]` and
`QaqRequestState.ffn_routes[layer]`. It rejects missing or unexpected keys,
duplicate keys, `None` or unsupported precisions, layer/unit mismatches, and
coverage other than exactly 36 attention plus 36 FFN routes (72 total) per
request. The weaker soft-derived comparison remains separately named in the
result artifact as `soft_derived_hard_route_comparison`.

Exact verification command from the repository root (the isolated worktree
used its absolute worktree-root equivalent for `cd projects/QAQ`):

```text
source ~/.venv/bin/activate
which python
python --version
nvidia-smi
PYTHONPATH=src:third_party/any-precision-llm python scripts/verify_s07b_roundtrip.py --device cuda:3 --result docs/results/s07_router_training.json
```

Measured result in `docs/results/s07_router_training.json`: checkpoint
SHA-256 `08bf646f19759c0d7949e159bdbe4f96bbea737204b96f8760d205c8d6fd1949`
matched the required identity; both validation requests had complete keyed
coverage of 36 attention and 36 FFN recorded/actual routes; exact matches
were attention `72/72`, FFN `72/72`, total `144/144`; mismatch count was `0`;
probabilities and soft-derived bits matched; repeated actual route maps and
selected precisions matched; repeated hard logits were bitwise equal; logits
were finite; and packed-student invariants remained unchanged. The focused
regression passed `1`, existing S07 checkpoint/round-trip tests passed `9`,
and the relevant S06/S07 structural router suite passed `9`.

No S07-B training or retraining occurred. S10-B was not started; the next
action is **Begin S10-B: Three-Way Router Semantics.**

S08-A evidence:
- The implementation subdivision S08-A established a synchronous loader for
  one concrete request state and retained no process-global request cache.
- The real S01 pinned fixture keeps `[8,64,32]` `torch.int32` qweight and
  `torch.float16` row LUTs on CPU before first use.
- First-use bytes were 34,816 for 4-bit, 98,304 for fresh 8-bit, and 65,536
  incremental bytes for a 4-to-8 upgrade. Reuse events transferred zero
  bytes.
- Resident and transferred 4-bit and 8-bit fixture outputs were bitwise equal.
  Request end released all retained GPU references, and duplicate textual IDs
  used independent request-state ownership.
- Focused S08-A tests passed: 8. Ruff passed for changed source and tests.
- No full-model Qwen3 on-demand evaluation, memory comparison, latency
  comparison, or S09 work was performed.

S08-A gate: CONTINUE.

S08-B evidence:
- The external Codex service-overload interruption was classified as infrastructure interruption, not a QAQ defect.
- The required S03-B packed artifact, S07 router checkpoint, pinned Qwen3 snapshot, pinned Any-Precision revision, and CUDA device were present and matched their recorded hashes.
- Real Qwen3 on-demand execution used 252 CPU-authoritative packed sources, with zero remaining `AnyPrecisionLinear` modules and no complete packed GPU copy.
- Resident and on-demand hard routes matched for both locked S07 validation requests.
- Both routes produced finite logits that were bitwise equal, with zero mean and maximum absolute logit difference.
- Four-token deterministic greedy generation matched between resident and on-demand modes for both requests, and routes remained fixed during decode.
- On-demand transfer accounting was `3,817,717,760` bytes for `validation-3` and `3,835,002,880` bytes for `validation-1000`; both matched the independent expected-byte calculation exactly.
- All transfer occurred during prefill, with zero decode transfer bytes and zero reuse transfer bytes.
- Each request retained 252 entries and 504 packed GPU buffers before cleanup, then retained zero entries, buffers, or packed bytes after `end_request()`.
- A later fresh request transferred its selected packed buffers again, proving request isolation.
- Synchronized two-repeat measurements recorded resident median prefill/decode/end-to-end latencies of `0.145354`/`0.187833`/`0.332110` seconds and on-demand medians of `5.815631`/`0.229669`/`6.031509` seconds.
- Resident peak allocated memory was `5,724,945,408` bytes at maximum across repeats; on-demand peak allocated memory was `4,806,114,304` bytes.
- Focused S08-B real tests passed: `3 passed in 438.03s`; S08-A focused tests remained `8 passed in 8.55s`; Ruff passed for all changed S08 files.
- The valid recorded S08-B regression result remains `8 passed in 651.74s`; it was not rerun because no relevant implementation or execution-path change invalidated it.
- Complete evidence and provenance are recorded in `docs/results/s08_on_demand.json`, including code snapshot hashes, model and artifact revisions, request digests, method, transfer records, allocator measurements, and commands.

S08 gate: COMPLETE.

Passing S08 implementation and evidence commit: `ee0d5e22b64713e97fb33596f60f0080f3b26df3`.
Next action at the S08 gate was to define S09-A; that protocol and its review
follow-up are recorded above. S09-B remains deferred until the current gate
is completed.

## S09-B3 routed decode diagnosis — REVISE

S09-B2 preserved all five mode results, but routed resident/on-demand logits
were not equivalent under the frozen bitwise criterion. Artifact-only analysis
found matching route maps and generated token IDs for all seven requests;
resident `s03-quality-3` repeated generation diverged at zero-based generated
token position 6, while on-demand generated tokens remained stable.

The narrow S09-B3 diagnostic used only routed resident and synchronous
on-demand modes, `s03-quality-3` and `validation-3`, on `cuda:3`, with seed
1729. Prefill logits were bitwise equal. Decode logits diverged at the first
step while selected tokens still matched. At the representative real shape
`[1,1,9728]` with 8-bit routing, repeated pinned `matmul_kbit` outputs were
not bitwise stable and resident/on-demand outputs differed; repeated
`dequant_kbit` plus `torch.matmul` outputs were bitwise stable and equal.
The pinned kernel's `M=1`, `K>4096`, 8-bit k-split path uses atomic accumulation.

No frozen protocol/configuration, production execution code, or preserved
S09-B2 result file was changed. S09 remains IN_PROGRESS. The next action is a
separately authorized narrow repair decision and targeted routed re-evaluation;
the repair has not been tested and S09 must not be marked complete.

## S09-B4 deterministic routed packed execution repair — CONTINUE

The pinned kernel dispatch was source-verified: on non-Orin devices it uses
atomic k-split accumulation exactly for effective `M == 1`, packed input width
`K > 4096`, and `w_bits >= 7`. Under QAQ's locked 4/8-bit routes, the affected
family is 8-bit one-row calls with `K > 4096`. The shared helper in
`qaq.s08_loader` uses pinned `dequant_kbit` plus `torch.matmul` only for that
family and preserves the existing packed path elsewhere. Resident
`_RoutedPackedLinear` and synchronous on-demand loader calls share the helper.

The Qwen3 target inventory contains 252 projections. The fallback can apply
to the 36 `model.layers.<i>.mlp.down_proj` projections (`in_features=9728`)
when selected at 8-bit. The other 216 targeted projections have
`in_features=2560` and retain `matmul_kbit` for both supported precisions.
The FP teacher and static packed paths are untouched by the diff.

Focused real-shape dispatch tests passed `2`; the relevant regression selection
passed `56`; the real S08 hard-routed regression passed `3`; and the follow-up
tiny Qwen3/backend selection passed `8`. Ruff passed. The frozen protocol
validator passed. Narrow CUDA validation on `cuda:3` passed for
`s03-quality-3` and `validation-3`: prefill and all eight decode logits were
finite and bitwise equal between resident and on-demand, route maps and tokens
matched, and five repeated `s03-quality-3` generations were stable in both
modes with matching per-step logits digests and sequences. On-demand transfer
remained packed-only and matched expected bytes exactly (`3,835,002,880` for
`s03-quality-3`; `3,817,717,760` for `validation-3`), decode transfer was zero,
cleanup returned entries, buffers, and bytes to zero, and the hidden-copy audit
passed. No persistent dense/dequantized model state was introduced.

The pinned Any-Precision submodule remains clean at
`a3257d02740cc5757c78673da534b0630ff3a4ea`. The frozen config/input hashes
remain `01ca65c6b3b7e16d7af66f1533140b1c9f31749c90bc91e097d096d463bf2e1c`
and `da1d33f0f2330cfc341c38945fe4b205f946223f8c9069c35d44999d400fbb49`.
All six failed S09-B2 artifacts remain byte-for-byte unchanged. No final S09
rerun was executed. Corrected routed quality, resource, and latency results
remain unknown; the original routed S09-B2 results are invalidated, while
unaffected FP/static evidence remains usable only after the execution-path
check recorded in D036.

Current stage: S09
Status: COMPLETE

## S09-C final evidence review and baseline freeze

The S09-B5 committed aggregation is `CONTINUE` with no errors. The read-only
closeout aggregation also returned `CONTINUE` with no errors. The frozen
protocol and fixed-input hashes remain unchanged.

Passing routed repair commit: `4a0dc702178fef0f84eb9ffd9bd6d1810e5dc564`.
Passing final evidence commit: `443f6994582500857afca9bad6032cc285448a86`.
Canonical final evidence: `docs/results/s09b_b5/`.
Frozen protocol SHA-256:
`01ca65c6b3b7e16d7af66f1533140b1c9f31749c90bc91e097d096d463bf2e1c`.
Frozen fixed-input SHA-256:
`da1d33f0f2330cfc341c38945fe4b205f946223f8c9069c35d44999d400fbb49`.

`docs/results/s09b/` is preserved failed S09-B2 evidence and is not the
canonical final baseline. No production code, measurement code, configs,
frozen inputs, pinned dependencies, or result JSON changed during closeout.
The focused closeout suite passed `28 passed`.

Established: S09-A froze the protocol before final results; S09-B2 returned
REVISE because routed decode logits were not reproducible; S09-B3 isolated the
pinned atomic k-split `matmul_kbit` path; S09-B4 repaired only the proven routed
dispatch family; and S09-B5 reused the unaffected FP/static results while
rerunning only the invalidated routed modes. B5 passed finite-output,
deterministic-repeat, route-map, generated-token, logits-digest, transfer,
cleanup, and hidden-copy criteria, plus both frozen quality gates.

Unknown and not claimed: this is not an exact QAQ paper-score reproduction;
route diversity remains observational `OTHER`; no post-baseline asynchronous,
prefetch, caching, or other optimization was tested; and no claim is made that
synchronous on-demand loading is faster than the resident baseline.

Next action: Baseline frozen. Stop. Define an explicit post-baseline stage and
decision before implementing any optimization or additional research mechanism.

Current stage: S09
Status: COMPLETE

## S10-A — static six-bit execution

Current stage: S10-A
Status: COMPLETE

Gate outcome: CONTINUE.

S10-A is complete on implementation commit
`b7300e1621f9c5d2ac5c8c9e1b0c01fb092f6426`. The existing identity-matched
Qwen3 artifact remains at
`quantized/s03b_qwen3_4b/backend_cache/packed/anyprec-(1cfa9a7208912126459214e8b04321603b3df60c)-w8_orig4-gc1-c4_s1_blk64`
with recorded model hash
`29d9bc526b3da0bd39daf2f82afd141f82d005ca1232cabc75cfe9d9ecc1cfee`.
All 252 targets retain one `[8,N,K//32]` `torch.int32` parent qweight;
LUT6 inventory is 252 finite `torch.float16` `[N,64]` tensors totaling
141,557,760 bytes, and the selected six-plane payload is 2,724,986,880
bytes. The pinned Any-Precision submodule is clean at
`a3257d02740cc5757c78673da534b0630ff3a4ea`.

Real precision-6 backend execution matched the pinned dequantizer/reference
under `atol=0.05`, `rtol=0.01` (`max_abs_error=0.015625`), was bitwise
deterministic, and had no persistent dense weight. Full Qwen3 static-6 smoke
returned finite `[1,8,151936]` logits with deterministic digest
`4e0856454ebab64588183a1e72acc2fc34ffea68d82c590526624edd804e3390`.
Unit, S10-A integration, existing static 4/8/inventory/duplicate/byte/
checkpoint, and S06/S07 structural suites passed as recorded in
`docs/stages/S10_6BIT_ROUTING.md`. Router semantics remain 4/8; no 6-bit
routing stage was started.

Next action: S07C-EVIDENCE-005 is resolved; await separate instruction before
beginning S10-B, the next 6-bit routing stage.

## S10-B — Three-Way Router Semantics

Current stage: S10-B
Status: COMPLETE

Gate outcome: CONTINUE.

S10-B is complete on implementation commit `f9e7c38`. Learned-router
candidate ordering is explicit and validated as exactly `(4,8)` or `(4,6,8)`.
The historical default remains `(4,8)` with probability order `[p4,p8]`; the
new explicit router emits `[p4,p6,p8]`, stores matching request-owned state,
executes real packed 4/6/8 mixtures, maps hard argmax index 1 to 6, and records
candidate ordering in traces, route observations, and checkpoint metadata.

The historical router count remains 72. Verified counts are 23,620,752 scalars
for `(4,8)` and 23,630,040 for `(4,6,8)`, an increase of 9,288. The historical
S07 checkpoint SHA-256 remains
`08bf646f19759c0d7949e159bdbe4f96bbea737204b96f8760d205c8d6fd1949`; a fresh
historical checkpoint load passed, and synthetic three-way checkpoint
round-trip plus both mismatch directions were rejected correctly. The pinned
Any-Precision revision remains clean at
`a3257d02740cc5757c78673da534b0630ff3a4ea`.

Verification passed: full unit suite `120 passed`; focused lifecycle and S08
regressions `14 passed`; real pinned packed three-way fixture `1 passed`; and
artifact-backed Qwen3 three-way forced 4/6/8 endpoints `1 passed in 421.02s`.
Ruff passed for all changed source and S10-B tests. The required Python
preflight resolved `/nfs/home/s314511048/.venv/bin/python`, Python `3.12.3`,
and RTX 3090 GPUs through `nvidia-smi`.

No training or retraining occurred. No cost-aware objective or penalty
coefficient was added. S08 on-demand loading remains 4/8-only; no 6-bit
on-demand support was introduced. Historical S07/S09 results, the packed
artifact, Any-Precision source, and historical checkpoint were not modified.
No quality, latency, memory, transfer, or routing-quality evaluation was run.

Historical next action: Begin S10-C: define and validate the cost-aware 4/6/8
router objective.

## S10-C — Cost-Aware 4/6/8 Router Objective

Current stage: S10-C
Status: COMPLETE

Gate outcome: CONTINUE.

S10-C adds only reusable normalized cost-objective composition primitives.
S07's `masked_kl_distillation_loss()` remains unchanged and remains the
completion-only KL objective. `expected_bit_cost()` constructs explicit costs
`[0.0, 0.5, 1.0]` for `(4,6,8)` and `[0.0, 1.0]` for historical `(4,8)`;
`mean_expected_bit_cost()` and `request_state_expected_bit_cost()` compute the
unweighted mean across every included decision, exactly once per attention and
FFN layer. Three-way diagnostic width is `4 + 4*L_bit`.

`cost_aware_distillation_loss()` composes `L_total = L_KD + lambda_bit*L_bit`.
The cost weight is explicitly validated as finite, numeric, non-negative, and
non-boolean. Zero is the backwards-compatible default and no nonzero
production lambda was selected. Request-state probability clones remain
attached to autograd. The objective is a normalized bit-plane-count surrogate,
not latency, memory, transfer, energy, or kernel-runtime weighting.

Focused S10-C tests passed `9`; S07 distillation, request-state, and S10-B unit
regressions passed `34`; the real pinned packed S10-B fixture passed `1`; the
full unit suite passed `127`; and Ruff passed for changed source and tests.
The required preflight resolved `/nfs/home/s314511048/.venv/bin/python`,
Python `3.12.3`, and healthy RTX 3090 visibility. No training, checkpoint
creation, production lambda selection, artifact-backed Qwen3 execution, S08
loader/artifact/Any-Precision change, historical-result rewrite, or unrelated
refactor occurred. Changed paths are limited to the objective/state seam, its
focused tests, and stage/decision/status documentation.

Next action: Begin S10-D. Do not begin it in this task.

## S10-D — Bit-Cost Coefficient Calibration

Status: COMPLETE
Gate outcome: CONTINUE.

S10-D executed the complete locked lambda grid on the required starting
commit `41e598b0e00e9b72444b498c5cd39b2f335c2257`, using the identity-matched
Qwen3-4B teacher, packed artifact, Wikitext revision, clean Any-Precision
`a3257d02740cc5757c78673da534b0630ff3a4ea`, and free `cuda:0`. Static 4/6/8
references were measured first with finite logits. Every trial reset the same
seed-1729 three-way router-only state, verified 72 routers and 23,630,040
scalars, used a fresh AdamW, and ran exactly four updates with the locked S07
examples, order, masks, temperatures, and optimizer values.

Evidence:
- Focused S10-D plus S10-C/S10-B/S07/request-state regressions passed `44`;
  Ruff passed for the runner and focused tests.
- All five grid points completed: `0.0, 0.003, 0.01, 0.03, 0.1`.
- No adaptive point was authorized by the observed triggers.
- Initial router hashes matched across all trials; teacher and packed base
  hashes were unchanged; optimizer audits were router-only and fresh; all
  losses, gradients, widths, probabilities, and logits were finite.
- Hard routing selected 6 on validation for every trial. The hard frontier is
  observed at `0.03` and `0.1`; the soft frontier at `0.0, 0.003, 0.03, 0.1`.
  These are not a production selection.

Canonical result: `docs/results/s10d_lambda_calibration.json`.
Protocol/config: `configs/s10d_lambda_calibration.json`.
Stage procedure and limitations: `docs/stages/S10_6BIT_ROUTING.md`.
No historical result, production checkpoint/lambda, S08 loader, packed
artifact, Any-Precision source, or S07 runner was changed.

Next action: firstmate/captain reviews the observed frontier and decides
whether to refine, confirm, or begin full training.
