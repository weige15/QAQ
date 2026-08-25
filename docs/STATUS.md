Current stage: S11-D1
Status: COMPLETE — PAIRED 4/6/8 TRAINING PROTOCOL FROZEN; NO TRAINING EXECUTED

## S11-D1 paired lookahead-specific 4/6/8 training protocol freeze

S11-D1 freezes the pre-result protocol in
[`docs/stages/S11D_PAIRED_LOOKAHEAD_468_TRAINING.md`](stages/S11D_PAIRED_LOOKAHEAD_468_TRAINING.md).
The exact question is: **Can a 4/6/8 router trained for one-layer-lookahead
attention routing achieve meaningfully lower selected precision while
preserving acceptable teacher-relative quality?** The protocol does not assume
that lookahead reduces width; canonical S11-C3 instead observed a
`+0.06481481481481488`-bit overall treatment delta for the historical 4/8
checkpoint.

The smallest fair experiment is a predeclared `2 timings × 2 lambdas × 3
seeds` matrix: `same_unit_468_control` and
`lookahead_attention_one_unit_468_treatment`, each at `lambda_bit=0.0` and
`0.03`, for seeds `1729/1730/1731`. Within each seed all four cells clone the
same canonical three-way initialization and hold exact S10-H 24-example
training data, 12-example validation data, AdamW values, 24-step budget,
`[p4,p6,p8]` order, teacher/base freeze, metric operations, and final-state
selection equal. `0.03` is retained only as the smallest S10-supported
positive-cost probe; S10-H showed a `-0.4907407407407405`-bit median hard-width
change but a quality-failing `+0.014972516723598044` hard-KL delta. It is not a
production selection, and the known-unfavorable `0.1` point is excluded.

Quality and precision remain separate. Quality is judged primarily against the
lookahead arm's paired zero-cost cells using S10's median hard-KL non-degradation
rule, with S11's per-request `1.25` KL and aggregate `1.10` mean-error timing
safeguards. Meaningful precision requires the lookahead `0.03` median hard
selected width to be at least `0.4907407407407405` bits below its paired
zero-cost reference, with negative deltas in at least two seeds. The
contemporaneous same-unit `0.03` cells expose the fixed-cost timing contrast;
no combined score is allowed. Frozen CONTINUE, REFINE, REVISE, PAUSE, and STOP
rules prevent post-result changes.

No executor, config, checkpoint, result artifact, model/dataset load, CUDA
work, training, evaluation, loader change, prefetch, asynchronous transfer, or
latency/transfer/memory/throughput measurement occurred. Next action: stop.
A separate S11-D2 may implement and structurally validate an executor and
non-executing plan without changing this protocol; real training remains a
later, separately authorized S11-D3.

Current stage: S11-C3
Status: COMPLETE — CONTINUE; FROZEN BROADER QUALITY MARGINS PASS

## S11-C3 frozen twelve-request broader-quality execution

S11-C3 executed the unchanged S11-C1/D058 protocol with the verified S11-C2
executor. Preflight passed before the result parent was created: the protected
protocol and fixture SHA-256 values remained
`320c42901046d26c310d97fe1d3331d8653ce7c913daf3bff0bab7df02e585b5` and
`a33cb9a7373f6ed68216e31249317ee35f25dc86d1e095b6428843671e8f3a08`;
the canonical S10-H result and S11-B3 control matched their protected hashes;
and the pinned model/tokenizer, packed artifact, historical S07 checkpoint,
and Any-Precision identities were present and exact. The inert plan was
byte-identical across two invocations, named only the two frozen fresh-process
modes followed by aggregation, and reported no execution or write activity.

The exact execution order was `same_unit_control`,
`lookahead_attention_one_unit_treatment`, then paired aggregation. Both fresh
mode processes used `cuda:0`, physical GPU
`GPU-384b6377-8f0c-e3d2-8b3a-b3408b54fd53` (NVIDIA GeForce RTX 3090,
driver `580.159.03`), with identical recorded hardware/software identities,
seed `1729`, all twelve frozen requests, and two immediate deterministic
repeats. Independent validation passed both mode schemas and the aggregation;
each mode contains 864 target-owned 4/8 routes, bitwise-equal repeat logits,
identical repeat routes and provenance, finite quality evidence, exact S11-B3
control overlap, immutable teacher/base/packed/router state, complete cleanup,
and no prohibited work. Independent aggregation recomputation was byte-for-
value equal to the canonical aggregation and reported no errors.

The complete valid result is **CONTINUE**. Treatment aggregate KL was
`0.053326172598948084` versus control `0.05696516142537197`, ratio
`0.9361190465300235`, passing the frozen `<= 1.10` factor. All twelve paired
request KL ratios passed the frozen `<= 1.25` factor, in request order:
`0.9431116083999646`, `0.8171828196666195`, `0.8609327298114132`,
`0.9519197144964956`, `1.0`, `1.0`, `0.9851409897234885`,
`0.881359038358788`, `0.9528381087573128`, `1.0`, `0.8712474506479513`,
and `0.8196586142853123`. Treatment aggregate mean absolute logit error was
`0.2549600688119729` versus control `0.2632628021140893`, ratio
`0.968462185939515`, passing the frozen `<= 1.10` factor. Aggregate maximum
absolute error was diagnostic only: control `4.119384765625`, treatment
`4.208902994791667`.

Across 864 paired target decisions, 18 attention routes changed and no FFN
route changed: 16 transitions were 4-to-8 and two were 8-to-4. Overall Hamming
distance was `0.020833333333333332`; attention distance was
`0.041666666666666664`; FFN distance was `0.0`. Treatment-minus-control mean
selected width was `0.06481481481481488` bits overall,
`0.12962962962962976` for attention, and `0.0` for FFN. These route and width
diagnostics neither strengthen nor weaken the quality classification and do
not establish selected-precision savings.

Canonical evidence:

- `docs/results/s11c_broader_quality/same_unit_control.json`, SHA-256
  `df2035e67e780ee9fd148eba73846ce49cb906aab6c7213c67ca9fc79287aa9c`;
- `docs/results/s11c_broader_quality/lookahead_attention_one_unit_treatment.json`,
  SHA-256 `2a0acd8091473f1b9e60b07fa60e98f7b9918e7913cac061835722ad3c3a06f0`;
- `docs/results/s11c_broader_quality/aggregation.json`, SHA-256
  `49131444e8d25a884c15f1803b8f5d430158ee5142f2689cbffef2b193e532a2`.

CONTINUE establishes only that the historical 4/8 checkpoint passed the frozen
broader teacher-relative quality margins under one-unit-lookahead attention
timing. It does not prove precision savings or authorize training. Next action:
stop. A paired lookahead-specific 4/6/8 training stage may be defined only as a
separate pre-result protocol and separately authorized execution; do not begin
it automatically.

Current stage: S11-C2
Status: COMPLETE — EXECUTOR READY; BROADER EXPERIMENT NOT EXECUTED

## S11-C2 fail-closed broader-quality executor readiness

S11-C2 implements the frozen S11-C1 twelve-request 4/8 paired-quality contract
without changing its research question, requests, modes, metrics, thresholds,
route diagnostics, classifications, or advancement rule. The byte-protected
machine contract is `configs/lookahead_broader_quality.json`; the byte-protected
fixed fixture is `configs/lookahead_broader_quality_inputs.json`, containing all
twelve exact 64-token arrays whose little-endian signed-64-bit SHA-256 digests
match both S11-C1 and the canonical S10-H validation manifest.

The thin default command is `scripts/run_lookahead_broader_quality.py`. Its
standard-library-only plan prints two fresh-process mode commands in frozen
order followed by aggregation and reports false model, dataset, tokenization,
CUDA, experiment, training, benchmark, and write activity. Explicit execution
reuses the S11-B2 `ProductionRuntime` only after exact config, mode, CUDA device,
destination, parent, and no-overwrite checks. The scheduler, validators,
aggregation, detailed route diagnostics, four-way classification, and shared
same-directory atomic hard-link persistence all fail closed on incomplete,
drifted, non-deterministic, mutable, prohibited, or unsafe evidence.

CPU-only injected verification covered 12 requests, two repeats per mode, 864
routes per mode, overlap equality with canonical S11-B3 control, all three
quality-margin failures, PAUSE/REVISE/STOP/CONTINUE behavior, route transitions,
selected-width deltas, dispatch-before-import, deterministic inert planning,
and atomic no-overwrite persistence. Existing S11-B and S10 broader-validation
regressions remain required readiness evidence.

No Qwen3-4B model, packed artifact, production teacher/student inference, CUDA,
training, retraining, optimizer, checkpoint creation, 6-bit route, loader or
prefetch change, generation, decode, perplexity, benchmark, profiler, or
performance measurement ran. No canonical `docs/results/s11c_broader_quality/`
parent or result exists. Quality, selected-precision benefit, and advancement
remain unknown. Next action: stop and obtain separate authorization before
executing the two frozen real mode commands and aggregation.

Current stage: S11-C1
Status: COMPLETE — BROADER QUALITY PROTOCOL FROZEN; EXECUTION NOT STARTED

## S11-C1 broader lookahead quality protocol freeze

The exact question is: **Does the historical S07 4/8 checkpoint continue to
preserve acceptable teacher-relative quality under one-layer-lookahead
attention routing on a meaningfully broader fixed evaluation set than the
two-request S11-B pilot?**

The protocol-only stage is frozen in
[`docs/stages/S11C_BROADER_QUALITY.md`](stages/S11C_BROADER_QUALITY.md). It
compares `same_unit_control` with
`lookahead_attention_one_unit_treatment`, in that order, using the unchanged
historical S07 4/8 checkpoint and twelve exact 64-token validation requests
from the canonical S10 broader-validation manifest. The fixture retains both
S11-B requests and adds ten source-fixed requests in deterministic order, for
six times the pilot coverage. Model, tokenizer, packed artifact, backend,
checkpoint, resident execution, seed `1729`, two exact repeats, metric
operations, and all non-timing behavior are held equal.

CONTINUE requires valid deterministic evidence plus treatment aggregate KL
`<= 1.10 *` control, each of twelve treatment request KL values
`<= 1.25 *` its paired control, and treatment aggregate mean absolute logit
error `<= 1.10 *` control. PAUSE covers unavailable prerequisites or incomplete
external evidence; REVISE covers invalid protocol/executor/integrity or repeat
evidence; STOP covers complete valid evidence that misses any quality margin.
Route transitions and treatment-minus-control selected-width deltas are
required diagnostics but cannot change the quality classification.

S11-B3's one changed decision—`validation-3` attention layer 23 changed from 4
to 8 bits—did not reduce selected precision in the two-request pilot. That one
transition is neither proof nor disproof that lookahead can save precision.
Only a future CONTINUE result from this frozen broader 4/8 check is sufficient
to justify opening a separately scoped paired lookahead-specific 4/6/8
router-training stage; that stage still requires its own pre-result protocol
and execution authorization, and CONTINUE is not itself evidence of savings.

No executor, result schema, result artifact, model/dataset load, CUDA work,
quality experiment, training, 4/6/8 optimization, prefetch, asynchronous
loading, on-demand change, latency, memory, transfer, throughput, generation,
decode, or perplexity work occurred. Next action: separately implement and
structurally validate an executor and non-executing plan without changing this
protocol; real execution remains a later authorization.

Current stage: S11-B3
Status: COMPLETE — ADVANCE_TO_BROADER_QUALITY_CHECK

## S11-B3 frozen paired quality pilot

S11-B3 executed the unchanged frozen S11-B protocol, config SHA-256
`21a664424debe4892c3577c490158228dd5399bb4b425611db728070d23a5051`,
from the complete S11-B2 executor base. The exact order was
`same_unit_control`, `lookahead_attention_one_unit_treatment`, then aggregation.
Each mode ran once in a fresh process with two deterministic repeats on
`cuda:2`, physical GPU
`GPU-74d97f46-6284-1055-698a-e2db4e9c744b` (NVIDIA GeForce RTX 3090,
driver `580.159.03`). Both mode records contain identical hardware/software
identity evidence and passed independent schema, identity, input, finite-value,
route, historical-control, provenance, cleanup, freeze, determinism, and
prohibited-work validation.

The aggregation is valid and has no errors. All three frozen quality checks
passed: aggregate KL ratio `0.9824730210816205`; per-request KL ratios
`0.9431116083999646` and `1.0`; aggregate mean absolute logit-error ratio
`0.9929261653206376`. One target route changed: `validation-3` attention layer
23 selected 8 bits under treatment instead of 4; all other target routes were
equal, including every FFN and the required layer-0 units. The frozen
classification is **ADVANCE_TO_BROADER_QUALITY_CHECK**.

Canonical evidence:

- `docs/results/s11b_quality_pilot/same_unit_control.json`, SHA-256
  `ba748dd09b8319c1ff395f65be130ecbb0bea1571c1afb76e0016a88b6e5a073`;
- `docs/results/s11b_quality_pilot/lookahead_attention_one_unit_treatment.json`,
  SHA-256 `742450cfe5dda791cbbbdc59adf1541a2d897f227b9be094909f36b7760c402c`;
- `docs/results/s11b_quality_pilot/aggregation.json`, SHA-256
  `2b1755345bb0a8bbae3110bbdca86bf7dc75edef9c3460e83e37b0297fe626a7`.

This two-request result establishes only that the historical same-unit-trained
checkpoint passed the frozen pilot margins under one-unit-lookahead attention
timing. It does not establish general quality, performance, overlap, transfer,
or prefetch benefit. Next action: stop and separately define a broader quality
check; do not execute that later stage from S11-B3.

Current stage: S11-B2
Status: COMPLETE — executor ready, pilot not executed

## S11-B2 fail-closed paired quality-pilot executor

S11-B2 implements the frozen S11-B1 protocol on the current semantic component
layout. Commit `75ea036b826a534e940cc726b72268cf81f3bc08` was inspected only as
prior implementation evidence; no unrelated change from its history was
ported. The protocol config remains schema `qaq-s11b-quality-pilot-v1`, exact
SHA-256 `21a664424debe4892c3577c490158228dd5399bb4b425611db728070d23a5051`.

The exact planned result paths and their parent remained absent throughout:

- `docs/results/s11b_quality_pilot/same_unit_control.json`;
- `docs/results/s11b_quality_pilot/lookahead_attention_one_unit_treatment.json`;
- `docs/results/s11b_quality_pilot/aggregation.json`.

Implementation paths are `scripts/run_lookahead_quality_pilot.py`,
`qaq.evaluation.lookahead_quality_protocol`,
`qaq.evaluation.lookahead_quality_runner`, and
`qaq.evaluation.lookahead_quality_runtime`. The existing compatibility
validator remains a thin adapter. The default path is deterministic and
standard-library-only, validates the canonical protocol, prints the exact two
one-mode child commands and aggregation command, and reports false
model/CUDA/pilot/write activity. Exact-mode execution is behind validated
mode/device/output dispatch and a lazy production import. Per-mode validation,
keyed historical control comparison, paired aggregation, classification,
complete state audits, cleanup proof, and same-directory atomic no-overwrite
hard-link persistence all reject incomplete or inconsistent evidence.

CPU-only synthetic verification:

1. canonical B1 validator passed;
2. two inert plans were byte-identical, with normalized SHA-256
   `c848870a429940efbc194838e929da543b66c7553e0abc0a71ac245b2d02461c`;
3. focused B2 structural tests passed `107`, including direct heavy-import,
   canonical dispatch/no-overwrite, exact environment/CUDA resource
   classification, resident representation drift, resource-versus-identity
   preflight, and result-identity mutation coverage;
4. unchanged B1 protocol behavior plus the source-mode check passed `51`;
5. S11-A/request-state/router regressions passed `24`;
6. masked-KL/hard-route, result aggregation, and injected-runtime persistence
   regressions passed `31`;
7. dependency-direction and semantic-name checks passed `4`;
8. the complete safe CPU unit selection passed `470`, with one established
   duplicate-optimizer warning;
9. Ruff and `git diff --check` passed.

A deliberately CPU-forced invocation of the unfiltered unit directory reported
`411` passing and `12` expected failures from nine legacy CUDA-required files;
those GPU tests were not rerun because this stage explicitly forbids CUDA work.

No artifact-backed test or CUDA operation ran. No Qwen3-4B model, packed artifact,
S07 checkpoint, CUDA kernel, teacher/student inference, real metric, generation,
decode, perplexity, training, evaluation, benchmark, profiler, production
aggregation, or result path ran. Historical result/config/protocol bytes,
`papers/`, and `third_party/` remain unchanged. Quality transfer and every real
pilot classification remain unknown.

Next action: S11-B3 requires separate authorization. Provision only the empty
frozen result parent, execute the two printed one-mode child commands on one
explicit comparable GPU in frozen order, then execute the printed aggregation
command. Do not change the protocol, infer quality from structural tests, begin
broader quality work, or add asynchronous transfer/prefetch.

## Historical repository status

Repository organization: semantic naming migration COMPLETE

Active scripts, configuration files, tests, and reusable Python implementation
use behavior- or component-based names. Reusable implementation remains in the
PR1 package structure under `src/qaq`; no module was relocated or redesigned.
Former stage-numbered script commands are thin compatibility aliases for frozen
protocol and result-provenance commands, with no duplicate implementation. The
nine retained aliases are `run_s03b.py`, `run_s03c.py`, `run_s07b.py`,
`run_s08b.py`, `run_s09b.py`, `run_s10d.py`, `run_s10h.py`,
`validate_s09_protocol.py`, and `validate_s11b_protocol.py`; repository search
found no internal command record requiring the removed `run_s10f.py`,
`verify_s07b_roundtrip.py`, or `provision_s03_artifact.py` entry points.
Production dependency checks continue to enforce `src/qaq -> src/qaq` and
`scripts -> src/qaq`.

The migration changes no calculations, frozen protocol bytes, thresholds,
seeds, hashes, dataset/model revisions, result semantics, historical stage
documents, or frozen result artifacts. Evidence: the full unit suite passed
`372` with one established optimizer warning; the full integration suite
passed `63` with three expected external-artifact skips; all 12 semantic and
nine compatibility script entry points accepted `--help`; both protocol
validators and both baseline-plan command forms passed; Ruff passed on all
changed Python paths; and the final identifier audit found no stage-numbered
Python identifiers or active filenames outside the documented compatibility
aliases.

Current stage: S11-B1
Status: COMPLETE - protocol frozen

## S11-B1 deterministic paired quality-pilot protocol freeze

The frozen machine-readable protocol is
`configs/lookahead_quality_pilot.json`, schema `qaq-s11b-quality-pilot-v1`,
SHA-256 `21a664424debe4892c3577c490158228dd5399bb4b425611db728070d23a5051`.
It compares exactly `same_unit_control` then
`lookahead_attention_one_unit_treatment` on the two fixed 64-token S07
validation requests using the exact S09 model/tokenizer/packed/backend and S07
4/8 checkpoint identities. It freezes resident hard packed execution, two
repeats, completion-only temperature-2 KL, full-logit mean/maximum error,
complete target-owned routes/provenance, freeze audits, and the
`INVALID_EVIDENCE`/`PAUSE`/`ADVANCE_TO_BROADER_QUALITY_CHECK`/
`CHECKPOINT_REUSE_DEGRADES` interpretation contract before results.

The canonical standard-library validator passed. Focused S11-B1 tests passed
`50`; S11-A/request-state/router regressions passed `24`; the full unit suite
passed `364` with one established optimizer warning; Ruff passed on the two
new Python paths; and `git diff --check` passed. The tested repository base and
required S11-A ancestry are
`ea335d57635ed8b38051169b8f0e770b3fe46459`. The planned result paths were
absent before and after deterministic repeated validator smoke checks.

S11-B1 created no executor, output directory, placeholder, candidate, or result
artifact. It ran no pilot or production model/router/teacher/packed-artifact
inference, training, evaluation, generation, decode, perplexity, CUDA workload,
benchmark, or performance measurement. Required CPU unit fixtures were
structural regression evidence only. Quality transfer remains unknown;
structural tests are not quality evidence, and two future requests cannot
establish general quality.

Next action: begin S11-B2 implementation of the fail-closed executor and
non-executing plan from frozen B1, while explicitly not running the real
Qwen3-4B pilot in B2.

## S11-A one-unit-lookahead attention routing semantics

Historical status: COMPLETE — semantics only.

S11-A adds explicit request-owned `same_unit` (unchanged default) and
`lookahead_attention_one_unit` timing. In the lookahead mode, layer 0 attention
remains same-layer; source layers 0–34 compute target attention decisions 1–35
from the detached masked mean after source attention residual completion and
post-attention normalization but before source FFN. Target ownership,
provenance, one-time consumption before packed target attention, same-layer
FFN routing, hard first-maximum behavior, soft autograd, and fixed-route decode
reuse are validated. There is no target beyond layer 35.

Evidence: focused S11/S05/S06 checks passed `15`; the required S05/S06/decode/
isolation selection plus the new 36-layer tiny real pinned packed integration
passed `14`; the full unit suite passed `314` with one established optimizer
warning; Ruff passed on every changed Python path; and `git diff --check`
passed. The real tiny integration executed 252 physically packed projections,
observed 72 target-owned decisions, exact source-to-target event order, one
call per target router, same-layer FFNs, and bitwise deterministic repeats.
The soft test observed finite target-router gradients and a real optimizer
update, no source-router gradient from the detached target feature, and
unchanged frozen packed/base parameters.

No Qwen3-4B model was loaded and no quality pilot, real training/evaluation,
S10 rerun, production checkpoint/lambda selection, asynchronous transfer,
prefetch, caching, scheduling, performance measurement, or historical evidence
change occurred. Quality parity and any execution benefit remain unknown.

Historical next action at the S11-A gate: define a separately scoped paired quality pilot comparing same-unit routing with one-unit-lookahead attention routing.

## S10-H2-B2 repaired canonical broader-validation retry

Operational attempt 2 consumed the one authorized repaired B2 retry exactly
once from `b1aca71bcc584f0e3559e5fe7caf142c2f750db3`. The command exited `0` and
the final closeout independently validated the complete nine-trial evidence as
`REFINE` with no errors. Canonical result:
`docs/results/s10h_broader_validation.json`, SHA-256
`7d9e0aff3b686570be0d1d57b5513ee921d60bd5470f275b0cd7cbb4fd63db20`.

All nine ordered seed/lambda trials completed 24 updates, exact router-only
fresh-AdamW and teacher/base-freeze audits, twelve 72-unit route maps, finite
nonzero gradients, unchanged-state reproducibility, inherited regressions, and
prohibited-work checks. A separate full audit passed 3,306 checks with zero
errors. Lambda `0.03` was on all three per-seed hard frontiers and reduced the
paired median hard width by `-0.4907407407407405`, with zero reproducibility
failures, but its paired median hard-KL delta was
`0.014972516723598044`, failing the frozen `<= 0.0` condition. This is a valid
**REFINE**, not REVISE or PAUSE, and it selects no production lambda.

The two pre-execution PAUSE branches, the failed pre-workspace Herdr lab, the
post-execution expected-candidate clean-status PAUSE, and the first closeout
pane's wrong commit-path assertion PAUSE remain preserved. None reran the
experiment. The final captain-authorized closeout ran no test, plan, model,
training, evaluation, profiler, monitor, or execution command; it validated,
audited, promoted by same-directory no-overwrite hard link, documented, and
closed the existing evidence only.

Next action: define a later, separately frozen refinement protocol. Do not
select a production lambda, run refinement, train a production router, or begin
later loss work from this closeout.

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
python scripts/validate_baseline_evaluation_protocol.py --config configs/baseline_evaluation.json
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
PYTHONPATH=src:. pytest -q tests/unit/test_baseline_evaluation_protocol.py tests/integration/test_baseline_evaluation_protocol_inputs.py tests/integration/test_perplexity_evaluator.py
```

Ruff passed for the validator and focused S09-A test files. At S09-A closeout, no S09-B benchmark, five-mode baseline evaluation, or
final result artifact existed. At that point S09-B execution machinery was
**MISSING**; the committed S09 script was only the non-benchmark protocol
validator.

## S09-B1 runner implementation — CONTINUE

S09-B1 adds `scripts/run_baseline_evaluation.py` and `qaq.evaluation.runner`. The parent resolves the
five frozen mode IDs and launches one explicit `--execute-mode` child per mode,
so no process can retain models for a second mode. The default path is the
non-executing plan, which invokes the canonical validator, prints child and
aggregation commands, and writes no result.

The runner consumes `configs/baseline_evaluation_prompts.json`, passes S09's explicit
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
- `python scripts/run_baseline_evaluation.py --plan --config configs/baseline_evaluation.json`
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
the corrected and verified S09-B runner and configs/baseline_evaluation.json, then
evaluate the frozen release gates.

## S09-A protocol owner

The authoritative machine-readable protocol is
`configs/baseline_evaluation.json`; its fixed inputs are
`configs/baseline_evaluation_prompts.json`. The detailed human-readable procedure and
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
  `configs/baseline_router_training.json`: four deterministic Wikitext training
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
PYTHONPATH=src:third_party/any-precision-llm python scripts/verify_router_checkpoint_roundtrip.py --device cuda:3 --result docs/results/s07_router_training.json
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
`qaq.loading.loader` uses pinned `dequant_kbit` plus `torch.matmul` only for that
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

Historical next action: Begin S10-D. The completed S10-D gate is recorded below.

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
Protocol/config: `configs/router_cost_calibration.json`.
Stage procedure and limitations: `docs/stages/S10_6BIT_ROUTING.md`.
No historical result, production checkpoint/lambda, S08 loader, packed
artifact, Any-Precision source, or S07 runner was changed.

Review repair: the runner now rejects any config bytes other than the locked
protocol, consumes configured KD/entropy/adaptive values, requires the exact
pinned Hugging Face snapshot path, and rejects missing router gradients.
Focused repair verification passed `11` tests in
`tests/unit/test_router_cost_calibration.py`.

Next action: firstmate/captain reviews the observed frontier and decides
whether to refine, confirm, or begin full training.

## S10-E — Frontier Confirmation Protocol Freeze

Current stage: S10-E
Status: COMPLETE
Gate outcome: CONTINUE.

Passing commit: `7a3548973cbe784657a41c0c6192c155909027c5`.
The frozen protocol is `configs/router_frontier_confirmation.json`; focused
protocol tests are in `tests/unit/test_router_frontier_protocol.py`. The
protocol records the merged S10-D/PR #9 starting point, exact candidate bits
`[4,6,8]`, lambdas `[0.0,0.03,0.1]`, captain-selected seeds
`[1729,1730,1731]`, nine paired trials, inherited S10-D/S07 data and training
values, router and objective invariants, exact future measurements, and
frozen CONTINUE/REFINE/PAUSE rules. Seeds and the three-candidate confirmation
are implementation choices, not source-paper facts.

The focused S10-E test passed `35`; the S10-D/S10-C/S10-B/S07/request-state
unit regression selection passed `49`; Ruff passed for the focused test; and
`git diff --check` passed. Hash comparison against the required starting
commit confirmed unchanged S10-D canonical config/result/runner/test files
and unchanged established S07/S08/router/objective/loading surfaces. No
S10-E trial, model inference, CUDA execution, router training, S10-D runner,
full training, adaptive extension, production lambda selection, or S10-F work
was performed. No `scripts/run_s10e.py` exists.

Next action: Begin S10-F: execute the frozen three-seed frontier confirmation protocol.

## S10-F — Frozen three-seed frontier confirmation

Status: REVISE. The exact nine ordered pairs completed on one explicit
`cuda:0` NVIDIA GeForce RTX 3090 from merged implementation base
`7fc136eabdba302e199354ae001cd1e1cd42199f`. The frozen S10-E config remained
byte-identical (`fe5ff8826f17605ca8b2dc7d83555e858d3d9f5fa67d14b49bb09b7cbf66a879`).
Pinned model/tokenizer, Wikitext, packed-artifact, and Any-Precision identities
were verified; the packed artifact and backend were consumed through explicit
read-only overrides. No S10-D static references, historical S07 checkpoint,
S08 loader, adaptive lambda, production selection, or prohibited
serving/resource measurement occurred.

The canonical result is `docs/results/s10f_frontier_confirmation.json` with
SHA-256 `d68f041e0a3dc32c465e8b8068ca3ab230d39253757e30f3019ca7e681b14233`.
It records all nine trials, paired fresh initializations and AdamW audits,
four steps per trial, finite/freeze/base audits, both 72-entry validation
route maps, route variation, collapse labels, soft/hard metrics, and one
immediate same-state hard-validation repeat per trial. Observed aggregates:
`0.03` frontier membership `2/3`, paired hard KD delta median
`-0.004020056687295437`, paired hard width delta median
`-0.16666666666666696`, and reproducibility failures `0`.

The runner falsely serialized `router_only_optimizer_audit` and
`fresh_adamw_audit` as `false` for all nine trials by comparing the inherited
Python tuple `("routers.",)` only to a list. Raw audit records show fresh state
and the `routers.` prefix, but this post-trial defect can affect gate validity.
The artifact's generated classification is `REFINE`; the worker classification
is `REVISE`. All nine records are preserved, no repair or rerun was performed,
and the next action is for firstmate to resolve the runner defect and evidence
policy before any broader validation.

### S10-F audit repair — PAUSE / RERUN_REQUIRED

The captain-authorized repair began with the mandated preflight: Python was
`/nfs/home/s314511048/.venv/bin/python` at version `3.12.3`, and `nvidia-smi`
reported eight idle NVIDIA GeForce RTX 3090 GPUs. The original packed artifact
hash matched `29d9bc526b3da0bd39daf2f82afd141f82d005ca1232cabc75cfe9d9ecc1cfee`.
The preserved S10-F result remained unchanged at its original SHA-256
`d68f041e0a3dc32c465e8b8068ca3ab230d39253757e30f3019ca7e681b14233`.

The repaired expression accepts the inherited tuple `("routers.",)` and the
JSON list `["routers."]` as the same one-prefix audit, while retaining exact
identity-based rejection of missing router parameters, extra non-router
parameters, and duplicate tensors. Fresh AdamW coverage requires empty state
before the first step and rejects a reused optimizer with state. The original
REVISE outcome and all measured frontier values remain measured-original; the
result JSON was not rewritten with repair-corrected fields.

Historical evidence is insufficient for Branch A. The result preserves only
per-trial prefix/count summaries and a fresh-state boolean, not the actual
included parameter identities/names, group membership, duplicate audit, or an
independent optimizer-state snapshot. Reclassifying either historical audit
would infer runtime proof from source/tests, so the resulting primary outcome
is `PAUSE / RERUN_REQUIRED`. No canonical training or evaluation rerun, extra
trial, broader validation, production-lambda selection, or success commit was
performed.

Repair verification: the focused repair subset passed `4`; the S10-E/S10-F
focused suite passed `65`; the inherited S10-F regression selection passed
`46`; Ruff passed on the two changed Python files; and `git diff --check`
passed.

### S10-F canonical rerun — COMPLETE / CONTINUE

Attempt 2 completed exactly the nine fresh ordered trials on the same explicit
`cuda:0` RTX 3090 under the repaired optimizer audits. Attempt 1 remains at
`docs/results/s10f_frontier_confirmation.json` with its original SHA-256
`d68f041e0a3dc32c465e8b8068ca3ab230d39253757e30f3019ca7e681b14233`; it was
not overwritten or used as attempt-2 evidence. The new artifact is
`docs/results/s10f_frontier_confirmation_rerun.json` with SHA-256
`b3bcc0e45d45852ac5060209c4789453ed452462f528f7bffd4cb80fb1ef58cb`.

All runtime audits passed for every trial, including identity-based
router-only membership, zero missing/extra/duplicate parameters, fresh AdamW
construction serials with zero state before training, finite loss/gradients,
teacher/base freeze, exact four-step budgets, and reproducibility repeats.
The frozen aggregates are `0.03` frontier membership `2/3`, paired hard KD
median delta `-0.004020056687295437`, paired hard selected-width median delta
`-0.16666666666666696`, and zero reproducibility failures. Focused tests passed
`65`, inherited regressions passed `46`, Ruff passed, and `git diff --check`
passed. The S10-F gate outcome is **CONTINUE**; the next action is a
separately scoped broader-validation decision, not execution here.

## S10-G — Broader-validation protocol definition and freeze

Status: CONTINUE (protocol freeze only; no S10-G experiment result exists).

The authoritative machine-readable protocol is
`configs/broader_router_validation.json`; focused tests are in
`tests/unit/test_broader_router_validation_protocol.py`. S10-A through S10-F are
established complete. S10-F attempt 1 remains preserved and attempt 2 is
present and classified CONTINUE; no production lambda was selected. Attempt 2
authorized only this separately scoped broader-validation decision, and no
broader validation has run.

The detailed frozen data, training, audit, route-map, reproducibility, gate,
and prohibition contract is owned by the machine-readable config and the
current stage document; D049 records the implementation assumptions.

S10-G itself created no runner, result JSON, or execution path and performed no
training, evaluation, GPU evaluation, or hardware/resource measurement. The
focused S10-G test passed `53`; S10-D/S10-E/S10-F predecessor regressions
passed `121`; Ruff and `git diff --check` passed.

Next action: obtain a separately authorized decision before any broader
validation execution; do not select a production lambda.

## S10-H1 — Protocol-locked broader-validation runner (historical)

Historical gate: CONTINUE for the H1 implementation and review repair. H1
added the fail-closed validator and non-executing plan in `scripts/run_broader_router_validation.py`,
including frozen S10-G/provenance checks, future-result validation, and
canonical-result overwrite refusal. Its detailed contract, provisioning
procedure, and recorded evidence remain in
[`docs/stages/S10_6BIT_ROUTING.md`](stages/S10_6BIT_ROUTING.md) and decisions
D050–D052. H1 performed no model, real-data, CUDA, training, result, resource,
or production-lambda work.

## S10-H2-A — Real executor seam

Status: COMPLETE (implementation only; no H2 experiment).

H2-A adds the lazy `qaq.router.s10h_executor` runtime boundary and shared
nine-trial/24-update scheduler. `--execute` requires an explicit device and a
temporary noncanonical destination; the canonical H2 result path is refused.
The injected deterministic runtime exercised the unchanged validator and
output-safety path without making a Qwen or quality claim. Focused S10-H tests
passed 36, the S10-G/S10-F/S10-E/S10-D regression selection passed 134, the
full unit suite passed 299 with one existing optimizer warning, Ruff passed on
changed Python files, and `git diff --check` passed.

The canonical H2 path was refused and
`docs/results/s10h_broader_validation.json` remains absent. No real-Qwen trial,
canonical experiment, production-lambda selection, or resource measurement was
performed. The next action is to stop at the H2-A decision gate and separately
authorize H2-B; do not infer broader-validation quality from the injected smoke
or the H2-A implementation.

## S10-H2-B attempt 1 and S10-H2-BR1 production contract repair

The current stage is recorded at the top of this file. The complete attempt and
repair evidence is owned by
[`docs/EXPERIMENTS.md`](EXPERIMENTS.md); the production contract and gate
boundary are owned by
[`docs/stages/S10_6BIT_ROUTING.md`](stages/S10_6BIT_ROUTING.md).

S10-H quality remains unknown. The next action is separate authorization for
S10-H2-B2 from the repaired, reviewed, merged commit; do not begin it from this
repair task.
