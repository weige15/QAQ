# Freeze the paired lookahead-specific 4/6/8 training protocol

_Legacy work-item reference: S11-D1._

Legacy identifiers elsewhere in this record are retained only for historical cross-reference to frozen decisions, evidence, paths, and machine-facing contracts.

## Exact research question

**Can a 4/6/8 router trained for one-layer-lookahead attention routing achieve
meaningfully lower selected precision while preserving acceptable
teacher-relative quality?**

This question does not assume that lookahead timing reduces selected precision.
S11-C3 showed the opposite direction for the historical 4/8 checkpoint:
lookahead increased mean selected width by `0.06481481481481488` bits overall
while passing the frozen teacher-relative quality margins.

## Goal

Freeze, before any new result is observed, the smallest fair paired training
experiment that can separate routing-timing effects from cost-pressure effects.
The paired lookahead-specific 4/6/8 training protocol freeze (legacy work item S11-D1) defines the arms, fixed coefficient set, paired initialization and
training contract, evaluation metrics, separate quality and selected-precision
gates, and outcome rules. It creates no executor, checkpoint, or result and
runs no training, model evaluation, CUDA workload, or performance measurement.

The exact Goal for the next `/goal-driven` is:

> Implement and structurally validate an S11-D2 executor and deterministic
> non-executing plan that consumes this S11-D1 contract without changing its
> arms, coefficient set, pairing, data, seeds, budget, metrics, thresholds, or
> classifications. The deterministic paired-training plan and dispatcher (legacy work item S11-D2) must not execute real training or evaluation; real
> Real paired-training execution (legacy work item S11-D3) requires separate authorization.

## Paired control and treatment

The experiment has exactly two arms:

1. `same_unit_468_control`, with `routing_timing="same_unit"` during both soft
   router training and final soft/hard evaluation;
2. `lookahead_attention_one_unit_468_treatment`, with
   `routing_timing="lookahead_attention_one_unit"` during both soft router
   training and final soft/hard evaluation.

Both arms use explicit candidates `[4,6,8]` and probability order
`[p4,p6,p8]`. In treatment, attention layer 0 remains same-unit; source
attention layers 0–34 predict target attention layers 1–35 at
`post_attention_pre_ffn`; all FFN routes remain same-unit. Target ownership,
feature detachment, one-time probability consumption, and provenance are the
unchanged S11-A semantics. The current implementation supports the required
soft gradient path: `src/qaq/model/manual.py` stores the target router's
lookahead probability, and
`tests/integration/test_soft_routing.py::test_soft_lookahead_updates_only_the_target_router_and_keeps_packed_base_frozen`
proves finite target-router gradients and a frozen packed base.

Each arm contains exactly two predeclared cost conditions, in order:
`lambda_bit=[0.0,0.03]`. The `0.03` cells are the primary control and treatment;
the corresponding `0.0` cells are within-timing quality and selected-width
references, not extra candidate searches. Thus the complete experiment is the
small `2 timings × 2 lambdas × 3 seeds = 12 trials` factorial, not an adaptive
sweep.

### Cost-coefficient rationale

The canonical broader-validation result (legacy work item S10-H) is complete evidence, not a production-lambda selection. Its
same-unit `lambda=0.03` trials were on the hard frontier in all three seeds and
reduced paired median hard width by `0.4907407407407405` bits versus
`lambda=0.0`, but increased paired median hard KL by
`0.014972516723598044` and therefore returned `REFINE`. `lambda=0.1` reduced
width further but had much worse median hard KL (`0.07732601106787722` versus
`0.01439695991575718` at zero cost). Repeating the full S10 grid or including
`0.1` would retest a known unfavorable tradeoff without a new justification.
Using only `0.03` would omit the zero-cost reference needed to decide whether
lookahead training preserves quality while reducing width.

Accordingly, `0.0` and `0.03` are the smallest evidence-backed set that
separates timing from cost pressure. `0.03` is an experimental probe, not a
production-ready value. No value may be added, removed, interpolated, selected
adaptively, or retuned after results.

## Training and evaluation protocol to freeze

### Identities and data

All twelve trials use the pinned Qwen3-4B model/tokenizer revision
`1cfa9a7208912126459214e8b04321603b3df60c`, physically packed artifact
SHA-256
`29d9bc526b3da0bd39daf2f82afd141f82d005ca1232cabc75cfe9d9ecc1cfee`,
Any-Precision commit `a3257d02740cc5757c78673da534b0630ff3a4ea`, and
Wikitext revision `b08601e04326c79dfdd32d625aee71d232d685c3`.
The full-precision teacher and packed non-router student base are frozen; only
all 72 three-way routers are trainable and optimizer-owned. The historical
S07 two-way checkpoint is not loaded or converted.

Reuse exactly the canonical S10-H manifests in their recorded order. Training
uses these 24 IDs:

`train-3`, `train-1003`, `train-2002`, `train-3037`, `train-4005`,
`train-5002`, `train-6001`, `train-7001`, `train-8000`, `train-9068`,
`train-10001`, `train-11003`, `train-12003`, `train-13000`, `train-14000`,
`train-15000`, `train-16002`, `train-17002`, `train-18000`, `train-19001`,
`train-20006`, `train-21000`, `train-22000`, `train-23000`.

Evaluation uses these 12 IDs:

`validation-3`, `validation-270`, `validation-500`, `validation-761`,
`validation-1000`, `validation-1252`, `validation-1500`, `validation-1759`,
`validation-2000`, `validation-2250`, `validation-2500`,
`validation-2755`.

Every source row, offset, text digest, 64-token input digest, and order must be
byte-for-value equal to the manifests in canonical
`docs/results/s10h_broader_validation.json` (SHA-256
`7d9e0aff3b686570be0d1d57b5513ee921d60bd5470f275b0cd7cbb4fd63db20`).
Tokenization uses `add_special_tokens=false`, first 64 tokens, prompt `[0,32)`,
completion `[32,64)`, and causal completion logits `[31,63)`. No row may be
resampled or replaced.

### Pairing and optimization

Use seeds `[1729,1730,1731]`, in that order. For each seed, create one fresh
canonical three-way router initialization and clone it byte-identically into
all four timing/lambda cells before any update. The expected initial router
hashes are the canonical S10-H hashes:

- `1729`: `7b5b5bd2a1ed89b98c0c1358e6a38f5579d0919d0ffc980e06aa7ad09a434123`;
- `1730`: `cca1b7cf3c06679fa4b2178ee2e8dfa4100a07738d0f1d4c9e928b4a08c0d55a`;
- `1731`: `c96ce0f8da7541ecb13594458772d8a254bcb8c378d52096554dd53257b8baf1`.

Within each seed, execute in exact order: zero-cost same-unit, zero-cost
lookahead, `0.03` same-unit, `0.03` lookahead. Every cell gets a fresh AdamW
with identity-audited router-only membership and empty state before its first
step. Warm starts and optimizer reuse are forbidden.

Each cell receives the same 24 examples once, in frozen order: batch size 1,
gradient accumulation 1, one epoch, exactly 24 optimizer steps, learning rate
`0.001`, weight decay `0.0`, betas `[0.9,0.999]`, epsilon `1e-8`,
`amsgrad=false`, no scheduler, routing temperature `1.0`, and distillation
temperature `2.0`. The loss is exactly completion-only
`L_total = L_KD + lambda_bit * L_bit`, where the normalized candidate costs
are `[0.0,0.5,1.0]` and `L_bit` is the unweighted mean across all 36 attention
and 36 FFN decisions. There is no entropy term, width-specific attention
weight, checkpoint selection, early stopping, extra epoch, or retry chosen
from observed metrics.

The final state after step 24 is the only evaluated state. Record initial and
final router hashes, all updates, finite losses/gradients, target-router
coverage and provenance, fresh optimizer evidence, and before/after hashes and
gradient absence for teacher and packed base. No trial creates or selects a
production checkpoint.

### Evaluation and reproducibility

Evaluate every final state on all twelve fixed validation requests in frozen
order using resident physically packed soft execution and deterministic hard
argmax execution. Use batch size one, teacher-forced 64-token forwards,
`use_cache=false`, and no generation or decode. For every request and trial,
record:

- completion-only temperature-2 masked teacher-relative KL;
- full-logit mean and maximum absolute teacher error;
- soft expected width and hard mean selected width;
- hard 4/6/8 counts and fractions;
- complete 72-unit layer-major hard route map, attention then FFN per layer;
- attention-only, FFN-only, and overall selected-width summaries; and
- treatment/control and positive/zero-cost paired route transitions.

Aggregate KL, mean absolute error, and selected width as unweighted arithmetic
means of the twelve request values. Maximum absolute error remains diagnostic.
Repeat each final hard evaluation immediately once at unchanged state; logits,
routes, metrics, input/teacher digests, and provenance must be identical.
Report per-seed values and the median of the three paired seed deltas. Do not
average the seeds into a synthetic trial or combine quality and precision.

## Measurable quality criteria

The primary quality reference is the treatment arm's same-seed, same-timing
`lambda=0.0` cell. This is preferable to treating the known quality-failing
same-unit `lambda=0.03` cell as acceptable. The contemporaneous same-unit
`lambda=0.03` arm remains necessary to expose whether timing changes the
tradeoff under identical cost pressure; canonical S10-H values are historical
comparators, not substitutes for current paired cells.

Quality passes only if all of these conditions hold:

1. The median across seeds of
   `lookahead_lambda_0.03 aggregate hard KL - lookahead_lambda_0.0 aggregate hard KL`
   is `<= 0.0`, reusing the exact S10 broader-quality rule that `0.03` failed.
2. For every seed and every one of the twelve requests, lookahead `0.03` hard
   KL is `<= 1.25 ×` its paired lookahead zero-cost hard KL.
3. For every seed, lookahead `0.03` aggregate hard mean absolute logit error is
   `<= 1.10 ×` its paired lookahead zero-cost aggregate.
4. Against the same-seed same-unit `0.03` control, every lookahead `0.03`
   aggregate hard KL and aggregate hard mean absolute error is `<= 1.10 ×`
   control and every paired request hard KL is `<= 1.25 ×` control. These are
   the established S11 paired timing safeguards, not universal guarantees.

All values must be finite. Maximum absolute error and soft metrics are required
diagnostics but are not thresholded. No selected-width result may compensate
for a quality failure.

## Measurable selected-precision criteria

The primary selected-precision reference is the treatment arm's paired
same-seed `lambda=0.0` cell, because it isolates the effect of the predeclared
cost pressure within the exact timing being tested. The same-unit `0.03` arm
is a fixed-cost timing comparator, not the primary savings baseline.

Selected precision passes only if:

1. The median across seeds of
   `lookahead_lambda_0.03 hard mean selected width - lookahead_lambda_0.0 hard mean selected width`
   is `<= -0.4907407407407405` bits overall.
2. At least two of three individual seed-level deltas are strictly negative.
3. All route maps have exactly 864 decisions per trial, valid 4/6/8 values,
   and exact attention/FFN coverage; no soft expected-width value substitutes
   for hard selected precision.

The `0.4907407407407405`-bit threshold is the canonical S10-H median reduction
at `lambda=0.03`. Matching or exceeding an already observed cost-aware effect
is the smallest repository-grounded definition of meaningful here; it is not
a latency, transfer, memory, throughput, or execution threshold. Report the
paired same-unit `0.03` versus lookahead `0.03` width delta separately, but do
not let that diagnostic replace the zero-cost savings gate.

## Outcome conditions

Apply outcomes in this order: `PAUSE`, `REVISE`, `CONTINUE`, `REFINE`, `STOP`.
All requests, arms, lambdas, seeds, identities, initial hashes, update counts,
metrics, thresholds, aggregation rules, and outcome boundaries above are frozen
before execution.

### PAUSE

Use **PAUSE** when a required model, tokenizer, packed artifact, dataset
fixture, exact initial state, comparable CUDA resource, or required evidence is
unavailable or incomplete; an interruption leaves a trial incomplete; or a
material scientific ambiguity prevents interpretation. PAUSE makes no result
claim and permits no substitution.

### REVISE

Use **REVISE** when evidence is complete enough to identify invalid protocol,
identity, pairing, initialization, ordering, optimizer, update-count, freeze,
gradient, route/provenance, repeat, persistence, regression, or prohibited-work
evidence. Correct only the bounded implementation defect and invalidate its
affected evidence. Changing a scientific threshold or arm requires a new
separately authorized pre-result work item.

### CONTINUE

Use **CONTINUE** only when all evidence and audits are complete and valid and
every measurable quality and selected-precision criterion passes. CONTINUE
answers the exact research question positively for this fixed experiment only;
it does not select `0.03` for production or establish any execution benefit.

### REFINE

Use **REFINE** only for complete, valid evidence in one of these two
predeclared near-miss regions, with every other scientific criterion passing:

- precision passes, all factor safeguards pass, and the lookahead paired median
  hard-KL delta is positive but strictly less than S10-H's failed
  `0.014972516723598044` delta; or
- quality passes, all three seed width deltas are negative, and the median
  width reduction is at least `0.24537037037037025` bits but less than
  `0.4907407407407405` bits.

REFINE may justify one later protocol proposal aimed only at the missed axis.
It does not permit changing these thresholds after seeing the result, adding
an adaptive lambda in the current evidence, or rerunning selected cells.

### STOP

Use **STOP** for complete valid evidence that is neither CONTINUE nor one of
the exact REFINE regions. This includes no selected-width reduction, a width
reduction below `0.24537037037037025` bits, quality degradation at least as
large as S10-H's failed median hard-KL delta, failure of any `1.10`/`1.25`
safeguard, or failures on both axes. STOP does not prove all lookahead training
impossible; it ends this bounded formulation without another automatic trial.

## Explicit exclusions

The paired-training protocol freeze (legacy work item S11-D1) excludes:

- real training, evaluation, model/dataset loading, CUDA work, executor code,
  checkpoint creation, result files, or output directories;
- any change to canonical S10-H or S11-C3 evidence or their historical
  interpretation;
- adaptive coefficient search, `lambda=0.1`, extra seeds/examples/epochs,
  warm starts, early stopping, checkpoint selection, entropy terms, combined
  quality/precision scores, or post-result threshold changes;
- training the teacher or packed base, loading/converting the S07 two-way
  checkpoint, changing candidates/order/costs, or non-router optimizer state;
- prefetch, asynchronous loading or transfer, overlap, prediction, caching,
  batching, scheduling, or changes to the synchronous on-demand loader;
- latency, transfer, memory, throughput, energy, kernel, profiler, or other
  execution/resource benchmarking; and
- any claim that reduced selected bits alone improve execution.

## Evidence required before real 6-bit loading or prefetch work

Only a valid S11-D3 **CONTINUE** result may justify defining the next bounded
real-loading objective. To justify testing an actual 6-bit loader path, that result
must also show at least one hard 6-bit selection in every treatment seed and
complete reproducible target-owned route maps; otherwise the loading question
has no observed treatment route to exercise.

The first follow-up objective would add and verify genuine synchronous on-demand
`qweight[:6] + lut6` loading only: resident parity, packed-only transfer,
physical byte accounting, request lifetime, reuse, cleanup, and deterministic
hard-route execution. Reduced selected width is only a routing observation and
cannot substitute for those checks.

Prefetch or asynchronous loading and any execution-benefit claim require a
further separately frozen comparison after real 6-bit synchronous correctness.
That later protocol must measure actual comparable latency, transfer, memory,
and/or throughput against the synchronous reference. S11-D success authorizes
asking and testing those later questions; it does not answer them.
