# S11-C1 — Broader lookahead quality protocol freeze

## Exact research question

**Does the historical S07 4/8 checkpoint continue to preserve acceptable
teacher-relative quality under one-layer-lookahead attention routing on a
meaningfully broader fixed evaluation set than the two-request S11-B pilot?**

## Goal

Freeze the smallest defensible paired broader-quality protocol before any new
result is observed. S11-C1 defines the evaluation set, control and treatment,
metrics, quality margins, repeats, route diagnostics, and classification rules.
It creates no executor and runs no model, dataset selection, CUDA workload, or
experiment.

The next `/goal-driven` implementation goal, after this protocol-only stage is
accepted, is a separately bounded S11-C2 executor and non-executing plan that
must consume this contract without changing it. Real GPU execution remains a
later, separately authorized stage.

## Authoritative established facts

The following are repository-established rather than choices made by S11-C1:

- S11-B3 completed with `ADVANCE_TO_BROADER_QUALITY_CHECK` under frozen config
  SHA-256
  `21a664424debe4892c3577c490158228dd5399bb4b425611db728070d23a5051`.
  Its two-request treatment/control ratios were `0.9824730210816205` for
  aggregate KL, `0.9431116083999646` and `1.0` per request, and
  `0.9929261653206376` for aggregate mean absolute logit error.
- S11-A owns the routing semantics. `same_unit` is the unchanged control.
  Treatment attention layer 0 remains same-unit; source attention layers 0–34
  predict target attention layers 1–35 at `post_attention_pre_ffn`; FFNs remain
  same-layer and target-owned.
- S07 owns the historical router-only 4/8 checkpoint, candidate order `[4,8]`,
  hard argmax behavior, and completion-only temperature-2 masked KL operation.
- S09 owns the pinned Qwen3-4B model/tokenizer, packed artifact,
  Any-Precision, historical checkpoint, and fixed-input identities.
- S10-G/H broader validation already froze and executed a deterministic
  twelve-request validation manifest. Its canonical result is
  `docs/results/s10h_broader_validation.json`, SHA-256
  `7d9e0aff3b686570be0d1d57b5513ee921d60bd5470f275b0cd7cbb4fd63db20`.
  S11-C1 reuses only that fixed validation fixture precedent; it does not reuse
  S10's three-way router, training, lambda, or optimization result.

S11-B3 changed one of 144 target decisions: `validation-3` attention layer 23
selected 8 bits under treatment instead of 4. The treatment therefore did not
reduce average selected precision in that two-request pilot. One 4-to-8 change
is too small to prove or disprove that lookahead can save precision on a
broader workload or after separately paired training, and it is not a quality
gate here.

## Frozen evaluation set

### Choice and rationale

Use exactly the twelve validation requests already frozen by the S10 broader
validation precedent. Twelve is six times the S11-B pilot count, retains both
pilot requests, spans twelve fixed source locations instead of adding generated
or hand-picked prompts, and is the smallest existing repository-backed broader
fixture. No post-result request replacement, resampling, extension, or subset
classification is allowed.

The source is `Salesforce/wikitext`, configuration `wikitext-2-raw-v1`,
validation split, revision
`b08601e04326c79dfdd32d625aee71d232d685c3`, tokenized with
`Qwen/Qwen3-4B` revision
`1cfa9a7208912126459214e8b04321603b3df60c`, `add_special_tokens=false`.
For each frozen source row, use exactly its first 64 token IDs: prompt `[0,32)`,
completion `[32,64)`, and causal completion loss logits `[31,63)`. Batch size is
one and there is no padding. The IDs and digests below come from the canonical
S10-H validation manifest; later fixture materialization must reproduce every
digest before execution and must persist the exact token arrays so execution
performs no selection or replacement.

| Order | Request | Frozen offset | Frozen source row | 64-token input SHA-256 |
| ---: | --- | ---: | ---: | --- |
| 1 | `validation-3` | 0 | 3 | `bbd7a25c172570f90d29d6fff0efc65975139ab7d65bb22409e87d10094f404b` |
| 2 | `validation-270` | 250 | 270 | `23e957c1cb5713a17c5332c2fd2bcb080c8d752d29cb51a5acb436fc8842f604` |
| 3 | `validation-500` | 500 | 500 | `dfe59fb1e0c1689410f5295037850be536ab56ce60dbbde4c8b6430969004b79` |
| 4 | `validation-761` | 750 | 761 | `d816c84ccd24ae11ca9a8124dd92130699393338847f900aa6cf46c7368c871b` |
| 5 | `validation-1000` | 1000 | 1000 | `99c0183a064c79daea4cb461de16ddeb2144dbbe2af64b375f6f2088bb6e659e` |
| 6 | `validation-1252` | 1250 | 1252 | `84ccbabf826875b899036c663e07080b558d1c0f047268b860e96da8a1bf7d17` |
| 7 | `validation-1500` | 1500 | 1500 | `735f2670539c002602b9e7500a4288e1393f91bd7e8cb8617d3b8f34ba625d5c` |
| 8 | `validation-1759` | 1750 | 1759 | `e6c401f8f0dad17504e55c4a9db5c2436a213786a5824c56f54826b4f1a8febc` |
| 9 | `validation-2000` | 2000 | 2000 | `37f62b98ee3bc4466d0cbf64866d8ae6bc27a0cf723321aa5247cfd93bf703be` |
| 10 | `validation-2250` | 2250 | 2250 | `6f6bf29d4ec5962df94dd58cadb51ea9a8eea484e4aa9f22c069f2aa2ed26378` |
| 11 | `validation-2500` | 2500 | 2500 | `360df7b59d764cda62bad990346db508d7bfbe059a5bb0d2593fcf3a5540d4b8` |
| 12 | `validation-2755` | 2750 | 2755 | `4dad3d315098a80c4d31a6198a7b120ff7cec5af66e481609a69b3adf0b659a4` |

Token digests use the S11-B encoding: SHA-256 over source-order little-endian
signed 64-bit token IDs. Any ID, order, source row, range, token count, or
digest mismatch is evidence drift, not permission to choose a replacement.

## Frozen control, treatment, and held constants

Execute and aggregate in this exact order:

1. `same_unit_control` with routing timing `same_unit`;
2. `lookahead_attention_one_unit_treatment` with routing timing
   `lookahead_attention_one_unit`;
3. paired aggregation.

Both modes use the historical S07 router checkpoint read-only, SHA-256
`08bf646f19759c0d7949e159bdbe4f96bbea737204b96f8760d205c8d6fd1949`,
with candidates `[4,8]`; the pinned model/tokenizer revision; packed artifact
SHA-256
`29d9bc526b3da0bd39daf2f82afd141f82d005ca1232cabc75cfe9d9ecc1cfee`;
and Any-Precision commit
`a3257d02740cc5757c78673da534b0630ff3a4ea`. Both are resident physically
packed hard query-level execution. There is no training, retraining, optimizer,
gradient, checkpoint output, or 6-bit candidate.

Each mode runs in one fresh process on the same explicit physical GPU with
identical hardware/software identity, request order, seed `1729`, batch size
one, full teacher-forced 64-token forwards, `use_cache=false`, and no sampling,
generation, decode, perplexity, or on-demand loading. Apart from mode ID and
routing timing, all inputs, identities, settings, teacher outputs, persistence
rules, and audit requirements are equal. For the two overlapping pilot
requests, the control routes and teacher-relative metrics must also equal the
canonical S11-B3 control evidence; the ten added requests have no invented
historical route target.

## Measurable Criteria for Success

A future result satisfies the protocol only when all of the following are
computable and pass:

1. All twelve requests execute in frozen order for both modes with exact input,
   model, artifact, checkpoint, backend, device, and software identities.
2. For every request and mode, record finite:
   - completion-only
     `qaq.router.distillation.masked_kl_distillation_loss` at temperature `2.0`
     over causal logit positions `[31,63)`;
   - mean absolute student-versus-teacher error over all 64 positions and the
     complete vocabulary;
   - maximum absolute error over that same tensor, as a diagnostic rather than
     a threshold.
3. Aggregate KL and aggregate mean absolute error are the unweighted arithmetic
   means of the twelve per-request values in frozen order. No selected-width or
   route-distance term may be combined with quality.
4. Every quality margin passes, reusing the pre-result S11-B margins without
   tightening or loosening them:
   - treatment aggregate KL `<= 1.10 *` control aggregate KL;
   - each of the twelve treatment request KL values `<= 1.25 *` its paired
     control KL;
   - treatment aggregate mean absolute logit error `<= 1.10 *` control
     aggregate mean absolute logit error.
5. Each mode performs exactly two immediate deterministic repeats in its fresh
   process. Both repeats must have identical input and teacher-logit digests,
   bitwise-equal student logits, identical complete route maps and provenance,
   finite metrics, and unchanged parameter/buffer hashes. Seed variation is
   not added because this is deterministic checkpoint inference, not training;
   seed `1729` is fixed for both modes.
6. Every request has exactly 72 unique target-owned routes, serialized
   layer-major with attention then FFN. Selected bits are only 4 or 8. Layer-0
   attention and FFN are equal across modes, and treatment source/target
   provenance satisfies S11-A.
7. Before/after hashes for the teacher, packed weights and buffers, non-router
   base, and router are equal; all are non-trainable, gradients and optimizer
   are absent, cleanup passes, and no prohibited work is observed.

The `1.10` and `1.25` factors, twelve-request fixture, repeat count, and this
gate are project implementation choices, not source-paper facts or universal
quality guarantees.

## Frozen route diagnostics

Routes are descriptive evidence, not a quality threshold. Record, by request
and in aggregate across all 864 target decisions:

- complete control and treatment maps keyed by
  `(request_id,target_layer,unit_type)`;
- overall, attention, and FFN 4/8 counts and fractions;
- per-request and aggregate unweighted mean selected bit width;
- paired overall, attention, and FFN Hamming counts and normalized distances;
- every changed unit with control bit, treatment bit, target layer/unit,
  lookahead source layer/source point, and request identity;
- directional `4 -> 8` and `8 -> 4` transition counts by request and scope;
- treatment-minus-control mean selected-width deltas by request, attention,
  FFN, and overall; and
- distinct route-map counts across requests.

A width increase, equality, or decrease cannot override a quality
classification. In particular, the one S11-B3 `4 -> 8` transition is retained
as prior context only and is neither proof nor disproof of precision-saving
ability.

## Classification rules

Evaluate in this precedence order. Thresholds and request membership may not be
changed after any result is observed.

### PAUSE

Use **PAUSE** when a required external model/tokenizer snapshot, packed
artifact, historical checkpoint, pinned backend, exact fixed fixture, or
comparable CUDA device is unavailable; an execution interruption leaves
required evidence incomplete; or evidence cannot be interpreted without a
material scientific decision. PAUSE makes no quality conclusion and consumes
no permission to substitute a resource, input, or threshold.

### REVISE

Use **REVISE** when the executor, protocol representation, identity, input
order, schema, pairing, determinism, repeat equality, freeze, route coverage,
provenance, cleanup, regression, or prohibited-work audit is invalid. A repeat
mismatch is REVISE because deterministic inference evidence is then invalid,
not a quality sample to average. Correct the bounded defect, invalidate affected
evidence, and revalidate the unchanged frozen scientific contract; any change
to inputs, metrics, margins, or interpretation requires a new pre-result
protocol decision.

### STOP

Use **STOP** only for complete, valid, reproducible evidence that fails one or
more frozen quality margins. Failure of either aggregate `1.10` margin or any
single paired-request `1.25` KL margin is a material broader-quality failure of
historical 4/8 checkpoint reuse under the current lookahead formulation. STOP
does not claim that lookahead is impossible after paired retraining, but it
does not justify proceeding directly to 4/6/8 lookahead training from this
checkpoint-reuse path.

### CONTINUE

Use **CONTINUE** only when the evidence is complete and valid, every integrity
and repeat criterion passes, and all three frozen quality conditions pass.
Selected-width direction and route-change count are reported but are not
additional CONTINUE requirements.

## Explicit exclusions

S11-C1 and its future quality check exclude:

- real experiment or GPU execution during protocol freeze;
- executor, validator, result-schema, or persistence implementation in S11-C1;
- training or retraining, including lookahead-specific training;
- 4/6/8 router construction, optimization, lambda selection, objective change,
  or production-checkpoint selection;
- prefetch, asynchronous loading or transfer, overlap, transfer prediction,
  caching, batching, scheduling, or on-demand-loader changes;
- latency, memory, transfer, throughput, energy, kernel, or other performance
  measurement;
- generation, decode, perplexity, subjective text scoring, and unrelated
  evaluation expansion; and
- post-result input replacement, new seeds, adaptive thresholds, or a combined
  quality/precision score.

## Rule for moving to paired 4/6/8 lookahead training

A future **CONTINUE** classification from this exact broader 4/8 protocol is
sufficient to justify opening the next separately scoped paired
lookahead-specific 4/6/8 router-training stage. That later stage may ask
whether a lookahead-trained 4/6/8 treatment reduces average selected precision
relative to a paired same-unit 4/6/8 control while preserving teacher-relative
quality.

CONTINUE opens that later stage for its own pre-result protocol freeze and
separate execution authorization; it is not evidence that 4/6/8 will save
precision and does not itself authorize training or execution. REVISE, PAUSE,
and STOP are not sufficient to advance.
