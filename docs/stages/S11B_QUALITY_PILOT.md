# Freeze the paired lookahead quality-pilot protocol

_Legacy work-item reference: S11-B1._

Legacy identifiers elsewhere in this record are retained only for historical cross-reference to frozen decisions, evidence, paths, and machine-facing contracts.

## Gate result

**COMPLETE — protocol frozen.** The paired lookahead quality-pilot protocol freeze (legacy work item S11-B1) defines and structurally validates the
paired quality-only comparison of `same_unit` and
`lookahead_attention_one_unit`. It implements no executor and produces no
quality evidence. The frozen config is
[`configs/lookahead_quality_pilot.json`](../../configs/lookahead_quality_pilot.json),
schema `qaq-s11b-quality-pilot-v1`, SHA-256
`21a664424debe4892c3577c490158228dd5399bb4b425611db728070d23a5051`.

The validator loaded no model, tokenizer, checkpoint, packed artifact,
dataset, Torch, Transformers, CUDA runtime, or Any-Precision code. Required
CPU unit regressions imported existing Torch modules and exercised test
fixtures only; they performed no model/backend/artifact/checkpoint integration
or CUDA work. No pilot, production training/inference, generation, decode,
perplexity, benchmark, or performance measurement ran, and no planned result
path or directory was created.

## Goal and evidence boundary

The unresolved question after S11-A is narrow: does the historical S07
same-unit-trained 4/8 checkpoint retain acceptable teacher-relative quality
when only attention routing timing changes to one-unit lookahead? S11-B1
freezes the comparison before observing a result so that inputs, metrics,
routes, margins, and classifications cannot be adapted to outcomes.

This is a two-example transfer pilot, not evidence of general quality,
workload coverage, useful overlap, transfer savings, or latency improvement.
A passing pilot permits only a separately defined broader quality check. It
never authorizes asynchronous transfer or prefetching.

## Established project facts

- S07 established the router-only 4/8 checkpoint, candidate order `[4,8]`,
  completion-only masked teacher/student KL operation, and the two fixed
  validation requests.
- S09 froze the exact Qwen3-4B model/tokenizer revision, packed artifact,
  Any-Precision revision, S07 checkpoint identity, and fixed-input file.
- S11-A is `COMPLETE — semantics only`; `same_unit` remains the default.
  Lookahead attention layer 0 remains same-unit, attention targets 1–35 are
  predicted from sources 0–34 at `post_attention_pre_ffn`, and every FFN
  remains same-layer and target-owned.
- The required S11-A commit
  `ea335d57635ed8b38051169b8f0e770b3fe46459` is an ancestor of this work item.
- No S11-B/lookahead-pilot result existed before the freeze.

The reviewed sources do not establish this exact lookahead timing, pilot size,
checkpoint-transfer criterion, repeat count, or quality margin. Those are
project implementation choices, not paper facts.

## Frozen identities and inputs

Both modes use:

- model and tokenizer `Qwen/Qwen3-4B` at
  `1cfa9a7208912126459214e8b04321603b3df60c`;
- the resident physically packed S03 artifact at
  `quantized/s03b_qwen3_4b/backend_cache/packed/anyprec-(1cfa9a7208912126459214e8b04321603b3df60c)-w8_orig4-gc1-c4_s1_blk64`,
  `pytorch_model.bin` SHA-256
  `29d9bc526b3da0bd39daf2f82afd141f82d005ca1232cabc75cfe9d9ecc1cfee`;
- Any-Precision path `third_party/any-precision-llm` at
  `a3257d02740cc5757c78673da534b0630ff3a4ea`;
- the historical S07 checkpoint, overridable only through
  `QAQ_S07_ROUTER_CHECKPOINT`, SHA-256
  `08bf646f19759c0d7949e159bdbe4f96bbea737204b96f8760d205c8d6fd1949`;
- fixed inputs from `configs/s09_baseline_prompts.json` using
  `full_input_ids` only.

Inputs are exactly, in order:

| Request | Tokens | Token SHA-256 | Prompt | Completion | Causal loss logits |
| --- | ---: | --- | --- | --- | --- |
| `validation-3` | 64 | `bbd7a25c172570f90d29d6fff0efc65975139ab7d65bb22409e87d10094f404b` | `[0,32)` | `[32,64)` | `[31,63)` |
| `validation-1000` | 64 | `99c0183a064c79daea4cb461de16ddeb2144dbbe2af64b375f6f2088bb6e659e` | `[0,32)` | `[32,64)` | `[31,63)` |

Digests are SHA-256 over source-order little-endian signed 64-bit token IDs.
The protocol duplicates no token arrays. Batch size is one; there is no
padding, runtime tokenization, dataset access, input generation, or
replacement.

## Mode and execution contract

The order is exact:

1. `same_unit_control`, `routing_timing: same_unit`;
2. `lookahead_attention_one_unit_treatment`,
   `routing_timing: lookahead_attention_one_unit`.

Apart from mode ID and routing timing, every mode field is equal: resident
physically packed hard 4/8 execution, candidate order `[4,8]`, no training,
no on-demand loader, and no generation, decode, perplexity, or resource work.
Each mode receives one fresh child process on an explicit CUDA device. The two
children must use the same physical GPU identity and identical setup. Each
child performs two deterministic full teacher-forced 64-token forwards with
seed 1729 and `use_cache=false`. S11-B1 validates only this description and
runs none of it.

Checkpoint reuse without retraining is deliberate. There is no optimizer,
learning rate, scheduler, gradient work, checkpoint output, or mutable model
state in the future pilot.

## Metrics

For each request and mode, future evidence records finite teacher and student
logits plus:

- completion-only `masked_kl_distillation_loss` at temperature `2.0`, using
  the existing `T^2 * masked KL(teacher || student)` operation on causal logit
  positions `[31,63)`;
- mean absolute error over every element of the full 64-position logit tensor;
- maximum absolute error over that same full logit tensor.

Each aggregate is the unweighted arithmetic mean of its two per-request
values. There is no width-combined quality scalar, and route distance is not a
quality metric.

## Routes and provenance

Every mode must record a complete target-owned map keyed by
`(request_id,target_layer,unit_type)`, serialized layer-major with attention
then FFN. Each request has exactly 72 unique records: 36 attention and 36 FFN,
with selected bits restricted to 4 or 8. Evidence includes overall,
attention, and FFN 4/8 fractions; unweighted mean selected width; paired
normalized Hamming distances for all three scopes; and every changed
target-unit record.

The same-unit control must equal the historical S07 hard routes by keyed
comparison. Their frozen canonical digests are:

- `validation-3`:
  `0b6c319dc15d41008e3810f991f706d3343826549d47ac88920c6b908e6d6aae`;
- `validation-1000`:
  `8e04d8e2e8e0bb9306dfdd5595176a019992bcd0be5712c9e90625a4d406b101`.

Across modes, only layer-0 attention and layer-0 FFN equality are required.
Later differences are recorded without normalization or rejection. In
particular, later FFN equality is not required even though FFN timing remains
same-layer.

Treatment provenance preserves target identity and request ownership: layer-0
attention is same-unit; target attention `i=1..35` comes from source `i-1` at
`post_attention_pre_ffn`; the target remains `attention_i`; FFNs stay
same-layer; and the five S11-A provenance fields and `[4,8]` ordering remain
intact.

## Repeats and freeze audit

The two repeats per mode must have identical input digests, bitwise-equal
logits, identical 72-unit route maps, identical provenance, and finite logits
and metrics. Future evidence hashes every named parameter and persistent
buffer before and after execution, covering teacher, packed weights/buffers,
non-router base, and router. All complete before/after hashes must match.
There is no optimizer, no gradient, and no teacher/packed/base/router state
change.

## Classification

Integrity takes precedence over quality:

1. `INVALID_EVIDENCE` for any integrity, control, determinism, freeze,
   coverage, provenance, result-schema, or prohibited-work failure;
2. `PAUSE` when a required external model, tokenizer, artifact, checkpoint,
   CUDA device, or identical physical GPU resource is unavailable;
3. `ADVANCE_TO_BROADER_QUALITY_CHECK` only if all preceding checks pass and
   treatment aggregate KL is `<= 1.10 *` control, each treatment request KL is
   `<= 1.25 *` its paired control, and treatment aggregate mean absolute logit
   error is `<= 1.10 *` control;
4. `CHECKPOINT_REUSE_DEGRADES` when complete valid evidence fails any quality
   margin.

The `1.10` and `1.25` factors are implementation choices, never paper facts.
A failure means only that this historical same-unit-trained checkpoint did not
transfer cleanly in this two-example pilot; it does not invalidate lookahead
after separately paired retraining.

## Frozen future outputs

The paired quality-pilot protocol freeze (legacy work item S11-B1) does not create:

- `docs/results/s11b_quality_pilot/same_unit_control.json`, schema
  `qaq-s11b-quality-pilot-mode-result-v1`;
- `docs/results/s11b_quality_pilot/lookahead_attention_one_unit_treatment.json`,
  the same per-mode schema;
- `docs/results/s11b_quality_pilot/aggregation.json`, schema
  `qaq-s11b-quality-pilot-aggregation-v1`.

The config freezes required identity, input, repeat, quality, route,
provenance, freeze, prohibition, pairing, threshold, error, and classification
fields. Paths are unique and project-relative; overwrite is forbidden.

## Structural validation evidence

The standard-library-only validator is
`scripts/validate_s11b_protocol.py`. It parses JSON fail-closed, rejects
non-finite and duplicate JSON values, enforces the frozen config bytes and all
semantic contracts, derives identities and inputs from authoritative project
files, recomputes token and historical-route digests, and refuses any existing
planned result. It writes no file and imports no ML/backend runtime.

Focused tests in `tests/unit/test_s11b_protocol.py` passed `50`. They cover
canonical and deterministic CLI success, no writes/results, mode and input
order, source agreement, interpretation and paths, import without prohibited
packages, malformed/non-finite JSON, simulated pre-existing evidence, and the
requested mutation classes. The S11-A/request-state/router regression
selection passed `24`; the full unit suite passed `364` with one established
optimizer warning; Ruff passed on both new Python paths; and `git diff --check`
passed. These structural checks are not pilot quality evidence.

## Limitations and exact next action

The quality question remains unresolved. Two requests cannot establish
lookahead quality generally, and no result may be inferred from the lookahead-routing semantic checks (legacy work item S11-A)
or the protocol-freeze structural tests (legacy work item S11-B1).

Next action: implement the fail-closed paired quality-pilot executor and
non-executing plan from the frozen protocol (legacy work item S11-B2), while explicitly not running the real
Qwen3-4B pilot during that executor-readiness work (legacy work item S11-B2).

## Make the paired quality-pilot executor ready

_Legacy work-item reference: S11-B2._

**COMPLETE — executor ready, pilot not executed.** The paired quality-pilot executor-readiness work (legacy work item S11-B2) implements the
frozen protocol (legacy work item S11-B1) without changing its config, identities, inputs, timing
semantics, margins, classifications, schemas, or output paths. Reusable code
uses the semantic components `qaq.evaluation.lookahead_quality_protocol`,
`qaq.evaluation.lookahead_quality_runner`, and
`qaq.evaluation.lookahead_quality_runtime`; the CLI is
`scripts/run_lookahead_quality_pilot.py`.

### Established facts and implementation choices

Established by the protocol freeze (legacy work item S11-B1) and preserved here: exact mode/request order, one fresh
child per mode, two repeats in each child, seed 1729, full 64-token fixed
inputs, prompt/completion/causal ranges, candidate order `[4,8]`, resident hard
routing, S11-A target ownership, historical S07 control routes, completion-only
temperature-2 KL, full-logit errors, state freeze requirements, quality
margins, classification precedence, schemas, and planned paths.

Implementation choices for the paired quality-pilot executor-readiness work (legacy work item S11-B2) are recorded in D057. The default path applies
the canonical validator and prints deterministic one-mode child and
aggregation commands. It imports no ML/backend package, opens no external
model resource, creates no path, and explicitly reports false model, CUDA,
pilot, and write activity. Unknown or ambiguous dispatch fails before the
production runtime import. Explicit mode execution requires one exact frozen
mode, an explicit `cuda:<index>`, its exact frozen output, and an already
existing non-symlink frozen output parent.

Each per-mode result uses schema `qaq-s11b-quality-pilot-mode-result-v1` and
retains raw ordered request/repeat evidence, teacher/student logit digests,
complete target-owned route maps and provenance, cleanup proof, recomputed
quality and route summaries, and sorted value-hash entries for teacher,
resident packed state, non-router base, and router parameters/persistent
buffers. The validator requires dtype, shape, non-trainability, absent
gradients, absent optimizer, per-component and aggregate before/after equality,
and rejects prohibited result data.

Aggregation validates both modes independently in frozen order, requires equal
protocol/input/device/software/teacher identities, and recomputes paired
quality margins, route distances, changed units, and layer-0 attention/FFN
equality. Missing external results return `PAUSE`; malformed complete evidence
returns `INVALID_EVIDENCE`; valid quality evidence alone can return
`ADVANCE_TO_BROADER_QUALITY_CHECK` or `CHECKPOINT_REUSE_DEGRADES`.

Persistence accepts only complete in-memory validated evidence. It checks the
exact destination and parent before work and promotion, refuses every existing
file/directory/symlink, writes and fsyncs a same-directory temporary file,
rereads/reparses/revalidates it, promotes with atomic no-overwrite `os.link`,
verifies promoted bytes and SHA-256, and cleans temporary files on safe failure.
Tests exercise this only in temporary directories.

### Structural verification boundary

Focused tests use an explicit deterministic injected runtime through the same
schedule/result construction boundary and label its output **test-only
structural evidence**. They cover inert plans and forbidden imports, including
direct import of the production runtime module; lazy dispatch; exact canonical
absent-parent, existing-destination, destination-symlink, and immediate-linked-
parent refusal before production imports; exact scheduling; per-mode identity,
execution, metric, route, provenance, cleanup, state, and prohibited-work
rejection; keyed historical control checks; both valid quality classifications
and each independent quality margin; tokenizer, packed-artifact,
Any-Precision, checkpoint metadata/order, fixed-input/range, and
hardware/software identity mutations; resource-versus-identity preflight for
model, packed and router checkpoints, backend checkout, architecture, and
comparable GPU; missing/wrong environment and CUDA availability/index/driver
resource classification; exact 72-router, 23,620,752-scalar, and 252 resident
packed-target representation validation; malformed evidence; missing-result
`PAUSE`; and atomic success, overwrite, race, malformed serialization, cleanup,
byte/digest, parent, link, and unrelated-path safety. Focused executor-readiness tests (legacy work item S11-B2) pass
`107`; protocol-freeze behavior (legacy work item S11-B1) and the reusable source-file mode check pass `51`;
the safe CPU selection passes `470` with one established warning.

No Qwen3-4B model, packed artifact, S07 checkpoint, CUDA kernel, teacher or
student inference, metric, generation, decode, perplexity, training,
evaluation, benchmark, profiler, production aggregation, or real result path
ran. No quality or performance claim is made, and no file under
`docs/results/`, `papers/`, or `third_party/` changed.

The quality question remains unknown. The exact next action is to execute the frozen paired quality pilot (legacy work item S11-B3): under a
separate authorization, provision the frozen empty result parent, execute the
two printed one-mode child commands on one explicit comparable GPU in frozen
order, then run the printed aggregation command. Do not infer or execute that pilot
from the paired-executor readiness evidence (legacy work item S11-B2) alone.

## Execute the frozen paired quality pilot

_Legacy work-item reference: S11-B3._

**COMPLETE — `ADVANCE_TO_BROADER_QUALITY_CHECK`.** The frozen paired quality-pilot execution (legacy work item S11-B3) used the unchanged
frozen protocol SHA-256
`21a664424debe4892c3577c490158228dd5399bb4b425611db728070d23a5051` and the
paired quality-pilot executor (legacy work item S11-B2). Preflight verified the pinned model/tokenizer snapshot, packed
artifact SHA-256, S07 checkpoint SHA-256, clean pinned Any-Precision revision,
mandatory `~/.venv`, absent result paths, and an empty non-symlink result
parent. No frozen input, identity, threshold, schema, or implementation changed.

The one authorized execution used this exact order: `same_unit_control`, then
`lookahead_attention_one_unit_treatment`, then aggregation. Each mode ran once
in its own fresh process and performed the frozen two repeats. Both used
`cuda:2`, physical GPU `GPU-74d97f46-6284-1055-698a-e2db4e9c744b`, an NVIDIA
GeForce RTX 3090 with driver `580.159.03`; their persisted hardware/software
identity records are equal. The mode and aggregation commands exited zero and
wrote through the executor's validated no-overwrite persistence path.

Canonical evidence:

- `docs/results/s11b_quality_pilot/same_unit_control.json`, SHA-256
  `ba748dd09b8319c1ff395f65be130ecbb0bea1571c1afb76e0016a88b6e5a073`;
- `docs/results/s11b_quality_pilot/lookahead_attention_one_unit_treatment.json`,
  SHA-256 `742450cfe5dda791cbbbdc59adf1541a2d897f227b9be094909f36b7760c402c`;
- `docs/results/s11b_quality_pilot/aggregation.json`, SHA-256
  `2b1755345bb0a8bbae3110bbdca86bf7dc75edef9c3460e83e37b0297fe626a7`.

Independent post-write validation reloaded all three files, validated each mode,
and recomputed the aggregation byte-for-byte. Both repeats, freeze audits,
historical control routes, complete route/provenance coverage, layer-0 pairing,
teacher logits, cleanup, and prohibited-work checks passed. Aggregation errors
are empty. Aggregate KL changed from `0.0631424393504858` to
`0.06203574314713478` (ratio `0.9824730210816205`); per-request KL ratios are
`0.9431116083999646` and `1.0`; aggregate mean absolute logit error changed
from `0.2928081303834915` to `0.2907368540763855` (ratio
`0.9929261653206376`). Every frozen quality margin passed.

Route comparison found one change: `validation-3` target attention layer 23
selected 8 bits under treatment instead of 4. Its overall and attention
normalized distances are `1/72` and `1/36`; `validation-1000` and all FFN
routes are unchanged. Required layer-0 attention and FFN equality passed.

This is a valid completed two-request pilot, not general quality evidence and
not a performance, overlap, transfer, or prefetch claim. It authorizes only the
separate definition of a broader quality check. No broader evaluation,
retraining, 6-bit routing, on-demand loading, asynchronous transfer, prefetch,
generation, decode, perplexity, or performance measurement began.
