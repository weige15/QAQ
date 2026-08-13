# S07 — Router distillation

## S07-A — reusable distillation machinery and deterministic smoke gate

**Status: COMPLETE — S07 gate passed; do not begin S08 in this task.** This section implements the
teacher-student router-training seams and validates them on a deliberately tiny
local fixture. It does not run real dataset-scale training, evaluate routing
quality, add a width/latency/transfer/entropy penalty, or implement S07-B/S08.

The S07-A implementation choices are recorded in
[`docs/DECISIONS.md` — D025](../DECISIONS.md#d025--s07-a-distillation-seams-2026-08-11).

### Loss and data contract

The implementation is `src/qaq/s07_distillation.py` and uses

```text
L_KD = T^2 * mean_{(b,s): m[b,s]=1} KL(
    softmax(z_teacher[b,s,:] / T) || softmax(z_student[b,s,:] / T)
)
```

Teacher and student logits are `[batch, sequence, vocabulary]`; the final
axis is the vocabulary axis. For causal models, `target_ids[t]` is the token
predicted by logit position `t`, so `target_ids[t] = input_ids[t+1]` for
linked valid tokens and `-100` otherwise. A completion token range `[a,b)`
therefore maps to the explicit logit mask `[a-1,b-1)`. The implementation
computes float32
`log_softmax` values and calls `torch.nn.functional.kl_div` with
`reduction="none"`, `log_target=True`, teacher log-probabilities, and a sum
over vocabulary before applying the explicit completion mask. The denominator
is the count of true completion-mask entries. Prompt positions and padding are
excluded; the mask is never inferred from `attention_mask`. Zero valid
completion targets fail clearly. The prompt-only feature mask remains in input
token coordinates and is intentionally separate from the causal logit mask.

`DistillationExample` and `DistillationBatch` carry tokenizer revision, causally
aligned input/target IDs, full attention mask, explicit completion-logit mask,
sequence positions, and either prompt/completion text plus an explicit prompt
mask or their token ranges. Range-based examples require prompt tokens to end
before completion tokens start and require the completion mask to be the causal
shift of that range. When a completion range is supplied without a prompt
range, the explicit prompt mask must still select only preceding tokens. The
S06 execution receives the full model attention mask and the separate
prompt-only mask, preserving the S05 feature timing.

### Freeze and optimizer evidence

`freeze_teacher_and_packed_student` freezes every full-precision teacher
parameter and every S06 student parameter outside `routers.`. Snapshots verify
`requires_grad=False`, no gradient, and unchanged values before and after
backward and optimizer update. The explicit optimizer constructor includes
router names only and audits exact tensor and scalar counts plus included name
prefixes; it does not select parameters solely by `requires_grad`.

The smoke fixture has 72 routers, 288 router tensors, and 5,616 router scalar
parameters (hidden width 4, fixture feature width 16). The production Qwen3-4B
router architecture/count remains the S06 value of 72 routers and 23,620,752
scalars.

### Fixture, command, and measured result

Fixture: deterministic 36-layer Qwen3-shaped local model with a tiny
full-precision teacher and the existing S06 soft execution seam, sequence
length 4, prompt range `[0,2)`, completion range `[2,4)`, shifted target IDs
`[2,3,4,-100]`, explicit completion-logit mask `[0,1,1,0]`, tokenizer
revision `tok-r1`, and seed `1729`. Smoke-only
settings are temperature `2.0`, SGD learning rate `1e-2`, and two steps; these
are not baseline decisions.

```text
source ~/.venv/bin/activate
which python                         # resolves inside ~/.venv
python --version                     # Python 3.12.3
PYTHONPATH=src pytest -q tests/unit/test_s07_distillation.py tests/integration/test_s07_distillation_smoke.py
                                     # 9 passed
```

Measured smoke results from the same fixture and command path:

- both smoke steps had finite KD loss and router gradient norm, and router
  parameters changed;
- teacher and packed student base parameters stayed frozen and unchanged;
- 72/72 route records were emitted exactly once;
- router-only checkpoint probabilities and deterministic hard routes matched
  after load.

### Hard routes, logs, and observational statistics

`hard_route` uses ordinary `torch.argmax([p4,p8])`, maps index 0 to 4-bit and
index 1 to 8-bit, and therefore resolves ties to 4-bit because the first
maximum wins. It performs no sampling. Compact route records contain only
request ID, layer, unit type, `p4`, `p8`, hard bit, entropy, and optional soft
average width. Coverage validation requires exactly one attention and one FFN
record per layer and request. Statistics report KD loss separately, entropy
with documented log base (the smoke uses base 2), `4*p4 + 8*p8`, hard 4/8
fractions, per-layer precision distribution, attention-vs-FFN distribution,
and hard-route variation across prompts. These statistics are observational and
do not affect the KD loss.

Router checkpoints contain only router state, optional optimizer state, and
metadata for model repository/revision, packed checkpoint ID/hash,
Any-Precision revision, router architecture, candidate ordering, and training
step metadata. Teacher and packed student weights are never serialized.

### S07-B locked baseline configuration

The locked configuration is in `configs/s07_router_training.json` and is an
implementation choice, not a QAQ-paper fact. It uses `Salesforce/wikitext`,
configuration `wikitext-2-raw-v1`, revision
`b08601e04326c79dfdd32d625aee71d232d685c3`, train split rows selected at fixed
offsets `[0,1000,2000,3000]`, and validation split rows selected at fixed
offsets `[0,1000]`. At each offset the first non-empty row with at least 64
pinned-tokenizer tokens is selected. Raw text is tokenized with the pinned
Qwen3-4B tokenizer at revision `1cfa9a7208912126459214e8b04321603b3df60c`,
without special tokens; the first 64 tokens are retained, prompt tokens are
`[0,32)`, and completion tokens are `[32,64)`. No generated tokens are used.

The baseline has four training examples, two validation examples, sequence
length 64, batch size 1, gradient accumulation 1, seed 1729, one epoch/four
optimizer steps, AdamW, learning rate `1e-3`, weight decay 0, no scheduler, KD
temperature 2.0, fixed routing temperature 1.0, per-step logging, and
checkpoint/evaluation at the final step 4 only. The teacher logits are
precomputed with the frozen teacher under `no_grad` and kept on CPU before
optimization to fit the resident packed student on the 24-GiB GPU. This is
an execution-memory measure, not a new objective.

### S07-B first and corrected training runs

The first baseline run used the unchanged Qwen3-4B revision
`1cfa9a7208912126459214e8b04321603b3df60c`, packed student artifact hash
`29d9bc526b3da0bd39daf2f82afd141f82d005ca1232cabc75cfe9d9ecc1cfee`, and
Any-Precision revision `a3257d02740cc5757c78673da534b0630ff3a4ea`. It used
`no_grad`, excluded the teacher from the optimizer, and left teacher values
unchanged, but did not explicitly set teacher parameters to
`requires_grad=False` before the freeze audit. D027 therefore records that
run as **REVISE**; it remains part of the history and was not erased.

D008-1 authorized exactly one corrected rerun with the same locked data,
model, packed artifact, optimizer, temperatures, seed, and four-step schedule.
The production path now invokes the audited teacher/student freeze seam before
teacher-logit precomputation and records teacher before/after hashes and
post-run gradient absence. The corrected run passed those audits: teacher
parameters were explicitly frozen and unchanged, packed-student non-router
parameters and buffers were unchanged, and the optimizer contained only the
23,620,752 router scalars. All four KD losses and router gradient norms were
finite, and router parameters changed.

The corrected run's KD loss was finite at every step and decreased from
`0.1730574965` to `0.0317778103`. The router-only final checkpoint is external
to Git at `~/.cache/qaq/s07b/final_router.pt` with SHA-256
`08bf646f19759c0d7949e159bdbe4f96bbea737204b96f8760d205c8d6fd1949`.
Fresh-process reload matched probabilities and hard routes, and fixed-subset
hard-route repeats matched route maps, selected precisions, and logits by
bitwise comparison.

### S07-B soft and hard routing observations

On the two validation examples, soft routing had final validation KD loss
`0.0386699643` and mean absolute logit error `0.2430240735` against the
full-precision teacher. Deterministic hard routing had KD loss `0.0631424394`
and mean absolute logit error `0.2928081304`; static 4-bit and static 8-bit
errors were `0.7434162199` and `0.0910567641`. Thus hard-minus-soft error was
`0.0497840569`, hard-minus-static-8 error was `0.2017513663`, and
hard-minus-static-4 error was `-0.4506080896`.

Hard routing selected 4 bits for `20.1389%` of units and 8 bits for `79.8611%`.
Attention fractions were 4-bit `16.6667%` and 8-bit `83.3333%`; FFN fractions
were 4-bit `23.6111%` and 8-bit `76.3889%`. There were two unique hard route
maps, 72 route records per request with complete coverage, mean hard width
`6.472222`, mean soft width `6.455843`, and mean prompt-to-prompt route
distance `0.0138889`. Parameter-weighted width was not supported by the
existing S07-A statistics. The observation is `OTHER`: mixed routes exist but
the two validation route maps did not meet the recorded material-variation
threshold. Query-adaptive routing was **not demonstrated**; this is not by
itself a router implementation failure.

Engineering gate: **CONTINUE**. Query-adaptivity demonstrated: **NO**.
The existing classification remains `OTHER` and is non-blocking under this
S07 gate. No S08 work was started.

## S07C-EVIDENCE-005 — hard-route checkpoint round-trip repair

This evidence repair is resolved with **CONTINUE**. The decision and exact
direct actual-route comparison are owned by
[`D039`](../DECISIONS.md#d039--s07c-evidence-005-direct-hard-route-round-trip);
the machine-readable evidence is recorded in
[`docs/results/s07_router_training.json`](../results/s07_router_training.json),
and the current status and next action remain owned by
[`docs/STATUS.md`](../STATUS.md). No router training, router-semantic change,
six-bit routing, or S10-B execution was part of this repair.

## Goal

Train the router from a full-precision teacher using the documented baseline objective and evaluate deterministic hard routes.

## Tasks

- Define the full-precision teacher and reproducible evaluation inputs.
- Train only router parameters using teacher-student logit distillation.
- Apply D008: no bit-width penalty in the baseline objective.
- Convert soft routing to hard inference routing using argmax per D009.
- Evaluate route stability, quality, and divergence from teacher and static baselines.
- Keep quantized model weights frozen throughout training.

## Tests

- Training updates router parameters only.
- The objective contains no bit-width cost penalty.
- Hard argmax routes are deterministic under fixed seeds and inputs.
- Teacher, soft-router, and hard-route outputs are compared with documented tolerances.
- Separate attention and FFN routes are evaluated.

## Required outputs

- Training configuration and exact command.
- Reproducible router checkpoint or generation command.
- Distillation and hard-route evaluation report.
- Route statistics and failure cases.
- Updated decisions and status.

## Known uncertainties

- Teacher data, loss reduction, optimizer, schedule, and convergence thresholds remain to be fixed and recorded.
- The source papers may not prescribe all training details needed by this baseline.

## CONTINUE condition

The router trains reproducibly with frozen quantized weights, no bit-width penalty, and deterministic hard routes meeting the documented quality gate.

## PAUSE condition

Teacher execution, data, or compute is unavailable.

## REVISE condition

A training or hard-routing assumption needs evidence-based correction.

## STOP condition

The baseline cannot train router-only, requires a cost penalty before freeze, or hard routes are not reproducible or usable.
