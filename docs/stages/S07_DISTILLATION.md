# S07 — Router distillation

## S07-A — reusable distillation machinery and deterministic smoke gate

**Status: COMPLETE — CONTINUE to S07-B.** This section implements the
teacher-student router-training seams and validates them on a deliberately tiny
local fixture. It does not run real dataset-scale training, evaluate routing
quality, add a width/latency/transfer/entropy penalty, or implement S07-B/S08.

### Loss and data contract

The implementation is `src/qaq/s07_distillation.py` and uses

```text
L_KD = T^2 * mean_{(b,s): m[b,s]=1} KL(
    softmax(z_teacher[b,s,:] / T) || softmax(z_student[b,s,:] / T)
)
```

Teacher and student logits are `[batch, sequence, vocabulary]`; the final
axis is the vocabulary axis. The implementation computes float32
`log_softmax` values and calls `torch.nn.functional.kl_div` with
`reduction="none"`, `log_target=True`, teacher log-probabilities, and a sum
over vocabulary before applying the explicit completion mask. The denominator
is the count of true completion-mask entries. Prompt positions and padding are
excluded; the mask is never inferred from `attention_mask`. Zero valid
completion targets fail clearly.

`DistillationExample` and `DistillationBatch` carry tokenizer revision, aligned
input/target IDs, full attention mask, explicit completion-loss mask, sequence
positions, and either prompt/completion text plus an explicit prompt mask or
their token ranges. The S06 execution receives the full model attention mask
and the separate prompt-only mask, preserving the S05 feature timing.

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
length 4, prompt range `[0,2)`, completion range `[2,4)`, explicit completion
mask `[0,0,1,1]`, tokenizer revision `tok-r1`, and seed `1729`. Smoke-only
settings are temperature `2.0`, SGD learning rate `1e-2`, and two steps; these
are not baseline decisions.

```text
source ~/.venv/bin/activate
which python                         # /nfs/home/s314511048/.venv/bin/python
python --version                     # Python 3.12.3
PYTHONPATH=src pytest -q tests/unit/test_s07_distillation.py tests/integration/test_s07_distillation_smoke.py
                                     # 8 passed
```

Measured smoke results from the same fixture and command path:

- step 1 KD loss `0.00010941564687527716`; router gradient norm
  `0.00015089756434509636`; router parameter changed: `True`;
- step 2 KD loss `0.00010947752161882818`; router gradient norm
  `0.00015089709935220638`; router parameter changed: `True`;
- all losses and gradients were finite;
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

### S07-B unresolved choices

The real training source/data subset, tokenizer data command, final
temperature, optimizer and learning rate, sequence length, batch size,
number of epochs/steps, scheduler, and convergence/evaluation thresholds
remain unresolved or proposed. The smoke values above must not be promoted to
baseline decisions. S07-B must lock those choices, run the one real baseline
router-distillation job, then evaluate soft and deterministic hard routes.

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
