# S07 — Router distillation

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
