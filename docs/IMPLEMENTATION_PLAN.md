# QAQ implementation plan

## Objective

Build a reproducible, paper-guided QAQ baseline rather than claiming an exact reproduction.
The baseline will use physically bit-packed nested multi-precision weights, query-conditioned precision routing, separate attention and FFN routes, teacher-student router training, hard query-level inference routes, and synchronous on-demand loading of selected packed planes from CPU to GPU.

## Evidence discipline

Each stage must record which behavior is supported by a source paper and which detail is an implementation choice made because the sources leave it unspecified.
Unresolved questions remain unresolved until a stage produces evidence.

## Stage order

1. S00 — Lock environment and specification.
2. S01 — Validate the Any-Precision backend.
3. S02 — Specify and verify physical bit packing.
4. S03 — Establish static 4-bit and 8-bit model baselines.
5. S04 — Prove manual independent attention/FFN precision plans.
6. S05 — Implement query features and request state without a learned router.
7. S06 — Implement a trainable soft router with router-only trainable parameters.
8. S07 — Distill the router from a full-precision teacher and evaluate hard routes.
9. S08 — Add synchronous on-demand loading of selected packed planes.
10. S09 — Compare modes and freeze the reproducible baseline.
11. S10-A — Enable static 6-bit execution.
12. S10-B — Define three-way router semantics.
13. S10-C — Add the cost-aware 4/6/8 router objective.
14. S10-D — Calibrate the bit-cost coefficient.
15. S10-E — Freeze frontier confirmation.
16. S10-F — Execute frozen frontier confirmation.
17. S10-G — Define and freeze broader validation.
18. S10-H1 — Implement and validate the fail-closed broader-validation runner and non-executing plan.
19. S10-H2-A — Implement the lazy, auditable real executor with explicit device selection and temporary noncanonical output.
20. S10-H2-B — Separately authorize and execute the frozen broader-validation protocol.
21. S11-A — Define and validate one-unit-lookahead attention routing semantics without running the quality pilot.
22. S11-B1 - Freeze the paired lookahead quality-pilot protocol before execution.

No stage may begin automatically. A stage stops at its decision gate and updates `docs/STATUS.md` before handoff.
The immediate next action is to begin S11-B2 implementation of the fail-closed executor and non-executing plan from frozen B1, while explicitly not running the real Qwen3-4B pilot in B2.

## Initial implementation choices

The decision ledger in `docs/DECISIONS.md` is authoritative for the initial choices, including the upstream Any-Precision starting point, the initial 4/8-bit router candidate scope, unit-level routing, BF16/FP16 non-quantized components, prompt feature construction, distillation objective, hard routing, synchronous request-scoped loading, batch-size-one scope, and baseline freeze boundaries.

## Comparative evaluation at S09

The final baseline comparison must include the full-precision teacher, static 4-bit, static 8-bit, routed resident, and routed on-demand modes.
Record quality, route selections, memory, transfer bytes, and latency with exact commands, deterministic seeds, and environment versions.
