Current stage: S07
Status: IN_PROGRESS

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

Next action: S07-B: lock real training data and hyperparameters, perform the single baseline router-distillation run, then evaluate soft and deterministic hard routes.
