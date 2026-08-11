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
- The locked configuration is `configs/s07_router_training.json`; it uses four
  deterministic Wikitext train examples, two separate validation examples,
  explicit 32-token prompt/completion boundaries, sequence length 64, batch
  size 1, AdamW, four steps, KD temperature 2.0, routing temperature 1.0, and
  seed 1729.
- Exactly one baseline training run completed. KD loss decreased from
  `0.1730574965` to `0.0317778103`; all losses and router gradients were
  finite. The packed student base was unchanged and the optimizer contained
  only the 23,620,752 router scalars.
- Soft validation KD loss was `0.0386699643` with mean absolute logit error
  `0.2430240735`; hard validation KD loss was `0.0631424394` with error
  `0.2928081304`. Hard routing selected 4 bits for `20.1389%` and 8 bits for
  `79.8611%` of routing units, with complete 72-unit logs per request.
- Fresh-process router checkpoint reload and fixed-subset hard-route
  determinism passed by bitwise comparison. The final router checkpoint is
  external to Git and its SHA-256 is recorded in the result artifact.
- The teacher was evaluated under `no_grad` and did not change, but its
  parameters were not explicitly set to `requires_grad=False` before the
  freeze audit. This is a gate defect, so S07 remains IN_PROGRESS with
  engineering result REVISE. The corrected script is ready, but the one-run
  rule forbids a silent rerun.
- Query-adaptive routing was not demonstrated: the result is `OTHER` because
  mixed routes existed but prompt-to-prompt variation was below the recorded
  material-variation threshold. No S08 work was started.

Result artifact: `docs/results/s07_router_training.json`.

Next action: obtain explicit authorization for one corrected S07-B rerun using the unchanged locked configuration, then re-evaluate the S07 gate. Do not begin S08.
