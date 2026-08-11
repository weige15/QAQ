Current stage: S07
Status: COMPLETE

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
  `configs/s07_router_training.json`: four deterministic Wikitext training
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

Next action: begin the repository-defined S08 work in a later task. This task
stops at the completed S07 gate and does not begin S08.
