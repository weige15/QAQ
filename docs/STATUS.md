Current stage: S08
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

S08-A evidence:
- The implementation subdivision S08-A established a synchronous loader for
  one concrete request state and retained no process-global request cache.
- The real S01 pinned fixture keeps `[8,64,32]` `torch.int32` qweight and
  `torch.float16` row LUTs on CPU before first use.
- First-use bytes were 34,816 for 4-bit, 98,304 for fresh 8-bit, and 65,536
  incremental bytes for a 4-to-8 upgrade. Reuse events transferred zero
  bytes.
- Resident and transferred 4-bit and 8-bit fixture outputs were bitwise equal.
  Request end released all retained GPU references, and duplicate textual IDs
  used independent request-state ownership.
- Focused S08-A tests passed: 8. Ruff passed for changed source and tests.
- No full-model Qwen3 on-demand evaluation, memory comparison, latency
  comparison, or S09 work was performed.

S08-A gate: CONTINUE.

S08-B evidence:
- The external Codex service-overload interruption was classified as infrastructure interruption, not a QAQ defect.
- The required S03-B packed artifact, S07 router checkpoint, pinned Qwen3 snapshot, pinned Any-Precision revision, and CUDA device were present and matched their recorded hashes.
- Real Qwen3 on-demand execution used 252 CPU-authoritative packed sources, with zero remaining `AnyPrecisionLinear` modules and no complete packed GPU copy.
- Resident and on-demand hard routes matched for both locked S07 validation requests.
- Both routes produced finite logits that were bitwise equal, with zero mean and maximum absolute logit difference.
- Four-token deterministic greedy generation matched between resident and on-demand modes for both requests, and routes remained fixed during decode.
- On-demand transfer accounting was `3,817,717,760` bytes for `validation-3` and `3,835,002,880` bytes for `validation-1000`; both matched the independent expected-byte calculation exactly.
- All transfer occurred during prefill, with zero decode transfer bytes and zero reuse transfer bytes.
- Each request retained 252 entries and 504 packed GPU buffers before cleanup, then retained zero entries, buffers, or packed bytes after `end_request()`.
- A later fresh request transferred its selected packed buffers again, proving request isolation.
- Synchronized two-repeat measurements recorded resident median prefill/decode/end-to-end latencies of `0.145354`/`0.187833`/`0.332110` seconds and on-demand medians of `5.815631`/`0.229669`/`6.031509` seconds.
- Resident peak allocated memory was `5,724,945,408` bytes at maximum across repeats; on-demand peak allocated memory was `4,806,114,304` bytes.
- Focused S08-B real tests passed: `3 passed in 438.03s`; S08-A focused tests remained `8 passed in 8.55s`; Ruff passed for all changed S08 files.
- The valid recorded S08-B regression result remains `8 passed in 651.74s`; it was not rerun because no relevant implementation or execution-path change invalidated it.
- Complete evidence and provenance are recorded in `docs/results/s08_on_demand.json`, including code snapshot hashes, model and artifact revisions, request digests, method, transfer records, allocator measurements, and commands.

S08 gate: COMPLETE.

Passing S08 implementation and evidence commit: `ae5e991`.
Next action: S09. Do not execute S09 in this task.
