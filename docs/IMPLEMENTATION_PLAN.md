# QAQ implementation plan

## Objective

Build a reproducible, paper-guided QAQ baseline rather than claiming an exact reproduction.
The baseline will use physically bit-packed nested multi-precision weights, query-conditioned precision routing, separate attention and FFN routes, teacher-student router training, hard query-level inference routes, and synchronous on-demand loading of selected packed planes from CPU to GPU.

## Evidence discipline

Each work item must record which behavior is supported by a source paper and which detail is an implementation choice made because the sources leave it unspecified.
Unresolved questions remain unresolved until a work item produces evidence.

## Objective sequence

Legacy identifiers are retained in parentheses for links to frozen decisions, evidence, paths, and machine-facing contracts.

1. Lock the environment and specification (legacy work item S00).
2. Validate the Any-Precision backend (legacy work item S01).
3. Specify and verify physical bit packing (legacy work item S02).
4. Establish static 4-bit and 8-bit model baselines (legacy work item S03).
5. Prove manual independent attention/FFN precision plans (legacy work item S04).
6. Implement query features and request state without a learned router (legacy work item S05).
7. Implement a trainable soft router with router-only trainable parameters (legacy work item S06).
8. Distill the router from a full-precision teacher and evaluate hard routes (legacy work item S07).
9. Add synchronous on-demand loading of selected packed planes (legacy work item S08).
10. Compare modes and freeze the reproducible baseline (legacy work item S09).
11. Enable static 6-bit execution (legacy work item S10-A).
12. Define three-way router semantics (legacy work item S10-B).
13. Add the cost-aware 4/6/8 router objective (legacy work item S10-C).
14. Calibrate the bit-cost coefficient (legacy work item S10-D).
15. Freeze frontier confirmation (legacy work item S10-E).
16. Execute frozen frontier confirmation (legacy work item S10-F).
17. Define and freeze broader validation (legacy work item S10-G).
18. Implement and validate the fail-closed broader-validation runner and non-executing plan (legacy work item S10-H1).
19. Implement the lazy, auditable real executor with explicit device selection and temporary noncanonical output (legacy work item S10-H2-A).
20. Separately authorize and execute the frozen broader-validation protocol (legacy work item S10-H2-B).
21. Define and validate one-unit-lookahead attention routing semantics without running the quality pilot (legacy work item S11-A).
22. Freeze the paired lookahead quality-pilot protocol before execution (legacy work item S11-B1).
23. Implement and structurally validate the fail-closed paired pilot executor, inert plan, result validators, aggregation, and atomic no-overwrite persistence without running the pilot (legacy work item S11-B2).
24. Separately authorize and execute the frozen paired pilot, then classify its real evidence (legacy work item S11-B3).
25. Freeze the broader lookahead quality protocol (legacy work item S11-C1).
26. Implement the broader lookahead quality executor and persistence boundary (legacy work item S11-C2).
27. Execute and classify the canonical broader lookahead quality comparison (legacy work item S11-C3).
28. Freeze the paired lookahead-specific 4/6/8 training protocol (legacy work item S11-D1).
29. Implement the deterministic paired-training plan and fail-closed dispatcher (legacy work item S11-D2).
30. Implement the production runtime for paired lookahead-specific 4/6/8 router training and evaluation (legacy work item S11-D3; current objective).

No objective may begin automatically. A work item stops at its decision gate and updates `docs/STATUS.md` before handoff.
The authoritative current objective and its follow-up action are recorded in `docs/STATUS.md`; historical codes do not grant authority to begin either one.

## Initial implementation choices

The decision ledger in `docs/DECISIONS.md` is authoritative for the initial choices, including the upstream Any-Precision starting point, the initial 4/8-bit router candidate scope, unit-level routing, BF16/FP16 non-quantized components, prompt feature construction, distillation objective, hard routing, synchronous request-scoped loading, batch-size-one scope, and baseline freeze boundaries.

## Comparative five-mode baseline evaluation (legacy work item S09)

The final baseline comparison must include the full-precision teacher, static 4-bit, static 8-bit, routed resident, and routed on-demand modes.
Record quality, route selections, memory, transfer bytes, and latency with exact commands, deterministic seeds, and environment versions.
