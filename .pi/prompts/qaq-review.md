---
description: Review one QAQ work item for correctness and evidence
argument-hint: "<objective or work item>"
---

Review exactly QAQ work item `$1`.

This is a review task, not an implementation task and not authorization to
begin a follow-up objective.

Follow `AGENTS.md`, including its environment requirements.

Read:

- `docs/STATUS.md`;
- `docs/DECISIONS.md`;
- the document for work item `$1` under `docs/stages/`;
- the relevant source notes and papers;
- the changed source, tests, configuration, and documentation; and
- the evidence produced for the work item.

Review the work item against what it actually claims to establish.

Check:

- source claims against the cited evidence;
- implementation behavior against the work-item requirements;
- acceptance conditions against current test and result evidence;
- unstated assumptions;
- deterministic seeds and exact commands where required;
- frozen identities, hashes, protocols, and thresholds where required;
- scope boundaries and prohibited work;
- whether tests exercise the claimed behavior rather than merely nearby
  behavior; and
- whether `docs/STATUS.md` describes the verified state without overstating
  unresolved evidence.

Where applicable, verify the QAQ baseline invariants in `AGENTS.md`, including
physical bit-packing, frozen quantized weights during router training, and
transfer of packed planes rather than unpacked weights.

Separate the result into:

### Established

State what the reviewed evidence supports.

### Problems

State concrete correctness, evidence, reproducibility, or scope problems.
For each problem, identify the requirement it conflicts with and the evidence.

### Unknown

State questions the current evidence does not resolve.
Do not infer an answer merely because the existing implementation is
consistent with one.

### Assumptions

State any material implementation assumption not established by the cited
sources or frozen project decisions.

### Recommendation

Recommend exactly one:

- `ACCEPT` — no material review issue remains;
- `REVISE` — bounded corrections to the current objective are required;
- `PAUSE` — required evidence or external material is unavailable; or
- `STOP` — proceeding would cross a material project boundary.

Explain what evidence would change a `REVISE`, `PAUSE`, or `STOP`
recommendation.

Do not implement fixes.

Do not begin a follow-up objective.

Do not modify PDFs under `papers/`.

Record a newly discovered assumption in `docs/DECISIONS.md` only when it is a
durable material project decision rather than a routine implementation detail.

Update `docs/STATUS.md` only when the review establishes a durable change in
verified project state.
