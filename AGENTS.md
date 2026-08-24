# QAQ contribution rules

This file is authoritative for every worker operating in this repository.

## FirstMate and worker authority

When a worker is launched by FirstMate, the active FirstMate task brief owns the
worktree, branch, status reporting, delivery mode, and merge authority. Do not
create a second orchestration process inside QAQ and do not wait for the captain
between routine steps already covered by the brief.

A request to implement or complete the current QAQ stage authorizes one complete
stage-delivery cycle on the assigned feature branch:

1. inspect the current stage and its prerequisites;
2. make the smallest reversible implementation choices consistent with the
   stage documents;
3. edit, test, retry failed checks, and recover a lost worker session;
4. update the durable status or decision record when the evidence requires it;
5. commit the stage work; and
6. push the feature branch and open or update its PR when the active FirstMate
   delivery contract requires a PR.

Do not request permission between those routine actions. Continue until the
stage is PR-ready, the active delivery contract reaches its completion point, or
a material decision is genuinely required.

Escalate only when at least one of these conditions applies:

- the scientific claim, frozen protocol, acceptance threshold, current-stage
  scope, or preserved behavior would change;
- a later stage would begin;
- work would be discarded, overwritten, made irreversible, or shared history
  would be rewritten;
- the default branch would be merged, a force-push would be used, or the active
  FirstMate brief reserves another action to the captain;
- credentials, unavailable external artifacts, unbounded compute cost, or a
  real execution outside the documented stage authorization is required; or
- conflicting evidence or a missing prerequisite cannot be resolved by safe,
  read-only inspection or a bounded reversible correction.

Unknowns that do not affect safety, scientific interpretation, acceptance
criteria, reproducibility, external cost, or public behavior are worker choices.
Choose the smallest reversible option, cover it with tests, and continue. Record
only durable research, interface, or protocol choices in `docs/DECISIONS.md`;
do not turn routine naming, file placement, test selection, local refactoring,
or retry strategy into captain decisions.

## Mandatory environment

Before implementation, testing, Python inspection, dependency work, or
execution in each fresh shell, activate and verify the project environment:

```bash
source ~/.venv/bin/activate
which python
python --version
```

Stop the project command if activation fails or if `which python` does not
resolve inside `~/.venv`. A clean shell relaunch and another preflight are
routine recovery and do not require captain permission. Do not create another
virtual environment and do not use system Python.

## Required reading and stage discipline

Before starting work, read `docs/STATUS.md`, `docs/DECISIONS.md`, the current
stage document under `docs/stages/`, and relevant source notes and papers when
the stage requires them.

Work on the current stage only. Complete its bounded delivery cycle without
stopping at internal checks. Do not automatically begin a later stage. A stage
gate is a captain gate only when it meets the material-decision boundary above;
otherwise it is a worker decision point and the worker continues, revises, or
retries within the authorized stage.

State what is known from sources, what remains unknown, and any material
implementation assumptions. Do not silently change a frozen protocol or claim
that unresolved research evidence is established.

## Completion and reproducibility

Add or update tests before declaring a stage complete. Preserve deterministic
seeds and exact commands. Update `docs/STATUS.md` before delivery when the
verified project state changed. The status update must identify the evidence
and next stage without claiming unresolved work is complete.

A worker may report completion only at the active delivery contract's real end:
for example, a committed branch, a PR with required checks green, or a verified
local-only branch. Intermediate implementation, validation, commit, session
recovery, branch refresh, and PR update steps are not separate captain gates.

## Baseline boundaries

This project is a paper-guided QAQ baseline, not an asserted exact reproduction.
Production bit planes must be physically bit-packed. Byte-per-bit
representations are test/reference-only. Fake quantization cannot support
memory, transfer, or latency claims. Quantized model weights remain frozen
during router training. The baseline loader is synchronous. Packed planes, not
unpacked weights, must be transferred.

Before the baseline is frozen, do not add asynchronous transfers, prefetching,
transfer prediction, bit-width cost penalties, cross-request caching,
multi-query batching, or unrelated research improvements. Do not add the
Any-Precision dependency until the project has completed its source/backend
review and the exact upstream commit is pinned.

## Scope

The initial baseline targets genuinely packed nested multi-precision weights,
query-conditioned routing, separate attention and FFN routes, teacher-student
router training, hard query-level inference routes, and synchronous on-demand
CPU-to-GPU loading of selected packed planes.

Every stage must preserve the distinction between behavior supported by source
papers and implementation decisions made where the papers leave details
unspecified.

## Maintaining this file

Keep this file for knowledge useful to almost every future agent session in
this project. Do not repeat what the codebase already shows; point to the
authoritative file or command instead. Prefer rewriting or pruning existing
entries over appending new ones. When updating this file, preserve this bar for
all agents and keep entries concise.

## Worktree artifact provisioning

Before artifact-backed validation, provision the ignored S03 artifact with
`scripts/provision_packed_model_artifact.py`.

`QAQ_S03_ARTIFACT_SOURCE` must name the exact external directory containing the
frozen `pytorch_model.bin`. The script is authorized to create only the ignored
worktree-local symlink under `quantized/`; it must never copy, alter, regenerate,
or commit the packed artifact.

Stop on a missing source, hash mismatch, non-empty conflicting destination, or
incorrect pinned backend revision. Missing material that cannot be recovered
from the documented source is a real blocker and must be escalated once with
the exact failed check.
