# QAQ contribution rules

This file is authoritative for repository-wide rules that should apply across
future QAQ agent sessions.

Task-specific instructions belong in the active task brief or user request.
Current project state belongs in `docs/STATUS.md`.
Work-item-specific requirements, acceptance criteria, and scientific boundaries
belong in `docs/stages/` (the directory name is retained for compatibility).

## Authority and current-objective work

When a worker is launched by FirstMate, the active FirstMate task brief owns:

- worktree identity;
- branch naming;
- status reporting;
- delivery mode; and
- merge authority.

Do not create a competing orchestration, worktree, landing, or permission
process inside QAQ.

When no FirstMate brief exists, follow the user's active request and these
repository rules. For implementation work, use an isolated feature branch
unless the user explicitly requests local-only work. Never merge the default
branch or force-push without explicit current authorization.

A request to implement or complete the current QAQ objective authorizes one
complete current-objective delivery cycle:

1. inspect the current state and required prerequisites;
2. make reversible implementation choices inside the documented objective scope;
3. edit code, tests, configuration, and documentation required by the work item;
4. run focused checks and the relevant broader checks;
5. diagnose ordinary failures, make bounded corrections, and retry;
6. update durable project records when verified state or a material decision
   changed;
7. commit the completed work-item changes; and
8. push or open/update a PR when required by the active delivery contract.

Do not request permission between those routine actions.

Continue while the work remains inside the current objective, is reversible, and
preserves its documented scientific meaning, acceptance criteria, and
non-goals.

Do not automatically begin a follow-up objective.

## Material decision boundary

Escalate only when proceeding would require at least one of the following:

- changing a scientific claim, frozen protocol, acceptance threshold,
  current-objective scope, non-goal, or preserved behavior;
- beginning a follow-up objective;
- discarding or overwriting meaningful work;
- performing an irreversible external action;
- rewriting shared history, force-pushing, or merging the default branch;
- using credentials or unavailable external material that cannot be recovered
  through documented means;
- accepting materially larger or unbounded compute or external cost;
- performing a real execution that the current objective or active task brief does
  not already authorize with bounded inputs and scope; or
- choosing between conflicting requirements when the choice would materially
  change the result.

Unknowns that do not affect safety, scientific interpretation, acceptance
criteria, reproducibility, external cost, or public behavior are worker
choices.

For such routine choices, use the smallest coherent reversible option, cover
it with appropriate tests, and continue.

Routine naming, file placement within scope, test selection, local
refactoring, ordinary dependency inspection, retry strategy, shell/session
recovery, status wording, and similar mechanics are not captain decisions.

For GPU or other expensive execution, an objective request authorizes the real run
only when the current work-item document or active task brief identifies the run,
its fixed inputs, and a bounded execution scope. Otherwise treat execution as
a material decision.

## Mandatory environment

Before implementation, testing, Python inspection, dependency work, or project
execution in each fresh shell, run:

```bash
source ~/.venv/bin/activate
which python
python --version
