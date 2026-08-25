# QAQ objective execution and verification rules

The filename is retained for compatibility; this rule governs objectives and work items.

This file owns current-objective scope, evidence, implementation choices, testing,
documentation, and completion. It intentionally does not create a second
FirstMate lifecycle.

## Current-objective contract

For the current objective, establish from repository evidence:

- the work-item name, intended outcome, and prerequisite;
- the active worktree, feature branch, destination branch, and delivery mode;
- exact in-scope areas and non-goals;
- behavior that must remain unchanged;
- required tests, fixed inputs, seeds, identities, and result paths; and
- whether a bounded real GPU or external-artifact run is part of the authorized
  objective.

Then execute the whole bounded objective-delivery cycle. Do not begin a follow-up objective.

## Known facts and worker choices

Separate known facts, material unknowns, and assumptions that can affect the
scientific claim, scope, acceptance criteria, reproducibility, external cost, or
safety. Escalate a material unknown only when repository inspection and one
bounded reversible correction cannot resolve it.

Everything else is a worker choice. Prefer the smallest reversible design that
matches existing code and work-item documents. Cover it with tests and continue.
Do not record routine mechanics in `docs/DECISIONS.md`; reserve that file for
durable research, protocol, or interface decisions.

## Execution and correction

Use this sequence without returning for routine permission:

1. verify environment, worktree identity, destination, and prerequisites;
2. implement the smallest coherent work-item change;
3. run focused tests;
4. run relevant integration, smoke, audit, and regression checks;
5. diagnose ordinary failures and make bounded in-scope corrections;
6. rerun affected checks;
7. update durable documentation when verified state changed;
8. commit; and
9. finish the active FirstMate delivery contract or the direct-Pi PR fallback.

Continue while checks pass or a bounded correction preserves the objective.
Revise automatically for ordinary code, test, formatting, documentation, or
clean private-branch integration failures. Use a blocked or needs-decision
status only at the material boundary in `AGENTS.md`.

Do not stop after implementation, validation, commit, worker recovery,
destination movement, branch refresh, feature-branch push, PR creation, or PR
update. Those are internal phases of one authorized objective delivery.

## Verification

Use repository-supported commands in this order when applicable:

1. focused tests;
2. relevant integration tests;
3. smoke check;
4. state or output audit;
5. regression tests for preserved behavior; and
6. formatting or static checks for changed code.

Run `.pi/rules/qaq-runtime.md` in each fresh shell. Report exact commands, exit
statuses, and relevant evidence. Do not claim success for work merely reported
by another process; verify it from repository or delivery-system evidence.

A failed check is not automatically a captain decision. Diagnose it, correct it
within scope, and rerun. Escalate only when the correction would cross the
material boundary, the required resource is unavailable, or repeated attempts
produce no new evidence.

## Documentation and results

Update `docs/STATUS.md` when the verified durable objective state changes. Update
`docs/DECISIONS.md` only for a material durable choice. Preserve exact result
paths and never overwrite frozen or canonical evidence without explicit
current authorization.

Structural tests are not experimental quality evidence. State remaining
research unknowns without turning them into implementation blockers when the
work item does not require resolving them.

## Completion

The work item is ready to report only when:

- the authorized current objective is met;
- required tests and audits passed or an exact material blocker is preserved;
- changed paths remain in scope;
- no unexplained worktree changes remain;
- work is committed; and
- the active delivery contract has reached PR-ready, checks-green, or
  local-branch-ready completion.

Return a concise final report with the objective, final commit, PR or branch, changed
paths, test evidence, GPU status, documentation changes, remaining unknowns,
and any next captain-level decision. Do not append artificial build-result or
landing-result stop markers.
