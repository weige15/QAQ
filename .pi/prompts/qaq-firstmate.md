---
description: Complete the current QAQ stage end-to-end
argument-hint: "[stage or task]"
---

Complete `${ARGUMENTS:-the current QAQ stage identified in docs/STATUS.md}`
end-to-end.

Use `AGENTS.md` as the repository-wide operating authority.

Before changing anything, establish the current state from:

- `docs/STATUS.md`;
- `docs/DECISIONS.md`;
- the applicable document under `docs/stages/`;
- the source, configuration, tests, papers, and artifacts required by that
  stage; and
- the active FirstMate task brief when one exists.

When FirstMate launched this worker, its active brief remains the controller
for worktree, branch, status reporting, delivery mode, and merge authority.
Do not create a competing orchestration or permission process.

Determine the intended stage outcome, acceptance conditions, preserved
behavior, non-goals, and currently unresolved questions from the authoritative
repository evidence.

Then complete the whole authorized current-stage cycle.

Continue through routine implementation, testing, bounded corrections,
retries, documentation updates, commits, branch delivery, and PR updates
without asking for permission between those actions.

Pause and report only when the material-decision boundary in `AGENTS.md` is
reached or when repository evidence proves that the current stage cannot
proceed safely.

Do not begin a later stage automatically.

At completion, report concisely:

- the stage and achieved outcome;
- the final commit and PR or local branch;
- materially changed paths;
- verification commands and outcomes;
- real execution result, or `NOT_REQUIRED`;
- durable status or decision changes;
- remaining unknowns; and
- any next action that requires new authorization.
