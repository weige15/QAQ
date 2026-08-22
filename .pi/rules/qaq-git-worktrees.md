# QAQ Git, worktree, and landing rules

This file is authoritative for repository topology, worktrees, feature branches, prerequisites, commits, landing, and pushing.

## Operating model

A worktree is a temporary checkout. A feature branch is the record of the current stage. Neither becomes invalid merely because the destination branch moves.

Record `<STAGE_BASE>` when the stage starts and keep it fixed for that implementation. Later destination commits do not change the stage goal, erase the feature branch, or require a rebase. Integration with the current destination is handled during the separate landing operation.

Preserve committed work. Never reset, overwrite, delete, or recreate a valid current-stage branch or worktree merely to make commit ancestry look linear.

## Controller root and stage worktree

Keep control state separate from stage code:

- `<CONTROLLER_ROOT>` is the physical worktree that has the destination branch checked out. Its current `.pi/prompts/` and `.pi/rules/` files are authoritative for controller behavior.
- `<STAGE_WORKTREE>` is the physical worktree that has the feature branch checked out. Its source, tests, stage documents, and stage changes are authoritative for implementation.
- Copies of `.pi/` inside an older feature branch are snapshots, not the active controller source.

At the start of every operation and after every worker recovery, read the controller prompt and rules from `<CONTROLLER_ROOT>`, then execute project commands in `<STAGE_WORKTREE>` when the operation is implementation or revision.

Do not merge, rebase, recreate, or modify the feature branch merely to obtain newer controller files. Controller updates on the destination branch change how the work is managed; they do not change the stage's code base.

## Authorization boundaries

Implementation or revision authorization permits only the current stage's feature-worktree operations:

- create or reuse the authorized feature worktree and branch;
- inspect the destination branch read-only;
- edit, test, explicitly stage, and commit authorized paths;
- perform the worker-session recovery allowed by `.pi/rules/qaq-worker-sessions.md`;
- perform a bounded feature-branch rebase only when the current stage actually needs destination changes to continue, as defined below.

Implementation authorization does not permit:

- moving the destination branch;
- landing into the destination branch;
- resetting or force-updating any branch;
- pushing;
- starting the next stage.

Landing and pushing remain separate operations. Never force-push. Never push without explicit authorization for that exact push.

When explicitly asked to execute through Firstmate, allow only one write-enabled implementation worktree. Read-only scouts may inspect, but they may not edit, land, or start another stage.

Never assume the destination branch is `main`; determine it from the user or repository policy.

## Repository and prerequisite gate

Before creating or authorizing a stage branch or worktree, run:

```bash
git worktree list --porcelain
pwd -P
git rev-parse --show-toplevel
git branch --show-current
git rev-parse HEAD
git status --short
git rev-parse <DESTINATION_BRANCH>
```

Record:

- the physical worktree;
- the repository root;
- the current branch and `HEAD`;
- the destination branch and its `HEAD`;
- the prerequisite commit, or `NONE`;
- whether the destination contains the prerequisite.

For a stage with a prerequisite, run:

```bash
git merge-base --is-ancestor <PREREQUISITE_COMMIT> <DESTINATION_BRANCH>
```

Continue only if it succeeds. Set `<STAGE_BASE>` to the exact verified destination `HEAD`.

For the first stage, record `Prerequisite: NONE`.

If the prerequisite check fails, use `PAUSE`. Report the destination branch, destination `HEAD`, missing prerequisite, its current branch or worktree if known, and the required landing order. Do not create the later-stage branch or worktree.

Do not bypass this gate by stacking, partial landing, changing destination, or assuming the prerequisite will land later. A stacked stage is allowed only with this complete user authorization:

```text
STACKED_STAGE_AUTHORIZATION: YES
BASE_COMMIT: <exact prerequisite commit>
DESTINATION_MISSING_PREREQUISITE_ACKNOWLEDGED: YES
REQUIRED_LANDING_ORDER:
1. <prerequisite commit or range>
2. <current-stage commit or range>
```

Never treat “the commit exists” as “the destination contains the commit.”

## Worktree startup and continuity

For a new stage, use an unused feature branch name and an unused worktree path:

```bash
git worktree add -b <FEATURE_BRANCH> <WORKTREE_PATH> <STAGE_BASE>
```

Enter the worktree and complete `.pi/rules/qaq-runtime.md`. Continue only if:

- the physical path and repository root match the authorization;
- the checked-out branch is `<FEATURE_BRANCH>`;
- `HEAD` equals `<STAGE_BASE>`;
- the status is clean.

If a proposed branch name or worktree path already exists, inspect it. Reuse it when it is the recorded current-stage branch and worktree and its state is understood. Otherwise choose a new unused name or path. Do not reset, overwrite, or delete preserved work to keep a proposed name.

The worker must remain in its recorded physical worktree and branch. If either identity changes unexpectedly, use `PAUSE` and report the before-and-after values.

A missing worker window is governed by `.pi/rules/qaq-worker-sessions.md`; it is not a reason to replace the branch or worktree.

## Destination movement during implementation

The destination branch may advance after `<STAGE_BASE>` is recorded. Check it read-only when needed:

```bash
git status --short
git rev-parse HEAD
git rev-parse <DESTINATION_BRANCH>
git merge-base --is-ancestor <STAGE_BASE> <FEATURE_BRANCH>
git merge-base --is-ancestor <STAGE_BASE> <DESTINATION_BRANCH>
git diff --name-only <STAGE_BASE>..<FEATURE_BRANCH>
git diff --name-only <STAGE_BASE>..<DESTINATION_BRANCH>
```

Continue the current implementation without rebasing when:

- the current worktree and feature branch are the recorded current-stage ones;
- `<STAGE_BASE>` remains an ancestor of the feature branch;
- `<STAGE_BASE>` remains an ancestor of the destination branch;
- the destination still contains the prerequisite;
- current-stage changes remain inside authorized paths;
- the worktree state is understood and safe.

Destination advancement by itself is not `REVISE`, `PAUSE`, or a reason to recreate the worktree. Record it and defer integration to landing.

A feature-branch rebase is exceptional. Use one only when the current stage genuinely requires code or interfaces added to the destination after `<STAGE_BASE>`, or when destination changes make the stage impossible to validate in isolation. Do not rebase merely to recover fast-forward ancestry or keep history linear.

When such a rebase is necessary, it is already authorized without another user round-trip only if:

- the current worktree and branch are the recorded current-stage ones;
- the worktree is clean;
- no Git operation is in progress;
- `<STAGE_BASE>` is an ancestor of both the feature branch and destination;
- the destination contains the prerequisite;
- every commit in `<STAGE_BASE>..<FEATURE_BRANCH>` belongs only to the current stage;
- the range contains no merge commit;
- all changed paths remain in scope;
- the feature branch was never pushed or shared.

Record the old feature `HEAD` and destination `HEAD`, then run:

```bash
git rebase <DESTINATION_HEAD>
```

If a conflict or unexpected change occurs, run `git rebase --abort`. Continue only if the abort restores the recorded feature `HEAD` and a clean status. After a successful rebase, rerun the required stage tests and record why the stage needed destination changes.

Never rebase the destination branch, a pushed or shared branch, a branch containing other work, or a range with unresolved ownership.

## Commit and implementation completion checks

Use explicit staging:

```bash
git add -- <AUTHORIZED_PATHS>
```

Do not stage unrelated paths.

At implementation completion, run:

```bash
git worktree list --porcelain
git branch --show-current
git rev-parse HEAD
git rev-parse <DESTINATION_BRANCH>
git merge-base --is-ancestor <STAGE_BASE> <STAGE_COMMIT>
git merge-base --is-ancestor <STAGE_BASE> <DESTINATION_BRANCH>
git merge-base --is-ancestor <PREREQUISITE_COMMIT> <DESTINATION_BRANCH>
git rev-list --count <STAGE_BASE>..<STAGE_COMMIT>
git log --oneline --reverse <STAGE_BASE>..<STAGE_COMMIT>
git rev-list --merges <STAGE_BASE>..<STAGE_COMMIT>
git diff --name-only <STAGE_BASE>..<STAGE_COMMIT>
git diff --name-only <STAGE_BASE>..<DESTINATION_BRANCH>
git status --short
```

Skip only the prerequisite check when the prerequisite is `NONE`, and mark it not applicable.

Implementation is ready to land when:

- the stage commit descends from `<STAGE_BASE>`;
- the current destination also descends from `<STAGE_BASE>`;
- the destination contains every prerequisite;
- the stage commit range belongs only to the current stage and contains no merge commit;
- changed paths remain in scope;
- required tests passed;
- no unexplained worktree change remains.

The current destination does not need to be an ancestor of the stage commit.

Choose the recommended landing method:

- `FAST_FORWARD` when the current destination is an ancestor of the stage commit;
- `MERGE_COMMIT` when both the destination and stage commit descend from `<STAGE_BASE>` but neither contains the other.

Classify landing as follows:

- `LANDABLE_DIRECTLY`: every readiness condition above passes;
- `PREREQUISITE_MUST_LAND_FIRST`: the destination does not contain a required prerequisite;
- `STACKED_LANDING_REQUIRED`: the user explicitly authorized a stacked stage and the required order must be preserved;
- `NOT_LANDABLE`: any other safety, scope, ownership, verification, or cleanliness condition fails.

Return the collected facts through the implementation completion report in `.pi/rules/qaq-stage-execution.md`.

## Separate landing operation

When implementation is complete, verify the report and current repository state.

Use `PAUSE` only when evidence is missing or the stage is not safely landable. Do not use `REVISE` or `PAUSE` merely because the destination advanced after `<STAGE_BASE>`.

A landing operation must state:

- the source branch and final stage commit;
- `<STAGE_BASE>`;
- the destination worktree and branch;
- the destination `HEAD` before landing;
- the prerequisite result;
- the exact stage commit range and count;
- the changed paths;
- the selected landing method;
- the post-landing checks;
- the no-push boundary.

Enter the worktree that already has the destination branch checked out and complete `.pi/rules/qaq-runtime.md`. Do not silently check the destination branch out elsewhere.

Recheck:

```bash
git status --short
git rev-parse <DESTINATION_BRANCH>
git merge-base --is-ancestor <STAGE_BASE> <STAGE_COMMIT>
git merge-base --is-ancestor <STAGE_BASE> <DESTINATION_BRANCH>
git merge-base --is-ancestor <PREREQUISITE_COMMIT> <DESTINATION_BRANCH>
```

Skip the prerequisite check only when it is `NONE`.

If the destination is an ancestor of the stage commit, land with:

```bash
git merge --ff-only <STAGE_COMMIT>
```

Otherwise, when both the destination and stage commit descend from `<STAGE_BASE>`, land with:

```bash
git merge --no-ff --no-edit <STAGE_COMMIT>
```

Do not rebase the feature branch merely to make the landing fast-forward.

If the merge conflicts, run:

```bash
git merge --abort
```

Use `REVISE` only when one bounded source correction can resolve the conflict while preserving the stage goal. Otherwise use `PAUSE`. Preserve both branches and all evidence.

After landing, run:

```bash
git rev-parse <DESTINATION_BRANCH>
git merge-base --is-ancestor <STAGE_COMMIT> <DESTINATION_BRANCH>
git status --short
```

Rerun the required stage checks on the destination. Run the GPU gate first only when those checks require GPU.

Return:

```text
Stage:
Source feature branch:
Source stage commit:
Stage base commit:
Destination worktree:
Destination branch:
Destination HEAD before landing:
Landing method: FAST_FORWARD | MERGE_COMMIT
Destination HEAD after landing:
Source stage commit contained in destination: yes/no
Prerequisite contained in destination: yes/no/not-applicable
Post-landing tests and exit statuses:
GPU check: NOT_REQUIRED or result summary
Destination status:
Push performed: no
Landing result: LANDED_AND_VERIFIED | REVISE | PAUSE
```

End the landing operation with:

```text
HARD STOP: Perform only the authorized landing and its checks. Do not push, delete branches or worktrees, or begin the next stage. Return the landing result and wait for the user.
WAITING_FOR_LANDING_RESULT
```

Give the next implementation operation only after post-landing checks pass and this succeeds:

```bash
git merge-base --is-ancestor <PASSING_STAGE_COMMIT> <DESTINATION_BRANCH>
```
