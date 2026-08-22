# QAQ Git, worktree, branch-refresh, and landing rules

This file is authoritative for repository topology, worktrees, feature branches, prerequisites, rebasing, commits, landing, and pushing. No other controller file should redefine when a rebase, landing, or push is allowed.

## Authorization boundaries

Implementation authorization permits only the current stage's feature-worktree operations:

- create and use the authorized feature worktree;
- perform the safe cleanup or recreation defined below;
- perform one bounded feature-branch refresh when every condition below passes;
- edit, test, explicitly stage, and commit authorized paths.

Implementation authorization does not permit:

- moving the destination branch;
- merging or cherry-picking into the destination branch;
- resetting or force-updating any branch;
- rebasing outside **Bounded feature-branch refresh**;
- pushing;
- starting the next stage.

When explicitly asked to execute through Firstmate, allow only one write-enabled implementation worktree. Read-only scouts may inspect, but they may not edit, land, or start another stage.

Never assume the destination branch is `main`; determine it from the user or repository policy.

Never force-push. Never push without explicit authorization for that exact push.

## Repository and prerequisite gate

Before authorizing or creating a stage branch or worktree, run:

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
- the current branch;
- the starting `HEAD`;
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

Do not bypass this gate by stacking, partial cherry-picking, changing destination, or assuming the prerequisite will land later. A bounded feature-branch refresh is not a bypass because the destination must already contain the prerequisite.

A stacked stage is allowed only with this complete user authorization:

```text
STACKED_STAGE_AUTHORIZATION: YES
BASE_COMMIT: <exact prerequisite commit>
DESTINATION_MISSING_PREREQUISITE_ACKNOWLEDGED: YES
REQUIRED_LANDING_ORDER:
1. <prerequisite commit or range>
2. <current-stage commit or range>
```

The worker must remain in its recorded physical worktree and branch. If the worktree, repository root, branch, or `HEAD` changes unexpectedly, use `PAUSE` and report the exact before-and-after values.

Never treat “the commit exists” as “the destination contains the commit.”

## Fresh worktree startup

Use an unused feature branch name and an unused worktree path. Create the new branch directly at the exact verified base:

```bash
git worktree add -b <FEATURE_BRANCH> <WORKTREE_PATH> <STAGE_BASE>
```

This creation is part of implementation authorization. Do not ask for rebase authorization merely to start the worktree.

Do not use the current checkout `HEAD` or an earlier feature branch as an implicit base.

Enter the new worktree and run the complete preflight from `.pi/rules/qaq-runtime.md`. Continue only if:

- the physical path and branch match the authorization;
- `HEAD` equals `<STAGE_BASE>`;
- the status is clean.

If the proposed branch name or path already exists, inspect it instead of assuming it is reusable. Do not reset, rebase, or overwrite existing work merely to keep the proposed name. Choose a new unique branch name or path for the same stage and record the actual values.

Firstmate may remove and recreate only a worktree and branch that it created during the current attempt, and only when all of these are true:

- the worktree is clean;
- the branch has no commit outside `<STAGE_BASE>`;
- the branch was never pushed or shared;
- non-force worktree removal succeeds;
- `git branch -d` succeeds.

This safe cleanup and recreation is part of implementation authorization. Otherwise use `PAUSE` and preserve the existing work.

## Bounded feature-branch refresh

Refresh only when the verified destination `HEAD` advanced after `<STAGE_BASE>` was recorded.

Before any refresh, run:

```bash
git status --short
git rev-parse HEAD
git rev-parse <DESTINATION_BRANCH>
git merge-base --is-ancestor <STAGE_BASE> <FEATURE_BRANCH>
git merge-base --is-ancestor <STAGE_BASE> <DESTINATION_BRANCH>
git rev-list --count <STAGE_BASE>..<FEATURE_BRANCH>
git log --oneline --reverse <STAGE_BASE>..<FEATURE_BRANCH>
git rev-list --merges <STAGE_BASE>..<FEATURE_BRANCH>
git diff --name-only <STAGE_BASE>..<FEATURE_BRANCH>
```

If the feature branch has no stage commit and the worktree is clean, recreate it at the new destination `HEAD` under **Fresh worktree startup**. Do not run an empty rebase.

If the feature branch has stage commits, one rebase onto the new destination `HEAD` is already authorized without another user round-trip only when every condition below is established:

- the current physical worktree and branch are the recorded current-stage worktree and feature branch;
- the worktree is clean;
- no rebase, merge, cherry-pick, or revert is already in progress;
- `<STAGE_BASE>` is an ancestor of both the feature branch and the destination branch;
- the destination contains the prerequisite;
- every commit in `<STAGE_BASE>..<FEATURE_BRANCH>` belongs only to the current stage;
- the range contains no merge commit;
- all changed paths remain in scope;
- the feature branch was created for the current stage;
- the feature branch was never pushed or shared.

Record the old destination `HEAD` and old feature `HEAD`, then run:

```bash
git rebase <NEW_DESTINATION_HEAD>
```

If a conflict or unexpected change occurs, run:

```bash
git rebase --abort
```

Continue only if abort restores the recorded feature `HEAD` and a clean status. Otherwise use `PAUSE` and preserve all evidence.

After a successful rebase, run:

```bash
git merge-base --is-ancestor <NEW_DESTINATION_HEAD> HEAD
git range-diff <STAGE_BASE>..<OLD_FEATURE_HEAD> <NEW_DESTINATION_HEAD>..HEAD
git diff --name-only <NEW_DESTINATION_HEAD>..HEAD
git status --short
```

Continue only when:

- the new branch is based on the exact destination `HEAD`;
- the range-diff shows the intended stage commits were replayed without unexplained changes;
- changed paths remain in scope;
- the status is clean.

Rerun all required stage tests before reporting completion.

Do not ask “may I rebase?” when every condition in this section passes. The current implementation or revision authorization already covers this one feature-branch refresh.

Never use this standing authorization to rebase:

- the destination branch;
- a pushed or shared branch;
- a branch containing other work;
- a commit range with unresolved ownership.

If the destination advances after the topology report but before landing, use `REVISE` and issue one bounded refresh step when these conditions pass. Do not use `PAUSE` solely to request rebase permission. Regenerate the topology report after the refresh and tests.

## Commit and topology checks

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
git merge-base --is-ancestor <DESTINATION_BRANCH> <STAGE_COMMIT>
git merge-base --is-ancestor <PREREQUISITE_COMMIT> <DESTINATION_BRANCH>
git rev-list --count <DESTINATION_BRANCH>..<STAGE_COMMIT>
git log --oneline --reverse <DESTINATION_BRANCH>..<STAGE_COMMIT>
git diff --name-only <DESTINATION_BRANCH>..<STAGE_COMMIT>
git status --short
```

Skip only the prerequisite check when the prerequisite is `NONE`, and mark it not applicable.

Return the collected topology facts through the implementation completion report in `.pi/rules/qaq-stage-execution.md`.

Classify landing as follows:

- `LANDABLE_DIRECTLY`: the destination contains every prerequisite, the destination `HEAD` is an ancestor of the final stage commit, the commit range and paths are understood, tests passed, and no unexplained change remains.
- `PREREQUISITE_MUST_LAND_FIRST`: the required prerequisite is not contained in the destination.
- `STACKED_LANDING_REQUIRED`: the user explicitly authorized a stacked stage and the required landing order must be preserved.
- `NOT_LANDABLE`: any other safety, scope, ancestry, ownership, verification, or cleanliness condition fails.

## Separate landing step

When implementation is reported complete, verify the report and repository state when tools permit.

- Use `REVISION` when one bounded correction is needed.
- When the destination advanced and **Bounded feature-branch refresh** conditions pass, issue that refresh as `REVISION`; do not request separate rebase authorization.
- Use `PAUSE` when evidence is missing, refresh conditions fail, or the stage is otherwise unsafe or not landable.
- Issue one `LANDING` step only when the stage is `LANDABLE_DIRECTLY`.
- Never issue the next implementation step before landing is verified.

A landing step must state:

- the source branch and commit;
- the destination worktree and branch;
- the pre-landing destination `HEAD`;
- the prerequisite result;
- the exact commit range and count;
- the changed paths;
- the one authorized operation;
- the post-landing checks;
- the no-push boundary.

Enter the worktree that already has the destination branch checked out and run the complete preflight from `.pi/rules/qaq-runtime.md`. Do not silently check the destination branch out elsewhere.

Default landing policy: authorize only a fast-forward when the destination contains all prerequisites and the stage commit is a valid descendant.

If the destination advanced, return to **Bounded feature-branch refresh** before landing.

In the destination worktree, do not perform a merge commit, cherry-pick, rebase, reset, force operation, or push. A feature-branch rebase is allowed only before `READY_TO_LAND` under **Bounded feature-branch refresh**. Any alternative landing method requires explicit approval for that exact operation and must not violate the entrypoint's non-negotiable restrictions.

Normal landing command:

```bash
git merge --ff-only <STAGE_COMMIT>
```

Use it only after verifying the destination worktree, branch, expected `HEAD`, prerequisite containment, and fast-forward relationship.

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
Destination worktree:
Destination branch:
Destination HEAD before landing:
Authorized operation:
Destination HEAD after landing:
Stage commit contained in destination: yes/no
Prerequisite contained in destination: yes/no/not-applicable
Post-landing tests and exit statuses:
GPU check: NOT_REQUIRED or result summary
Destination status:
Push performed: no
Landing result: LANDED_AND_VERIFIED | REVISE | PAUSE
```

End the landing step with:

```text
HARD STOP: Perform only the authorized landing and its checks. Do not push, delete branches or worktrees, or begin the next stage. Return the landing result and wait for the user.
WAITING_FOR_LANDING_RESULT
```

Give the next implementation step only after post-landing checks pass and this succeeds:

```bash
git merge-base --is-ancestor <PASSING_STAGE_COMMIT> <DESTINATION_BRANCH>
```
