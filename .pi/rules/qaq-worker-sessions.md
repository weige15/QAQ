# QAQ worker-session and recovery rules

This file is authoritative for launching, relaunching, attaching, and recovering Pi/Firstmate worker processes, terminals, and windows. A worker session is temporary execution state; it is not a branch, worktree, commit, or authorization boundary. Git and worktree state remain governed by `.pi/rules/qaq-git-worktrees.md`.

## Standing authorization

An authorized `IMPLEMENTATION` or `REVISION` operation already permits:

- launching the write-enabled worker for the current stage;
- relaunching it after a crash, exit, missing terminal, or missing window;
- recreating temporary session or window metadata and binding the replacement to the exact recorded worktree;
- running read-only identity checks and the required runtime preflight;
- resuming the same authorized edits, tests, commit, and report.

These actions do not require another user confirmation when the recovery gate below passes.

The controller's one-step rule limits stage scope and later-stage disclosure. It does not limit the operation to one shell command, one worker launch, or one user round-trip.

## Recovery gate

Before relaunching or reattaching, establish all of the following without changing repository state:

- the recorded physical worktree still exists;
- `pwd -P` and `git rev-parse --show-toplevel` resolve to the recorded paths;
- the checked-out feature branch and `HEAD` match the recorded current-stage values;
- no other live write-enabled worker is attached to that worktree;
- no unexpected rebase, merge, cherry-pick, revert, or other Git operation is in progress;
- the status is clean, or every existing change can be identified as current-stage work inside the authorized paths and recovery will preserve it unchanged;
- recovery will not delete, reset, clean, stash, rebase, replace, or recreate the preserved worktree, branch, commits, or files.

The ordinary recovery case is: the old window is gone, the recorded worktree still exists, the branch and `HEAD` match, the status is clean, and no live writer remains. Relaunch in place immediately. Preserved commits make this case safer; they are not a reason to pause.

When every condition passes:

- use `CONTINUE` or `REVISE`, not `PAUSE`;
- create a fresh worker session or window identifier and bind it to the exact existing worktree;
- run `.pi/rules/qaq-runtime.md` before project commands;
- continue the same authorized operation;
- preserve all commits and any recorded in-scope uncommitted changes;
- report the recovery in the next normal result rather than requesting permission first.

A missing worker window alone is never a reason to recreate or delete the worktree or branch.

## Pause boundary

Use `PAUSE` only when at least one recovery condition cannot be established or recovery would require new authority. Examples include:

- the recorded worktree is missing;
- the repository root, branch, or `HEAD` differs from the recorded state;
- another write-enabled worker may still be active;
- an unexpected Git operation or unexplained out-of-scope change exists;
- the launcher can proceed only by deleting, overwriting, resetting, stashing, rebasing, or replacing preserved work;
- credentials, external resources, scope, acceptance criteria, landing, or pushing require authorization not already granted.

Do not request permission for read-only inspection, a replacement terminal or window, a clean relaunch, a runtime preflight, or resuming the same operation after the recovery gate passes.
