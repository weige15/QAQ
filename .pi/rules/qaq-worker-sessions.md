# QAQ worker-session and recovery rules

This file is authoritative for launching, relaunching, attaching, and recovering Pi/Firstmate worker processes, terminals, and windows. A worker session is temporary execution state; it is not a branch, worktree, commit, or authorization boundary. Git and worktree state remain governed by `.pi/rules/qaq-git-worktrees.md`.

## Controller source on launch and recovery

Record two separate paths:

- `<CONTROLLER_ROOT>`: the physical worktree that has the destination branch checked out and supplies the current `.pi/prompts/` and `.pi/rules/` files;
- `<STAGE_WORKTREE>`: the physical worktree that has the current feature branch checked out and supplies stage source, tests, documentation, and changes.

Launch and recovery must load the controller entrypoint and mandatory rules from the current `<CONTROLLER_ROOT>`, while setting the worker's project working directory to `<STAGE_WORKTREE>`.

A feature branch may contain an older snapshot of `.pi/`. That snapshot is not the active controller. Do not rebase, merge, recreate, or modify the feature branch merely to obtain controller updates.

On every launch or recovery:

1. identify the destination worktree and record its `HEAD` as the controller rules commit;
2. reread the controller entrypoint and mandatory rules from that worktree;
3. verify the stage worktree identity and state;
4. launch or resume the worker in the stage worktree.

## Standing authorization

An authorized `IMPLEMENTATION` or `REVISION` operation already permits:

- launching the write-enabled worker for the current stage;
- relaunching it after a crash, exit, missing terminal, or missing window;
- recreating temporary session or window metadata and binding the replacement to the exact recorded stage worktree;
- reading the current controller files from the destination worktree;
- running read-only identity checks and the required runtime preflight;
- resuming the same authorized edits, tests, commit, and report.

These actions do not require another user confirmation when the recovery gate below passes.

The controller's one-step rule limits stage scope and later-stage disclosure. It does not limit the operation to one shell command, one worker launch, or one user round-trip.

## Recovery gate

Before relaunching or reattaching, establish all of the following without changing repository state:

- the recorded controller root and stage worktree still exist;
- the controller root still has the recorded destination branch checked out;
- `pwd -P` and `git rev-parse --show-toplevel` in the stage worktree resolve to the recorded paths;
- the checked-out feature branch and `HEAD` match the recorded current-stage values;
- no other live write-enabled worker is attached to that stage worktree;
- no unexpected rebase, merge, cherry-pick, revert, or other Git operation is in progress;
- the status is clean, or every existing change can be identified as current-stage work inside the authorized paths and recovery will preserve it unchanged;
- recovery will not delete, reset, clean, stash, rebase, replace, or recreate the preserved worktree, branch, commits, or files.

The ordinary recovery case is: the old window is gone, both recorded worktrees still exist, the feature branch and `HEAD` match, the status is clean, and no live writer remains. Reload the controller from the destination worktree and relaunch in place immediately. Preserved commits make this case safer; they are not a reason to pause.

When every condition passes:

- use `CONTINUE` or `REVISE`, not `PAUSE`;
- create a fresh worker session or window identifier;
- load current controller files from the destination worktree;
- bind the worker to the exact existing stage worktree;
- run `.pi/rules/qaq-runtime.md` before project commands;
- continue the same authorized operation;
- preserve all commits and any recorded in-scope uncommitted changes;
- report the recovery and controller rules commit in the next normal result rather than requesting permission first.

A missing worker window alone is never a reason to recreate or delete the worktree or branch.

## Pause boundary

Use `PAUSE` only when at least one recovery condition cannot be established or recovery would require new authority. Examples include:

- the controller root or stage worktree is missing;
- the destination branch, repository root, feature branch, or feature `HEAD` differs from the recorded state in an unexplained way;
- another write-enabled worker may still be active;
- an unexpected Git operation or unexplained out-of-scope change exists;
- the launcher can proceed only by deleting, overwriting, resetting, stashing, rebasing, or replacing preserved work;
- credentials, external resources, scope, acceptance criteria, landing, or pushing require authorization not already granted.

Do not request permission for read-only inspection, loading current controller rules, a replacement terminal or window, a clean relaunch, a runtime preflight, or resuming the same operation after the recovery gate passes.
