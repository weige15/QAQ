# QAQ Git, worktree, and delivery rules

This file defines project-specific safety boundaries. When a worker is launched
by FirstMate, the active task brief and FirstMate launcher own the worktree,
branch, status channel, delivery mode, PR flow, and merge authority. Do not
replace them with a second project-local landing controller.

## Standing authorization

Authorization to implement the current objective includes, without another captain
round-trip:

- using the assigned isolated worktree and feature branch;
- creating the feature branch when a direct Pi session has no FirstMate-created
  branch;
- inspecting destination and prerequisite commits read-only;
- editing, testing, explicitly staging, and committing in-scope paths;
- recovering the same preserved worktree after a worker/session failure;
- rebasing an unshared, clean feature branch only when current destination code
  is genuinely required to complete or validate the work item;
- pushing only the feature branch; and
- opening or updating the PR required by the active delivery contract.

These are routine delivery mechanics, not separate captain decisions.

## Hard boundaries

Never:

- edit from the primary checkout when an isolated worktree was assigned;
- push directly to the default branch;
- merge the default branch unless the active FirstMate merge authority performs
  it or the captain explicitly authorizes that exact merge;
- force-push or rewrite shared history;
- reset, clean, stash, overwrite, or delete preserved work to make topology look
  simpler; or
- begin a follow-up objective from the current task.

A destructive recovery, default-branch merge, force operation, or discarded
work is a material decision. Feature-branch push and PR update are not.

## Identity and prerequisite checks

At startup and recovery, verify:

```bash
pwd -P
git rev-parse --show-toplevel
git branch --show-current
git rev-parse HEAD
git status --short
git worktree list --porcelain
```

Use the active brief or repository policy to identify the destination branch;
do not assume its name. Verify every documented prerequisite is contained in
the destination before starting dependent implementation.

If a prerequisite is missing, report one material blocker with the exact
commits and required order. Do not invent a stacked objective. If all prerequisites
remain satisfied, continue without asking about routine topology details.

## Destination movement

A destination branch advancing does not invalidate a valid feature worktree or
objective base. Continue current-objective work when the feature branch, worktree,
prerequisites, and changed paths remain understood.

Refresh the feature branch only when destination changes are actually needed by
the work item. For an unshared clean branch, a safe rebase and required test rerun
are routine. For a pushed/shared branch, conflicts, uncertain ownership, or a
history rewrite requirement, preserve the work and escalate rather than force.

## Delivery

Commit only authorized paths and keep the worktree explainable. Complete the
active delivery contract:

- `no-mistakes`: commit, follow FirstMate's validation instruction, respond to
  evidence-based non-human gates, and deliver the PR with required checks green;
- `direct-PR`: commit, push the feature branch, and open or update the PR;
- `local-only`: commit the feature branch and report it ready for FirstMate's
  guarded local merge.

Do not create a separate implementation-to-landing permission gate. The worker
finishes at PR-ready or local-branch-ready delivery; FirstMate owns any later
merge decision and cleanup.
