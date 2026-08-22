# QAQ stage execution and verification rules

This file is authoritative for stage identity, scope, evidence, implementation planning, tests, documentation, completion criteria, and the implementation completion report. Git history and landing are governed by `.pi/rules/qaq-git-worktrees.md`.

## Stage identity and scope

For every implementation or revision operation, state:

- state: `READY_TO_BUILD`, `REVISE`, `READY_TO_LAND`, `PAUSE`, or `COMPLETE`;
- operation: `IMPLEMENTATION`, `REVISION`, `LANDING`, `PAUSE`, or `COMPLETE`;
- stage identifier and title;
- controller root and controller rules commit;
- repository entry path and physical stage worktree path;
- feature branch and destination branch;
- prerequisite commit, or `NONE`;
- stage base commit;
- exact in-scope files or areas;
- exact non-goals.

Give one self-contained operation only. It may contain the complete bounded sequence of routine commands, checks, worker launches or relaunches, and authorized recovery needed to reach the next genuine decision gate. “One step” does not mean one command or one permission prompt. Do not include a later-stage preview.

## Known facts, unknowns, assumptions, and proposal

Separate:

- `KNOWN`: facts established by the user or repository evidence;
- `UNKNOWN`: facts not yet established that could change safety, scope, acceptance criteria, ancestry, or resources;
- `ASSUMPTIONS`: minimal temporary choices, each labeled `controller assumption` with a reason;
- `PROPOSED CURRENT STEP`: the one operation now authorized, labeled `controller proposal`.

Do not silently fill gaps or invent missing product requirements.

## Goal and preserved behavior

Choose the smallest coherent change that advances the documented QAQ goal and can be checked independently.

State:

- the single intended outcome;
- behavior that must remain unchanged;
- behavior intentionally changed;
- exact non-goals.

Base the operation on the user's instruction and repository evidence, including `AGENTS.md`, README files, design notes, stage records, source, tests, source notes, and papers required by the stage.

## Design and critical rules

Describe only what the current stage needs:

- affected components or files and their responsibilities;
- inputs and outputs;
- required error handling;
- observable conditions that must remain true;
- compatibility limits.

Label an unsupported implementation decision as `worker choice` and give a brief reason.

## Implementation sequence and decision points

Give a bounded sequence for the current stage only.

At each meaningful point:

- continue when the check passes;
- revise when one bounded correction preserves the goal;
- pause when scope, prerequisites, environment, GPU availability, repository safety, or acceptance criteria remain unresolved and no standing recovery rule applies;
- stop when the stage is committed and its completion report is returned.

Execute the whole bounded sequence without returning for permission between normal checks, test reruns, clean worker launches or relaunches, or other actions already covered by the current operation.

Do not revise, pause, recreate the worktree, or rebase solely because the destination branch advanced after the stage base was recorded. Continue the implementation on its existing branch and let the separate landing operation integrate it.

Rebase only when the stage genuinely requires destination changes to continue under `.pi/rules/qaq-git-worktrees.md`.

Do not pause solely because the prior worker window is missing. Apply `.pi/rules/qaq-worker-sessions.md`; when its **Recovery gate** passes, recover the session in the existing worktree and continue the same operation.

Do not include landing commands or later-stage work in an implementation or revision operation.

## Tests and verification

Specify exact repository-supported commands in this order when applicable:

1. focused tests;
2. relevant integration tests;
3. smoke check;
4. state or output audit;
5. regression tests for preserved behavior.

In every fresh shell, complete `.pi/rules/qaq-runtime.md` before project commands. Run its GPU gate first for GPU-dependent checks.

Report each command, exit status, and relevant output.

If an expected command does not exist, use `REVISE` or `PAUSE` instead of claiming success.

Treat work completed in another tool, worker, or session as unverified until evidence or repository inspection confirms it.

## Documentation and result handling

Update documentation only when the stage changes behavior, interfaces, setup, or recorded decisions.

Record:

- implementation status;
- decision sources;
- experiments;
- exact result paths;
- reproducibility details;
- whether generated output is safe to commit.

Do not overwrite existing results without explicit authorization.

## Stage completion gate

Mark implementation complete only when:

- the authorized goal is met;
- required tests passed and are reported;
- changed paths remain in scope;
- no unexplained worktree changes remain;
- the feature branch is committed;
- the final stage commit is a descendant of the recorded stage base;
- the current destination is also a descendant of the recorded stage base;
- the destination contains every prerequisite;
- branch, base, prerequisite, destination, commit range, and landing method are recorded;
- the topology report is complete.

The current destination does not need to be an ancestor of the final stage commit. Destination movement is integrated during landing.

Use `REVISE` for one bounded correction to the stage work.

Use `PAUSE` for missing evidence, a failed prerequisite, unsafe state, unavailable required GPU, unauthorized scope, or uncertain ownership. Do not use `PAUSE` merely for destination movement that the landing rules can handle.

Use `STOP` after returning the required report.

## Implementation completion report

Populate topology fields from `.pi/rules/qaq-git-worktrees.md` and return:

```text
Stage:
Controller root:
Controller rules commit:
Feature branch:
Physical stage worktree path:
Repository root:
Starting commit:
Stage base commit:
Worktree start: CREATED_AT_VERIFIED_BASE | REUSED_EXISTING_AUTHORIZED_WORKTREE
Worker session recovery: NOT_REQUIRED | RELAUNCHED_EXISTING_WORKTREE
Branch rebase: NOT_REQUIRED | REQUIRED_FOR_STAGE_DEPENDENCY
Pre-rebase feature HEAD: NONE or <commit>
Post-rebase feature HEAD: NONE or <commit>
Feature branch pushed or shared: no/yes/unknown
Prerequisite commit:
Destination branch:
Destination HEAD at stage start:
Destination HEAD at report:
Destination advanced since stage start: yes/no
Stage base is ancestor of final stage commit: yes/no
Stage base is ancestor of destination HEAD: yes/no
Destination HEAD is ancestor of final stage commit: yes/no
Final stage commit:
Commit range to land:
Commit count:
Ordered commits in range:
Changed paths:
Destination changes since stage base:
Destination contains prerequisite: yes/no/not-applicable
Tests and exit statuses:
GPU check: NOT_REQUIRED or result summary
Recommended landing method: FAST_FORWARD | MERGE_COMMIT
Recommended landing order:
Independently landable: yes/no
Landing classification: LANDABLE_DIRECTLY | PREREQUISITE_MUST_LAND_FIRST | STACKED_LANDING_REQUIRED | NOT_LANDABLE
Unexpected changes remaining:
```

End implementation with:

```text
HARD STOP: Complete only this implementation stage. Do not land, push, create the next branch or worktree, or begin later work. Return the topology report and wait for the user.
WAITING_FOR_BUILD_RESULT
```
