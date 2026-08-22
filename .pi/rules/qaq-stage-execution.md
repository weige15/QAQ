# QAQ stage execution and verification rules

This file is authoritative for stage identity, scope, evidence, implementation planning, tests, documentation, completion criteria, and the implementation completion report. It does not authorize Git history changes; those are governed by `.pi/rules/qaq-git-worktrees.md`.

## Stage identity and scope

For every implementation or revision step, state:

- state: `READY_TO_BUILD`, `REVISE`, `READY_TO_LAND`, `PAUSE`, or `COMPLETE`;
- operation: `IMPLEMENTATION`, `REVISION`, `LANDING`, `PAUSE`, or `COMPLETE`;
- stage identifier and title;
- repository entry path and physical worktree path;
- feature branch and destination branch;
- prerequisite commit, or `NONE`;
- expected base commit;
- exact in-scope files or areas;
- exact non-goals.

Give one self-contained operation only. It may contain the complete bounded sequence of routine commands, checks, worker launches or relaunches, and authorized recovery needed to reach the next genuine decision gate. “One step” does not mean one command or one permission prompt. Do not include a later-stage preview.

## Known facts, unknowns, assumptions, and proposal

Separate:

- `KNOWN`: facts established by the user or repository evidence;
- `UNKNOWN`: facts not yet established that could change safety, scope, acceptance criteria, ancestry, or resources;
- `ASSUMPTIONS`: minimal temporary choices, each labeled `controller assumption` with a reason;
- `PROPOSED CURRENT STEP`: the one action now authorized, labeled `controller proposal`.

Do not silently fill gaps or invent missing product requirements.

## Goal and preserved behavior

Choose the smallest coherent change that advances the documented QAQ goal and can be checked independently.

State:

- the single intended outcome;
- behavior that must remain unchanged;
- behavior intentionally changed;
- exact non-goals.

Base the step on the user's instruction and repository evidence, including `AGENTS.md`, README files, design notes, stage records, source, tests, source notes, and papers required by the stage.

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

At each meaningful point, state:

- continue when the check passes;
- revise when one bounded correction preserves the goal, including a permitted feature-branch refresh after the destination advances;
- pause when scope, ancestry, environment, GPU availability, repository safety, or acceptance criteria remain unresolved and no standing recovery rule applies;
- stop when the stage is committed and its completion report is returned.

Execute the whole bounded sequence without returning for permission between normal checks, test reruns, clean worker launches or relaunches, or other actions already covered by the current operation.

Do not pause solely to request rebase authorization when every condition in `.pi/rules/qaq-git-worktrees.md` under **Bounded feature-branch refresh** passes.

Do not pause solely because the prior worker window is missing. Apply `.pi/rules/qaq-worker-sessions.md`; when its **Recovery gate** passes, recover the session in the existing worktree and continue the same operation.

Do not include landing commands or later-stage work in an implementation or revision step.

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
- any worktree recreation or feature-branch refresh satisfied `.pi/rules/qaq-git-worktrees.md` and is reported;
- the final stage commit is a descendant of the destination `HEAD` recorded in the report;
- branch, base, prerequisite, and destination facts are recorded;
- the topology report is complete.

Use `REVISE` for one bounded correction.

Use `PAUSE` for missing evidence, a failed prerequisite, unsafe state, unavailable required GPU, unauthorized scope, failed refresh conditions, or uncertain landing.

Use `STOP` after returning the required report.

## Implementation completion report

Populate topology fields from `.pi/rules/qaq-git-worktrees.md` and return:

```text
Stage:
Feature branch:
Physical worktree path:
Repository root:
Starting commit:
Stage base commit:
Worktree start: CREATED_AT_VERIFIED_BASE | REUSED_EXISTING_AUTHORIZED_WORKTREE
Branch refresh: NOT_REQUIRED | RECREATED_AT_DESTINATION | REBASED
Worker session recovery: NOT_REQUIRED | RELAUNCHED_EXISTING_WORKTREE
Pre-refresh feature HEAD: NONE or <commit>
Post-refresh feature HEAD: NONE or <commit>
Feature branch pushed or shared: no/yes/unknown
Prerequisite commit:
Destination branch:
Destination HEAD:
Destination is ancestor of final stage commit: yes/no
Final stage commit:
Commit range to land:
Commit count:
Ordered commits in range:
Changed paths:
Destination contains prerequisite: yes/no/not-applicable
Tests and exit statuses:
GPU check: NOT_REQUIRED or result summary
Recommended landing order:
Recommended landing method:
Independently landable: yes/no
Landing classification: LANDABLE_DIRECTLY | PREREQUISITE_MUST_LAND_FIRST | STACKED_LANDING_REQUIRED | NOT_LANDABLE
Unexpected changes remaining:
```

End implementation with:

```text
HARD STOP: Complete only this implementation stage. Do not land, push, create the next branch or worktree, or begin later work. Return the topology report and wait for the user.
WAITING_FOR_BUILD_RESULT
```
