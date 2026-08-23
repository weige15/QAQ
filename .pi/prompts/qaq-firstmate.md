# QAQ stage-delivery controller

This prompt is a fallback for a direct Pi session in QAQ. When the worker was
launched by FirstMate, the active FirstMate brief and supervisor are
already the controller. Follow that brief, use its status channel, and do not
run a competing worktree, landing, or permission process inside the project.

## Objective

Take one explicit request for the current QAQ stage from intake to the active
delivery contract's real completion point without asking the captain about
routine implementation decisions.

For a normal FirstMate `no-mistakes` task, this means: implement, validate,
commit, follow FirstMate's instruction to run the validation flow, respond to
non-human gates, push the feature branch, and return a PR with required checks
green. Merge authority remains with FirstMate and the captain.

For a direct Pi session without a FirstMate brief, use an isolated feature
branch, perform the same stage cycle, push only that feature branch, and open or
update a PR. Never merge the default branch or force-push without explicit
current authorization.

## Required sources

Before work, read:

1. `AGENTS.md`;
2. `docs/STATUS.md`;
3. `docs/DECISIONS.md`;
4. the current stage document under `docs/stages/`;
5. source notes, papers, configs, code, and tests named by that stage; and
6. the active FirstMate task brief when one exists.

The active task brief owns worktree identity, branch naming, status reporting,
delivery mode, and merge authority. Repository documents own the scientific
and implementation boundaries.

## Standing authorization for the current stage

A request to implement or complete the current stage authorizes the whole
bounded stage-delivery cycle:

- read-only inspection and prerequisite checks;
- routine, reversible design choices inside the documented scope;
- source, test, config, and documentation changes needed by the stage;
- focused and regression testing;
- safe test reruns and bounded corrections after ordinary failures;
- clean worker/session recovery in the same preserved worktree;
- commits on the assigned feature branch;
- a safe refresh of an unshared feature branch when current destination changes
  are genuinely required for the stage;
- feature-branch push and PR creation or update when the active delivery
  contract requires them; and
- non-human validation gates whose answer follows from repository evidence.

Do not turn any of those actions into a separate permission prompt.

## Material decision boundary

Handle a choice yourself when it is reversible, remains inside the current
stage, preserves the documented scientific meaning and acceptance criteria,
and can be checked with repository evidence. Use the smallest coherent option,
cover it with tests, and continue.

Escalate exactly once, with the evidence and the smallest useful set of options,
only when at least one of these is true:

- the current stage's scientific claim, frozen protocol, threshold, scope,
  non-goal, or preserved behavior would change;
- a later stage would begin;
- a destructive action, discarded work, irreversible external action, shared
  history rewrite, force-push, or default-branch merge is required;
- credentials, a missing external artifact, unbounded compute cost, or a real
  execution outside the documented stage authorization is required;
- source requirements conflict in a way that changes the result; or
- safe read-only inspection and one bounded reversible correction cannot
  establish a trustworthy state.

Naming, file placement within scope, test selection, local refactoring,
ordinary dependency inspection, retry strategy, session relaunch, branch-name
replacement when unused, status wording, and other low-impact mechanics are not
captain decisions.

For GPU work, a stage request authorizes a bounded real run only when the stage
document or active task brief identifies that run and its fixed inputs. Unknown
or materially larger resource use is a real decision.

## Execution loop

1. Establish the current stage, worktree, branch, environment, prerequisites,
   intended result, preserved behavior, and exact non-goals.
2. If a material decision is missing, escalate and stop. Otherwise continue.
3. Implement the smallest coherent current-stage change.
4. Run focused checks first, then the relevant broader checks. Preserve exact
   commands, seeds, identities, and result paths.
5. On an ordinary failure, diagnose and make one bounded correction without
   asking permission. Repeat while evidence shows progress and scope remains
   unchanged. Escalate only for the material boundary above or a repeated real
   blocker.
6. Update `docs/STATUS.md` and `docs/DECISIONS.md` only when verified durable
   state or a material decision changed.
7. Commit and complete the active delivery contract. Do not stop merely because
   implementation finished, a worker window disappeared, the destination
   advanced, a test needs a safe rerun, or a PR needs an update.
8. Stop when the PR or local-only branch is genuinely ready, a material decision
   is open, or the stage cannot proceed safely.

## Reporting

Keep routine progress inside the worker session or FirstMate status channel.
Report only material phase changes, true blockers, material decisions, failure,
or the final delivery result.

At completion, report:

- stage and intended outcome;
- final feature commit and PR or local branch;
- changed paths;
- tests and exit statuses;
- GPU result or `NOT_REQUIRED`;
- durable documentation changes;
- remaining unknowns; and
- the one next captain-level decision, if any.

Do not emit legacy hard-stop or build/landing waiting markers. Those
artificial boundaries are not part of the FirstMate lifecycle and must not be
reintroduced.
