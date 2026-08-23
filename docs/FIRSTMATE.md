# FirstMate workflow for QAQ

FirstMate is a separate supervisor distribution. QAQ should be registered as a
project in a FirstMate home; the project-local `.pi/prompts/qaq-firstmate.md` is
only a fallback for a direct Pi session. Copying that prompt into QAQ does not
provide FirstMate's watcher, isolated crew worktrees, status channel, delivery
flow, or session recovery.

## Recommended setup

From a cloned and current FirstMate home, launch a supported primary agent and
ask it to add:

```text
https://github.com/weige15/QAQ
```

Use `no-mistakes` as QAQ's standing delivery posture. Keep automatic merge off
unless the captain explicitly wants FirstMate's `+yolo` merge authority. With
normal merge authority, a worker should run unattended through implementation,
validation, correction, commit, feature-branch push, PR creation, and required
checks; the captain is asked only for a material project decision or the final
merge.

For unattended supervision, use FirstMate's `/afk` mode after the QAQ task is
under way. It handles routine worker notifications and reports only genuine
escalations or the final outcome.

## Task wording

A stage request should authorize a complete current-stage delivery rather than
a single command. For example:

```text
Complete QAQ's current documented stage through a green no-mistakes PR. Make
routine reversible implementation decisions yourself. Escalate only if the
scientific protocol, acceptance criteria, current-stage scope, external
resource authorization, destructive state, or merge authority must change. Do
not begin the next stage.
```

The repository's `AGENTS.md` defines the same boundary for every worker.

## Expected stops

A healthy task stops for one of these reasons:

- a PR is ready with its required checks complete;
- a material research or product decision is required;
- a required external artifact, credential, or bounded compute resource is
  unavailable;
- safe repository state cannot be established without discarding or rewriting
  work; or
- the current stage is complete and starting the next stage needs a new request.

It should not stop for naming, file placement, test selection, formatting,
ordinary test repair, branch refresh, worker-window loss, feature-branch push,
PR creation, or another reversible in-scope implementation choice.
