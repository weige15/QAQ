# QAQ runtime and environment rules

This file owns shell setup, repository identity checks, Python selection, and
GPU availability for the direct-Pi fallback controller. An active FirstMate
brief may add worktree-specific checks but does not weaken these project
requirements.

## Fresh-shell preflight

Before the first project command in every fresh shell, enter the intended QAQ
worktree and run:

```bash
source ~/.venv/bin/activate
which python
python --version
pwd -P
git rev-parse --show-toplevel
git branch --show-current
git rev-parse HEAD
git status --short
```

Continue when Python resolves inside `~/.venv`, the physical repository and
branch match the assigned worktree, and no unexpected work would be overwritten.
Record the exact identity in the normal task evidence.

A missing shell, terminal, or process is temporary execution state. Relaunch a
clean shell, repeat this preflight, and continue without captain permission.
If activation or repository identity still fails after the clean retry, report
the exact failed check as a real blocker. Do not create another environment,
use system Python, reset the branch, or discard preserved work as a workaround.

## GPU gate

Before a GPU-dependent command, run in the same shell:

```bash
nvidia-smi
```

A current-stage request authorizes the run when the stage document or active
task brief fixes the run, inputs, and expected scope. Continue when the command
succeeds and the documented capacity is available.

When capacity is temporarily unavailable, use the active FirstMate wait or
blocked status rather than asking for a new implementation decision. Escalate
only when the required capacity is unknown, the run materially exceeds the
stage's documented scope, or a different device or execution mode would change
the evidence. Never silently switch to CPU or alter the experiment.

For CPU-only work, record `GPU check: NOT_REQUIRED`.
