# QAQ runtime and environment rules

This file is authoritative for shell setup, repository-entry checks, Python selection, and GPU availability. Other controller files must reference this file rather than repeat its commands.

## Fresh-shell preflight

Before any project command in every fresh shell, enter the intended QAQ repository or worktree and run exactly:

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

Record:

- the exact Python path;
- the Python version;
- the physical directory;
- the repository root;
- the current branch;
- `HEAD`;
- the short status.

Continue only when all of these are true:

- activation succeeds;
- `which python` resolves inside `~/.venv`;
- the physical path, repository root, branch, and `HEAD` match the current authorization;
- the expected base and ancestry checks pass;
- no unexpected change would be overwritten.

Use `PAUSE` when any condition fails. Do not create another virtual environment and do not use system Python.

Repeat the complete preflight:

- in every new shell;
- after entering a newly created or manually created worktree;
- before project commands after changing to another authorized worktree.

Do not assume shell state carries over between commands, workers, tools, or sessions.

## GPU gate

For a GPU-dependent experiment, test, training run, inference run, or build, run this in the same shell before the first GPU command:

```bash
nvidia-smi
```

Record the visible GPUs, utilization, and relevant free memory.

Continue only when the command succeeds and the required capacity is available. If capacity requirements are unknown, use `PAUSE`. Do not silently switch to CPU, another GPU, or a different execution mode.

For CPU-only work, record:

```text
GPU check: NOT_REQUIRED
```

## Missing repository or environment

If `projects/QAQ` does not exist or is not a Git repository, report the exact failed check and issue one bounded setup step. Do not pretend the repository checks succeeded.

If `~/.venv` is unavailable, activation fails, or Python resolves outside `~/.venv`, use `PAUSE`. Do not work around the required environment.
