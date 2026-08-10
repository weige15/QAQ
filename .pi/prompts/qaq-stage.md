# `/qaq-stage <stage>`

## Mandatory environment check

At the very beginning, run exactly:

```bash
source ~/.venv/bin/activate
which python
python --version
```

Stop immediately if activation fails or if `which python` does not resolve inside `~/.venv`.
Do not create another virtual environment or use system Python.

## Instructions

Perform exactly one requested stage (`S00` through `S09`).
Read `AGENTS.md`, `docs/STATUS.md`, `docs/DECISIONS.md`, the requested `docs/stages/S##_*.md`, and relevant source notes and papers where that stage requires them.

Before changing anything, state the source-supported behavior, the unresolved questions, and the implementation assumptions for this stage.
Record assumptions in `docs/DECISIONS.md`; never silently choose unspecified behavior.
Follow the current stage document's goal, tasks, tests, outputs, and decision gates.
Add or update tests before declaring the stage complete, preserve deterministic seeds and exact commands, and update `docs/STATUS.md` before stopping.

Work on the current stage only.
Never begin a later stage automatically, even if the current stage finishes early.
Stop at the current stage's CONTINUE, PAUSE, REVISE, or STOP decision gate and report the evidence.
Do not modify PDFs under `papers/`.
