# `/qaq-review <stage>`

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

Review exactly one requested current stage.
Read `AGENTS.md`, `docs/STATUS.md`, `docs/DECISIONS.md`, the current `docs/stages/S##_*.md`, relevant source notes and papers, and the stage's changed files.

Check source claims against cited evidence, identify unstated assumptions, verify the required tests and exact commands, and check that the stage has not crossed its scope boundary.
Confirm that quantized weights remain frozen during router training where applicable, production bit planes are physically packed, and packed planes rather than unpacked weights are transferred where applicable.
Record any newly discovered assumption in `docs/DECISIONS.md` and update `docs/STATUS.md` before stopping when the review changes project state.

Review only the current stage.
Do not implement or start a later stage automatically.
Do not modify PDFs under `papers/`.
