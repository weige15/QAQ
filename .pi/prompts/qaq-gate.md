# `/qaq-gate <stage>`

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

Evaluate exactly one requested current stage against its document in `docs/stages/`.
Read `AGENTS.md`, `docs/STATUS.md`, `docs/DECISIONS.md`, the stage document, relevant source notes and papers, and the exact test and result artifacts.

Check every required task, test, output, uncertainty, and CONTINUE/PAUSE/REVISE/STOP condition.
Distinguish source-supported behavior from implementation choices and require assumptions to be recorded rather than inferred.
Verify deterministic seeds, exact commands, physically bit-packed production planes, frozen quantized weights during router training where applicable, and synchronous transfer of packed planes where applicable.

Produce a pass, pause, revise, or stop recommendation with evidence.
Update `docs/STATUS.md` before stopping if the gate result changes the durable project state.

Gate only the current stage.
Never begin a later stage automatically.
Do not modify PDFs under `papers/`.
