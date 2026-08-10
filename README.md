# QAQ

QAQ is a paper-guided baseline for query-conditioned mixed-precision language-model inference.
It is an implementation scaffold and research-control repository, not an asserted exact reproduction of any source paper.

The intended baseline combines genuinely packed nested multi-precision weights represented as bit planes, query-conditioned precision routing, separate attention and FFN decisions, teacher-student router training, hard query-level inference routes, and synchronous on-demand transfer of selected packed planes from CPU to GPU.

## Current state

The repository is at **S00 — Lock environment and specification** and has no implementation stage underway.
Read `docs/STATUS.md`, `docs/DECISIONS.md`, and the current stage document before doing any work.
Workers must complete only the current stage and stop at its decision gate.

The source PDFs under `papers/` are preserved project material and must not be modified, renamed, summarized, or reinterpreted as part of scaffold setup.

## Project layout

- `papers/` — source PDFs and their existing inventory.
- `docs/` — plan, decisions, source inventory, bit-plane contract, experiment plan, and stage gates.
- `.pi/prompts/` — reusable stage, review, and gate instructions.
- `configs/` — reserved for explicit, versioned configuration.
- `src/qaq/` — implementation package scaffold.
- `tests/` — unit, integration, and system test scaffolds.
- `scripts/` — reproducible project scripts.
- `third_party/` — pinned external source or integration material.

## Environment

Every worker performing implementation, testing, Python inspection, dependency work, or execution must run:

```bash
source ~/.venv/bin/activate
which python
python --version
```

`which python` must resolve inside `~/.venv`; otherwise stop and report the problem.

## Scope boundary

This initial scaffold does not implement quantization, model integration, routing, training, CUDA kernels, CPU/GPU loading, or experiments.
