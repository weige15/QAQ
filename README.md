# QAQ

QAQ is a paper-guided baseline for query-conditioned mixed-precision language-model inference.
It is an implementation scaffold and research-control repository, not an asserted exact reproduction of any source paper.

The intended baseline combines genuinely packed nested multi-precision weights represented as bit planes, query-conditioned precision routing, separate attention and FFN decisions, teacher-student router training, hard query-level inference routes, and synchronous on-demand transfer of selected packed planes from CPU to GPU.

## Current state

See [`docs/STATUS.md`](docs/STATUS.md) for the authoritative implementation
stage, evidence, and next action.
Read `docs/STATUS.md`, `docs/DECISIONS.md`, and the current stage document before doing any work.
Workers complete only the current stage, but a request for that stage covers the
full bounded cycle through implementation, validation, correction, commit, and
PR-ready delivery. Internal checks are worker decision points, not separate
permission prompts. Workers escalate only material research, scope, resource,
destructive, or merge decisions and never begin the next stage automatically.

For the intended supervisor workflow, see [`docs/FIRSTMATE.md`](docs/FIRSTMATE.md).

The source PDFs under `papers/` are preserved project material and must not be modified, renamed, summarized, or reinterpreted as part of scaffold setup.

## Project layout

- `papers/` — source PDFs and their existing inventory.
- `docs/` — plan, decisions, source inventory, bit-plane contract, experiment plan, and stage gates.
- `.pi/prompts/` — reusable direct-Pi stage, review, and gate instructions; an active FirstMate task brief takes precedence over the direct-Pi fallback controller.
- `configs/` — reserved for explicit, versioned configuration.
- `src/qaq/` — implementation package, organized by reusable model, router, quantization, loading, and evaluation concerns.
- `tests/` — unit, integration, and system test scaffolds.
- Reusable implementation modules live under `qaq/model/`, `qaq/router/`, `qaq/quantization/`, `qaq/loading/`, and `qaq/evaluation/`; stage-era paths are not part of the active import surface.
- `scripts/` — reproducible project scripts.
- `third_party/` — pinned external source or integration material.

## Environment

Every worker performing implementation, testing, Python inspection, dependency work, or execution must run:

```bash
source ~/.venv/bin/activate
which python
python --version
```

`which python` must resolve inside `~/.venv`; otherwise stop the project command and report the problem.
A clean worker or shell relaunch is routine recovery and does not require another permission prompt.

## Scope boundary

The authoritative current implementation stage, supported capabilities, and
remaining work are tracked in [`docs/STATUS.md`](docs/STATUS.md). The physical
format and its production-storage boundary are documented in
`docs/BITPLANE_FORMAT.md`.
