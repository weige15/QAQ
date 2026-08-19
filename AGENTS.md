# QAQ contribution rules

This file is authoritative for every worker operating in this repository.

## Mandatory environment

Before any implementation, testing, Python inspection, dependency work, or execution, every worker must activate the project environment exactly as follows:

```bash
source ~/.venv/bin/activate
which python
python --version
```

Stop immediately if activation fails or if `which python` does not resolve inside `~/.venv`.
Do not create another virtual environment and do not use system Python.

## Required reading and stage discipline

Before starting work, read `docs/STATUS.md`, `docs/DECISIONS.md`, the current stage document under `docs/stages/`, and relevant source notes and papers when the stage requires them.
Work on the current stage only.
Do not automatically begin a later stage.
Stop at the current stage's decision gate.
State what is known from sources, what remains unknown, and which implementation assumptions are being made.
Record every assumption in `docs/DECISIONS.md` rather than silently choosing behavior.

## Completion and reproducibility

Add or update tests before declaring a stage complete.
Preserve deterministic seeds and exact commands.
Update `docs/STATUS.md` before stopping.
The status update must identify the evidence and the next action without claiming unresolved work is complete.

## Baseline boundaries

This project is a paper-guided QAQ baseline, not an asserted exact reproduction.
Production bit planes must be physically bit-packed.
Byte-per-bit representations are test/reference-only.
Fake quantization cannot support memory, transfer, or latency claims.
Quantized model weights remain frozen during router training.
The baseline loader is synchronous.
Packed planes, not unpacked weights, must be transferred.

Before the baseline is frozen, do not add asynchronous transfers, prefetching, transfer prediction, bit-width cost penalties, cross-request caching, multi-query batching, or unrelated research improvements.
Do not add the Any-Precision dependency until the project has completed its source/backend review and the exact upstream commit is pinned.

## Scope

The initial baseline targets genuinely packed nested multi-precision weights, query-conditioned routing, separate attention and FFN routes, teacher-student router training, hard query-level inference routes, and synchronous on-demand CPU-to-GPU loading of selected packed planes.

Every stage must preserve the distinction between behavior supported by source papers and implementation decisions made where the papers leave details unspecified.

## Maintaining this file

Keep this file for knowledge useful to almost every future agent session in this project.
Do not repeat what the codebase already shows; point to the authoritative file or command instead.
Prefer rewriting or pruning existing entries over appending new ones.
When updating this file, preserve this bar for all agents and keep entries concise.

## Worktree artifact provisioning

Before artifact-backed validation, provision the ignored S03 artifact with
`scripts/provision_s03_artifact.py`.

`QAQ_S03_ARTIFACT_SOURCE` must name the exact external directory containing
the frozen `pytorch_model.bin`. The script is authorized to create only the
ignored worktree-local symlink under `quantized/`; it must never copy, alter,
regenerate, or commit the packed artifact.

Stop on a missing source, hash mismatch, non-empty conflicting destination,
or incorrect pinned backend revision.
