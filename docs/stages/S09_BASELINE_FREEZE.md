# S09 — Evaluate and freeze the baseline

## Goal

Compare teacher, static 4-bit, static 8-bit, routed resident, and routed on-demand modes; record quality, routing, memory, transfer, and latency results; freeze the reproducible baseline.

## Tasks

- Define the final evaluation matrix, datasets, prompts, seeds, and environment.
- Run the five required modes with identical applicable inputs.
- Record quality, route selections, GPU memory, actual packed transfer bytes, and latency.
- Analyze failure cases and document limitations without broadening the baseline.
- Freeze the exact source revisions, configuration, model artifact references, commands, and result artifacts.
- Explicitly defer asynchronous loading, prefetching, transfer prediction, cost penalties, cross-request caching, multi-query batching, token schedulers, and unrelated improvements.

## Tests

- All five comparison modes execute reproducibly.
- Results can be regenerated from the recorded commands and deterministic seeds.
- Resource metrics distinguish packed storage from unpacked or fake references.
- Final quality and performance comparisons pass the documented release criteria.
- A clean rerun confirms the frozen baseline.

## Required outputs

- Complete comparison report in `docs/EXPERIMENTS.md` or a linked artifact.
- Frozen configuration and revision manifest.
- Limitations and deferred-work record.
- Final status update naming the passing commit and reproducibility evidence.

## Known uncertainties

- Quality thresholds, hardware variance, and final evaluation inputs are not established by this scaffold.
- The baseline may reveal limitations that require a revised scope rather than a silent implementation change.

## CONTINUE condition

The reproducible baseline comparison is complete, all required evidence is recorded, and the baseline is explicitly frozen.

## PAUSE condition

A required evaluation resource or external artifact is unavailable.

## REVISE condition

A comparison or reproducibility defect can be corrected without adding pre-freeze mechanisms.

## STOP condition

The modes are not comparable, resource claims are not grounded in packed data, or the baseline cannot be frozen without unresolved critical assumptions.
