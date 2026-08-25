# QAQ Status

Current objective: **Make the defined same-unit block-sensitivity study structurally executable**
Source protocol/runtime references: **S11-D1 through S11-D3 and the route-policy diagnostic**
Status: **COMPLETE — deterministic executor ready; sensitivity study not executed**

## Completed route-policy diagnostic

The read-only diagnostic in
`docs/stages/S11D_ROUTE_POLICY_DIAGNOSTIC.md` uses only the twelve byte-identified
canonical S11-D trial files. Across `10,368` hard decisions, `86.11%` of the
`864` fixed trial/unit cells are invariant across all twelve requests, a modal
static unit policy explains `96.74%` of decisions, and unordered within-cell
request pairs disagree `4.96%` of the time. The evidence therefore supports a
mostly static unit/layer policy with real but secondary request variation.

The reproducible derived artifact is
`docs/results/s11d_route_policy_diagnostic.json`. It reports 4/6/8 use and route
variation by layer, attention/FFN unit, request, seed, timing, and cost, plus
matched directional transitions and explicit early/middle/late localization.
Pooled lookahead choices are `0.03665` bits higher than same-unit, but the sign
is not consistent across seeds, so conservatism is not systematic. These are
observational associations and cannot identify which individual downgrade
caused quality loss.

A smallest future same-unit one-block-at-a-time sensitivity study is defined for
the `42` units with observed same-unit cost downgrades.

## Completed executor readiness

The standard-library-only contract in
`src/qaq/evaluation/block_sensitivity.py` and command
`scripts/run_s11d_block_sensitivity.py` now derive a byte-deterministic inert
plan from the exact diagnostic and canonical same-unit lambda-zero route maps.
The plan preserves all `42` ordered targets, seed contexts `[1729,1730,1731]`,
twelve ordered requests, `36` paired contexts per intervention, target-forced-8
controls, 4-first/6-only-after-failure scheduling, immediate repeats, and the
existing per-seed `1.10` aggregate KL/mean-error and per-request `1.25` KL
factors.

Future evidence is complete only when one atomic unit/precision file contains
all 36 ordered control/treatment pairs, exact source/input/teacher/route
identities, the complete established S11 hardware/software fields on a
compatible RTX 3090, finite metrics, repeated logits and metrics, three
independently recomputed seed summaries, and passing route-isolation and
prohibited-work audits. Precision 6 dispatch requires a complete valid failed
precision 4 file; it is rejected after a precision 4 pass. Persistence validates
before writing, uses a same-directory fsynced temporary file and atomic
no-overwrite hard-link
promotion, verifies promoted bytes, and never treats a temporary file as
complete. The non-mutating resume command scans canonical paths in target order,
classifies exactly one next action per unit, and rejects temporary, linked,
wrongly named, malformed, mixed-study, or cross-execution-provenance evidence.
Aggregation independently revalidates every consumed canonical file, requires
one shared hardware/software identity, all 42 precision-4 results, and exactly
the required fallbacks; it rejects mixed, missing, duplicate, orphaned, or extra
evidence and emits one lowest-safe precision per target.

Focused CPU-only structural tests pass without importing model, dataset, Torch,
Transformers, Any-Precision, or CUDA runtime packages. The default command only
prints the deterministic plan. The dispatcher validates a future explicit
execution request but deliberately contains no production execution import.
No sensitivity trial, model/CUDA execution, router training, lambda retuning,
lookahead work, or canonical sensitivity result was performed or created.

## Frozen S11-D source state

The separately authorized real execution used the frozen protocol in
`configs/lookahead_468_training.json` (SHA-256
`4a62aeb7d8ae90a6349dc9dc8aab6dda4196b54876c4d0546c05808936fefe92`),
the validated dispatcher in `scripts/run_lookahead_468_training.py`, and the
production runtime in `src/qaq/evaluation/lookahead_468_runtime.py` on
`cuda:2`, an NVIDIA GeForce RTX 3090.

All twelve trials completed exactly once in the frozen order. Every trial
persisted complete canonical evidence, completed the 24 ordered optimizer
updates, evaluated the twelve frozen validation requests, retained 864 hard
route decisions, passed identity/data-order/pairing/optimizer/freeze/gradient/
route-provenance/repeat/prohibited-work audits, and created no production
checkpoint. No trial was retried, substituted, reordered, or added.

Only after all twelve canonical trial files existed did the validated
aggregation command run. The canonical aggregate is
`docs/results/s11d_paired_468/aggregation.json` (SHA-256
`ad40dc13276b83aef5ea0d58d1920c4e472ba3f8817c691e5ea5fa5b1881ef04`).
Its completeness, finiteness, and pairing audits all pass.

## Frozen outcome

The aggregate applies the frozen outcome rules and returns **STOP**:

* median paired lookahead `0.03 - 0.0` hard KL delta:
  `0.010411208301472167` (quality requires `<= 0.0`);
* median paired lookahead `0.03 - 0.0` hard selected-width delta:
  `-0.4375` bits (precision requires `<= -0.4907407407407405`);
* all three seed width deltas are negative, but the frozen median-width gate
  fails;
* the frozen factor safeguards do not all pass; and
* neither exact `REFINE` region applies.

The per-seed paired hard-KL deltas are
`[0.010565421544015408, 0.010202943192174036, 0.010411208301472167]` for
seeds `[1729,1730,1731]`. The corresponding selected-width deltas are
`[-0.4884259259259247, -0.4375, -0.4120370370370363]` bits.

This valid complete result answers the bounded frozen formulation negatively.
It does not prove all lookahead-specific training impossible and does not
select or recommend a production lambda.

## Decision gate

Stop. Executor readiness is complete, and the frozen S11-D `STOP` outcome
remains unchanged. The sensitivity result directory remains absent. No
sensitivity execution, retuning, selective or additional S11-D run, performance
work, loader work, or lookahead objective is authorized or under way.

## Authoritative references

* Repository rules: `AGENTS.md`
* Frozen protocol: `docs/stages/S11D_PAIRED_LOOKAHEAD_468_TRAINING.md`
* Frozen machine contract: `configs/lookahead_468_training.json`
* Validated dispatcher/plan: `src/qaq/evaluation/lookahead_468_executor.py`
* Production runtime: `src/qaq/evaluation/lookahead_468_runtime.py`
* Command entry point: `scripts/run_lookahead_468_training.py`
* Canonical trial and aggregate evidence: `docs/results/s11d_paired_468/`
* Route-policy diagnostic: `docs/stages/S11D_ROUTE_POLICY_DIAGNOSTIC.md`
* Read-only analyzer: `scripts/analyze_s11d_route_policy.py`
* Derived diagnostic artifact: `docs/results/s11d_route_policy_diagnostic.json`
* Sensitivity executor/plan contract: `src/qaq/evaluation/block_sensitivity.py`
* Sensitivity structural command: `scripts/run_s11d_block_sensitivity.py`
* Sensitivity structural tests: `tests/unit/test_s11d_block_sensitivity.py`
* Experiment record: `docs/EXPERIMENTS.md`
* Durable decisions: `docs/DECISIONS.md`
