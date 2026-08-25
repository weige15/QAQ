# QAQ Status

Current objective: **Diagnose request dependence in completed canonical S11-D routes**
Source protocol/runtime references: **S11-D1 through S11-D3**
Status: **COMPLETE — mostly static unit/layer policy; sensitivity study not executed**

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
the `42` units with observed same-unit cost downgrades. It was not executed and
is not under way. No trial, training, retuning, lambda selection, lookahead
experiment, or canonical result was created or changed.

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

Stop. The route-policy diagnostic is complete, and the frozen S11-D `STOP`
outcome remains unchanged. No sensitivity execution, retuning, selective or
additional S11-D run, performance work, loader work, or lookahead objective is
authorized or under way.

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
* Experiment record: `docs/EXPERIMENTS.md`
* Durable decisions: `docs/DECISIONS.md`
