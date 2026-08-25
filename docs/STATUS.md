# QAQ Status

Current objective: **Execute and classify the frozen paired lookahead-specific 4/6/8 router experiment**
Legacy protocol/runtime references: **S11-D1 through S11-D3**
Status: **COMPLETE — frozen outcome `STOP`**

## Current state

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

Stop. The frozen `STOP` outcome ends this objective. No retuning, selective or
additional run, performance work, protocol change, loader work, or follow-up
objective is authorized or under way.

## Authoritative references

* Repository rules: `AGENTS.md`
* Frozen protocol: `docs/stages/S11D_PAIRED_LOOKAHEAD_468_TRAINING.md`
* Frozen machine contract: `configs/lookahead_468_training.json`
* Validated dispatcher/plan: `src/qaq/evaluation/lookahead_468_executor.py`
* Production runtime: `src/qaq/evaluation/lookahead_468_runtime.py`
* Command entry point: `scripts/run_lookahead_468_training.py`
* Canonical trial and aggregate evidence: `docs/results/s11d_paired_468/`
* Experiment record: `docs/EXPERIMENTS.md`
* Durable decisions: `docs/DECISIONS.md`
