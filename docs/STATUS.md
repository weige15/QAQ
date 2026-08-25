# QAQ Status

Current objective: **Structurally validate the production runtime for paired lookahead-specific 4/6/8 router training and evaluation**
Legacy work-item reference: **S11-D3**
Status: **READY — structural validation passed; no paired training trial has executed**

## Current state

The frozen paired lookahead-specific 4/6/8 training protocol (legacy work item S11-D1) and deterministic paired-training plan and dispatcher (legacy work item S11-D2) remain the execution authority.

A production runtime now exists at `src/qaq/evaluation/lookahead_468_runtime.py`. The standard-library dispatcher validates the exact frozen configuration, trial ID, explicit CUDA device, canonical destination, and complete aggregation inputs before the command imports that runtime. Default and `--plan` modes remain deterministic, standard-library-only, and inert.

Deterministic tiny-object structural validation proves the runtime scheduler's required boundaries without loading the production model or dataset and without doing CUDA work:

* one deterministic seed initialization is restored byte-identically into every paired cell;
* every cell receives a fresh router-only AdamW with empty initial state;
* the teacher and packed base remain frozen;
* every trial performs exactly 24 observed, ordered AdamW calls with finite losses and router gradients;
* training, soft evaluation, and hard evaluation explicitly audit complete request state, one-time lookahead consumption, established layer-0 handling, and request cleanup;
* same-unit and one-unit-lookahead target ownership and provenance cover all 72 units;
* soft and hard evaluation cover all twelve fixed validation requests in order;
* every hard evaluation contains 72 layer-major decisions, for 864 decisions per trial, with complete required metrics;
* the immediate unchanged-state hard repeat is byte-identical;
* the post-trial evaluator validates all twelve trial identities in frozen order, computes treatment/control and positive/zero-cost route transitions for every request, reports per-seed and median paired deltas, and applies the frozen `CONTINUE`/`REFINE`/`STOP` boundaries; and
* injected identity, data/order, initialization, optimizer membership/freshness, update-count, freeze, gradient, request-state, route/provenance, repeat, aggregation, and persistence defects return `PAUSE` or `REVISE` without partial canonical evidence.

The existing lookahead regression tests continue to prove detached features, one-time probability consumption, target-owned routing, provenance, and the soft-gradient path while keeping the packed base frozen.

Therefore:

* zero of twelve real trials executed;
* no model training or production evaluation occurred;
* no production checkpoint exists;
* no canonical trial evidence or aggregate exists;
* `docs/results/s11d_paired_468/` remains absent; and
* all scientific outcomes remain unknown.

## Decision gate

The bounded implementation and structural checks pass. The runtime is ready for a separately authorized real execution objective.

Do not begin the frozen twelve-trial execution automatically. The next action, only after explicit authorization, is to submit the first frozen trial through the validated dispatcher and stop safely on any runtime prerequisite or evidence mismatch.

## Frozen boundaries

Do not:

* change the two arms, `lambda_bit` values, seeds, candidate bits, pairing, trial order, data, training budget, metrics, thresholds, aggregation rules, or outcome definitions;
* substitute another executor or bypass the dispatcher;
* convert or reuse the historical two-way checkpoint;
* alter synchronous on-demand loading or add prefetch, asynchronous loading or transfer, caching, batching, scheduling, or performance work;
* manually construct missing evidence;
* selectively execute or rerun cells;
* interpret structural validation as scientific evidence; or
* begin a follow-up objective automatically.

## Authoritative references

* Repository rules: `AGENTS.md`
* Frozen paired-training protocol: `docs/stages/S11D_PAIRED_LOOKAHEAD_468_TRAINING.md` (legacy work item S11-D1)
* Frozen machine contract: `configs/lookahead_468_training.json`
* Deterministic dispatcher/plan: `src/qaq/evaluation/lookahead_468_executor.py` (legacy work item S11-D2)
* Production runtime: `src/qaq/evaluation/lookahead_468_runtime.py`
* Command entry point: `scripts/run_lookahead_468_training.py`
* Structural tests: `tests/unit/test_lookahead_468_runtime.py`
* Canonical prior data evidence: `docs/results/s10h_broader_validation.json`
* Durable decisions: `docs/DECISIONS.md`

`docs/STATUS.md` records only the current handoff state. Historical work-item details and evidence belong in the work-item documents, decisions, results, experiments, and Git history.
