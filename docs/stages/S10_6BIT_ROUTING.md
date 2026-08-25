# Enable static 6-bit execution

_Legacy work-item reference: S10-A._

Legacy identifiers elsewhere in this record are retained only for historical cross-reference to frozen decisions, evidence, paths, and machine-facing contracts.

## Gate result

**CONTINUE.** S10-A enables public static precision `6` against the existing
nested Qwen3 4→8 artifact. The implementation change is limited to
`qaq.model.static`: its public static precision set is now exactly `(4, 6, 8)`.
The pinned Any-Precision source, artifact, router source, request loading,
hard routing, router checkpoints, and S09 evidence were not changed.

The artifact-backed results below are the recorded implementation-gate
evidence. The exact identity-matched S03-B artifact was authorized read-only
from the original QAQ worktree through `QAQ_S03_ARTIFACT`, and its
`pytorch_model.bin` SHA-256 was verified. The focused integration and
preservation reruns passed; no artifact was regenerated or substituted.

## Sources, identity, and artifact inventory

The authoritative S09 closeout establishes S01–S08 complete and S09 frozen and
closed. The artifact used for this gate is:

```text
quantized/s03b_qwen3_4b/backend_cache/packed/
  anyprec-(1cfa9a7208912126459214e8b04321603b3df60c)-w8_orig4-gc1-c4_s1_blk64
```

Its `pytorch_model.bin` SHA-256 is
`29d9bc526b3da0bd39daf2f82afd141f82d005ca1232cabc75cfe9d9ecc1cfee`,
matching `docs/quantized_model_manifest.json`. The pinned Any-Precision
submodule is `a3257d02740cc5757c78673da534b0630ff3a4ea` and was clean.

The recorded artifact inventory was loaded without regenerating or
requantizing it. All 252 expected Qwen3 targets had exactly one parent
`<target>.qweight` plus `lut4`, `lut6`, and `lut8`; the state dictionary also
retains the natural pinned-backend `lut5` and `lut7` buffers. There were no
missing or extra quantized keys.

| inventory | result |
| --- | ---: |
| target / qweight count | 252 |
| qweight shape and dtype | `[8, out_features, in_features // 32]`, `torch.int32` |
| qweight parent payload | 3,633,315,840 bytes |
| selected six-plane payload | 2,724,986,880 bytes (`qweight[:6]`) |
| LUT6 count | 252 |
| LUT6 shapes | 72 × `[2560,64]`; 72 × `[9728,64]`; 72 × `[1024,64]`; 36 × `[4096,64]` |
| LUT6 dtype / values | `torch.float16`; all finite |
| LUT6 total | 141,557,760 bytes |

Representative tensor evidence (SHA-256 is over the contiguous tensor bytes):

| target | qweight shape / bytes / digest | LUT6 shape / bytes / digest |
| --- | --- | --- |
| `model.layers.0.mlp.down_proj` | `[8,2560,304]` / 24,903,680 / `0c9b602586778d37a29cbb9ae90b9d6506de24d31e6161ae9e4cc1884c1b2c7d` | `[2560,64]` / 327,680 / `4af0512d26f8339b7be02ca0cb3bfa1f51263b784876d15415351829fa0b6610` |
| `model.layers.0.mlp.gate_proj` | `[8,9728,80]` / 24,903,680 / `a694efd7223cace865825e893b1ce8607877bd5b6e7bd10a93ac73929cd212d2` | `[9728,64]` / 1,245,184 / `d9c5a8e2fdbeafeada1b23549d254f47d1f4b84d47c0c12a7353f5e7df88d920` |
| `model.layers.0.self_attn.o_proj` | `[8,2560,128]` / 10,485,760 / `c45f3b7214f051e976dc8bb75b2a257688d341e94a14353b299b98306c7ab163` | `[2560,64]` / 327,680 / `6814e0fac3a9b145cce9db57d6dd4caa5898aeef1ec9593509a2c16176deb412` |
| `model.layers.0.self_attn.q_proj` | `[8,4096,80]` / 10,485,760 / `e7b3060bc87b050e7eb54fef50930a0ca7972c1c34d844411d6249861d965bf3` | `[4096,64]` / 524,288 / `12052f80149d0fe9bb75e41afba7ff02ec318c346c960bb6ebbaa0600de80c5e` |

## Six-bit semantics and backend evidence

Precision 6 uses the existing nested parent and matching lookup table exactly:

```text
4 bits: qweight[:4] + lut4
6 bits: qweight[:6] + lut6
8 bits: qweight[:8] + lut8
```

The parent is physically packed; no byte-per-bit representation, separate
six-bit qweight, persistent dequantized weight, or duplicate model is added.
The pinned backend continues to load the inclusive 4–8 buffers, including its
internal `lut5` and `lut7` buffers, while QAQ's public static gate exposes only
4, 6, and 8.

The real representative `model.layers.0.mlp.down_proj` was executed on
`cuda:0` (NVIDIA GeForce RTX 3090) with deterministic CPU-seeded FP16 input
`[4,9728]`, using the pinned backend at precision 6. The output was finite
FP16 `[4,2560]` on `cuda:0`, and the independent pinned `dequant_kbit` plus
FP16 `torch.matmul` reference was finite and shape-equal. With the established
S01 tolerance `atol=0.05`, `rtol=0.01`, `max_abs_error=0.015625`,
`mean_abs_error=0.0017089294`, and `allclose=True`. Repeated backend output
was bitwise equal with digest
`e25d9c8d531d22269c18b1ef27ebe17dc6ceb34223326a6f25824f154e02778f`.
The module had no parameters and no registered dense weight; the temporary
reference reconstruction was not part of persistent module state.

## Full-model static smoke and audit

The deterministic full Qwen3 static loader was run at precision 6 on the same
RTX 3090. It loaded the recorded artifact normally with no state-dict errors
and returned finite FP16 logits of shape `[1,8,151936]`. The first and repeated
logits digest was
`4e0856454ebab64588183a1e72acc2fc34ffea68d82c590526624edd804e3390`.

The loaded graph contained 252 `AnyPrecisionLinear` modules, 252 qweight
buffers, 252 LUT6 buffers, and zero `qweight6` buffers. Quantized modules had
zero dense parameters. The existing loader graph reported 389,152,256 total
parameters, all currently marked trainable by that loader; S07's established
router-training freeze machinery remains unchanged and was not re-run here.
The precision-6 smoke was repeated bitwise deterministically. Existing static
4-bit and 8-bit forward, checkpoint, target-inventory, duplicate-model, and
physical-byte regressions also passed.

## Recorded validation commands

Every command was preceded by the mandated `~/.venv` activation, interpreter
check, and successful `nvidia-smi` check. `PYTHONPATH=src:.` was used where
needed to avoid an ambient unrelated `scripts` package on `PYTHONPATH`.

```text
PYTHONPATH=src:. pytest -q tests/unit
110 passed

PYTHONPATH=src:third_party/any-precision-llm:. \
  QAQ_S03_ARTIFACT=<identity-matched S03-B artifact> \
  QAQ_MODEL_DEVICE=cuda:0 pytest -q tests/integration/test_s10a_static6.py
3 passed in 221.42s

PYTHONPATH=src:. QAQ_S03_ARTIFACT=<identity-matched S03-B artifact> \
  QAQ_MODEL_DEVICE=cuda:0 pytest -q \
  tests/integration/test_static4_forward.py \
  tests/integration/test_static8_forward.py \
  tests/integration/test_expected_modules_quantized.py \
  tests/integration/test_no_duplicate_precision_models.py \
  tests/integration/test_manifest_byte_count.py \
  tests/integration/test_checkpoint_roundtrip.py
10 passed in 640.26s

PYTHONPATH=src:. pytest -q \
  tests/integration/test_s06_soft_routing.py \
  tests/integration/test_s07_distillation_smoke.py
3 passed in 9.37s

ruff check src/qaq/model/static.py \
  tests/unit/test_static_precision_validation.py \
  tests/integration/test_s10a_static6.py
All checks passed
```

## Limitations and boundary

Static 6-bit execution (legacy work item S10-A) establishes that capability only. It does not add 6-bit routing,
change `CANDIDATE_BITS`, change hard-route choices, alter request loading or
state ownership, retrain or reload router checkpoints, or assess quality,
latency, transfer, or memory claims. Router semantics remain exactly 4/8
after S10-A. A separate follow-up objective and decision is required for any 6-bit
routing behavior.

## Define three-way 4/6/8 router semantics

_Legacy work-item reference: S10-B._

**Gate: CONTINUE.** Three-way 4/6/8 router semantics (legacy work item S10-B) add explicit, backwards-compatible learned-router
candidate semantics for historical `(4, 8)` and new `(4, 6, 8)` routers. It does
not train a router or add the cost-aware objective.

### Established facts and implementation choice

S06 established the historical learned-router architecture, 72 separate
attention/FFN routers, `[p4, p8]` ordering, RMS epsilon `1e-6`, hidden width
`128`, GELU, temperature behavior, detached features, and frozen packed base.
S10-A established static six-bit execution through `qweight[:6] + lut6`.
S04 manual `PrecisionPlan` remains exactly 4/8-only, and S08 synchronous
on-demand loading remains exactly 4/8-only.

The S10-B implementation choice is one authoritative
`validate_candidate_bits()` validator accepting only `(4, 8)` and `(4, 6, 8)`.
Candidate ordering is explicit data, carried on request state and attached to
soft traces, route logs, and checkpoint metadata. Vector length is never used
to infer bit meaning.

### Router, state, and probability semantics

`SoftPrecisionRouter` defaults to `(4, 8)` and sizes its output projection from
the validated candidate tuple. An explicit `(4, 6, 8)` router emits exactly
three probabilities in canonical order `[p4, p6, p8]`. Probability validation
requires a matching final dimension, finite non-negative values, and a unit sum.

`QaqRequestState` defaults to `(4, 8)` and accepts an explicit `(4, 6, 8)`
learned state. Stored probability vectors and hard routes must belong to that
state's ordering. A three-way resident learned hard route can therefore store
and reuse `6`; historical state and manual policy behavior remain unchanged.

The soft packed execution is the candidate-aware sum:

```text
output = p4 * packed(inputs, 4) + p6 * packed(inputs, 6) + p8 * packed(inputs, 8)
```

Only configured precisions execute, each once. Forced one-hot endpoints select
the corresponding packed output, and probabilities remain attached to the
mixture for gradient flow. Soft trace records include `candidate_bits`.

### Counts and route observations

The router count remains 72. With feature dimension `2560` and hidden width
`128`, the historical per-router output head is `Linear(128, 2)` and the
three-way head is `Linear(128, 3)`, adding `129` scalars per router. The
verified full counts are:

```text
historical (4,8): 23,620,752
three-way (4,6,8): 23,630,040
increase: 9,288
```

Route records preserve historical `p4`/`p8` records and add explicit `p6` plus
candidate ordering for three-way records. Statistics expose hard fractions for
4, 6, and 8, candidate-aware layer and attention/FFN distributions, entropy,
and `soft_average_width`. The width equations are `4*p4 + 8*p8` historically
and `4*p4 + 6*p6 + 8*p8` for three-way records. Historical records are not
reinterpreted as three-way records.

### Hard routes and checkpoints

Hard routing remains deterministic ordinary `torch.argmax`: index 0 maps to 4,
index 1 maps to 6 in three-way mode, and index 2 maps to 8. First-maximum ties
therefore map `[0.5,0.5,0]` to 4, `[0,0.5,0.5]` to 6, and equal three-way
probabilities to 4; historical `[0.5,0.5]` remains 4.

Checkpoint format version 1 is retained because existing metadata already
serializes candidate ordering. Matched historical and synthetic three-way
router states round-trip with equal probabilities and hard routes. Metadata
mismatch and state-shape mismatch reject 4/8-to-4/6/8 and reverse loads; no
weights are padded, copied, interpolated, or silently converted.

### Verification and boundaries

The exact focused commands were:

```text
source ~/.venv/bin/activate
which python
python --version
nvidia-smi
PYTHONPATH=src:. pytest -q tests/unit
PYTHONPATH=src:. pytest -q tests/integration/test_s06_soft_routing.py tests/integration/test_s07_distillation_smoke.py tests/integration/test_s05_tiny_qwen3_execution.py tests/integration/test_s08_sync_transfer.py tests/integration/test_s08_request_lifetime.py tests/integration/test_request_state_isolation.py tests/integration/test_route_fixed_during_decode.py
PYTHONPATH=src:third_party/any-precision-llm:. pytest -q tests/integration/test_s10b_soft_packed.py
PYTHONPATH=src:third_party/any-precision-llm:. QAQ_MODEL_DEVICE=cuda:0 pytest -q tests/integration/test_s06_soft_packed.py -k three_way
```

Focused unit coverage passed `120 passed` across the full unit suite, including
candidate validation, parameter counts, request state, ties, logs/statistics,
checkpoint compatibility, historical `PrecisionPlan`, and frozen-state audits.
The S10-B focused integration command passed `1 passed` for real pinned packed
4/6/8 endpoint and gradient execution. The S06/S07 lifecycle and S08
request/loader regression selection passed `14 passed`; the S06 three-way
production seam passed as part of `4 passed` with the S10-B packed test.
Ruff passed for all changed source and S10-B test files.

The mandatory environment preflight resolved Python to
`/nfs/home/s314511048/.venv/bin/python`, Python `3.12.3`, and NVIDIA GeForce
RTX 3090 devices through `nvidia-smi`; the real packed fixture used `cuda:0`.
The repository is based at S07C-R1 commit
`0747b0aa605500c93332ab847344be52494f07e7`, with S10-A complete, the S07C
checkpoint identity preserved, Any-Precision clean at
`a3257d02740cc5757c78673da534b0630ff3a4ea`, and no unexpected source changes
before implementation.

No training or retraining occurred. No cost-aware objective or penalty
coefficient was added. The S08 loader was not modified and no 6-bit on-demand
support is claimed. The packed artifact, Any-Precision source, S07/S09 results,
and historical S07 checkpoint were not modified. No quality, latency, memory,
transfer, or routing-quality evaluation was performed.

**Historical next action:** define and validate the cost-aware 4/6/8 router objective (legacy work item S10-C). The completed cost-aware-objective gate is recorded below.

## Add the cost-aware 4/6/8 router objective

_Legacy work-item reference: S10-C._

**Gate: CONTINUE.** The cost-aware 4/6/8 router objective (legacy work item S10-C) adds reusable objective composition primitives only;
it does not train or retrain a router and does not select a production cost
coefficient.

### Established facts and implementation choice

S07's established training objective remains completion-only teacher-student
`T^2 * masked KL(teacher || student)`. `masked_kl_distillation_loss()` is
unchanged and remains the sole KD implementation. Three-way router semantics (legacy work item S10-B) establish explicit
learned-router orderings `(4, 8)` and `(4, 6, 8)`, with the three-way order
`[p4, p6, p8]`, and all 72 attention/FFN routing units expose differentiable
probabilities. S10-C records the implementation choice to add a normalized
bit-plane-count surrogate as a composable auxiliary term. Its reduction is an
unweighted arithmetic mean over every included attention and FFN decision;
this is not measured hardware weighting.

For an explicit candidate tuple, the normalized candidate cost is:

```text
c(bit) = (bit - 4) / (8 - 4)
C(p) = sum_b p_b * c(b)
L_bit = mean_r C(p_r)
L_total = L_KD + lambda_bit * L_bit
```

The cost vectors are `[0.0, 0.5, 1.0]` for `(4, 6, 8)` and `[0.0, 1.0]`
for historical `(4, 8)`. The implementation constructs this vector from the
explicit ordering, never from vector length. For three-way routing,
`expected_bit_width = 4 + 4 * L_bit` is a diagnostic relationship only.
`expected_bit_cost()`, `mean_expected_bit_cost()`,
`request_state_expected_bit_cost()`, and
`cost_aware_distillation_loss()` keep the probability tensors attached to
autograd. Request-state aggregation includes each stored attention and FFN
probability exactly once and gives every decision equal weight.

`lambda_bit` is validated as numeric, finite, non-negative, and not boolean.
Zero is the optional backwards-compatible default and returns the KD scalar
unchanged, including equivalent KD gradients. A clearly labeled positive
lambda is used only in tests to verify `4 < 6 < 8` objective ordering and
cost-gradient pressure away from an 8-bit-dominant softmax toward lower-cost
alternatives. No nonzero production lambda is established.

Candidate validation remains restricted to `(4, 6, 8)` and historical `(4, 8)`;
5-bit, 7-bit, reordered, malformed, non-finite, negative, wrongly shaped, or
non-unit-sum probabilities are rejected. This objective is a normalized
bit-plane-count surrogate, not a claim about latency, memory, transfer,
energy, kernel runtime, or any other hardware cost.

Focused S10-C tests cover exact endpoint and mixed/uniform costs, historical
scaling, bounded finite aggregation, explicit ordering and probability/weight
validation, expected-width diagnostics, positive test-only lambda ordering,
lambda-zero scalar and gradient compatibility, 8-bit-dominant softmax
pressure, complete request-state aggregation with gradients, finite combined
gradients, and frozen-state preservation. Existing S07 and S10-B regression
suites remain required; no artifact-backed Qwen3 execution is needed for this
objective-only work item.

## Calibrate the bit-cost coefficient

_Legacy work-item reference: S10-D._

**Gate: CONTINUE.** Bit-cost coefficient calibration (legacy work item S10-D) calibrates observations for the cost-aware router coefficient (legacy work item S10-C); it
does not choose a production coefficient or start full router training. The
machine-readable protocol is `configs/s10d_lambda_calibration.json`, the
runner is `scripts/run_s10d.py`, and the complete measurements are in
`docs/results/s10d_lambda_calibration.json`.

### Source facts and locked protocol

S07 supplies the unchanged completion-only `masked_kl_distillation_loss`, the
frozen teacher/packed-student boundary, the four-step data/training contract,
and the fixed Wikitext rows. S10-B supplies explicit three-way ordering
`(4,6,8)` / `[p4,p6,p8]`, 72 separate routers, hard index mapping, and resident
three-way execution. S10-C supplies the normalized cost vector `[0,0.5,1]`
and `L_total = L_KD + lambda_bit * L_bit`. These primitives were reused; the
historical S07 runner and objective were not changed.

The run used the identity-matched Qwen3-4B teacher revision
`1cfa9a7208912126459214e8b04321603b3df60c`, packed artifact SHA-256
`29d9bc526b3da0bd39daf2f82afd141f82d005ca1232cabc75cfe9d9ecc1cfee`,
Wikitext revision `b08601e04326c79dfdd32d625aee71d232d685c3`, and clean
Any-Precision `a3257d02740cc5757c78673da534b0630ff3a4ea`. It used exactly four
train examples, two validation examples, source order, sequence 64,
prompt/completion 32/32, seed 1729, AdamW (`lr=0.001`, weight decay 0),
batch size 1, four optimizer steps, KD temperature 2.0, and routing
temperature 1.0. The explicit free GPU was `cuda:0` (RTX 3090).

Before learned-route interpretation, static 4/6/8 was run once on the same
validation examples. All logits were finite. Mean masked KD / mean absolute
logit error were respectively: static 4 `0.4631383568 / 0.7434162199`,
static 6 `0.0803938322 / 0.2947778329`, and static 8 `0.0050811910 /
0.0910567641`.

Every trial reset a fresh canonical three-way router initialized after seed
1729, verified 72 routers and 23,630,040 scalars, and reloaded a clone of the
same router-only state hash `15da263c1c60fccd89c306a41b4eef9da3739d45bdddcdcdca59ec4e02cfb758`.
No historical two-way checkpoint was loaded. Each trial built a new AdamW,
used the same examples/order and precomputed teacher targets, recorded the
separate initial KD/bit-cost gradient norms and lambda-weighted ratios, and
ran exactly four updates. Frozen teacher and packed-student hashes matched;
all gradients, losses, logits, expected widths, and probabilities were finite.
The runner's local backward seam recomputes frozen packed weights through the
existing `execute_packed_linear` helper rather than retaining dense weights;
this is an execution-memory measure only and does not modify production model
or backend code.

The full grid completed before extension decisions. No permitted extension was
triggered: lambda `0.003` was `OTHER` rather than a >=95% collapse to 4 or 6;
lambda `0.1` did not have the exact lambda-zero hard map and its soft-width
delta was `0.4878288171` bits, not below `0.001`. Thus only
`0.0, 0.003, 0.01, 0.03, 0.1` were performed.

| lambda | soft KD | soft width | hard KD | hard width | hard 4/6/8 fractions | label |
|---:|---:|---:|---:|---:|---|---|
| 0.0 | 0.0498826504 | 6.8869917558 | 0.0692504579 | 7.1944444444 | 0.083333/0.236111/0.680556 | OTHER |
| 0.003 | 0.0508279633 | 6.8676936171 | 0.0694124866 | 7.1944444444 | 0.083333/0.236111/0.680556 | OTHER |
| 0.01 | 0.0556957237 | 6.8241818363 | 0.0681527648 | 7.1805555556 | 0.090278/0.229167/0.680556 | OTHER |
| 0.03 | 0.0529664997 | 6.7444918938 | 0.0645516384 | 7.0277777778 | 0.090278/0.305556/0.604167 | ADAPTIVE_OBSERVED |
| 0.1 | 0.0620286539 | 6.3991629386 | 0.0736745223 | 6.5694444444 | 0.131944/0.451389/0.416667 | OTHER |

The result records mean/max logit errors, entropy, p4/p6/p8, hard fractions
summing to one, explicit hard 6 fractions, mean p6, whether validation
routes select 6, per-validation route maps, route variation, unique map counts,
frozen-state audits, baseline deltas, and deterministic soft/hard Pareto
frontiers. The observed hard frontier contains lambdas 0.03 and 0.1; the soft
frontier contains 0.0, 0.003, 0.03, and 0.1. These are observations only, not
a scalarized selection.

### Verification and limitations

The pre-sweep command passed 44 tests: focused S10-D tests plus S10-C,
S10-B, S07 distillation, and request-state regressions. Ruff passed for the
runner and focused tests. The execution command was:

```text
source ~/.venv/bin/activate && which python && python --version && nvidia-smi
PYTHONPATH=src:third_party/any-precision-llm:. python scripts/run_s10d.py \
  --config configs/s10d_lambda_calibration.json --device cuda:0 \
  --output docs/results/s10d_lambda_calibration.json
```

This is a four-step calibration observation on six fixed examples, not full
router training or a paper-score reproduction. It makes no latency, memory,
transfer, energy, or kernel-runtime claim; no S08 loader was invoked, and no
production checkpoint or lambda was created. Extra seeds, epochs, data,
tuning, quotas, entropy terms, temperature changes, optimizer changes, and
extensions not authorized by the protocol were not run. The next decision is
owned by firstmate/captain: refine, confirm, or begin full training after
reviewing the observed frontier.

## Freeze the frontier-confirmation protocol

_Legacy work-item reference: S10-E._

**Gate: CONTINUE.** The frontier-confirmation protocol freeze (legacy work item S10-E) freezes the machine-readable confirmation protocol in
`configs/s10e_frontier_confirmation.json`. This work item defines the future
confirmation trial only; it does not execute a trial, train a router, select a
production lambda, or begin S10-F.

### Source/project-established facts

S10-A through S10-D are complete at the merged starting point
`e718f27fe6b02082709d65665396640e251e602c`, which is PR #9/S10-D in
`origin/main`. S10-D's canonical result is
`docs/results/s10d_lambda_calibration.json` and its locked configuration is
`configs/s10d_lambda_calibration.json`. That evidence completed exactly
`0.0, 0.003, 0.01, 0.03, 0.1`, performed no adaptive extension, selected no
production lambda, and observed the hard KD/width frontier at `0.03` and
`0.1`. Three-way router semantics (legacy work item S10-B) establish the explicit resident three-way `(4,6,8)` router,
72 request-routed attention/FFN units, and deterministic hard routing; S10-C
establishes the unchanged completion-only KD composition with normalized
bit-plane cost; S07 establishes the frozen teacher/packed-base and locked data
and training contract.

The S10-D source/result documents establish the inherited identities and
contract, but they do not establish the S10-E seeds, candidate subset, paired
controls, or confirmation decision rule.

### Implementation choice — captain-selected controls

The captain-selected confirmation controls are exactly three seeds
`[1729, 1730, 1731]`, candidates/lambdas `[0.0, 0.03, 0.1]`, and nine paired
trials. For each seed, one fresh canonical three-way router initialization is
cloned identically before each lambda; every lambda gets a fresh AdamW in the
same order, with no warm start and no historical two-way S07 checkpoint.
These seeds and the three-candidate confirmation are implementation choices,
not source-paper facts.

The protocol requires initial/final router hashes, initial KD and bit-cost
gradient norms, the lambda-weighted gradient ratio, finite-value and frozen-
component/router-only optimizer audits, soft and hard validation metrics,
explicit hard fraction 6, both validation route maps, route variation,
distinct map count, and cross-seed aggregates. It explicitly forbids latency,
memory, transfer, throughput, and energy measurements.

Both validation route maps use the inherited S10-D serializer's layer-major
order: layer 0 attention, layer 0 FFN, then layer 1 attention, layer 1 FFN,
through layer 35. This ordering is frozen for S10-E and keeps route-map
interpretation compatible with the existing S10-D evidence.

Success requires all nine trials and audits, no invalid or degenerate collapse,
`0.03` on the per-seed hard KD/width frontier in at least two of three seeds,
paired-control median hard KD delta (candidate minus `0.0`) no greater than
`0.0`, strictly lower paired-control median hard selected width, and no
reproducibility failure. There is no scalar combined score and no arbitrary
quality-loss threshold. Success authorizes only later broader validation;
failure is `REFINE`; incomplete evidence is `PAUSE`.

### Verification boundary

The focused protocol tests validate exact identities, ordering, counts,
training values, paired-initialization semantics, measurement fields,
prohibitions, and gate rules by rejecting missing, extra, reordered, or
reintroduced values. They are lightweight configuration tests and execute no
experiment. The frontier-confirmation protocol work (legacy work item S10-E) stops after this freeze; the next action after a passing
commit is to **execute the frozen three-seed frontier-confirmation protocol (legacy work item S10-F).**

## Execute the frozen three-seed frontier confirmation

_Legacy work-item reference: S10-F._

**Worker classification: REVISE.** The canonical matrix completed all nine
ordered trials, but a runner audit defect invalidated the gate evidence after
execution. No trial was repaired or rerun.

### Exact execution and identities

The run used implementation base
`7fc136eabdba302e199354ae001cd1e1cd42199f`, the merged S10-E commit, rather
than the historical `required_starting_commit` retained in the frozen config.
The frozen config remained byte-identical at SHA-256
`fe5ff8826f17605ca8b2dc7d83555e858d3d9f5fa67d14b49bb09b7cbf66a879`.
All trials used one explicit `cuda:0` NVIDIA GeForce RTX 3090. The pinned
Qwen3-4B model and tokenizer revision was
`1cfa9a7208912126459214e8b04321603b3df60c`; the Wikitext revision was
`b08601e04326c79dfdd32d625aee71d232d685c3`; the packed artifact SHA-256 was
`29d9bc526b3da0bd39daf2f82afd141f82d005ca1232cabc75cfe9d9ecc1cfee`; and
the Any-Precision revision was `a3257d02740cc5757c78673da534b0630ff3a4ea`.
The artifact and backend source were read-only explicit overrides because the
disposable worktree's gitlink was uninitialized; neither was copied or
modified.

The exact ordered matrix was, for each seed in `[1729,1730,1731]`, lambdas
`[0.0,0.03,0.1]`: nine pairs total. Each seed used one fresh canonical
three-way router state cloned identically across lambdas, a fresh AdamW, and
exactly four optimizer steps. The teacher and packed student base remained
frozen, and no historical S07 checkpoint, static S10-D reference, S08 loader,
adaptive lambda, or prohibited serving/resource measurement was used.

### Observed evidence and gate values

The result is `docs/results/s10f_frontier_confirmation.json`, SHA-256
`d68f041e0a3dc32c465e8b8068ca3ab230d39253757e30f3019ca7e681b14233`. It
contains all nine per-trial records, both exact 72-entry layer-major hard
route maps for each validation ID, route variation, distinct-map counts,
finite/freeze/base/optimizer fields, soft and hard metrics, and the one
immediate same-state hard-validation reproducibility repeat per trial.

Measured aggregates were:

- `0.03` was on the hard KD/selected-width frontier in `2/3` seeds.
- Paired hard KD delta median (`0.03 - 0.0`) was
  `-0.004020056687295437`.
- Paired hard selected-width delta median (`0.03 - 0.0`) was
  `-0.16666666666666696`.
- Reproducibility failures: `0`.
- No invalid or degenerate collapse was observed; collapse labels were
  `ADAPTIVE_OBSERVED` or `OTHER`.

The generated artifact reports `REFINE` because its nine
`router_only_optimizer_audit` and `fresh_adamw_audit` fields are `false`.
This is a software defect in result interpretation: the inherited optimizer
audit is a Python tuple `("routers.",)` while the runner compared it only to
`["routers."]`. The raw audit simultaneously records fresh optimizer state,
the `routers.` prefix, and the expected 23,630,040 router scalars. Because the
defect could affect gate validity and was discovered after canonical trials,
the worker classification is `REVISE`; all nine records are preserved but not
treated as a valid CONTINUE gate.

### Limitations and next action

This is fixed four-example/four-step confirmation evidence only. It does not
select a production lambda or checkpoint and does not authorize broader
validation. Firstmate must resolve the runner audit interpretation and decide
what evidence policy applies; this task performs no repair or rerun.

### Repair the frontier-confirmation audit and preserve historical evidence

_Legacy work-item reference: S10-F._

The repair preflight used `/nfs/home/s314511048/.venv/bin/python` with Python
`3.12.3` and healthy RTX 3090 visibility. The original packed artifact hash
remained `29d9bc526b3da0bd39daf2f82afd141f82d005ca1232cabc75cfe9d9ecc1cfee`.
The preserved result remained byte-identical at
`d68f041e0a3dc32c465e8b8068ca3ab230d39253757e30f3019ca7e681b14233`.

The smallest deterministic reproduction showed that the runner evaluated a
tuple operand `("routers.",)` against a list operand `["routers."]`; contents
match but Python equality is false. Existing parameter-identity auditing
rejects missing router parameters, extra non-router parameters, and duplicate
tensors. The S10-F repair normalizes only the prefix container before the
exact one-prefix comparison and adds strict fresh-state classification. No
S10-D, router, packed-artifact, Any-Precision, dataset, or frozen-config
semantics changed.

The repair tests passed `4`; the S10-E/S10-F focused suite passed `65`; the
inherited regression selection passed `46`; Ruff passed for the changed
Python files; and `git diff --check` passed. The original nine-trial result
and all measured frontier values were not rewritten.

Historical revalidation cannot take Branch A. For router membership, the
preserved result has only per-trial prefix/count summaries and omits the
included parameter names/identities, group membership, and duplicate audit.
For fresh AdamW, it has only a boolean fresh-state field and no independent
preserved optimizer-state snapshot. Neither audit therefore has sufficient
preserved runtime evidence without inferring behavior from source/tests.
The resulting primary outcome is **PAUSE / RERUN_REQUIRED**. No canonical
training or evaluation rerun, extra trial, broader validation, production
lambda selection, or success commit occurred; the frozen three-seed frontier
confirmation (legacy work item S10-F) remains the current objective.

### Complete the canonical frontier-confirmation rerun

_Legacy work-item reference: S10-F._

Attempt 2 executed exactly the frozen nine ordered pairs on `cuda:0`, preserving
attempt 1 at `docs/results/s10f_frontier_confirmation.json` with SHA-256
`d68f041e0a3dc32c465e8b8068ca3ab230d39253757e30f3019ca7e681b14233`. The new
artifact is `docs/results/s10f_frontier_confirmation_rerun.json`, SHA-256
`b3bcc0e45d45852ac5060209c4789453ed452462f528f7bffd4cb80fb1ef58cb`.

Every trial records expected and actual optimizer parameter counts and stable
name digests, identity-based membership, missing/extra/duplicate counts,
construction serial, zero state entries before training, and fresh AdamW
status. All runtime, freeze, finite-value, four-step, route-map, and
reproducibility audits passed. The frozen aggregates are `0.03` frontier `2/3`,
paired hard KD median delta `-0.004020056687295437`, paired hard width median
delta `-0.16666666666666696`, and zero reproducibility failures. Focused tests
passed `65`, inherited regressions passed `46`, Ruff passed, and
`git diff --check` passed. The frozen three-seed frontier-confirmation gate (legacy work item S10-F) outcome is **CONTINUE**. No follow-up objective,
production lambda, or broader validation was started.

## Define and freeze the broader-validation protocol

_Legacy work-item reference: S10-G._

**Gate: CONTINUE (protocol freeze only; no broader-validation experiment result exists; legacy work item S10-G).**

The machine-readable protocol is `configs/s10g_broader_validation.json` and
its focused, configuration-only tests are
`tests/unit/test_s10g_broader_validation_protocol.py`. S10-A through S10-F
are established complete, S10-F attempt 2 is present with its original
attempt-1 artifact preserved, attempt 2 is classified CONTINUE, no production
lambda was selected, and no broader validation has run. S10-F authorized only
this separately scoped broader-validation decision.

### Source/project-established facts

The protocol inherits the pinned Qwen3-4B and tokenizer revision, Wikitext
revision, clean Any-Precision revision, packed artifact identity, 72
attention/FFN routers, explicit `[p4,p6,p8]` ordering, normalized cost vector
`[0.0,0.5,1.0]`, completion-only KD objective, frozen teacher/packed base, and
router-only optimization from the preceding work items. It also preserves the
S10-F candidate/lambda/seed matrix, paired initialization semantics, fresh
AdamW requirement, layer-major 72-unit route-map order, and two-axis hard
KD/selected-width gate. These facts and the preserved attempt hashes are
recorded in the protocol rather than inferred from a future run.

### Implementation choices and frozen contract

The broader data choice is exactly 24 train examples and 12 validation
examples selected by the pinned tokenizer and deterministic ascending offsets:
train offsets `0,1000,...,23000`, validation offsets
`0,250,...,2750`. The exact selected row indices, IDs, split/revision, source
order, sequence length, and prompt/completion boundaries are frozen in the
config. This is six times each S10-F count while retaining the same
first-qualifying-row rule. The future run is exactly one pass over 24 examples
and 24 AdamW updates, batch size 1, with S10-F's learning rate, weight decay,
temperatures, scheduler, and other optimizer settings preserved. The only
additional optimizer values are the explicit PyTorch AdamW defaults
`betas=[0.9,0.999]`, `eps=1e-8`, and `amsgrad=false`.

The exact nine ordered trials remain seeds `[1729,1730,1731]` and lambdas
`[0.0,0.03,0.1]`. Each seed has one fresh canonical three-way router state,
cloned identically for each lambda, and each lambda gets a fresh AdamW with
router-only membership. The teacher and packed base remain frozen. Each trial
records finite/freeze/optimizer audits, training count and update count, soft
and hard KD/logit/width/probability/entropy metrics, fractions, all twelve
validation route maps (each exactly 72 entries in layer-major order), route
variation, cross-seed paired deltas, and one immediate same-state hard
reproducibility repeat. The schema freezes base-2 entropy, an explicit
collapse audit, structured optimizer proof rather than optimizer booleans,
run-level inherited-regression evidence, and run-level prohibited-work
evidence.

The future gate has exactly two lower-is-better axes: hard validation KD and
hard selected width. CONTINUE requires all nine valid trials and audits, no
invalid collapse, at least two per-seed frontier memberships for `0.03`,
paired median hard KD delta `<=0.0`, paired median hard-width delta `<0.0`,
passing inherited regressions, and zero reproducibility failures. REFINE is a
complete structurally valid matrix that fails one or more two-axis result
conditions; with PAUSE evaluated first, REVISE covers complete evidence with
failed audits, inherited regressions, invalid collapse, reproducibility, or
prohibited work; CONTINUE requires every condition. No scalar combined score
or production-lambda selection is permitted.

The broader-validation protocol freeze (legacy work item S10-G) created no runner, result JSON, or execution path, and performed
no training, model evaluation, CUDA evaluation, or hardware/resource
measurement. Adaptive lambda search, post-result replacement, warm starts,
S07 conversion, teacher/base training, non-router optimizer membership,
candidate or cost changes, S08 changes, six-bit on-demand loading, and S10-H
execution are explicitly prohibited.

Verification after the freeze passed 53 focused S10-G tests, the S10-D/S10-E/
S10-F predecessor regression selection passed 121 tests, Ruff passed for the
changed Python test, and `git diff --check` passed. The next action is a
separately authorized follow-up execution decision; this work item claims no broader
validation result and selects no production lambda.

## Implement the protocol-locked broader-validation runner and pre-execution validation

_Legacy work-item reference: S10-H1._

**Historical gate: CONTINUE for H1 implementation only.** H1 added
`scripts/run_s10h.py` and `tests/unit/test_s10h_broader_validation.py`. It does
not execute S10-H2, train routers, load Qwen3 or packed weights, consume real
broader-validation rows, create `docs/results/s10h_broader_validation.json`,
select a lambda, or measure latency, memory, transfer, throughput, energy, or
hardware cost.

The runner reads the exact frozen S10-G config bytes (SHA-256
`fcb66902174558e5d3f9198f34a8430b685568fd4e21e1632b40f6870aa4aec7`), checks
that `7fc136eabdba302e199354ae001cd1e1cd42199f` is an ancestor, preserves both
S10-F attempt hashes, and checks the pinned model/tokenizer, Wikitext,
Any-Precision, packed-artifact, and manifest identities without importing a
model runtime; pre-execution identity checking hashes the actual packed
`pytorch_model.bin` bytes. Its future-result validator also requires the
packed-artifact path and manifest digest, and is independent of the S10-D/F
four-step execution helpers: it requires the exact nine ordered pairs
(seeds `1729/1730/1731`, lambdas `0.0/0.03/0.1`), exact 24/12 ordered data
manifests, 24 examples and updates, batch/accumulation `1/1`, one epoch,
AdamW values, paired canonical initialization, fresh identity-audited
router-only optimizers, teacher/base freeze proofs, twelve ordered 72-unit
route maps restricted to 4/6/8, repeat evidence, run-level regressions and
prohibition audits, and recomputed cross-seed aggregates. Gate classification
uses the frozen PAUSE > REVISE > REFINE > CONTINUE precedence.

The default invocation and explicit `--plan` validate and print protocol
identity, ancestry, frozen revisions/artifact identity, data/trial counts,
training contract, future output path, prohibitions, thresholds, and the
explicit future `--execute` command. Plan mode loads no model or dataset,
trains nothing, evaluates no CUDA behavior, writes no result, and cannot alter
frozen artifacts. `--execute` is an explicit H2 opt-in but returns PAUSE in H1
because the real executor is intentionally not present. A canonical result
path is never overwritten. A deterministic synthetic structural fixture is
used only by focused validator tests and is not experiment evidence.

Focused H1 verification passed `31` tests; the combined H1/S10-G protocol
selection passed `81`, and the full unit suite passed `291` tests with one
existing duplicate-optimizer warning. Ruff and `git diff --check` passed. No
model, real data, CUDA behavior, training, result JSON, or production lambda
was used. At that historical gate, the next action was a separately authorized
broader-validation implementation and execution work item (legacy work item S10-H2). The real-executor implementation (legacy work item S10-H2-A) is recorded below; do not infer a
broader-validation result from the plan or synthetic fixture.

## Implement the real broader-validation executor seam

_Legacy work-item reference: S10-H2-A._

**Gate: COMPLETE for implementation only; no broader-validation experiment was run (legacy work item S10-H2-A).** The real broader-validation executor seam (legacy work item S10-H2-A) replaces the
earlier protocol runner's `--execute` PAUSE seam (legacy work item S10-H1) with a lazy call to
`qaq.router.s10h_executor.execute_production`. `scripts/run_s10h.py` remains
standard-library-only on its default and `--plan` paths: those paths do not
import torch, Transformers, datasets, CUDA, Qwen, packed execution, or result
writers. `--execute` requires an explicit `--device`; there is no automatic
GPU selection. The production Qwen/data loader is behind the replaceable
`S10HRuntime` boundary, and the focused tests inject a deterministic tiny
runtime rather than claiming Qwen quality or running a reduced Qwen trial.

The shared scheduler owns the frozen nine ordered trials
`(seed, lambda_bit)`, one fresh 72-router three-way initialization per seed,
paired cloned lambda starts, fresh AdamW identity audits, exactly 24 examples
and 24 updates, completion-only KL at `T=2.0`, routing at `T=1.0`, normalized
costs `[0.0, 0.5, 1.0]`, request-local hard routes, ordered twelve 72-entry
maps, and one immediate same-state hard repeat. It records actual router and
optimizer tensor/name/scalar evidence, construction serials and state counts,
finite/nonzero gradients, teacher/base requires-grad and hash evidence,
per-trial hashes, update order, and ordered data manifests. The unmodified H1
`validate_result()` is run before any output operation; CONTINUE and REFINE
may be promoted, while PAUSE and REVISE return without writing.

Output promotion validates the destination parent and refuses existing paths,
creates a temporary file on that parent filesystem, flushes and closes it,
re-validates the serialized bytes, then uses an atomic same-filesystem
no-overwrite link. All temporary files are removed on failure or interruption.
The canonical H2 path is deliberately refused during H2-A, so this work item did
not invoke production output handling and
`docs/results/s10h_broader_validation.json` remains absent.

The deterministic smoke covers all nine trials through the injected runtime,
216 calls to the unchanged KL/cost primitives, 24 updates per trial, paired
and distinct initialization, optimizer membership/freshness, frozen state,
finite gradients, route/cost/order audits, validator propagation, output
cleanup, and overwrite refusal. It writes only to a temporary test directory;
it is not a Qwen/H2 result. Commands used for this work item are the mandated
`~/.venv`/`nvidia-smi` preflight, focused S10-H tests, the S10-G/S10-F/S10-E/
S10-D regression selection, `PYTHONPATH=src:. pytest -q tests/unit`, the
non-executing `python scripts/run_s10h.py --plan` smoke, Ruff on changed Python
files, and `git diff --check`. No canonical H2 experiment, real-Qwen trial,
production lambda selection, resource measurement, or H2 result was run or
created.

## Repair the production example-ID contract

_Legacy work-item reference: S10-H2-BR1._

**Gate: COMPLETE — repair only.** The production example-ID contract repair (legacy work item S10-H2-BR1), including the failed first-attempt evidence,
is recorded in [`docs/EXPERIMENTS.md`](../EXPERIMENTS.md). This work item owns the
repaired production contract: `_select_examples()` supplies
`DistillationExample` objects, while its separate manifest remains metadata.
The order-validation boundary reads each selected ID only through
`example.example_id`, requires a non-empty string, and compares the resulting
list directly with the unchanged frozen train and validation arrays. Missing,
empty, non-string, reordered, and dictionary-substitute values fail with
structured `ExecutorError("REVISE", ...)`.

The boundary does not infer IDs from manifests, reorder or mutate selections,
subscribe to examples, or pass non-`DistillationExample` selections to
`_device_example()`. Valid selected objects retain their identity and type
through device conversion. This repair changes no frozen protocol, model,
training, loader, validator, result, or submodule behavior; the evidence
record owns the exact preservation and non-execution claims. The repaired canonical broader-validation retry (legacy work item S10-H2-B2) remains
outside this gate and requires separate authorization from the repaired,
reviewed, merged commit.

## Complete the repaired canonical broader-validation retry

_Legacy work-item reference: S10-H2-B2._

**Gate: COMPLETE — REFINE.** The repaired canonical broader-validation retry (legacy work item S10-H2-B2) consumed the one authorized
retry during operational attempt 2 from exact commit
`b1aca71bcc584f0e3559e5fe7caf142c2f750db3`. The nine-trial result is complete,
independently valid, and canonical, but it misses the frozen paired hard-KL
condition. This outcome selects and recommends no production lambda.

### Authorization, execution, and preserved PAUSE history

Attempt 1 remains **REVISE before training** at `87786fe6...`; BR1 repaired its
`DistillationExample.example_id` consumer contract and PR #15 merged that
repair at `b1aca71...`. The pre-execution PAUSE branches remain unchanged:
`fm/qaq-s10h2b2-20260819T151818Z` at `97b461aed06feb189821b0eb8cd956ba64b1a3ab`
and `fm/qaq-s10h2b2-reauthorized-20260819` at
`c13ef4f3c8ce3e0727848733ecfee432ffa51ad8`. They ran no trial and consumed no
retry. A later Herdr service-loss lab established no workspace and ran no
repository command.

The one real retry started at `2026-08-19T17:45:47Z` in named Herdr session
`fm-lab-qaq-s10h2b2-2026-1101046-3135`, workspace `w1`, tab `w1:t1`, pane
`w1:p1`. It exited `0` and wrote the noncanonical candidate. The execution
shell then exited because an over-strict clean-status assertion rejected that
expected untracked candidate; this was a post-execution orchestration PAUSE,
not a candidate or validator defect. The first authorized closeout pane,
control directory `/tmp/qaq-s10h2b2-closeout.keCf2i`, independently validated
`REFINE` with no errors, then exited because an extra assertion searched the
wrong commit locations instead of `ancestry.commit`. It did not promote or
modify project files.

The captain authorized one additional closeout-only pane. It used new control
directory `/tmp/qaq-s10h2b2-closeout.7vNem5`, named session
`fm-lab-qaq-s10h2b2-clos-1146146-961`, workspace `w1`, tab `w1:t1`, pane
`w1:p1`, and active bash PID `1146397`. The pane was visible, inspectable, and
bound to this isolated worktree. It ran no test, plan, model load, evaluation,
training, profiler, monitoring loop, or `--execute` command. All earlier
reports and the execution log were preserved.

### Environment, identities, and pre-execution evidence

Both the consumed execution and final closeout used
`/nfs/home/s314511048/.venv/bin/python`, Python `3.12.3`. Execution selected
only `cuda:0`, directly mapped to physical GPU 0 because
`CUDA_VISIBLE_DEVICES` was unset: NVIDIA GeForce RTX 3090,
UUID `GPU-384b6377-8f0c-e3d2-8b3a-b3408b54fd53`. Immediately before execution
all eight GPUs reported 24,576 MiB total, 24,124 MiB free, 0% utilization, and
no compute process. The PyTorch mapping check reported 25,017,974,784 free of
25,296,044,032 bytes on logical device 0.

Frozen identities matched: config
`fcb66902174558e5d3f9198f34a8430b685568fd4e21e1632b40f6870aa4aec7`, manifest
`1e2b3515072e22d71ac35a35a3002e3a1dcd5ce44887c554b1408f735c928530`, packed
model `29d9bc526b3da0bd39daf2f82afd141f82d005ca1232cabc75cfe9d9ecc1cfee`,
model/tokenizer `1cfa9a7208912126459214e8b04321603b3df60c`, Wikitext
`b08601e04326c79dfdd32d625aee71d232d685c3`, Any-Precision
`a3257d02740cc5757c78673da534b0630ff3a4ea`, and historical S10-F artifacts
`d68f041e0a3dc32c465e8b8068ca3ab230d39253757e30f3019ca7e681b14233` and
`b3bcc0e45d45852ac5060209c4789453ed452462f528f7bffd4cb80fb1ef58cb`.
The logical packed artifact already existed and was verified in place; it was
not copied, regenerated, requantized, reprovisioned, or overridden through
`QAQ_S03_ARTIFACT`. The pinned backend remained clean.

The authoritative pre-execution evidence at `b1aca71...` was: focused repaired
selection `9 passed, 5 deselected`; combined executor/validator `47 passed`;
S07 distillation plus executor `22 passed`; four predecessor protocols
`134 passed` with only the established duplicate-optimizer warning; full unit
suite `310 passed` with that same warning; Ruff and `git diff --check` passed;
and the non-executing plan reported the exact identities, 24/12 manifests,
nine-trial order, absent results, and no load/train/evaluate/write behavior.

### Independent audit and canonical promotion

The execution log is `/tmp/qaq-s10h2b2-execution.log`, SHA-256
`1f3da7860eb44dd7f710762d2be41357deb8af8fd2ecb6c4a37e12c006e04f55`; its
exit-code file contains `0`. The candidate SHA-256 was
`7d9e0aff3b686570be0d1d57b5513ee921d60bd5470f275b0cd7cbb4fd63db20`.
The unmodified `scripts.run_s10h.validate_result` was invoked independently,
read the execution commit only from `payload.get("ancestry", {}).get("commit")`,
found exact `b1aca71...`, exact ordered pairs, classification `REFINE`, and no
errors.

A complete independent audit passed 3,306 checks with zero errors. It verified
the format, stage, all frozen identities, `(4,6,8)` order, 72 routers,
23,630,040 router scalars, 288 canonical `routers.` parameter tensors, exact
24/12 manifests, exact nine ordered trials, 24 history entries and optimizer
updates per trial, finite/nonzero router gradients, finite losses, teacher/base
freeze hashes and gradient absence, paired initializations, unique fresh AdamW
serials `1..9`, empty initial optimizer state, zero missing/unexpected/duplicate
optimizer members, twelve ordered valid 72-entry route maps per trial,
soft/hard metrics, route aggregates, unchanged-state reproducibility, inherited
regressions, prohibited-work evidence, exact aggregate recomputation, and no
forbidden measurement field. Initial router hashes were identical across
lambdas within each seed and distinct across seeds:
`7b5b5bd2...` (1729), `cca1b7cf...` (1730), and `c96ce0f8...` (1731).

After that audit, the candidate and canonical paths were confirmed to share a
parent. The closeout revalidated the candidate, called only
`os.link(candidate, canonical)` with no overwrite or fallback, verified exact
bytes and SHA-256, and unlinked the candidate. It did not use `cp`, rename
overwrite, or a cross-filesystem fallback. A separate post-promotion validator
again returned `REFINE` with no errors and exact `ancestry.commit`. Canonical
result: `docs/results/s10h_broader_validation.json`, SHA-256
`7d9e0aff3b686570be0d1d57b5513ee921d60bd5470f275b0cd7cbb4fd63db20`.

### Aggregate observations

- Median hard validation KL by lambda: `0.0 = 0.01439695991575718`,
  `0.03 = 0.028918379141638677`, `0.1 = 0.07732601106787722`.
- Median hard selected width by lambda: `0.0 = 7.643518518518518`,
  `0.03 = 7.1342592592592595`, `0.1 = 6.150462962962963`.
- Lambda `0.03` hard-frontier membership: seed 1729 `true`, seed 1730 `true`,
  seed 1731 `true`; count `3/3`.
- Paired median hard-KL delta (`0.03 - 0.0`):
  `0.014972516723598044`.
- Paired median hard-width delta (`0.03 - 0.0`):
  `-0.4907407407407405`.
- Reproducibility failures: `0`.

The positive paired hard-KL delta fails the frozen `<= 0.0` condition. The
frontier count, strict width reduction, and reproducibility conditions pass.
With complete valid evidence and no integrity failure, the prescribed outcome
is therefore **REFINE**, not REVISE or PAUSE.

### Per-trial observations

| seed | lambda | hard KL | hard fractions 4 / 6 / 8 | soft expected width | hard selected width | collapse | changed units / fraction | distinct maps |
|---:|---:|---:|---|---:|---:|---|---|---:|
| 1729 | 0.0 | 0.012808105093427002 | 0.0 / 0.16435185185185186 / 0.8356481481481481 | 7.610388476508249 | 7.671296296296297 | OTHER | 7 / 0.09722222222222222 | 10 |
| 1729 | 0.03 | 0.028918379141638677 | 0.018518518518518517 / 0.4375 / 0.5439814814814815 | 7.067989198371542 | 7.050925925925926 | OTHER | 15 / 0.20833333333333334 | 12 |
| 1729 | 0.1 | 0.07732601106787722 | 0.10069444444444445 / 0.7743055555555556 / 0.125 | 6.089355673305036 | 6.048611111111111 | OTHER | 12 / 0.16666666666666666 | 12 |
| 1730 | 0.0 | 0.015681098991384108 | 0.0 / 0.1875 / 0.8125 | 7.55385224689104 | 7.625 | OTHER | 5 / 0.06944444444444445 | 6 |
| 1730 | 0.03 | 0.030653615714982152 | 0.01273148148148148 / 0.4074074074074074 / 0.5798611111111112 | 7.117284387127496 | 7.1342592592592595 | OTHER | 12 / 0.16666666666666666 | 11 |
| 1730 | 0.1 | 0.07818099204450846 | 0.09490740740740741 / 0.7175925925925926 / 0.1875 | 6.174413268594149 | 6.185185185185185 | OTHER | 13 / 0.18055555555555555 | 11 |
| 1731 | 0.0 | 0.01439695991575718 | 0.0 / 0.17824074074074073 / 0.8217592592592593 | 7.5755595256877255 | 7.643518518518518 | OTHER | 9 / 0.125 | 9 |
| 1731 | 0.03 | 0.017601857039456565 | 0.0023148148148148147 / 0.3761574074074074 / 0.6215277777777778 | 7.122825561146156 | 7.238425925925926 | OTHER | 13 / 0.18055555555555555 | 12 |
| 1731 | 0.1 | 0.05551074305549264 | 0.07523148148148148 / 0.7743055555555556 / 0.15046296296296297 | 6.139337889684935 | 6.150462962962963 | OTHER | 13 / 0.18055555555555555 | 12 |

### Boundary and next action

No latency, memory, transfer, throughput, energy, or hardware-cost measurement
was recorded. No adaptive search, warm start, data/seed replacement, S07
conversion, teacher/base training, non-router optimizer membership, S08
change, six-bit on-demand loading, production-lambda selection, or follow-up objective
work occurred. The next action is a later, separately frozen refinement
protocol. Do not choose a production lambda or begin refinement execution from
this closeout.
