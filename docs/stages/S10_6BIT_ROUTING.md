# S10-A — Enable static 6-bit execution

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

S10-A establishes static 6-bit execution only. It does not add 6-bit routing,
change `CANDIDATE_BITS`, change hard-route choices, alter request loading or
state ownership, retrain or reload router checkpoints, or assess quality,
latency, transfer, or memory claims. Router semantics remain exactly 4/8
after S10-A. A separate later stage and decision is required for any 6-bit
routing behavior.

## S10-B — Three-Way Router Semantics

**Gate: CONTINUE.** S10-B adds explicit, backwards-compatible learned-router
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

**Historical next action:** Begin S10-C: define and validate the cost-aware
4/6/8 router objective. The completed S10-C gate is recorded below.

## S10-C — Cost-aware 4/6/8 router objective

**Gate: CONTINUE.** S10-C adds reusable objective composition primitives only;
it does not train or retrain a router and does not select a production cost
coefficient.

### Established facts and implementation choice

S07's established training objective remains completion-only teacher-student
`T^2 * masked KL(teacher || student)`. `masked_kl_distillation_loss()` is
unchanged and remains the sole KD implementation. S10-B establishes explicit
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
objective-only stage.

## S10-D — Bit-cost coefficient calibration

**Gate: CONTINUE.** S10-D calibrates observations for the S10-C coefficient; it
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
