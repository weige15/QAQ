# Implement the trainable soft router

_Legacy work-item reference: S06._

Legacy identifiers elsewhere in this record are retained only for historical cross-reference to frozen decisions, evidence, paths, and machine-facing contracts.

## Gate result

**Status: COMPLETE — CONTINUE.** The trainable soft routers (legacy work item S06) produce finite two-way
probabilities, execute both verified packed precision paths, preserve the S05
`same_unit` prompt-only feature timing, and receive gradients while the packed
model stays frozen. S11-A documents the separate attention-only lookahead
timing.

## Router architecture

- Feature dimension: the model hidden size, `2560` for Qwen3-4B.
- Ownership: one distinct router per layer and routing unit; no cross-layer or
  attention/FFN sharing.
- Router count: `36` attention routers plus `36` FFN routers, `72` total.
- Architecture: parameter-free feature RMS normalization, `Linear(2560, 128)`,
  GELU, and `Linear(128, 2)`.
- Hidden width: `128`.
- Activation: GELU.
- Normalization: `feature / sqrt(mean(feature**2) + 1e-6)`, accumulated in
  float32, with no trainable parameters.
- Temperature: fixed configurable positive scalar, baseline `1.0`; no schedule.
- Initialization: PyTorch `nn.Linear.reset_parameters()` defaults, controlled
  by the caller's deterministic seed.
- Candidate ordering is canonical and explicit: output index `0` is 4-bit and
  output index `1` is 8-bit.
- Full Qwen3-4B router parameter count: `23,620,752`.

For `same_unit`, the S05 `masked_mean_pool` feature is detached at the router
boundary and computed from the incoming attention or FFN hidden states using
only the prompt mask. The optional S11-A attention-only timing is documented
in [`S11_LOOKAHEAD_ROUTING.md`](S11_LOOKAHEAD_ROUTING.md).

## Soft packed execution

For every attention or FFN unit, one probability tensor is computed and passed
unchanged to all projections in that unit. The packed boundary executes both
real pinned-backend calls:

```python
y4 = packed_linear(inputs, precision=4)
y8 = packed_linear(inputs, precision=8)
y = p4 * y4 + p8 * y8
```

The probabilities are cast only to the packed output dtype for numerical
compatibility, preserving autograd to the router. No unpacked persistent model
weights, hard argmax route, distillation, cost penalty, dataset training, or
CPU-to-GPU loading was added.

## Evidence

The deterministic unit and tiny-Qwen3 tests passed:

```text
source ~/.venv/bin/activate
which python                         # /nfs/home/s314511048/.venv/bin/python
python --version                     # Python 3.12.3
pytest -q tests/unit/test_s06_router.py tests/integration/test_s06_soft_routing.py
                                     # 13 passed
```

The real pinned packed backend endpoint and probability-gradient checks passed:

```text
source ~/.venv/bin/activate
which python
python --version
pytest -q tests/integration/test_s06_soft_packed.py -k 'not qwen3'
                                     # 2 passed
```

The full Qwen3-4B artifact endpoint test passed on `cuda:3`, using the supplied
S03-B artifact and the actual packed 4-bit/8-bit calls:

```text
QAQ_S03_ARTIFACT=<S03-B artifact> QAQ_MODEL_DEVICE=cuda:3 \
pytest -q tests/integration/test_s06_soft_packed.py
                                     # 3 passed in 419.02s
```

Both forced endpoints matched their corresponding S03/S04 hard executions
within `atol=1e-3`, `rtol=1e-3`; the direct synthetic pinned-backend endpoints
were bitwise equal. All 252 soft projection calls were observed.

Probability and temperature evidence:

- One-feature and batched router shapes were `[2]` and `[3, 2]`.
- All probabilities were finite, non-negative, and summed to one within
  `1e-6` in the router tests and `1e-5` at the packed boundary.
- A fixed logits pair `[2, 0]` produced greater concentration toward index 0 at
  temperature `0.5` than at `2.0`.
- Zero and ordinary feature vectors remained finite under normalization.

Sharing evidence:

- Each attention layer had exactly four soft records for q/k/v/o with one
  shared probability tensor object.
- Each FFN layer had exactly three soft records for gate/up/down with one
  shared probability tensor object.
- The full tiny model observed `252` soft calls and no hard precision calls.

Gradient and freeze evidence:

- The deterministic soft-mixture loss produced finite, nonzero gradients on
  router parameters.
- The trainable audit reported only names under `routers.`.
- Full tiny-model router count was `72`; the parameter-count formula matched
  the configured architecture.
- One SGD step changed router parameters and changed no frozen model
  parameter.
- Packed weights, embeddings, normalizations, output head, lookup tables, and
  quantization metadata were all frozen; non-router gradients were absent.

Regression evidence:

```text
pytest -q tests/unit
                                     # 67 passed
QAQ_S03_ARTIFACT=<S03-B artifact> QAQ_MODEL_DEVICE=cuda:3 \
pytest -q tests/integration/test_s04_manual_routing.py \
  tests/integration/test_s05_manual_routing.py \
  tests/integration/test_static4_forward.py \
  tests/integration/test_static8_forward.py
                                     # 12 passed in 431.49s
```

S05 prompt-feature, request-isolation, and tiny lifecycle tests remained
passing in the targeted regression run. No real dataset, router training run,
distillation, S07 hard routing, or on-demand loading was performed.

## Limitations and next action

This work item proves router structure, probability behavior, differentiable soft
execution, gradient flow, and frozen-model behavior only. It does not prove
learned routing quality. The soft wrapper intentionally supports prefill
training only; production hard argmax inference belongs to S07.

The trainable-soft-router gate (legacy work item S06) is complete. The current repository objective and follow-up action are
authoritative in [`docs/STATUS.md`](../STATUS.md); S07-A has since completed
its reusable distillation smoke machinery, and S07-B remains.
