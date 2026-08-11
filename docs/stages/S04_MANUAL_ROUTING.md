# S04 — Manual attention/FFN precision plans

## Gate result

**Status: COMPLETE — CONTINUE.** The verified S03-B nested checkpoint executes
with immutable, explicit per-layer attention and FFN plans. All required route
isolation, mixed-plan, serialization, parity, and state-leakage checks passed.

This stage remains a resident packed-model baseline. It does not add query
features, a learned router, request-specific route generation, or CPU-to-GPU
on-demand loading.

## Locked prerequisites

- Model: `Qwen/Qwen3-4B`, revision
  `1cfa9a7208912126459214e8b04321603b3df60c`.
- Any-Precision revision:
  `a3257d02740cc5757c78673da534b0630ff3a4ea`.
- Model structure: 36 transformer layers, with four attention projections
  and three FFN projections per layer, for 252 packed targets.
- Verified S03-B artifact from `docs/quantized_model_manifest.json`:
  `quantized/s03b_qwen3_4b/backend_cache/packed/anyprec-(1cfa9a7208912126459214e8b04321603b3df60c)-w8_orig4-gc1-c4_s1_blk64`.
  The disposable worktree does not contain ignored model weights; tests use
  the artifact through `QAQ_S03_ARTIFACT` without modifying it.

## Implementation

`src/qaq/s04_manual.py` defines the immutable `PrecisionPlan`:

```python
PrecisionPlan(
    attention_bits=tuple[int, ...],  # exactly 36 entries
    ffn_bits=tuple[int, ...],        # exactly 36 entries
)
```

Construction and the defensive pre-forward validation both require tuples of
exactly 36 non-boolean integers, and every value must be exactly `4` or `8`.
Missing, extra, malformed, or unsupported entries raise a clear `TypeError` or
`ValueError`; values are never clamped or substituted. `PrecisionPlan` is a
frozen, slotted dataclass. `to_dict()` returns a JSON-compatible copy, while
`to_json()` uses sorted keys and compact separators. `from_dict()` and
`from_json()` reject missing or extra top-level fields and reconstruct tuples.

The manual loader first uses the existing S03 loader, then wraps each verified
`AnyPrecisionLinear` in a routed packed-linear boundary. The boundary requires
`precision=4` or `precision=8` on every call and forwards that explicit argument
to the pinned backend. It never calls `set_precision()` and stores no selected
plan on the model. Each `ManualRoutedQwen3ForCausalLM` call requires a
`precision_plan`; a new optional `PrecisionTrace` is created per call when one
is not supplied.

Propagation is explicit:

```text
PrecisionPlan
  -> _ManualBaseModel layer index
  -> _ManualDecoderLayer attention_bits / ffn_bits
  -> _ManualAttention or _ManualMLP selected_bits
  -> _RoutedPackedLinear(precision=selected_bits)
  -> AnyPrecisionLinear(..., precision=selected_bits)
```

`PrecisionTrace` is a per-forward collector, not model or module state. Every
packed call records `layer_index`, `unit_type`, `module_path`, and
`selected_bits`; trace comparison is independent of final logits.

## Route scopes and exclusions

For layer `i`, the attention route controls exactly:

- `model.layers.i.self_attn.q_proj`
- `model.layers.i.self_attn.k_proj`
- `model.layers.i.self_attn.v_proj`
- `model.layers.i.self_attn.o_proj`

The FFN route controls exactly:

- `model.layers.i.mlp.gate_proj`
- `model.layers.i.mlp.up_proj`
- `model.layers.i.mlp.down_proj`

The Qwen3 attention reshaping, Q/K RMS normalization, rotary processing,
attention operation, and KV-cache update are unchanged. Embeddings, tied
`lm_head`, final and per-layer RMS normalizations, activation/gating, and KV
cache remain outside the packed target set, as specified by S03. The manual
wrapper mirrors the verified Qwen3 forward path only to carry the plan through
the decoder layer; it does not replace or quantize excluded components.

## Exact manual plans

All plans below contain one entry per layer, in layer-index order `0..35`:

| Name | `attention_bits` | `ffn_bits` |
| --- | --- | --- |
| all-4 | `(4,) * 36` | `(4,) * 36` |
| all-8 | `(8,) * 36` | `(8,) * 36` |
| attention-8/FFN-4 | `(8,) * 36` | `(4,) * 36` |
| attention-4/FFN-8 | `(4,) * 36` | `(8,) * 36` |
| alternating | `4` on even layers, `8` on odd layers | `8` on even layers, `4` on odd layers |
| attention isolation | all `4`, except layer `7` is `8` | all `4` |
| FFN isolation | all `4` | all `4`, except layer `19` is `8` |

## Measured results

The measurement used the exact S03-B artifact, deterministic smoke inputs from
`smoke_inputs()`, `use_cache=False`, and CUDA device `cuda:3` (NVIDIA GeForce
RTX 3090). Manual and static models were loaded in one process. The
all-4/all-8 comparisons use underlying logits, not generated text.

### Serialization

`tests/unit/test_s04_precision_plan.py`: **7 passed**. Validation rejected a
list in place of a tuple, short and long route arrays, unsupported bit `6`,
boolean bits, missing fields, and extra fields. A 36-layer alternating plan
round-tripped through the canonical JSON representation, and repeated
serialization was byte-identical.

### Static parity

The comparison tolerance was selected before the measurement as `atol=1e-3`,
`rtol=1e-3`. Both manual results were exactly equal to their S03 static
counterparts, so measured mean and maximum absolute logit errors were zero:

| Plan | S03 static digest | Manual digest | Mean error | Max error | Trace |
| --- | --- | --- | ---: | ---: | --- |
| all-4 | `8b28d8ae1cf0d27462b0704d2661ebe90f67073c4435bbd8e21ad2ef19a6aa5d` | same | `0.0` | `0.0` | 252 exact calls |
| all-8 | `9337bad41bf1f9294aca8ba7721a313ad5abfe14e279970e2cf45142946f04c3` | same | `0.0` | `0.0` | 252 exact calls |

Both outputs were finite and passed the documented tolerance. The full
measurement process loaded in `415.39432963589206` seconds; this is checkpoint
I/O and model construction time, not a latency claim.

### Isolation and trace scope

- Changing only attention route layer `7` from 4 to 8 changed exactly four
  trace records, all in `model.layers.7.self_attn`, with paths `q_proj`,
  `k_proj`, `v_proj`, and `o_proj`. The final output changed and the changed
  trace matched the complete expected 252-call trace.
- Changing only FFN route layer `19` from 4 to 8 changed exactly three trace
  records, all in `model.layers.19.mlp`, with paths `gate_proj`, `up_proj`, and
  `down_proj`. The final output changed and the changed trace matched the
  complete expected 252-call trace.

### Mixed plans and state leakage

Attention-8/FFN-4, attention-4/FFN-8, and the exact even/odd alternating plan
all produced finite outputs, bitwise deterministic repeats, 252 trace calls,
and exact expected trace matches.

In one process, the sequence all-4 → all-8 → all-4 → attention-8/FFN-4 →
all-8 reproduced the first all-4 output exactly at the third position and the
first all-8 output exactly at the fifth position. Every call had the exact
plan-specific trace. This is the state-leakage gate; no cross-request or
sequential plan state was observed.

## Commands and regression evidence

All commands began with the required `~/.venv` activation block. The primary
S04 command was:

```bash
QAQ_S03_ARTIFACT=<verified-manifest-artifact> QAQ_MODEL_DEVICE=cuda:3 \
pytest -q tests/integration/test_s04_manual_routing.py
```

Result: **8 passed in 425.13s**. Ruff passed for all S04 source and tests.
The unit/S01/S02 regression command produced **43 passed, 5 skipped** (the
skips were artifact-dependent integration fixtures when the artifact path was
not supplied). With the artifact supplied, the S03 static and checkpoint
regression selection produced **13 passed in 641.97s** and no artifact skips.

The S04 implementation does not alter the pinned Any-Precision source,
physical packed storage, S03 checkpoint, or S01/S02 reference code.

## Limitations and next gate

This is an explicit resident manual policy only. It does not derive plans from
prompts, train or invoke a router, maintain request-specific route state,
transfer selected planes on demand, prefetch, batch multiple queries, or make
latency/memory-transfer claims. The adapter is specialized to the verified
Qwen3-4B 36-layer structure and the pinned Transformers/backend execution
path. S05 is the next stage and is intentionally not implemented here.

**Next action:** Begin S05: implement prompt-derived query features and
request-specific route state using a deterministic manual route policy.
