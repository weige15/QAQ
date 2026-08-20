# S11-A — One-Unit-Lookahead Attention Routing Semantics

## Gate result

**COMPLETE — semantics only.** S11-A adds an explicit request-owned routing
mode with `same_unit` as the unchanged default and
`lookahead_attention_one_unit` as the attention-only variant. No quality
pilot, Qwen3-4B model load, training run, production checkpoint, lambda
selection, or hardware/resource measurement is part of this stage.

## Hypothesis and evidence boundary

The implementation hypothesis is that an attention decision for target layer
`t=s+1` can be made from source layer `s` after attention, residual addition,
and post-attention normalization, while preserving target ownership and all
other established routing semantics. S11-A validates that execution ordering
and state contract only. It does not establish equal or better quality, useful
overlap, reduced latency, reduced transfer, or superiority over `same_unit`.

The source papers do not establish this exact timing. The timing mode, source
point, layer-0 fallback, and attention-only scope are implementation choices
recorded in D055.

## Control and lookahead timing

`same_unit` remains byte-for-byte the default mode. Attention continues to use
its same-layer normalized input feature and FFN continues to use its same-layer
post-attention feature. Existing S05/S06 numerical outputs and trace sequences
remain unchanged.

For `lookahead_attention_one_unit`:

- layer-0 attention uses the unchanged same-layer fallback;
- for each source layer `s=0..34`, the source attention executes, its residual
  addition completes, and `post_attention_layernorm` produces the established
  FFN input representation;
- that representation is masked-mean pooled over prompt tokens and detached;
- target attention router `s+1`, never source attention router `s`, is invoked;
- feature and hard route or soft probability are stored under target attention
  layer `s+1` with provenance;
- source FFN `s` is then routed and executed with unchanged same-layer timing;
- target layer `s+1` requires and consumes the stored decision before any
  target packed attention projection executes;
- layer 35 creates no target beyond the 36-layer model.

The semantic schedule is:

```text
source attention s executes
  -> source attention residual completes
  -> post-attention normalization at s
  -> feature(source=s, point=post_attention_pre_ffn)
  -> target attention route/probability(t=s+1) available
  -> same-layer FFN s executes
  -> target layer t enters
  -> target attention decision t consumed
  -> packed target attention t executes
```

This provides exactly one source-FFN execution window of semantic lead time.
There is no asynchronous transfer, prefetch, overlap, scheduler, cache, or
measured benefit.

## Request-owned state and provenance

`QaqRequestState.routing_timing` accepts exactly `same_unit` and
`lookahead_attention_one_unit`. Features, probabilities, hard routes,
provenance, and one-time consumption flags remain owned by the concrete
request state; textual request IDs are still metadata and no process-global
registry exists.

Route/probability ownership is `(target_layer, target_unit_type)`. Lookahead
feature provenance is `(source_layer, source_point)`, represented by all five
fields:

```text
source_layer
target_layer
target_unit_type = attention
source_point = post_attention_pre_ffn
routing_timing = lookahead_attention_one_unit
```

For every lookahead record, `target_layer == source_layer + 1`. Target-layer
execution consumes but does not overwrite the stored feature or decision.
Duplicate prediction, duplicate consumption, absent provenance, and missing
early route/probability fail closed. `end_request()` runs registered cleanup
and releases request-owned features, decisions, provenance, and consumption
state.

## Hard, soft, and decode behavior

Hard prefill invokes the target-layer policy once at the source point, retains
the explicit `(4,8)` or `(4,6,8)` candidate order, and uses the unchanged
first-maximum `argmax` mapping. The target consumes the selected bit once and
does not invoke the policy again.

Soft prefill invokes the target `SoftPrecisionRouter` once at the source point.
The detached source feature prevents target-router gradients from flowing back
into the source attention router through hidden-state production. The stored
probability clone remains attached to the target-router graph. Packed
candidates execute only when target attention executes. Request-level bit-cost
reduction sees exactly 36 target-owned attention and 36 same-layer FFN
probabilities, each once; no objective or cost definition changed.

Decode computes no feature, invokes no router or policy, and reuses all 72 hard
prefill routes unchanged. Soft decode remains unsupported, as before.
Quantized packed/model parameters remain frozen during router optimization.

## Deterministic trace contract

Lookahead provenance events use this exact per-source order for every
`s=0..34`:

1. `source_attention_execution`
2. `source_attention_residual_completion`
3. `lookahead_target_feature_computed`
4. `lookahead_target_route_available`
5. `source_ffn_execution`
6. `target_layer_entry`
7. `target_route_consumed`
8. `target_attention_execution`

Each event carries the five provenance fields above. The route-available event
precedes both the source FFN execution marker and its existing `unit_execute`
marker. These events establish ordering only; they do not claim concurrency or
performance.

## Verification

Focused deterministic tests cover timing-mode validation/defaults,
source-to-target mapping and target router identity, layer-0 fallback, final
layer bounds, target-indexed feature/decision/probability coverage, duplicate
and missing-decision failures, request isolation and cleanup, full provenance
event order, same-layer FFN calls, candidate order/tie behavior, decode reuse,
finite soft outputs and gradients, a real target-router optimizer update,
source-router gradient isolation, and frozen packed/base state.

The tiny real execution-path integration uses 36 Qwen3 decoder layers whose
252 routed projections are real physically packed pinned
`AnyPrecisionLinear` modules. It proves route 1 is produced during layer 0
before FFN 0, attention 1 consumes it, attention routers 0 and 1 are each
invoked exactly once with no substitution, all FFNs remain same-layer, all 72
decisions are present, all 252 packed calls execute, and repeated runs are
bitwise deterministic.

Regression coverage runs the exact listed S05/S06/decode/isolation tests, the
new tiny packed lookahead test, the full unit suite, Ruff on every changed
Python path, and `git diff --check`.

## Limitations and deferred alternatives

S11-A makes no quality or resource claim. It changes no layer-0 semantics,
FFN timing, decode reuse, router loss, masked KL, normalized bit cost,
candidate bits/order, packed kernels, Any-Precision source, loader transfer
policy, historical S10 evidence, or frozen model state.

The following alternatives are documented and deliberately not implemented:

- **A — boundary attention scheduling:** use a near/exact historical attention
  feature at a later boundary; this may preserve feature recency but offers
  little semantic lead time.
- **B — full-layer lookahead:** predict a later layer from an older full-layer
  representation; this offers more semantic lead time but increases feature
  age and quality risk.
- **C — early FFN prediction:** predict FFN before same-layer attention; this
  removes established same-layer attention information and changes a second
  routing semantic.

No alternative is claimed superior without a separately scoped measurement.
The next action is exactly: **Define and execute a separately scoped paired
quality pilot comparing same-unit routing with one-unit-lookahead attention
routing.** It was not started in S11-A.
