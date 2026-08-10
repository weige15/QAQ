# QAQ decision ledger

This ledger separates source-supported behavior from implementation choices.
Every entry below is an **implementation choice**, not a paper-established fact, unless a later source review adds direct evidence and cites the source explicitly.
Unspecified details must not be silently filled in.

## Initial implementation choices

### D001 — Any-Precision backend

**Choice:** Use the official Any-Precision LLM implementation as the starting nested-quantization, packing, and CUDA-kernel backend.
**Status:** Open until S00 verifies source, compatibility, and viability.
**Source basis:** Not established by this scaffold; source review must document what the papers do and do not specify.

### D002 — Pin upstream revision

**Choice:** Pin the exact Any-Precision commit before modifying or wrapping it.
**Status:** Open until S00 records the revision and provenance.
**Source basis:** Implementation-control requirement, not a paper-established fact.

### D003 — Supported routes

**Choice:** Initially support only 4-bit and 8-bit routes.
**Status:** Baseline scope.
**Source basis:** Implementation scope choice; not asserted as a limitation of the papers.

### D004 — Separate unit routes

**Choice:** Route attention and FFN separately. All projections inside one selected unit use the same precision.
**Status:** Baseline scope.
**Source basis:** Implementation choice unless a later source review establishes direct support.

### D005 — Non-quantized components

**Choice:** Keep embeddings, normalization, activations, KV cache, and output head in BF16/FP16.
**Status:** Baseline scope.
**Source basis:** Implementation choice; the source papers do not establish this exact component policy here.

### D006 — Route reuse

**Choice:** During prefill, route each attention or FFN unit using its incoming prompt hidden states; store the selected route and reuse it during decoding.
**Status:** Baseline scope.
**Source basis:** Implementation choice; exact route timing and reuse are not established by this scaffold.

### D007 — Prompt feature

**Choice:** Initially mean-pool only non-padding prompt positions for the router feature.
**Status:** Baseline scope.
**Source basis:** Implementation choice for an unspecified feature-construction detail.

### D008 — Router objective

**Choice:** Initially train using teacher-student logit distillation without a bit-width penalty.
**Status:** Baseline scope.
**Source basis:** Implementation choice; no cost penalty is permitted before baseline freeze.

### D009 — Hard routes

**Choice:** Convert soft routing to hard inference routing using argmax.
**Status:** Baseline scope.
**Source basis:** Implementation choice unless later source review finds direct support.

### D010 — On-demand storage and lifetime

**Choice:** In on-demand mode, CPU packed storage is authoritative; selected packed planes are synchronously transferred on first use and retained until that request ends.
**Status:** Baseline scope.
**Source basis:** Implementation choice; loading lifetime and authority are not established here.

### D011 — Batch size

**Choice:** Initial inference supports batch size one only.
**Status:** Baseline scope.
**Source basis:** Implementation scope choice.

### D012 — Baseline freeze boundary

**Choice:** Do not add asynchronous loading, prefetching, layer-dependency signals, token schedulers, or new router objectives before the baseline is frozen.
**Status:** Baseline boundary.
**Source basis:** Implementation-control choice, not a claim about the papers.

## Decision protocol

A worker must add a dated or commit-linked entry when a stage resolves an unknown or introduces a new assumption.
The entry must state the evidence, alternatives considered when material, consequence, and reversal path.
A stage cannot be declared complete while its required decision gate contains an unresolved blocker.
