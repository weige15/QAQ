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

### D013 — Environment capture (S00, 2026-08-10)

**Observation:** The captured host has Python 3.12.3 in `/nfs/home/s314511048/.venv`, CUDA Toolkit 12.4 from `/usr/local/cuda-12.4/bin/nvcc`, GCC 12.4.0, eight NVIDIA GeForce RTX 3090 GPUs with 24,576 MiB each, and PyTorch 2.4.0+cu124. The PyTorch CUDA smoke check passed. `transformers` 5.12.1 is present.

**Preliminary prerequisite results:** Python 3.11 — **FAIL** because the active interpreter is 3.12.3; CUDA Toolkit 12 or newer — **PASS**; GCC 9 or newer — **PASS**. This is only a documented-prerequisite comparison and does not establish Any-Precision or extension-build compatibility.

**Evidence:** `docs/environment.json`; exact inspection commands are recorded there. No packages were installed or upgraded, no model was downloaded, no CUDA extension was compiled, and no quantization or implementation work was performed.

**Consequence:** The remaining S00 evidence is still required. Do not begin S01 or modify the environment as part of this capture.

**Reversal path:** Re-run the inspection after an explicitly authorized environment change; do not change the environment during this task.

## Decision protocol

A worker must add a dated or commit-linked entry when a stage resolves an unknown or introduces a new assumption.
The entry must state the evidence, alternatives considered when material, consequence, and reversal path.
A stage cannot be declared complete while its required decision gate contains an unresolved blocker.
