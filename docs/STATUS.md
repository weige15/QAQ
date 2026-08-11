Current stage: S05
Status: COMPLETE

S00, S01, S02, and S03 are COMPLETE. S03-A and S03-B were complete with CONTINUE evidence:
- The exact pinned `Qwen/Qwen3-4B` revision `1cfa9a7208912126459214e8b04321603b3df60c` is cached locally.
- The full-precision BF16 `Qwen3ForCausalLM` loaded on `cuda:3` under Transformers `4.51.0`.
- The real module tree contains 144 attention and 108 FFN target projections, 252 total, matching the S00 mapping in names, classes, dimensions, biases, and layer indices.
- Embeddings, output head, normalizations, Q/K normalization, rotary processing, activations, and KV-cache structures remain excluded.
- Instantiated embedding/output-head tied storage was verified.
- The unmodified model passed the deterministic short full-precision smoke forward with finite logits.
- No query-derived routing, learned router, or on-demand loading was performed.

Evidence: `docs/actual_model_modules.json` and `docs/stages/S03_STATIC_MODEL.md`.

S03-B produced one verified nested 4-bit/8-bit packed artifact and passed its target, byte-accounting, reload, static smoke, and numerical sanity checks. Evidence: `docs/quantized_model_manifest.json`, `docs/stages/S03_STATIC_MODEL.md`, and `docs/EXPERIMENTS.md`.

S03-C static-baseline quality requirements are complete. Five fixed prompts, a 512-token WikiText-2 development perplexity sample, deterministic generation, fresh-process checkpoint reload, and full regression evidence all passed. Passing S03-C implementation commit: `842890d7580898db3846cb11c2d71f291579d1be`.

S04 manual routing is COMPLETE at passing implementation commit `a5802358acd756751d4006705ebea961a27b0f8c`. The immutable 36-layer attention/FFN plan, explicit packed-linear propagation, trace instrumentation, all-4/all-8 numerical parity, route isolation, mixed-plan determinism, serialization, and sequential leakage tests passed. Evidence: `docs/stages/S04_MANUAL_ROUTING.md`, `docs/EXPERIMENTS.md`, and `tests/integration/test_s04_manual_routing.py`.

S05 implementation is present in `src/qaq/model/request_state.py`,
`src/qaq/router/features.py`, and the S04 execution seam. Focused pooling,
padding, request-state, timing, no-completion-leakage, decode-reuse, and
request-isolation checks pass (`18 passed`, including a real tiny Qwen3
prefill/decode wrapper check). Artifact-dependent CUDA parity and full
cache-backed decode evidence is covered by the request route-reuse tests. The
corrected read-only packed artifact passed the unchanged S04 regressions and
S05 all-4/all-8 request-prefill parity (`10 passed in 422.84s`); S03 static
4-bit/8-bit regressions also passed (`2 passed in 218.23s`).

Passing S05 implementation commit: `7ec07e6`.

Next action: Begin S06: implement the trainable soft router and differentiable 4-bit/8-bit mixture path.
