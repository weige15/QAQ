Current stage: S03
Status: IN_PROGRESS

S00, S01, and S02 are COMPLETE. S03-A is complete with CONTINUE evidence:
- The exact pinned `Qwen/Qwen3-4B` revision `1cfa9a7208912126459214e8b04321603b3df60c` is cached locally.
- The full-precision BF16 `Qwen3ForCausalLM` loaded on `cuda:3` under Transformers `4.51.0`.
- The real module tree contains 144 attention and 108 FFN target projections, 252 total, matching the S00 mapping in names, classes, dimensions, biases, and layer indices.
- Embeddings, output head, normalizations, Q/K normalization, rotary processing, activations, and KV-cache structures remain excluded.
- Instantiated embedding/output-head tied storage was verified.
- The unmodified model passed the deterministic short full-precision smoke forward with finite logits.
- No quantization, Qwen module replacement, routing, or S04 work was performed.

Evidence: `docs/actual_model_modules.json` and `docs/stages/S03_STATIC_MODEL.md`.

Next action: S03-B: quantize the verified Qwen3 target modules into one nested 4-bit/8-bit packed representation.
