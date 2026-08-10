# S03 — Static 4-bit and 8-bit model baselines

## Goal

Quantize the target model into one nested representation and validate static 4-bit and 8-bit inference.

## Tasks

- Use the S02 physical format and the pinned backend.
- Quantize the target model once into the nested representation.
- Keep embeddings, normalization, activations, KV cache, and output head in BF16/FP16 as specified by D005.
- Run static 4-bit and static 8-bit inference with deterministic inputs.
- Record quality, memory, latency, and serialization evidence with exact commands.
- Keep quantized weights frozen; no router training occurs in this stage.

## Tests

- Model conversion is deterministic and reproducible.
- Static 4-bit inference runs and meets the documented correctness gate.
- Static 8-bit inference runs and meets the documented correctness gate.
- Output and resource measurements use physically packed weights.
- Non-quantized component dtypes match the recorded decision.

## Required outputs

- Nested packed model artifact or reproducible generation command.
- Static 4-bit and 8-bit correctness and resource reports.
- Tests for representative model paths.
- Updated decisions and status.

## Known uncertainties

- Target model conversion compatibility and quality thresholds remain unverified.
- Exact evaluator and reference tolerances must come from S00 evidence.

## CONTINUE condition

Both static routes are reproducible, correct under the documented gate, and measured with real packed storage.

## PAUSE condition

Required model artifacts or evaluation resources are unavailable.

## REVISE condition

A conversion or dtype assumption needs correction while preserving the S03 objective.

## STOP condition

Static 4-bit or 8-bit inference cannot be made trustworthy, or the implementation would require fake packing or unfrozen quantized weights.
