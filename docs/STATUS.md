Current stage: S01
Status: COMPLETE
Last passing commit: to be recorded immediately after the S01 evidence commit.

S00 is COMPLETE. Its environment, target identity, architecture mapping, and
Any-Precision provenance are recorded in the S00 documents. The exact pinned
Any-Precision gitlink is `a3257d02740cc5757c78673da534b0630ff3a4ea`.

Completed in S01:
- Imported the pinned `AnyPrecisionLinear`, `any_precision_ext.matmul_kbit`, and `any_precision_ext.dequant_kbit` interfaces.
- Validated one deterministic synthetic, physically packed linear operation at exactly 4 and 8 bits on an NVIDIA GeForce RTX 3090.
- Compared both packed CUDA outputs with the pinned `dequant_kbit` helper followed by an independent `torch.matmul` reference.
- Recorded packed storage, lookup-table storage, output samples, stable output digests, errors, tolerance, deterministic repeats, and distinct precision-path evidence.
- Added focused CUDA-required tests and `scripts/validate_backend.py`; no model, Qwen3 weights, dataset, or network-dependent path was used.

Validation outcome: S01 CONTINUE. Import, 4-bit execution, 8-bit execution,
both references, both determinism checks, packed-storage consumption, and
distinct precision-path evidence all pass.

Next action: Begin S02: experimentally specify and verify physical bit-plane packing.
