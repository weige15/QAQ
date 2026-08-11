# S01 — Validate the Any-Precision backend

## Outcome

**Status: COMPLETE — CONTINUE to S02.**

S01 validates one small synthetic packed linear operation at exactly 4 and 8
bits. It does not load Qwen3, model weights, a dataset, or any full model.

## Pinned source and actual API

- Upstream: `https://github.com/SNU-ARC/any-precision-llm.git`
- Exact gitlink/source commit: `a3257d02740cc5757c78673da534b0630ff3a4ea`
- Source checkout status: clean; `third_party/` was initialized only to populate the already-pinned gitlink and was not modified.
- `AnyPrecisionLinear` constructor: `AnyPrecisionLinear(in_features, out_features, supported_bits, bias=True, precisions=None, device=None, dtype=None)` (`any_precision/modules/AnyPrecisionLinear.py:10-38`).
- The constructor allocates `qweight` as `torch.int32` with shape `(max(supported_bits), out_features, in_features // 32)` and one `float16` `lut{bit}` buffer with shape `(out_features, 2**bit)` per supported bit.
- `forward(x, precision=bit)` is the execution-time interface. For `M <= 8`, the pinned implementation invokes `matmul_kbit`; larger flattened batches use the pinned `dequant_kbit` helper followed by `torch.matmul` (`AnyPrecisionLinear.py:54-69`).
- The CUDA extension exports `matmul_kbit(in, qweight, lut, w_bits)` and `dequant_kbit(qweight, lut, w_bits)` (`modules/kernels/main.cu:60-147`). Both kernels assert 3–8 bit precision; S01 exercises only 4 and 8.
- The pinned pack helper constructs parent-bit bitmaps and applies `_permute_bitmaps_int32` (`quantization/pack.py:90-112`). The S01 test uses that helper as an opaque pinned implementation detail; it does not independently characterize plane order, padding, signed encoding, or serialization endianness. Those questions were deferred to S02 and are resolved in [`docs/BITPLANE_FORMAT.md`](../BITPLANE_FORMAT.md).
- The pinned source supports an optional bias buffer. S01 constructs `bias=False`, so no bias is added to either output.

## Deterministic synthetic operation

The operation is deliberately larger than a toy scalar but remains small:

| quantity | value | rationale |
| --- | ---: | --- |
| batch `M` | 4 | stays on the pinned packed matmul path (`M <= 8`) and exercises its four-row mode |
| output `N` | 64 | divisible by `num_rows * multi_row = 16`, giving four CUDA grid blocks |
| input `K` | 1024 | divisible by 32, giving 32 packed `int32` words per output row and one full kernel tile |
| seed | 1729 | fixed CPU generator seed for source weights and inputs |
| input/source dtype | `float16` inputs; `float32` synthetic source weights | inputs match the pinned kernel; source weights are only the deterministic quantization source |
| LUT dtype | `float16` | required by the pinned extension |
| packed dtype | `int32` | required by the pinned extension and physically packed representation |

The same generated source matrix and input matrix feed both precision
representations. The test-only synthetic construction derives nested 8-bit
parent labels and 4-/8-bit row LUTs, then calls the pinned packing helper. It
does not claim to be the project's production quantizer.

## Reference and tolerance

For each precision, the reference calls the pinned `dequant_kbit` helper on the
same CUDA `qweight` and selected LUT, then computes
`inputs @ dequantized_weight.T` in `float16`. This is the documented pinned
dequantization helper, not a second guessed bit-plane decoder.

The tolerance was fixed before judging outputs: `atol=0.05`, `rtol=0.01`.
This allows the expected difference between the pinned CUDA kernel's `float16`
accumulation and the independent `float16` matrix multiply for `K=1024`, while
remaining tight relative to the observed multi-unit outputs. Meaningful
relative error excludes reference magnitudes below `0.01` to avoid division by
near-zero values.

## Measured evidence

Hardware: CUDA device 0, NVIDIA GeForce RTX 3090. Both outputs are shape
`[4, 64]`, dtype `torch.float16`, device `cuda:0`.

| precision | max abs error | mean abs error | meaningful max relative error | allclose | output digest |
| ---: | ---: | ---: | ---: | --- | --- |
| 4 | 0.00872802734375 | 0.00251007080078125 | 0.1104247123003006 | true | `7a7d75ef8b5a56ff91f230f4c60ac49df46cdead833bc3cf6d8af0be9d146001` |
| 8 | 0.01171875 | 0.0023452043533325195 | 0.048128340393304825 | true | `7b218306e70f434aca7a7101ff57d973f9ffc120c8a1ac7b5b08ffad9f6d121c` |

Output samples (first eight flattened values) were:

- 4-bit: `[0.35498046875, 6.76953125, -0.818359375, -1.3330078125, 5.796875, 1.4208984375, 0.998046875, 3.296875]`
- 8-bit: `[0.494140625, 7.25390625, -0.5244140625, -1.6728515625, 6.02734375, 1.474609375, 1.0458984375, 3.18359375]`

Reference output digests were `b1aa02114aaae971dbbc89bb7edc7f48cc2fdd6d7aab0bca2f978c982979b6c9`
(4-bit) and `e7f32bb35ac0cc7bef7859a58c366098cfceaa3fd397549b197ffd56697d291c`
(8-bit).

### Storage observations

| object | shape | dtype/device | bytes |
| --- | --- | --- | ---: |
| full `qweight` | `[8, 64, 32]` | `int32` / `cuda:0`, contiguous | 65,536 |
| selected 4-bit packed prefix | four parent planes | `int32` / `cuda:0` | 32,768 |
| selected 8-bit packed planes | eight parent planes | `int32` / `cuda:0` | 65,536 |
| `lut4` | `[64, 16]` | `float16` / `cuda:0` | 2,048 |
| `lut8` | `[64, 256]` | `float16` / `cuda:0` | 32,768 |

The 8-bit qweight suffix was nonzero. Pinned-helper dequantization produced
distinct effective weights: 4-bit digest
`57f335dd77034c2ff18cb29c4c6335f0fc6ae50687b942744328b37c65d05cd7`, 8-bit
digest `3d2891837cdb60ab41e003917f862aed6f427c9cdde32464e1a44f7ce8f84602`,
with maximum absolute difference `0.04443359375`. This establishes distinct
precision paths from API, LUT/storage, selected parent-plane, and effective
dequantized-weight evidence; it does not rely on nondeterministic output
differences.

Repeated 4-bit executions were bitwise equal with digest
`7a7d75ef8b5a56ff91f230f4c60ac49df46cdead833bc3cf6d8af0be9d146001` on both
runs. Repeated 8-bit executions were bitwise equal with digest
`7b218306e70f434aca7a7101ff57d973f9ffc120c8a1ac7b5b08ffad9f6d121c` on both
runs. Flipping one packed qweight word changed the 4-bit output and restoring
the word restored the original path.

## Exact validation commands

Every command was run after:

```bash
source ~/.venv/bin/activate
which python
python --version
```

The interpreter was `/nfs/home/s314511048/.venv/bin/python`, Python 3.12.3.
The ordered results were:

```text
pytest -q tests/unit/test_backend_import.py                         PASS: 1 passed
pytest -q tests/unit/test_single_linear_precision4.py                PASS: 2 passed
pytest -q tests/unit/test_single_linear_precision8.py                PASS: 1 passed
pytest -q tests/unit/test_cuda_vs_dequantized_reference.py           PASS: 1 passed
pytest -q tests/unit/test_deterministic_output.py                    PASS: 2 passed
pytest -q tests/unit/test_backend_import.py tests/unit/test_single_linear_precision4.py tests/unit/test_single_linear_precision8.py tests/unit/test_cuda_vs_dequantized_reference.py tests/unit/test_deterministic_output.py
                                                                    PASS: 7 passed
python scripts/validate_backend.py                                  PASS: JSON evidence emitted; exit 0
ruff check src/qaq tests/unit/test_backend_import.py tests/unit/test_single_linear_precision4.py tests/unit/test_single_linear_precision8.py tests/unit/test_cuda_vs_dequantized_reference.py tests/unit/test_deterministic_output.py scripts/validate_backend.py
                                                                    PASS: clean
```

## Known limitations and gate

- This is a backend validation only. Qwen3 integration, model loading, router behavior, and production quantization remain unimplemented. S02 physical bit-plane characterization is complete and documented in [`docs/BITPLANE_FORMAT.md`](../BITPLANE_FORMAT.md).
- The reference trusts the pinned `dequant_kbit` helper because it is present; S01 does not duplicate or experimentally infer its internal bit-plane mapping.
- The CUDA extension was already built and installed during S00; S01 validates execution rather than rebuilding the upstream source.
- Results are empirical for the recorded environment and RTX 3090. CUDA-unavailable environments must fail the tests explicitly and are not a pass.

The S01 CONTINUE condition is satisfied: import, both packed precisions, both
independent helper-based references, both determinism checks, distinct
precision-path evidence, physical storage observations, and no-model scope all
pass. S02 subsequently resolved the deferred physical-format questions; the
current stage and next action are tracked in [`docs/STATUS.md`](../STATUS.md).
