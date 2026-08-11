# S02 — Specify and verify physical bit-plane packing

## Outcome

**Status: COMPLETE — CONTINUE to S03.**

The pinned Any-Precision backend maintains genuinely packed `torch.int32`
planes. Source inspection and deterministic CPU/CUDA experiments establish the
physical plane order, warp byte permutation, 4/8 prefix relationship, direct
LUT reconstruction, alignment boundary, serialized endianness, and byte
formulas. The complete contract is in [`docs/BITPLANE_FORMAT.md`](../BITPLANE_FORMAT.md).

No Qwen3 tensor, full packed model, router, loader, asynchronous path, kernel
optimization, or latency experiment was started. The pinned upstream source
was not modified.

## Gate evidence

- Current branch starts from S01 passing commit `8ea379e`; submodule HEAD is
  exactly `a3257d02740cc5757c78673da534b0630ff3a4ea` and clean.
- `AnyPrecisionLinear` source allocates `qweight` as `int32` `[P,N,K//32]`;
  pinned CUDA source consumes `w_bits` leading planes and row-wise `float16`
  LUTs `[N,2**w_bits]`.
- Known patterns on `(N,K)=(4,32)` match the backend: all zeros, all ones,
  positions 0/1/31, alternating positions, one populated plane, and adjacent
  planes. Example LSB-plane word: positions 0/1/31 -> `0xC0000001`.
- Direct pinned CUDA dequantization matches the independent codec for both 4
  and 8 bits, including signed LUT values. The 4-bit result equals the
  leading-plane code prefix `parent_code >> 4`.
- The actual pinned nested quantizer on deterministic `(4,32)` data shares one
  parent label tensor and emits distinct `[4,16]`/`[4,256]` LUTs; both
  reconstructions are distinct and digest-checked.
- `K=32`/`64` are accepted. Source-style packing rejects `K=31`, `33`, `40`,
  and `56`; no implicit padding is introduced. Non-aligned constructor floor
  division is outside the contract.
- Direct byte measurements match `tensor.numel()*tensor.element_size()`.
  For S01 `[8,64,32]` qweight: 65,536 bytes; selected 4-plane prefix:
  32,768 bytes; selected 8-plane payload: 65,536 bytes; LUT4: 2,048 bytes;
  LUT8: 32,768 bytes.
- `torch.save` evidence has `archive/byteorder = little`; extracted tensor
  bytes match contiguous little-endian `int32` bytes. Full archive naming is
  intentionally not treated as a portable wire format.

## Deterministic fixture digests

SHA-256 is over C-contiguous packed `int32` bytes on the recorded little-endian
host. Known-word fixture shape is `[8,4,1]`:

| fixture | digest |
| --- | --- |
| all zeros | `38723a2e5e8a17aa7950dc008209944e898f69a7bd10a23c839d341e935fd5ca` |
| all ones | `e9175db65a9789096ca9cb5524d3abc2107df03e3c9ba3af1aca628f9c5d3bd2` |
| positions 0/1/31 | `5bbd124bb7a8d14b52ab941119dab7af0bc4b5c8da168d5de56f5d3b28b8c6d3` |
| alternating | `7cf87f2f95dda7d35c75e2b031267703c7b169c93a4df2f29faa6e4aa75d8a16` |
| one populated plane | `60b8df3bc1cdfda128c6aa5a9300682f972b5aac5ab1b0339b952331febc4ea3` |
| adjacent planes | `ed5fc93d086b4717f5537866a9428ab8757d1bee0ddf9635ef0d137dfbf38e77` |

The random fixture uses seed `20260811`, codes shape `[5,1024]`, code digest
`476bcfc7f78e4bf89bec37847f6f3a49370fb5d09cba0dc28b6b30af50171334`, and
packed digest `1981743588b29dd277ab68f7f69bbdff7bce90e3d3ce4a40bc90066691d9a648`.

The pinned nested quantizer fixture uses NumPy seed `20260811` and
`random_state=1729`: parent digest
`5a31c268934a617b8b55d6198abeb35078a0214ff0d8ceb49fe26b55dde66010`, LUT4
digest `3e545d5c27ce357903878ddc1380e8e1c8edf776476aa4cbc9c318b39efe8204`,
and LUT8 digest
`500c4cbe06f7f4dfecca4f588a0f90797c1deff551f71630f09e8eeffbf20791`.

## Exact validation commands and results

Every Python, CUDA, and test command below was preceded by exactly:

```bash
source ~/.venv/bin/activate
which python
python --version
```

The interpreter was `/nfs/home/s314511048/.venv/bin/python`, Python 3.12.3.

The focused S02 sequence was:

```text
pytest -q tests/unit/test_pack_unpack_known_pattern.py                         PASS: 1 passed
pytest -q tests/unit/test_backend_known_patterns.py                             PASS: 1 passed
pytest -q tests/unit/test_pack_unpack_random.py                                  PASS: 1 passed
pytest -q tests/unit/test_plane_order.py                                         PASS: 2 passed
pytest -q tests/unit/test_prefix_precision.py                                    PASS: 1 passed
pytest -q tests/unit/test_padding.py                                             PASS: 9 passed
pytest -q tests/unit/test_packed_byte_count.py                                   PASS: 1 passed
pytest -q tests/unit/test_no_byte_per_bit_production_storage.py                  PASS: 1 passed
pytest -q tests/unit/test_reference_backend_agreement.py                         PASS: 1 passed
pytest -q tests/unit/test_nested_quantization_metadata.py                        PASS: 1 passed
pytest -q tests/unit/test_serialization_order.py                                 PASS: 1 passed
ruff check src/qaq/quantization tests/unit/test_pack_unpack_known_pattern.py tests/unit/test_pack_unpack_random.py tests/unit/test_backend_known_patterns.py tests/unit/test_plane_order.py tests/unit/test_prefix_precision.py tests/unit/test_padding.py tests/unit/test_packed_byte_count.py tests/unit/test_no_byte_per_bit_production_storage.py tests/unit/test_reference_backend_agreement.py tests/unit/test_nested_quantization_metadata.py tests/unit/test_serialization_order.py
                                                                                PASS: clean
```

The exact source/revision checks were:

```text
git rev-parse HEAD                                                     PASS: S01 base before S02 changes
git -C third_party/any-precision-llm rev-parse HEAD                    PASS: a3257d02740cc5757c78673da534b0630ff3a4ea
git -C third_party/any-precision-llm status --porcelain=v1              PASS: clean
```

The final complete S01+S02 command is recorded in this document immediately
after it passes:

```text
pytest -q tests/unit/test_backend_import.py tests/unit/test_single_linear_precision4.py tests/unit/test_single_linear_precision8.py tests/unit/test_cuda_vs_dequantized_reference.py tests/unit/test_deterministic_output.py tests/unit/test_pack_unpack_known_pattern.py tests/unit/test_backend_known_patterns.py tests/unit/test_pack_unpack_random.py tests/unit/test_plane_order.py tests/unit/test_prefix_precision.py tests/unit/test_padding.py tests/unit/test_packed_byte_count.py tests/unit/test_no_byte_per_bit_production_storage.py tests/unit/test_reference_backend_agreement.py tests/unit/test_nested_quantization_metadata.py tests/unit/test_serialization_order.py
                                                                                PASS: 27 passed, 2 warnings
```

## Unknowns and boundary

The full-model PyTorch archive member order and big-endian behavior remain
outside the contract; consumers must use loaded tensors and the explicit
little-endian v1 payload. Grouped LUTs are unsupported by the pinned packer.
These do not block the supported 4/8-bit physical contract or the S02 gate.

Next action after this commit: Begin S03 static 4-bit and 8-bit Qwen3 model
baselines from one nested packed representation. S03 is not executed here.
