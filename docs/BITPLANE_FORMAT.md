# QAQ bit-plane format contract v1

This is the S02 contract for the pinned Any-Precision source at
`a3257d02740cc5757c78673da534b0630ff3a4ea`.  It describes the physical
`qweight` payload and the row-wise LUTs used by that backend.  It is not a
claim that QAQ reproduces the authors' complete quantization pipeline.

The independent reference implementation is
[`src/qaq/quantization/reference_codec.py`](../src/qaq/quantization/reference_codec.py).
It is deliberately slow and may use byte/boolean arrays for correctness
checks only.  It is never production inference storage and must not support
memory, transfer, or latency claims.

## Verified facts

### Physical tensor contract

The pinned `AnyPrecisionLinear` constructor allocates:

| object | exact shape | dtype | meaning |
| --- | --- | --- | --- |
| `qweight` | `[P, N, K // 32]` | `torch.int32` | `P` parent bit planes, `N` output rows, and packed words along input `K` |
| `lut4` | `[N, 16]` | `torch.float16` | one direct lookup table per output row for 4-bit codes |
| `lut8` | `[N, 256]` | `torch.float16` | one direct lookup table per output row for 8-bit codes |
| optional `bias` | `[N]` | `torch.float16` | separate linear bias, not part of the packed payload |

For QAQ's nested 4/8-bit representation, `P=8`.  The CUDA entry points read
the `int32` tensor as `uint32_t*`, and derive the logical input width as
`K = qweight.size(2) * 32`.  An `int32` storage word therefore contains 32
physical plane bits, one bit for each of 32 logical code positions, although
the warp permutation described below makes those positions non-contiguous
within a word when a row is longer than 32.

The physical tensor is plane-major, then output-row-major, then storage-word
major.  It is C-contiguous, so its payload byte count is directly:

```text
qweight_bytes = qweight.numel() * qweight.element_size()
              = P * N * (K / 32) * 4
              = P * N * K / 8
```

The backend's `P=8` payload is physically packed.  A byte-per-plane-bit
oracle for the same shape would use `P*N*K` bytes and is reference-only.

### Plane order and code significance

The pinned packer (`any_precision/quantization/pack.py:90-99`) starts with
the most-significant parent-code mask and shifts it right for each plane.
Therefore:

```text
qweight[0] = code bit P-1  (MSB)
qweight[1] = code bit P-2
...
qweight[P-1] = code bit 0  (LSB)
```

For an 8-bit parent code `c8`, the 4-bit route consumes the leading four
planes and reconstructs:

```text
c4 = (c8 >> 4) = 8*b0 + 4*b1 + 2*b2 + b3
c8 = 128*b0 + 64*b1 + ... + 2*b6 + b7
```

The CUDA kernels load exactly `w_bits` plane rows (`dequant.cuh` and
`matmul.cuh`), and `AnyPrecisionLinear.forward(precision=...)` passes the
selected LUT and bit count to that kernel.  The 4-bit route is thus a prefix
of the same physical 8-plane parent payload; it is not a separately packed
suffix or a different sign encoding.

### Bit and byte ordering inside words

Before the backend's warp permutation, each plane/row is packed with
NumPy's MSB-first `packbits`: byte `B_j` has logical plane bits for positions
`8*j` through `8*j+7`, with position `8*j` in bit 7 and position `8*j+7` in
bit 0.

The source then applies `_permute_bitmaps_int32` (`pack.py:12-76`).  For each
block of `4W` logical bytes, where `W=32` for a complete warp block and
`W=(remaining_bytes/4)` for the final complete-word remainder, physical
storage word `t` has memory-order bytes:

```text
[B_(3W+t), B_(2W+t), B_(W+t), B_t]       (low address to high address)
```

On the verified little-endian host, the corresponding unsigned numeric word
is:

```text
q = B_t << 24 | B_(W+t) << 16 | B_(2W+t) << 8 | B_(3W+t)
```

For the smallest aligned `K=32` case (`W=1`), this reduces to logical
position `i` mapping to numeric word bit `31-i`.  Thus logical positions 0,
1, and 31 produce `0x80000000`, `0x40000000`, and `0x00000001` when set in
one plane.  The CUDA dequantizer's masks and shifts were compared with these
patterns and with full `K=64` and `K=1024` reconstructions.

### Reconstruction, signs, LUTs, and nesting

For precision `p`, output row `n`, and logical input position `k`:

```text
code_p[n,k] = sum(i=0..p-1, plane_bit[i,n,k] * 2**(p-1-i))
W_p[n,k]    = LUT_p[n, code_p[n,k]]
```

The pinned quantizer creates parent labels as `uint8` values, stores LUT
centroids as `float32` while quantizing, and saves them as `float16`
(`quantize.py:78-188,201-219`; `pack.py:104-112`).  The pinned packer uses
the direct row LUT values; it does not serialize a separate scale, zero
point, or sign plane.  Signed reconstructed values are simply negative or
positive floating LUT entries.  There is no invented sign plane and the
packed codes themselves are unsigned indices.

The actual pinned nested quantizer was run on a deterministic `(N,K)=(4,32)`
fixture with seed `20260811` and quantizer `random_state=1729`.  It produced
one shared parent-label tensor of shape `[4,32]` (returned internally as
`float32`, then converted to `uint8` for storage), a `[4,16]` 4-bit LUT, and
a `[4,256]` 8-bit LUT.  Reconstructing with `parent >> 4` and `lut4` versus
`parent` and `lut8` produced distinct signed-float matrices.  The LUTs are
not equal prefixes: each precision has its own centroids, including duplicate
8-bit entries where the pinned upscaler leaves a cluster empty.

### Padding and alignment

No padding is inserted by the pinned packer.  Its source path requires:

1. `K` divisible by 8 for the `packbits` result and reshape; and
2. `K/8` divisible by 4 for `_permute_bitmaps`, which is exactly `K`
   divisible by 32.

The aligned `K=32` and `K=64` fixtures were accepted.  Source-style probes
rejected `K=31` and `K=33` during `packbits` assignment, and rejected `K=40`
and `K=56` at the multiple-of-four-byte assertion.  The constructor itself
uses `in_features // 32` and therefore exposes a floor-sized qweight for a
non-aligned input instead of padding it; such a tensor is outside this
contract and must be rejected by QAQ validation rather than silently used.

### Serialization and endianness

The production pack path writes `torch.from_numpy(weighttensor)` into a
state dictionary, with each tensor's C-order dimensions `[plane,row,word]`,
then calls `torch.save` (`pack.py:154-177`).  This establishes the tensor
serialization order relevant to later `torch.load` use.  In a direct
`torch.save` experiment on the recorded environment, PyTorch emitted its
zip-based archive with:

```text
archive/byteorder = little
archive/version   = 3
```

The extracted `archive/data/0` payload exactly matched the tensor's direct
little-endian `int32` bytes.  For example, words
`[0x80000000, 0x40000000, 0xC0000001, 0xFFFFFFFF]` serialized as
`0000008000000040010000c0ffffffff`.

The physical contract is therefore explicitly little-endian for serialized
`int32` payloads.  The PyTorch archive is an implementation container, not a
new QAQ wire format: archive member names and container metadata remain
version-dependent, while the qweight tensor shape, dtype, C-order, and raw
payload order are the contract.

### Stable fixtures and observed words

All hashes below are SHA-256 over the C-contiguous `int32` qweight bytes on
the verified little-endian host.  The fixture has `N=4`, `K=32`, `P=8` and
repeats the row-0 pattern across all rows.

| fixture | row-0 words by plane (hex) | packed SHA-256 |
| --- | --- | --- |
| all zeros | `00,00,00,00,00,00,00,00` | `38723a2e5e8a17aa7950dc008209944e898f69a7bd10a23c839d341e935fd5ca` |
| all ones (`c8=255`) | `FFFFFFFF` on every plane | `e9175db65a9789096ca9cb5524d3abc2107df03e3c9ba3af1aca628f9c5d3bd2` |
| only positions 0,1,31 set in the LSB plane | `00,00,00,00,00,00,00,C0000001` | `5bbd124bb7a8d14b52ab941119dab7af0bc4b5c8da168d5de56f5d3b28b8c6d3` |
| alternating even logical positions in the LSB plane | `00,00,00,00,00,00,00,AAAAAAAA` | `7cf87f2f95dda7d35c75e2b031267703c7b169c93a4df2f29faa6e4aa75d8a16` |
| one populated plane (`c8=16`, plane 3) | `00,00,00,FFFFFFFF,00,00,00,00` | `60b8df3bc1cdfda128c6aa5a9300682f972b5aac5ab1b0339b952331febc4ea3` |
| adjacent planes 0/1 alternating | `AAAAAAAA,55555555,00,00,00,00,00,00` | `ed5fc93d086b4717f5537866a9428ab8757d1bee0ddf9635ef0d137dfbf38e77` |

The deterministic random fixture uses seed `20260811`, shape `[5,1024]`, and
has code digest
`476bcfc7f78e4bf89bec37847f6f3a49370fb5d09cba0dc28b6b30af50171334` and
packed digest
`1981743588b29dd277ab68f7f69bbdff7bce90e3d3ce4a40bc90066691d9a648`.

### Measured byte accounting

For the S01 fixture (`P=8`, `N=64`, `K=1024`, `float16` LUTs):

| object | formula | measured bytes |
| --- | --- | ---: |
| full qweight | `8*64*(1024/32)*4` | 65,536 |
| selected 4-plane payload | `4*64*(1024/32)*4` | 32,768 |
| selected 8-plane payload | `8*64*(1024/32)*4` | 65,536 |
| `lut4` | `64*16*2` | 2,048 |
| `lut8` | `64*256*2` | 32,768 |
| optional bias | `64*2` | 128 |

For the small known-word fixture (`N=4`, `K=32`), full qweight is 128 bytes,
the 4-plane prefix is 64 bytes, the 8-plane prefix is 128 bytes, `lut4` is
128 bytes, and `lut8` is 2,048 bytes.  The byte-count test verifies the
packed tensor directly as `tensor.numel() * tensor.element_size()` and keeps
LUT bytes separate from qweight bytes.

## Unknowns

- The exact full-model `torch.save` member order across every model/state-dict
  key is not measured here; only the source insertion behavior and direct
  qweight payload order are established.  QAQ must use `torch.load`/state-dict
  tensors rather than depend on archive member naming or offsets.
- Big-endian host behavior is not experimentally established.  The v1
  contract is intentionally little-endian and the reference codec rejects a
  big-endian host rather than guessing.
- Grouped LUT layouts (`group_count != 1`) are not part of this contract.  The
  pinned packer explicitly raises `NotImplementedError` for those layouts;
  S02 verifies the supported `group_count=1` row-wise LUT shape only.
- No Qwen3 tensor was loaded or enumerated in S02.  Whether every later target
  dimension satisfies this alignment contract remains a later-stage concern.

These unknowns do not make the physical 4/8-bit contract ambiguous for the
supported pinned path: plane order, bit order, reconstruction, alignment,
serialized tensor bytes, and byte accounting all have source and experiment
evidence.

## Implementation consequences

- Keep production qweight tensors as physically packed `torch.int32` planes.
  The reference codec's `uint8` codes and unpacked bitmaps are correctness
  oracles only.
- Treat `K % 32 != 0` as an input validation error.  Do not rely on the
  constructor's floor division as padding, and do not claim support for a
  silently truncated tensor.
- For a nested parent payload, select the leading `qweight[:4]` planes and
  `lut4` for 4-bit execution, or all eight planes and `lut8` for 8-bit
  execution.  Keep the two LUTs explicit and separate.
- Account for selected packed planes with `P*N*K/8` bytes and account for
  LUTs, biases, metadata, and temporary reconstructed weights separately.
  A reconstructed FP16 temporary returned by the backend is not the
  persistent packed representation.
- Later code that serializes or transfers weights must operate on packed
  planes/tensors, not on the reference codec's logical codes or unpacked
  weights.  No asynchronous loading, routing, model integration, or kernel
  optimization is introduced by S02.
