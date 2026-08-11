Current stage: S02
Status: COMPLETE
Last passing commit: `6bd50a1` (`qaq: verify S02 physical bit-plane format`).

S00 and S01 are COMPLETE. Their environment, target identity, architecture
mapping, backend provenance, and S01 CUDA evidence are recorded in the S00/S01
documents. The exact pinned Any-Precision gitlink remains
`a3257d02740cc5757c78673da534b0630ff3a4ea`.

Completed in S02:
- Inspected the pinned packer, linear module, CUDA entry points, dequantization
  masks, matmul indexing, and nested quantizer at the exact pinned revision.
- Experimentally established `int32` plane-major storage `[P,N,K//32]`,
  MSB-first plane order, the warp byte permutation, little-endian serialized
  payload order, and the strict `K % 32 == 0` alignment boundary.
- Verified leading-plane 4-bit selection from the shared 8-bit parent,
  direct row-wise LUT reconstruction, negative LUT values without a sign
  plane, distinct nested LUTs, and separate bias/LUT/metadata accounting.
- Added the slow independent reference codec with pack/unpack/reconstruct,
  deterministic known-word/random fixtures, source/backend agreement tests,
  serialization and byte-count tests, alignment tests, and a production
  packed-storage guard.
- Preserved and reran every S01 unit test. The explicit complete S01+S02 run
  passed 27 tests with 2 known upstream import deprecation warnings.

Validation outcome: S02 CONTINUE. The physical format is versioned in
`docs/BITPLANE_FORMAT.md`; all required ordering, reconstruction, prefix,
nested, alignment, serialization, byte accounting, reference/backend, and
production-storage evidence passed. Full-model archive member naming and
big-endian hosts remain documented non-contract unknowns.

Next action: Begin S03: create static 4-bit and 8-bit Qwen3 model baselines from one nested packed representation.
