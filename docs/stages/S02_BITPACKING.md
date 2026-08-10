# S02 — Specify and verify physical bit packing

## Goal

Experimentally establish plane order, packing layout, reconstruction, padding, signs, lookup representation, and measured byte counts.

## Tasks

- Derive the representation from the pinned backend and verify it experimentally.
- Specify plane order and bit significance for nested 4-bit and 8-bit routes.
- Specify word/byte packing, padding, alignment, signs, lookup values, and reconstruction.
- Compare packed reconstruction with an independent reference on representative shapes, including awkward dimensions.
- Measure serialized bytes and distinguish metadata from packed plane storage.
- Define a versioned contract in `docs/BITPLANE_FORMAT.md`.

## Tests

- Round-trip reconstruction passes for representative signed and unsigned cases.
- 4-bit and 8-bit nested relationships are verified.
- Padding and non-aligned shapes are covered.
- Measured byte counts match the serialized representation.
- A byte-per-bit oracle is used only for correctness, never for resource claims.

## Required outputs

- Normative bit-plane format contract.
- Packing/reconstruction tests and independent reference.
- Measured byte-count report.
- Updated decisions and status with all unresolved convention choices explicit.

## Known uncertainties

- Plane ordering, padding, sign conventions, lookup representation, and metadata layout are unknown until measurement.
- Backend terminology may not uniquely determine physical storage.

## CONTINUE condition

The physical representation is documented, experimentally verified, versioned, and sufficient for S03.

## PAUSE condition

The backend or hardware is unavailable for the measurements.

## REVISE condition

Measurements disprove a format assumption but support a corrected format within the same baseline scope.

## STOP condition

A trustworthy physical packing contract cannot be established, or resource claims would rely on an unpacked or fake representation.
