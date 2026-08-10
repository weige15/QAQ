# S01 — Validate the Any-Precision backend

## Goal

Build and test one deterministic packed linear operation at 4 and 8 bits against a reference.

## Tasks

- Use only the exact Any-Precision commit pinned in S00.
- Add the smallest project wrapper or adapter needed for one packed linear operation.
- Establish inputs, scales, signs, device, dtype, and shape conventions explicitly.
- Compare 4-bit and 8-bit packed operations against an independently defined reference.
- Keep production data physically packed and keep any byte-per-bit oracle test-only.

## Tests

- Deterministic 4-bit packed linear output matches the reference within documented tolerance.
- Deterministic 8-bit packed linear output matches the reference within documented tolerance.
- Shape, dtype, sign, and device edge cases are covered.
- The test proves packed input is consumed; it does not substitute fake quantization.

## Required outputs

- Minimal backend adapter and focused tests.
- Reference definition and tolerance rationale.
- Exact commands, seed, tensor shapes, and backend revision.
- Measured result report.
- Updated decisions and status.

## Known uncertainties

- The backend's actual public and internal interfaces are unverified until S01.
- Exact kernel support, dtype restrictions, and error tolerances remain unknown.
- The reference representation must not be confused with production packing.

## CONTINUE condition

Both deterministic 4-bit and 8-bit packed linear operations agree with the reference and the adapter's assumptions are recorded.

## PAUSE condition

A required backend build, CUDA resource, or upstream artifact is temporarily unavailable.

## REVISE condition

The backend interface or reference contract differs from the initial specification and can be corrected without changing the project objective.

## STOP condition

The pinned backend cannot provide a trustworthy packed 4/8-bit operation for the target environment, or correctness requires an unrecorded assumption.
