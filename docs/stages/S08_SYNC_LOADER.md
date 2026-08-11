# S08 — Synchronous on-demand loading

## Goal

Keep packed maximum-precision storage on CPU, transfer only selected packed planes synchronously, retain them for the request, and measure actual transfer bytes and GPU memory.

## S08-A — synchronous request-scoped packed-plane loader contract

**Status: COMPLETE — CONTINUE to a later S08 integration and measurement work unit.**
S08-A is the implementation subdivision introduced by this work. It proves the
loader contract on a small real pinned-backend fixture only; it does not
complete S08, run the Qwen3-4B on-demand evaluation, or make memory/latency
comparisons.

### Verified representation and transfer granularity

The fixture reuses the S01 deterministic Any-Precision representation:
`qweight` is CPU-authoritative, contiguous `torch.int32` with shape
`[8,64,32]`, and the row-wise `lut4` and `lut8` are CPU `torch.float16`
tensors with shapes `[64,16]` and `[64,256]`. This is the pinned nested
representation, not a new encoding or a dense weight materialization.

The pinned backend reads only leading planes for a selected precision. A
4-bit first use transfers `qweight[:4]` and `lut4`; an 8-bit first use
transfers `qweight[:8]` and `lut8`. The loader includes an optional bias when
one exists. A 4-to-8 upgrade transfers only the missing `qweight[4:8]` and
`lut8`, concatenating the new GPU suffix with the retained prefix. It does not
repeat the CPU transfer of the first four planes. Exact bytes are measured on
the actual destination tensors as `numel() * element_size()` for every event.

### Ownership, lifecycle, and evidence

Each `SynchronousPackedPlaneLoader` is owned by one concrete
`QaqRequestState`; textual request IDs are descriptive only. The request state
exposes `end_request()`, which releases the loader's retained GPU references.
No process-global cache exists. Each call records request ID and state identity,
projection identity, precision, CPU source, CUDA destination, first-use versus
reuse, exact bytes, and transferred buffer names, dtypes, and shapes without
recording weight contents.

The real packed S01 fixture passed focused tests for CPU authority, 4-bit and
8-bit first use, 4-bit reuse, fresh 8-bit use, 4-to-8 upgrade, cleanup,
post-cleanup rejection, duplicate textual IDs, invalid precision, finite
outputs, and resident-versus-on-demand correctness. Measured bytes were
`34,816` for 4-bit first use, `98,304` for fresh 8-bit first use, and `65,536`
for the incremental 4-to-8 upgrade. The resident and transferred outputs were
bitwise equal for both precisions under the existing S01 pinned-backend
execution criterion.

All copies are ordinary synchronous `.to(device=...)` operations followed by
`torch.cuda.synchronize`; no non-blocking copies, CUDA streams, futures,
prefetching, or background workers are present. This fixture-level result does
not establish allocator release behavior, full-model transfer totals, memory
savings, or latency benefit.

## Tasks

- Make CPU packed storage authoritative for on-demand mode.
- Transfer only the selected packed planes synchronously on first use.
- Retain selected packed planes until that request ends, then release request state.
- Transfer packed planes, never unpacked weights.
- Support batch size one only and do not add cross-request caching.
- Measure actual transfer bytes, GPU memory, and latency.

## Tests

- First use transfers the selected packed planes exactly once for a request.
- Later use within the request reuses retained planes.
- Request end releases request-scoped GPU plane storage.
- A later request does not see prior request data.
- Transfer byte counts come from physically packed buffers.
- No asynchronous transfer, prefetch, prediction, or token scheduler is present.

## Required outputs

- Synchronous loader and request-lifetime tests.
- Actual transfer-byte and GPU-memory report.
- Latency comparison against resident routed mode.
- Updated decisions and status.

## Known uncertainties

- Exact CPU storage layout, transfer granularity, allocator behavior, and measurement instrumentation remain to be verified.
- Request-end cleanup behavior may depend on the target runtime.

## CONTINUE condition

Synchronous request-scoped loading transfers only selected packed planes, produces trustworthy measurements, and preserves correctness.

## PAUSE condition

Required GPU instrumentation or runtime support is temporarily unavailable.

## REVISE condition

A storage, lifetime, or measurement assumption must be corrected and retested.

## STOP condition

The loader transfers unpacked data, cannot enforce request lifetime, or requires asynchronous behavior before baseline freeze.
