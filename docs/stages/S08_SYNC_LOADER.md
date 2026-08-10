# S08 — Synchronous on-demand loading

## Goal

Keep packed maximum-precision storage on CPU, transfer only selected packed planes synchronously, retain them for the request, and measure actual transfer bytes and GPU memory.

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
