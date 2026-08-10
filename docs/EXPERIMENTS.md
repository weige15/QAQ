# Experiments

This file is an experiment-plan scaffold.
No experiments are run or claimed complete by the project-control setup.

## Required comparison at S09

Compare:

1. full-precision teacher;
2. static 4-bit model;
3. static 8-bit model;
4. routed resident mode;
5. routed synchronous on-demand mode.

Record quality, selected routes, GPU memory, actual packed transfer bytes, and latency.
Every result must include the exact command, environment versions, model and data identifiers, deterministic seed, and relevant configuration.

## Boundaries before baseline freeze

Do not introduce asynchronous transfers, prefetching, transfer prediction, bit-width cost penalties, cross-request caching, multi-query batching, or unrelated research improvements.
