# Experiments

This file records experiment plans and links to completed stage evidence.
S02's physical-format evidence is authoritative in
[`docs/stages/S02_BITPACKING.md`](stages/S02_BITPACKING.md) and
[`docs/BITPLANE_FORMAT.md`](BITPLANE_FORMAT.md).

## S01 pinned backend evidence (2026-08-11)

This is the only experiment recorded by S01. It uses no model, Qwen3 weight,
dataset, or network-dependent input.

- Command: `python scripts/validate_backend.py`, after the mandatory environment activation block in `docs/stages/S01_BACKEND.md`.
- Any-Precision source: `a3257d02740cc5757c78673da534b0630ff3a4ea`.
- Hardware: CUDA device 0, NVIDIA GeForce RTX 3090.
- Seed and dimensions: seed `1729`, `M=4`, `N=64`, `K=1024`.
- Dtypes: input/LUT/output `float16`; packed qweight `int32`; bias disabled.
- 4-bit result: output digest `7a7d75ef8b5a56ff91f230f4c60ac49df46cdead833bc3cf6d8af0be9d146001`; max absolute error `0.00872802734375`; mean absolute error `0.00251007080078125`; meaningful max relative error `0.1104247123003006`; allclose `true` with `atol=0.05`, `rtol=0.01`.
- 8-bit result: output digest `7b218306e70f434aca7a7101ff57d973f9ffc120c8a1ac7b5b08ffad9f6d121c`; max absolute error `0.01171875`; mean absolute error `0.0023452043533325195`; meaningful max relative error `0.048128340393304825`; allclose `true` with the same tolerance.
- Storage: full qweight `[8,64,32]` int32, `65,536` bytes; selected packed 4-bit prefix `32,768` bytes; selected 8-bit planes `65,536` bytes; LUT4 `[64,16]` float16, `2,048` bytes; LUT8 `[64,256]` float16, `32,768` bytes.
- Determinism: repeated 4-bit and 8-bit outputs were bitwise equal with the digests above.
- Distinct paths: nonzero 8-bit qweight suffix, different LUT shapes, and different pinned-helper effective-weight digests; maximum effective-weight delta `0.04443359375`.

## Required comparison at S09

Compare:

1. full-precision teacher;
2. static 4-bit model;
3. static 8-bit model;
4. routed resident mode;
5. routed synchronous on-demand mode.

Record quality, selected routes, GPU memory, actual packed transfer bytes, and latency.
Every result must include the exact command, environment versions, model and data identifiers, deterministic seed, and relevant configuration.

## S03-B nested Qwen3 static baseline (2026-08-11)

- Command: `source ~/.venv/bin/activate && which python && python --version && PYTHONPATH=src:third_party/any-precision-llm python scripts/run_s03b.py --overwrite-artifact`.
- Model: `Qwen/Qwen3-4B`, immutable revision `1cfa9a7208912126459214e8b04321603b3df60c`; tokenizer uses the same revision. Any-Precision commit: `a3257d02740cc5757c78673da534b0630ff3a4ea`.
- Mapping and targets: explicit `configs/qwen3_any_precision.yaml`; exact S03-A target-set check passed for 252 projections, with no omitted, unexpected, duplicate, or excluded targets.
- Quantizer settings: seed `4`, parent `8`, group count `1`, random state `1729`; pinned C4 train shard, one 64-token sample, tokenizer first-64-token truncation.
- Quantization runtime: `197.52182836900465` seconds on CUDA `cuda:3`, NVIDIA GeForce RTX 3090.
- Physical storage: packed parent planes `3633315840` bytes; selected 4-plane prefix `1816657920` bytes; selected 8-plane payload `3633315840` bytes; LUT4 `35389440` bytes; LUT8 `566231040` bytes; scales `0` bytes; lookup/scale/metadata `617504066` bytes; total artifact `5525158010` bytes.
- Static 4-bit: logits `[1,8,151936]`, finite, digest `8b28d8ae1cf0d27462b0704d2661ebe90f67073c4435bbd8e21ad2ef19a6aa5d`, peak allocated GPU memory `5585867264` bytes.
- Static 8-bit: logits `[1,8,151936]`, finite, digest `9337bad41bf1f9294aca8ba7721a313ad5abfe14e279970e2cf45142946f04c3`, peak allocated GPU memory `5588298240` bytes.
- BF16 full-precision comparison digest: `a59aa0c2a7d31a8e4a5e9687ce229f9fcaa461344d3ea68f506867355fd73a18`.
- FP-vs-4 error: mean/max absolute `0.38069865107536316` / `3.34375`. FP-vs-8 error: mean/max absolute `0.04913947731256485` / `0.6796875`.
- Artifact: `quantized/s03b_qwen3_4b/backend_cache/packed/anyprec-(1cfa9a7208912126459214e8b04321603b3df60c)-w8_orig4-gc1-c4_s1_blk64`; complete hashes are in `docs/quantized_model_manifest.json`.
- Fresh-process round-trip and manifest integration tests: `9 passed`. Relevant S01/S02 regression tests: `27 passed`. Ruff: clean.

## Boundaries before baseline freeze

Do not introduce asynchronous transfers, prefetching, transfer prediction, bit-width cost penalties, cross-request caching, multi-query batching, or unrelated research improvements.
