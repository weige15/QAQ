# S09 — Evaluate and freeze the baseline

## Goal

S09-A defines and freezes the reproducible five-mode evaluation protocol before
any final S09-B comparison results. S09-B will later execute the frozen
protocol and report quality, routing, memory, transfer, and latency results.

## Tasks

- Define the final evaluation matrix, datasets, prompts, seeds, and environment.
- S09-A: define the five required modes, identical applicable inputs, methods,
  release criteria, and a clean validation command.
- S09-B (not part of this work unit): run the five modes from the frozen config
  without changing inputs, gates, or mechanisms.
- Analyze failure cases and document limitations without broadening the baseline.
- Freeze the exact source revisions, configuration, model artifact references, commands, and result artifacts.
- Explicitly defer asynchronous loading, prefetching, transfer prediction, cost penalties, cross-request caching, multi-query batching, token schedulers, and unrelated improvements.

## Tests

- All five comparison modes execute reproducibly.
- Results can be regenerated from the recorded commands and deterministic seeds.
- Resource metrics distinguish packed storage from unpacked or fake references.
- The protocol parses, resolves the locked artifacts, and rejects incomplete or
  ambiguous variants without loading the five models.
- S09-B must later provide final quality and performance evidence against the
  frozen release criteria.

## Required outputs

- Complete comparison report in `docs/EXPERIMENTS.md` or a linked artifact.
- Frozen configuration and revision manifest.
- Limitations and deferred-work record.
- Final status update naming the passing commit and reproducibility evidence.

## Known uncertainties

- The final comparison values do not exist in S09-A by design.
- Hardware availability during S09-B is unknown; a missing or incomparable GPU
  is PAUSE, not a mixed-device result.
- The S07 adaptivity classification remains OTHER and has no new diversity
  threshold in this protocol.

## S09-A workflow subdivision — protocol freeze before results

**Status: IN_PROGRESS — protocol frozen before final results.** The machine-readable
owner is [`configs/s09_baseline_eval.json`](../../configs/s09_baseline_eval.json),
schema `qaq-s09-baseline-eval-v1`, with fixed inputs in
[`configs/s09_baseline_prompts.json`](../../configs/s09_baseline_prompts.json).
The protocol records `protocol_frozen_before_final_results: true`. Its SHA-256
is recorded in `docs/STATUS.md` and `docs/EXPERIMENTS.md` after each intentional
protocol edit; S09-B must use the committed value without modification.

### Evaluation matrix and locked identities

The matrix contains exactly these five modes, once each:

| Mode | Model/storage | Routing | Loader |
| --- | --- | --- | --- |
| full-precision BF16 teacher | `Qwen/Qwen3-4B` BF16 | none | resident |
| static packed 4-bit | exact S03 nested artifact | static 4-bit | resident |
| static packed 8-bit | exact S03 nested artifact | static 8-bit | resident |
| hard-routed resident packed model | exact S03 artifact + exact S07 router | hard query-level | resident |
| hard-routed synchronous on-demand packed model | exact S03 artifact + exact S07 router | hard query-level | synchronous request-scoped |

Every applicable mode uses model/tokenizer revision
`1cfa9a7208912126459214e8b04321603b3df60c`, identical input token IDs, and
identical generation settings. The packed checkpoint is the manifest artifact
with `pytorch_model.bin` SHA-256
`29d9bc526b3da0bd39daf2f82afd141f82d005ca1232cabc75cfe9d9ecc1cfee`.
The Any-Precision submodule is
`a3257d02740cc5757c78673da534b0630ff3a4ea`. Routed modes use the S07 router
checkpoint SHA-256
`08bf646f19759c0d7949e159bdbe4f96bbea737204b96f8760d205c8d6fd1949`.
Resident and on-demand routes must be identical; no soft final mode,
alternate router, alternate checkpoint, or sixth mode is permitted.

### Dataset and token-weighted quality sample

The protocol reuses the S03 evaluator path
`qaq.s03_quality.build_perplexity_windows` and
`qaq.s03_quality.evaluate_perplexity`. It freezes Salesforce/wikitext,
configuration `wikitext-2-raw-v1`, revision
`b08601e04326c79dfdd32d625aee71d232d685c3`, split `test`, and the pinned
tokenizer revision. The selector concatenates non-empty rows in source order,
uses no random sampling, and takes the first 32 windows with source length 129,
sequence length 128, and stride 128. Target ranges do not overlap, producing
exactly 4096 evaluated target tokens. Padding and generated tokens are
excluded. Labels are `window[1:]` for logits from `window[:-1]`; float32 loss
is accumulated as a token-weighted sum and divided by the exact target count.

The existing S03 default remains unchanged for its historical four-window,
stride-129 result. S09 invokes the same evaluator with explicit
`sample_count=32, stride=128` arguments; no divergent perplexity calculation
is introduced.

### Fixed prompts, route records, and generation

Seven fixed requests are committed: five S03 quality prompts plus the S07/S08
validation requests `validation-3` and `validation-1000`. Exact token IDs,
prompt lengths, source metadata, and the two 64-token validation inputs are in
the fixed-input file. Runtime prompt generation is forbidden. Each routed
request records complete 72-unit maps (36 attention and 36 FFN), attention/FFN/
overall 4-bit and 8-bit fractions, a canonical route-map digest, and an
observational prompt-to-prompt diversity summary. S07's OTHER limitation is
retained without a new diversity threshold.

Generation is batch-one greedy decoding with `do_sample=false`, `num_beams=1`,
temperature not applicable, and `max_new_tokens=8`. Every mode uses the same
fixed input IDs and settings. Results record input ID/digest, generated token
IDs, output digest, finite-value check, and normal termination. Text output is
descriptive only and receives no subjective score.

### Hardware, memory, transfer, and latency methods

All modes prefer one fixed CUDA device (the recorded S08 device is index 3,
NVIDIA GeForce RTX 3090). If that device class is unavailable during S09-B,
the outcome is PAUSE rather than a mixed-GPU comparison. An identical RTX 3090
is acceptable only with pre-recorded identity/comparability. Device index,
GPU/driver, CUDA runtime, PyTorch, Transformers, and Python versions are
recorded.

Each mode runs in a fresh process. For every request, record allocated and
reserved memory before measured execution, peak allocated/reserved, and
allocated/reserved after cleanup. Synchronize before and after every measured
CUDA interval and call `torch.cuda.reset_peak_memory_stats()` immediately
before the measured interval. Do not call `empty_cache()` inside it; any
between-run use is recorded. Allocated memory, reserved allocator memory,
physically resident packed bytes, and request-owned on-demand bytes remain
separate; reserved memory is not live residency.

For on-demand requests, record selected packed bytes immediately before request
end, retained entries/buffers before and after cleanup, retained bytes after
cleanup, and actual CPU-to-GPU packed transfer bytes. Transfer accounting
records first-use, reuse, prefill, decode, attention, FFN, total bytes/events,
and independently expected physical bytes. Expected bytes use the actual hard
route map, actual S08 buffer layout, and D029's selected-plane-plus-LUT rule;
actual bytes must equal the independent physical expectation. No complete
packed parent may be resident on the on-demand GPU path.

Latency uses one warm-up request, fully ending on-demand warm-up before
measurement, followed by five measured repeats for each fixed latency request.
All raw values are retained and the headline is the median; slow runs are not
removed and transfer time is not subtracted. Record synchronized prefill,
decode, and end-to-end latency; on-demand end-to-end includes transfers.

### Release criteria and deferred mechanisms

Structural/reproducibility failures are **REVISE**: any mode missing or
non-finite output, identity/hash mismatch, input mismatch, failed deterministic
repeat, route/output mismatch, transfer accounting mismatch, uncleared
request-owned references, hidden complete on-demand GPU copy, insufficient
rerun record, or relevant regression failure invalidates the comparison.

Quality gates are implementation gates, not paper-score claims:

- static 8-bit perplexity `<= 1.10 *` static 4-bit perplexity;
- routed resident perplexity `<= 1.10 *` static 4-bit perplexity;
- routed on-demand perplexity agrees with resident under the established
  execution-equivalence criterion (finite bitwise-equal logits, matching
  generated token IDs, and matching route maps).

The 10% margin is frozen. Performance is measurement validity, not a speed
target: synchronized comparable latency, allocator memory observations,
physical residency/transfer accounting, and no complete on-demand parent copy.
No memory-reduction percentage is required; exact reductions, if any, are
reported, and slower on-demand execution is a baseline limitation.

Asynchronous loading, prefetching, transfer prediction, bit-width cost
penalties, cross-request caching, multi-query batching, schedulers,
post-baseline optimization, soft final modes, and alternate routers/checkpoints
are explicitly deferred. If a genuine defect is found after S09-B starts, the
required outcome is REVISE with invalidation of affected results, never a silent
protocol edit.

### S09-A validation gate

The validator parses the config, checks exact five-mode completeness and
identity agreement with manifests/results, validates fixed inputs and sample
arithmetic, checks all release/quality/latency gates and forbidden mode
mechanisms, and resolves the practical model/artifact/checkpoint identities.
It does not load the five Qwen models or run the final comparison.

```text
source ~/.venv/bin/activate
which python
python --version
python scripts/validate_s09_protocol.py --config configs/s09_baseline_eval.json
```

## CONTINUE condition

S09-A continues to S09-B only after the protocol validator and focused tests
pass. S09-B may freeze the baseline only after the five-mode comparison
provides all required evidence and passes the frozen release criteria.

## PAUSE condition

A required evaluation resource or external artifact is unavailable.

## REVISE condition

A comparison or reproducibility defect can be corrected without adding pre-freeze mechanisms.

## STOP condition

The modes are not comparable, resource claims are not grounded in packed data, or the baseline cannot be frozen without unresolved critical assumptions.
