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

**Status: COMPLETE — S09-A validation gate passed; protocol frozen before final results.** The machine-readable
owner is [`configs/s09_baseline_eval.json`](../../configs/s09_baseline_eval.json),
schema `qaq-s09-baseline-eval-v1`, with fixed inputs in
[`configs/s09_baseline_prompts.json`](../../configs/s09_baseline_prompts.json).
The protocol records `protocol_frozen_before_final_results: true`. Its SHA-256
is recorded in `docs/STATUS.md` and `docs/EXPERIMENTS.md` after each intentional
protocol edit; S09-B must use the committed value without modification.

S09-A closeout evidence: PR #5 merge commit
`0f5802a777983c210b6f65ca26fd55368f49bf51` is present; the canonical validator
passed with hashes enabled and exited `0`; the packed artifact, S07 router,
Qwen3-4B model/tokenizer, and Any-Precision identities matched their frozen
values; and the focused protocol/input/evaluator suite passed `18 passed`.
No S09-B benchmark or final result artifact exists. S09-B execution machinery is
**MISSING**, so the next action is: implement the minimal S09-B evaluation
runner required to execute the frozen `configs/s09_baseline_eval.json`
contract, without running the final evaluation yet.

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
identity agreement with manifests/results, validates the complete routed
recording contract, fixed inputs, generation/seed policy, sample arithmetic,
physical transfer rule and expected-byte sources, live Any-Precision
submodule revision, fixed-GPU comparability identity, all release/quality/
latency gates, post-result invalidation policy, complete deferred-mechanisms
list, and forbidden mode mechanisms. It resolves practical model/artifact/
checkpoint identities without loading the five Qwen models or running the
final comparison.

```text
source ~/.venv/bin/activate
which python
python --version
python scripts/validate_s09_protocol.py --config configs/s09_baseline_eval.json
```

## S09-B1 runner implementation

S09-B1 is an implementation-only continuation of the frozen S09-A protocol.
It does not modify either frozen JSON file and does not produce comparison
results.

The entry point is `scripts/run_s09b.py`, backed by `qaq.s09_runner`.
The parent process validates the frozen protocol and launches one fresh
`--execute-mode <mode>` child for each mode.
The child owns one model and one per-mode JSON result, then exits.
No service, worker pool, persistent model process, or cross-request model cache
is introduced.

The five adapters are derived from the config in frozen order:
`full_precision_bf16_teacher`, `static_packed_4bit`,
`static_packed_8bit`, `hard_routed_resident_packed`, and
`hard_routed_synchronous_on_demand_packed`.
S03 loading, static precision selection, perplexity, and cleanup are reused;
S07 hard routing and checkpoint loading are reused; and S08's request-owned
synchronous packed loader and physical transfer sources are reused.

Each per-mode result uses schema `qaq-s09b-per-mode-result-v1` and records
provenance, frozen identities and hash, fixed input digests, S09 perplexity
setup and metrics, fixed greedy generation, memory boundaries, five raw
latency repeats, deterministic evidence, complete routed maps and fractions,
and on-demand transfer and cleanup evidence where applicable.
The aggregator validates all five result files and returns `PAUSE` for missing
external results, `REVISE` for structural, identity, quality, route, transfer,
cleanup, or deterministic failures, and `CONTINUE` only when all frozen gates
pass.

The safe non-executing command is:

```text
source ~/.venv/bin/activate && which python && python --version && python scripts/run_s09b.py --plan --config configs/s09_baseline_eval.json
```

It runs the existing protocol validator, resolves output locations, prints the
five exact child commands and final aggregation command, and writes no result.
The later execution command shape is:

```text
source ~/.venv/bin/activate && which python && python --version && python scripts/run_s09b.py --execute --config configs/s09_baseline_eval.json
```

The execution command is intentionally not run in S09-B1.
Focused runner tests are in `tests/unit/test_s09_runner.py` and
`tests/integration/test_s09_runner_plan.py`.
No final S09-B five-mode evaluation, quality result, memory result, latency
result, routing result, transfer result, or final artifact was produced.

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

## S09-B4 — narrow routed decode repair

S09-B2's routed result files are preserved but invalidated because D035
identified nondeterministic pinned `matmul_kbit` accumulation. D036 records the
source-verified dispatch and repair gate. The repair uses a shared helper for
resident and synchronous on-demand routed calls only: non-Orin, one effective
row, packed `K > 4096`, and precision at least 7. For the locked QAQ precisions,
this means 8-bit calls. It falls back to pinned `dequant_kbit` plus
`torch.matmul` with a temporary per-call CUDA weight. Static and full-precision
paths remain outside the diff, and no frozen protocol or final result file is
changed.

The Qwen3 inventory audit found 36 affected `mlp.down_proj` targets at
`in_features=9728`; the remaining 216 targets are `in_features=2560` and do
not satisfy the condition. Narrow validation covered only `s03-quality-3` and
`validation-3`, with eight decode steps and five repeated generations for
`s03-quality-3`. It passed bitwise resident/on-demand logits parity, route and
token parity, repeat stability, packed transfer equality, zero decode transfer,
request cleanup, and the hidden-copy audit. Corrected routed quality and
resource results remain unknown.

The next action is only to rerun the invalidated routed resident and routed
synchronous on-demand S09-B evidence. The five-mode final evaluation must not
be rerun as part of this repair.

## S09-C — final evidence review, baseline freeze, and completion

**Gate: CONTINUE.** The committed S09-B5 aggregation is `CONTINUE` with
`errors: []`. A read-only aggregation against a new temporary copy also
returned `CONTINUE` with `errors: []`. The closeout did not run another
benchmark or model mode.

### Canonical evidence and integrity

The canonical final evidence directory is `docs/results/s09b_b5/`.
The preserved `docs/results/s09b/` directory is failed S09-B2 evidence and is
not the canonical baseline. The final evidence commit is
`443f6994582500857afca9bad6032cc285448a86`; the routed repair commit is
`4a0dc702178fef0f84eb9ffd9bd6d1810e5dc564`.
The frozen protocol SHA-256 is
`01ca65c6b3b7e16d7af66f1533140b1c9f31749c90bc91e097d096d463bf2e1c`.
The fixed-input SHA-256 is
`da1d33f0f2330cfc341c38945fe4b205f946223f8c9069c35d44999d400fbb49`.
The validator passed with the five modes, seven requests, 32 windows, and
4096 evaluated target tokens.

Before documentation edits, every committed result JSON was hashed.
The hashes remain unchanged after the read-only aggregation:

```text
0f01a9097f921f4ae397d6c3497c1375cd17904e007887176fc3394c73425156  docs/results/s09b_b5/aggregation.json
2211305583baff321a35ae0d76848e9b26505164816e066fe4e5c0c7ce31cb60  docs/results/s09b/full_precision_bf16_teacher.json
2211305583baff321a35ae0d76848e9b26505164816e066fe4e5c0c7ce31cb60  docs/results/s09b_b5/full_precision_bf16_teacher.json
38a7e83df0f2f5155b1528e17b0c5ee1c6bbc3989c01fc2d93245c8434bf13fb  docs/results/s09b/aggregation.json
423f3f496c461ca872abf010d1746bcf5b88e3e28cf4973dc69f88467c811f9e  docs/results/s09b_b5/hard_routed_resident_packed.json
936b953faa4ad270c0f7ff4b59775e678c91baaa10bec048e0edcc8a12edb44c  docs/results/s09b_b5/static_packed_8bit.json
936b953faa4ad270c0f7ff4b59775e678c91baaa10bec048e0edcc8a12edb44c  docs/results/s09b/static_packed_8bit.json
9767c347366bec11bb15e1cede47c2eac29f646254dcebad2df008ddbdfee61a  docs/results/s09b_b5/static_packed_4bit.json
9767c347366bec11bb15e1cede47c2eac29f646254dcebad2df008ddbdfee61a  docs/results/s09b/static_packed_4bit.json
9958c47ec2fe2370f029cd12c77ed56cbf1d5a2ca817a8868b1deb97eeb1f214  docs/results/s09b_b5/hard_routed_synchronous_on_demand_packed.json
9fc9697f3e6c17be142ceaafd585c431bcd5da4001c746226fdcea3a03cf552f  docs/results/s09b/hard_routed_resident_packed.json
c04cc713edd8a13c51d2281e3a264df7097db7b3d3dad91b2e7d75bef789c8d6  docs/results/s09b/hard_routed_synchronous_on_demand_packed.json
```

The B5 diff from `4a0dc702178fef0f84eb9ffd9bd6d1810e5dc564` to
`443f6994582500857afca9bad6032cc285448a86` added only `docs/results/s09b_b5/`.
The execution-path diff across `scripts/run_s09b.py`, `src/qaq/s09_runner.py`,
`src/qaq/s03_quality.py`, and `src/qaq/s03_static.py` from the original B2
base through the repair is empty. Therefore the FP/static three-mode B2
results are valid for reuse; the two routed modes were invalidated and rerun.
The temporary aggregation's `results_dir` is execution provenance and was not
rewritten in the committed `aggregation.json`.

### Five-mode quality comparison

All values below are read from the committed B5 per-mode JSON files.

| Mode | Perplexity |
| --- | ---: |
| full-precision BF16 teacher | 30.648146290315317 |
| static packed 4-bit | 32.53290622283182 |
| static packed 8-bit | 30.57498909612196 |
| hard-routed resident packed | 30.678448224528175 |
| hard-routed synchronous on-demand packed | 30.678448224528175 |

`static8/static4 = 0.9398173310032849` and
`routed-resident/static4 = 0.9429974689134236`.
Both frozen quality gates pass. The routed resident and on-demand values are
identical.

All five modes had finite outputs and metrics. Every mode recorded five
agreeing deterministic repeats for each of the seven fixed requests.
Resident/on-demand route maps matched, generated token IDs matched, and all
seven resident/on-demand logits digests matched. Transfer equality, cleanup,
and hidden-copy checks also passed.

### Routing evidence

Resident and on-demand route statistics were identical:

| Unique maps | Changed units | Changed fraction | Mean pairwise distance | Classification |
| ---: | ---: | ---: | ---: | --- |
| 5 | 4 | 0.05555555555555555 | 0.022486772486772486 | OTHER |

Per-request fractions are attention 4/8, FFN 4/8, and overall 4/8:

| Request | Attention | FFN | Overall |
| --- | --- | --- | --- |
| s03-quality-0 | 0.2777777777777778 / 0.7222222222222222 | 0.1111111111111111 / 0.8888888888888888 | 0.19444444444444445 / 0.8055555555555556 |
| s03-quality-1 | 0.2777777777777778 / 0.7222222222222222 | 0.1111111111111111 / 0.8888888888888888 | 0.19444444444444445 / 0.8055555555555556 |
| s03-quality-2 | 0.25 / 0.75 | 0.08333333333333333 / 0.9166666666666666 | 0.16666666666666666 / 0.8333333333333334 |
| s03-quality-3 | 0.2777777777777778 / 0.7222222222222222 | 0.1111111111111111 / 0.8888888888888888 | 0.19444444444444445 / 0.8055555555555556 |
| s03-quality-4 | 0.3055555555555556 / 0.6944444444444444 | 0.1388888888888889 / 0.8611111111111112 | 0.2222222222222222 / 0.7777777777777778 |
| validation-3 | 0.3055555555555556 / 0.6944444444444444 | 0.1111111111111111 / 0.8888888888888888 | 0.20833333333333334 / 0.7916666666666666 |
| validation-1000 | 0.2777777777777778 / 0.7222222222222222 | 0.1111111111111111 / 0.8888888888888888 | 0.19444444444444445 / 0.8055555555555556 |

`OTHER` is descriptive and is not a release failure. It does not establish
route diversity beyond this observation.

### Memory and transfer evidence

Peak allocator values are the maxima across the committed request records.
Reserved allocator memory is not live packed residency.

| Mode | Peak allocated bytes | Peak reserved bytes | Physical packed residency |
| --- | ---: | ---: | ---: |
| full-precision BF16 teacher | 8125394944 | 8355053568 | 0 |
| static packed 4-bit | 5622764544 | 5811208192 | 4234936320 |
| static packed 8-bit | 5622764544 | 5811208192 | 4234936320 |
| hard-routed resident packed | 5726520832 | 5918162944 | 4234936320 |
| hard-routed synchronous on-demand packed | 4886706176 | 5167382528 | 0 |

On-demand peak request-owned packed bytes were `3900211200`.
Complete packed residency was zero. There were 252 retained entries and 504
retained buffers before cleanup, and zero entries, buffers, and bytes after
cleanup. All source qweights and LUTs remained CPU-resident; the hidden-copy
audit passed and no complete packed parent was on GPU.

On-demand physical transfer was `134138675200` bytes actual and
`134138675200` bytes independently expected, so equality passed.
Prefill transfer was `134138675200` bytes, decode transfer was `0` bytes,
first-use transfer was `134138675200` bytes, and reuse transfer was `0` bytes.
This is physical packed transfer evidence only; it is not a claim about future
asynchronous transfer savings.

### Latency evidence

The following values are the exact committed five-repeat medians in the fixed
request order `s03-quality-0`, `s03-quality-1`, `s03-quality-2`,
`s03-quality-3`, `s03-quality-4`, `validation-3`, `validation-1000`.
Each tuple is `prefill / decode / end_to_end` seconds. Transfer is included in
on-demand end-to-end latency, and no speed gate is imposed by the protocol.

```text
full_precision_bf16_teacher
0.03852094989269972 / 0.3015937090385705 / 0.3404704499989748
0.0353640781249851  / 0.2844479589257389 / 0.319878710899502
0.03574147191829979  / 0.2837038200814277 / 0.3198872189968824
0.03537587006576359  / 0.2838846899103373 / 0.3197095841169357
0.03534690081141889  / 0.2843983361963183 / 0.3198142009787261
0.03528881608508527  / 0.2838159708771855 / 0.3191385569516569
0.03538861614651978  / 0.2837730711326003 / 0.3193186500575393

static_packed_4bit
0.0448451810516417   / 0.2975339908152819 / 0.341828634031117
0.0446124579757452   / 0.2933125318959355 / 0.3379261489026248
0.04468805016949773  / 0.29219507612288   / 0.336816867114976
0.04389741295017302  / 0.293540304992348  / 0.337433269014582
0.04456966300494969  / 0.2902058698236942 / 0.3353296949062496
0.0447898309212178   / 0.2941319830715656 / 0.3389206018764526
0.04587049502879381  / 0.3024317480158061 / 0.3483035841491073

static_packed_8bit
0.04424904193729162  / 0.297714252024889  / 0.3418716988526285
0.04461880400776863  / 0.295672673964873  / 0.3404010799713433
0.04483346920460463  / 0.2965151828248054 / 0.3411716229747981
0.04444831586442888  / 0.2948019709438086 / 0.339408746920526
0.04465175699442625  / 0.2939406740479171 / 0.3384424760006368
0.04459544806741178  / 0.2913767248392105 / 0.3359937539789826
0.04510899819433689  / 0.2936529919970781 / 0.3386343871243298

hard_routed_resident_packed
0.2178544132038951   / 0.6203489559702575 / 0.8288311799988151
0.215908891055733    / 0.60264105303213   / 0.8126787338405848
0.229004189837724    / 0.6012733518145978 / 0.8079777988605201
0.2200271300971508   / 0.5983383841812611 / 0.8216986409388483
0.2183065507560968   / 0.5950753316283226 / 0.8036964708007872
0.2161036841571331   / 0.5950823440216482 / 0.7974853022024035
0.1751139578409493   / 0.4682606239803135 / 0.6399037949740887

hard_routed_synchronous_on_demand_packed
7.694171818904579   / 0.4856082745827734 / 8.18009682232514
6.561490010935813   / 0.4818590441718698 / 7.0792201817967
5.518259800970554   / 0.4713857369497418 / 6.008487404789776
5.299130752217025   / 0.4959729607217014 / 5.826189012732357
5.182901333086193   / 0.4745767489075661 / 5.664833359885961
5.403375862631947   / 0.4650539942085743 / 5.871282609645277
5.290773496031761   / 0.4813977479934692 / 5.762501392979175
```

Routed latency changed after the deterministic fallback. The frozen protocol
has no speed target, so the slower synchronous on-demand baseline is reported
as a limitation rather than a failed gate.

### B2 to B5 failure and correction history

S09-B2 returned `REVISE`. Quality, routes, transfers, cleanup, and generated-
token parity were otherwise sensible, but routed decode logits were not
bitwise reproducible/equivalent. S09-B3 isolated atomic k-split
`matmul_kbit` nondeterminism at the real routed shape. S09-B4 introduced a
narrow deterministic `dequant_kbit` plus `torch.matmul` fallback only for the
proven dispatch family. The pinned Any-Precision source itself remained
unchanged. S09-B5 reused the three unaffected FP/static results and reran only
the two invalidated routed modes. B5 passed deterministic and
resident/on-demand equivalence gates.

Provenance remains mixed by design: FP/static evidence comes from the
unaffected S09-B2 execution provenance at `a2e31188be952f97a1439ff7df46d9f43100bae5`;
corrected routed evidence comes from repair commit
`4a0dc702178fef0f84eb9ffd9bd6d1810e5dc564`; and the assembled/frozen B5 result
commit is `443f6994582500857afca9bad6032cc285448a86`. Provenance fields in the
JSON were not rewritten.

### Limitations, deferred work, and completion

This is not an exact reproduction of QAQ paper scores. Route diversity remains
observational `OTHER`. No post-baseline asynchronous loading, prefetching,
transfer prediction, caching, cost penalty, batching, scheduler, or other
research mechanism was tested. There is no claim that synchronous on-demand
loading is faster than the resident baseline, and no conclusion about later
optimized-loader performance is permitted.

No later authoritative stage is defined in `docs/stages/`; therefore the exact
next action is: **Baseline frozen. Stop. Define an explicit post-baseline stage
and decision before implementing any optimization or additional research
mechanism.**

S09 is **COMPLETE**.
