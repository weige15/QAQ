# Diagnose request dependence in completed S11-D routes

## Outcome

**COMPLETE — predominantly static by trained unit/layer, with real but secondary
request variation.** This is a read-only post-result diagnostic, not another
experiment and not a revision of the frozen S11-D `STOP` outcome.

Across all `10,368` canonical hard decisions, 4/6/8 usage is
`54 / 2,939 / 7,375` (`0.52% / 28.35% / 71.13%`). Within a fixed trained
seed/timing/cost trial, `744/864 = 86.11%` of layer/unit cells use one bit for
all twelve requests. A per-cell modal static policy explains `96.74%` of all
decisions, and only `2,826/57,024 = 4.96%` of within-cell unordered request
pairs disagree. Under the explicit descriptive rule below, the observed router
is therefore **mostly a static unit/layer policy**, not predominantly
request-dependent.

That label is descriptive. Request dependence is nonzero, especially under
`lambda_bit=0.03` and in late layers. The completed evidence is observational:
it cannot determine which individual downgrade caused S11-D's quality loss.

## Exact evidence boundary

The route statistics use only the twelve completed canonical trial files. The
canonical aggregate is used only to establish trial membership and request
order; its precomputed transition summaries are not treated as analysis input.
The analyzer requires exact SHA-256 identities and rejects missing, extra,
incomplete, audit-failing, repeat-mismatched, or malformed canonical JSON.

| Canonical input | SHA-256 |
|---|---|
| `docs/results/s11d_paired_468/aggregation.json` | `ad40dc13276b83aef5ea0d58d1920c4e472ba3f8817c691e5ea5fa5b1881ef04` |
| `seed-1729__same_unit_468_control__lambda-0.json` | `af6c28e4a425a7e8706b05dd60ccd2680304ca6cd53cfc4040147b00bd39b95e` |
| `seed-1729__lookahead_attention_one_unit_468_treatment__lambda-0.json` | `29e797fab9e2b741a0e5a70377e1879e79827d0bd9a0d33d0129beafd624f164` |
| `seed-1729__same_unit_468_control__lambda-0p03.json` | `654c6efa85d5b6a4f8f37c9bc2ea21f747ed3f7d04b3209a5c55aac810b757d5` |
| `seed-1729__lookahead_attention_one_unit_468_treatment__lambda-0p03.json` | `a1265186641db3d2fc020af5e0cbfa2158fc46151587cf8ecb3c8d5b41ecde83` |
| `seed-1730__same_unit_468_control__lambda-0.json` | `c56a740629bef0129c6e66eae471375ca0b7446a02d7546d3b27cbd48ceca23f` |
| `seed-1730__lookahead_attention_one_unit_468_treatment__lambda-0.json` | `c7e0da388a49966901e4c3da77d124ccccac8e7c6e5324033b2f8ae409dfd8cb` |
| `seed-1730__same_unit_468_control__lambda-0p03.json` | `36e1d77720203be9a81fee93e6657faff5e27edeb4bef1c74cc51b0fd019051c` |
| `seed-1730__lookahead_attention_one_unit_468_treatment__lambda-0p03.json` | `386adb6afd48261b8a27b8805e1a729ca259d09038da5bdef34e463d89fc34eb` |
| `seed-1731__same_unit_468_control__lambda-0.json` | `fda1cee6b057afa230fab7dd01462e94ce92904a37e635161fd6fb48075da0c5` |
| `seed-1731__lookahead_attention_one_unit_468_treatment__lambda-0.json` | `210d8339a24209585534a59d489d084f800ed16c36db95de2ad182ce8151a401` |
| `seed-1731__same_unit_468_control__lambda-0p03.json` | `831eb7c429218975533664350a4e398765fe7134098788bc0c61acafeaae3d8a` |
| `seed-1731__lookahead_attention_one_unit_468_treatment__lambda-0p03.json` | `a109e52511bb2cf2b5b276b3bd38b51b7ae0ebb24b729eb68212f9cceecd0b4d` |

All trial paths have parent `docs/results/s11d_paired_468/`. The byte-derived,
noncanonical analysis artifact is
`docs/results/s11d_route_policy_diagnostic.json`. It includes every source path
and digest, all `864` unit-trial request-variation records, and all detailed
breakdowns. It is deliberately outside the canonical trial directory.

Excluded evidence and actions are soft routes, training-history associations,
noncanonical or superseded files, model/data/CUDA execution, retraining,
lambda retuning, S11-D reruns, new lookahead work, and sensitivity execution.
No canonical file was changed.

## Reproducible definitions and procedure

The analyzer is `scripts/analyze_s11d_route_policy.py`, backed by the
standard-library-only module `qaq.evaluation.s11d_route_diagnostic`. Rebuild and
verify the derived result without loading Torch, a model, a dataset, or CUDA:

```bash
source ~/.venv/bin/activate
PYTHONPATH=src:. python scripts/analyze_s11d_route_policy.py \
  --output docs/results/s11d_route_policy_diagnostic.json
PYTHONPATH=src:. python scripts/analyze_s11d_route_policy.py \
  --check docs/results/s11d_route_policy_diagnostic.json
```

Definitions:

* A **unit-trial** is one target `(layer, attention-or-FFN)` under one fixed
  seed, timing, and cost condition, observed on all twelve requests.
* A unit-trial is **invariant** when every request selects the same bit.
* **Modal fidelity** is the fraction of its twelve decisions equal to its most
  frequent bit, pooled across unit-trials.
* **Request-pair disagreement** is the fraction of unordered request pairs that
  choose different bits within the same unit-trial. Request order is not given
  a causal direction.
* **Mostly static** is the post-result descriptive label used when pooled modal
  fidelity is at least `0.90` and request-pair disagreement is at most `0.10`.
  This definition is not a frozen quality, precision, or acceptance threshold.
* Equal layer thirds are explicit: early `0–11`, middle `12–23`, and late
  `24–35`.
* Timing transitions are matched `same_unit -> lookahead` at fixed seed, cost,
  request, layer, and unit. Cost transitions are matched
  `lambda 0.0 -> 0.03` at fixed seed, timing, request, layer, and unit. Seed and
  request comparisons are unordered because neither has a scientific
  high-to-low direction.

`usage.by_layer`, `usage.by_layer_and_unit_type`, `usage.by_seed`,
`usage.by_request`, `usage.by_timing`, `usage.by_cost_condition`, and the joint
`usage.by_seed_timing_cost_trial`, `usage.by_request_timing_cost`, and
`usage.by_layer_unit_timing_cost` records in the derived JSON contain counts,
normalized fractions, decision totals, and mean selected bits. This preserves
all requested marginal and matched views rather than hiding them in an overall
mean.

## 4/6/8 use

### Overall, unit type, timing, cost, and region

| View | n | 4 count (%) | 6 count (%) | 8 count (%) | mean bits |
|---|---:|---:|---:|---:|---:|
| overall | 10,368 | 54 (0.52) | 2,939 (28.35) | 7,375 (71.13) | 7.4122 |
| attention | 5,184 | 47 (0.91) | 1,742 (33.60) | 3,395 (65.49) | 7.2917 |
| FFN | 5,184 | 7 (0.14) | 1,197 (23.09) | 3,980 (76.77) | 7.5328 |
| same-unit | 5,184 | 29 (0.56) | 1,513 (29.19) | 3,642 (70.25) | 7.3939 |
| lookahead | 5,184 | 25 (0.48) | 1,426 (27.51) | 3,733 (72.01) | 7.4306 |
| lambda 0.0 | 5,184 | 2 (0.04) | 903 (17.42) | 4,279 (82.54) | 7.6501 |
| lambda 0.03 | 5,184 | 52 (1.00) | 2,036 (39.27) | 3,096 (59.72) | 7.1744 |
| early 0–11 | 3,456 | 2 (0.06) | 927 (26.82) | 2,527 (73.12) | 7.4612 |
| middle 12–23 | 3,456 | 14 (0.41) | 782 (22.63) | 2,660 (76.97) | 7.5312 |
| late 24–35 | 3,456 | 38 (1.10) | 1,230 (35.59) | 2,188 (63.31) | 7.2442 |

Seed mean widths are tightly ranged but not identical: seed 1729 is `7.4057`
(`22/983/2451` at 4/6/8), seed 1730 is `7.3999` (`18/1001/2437`), and seed
1731 is `7.4311` (`14/955/2487`), each with `n=3,456`.

Across requests (`n=864` each), mean width ranges from `7.3611` for
`validation-270` to `7.4329` for `validation-3`. The derived artifact records
all twelve request counts: 4-bit use ranges from `2` to `10`, 6-bit use from
`237` to `257`, and 8-bit use from `598` to `622`. These narrow request-level
margins are consistent with, but do not by themselves prove, static behavior.

## Request dependence versus other variation

| Fixed-cell diagnostic | Estimate | n / uncertainty |
|---|---:|---|
| invariant unit-trials | 86.11% | 744/864; Wilson 95% 83.64–88.26% |
| modal static-policy fidelity | 96.74% | 10,368 decisions |
| request-pair disagreement | 4.96% | 2,826/57,024; Wilson 95% 4.78–5.14% |
| matched seed-pair disagreement | 17.79% | 1,844/10,368; Wilson 95% 17.06–18.53% |
| matched timing changes | 11.42% | 592/5,184 |
| matched cost changes | 22.88% | 1,186/5,184 |

The request-pair disagreement is `5.32%` for attention and `4.59%` for FFN.
Attention modal fidelity is `96.39%`; FFN is `97.09%`. Across seeds, request
pair disagreement ranges from `4.20%` to `5.53%`. Same-unit (`4.99%`) and
lookahead (`4.92%`) are nearly equal. Cost pressure has the clearest effect on
request variation: lambda 0.0 has `3.17%` request-pair disagreement and
`90.51%` invariant unit-trials, versus `6.74%` and `81.71%` at lambda 0.03.

Layer position matters. Request-pair disagreement is `2.91%` early, `4.41%`
middle, and `7.55%` late; invariant unit-trials are `91.67%`, `88.19%`, and
`78.47%`, respectively. Thus the router is not perfectly static, but request
variation is smaller than matched seed, timing, and especially cost variation,
and is concentrated toward the late third.

Wilson intervals in the artifact are descriptive binomial intervals with every
numerator and denominator shown. Decisions are clustered within only three
seed contexts and twelve fixed requests, so they are not population-level
inferential guarantees. Per-seed records expose the small-`n` range rather than
averaging seeds into a synthetic trial.

## Directional changes and layer concentration

### Cost: lambda 0.0 -> 0.03

Of `5,184` matched decisions, cost pressure lowers `1,185`, leaves `3,998`
equal, and raises one; mean selected width changes by `-0.4757` bits. The
`1,134` 8→6 changes are balanced between attention (`573`) and FFN (`561`).
The deeper changes are attention-heavy: 8→4 occurs `49` times (`42` attention,
`7` FFN), and 6→4 occurs twice (both attention).

* 8→6: early `335`, middle `346`, late `453`;
* 8→4: early `2`, middle `14`, late `33`;
* 6→4: early `0`, middle `0`, late `2`.

Overall cost-change rates are `19.50%` early, `20.83%` middle, and `28.30%`
late. The most frequent 8→6 layers are 5 (`77`), 11 (`72`), 30 (`72`), 21
(`69`), and 32 (`68`). The 8→4 sites are layers 2 (`2`), 21 (`14`), 25 (`13`),
27 (`12`), 32 (`6`), and 34 (`2`); both 6→4 changes are layer 34 attention.

### Timing: same-unit -> lookahead

Of `5,184` matched decisions, lookahead lowers `245`, leaves `4,592` equal,
and raises `347`; pooled mean width changes by `+0.03665` bits. Among the `592`
changes, the lookahead choice is higher in `58.61%` (Wilson 95%
`54.60–62.51%`). This is a modest pooled conservative association, **not a
systematic result across seeds**: seed-aggregated timing deltas span
`-0.01968` to `+0.08912` bits, with one of three seeds negative. The per-cell
breakdowns in the artifact also show that the sign is positive for two seeds
and negative for one at both cost conditions.

Timing changes are most frequent by rate in the middle third (`13.14%`), then
early (`11.17%`) and late (`9.95%`). Direction matters:

* 8→6: early `123`, middle `48`, late `56` (`142` attention, `85` FFN);
* 8→4: early `0`, middle `3`, late `11` (`13` attention, `1` FFN);
* 6→4: early `0`, middle `1`, late `3` (all attention).

Attention timing changes are almost balanced (`159` lower, `183` higher; mean
`+0.01235` bits). FFN changes more often go upward (`86` lower, `164` higher;
mean `+0.06096`). This demonstrates that changing attention timing can remain
associated with FFN route differences after separate training; it is not a
claim that lookahead directly controls FFN routing.

Every transition is further localized by layer, unit type, seed, request, and
region in
`matched_variation_and_transitions.*.downgrade_localization`. The corresponding
breakdowns retain equal and upward transitions too, so downgrade-only summaries
do not conceal the comparison denominator.

## Smallest causal same-unit block-sensitivity study (defined, not executed)

The present association analysis cannot identify an individual harmful
downgrade. The smallest interpretable follow-up must change one layer/unit block
at a time; grouping blocks would preserve the same attribution problem.

Limit the evidence-directed candidate set to the `42` distinct layer/unit
blocks that exhibit at least one downgrade under the canonical **same-unit**
`lambda 0.0 -> 0.03` comparison. Their exact layers, unit types, regions, and
observed transition counts are recorded in
`proposed_same_unit_block_sensitivity.candidate_units`. The other `30` units
remain conservatively at 8-bit and receive no lower-safe label; probing them
would widen the smallest study beyond observed same-unit downgrade evidence.

For each candidate block:

1. For every canonical seed-context and request, replay the canonical same-unit
   lambda-zero route map. Force the target block to 8 bits for the paired
   control; do not change any other unit.
2. Test target precision 4 first. If it passes, label 4 as the lowest safe
   precision. If it fails, test target precision 6. Label 6 if it passes;
   otherwise label 8. This requires `42` initial intervention cells and at most
   `84`, rather than pre-running both lower precisions everywhere.
3. Use all three existing seed-context route maps and all twelve canonical
   requests (`36` paired contexts per block), with one immediate exact repeat.
   These are context replicates, not new training seeds. Do not train, sample,
   tune lambda, or add a timing arm.
4. Measure completion-only temperature-2 masked teacher-relative KL, full-logit
   mean absolute teacher error, and diagnostic maximum absolute error against
   the same full-precision teacher.
5. A target precision passes only if, for **each** of the three seed contexts,
   aggregate KL is `<= 1.10 ×` its paired target-forced-8 control, aggregate
   mean absolute error is `<= 1.10 ×` control, every request KL is
   `<= 1.25 ×` control, all values are finite, the route maps differ only at the
   target unit, and the immediate repeat is identical. These are the existing
   S11 quality factors, not retuned post-result margins.

Pairing within seed-context/request fixes observed request and seed-associated
route context. Forced same-unit maps remove timing and cost condition as
varying factors. The study therefore isolates a single block precision contrast
without another lookahead experiment or another S11-D training run.

This document only defines that future protocol. It does **not** authorize or
execute it, and it does not label any block safe today.

## Scientific boundary

The analysis describes associations among route selections, requests, seeds,
timing, and cost pressure. It cannot determine which individual 8→6, 8→4, or
6→4 downgrade caused quality loss, cannot infer execution benefit from selected
bits, and cannot turn the failed S11-D `STOP` result into a production lambda or
precision policy. The frozen S11-D protocol, thresholds, canonical evidence,
and outcome remain unchanged.

## Same-unit block-sensitivity executor readiness

**COMPLETE — structurally ready, not executed.**
`qaq.evaluation.block_sensitivity` validates this document's exact
byte-identified definition against the canonical route evidence and emits a
deterministic non-executing plan. The plan contains exactly the ordered 42
candidate units, three existing seed contexts, twelve canonical requests, 36
paired contexts per intervention, one target-forced-8 control for every
treatment, precision 4 first, and precision 6 only after complete valid
precision-4 failure. It preserves completion-only temperature-2 masked
teacher-relative KL, full-logit mean and diagnostic maximum absolute teacher
error, the per-seed aggregate factors `1.10`, per-request KL factor `1.25`,
finiteness, exact immediate repeat, and target-only route-difference rules.

A complete future unit/precision result is one independently valid atomic file.
It binds the study, model/tokenizer/artifact, the established exact S11
hardware/software fields on a compatible RTX 3090, unit, precision, all ordered
source contexts, input and teacher digests, expected forced route
maps, paired control/treatment logits and immediate repeats, metrics, three
recomputed seed summaries, classification, and the no-training/no-retuning/
no-lookahead audit. Missing, duplicate, reordered, non-finite, mismatched,
partially repeated, cross-study, or route-inconsistent evidence is invalid.

The dispatch boundary accepts only an exact known unit, precision 4 or 6, an
explicit `cuda:<index>`, and the exact absent output. A precision-6 request also
requires an already persisted complete valid failed precision-4 result and is
rejected when precision 4 passed. Persistence validates before mutation, writes
and fsyncs a same-directory temporary file, revalidates it, promotes with an
atomic no-overwrite hard link, verifies promoted bytes and SHA-256, fsyncs the
directory, and cleans its temporary file. Existing destinations are never
overwritten; an interrupted temporary file never counts as complete. The
non-mutating `--resume-plan` scan walks the authoritative target order and emits
exactly one next action per unit: run precision 4, run precision 6 after complete
valid failed 4, or complete. Absent result state deterministically means all 42
precision-4 actions remain. Temporary, linked, wrongly named, malformed,
mixed-study, or cross-execution-provenance state is rejected.

Aggregation revalidates every canonical result path and accepts only all 42
mandatory precision-4 results plus exactly those precision-6 fallbacks required
by failed precision-4 results. Fallback forced-8 controls must match the first
attempt.
All files must use their exact unit/precision names and share one complete
hardware/software execution identity. Missing, duplicate, orphaned, unexpected,
or mixed-study evidence is rejected. Only then may aggregation assign 4 after a
precision-4 pass, 6 after a failed 4 and passed 6, or conservative 8 after both
fail.

The command `scripts/run_s11d_block_sensitivity.py --plan` is inert and
standard-library-only. Structural tests block imports of Torch, Transformers,
datasets, Any-Precision, and CUDA extensions while validating deterministic
plan bytes. No sensitivity result parent was created; no model/CUDA execution,
router training, lambda retuning, S11-D rerun, or lookahead work occurred.
