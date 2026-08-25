"""Read-only diagnostics for the completed canonical S11-D hard route maps.

This module never imports model, training, dataset, or CUDA code.  It accepts only
the byte-identified canonical aggregate and twelve canonical trial files.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
CANONICAL_PARENT = ROOT / "docs/results/s11d_paired_468"
AGGREGATION_PATH = CANONICAL_PARENT / "aggregation.json"
DERIVED_OUTPUT = ROOT / "docs/results/s11d_route_policy_diagnostic.json"
EXPECTED_PROTOCOL_SHA256 = "4a62aeb7d8ae90a6349dc9dc8aab6dda4196b54876c4d0546c05808936fefe92"
EXPECTED_AGGREGATION_SHA256 = "ad40dc13276b83aef5ea0d58d1920c4e472ba3f8817c691e5ea5fa5b1881ef04"
EXPECTED_TRIAL_SHA256 = {
    "seed-1729__same_unit_468_control__lambda-0": "af6c28e4a425a7e8706b05dd60ccd2680304ca6cd53cfc4040147b00bd39b95e",
    "seed-1729__lookahead_attention_one_unit_468_treatment__lambda-0": "29e797fab9e2b741a0e5a70377e1879e79827d0bd9a0d33d0129beafd624f164",
    "seed-1729__same_unit_468_control__lambda-0p03": "654c6efa85d5b6a4f8f37c9bc2ea21f747ed3f7d04b3209a5c55aac810b757d5",
    "seed-1729__lookahead_attention_one_unit_468_treatment__lambda-0p03": "a1265186641db3d2fc020af5e0cbfa2158fc46151587cf8ecb3c8d5b41ecde83",
    "seed-1730__same_unit_468_control__lambda-0": "c56a740629bef0129c6e66eae471375ca0b7446a02d7546d3b27cbd48ceca23f",
    "seed-1730__lookahead_attention_one_unit_468_treatment__lambda-0": "c7e0da388a49966901e4c3da77d124ccccac8e7c6e5324033b2f8ae409dfd8cb",
    "seed-1730__same_unit_468_control__lambda-0p03": "36e1d77720203be9a81fee93e6657faff5e27edeb4bef1c74cc51b0fd019051c",
    "seed-1730__lookahead_attention_one_unit_468_treatment__lambda-0p03": "386adb6afd48261b8a27b8805e1a729ca259d09038da5bdef34e463d89fc34eb",
    "seed-1731__same_unit_468_control__lambda-0": "fda1cee6b057afa230fab7dd01462e94ce92904a37e635161fd6fb48075da0c5",
    "seed-1731__lookahead_attention_one_unit_468_treatment__lambda-0": "210d8339a24209585534a59d489d084f800ed16c36db95de2ad182ce8151a401",
    "seed-1731__same_unit_468_control__lambda-0p03": "831eb7c429218975533664350a4e398765fe7134098788bc0c61acafeaae3d8a",
    "seed-1731__lookahead_attention_one_unit_468_treatment__lambda-0p03": "a109e52511bb2cf2b5b276b3bd38b51b7ae0ebb24b729eb68212f9cceecd0b4d",
}
BITS = (4, 6, 8)
UNITS = ("attention", "ffn")
SEEDS = (1729, 1730, 1731)
TIMINGS = ("same_unit", "lookahead_attention_one_unit")
LAMBDAS = (0.0, 0.03)


class DiagnosticError(ValueError):
    """Canonical evidence failed the read-only diagnostic boundary."""


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _load_json(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise DiagnosticError(f"expected JSON object: {path}")
    return value, raw


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DiagnosticError(message)


def _region(layer: int) -> str:
    if 0 <= layer <= 11:
        return "early_0_11"
    if 12 <= layer <= 23:
        return "middle_12_23"
    if 24 <= layer <= 35:
        return "late_24_35"
    raise DiagnosticError(f"invalid layer: {layer}")


def _rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def load_canonical_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate and flatten only the twelve byte-identified trial route maps."""

    aggregation, aggregation_raw = _load_json(AGGREGATION_PATH)
    _require(
        _sha256(aggregation_raw) == EXPECTED_AGGREGATION_SHA256,
        "canonical aggregation SHA-256 mismatch",
    )
    _require(
        aggregation.get("schema") == "qaq-s11d-paired-lookahead-468-aggregation-v1",
        "aggregation schema mismatch",
    )
    _require(
        aggregation.get("protocol_sha256") == EXPECTED_PROTOCOL_SHA256,
        "aggregation protocol mismatch",
    )
    trial_order = aggregation.get("trial_order")
    request_order = aggregation.get("request_order")
    _require(
        isinstance(trial_order, list)
        and trial_order == list(EXPECTED_TRIAL_SHA256),
        "canonical trial order or membership mismatch",
    )
    _require(isinstance(request_order, list) and len(request_order) == 12, "request order mismatch")

    actual_json = {path.name for path in CANONICAL_PARENT.glob("*.json")}
    expected_json = {"aggregation.json", *(f"{trial_id}.json" for trial_id in trial_order)}
    _require(actual_json == expected_json, "canonical result directory has unexpected/missing JSON")

    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for index, trial_id in enumerate(trial_order, start=1):
        path = CANONICAL_PARENT / f"{trial_id}.json"
        trial, raw = _load_json(path)
        digest = _sha256(raw)
        _require(digest == EXPECTED_TRIAL_SHA256[trial_id], f"trial SHA-256 mismatch: {trial_id}")
        _require(trial.get("schema") == "qaq-s11d-paired-lookahead-468-trial-v1", "trial schema mismatch")
        _require(trial.get("trial_id") == trial_id, "trial id mismatch")
        _require(trial.get("trial_index") == index, "trial index mismatch")
        _require(trial.get("protocol_sha256") == EXPECTED_PROTOCOL_SHA256, "trial protocol mismatch")
        _require(trial.get("route_decisions") == 864, "trial route count mismatch")
        _require(trial.get("audits", {}).get("passed") is True, "trial audits incomplete")
        _require(trial.get("immediate_hard_repeat", {}).get("identical") is True, "repeat mismatch")
        _require(trial.get("production_checkpoint_created") is False, "unexpected checkpoint")
        hard_validation = trial.get("hard_validation")
        _require(isinstance(hard_validation, list) and len(hard_validation) == 12, "request evidence incomplete")
        _require([item.get("request_id") for item in hard_validation] == request_order, "request order drifted")
        for request in hard_validation:
            route_map = request.get("route_map")
            _require(isinstance(route_map, list) and len(route_map) == 72, "route map incomplete")
            expected_units = [(layer, unit) for layer in range(36) for unit in UNITS]
            actual_units = [(item.get("target_layer"), item.get("unit_type")) for item in route_map]
            _require(actual_units == expected_units, "route map unit order/coverage mismatch")
            _require(
                all(item.get("selected_bits") in BITS for item in route_map),
                "route map contains invalid precision",
            )
            for route in route_map:
                rows.append(
                    {
                        "trial_id": trial_id,
                        "seed": trial["seed"],
                        "timing": trial["routing_timing"],
                        "lambda_bit": trial["lambda_bit"],
                        "request_id": request["request_id"],
                        "layer": route["target_layer"],
                        "region": _region(route["target_layer"]),
                        "unit_type": route["unit_type"],
                        "bits": route["selected_bits"],
                    }
                )
        sources.append({"path": _rel(path), "sha256": digest, "trial_id": trial_id})

    _require(len(rows) == 12 * 12 * 72, "flattened route coverage mismatch")
    source = {
        "selection_rule": (
            "aggregation.trial_order plus exact recorded SHA-256; every trial must pass schema, "
            "protocol, completeness, audit, route-coverage, and immediate-repeat checks"
        ),
        "aggregation_manifest": {
            "path": _rel(AGGREGATION_PATH),
            "sha256": EXPECTED_AGGREGATION_SHA256,
            "used_for": "canonical membership and request order only; route statistics use trial files",
        },
        "trial_files": sources,
        "excluded": [
            "aggregation paired-transition summaries as a statistical source",
            "incomplete, noncanonical, superseded, or checksum-mismatched files",
            "training history, soft routes, checkpoints, model execution, and newly generated trials",
        ],
    }
    return rows, source


def _usage_record(group: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts = Counter(int(row["bits"]) for row in rows)
    total = len(rows)
    return {
        **group,
        "decisions": total,
        "counts": {str(bit): counts[bit] for bit in BITS},
        "fractions": {str(bit): counts[bit] / total for bit in BITS},
        "mean_selected_bits": sum(int(row["bits"]) for row in rows) / total,
    }


def _group_usage(rows: Sequence[Mapping[str, Any]], dimensions: Sequence[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[name] for name in dimensions)].append(row)
    records = []
    for key in sorted(grouped, key=lambda item: tuple(str(value) for value in item)):
        records.append(_usage_record(dict(zip(dimensions, key, strict=True)), grouped[key]))
    return records


def _wilson(successes: int, total: int) -> dict[str, float | int]:
    _require(total > 0, "Wilson interval requires observations")
    z = 1.959963984540054
    p = successes / total
    denominator = 1.0 + z * z / total
    centre = (p + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return {
        "successes": successes,
        "n": total,
        "fraction": p,
        "wilson_95_low": max(0.0, centre - radius),
        "wilson_95_high": min(1.0, centre + radius),
    }


def _request_dependence(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    dimensions = ("trial_id", "seed", "timing", "lambda_bit", "layer", "region", "unit_type")
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[name] for name in dimensions)].append(row)

    details: list[dict[str, Any]] = []
    for key in sorted(grouped, key=lambda item: tuple(str(value) for value in item)):
        group = grouped[key]
        counts = Counter(int(row["bits"]) for row in group)
        _require(len(group) == 12, "unit-trial request coverage mismatch")
        modal_count = max(counts.values())
        pair_total = math.comb(len(group), 2)
        pair_disagreements = sum(
            counts[left] * counts[right] for left, right in itertools.combinations(BITS, 2)
        )
        details.append(
            {
                **dict(zip(dimensions, key, strict=True)),
                "request_count": len(group),
                "counts": {str(bit): counts[bit] for bit in BITS},
                "distinct_precision_count": len(counts),
                "invariant_across_requests": len(counts) == 1,
                "modal_precision": min(bit for bit, count in counts.items() if count == modal_count),
                "modal_count": modal_count,
                "modal_fidelity": modal_count / len(group),
                "request_pair_disagreements": pair_disagreements,
                "request_pair_total": pair_total,
                "request_pair_disagreement_rate": pair_disagreements / pair_total,
            }
        )

    def summarize(items: Sequence[Mapping[str, Any]], fields: Mapping[str, Any]) -> dict[str, Any]:
        invariant = sum(bool(item["invariant_across_requests"]) for item in items)
        pair_disagreements = sum(int(item["request_pair_disagreements"]) for item in items)
        pair_total = sum(int(item["request_pair_total"]) for item in items)
        decisions = sum(int(item["request_count"]) for item in items)
        modal_matches = sum(int(item["modal_count"]) for item in items)
        return {
            **fields,
            "unit_trials": len(items),
            "invariant_unit_trials": _wilson(invariant, len(items)),
            "modal_policy_fidelity": modal_matches / decisions,
            "request_pair_disagreement": _wilson(pair_disagreements, pair_total),
        }

    summary = summarize(details, {})
    summaries: dict[str, list[dict[str, Any]]] = {}
    for dimension in ("unit_type", "seed", "timing", "lambda_bit", "region"):
        values = sorted({item[dimension] for item in details}, key=str)
        summaries[f"by_{dimension}"] = [
            summarize([item for item in details if item[dimension] == value], {dimension: value})
            for value in values
        ]
    static = (
        summary["modal_policy_fidelity"] >= 0.9
        and summary["request_pair_disagreement"]["fraction"] <= 0.1
    )
    return {
        "definitions": {
            "eligible_unit_trial": "one target layer/unit in one fixed seed/timing/cost trial",
            "invariant": "all twelve canonical requests select the same bit",
            "modal_policy_fidelity": "decisions matching that unit-trial's most frequent bit / decisions",
            "request_pair_disagreement": "unordered request pairs selecting different bits / all request pairs",
            "descriptive_static_label_rule": (
                "mostly_static if pooled modal fidelity >= 0.90 and pooled request-pair disagreement <= 0.10; "
                "this post-result descriptive rule is not a frozen quality or acceptance threshold"
            ),
        },
        "overall": summary,
        "summaries": summaries,
        "unit_trial_details": details,
        "determination": {
            "label": "mostly_static_unit_layer_policy" if static else "materially_request_dependent_or_inconclusive",
            "rule_passed": static,
            "interpretation": (
                "Request dependence is real but secondary: compare the pooled modal fidelity and pairwise "
                "disagreement with the seed/timing/cost matched-comparator rates."
            ),
        },
    }


def _matched_rows(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[Any, ...], int]:
    return {
        (
            row["seed"],
            row["timing"],
            row["lambda_bit"],
            row["request_id"],
            row["layer"],
            row["unit_type"],
        ): int(row["bits"])
        for row in rows
    }


def _transition_summary(records: Sequence[Mapping[str, Any]], dimensions: Sequence[str] = ()) -> dict[str, Any]:
    transitions = Counter((int(item["from_bits"]), int(item["to_bits"])) for item in records)
    lower = sum(count for (source, target), count in transitions.items() if target < source)
    equal = sum(count for (source, target), count in transitions.items() if target == source)
    higher = sum(count for (source, target), count in transitions.items() if target > source)
    result: dict[str, Any] = {
        "comparisons": len(records),
        "transition_counts": {
            f"{source}_to_{target}": transitions[source, target]
            for source in BITS
            for target in BITS
        },
        "lower_equal_higher": {"lower": lower, "equal": equal, "higher": higher},
        "changed_fraction": (lower + higher) / len(records),
        "mean_target_minus_source_bits": sum(
            int(item["to_bits"]) - int(item["from_bits"]) for item in records
        )
        / len(records),
    }
    if dimensions:
        result["breakdowns"] = {
            f"by_{dimension}": [
                {
                    dimension: value,
                    **_transition_summary(
                        [item for item in records if item[dimension] == value]
                    ),
                }
                for value in sorted({item[dimension] for item in records}, key=str)
            ]
            for dimension in dimensions
        }
    return result


def _downgrade_localization(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for source, target in ((8, 6), (8, 4), (6, 4)):
        selected = [
            item
            for item in records
            if item["from_bits"] == source and item["to_bits"] == target
        ]
        output[f"{source}_to_{target}"] = {
            "count": len(selected),
            "by_layer": dict(sorted(Counter(str(item["layer"]) for item in selected).items(), key=lambda x: int(x[0]))),
            "by_region": dict(sorted(Counter(item["region"] for item in selected).items())),
            "by_unit_type": dict(sorted(Counter(item["unit_type"] for item in selected).items())),
            "by_seed": dict(sorted(Counter(str(item["seed"]) for item in selected).items())),
            "by_request": dict(sorted(Counter(item["request_id"] for item in selected).items())),
        }
    return output


def _comparators(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    index = _matched_rows(rows)
    # Preserve canonical request order through first appearance.
    requests = list(dict.fromkeys(str(row["request_id"]) for row in rows))
    timing_records: list[dict[str, Any]] = []
    cost_records: list[dict[str, Any]] = []
    for seed in SEEDS:
        for lambda_bit in LAMBDAS:
            for request_id in requests:
                for layer in range(36):
                    for unit_type in UNITS:
                        base = (seed, "same_unit", lambda_bit, request_id, layer, unit_type)
                        treatment = (
                            seed,
                            "lookahead_attention_one_unit",
                            lambda_bit,
                            request_id,
                            layer,
                            unit_type,
                        )
                        timing_records.append(
                            {
                                "seed": seed,
                                "lambda_bit": lambda_bit,
                                "request_id": request_id,
                                "layer": layer,
                                "region": _region(layer),
                                "unit_type": unit_type,
                                "from_bits": index[base],
                                "to_bits": index[treatment],
                            }
                        )
    for seed in SEEDS:
        for timing in TIMINGS:
            for request_id in requests:
                for layer in range(36):
                    for unit_type in UNITS:
                        zero = (seed, timing, 0.0, request_id, layer, unit_type)
                        positive = (seed, timing, 0.03, request_id, layer, unit_type)
                        cost_records.append(
                            {
                                "seed": seed,
                                "timing": timing,
                                "request_id": request_id,
                                "layer": layer,
                                "region": _region(layer),
                                "unit_type": unit_type,
                                "from_bits": index[zero],
                                "to_bits": index[positive],
                            }
                        )

    timing_summary = _transition_summary(
        timing_records, ("unit_type", "seed", "lambda_bit", "request_id", "region")
    )
    changed = timing_summary["lower_equal_higher"]["lower"] + timing_summary["lower_equal_higher"]["higher"]
    higher = timing_summary["lower_equal_higher"]["higher"]
    timing_summary["lookahead_more_conservative_among_changes"] = _wilson(higher, changed)
    per_seed_deltas = [
        item["mean_target_minus_source_bits"] for item in timing_summary["breakdowns"]["by_seed"]
    ]
    timing_summary["systematic_conservatism_determination"] = {
        "label": "not_systematic_across_seeds",
        "reason": (
            "pooled lookahead choices are modestly higher, but the per-seed mean deltas do not all "
            "have the same sign"
        ),
        "per_seed_mean_delta_min": min(per_seed_deltas),
        "per_seed_mean_delta_max": max(per_seed_deltas),
    }
    timing_summary["downgrade_localization"] = _downgrade_localization(timing_records)

    cost_summary = _transition_summary(
        cost_records, ("unit_type", "seed", "timing", "request_id", "region")
    )
    cost_summary["downgrade_localization"] = _downgrade_localization(cost_records)

    # Seed is unordered. Count all three same-condition seed pairs without assigning direction.
    seed_disagreements = 0
    seed_pairs = 0
    for timing in TIMINGS:
        for lambda_bit in LAMBDAS:
            for request_id in requests:
                for layer in range(36):
                    for unit_type in UNITS:
                        values = [
                            index[seed, timing, lambda_bit, request_id, layer, unit_type]
                            for seed in SEEDS
                        ]
                        for left, right in itertools.combinations(values, 2):
                            seed_pairs += 1
                            seed_disagreements += left != right

    return {
        "matched_timing_same_unit_to_lookahead": timing_summary,
        "matched_cost_zero_to_0p03": cost_summary,
        "matched_seed_pair_disagreement": _wilson(seed_disagreements, seed_pairs),
        "comparison_directions": {
            "timing": "same_unit -> lookahead at fixed seed, cost, request, layer, and unit",
            "cost": "lambda 0.0 -> 0.03 at fixed seed, timing, request, layer, and unit",
            "seed": "unordered seed pairs at every fixed timing, cost, request, layer, and unit",
            "requests": "unordered request pairs within fixed trial/unit; see request_dependence",
        },
    }


def _future_study_candidates(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    index = _matched_rows(rows)
    requests = list(dict.fromkeys(str(row["request_id"]) for row in rows))
    candidates: dict[tuple[int, str], Counter[tuple[int, int]]] = defaultdict(Counter)
    for seed in SEEDS:
        for request_id in requests:
            for layer in range(36):
                for unit_type in UNITS:
                    source = index[seed, "same_unit", 0.0, request_id, layer, unit_type]
                    target = index[seed, "same_unit", 0.03, request_id, layer, unit_type]
                    if target < source:
                        candidates[layer, unit_type][source, target] += 1
    output = []
    for (layer, unit_type), transitions in sorted(candidates.items()):
        output.append(
            {
                "layer": layer,
                "region": _region(layer),
                "unit_type": unit_type,
                "observed_same_unit_cost_downgrades": {
                    f"{source}_to_{target}": count
                    for (source, target), count in sorted(transitions.items())
                },
            }
        )
    _require(len(output) == 42, "same-unit sensitivity candidate count drifted")
    return output


def build_diagnostic() -> dict[str, Any]:
    rows, source = load_canonical_rows()
    request_dependence = _request_dependence(rows)
    comparators = _comparators(rows)
    candidate_units = _future_study_candidates(rows)
    usage = {
        "overall": _usage_record({}, rows),
        "by_layer": _group_usage(rows, ("layer",)),
        "by_layer_and_unit_type": _group_usage(rows, ("layer", "unit_type")),
        "by_region": _group_usage(rows, ("region",)),
        "by_unit_type": _group_usage(rows, ("unit_type",)),
        "by_seed": _group_usage(rows, ("seed",)),
        "by_request": _group_usage(rows, ("request_id",)),
        "by_timing": _group_usage(rows, ("timing",)),
        "by_cost_condition": _group_usage(rows, ("lambda_bit",)),
        "by_seed_timing_cost_trial": _group_usage(rows, ("seed", "timing", "lambda_bit")),
        "by_request_timing_cost": _group_usage(rows, ("request_id", "timing", "lambda_bit")),
        "by_layer_unit_timing_cost": _group_usage(
            rows, ("layer", "unit_type", "timing", "lambda_bit")
        ),
    }
    return {
        "schema": "qaq-s11d-route-policy-diagnostic-v1",
        "source_evidence": source,
        "coverage": {
            "trials": 12,
            "seeds": list(SEEDS),
            "timings": list(TIMINGS),
            "cost_conditions": list(LAMBDAS),
            "requests_per_trial": 12,
            "layers": 36,
            "unit_types": list(UNITS),
            "decisions": len(rows),
            "layer_regions": {
                "early_0_11": [0, 11],
                "middle_12_23": [12, 23],
                "late_24_35": [24, 35],
            },
        },
        "usage": usage,
        "request_dependence": request_dependence,
        "matched_variation_and_transitions": comparators,
        "attention_ffn_comparison": {
            "usage": usage["by_unit_type"],
            "request_dependence": request_dependence["summaries"]["by_unit_type"],
            "timing_transition": comparators["matched_timing_same_unit_to_lookahead"]["breakdowns"]["by_unit_type"],
            "cost_transition": comparators["matched_cost_zero_to_0p03"]["breakdowns"]["by_unit_type"],
            "uncertainty_note": (
                "Each record reports n; Wilson intervals are descriptive binomial intervals. Routes are "
                "clustered within three seed contexts and twelve fixed requests, so intervals are not "
                "population-level inferential guarantees. Per-seed breakdowns expose the n=3 range."
            ),
        },
        "causal_boundary": (
            "All route/quality relationships are observational associations. These files cannot determine "
            "which individual downgrade caused the S11-D quality loss."
        ),
        "proposed_same_unit_block_sensitivity": {
            "status": "defined_not_executed",
            "candidate_scope": (
                "42 layer/unit blocks with at least one 8->6, 8->4, or 6->4 change under the "
                "canonical same-unit lambda 0.0 -> 0.03 comparison"
            ),
            "candidate_units": candidate_units,
            "smallest_intervention": "one target attention or FFN block in one layer; never group blocks",
            "fixed_control": (
                "For each seed-context/request, replay its canonical same-unit lambda=0 route map, force "
                "the target block to 8 bits in the control, and change only that target block in treatment."
            ),
            "sequential_precision_rule": (
                "Test target=4 first. If 4 passes, label 4. If 4 fails, test target=6; label 6 if it "
                "passes, otherwise label 8. Thus 42 interventions are required initially and at most 84."
            ),
            "quality_measurements": [
                "completion-only temperature-2 masked teacher-relative KL",
                "full-logit mean absolute teacher error",
                "full-logit maximum absolute teacher error (diagnostic)",
            ],
            "lowest_safe_precision_rule": (
                "A target precision passes only if, for every one of the three seed-context route maps: "
                "aggregate KL <= 1.10 times its paired target-forced-8 control, aggregate mean absolute "
                "error <= 1.10 times control, every paired request KL <= 1.25 times control, all values "
                "are finite, route maps differ only at the target, and an immediate repeat is identical."
            ),
            "replication": (
                "three existing seed-context route maps x twelve canonical requests, each with one exact "
                "immediate repeat; no new training seeds, resampling, timing arm, or cost condition"
            ),
            "controls_for_existing_variation": (
                "Pairing within each seed-context/request holds request and seed-associated route context "
                "fixed. Forced same-unit maps remove timing and lambda as varying factors."
            ),
            "out_of_scope_units": (
                "The 30 units never downgraded in the same-unit cost comparison retain conservative 8-bit "
                "status and receive no lowest-safe label; testing lower bits for them would widen the "
                "smallest evidence-directed study."
            ),
            "prohibited_here": [
                "execution of this study",
                "training or retraining",
                "lambda tuning",
                "S11-D reruns",
                "lookahead evaluation",
                "canonical evidence mutation",
            ],
        },
    }


def serialize_diagnostic(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
