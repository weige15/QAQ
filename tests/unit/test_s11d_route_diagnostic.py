from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from qaq.evaluation import s11d_route_diagnostic as diagnostic

ROOT = Path(__file__).parents[2]
DERIVED = ROOT / "docs/results/s11d_route_policy_diagnostic.json"


@pytest.fixture(scope="module")
def result() -> dict:
    return diagnostic.build_diagnostic()


def test_uses_exact_complete_canonical_trial_evidence(result):
    source = result["source_evidence"]
    assert source["aggregation_manifest"] == {
        "path": "docs/results/s11d_paired_468/aggregation.json",
        "sha256": diagnostic.EXPECTED_AGGREGATION_SHA256,
        "used_for": "canonical membership and request order only; route statistics use trial files",
    }
    assert [item["trial_id"] for item in source["trial_files"]] == list(
        diagnostic.EXPECTED_TRIAL_SHA256
    )
    assert {item["sha256"] for item in source["trial_files"]} == set(
        diagnostic.EXPECTED_TRIAL_SHA256.values()
    )
    assert result["coverage"]["decisions"] == 12 * 12 * 72 == 10368


def test_quantifies_usage_and_request_dependence(result):
    overall = result["usage"]["overall"]
    assert overall["counts"] == {"4": 54, "6": 2939, "8": 7375}
    assert sum(overall["fractions"].values()) == pytest.approx(1.0)
    assert len(result["usage"]["by_layer_and_unit_type"]) == 72
    assert len(result["usage"]["by_seed_timing_cost_trial"]) == 12
    assert len(result["usage"]["by_request_timing_cost"]) == 48

    dependence = result["request_dependence"]
    assert len(dependence["unit_trial_details"]) == 864
    assert all(item["request_count"] == 12 for item in dependence["unit_trial_details"])
    assert dependence["overall"]["invariant_unit_trials"]["successes"] == 744
    assert dependence["overall"]["modal_policy_fidelity"] == pytest.approx(
        0.9673996913580247
    )
    assert dependence["overall"]["request_pair_disagreement"]["fraction"] == pytest.approx(
        0.04955808080808081
    )
    assert dependence["determination"]["label"] == "mostly_static_unit_layer_policy"


def test_localizes_matched_variation_and_does_not_overstate_lookahead(result):
    comparisons = result["matched_variation_and_transitions"]
    timing = comparisons["matched_timing_same_unit_to_lookahead"]
    cost = comparisons["matched_cost_zero_to_0p03"]
    assert timing["lower_equal_higher"] == {"lower": 245, "equal": 4592, "higher": 347}
    assert timing["systematic_conservatism_determination"]["label"] == (
        "not_systematic_across_seeds"
    )
    assert timing["downgrade_localization"]["8_to_6"]["by_region"] == {
        "early_0_11": 123,
        "late_24_35": 56,
        "middle_12_23": 48,
    }
    assert cost["lower_equal_higher"] == {"lower": 1185, "equal": 3998, "higher": 1}
    assert cost["downgrade_localization"]["8_to_4"]["by_unit_type"] == {
        "attention": 42,
        "ffn": 7,
    }
    assert comparisons["matched_seed_pair_disagreement"]["fraction"] == pytest.approx(
        0.17785493827160495
    )


def test_future_study_is_bounded_and_unexecuted(result):
    study = result["proposed_same_unit_block_sensitivity"]
    assert study["status"] == "defined_not_executed"
    assert len(study["candidate_units"]) == 42
    assert all(item["unit_type"] in {"attention", "ffn"} for item in study["candidate_units"])
    assert "execution of this study" in study["prohibited_here"]
    assert "cannot determine" in result["causal_boundary"]


def test_checked_in_derived_artifact_is_reproducible(result):
    assert DERIVED.read_bytes() == diagnostic.serialize_diagnostic(result)
    parsed = json.loads(DERIVED.read_text())
    assert parsed["schema"] == "qaq-s11d-route-policy-diagnostic-v1"


def test_cli_refuses_to_write_into_canonical_trial_directory(tmp_path):
    forbidden = ROOT / "docs/results/s11d_paired_468/diagnostic.json"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/analyze_s11d_route_policy.py",
            "--output",
            str(forbidden),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 2
    assert "must not be written inside the canonical S11-D directory" in completed.stderr
    assert not forbidden.exists()
