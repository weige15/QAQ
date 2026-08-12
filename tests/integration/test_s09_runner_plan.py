from __future__ import annotations

from pathlib import Path

from qaq import s09_runner as runner

ROOT = Path(__file__).resolve().parents[2]


def test_plan_validates_protocol_and_has_no_execution_side_effects(tmp_path, monkeypatch, capsys):
    calls = []

    def fake_validate(config_path, *, check_external, verify_hashes):
        calls.append((config_path, check_external, verify_hashes))
        return {"mode_count": 5}

    monkeypatch.setattr("scripts.validate_s09_protocol.validate_protocol", fake_validate)
    details = runner.plan(ROOT / "configs/s09_baseline_eval.json", tmp_path / "results", "cuda:3")
    assert details["safe"] is True
    assert details["model_loading"] is False
    assert details["cuda_inference"] is False
    assert details["benchmark"] is False
    assert details["writes_final_result"] is False
    assert len(details["child_commands"]) == 5
    assert details["aggregation_command"][details["aggregation_command"].index("--aggregate")] == "--aggregate"
    assert calls == [(ROOT / "configs/s09_baseline_eval.json", True, True)]
    assert not (tmp_path / "results").exists()
    assert capsys.readouterr().out == ""
