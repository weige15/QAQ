from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
POLICY_PATHS = (
    ROOT / "AGENTS.md",
    ROOT / "README.md",
    ROOT / ".pi/prompts/qaq-firstmate.md",
    ROOT / ".pi/rules/qaq-runtime.md",
    ROOT / ".pi/rules/qaq-git-worktrees.md",
    ROOT / ".pi/rules/qaq-worker-sessions.md",
    ROOT / ".pi/rules/qaq-stage-execution.md",
    ROOT / "docs/FIRSTMATE.md",
)


def _policy_text() -> str:
    return "\n".join(path.read_text(encoding="utf-8") for path in POLICY_PATHS)


def test_firstmate_policy_does_not_reintroduce_micro_permission_markers() -> None:
    text = _policy_text()
    forbidden = (
        "HARD STOP:",
        "WAITING_FOR_BUILD_RESULT",
        "WAITING_FOR_LANDING_RESULT",
        "Give exactly one current step, then stop and wait for the user",
        "Implementation, landing, and pushing are separate operations",
        "Never push without explicit authorization for that exact push",
    )

    for marker in forbidden:
        assert marker not in text


def test_firstmate_policy_authorizes_full_stage_delivery() -> None:
    agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    prompt = (ROOT / ".pi/prompts/qaq-firstmate.md").read_text(encoding="utf-8")

    required_agents = (
        "one\ncomplete current-objective delivery cycle",
        "Do not request permission between those routine actions",
        "push or open/update a PR when required by the active delivery contract",
        "Do not automatically begin a follow-up objective",
    )
    required_prompt = (
        "complete the whole authorized current-objective cycle",
        "without asking for permission between those actions",
        "material-decision boundary in `AGENTS.md`",
        "Do not begin a follow-up objective automatically",
    )

    for marker in required_agents:
        assert marker in agents
    for marker in required_prompt:
        assert marker in prompt


def test_firstmate_policy_limits_escalation_to_material_decisions() -> None:
    agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    stage_rules = (ROOT / ".pi/rules/qaq-stage-execution.md").read_text(
        encoding="utf-8"
    )

    assert "Escalate only when proceeding would require at least one" in agents
    assert "Everything else is a worker choice" in stage_rules
    assert "A failed check is not automatically a captain decision" in stage_rules
