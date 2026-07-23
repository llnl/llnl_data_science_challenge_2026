from pathlib import Path
import tomllib


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_threshold_optimizer_skill_defines_required_workflow() -> None:
    skill_path = (
        REPOSITORY_ROOT / ".agents" / "skills" / "threshold-optimizer" / "SKILL.md"
    )
    content = skill_path.read_text(encoding="utf-8")

    assert "name: threshold-optimizer" in content
    assert "segment_ct_dataset" in content
    assert "visualize_slice" in content
    for threshold in ("0.3", "0.5", "0.7"):
        assert threshold in content
    assert "threshold_comparison.md" in content


def test_segmentation_agent_declares_required_outputs_and_limits() -> None:
    agent_path = REPOSITORY_ROOT / ".codex" / "agents" / "segmentation_agent.toml"
    agent = tomllib.loads(agent_path.read_text(encoding="utf-8"))
    instructions = agent["developer_instructions"]

    assert agent["name"] == "segmentation_agent"
    assert agent["sandbox_mode"] == "workspace-write"
    for artifact in (
        "segment_lattice.py",
        "segmented_mask.tif",
        "slice_380.png",
        "report.md",
    ):
        assert artifact in instructions
    assert "10 total iterations" in instructions
    assert "3 consecutive iterations" in instructions


def test_segmentation_rubric_has_criteria_and_json_contract() -> None:
    rubric_path = REPOSITORY_ROOT / "evals" / "rubric_segmentation_1.md"
    content = rubric_path.read_text(encoding="utf-8")

    for criterion in (
        "Structural Integrity",
        "False Positives/Negatives",
        "Topology",
        "Noise and Artifacts",
    ):
        assert criterion in content
    assert '"reasoning"' in content
    assert '"score"' in content
    for score in range(6):
        assert f"**{score}:" in content
