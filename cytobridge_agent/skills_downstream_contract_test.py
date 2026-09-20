from pathlib import Path

from cytobridge_agent.tools.skills_loader import SkillsLoader


def test_downstream_skills_do_not_embed_external_file_contracts() -> None:
    root = Path(__file__).resolve().parent / "skills" / "downstream"
    forbidden = {
        "downstream-benchmark",
        "predicted_heldout",
        "holdout_prediction.csv",
        "velocity_field.csv",
        "growth_rates.csv",
        "per_cell_fate.json",
        "perturbation_results.json",
        "driver_genes.json",
    }

    offenders = []
    for path in root.rglob("*.md"):
        text = path.read_text(encoding="utf-8")
        hits = sorted(token for token in forbidden if token in text)
        if hits:
            offenders.append(f"{path.relative_to(root)}: {', '.join(hits)}")

    assert not offenders, "Downstream skills must not embed task-specific file contracts:\n" + "\n".join(offenders)


def test_downstream_skill_creator_moved_to_planner_domain() -> None:
    skills_root = Path(__file__).resolve().parent / "skills"

    assert not (skills_root / "downstream" / "skill-creator" / "SKILL.md").exists()
    assert (skills_root / "planner" / "skill-creator" / "SKILL.md").exists()


def test_downstream_runtime_docs_are_canonical() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    downstream_refs = Path(__file__).resolve().parent / "skills" / "downstream" / "references"
    docs_root = repo_root / "CytoBridge-main" / "docs" / "runtime" / "downstream"

    assert not downstream_refs.exists()
    for name in [
        "README.md",
        "semantics-and-evidence.md",
        "model-semantics.md",
        "api-reference.md",
        "recipes-and-artifacts.md",
        "checklist-and-failures.md",
    ]:
        assert (docs_root / name).exists(), name


def test_dynbench_skills_off_masks_downstream_skills(monkeypatch) -> None:
    monkeypatch.setenv("CYTOBRIDGE_DISABLE_DOWNSTREAM_SKILLS", "1")

    assert SkillsLoader(domain="downstream").discover() == []
    workflow_names = {item["name"] for item in SkillsLoader(domain="workflow").discover()}
    assert "downstream-analysis" not in workflow_names
    assert "downstream-visualization" not in workflow_names
    assert "preprocessing-execution" in workflow_names
