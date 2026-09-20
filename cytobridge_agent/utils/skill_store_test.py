import json

from cytobridge_agent.utils.skill_store import ensure_cellcompass_skills_migrated, _sha256


def _write(path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_skill_store_overwrites_stale_user_mirror_by_default(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("CYTOBRIDGE_SKILL_SYNC_MODE", raising=False)
    monkeypatch.delenv("CYTOBRIDGE_PRESERVE_USER_SKILLS", raising=False)
    builtin = tmp_path / "builtin"
    user = tmp_path / "user"
    rel = "workflow/downstream-analysis/SKILL.md"

    _write(builtin / rel, "repo v2\n")
    _write(user / rel, "stale user copy\n")
    _write(user / ".builtin_manifest.json", json.dumps({rel: "old-hash"}))

    ensure_cellcompass_skills_migrated(builtin_root=builtin, user_root=user)

    assert (user / rel).read_text(encoding="utf-8") == "repo v2\n"


def test_skill_store_removes_stale_files_by_default(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("CYTOBRIDGE_SKILL_SYNC_MODE", raising=False)
    monkeypatch.delenv("CYTOBRIDGE_PRESERVE_USER_SKILLS", raising=False)
    builtin = tmp_path / "builtin"
    user = tmp_path / "user"
    kept_rel = "workflow/kept/SKILL.md"
    stale_rel = "workflow/removed/SKILL.md"

    _write(builtin / kept_rel, "kept\n")
    _write(user / kept_rel, "old kept\n")
    _write(user / stale_rel, "removed from package\n")

    ensure_cellcompass_skills_migrated(builtin_root=builtin, user_root=user)

    assert (user / kept_rel).read_text(encoding="utf-8") == "kept\n"
    assert not (user / stale_rel).exists()


def test_skill_store_preserve_mode_keeps_local_edits(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CYTOBRIDGE_SKILL_SYNC_MODE", "preserve")
    builtin = tmp_path / "builtin"
    user = tmp_path / "user"
    rel = "workflow/downstream-analysis/SKILL.md"

    old_builtin = tmp_path / "old_builtin.md"
    _write(old_builtin, "repo v1\n")
    _write(builtin / rel, "repo v2\n")
    _write(user / rel, "local edit\n")
    _write(user / ".builtin_manifest.json", json.dumps({rel: _sha256(old_builtin)}))

    ensure_cellcompass_skills_migrated(builtin_root=builtin, user_root=user)

    assert (user / rel).read_text(encoding="utf-8") == "local edit\n"
