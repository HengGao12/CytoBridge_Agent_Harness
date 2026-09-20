"""Skill discovery/loading utilities for downstream prompt resources.

Skills are prompt resources (SKILL.md + optional references/scripts/assets).
They are not an execution engine.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import os
import re

from ..utils.skill_store import (
    ensure_cellcompass_skills_migrated,
    get_builtin_skills_root,
    get_cellcompass_skills_root,
)

DEPRECATED_SKILLS_BY_DOMAIN = {
    "planner": {
        "algorithm-development",
        "algorithm-proposal-theory",
        "algorithm-authoring",
        "algorithm-review-and-training",
        "algorithm-tuning-playbook",
    },
}


def _downstream_skills_disabled() -> bool:
    raw = os.environ.get("CYTOBRIDGE_DISABLE_DOWNSTREAM_SKILLS", "")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class SkillDefinition:
    name: str
    description: str
    body: str
    skill_md_path: str
    source: str  # "user" | "builtin"


class SkillsLoader:
    """Discover and load SKILL.md files from the CellCompass skill mirror."""

    def __init__(
        self,
        domain: str = "downstream",
        user_root: Optional[Path] = None,
        builtin_root: Optional[Path] = None,
    ) -> None:
        self.domain = domain
        self.user_root = Path(user_root).expanduser().resolve() if user_root else get_cellcompass_skills_root()
        self.builtin_root = Path(builtin_root).expanduser().resolve() if builtin_root else get_builtin_skills_root()
        ensure_cellcompass_skills_migrated(builtin_root=self.builtin_root, user_root=self.user_root)

    @staticmethod
    def _parse_frontmatter(text: str) -> Tuple[Dict[str, str], str]:
        """Parse YAML-like frontmatter without introducing extra dependencies."""
        stripped = text.lstrip()
        if not stripped.startswith("---"):
            return {}, text

        lines = stripped.splitlines()
        if len(lines) < 3:
            return {}, text

        end_idx = None
        for i in range(1, len(lines)):
            if lines[i].strip() == "---":
                end_idx = i
                break
        if end_idx is None:
            return {}, text

        fm_lines = lines[1:end_idx]
        body = "\n".join(lines[end_idx + 1 :]).strip()
        meta: Dict[str, str] = {}
        for line in fm_lines:
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip().strip("\"'")
        return meta, body

    @staticmethod
    def _normalize_token(raw: str) -> str:
        token = str(raw or "").strip().lower()
        token = re.sub(r"[^a-z0-9]+", "-", token)
        token = re.sub(r"-{2,}", "-", token).strip("-")
        return token

    @classmethod
    def _normalize_name(cls, raw_name: str, fallback_from_path: Path) -> str:
        fallback = cls._normalize_token(fallback_from_path.parent.name)
        name = cls._normalize_token(raw_name)
        return name or fallback or fallback_from_path.parent.name.lower()

    def _iter_skill_files(self) -> List[Path]:
        items: List[Path] = []
        if _downstream_skills_disabled() and self.domain == "downstream":
            return items
        user_dir = self.user_root / self.domain
        if user_dir.exists():
            for skill_md in sorted(user_dir.glob("*/SKILL.md")):
                if (
                    _downstream_skills_disabled()
                    and self.domain == "workflow"
                    and skill_md.parent.name.startswith("downstream-")
                ):
                    continue
                items.append(skill_md)
        return items

    def domain_description(self) -> str:
        """Return a short description for this skill domain, if provided."""
        candidates = [
            self.user_root / self.domain / "DESCRIPTION.md",
            self.builtin_root / self.domain / "DESCRIPTION.md",
        ]
        for path in candidates:
            if not path.exists():
                continue
            try:
                raw = path.read_text(encoding="utf-8")
            except Exception:
                continue
            fm, body = self._parse_frontmatter(raw)
            description = (fm.get("description", "") or "").strip()
            if description:
                return description
            first_line = next((ln.strip() for ln in body.splitlines() if ln.strip() and not ln.startswith("#")), "")
            if first_line:
                return first_line[:240]
        return ""

    def _infer_source(self, skill_md: Path) -> str:
        try:
            rel = skill_md.relative_to(self.user_root)
        except ValueError:
            return "user"
        if (self.builtin_root / rel).exists():
            return "builtin"
        return "user"

    def discover(self) -> List[Dict[str, str]]:
        """List available skills from the CellCompass mirror."""
        merged: Dict[str, Dict[str, str]] = {}
        for skill_md in self._iter_skill_files():
            try:
                raw = skill_md.read_text(encoding="utf-8")
            except Exception:
                continue
            fm, body = self._parse_frontmatter(raw)
            name = self._normalize_name(fm.get("name", ""), skill_md)
            if name in DEPRECATED_SKILLS_BY_DOMAIN.get(self.domain, set()):
                continue
            description = (fm.get("description", "") or "").strip()
            if not description:
                first_line = next((ln.strip() for ln in body.splitlines() if ln.strip()), "")
                description = first_line[:140]
            source = self._infer_source(skill_md)
            merged[name] = {
                "name": name,
                "description": description,
                "source": source,
                "skill_md_path": str(skill_md),
            }
        return sorted(merged.values(), key=lambda x: x["name"])

    def load(self, name: str) -> Optional[SkillDefinition]:
        """Load one skill by normalized name."""
        target = self._normalize_token(name)
        if not target:
            return None

        found = None
        for item in self.discover():
            if item["name"] == target:
                found = item
                break
        if not found:
            return None

        skill_md = Path(found["skill_md_path"])
        try:
            raw = skill_md.read_text(encoding="utf-8")
        except Exception:
            return None
        fm, body = self._parse_frontmatter(raw)
        normalized_name = self._normalize_name(fm.get("name", ""), skill_md)
        return SkillDefinition(
            name=normalized_name,
            description=(fm.get("description", "") or found.get("description", "")).strip(),
            body=body.strip(),
            skill_md_path=str(skill_md),
            source=found["source"],
        )
