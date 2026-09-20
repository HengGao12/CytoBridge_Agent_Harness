"""State-backed helper methods for loading skills as prompt resources."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set
import json
import re

from .skills_loader import SkillsLoader


def _now_iso() -> str:
    return datetime.now().isoformat()


class SkillsTools:
    """Manage loaded prompt skills in shared state."""

    def __init__(
        self,
        state: Dict[str, Any],
        domain: str = "downstream",
        event_sink: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        active_key: str = "active_skills",
        revision_key: str = "skills_revision",
        policy_key: str = "skill_policy",
    ) -> None:
        self.state = state
        self.domain = domain
        self.event_sink = event_sink
        self.active_key = active_key
        self.revision_key = revision_key
        self.policy_key = policy_key
        self.loader = SkillsLoader(domain=domain)
        self._ensure_defaults()

    def _ensure_defaults(self) -> None:
        self.state.setdefault(self.active_key, [])
        self.state.setdefault(self.revision_key, 0)
        self.state.setdefault(
            "hidden_skills",
            {
                "workflow": [],
                "planner": [],
                "downstream": [],
                "algorithm": [],
            },
        )
        self.state.setdefault(
            self.policy_key,
            {
                "enabled": True,
                "domain": self.domain,
                "prefer_user_dir": True,
                "inject_max_chars": 32000,
            },
        )

    def _emit(self, event_type: str, payload: Dict[str, Any]) -> None:
        if not self.event_sink:
            return
        try:
            self.event_sink(event_type, payload)
        except Exception:
            pass

    def _state_snapshot(self) -> Dict[str, Any]:
        return {
            "domain": self.domain,
            "active_key": self.active_key,
            "revision_key": self.revision_key,
            "policy_key": self.policy_key,
        }

    @staticmethod
    def _with_truncation_note(base: str, max_chars: int, note: str = "\n\n[...skills context truncated...]") -> str:
        if max_chars <= 0:
            return ""
        if len(base) + len(note) <= max_chars:
            return base + note
        if len(note) >= max_chars:
            return note[:max_chars]
        keep = max_chars - len(note)
        return base[:keep] + note

    def list_skills(self) -> str:
        skills = []
        for item in self.loader.discover():
            skills.append(
                {
                    "name": item.get("name", ""),
                    "description": item.get("description", ""),
                    "skill_md_path": item.get("skill_md_path", ""),
                }
            )
        payload = {
            "skills": skills,
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def domain_description(self) -> str:
        return self.loader.domain_description()

    def _hidden_names(self) -> Set[str]:
        raw = self.state.get("hidden_skills") or {}
        domain_raw = raw.get(self.domain, []) if isinstance(raw, dict) else []
        hidden: Set[str] = set()
        for item in domain_raw or []:
            name = str(item or "").strip().lower()
            if name:
                hidden.add(name)
        return hidden

    def _set_hidden_names(self, hidden: Set[str]) -> None:
        raw = dict(self.state.get("hidden_skills") or {})
        raw[self.domain] = sorted(hidden)
        self.state["hidden_skills"] = raw

    def set_skill_exposed(self, name: str, exposed: bool) -> str:
        target = (name or "").strip().lower()
        if not target:
            return "❌ Skill name is required."

        available = {
            str(item.get("name", "")).strip().lower()
            for item in self.loader.discover()
            if str(item.get("name", "")).strip()
        }
        if target not in available:
            return f"❌ Skill not found: {name}"

        hidden = set(self._hidden_names())
        if exposed:
            if target not in hidden:
                return f"✅ Skill already exposed: {target}"
            hidden.discard(target)
        else:
            if target in hidden:
                return f"✅ Skill already hidden: {target}"
            hidden.add(target)

        self._set_hidden_names(hidden)
        self.state[self.revision_key] = int(self.state.get(self.revision_key, 0)) + 1
        self._emit(
            "skill_visibility_updated",
            {
                "name": target,
                "exposed": bool(exposed),
                "domain": self.domain,
                **self._state_snapshot(),
            },
        )
        return f"✅ Skill {'exposed' if exposed else 'hidden'}: {target}"

    def load_skill(self, name: str) -> str:
        skill = self.loader.load(name)
        if not skill:
            return f"❌ Skill not found: {name}"

        active: List[Dict[str, Any]] = list(self.state.get(self.active_key, []))
        for item in active:
            if item.get("name") == skill.name:
                return f"✅ Skill already loaded: {skill.name}"

        active.append(
            {
                "name": skill.name,
                "description": skill.description,
                "source": skill.source,
                "skill_md_path": skill.skill_md_path,
                "content": skill.body,
                "loaded_at": _now_iso(),
            }
        )
        self.state[self.active_key] = active
        self.state[self.revision_key] = int(self.state.get(self.revision_key, 0)) + 1
        self._emit(
            "skill_loaded",
            {
                "name": skill.name,
                "source": skill.source,
                "path": skill.skill_md_path,
                **self._state_snapshot(),
            },
        )
        return f"✅ Loaded skill: {skill.name} ({skill.source})"

    def unload_skill(self, name: str) -> str:
        target = (name or "").strip().lower()
        active: List[Dict[str, Any]] = list(self.state.get(self.active_key, []))
        keep = [x for x in active if str(x.get("name", "")).lower() != target]
        if len(keep) == len(active):
            return f"❌ Skill is not loaded: {name}"
        self.state[self.active_key] = keep
        self.state[self.revision_key] = int(self.state.get(self.revision_key, 0)) + 1
        self._emit("skill_unloaded", {"name": target, **self._state_snapshot()})
        return f"✅ Unloaded skill: {target}"

    def show_loaded_skills(self) -> str:
        active = self.state.get(self.active_key, [])
        return json.dumps(
            {
                "count": len(active),
                "skills_revision": int(self.state.get(self.revision_key, 0)),
                "skills": active,
            },
            ensure_ascii=False,
            indent=2,
        )

    def render_prompt_context(self) -> str:
        """Legacy alias: return catalog + per-turn context without turn text."""
        return self.render_turn_context(turn_text="")

    def _discover_enabled(self) -> List[Dict[str, Any]]:
        """Return discovered skills filtered by optional active-skill allowlist."""
        discovered = self.loader.discover()
        hidden = self._hidden_names()
        discovered = [
            item
            for item in discovered
            if str(item.get("name", "")).strip().lower() not in hidden
        ]
        active_names = {
            str(item.get("name", "")).strip().lower()
            for item in self.state.get(self.active_key, [])
            if str(item.get("name", "")).strip()
        }
        if not active_names:
            return discovered
        out: List[Dict[str, Any]] = []
        for item in discovered:
            name = str(item.get("name", "")).strip().lower()
            if name in active_names:
                out.append(item)
        return out

    def _get_active_skill_names(self) -> List[str]:
        ordered: List[str] = []
        for item in self.state.get(self.active_key, []):
            name = str(item.get("name", "")).strip().lower()
            if name and name not in ordered:
                ordered.append(name)
        return ordered

    def _compute_turn_selection(
        self,
        turn_text: str,
        auto_skill_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        discovered = self.loader.discover()
        available_names = {
            str(item.get("name", "")).strip().lower()
            for item in discovered
            if str(item.get("name", "")).strip()
        }
        loaded_names = self._get_active_skill_names()
        prioritized_names: List[str] = []
        for name in loaded_names:
            if name in available_names and name not in prioritized_names:
                prioritized_names.append(name)

        return {
            "discovered": discovered,
            "prioritized_names": prioritized_names,
            "loaded_names": loaded_names,
            "auto_names": [],
            "mentioned_names": [],
            "path_hits": [],
        }

    def build_turn_context(
        self,
        turn_text: str,
        auto_skill_names: Optional[List[str]] = None,
    ) -> tuple[str, Dict[str, Any]]:
        policy = self.state.get(self.policy_key) or {}
        max_chars = int(policy.get("inject_max_chars", 32000))
        catalog = self.render_catalog_context()
        selection = self._compute_turn_selection(turn_text, auto_skill_names=auto_skill_names)
        loaded_meta: List[Dict[str, Any]] = []
        merged = catalog
        if len(merged) > max_chars:
            merged = self._with_truncation_note(catalog[:max_chars], max_chars)
        summary = {
            **self._state_snapshot(),
            "loaded": selection["loaded_names"],
            "auto": selection["auto_names"],
            "mentioned": selection["mentioned_names"],
            "path_hits": selection["path_hits"],
            "injected": [],
            "injected_meta": loaded_meta,
        }
        return merged, summary

    @staticmethod
    def _normalize_mention_name(name: str) -> str:
        s = str(name or "").strip().lower()
        s = re.sub(r"[^a-z0-9]+", "-", s)
        return re.sub(r"-{2,}", "-", s).strip("-")

    @staticmethod
    def _extract_skill_mentions(text: str) -> Dict[str, Any]:
        """Extract skill mentions by `$name` and `skill://...` / `.../SKILL.md` paths."""
        s = str(text or "")
        ordered_names: List[str] = []
        for m in re.finditer(r"\$([A-Za-z0-9_\-]+)", s):
            normalized = SkillsTools._normalize_mention_name(m.group(1))
            if normalized and normalized not in ordered_names:
                ordered_names.append(normalized)
        names = set(ordered_names)
        paths = set()
        for m in re.finditer(r"(skill://[^\s)\]]+|[^\s)\]]*SKILL\.md)", s, flags=re.IGNORECASE):
            paths.add(m.group(1).strip())
        return {"names": names, "paths": paths, "ordered_names": ordered_names}

    @staticmethod
    def _extract_plain_skill_mentions(text: str, available_names: Set[str]) -> List[str]:
        """Extract plain-text skill mentions like `skill creator` / `skill-creator`."""
        s = str(text or "").lower()
        if not s or not available_names:
            return []
        hits: List[tuple[int, str]] = []
        for name in available_names:
            if not name:
                continue
            variants = {
                name,
                name.replace("-", " "),
                name.replace("-", "_"),
            }
            best_index: Optional[int] = None
            for variant in variants:
                if not variant:
                    continue
                pattern = rf"(?<![a-z0-9]){re.escape(variant)}(?![a-z0-9])"
                match = re.search(pattern, s)
                if match:
                    idx = int(match.start())
                    if best_index is None or idx < best_index:
                        best_index = idx
            if best_index is not None:
                hits.append((best_index, name))
        hits.sort(key=lambda item: (item[0], item[1]))
        return [name for _, name in hits]

    def catalog_items(self, exclude_names: Optional[Set[str]] = None) -> List[Dict[str, Any]]:
        excluded = {str(name or "").strip().lower() for name in (exclude_names or set()) if str(name or "").strip()}
        hidden = self._hidden_names()
        return [
            item for item in self.loader.discover()
            if str(item.get("name", "")).strip().lower() not in excluded
            and str(item.get("name", "")).strip().lower() not in hidden
        ]

    def render_catalog_context(self, exclude_names: Optional[Set[str]] = None) -> str:
        """Render a compact codex/openclaw-style skill catalog (name/description/path only)."""
        skills = self.catalog_items(exclude_names=exclude_names)
        if not skills:
            return ""
        lines: List[str] = []
        lines.append("## Skills")
        lines.append("A skill is a local, file-backed operating manual stored in `SKILL.md`.")
        lines.append("Skills are not tools and are not automatically active in context.")
        lines.append("### Available skills")
        for item in skills:
            lines.append(
                f"- {item.get('name')}: {item.get('description', '')} (file: {item.get('skill_md_path')})"
            )
        lines.append("### Usage")
        lines.append("- When a skill is relevant, proactively read its `SKILL.md` with normal file tools before following it.")
        lines.append("- If the right skill or path is uncertain, call `list_skills()` first, then read the selected `SKILL.md`.")
        lines.append("- For the standard CytoBridge workflow, read `workflow-orchestrator` first, then only the current stage skill.")
        lines.append("- Do not load many skills at once unless the task genuinely spans multiple stages or needs a specialized auxiliary skill.")
        lines.append("- Prefer `read_file`, `find_files`, and `grep_files` to inspect skill docs on demand.")
        lines.append("- Do not assume a skill is active unless you have actually read its `SKILL.md` in this run.")
        return "\n".join(lines)

    def render_turn_context(self, turn_text: str, auto_skill_names: Optional[List[str]] = None) -> str:
        """Render codex-style skill context: catalog + current-turn explicit injections."""
        merged, _ = self.build_turn_context(turn_text=turn_text, auto_skill_names=auto_skill_names)
        return merged
