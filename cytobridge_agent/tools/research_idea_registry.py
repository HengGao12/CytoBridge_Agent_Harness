from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .training_tools import get_cellcompass_root


def research_idea_root(*, workspace_root: Optional[Path] = None) -> Path:
    root = workspace_root or get_cellcompass_root()
    return Path(root).expanduser().resolve() / "research_ideas"


def research_idea_dir(idea_id: str, *, workspace_root: Optional[Path] = None) -> Path:
    target = str(idea_id or "").strip().lower()
    return research_idea_root(workspace_root=workspace_root) / target


def research_idea_markdown_path(idea_id: str, *, workspace_root: Optional[Path] = None) -> Path:
    return research_idea_dir(idea_id, workspace_root=workspace_root) / "IDEA.md"


def research_idea_json_path(idea_id: str, *, workspace_root: Optional[Path] = None) -> Path:
    return research_idea_dir(idea_id, workspace_root=workspace_root) / "IDEA.json"


def research_idea_registry_dir(idea_id: str, *, workspace_root: Optional[Path] = None) -> Path:
    return research_idea_dir(idea_id, workspace_root=workspace_root) / "registry"


def research_idea_registry_index_path(idea_id: str, *, workspace_root: Optional[Path] = None) -> Path:
    return research_idea_registry_dir(idea_id, workspace_root=workspace_root) / "idea_registry.json"


def research_idea_revisions_dir(idea_id: str, *, workspace_root: Optional[Path] = None) -> Path:
    return research_idea_registry_dir(idea_id, workspace_root=workspace_root) / "revisions"


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def load_research_idea_record(idea_id: str, *, workspace_root: Optional[Path] = None) -> Dict[str, Any]:
    idea_path = research_idea_json_path(idea_id, workspace_root=workspace_root)
    payload = _read_json(idea_path, {})
    return payload if isinstance(payload, dict) else {}


def _summary_from_record(record: Dict[str, Any]) -> Dict[str, Any]:
    linked_algorithms = [
        str(item).strip().lower()
        for item in (record.get("linked_algorithms") or [])
        if str(item).strip()
    ]
    summary = {
        "idea_id": str(record.get("idea_id") or "").strip().lower(),
        "title": str(record.get("title") or "").strip(),
        "primary_track": str(record.get("primary_track") or "").strip(),
        "alternate_track": str(record.get("alternate_track") or "").strip(),
        "review_mode": str(record.get("review_mode") or "").strip(),
        "review_status": str(record.get("review_status") or "").strip(),
        "portfolio_status": str(record.get("portfolio_status") or "").strip(),
        "resolution_status": str(record.get("resolution_status") or "").strip(),
        "execution_status": str(record.get("execution_status") or "").strip(),
        "current_revision_id": str(record.get("current_revision_id") or "").strip(),
        "created_at": str(record.get("created_at") or "").strip(),
        "updated_at": str(record.get("updated_at") or "").strip(),
        "active": bool(record.get("active", False)),
        "linked_algorithms": linked_algorithms,
        "linked_algorithm_count": len(linked_algorithms),
        "attempt_count": int(record.get("attempt_count") or 0),
    }
    for key in ("problem_definition", "scientific_object", "falsifiable_success_criteria"):
        value = str(record.get(key) or "").strip()
        if value:
            summary[key] = value
    return summary


def list_research_ideas(*, workspace_root: Optional[Path] = None) -> List[Dict[str, Any]]:
    root = research_idea_root(workspace_root=workspace_root)
    if not root.exists():
        return []

    ideas: List[Dict[str, Any]] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        record = load_research_idea_record(child.name, workspace_root=workspace_root)
        if not record:
            continue
        summary = _summary_from_record(record)
        if summary.get("idea_id"):
            ideas.append(summary)
    ideas.sort(
        key=lambda item: (
            0 if item.get("active") else 1,
            str(item.get("updated_at") or ""),
            str(item.get("idea_id") or ""),
        ),
        reverse=False,
    )
    return ideas


def render_research_idea_catalog_context(
    *,
    workspace_root: Optional[Path] = None,
    active_idea_id: str = "",
    limit: int = 12,
) -> str:
    ideas = list_research_ideas(workspace_root=workspace_root)
    active_id = str(active_idea_id or "").strip().lower()
    if active_id:
        for item in ideas:
            if str(item.get("idea_id") or "").strip().lower() == active_id:
                item["active"] = True

    if not ideas:
        return (
            "## Research Ideas\n"
            "No research ideas have been recorded yet under `~/.cellcompass/research_ideas`."
        )

    capped = ideas[: max(1, int(limit))]
    lines = [
        "## Research Ideas",
        "Research ideas are persistent scientific-problem assets, separate from concrete algorithms.",
        "Use them to track question framing, review state, and linked algorithm attempts over time.",
    ]
    for item in capped:
        tags = [
            f"track={item.get('primary_track') or '(unset)'}",
            f"review={item.get('review_status') or '(unset)'}",
            f"portfolio={item.get('portfolio_status') or '(unset)'}",
            f"resolution={item.get('resolution_status') or '(unset)'}",
            f"execution={item.get('execution_status') or '(unset)'}",
        ]
        if item.get("active"):
            tags.append("active=true")
        lines.append(
            f"- {item.get('idea_id')}: {item.get('title') or '(untitled)'} "
            f"[{', '.join(tags)}; linked_algorithms={item.get('linked_algorithm_count', 0)}; "
            f"attempts={item.get('attempt_count', 0)}]"
        )
    if len(ideas) > len(capped):
        lines.append(f"- ... {len(ideas) - len(capped)} more idea(s) omitted")
    return "\n".join(lines)


__all__ = [
    "load_research_idea_record",
    "list_research_ideas",
    "render_research_idea_catalog_context",
    "research_idea_dir",
    "research_idea_json_path",
    "research_idea_markdown_path",
    "research_idea_registry_dir",
    "research_idea_registry_index_path",
    "research_idea_revisions_dir",
    "research_idea_root",
]
