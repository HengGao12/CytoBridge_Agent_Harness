from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from .skill_store import get_cellcompass_skills_root


def get_workspace_root() -> Path:
    override = os.getenv("CYTOBRIDGE_WORKSPACE_ROOT", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


def get_package_root() -> Path:
    return get_workspace_root() / "CytoBridge-main"


def render_runtime_paths_context(output_dir: Optional[str] = None) -> str:
    cwd = Path.cwd().resolve()
    workspace_root = get_workspace_root()
    package_root = get_package_root()
    docs_root = package_root / "docs"
    cellcompass_root = (Path.home() / ".cellcompass").resolve()
    skills_root = get_cellcompass_skills_root()
    training_algorithms_root = (cellcompass_root / "training_algorithms").resolve()
    resolved_output = Path(output_dir).expanduser().resolve() if output_dir else None
    output_text = str(resolved_output) if resolved_output else "(not set yet)"
    scripts_text = str((resolved_output / "scripts").resolve()) if resolved_output else "(not set yet)"

    return "\n".join(
        [
            f"- Current working directory: `{cwd}`",
            f"- Workspace root: `{workspace_root}`",
            f"- CytoBridge package directory: `{package_root}`",
            f"- CytoBridge docs directory: `{docs_root}`",
            f"- CellCompass home directory: `{cellcompass_root}`",
            f"- CellCompass skills root (absolute path): `{skills_root}`",
            f"- Custom training algorithms root (absolute path): `{training_algorithms_root}`",
            f"- Current output directory: `{output_text}`",
            f"- Output scripts root (absolute path): `{scripts_text}`",
            "- Resolve repo-relative paths in prompts, docs, and skills against the workspace root.",
            "- When referring to custom algorithm files, prefer the exact absolute path above or workspace-relative paths produced by tools.",
            "- Do not invent the literal path `/.cellcompass/...`; that is invalid. The home directory prefix is required.",
        ]
    )
