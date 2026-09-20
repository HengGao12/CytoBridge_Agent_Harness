from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatch
import os
from pathlib import Path
from typing import Any, Optional

from ..utils.skill_store import get_cellcompass_skills_root
from ..utils.runtime_paths import get_workspace_root
from .training_tools import get_cellcompass_root


def get_cytobridge_portfolio_root() -> Path:
    override = os.getenv("CYTOBRIDGE_PORTFOLIO_ROOT", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return Path("/data/cytobridge/portfolio").resolve()


@dataclass
class WorkspacePolicy:
    workspace_root: Path
    output_root: Path
    read_roots: list[Path]
    write_roots: list[Path]
    blocked_roots: list[Path]
    unrestricted_reads: bool = True
    writable_output_dirs: tuple[str, ...] = ("training_runs", "figures", "scripts", "paper")
    writable_output_patterns: tuple[str, ...] = ("report*.html", "report*.md")

    def resolve_user_path(self, raw_path: str) -> Path:
        candidate = Path(str(raw_path or "").strip()).expanduser()
        if candidate.is_absolute():
            return candidate.resolve()
        return (self.workspace_root / candidate).resolve()

    @staticmethod
    def _is_within(path: Path, root: Path) -> bool:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False

    def ensure_within_allowed_roots(self, path: Path, roots: list[Path], label: str) -> Path:
        for root in roots:
            if self._is_within(path, root):
                return path
        allowed = ", ".join(str(root) for root in roots)
        raise ValueError(f"Path '{path}' is outside allowed {label} roots: {allowed}")

    def _assert_not_blocked(self, path: Path) -> None:
        for root in self.blocked_roots:
            if self._is_within(path, root):
                raise ValueError(f"Path '{path}' is blocked by workspace policy (under {root})")

    def _is_allowed_output_write(self, path: Path) -> bool:
        if not self._is_within(path, self.output_root):
            return False
        rel = path.relative_to(self.output_root)
        if len(rel.parts) == 1 and any(fnmatch(rel.name, pattern) for pattern in self.writable_output_patterns):
            return True
        return bool(rel.parts) and rel.parts[0] in self.writable_output_dirs

    def validate_read_path(self, raw_path: str) -> Path:
        path = self.resolve_user_path(raw_path)
        if self.unrestricted_reads:
            return path
        return self.ensure_within_allowed_roots(path, self.read_roots, "read")

    def validate_write_path(self, raw_path: str) -> Path:
        path = self.resolve_user_path(raw_path)
        self._assert_not_blocked(path)
        for root in self.write_roots:
            if self._is_within(path, root):
                return path
        if self._is_allowed_output_write(path):
            return path
        allowed = ", ".join(str(root) for root in self.write_roots)
        raise ValueError(
            f"Path '{path}' is outside allowed write roots. Allowed roots: {allowed}; "
            "allowed output targets under output_dir: report*.html/report*.md, "
            "figures/**, training_runs/**, scripts/**, paper/**. "
            "Treat _workspace/ as a read-only task input/provenance area; save generated "
            "scripts under output_dir/scripts/** instead."
        )

    def to_state_dict(self) -> dict[str, Any]:
        return {
            "workspace_root": str(self.workspace_root),
            "output_root": str(self.output_root),
            "read_roots": [str(root) for root in self.read_roots],
            "write_roots": [str(root) for root in self.write_roots],
            "blocked_roots": [str(root) for root in self.blocked_roots],
            "unrestricted_reads": bool(self.unrestricted_reads),
            "writable_output_dirs": list(self.writable_output_dirs),
            "writable_output_patterns": list(self.writable_output_patterns),
        }

    def render_prompt_block(self) -> str:
        data = self.to_state_dict()
        lines = ["Planner workspace policy:"]
        lines.append(f"- workspace_root: {data['workspace_root']}")
        lines.append(f"- output_root: {data['output_root']}")
        lines.append(f"- unrestricted_reads: {data['unrestricted_reads']}")
        lines.append("- writable roots:")
        lines.extend(f"  - {root}" for root in data["write_roots"])
        lines.append("- blocked roots:")
        lines.extend(f"  - {root}" for root in data["blocked_roots"])
        lines.append("- additional writable output patterns:")
        for pattern in data["writable_output_patterns"]:
            lines.append(f"  - {pattern}")
        for name in data["writable_output_dirs"]:
            lines.append(f"  - {name}/**")
        return "\n".join(lines)


def build_planner_workspace_policy(
    state: dict[str, Any],
    workspace_root: Optional[Path] = None,
) -> WorkspacePolicy:
    def _include_resolved_aliases(roots: list[Path]) -> list[Path]:
        expanded: list[Path] = []
        for root in roots:
            expanded.append(root)
            resolved = root.expanduser().resolve()
            if resolved != root:
                expanded.append(resolved)
        return expanded

    workspace_root = Path(workspace_root or get_workspace_root()).expanduser().resolve()
    output_root = Path(state.get("output_dir") or (workspace_root / "cytobridge_output")).expanduser().resolve()
    user_algo_root = get_cellcompass_root() / "training_algorithms"
    user_skills_root = get_cellcompass_skills_root()
    portfolio_root = get_cytobridge_portfolio_root()

    read_roots = [workspace_root, output_root, Path.home() / ".cellcompass"]
    write_roots = [
        user_algo_root,
        user_skills_root,
        portfolio_root,
        output_root / "training_runs",
        output_root / "figures",
        output_root / "paper",
    ]
    write_roots = _include_resolved_aliases(write_roots)
    blocked_roots = [
        workspace_root / "CytoBridge-main" / "CytoBridge",
        workspace_root / "cytobridge_agent",
        workspace_root / "web",
        workspace_root / ".git",
    ]
    return WorkspacePolicy(
        workspace_root=workspace_root,
        output_root=output_root,
        read_roots=read_roots,
        write_roots=write_roots,
        blocked_roots=blocked_roots,
        unrestricted_reads=True,
    )
