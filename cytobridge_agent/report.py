from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Dict, Any

from .schemas import PilotResult


def render_report(
    summary: Dict[str, Any],
    decision: Dict[str, Any],
    pilot_results: List[PilotResult],
    final_cfg_path: Path,
    metrics_path: Path,
    figures_dir: Path,
) -> str:
    lines = []
    lines.append("# CytoBridge Agent Report")
    lines.append("")
    lines.append("## Data Summary")
    lines.append(f"- Cells: {summary.get('n_obs')}, Genes: {summary.get('n_vars')}")
    tcs = summary.get("time_candidates", [])
    if tcs:
        lines.append("- Time fields:")
        for tc in tcs:
            lines.append(
                f"  - {tc['key']}: {len(tc['levels'])} levels, counts {tc['counts']}"
            )
    lines.append("")
    lines.append("## LLM / Heuristic Decision")
    lines.append(f"- Model family: {decision.get('model_family')}")
    lines.append(f"- Rationale: {decision.get('why')}")
    lines.append("")
    lines.append("## Pilot Results")
    if pilot_results:
        lines.append("| candidate | success | W1 | time(s) | error |")
        lines.append("|---|---|---|---|---|")
        for r in pilot_results:
            w1 = "" if r.w1 is None else ";".join(f"{x:.4f}" for x in r.w1)
            lines.append(
                f"| {r.name} | {r.success} | {w1} | {r.runtime_sec or ''} | {r.error or ''} |"
            )
    else:
        lines.append("- No pilot runs recorded.")
    lines.append("")
    lines.append("## Final Outputs")
    lines.append(f"- Final config: `{final_cfg_path}`")
    lines.append(f"- Metrics: `{metrics_path}`")
    lines.append(f"- Figures: `{figures_dir}`")
    return "\n".join(lines)

