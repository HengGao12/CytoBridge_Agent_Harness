from __future__ import annotations

from typing import Any, Dict


def render_phase_summary(state: Dict[str, Any]) -> str:
    phase = state.get("workflow_phase", "intake")
    lines = [f"Current workflow phase: {phase}"]
    status = state.get("phase_status") or {}
    final_config = state.get("final_config") or {}
    if not isinstance(final_config, dict):
        final_config = {}
    if status:
        lines.append("Phase status:")
        for key, value in status.items():
            lines.append(f"- {key}: {value}")
    if state.get("preprocessed_path"):
        lines.append(f"Preprocessed path: {state.get('preprocessed_path')}")
    if final_config.get("path"):
        lines.append(f"Trained model path: {final_config.get('path')}")
    if state.get("report_path"):
        lines.append(f"Report path: {state.get('report_path')}")
    return "\n".join(lines)
