from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

from .paper_review_freshness import sync_paper_reviewer_gate_from_review


_PHASE_ORDER = [
    "intake",
    "preprocessing",
    "theory_selection",
    "training",
    "downstream_analysis",
    "reporting",
    "completed",
]


class WorkflowCommitter:
    def __init__(self, state: Dict[str, Any], event_sink=None):
        self.state = state
        self.event_sink = event_sink

    def _emit(self, event_type: str, payload: Dict[str, Any]) -> None:
        if self.event_sink:
            try:
                self.event_sink(event_type, payload)
            except Exception:
                pass

    def _normalize_updates(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        normalized = dict(updates)
        final_config = normalized.get("final_config")
        if final_config is None or isinstance(final_config, dict):
            return normalized

        latest_run = None
        candidate_runs = normalized.get("training_runs", self.state.get("training_runs")) or []
        if candidate_runs and isinstance(candidate_runs[-1], dict):
            latest_run = candidate_runs[-1]

        final_config_dict: Dict[str, Any] = {"name": str(final_config)}
        if latest_run:
            model_path = latest_run.get("model_artifact_path") or latest_run.get("trained_model_path")
            if model_path:
                final_config_dict["path"] = model_path
            if latest_run.get("run_id"):
                final_config_dict["run_id"] = latest_run["run_id"]
            if latest_run.get("run_dir"):
                final_config_dict["run_dir"] = latest_run["run_dir"]
            if latest_run.get("resolved_config_path"):
                final_config_dict["resolved_config_path"] = latest_run["resolved_config_path"]
        normalized["final_config"] = final_config_dict
        return normalized

    def commit(
        self,
        phase: str,
        updates: Dict[str, Any] | None = None,
        artifacts: Dict[str, Any] | None = None,
        summary: str = "",
        mark_phase_complete: bool = True,
    ) -> str:
        updates = self._normalize_updates(dict(updates or {}))
        artifacts = dict(artifacts or {})

        self.state.update(updates)
        artifact_index = dict(self.state.get("artifact_index") or {})
        if artifacts:
            phase_artifacts = dict(artifact_index.get(phase) or {})
            phase_artifacts.update(artifacts)
            artifact_index[phase] = phase_artifacts
            self.state["artifact_index"] = artifact_index

        paper_review = updates.get("paper_review")
        if isinstance(paper_review, dict):
            relevant_paths = []
            manifest = paper_review.get("reviewed_file_hashes")
            if isinstance(manifest, dict):
                relevant_paths.extend(manifest.get("source_paths") or [])
            output_dir = self.state.get("output_dir")
            if output_dir:
                relevant_paths.append(output_dir)
            gate_sync = sync_paper_reviewer_gate_from_review(paper_review, relevant_paths)
            self.state["paper_reviewer_gate_sync"] = gate_sync

        phase_status = dict(self.state.get("phase_status") or {})
        if phase:
            phase_status.setdefault(phase, "pending")
            phase_status[phase] = "completed" if mark_phase_complete else "in_progress"
        self.state["phase_status"] = phase_status

        next_phase = phase
        if mark_phase_complete and phase in _PHASE_ORDER:
            idx = _PHASE_ORDER.index(phase)
            next_phase = _PHASE_ORDER[min(idx + 1, len(_PHASE_ORDER) - 1)]
        self.state["workflow_phase"] = next_phase

        entry = {
            "phase": phase,
            "summary": str(summary or ""),
            "updates": sorted(updates.keys()),
            "artifacts": artifacts,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "next_phase": next_phase,
        }
        audit = list(self.state.get("workflow_audit_log") or [])
        audit.append(entry)
        self.state["workflow_audit_log"] = audit

        self._emit(
            "workflow_state_committed",
            {
                "phase": phase,
                "summary": summary,
                "updates": updates,
                "artifacts": artifacts,
                "next_phase": next_phase,
                "phase_status": phase_status,
            },
        )
        self._emit(
            "stage",
            {
                "stage": next_phase,
                "message": f"workflow phase advanced to {next_phase}",
            },
        )
        return (
            f"✅ Workflow state committed for phase='{phase}'. "
            f"Next phase: {next_phase}. Updated keys: {sorted(updates.keys())}"
        )
