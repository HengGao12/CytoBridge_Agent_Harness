"""Canonical runtime snapshots shared by CLI, TUI, Web, and supervisors."""
from __future__ import annotations

from typing import Any, Dict, Optional

from .campaign_state import resolve_active_campaign, state_with_resolved_campaign
from .job_registry import JobRegistry
from .lifecycle_state import summarize_algorithm_lifecycle
from .runtime_events import _json_safe
from .runtime_state import RuntimeState


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _first_nonempty(*values: Any) -> Optional[Any]:
    for value in values:
        if value not in (None, ""):
            return value
    return None


def _active_algorithm_id(state: Dict[str, Any]) -> Optional[str]:
    active_context = _as_dict(state.get("active_algorithm_context"))
    return _first_nonempty(
        active_context.get("algorithm_id"),
        state.get("planner_algorithm_workspace") and str(state.get("planner_algorithm_workspace")).rstrip("/").split("/")[-1],
        _as_dict(state.get("active_experiment_registry")).get("algorithm_id"),
        state.get("latest_algorithm_proposal_id"),
        state.get("latest_training_algorithm_id"),
    )


def _artifact_paths(state: Dict[str, Any]) -> Dict[str, Any]:
    final_config = _as_dict(state.get("final_config"))
    return {
        "final_config_path": final_config.get("path"),
        "latest_training_run_dir": state.get("latest_training_run_dir"),
        "report_path": state.get("report_path"),
        "paper_path": state.get("paper_path") or state.get("latest_paper_path"),
    }


def _algorithm_snapshot(state: Dict[str, Any]) -> Dict[str, Any]:
    active_campaign = _as_dict(state.get("active_algorithm_campaign"))
    active_context = _as_dict(state.get("active_algorithm_context"))
    campaign_id = _first_nonempty(state.get("active_algorithm_campaign_id"), active_campaign.get("campaign_id"))
    lifecycle = summarize_algorithm_lifecycle(state)
    return {
        "active_algorithm_id": _active_algorithm_id(state),
        "active_algorithm_context": active_context,
        "latest_algorithm_proposal_id": state.get("latest_algorithm_proposal_id"),
        "latest_training_algorithm_id": state.get("latest_training_algorithm_id"),
        "active_campaign_id": campaign_id,
        "active_campaign_stage": active_campaign.get("current_stage"),
        "active_campaign_status": active_campaign.get("status"),
        "baseline_mode": _first_nonempty(
            active_campaign.get("baseline_mode"),
            _as_dict(active_campaign.get("baseline_policy")).get("mode"),
            state.get("campaign_baseline_mode"),
        ),
        "lifecycle": lifecycle,
    }


def _session_file_tools(session: Any) -> Optional[Any]:
    planner = getattr(session, "planner", None)
    tools_handler = getattr(planner, "tools_handler", None)
    return getattr(tools_handler, "planner_file_tools", None)


def _workflow_snapshot(state: Dict[str, Any], *, stop_requested: bool) -> Dict[str, Any]:
    return {
        "conversation_turn": state.get("conversation_turn"),
        "planner_phase": state.get("planner_phase"),
        "workflow_phase": state.get("workflow_phase"),
        "context_policy": _as_dict(state.get("context_policy")),
        "llm_context_usage": _as_dict(state.get("llm_context_usage")),
        "stop_requested": bool(stop_requested),
        "stop_hook_enabled": state.get("stop_hook_enabled"),
        "stop_hook_mode": state.get("stop_hook_mode"),
        "stop_hook_max_triggers": state.get("stop_hook_max_triggers"),
    }


def build_runtime_snapshot(
    *,
    session: Any,
    runtime_state: RuntimeState,
    jobs: JobRegistry,
    initial_config: Optional[Dict[str, Any]] = None,
    stop_requested: bool = False,
    ownership: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the single status object adapters should render.

    The active ``InteractiveSession.state`` remains the mutable source for agent
    workflow state. This function only normalizes it into stable sections so
    adapters and supervisors stop reconstructing slightly different meanings.
    """

    cfg = dict(initial_config or {})
    raw_state = _as_dict(getattr(session, "state", None))
    campaign_resolution = resolve_active_campaign(raw_state, file_tools=_session_file_tools(session))
    state = state_with_resolved_campaign(raw_state, campaign_resolution)
    session_id = getattr(session, "session_id", None)
    active_jobs = jobs.list(session_id=session_id, active_only=True) if session_id else jobs.list(active_only=True)
    stale_jobs = jobs.stale_jobs()
    input_path = _first_nonempty(getattr(session, "input_path", None), state.get("input_path"))
    output_dir = _first_nonempty(getattr(session, "output_dir", None), state.get("output_dir"))
    final_config = _as_dict(state.get("final_config"))

    snapshot = {
        "schema_version": 1,
        "has_active_session": session is not None,
        "state_sources": {
            "runtime": "process_local_runtime_state",
            "session": "active_interactive_session",
            "workflow": "active_session.state",
            "jobs": str(jobs.path),
        },
        "runtime": runtime_state.to_dict(),
        "ownership": _as_dict(ownership),
        "session": {
            "session_id": session_id,
            "input_path": input_path,
            "output_dir": output_dir,
        },
        "model": {
            "provider": cfg.get("llm_provider") or state.get("llm_provider") or "auto",
            "model": cfg.get("llm_model") or state.get("llm_model") or "unknown",
            "thinking_level": cfg.get("llm_thinking_level") or state.get("llm_thinking_level") or "low",
            "auth_mode": cfg.get("llm_auth_mode") or state.get("llm_auth_mode"),
            "profile_id": cfg.get("llm_profile_id") or state.get("llm_profile_id"),
        },
        "workflow": _workflow_snapshot(state, stop_requested=stop_requested),
        "data": {
            "input_path": input_path,
            "preprocessed_path": state.get("preprocessed_path"),
            "final_config_path": final_config.get("path"),
            "active_data_path": _first_nonempty(final_config.get("path"), state.get("preprocessed_path"), input_path),
        },
        "algorithm": _algorithm_snapshot(state),
        "artifacts": _artifact_paths(state),
        "jobs": {
            "active_count": len(active_jobs),
            "stale_count": len(stale_jobs),
            "active": active_jobs,
            "stale": stale_jobs,
        },
    }
    snapshot["algorithm"]["campaign_resolution"] = {
        key: value
        for key, value in campaign_resolution.items()
        if key != "campaign"
    }
    return _json_safe(snapshot)


def build_saved_session_snapshot(
    *,
    session_id: str,
    metadata: Optional[Dict[str, Any]],
    agent_state: Optional[Dict[str, Any]],
    jobs: JobRegistry,
    ownership: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a canonical snapshot for a stored conversation.

    This is intentionally marked as ``saved`` rather than process-running:
    supervisors can compare it with the active runtime snapshot without
    confusing checkpoint state for a live agent process.
    """

    meta = _as_dict(metadata)
    raw_state = _as_dict(agent_state)
    campaign_resolution = resolve_active_campaign(raw_state, file_tools=None)
    state = state_with_resolved_campaign(raw_state, campaign_resolution)
    active_jobs = jobs.list(session_id=session_id, active_only=True)
    stale_jobs = [job for job in jobs.stale_jobs() if job.get("session_id") == session_id]
    input_path = _first_nonempty(meta.get("input_path"), state.get("input_path"))
    output_dir = _first_nonempty(meta.get("output_dir"), state.get("output_dir"))
    final_config = _as_dict(state.get("final_config"))
    updated_at = _first_nonempty(meta.get("updated_at"), state.get("updated_at"))

    snapshot = {
        "schema_version": 1,
        "has_active_session": False,
        "state_sources": {
            "runtime": "stored_conversation_snapshot",
            "session": "conversation_store.metadata",
            "workflow": "conversation_store.agent_state",
            "jobs": str(jobs.path),
        },
        "runtime": {
            "status": "saved",
            "session_id": session_id,
            "message": "stored conversation snapshot",
            "since": updated_at,
            "last_error": None,
            "active_turn_id": None,
            "active_turn_started_at": None,
            "active_elapsed_seconds": 0.0,
            "is_terminal": False,
        },
        "ownership": _as_dict(ownership),
        "session": {
            "session_id": session_id,
            "input_path": input_path,
            "output_dir": output_dir,
            "title": meta.get("title"),
            "updated_at": updated_at,
        },
        "model": {
            "provider": state.get("llm_provider") or meta.get("llm_provider") or "unknown",
            "model": state.get("llm_model") or meta.get("llm_model") or "unknown",
            "thinking_level": state.get("llm_thinking_level") or meta.get("llm_thinking_level") or "unknown",
            "auth_mode": state.get("llm_auth_mode") or meta.get("llm_auth_mode"),
            "profile_id": state.get("llm_profile_id") or meta.get("llm_profile_id"),
        },
        "workflow": _workflow_snapshot(state, stop_requested=False),
        "data": {
            "input_path": input_path,
            "preprocessed_path": state.get("preprocessed_path"),
            "final_config_path": final_config.get("path"),
            "active_data_path": _first_nonempty(final_config.get("path"), state.get("preprocessed_path"), input_path),
        },
        "algorithm": _algorithm_snapshot(state),
        "artifacts": _artifact_paths(state),
        "jobs": {
            "active_count": len(active_jobs),
            "stale_count": len(stale_jobs),
            "active": active_jobs,
            "stale": stale_jobs,
        },
    }
    snapshot["algorithm"]["campaign_resolution"] = {
        key: value
        for key, value in campaign_resolution.items()
        if key != "campaign"
    }
    return _json_safe(snapshot)


def legacy_status_from_snapshot(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Compatibility fields for existing callers during adapter migration."""

    session = _as_dict(snapshot.get("session"))
    model = _as_dict(snapshot.get("model"))
    workflow = _as_dict(snapshot.get("workflow"))
    jobs = _as_dict(snapshot.get("jobs"))
    return {
        "has_active_session": snapshot.get("has_active_session"),
        "session_id": session.get("session_id"),
        "runtime": snapshot.get("runtime"),
        "ownership": snapshot.get("ownership") or {},
        "input_path": session.get("input_path"),
        "output_dir": session.get("output_dir"),
        "provider": model.get("provider"),
        "model": model.get("model"),
        "thinking_level": model.get("thinking_level"),
        "stop_requested": workflow.get("stop_requested"),
        "active_jobs": jobs.get("active") or [],
        "stale_jobs": jobs.get("stale") or [],
        "conversation_turn": workflow.get("conversation_turn"),
        "planner_phase": workflow.get("planner_phase"),
        "stop_hook_enabled": workflow.get("stop_hook_enabled"),
        "stop_hook_mode": workflow.get("stop_hook_mode"),
        "stop_hook_max_triggers": workflow.get("stop_hook_max_triggers"),
    }
