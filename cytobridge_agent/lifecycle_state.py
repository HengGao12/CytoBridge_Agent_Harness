"""Normalized algorithm lifecycle state for runtime adapters and guards."""
from __future__ import annotations

from typing import Any, Dict, List, Optional


DEFAULT_ALGORITHM_STAGE_ORDER = [
    "stage1_feasibility",
    "stage2_claim_validation",
    "stage3_tuning",
    "final_regression",
]


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


def _text(value: Any) -> str:
    return str(value or "").strip()


def _first_nonempty(*values: Any) -> Optional[Any]:
    for value in values:
        if value not in (None, ""):
            return value
    return None


def _active_algorithm_id(state: Dict[str, Any], active_context: Dict[str, Any]) -> str:
    workspace = _text(state.get("planner_algorithm_workspace"))
    return _text(
        _first_nonempty(
            active_context.get("algorithm_id"),
            workspace.rstrip("/").split("/")[-1] if workspace else "",
            _as_dict(state.get("active_experiment_registry")).get("algorithm_id"),
            state.get("latest_training_algorithm_id"),
            state.get("latest_algorithm_proposal_id"),
        )
    )


def _stage_order(campaign: Dict[str, Any]) -> List[str]:
    order = [_text(item) for item in _as_list(campaign.get("stage_order")) if _text(item)]
    return order or list(DEFAULT_ALGORITHM_STAGE_ORDER)


def summarize_lifecycle_stage(stage_name: str, stage_state: Dict[str, Any], status_view: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    status_view = _as_dict(status_view)
    last_gate = _as_dict(stage_state.get("last_gate_check"))
    gate_evidence = _as_dict(stage_state.get("stage_gate_evidence"))
    blockers = [
        _text(item)
        for item in _as_list(last_gate.get("blockers") or gate_evidence.get("blockers"))
        if _text(item)
    ]
    active_best_trial_id = _text(
        _first_nonempty(
            status_view.get("active_best_trial_id"),
            stage_state.get("active_best_trial_id"),
            stage_state.get("gate_passed_trial_id"),
        )
    )
    stage_status = _text(status_view.get("stage_status") or stage_state.get("status"))
    gate_ready = bool(status_view.get("gate_ready") or stage_state.get("gate_ready"))
    last_gate_ok = bool(last_gate.get("ok", False))
    locked = stage_status == "locked"
    passed = bool(gate_ready or last_gate_ok or locked)
    return {
        "stage": stage_name,
        "status": stage_status,
        "stage_internal_status": _text(status_view.get("stage_internal_status")),
        "stage_gate_status": _text(status_view.get("stage_gate_status")),
        "passed": passed,
        "locked": locked,
        "gate_ready": gate_ready,
        "last_gate_ok": last_gate_ok,
        "last_gate_checked_at": _text(last_gate.get("checked_at")),
        "last_gate_blockers": blockers,
        "active_best_trial_id": active_best_trial_id,
        "current_trial_id": _text(status_view.get("current_trial_id") or stage_state.get("current_trial_id")),
        "trial_count": int(stage_state.get("trial_count") or 0),
        "promote_count": int(stage_state.get("promote_count") or 0),
        "reject_count": int(stage_state.get("reject_count") or 0),
        "gate_passed_trial_id": _text(stage_state.get("gate_passed_trial_id")),
        "gate_passed_at": _text(stage_state.get("gate_passed_at")),
        "next_required_action": _text(status_view.get("next_required_action")),
        "user_facing_status": _text(status_view.get("user_facing_status")),
    }


def summarize_algorithm_lifecycle(state: Dict[str, Any]) -> Dict[str, Any]:
    """Return a single finite-state lifecycle view from mutable planner state."""

    state = _as_dict(state)
    active_context = _as_dict(state.get("active_algorithm_context"))
    campaign = _as_dict(state.get("active_algorithm_campaign"))
    algorithm_id = _active_algorithm_id(state, active_context)
    campaign_id = _text(_first_nonempty(state.get("active_algorithm_campaign_id"), campaign.get("campaign_id")))
    raw_lifecycle_status = _text(
        _first_nonempty(
            campaign.get("algorithm_lifecycle_status"),
            active_context.get("algorithm_lifecycle_status"),
            state.get("algorithm_lifecycle_status"),
        )
    ) or ("developing" if algorithm_id else "")
    proposal_status = _text(active_context.get("proposal_status") or state.get("latest_algorithm_proposal_status"))
    campaign_status = _text(campaign.get("status"))
    current_stage = _text(campaign.get("current_stage"))
    order = _stage_order(campaign)
    stages_raw = _as_dict(campaign.get("stages"))
    status_views = _as_dict(campaign.get("stage_statuses"))
    stages = {
        stage_name: summarize_lifecycle_stage(
            stage_name,
            _as_dict(stages_raw.get(stage_name)),
            _as_dict(status_views.get(stage_name)),
        )
        for stage_name in order
    }
    current_stage_summary = stages.get(current_stage) if current_stage else None
    locked_release = _as_dict(campaign.get("locked_release") or active_context.get("completed_release") or state.get("completed_release"))
    release_locked = bool(locked_release) or campaign_status in {"locked", "complete", "completed"}
    lifecycle_complete = raw_lifecycle_status == "complete" and release_locked
    blockers: List[str] = []
    next_actions: List[str] = []

    if not algorithm_id:
        status = "no_algorithm"
        next_actions.append("propose_or_select_algorithm")
    elif raw_lifecycle_status == "failed" or campaign_status == "failed":
        status = "failed"
        blockers.append(_text(campaign.get("failure_reason") or active_context.get("algorithm_lifecycle_status_reason")) or "algorithm marked failed")
        next_actions.append("revise_proposal_or_restore_meaningful_algorithm")
    elif lifecycle_complete:
        status = "complete"
    elif not campaign_id:
        status = "proposal_or_implementation"
        next_actions.append("start_algorithm_campaign")
    elif current_stage == "final_regression":
        status = "final_regression"
        if current_stage_summary and current_stage_summary["last_gate_blockers"]:
            blockers.extend(current_stage_summary["last_gate_blockers"])
        next_actions.append("lock_final_regression_release")
    elif current_stage_summary and current_stage_summary["last_gate_blockers"]:
        status = "gate_blocked"
        blockers.extend(current_stage_summary["last_gate_blockers"])
        if current_stage_summary.get("next_required_action"):
            next_actions.append(str(current_stage_summary["next_required_action"]))
        else:
            next_actions.append(f"resolve_{current_stage}_gate_blockers")
    elif campaign_status in {"running", "active", "developing", ""}:
        status = "campaign_running"
        next_actions.append(f"continue_{current_stage or 'campaign'}")
    else:
        status = "campaign_pending"
        next_actions.append(f"inspect_campaign_status_{campaign_status}")

    if raw_lifecycle_status == "complete" and not release_locked:
        status = "inconsistent"
        blockers.append("algorithm_lifecycle_status is complete but no locked release is recorded")
        next_actions.append("repair_lifecycle_bookkeeping_or_run_final_regression")

    try:
        current_stage_index = order.index(current_stage) if current_stage else -1
    except ValueError:
        current_stage_index = -1

    return {
        "schema_version": 1,
        "status": status,
        "algorithm_id": algorithm_id,
        "proposal_status": proposal_status,
        "raw_lifecycle_status": raw_lifecycle_status,
        "lifecycle_complete": lifecycle_complete,
        "release_locked": release_locked,
        "campaign_id": campaign_id,
        "campaign_status": campaign_status,
        "current_stage": current_stage,
        "current_stage_index": current_stage_index,
        "stage_order": order,
        "stages": stages,
        "algorithm_validated": bool(campaign.get("algorithm_validated")),
        "locked_release": locked_release,
        "blockers": blockers,
        "next_actions": next_actions,
        "completion_criteria": {
            "complete_requires": "algorithm_lifecycle_status=complete plus final_regression locked_release",
            "stage_gate": "trial promotion alone is not completion; formal gate state must be recorded",
            "paper": "paper/report claims must be derived from locked final regression artifacts",
        },
    }
