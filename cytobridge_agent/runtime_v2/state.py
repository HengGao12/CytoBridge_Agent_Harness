from __future__ import annotations

from typing import Any, Dict, Literal, TypedDict


DEFAULT_CONTEXT_POLICY: Dict[str, Any] = {
    "enabled": True,
    "context_window": 200000,
    "context_window_cap": 256000,
    "effective_context_window": 200000,
    "allow_large_context_window": False,
    "trigger_ratio": 0.82,
    "keep_last_turns": 6,
    "max_tool_chars": 1500,
    "microcompact_enabled": True,
    "llm_compact_enabled": True,
    "llm_compact_input_max_chars": 24000,
}

DEFAULT_RUNTIME_RECURSION_LIMIT = 10000
DEFAULT_STOP_HOOK_ENABLED = False
DEFAULT_STOP_HOOK_MODE = "prompt"
DEFAULT_STOP_HOOK_MAX_TRIGGERS = 20
DEFAULT_STOP_HOOK_PROMPT = (
    "Before allowing the planner to finish, check whether an automated custom "
    "algorithm-design task has completed a meaningful algorithm lifecycle: "
    "the proposal was reviewed or intentionally skipped by policy, the "
    "workspace implementation matches the approved proposal, required "
    "implementation/inference reviews passed, campaign evidence was collected "
    "through the configured gates, Stage 1 feasibility passed, Stage 2 claim "
    "validation passed, Stage 3 tuning/generalization was completed or "
    "intentionally advanced according to policy, final_regression produced "
    "a final result / locked release, and any required paper-level deliverable "
    "was produced and approved by a fresh `paper_reviewer` review with no "
    "blocking issues. The structured snapshot may include "
    "`algorithm_lifecycle_status`; for an algorithm-lifecycle task, pass only "
    "when that status is `complete`, but do not rely on that flag alone when "
    "the visible transcript or artifacts indicate an invalid completion. Return "
    "block if the final release appears to use diagnostic degeneration/ablation "
    "settings instead of the approved-proposal mechanism, if a claim/custom "
    "metric is mislabeled relative to what it actually computes, or if the "
    "reported evidence is not the locked final-regression result, if the "
    "paper package is missing when the task requires one, or if the current "
    "paper has no fresh `paper_reviewer` approval with a file-hash manifest. "
    "If it is "
    "`developing`, return block and "
    "make the planner continue the campaign lifecycle. If it is `failed`, "
    "return block and make the planner revise/fix the algorithm or design a "
    "new algorithm that satisfies the user goal. Passing Stage 1 alone, or "
    "having only a promoted trial without a formal stage gate, is not "
    "completion. This stop hook is an autonomy guard: do not ask the user for "
    "clarification or confirmation, and do not allow the planner to stop just "
    "because it wants user input. If the current user request is not an "
    "algorithm-lifecycle task, pass only when that concrete request is "
    "complete. Return block whenever the planner is trying to stop, ask the "
    "user, or wait for confirmation before the relevant lifecycle or requested "
    "task is actually complete. A terminal response that only says the planner "
    "will inspect, edit, run, test, train, evaluate, or otherwise do work next "
    "is not an acceptable final answer; return block so the planner continues "
    "executing."
)

LEGACY_DEFAULT_CONTEXT_POLICY: Dict[str, Any] = {
    **DEFAULT_CONTEXT_POLICY,
    "context_window": 128000,
}

_MIN_CONTEXT_WINDOW = 1024
_MIN_TRIGGER_RATIO = 0.2
_MAX_TRIGGER_RATIO = 0.99
_DEFAULT_CONTEXT_WINDOW_CAP = 256000


WorkflowPhase = Literal[
    "intake",
    "preprocessing",
    "theory_selection",
    "training",
    "downstream_analysis",
    "reporting",
    "completed",
]


class SingleAgentState(TypedDict, total=False):
    messages: list[Any]
    workflow_phase: WorkflowPhase
    task_profile: Dict[str, Any]
    phase_status: Dict[str, str]
    artifact_index: Dict[str, Any]
    workflow_audit_log: list[dict]


DEFAULT_TASK_PROFILE: Dict[str, Any] = {
    "current_stage": "intake",
    "primary_goal": "analysis",
    "facets": {
        "reproduction": False,
        "new_algorithm": False,
        "tuning": False,
        "integration": False,
        "analysis": True,
        "idea_management": False,
    },
    "change_axes": {
        "data_contract": False,
        "solver": False,
        "coupling": False,
        "path": False,
        "mass": False,
        "loss": False,
        "evaluation": False,
        "downstream": False,
    },
    "risk_flags": {
        "scalability_sensitive": False,
        "semantics_sensitive": False,
        "requires_baseline": False,
        "requires_literature": False,
    },
    "notes": "",
}


def _normalize_context_policy(policy: Any) -> Dict[str, Any]:
    if not isinstance(policy, dict):
        return dict(DEFAULT_CONTEXT_POLICY)

    candidate = dict(policy)
    if candidate == LEGACY_DEFAULT_CONTEXT_POLICY:
        return dict(DEFAULT_CONTEXT_POLICY)

    normalized = dict(DEFAULT_CONTEXT_POLICY)
    normalized.update(candidate)

    try:
        normalized["context_window"] = int(normalized.get("context_window", DEFAULT_CONTEXT_POLICY["context_window"]))
    except Exception:
        normalized["context_window"] = int(DEFAULT_CONTEXT_POLICY["context_window"])
    if normalized["context_window"] < _MIN_CONTEXT_WINDOW:
        normalized["context_window"] = int(DEFAULT_CONTEXT_POLICY["context_window"])

    try:
        normalized["context_window_cap"] = int(
            normalized.get("context_window_cap", DEFAULT_CONTEXT_POLICY["context_window_cap"])
        )
    except Exception:
        normalized["context_window_cap"] = int(DEFAULT_CONTEXT_POLICY["context_window_cap"])
    if normalized["context_window_cap"] < _MIN_CONTEXT_WINDOW:
        normalized["context_window_cap"] = _DEFAULT_CONTEXT_WINDOW_CAP

    normalized["allow_large_context_window"] = bool(
        normalized.get("allow_large_context_window", DEFAULT_CONTEXT_POLICY["allow_large_context_window"])
    )
    if normalized["allow_large_context_window"]:
        normalized["effective_context_window"] = int(normalized["context_window"])
    else:
        normalized["effective_context_window"] = int(
            min(int(normalized["context_window"]), int(normalized["context_window_cap"]))
        )

    try:
        normalized["trigger_ratio"] = float(normalized.get("trigger_ratio", DEFAULT_CONTEXT_POLICY["trigger_ratio"]))
    except Exception:
        normalized["trigger_ratio"] = float(DEFAULT_CONTEXT_POLICY["trigger_ratio"])
    if not (_MIN_TRIGGER_RATIO <= normalized["trigger_ratio"] < _MAX_TRIGGER_RATIO):
        normalized["trigger_ratio"] = float(DEFAULT_CONTEXT_POLICY["trigger_ratio"])

    try:
        normalized["keep_last_turns"] = max(1, int(normalized.get("keep_last_turns", DEFAULT_CONTEXT_POLICY["keep_last_turns"])))
    except Exception:
        normalized["keep_last_turns"] = int(DEFAULT_CONTEXT_POLICY["keep_last_turns"])

    try:
        normalized["max_tool_chars"] = max(128, int(normalized.get("max_tool_chars", DEFAULT_CONTEXT_POLICY["max_tool_chars"])))
    except Exception:
        normalized["max_tool_chars"] = int(DEFAULT_CONTEXT_POLICY["max_tool_chars"])

    try:
        normalized["llm_compact_input_max_chars"] = max(
            2000,
            int(normalized.get("llm_compact_input_max_chars", DEFAULT_CONTEXT_POLICY["llm_compact_input_max_chars"])),
        )
    except Exception:
        normalized["llm_compact_input_max_chars"] = int(DEFAULT_CONTEXT_POLICY["llm_compact_input_max_chars"])

    normalized["enabled"] = bool(normalized.get("enabled", DEFAULT_CONTEXT_POLICY["enabled"]))
    normalized["microcompact_enabled"] = bool(normalized.get("microcompact_enabled", DEFAULT_CONTEXT_POLICY["microcompact_enabled"]))
    normalized["llm_compact_enabled"] = bool(normalized.get("llm_compact_enabled", DEFAULT_CONTEXT_POLICY["llm_compact_enabled"]))
    return normalized


def _normalize_task_profile(profile: Any) -> Dict[str, Any]:
    normalized = {
        "current_stage": DEFAULT_TASK_PROFILE["current_stage"],
        "primary_goal": DEFAULT_TASK_PROFILE["primary_goal"],
        "facets": dict(DEFAULT_TASK_PROFILE["facets"]),
        "change_axes": dict(DEFAULT_TASK_PROFILE["change_axes"]),
        "risk_flags": dict(DEFAULT_TASK_PROFILE["risk_flags"]),
        "notes": DEFAULT_TASK_PROFILE["notes"],
    }
    if not isinstance(profile, dict):
        return normalized

    current_stage = str(profile.get("current_stage") or "").strip()
    if current_stage:
        normalized["current_stage"] = current_stage
    primary_goal = str(profile.get("primary_goal") or "").strip()
    if primary_goal:
        normalized["primary_goal"] = primary_goal

    facets = profile.get("facets")
    if isinstance(facets, dict):
        for key in normalized["facets"]:
            if key in facets:
                normalized["facets"][key] = bool(facets[key])

    change_axes = profile.get("change_axes")
    if isinstance(change_axes, dict):
        for key in normalized["change_axes"]:
            if key in change_axes:
                normalized["change_axes"][key] = bool(change_axes[key])

    risk_flags = profile.get("risk_flags")
    if isinstance(risk_flags, dict):
        for key in normalized["risk_flags"]:
            if key in risk_flags:
                normalized["risk_flags"][key] = bool(risk_flags[key])

    normalized["notes"] = str(profile.get("notes") or normalized["notes"])
    return normalized


def ensure_runtime_v2_state(state: Dict[str, Any]) -> Dict[str, Any]:
    state.setdefault("workflow_phase", "intake")
    state["task_profile"] = _normalize_task_profile(state.get("task_profile"))
    state.setdefault(
        "phase_status",
        {
            "preprocessing": "pending",
            "theory_selection": "pending",
            "training": "pending",
            "downstream_analysis": "pending",
            "reporting": "pending",
        },
    )
    state.setdefault("artifact_index", {})
    state.setdefault("workflow_audit_log", [])
    state.setdefault("messages", [])
    state.setdefault("training_runs", [])
    state.setdefault("downstream_results", [])
    state.setdefault("downstream_figures", [])
    state.setdefault("plan_decision", {})
    state.setdefault("candidates", [])
    state.setdefault(
        "hidden_skills",
        {
            "workflow": [],
            "planner": [],
            "downstream": [],
        },
    )
    state.setdefault("hidden_training_algorithms", [])
    state.setdefault("algorithm_proposal_review_mode", "agent_decide")
    state.setdefault("idea_review_mode", "agent_decide")
    state.setdefault("algorithm_proposals", {})
    state.setdefault("latest_algorithm_proposal_id", "")
    state.setdefault("research_ideas", {})
    state.setdefault("latest_research_idea_id", "")
    state.setdefault("active_research_idea_id", "")
    state.setdefault("active_idea_registry", {})
    state.setdefault("decision_log_summary", [])
    state.setdefault("obsolete_results_summary", [])
    state.setdefault("active_experiment_registry", {})
    state.setdefault(
        "active_algorithm_context",
        {
            "algorithm_id": "",
            "workspace_path": "",
            "proposal_id": "",
            "proposal_status": "",
            "primary_idea_id": "",
            "registry_path": "",
            "active_snapshot_id": "",
            "dirty_since_snapshot": False,
            "dirty_paths": [],
            "last_mutation_at": "",
            "last_verified_at": "",
        },
    )
    state.setdefault("active_proposal_id", "")
    state.setdefault("active_workspace_snapshot_id", "")
    state.setdefault("active_baseline_run_id", "")
    state.setdefault("active_algorithm_campaign_id", "")
    state.setdefault("active_algorithm_campaign", {})
    state.setdefault("runtime_action", {})
    state.setdefault("stop_hook_enabled", DEFAULT_STOP_HOOK_ENABLED)
    state.setdefault("stop_hook_mode", DEFAULT_STOP_HOOK_MODE)
    state.setdefault("stop_hook_prompt", DEFAULT_STOP_HOOK_PROMPT)
    state.setdefault("stop_hook_max_triggers", DEFAULT_STOP_HOOK_MAX_TRIGGERS)
    state.setdefault("subagent_registry", {})
    state.setdefault("subagent_counter", 0)
    state.setdefault("subagent_run_log", [])
    state["context_policy"] = _normalize_context_policy(state.get("context_policy"))
    return state
