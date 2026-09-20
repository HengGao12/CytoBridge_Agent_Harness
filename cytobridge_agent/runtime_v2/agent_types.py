from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


SPAWN_SUBAGENT_TOOL_NAME = "spawn_subagent"
SUBMIT_SUBAGENT_RESULT_TOOL_NAME = "submit_subagent_result"
SUBMIT_PROPOSAL_REVIEW_TOOL_NAME = "submit_proposal_review"
SUBMIT_RESEARCH_IDEA_REVIEW_TOOL_NAME = "submit_research_idea_review"
SUBMIT_IMPLEMENTATION_REVIEW_TOOL_NAME = "submit_implementation_review"

DEFAULT_GENERAL_DISALLOWED_TOOLS = {
    SPAWN_SUBAGENT_TOOL_NAME,
    "commit_workflow_state",
    "set_plan_from_text",
    "update_plan",
    "create_algorithm_proposal",
    "review_algorithm_proposal",
    "init_training_algorithm_workspace",
    "record_decision",
    "mark_algorithm_failed",
    "mark_result_obsolete",
    "snapshot_active_algorithm_workspace",
    "rollback_algorithm_workspace",
    "set_active_baseline_run",
    "record_algorithm_benchmark_baseline",
    "register_algorithm_benchmark_dataset",
    "start_algorithm_campaign",
    "get_algorithm_campaign_status",
    "start_campaign_trial",
    "run_campaign_trial",
    "resume_rejected_trial",
    "list_campaign_trials",
    "check_campaign_stage_gate",
    "create_workspace_file",
    "create_algorithm_workspace_artifact",
    "replace_workspace_file",
    "apply_workspace_patch",
    "preview_workspace_diff",
    "run_training",
    "persist_runtime_adata",
}


@dataclass(frozen=True)
class SubagentTypeSpec:
    name: str
    description: str
    prompt_name: str
    allowed_tools: List[str]
    disallowed_tools: List[str]

    def build_tool_policy(self) -> Dict[str, Any]:
        return {
            "subagent_type": self.name,
            "allowed_tools": list(self.allowed_tools),
            "disallowed_tools": list(self.disallowed_tools),
        }


_SUBAGENT_TYPES: Dict[str, SubagentTypeSpec] = {
    "general": SubagentTypeSpec(
        name="general",
        description="General bounded worker for delegated side tasks.",
        prompt_name="subagent_worker",
        allowed_tools=[],
        disallowed_tools=sorted(DEFAULT_GENERAL_DISALLOWED_TOOLS),
    ),
    "proposal_evaluator": SubagentTypeSpec(
        name="proposal_evaluator",
        description="Strict mathematical reviewer for custom algorithm proposals.",
        prompt_name="subagent_proposal_evaluator",
        allowed_tools=[
            "read_file",
            "find_files",
            "grep_files",
            "list_path",
            "list_skills",
            "get_current_workflow_context",
            "get_algorithm_proposal_template",
            "get_algorithm_proposal_status",
            SUBMIT_PROPOSAL_REVIEW_TOOL_NAME,
        ],
        disallowed_tools=[],
    ),
    "idea_evaluator": SubagentTypeSpec(
        name="idea_evaluator",
        description="Strict research-problem reviewer for persistent research ideas.",
        prompt_name="subagent_idea_evaluator",
        allowed_tools=[
            "read_file",
            "find_files",
            "grep_files",
            "list_path",
            "get_research_idea_status",
            "list_research_ideas",
            "list_skills",
            "get_current_workflow_context",
            "get_algorithm_proposal_status",
            "list_experiment_history",
            SUBMIT_RESEARCH_IDEA_REVIEW_TOOL_NAME,
        ],
        disallowed_tools=[],
    ),
    "inference_evaluator": SubagentTypeSpec(
        name="inference_evaluator",
        description="Strict read-only reviewer for custom inference/evaluation code before trusted training.",
        prompt_name="subagent_inference_evaluator",
        allowed_tools=[
            "read_file",
            "find_files",
            "grep_files",
            "list_path",
            "list_skills",
            "inspect_h5ad_contract",
            "get_algorithm_benchmark_dataset",
            "get_current_workflow_context",
            "get_algorithm_proposal_status",
            "list_experiment_history",
            SUBMIT_SUBAGENT_RESULT_TOOL_NAME,
        ],
        disallowed_tools=[],
    ),
    "implementation_evaluator": SubagentTypeSpec(
        name="implementation_evaluator",
        description="Strict read-only reviewer for proposal-to-implementation semantic alignment before trusted training.",
        prompt_name="subagent_implementation_evaluator",
        allowed_tools=[
            "read_file",
            "find_files",
            "grep_files",
            "list_path",
            "list_skills",
            "get_current_workflow_context",
            "get_algorithm_proposal_status",
            "list_experiment_history",
            SUBMIT_IMPLEMENTATION_REVIEW_TOOL_NAME,
        ],
        disallowed_tools=[],
    ),
    "paper_reviewer": SubagentTypeSpec(
        name="paper_reviewer",
        description="Strict read-only reviewer for paper drafts, citation provenance, and manuscript completion gates.",
        prompt_name="subagent_paper_reviewer",
        allowed_tools=[
            "read_file",
            "find_files",
            "grep_files",
            "list_path",
            "list_skills",
            "get_current_workflow_context",
            "query_campaign_baseline_metrics",
            SUBMIT_SUBAGENT_RESULT_TOOL_NAME,
        ],
        disallowed_tools=[],
    ),
}


def resolve_subagent_type(name: str) -> SubagentTypeSpec:
    key = str(name or "general").strip().lower() or "general"
    return _SUBAGENT_TYPES.get(key, _SUBAGENT_TYPES["general"])


def default_subagent_tool_policy(subagent_type: str = "general") -> Dict[str, Any]:
    return resolve_subagent_type(subagent_type).build_tool_policy()


def available_subagent_type_names() -> List[str]:
    return sorted(_SUBAGENT_TYPES.keys())
