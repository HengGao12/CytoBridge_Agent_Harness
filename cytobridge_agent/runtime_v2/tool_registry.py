from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import traceback
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Literal, Tuple

from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, field_validator

from ..display import DisplayManager
from ..schemas import AgentState
from ..tools.adata_manager import AnnDataManager
from ..tools.file_tools import ReadResult
from ..tools.guarded_terminal import (
    TERMINAL_TOOL_NAME,
    SUPPORTED_COMMAND_SUMMARY,
    execute_guarded_terminal_command,
)
from ..tools.web_fetch import fetch_web_content
from ..tools.web_search import search_web_free
from ..tools.bohrium_paper_search import search_bohrium_papers
from ..tools.planner_tools import PlannerTools
from ..utils.codex_search import codex_native_web_search
from .agent_types import (
    SPAWN_SUBAGENT_TOOL_NAME,
    SUBMIT_IMPLEMENTATION_REVIEW_TOOL_NAME,
    SUBMIT_PROPOSAL_REVIEW_TOOL_NAME,
    SUBMIT_RESEARCH_IDEA_REVIEW_TOOL_NAME,
    SUBMIT_SUBAGENT_RESULT_TOOL_NAME,
    available_subagent_type_names,
    default_subagent_tool_policy,
    resolve_subagent_type,
)
from .commit import WorkflowCommitter
from .subagent_manager import SubagentRuntimeManager, is_retryable_subagent_infrastructure_error
from .tool_isolation import IsolatedToolResult, run_tool_in_subprocess


DEFAULT_TOOL_TIMEOUT_SEC = 300
TRAINING_TOOL_TIMEOUT_SEC = 7200
PREVIEW_TRAINING_TOOL_TIMEOUT_SEC = int(os.getenv("CYTOBRIDGE_PREVIEW_TRAINING_TOOL_TIMEOUT_SEC", "600"))
LONG_IO_TOOL_TIMEOUT_SEC = 1800

_STATE_MUTATING_TOOL_NAMES = {
    "load_or_switch_adata",
    "persist_runtime_adata",
    "create_algorithm_proposal",
    "revise_algorithm_proposal",
    "create_research_idea",
    "revise_research_idea",
    "review_research_idea",
    "set_active_research_idea",
    "update_research_idea_progress",
    "link_algorithm_to_idea",
    "init_training_algorithm_workspace",
    "record_decision",
    "mark_algorithm_failed",
    "mark_result_obsolete",
    "activate_algorithm_workspace",
    "snapshot_active_algorithm_workspace",
    "rollback_algorithm_workspace",
    "set_active_baseline_run",
    "record_algorithm_benchmark_baseline",
    "register_algorithm_benchmark_dataset",
    "register_stage2_simulation_dataset",
    "start_algorithm_campaign",
    "update_campaign_claim_metric_spec",
    "set_campaign_stage_panel",
    "switch_campaign_stage_panel",
    "refresh_campaign_stage_baselines",
    "update_campaign_control_baseline",
    "run_campaign_control_baseline",
    "run_campaign_locked_algorithm_baseline",
    "compute_campaign_claim_metric_for_baselines",
    "start_campaign_trial",
    "run_campaign_trial",
    "abort_current_campaign_trial",
    "resume_rejected_trial",
    "check_campaign_stage_gate",
    "create_workspace_file",
    "create_algorithm_workspace_artifact",
    "replace_workspace_file",
    "patch_algorithm_config",
    "apply_workspace_patch",
    "set_plan_from_text",
    "update_plan",
    "set_task_profile",
    "check_workflow_gate",
    "preview_training_run",
    "run_training",
    "commit_workflow_state",
    SPAWN_SUBAGENT_TOOL_NAME,
    SUBMIT_PROPOSAL_REVIEW_TOOL_NAME,
    SUBMIT_RESEARCH_IDEA_REVIEW_TOOL_NAME,
    SUBMIT_IMPLEMENTATION_REVIEW_TOOL_NAME,
    SUBMIT_SUBAGENT_RESULT_TOOL_NAME,
}

_ADATA_READING_TOOL_NAMES = {
    "inspect_adata_state",
    "load_or_switch_adata",
    "persist_runtime_adata",
    "inspect_h5ad_contract",
    "materialize_paper_dataset",
    "execute_python",
    "run_saved_python_script",
    "preview_training_run",
    "run_training",
    "run_campaign_trial",
    "run_campaign_control_baseline",
    "run_campaign_locked_algorithm_baseline",
    "refresh_campaign_stage_baselines",
    "compute_campaign_claim_metric_for_baselines",
}

_ARTIFACT_WRITING_TOOL_NAMES = {
    "materialize_paper_dataset",
    "execute_python",
    "run_saved_python_script",
    "persist_runtime_adata",
    "init_training_algorithm_workspace",
    "snapshot_active_algorithm_workspace",
    "rollback_algorithm_workspace",
    "record_algorithm_benchmark_baseline",
    "register_algorithm_benchmark_dataset",
    "register_stage2_simulation_dataset",
    "start_algorithm_campaign",
    "run_campaign_trial",
    "run_campaign_control_baseline",
    "run_campaign_locked_algorithm_baseline",
    "abort_current_campaign_trial",
    "update_campaign_control_baseline",
    "create_workspace_file",
    "create_algorithm_workspace_artifact",
    "replace_workspace_file",
    "patch_algorithm_config",
    "apply_workspace_patch",
    "preview_training_run",
    "run_training",
    "commit_workflow_state",
}

_GPU_TOOL_NAMES = {
    "preview_training_run",
    "run_training",
    "run_campaign_trial",
    "run_campaign_control_baseline",
    "run_campaign_locked_algorithm_baseline",
}

_MANAGED_SUBPROCESS_TOOL_NAMES = {
    "execute_python",
    "run_saved_python_script",
    "preview_training_run",
    "run_training",
    "run_campaign_trial",
    "run_campaign_control_baseline",
    "run_campaign_locked_algorithm_baseline",
    "refresh_campaign_stage_baselines",
    "compute_campaign_claim_metric_for_baselines",
}


class _PlanItemInput(BaseModel):
    step: str = Field(description="Plan step text.")
    status: Literal["pending", "in_progress", "completed"] = Field(
        description="Plan status: pending | in_progress | completed."
    )


def _json_string_to_object(value: Any, field_name: str) -> Any:
    if value is None or not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{field_name} must be a JSON object, not a raw string.") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{field_name} must be a JSON object.")
    return parsed


def _json_string_to_list(value: Any, field_name: str) -> Any:
    if value is None or not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{field_name} must be a JSON array, not a raw string.") from exc
    if not isinstance(parsed, list):
        raise ValueError(f"{field_name} must be a JSON array.")
    return parsed


class _SetPlanFromTextInput(BaseModel):
    plan_text: str = Field(description="Full free-text plan to parse into structured items.")
    explanation: str = Field(default="", description="Optional concise explanation.")


class _UpdatePlanInput(BaseModel):
    plan: List[_PlanItemInput] = Field(
        description="FULL plan list. Every item must include both step and status."
    )
    explanation: str = Field(default="", description="Optional concise explanation.")


class _CreateAlgorithmProposalInput(BaseModel):
    algorithm_id: str = Field(description="Custom algorithm id (directory name) under ~/.cellcompass/training_algorithms.")
    abstract: str = Field(description="Short abstract summarizing the target problem, core algorithm idea, and expected validation evidence.")
    literature_and_package_grounding: str = Field(
        default="",
        description=(
            "Literature/package grounding for the proposal. For genuinely new algorithms, cite at least 10 directly "
            "relevant papers/notes and explain why each matters. `literature` is accepted as an alias when preferred."
        ),
    )
    literature: str = Field(
        default="",
        description="Alias/additional text for literature_and_package_grounding. Use this if the caller naturally names the field `literature`.",
    )
    references: str = Field(
        default="",
        description="Optional bibliography/reference list for the proposal. Include paper titles, local literature-note paths, PDF paths, or stable citations when available.",
    )
    problem_statement: str = Field(description="Specific problem or method gap the proposal claims to solve.")
    problem_mathematical_form: str = Field(
        default="",
        description="Optional concrete mathematical problem/objective/dynamic formulation. Leave empty only when the method has no clean global form, but still define the problem in prose.",
    )
    mathematical_derivation_to_algorithm_design: str = Field(
        default="",
        description=(
            "Optional derivation from the proposed mathematical/dynamic problem to the concrete algorithm design. "
            "For new mathematical algorithms, make this self-contained: define variables, derive the computable representation, "
            "derive supervision targets/losses, inference rule, distribution recovery, mass recovery when applicable, "
            "and all proposal-stage assumptions/approximations."
        ),
    )
    novelty_and_contributions: str = Field(
        default="",
        description=(
            "Optional novelty/contribution statement. For genuinely new algorithms, state concrete novelty relative to builtin CytoBridge methods "
            "and directly relevant literature; if there is no algorithmic novelty claim, explain why it is not applicable."
        ),
    )
    claimed_capability: str = Field(description="What the algorithm claims it can solve/improve, and what evidence would validate that claim.")
    objective: str = Field(description="What scientific/modeling gap the algorithm addresses.")
    theoretical_core: str = Field(description="Core theoretical rationale (conceptual, non-code).")
    mathematical_abstraction: str = Field(description="Mathematical abstraction of the method (non-implementation detail).")
    algorithm_semantics_table: str = Field(
        description=(
            "Markdown table describing the algorithm's semantic objects. Include columns such as "
            "`Mathematical Object`, `Runtime Layer`, and `Semantic Definition`."
        )
    )
    implementation_pseudocode: str = Field(
        description=(
            "Implementation-oriented pseudocode, preferably with global Inputs/Outputs, labeled steps P1/P2/..., "
            "per-step Inputs/Outputs/Invariants/Forbidden deviations, and key equations or update rules. "
            "Formatting issues may produce warnings, but real implementation code is not allowed."
        )
    )
    unbalanced_decision: str = Field(
        description=(
            "Explicit decision on whether unbalanced mass is enabled. Must include rationale for either choice. "
            "If enabled, the proposal must later justify why weighted-particle inference can recover observed "
            "cell-count / total-mass changes."
        )
    )
    stochasticity_decision: str = Field(
        description="Explicit decision on whether stochasticity/noise is enabled. Must include rationale for either choice."
    )
    evaluation_plan: str = Field(
        description="Evaluation plan that preserves builtin W1/TMV and defines any additive custom metrics needed to validate the proposal."
    )
    expected_evaluation_outcome: str = Field(
        description=(
            "Expected evaluation outcome: specify the data/simulation scenario where the algorithm should work, "
            "the expected result pattern beyond builtin W1/TMV, and why that pattern validates the claimed capability."
        )
    )
    inductive_generalization_argument: str = Field(
        description=(
            "Argument that the runtime dynamics apply to new valid t=0 cells/particles under the stated data contract. "
            "State the learned rule, allowed inference-time inputs/context, training-only supervision signals, and why "
            "the method does not rely on training-cell ids, memorized OT rows, barcode-specific hardcoding, future snapshots, "
            "or target-specific lookup/correction."
        )
    )
    mass_modeling_scope: Literal["balanced_only", "models_unbalanced_mass"] = Field(
        description=(
            "Machine-readable proposal-stage algorithm attribute. Use `balanced_only` when the algorithm deliberately does not model "
            "cell-count/total-mass changes; use `models_unbalanced_mass` when TMV should be a hard campaign/training gate."
        )
    )
    distribution_recovery_argument: str = Field(
        description=(
            "Argument for why inference from t_k to t_{k+1} can recover the observed next-time distribution. "
            "If unbalanced mass is enabled, this argument must also explain why the final weighted-particle "
            "measure can recover the observed cell-count / total-mass change."
        )
    )
    overengineering_self_check: str = Field(
        description="Self-check proving the design is not unnecessarily complex and keeps a simple effective core."
    )
    uncertainty_and_risks: str = Field(default="", description="Known assumptions/risks/limits.")
    primary_idea_id: str = Field(
        default="",
        description="Optional research idea id that this algorithm proposal primarily addresses.",
    )
    requires_user_review: bool = Field(
        default=False,
        description="In agent_decide mode, set true when uncertainty is high and user review should be mandatory.",
    )


class _ReviewAlgorithmProposalInput(BaseModel):
    algorithm_id: str = Field(description="Algorithm proposal id to review.")
    proposal_id: str = Field(
        default="",
        description="Exact proposal version id to review. Omit only when reviewing the current active proposal.",
    )
    decision: Literal["approve", "reject", "revise"] = Field(description="User review decision.")
    reviewer_feedback: str = Field(default="", description="Optional concise user feedback.")


class _ReviseAlgorithmProposalInput(BaseModel):
    algorithm_id: str = Field(description="Algorithm id whose current active proposal should be revised.")
    revision_note: str = Field(
        description="Concise explanation of why this revision is needed. This is stored in proposal history."
    )
    proposal_markdown: str = Field(
        default="",
        description=(
            "Full revised proposal markdown. Use this for substantial rewrites or when patch context is stale. "
            "Do not include implementation code."
        ),
    )
    proposal_patch: str = Field(
        default="",
        description=(
            "Optional Codex-style or unified diff patch against the current editable PROPOSAL.md. "
            "Use this for small section edits. Provide either proposal_markdown or proposal_patch, not both."
        ),
    )
    requires_user_review: bool = Field(
        default=False,
        description="Request mandatory user review for this revision when uncertainty is high.",
    )


class _GetAlgorithmProposalStatusInput(BaseModel):
    algorithm_id: str = Field(default="", description="Optional algorithm id. Empty means list all proposals.")


class _CreateResearchIdeaInput(BaseModel):
    title: str = Field(description="Human-readable research idea title.")
    primary_track: Literal["biology-journal", "ml-topconf"] = Field(
        description="Primary paper-positioning track for this idea."
    )
    alternate_track: str = Field(default="", description="Optional alternate track: biology-journal or ml-topconf.")
    problem_definition: str = Field(description="Concrete scientific problem statement.")
    scientific_object: str = Field(description="Exact object of inference or explanation.")
    current_method_failure_mode: str = Field(description="Specific way current methods fail or remain ambiguous.")
    prior_work: str = Field(description="What prior methods already achieved, what they still miss, and where the remaining improvement space is.")
    why_this_matters: str = Field(description="Why answering this question matters scientifically.")
    why_now: str = Field(description="Why the question is timely or newly answerable now.")
    falsifiable_success_criteria: str = Field(description="Concrete, falsifiable success criteria.")
    non_goals: str = Field(description="Claims the idea should explicitly avoid.")
    evidence_basis: str = Field(description="Traceable evidence supporting the idea.")
    feasible_direction_families: str = Field(
        description="1-5 bounded direction families, not full algorithm implementations."
    )
    feasibility_constraints: str = Field(description="Key feasibility constraints or impossible conditions.")
    idea_id: str = Field(default="", description="Optional explicit idea id. If omitted, one is derived from the title.")
    requires_user_review: bool = Field(default=False, description="Force user review in agent_decide mode.")


class _ReviseResearchIdeaInput(BaseModel):
    idea_id: str = Field(description="Existing research idea id.")
    revision_note: str = Field(description="Why the idea is being revised.")
    title: str = Field(default="", description="Optional replacement title.")
    primary_track: str = Field(default="", description="Optional replacement primary track.")
    alternate_track: str = Field(default="", description="Optional replacement alternate track.")
    problem_definition: str = Field(default="", description="Optional replacement problem statement.")
    scientific_object: str = Field(default="", description="Optional replacement scientific object.")
    current_method_failure_mode: str = Field(default="", description="Optional replacement failure mode.")
    prior_work: str = Field(default="", description="Optional replacement prior-work summary and remaining headroom.")
    why_this_matters: str = Field(default="", description="Optional replacement motivation.")
    why_now: str = Field(default="", description="Optional replacement why-now statement.")
    falsifiable_success_criteria: str = Field(default="", description="Optional replacement success criteria.")
    non_goals: str = Field(default="", description="Optional replacement non-goals.")
    evidence_basis: str = Field(default="", description="Optional replacement evidence basis.")
    feasible_direction_families: str = Field(default="", description="Optional replacement 1-5 direction families.")
    feasibility_constraints: str = Field(default="", description="Optional replacement feasibility constraints.")
    requires_user_review: bool = Field(default=False, description="Force user review in agent_decide mode.")


class _ReviewResearchIdeaInput(BaseModel):
    idea_id: str = Field(description="Research idea id to review.")
    decision: Literal["approve", "reject", "revise"] = Field(description="User review decision.")
    reviewer_feedback: str = Field(default="", description="Optional concise user feedback.")


class _GetResearchIdeaStatusInput(BaseModel):
    idea_id: str = Field(default="", description="Optional idea id. Empty means list all ideas.")


class _ListResearchIdeasInput(BaseModel):
    track: str = Field(default="", description="Optional primary track filter.")
    include_inactive: bool = Field(default=True, description="Whether to include inactive ideas.")


class _SetActiveResearchIdeaInput(BaseModel):
    idea_id: str = Field(description="Research idea id to mark active.")


class _UpdateResearchIdeaProgressInput(BaseModel):
    idea_id: str = Field(description="Research idea id to update.")
    progress_summary: str = Field(description="Concrete progress summary tied to evidence or attempts.")
    resolution_status: str = Field(default="", description="Optional new resolution status.")
    execution_status: str = Field(default="", description="Optional new execution status.")
    portfolio_status: str = Field(default="", description="Optional new portfolio status.")
    linked_algorithm_id: str = Field(default="", description="Optional linked algorithm id.")
    proposal_id: str = Field(default="", description="Optional proposal id tied to the progress update.")
    run_id: str = Field(default="", description="Optional run id tied to the progress update.")
    evidence_refs: List[str] = Field(default_factory=list, description="Optional evidence refs supporting the update.")
    next_step: str = Field(default="", description="Optional next-step note.")


class _LinkAlgorithmToIdeaInput(BaseModel):
    idea_id: str = Field(description="Research idea id.")
    algorithm_id: str = Field(description="Algorithm id to link.")
    proposal_id: str = Field(default="", description="Optional proposal id for the link.")
    summary: str = Field(default="", description="Optional concise summary of why the link exists.")


class _SetTaskProfileInput(BaseModel):
    current_stage: Literal["intake", "proposal", "authoring", "review", "training", "downstream", "report"] = Field(
        description="Current algorithm/workflow stage."
    )
    primary_goal: Literal["tuning", "reproduction", "new_algorithm", "integration", "analysis", "idea_development"] = Field(
        default="analysis",
        description="Primary goal driving the current work."
    )
    facets: Dict[str, bool] = Field(
        default_factory=dict,
        description="Optional additional task facets, for example {'reproduction': true, 'integration': true}.",
    )
    change_axes: Dict[str, bool] = Field(
        default_factory=dict,
        description=(
            "Which layers are being changed. Supported keys: "
            "data_contract, solver, coupling, path, mass, loss, evaluation, downstream."
        ),
    )
    notes: str = Field(default="", description="Optional concise notes about scope or constraints.")


class _CheckWorkflowGateInput(BaseModel):
    stage: str = Field(
        default="",
        description=(
            "Optional stage override for gate checking. "
            "Defaults to task_profile.current_stage. "
            "Valid values: intake, proposal, authoring, review, training, downstream, report."
        ),
    )


class _PreviewTrainingRunInput(BaseModel):
    candidate_name: str = Field(default="", description="Builtin config/family name. Mutually exclusive with training_algorithm_id.")
    training_algorithm_id: str = Field(default="", description="Custom training algorithm id. Mutually exclusive with candidate_name.")
    stage: Literal["pilot", "final"] = Field(default="final", description="Preview context label: pilot or final.")
    config_overrides: Dict[str, Any] = Field(default_factory=dict, description="Optional config overrides merged on top of the target config.")
    adata_path: str = Field(default="", description="Optional .h5ad path override for preview.")
    device: str = Field(default="", description="Optional device override such as cuda or cpu.")
    run_smoke_test: bool = Field(
        default=True,
        description="Run a 1-epoch training/inference/evaluation smoke test to catch broken metric or inference code.",
    )


class _ReadFileInput(BaseModel):
    file_path: str = Field(description="Absolute path or current-workdir-relative path to the file.")
    offset: int = Field(default=1, description="1-indexed starting line for text-like files.")
    limit: Optional[int] = Field(default=None, description="Maximum number of text lines to return for text-like files.")
    pages: Optional[str] = Field(
        default=None,
        description="Optional PDF page selection like '1-3,5'. Required for PDFs longer than 10 pages.",
    )
    pdf_mode: Literal["render", "text"] = Field(
        default="render",
        description="PDF read mode: `render` attaches rendered page images; `text` extracts textual content from selected pages.",
    )


class _InspectH5adContractInput(BaseModel):
    file_path: str = Field(description="Absolute path to a .h5ad file to inspect read-only.")
    time_key: str = Field(default="", description="Optional obs column to treat as time. Defaults to common time keys when present.")
    label_keys: List[str] = Field(
        default_factory=list,
        description="Optional obs columns to summarize as labels, lineage, fate, branch, condition, or grouping keys.",
    )
    max_categories: int = Field(default=20, description="Maximum categories/count rows to include per column.")


class _InspectPaperSourceInput(BaseModel):
    input_source: str = Field(
        default="",
        description="Optional paper PDF path, manuscript-derived text path, or article URL. Defaults to current input_path when it points to a paper source.",
    )
    fetch_online_if_needed: bool = Field(
        default=True,
        description="Whether manuscript inspection may fetch article pages or linked dataset landing pages when needed.",
    )
    article_url: str = Field(
        default="",
        description="Optional explicit article/full-text URL to pair with the local PDF.",
    )
    dataset_hint: str = Field(
        default="",
        description="Optional short structured hint to bias candidate selection toward a specific experiment, figure, tissue, or accession.",
    )
    target_dataset_prompt: str = Field(
        default="",
        description=(
            "Optional binding natural-language scope prompt from the user/main agent, e.g. "
            "'download only the in vitro culture arm; exclude in vivo transplantation and cytokine perturbation'. "
            "When provided, paper intake should select and materialize only assets needed for that subset."
        ),
    )
    use_llm: bool = Field(
        default=True,
        description="Whether to use the current runtime LLM for the manuscript inspection rounds.",
    )


class _MaterializePaperDatasetInput(BaseModel):
    input_source: str = Field(
        default="",
        description="Optional paper PDF path, manuscript-derived text path, or article URL. Defaults to current input_path when it points to a paper source.",
    )
    dataset_hint: str = Field(
        default="",
        description="Optional short structured hint to bias selection toward a specific experiment, figure, tissue, or accession.",
    )
    target_dataset_prompt: str = Field(
        default="",
        description=(
            "Optional binding natural-language scope prompt generated from the user request, e.g. "
            "'materialize the in vitro arm only, days 2/4/6; exclude in vivo and full-study archives'. "
            "Use this whenever the user asks for a specific part of the paper data. Leave empty for default full-paper auto-selection."
        ),
    )
    selected_asset_url: str = Field(
        default="",
        description="Optional exact candidate asset URL to materialize. When provided, paper intake locks onto this asset instead of auto-switching to another candidate.",
    )
    fetch_online_if_needed: bool = Field(
        default=True,
        description="Whether manuscript inspection may fetch article pages or linked dataset landing pages when needed.",
    )
    article_url: str = Field(
        default="",
        description="Optional explicit article/full-text URL to pair with the local PDF.",
    )
    prefer_processed: bool = Field(
        default=True,
        description="Prefer processed assets such as h5ad, 10x h5, MEX, loom, or dense count tables over raw FASTQ/SRA assets.",
    )
    allow_raw: bool = Field(
        default=False,
        description="Allow raw FASTQ/SRA fallback when no suitable processed asset exists.",
    )
    require_downstream_ready: bool = Field(
        default=True,
        description="Require the resulting paper_input.h5ad to include enough metadata for CytoBridge preprocessing readiness.",
    )
    auto_load: bool = Field(
        default=True,
        description="Automatically load the resulting paper_input.h5ad into the runtime AnnData manager after materialization.",
    )
    use_llm: bool = Field(
        default=True,
        description="Whether to use the current runtime LLM for manuscript understanding and paper-specific materialization.",
    )


class _WebSearchInput(BaseModel):
    query: str = Field(description="Web search query.")
    count: int = Field(default=5, description="Maximum number of results or sources to surface.")
    allowed_domains: List[str] = Field(
        default_factory=list,
        description="Optional domain allowlist. Subdomains are allowed automatically.",
    )


class _BohriumPaperSearchInput(BaseModel):
    query: str = Field(description="Natural-language literature question to search in the Bohrium paper RAG index.")
    words: List[str] = Field(
        default_factory=list,
        description="Optional English keyword list. If omitted, keywords are derived from query.",
    )
    page_size: int = Field(default=10, description="Maximum number of papers to return, 1-100.")
    start_time: str = Field(default="", description="Optional start date in YYYY-MM-DD format.")
    end_time: str = Field(default="", description="Optional end date in YYYY-MM-DD format.")
    search_type: int = Field(default=5, description="Bohrium paper search type. Use 5 for title/abstract/corpus/image/target search.")
    jcr_zones: List[str] = Field(default_factory=list, description="Optional JCR filters such as Q1 or Q2.")
    include_dbs: List[str] = Field(default_factory=list, description="Optional database filters such as SCI.")


class _SearchTheoryInput(BaseModel):
    query: str = Field(description="Mathematical theory query, for example dynamic OT, WFR, Benamou-Brenier, or Schrodinger bridge.")
    top_k: int = Field(default=5, description="Maximum number of theory-book excerpts to return.")


class _WebFetchInput(BaseModel):
    url: str = Field(description="Public http/https URL to fetch.")
    max_chars: int = Field(
        default=12000,
        description="Maximum number of response characters to return after extraction/truncation.",
    )
    extract_mode: Literal["readable", "raw_text", "html"] = Field(
        default="readable",
        description="Readable article extraction, raw visible text extraction, or raw HTML.",
    )
    timeout_sec: float = Field(
        default=20.0,
        description="Request timeout in seconds.",
    )
    allowed_domains: List[str] = Field(
        default_factory=list,
        description="Optional domain allowlist. Subdomains are allowed automatically.",
    )


class _TerminalCommandInput(BaseModel):
    command: str = Field(
        description=(
            "Guarded terminal command to execute. Use `cwd` instead of `cd`; shell operators and write/edit "
            "commands are blocked. Allowed commands are a narrow whitelist of read-only inspection commands, "
            "restricted `curl`, and restricted `git clone`. "
            f"{SUPPORTED_COMMAND_SUMMARY}"
        )
    )
    cwd: str = Field(
        default="",
        description="Optional working directory for this one command. This does not persist across calls.",
    )
    timeout_sec: float = Field(
        default=20.0,
        description="Per-command timeout in seconds. The process group is killed when the timeout is reached.",
    )


class _SpawnSubagentInput(BaseModel):
    subagent_type: str = Field(
        default="general",
        description=(
            "Subagent role type. Current built-ins: general, proposal_evaluator, idea_evaluator, "
            "inference_evaluator, implementation_evaluator, paper_reviewer."
        ),
    )
    task: str = Field(description="Bounded task description for the subagent.")
    success_criteria: List[str] = Field(
        default_factory=list,
        description="Explicit completion criteria the subagent must satisfy before finishing.",
    )
    context_notes: str = Field(default="", description="Optional additional context from the parent planner.")
    relevant_paths: List[str] = Field(
        default_factory=list,
        description="Optional relevant absolute or workspace-relative paths for the task.",
    )


class _SubmitSubagentResultInput(BaseModel):
    status: Literal["completed", "needs_input", "failed"] = Field(
        description="Subagent completion status."
    )
    summary: str = Field(description="Concise summary of the result or failure.")
    findings: List[str] = Field(default_factory=list, description="Concrete findings discovered by the subagent.")
    artifact_refs: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Referenced artifacts or paths discovered/generated by the subagent.",
    )
    proposed_state_updates: Dict[str, Any] = Field(
        default_factory=dict,
        description="Suggested state updates for the parent planner to review and apply explicitly.",
    )
    needs_input_question: str = Field(
        default="",
        description="Concrete follow-up question when status is `needs_input`.",
    )


class _SubmitProposalReviewInput(BaseModel):
    decision: Literal["approve", "revise", "reject"] = Field(
        description="High-standard proposal review decision."
    )
    summary: str = Field(
        description="Concise overall verdict summary for the parent planner."
    )
    reviewer_feedback: str = Field(
        description="Detailed review feedback explaining the mathematical judgment and any required theoretical revisions."
    )
    findings: List[str] = Field(
        default_factory=list,
        description="Concrete review findings, concerns, or strengths discovered in the proposal.",
    )
    implementation_risks: List[str] = Field(
        default_factory=list,
        description="Concrete implementation or optimization risk points, sorted by expected severity from highest to lowest. These are advisory and must not by themselves determine approve/revise/reject.",
    )
    risk_assessment: str = Field(
        default="",
        description=(
            "Severity-ranked markdown risk assessment written by the proposal evaluator. It should describe each risk, "
            "its severity rationale, how it may manifest in the CytoBridge training/campaign framework, and practical diagnostics."
        ),
    )
    confidence: float = Field(
        default=0.0,
        description="Optional confidence score in [0, 1] for the decision.",
    )


class _SubmitImplementationReviewInput(BaseModel):
    decision: Literal["approve", "revise", "reject"] = Field(
        description="Proposal-to-implementation alignment verdict."
    )
    summary: str = Field(
        description="Concise verdict summary for the parent planner."
    )
    reviewer_feedback: str = Field(
        description="Actionable explanation of whether the implementation faithfully realizes the approved proposal."
    )
    findings: List[str] = Field(
        default_factory=list,
        description="Concrete code-grounded observations discovered during review.",
    )
    blocking_issues: List[str] = Field(
        default_factory=list,
        description="Issues that block trusted training/campaign evidence.",
    )
    advisory_risks: List[str] = Field(
        default_factory=list,
        description="Non-blocking implementation or scientific risks, sorted by severity when possible.",
    )
    efficiency_recommendations: List[str] = Field(
        default_factory=list,
        description="Non-blocking safe acceleration suggestions such as GPU/vectorization, mini-batch, chunking, caching, or parallelization.",
    )
    generalization_shortcut_risks: List[str] = Field(
        default_factory=list,
        description="Risks that the implementation/training strategy only works because of prior fitted state, warm starts, or target-specific cached artifacts.",
    )
    risk_assessment: str = Field(
        default="",
        description=(
            "Optional severity-ranked markdown notes for implementation-specific risks. Focus on code/config, "
            "scalability, POT/chunked OT alignment, runtime manifestation, and practical diagnostics."
        ),
    )


class _SubmitResearchIdeaReviewInput(BaseModel):
    decision: Literal["approve", "revise", "reject"] = Field(
        description="High-standard research-idea review decision."
    )
    summary: str = Field(
        description="Concise overall verdict summary for the parent planner."
    )
    reviewer_feedback: str = Field(
        description="Detailed review feedback explaining the scope, package-fit, and problem-formulation judgment."
    )
    findings: List[str] = Field(
        default_factory=list,
        description="Concrete review findings, concerns, or strengths discovered in the idea artifact.",
    )
    confidence: float = Field(
        default=0.0,
        description="Optional confidence score in [0, 1] for the decision.",
    )


class _RunSavedPythonScriptInput(BaseModel):
    script_path: str = Field(
        description="Absolute path or output_dir/scripts-relative path to a saved Python script."
    )
    reason: str = Field(default="", description="Optional concise reason for running the saved script.")
    timeout: int = Field(default=300, description="Max execution time in seconds.")
    timeout_mode: Literal["shared", "isolated", "auto"] = Field(
        default="isolated",
        description="Timeout strategy, matching execute_python.",
    )


class _RecordDecisionInput(BaseModel):
    algorithm_id: str = Field(description="Custom algorithm id.")
    phase: str = Field(description="Workflow phase such as proposal, authoring, review, or training.")
    decision: str = Field(description="Short machine-readable decision label.")
    rationale: str = Field(description="Concise reason for the decision.")
    status: str = Field(default="final", description="Decision status such as final or provisional.")
    alternatives: List[str] = Field(default_factory=list, description="Optional alternative options considered.")
    evidence: List[str] = Field(default_factory=list, description="Optional evidence items supporting the decision.")


class _MarkAlgorithmFailedInput(BaseModel):
    algorithm_id: str = Field(description="Custom algorithm id.")
    reason: str = Field(description="Why the current algorithm lifecycle should be marked failed.")
    evidence: List[str] = Field(
        default_factory=list,
        description="Optional evidence items such as failed campaign ids, diagnostic paths, or reviewer findings.",
    )


class _MarkResultObsoleteInput(BaseModel):
    algorithm_id: str = Field(description="Custom algorithm id.")
    artifact_type: Literal["proposal", "run", "comparison"] = Field(description="Artifact type to mark obsolete.")
    artifact_id: str = Field(description="Artifact id, for example proposal_id or run_id.")
    reason: str = Field(description="Why the artifact is no longer valid evidence.")
    replaced_by: str = Field(default="", description="Optional replacement artifact id.")


class _ListExperimentHistoryInput(BaseModel):
    algorithm_id: str = Field(description="Custom algorithm id.")
    include_obsolete: bool = Field(default=False, description="Whether to include obsolete proposals/runs in the listing.")
    limit: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum number of recent proposals, snapshots, runs, decisions, and obsolete records to return per section. Defaults to 10.",
    )
    max_field_chars: int = Field(
        default=1200,
        ge=200,
        le=4000,
        description="Maximum characters for any long string field in the concise summary. Defaults to 1200.",
    )


class _RollbackAlgorithmWorkspaceInput(BaseModel):
    algorithm_id: str = Field(description="Custom algorithm id.")
    target_snapshot_id: str = Field(default="", description="Optional explicit workspace snapshot id to restore.")
    target_proposal_id: str = Field(default="", description="Optional proposal id; restores the latest snapshot linked to that proposal.")


class _ActivateAlgorithmWorkspaceInput(BaseModel):
    algorithm_id: str = Field(description="Custom algorithm id to make active for review/authoring/training workflow state.")
    target_snapshot_id: str = Field(default="", description="Optional explicit workspace snapshot id to bind as active without restoring files.")
    target_proposal_id: str = Field(default="", description="Optional proposal id to bind as active; if a matching snapshot exists it is also selected.")


class _SnapshotActiveAlgorithmWorkspaceInput(BaseModel):
    reason: str = Field(
        default="",
        description="Why this clean snapshot is being created, for example before review or training.",
    )


class _SetActiveBaselineRunInput(BaseModel):
    algorithm_id: str = Field(description="Custom algorithm id.")
    run_id: str = Field(description="Registered run_id to promote to active baseline.")
    reason: str = Field(description="Why this run is becoming the active baseline.")


class _CompareAlgorithmRunsInput(BaseModel):
    algorithm_id: str = Field(description="Custom algorithm id.")
    run_ids: List[str] = Field(default_factory=list, description="Optional explicit run ids to compare.")
    include_baseline: bool = Field(default=True, description="Include the active baseline run when available.")


class _RegisterAlgorithmBenchmarkDatasetInput(BaseModel):
    dataset_id: str = Field(description="Stable benchmark dataset id.")
    source_path: str = Field(
        description=(
            "Prepared .h5ad path to copy into ~/.cellcompass/algorithm_benchmarks. "
            "The file must already contain obs['time_point_processed'] and obsm['X_latent']; "
            "this tool validates and registers only, it does not preprocess."
        )
    )
    title: str = Field(default="", description="Optional human-readable dataset title.")
    description: str = Field(default="", description="Optional dataset-card description.")
    tags: List[str] = Field(default_factory=list, description="Optional dataset tags, e.g. simulation, tiny, 2d.")
    stage_relevance: List[str] = Field(
        default_factory=list,
        description="Optional campaign stages this dataset is useful for, e.g. stage1_feasibility.",
    )
    overwrite: bool = Field(default=False, description="Refresh an existing dataset card and copied data.")
    preprocessing_script_path: str = Field(
        default="",
        description=(
            "Optional path to the exact preprocessing script that produced source_path. "
            "When provided, the tool copies it into datasets/<dataset_id>/scripts/preprocessing/ "
            "and records original path, copied path, and SHA256 in dataset.json."
        ),
    )


class _RegisterStage2SimulationDatasetInput(BaseModel):
    dataset_id: str = Field(description="Stable benchmark dataset id for the generated simulation.")
    source_path: str = Field(
        description=(
            "Prepared generated .h5ad path. It must already contain obs['time_point_processed'] "
            "and obsm['X_latent']; this tool validates and registers only."
        )
    )
    generator_path: str = Field(description="Path to the simulation generator script or source directory to fingerprint and archive.")
    algorithm_id: str = Field(default="", description="Optional custom algorithm id this simulation was designed to validate.")
    proposal_id: str = Field(default="", description="Optional proposal id whose claim this simulation targets.")
    simulation_version: str = Field(
        default="",
        description="Optional explicit frozen simulation version. Empty derives one from dataset and generator hashes.",
    )
    claim_metric_name: str = Field(default="", description="Optional campaign claim metric name expected for this simulation.")
    title: str = Field(default="", description="Optional human-readable dataset title.")
    description: str = Field(default="", description="Optional dataset-card description.")
    tags: List[str] = Field(default_factory=list, description="Optional extra dataset tags; simulation/stage2/claim are added automatically.")
    overwrite: bool = Field(default=False, description="Refresh the dataset card only when intentionally replacing the generated data.")
    notes: str = Field(default="", description="Optional concise provenance or simulation-design notes.")


class _ListAlgorithmBenchmarksInput(BaseModel):
    stage: str = Field(default="", description="Optional stage filter such as stage1_feasibility.")
    tags: List[str] = Field(default_factory=list, description="Optional tag filters; any matching tag is included.")
    ready_only: bool = Field(default=False, description="If true, only return datasets satisfying the training data contract.")


class _GetAlgorithmBenchmarkDatasetInput(BaseModel):
    dataset_id: str = Field(description="Benchmark dataset id.")


class _GetAlgorithmBenchmarkBaselinesInput(BaseModel):
    dataset_id: str = Field(description="Benchmark dataset id.")


class _ListAlgorithmBenchmarkBaselinesInput(BaseModel):
    dataset_id: str = Field(description="Benchmark dataset id.")


class _RecordAlgorithmBenchmarkBaselineInput(BaseModel):
    dataset_id: str = Field(description="Benchmark dataset id.")
    algorithm_name: str = Field(description="Builtin or reference algorithm name.")
    metrics: Dict[str, Any] = Field(description="W1/TMV metrics payload, optionally including runtime_sec and memory_peak_mb.")
    baseline_type: Literal["builtin", "reference"] = Field(
        default="builtin",
        description="Baseline type. Custom claim metrics must stay in campaign registries, not benchmark leaderboards.",
    )
    run_id: str = Field(default="", description="Optional training run id that produced these metrics.")
    config_path: str = Field(default="", description="Optional config path used for this baseline run.")
    notes: str = Field(default="", description="Optional concise provenance note.")


class _MakeBenchmarkDatasetConfigInput(BaseModel):
    dataset_ids: Optional[List[str]] = Field(default=None, description="Benchmark dataset ids. Empty means ready datasets matching stage.")
    stage: str = Field(default="", description="Optional stage filter used only when dataset_ids is empty.")
    common_config_overrides: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Config overrides applied to every dataset run in the panel. Pass sparse leaf overrides: "
            "when changing one hyperparameter, include only that leaf path/value, not a copied training/model config block."
        ),
    )
    per_dataset_config_overrides: Optional[Dict[str, Dict[str, Any]]] = Field(
        default=None,
        description=(
            "Dataset-id keyed config overrides for dataset-specific custom algorithm configs. "
            "Use this when simulation_gene_2d and weinreb_rawrebuild_full_k10mindiff1 need different batch size, coupling, or solver hyperparameters. "
            "Each dataset override should also be sparse leaf paths/values, for example {'training.plan[0].lr': 0.002}."
        ),
    )

    @field_validator("dataset_ids", mode="before")
    @classmethod
    def _coerce_dataset_ids(cls, value: Any) -> Any:
        return _json_string_to_list(value, "dataset_ids")

    @field_validator("common_config_overrides", "per_dataset_config_overrides", mode="before")
    @classmethod
    def _coerce_config_objects(cls, value: Any) -> Any:
        return _json_string_to_object(value, "config overrides")


class _StartAlgorithmCampaignInput(BaseModel):
    algorithm_id: str = Field(description="Custom algorithm id.")
    proposal_id: str = Field(default="", description="Optional exact approved proposal id; defaults to the active proposal.")
    claim_metric_spec: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Claim metric contract for Stage 2. For a custom Stage 2 claim metric, name/direction alone is not enough: "
            "include evaluator_path (and evaluator_function when not using the default) so the same metric can be computed "
            "from EvaluationMetricsContext.evaluation_trajectory for the candidate and builtin/reference baselines. "
            "For a formal held-out timepoint Stage 2 claim metric, use "
            "evaluator_path='builtin:holdout_time_w1', direction='lower', and include "
            "holdout_time_evaluation/time_points/mode/time_key/latent_key in claim_metric_spec; the campaign will run "
            "the same auxiliary split protocol for candidate and baselines and preserve evaluator provenance. "
            "May include external_baselines/stage_baselines for stage gates. TMV gate policy is inherited from "
            "the proposal's machine-readable mass_modeling_scope unless explicitly overridden. "
            "For agent-designed Stage 2 simulations, include simulation_dataset_id or simulation_version from "
            "register_stage2_simulation_dataset(...); baselines must carry the same simulation_version. "
            "If a builtin/reference baseline needs metric I/O adaptation, use baseline_metric_adapters for "
            "output standardization or explicit unsupported declarations; do not provide replacement numeric "
            "claim metrics or change the metric definition per baseline."
        ),
    )


class _UpdateCampaignClaimMetricSpecInput(BaseModel):
    campaign_id: str = Field(default="", description="Campaign id. Defaults to the active campaign.")
    claim_metric_spec: Dict[str, Any] = Field(
        description=(
            "Authoritative claim metric contract repair for an existing active campaign. "
            "Use to add or correct evaluator_path/evaluator_function/provenance without starting a duplicate campaign."
        )
    )
    reason: str = Field(default="", description="Why this campaign claim metric contract repair is needed.")

    @field_validator("claim_metric_spec", mode="before")
    @classmethod
    def _coerce_claim_metric_spec(cls, value: Any) -> Any:
        return _json_string_to_object(value, "claim_metric_spec")


class _SetCampaignStagePanelInput(BaseModel):
    campaign_id: str = Field(default="", description="Campaign id. Defaults to the active campaign.")
    stage: str = Field(default="", description="Campaign stage. Empty means the current stage.")
    dataset_config_overrides: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Frozen stage panel payload. First call make_benchmark_dataset_config(...), then pass its returned "
            "dataset_config_overrides here. It must include datasets or target_dataset_ids. Dataset ids are fixed "
            "for the stage; later trials may still change common_config_overrides/per_dataset config values for those same ids. "
            "Use sparse leaf config overrides for tuning; do not copy the full training/model config block."
        ),
    )
    overwrite: bool = Field(default=False, description="Only allowed before completed trials; replaces an existing same-stage panel.")

    @field_validator("dataset_config_overrides", mode="before")
    @classmethod
    def _coerce_dataset_config_overrides(cls, value: Any) -> Any:
        return _json_string_to_object(value, "dataset_config_overrides")


class _SwitchCampaignStagePanelInput(BaseModel):
    campaign_id: str = Field(default="", description="Campaign id. Defaults to the active campaign.")
    stage: str = Field(default="", description="Campaign stage. Empty means the current stage.")
    dataset_config_overrides: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "New stage panel payload from make_benchmark_dataset_config(...). This intentionally changes dataset ids "
            "inside the same stage without resetting trial_count/promote_count/reject_count. It clears the current "
            "stage active best and gate/baseline evidence so the next trial and baseline refresh use the new panel. "
            "Config overrides in the new payload should be sparse leaf paths/values."
        ),
    )
    reason: str = Field(
        default="",
        description="Why changing datasets mid-stage is useful, for example adding a real scalability benchmark.",
    )

    @field_validator("dataset_config_overrides", mode="before")
    @classmethod
    def _coerce_dataset_config_overrides(cls, value: Any) -> Any:
        return _json_string_to_object(value, "dataset_config_overrides")


class _RefreshCampaignStageBaselinesInput(BaseModel):
    campaign_id: str = Field(default="", description="Campaign id. Defaults to the active campaign.")
    stage: str = Field(default="", description="Campaign stage. Empty means the current stage.")
    baseline_algorithms: Optional[List[str]] = Field(
        default=None,
        description=(
            "Optional explicit builtin/reference candidate names to run or refresh as a repair scope. Empty uses all runnable builtin baselines. In strict_all_builtin mode, gates still audit the full fixed builtin comparator set and any required baseline outside this repair scope remains an explicit blocker."
        ),
    )
    run_missing: bool = Field(
        default=True,
        description=(
            "Run truly missing baseline records on every dataset in the frozen stage panel. If false, only existing records are used. "
            "For Stage 2, an existing W1/TMV baseline without the custom claim metric is not retrained just because evaluator_path is missing; "
            "provide claim_metric_spec.evaluator_path so the claim metric can be computed posthoc from saved trajectories/models."
        ),
    )
    overwrite_existing: bool = Field(
        default=False,
        description="Rerun baseline records even when matching dataset/algorithm baseline records already exist.",
    )
    baseline_type: Literal["builtin", "reference"] = Field(
        default="builtin",
        description="Baseline record type to read/write.",
    )

    @field_validator("baseline_algorithms", mode="before")
    @classmethod
    def _coerce_baseline_algorithms(cls, value: Any) -> Any:
        return _json_string_to_list(value, "baseline_algorithms")


class _RegisterCampaignControlBaselineInput(BaseModel):
    campaign_id: str = Field(default="", description="Campaign id. Defaults to the active campaign.")
    baseline_id: str = Field(
        default="",
        description="Stable id for this frozen control/ablation baseline. Leave empty to derive from baseline_name.",
    )
    baseline_name: str = Field(
        description="Human-readable control baseline name, for example shuffle_control or no_lag_ablation."
    )
    baseline_type: str = Field(
        default="control",
        description="Control kind such as control, ablation, shuffle, negative_control, or reference_control.",
    )
    stage: str = Field(default="", description="Campaign stage. Empty means the current stage.")
    metrics: Dict[str, Any] = Field(
        description=(
            "Measured metrics from the actual control/ablation run on the frozen stage panel. "
            "Include w1_mean or w1_scores and any claim metric/custom_metrics needed for gates."
        ),
    )
    target_dataset_ids: Optional[List[str]] = Field(
        default=None,
        description="Dataset ids used by the control run. Must match the frozen stage panel when one exists.",
    )
    metric_roles: Optional[List[str]] = Field(
        default=None,
        description="Which gate metrics this baseline should compare: both, w1, claim, or exact metric names.",
    )
    required_for_gate: bool = Field(
        default=True,
        description="If true, include this frozen control baseline in strongest-comparator gate selection.",
    )
    source_run_id: str = Field(default="", description="Run id that produced the control metrics, when available.")
    source_trial_id: str = Field(default="", description="Campaign trial id associated with the control run, when available.")
    source_metrics_path: str = Field(default="", description="Path to the metrics JSON for audit, when available.")
    source_config_path: str = Field(default="", description="Path to the control config/resolved config for audit, when available.")
    source_artifact_paths: Optional[Dict[str, str]] = Field(
        default=None,
        description="Additional provenance artifact paths, such as logs, model artifacts, or notebooks.",
    )
    control_code_source: Dict[str, Any] = Field(
        description=(
            "Required code/config source for this control. Use {'kind':'config_overrides','config_overrides': {...}} "
            "or {'kind':'workspace_patch','patch': '...'} / {'kind':'workspace_patch','patch_path':'...'}."
        )
    )
    reason: str = Field(
        default="",
        description=(
            "Why this control matters scientifically, for example proving performance exceeds a shuffled-label baseline."
        ),
    )

    @field_validator("metrics", "source_artifact_paths", "control_code_source", mode="before")
    @classmethod
    def _coerce_metric_objects(cls, value: Any) -> Any:
        return _json_string_to_object(value, "metrics/source_artifact_paths/control_code_source")

    @field_validator("target_dataset_ids", "metric_roles", mode="before")
    @classmethod
    def _coerce_lists(cls, value: Any) -> Any:
        return _json_string_to_list(value, "target_dataset_ids/metric_roles")


class _UpdateCampaignControlBaselineInput(BaseModel):
    campaign_id: str = Field(default="", description="Campaign id. Defaults to the active campaign.")
    baseline_id: str = Field(description="Registered control baseline id to deactivate, reactivate, update, or replace.")
    stage: str = Field(default="", description="Campaign stage. Empty means the current stage.")
    action: Literal["deactivate", "reactivate", "update", "replace"] = Field(
        default="deactivate",
        description="deactivate removes the control from gate selection; replace updates metrics/code source with revision history.",
    )
    metrics: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Replacement measured metrics for update/replace actions.",
    )
    target_dataset_ids: Optional[List[str]] = Field(
        default=None,
        description="Dataset ids for replacement metrics; must match the frozen stage panel when one exists.",
    )
    metric_roles: Optional[List[str]] = Field(
        default=None,
        description="Optional replacement roles: both, w1, claim, or exact metric names.",
    )
    required_for_gate: Optional[bool] = Field(
        default=None,
        description="Optional replacement gate participation flag.",
    )
    source_run_id: str = Field(default="", description="Replacement run id, when available.")
    source_trial_id: str = Field(default="", description="Replacement trial id, when available.")
    source_metrics_path: str = Field(default="", description="Replacement metrics path, when available.")
    source_config_path: str = Field(default="", description="Replacement resolved config path, when available.")
    source_artifact_paths: Optional[Dict[str, str]] = Field(default=None, description="Replacement artifact provenance.")
    control_code_source: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Replacement code/config source for update/replace actions.",
    )
    reason: str = Field(default="", description="Reason for the revision/deactivation.")

    @field_validator("metrics", "source_artifact_paths", "control_code_source", mode="before")
    @classmethod
    def _coerce_objects(cls, value: Any) -> Any:
        return _json_string_to_object(value, "metrics/source_artifact_paths/control_code_source")

    @field_validator("target_dataset_ids", "metric_roles", mode="before")
    @classmethod
    def _coerce_lists(cls, value: Any) -> Any:
        return _json_string_to_list(value, "target_dataset_ids/metric_roles")


class _RunCampaignControlBaselineInput(BaseModel):
    campaign_id: str = Field(default="", description="Campaign id. Defaults to the active campaign.")
    baseline_id: str = Field(default="", description="Stable id for this control baseline.")
    baseline_name: str = Field(description="Human-readable control baseline name.")
    baseline_type: str = Field(default="ablation", description="Control kind such as ablation, shuffle, no_lag, or negative_control.")
    stage: str = Field(default="", description="Campaign stage. Empty means the current stage.")
    control_code_source: Dict[str, Any] = Field(
        description=(
            "Declared control source. Use kind=config_overrides with config_overrides for sparse ablations, "
            "or kind=workspace_patch with patch/patch_path for code ablations."
        )
    )
    metric_roles: Optional[List[str]] = Field(default=None, description="Gate metric roles: both, w1, claim, or exact metric names.")
    required_for_gate: bool = Field(default=True, description="If true, include the resulting control in gate selection.")
    replace_existing: bool = Field(
        default=True,
        description="If true and baseline_id already exists, replace it with an audited revision instead of failing.",
    )
    dataset_config_overrides: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional frozen-panel-compatible dataset payload. Usually omit to reuse the stage panel.",
    )
    seed: Optional[int] = Field(default=None, description="Optional random seed for the control training run.")
    reason: str = Field(default="", description="Why this control is scientifically required.")

    @field_validator("control_code_source", "dataset_config_overrides", mode="before")
    @classmethod
    def _coerce_objects(cls, value: Any) -> Any:
        return _json_string_to_object(value, "control_code_source/dataset_config_overrides")

    @field_validator("metric_roles", mode="before")
    @classmethod
    def _coerce_lists(cls, value: Any) -> Any:
        return _json_string_to_list(value, "metric_roles")


class _RunCampaignLockedAlgorithmBaselineInput(BaseModel):
    campaign_id: str = Field(default="", description="Campaign id. Defaults to the active campaign.")
    reference_algorithm_id: str = Field(
        description=(
            "Completed custom algorithm id to use as an audited baseline. The algorithm must have "
            "algorithm_lifecycle_status=complete and a final_regression locked release."
        ),
    )
    reference_campaign_id: str = Field(
        default="",
        description="Optional completed campaign id for the reference algorithm; defaults to its registry completed_campaign_id.",
    )
    baseline_id: str = Field(default="", description="Stable id for this locked algorithm baseline.")
    baseline_name: str = Field(default="", description="Human-readable baseline name. Defaults to reference_algorithm_id.")
    baseline_type: str = Field(default="locked_algorithm", description="Baseline kind label for audit records.")
    stage: str = Field(default="", description="Campaign stage. Empty means the current stage.")
    metric_roles: Optional[List[str]] = Field(default=None, description="Gate metric roles: both, w1, claim, or exact metric names.")
    required_for_gate: bool = Field(default=True, description="If true, include the resulting baseline in gate selection.")
    replace_existing: bool = Field(
        default=True,
        description="If true and baseline_id already exists, replace it with an audited revision instead of failing.",
    )
    dataset_config_overrides: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional frozen-panel-compatible dataset payload. Usually omit to reuse the stage panel.",
    )
    reference_config_overrides: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Deprecated and rejected. Do not guess compatibility through config overrides; use reference_dataset_adapter "
            "to materialize an adapted h5ad with an explicit script."
        ),
    )
    reference_dataset_adapter: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Optional explicit dataset adapter for the locked reference algorithm. Use "
            "{'kind':'python_script','script_path':'...'} or {'kind':'python_script','script':'...'}; the script is called "
            "with --input-adata, --output-adata, --dataset-id, --reference-algorithm-id, --current-algorithm-id, and "
            "must write the adapted .h5ad. This adapts data format only; it must not edit or retune the locked algorithm."
        ),
    )
    reference_claim_metric_adapter: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Optional explicit claim-metric context wrapper for evaluating the locked reference under the current "
            "campaign claim_metric_spec. Use {'kind':'claim_metric_adapter','adapter_path':'...'} or inline "
            "{'kind':'python_script','script':'...'} defining adapt_baseline_metric_context(context). This adapts the "
            "evaluation context before the current campaign evaluator runs; it must not replace the evaluator or retune "
            "the locked algorithm."
        ),
    )
    seed: Optional[int] = Field(default=None, description="Optional random seed for the reference baseline run.")
    reason: str = Field(default="", description="Why this completed algorithm is the appropriate comparator.")

    @field_validator(
        "dataset_config_overrides",
        "reference_config_overrides",
        "reference_dataset_adapter",
        "reference_claim_metric_adapter",
        mode="before",
    )
    @classmethod
    def _coerce_objects(cls, value: Any) -> Any:
        return _json_string_to_object(
            value,
            "dataset_config_overrides/reference_config_overrides/reference_dataset_adapter/reference_claim_metric_adapter",
        )

    @field_validator("metric_roles", mode="before")
    @classmethod
    def _coerce_lists(cls, value: Any) -> Any:
        return _json_string_to_list(value, "metric_roles")


class _ComputeCampaignClaimMetricForBaselinesInput(BaseModel):
    campaign_id: str = Field(default="", description="Campaign id. Defaults to the active campaign.")
    stage: str = Field(default="stage2_claim_validation", description="Usually stage2_claim_validation.")
    baseline_algorithms: Optional[List[str]] = Field(
        default=None,
        description="Optional builtin/reference baselines. Empty uses all default runnable builtin baselines.",
    )
    use_saved_trajectory: bool = Field(
        default=True,
        description="Prefer saved baseline evaluation trajectories/models. This is the default and should usually remain true.",
    )
    regenerate_trajectory_if_missing: bool = Field(
        default=False,
        description="If true, train/run missing baselines. If false, report structured missing records instead of launching new training.",
    )
    baseline_type: Literal["builtin", "reference"] = Field(default="builtin", description="Baseline record type.")

    @field_validator("baseline_algorithms", mode="before")
    @classmethod
    def _coerce_baseline_algorithms(cls, value: Any) -> Any:
        return _json_string_to_list(value, "baseline_algorithms")


class _QueryCampaignBaselineMetricsInput(BaseModel):
    campaign_id: str = Field(default="", description="Campaign id. Defaults to the active campaign.")
    stage: str = Field(default="", description="Optional campaign stage filter, such as stage2_claim_validation.")
    dataset_id: str = Field(default="", description="Optional dataset id filter.")
    metric_names: Optional[List[str]] = Field(
        default=None,
        description="Optional metric names to extract explicitly, for example ['LCFDE', 'w1_mean'].",
    )
    output_dir: str = Field(
        default="",
        description="Optional output_dir to scan for local baseline training_run metrics.json files. Empty uses session output_dir.",
    )
    include_training_run_scan: bool = Field(default=True, description="Scan output_dir/training_runs for baseline metrics.json files.")
    include_benchmark_registry: bool = Field(default=True, description="Include benchmark registry baseline records when available.")

    @field_validator("metric_names", mode="before")
    @classmethod
    def _coerce_metric_names(cls, value: Any) -> Any:
        return _json_string_to_list(value, "metric_names")


class _GetAlgorithmCampaignStatusInput(BaseModel):
    campaign_id: str = Field(default="", description="Campaign id. Defaults to the active campaign.")
    recent_trials_limit: int = Field(
        default=3,
        ge=0,
        le=20,
        description="Number of most recent trials to include in the concise summary.",
    )


class _StartCampaignTrialInput(BaseModel):
    campaign_id: str = Field(default="", description="Campaign id. Defaults to the active campaign.")
    base_trial_id: str = Field(
        default="",
        description=(
            "Optional rejected trial id to resume as non-active working base. "
            "If a campaign already has an open trial, start_campaign_trial is idempotent and returns that same trial instead of creating a new one."
        ),
    )


class _RunCampaignTrialInput(BaseModel):
    campaign_id: str = Field(default="", description="Campaign id. Defaults to the active campaign.")
    dataset_config_overrides: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Dataset-specific config archived with this trial. For a panel, pass "
            "{'datasets':[{'dataset_id':'small','adata_path':'...','config_overrides':{...}}, ...], "
            "'config_overrides':{...common...}}. Prefer make_benchmark_dataset_config(...) instead of hand-writing paths. "
            "After a stage panel is frozen, config-only payloads are allowed: either {'config_overrides':{...}} or "
            "{'common_config_overrides':{...}, 'per_dataset_config_overrides':{'dataset_id':{...}}}; these are merged onto the frozen dataset entries. "
            "Pass sparse overrides only: when tuning one parameter, pass one leaf path/value such as training.plan[0].lr, "
            "not a copied training/model config block. Broad copied config blocks are pruned to changed leaf values before training. "
            "Override paths may use nested dicts, dot notation, or bracket list notation such as training.plan[0].lr. "
            "Epoch edits need an explicit reason; put __allow_epoch_override and __epoch_override_reason either at this payload top level "
            "or inside config_overrides. "
            "The first panel used in a stage freezes that stage's dataset ids. Later trials may tune config values but must keep the same dataset ids. "
            "If omitted after a panel is frozen, the frozen panel payload is reused. The agent does not pass decision/promote/reject; "
            "the campaign controller decides automatically after training."
        ),
    )
    holdout_time_evaluation: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Optional default-off trajectory generalization diagnostic. Use only when the algorithm claim needs "
            "held-out timepoint validation or trajectory reasonableness evidence. Shape: "
            "{'enabled': true, 'mode': 'single'|'sequential'|'simultaneous', 'time_points':[...], "
            "'time_key':'time_point_processed', 'latent_key':'X_latent', 'allow_initial_time': false}. "
            "The campaign tool always runs the normal non-holdout trial first so standard gate logic and fixed "
            "baseline comparison are preserved; holdout evaluation then trains auxiliary split(s), predicts the "
            "held-out distribution, and records heldout W1 under the run's holdout_time_evaluation artifacts. "
            "Leave disabled if a stronger claim metric is already available. attach_to_custom_metrics/use_as_claim_metric "
            "only copies the diagnostic mean W1 into custom_metrics; it does not replace campaign claim_metric_spec "
            "or its evaluator provenance. For a formal Stage 2 claim metric, provide a real campaign evaluator and "
            "compute the same holdout protocol for comparable baselines."
        ),
    )

    @field_validator("dataset_config_overrides", mode="before")
    @classmethod
    def _coerce_dataset_config_overrides(cls, value: Any) -> Any:
        return _json_string_to_object(value, "dataset_config_overrides")

    @field_validator("holdout_time_evaluation", mode="before")
    @classmethod
    def _coerce_holdout_time_evaluation(cls, value: Any) -> Any:
        return _json_string_to_object(value, "holdout_time_evaluation")


class _AbortCurrentCampaignTrialInput(BaseModel):
    campaign_id: str = Field(default="", description="Campaign id. Defaults to the active campaign.")
    reason: str = Field(
        default="",
        description=(
            "Why the open trial is being aborted. Use this only for stale/incomplete current trials, "
            "for example after a stop/OOM/tool interruption leaves current_trial.status=running with no live process."
        ),
    )
    restore_active_best: bool = Field(
        default=True,
        description=(
            "If true, restore the stage active-best workspace snapshot after aborting. "
            "This is the safe default for stale running trials and does not change the active-best pointer."
        ),
    )


class _PatchAlgorithmConfigInput(BaseModel):
    algorithm_id: str = Field(
        default="",
        description="Custom algorithm id. Empty uses the active algorithm context.",
    )
    updates: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Typed config patch as path -> value entries. Prefer leaf paths such as "
            "training.plan[0].lr, training.plan.0.batch_size, or model.hidden_dims[0]. "
            "Do not pass whole training.plan lists or nested config blocks unless allow_list_replace=true."
        ),
    )
    dry_run: bool = Field(
        default=False,
        description="If false, write config.yaml and return the resolved diff. Set true to preview without writing.",
    )
    reason: str = Field(default="", description="Why this persistent config change is being made.")
    allow_create: bool = Field(
        default=False,
        description="Allow creating missing config keys. Default false catches mistyped paths.",
    )
    allow_list_replace: bool = Field(
        default=False,
        description="Allow replacing whole list/dict values. Default false prevents accidental training.plan replacement.",
    )
    strict_types: bool = Field(
        default=True,
        description="Require patched leaf values to match the existing scalar type, except int/float are compatible.",
    )
    allow_epoch_override: bool = Field(
        default=False,
        description="Allow changing epochs. Default false preserves the 3000-epoch convergence discipline.",
    )
    epoch_override_reason: str = Field(
        default="",
        description="Required when allow_epoch_override=true; explain convergence evidence for changing epochs.",
    )


class _CreateAlgorithmWorkspaceArtifactInput(BaseModel):
    relative_path: str = Field(
        description=(
            "Path relative to the active algorithm workspace, for example "
            "diagnostics/check_mass_drift.py or diagnostics/mass_drift_summary.json. "
            "Absolute paths and '..' are rejected."
        ),
    )
    kind: Literal["file", "directory"] = Field(
        default="file",
        description="Create a file or directory under the active algorithm workspace.",
    )
    content: str = Field(default="", description="File content. Ignored for kind='directory'.")
    overwrite: bool = Field(default=False, description="If true, replace an existing file. Directories are idempotent.")
    reason: str = Field(default="", description="Why this algorithm workspace artifact is being created.")
    algorithm_id: str = Field(
        default="",
        description="Optional algorithm id. Empty uses the active algorithm context; if provided it must match the active workspace.",
    )


class _ResumeRejectedTrialInput(BaseModel):
    campaign_id: str = Field(default="", description="Campaign id. Defaults to the active campaign.")
    trial_id: str = Field(description="Rejected trial id to resume.")


class _ListCampaignTrialsInput(BaseModel):
    campaign_id: str = Field(default="", description="Campaign id. Defaults to the active campaign.")
    decision: str = Field(default="", description="Optional decision filter: promote or reject.")
    stage: str = Field(default="", description="Optional stage filter, for example stage1_feasibility or stage3_tuning.")
    limit: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum number of concise trial summaries to return. Defaults to 10 and is capped at 50.",
    )
    offset: int = Field(
        default=0,
        ge=0,
        description="Pagination offset after filtering and sorting by most-recent update first.",
    )


class _CheckCampaignStageGateInput(BaseModel):
    campaign_id: str = Field(default="", description="Campaign id. Defaults to the active campaign.")
    advance: bool = Field(
        default=True,
        description=(
            "If true, a passing gate advances to the next stage or locks final regression. "
            "If false, only checks readiness. Stage 3 has an optional SOTA/Pareto gate for early completion, "
            "but missing that optional gate after budget exhaustion does not fail the already Stage-2-validated algorithm; "
            "advance=true moves from Stage 3 to final regression when the optional gate passes or the tuning budget is exhausted. "
            "Final regression is a frozen benchmark/reporting stage, not an external-baseline pass/fail gate."
        ),
    )


class SingleAgentTools(PlannerTools):
    def __init__(
        self,
        llm: ChatOpenAI,
        state: AgentState,
        event_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        checkpoint_callback: Optional[Callable[[], None]] = None,
        *,
        agent_role: str = "planner",
        agent_id: str = "planner",
        parent_agent_id: str = "",
        subagent_type: str = "",
        tool_policy: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(llm=llm, state=state, event_callback=event_callback)
        self.agent_role = str(agent_role or "planner").strip().lower() or "planner"
        self.agent_id = str(agent_id or "planner").strip() or "planner"
        self.parent_agent_id = str(parent_agent_id or "").strip()
        resolved_subagent_type = str(subagent_type or "").strip().lower()
        if self.agent_role == "subagent":
            resolved_subagent_type = (
                resolved_subagent_type
                or str((tool_policy or {}).get("subagent_type") or "").strip().lower()
                or "general"
            )
            self.subagent_type = resolve_subagent_type(resolved_subagent_type).name
            base_policy = default_subagent_tool_policy(self.subagent_type)
            merged_policy = dict(base_policy)
            candidate_policy = dict(tool_policy or {})
            merged_policy.update(
                {
                    key: value
                    for key, value in candidate_policy.items()
                    if key not in {"allowed_tools", "disallowed_tools", "subagent_type"}
                }
            )
            merged_policy["allowed_tools"] = sorted(
                {
                    str(item).strip()
                    for item in list(base_policy.get("allowed_tools") or []) + list(candidate_policy.get("allowed_tools") or [])
                    if str(item).strip()
                }
            )
            merged_policy["disallowed_tools"] = sorted(
                {
                    str(item).strip()
                    for item in list(base_policy.get("disallowed_tools") or []) + list(candidate_policy.get("disallowed_tools") or [])
                    if str(item).strip()
                }
            )
            merged_policy["subagent_type"] = self.subagent_type
            self.tool_policy = merged_policy
        else:
            self.subagent_type = ""
            self.tool_policy = dict(tool_policy or {})
        self.checkpoint_callback = checkpoint_callback
        self.committer = WorkflowCommitter(state, event_sink=self._emit_event)
        self._rag_manager: Optional[Any] = None
        self._submitted_subagent_result: Optional[Dict[str, Any]] = None
        self.subagent_manager: Optional[SubagentRuntimeManager] = None
        if self.agent_role == "planner":
            self.subagent_manager = SubagentRuntimeManager(
                llm=llm,
                state=state,
                parent_agent_id=self.agent_id,
                event_sink=self._emit_event,
                checkpoint_callback=self.checkpoint_callback,
            )

    @property
    def rag_manager(self) -> Any:
        if self._rag_manager is None:
            from ..rag.rag_main import RAGManager

            self._rag_manager = RAGManager(
                llm_client=self.llm,
            )
        return self._rag_manager



    def _generate_parent_hyde_queries(self, query: str, max_expansions: int = 2) -> Tuple[List[str], str]:
        base_query = str(query or "").strip()
        queries: List[str] = [base_query] if base_query else []
        if os.environ.get("CYTOBRIDGE_RAG_PARENT_HYDE", "1").strip() == "0" or not self.llm:
            return queries, "disabled"
        try:
            from ..rag.config import call_llm
            from ..rag.rag_main import HYDE_TEMPLATES

            for template in list(HYDE_TEMPLATES)[: max(0, int(max_expansions))]:
                user_prompt = str(template.get("prompt") or "").format(query=base_query)
                response = call_llm(
                    llm_client=self.llm,
                    user_prompt=user_prompt,
                    system_prompt=str(template.get("system") or ""),
                    temperature=0.7,
                    agent="rag_hyde",
                )
                generated = str(response or "").strip().replace("\n", " ")
                if len(generated) > 50 and generated not in queries:
                    queries.append(generated)
            return queries, "generated" if len(queries) > 1 else "no_expansion"
        except BaseException as exc:  # noqa: BLE001
            return queries, f"fallback_original_query:{type(exc).__name__}:{str(exc)[:240]}"

    def _rerank_literature_candidates(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> Tuple[List[Dict[str, Any]], str]:
        if not candidates:
            return [], "no_candidates"
        top_k = max(1, min(int(top_k or 5), len(candidates)))
        if os.environ.get("CYTOBRIDGE_RAG_PARENT_RERANK", "1").strip() == "0" or not self.llm:
            return list(candidates[:top_k]), "disabled"
        try:
            from ..rag.config import call_llm
            from ..rag.knowledge_searcher import parse_llm_rerank_indices

            blocks: List[str] = []
            for idx, item in enumerate(candidates, 1):
                excerpt_lines: List[str] = []
                for chunk in list(item.get("fine_grained_chunks") or [])[:2]:
                    page = chunk.get("page", "?")
                    text = str(chunk.get("text") or "").strip().replace("\n", " ")[:360]
                    if text:
                        excerpt_lines.append(f"- p.{page}: {text}")
                blocks.append(
                    "\n".join(
                        [
                            f"Document {idx}",
                            f"Title: {item.get('title') or 'Unknown'}",
                            f"Summary: {str(item.get('summary') or '')[:420]}",
                            f"Methods: {', '.join(map(str, list(item.get('key_methods') or [])[:4]))}",
                            "Evidence:",
                            "\n".join(excerpt_lines) if excerpt_lines else "- no excerpt attached",
                        ]
                    )
                )
            prompt = f"""Rank these local-literature search candidates for relevance to the query.

Query: {query!r}

Keep only candidates that are directly useful for later scientific reasoning. Return only JSON:
{{"ranked_indices": [2, 1, 4]}}

Candidates:
{chr(10).join(blocks)}
"""
            response = call_llm(
                llm_client=self.llm,
                user_prompt=prompt,
                system_prompt="You are a precise scientific literature relevance judge.",
                temperature=0.1,
                agent="rag_rerank",
            )
            indices = parse_llm_rerank_indices(str(response or ""), len(candidates))
            if not indices:
                return list(candidates[:top_k]), "fallback_no_valid_indices"
            selected = [dict(candidates[idx - 1], rerank_position=pos) for pos, idx in enumerate(indices, 1)]
            return selected[:top_k], "reranked"
        except BaseException as exc:  # noqa: BLE001
            return list(candidates[:top_k]), f"fallback_original_order:{type(exc).__name__}:{str(exc)[:240]}"

    def _format_literature_results(self, results: List[Dict[str, Any]]) -> str:
        try:
            from ..rag.rag_main import RAGManager

            manager = RAGManager(llm_client=None)
            return manager.format_literature_context(results, max_excerpts_per_paper=3)
        except BaseException:  # noqa: BLE001
            lines = ["Literature Retrieval Results", f"Found {len(results)} relevant papers", ""]
            for idx, item in enumerate(results, 1):
                lines.extend([
                    f"[{idx}] {item.get('title') or 'Unknown title'}",
                    f"Relevance: {item.get('relevance_score')}",
                    f"File path: {item.get('pdf_path') or item.get('file_path') or ''}",
                    str(item.get('summary') or ''),
                    "",
                ])
            return "\n".join(lines)

    def _run_clean_subprocess_tool(
        self,
        tool_name: str,
        payload: Dict[str, Any],
        timeout: float,
    ) -> Dict[str, Any]:
        env = os.environ.copy()
        if sys.platform == "darwin":
            env.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        for key in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            env.setdefault(key, "1")
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "cytobridge_agent.runtime_v2.tool_subprocess_entry", tool_name],
                input=json.dumps(payload, ensure_ascii=False),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=max(1.0, float(timeout)),
                env=env,
                cwd=str(Path(__file__).resolve().parents[2]),
            )
        except subprocess.TimeoutExpired as exc:
            return {
                "ok": False,
                "error_type": "TimeoutError",
                "error": f"clean subprocess tool timed out after {timeout:.1f}s",
                "stderr": (exc.stderr or "")[:12000],
            }
        if proc.returncode != 0:
            return {
                "ok": False,
                "error_type": "ToolProcessCrashed",
                "error": f"clean subprocess exited with returncode={proc.returncode}",
                "returncode": proc.returncode,
                "stderr": (proc.stderr or "")[:12000],
                "stdout": (proc.stdout or "")[:12000],
            }
        try:
            return json.loads(proc.stdout or "{}")
        except json.JSONDecodeError as exc:
            return {
                "ok": False,
                "error_type": "JSONDecodeError",
                "error": str(exc),
                "stdout": (proc.stdout or "")[:12000],
                "stderr": (proc.stderr or "")[:12000],
            }

    def _render_clean_subprocess_error(self, tool_name: str, result: Dict[str, Any]) -> str:
        parts = [
            f"Error: tool '{tool_name}' failed in clean subprocess.",
            f"error_type: {result.get('error_type') or 'unknown'}",
            f"error: {result.get('error') or '(no error text returned)'}",
        ]
        if result.get("returncode") is not None:
            parts.append(f"returncode: {result.get('returncode')}")
        if result.get("stderr"):
            parts.append(f"stderr:\n{str(result.get('stderr'))[:12000]}")
        if result.get("traceback"):
            parts.append(f"Traceback:\n{str(result.get('traceback'))[:12000]}")
        return "\n".join(parts)

    def restore_from_snapshot(
        self,
        agent_snapshots: Optional[Dict[str, Any]],
        agent_histories: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not self.subagent_manager:
            return {"restored": {}, "degraded": False, "reasons": []}
        return self.subagent_manager.restore_from_snapshot(agent_snapshots, agent_histories=agent_histories)

    def collect_subagent_histories(self) -> Dict[str, List[Any]]:
        if not self.subagent_manager:
            return {}
        return self.subagent_manager.collect_histories()

    def collect_subagent_snapshots(self) -> Dict[str, Dict[str, Any]]:
        if not self.subagent_manager:
            return {}
        return self.subagent_manager.collect_snapshots()

    def peek_subagent_result(self, *, copy_result: bool = False) -> Optional[Dict[str, Any]]:
        if self._submitted_subagent_result is None:
            return None
        return dict(self._submitted_subagent_result) if copy_result else self._submitted_subagent_result

    def restore_subagent_result(self, payload: Optional[Dict[str, Any]]) -> None:
        self._submitted_subagent_result = dict(payload or {}) if payload else None

    @staticmethod
    def _render_structured_retry_evidence(
        result: Dict[str, Any],
        review_key: str,
        *,
        max_chars: int = 12000,
    ) -> str:
        proposed_updates = result.get("proposed_state_updates") or {}
        evidence = {
            "status": result.get("status"),
            "subagent_id": result.get("subagent_id"),
            "summary": result.get("summary"),
            "findings": result.get("findings") or [],
            "artifact_refs": result.get("artifact_refs") or [],
            "previous_structured_payload": proposed_updates.get(review_key),
            "needs_input_question": result.get("needs_input_question") or "",
        }
        try:
            text = json.dumps(evidence, ensure_ascii=False, indent=2, default=str)
        except Exception:
            text = str(evidence)
        if len(text) > max_chars:
            return text[:max_chars] + "\n... [truncated previous subagent result]"
        return text

    @staticmethod
    def _parse_json_object_fragment(raw_text: str) -> Dict[str, Any]:
        raw = str(raw_text or "").strip()
        start = raw.find("{")
        if start < 0:
            return {}
        try:
            parsed, _ = json.JSONDecoder().raw_decode(raw[start:])
        except Exception:
            return {}
        return dict(parsed) if isinstance(parsed, dict) else {}

    @staticmethod
    def _parse_review_value(value: str) -> Any:
        raw = str(value or "").strip()
        if not raw:
            return ""
        if raw[:1] in {"[", "{"}:
            try:
                return json.loads(raw)
            except Exception:
                return raw
        return raw

    @staticmethod
    def _extract_flat_review_fields(mapping: Dict[str, Any], review_key: str) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        prefixes = (
            f"{review_key}.",
            f"proposed_state_updates.{review_key}.",
        )
        for key, value in mapping.items():
            key_text = str(key or "").strip()
            for prefix in prefixes:
                if key_text.startswith(prefix):
                    field = key_text[len(prefix) :].strip()
                    if field:
                        payload[field] = value
                    break
        return payload

    @classmethod
    def _extract_review_payload(cls, result: Dict[str, Any], review_key: str) -> Dict[str, Any]:
        proposed = result.get("proposed_state_updates")
        if isinstance(proposed, dict):
            nested = proposed.get(review_key)
            if isinstance(nested, dict):
                return dict(nested)
            flattened = cls._extract_flat_review_fields(proposed, review_key)
            if flattened:
                return flattened

        direct = result.get(review_key)
        if isinstance(direct, dict):
            return dict(direct)

        flattened = cls._extract_flat_review_fields(result, review_key)
        if flattened:
            return flattened

        payload: Dict[str, Any] = {}
        text_items = [str(item or "") for item in (result.get("findings") or [])]
        if result.get("summary"):
            text_items.append(str(result.get("summary") or ""))
        json_markers = (
            f"proposed_state_updates.{review_key}=",
            f"{review_key}=",
        )
        json_patterns = tuple(
            re.compile(rf"{re.escape(marker[:-1])}\s*=\s*")
            for marker in json_markers
        )
        field_markers = (
            f"proposed_state_updates.{review_key}.",
            f"{review_key}.",
        )
        for text in text_items:
            stripped = text.strip()
            for marker in json_markers:
                marker_index = stripped.find(marker)
                if marker_index >= 0:
                    candidate = cls._parse_json_object_fragment(stripped[marker_index + len(marker) :])
                    if candidate:
                        payload.update(candidate)
            for pattern in json_patterns:
                match = pattern.search(stripped)
                if match:
                    candidate = cls._parse_json_object_fragment(stripped[match.end() :])
                    if candidate:
                        payload.update(candidate)
            for line in stripped.splitlines() or [stripped]:
                candidate_line = line.strip()
                for marker in field_markers:
                    if not candidate_line.startswith(marker) or "=" not in candidate_line:
                        continue
                    field, value = candidate_line[len(marker) :].split("=", 1)
                    field = field.strip()
                    if field:
                        payload[field] = cls._parse_review_value(value)
                    break
        return payload

    def _is_tool_allowed(self, tool_name: str) -> bool:
        name = str(tool_name or "").strip()
        if not name:
            return False
        allowed = {
            str(item).strip()
            for item in (self.tool_policy.get("allowed_tools") or [])
            if str(item).strip()
        }
        disallowed = {
            str(item).strip()
            for item in (self.tool_policy.get("disallowed_tools") or [])
            if str(item).strip()
        }
        if allowed and name not in allowed:
            return False
        return name not in disallowed

    @staticmethod
    def _is_subagent_infrastructure_failure(result: Dict[str, Any]) -> bool:
        if str(result.get("status") or "").strip().lower() != "failed":
            return False
        text = "\n".join(
            [
                str(result.get("summary") or ""),
                "\n".join(str(item or "") for item in (result.get("findings") or [])),
            ]
        )
        return is_retryable_subagent_infrastructure_error(text)

    def spawn_subagent(
        self,
        task: str,
        subagent_type: str = "general",
        success_criteria: Optional[List[str]] = None,
        context_notes: str = "",
        relevant_paths: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        if self.agent_role != "planner" or self.subagent_manager is None:
            raise RuntimeError("spawn_subagent(...) is only available to the planner runtime.")
        resolved_type = str(subagent_type or "general").strip().lower() or "general"
        result = self.subagent_manager.run_sync(
            subagent_type=str(subagent_type or "general").strip().lower() or "general",
            task=str(task or "").strip(),
            success_criteria=list(success_criteria or []),
            context_notes=str(context_notes or "").strip(),
            relevant_paths=list(relevant_paths or []),
        )
        if resolve_subagent_type(resolved_type).name == "proposal_evaluator":
            errors = self._proposal_review_contract_errors(result)
            if errors and not self._is_subagent_infrastructure_failure(result):
                self._emit_event(
                    "structured_subagent_result_retry",
                    {
                        "subagent_type": "proposal_evaluator",
                        "review_key": "proposal_review",
                        "subagent_id": str(result.get("subagent_id") or ""),
                        "errors": list(errors),
                    },
                )
                retry_context = (
                    f"{str(context_notes or '').strip()}\n\n"
                    "STRUCTURED OUTPUT CONTRACT VIOLATION:\n"
                    + "\n".join(f"- {error}" for error in errors)
                    + "\n\n"
                    "Repair the proposal review output now. Reuse the previous subagent analysis below; "
                    "do not redo the full proposal review unless that analysis is insufficient. "
                    "Finish by calling submit_proposal_review(...) with "
                    "decision, summary, reviewer_feedback, implementation_risks sorted by severity, "
                    "risk_assessment, and confidence. Do not return free text only.\n\n"
                    "Previous invalid subagent result to reuse:\n"
                    f"{self._render_structured_retry_evidence(result, 'proposal_review')}"
                )
                result = self.subagent_manager.run_sync(
                    subagent_type="proposal_evaluator",
                    task=(
                        "Repair the previous structured output for this already-run proposal review. "
                        "Use the prior result context; only inspect files again if needed to fill missing required fields.\n\n"
                        f"Original task:\n{str(task or '').strip()}"
                    ),
                    success_criteria=list(success_criteria or [])
                    + [
                        "Repair the previous structured-output contract violation before returning.",
                        "Reuse the prior review analysis instead of restarting from scratch when it is sufficient.",
                    ],
                    context_notes=retry_context,
                    relevant_paths=list(relevant_paths or []),
                )
        resolved_name = resolve_subagent_type(resolved_type).name
        if resolved_name == "proposal_evaluator":
            return self._maybe_apply_spawned_proposal_review(
                result,
                subagent_type=resolved_type,
                task=str(task or ""),
                relevant_paths=list(relevant_paths or []),
            )
        if resolved_name == "implementation_evaluator":
            errors = self._implementation_review_contract_errors(result)
            if errors and not self._is_subagent_infrastructure_failure(result):
                self._emit_event(
                    "structured_subagent_result_retry",
                    {
                        "subagent_type": "implementation_evaluator",
                        "review_key": "implementation_review",
                        "subagent_id": str(result.get("subagent_id") or ""),
                        "errors": list(errors),
                    },
                )
                retry_context = (
                    f"{str(context_notes or '').strip()}\n\n"
                    "STRUCTURED OUTPUT CONTRACT VIOLATION:\n"
                    + "\n".join(f"- {error}" for error in errors)
                    + "\n\n"
                    "Repair the implementation review output now. Reuse the previous subagent analysis below; "
                    "do not redo the full implementation review unless that analysis is insufficient. "
                    "Finish by calling submit_implementation_review(...) with decision, summary, reviewer_feedback, "
                    "findings, blocking_issues, advisory_risks, efficiency_recommendations, "
                    "generalization_shortcut_risks, and optional risk_assessment. Use empty lists when a category has no issues.\n\n"
                    "Previous invalid subagent result to reuse:\n"
                    f"{self._render_structured_retry_evidence(result, 'implementation_review')}"
                )
                result = self.subagent_manager.run_sync(
                    subagent_type="implementation_evaluator",
                    task=(
                        "Repair the previous structured output for this already-run implementation review. "
                        "Use the prior result context; only inspect files again if needed to fill missing required fields.\n\n"
                        f"Original task:\n{str(task or '').strip()}"
                    ),
                    success_criteria=list(success_criteria or [])
                    + [
                        "Repair the previous structured-output contract violation before returning.",
                        "Reuse the prior review analysis instead of restarting from scratch when it is sufficient.",
                    ],
                    context_notes=retry_context,
                    relevant_paths=list(relevant_paths or []),
                )
            return self._maybe_apply_spawned_implementation_review(
                result,
                relevant_paths=list(relevant_paths or []),
            )
        if resolved_name == "inference_evaluator":
            errors = self._inference_review_contract_errors(result)
            if errors and not self._is_subagent_infrastructure_failure(result):
                self._emit_event(
                    "structured_subagent_result_retry",
                    {
                        "subagent_type": "inference_evaluator",
                        "review_key": "inference_review",
                        "subagent_id": str(result.get("subagent_id") or ""),
                        "errors": list(errors),
                    },
                )
                retry_context = (
                    f"{str(context_notes or '').strip()}\n\n"
                    "STRUCTURED OUTPUT CONTRACT VIOLATION:\n"
                    + "\n".join(f"- {error}" for error in errors)
                    + "\n\n"
                    "Repair the inference review output now. Reuse the previous subagent analysis below; "
                    "do not redo the full inference review unless that analysis is insufficient. "
                    "Finish by calling submit_subagent_result(...) with proposed_state_updates.inference_review "
                    "containing decision, reviewer_feedback, blocking_issues, and advisory_risks. "
                    "Use empty lists when a category has no issues.\n\n"
                    "Previous invalid subagent result to reuse:\n"
                    f"{self._render_structured_retry_evidence(result, 'inference_review')}"
                )
                result = self.subagent_manager.run_sync(
                    subagent_type="inference_evaluator",
                    task=(
                        "Repair the previous structured output for this already-run inference review. "
                        "Use the prior result context; only inspect files again if needed to fill missing required fields.\n\n"
                        f"Original task:\n{str(task or '').strip()}"
                    ),
                    success_criteria=list(success_criteria or [])
                    + [
                        "Repair the previous structured-output contract violation before returning.",
                        "Reuse the prior review analysis instead of restarting from scratch when it is sufficient.",
                    ],
                    context_notes=retry_context,
                    relevant_paths=list(relevant_paths or []),
                )
            return self._maybe_apply_spawned_inference_review(
                result,
                relevant_paths=list(relevant_paths or []),
            )
        return result

    @staticmethod
    def _proposal_review_contract_errors(result: Dict[str, Any]) -> List[str]:
        payload = SingleAgentTools._extract_review_payload(result, "proposal_review")
        errors: List[str] = []
        if str(result.get("status") or "").strip().lower() != "completed":
            errors.append(f"subagent status is {result.get('status') or 'missing'}, expected completed")
        if str(payload.get("decision") or "").strip().lower() not in {"approve", "revise", "reject"}:
            errors.append("missing or invalid decision; expected approve|revise|reject")
        if not str(payload.get("reviewer_feedback") or "").strip():
            errors.append("missing or empty `reviewer_feedback`")
        risks = [str(item).strip() for item in (payload.get("implementation_risks") or []) if str(item).strip()]
        if not risks:
            errors.append("missing or empty `implementation_risks`")
        if not str(payload.get("risk_assessment") or "").strip():
            errors.append("missing or empty `risk_assessment`")
        return errors

    @staticmethod
    def _implementation_review_contract_errors(result: Dict[str, Any]) -> List[str]:
        payload = SingleAgentTools._extract_review_payload(result, "implementation_review")
        errors: List[str] = []
        if str(result.get("status") or "").strip().lower() != "completed":
            errors.append(f"subagent status is {result.get('status') or 'missing'}, expected completed")
        if str(payload.get("decision") or "").strip().lower() not in {"approve", "revise", "reject"}:
            errors.append("missing or invalid decision; expected approve|revise|reject")
        if not str(payload.get("reviewer_feedback") or "").strip():
            errors.append("missing or empty `reviewer_feedback`")
        for field_name in (
            "blocking_issues",
            "advisory_risks",
            "efficiency_recommendations",
            "generalization_shortcut_risks",
        ):
            if field_name not in payload:
                errors.append(f"missing `{field_name}` list; use [] when there are no items")
            elif payload.get(field_name) is not None and not isinstance(payload.get(field_name), list):
                errors.append(f"`{field_name}` must be a list")
        return errors

    @staticmethod
    def _inference_review_contract_errors(result: Dict[str, Any]) -> List[str]:
        payload = SingleAgentTools._extract_review_payload(result, "inference_review")
        errors: List[str] = []
        if str(result.get("status") or "").strip().lower() != "completed":
            errors.append(f"subagent status is {result.get('status') or 'missing'}, expected completed")
        if str(payload.get("decision") or "").strip().lower() not in {"approve", "revise", "reject"}:
            errors.append("missing or invalid decision; expected approve|revise|reject")
        if not str(payload.get("reviewer_feedback") or "").strip():
            errors.append("missing or empty `reviewer_feedback`")
        for field_name in ("blocking_issues", "advisory_risks"):
            if field_name not in payload:
                errors.append(f"missing `{field_name}` list; use [] when there are no items")
            elif payload.get(field_name) is not None and not isinstance(payload.get(field_name), list):
                errors.append(f"`{field_name}` must be a list")
        return errors

    def _proposal_review_target_from_paths(self, relevant_paths: List[str]) -> Dict[str, str]:
        for raw_path in relevant_paths:
            text = str(raw_path or "").strip()
            if not text:
                continue
            path = Path(text)
            parts = list(path.parts)
            algorithm_id = ""
            for idx, part in enumerate(parts):
                if part == "training_algorithms" and idx + 1 < len(parts):
                    algorithm_id = str(parts[idx + 1]).strip().lower()
                    break
            proposal_id = ""
            if path.name.startswith("proposal_"):
                proposal_id = path.stem.replace(".risk", "")
            if path.exists() and path.suffix == ".json":
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    payload = {}
                if isinstance(payload, dict):
                    proposal_id = str(payload.get("proposal_id") or proposal_id).strip()
                    algorithm_id = str(payload.get("algorithm_id") or algorithm_id).strip().lower()
            if algorithm_id and not proposal_id:
                record = self.planner_file_tools._get_proposal_record(algorithm_id)
                if isinstance(record, dict):
                    proposal_id = str(record.get("proposal_id") or "").strip()
            if algorithm_id:
                return {"algorithm_id": algorithm_id, "proposal_id": proposal_id}
        return {}

    def _resolve_spawned_proposal_review_target(self, relevant_paths: List[str]) -> Dict[str, str]:
        target = self._proposal_review_target_from_paths(relevant_paths)
        if target.get("algorithm_id"):
            return target
        active_context = self.planner_file_tools.get_active_algorithm_context()
        active_algorithm = str(active_context.get("algorithm_id") or "").strip().lower()
        active_proposal = str(active_context.get("proposal_id") or "").strip()
        if active_algorithm:
            return {"algorithm_id": active_algorithm, "proposal_id": active_proposal}
        latest_algorithm = str(self.state.get("latest_algorithm_proposal_id") or "").strip().lower()
        latest_proposal = str(self.state.get("active_proposal_id") or "").strip()
        if latest_algorithm:
            return {"algorithm_id": latest_algorithm, "proposal_id": latest_proposal}
        return {}

    def _maybe_apply_spawned_proposal_review(
        self,
        result: Dict[str, Any],
        *,
        subagent_type: str,
        task: str,
        relevant_paths: List[str],
    ) -> Dict[str, Any]:
        if resolve_subagent_type(subagent_type).name != "proposal_evaluator":
            return result
        contract_errors = self._proposal_review_contract_errors(result)
        if contract_errors:
            applied_updates = dict(result.get("applied_state_updates") or {})
            infra_failure = self._is_subagent_infrastructure_failure(result)
            applied_updates["proposal_review"] = {
                "applied": False,
                "decision": "",
                "errors": list(contract_errors),
                "error": (
                    "proposal_evaluator failed because reviewer infrastructure did not produce a structured verdict; "
                    "review_algorithm_proposal(...) was not called."
                    if infra_failure
                    else "proposal_evaluator result did not satisfy the structured review contract after retry; "
                    "review_algorithm_proposal(...) was not called."
                ),
            }
            result["applied_state_updates"] = applied_updates
            self._emit_event(
                "structured_subagent_result_invalid",
                {
                    "subagent_type": "proposal_evaluator",
                    "review_key": "proposal_review",
                    "subagent_id": str(result.get("subagent_id") or ""),
                    "errors": list(contract_errors),
                },
            )
            return result
        review_payload = self._extract_review_payload(result, "proposal_review")
        decision = str(review_payload.get("decision") or "").strip().lower()
        feedback = str(review_payload.get("reviewer_feedback") or "").strip()
        if result.get("status") != "completed" or decision not in {"approve", "revise", "reject"} or not feedback:
            return result
        target = self._resolve_spawned_proposal_review_target(relevant_paths)
        algorithm_id = str(target.get("algorithm_id") or "").strip().lower()
        proposal_id = str(target.get("proposal_id") or "").strip()
        applied_updates = dict(result.get("applied_state_updates") or {})
        review_application: Dict[str, Any] = {
            "applied": False,
            "algorithm_id": algorithm_id,
            "proposal_id": proposal_id,
            "decision": decision,
        }
        if not algorithm_id:
            review_application["error"] = (
                "Could not resolve target algorithm for proposal_evaluator result. "
                "Call review_algorithm_proposal(...) explicitly with algorithm_id and proposal_id."
            )
            applied_updates["proposal_review"] = review_application
            result["applied_state_updates"] = applied_updates
            return result
        apply_result = self.review_algorithm_proposal(
            algorithm_id=algorithm_id,
            proposal_id=proposal_id,
            decision=decision,
            reviewer_feedback=feedback,
            implementation_risks=[
                str(item).strip()
                for item in (review_payload.get("implementation_risks") or [])
                if str(item).strip()
            ],
            risk_assessment=str(review_payload.get("risk_assessment") or "").strip(),
        )
        if isinstance(apply_result, str) and apply_result.startswith("Error:"):
            review_application["error"] = apply_result
        else:
            review_application["applied"] = True
            review_application["result"] = apply_result
            self._emit_event(
                "algorithm_proposal_spawn_review_applied",
                {
                    "algorithm_id": algorithm_id,
                    "proposal_id": proposal_id,
                    "decision": decision,
                    "subagent_id": str(result.get("subagent_id") or ""),
                },
            )
        applied_updates["proposal_review"] = review_application
        result["applied_state_updates"] = applied_updates
        return result

    def _maybe_apply_spawned_implementation_review(
        self,
        result: Dict[str, Any],
        *,
        relevant_paths: List[str],
    ) -> Dict[str, Any]:
        contract_errors = self._implementation_review_contract_errors(result)
        applied_updates = dict(result.get("applied_state_updates") or {})
        if contract_errors:
            infra_failure = self._is_subagent_infrastructure_failure(result)
            applied_updates["implementation_review"] = {
                "applied": False,
                "decision": "",
                "errors": list(contract_errors),
                "error": (
                    "implementation_evaluator failed because reviewer infrastructure did not produce a structured verdict; "
                    "implementation review was not recorded."
                    if infra_failure
                    else "implementation_evaluator result did not satisfy the structured review contract after retry; "
                    "implementation review was not recorded."
                ),
            }
            result["applied_state_updates"] = applied_updates
            self._emit_event(
                "structured_subagent_result_invalid",
                {
                    "subagent_type": "implementation_evaluator",
                    "review_key": "implementation_review",
                    "subagent_id": str(result.get("subagent_id") or ""),
                    "errors": list(contract_errors),
                },
            )
            return result

        target = self._resolve_spawned_proposal_review_target(relevant_paths)
        algorithm_id = str(target.get("algorithm_id") or "").strip().lower()
        proposal_id = str(target.get("proposal_id") or "").strip()
        review_payload = self._extract_review_payload(result, "implementation_review")
        decision = str(review_payload.get("decision") or "").strip().lower()
        review_application: Dict[str, Any] = {
            "applied": False,
            "algorithm_id": algorithm_id,
            "proposal_id": proposal_id,
            "decision": decision,
        }
        if not algorithm_id:
            review_application["error"] = (
                "Could not resolve target algorithm for implementation_evaluator result. "
                "Trigger the review from the training/campaign gate or include the algorithm workspace path in relevant_paths."
            )
            applied_updates["implementation_review"] = review_application
            result["applied_state_updates"] = applied_updates
            return result

        fingerprint = self._build_implementation_review_fingerprint(algorithm_id, proposal_id=proposal_id)
        if not bool(fingerprint.get("requires_review")):
            review_application["error"] = (
                "Could not compute an implementation review fingerprint for the target algorithm/proposal."
            )
            applied_updates["implementation_review"] = review_application
            result["applied_state_updates"] = applied_updates
            return result

        payload = dict(fingerprint.get("payload") or {})
        target_proposal_id = str((payload.get("proposal") or {}).get("proposal_id") or proposal_id).strip()
        findings = [str(item).strip() for item in (result.get("findings") or []) if str(item).strip()]
        try:
            record = self.planner_file_tools.record_implementation_review(
                algorithm_id,
                proposal_id=target_proposal_id,
                review_hash=str(fingerprint.get("review_hash") or ""),
                decision=decision,
                reviewer_feedback=str(review_payload.get("reviewer_feedback") or "").strip(),
                findings=findings,
                blocking_issues=[
                    str(item).strip()
                    for item in (review_payload.get("blocking_issues") or [])
                    if str(item).strip()
                ],
                advisory_risks=[
                    str(item).strip()
                    for item in (review_payload.get("advisory_risks") or [])
                    if str(item).strip()
                ],
                efficiency_recommendations=[
                    str(item).strip()
                    for item in (review_payload.get("efficiency_recommendations") or [])
                    if str(item).strip()
                ],
                generalization_shortcut_risks=[
                    str(item).strip()
                    for item in (review_payload.get("generalization_shortcut_risks") or [])
                    if str(item).strip()
                ],
                risk_assessment=str(review_payload.get("risk_assessment") or "").strip(),
                subagent_id=str(result.get("subagent_id") or ""),
                reviewed_paths=list(fingerprint.get("source_paths") or relevant_paths or []),
                review_summary=str(result.get("summary") or "").strip(),
                review_hash_scope=str(fingerprint.get("review_hash_scope") or "proposal_semantics"),
                implementation_review_policy_version=int(
                    (payload.get("implementation_review_policy_version") or 0)
                ),
            )
        except Exception as exc:  # noqa: BLE001
            review_application["error"] = f"Failed to record implementation review: {exc}"
            applied_updates["implementation_review"] = review_application
            result["applied_state_updates"] = applied_updates
            return result

        review_application["applied"] = True
        review_application["proposal_id"] = target_proposal_id
        review_application["review_hash"] = str(record.get("review_hash") or "")
        review_application["status"] = str(record.get("status") or "")
        review_application["result"] = record
        applied_updates["implementation_review"] = review_application
        result["applied_state_updates"] = applied_updates
        self._emit_event(
            "algorithm_implementation_spawn_review_applied",
            {
                "algorithm_id": algorithm_id,
                "proposal_id": target_proposal_id,
                "decision": decision,
                "status": str(record.get("status") or ""),
                "review_hash": str(record.get("review_hash") or ""),
                "subagent_id": str(result.get("subagent_id") or ""),
            },
        )
        return result

    def _spawned_inference_review_fingerprint(
        self,
        algorithm_id: str,
        *,
        proposal_id: str = "",
    ) -> Dict[str, Any]:
        from ..tools.training_tools import (
            get_cellcompass_root,
            get_workspace_root,
            materialize_training_config,
            resolve_training_target,
        )

        algo_id = str(algorithm_id or "").strip().lower()
        if not algo_id:
            return {"ok": False, "error": "algorithm_id is required"}
        output_dir = Path(str(self.state.get("output_dir") or "cytobridge_output")).expanduser().resolve()
        data_path = str(self.state.get("preprocessed_path") or self.state.get("input_path") or "").strip()
        purpose = "training"
        training_stage = "pilot"
        config_overrides: Dict[str, Any] = {}
        campaign_id = str(self.state.get("active_algorithm_campaign_id") or "").strip()
        campaign: Dict[str, Any] = {}
        if campaign_id:
            try:
                maybe_campaign = self.planner_file_tools.get_algorithm_campaign_status(campaign_id)
            except Exception:
                maybe_campaign = {}
            if isinstance(maybe_campaign, dict) and str(maybe_campaign.get("algorithm_id") or "").strip().lower() == algo_id:
                campaign = maybe_campaign
                current_stage = str(campaign.get("current_stage") or "stage1_feasibility")
                policy = dict((campaign.get("stage_policies") or {}).get(current_stage) or {})
                training_stage = str(policy.get("training_stage") or training_stage)
                purpose = "campaign"
                try:
                    dataset_payload = self.planner_file_tools.resolve_campaign_trial_dataset_payload(campaign_id, {})
                except Exception:
                    dataset_payload = {}
                first_dataset: Dict[str, Any] = {}
                raw_datasets = dataset_payload.get("datasets") if isinstance(dataset_payload, dict) else None
                if isinstance(raw_datasets, list) and raw_datasets:
                    first_dataset = dict(raw_datasets[0] or {})
                common_overrides = (
                    dataset_payload.get("config_overrides")
                    if isinstance(dataset_payload, dict) and isinstance(dataset_payload.get("config_overrides"), dict)
                    else {}
                )
                entry_overrides = (
                    first_dataset.get("config_overrides")
                    if isinstance(first_dataset.get("config_overrides"), dict)
                    else first_dataset
                )
                config_overrides = deepcopy(common_overrides or {})
                for key, value in dict(entry_overrides or {}).items():
                    if key in {
                        "dataset_id",
                        "id",
                        "adata_path",
                        "target_dataset_ids",
                        "dataset_ids",
                        "stage_panel_dataset_ids",
                        "config_overrides",
                        "common_config_overrides",
                        "per_dataset_config_overrides",
                        "simulation_version",
                        "simulation_generator",
                    }:
                        continue
                    config_overrides[key] = deepcopy(value)
                data_path = str(first_dataset.get("adata_path") or data_path).strip()
        training_target = resolve_training_target(
            candidate=None,
            training_algorithm_id=algo_id,
            input_adata_path=data_path,
            output_dir=str(output_dir),
            stage=training_stage,
            metadata={
                "planner_state_keys": sorted(self.state.keys()),
                "purpose": "manual_spawn_inference_review",
                "campaign_id": campaign_id,
            },
            search_roots=[get_cellcompass_root() / "training_algorithms"],
            workspace_root=get_workspace_root(),
            config_overrides=config_overrides,
        )
        review_config = materialize_training_config(
            training_target,
            stage=training_stage,
            checkpoints_dir=output_dir / ".runtime" / "inference_review_preflight",
        )
        fingerprint = self._build_inference_review_fingerprint(
            training_target,
            resolved_config=review_config,
            purpose=purpose,
        )
        source_paths = list(fingerprint.get("source_paths") or [])
        workspace_path = str(self.planner_file_tools._proposal_dir(algo_id))
        for path in [
            workspace_path,
            str(self.planner_file_tools._proposal_dir(algo_id) / "PROPOSAL.md"),
            str(Path(__file__).resolve().parents[2] / "CytoBridge-main" / "CytoBridge" / "tl" / "trainer.py"),
            str(Path(__file__).resolve().parents[2] / "CytoBridge-main" / "CytoBridge" / "tl" / "analysis.py"),
            str(Path(__file__).resolve().parents[2] / "CytoBridge-main" / "CytoBridge" / "tl" / "training_algorithm.py"),
        ]:
            if path and path not in source_paths:
                source_paths.append(path)
        return {
            "ok": True,
            "algorithm_id": algo_id,
            "proposal_id": str(proposal_id or "").strip(),
            "campaign_id": campaign_id if campaign else "",
            "purpose": purpose,
            "training_stage": training_stage,
            "fingerprint": fingerprint,
            "source_paths": source_paths,
        }

    def _maybe_apply_spawned_inference_review(
        self,
        result: Dict[str, Any],
        *,
        relevant_paths: List[str],
    ) -> Dict[str, Any]:
        contract_errors = self._inference_review_contract_errors(result)
        applied_updates = dict(result.get("applied_state_updates") or {})
        if contract_errors:
            infra_failure = self._is_subagent_infrastructure_failure(result)
            applied_updates["inference_review"] = {
                "applied": False,
                "decision": "",
                "errors": list(contract_errors),
                "error": (
                    "inference_evaluator failed because reviewer infrastructure did not produce a structured verdict; "
                    "inference review was not recorded."
                    if infra_failure
                    else "inference_evaluator result did not satisfy the structured review contract after retry; "
                    "inference review was not recorded."
                ),
            }
            result["applied_state_updates"] = applied_updates
            self._emit_event(
                "structured_subagent_result_invalid",
                {
                    "subagent_type": "inference_evaluator",
                    "review_key": "inference_review",
                    "subagent_id": str(result.get("subagent_id") or ""),
                    "errors": list(contract_errors),
                },
            )
            return result

        target = self._resolve_spawned_proposal_review_target(relevant_paths)
        algorithm_id = str(target.get("algorithm_id") or "").strip().lower()
        proposal_id = str(target.get("proposal_id") or "").strip()
        review_payload = self._extract_review_payload(result, "inference_review")
        decision = str(review_payload.get("decision") or "").strip().lower()
        review_application: Dict[str, Any] = {
            "applied": False,
            "algorithm_id": algorithm_id,
            "proposal_id": proposal_id,
            "decision": decision,
        }
        if not algorithm_id:
            review_application["error"] = (
                "Could not resolve target algorithm for inference_evaluator result. "
                "Trigger the review from the training/campaign gate or include the algorithm workspace path in relevant_paths."
            )
            applied_updates["inference_review"] = review_application
            result["applied_state_updates"] = applied_updates
            return result
        try:
            context = self._spawned_inference_review_fingerprint(algorithm_id, proposal_id=proposal_id)
        except Exception as exc:  # noqa: BLE001
            review_application["error"] = f"Could not compute inference review fingerprint: {exc}"
            applied_updates["inference_review"] = review_application
            result["applied_state_updates"] = applied_updates
            return result
        if not bool(context.get("ok")):
            review_application["error"] = str(context.get("error") or "Could not compute inference review fingerprint.")
            applied_updates["inference_review"] = review_application
            result["applied_state_updates"] = applied_updates
            return result
        fingerprint = dict(context.get("fingerprint") or {})
        if not bool(fingerprint.get("requires_review")):
            review_application["error"] = (
                "The current algorithm/config has no custom inference surface requiring inference review."
            )
            review_application["required"] = False
            applied_updates["inference_review"] = review_application
            result["applied_state_updates"] = applied_updates
            return result
        try:
            record = self.planner_file_tools.record_inference_review(
                algorithm_id,
                review_hash=str(fingerprint.get("review_hash") or ""),
                decision=decision,
                reviewer_feedback=str(review_payload.get("reviewer_feedback") or "").strip(),
                findings=[str(item).strip() for item in (result.get("findings") or []) if str(item).strip()],
                blocking_issues=[
                    str(item).strip()
                    for item in (review_payload.get("blocking_issues") or [])
                    if str(item).strip()
                ],
                advisory_risks=[
                    str(item).strip()
                    for item in (review_payload.get("advisory_risks") or [])
                    if str(item).strip()
                ],
                subagent_id=str(result.get("subagent_id") or ""),
                reviewed_paths=list(context.get("source_paths") or relevant_paths or []),
                review_summary=str(result.get("summary") or "").strip(),
            )
        except Exception as exc:  # noqa: BLE001
            review_application["error"] = f"Failed to record inference review: {exc}"
            applied_updates["inference_review"] = review_application
            result["applied_state_updates"] = applied_updates
            return result
        review_application["applied"] = True
        review_application["required"] = True
        review_application["review_hash"] = str(record.get("review_hash") or "")
        review_application["status"] = str(record.get("status") or "")
        review_application["purpose"] = str(context.get("purpose") or "")
        review_application["campaign_id"] = str(context.get("campaign_id") or "")
        review_application["result"] = record
        applied_updates["inference_review"] = review_application
        result["applied_state_updates"] = applied_updates
        self._emit_event(
            "algorithm_inference_spawn_review_applied",
            {
                "algorithm_id": algorithm_id,
                "proposal_id": proposal_id,
                "decision": decision,
                "status": str(record.get("status") or ""),
                "review_hash": str(record.get("review_hash") or ""),
                "purpose": str(context.get("purpose") or ""),
                "campaign_id": str(context.get("campaign_id") or ""),
                "subagent_id": str(result.get("subagent_id") or ""),
            },
        )
        return result

    def submit_subagent_result(
        self,
        status: str,
        summary: str,
        findings: Optional[List[str]] = None,
        artifact_refs: Optional[List[Dict[str, Any]]] = None,
        proposed_state_updates: Optional[Dict[str, Any]] = None,
        needs_input_question: str = "",
    ) -> Dict[str, Any]:
        if self.agent_role != "subagent":
            raise RuntimeError("submit_subagent_result(...) is only available inside subagent runtimes.")

        normalized_status = str(status or "").strip().lower()
        payload = {
            "status": normalized_status,
            "summary": str(summary or "").strip(),
            "findings": [str(item).strip() for item in (findings or []) if str(item).strip()],
            "artifact_refs": [dict(item or {}) for item in (artifact_refs or []) if isinstance(item, dict)],
            "proposed_state_updates": dict(proposed_state_updates or {}),
            "needs_input_question": str(needs_input_question or "").strip(),
        }
        self._submitted_subagent_result = payload
        self._emit_event(
            "subagent_result_submitted",
            {
                "subagent_id": self.agent_id,
                "status": normalized_status,
                "summary": payload["summary"],
            },
        )
        return {"accepted": True, "status": normalized_status}

    def submit_proposal_review(
        self,
        decision: str,
        summary: str,
        reviewer_feedback: str,
        findings: Optional[List[str]] = None,
        implementation_risks: Optional[List[str]] = None,
        risk_assessment: str = "",
        confidence: float = 0.0,
    ) -> Dict[str, Any]:
        if self.agent_role != "subagent":
            raise RuntimeError("submit_proposal_review(...) is only available inside evaluator subagent runtimes.")

        if self.subagent_type != "proposal_evaluator":
            raise RuntimeError("submit_proposal_review(...) is only available to proposal_evaluator subagents.")

        choice = str(decision or "").strip().lower()
        if choice not in {"approve", "revise", "reject"}:
            raise ValueError("decision must be one of approve | revise | reject")

        try:
            confidence_value = float(confidence)
        except Exception:
            confidence_value = 0.0
        confidence_value = max(0.0, min(1.0, confidence_value))

        payload = {
            "status": "completed",
            "summary": str(summary or "").strip(),
            "findings": [str(item).strip() for item in (findings or []) if str(item).strip()],
            "artifact_refs": [],
            "proposed_state_updates": {
                "proposal_review": {
                    "decision": choice,
                    "reviewer_feedback": str(reviewer_feedback or "").strip(),
                    "implementation_risks": [
                        str(item).strip()
                        for item in (implementation_risks or [])
                        if str(item).strip()
                    ],
                    "risk_assessment": str(risk_assessment or "").strip(),
                    "confidence": confidence_value,
                }
            },
            "needs_input_question": "",
        }
        self._submitted_subagent_result = payload
        self._emit_event(
            "proposal_review_submitted",
            {
                "subagent_id": self.agent_id,
                "decision": choice,
                "summary": payload["summary"],
                "implementation_risks": list(
                    ((payload.get("proposed_state_updates") or {}).get("proposal_review") or {}).get("implementation_risks")
                    or []
                ),
                "confidence": confidence_value,
            },
        )
        return {"accepted": True, "decision": choice, "confidence": confidence_value}

    def submit_implementation_review(
        self,
        decision: str,
        summary: str,
        reviewer_feedback: str,
        findings: Optional[List[str]] = None,
        blocking_issues: Optional[List[str]] = None,
        advisory_risks: Optional[List[str]] = None,
        efficiency_recommendations: Optional[List[str]] = None,
        generalization_shortcut_risks: Optional[List[str]] = None,
        risk_assessment: str = "",
    ) -> Dict[str, Any]:
        if self.agent_role != "subagent":
            raise RuntimeError("submit_implementation_review(...) is only available inside evaluator subagent runtimes.")

        if self.subagent_type != "implementation_evaluator":
            raise RuntimeError("submit_implementation_review(...) is only available to implementation_evaluator subagents.")

        choice = str(decision or "").strip().lower()
        if choice not in {"approve", "revise", "reject"}:
            raise ValueError("decision must be one of approve | revise | reject")

        payload = {
            "status": "completed",
            "summary": str(summary or "").strip(),
            "findings": [str(item).strip() for item in (findings or []) if str(item).strip()],
            "artifact_refs": [],
            "proposed_state_updates": {
                "implementation_review": {
                    "decision": choice,
                    "reviewer_feedback": str(reviewer_feedback or "").strip(),
                    "blocking_issues": [
                        str(item).strip()
                        for item in (blocking_issues or [])
                        if str(item).strip()
                    ],
                    "advisory_risks": [
                        str(item).strip()
                        for item in (advisory_risks or [])
                        if str(item).strip()
                    ],
                    "efficiency_recommendations": [
                        str(item).strip()
                        for item in (efficiency_recommendations or [])
                        if str(item).strip()
                    ],
                    "generalization_shortcut_risks": [
                        str(item).strip()
                        for item in (generalization_shortcut_risks or [])
                        if str(item).strip()
                    ],
                    "risk_assessment": str(risk_assessment or "").strip(),
                }
            },
            "needs_input_question": "",
        }
        self._submitted_subagent_result = payload
        self._emit_event(
            "implementation_review_submitted",
            {
                "subagent_id": self.agent_id,
                "decision": choice,
                "summary": payload["summary"],
                "blocking_issues": list(
                    ((payload.get("proposed_state_updates") or {}).get("implementation_review") or {}).get("blocking_issues")
                    or []
                ),
            },
        )
        return {"accepted": True, "decision": choice}

    def submit_research_idea_review(
        self,
        decision: str,
        summary: str,
        reviewer_feedback: str,
        findings: Optional[List[str]] = None,
        confidence: float = 0.0,
    ) -> Dict[str, Any]:
        if self.agent_role != "subagent":
            raise RuntimeError("submit_research_idea_review(...) is only available inside evaluator subagent runtimes.")

        if self.subagent_type != "idea_evaluator":
            raise RuntimeError("submit_research_idea_review(...) is only available to idea_evaluator subagents.")

        choice = str(decision or "").strip().lower()
        if choice not in {"approve", "revise", "reject"}:
            raise ValueError("decision must be one of approve | revise | reject")

        try:
            confidence_value = float(confidence)
        except Exception:
            confidence_value = 0.0
        confidence_value = max(0.0, min(1.0, confidence_value))

        payload = {
            "status": "completed",
            "summary": str(summary or "").strip(),
            "findings": [str(item).strip() for item in (findings or []) if str(item).strip()],
            "artifact_refs": [],
            "proposed_state_updates": {
                "research_idea_review": {
                    "decision": choice,
                    "reviewer_feedback": str(reviewer_feedback or "").strip(),
                    "confidence": confidence_value,
                }
            },
            "needs_input_question": "",
        }
        self._submitted_subagent_result = payload
        self._emit_event(
            "research_idea_review_submitted",
            {
                "subagent_id": self.agent_id,
                "decision": choice,
                "summary": payload["summary"],
                "confidence": confidence_value,
            },
        )
        return {"accepted": True, "decision": choice, "confidence": confidence_value}

    def inspect_adata_state(self) -> str:
        manager = AnnDataManager()
        path = manager.get_path()
        payload = {
            "loaded_in_main_process": False,
            "path": path,
            "path_bound_for_isolated_tools": bool(path),
            "message": (
                "The runtime no longer exposes an in-process AnnData object. "
                "Use the bound path with isolated tools such as inspect_h5ad_contract, "
                "execute_python, run_saved_python_script, training, or downstream workers."
            ),
        }
        self._emit_event("file_activity", {"scope": "runtime", "action": "inspect_adata_state", "path": manager.get_path()})
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def inspect_h5ad_contract(
        self,
        file_path: str,
        time_key: str = "",
        label_keys: Optional[List[str]] = None,
        max_categories: int = 20,
    ) -> str:
        """Read-only AnnData contract summary for reviewers."""
        target = Path(str(file_path or "")).expanduser()
        if not target.exists():
            return json.dumps({"ok": False, "error": f"path does not exist: {target}"}, ensure_ascii=False, indent=2)
        if target.suffix.lower() != ".h5ad":
            return json.dumps({"ok": False, "error": f"expected .h5ad file, got: {target}"}, ensure_ascii=False, indent=2)
        payload = {
            "file_path": str(target),
            "time_key": time_key,
            "label_keys": list(label_keys or []),
            "max_categories": max_categories,
        }
        clean_result = self._run_clean_subprocess_tool("inspect_h5ad_contract", payload, timeout=240)
        if not clean_result.get("ok"):
            return self._render_clean_subprocess_error("inspect_h5ad_contract", clean_result)
        self._emit_event("file_activity", {"scope": "runtime", "action": "inspect_h5ad_contract", "path": str(target)})
        return str(clean_result.get("content") or "")
        max_items = max(3, min(int(max_categories or 20), 50))
        try:
            import anndata as ad  # type: ignore
        except Exception as exc:  # noqa: BLE001
            return json.dumps({"ok": False, "error": f"anndata import failed: {exc}"}, ensure_ascii=False, indent=2)

        adata = None
        try:
            adata = ad.read_h5ad(str(target), backed="r")
            obs = adata.obs.copy()
            obs_columns = list(map(str, obs.columns))
            requested_time = str(time_key or "").strip()
            state_time = str(self.state.get("time_key") or "").strip()
            common_time_keys = [
                requested_time,
                state_time,
                "time_point_processed",
                "time",
                "day",
                "stage",
                "Time point",
            ]
            selected_time_key = next((key for key in common_time_keys if key and key in obs.columns), "")
            common_label_keys = [
                "lineage",
                "clone",
                "barcode",
                "fate",
                "cell_type",
                "cell_type_broad",
                "label",
                "condition",
                "branch",
                "state",
            ]
            keys: List[str] = []
            for key in list(label_keys or []) + common_label_keys:
                key = str(key or "").strip()
                if key and key in obs.columns and key not in keys:
                    keys.append(key)
            if selected_time_key and selected_time_key not in keys:
                keys.insert(0, selected_time_key)

            def _value_counts(series: Any) -> Dict[str, int]:
                counts = series.astype(str).fillna("<NA>").value_counts(dropna=False).head(max_items)
                return {str(index): int(value) for index, value in counts.items()}

            obs_summary: Dict[str, Any] = {}
            for key in keys[:20]:
                series = obs[key]
                summary: Dict[str, Any] = {
                    "dtype": str(series.dtype),
                    "unique_count": int(series.astype(str).nunique(dropna=False)),
                    "top_counts": _value_counts(series),
                }
                if selected_time_key and key != selected_time_key:
                    try:
                        grouped = (
                            obs[[selected_time_key, key]]
                            .astype(str)
                            .groupby(selected_time_key)[key]
                            .nunique(dropna=False)
                            .head(max_items)
                        )
                        summary["unique_by_time"] = {str(index): int(value) for index, value in grouped.items()}
                    except Exception:
                        pass
                obs_summary[key] = summary

            time_counts = _value_counts(obs[selected_time_key]) if selected_time_key else {}
            payload = {
                "ok": True,
                "path": str(target.resolve()),
                "shape": [int(adata.n_obs), int(adata.n_vars)],
                "obs_columns": obs_columns[:120],
                "obs_columns_truncated": len(obs_columns) > 120,
                "selected_time_key": selected_time_key,
                "time_counts": time_counts,
                "summarized_obs_keys": keys[:20],
                "obs_summary": obs_summary,
                "obsm_keys": {str(key): list(map(int, adata.obsm[key].shape)) for key in list(adata.obsm.keys())[:30]},
                "layers": list(map(str, adata.layers.keys()))[:30],
                "uns_keys": list(map(str, adata.uns.keys()))[:50],
                "read_only": True,
            }
            self._emit_event("file_activity", {"scope": "runtime", "action": "inspect_h5ad_contract", "path": str(target)})
            return json.dumps(payload, ensure_ascii=False, indent=2)
        except Exception as exc:  # noqa: BLE001
            return json.dumps({"ok": False, "path": str(target), "error": str(exc)}, ensure_ascii=False, indent=2)
        finally:
            if adata is not None and getattr(adata, "isbacked", False):
                try:
                    adata.file.close()
                except Exception:
                    pass

    def load_or_switch_adata(self, path: str, force_reload: bool = False) -> str:
        target = Path(path).expanduser().resolve()
        if not target.exists():
            return f"Error: path does not exist: {target}"
        manager = AnnDataManager()
        manager.bind_path(str(target))
        self.state["input_path"] = str(target)
        self.state["pending_input_path"] = None
        self.state["data_switch_note"] = f"Switched active adata to {target}"
        self._emit_event("file_activity", {"scope": "runtime", "action": "bind_adata_path", "path": str(target), "force_reload": bool(force_reload)})
        return f"✅ Bound adata path without loading it in the main agent process: {target}"

    def persist_runtime_adata(self, save_path: str, sanitize_on_error: bool = True) -> str:
        manager = AnnDataManager()
        source_path = manager.get_path()
        if not source_path:
            return "Error: no adata path is bound"
        source = Path(source_path).expanduser().resolve()
        if not source.exists():
            return f"Error: bound adata path does not exist: {source}"
        target = Path(save_path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(source, target)
            manager.bind_path(str(target))
            saved = str(target)
        except Exception as exc:
            return f"Error: failed to persist bound adata file: {exc}"
        self._emit_event("file_activity", {"scope": "runtime", "action": "persist_runtime_adata", "path": saved})
        return f"✅ Saved runtime adata to {saved}"

    def search_literature(self, query: str) -> str:
        output_dir = str(self.state.get("output_dir") or "").strip() or None
        top_k = max(1, min(int(os.environ.get("CYTOBRIDGE_RAG_TOP_K", "5") or "5"), 12))
        queries, hyde_status = self._generate_parent_hyde_queries(query, max_expansions=2)
        payload = {
            "query": query,
            "queries": queries,
            "output_dir": output_dir,
            "top_k": max(top_k * 2, top_k + 3),
        }
        result = self._run_clean_subprocess_tool("search_literature", payload, timeout=180)
        if not result.get("ok"):
            return self._render_clean_subprocess_error("search_literature", result)
        candidates = list(result.get("candidates") or [])
        selected, rerank_status = self._rerank_literature_candidates(query, candidates, top_k=top_k)
        content = self._format_literature_results(selected) if selected else str(result.get("content") or "")
        trace = dict(result.get("trace") or {})
        trace.update(
            {
                "query": query,
                "hyde_expansions": queries,
                "parent_hyde_status": hyde_status,
                "parent_rerank_status": rerank_status,
                "selected_results": [
                    {
                        "title": item.get("title"),
                        "pdf_filename": item.get("pdf_filename"),
                        "relevance_score": item.get("relevance_score"),
                        "rerank_position": item.get("rerank_position"),
                        "pdf_path": item.get("pdf_path"),
                    }
                    for item in selected
                ],
                "formatted_context": content,
                "clean_subprocess_candidates": len(candidates),
            }
        )
        self._append_literature_trace("search_literature", trace)
        return content

    def search_bohrium_paper(
        self,
        query: str,
        words: Optional[List[str]] = None,
        page_size: int = 10,
        start_time: str = "",
        end_time: str = "",
        search_type: int = 5,
        jcr_zones: Optional[List[str]] = None,
        include_dbs: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        payload = search_bohrium_papers(
            query=query,
            words=words,
            page_size=page_size,
            start_time=start_time,
            end_time=end_time,
            search_type=search_type,
            jcr_zones=jcr_zones,
            include_dbs=include_dbs,
        )
        self._append_literature_trace(
            "search_bohrium_paper",
            {
                "query": query,
                "words": payload.get("words", []),
                "success": bool(payload.get("success")),
                "error": payload.get("error", ""),
                "selected_results": [
                    {
                        "title": item.get("title"),
                        "doi": item.get("doi"),
                        "paper_id": item.get("paper_id"),
                        "journal": item.get("journal"),
                        "published_at": item.get("published_at"),
                        "citation_count": item.get("citation_count"),
                    }
                    for item in payload.get("results", [])
                ],
                "summary": payload.get("summary", ""),
            },
        )
        return payload

    def search_theory(self, query: str, top_k: int = 5) -> str:
        output_dir = str(self.state.get("output_dir") or "").strip() or None
        payload = {"query": query, "top_k": top_k, "output_dir": output_dir}
        result = self._run_clean_subprocess_tool("search_theory", payload, timeout=180)
        if not result.get("ok"):
            return self._render_clean_subprocess_error("search_theory", result)
        trace = dict(result.get("trace") or {})
        if trace:
            self._append_literature_trace("search_theory", trace)
        return str(result.get("content") or "")

    def _append_literature_trace(self, event_type: str, payload: Dict[str, Any]) -> None:
        output_dir = Path(self.state.get("output_dir") or (Path.cwd() / "cytobridge_output")).expanduser().resolve()
        log_dir = output_dir / "rag_log"
        log_dir.mkdir(parents=True, exist_ok=True)
        target = log_dir / "literature_trace.jsonl"
        record = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "payload": payload,
        }
        with target.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _uses_codex_native_web_search(self) -> bool:
        return str(getattr(self.llm, "_llm_type", "") or "").strip().lower() == "codex-oauth-sidecar"

    def web_search(
        self,
        query: str,
        count: int = 5,
        allowed_domains: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        normalized_query = str(query or "").strip()
        normalized_domains = [str(item).strip() for item in (allowed_domains or []) if str(item).strip()]
        if not normalized_query:
            return {
                "success": False,
                "backend": "none",
                "query": "",
                "summary": "",
                "results": [],
                "error": "query must not be empty",
            }

        if self._uses_codex_native_web_search():
            try:
                result = codex_native_web_search(
                    query=normalized_query,
                    model=str(getattr(self.llm, "model_name", "") or "").strip(),
                    preferred_profile_id=getattr(self.llm, "preferred_profile_id", None),
                    count=count,
                    allowed_domains=normalized_domains,
                    context_size="medium",
                    mode="cached",
                )
                sources = [
                    {
                        "title": str(item.get("title") or "").strip(),
                        "url": str(item.get("url") or "").strip(),
                        "snippet": "",
                    }
                    for item in (result.get("sources") or [])
                    if isinstance(item, dict) and (item.get("url") or item.get("title"))
                ]
                payload = {
                    "success": True,
                    "backend": "openai_codex_native",
                    "query": normalized_query,
                    "summary": str(result.get("content") or "").strip(),
                    "results": sources[: max(1, int(count or 5))],
                    "usage": dict(result.get("usage") or {}),
                    "mode": str(result.get("mode") or "cached"),
                }
            except Exception as exc:
                payload = search_web_free(
                    normalized_query,
                    count=count,
                    allowed_domains=normalized_domains,
                    searxng_base_url=os.environ.get("SEARXNG_BASE_URL"),
                )
                notes = list(payload.get("notes") or [])
                notes.insert(
                    0,
                    f"OpenAI/Codex native web_search failed and fell back to managed free search: {exc}",
                )
                payload["notes"] = notes
                payload["fallback_from"] = "openai_codex_native"
        else:
            payload = search_web_free(
                normalized_query,
                count=count,
                allowed_domains=normalized_domains,
                searxng_base_url=os.environ.get("SEARXNG_BASE_URL"),
            )

        self._emit_event(
            "web_search",
            {
                "scope": "runtime",
                "query": normalized_query,
                "backend": payload.get("backend"),
                "success": bool(payload.get("success")),
                "result_count": len(payload.get("results") or []),
            },
        )
        return payload

    def web_fetch(
        self,
        url: str,
        max_chars: int = 12000,
        extract_mode: Literal["readable", "raw_text", "html"] = "readable",
        timeout_sec: float = 20.0,
        allowed_domains: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        payload = fetch_web_content(
            url,
            max_chars=max_chars,
            extract_mode=extract_mode,
            timeout_sec=timeout_sec,
            allowed_domains=allowed_domains,
        )
        self._emit_event(
            "web_fetch",
            {
                "scope": "runtime",
                "url": str(url or "").strip(),
                "backend": payload.get("backend"),
                "success": bool(payload.get("success")),
                "status_code": payload.get("status_code"),
                "final_url": payload.get("final_url"),
                "truncated": bool(payload.get("truncated")),
            },
        )
        return payload

    def run_terminal_command(
        self,
        command: str,
        cwd: str = "",
        timeout_sec: float = 20.0,
    ) -> Dict[str, Any]:
        payload = execute_guarded_terminal_command(
            command,
            cwd=str(cwd or "").strip() or None,
            timeout_sec=timeout_sec,
            output_dir=str(self.state.get("output_dir") or "").strip() or None,
        )
        self._emit_event(
            "terminal_command",
            {
                "scope": "runtime",
                "success": bool(payload.get("success")),
                "command": str(payload.get("command") or command or "").strip(),
                "argv": list(payload.get("argv") or []),
                "cwd": str(payload.get("cwd") or cwd or "").strip(),
                "timeout_sec": payload.get("timeout_sec"),
                "timed_out": bool(payload.get("timed_out")),
                "exit_code": payload.get("exit_code"),
                "clone_destination": payload.get("clone_destination", ""),
                "stdout": str(payload.get("stdout") or ""),
                "stderr": str(payload.get("stderr") or ""),
                "error": str(payload.get("error") or ""),
                "stdout_truncated": bool(payload.get("stdout_truncated")),
                "stderr_truncated": bool(payload.get("stderr_truncated")),
                "duration_sec": payload.get("duration_sec"),
                "supported_commands": dict(payload.get("supported_commands") or {}),
                "supported_commands_summary": str(payload.get("supported_commands_summary") or ""),
                "usage_hint": str(payload.get("usage_hint") or ""),
            },
        )
        return payload

    def read_file(
        self,
        file_path: str,
        offset: int = 1,
        limit: Optional[int] = None,
        pages: Optional[str] = None,
        pdf_mode: Literal["render", "text"] = "render",
    ) -> Any:
        result = super().read_file(
            file_path=file_path,
            offset=offset,
            limit=limit,
            pages=pages,
            pdf_mode=pdf_mode,
        )
        return result

    @staticmethod
    def _looks_like_url(value: str) -> bool:
        lowered = str(value or "").strip().lower()
        return lowered.startswith("http://") or lowered.startswith("https://")

    def _resolve_paper_source(self, input_source: str = "") -> Tuple[Optional[str], Optional[str]]:
        raw_value = str(input_source or self.state.get("input_path") or "").strip()
        if not raw_value:
            return None, (
                "Error: no paper source was provided. Pass a local paper path/article URL or set input_path to a paper source first."
            )
        if self._looks_like_url(raw_value):
            return raw_value, None
        target = Path(raw_value).expanduser().resolve()
        if not target.exists():
            return None, f"Error: paper source path does not exist: {target}"
        return str(target), None

    def inspect_paper_source(
        self,
        input_source: str = "",
        fetch_online_if_needed: bool = True,
        article_url: str = "",
        dataset_hint: str = "",
        target_dataset_prompt: str = "",
        use_llm: bool = True,
    ) -> str:
        source, err = self._resolve_paper_source(input_source)
        if err:
            return err

        from ..paper_intake import inspect_paper_source as inspect_paper_source_impl

        output_dir = Path(self.state.get("output_dir") or (Path.cwd() / "cytobridge_output")).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = inspect_paper_source_impl(
            input_source=str(source),
            output_dir=str(output_dir),
            fetch_online_if_needed=bool(fetch_online_if_needed),
            article_url=str(article_url or "").strip() or None,
            dataset_hint=str(dataset_hint or "").strip() or None,
            target_dataset_prompt=str(target_dataset_prompt or "").strip() or None,
            llm=self.llm,
            use_llm=bool(use_llm),
        )
        candidate_items = [item for item in list(payload.get("candidate_assets") or []) if isinstance(item, dict)]
        selected_asset_urls = [str(url or "").strip() for url in list(payload.get("selected_asset_urls") or []) if str(url or "").strip()]
        supporting_asset_urls = [str(url or "").strip() for url in list(payload.get("supporting_asset_urls") or []) if str(url or "").strip()]
        fallback_asset_urls = [str(url or "").strip() for url in list(payload.get("fallback_asset_urls") or []) if str(url or "").strip()]

        def preview_item(item: dict, role: str = "") -> dict:
            out = {
                "url": item.get("url"),
                "provider": item.get("provider"),
                "asset_kind": item.get("asset_kind"),
                "file_format": item.get("file_format"),
                "filename": item.get("filename"),
                "score": item.get("score"),
            }
            if role:
                out["selection_role"] = role
            return out

        by_url = {str(item.get("url") or ""): item for item in candidate_items}
        preview_assets = []
        seen_urls = set()
        for role, urls in (
            ("selected", selected_asset_urls),
            ("supporting", supporting_asset_urls),
            ("fallback", fallback_asset_urls),
        ):
            for url in urls:
                if not url or url in seen_urls:
                    continue
                item = by_url.get(url)
                preview_assets.append(preview_item(item or {"url": url}, role=role))
                seen_urls.add(url)
        for item in candidate_items:
            url = str(item.get("url") or "")
            if url in seen_urls:
                continue
            preview_assets.append(preview_item(item))
            seen_urls.add(url)
            if len(preview_assets) >= 15:
                break
        response = {
            "input_source": str(source),
            "title": payload.get("title") or "",
            "doi": payload.get("doi") or "",
            "summary_path": payload.get("summary_path") or "",
            "inspection_bundle_path": payload.get("inspection_bundle_path") or "",
            "asset_report_path": payload.get("asset_report_path") or "",
            "llm_pdf_analysis_path": payload.get("llm_pdf_analysis_path") or "",
            "llm_web_analysis_path": payload.get("llm_web_analysis_path") or "",
            "llm_discovery_trace_path": payload.get("llm_discovery_trace_path") or "",
            "llm_enabled": bool(payload.get("llm_enabled")),
            "selected_asset_urls": selected_asset_urls,
            "supporting_asset_urls": supporting_asset_urls,
            "fallback_asset_urls": fallback_asset_urls,
            "resolved_dataset_hint": payload.get("resolved_dataset_hint") or "",
            "target_dataset_prompt": payload.get("target_dataset_prompt") or "",
            "candidate_asset_count": len(payload.get("candidate_assets") or []),
            "accessions": payload.get("accessions") or {},
            "candidate_assets": preview_assets,
        }
        return json.dumps(response, ensure_ascii=False, indent=2)

    def materialize_paper_dataset(
        self,
        input_source: str = "",
        dataset_hint: str = "",
        target_dataset_prompt: str = "",
        selected_asset_url: str = "",
        fetch_online_if_needed: bool = True,
        article_url: str = "",
        prefer_processed: bool = True,
        allow_raw: bool = False,
        require_downstream_ready: bool = True,
        auto_load: bool = True,
        use_llm: bool = True,
    ) -> str:
        source, err = self._resolve_paper_source(input_source)
        if err:
            return err

        from ..paper_intake import prepare_anndata_from_paper as prepare_anndata_from_paper_impl

        output_dir = Path(self.state.get("output_dir") or (Path.cwd() / "cytobridge_output")).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        result = prepare_anndata_from_paper_impl(
            input_path=str(source),
            output_dir=str(output_dir),
            dataset_hint=str(dataset_hint or "").strip() or None,
            target_dataset_prompt=str(target_dataset_prompt or "").strip() or None,
            selected_asset_url=str(selected_asset_url or "").strip() or None,
            fetch_online_if_needed=bool(fetch_online_if_needed),
            article_url=str(article_url or "").strip() or None,
            prefer_processed=bool(prefer_processed),
            allow_raw=bool(allow_raw),
            llm=self.llm,
            use_llm=bool(use_llm),
            require_downstream_ready=bool(require_downstream_ready),
        )
        load_note = ""
        if auto_load:
            load_note = self.load_or_switch_adata(result["adata_path"], force_reload=True)
        self.state["converted_path"] = str(result.get("adata_path") or "")
        response = {
            "input_source": str(source),
            "adata_path": result.get("adata_path") or "",
            "selected_asset": result.get("selected_asset") or {},
            "target_dataset_prompt": result.get("target_dataset_prompt") or "",
            "report_path": result.get("report_path") or "",
            "inspection_bundle_path": result.get("inspection_bundle_path") or "",
            "asset_report_path": result.get("asset_report_path") or "",
            "materialization_attempts_path": result.get("materialization_attempts_path") or "",
            "auto_loaded": bool(auto_load),
            "load_note": load_note,
        }
        return json.dumps(response, ensure_ascii=False, indent=2)

    def run_training(
        self,
        candidate_name: str = "",
        training_algorithm_id: str = "",
        config_overrides: Optional[Dict[str, Any]] = None,
        run_label: str = "",
        adata_path: str = "",
        device: str = "",
        seed: Optional[int] = None,
        decision: str = "provisional",
        decision_reason: str = "",
        holdout_time_evaluation: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Run CytoBridge training through the standardized training backend.

        Provide exactly one of:
        - candidate_name: builtin config / candidate family
        - training_algorithm_id: custom algorithm id under ~/.cellcompass/training_algorithms

        Data-selection rule:
        - if adata_path is provided, that file is used for this run
        - otherwise training uses the current workflow dataset path
          (typically preprocessed_path, falling back to input_path)

        Device rule:
        - if device is provided, use it for this run only
        - otherwise reuse the current session device
        - use this only when there is a concrete reason (for example forced CPU fallback)

        Internal data model:
        - raw AnnData handling belongs to `training_data_builder(...)`
        - custom model construction belongs to `model_builder(...)`
        - full custom stage execution belongs to `stage_runner(...)`
        - backend construction belongs to `flow_matching_backend_builder(...)`
        - evaluation-time prediction overrides belong to `simulation_hook(...)`
        - backend builders receive `build_context.training_data`, not raw adata

        Direct-run semantics:
        - this is a manual/debug training run, not campaign lifecycle completion
        - custom algorithms are complete only after campaign final_regression locks a release

        Optional hold-out time evaluation:
        - default disabled
        - if enabled, the normal full-data run completes first, then auxiliary
          split training runs estimate W1 at held-out time point(s)
        - use only for trajectory generalization/reasonableness claims, or when
          intentionally attaching the same protocol as a comparable claim metric

        This reuses the existing run-bundle logic so training outputs, metrics,
        and manifests remain consistent.
        """
        result = self.run_training_tool(
            candidate_name=candidate_name or None,
            training_algorithm_id=training_algorithm_id or None,
            config_overrides=config_overrides,
            run_label=run_label or None,
            adata_path=adata_path or None,
            device=device or None,
            seed=seed,
            decision=decision or "provisional",
            decision_reason=decision_reason or "",
            holdout_time_evaluation=holdout_time_evaluation,
        )
        if not isinstance(result, str):
            return str(result)
        result = result.replace(
            "🔴 CRITICAL: DO NOT call `run_training_tool` again. ",
            "🔴 CRITICAL: Avoid launching duplicate training runs unless you intentionally want another run. ",
        )
        return result

    def summarize_training_runs(self) -> str:
        runs = list(self.state.get("training_runs") or [])
        if not runs:
            return "No training runs recorded."
        lines = []
        for idx, run in enumerate(runs, start=1):
            lines.append(
                f"{idx}. run_id={run.get('run_id') or run.get('id','')} | stage={run.get('stage','')} | "
                f"mode={run.get('training_mode','')} | status={run.get('status','')} | "
                f"model={run.get('model_artifact_path') or run.get('trained_model_path','')}"
            )
        return "\n".join(lines)

    def summarize_downstream_artifacts(self) -> str:
        figures = list(self.state.get("downstream_figures") or [])
        results = list(self.state.get("downstream_results") or [])
        lines = [f"Downstream results: {len(results)}", f"Downstream figures: {len(figures)}"]
        for idx, fig in enumerate(figures[:50], start=1):
            lines.append(
                f"{idx}. path={fig.get('path','')} | caption={fig.get('caption','')} | analysis={fig.get('analysis','')}"
            )
        return "\n".join(lines)

    def commit_workflow_state(
        self,
        phase: str,
        updates: Dict[str, Any],
        artifacts: Optional[Dict[str, Any]] = None,
        summary: str = "",
        mark_phase_complete: bool = True,
    ) -> str:
        """
        Commit phase outputs into the unified workflow state.

        Prefer structured values in `updates`:
        - preprocessing:
          `preprocessed_path=...`
          `preprocessing_script_path=...`
          `preprocessing_script_sha256=...`
          `preprocessing_input_paths=[...]`
          `preprocessing_output_paths=[...]`
        - training:
          `final_config={"name": ..., "path": ..., "run_id": ..., "run_dir": ...}`
          `final_metrics={...}`
          `training_runs=[...]`
        - downstream:
          `downstream_results=[...]`
          `downstream_summary="..."`
          `downstream_figures=[{"path": ..., "caption": ..., "analysis": ...}, ...]`
        - reporting:
          `report_path="..."`
          `final_summary="..."`

        Avoid shorthand strings for keys that later phases treat as mappings,
        especially `final_config`.
        """
        return self.committer.commit(
            phase=phase,
            updates=updates,
            artifacts=artifacts,
            summary=summary,
            mark_phase_complete=mark_phase_complete,
        )

    def _tool_event_display(self) -> DisplayManager:
        return DisplayManager()

    def _stringify_tool_output(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, ReadResult):
            return value.tool_text
        if isinstance(value, dict) and "tool_text" in value and "media" in value:
            return str(value.get("tool_text") or "")
        if isinstance(value, (dict, list, tuple)):
            try:
                return json.dumps(value, ensure_ascii=False, indent=2)
            except Exception:
                return str(value)
        return str(value)

    def _build_tool(
        self,
        func: Callable[..., Any],
        name: str,
        description: str,
        emit_events: bool = True,
        args_schema: Optional[type[BaseModel]] = None,
        isolation: str = "in_process",
        isolation_timeout: Optional[float] = None,
        timeout_sec: Optional[float] = None,
        mutates_state: Optional[bool] = None,
        reads_adata: Optional[bool] = None,
        writes_artifacts: Optional[bool] = None,
        may_use_gpu: Optional[bool] = None,
    ) -> StructuredTool:
        tool_name = str(name or "").strip()
        effective_isolation = str(isolation or "in_process").strip() or "in_process"
        if tool_name in _MANAGED_SUBPROCESS_TOOL_NAMES and effective_isolation == "in_process":
            effective_isolation = "managed_subprocess"
        if effective_isolation == "in_process":
            timeout_sec = None
            isolation_timeout = None
        elif timeout_sec is None:
            if isolation_timeout is not None:
                timeout_sec = isolation_timeout
            elif tool_name == "preview_training_run":
                timeout_sec = PREVIEW_TRAINING_TOOL_TIMEOUT_SEC
            elif tool_name in _GPU_TOOL_NAMES or tool_name in _MANAGED_SUBPROCESS_TOOL_NAMES:
                timeout_sec = TRAINING_TOOL_TIMEOUT_SEC if tool_name in _GPU_TOOL_NAMES else DEFAULT_TOOL_TIMEOUT_SEC
            elif tool_name in {"materialize_paper_dataset", "inspect_paper_source"}:
                timeout_sec = LONG_IO_TOOL_TIMEOUT_SEC
            else:
                timeout_sec = DEFAULT_TOOL_TIMEOUT_SEC
        effective_timeout = max(1.0, float(timeout_sec)) if timeout_sec is not None else None
        effective_isolation_timeout = isolation_timeout
        if effective_isolation == "process" and effective_isolation_timeout is None:
            if effective_timeout is None:
                effective_timeout = float(DEFAULT_TOOL_TIMEOUT_SEC)
            effective_isolation_timeout = effective_timeout
        contract = {
            "cytobridge_contract_version": 1,
            "cytobridge_tool_name": tool_name,
            "cytobridge_isolation": effective_isolation,
            "cytobridge_timeout_sec": effective_timeout,
            "cytobridge_isolation_timeout": effective_isolation_timeout,
            "cytobridge_timeout_enforced": effective_isolation in {"process", "managed_subprocess"},
            "cytobridge_mutates_state": bool(mutates_state if mutates_state is not None else tool_name in _STATE_MUTATING_TOOL_NAMES),
            "cytobridge_reads_adata": bool(reads_adata if reads_adata is not None else tool_name in _ADATA_READING_TOOL_NAMES),
            "cytobridge_writes_artifacts": bool(
                writes_artifacts if writes_artifacts is not None else tool_name in _ARTIFACT_WRITING_TOOL_NAMES
            ),
            "cytobridge_may_use_gpu": bool(may_use_gpu if may_use_gpu is not None else tool_name in _GPU_TOOL_NAMES),
        }

        def _render_isolation_error(result: IsolatedToolResult) -> str:
            parts = [
                f"Error: tool '{name}' failed in isolated subprocess.",
                f"error_type: {result.error_type or 'unknown'}",
                f"error: {result.error or '(no error text returned)'}",
            ]
            if result.returncode is not None:
                parts.append(f"returncode: {result.returncode}")
            if result.signal_name:
                parts.append(f"signal: {result.signal_name}")
            if result.timed_out:
                parts.append("timed_out: true")
            if result.traceback:
                parts.append(f"Traceback:\n{result.traceback[:12000]}")
            parts.append(
                "The main agent process remained alive; inspect the tool inputs/artifacts and retry with a safer command or smaller workload."
            )
            return "\n".join(parts)

        @wraps(func)
        def instrumented(*args, **kwargs):
            display = self._tool_event_display()
            tool_call_id: Optional[str] = None
            if emit_events:
                tool_call_id = display.print_tool_start(name, kwargs)
            try:
                if effective_isolation == "process" and os.environ.get("CYTOBRIDGE_DISABLE_TOOL_PROCESS_ISOLATION") != "1":
                    isolated = run_tool_in_subprocess(
                        func,
                        *args,
                        isolation_timeout=effective_isolation_timeout,
                        tool_kwargs=kwargs,
                    )
                    if not isolated.ok:
                        rendered_error = _render_isolation_error(isolated)
                        if emit_events:
                            display.print_tool_output(rendered_error, tool_call_id=tool_call_id)
                        return rendered_error
                    result = isolated.result
                else:
                    result = func(*args, **kwargs)
            except BaseException as exc:  # noqa: BLE001
                trace_text = traceback.format_exc()
                rendered_error = (
                    f"Error: tool '{name}' failed: {exc}\n"
                    f"Traceback:\n{trace_text[:12000]}"
                )
                if emit_events:
                    display.print_tool_output(rendered_error, tool_call_id=tool_call_id)
                    display.finish_tool_call(tool_call_id)
                return rendered_error
            try:
                rendered = self._stringify_tool_output(result)
                if emit_events and rendered:
                    display.print_tool_output(rendered, tool_call_id=tool_call_id)
                return result
            finally:
                if emit_events:
                    display.finish_tool_call(tool_call_id)

        return StructuredTool.from_function(
            func=instrumented,
            name=name,
            description=description,
            args_schema=args_schema,
            metadata={
                **contract,
            },
        )

    def get_tools(self) -> List[StructuredTool]:
        tools = [
            self._build_tool(self.execute_python, "execute_python", "Execute Python code for exploratory data checks, analysis, training orchestration, or downstream workflows. The current adata is already managed for you, the CytoBridge package is preloaded as `cb` when available, and the runtime provides `output_dir`, `figures_dir`, and `scripts_dir`. By default this runs in an isolated subprocess with strict timeout enforcement; AnnData changes are synchronized back when possible, but temporary Python variables are not persisted. Use this for exploratory work; when code becomes important for reuse or review, save it under output_dir/scripts and rerun it with `run_saved_python_script(...)`. Optional `timeout` overrides the execution limit in seconds; default is 300. Timeout overruns are terminated and returned as tool output.", emit_events=False),
            self._build_tool(self.run_saved_python_script, "run_saved_python_script", "Execute a saved Python script from output_dir/scripts in true script mode. The script runs with `__name__ == '__main__'`, a fresh script scope, the current managed `adata`, and the runtime helpers (`output_dir`, `figures_dir`, `scripts_dir`, `save_pubfig`, etc.). Relative paths are resolved against output_dir/scripts; use this for durable preprocessing/downstream/reporting scripts after exploratory work stabilizes. Default timeout mode is isolated for strict subprocess timeout enforcement.", args_schema=_RunSavedPythonScriptInput, emit_events=False),
            self._build_tool(self.inspect_adata_state, "inspect_adata_state", "Inspect the currently loaded AnnData object, including obs/obsm/layers/uns keys and shape."),
            self._build_tool(self.load_or_switch_adata, "load_or_switch_adata", "Load or switch the active AnnData .h5ad file into the shared runtime manager."),
            self._build_tool(self.persist_runtime_adata, "persist_runtime_adata", "Persist the currently loaded runtime AnnData object to disk."),
            self._build_tool(
                self.list_path,
                "list_path",
                "List a file or directory path for repo/docs/data inspection.",
                isolation="process",
                isolation_timeout=60,
            ),
            self._build_tool(
                self.read_file,
                "read_file",
                "Unified file reader for text, images, and PDFs. For text, use line-based `offset`/`limit`. For PDFs, choose `pdf_mode='text'` to extract text or `pdf_mode='render'` to inspect rendered page images, and optionally narrow with `pages` like `1-3,5`. Image reads and `pdf_mode='render'` require multimodal to be enabled for the session. Use this for docs, prompts, skills, source code, images, and PDFs.",
                args_schema=_ReadFileInput,
                isolation="process",
                isolation_timeout=180,
            ),
            self._build_tool(
                self.inspect_h5ad_contract,
                "inspect_h5ad_contract",
                "Read-only inspection of an AnnData .h5ad file for reviewer use. Returns shape, obs columns, selected time key, label/lineage/fate/condition counts, obsm/layers/uns keys, and never loads it as the active runtime dataset or writes data.",
                args_schema=_InspectH5adContractInput,
                isolation="managed_subprocess",
                isolation_timeout=240,
            ),
            self._build_tool(
                self.inspect_paper_source,
                "inspect_paper_source",
                "Inspect a paper PDF/article/manuscript source and list candidate downloadable datasets without downloading or materializing them. The inspection is LLM-led when a runtime LLM is available: failed URL/page/provider attempts are recorded as recoverable observations in the discovery trace. If the user asks for a specific part of the paper data, pass that request as `target_dataset_prompt` so inspection scopes candidate selection to that subset. Use this when the user wants to discuss a paper, understand dataset availability, or choose which manuscript dataset to extract later. For routine PDF reading alone, prefer `read_file(...)`.",
                args_schema=_InspectPaperSourceInput,
            ),
            self._build_tool(
                self.materialize_paper_dataset,
                "materialize_paper_dataset",
                "Explicit on-demand paper-to-AnnData entrypoint. Resolve/download the requested manuscript dataset, materialize `paper_input.h5ad`, and optionally load it into the runtime. When the user requests a subset such as an experimental arm, tissue, figure, accession, or time course, pass a binding `target_dataset_prompt`; paper intake should download only the primary and sidecar assets needed for that target, not all paper datasets. LLM-selected assets take priority over heuristic scores, and recoverable download/materialization errors are fed back into retry rounds. Use this only when the user explicitly asks to extract, download, or continue analysis from paper-derived data. Provide `selected_asset_url` to lock materialization to one inspected candidate.",
                args_schema=_MaterializePaperDatasetInput,
            ),
            self._build_tool(
                self.find_files,
                "find_files",
                "Find files by glob pattern. Optional `timeout` limits directory scanning time in seconds; default is 15.",
                isolation="process",
                isolation_timeout=90,
            ),
            self._build_tool(
                self.grep_files,
                "grep_files",
                "Search a text pattern in files. Optional `timeout` limits scanning time in seconds; default is 15.",
                isolation="process",
                isolation_timeout=90,
            ),
            self._build_tool(
                self.web_search,
                "web_search",
                "Search the public web for up-to-date external information. For ordinary models this uses a free managed backend (prefer configured SearXNG, otherwise DuckDuckGo HTML fallback). For `codex_oauth` sessions it uses OpenAI/Codex native Responses `web_search`. Returns a concise summary when available plus structured sources/results.",
                args_schema=_WebSearchInput,
                isolation="process",
                isolation_timeout=120,
            ),
            self._build_tool(
                self.web_fetch,
                "web_fetch",
                "Fetch a public web page and return readable content. This is a managed HTTP fetch tool for all models; it supports readable extraction, raw visible text extraction, or raw HTML, enforces public-network URL safety checks, and rejects localhost/private-network targets.",
                args_schema=_WebFetchInput,
                isolation="process",
                isolation_timeout=120,
            ),
            self._build_tool(
                self.run_terminal_command,
                TERMINAL_TOOL_NAME,
                f"Execute one guarded terminal command with an explicit timeout. Use this only when dedicated tools like read_file/find_files/grep_files/web_search/web_fetch are insufficient; prefer those tools first. The command runner is intentionally narrow: use `cwd` instead of `cd`, shell chaining/redirection is blocked, write/edit/install commands are blocked, `git` is limited to read-only inspection plus restricted `git clone`, and `curl` is limited to safe read-only HTTP access. {SUPPORTED_COMMAND_SUMMARY}",
                args_schema=_TerminalCommandInput,
                emit_events=False,
                isolation="process",
                isolation_timeout=3600,
            ),
            self._build_tool(
                self.get_algorithm_proposal_template,
                "get_algorithm_proposal_template",
                "Read-only helper that returns the current standard CytoBridge algorithm proposal template and per-section writing guidance. Use before create_algorithm_proposal(...) or revise_algorithm_proposal(...).",
            ),
            self._build_tool(
                self.create_algorithm_proposal,
                "create_algorithm_proposal",
                "Create/update a custom algorithm proposal at theory/math level (no code details). For new algorithms, first ground the proposal with search_theory(...) for mathematical foundations and search_literature(...) for algorithm papers/baselines, then read the relevant page/section ranges with read_file(...). The proposal must include an abstract, claimed problem/capability, expected evaluation outcome, inductive generalization argument for new valid t0 cells, machine-readable mass_modeling_scope, distribution/mass recovery argument, and anti-overengineering self-check.",
                args_schema=_CreateAlgorithmProposalInput,
            ),
            self._build_tool(
                self.revise_algorithm_proposal,
                "revise_algorithm_proposal",
                "Create a new revision of an existing algorithm proposal and trigger the normal proposal review flow. Use proposal_markdown for full proposal rewrites and proposal_patch for small section edits; provide exactly one of them. This preserves proposal_id history, registry artifacts, risk artifacts, and review semantics better than editing PROPOSAL.md as a normal file.",
                args_schema=_ReviseAlgorithmProposalInput,
            ),
            self._build_tool(
                self.create_research_idea,
                "create_research_idea",
                "Create a persistent research idea with a sharp scientific question, evidence basis, and bounded direction families.",
                args_schema=_CreateResearchIdeaInput,
            ),
            self._build_tool(
                self.revise_research_idea,
                "revise_research_idea",
                "Create a new revision of an existing research idea while preserving unspecified fields from the latest revision.",
                args_schema=_ReviseResearchIdeaInput,
            ),
        ]
        if self.agent_role == "planner":
            tools.append(
                self._build_tool(
                    self.spawn_subagent,
                    SPAWN_SUBAGENT_TOOL_NAME,
                    (
                        "Spawn a bounded synchronous subagent with an explicit role type, brief, and success criteria. "
                        f"Current built-in types: {', '.join(available_subagent_type_names())}. "
                        "The subagent runs in an isolated runtime and returns a structured result for the planner to review and merge explicitly. "
                        "For proposal_evaluator subagents, a valid submitted proposal_review is automatically applied to the resolved target proposal."
                    ),
                    args_schema=_SpawnSubagentInput,
                )
            )
        elif self.agent_role == "subagent":
            if self.subagent_type == "proposal_evaluator":
                tools.append(
                    self._build_tool(
                        self.submit_proposal_review,
                        SUBMIT_PROPOSAL_REVIEW_TOOL_NAME,
                        "Submit the final proposal-review verdict and feedback for this evaluator run. This both finalizes the evaluator subagent and returns the review decision to the parent planner.",
                        args_schema=_SubmitProposalReviewInput,
                    )
                )
            elif self.subagent_type == "idea_evaluator":
                tools.append(
                    self._build_tool(
                        self.submit_research_idea_review,
                        SUBMIT_RESEARCH_IDEA_REVIEW_TOOL_NAME,
                        "Submit the final research-idea review verdict and feedback for this evaluator run. This both finalizes the evaluator subagent and returns the review decision to the parent planner.",
                        args_schema=_SubmitResearchIdeaReviewInput,
                    )
                )
            elif self.subagent_type == "implementation_evaluator":
                tools.append(
                    self._build_tool(
                        self.submit_implementation_review,
                        SUBMIT_IMPLEMENTATION_REVIEW_TOOL_NAME,
                        "Submit the final implementation-alignment review verdict and feedback for this evaluator run. This both finalizes the evaluator subagent and returns the review decision to the parent planner.",
                        args_schema=_SubmitImplementationReviewInput,
                    )
                )
            else:
                tools.append(
                    self._build_tool(
                        self.submit_subagent_result,
                        SUBMIT_SUBAGENT_RESULT_TOOL_NAME,
                        "Submit the authoritative structured result for this subagent run. Call this before finishing. The parent planner will review and merge any proposed_state_updates explicitly.",
                        args_schema=_SubmitSubagentResultInput,
                    )
                )
        tools.extend([
            self._build_tool(
                self.get_algorithm_proposal_status,
                "get_algorithm_proposal_status",
                "Get proposal status for one algorithm id or all proposals. For proposal revisions, prefer revise_algorithm_proposal(...). Use editable_proposal_path only if you must patch the root PROPOSAL.md directly.",
                args_schema=_GetAlgorithmProposalStatusInput,
            ),
            self._build_tool(
                self.review_research_idea,
                "review_research_idea",
                "Record user review decision for a research idea. decision: approve | reject | revise.",
                args_schema=_ReviewResearchIdeaInput,
            ),
            self._build_tool(
                self.get_research_idea_status,
                "get_research_idea_status",
                "Get research idea status for one idea id or all ideas.",
                args_schema=_GetResearchIdeaStatusInput,
            ),
            self._build_tool(
                self.list_research_ideas,
                "list_research_ideas",
                "List research ideas with optional track filter.",
                args_schema=_ListResearchIdeasInput,
            ),
            self._build_tool(
                self.set_active_research_idea,
                "set_active_research_idea",
                "Set the active research idea for the current session and prompt context.",
                args_schema=_SetActiveResearchIdeaInput,
            ),
            self._build_tool(
                self.update_research_idea_progress,
                "update_research_idea_progress",
                "Update research idea progress and explicitly adjust resolution state when supported by evidence.",
                args_schema=_UpdateResearchIdeaProgressInput,
            ),
            self._build_tool(
                self.link_algorithm_to_idea,
                "link_algorithm_to_idea",
                "Associate an algorithm with a research idea and persist the weak linkage.",
                args_schema=_LinkAlgorithmToIdeaInput,
            ),
            self._build_tool(self.init_training_algorithm_workspace, "init_training_algorithm_workspace", "Create a custom training algorithm workspace under ~/.cellcompass/training_algorithms after proposal approval. It generates manifest.yaml, algorithm.py, config.yaml, README.md, and IMPLEMENTATION_MAP.md; config.yaml is seeded from builtin vgfm."),
            self._build_tool(self.record_decision, "record_decision", "Record a structured experiment-management decision for a custom algorithm.", args_schema=_RecordDecisionInput),
            self._build_tool(self.mark_algorithm_failed, "mark_algorithm_failed", "Mark a developing custom algorithm lifecycle as failed when evidence shows this direction cannot satisfy the user goal. This does not complete the task; the planner must revise, repair, or design a replacement algorithm.", args_schema=_MarkAlgorithmFailedInput),
            self._build_tool(self.mark_result_obsolete, "mark_result_obsolete", "Mark a proposal/run/comparison as obsolete so later tools stop treating it as valid evidence.", args_schema=_MarkResultObsoleteInput),
            self._build_tool(self.list_experiment_history, "list_experiment_history", "List registry-backed proposal, snapshot, run, decision, and obsolete history for a custom algorithm.", args_schema=_ListExperimentHistoryInput),
            self._build_tool(self.activate_algorithm_workspace, "activate_algorithm_workspace", "Make one custom algorithm workspace the active review/authoring/training context. This updates session state to point workflow gates at that algorithm and can also bind a specific proposal id or workspace snapshot id.", args_schema=_ActivateAlgorithmWorkspaceInput),
            self._build_tool(self.snapshot_active_algorithm_workspace, "snapshot_active_algorithm_workspace", "Create a clean registry snapshot of the currently active algorithm workspace and clear its dirty flag. Use this after intentional workspace edits and before review/training when the active context reports dirty_since_snapshot=true.", args_schema=_SnapshotActiveAlgorithmWorkspaceInput),
            self._build_tool(self.rollback_algorithm_workspace, "rollback_algorithm_workspace", "Restore a custom algorithm workspace from a recorded workspace snapshot or proposal-linked snapshot.", args_schema=_RollbackAlgorithmWorkspaceInput),
            self._build_tool(self.set_active_baseline_run, "set_active_baseline_run", "Promote a registered custom-algorithm run to the active baseline.", args_schema=_SetActiveBaselineRunInput),
            self._build_tool(self.compare_algorithm_runs, "compare_algorithm_runs", "Compare registered custom-algorithm runs and report deltas versus the active baseline.", args_schema=_CompareAlgorithmRunsInput),
            self._build_tool(self.list_algorithm_benchmarks, "list_algorithm_benchmarks", "List prepared algorithm benchmark datasets under ~/.cellcompass/algorithm_benchmarks. Use before campaign tuning to choose a real, comparable frozen stage panel.", args_schema=_ListAlgorithmBenchmarksInput, isolation="process", isolation_timeout=90),
            self._build_tool(self.get_algorithm_benchmark_dataset, "get_algorithm_benchmark_dataset", "Inspect one benchmark dataset card, including data_path, available fields, README, and campaign dataset_config_overrides example.", args_schema=_GetAlgorithmBenchmarkDatasetInput, isolation="process", isolation_timeout=90),
            self._build_tool(self.get_algorithm_benchmark_baselines, "get_algorithm_benchmark_baselines", "Read builtin/reference W1/TMV baseline records and leaderboard for one benchmark dataset.", args_schema=_GetAlgorithmBenchmarkBaselinesInput, isolation="process", isolation_timeout=90),
            self._build_tool(self.list_algorithm_benchmark_baselines, "list_algorithm_benchmark_baselines", "List concise builtin/reference baseline metrics plus local benchmark-managed model/config/log paths for one dataset. Use this to find baseline checkpoints for debugging.", args_schema=_ListAlgorithmBenchmarkBaselinesInput, isolation="process", isolation_timeout=90),
            self._build_tool(self.record_algorithm_benchmark_baseline, "record_algorithm_benchmark_baseline", "Record builtin/reference W1/TMV baseline metrics for one benchmark dataset and update its leaderboard. Do not use for custom claim metrics.", args_schema=_RecordAlgorithmBenchmarkBaselineInput),
            self._build_tool(self.make_benchmark_dataset_config, "make_benchmark_dataset_config", "Build the dataset_config_overrides payload expected by run_campaign_trial from registered benchmark dataset ids. Supports common_config_overrides plus per_dataset_config_overrides; the stage freezes dataset ids, not per-dataset config values.", args_schema=_MakeBenchmarkDatasetConfigInput),
            self._build_tool(self.register_algorithm_benchmark_dataset, "register_algorithm_benchmark_dataset", "Admin tool: validate and register a prepared benchmark .h5ad. It does not preprocess; missing obs['time_point_processed'] or obsm['X_latent'] returns invalid_contract with preprocessing instructions. Pass preprocessing_script_path for newly prepared datasets so the exact routine is copied into the benchmark directory and recorded for audit.", args_schema=_RegisterAlgorithmBenchmarkDatasetInput),
            self._build_tool(self.register_stage2_simulation_dataset, "register_stage2_simulation_dataset", "Register an agent-designed Stage 2 claim-validation simulation dataset with a frozen generator hash and simulation_version.", args_schema=_RegisterStage2SimulationDatasetInput),
            self._build_tool(self.start_algorithm_campaign, "start_algorithm_campaign", "Start a new autoresearch-style custom algorithm tuning campaign with isolated git archive, staged policies, and automatic promote/reject. This is not a resume/update tool: if an active campaign already exists, continue it with get_algorithm_campaign_status, set/switch stage panel, run_campaign_trial, check_campaign_stage_gate, or abort_current_campaign_trial for stale interrupted trials instead of starting another campaign. For a custom Stage 2 claim metric, claim_metric_spec must include evaluator_path so refresh_campaign_stage_baselines can compute the same metric from saved candidate/baseline trajectories; name/direction alone is not comparable.", args_schema=_StartAlgorithmCampaignInput),
            self._build_tool(self.update_campaign_claim_metric_spec, "update_campaign_claim_metric_spec", "Repair the authoritative claim_metric_spec for an existing campaign. Use this when a campaign needs evaluator_path/evaluator_function/provenance corrected; do not start duplicate campaigns or patch campaign.json directly just to repair the evaluator.", args_schema=_UpdateCampaignClaimMetricSpecInput),
            self._build_tool(self.set_campaign_stage_panel, "set_campaign_stage_panel", "Freeze the dataset ids for one campaign stage before trials. Build the payload with make_benchmark_dataset_config(...), pass its dataset_config_overrides here, then run_campaign_trial(...) can omit the payload or tune config overrides for the same ids. This locks dataset ids, not per-dataset config values.", args_schema=_SetCampaignStagePanelInput),
            self._build_tool(self.switch_campaign_stage_panel, "switch_campaign_stage_panel", "Intentionally switch the dataset ids for the current campaign stage without resetting stage trial_count/promote_count/reject_count. Use after finishing or aborting any open trial. This invalidates current-stage active best, external baseline metrics, and gate evidence so the next run_campaign_trial(...) and refresh_campaign_stage_baselines(...) use the new panel.", args_schema=_SwitchCampaignStagePanelInput),
            self._build_tool(self.refresh_campaign_stage_baselines, "refresh_campaign_stage_baselines", "Refresh builtin/reference baselines for the current frozen stage panel, auto-select the comparable stage baseline, and write it into the campaign gate when required evidence is complete. Do not call this for read-only inspection: use query_campaign_baseline_metrics. If the strict ledger is already complete for the unchanged active best and no explicit repair scope is supplied, run_missing=false is a no-op. For Stage 2 custom claim metrics, candidate and baseline values must come from the same campaign claim_metric_spec evaluator and carry matching claim_metric_evaluator provenance; static direct fields and numeric adapters are not accepted as comparable evidence. In strict_all_builtin mode, baseline_algorithms is a repair scope for named missing/stale baselines; the gate still audits the full fixed builtin comparator set and blocks on any missing required baseline. Reuses existing records/trajectories/models when possible; run_missing only trains truly missing baselines and does not retrain an existing Stage 2 baseline just because claim_metric_spec lacks evaluator_path.", args_schema=_RefreshCampaignStageBaselinesInput),
            self._build_tool(self.update_campaign_control_baseline, "update_campaign_control_baseline", "Deactivate, reactivate, update, or replace a previously registered control/ablation baseline with audit history. Use deactivate when a control was mis-coded and should no longer participate in gates; use replace after rerunning corrected control code. Gate checks ignore inactive controls.", args_schema=_UpdateCampaignControlBaselineInput),
            self._build_tool(self.run_campaign_control_baseline, "run_campaign_control_baseline", "Run an agent-declared ablation/shuffle/no-lag control on the same frozen campaign stage panel, using either sparse config_overrides or a temporary workspace_patch, aggregate metrics, and automatically register or replace the control baseline. This never creates/promotes/rejects a campaign trial and never changes active best.", args_schema=_RunCampaignControlBaselineInput),
            self._build_tool(self.run_campaign_locked_algorithm_baseline, "run_campaign_locked_algorithm_baseline", "Run a completed final-regression locked custom algorithm as an audited reference baseline on the current campaign's frozen stage panel, then register or replace it as gate evidence. The reference algorithm must be complete/locked. For h5ad format incompatibility, provide reference_dataset_adapter as an explicit Python script that materializes an adapted h5ad. For claim-metric context incompatibility, provide reference_claim_metric_adapter defining adapt_baseline_metric_context(context); it wraps the current campaign evaluator and must not replace it. reference_config_overrides are rejected to avoid guess-based retuning.", args_schema=_RunCampaignLockedAlgorithmBaselineInput),
            self._build_tool(self.compute_campaign_claim_metric_for_baselines, "compute_campaign_claim_metric_for_baselines", "Stage 2 helper: recompute the campaign claim metric for builtin/reference baselines on the frozen panel using saved baseline trajectories/models when possible, then update the comparable stage baseline. Returns structured missing/unsupported records instead of silently doing nothing.", args_schema=_ComputeCampaignClaimMetricForBaselinesInput),
            self._build_tool(
                self.query_campaign_baseline_metrics,
                "query_campaign_baseline_metrics",
                "Read-only paper/reviewer evidence query for actual measured campaign baseline metrics. Reports selected gate comparator, compact scalar metrics, W1 backend provenance, strict-ledger completeness/repair guidance, benchmark registry records, and local baseline training_run metrics.json paths. Nested evaluator diagnostics are omitted from the agent-facing summary and remain available through metrics_path. Use this before writing or approving baseline/result tables; it does not train, refresh, tune, or mutate campaign state.",
                args_schema=_QueryCampaignBaselineMetricsInput,
                isolation="process",
                isolation_timeout=120,
            ),
            self._build_tool(self.get_algorithm_campaign_status_summary, "get_algorithm_campaign_status", "Return concise campaign status with current stage, gate status, active best, blocked reason, next action, and recent trials. This agent-facing tool does not return the raw campaign registry.", args_schema=_GetAlgorithmCampaignStatusInput),
            self._build_tool(self.start_campaign_trial, "start_campaign_trial", "Ensure the campaign has one open trial. If a trial is already open, returns that same trial; otherwise starts from stage active best or from a rejected trial as a non-active working base.", args_schema=_StartCampaignTrialInput),
            self._build_tool(self.run_campaign_trial, "run_campaign_trial", "Archive the current algorithm workspace/config, run the frozen stage panel or the provided make_benchmark_dataset_config(...) payload, including dataset-specific config overrides, and let the campaign controller automatically promote or reject the trial. The agent does not pass a run decision. Optional holdout_time_evaluation is default-off and should be used only for trajectory generalization/reasonableness evidence; the normal non-holdout trial always runs first, then auxiliary holdout split runs record heldout W1 diagnostics.", args_schema=_RunCampaignTrialInput),
            self._build_tool(self.abort_current_campaign_trial, "abort_current_campaign_trial", "Abort and clear the current open campaign trial when it is stale or incomplete, such as status=running after stop/OOM/tool interruption with no live process. It records decision=aborted, clears current_trial_id, optionally restores the stage active-best workspace snapshot, and never promotes/rejects or changes active best.", args_schema=_AbortCurrentCampaignTrialInput),
            self._build_tool(self.resume_rejected_trial, "resume_rejected_trial", "Resume a rejected trial as a working base without changing the campaign active best unless the new trial later promotes.", args_schema=_ResumeRejectedTrialInput),
            self._build_tool(self.list_campaign_trials, "list_campaign_trials", "List concise paginated campaign trial summaries, optionally filtering by decision or stage. Defaults to the 10 most recent matching trials and never returns raw full trial payloads.", args_schema=_ListCampaignTrialsInput),
            self._build_tool(self.check_campaign_stage_gate, "check_campaign_stage_gate", "Check whether the current stage active best passes its lifecycle gate and persist stage_gate_evidence in the campaign registry. Stage 1/2 compare to external baselines. A supervisor/runtime locked stage-gate spec, when configured, is an additional immutable gate and cannot be changed from the algorithm workspace or campaign tuning loop. Stage 3 has an optional SOTA/Pareto early-completion gate: active best should be non-worse than the same-panel SOTA baseline on both W1 and the validated claim metric; if the Stage 3 budget is exhausted without this optional gate, the Stage-2-validated algorithm can still proceed to final regression. Final regression is frozen benchmarking/reporting and locks an active best without an external-baseline gate. Use advance=false to inspect readiness without moving stages; use advance=true to advance or lock after a pass.", args_schema=_CheckCampaignStageGateInput),
            self._build_tool(self.list_workspace_tree, "list_workspace_tree", "List planner-writable workspace trees such as algorithm workspaces and training runs.", isolation="process", isolation_timeout=90),
            self._build_tool(self.read_workspace_file, "read_workspace_file", "Read a file from the writable workspace roots.", isolation="process", isolation_timeout=180),
            self._build_tool(self.create_workspace_file, "create_workspace_file", "Create a new file inside writable workspace roots."),
            self._build_tool(
                self.create_algorithm_workspace_artifact,
                "create_algorithm_workspace_artifact",
                "Create a file or directory under the active algorithm workspace using a relative path. Use this for diagnostics scripts and outputs such as diagnostics/check_risk.py or diagnostics/risk_summary.json; it refuses absolute paths, '..', inactive algorithms, and unapproved proposal workspaces.",
                args_schema=_CreateAlgorithmWorkspaceArtifactInput,
            ),
            self._build_tool(self.replace_workspace_file, "replace_workspace_file", "Rewrite a workspace file."),
            self._build_tool(
                self.patch_algorithm_config,
                "patch_algorithm_config",
                "Safely patch the active custom algorithm config.yaml with typed leaf path updates and return a resolved unified diff. Writes by default so campaign promote/reject archives the real config file; set dry_run=true only for preview. Use this as a shortcut for simple scalar hyperparameter edits; use apply_workspace_patch for complex config edits.",
                args_schema=_PatchAlgorithmConfigInput,
            ),
            self._build_tool(
                self.apply_workspace_patch,
                "apply_workspace_patch",
                "Apply patch edits to workspace files. Supported formats: Codex patch (`*** Begin Patch ... *** End Patch`) and unified diff (`---/+++` with `@@` hunks). Patch headers are real paths, not git path aliases: use absolute paths or tool-returned workspace paths; do not use `--- a/file` / `+++ b/file`. For proposal revisions, prefer revise_algorithm_proposal(...), which supports full markdown replacement and local patches while preserving review semantics. Direct PROPOSAL.md patches still create a new proposal_id and trigger proposal review, but must target editable_proposal_path and must not mix proposal edits with code/config edits.",
            ),
            self._build_tool(self.preview_workspace_diff, "preview_workspace_diff", "Preview a unified diff without applying it.", isolation="process", isolation_timeout=90),
            self._build_tool(
                self.set_plan_from_text,
                "set_plan_from_text",
                "Parse free-text plan into structured steps with statuses.",
                args_schema=_SetPlanFromTextInput,
            ),
            self._build_tool(
                self.update_plan,
                "update_plan",
                "Update structured execution plan. Submit FULL plan list with explicit step+status for every item.",
                args_schema=_UpdatePlanInput,
            ),
            self._build_tool(self.get_plan_status, "get_plan_status", "Get current execution plan status and progress.", isolation="process", isolation_timeout=60),
            self._build_tool(
                self.list_skills,
                "list_skills",
                "List available skills with name, description, and SKILL.md path. Optional domain: all | planner | workflow | downstream | algorithm. Use this only when the relevant skill path is uncertain, then read the chosen SKILL.md with read_file before following it.",
            ),
            self._build_tool(
                self.get_current_workflow_context,
                "get_current_workflow_context",
                "Read a compact current workflow context snapshot: active algorithm/proposal/snapshot/dirty state, active research idea, plan, key paths, external literature-tool availability, pending planner/runtime actions, and recent artifact counts. This is read-only and is the preferred first check when active bindings, paths, or optional external tool availability are uncertain.",
            ),
            self._build_tool(self.search_literature, "search_literature", "Search the local theory knowledge before literature-backed reasoning (especially theory selection and downstream analysis). The returned payload includes `pdf_path` / `pdf_filename` values that can be read directly with `read_file` when deeper inspection is needed.", isolation="managed_subprocess", isolation_timeout=180),
            self._build_tool(
                self.search_bohrium_paper,
                "search_bohrium_paper",
                "Search Bohrium's large external paper RAG index using ACCESS_KEY or BOHRIUM_ACCESS_KEY from the environment. Check get_current_workflow_context().external_literature_tools.bohrium_paper_search.available first; if unavailable, use search_literature and web_search instead. Use this alongside search_literature for algorithm grounding, SOTA comparison, and broad literature discovery. Returns structured paper metadata, abstracts, corpus snippets, figures, DOI, journal, date, and citation fields when available.",
                args_schema=_BohriumPaperSearchInput,
            ),
            self._build_tool(
                self.search_theory,
                "search_theory",
                "Search local long-form mathematical theory books. Use this for OT, dynamic OT, WFR/UOT, Schrodinger bridge, continuity-equation, variational-flow, and proof-level questions. Results include book id, page, recommended page range, pdf_path, and ready-to-use read_file(..., pdf_mode='text', pages='...') commands.",
                args_schema=_SearchTheoryInput,
                isolation="managed_subprocess",
                isolation_timeout=180,
            ),
            self._build_tool(
                self.set_task_profile,
                "set_task_profile",
                "Set a multi-axis task profile for the current work. Use this instead of a single task_mode when the task mixes reproduction, tuning, integration, or new algorithm design.",
                args_schema=_SetTaskProfileInput,
            ),
            self._build_tool(
                self.check_workflow_gate,
                "check_workflow_gate",
                "Check whether the current state satisfies the minimum gate for a stage such as proposal, authoring, review, training, downstream, or report.",
                args_schema=_CheckWorkflowGateInput,
            ),
            self._build_tool(
                self.preview_training_run,
                "preview_training_run",
                "Training preview with full-data chain inspection plus an optional 1-epoch inference/evaluation smoke test. Inspection may run backend setup/build_state/sample_pairs before any epoch logs; a timeout there means setup scalability or contract failure, not slow one-epoch training. The smoke test is only for code/range validation, not final metric quality.",
                args_schema=_PreviewTrainingRunInput,
            ),
            self._build_tool(self.run_training, "run_training", "Run CytoBridge training through the standardized backend and run-bundle pipeline. Provide exactly one of `candidate_name` (builtin config/family) or `training_algorithm_id` (custom algorithm workspace id). Optional `adata_path` lets you train on a specific .h5ad for this manual/debug run. Optional holdout_time_evaluation is default-off; if enabled, the normal full-data run finishes first, then auxiliary split runs compute heldout timepoint W1 diagnostics for trajectory generalization. For custom algorithms, this does not complete development; only a campaign final_regression locked release is complete and suitable for user-facing reporting or downstream analysis. For iterative custom-algorithm tuning, prefer campaign tools such as `run_campaign_trial(...)` so promote/reject is automatic. The legacy `decision` field is for manual/non-campaign runs. Epoch overrides are blocked by default unless explicitly allowed with reason in `config_overrides`."),
            self._build_tool(self.commit_workflow_state, "commit_workflow_state", "Commit phase outputs, artifact references, and summary into the unified workflow state. Use structured values: for training prefer final_config={'name','path','run_id','run_dir'}, for downstream prefer downstream_results/downstream_summary/downstream_figures, for reporting prefer report_path/final_summary."),
            self._build_tool(self.summarize_training_runs, "summarize_training_runs", "Summarize recorded training runs from workflow state."),
            self._build_tool(self.summarize_downstream_artifacts, "summarize_downstream_artifacts", "Summarize stored downstream artifacts and figure inventory from workflow state."),
        ])
        if self.agent_role == "planner" and self._normalize_idea_review_mode() == "always_user_review":
            tools = [tool for tool in tools if str(getattr(tool, "name", "")) != "review_research_idea"]
        return [tool for tool in tools if self._is_tool_allowed(str(getattr(tool, "name", "") or ""))]
