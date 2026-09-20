"""
LangGraph State/Action schemas for CytoBridge Agent.

This module defines:
- State: the global state passed through the LangGraph nodes
- Actions: structured outputs from LLM decision nodes
- Evidence: references and citations for scientific claims
"""
from __future__ import annotations

from typing import List, Dict, Optional, Any, Literal, TypedDict
from pydantic import BaseModel, Field
from enum import Enum


# ==============================================================================
# Data Summary Schemas
# ==============================================================================

class TimePointSummary(BaseModel):
    """Summary of a candidate time-point column in the AnnData."""
    key: str
    levels: List[str]
    counts: List[int]
    mass_variance: float = Field(default=0.0, description="Coefficient of variation of counts")


class DataSummary(BaseModel):
    """Summary of the input AnnData for LLM context."""
    n_obs: int
    n_vars: int
    time_candidates: List[TimePointSummary] = Field(default_factory=list)
    has_latent: bool = False
    has_pca: bool = False
    has_umap: bool = False
    obs_keys: List[str] = Field(default_factory=list)
    label_candidates: List[str] = Field(default_factory=list, description="Columns likely to be cell type labels")
    var_is_log_norm: Optional[bool] = None
    sparsity: Optional[float] = None


# ==============================================================================
# User Goal / Scientific Question
# ==============================================================================

class AnalysisScope(str, Enum):
    """Types of downstream analyses the user wants."""
    TRAJECTORY_FATE = "trajectory_fate"
    DRIVERS_GENES = "drivers_genes"
    GROWTH_MASS = "growth_mass"
    STOCHASTICITY_SCORE = "stochasticity_score"
    INTERACTION = "interaction"
    COUNTERFACTUAL = "counterfactual"
    LIMITATIONS = "limitations"
    BENCHMARK = "benchmark"


class UserGoal(BaseModel):
    """Parsed user scientific question and constraints."""
    raw_question: str = Field(default="", description="Original user question in natural language")
    requested_analyses: List[AnalysisScope] = Field(default_factory=list)
    label_key: Optional[str] = Field(default=None, description="User-specified cell type column")
    time_key: Optional[str] = Field(default=None, description="User-specified time column")
    force_interaction: bool = False
    device: str = "cuda"
    max_retries: int = 3
    report_format: Literal["md", "html"] = "html"
    gene_sets_gmt: Optional[str] = None
    disable_llm_overrides: bool = Field(default=False, description="禁用 LLM 配置覆写，使用 YAML 默认配置")


# ==============================================================================
# Theory RAG Evidence
# ==============================================================================

class TheoryChunk(BaseModel):
    """A retrieved chunk from npj_review.pdf with citation info."""
    chunk_id: str
    text: str
    page: Optional[int] = None
    relevance_score: float = 0.0
    section: Optional[str] = None
    block_id: Optional[str] = None
    char_start: Optional[int] = None
    char_end: Optional[int] = None
    chunk_type: Optional[str] = None
    neighbor_rank: Optional[int] = None


class TheoryContext(BaseModel):
    """Collection of retrieved theory chunks for the current query."""
    query: str
    chunks: List[TheoryChunk] = Field(default_factory=list)


# ==============================================================================
# Model Configuration Schemas
# ==============================================================================

class CandidateConfig(BaseModel):
    """A candidate model configuration derived from built-in yaml."""
    name: str  # base config name (e.g., ruot, unbalanced_ot, crufm)
    overrides: Dict[str, Any] = Field(default_factory=dict)
    rationale: str = Field(default="", description="Why this config is proposed")





class ModelFamily(str, Enum):
    """Available CytoBridge model families."""
    DYNAMICAL_OT = "dynamical_ot"
    UNBALANCED_OT = "unbalanced_ot"
    RUOT = "ruot"
    CRUFM = "crufm"
    BALANCED_OT_CFM = "balanced_ot_cfm"
    SF2M = "sf2m"
    VGFM = "vgfm"
    WFRFM = "wfrfm"
    CYTO_INTERACTION = "cyto_simulation"


# ==============================================================================
# LLM Decision Schemas
# ==============================================================================

class PlanDecision(BaseModel):
    """LLM output: initial planning decision for model family and candidates."""
    model_family: ModelFamily
    reasoning: str = Field(description="Why this model family is appropriate")
    theory_support: str = Field(default="", description="Relevant theory from RAG")
    candidates: List[CandidateConfig] = Field(min_length=1, max_length=6)


# ==============================================================================
# Pilot Results & Reflection
# ==============================================================================

class PilotResult(BaseModel):
    """Result from a single pilot training run."""
    candidate_index: int
    candidate_name: str
    success: bool
    w1_scores: Optional[List[float]] = None
    tmv_scores: Optional[List[float]] = None
    runtime_sec: Optional[float] = None
    error_summary: Optional[str] = None
    error_type: Optional[Literal["oom", "nan", "convergence", "other"]] = None


class ReflectionAction(str, Enum):
    """Possible actions after pilot reflection."""
    FINALIZE = "finalize"
    RETRY_PILOT = "retry_pilot"
    ABORT = "abort"


class ReflectionDecision(BaseModel):
    """LLM output: decision after reviewing pilot results."""
    action: ReflectionAction
    reasoning: str
    chosen_candidate_index: Optional[int] = Field(default=None, description="Index if finalize")
    new_candidates: Optional[List[CandidateConfig]] = Field(default=None, description="New configs if retry")
    adjustments: Optional[Dict[str, Any]] = Field(default=None, description="Parameter adjustments for retry")


# ==============================================================================
# Downstream Analysis Results
# ==============================================================================

class DownstreamResult(BaseModel):
    """Result from a downstream analysis tool."""
    analysis_type: AnalysisScope
    success: bool
    artifacts: Dict[str, str] = Field(default_factory=dict, description="Paths to generated files")
    summary: str = Field(default="", description="Brief summary of findings")
    error: Optional[str] = None


# ==============================================================================
# Visualization Brief
# ==============================================================================

class VisualizationBrief(TypedDict, total=False):
    """Structured visualization intent passed from Planner to Downstream."""
    viz_goal: Literal["publication", "exploratory", "presentation"]
    figure_requests: List[str]
    grouping: List[str]
    basis_preference: str
    strict_basis: bool
    style: Literal["nature_clean", "default_publication"]
    formats: List[str]
    main_text_figures: int
    include_all_in_appendix: bool
    must_use_existing_results: bool
    force_rerun_analysis_for_plot: bool
    rerun_reason: str
    notes: str


# ==============================================================================
# Biological Insight & Claim Support
# ==============================================================================

class SupportLevel(str, Enum):
    """Level of support for a scientific claim."""
    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    NOT_SUPPORTED = "not_supported"


class ClaimAssessment(BaseModel):
    """Assessment of whether a user claim is supported by the model."""
    claim: str
    support_level: SupportLevel
    evidence: List[str] = Field(default_factory=list, description="Artifact paths or figure refs")
    theory_refs: List[str] = Field(default_factory=list, description="Theory chunk IDs")
    reasoning: str
    limitations: str = Field(default="", description="Caveats and limitations")


class BiologicalInsight(BaseModel):
    """LLM-generated biological insight with evidence."""
    title: str
    description: str
    evidence_artifacts: List[str] = Field(default_factory=list)
    theory_support: str = Field(default="")
    confidence: Literal["high", "medium", "low"] = "medium"


class InsightReport(BaseModel):
    """Complete insight report from the LLM."""
    insights: List[BiologicalInsight] = Field(default_factory=list)
    claim_assessments: List[ClaimAssessment] = Field(default_factory=list)
    overall_limitations: str = Field(default="")


# ==============================================================================
# Final Artifacts
# ==============================================================================

class FinalArtifacts(BaseModel):
    """Paths to all final output artifacts."""
    config_path: str
    h5ad_path: str
    metrics_path: str
    report_path: str
    figures_dir: str
    downstream_results: List[DownstreamResult] = Field(default_factory=list)


# ==============================================================================
# LangGraph State (TypedDict for compatibility)
# ==============================================================================

class AgentState(TypedDict, total=False):
    """
    The global state passed through all LangGraph nodes.
    Use TypedDict for LangGraph compatibility.
    """
    # Input
    input_path: Optional[str]
    pending_input_path: Optional[str]
    converted_path: Optional[str]
    data_switch_note: str
    history_revision: int
    compaction_stats: Dict[str, Any]
    context_policy: Dict[str, Any]
    conversation_turn: int
    planner_phase: Literal["working", "needs_input", "final"]
    planner_need: Dict[str, Any]
    workflow_phase: Literal[
        "intake",
        "preprocessing",
        "theory_selection",
        "training",
        "downstream_analysis",
        "reporting",
        "completed",
    ]
    task_profile: Dict[str, Any]
    phase_status: Dict[str, str]
    artifact_index: Dict[str, Any]
    workflow_audit_log: List[dict]
    tool_catalog_revision: int
    tool_catalog: Dict[str, Any]
    pending_tools: List[dict]
    tool_activation_mode: Literal["auto_next_turn", "manual_review"]
    tool_harvest_enabled: bool
    tool_harvest_report: Dict[str, Any]
    tool_review_report: Dict[str, Any]
    active_skills: List[dict]
    skills_revision: int
    skill_policy: Dict[str, Any]
    planner_loaded_skills: List[dict]
    planner_skills_revision: int
    planner_skill_policy: Dict[str, Any]
    hidden_skills: Dict[str, List[str]]
    hidden_training_algorithms: List[str]
    planner_active_auto_skills: List[dict]
    planner_skill_injection_revision: int
    planner_workspace_policy: Dict[str, Any]
    planner_algorithm_workspace: str
    planner_last_patch_summary: Dict[str, Any]
    algorithm_proposal_review_mode: Literal["always_user_review", "agent_decide", "auto_approve"]
    idea_review_mode: Literal["always_user_review", "agent_decide", "auto_approve"]
    algorithm_proposals: Dict[str, Any]
    latest_algorithm_proposal_id: str
    research_ideas: Dict[str, Any]
    latest_research_idea_id: str
    active_research_idea_id: str
    active_idea_registry: Dict[str, Any]
    decision_log_summary: List[dict]
    obsolete_results_summary: List[dict]
    active_experiment_registry: Dict[str, Any]
    active_algorithm_context: Dict[str, Any]
    active_proposal_id: str
    active_workspace_snapshot_id: str
    active_baseline_run_id: str
    active_algorithm_campaign_id: str
    active_algorithm_campaign: Dict[str, Any]
    runtime_action: Dict[str, Any]
    subagent_registry: Dict[str, Any]
    subagent_counter: int
    subagent_run_log: List[dict]
    figure_quality_preset: Literal["publication", "balanced", "fast"]
    viz_brief_enabled: bool
    report_figure_policy: Literal["main_plus_appendix", "full_main", "main_only"]
    report_figure_snapshot: Dict[str, Any]
    report_policy_warning: str
    report_validation_warnings: List[str]
    viz_default_goal: Literal["publication", "exploratory", "presentation"]
    current_viz_brief: Dict[str, Any]
    umap_warmup_mode: Literal["eager", "lazy"]
    umap_param_mode: Literal["auto", "manual"]
    umap_params_override: Dict[str, Any]
    umap_force_recompute: bool
    user_goal: dict  # UserGoal.model_dump()
    output_dir: str  # Output directory for all results
    
    # Data inspection
    data_summary: dict  # DataSummary.model_dump()
    preprocessed_path: str
    time_key: str
    label_key: Optional[str]
    
    # Preprocessing strategy (LLM-decided)
    preprocessing_strategy: dict  # 预处理策略
    
    # Theory RAG
    theory_context: dict  # TheoryContext.model_dump()
    
    # Planning
    plan_decision: dict  # PlanDecision.model_dump()
    candidates: List[dict]  # List[CandidateConfig.model_dump()]
    execution_plan: str
    planner_mode_plan_draft: str
    execution_plan_items: List[dict]
    execution_plan_explanation: str
    execution_plan_updated_at: str
    downstream_execution_plan: str
    downstream_plan_draft: str
    downstream_execution_plan_items: List[dict]
    downstream_execution_plan_explanation: str
    downstream_execution_plan_updated_at: str
    
    # Pilot loop
    pilot_results: List[dict]  # List[PilotResult.model_dump()]
    reflection_decision: dict  # ReflectionDecision.model_dump()
    retry_count: int
    
    # Final training
    final_config: dict
    final_metrics: dict
    training_runs: List[dict]
    latest_training_run_id: str
    latest_training_run_dir: str
    latest_training_algorithm_id: str
    
    # Downstream
    downstream_results: List[dict]  # List[DownstreamResult.model_dump()]
    
    # Insight & Report
    insight_report: dict  # InsightReport.model_dump()
    final_artifacts: dict  # FinalArtifacts.model_dump()
    
    # Control flow
    current_node: str
    error_log: List[str]
    messages: List[dict]  # Chat history for LLM context
    final_summary: str



# ==============================================================================
# Helper functions
# ==============================================================================

def create_initial_state(
    input_path: Optional[str],
    user_goal: UserGoal,
) -> AgentState:
    """Create initial state for the LangGraph agent."""
    return AgentState(
        input_path=input_path,
        pending_input_path=None,
        converted_path=None,
        data_switch_note="",
        history_revision=0,
        compaction_stats={
            "count": 0,
            "last_at": "",
            "last_before_tokens": 0,
            "last_after_tokens": 0,
            "last_reason": "",
        },
        context_policy={
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
        },
        conversation_turn=0,
        planner_phase="working",
        planner_need={},
        workflow_phase="intake",
        task_profile={
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
        },
        phase_status={
            "preprocessing": "pending",
            "theory_selection": "pending",
            "training": "pending",
            "downstream_analysis": "pending",
            "reporting": "pending",
        },
        artifact_index={},
        workflow_audit_log=[],
        tool_catalog_revision=0,
        tool_catalog={},
        pending_tools=[],
        tool_activation_mode="auto_next_turn",
        tool_harvest_enabled=False,
        tool_harvest_report={},
        tool_review_report={},
        active_skills=[],
        skills_revision=0,
        skill_policy={
            "enabled": True,
            "domain": "downstream",
            "prefer_user_dir": True,
            "inject_max_chars": 32000,
        },
        planner_loaded_skills=[],
        planner_skills_revision=0,
        planner_skill_policy={
            "enabled": True,
            "domain": "planner",
            "prefer_user_dir": True,
            "inject_max_chars": 24000,
        },
        hidden_skills={
            "workflow": [],
            "planner": [],
            "downstream": [],
        },
        hidden_training_algorithms=[],
        planner_active_auto_skills=[],
        planner_skill_injection_revision=0,
        planner_workspace_policy={},
        planner_algorithm_workspace="",
        planner_last_patch_summary={},
        algorithm_proposal_review_mode="agent_decide",
        idea_review_mode="agent_decide",
        algorithm_proposals={},
        latest_algorithm_proposal_id="",
        research_ideas={},
        latest_research_idea_id="",
        active_research_idea_id="",
        active_idea_registry={},
        decision_log_summary=[],
        obsolete_results_summary=[],
        active_experiment_registry={},
        active_algorithm_context={
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
        active_proposal_id="",
        active_workspace_snapshot_id="",
        active_baseline_run_id="",
        active_algorithm_campaign_id="",
        active_algorithm_campaign={},
        subagent_registry={},
        subagent_counter=0,
        subagent_run_log=[],
        figure_quality_preset="publication",
        viz_brief_enabled=True,
        report_figure_policy="main_only",
        report_figure_snapshot={"items": []},
        report_policy_warning="",
        report_validation_warnings=[],
        viz_default_goal="publication",
        current_viz_brief={},
        umap_warmup_mode="lazy",
        umap_param_mode="auto",
        umap_params_override={},
        umap_force_recompute=False,
        user_goal=user_goal.model_dump(),
        data_summary={},
        preprocessed_path="",
        time_key="",
        label_key=None,
        preprocessing_strategy={},
        theory_context={},
        plan_decision={},
        candidates=[],
        execution_plan="",
        planner_mode_plan_draft="",
        execution_plan_items=[],
        execution_plan_explanation="",
        execution_plan_updated_at="",
        downstream_execution_plan="",
        downstream_plan_draft="",
        downstream_execution_plan_items=[],
        downstream_execution_plan_explanation="",
        downstream_execution_plan_updated_at="",
        pilot_results=[],
        reflection_decision={},
        retry_count=0,
        final_config={},
        final_metrics={},
        training_runs=[],
        latest_training_run_id="",
        latest_training_run_dir="",
        latest_training_algorithm_id="",
        downstream_results=[],
        insight_report={},
        final_artifacts={},
        current_node="start",
        error_log=[],
        messages=[],
        final_summary="",
    )
