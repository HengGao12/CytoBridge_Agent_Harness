"""
Unified Analysis Toolkit.

This module merges downstream_tools.py and analysis_toolkit.py into a single toolkit
for a React-style agent to access training metrics and analysis utilities.

Capabilities:
1. Data state inspection
2. Training metrics access
3. Trajectory and fate analysis
4. Driver gene analysis
5. Growth mass analysis
6. Stochasticity analysis
7. Cell interaction analysis
8. Gene regulatory network (GRN)
9. Code execution
10. Visualization helpers
"""
from __future__ import annotations

import csv
import json
import logging
import os
import re
import ast
import select
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable, Dict, Any, List, Optional, Set, Tuple, Union, Literal
import functools
from uuid import uuid4

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from anndata import AnnData
from scipy import sparse
import torch
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from .code_executor import CodeExecutor
from .file_tools import (
    list_path as list_path_impl,
    read_text_file as read_text_file_impl,
    find_files as find_files_impl,
    grep_files as grep_files_impl,
)
from .plan_system import (
    PlanService,
    render_plan_state,
)
from .tool_catalog import ToolCatalog
from .tool_router import GeneratedExecutor, ToolRouter
from .tool_refiner_agent import ToolRefinerAgent
from .skills_tools import SkillsTools
from .downstream_refactor_core import DownstreamRefactorCore, DownstreamResult
from .figure_policy import (
    apply_matplotlib_style,
    ensure_publication_bundle_for_artifacts,
    get_preset_config,
    normalize_preset,
    save_figure_bundle,
    save_plotly_bundle,
)
from .umap_policy import build_umap_policy, compact_policy_signature
from ..utils.llm_runtime import invoke_with_retry

logger = logging.getLogger(__name__)


def _load_adata_or_model_artifact(path: str | Path) -> tuple[AnnData, Path]:
    """Load a full trained h5ad or a compact CytoBridge model artifact."""

    import anndata as ad

    source = Path(path).expanduser()
    if source.suffix.lower() != ".json":
        return ad.read_h5ad(source), source
    try:
        manifest = json.loads(source.read_text(encoding="utf-8"))
    except Exception:
        return ad.read_h5ad(source), source
    if str(manifest.get("artifact_type") or "") != "cytobridge_model_artifact":
        return ad.read_h5ad(source), source
    reference_path = Path(str(manifest.get("reference_adata_path") or manifest.get("input_adata_path") or "")).expanduser()
    if not reference_path.is_file():
        raise FileNotFoundError(f"Model artifact reference AnnData is missing: {reference_path}")
    state_path = Path(str(manifest.get("model_state_path") or "")).expanduser()
    if not state_path.is_file():
        raise FileNotFoundError(f"Model artifact state file is missing: {state_path}")
    payload = torch.load(state_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("all_model"), dict):
        raise ValueError(f"Invalid CytoBridge model artifact state file: {state_path}")
    adata = ad.read_h5ad(reference_path)
    adata.uns["all_model"] = payload["all_model"]
    if "training_summary" in payload:
        adata.uns["training_summary"] = payload["training_summary"]
    adata.uns["model_artifact"] = {
        "manifest_path": str(source),
        "model_state_path": str(state_path),
        "reference_adata_path": str(reference_path),
    }
    return adata, reference_path


# 输出长度限制
MAX_OUTPUT_LENGTH = 3000
DEFAULT_TOOL_CANDIDATE_TAGS = ("# TOOL_CANDIDATE", "@tool_candidate")
TOOL_WORKER_RESULT_PREFIX = "__CYTOBRIDGE_TOOL_WORKER_RESULT__="
WORKER_EVENT_PREFIX = "__CYTOBRIDGE_WORKER_EVENT__="
GENERIC_OBS_KEYS = {
    "cell_type",
    "time",
    "time_point",
    "batch",
    "label",
    "cluster",
    "condition",
}


class _DownstreamPlanItemInput(BaseModel):
    step: str = Field(description="Plan step text.")
    status: Literal["pending", "in_progress", "completed"] = Field(
        description="Step status: pending | in_progress | completed."
    )


class _DownstreamCreatePlanInput(BaseModel):
    plan_text: str = Field(description="Full plan text to parse into structured items.")
    explanation: str = Field(default="", description="Optional explanation.")
    instruction: str = Field(
        default="",
        description="Legacy alias for plan_text; use plan_text by default.",
    )
    reason: str = Field(default="", description="Optional reasoning trace.")


class _DownstreamSetPlanFromTextInput(BaseModel):
    plan_text: str = Field(description="Full structured plan text.")
    explanation: str = Field(default="", description="Optional explanation.")
    reason: str = Field(default="", description="Optional reasoning trace.")


class _DownstreamUpdatePlanInput(BaseModel):
    plan: List[_DownstreamPlanItemInput] = Field(
        description="FULL plan list. Every item must include step + status."
    )
    explanation: str = Field(default="", description="Optional explanation.")
    reason: str = Field(default="", description="Optional reasoning trace.")

def log_reason(func):
    """
    装饰器：自动提取并打印工具调用时的 'reason' 参数。
    """
    import functools
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        # 提取 reason
        reason = kwargs.get('reason', "")
        
        # 打印日志
        if reason:
            print(f"\n{'='*20}  Agent Reasoning {'='*20}")
            print(f"🔧 调用工具: {func.__name__}")
            print(f"💭 思考过程: {reason}")
            print(f"{'='*60}\n")
            
        return func(self, *args, **kwargs)
    return wrapper


def safe_tool(timeout: int = 180):
    """
    装饰器：为慢速/高内存工具添加超时保护。
    
    Args:
        timeout: 超时时间（秒），超过后中断执行。
    """
    import functools
    import gc
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            try:
                from func_timeout import func_timeout as ft, FunctionTimedOut
                return ft(timeout, func, args=(self, *args), kwargs=kwargs)
            except FunctionTimedOut:
                # 尝试清理内存
                gc.collect()
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except:
                    pass
                logger.warning(f"Tool {func.__name__} timed out after {timeout}s")
                return f"⏳ {func.__name__} 执行超时（限制 {timeout}s）。建议：减少数据量、简化分析，或稍后重试。"
            except ImportError:
                # 没装 func_timeout 就直接运行
                return func(self, *args, **kwargs)
        return wrapper
    return decorator

def truncate_output(output: str, max_length: int = MAX_OUTPUT_LENGTH) -> str:
    """Truncate long text output and append a notice.

    Args:
        output: Raw output string to trim.
        max_length: Maximum number of characters to keep before truncation.

    Returns:
        The original output if within limit, otherwise a truncated string with a notice.
    """
    if len(output) <= max_length:
        return output
    return output[:max_length] + f"\n\n... [输出被截断，共 {len(output)} 字符，显示前 {max_length} 字符]"


# ==============================================================================
# Helper Functions (from downstream_tools.py)
# ==============================================================================

def _ensure_dir(path: Path) -> Path:
    """Create a directory if it does not exist and return the path.

    Args:
        path: Directory path to create.

    Returns:
        The same path after ensuring it exists.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def _available_cpu_count(default: int = 4) -> int:
    """Return CPU count available to current process (respecting cpuset affinity)."""
    try:
        affinity = os.sched_getaffinity(0)
        if affinity:
            return max(1, len(affinity))
    except Exception:
        pass
    return max(1, int(os.cpu_count() or default))


def _detect_gene_name_column(adata: AnnData) -> Tuple[List[str], str]:
    """Detect the best gene name column in AnnData.var.

    Args:
        adata: AnnData object to inspect.

    Returns:
        A tuple of (gene_names, source_column). "var_names" is used if no column is found.
    """
    gene_col_candidates = ['gene_symbol', 'gene_name', 'gene', 'symbol', 'Symbol', 'Gene', 'gene_id']
    for col in gene_col_candidates:
        if col in adata.var.columns:
            return list(adata.var[col]), col
    return list(adata.var_names), 'var_names'


def _get_gene_adata(adata: AnnData) -> Tuple[Optional[AnnData], str]:
    """Resolve a gene-level AnnData view for downstream analyses.

    Args:
        adata: AnnData to inspect.

    Returns:
        A tuple of (gene_adata, source). "raw" indicates raw expression was used;
        "latent_only" indicates no gene-level data is available.
    """
    # if adata.raw is not None and adata.raw.n_vars > 0:
    #     gene_adata = adata.raw.to_adata()
    #     gene_adata.obs = adata.obs.copy()
    #     return gene_adata, "raw"

    is_latent_only = False
    if "X_latent" in adata.obsm:
        try:
            is_latent_only = (
                adata.n_vars == adata.obsm["X_latent"].shape[1]
                and "highly_variable" not in adata.var.columns
            )
        except Exception:
            is_latent_only = False

    if is_latent_only:
        return None, "latent_only"
    return adata, "X"


def _get_pca_loadings(adata: AnnData, latent_dim: int) -> Optional[np.ndarray]:
    """Fetch PCA loadings truncated to the latent dimension.

    Args:
        adata: AnnData containing PCA loadings in ``varm["PCs"]``.
        latent_dim: Desired latent dimension.

    Returns:
        The loadings matrix of shape (n_genes, latent_dim) or None when unavailable.
    """
    if "PCs" not in adata.varm:
        return None
    pcs = adata.varm["PCs"]
    if pcs.shape[1] < latent_dim:
        return None
    return pcs[:, :latent_dim].astype(np.float32)


def _get_velocity_latent(adata: AnnData) -> Optional[np.ndarray]:
    """Get latent velocity embeddings from AnnData.

    Args:
        adata: AnnData to inspect.

    Returns:
        Latent velocity array if present in ``obsm`` or ``layers``, otherwise None.
    """
    # Prefer obsm (correct location for latent space data)
    if "velocity_latent" in adata.obsm:
        return np.asarray(adata.obsm["velocity_latent"])
    # Fallback to layers for backward compatibility (but this is incorrect)
    if "velocity_latent" in adata.layers:
        return np.asarray(adata.layers["velocity_latent"])
    return None


def _get_score_latent(adata: AnnData) -> Optional[np.ndarray]:
    """Get latent score embeddings from AnnData.

    Args:
        adata: AnnData to inspect.

    Returns:
        Latent score array if present in ``obsm``, otherwise None.
    """
    if "score_latent" in adata.obsm:
        return np.asarray(adata.obsm["score_latent"])
    if "score" in adata.obsm:
        return np.asarray(adata.obsm["score"])
    return None


def _get_interaction_force(adata: AnnData) -> Optional[np.ndarray]:
    """Get interaction force embeddings from AnnData.

    Args:
        adata: AnnData to inspect.

    Returns:
        Interaction force array if present, otherwise None.
    """
    if "interaction_force" in adata.layers:
        return np.asarray(adata.layers["interaction_force"])
    if "interaction_force" in adata.obsm:
        return np.asarray(adata.obsm["interaction_force"])
    return None


def _mean_projected_vectors(vectors: np.ndarray, W: np.ndarray, batch_size: int = 1024) -> Tuple[np.ndarray, np.ndarray]:
    """Compute mean and mean-absolute projected vectors in batches.

    Args:
        vectors: Latent vectors of shape (n_cells, n_dims).
        W: Projection matrix of shape (n_genes, n_dims).
        batch_size: Number of cells per batch.

    Returns:
        A tuple of (mean_vector, mean_abs_vector) in gene space.
    """
    if vectors.size == 0:
        n_genes = W.shape[0]
        return np.zeros(n_genes, dtype=np.float32), np.zeros(n_genes, dtype=np.float32)

    if batch_size is None or int(batch_size) <= 0:
        # Target ~192MB working tensor for proj = batch x n_genes float32.
        batch_size = max(64, min(4096, int((192 * 1024 * 1024) / max(4, W.shape[0] * 4))))
    else:
        batch_size = int(batch_size)

    n_cells = vectors.shape[0]
    n_genes = W.shape[0]
    sum_vec = np.zeros(n_genes, dtype=np.float64)
    sum_abs = np.zeros(n_genes, dtype=np.float64)
    for start in range(0, n_cells, batch_size):
        end = min(start + batch_size, n_cells)
        proj = np.asarray(vectors[start:end], dtype=np.float32) @ W.T
        sum_vec += proj.sum(axis=0)
        sum_abs += np.abs(proj).sum(axis=0)
    mean_vec = sum_vec / n_cells
    mean_abs = sum_abs / n_cells
    return mean_vec.astype(np.float32), mean_abs.astype(np.float32)


def _has_empty_neighbor_rows(conn: Any) -> bool:
    """Return True when a sparse connectivity matrix contains empty rows."""
    if conn is None:
        return True
    try:
        csr = conn.tocsr()
        row_nnz = np.diff(csr.indptr)
        return bool((row_nnz == 0).any())
    except Exception:
        return True


def _write_gene_summary(
    outdir: Path,
    prefix: str,
    gene_names: List[str],
    mean_vec: np.ndarray,
    mean_abs: np.ndarray,
    top_n: int = 20,
    figure_dir: Optional[Path] = None,
) -> Dict[str, str]:
    """Write gene summary CSV and a top-gene bar plot.

    Args:
        outdir: Output directory for artifacts.
        prefix: Filename prefix for generated artifacts.
        gene_names: List of gene labels aligned to vectors.
        mean_vec: Mean projected values per gene.
        mean_abs: Mean absolute projected values per gene.
        top_n: Number of top genes to plot.
        figure_dir: Optional directory for saving the plot image. If None,
            the plot is saved into ``outdir``.

    Returns:
        Mapping of artifact names to file paths.
    """
    artifacts: Dict[str, str] = {}
    summary_path = outdir / f"{prefix}_gene_summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["gene", "mean", "mean_abs"])
        for gene, mean_val, mean_abs_val in zip(gene_names, mean_vec, mean_abs):
            writer.writerow([gene, f"{mean_val:.6f}", f"{mean_abs_val:.6f}"])
    artifacts[f"{prefix}_gene_summary"] = str(summary_path)

    top_idx = np.argsort(mean_abs)[::-1][: min(top_n, len(gene_names))]
    top_genes = [gene_names[i] for i in top_idx]
    top_vals = mean_abs[top_idx]
    fig, ax = plt.subplots(figsize=(8, max(4, len(top_genes) * 0.3)))
    ax.barh(range(len(top_genes)), top_vals, color="#4c72b0")
    ax.set_yticks(range(len(top_genes)))
    ax.set_yticklabels(top_genes)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |value| in gene space")
    ax.set_title(f"Top {len(top_genes)} genes ({prefix})")
    fig.tight_layout()
    fig_outdir = figure_dir if figure_dir is not None else outdir
    fig_outdir.mkdir(parents=True, exist_ok=True)
    fig_path = fig_outdir / f"{prefix}_gene_top.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    artifacts[f"{prefix}_gene_top"] = str(fig_path)
    return artifacts


# ==============================================================================
# Main Class
# ==============================================================================

class DownstreamAnalysisToolkit:
    """Unified analysis toolkit combining downstream and analysis utilities.

    Capabilities:
    - Data state inspection
    - Training metrics access
    - Trajectory and fate analysis
    - Driver gene analysis
    - Growth mass analysis
    - Stochasticity analysis
    - Cell interaction analysis
    - Gene regulatory network
    - Code execution
    - Visualization helpers
    """

    BUILTIN_TOOL_METHODS = [
        "get_data_summary",
        "get_training_metrics",
        "evaluate_model_quality",
        "analyze_growth_mass",
        "analyze_growth_driver_genes",
        "analyze_velocity_driver_genes",
        "analyze_fate_probabilities",
        "analyze_stochasticity",
        "analyze_interaction",
        "analyze_grn",
        "analyze_vg_driver_genesW",
        "analyze_trajectory_fate",
        "create_plan",
        "set_plan_from_text",
        "update_plan",
        "get_plan_status",
        "execute_python",
        "compute_correlation",
        "get_obs_column",
        "get_gene_expression",
        "search_genes",
        "downsample_data",
        "analyze_gene_perturbation",
        "train_cell_classifier",
        "generate_trajectory_dataset",
        "extract_trajectory_gene_dynamics",
        "plot_cell_lineage_sankey",
        "list_path",
        "read_text_file",
        "find_files",
        "grep_files",
        "list_tool_catalog",
        "set_tool_activation_mode",
        "approve_pending_tools",
        "disable_generated_tool",
        "set_figure_quality_preset",
        "get_figure_quality_preset",
    ]
    SKILL_TOOL_METHODS = []
    SKILLS_MODE_ALLOWED_METHODS = [
        "get_data_summary",
        "get_training_metrics",
        "evaluate_model_quality",
        "create_plan",
        "set_plan_from_text",
        "update_plan",
        "get_plan_status",
        "execute_python",
        "list_path",
        "read_text_file",
        "find_files",
        "grep_files",
        "list_tool_catalog",
        "set_tool_activation_mode",
        "approve_pending_tools",
        "disable_generated_tool",
        "set_figure_quality_preset",
        "get_figure_quality_preset",
    ]
    
    def __init__(
        self,
        adata_path: str,
        training_metrics: Optional[Dict[str, Any]] = None,
        output_dir: Optional[str] = None,
        user_requested_analyses: Optional[List[str]] = None,
        device: str = "cpu",
        question: str = "",
        llm: Any = None,
        shared_state: Optional[Dict[str, Any]] = None,
        event_sink: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> None:
        """Initialize the unified analysis toolkit and load AnnData.

        Args:
            adata_path: Path to the trained ``.h5ad`` file.
            training_metrics: Optional training metrics dictionary.
            output_dir: Optional output directory for artifacts; defaults to ``<adata_dir>/react_agent``.
            user_requested_analyses: Optional list of user-preferred analyses.
            device: Torch device string such as "cpu" or "cuda".
            question: User's goal/question (context for planning).
            llm: Planner LLM instance.
        """
        self.adata_path = Path(adata_path)
        self._subprocess_adata_path = Path(adata_path)
        self._is_worker_process = os.environ.get("CYTOBRIDGE_DISABLE_SUBPROCESS_GUARD") == "1"
        self._adata, self._subprocess_adata_path = _load_adata_or_model_artifact(adata_path)
        self._set_gene_names()
        # print(self.adata.raw) removed
        self.training_metrics = training_metrics or {}
        self.user_requested_analyses = user_requested_analyses or []
        self.device = device
        self.question = question
        self.llm = llm
        self.shared_state = shared_state
        self.event_sink = event_sink
        
        self.output_dir = Path(output_dir) if output_dir else self.adata_path.parent / "react_agent"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir = self.output_dir / "figures"
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self._configure_plotly_chrome()

        # Runtime/memory policy: prefer latent-space analysis on large datasets.
        self.latent_first_default = True
        self.max_dense_projection_bytes = 768 * 1024 * 1024  # avoid accidental huge dense allocations
        self.fast_velocity_n_pcs = 50
        self.fast_velocity_reuse_neighbors = False

        if self._is_worker_process:
            logger.info("Worker process: skip UMAP bootstrap in init.")
        else:
            logger.info("Main process: preparing runtime snapshot (UMAP warmup mode applied).")
        velocity_space, velocity_reason = self._ensure_velocity_layer(prefer_gene=False)
        if velocity_space is not None and "velocity" in self.adata.layers:
            target_dim = self.adata.layers["velocity"].shape[1]
            _ = self._ensure_expression_layer(target_dim)
        else:
            logger.info("Velocity layer not prepared at init: %s", velocity_reason)

        
        # Initialize code executor (will get adata from property)
        self._code_executor = None
        
        # Analysis history and results
        self.analysis_history = []
        self.analysis_results = {}
        self._worker_captured_images: List[str] = []
        self._worker_captured_image_set: Set[str] = set()
        self._worker_stream_buffer: Dict[str, str] = {"tool": ""}
        self._tool_worker_proc: Optional[subprocess.Popen] = None
        self._tool_worker_init_payload: Optional[Path] = None
        if self.shared_state is None:
            self._local_plan_state = {
                "downstream_execution_plan_items": [],
                "downstream_execution_plan_explanation": "",
                "downstream_execution_plan_updated_at": "",
                "downstream_execution_plan": "",
                "downstream_plan_draft": "",
                "tool_catalog": {},
                "pending_tools": [],
                "tool_catalog_revision": 0,
                "tool_activation_mode": "auto_next_turn",
                "tool_harvest_enabled": False,
                "tool_harvest_report": {},
                "tool_review_report": {},
                "active_skills": [],
                "skills_revision": 0,
                "skill_policy": {
                    "enabled": True,
                    "domain": "downstream",
                    "prefer_user_dir": True,
                    "inject_max_chars": 32000,
                },
                "figure_quality_preset": "publication",
                "viz_brief_enabled": True,
                "report_figure_policy": "main_only",
                "viz_default_goal": "publication",
                "current_viz_brief": {},
                "umap_warmup_mode": "lazy",
                "umap_param_mode": "auto",
                "umap_params_override": {},
                "umap_force_recompute": False,
                "tool_subprocess_guard_enabled": True,
                "tool_subprocess_timeout": 300,
                "tool_subprocess_persistent": True,
            }
            state_ref = self._local_plan_state
        else:
            state_ref = self.shared_state
            state_ref.setdefault("downstream_execution_plan_items", [])
            state_ref.setdefault("downstream_execution_plan_explanation", "")
            state_ref.setdefault("downstream_execution_plan_updated_at", "")
            state_ref.setdefault("downstream_execution_plan", "")
            state_ref.setdefault("downstream_plan_draft", "")
            state_ref.setdefault("tool_catalog", {})
            state_ref.setdefault("pending_tools", [])
            state_ref.setdefault("tool_catalog_revision", 0)
            state_ref.setdefault("tool_activation_mode", "auto_next_turn")
            state_ref.setdefault("tool_harvest_enabled", False)
            state_ref.setdefault("tool_harvest_report", {})
            state_ref.setdefault("tool_review_report", {})
            state_ref.setdefault("active_skills", [])
            state_ref.setdefault("skills_revision", 0)
            state_ref.setdefault(
                "skill_policy",
                {
                    "enabled": True,
                    "domain": "downstream",
                    "prefer_user_dir": True,
                    "inject_max_chars": 32000,
                },
            )
            state_ref.setdefault("figure_quality_preset", "publication")
            state_ref.setdefault("viz_brief_enabled", True)
            state_ref.setdefault("report_figure_policy", "main_only")
            state_ref.setdefault("viz_default_goal", "publication")
            state_ref.setdefault("current_viz_brief", {})
            state_ref.setdefault("umap_warmup_mode", "lazy")
            state_ref.setdefault("umap_param_mode", "auto")
            state_ref.setdefault("umap_params_override", {})
            state_ref.setdefault("umap_force_recompute", False)
            state_ref.setdefault("tool_subprocess_guard_enabled", True)
            state_ref.setdefault("tool_subprocess_timeout", 300)
            state_ref.setdefault("tool_subprocess_persistent", True)
        self.figure_quality_preset = normalize_preset(state_ref.get("figure_quality_preset", "publication"))
        apply_matplotlib_style(self.figure_quality_preset)
        warmup_mode = str(state_ref.get("umap_warmup_mode", "lazy") or "lazy").strip().lower()
        if warmup_mode not in {"eager", "lazy"}:
            warmup_mode = "lazy"
        self.umap_warmup_mode = warmup_mode
        state_ref["umap_warmup_mode"] = self.umap_warmup_mode
        self.subprocess_guard_enabled = bool(state_ref.get("tool_subprocess_guard_enabled", True))
        configured_timeout = int(state_ref.get("tool_subprocess_timeout", 300))
        # Migrate old default (3600s) to new default (300s).
        if configured_timeout == 3600:
            configured_timeout = 300
            state_ref["tool_subprocess_timeout"] = 300
        self.subprocess_guard_timeout = configured_timeout
        self.subprocess_persistent = bool(state_ref.get("tool_subprocess_persistent", True))
        # Create a reusable runtime adata snapshot once in the parent process so
        # subprocess tools do not repeatedly reload/rebuild from the raw input file.
        if self.subprocess_guard_enabled and not self._is_worker_process:
            if self.umap_warmup_mode == "eager":
                self._prepare_runtime_snapshot_with_umap()
            else:
                self._emit_event(
                    "status",
                    {"message": "Downstream init: UMAP warmup mode=lazy; skip eager UMAP precompute."},
                )
                self._refresh_subprocess_adata_snapshot(force=True)
        self.plan_service = PlanService(state_ref, event_sink=self.event_sink, namespace="downstream")
        self.tool_catalog = ToolCatalog(state_ref, self.output_dir, event_sink=self.event_sink)
        self.tool_router = ToolRouter(self, self.tool_catalog, self.output_dir, event_sink=self.event_sink)
        self.generated_executor = GeneratedExecutor(self.output_dir)
        self.tool_refiner = ToolRefinerAgent(self.llm) if self.llm else None
        self.skills_tools = SkillsTools(state_ref, domain="downstream", event_sink=self.event_sink)
        self._downstream_core = DownstreamRefactorCore(self.output_dir, device=self.device)
        self._register_builtin_tool_specs()
        
        # Data state cache
        self._data_state = None
        
        logger.info("DownstreamAnalysisToolkit initialized")
        logger.info(f"  AnnData: {self.adata.n_obs} cells × {self.adata.n_vars} genes")
        logger.info(f"  Training metrics available: {len(self.training_metrics) > 0}")
        logger.info(f"  User requested analyses: {self.user_requested_analyses}")

    def close(self) -> None:
        """Terminate persistent subprocess workers."""
        self._stop_persistent_worker("tool")

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _runtime_snapshot_path(self) -> Path:
        return self.output_dir / "_downstream_runtime_adata.h5ad"

    def _refresh_subprocess_adata_snapshot(self, force: bool = False) -> Path:
        """Persist current in-memory adata for subprocess reuse."""
        snap = self._runtime_snapshot_path()
        if self._is_worker_process:
            self._subprocess_adata_path = self.adata_path
            return self._subprocess_adata_path
        if force or not snap.exists():
            try:
                self.adata.write_h5ad(snap)
                self._subprocess_adata_path = snap
                logger.info("Refreshed downstream runtime adata snapshot: %s", snap)
            except Exception:
                logger.warning("Failed to refresh runtime adata snapshot; fallback to original path", exc_info=True)
                self._subprocess_adata_path = self.adata_path
        return self._subprocess_adata_path

    def _prepare_runtime_snapshot_with_umap(self) -> Path:
        """Ensure runtime snapshot exists and includes X_umap when possible."""
        if self._is_worker_process:
            return self._resolve_subprocess_adata_path()
        try:
            if "X_umap" not in self.adata.obsm:
                self._emit_event(
                    "status",
                    {"message": "Downstream init: X_umap missing, computing UMAP once for runtime snapshot..."},
                )
                t0 = time.time()
                self._ensure_embedding(preferred="umap")
                self._emit_event(
                    "status",
                    {"message": f"Downstream init: UMAP ready ({time.time() - t0:.1f}s)."},
                )
            else:
                self._emit_event("status", {"message": "Downstream init: X_umap already present."})
        except Exception as exc:
            logger.warning("Failed to precompute UMAP during downstream init: %s", exc, exc_info=True)
            self._emit_event(
                "status",
                {"message": f"Downstream init: UMAP precompute skipped due to error: {exc}"},
            )
        return self._refresh_subprocess_adata_snapshot(force=True)

    def _resolve_subprocess_adata_path(self) -> Path:
        """Return the best on-disk adata path for subprocess workers."""
        cand = Path(self._subprocess_adata_path)
        if cand.exists():
            return cand
        orig = Path(self.adata_path)
        if orig.exists():
            self._subprocess_adata_path = orig
            return orig
        return self._refresh_subprocess_adata_snapshot(force=True)

    @staticmethod
    def _stringify_for_h5ad(value: Any) -> str:
        """Convert arbitrary python objects to stable strings for h5ad metadata."""
        if value is None:
            return ""
        if isinstance(value, (str, int, float, bool, np.integer, np.floating)):
            return str(value)
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, (dict, list, tuple, set)):
            try:
                return json.dumps(value, ensure_ascii=False, default=str)
            except Exception:
                return str(value)
        return str(value)

    def _sanitize_uns_for_h5ad(self, obj: Any) -> Any:
        """Recursively sanitize ``uns`` to avoid h5ad serialization failures."""
        if obj is None or isinstance(obj, (str, bytes, int, float, bool, np.generic, np.ndarray)):
            return obj
        if sparse.issparse(obj):
            return obj
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, dict):
            return {str(k): self._sanitize_uns_for_h5ad(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._sanitize_uns_for_h5ad(v) for v in obj]
        if isinstance(obj, tuple):
            return [self._sanitize_uns_for_h5ad(v) for v in obj]
        if isinstance(obj, set):
            return [self._sanitize_uns_for_h5ad(v) for v in sorted(list(obj), key=lambda x: str(x))]
        if isinstance(obj, pd.Series):
            s = obj.copy()
            if s.dtype == object:
                s = s.map(self._stringify_for_h5ad)
            return s
        if isinstance(obj, pd.DataFrame):
            df = obj.copy()
            for col in df.columns:
                if df[col].dtype == object:
                    df[col] = df[col].map(self._stringify_for_h5ad)
            return df
        return self._stringify_for_h5ad(obj)

    def _sanitize_adata_metadata_for_h5ad_inplace(self, adata: AnnData) -> Dict[str, int]:
        """Sanitize object-typed metadata fields in-place before writing h5ad."""
        stats = {"obs_cols": 0, "var_cols": 0}
        for col in list(adata.obs.columns):
            try:
                if adata.obs[col].dtype == object:
                    adata.obs[col] = adata.obs[col].map(self._stringify_for_h5ad)
                    stats["obs_cols"] += 1
            except Exception:
                continue
        for col in list(adata.var.columns):
            try:
                if adata.var[col].dtype == object:
                    adata.var[col] = adata.var[col].map(self._stringify_for_h5ad)
                    stats["var_cols"] += 1
            except Exception:
                continue
        try:
            adata.uns = self._sanitize_uns_for_h5ad(dict(adata.uns))
        except Exception:
            logger.debug("Failed to sanitize adata.uns; keeping original", exc_info=True)
        return stats

    def persist_runtime_adata(
        self,
        save_path: Optional[str] = None,
        sanitize_on_error: bool = True,
        reason: str = "",
    ) -> str:
        """Persist current downstream runtime adata once (typically after downstream completion).

        Args:
            save_path: Optional destination path. Defaults to ``<output_dir>/downstream_final_adata.h5ad``.
            sanitize_on_error: If write fails, sanitize metadata and retry once.
            reason: Optional caller reason.

        Returns:
            Human-readable status text.
        """
        guarded = self._invoke_tool_method_in_subprocess(
            method="persist_runtime_adata",
            kwargs={
                "save_path": save_path,
                "sanitize_on_error": bool(sanitize_on_error),
                "reason": reason,
            },
            timeout=max(300, self.subprocess_guard_timeout),
        )
        if guarded is not None:
            return self._consume_guarded_tool_result(guarded, "runtime adata persistence")

        target = Path(save_path).expanduser() if save_path else (self.output_dir / "downstream_final_adata.h5ad")
        if not target.is_absolute():
            target = (self.output_dir / target).resolve()
        else:
            target = target.resolve()
        target.parent.mkdir(parents=True, exist_ok=True)

        try:
            self.adata.write_h5ad(str(target))
            self._subprocess_adata_path = target
            self._data_state = None
            return f" Saved downstream runtime adata: {target}"
        except Exception as first_exc:
            if not sanitize_on_error:
                return f" Failed saving downstream runtime adata: {first_exc}"
            stats = self._sanitize_adata_metadata_for_h5ad_inplace(self.adata)
            try:
                self.adata.write_h5ad(str(target))
                self._subprocess_adata_path = target
                self._data_state = None
                return (
                    f" Saved downstream runtime adata after metadata sanitization: {target}\n"
                    f"Sanitized columns: obs={stats.get('obs_cols', 0)}, var={stats.get('var_cols', 0)}"
                )
            except Exception as second_exc:
                return (
                    " Failed saving downstream runtime adata even after metadata sanitization.\n"
                    f"first_error: {first_exc}\n"
                    f"second_error: {second_exc}"
                )
    
    @property
    def adata(self):
        """Get the toolkit-local AnnData object."""
        return self._adata

    @adata.setter
    def adata(self, value: AnnData) -> None:
        """Update the toolkit-local AnnData object."""
        self._adata = value
    
    @property
    def code_executor(self):
        """Lazy-create CodeExecutor with current adata."""
        if self._code_executor is None:
            self._code_executor = CodeExecutor(
                self.adata,
                self.output_dir,
                figure_quality_preset=self.figure_quality_preset,
            )
        return self._code_executor

    def _emit_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        if not self.event_sink:
            return
        try:
            self.event_sink(event_type, payload)
        except Exception:
            logger.debug("Downstream toolkit event emit failed", exc_info=True)

    def _record_worker_captured_images(self, worker_payload: Dict[str, Any]) -> None:
        captured = worker_payload.get("captured_images")
        if not isinstance(captured, list):
            return
        for raw_path in captured:
            try:
                p = Path(str(raw_path)).expanduser()
                if not p.is_absolute():
                    p = (self.output_dir / p).resolve()
                else:
                    p = p.resolve()
                key = str(p)
            except Exception:
                continue
            if key in self._worker_captured_image_set:
                continue
            self._worker_captured_image_set.add(key)
            self._worker_captured_images.append(key)

    def _worker_prefix(self, worker_type: str) -> str:
        _ = worker_type
        return TOOL_WORKER_RESULT_PREFIX

    def _worker_module(self, worker_type: str) -> str:
        _ = worker_type
        return "cytobridge_agent.tools.downstream_toolkit_worker"

    def _get_worker_proc(self, worker_type: str) -> Optional[subprocess.Popen]:
        _ = worker_type
        return self._tool_worker_proc

    def _set_worker_proc(self, worker_type: str, proc: Optional[subprocess.Popen]) -> None:
        _ = worker_type
        self._tool_worker_proc = proc

    def _set_worker_payload_path(self, worker_type: str, payload_path: Optional[Path]) -> None:
        _ = worker_type
        self._tool_worker_init_payload = payload_path

    def _stop_persistent_worker(self, worker_type: str) -> None:
        proc = self._get_worker_proc(worker_type)
        payload_path = self._tool_worker_init_payload
        if proc is None:
            if payload_path is not None:
                try:
                    payload_path.unlink(missing_ok=True)
                except Exception:
                    pass
                self._set_worker_payload_path(worker_type, None)
            return
        try:
            if proc.poll() is None and proc.stdin:
                try:
                    proc.stdin.write(json.dumps({"command": "shutdown"}, ensure_ascii=False) + "\n")
                    proc.stdin.flush()
                except Exception:
                    pass
                try:
                    proc.wait(timeout=2)
                except Exception:
                    proc.kill()
        except Exception:
            logger.debug("failed stopping %s worker", worker_type, exc_info=True)
        finally:
            self._set_worker_proc(worker_type, None)
            self._set_worker_payload_path(worker_type, None)
            self._worker_stream_buffer[worker_type] = ""
            if payload_path is not None:
                try:
                    payload_path.unlink(missing_ok=True)
                except Exception:
                    pass

    def _start_persistent_worker(self, worker_type: str) -> Optional[Dict[str, Any]]:
        self._stop_persistent_worker(worker_type)
        adata_path = str(self._resolve_subprocess_adata_path())
        init_payload: Dict[str, Any] = {
            "adata_path": adata_path,
            "output_dir": str(self.output_dir),
            "device": str(self.device),
            "umap_warmup": bool(self.umap_warmup_mode == "eager"),
        }
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
                payload_path = Path(f.name)
                json.dump(init_payload, f, ensure_ascii=False)
            env = dict(os.environ)
            env["CYTOBRIDGE_DISABLE_SUBPROCESS_GUARD"] = "1"
            cmd = [
                sys.executable,
                "-m",
                self._worker_module(worker_type),
                "--server",
                "--payload-file",
                str(payload_path),
            ]
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )
            self._set_worker_proc(worker_type, proc)
            self._set_worker_payload_path(worker_type, payload_path)
            return None
        except Exception as exc:
            return {"ok": False, "error": f"failed to start persistent {worker_type} worker: {exc}"}

    def _invoke_via_persistent_worker(
        self,
        worker_type: str,
        request: Dict[str, Any],
        timeout: int,
    ) -> Optional[Dict[str, Any]]:
        if not self.subprocess_persistent or self._is_worker_process:
            return None
        proc = self._get_worker_proc(worker_type)
        if proc is None or proc.poll() is not None:
            start_err = self._start_persistent_worker(worker_type)
            if start_err:
                return start_err
            proc = self._get_worker_proc(worker_type)
        if proc is None or proc.stdin is None or proc.stdout is None:
            return {"ok": False, "error": f"persistent {worker_type} worker unavailable"}

        try:
            proc.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
            proc.stdin.flush()
        except Exception as exc:
            self._stop_persistent_worker(worker_type)
            return {"ok": False, "error": f"failed writing to persistent {worker_type} worker: {exc}"}

        deadline = time.time() + int(max(1, timeout))
        prefix = self._worker_prefix(worker_type)
        tail_lines: List[str] = []
        stream_buf = self._worker_stream_buffer.get(worker_type, "")
        fd = None
        try:
            fd = proc.stdout.fileno()
            os.set_blocking(fd, False)
        except Exception:
            fd = None

        def _handle_line(raw_line: str) -> Optional[Dict[str, Any]]:
            line = raw_line.rstrip("\r")
            if not line:
                return None
            tail_lines.append(line)
            if line.startswith(WORKER_EVENT_PREFIX):
                raw_evt = line[len(WORKER_EVENT_PREFIX):].strip()
                try:
                    evt = json.loads(raw_evt)
                    evt_type = str(evt.get("type") or "status")
                    evt_data = evt.get("data") if isinstance(evt.get("data"), dict) else {}
                    self._emit_event(evt_type, evt_data)
                except Exception:
                    logger.debug("invalid worker event payload: %s", raw_evt[:200], exc_info=True)
                return None
            if not line.startswith(prefix):
                return None
            raw = line[len(prefix):].strip()
            try:
                payload = json.loads(raw)
            except Exception:
                payload = {"ok": False, "error": f"invalid persistent {worker_type} payload: {raw[:300]}"}
            self._record_worker_captured_images(payload)
            return payload

        while time.time() < deadline:
            while "\n" in stream_buf:
                one, stream_buf = stream_buf.split("\n", 1)
                result = _handle_line(one)
                if result is not None:
                    self._worker_stream_buffer[worker_type] = stream_buf
                    return result

            if proc.poll() is not None:
                rc = int(proc.returncode)
                self._stop_persistent_worker(worker_type)
                return {
                    "ok": False,
                    "error": f"persistent {worker_type} worker exited with code {rc}",
                    "stdout_tail": "\n".join(tail_lines[-40:]),
                }
            wait = max(0.05, min(0.3, deadline - time.time()))
            if fd is None:
                # Fallback path if fileno is unavailable.
                time.sleep(wait)
                continue
            ready, _, _ = select.select([fd], [], [], wait)
            if not ready:
                continue
            try:
                chunk = os.read(fd, 65536)
            except BlockingIOError:
                continue
            except Exception as exc:
                self._stop_persistent_worker(worker_type)
                return {"ok": False, "error": f"failed reading persistent {worker_type} worker output: {exc}"}
            if not chunk:
                continue
            decoded = chunk.decode("utf-8", errors="replace").replace("\r", "\n")
            stream_buf += decoded
            if len(stream_buf) > 1_000_000:
                stream_buf = stream_buf[-300_000:]

        self._worker_stream_buffer[worker_type] = stream_buf
        self._stop_persistent_worker(worker_type)
        return {"ok": False, "error": f"persistent {worker_type} worker timeout after {int(timeout)}s"}

    def pop_worker_captured_images(self) -> List[str]:
        """Return and clear image paths captured by subprocess savefig hooks."""
        out = list(self._worker_captured_images)
        self._worker_captured_images.clear()
        self._worker_captured_image_set.clear()
        return out

    def _invoke_core_in_subprocess(
        self,
        method: str,
        kwargs: Dict[str, Any],
        timeout: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Execute a DownstreamRefactorCore method in an isolated subprocess.

        This prevents heavy tool OOM from killing the main agent process.
        Returns:
            Worker payload dict, or None when subprocess guard is disabled.
        """
        if not self.subprocess_guard_enabled:
            return None
        if os.environ.get("CYTOBRIDGE_DISABLE_SUBPROCESS_GUARD") == "1":
            return None
        if not self.subprocess_persistent:
            return {
                "ok": False,
                "error": "persistent subprocess worker is required but disabled",
            }

        request = {
            "method": method,
            "adata_path": str(self._resolve_subprocess_adata_path()),
            "output_dir": str(self.output_dir),
            "device": str(self.device),
            "kwargs": self._json_safe(kwargs),
            "core_method": True,
        }
        return self._invoke_via_persistent_worker(
            worker_type="tool",
            request=request,
            timeout=int(timeout or self.subprocess_guard_timeout),
        )

    def _json_safe(self, obj: Any) -> Any:
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
        if isinstance(obj, dict):
            return {str(k): self._json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple, set)):
            return [self._json_safe(v) for v in obj]
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

    def _invoke_tool_method_in_subprocess(
        self,
        method: str,
        kwargs: Dict[str, Any],
        timeout: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Execute an arbitrary toolkit method in isolated subprocess."""
        if not self.subprocess_guard_enabled:
            return None
        if os.environ.get("CYTOBRIDGE_DISABLE_SUBPROCESS_GUARD") == "1":
            return None
        if not self.subprocess_persistent:
            return {
                "ok": False,
                "error": "persistent subprocess worker is required but disabled",
            }

        request = {
            "method": method,
            "adata_path": str(self._resolve_subprocess_adata_path()),
            "output_dir": str(self.output_dir),
            "kwargs": self._json_safe(kwargs),
            "analysis_results_seed": self._json_safe(self.analysis_results),
        }
        return self._invoke_via_persistent_worker(
            worker_type="tool",
            request=request,
            timeout=int(timeout or self.subprocess_guard_timeout),
        )

    def _consume_guarded_tool_result(self, guarded: Dict[str, Any], label: str) -> str:
        if not guarded.get("ok", False):
            return (
                f" {label} failed in isolated tool process.\n"
                f"error: {guarded.get('error', 'unknown')}\n"
                f"stderr: {str(guarded.get('stderr_tail', ''))[:800]}"
            )
        analysis_results = guarded.get("analysis_results")
        if isinstance(analysis_results, dict):
            self.analysis_results = analysis_results
        analysis_history = guarded.get("analysis_history")
        if isinstance(analysis_history, list):
            self.analysis_history = analysis_history
        updated_adata_path = guarded.get("updated_adata_path")
        if updated_adata_path and Path(str(updated_adata_path)).exists():
            try:
                self._manager.load(str(updated_adata_path))
                self._subprocess_adata_path = Path(str(updated_adata_path)).resolve()
                self._data_state = None
                if self._code_executor is not None:
                    self._code_executor.adata = self.adata
            except Exception:
                logger.warning("failed to sync updated adata from tool worker", exc_info=True)
        return str(guarded.get("text", ""))

    @staticmethod
    def _result_from_guarded_payload(guarded: Dict[str, Any], default_title: str) -> DownstreamResult:
        return DownstreamResult(
            title=str(guarded.get("title", default_title)),
            summary=list(guarded.get("summary") or []),
            artifacts=dict(guarded.get("artifacts") or {}),
            warnings=list(guarded.get("warnings") or []),
            payload=dict(guarded.get("payload") or {}),
        )

    def _configure_plotly_chrome(self) -> None:
        """Configure local Chrome path for Plotly/Kaleido static export."""
        if os.environ.get("BROWSER_PATH"):
            return
        candidates = [
            Path.home() / ".local/share/plotly_chrome/chrome-linux64/chrome",
            Path.home() / ".plotly/chrome/chrome-linux64/chrome",
        ]
        for c in candidates:
            if c.exists():
                os.environ["BROWSER_PATH"] = str(c)
                logger.info("Configured BROWSER_PATH for Plotly image export: %s", c)
                return

    def _register_builtin_tool_specs(self) -> None:
        for method_name in self.BUILTIN_TOOL_METHODS:
            if not hasattr(self, method_name):
                continue
            func = getattr(self, method_name)
            self.tool_catalog.register_builtin(
                name=method_name,
                description=func.__doc__ or f"Execute {method_name}",
                method_name=method_name,
            )

    def get_structured_tools(self) -> List[StructuredTool]:
        tools = self.tool_router.build_structured_tools()
        allowed = set(self.SKILLS_MODE_ALLOWED_METHODS)
        tools = [t for t in tools if t.name in allowed]
        return tools

    @log_reason
    def set_figure_quality_preset(self, preset: str = "publication", reason: str = "") -> str:
        """Set default figure quality preset: publication | balanced | fast."""
        selected = normalize_preset(preset)
        self.figure_quality_preset = selected
        apply_matplotlib_style(selected)
        if self.shared_state is not None:
            self.shared_state["figure_quality_preset"] = selected
        elif hasattr(self, "_local_plan_state"):
            self._local_plan_state["figure_quality_preset"] = selected
        if self._code_executor is not None:
            self._code_executor.figure_quality_preset = selected
        cfg = get_preset_config(selected)
        return f" figure_quality_preset={selected} (dpi={cfg['dpi']}, formats={list(cfg['formats'])})"

    @log_reason
    def get_figure_quality_preset(self, reason: str = "") -> str:
        """Get current default figure quality preset and effective export settings."""
        cfg = get_preset_config(self.figure_quality_preset)
        return json.dumps(
            {
                "figure_quality_preset": self.figure_quality_preset,
                "effective": cfg,
            },
            ensure_ascii=False,
            indent=2,
        )

    def _apply_publication_bundle_to_artifacts(self, artifacts: Dict[str, str]) -> Dict[str, str]:
        return ensure_publication_bundle_for_artifacts(
            artifacts or {},
            preset=self.figure_quality_preset,
            keep_original=True,
        )

    def activate_pending_tools(self, current_turn: int) -> Dict[str, Any]:
        return self.tool_catalog.activate_pending(current_turn=current_turn)

    @log_reason
    def list_tool_catalog(self, reason: str = "") -> str:
        """List builtin/generated tool registry and pending activations."""
        return self.tool_catalog.dump_json()

    @log_reason
    def set_tool_activation_mode(self, mode: str = "auto_next_turn", reason: str = "") -> str:
        """Set generated tool activation mode: auto_next_turn or manual_review."""
        try:
            selected = self.tool_catalog.set_activation_mode(mode)
        except ValueError as e:
            return f" {e}"
        self._emit_event("tool_activation_mode_updated", {"mode": selected})
        return f" tool_activation_mode set to {selected}"

    @log_reason
    def approve_pending_tools(self, tool_ids: Optional[List[str]] = None, reason: str = "") -> str:
        """Approve pending generated tools and activate those already eligible."""
        result = self.tool_catalog.approve_pending(tool_ids=tool_ids)
        current_turn = int((self.shared_state or {}).get("conversation_turn", 0))
        activation = self.tool_catalog.activate_pending(current_turn=current_turn)
        return (
            f" approved={result.get('approved', 0)} | "
            f"activated={len(activation.get('activated', []))}"
        )

    @log_reason
    def disable_generated_tool(self, name_or_id: str, reason: str = "") -> str:
        """Disable an active generated tool by name or tool_id."""
        result = self.tool_catalog.disable_generated(name_or_id)
        if not result.get("ok"):
            return f" {result.get('error', 'Unknown error')}"
        return f" Disabled generated tool: {result.get('name')}"

    @log_reason
    def list_path(
        self,
        path: str = ".",
        recursive: bool = False,
        max_entries: int = 200,
        show_hidden: bool = False,
        reason: str = "",
    ) -> str:
        """List files/folders for a given path."""
        result = list_path_impl(
            path=path,
            recursive=recursive,
            max_entries=max_entries,
            show_hidden=show_hidden,
        )
        self.event_sink(
            "file_activity",
            {
                "scope": "downstream",
                "action": "list_path",
                "path": path,
                "recursive": recursive,
                "max_entries": max_entries,
                "show_hidden": show_hidden,
                "preview": result[:800] + ("\n...[truncated]..." if len(result) > 800 else ""),
            },
        )
        return result

    @log_reason
    def read_text_file(
        self,
        path: str,
        start_line: int = 1,
        end_line: Optional[int] = None,
        max_chars: int = 20000,
        encoding: str = "utf-8",
        reason: str = "",
    ) -> str:
        """Read text file content with line slicing and truncation."""
        result = read_text_file_impl(
            path=path,
            start_line=start_line,
            end_line=end_line,
            max_chars=max_chars,
            encoding=encoding,
        )
        self.event_sink(
            "file_activity",
            {
                "scope": "downstream",
                "action": "read_text_file",
                "path": path,
                "start_line": start_line,
                "end_line": end_line,
                "max_chars": max_chars,
                "preview": result[:1200] + ("\n...[truncated]..." if len(result) > 1200 else ""),
            },
        )
        return result

    @log_reason
    def find_files(
        self,
        pattern: str = "*",
        path: str = ".",
        limit: int = 200,
        recursive: bool = True,
        show_hidden: bool = False,
        timeout: int = 15,
        reason: str = "",
    ) -> str:
        """Find files by glob pattern under a path."""
        result = find_files_impl(
            pattern=pattern,
            path=path,
            limit=limit,
            recursive=recursive,
            show_hidden=show_hidden,
            timeout=timeout,
        )
        self.event_sink(
            "file_activity",
            {
                "scope": "downstream",
                "action": "find_files",
                "path": path,
                "pattern": pattern,
                "limit": limit,
                "recursive": recursive,
                "show_hidden": show_hidden,
                "timeout": timeout,
                "preview": result[:1000] + ("\n...[truncated]..." if len(result) > 1000 else ""),
            },
        )
        return result

    @log_reason
    def grep_files(
        self,
        pattern: str = "",
        include: Optional[str] = None,
        path: str = ".",
        limit: int = 100,
        case_sensitive: bool = False,
        timeout: int = 15,
        reason: str = "",
    ) -> str:
        """Search file contents and return path:line matches."""
        result = grep_files_impl(
            pattern=pattern,
            include=include,
            path=path,
            limit=limit,
            case_sensitive=case_sensitive,
            timeout=timeout,
        )
        self.event_sink(
            "file_activity",
            {
                "scope": "downstream",
                "action": "grep_files",
                "path": path,
                "pattern": pattern,
                "include": include,
                "limit": limit,
                "case_sensitive": case_sensitive,
                "timeout": timeout,
                "preview": result[:1000] + ("\n...[truncated]..." if len(result) > 1000 else ""),
            },
        )
        return result

    def render_skills_context(self, turn_text: str = "") -> str:
        rendered, summary = self.skills_tools.build_turn_context(turn_text="")
        try:
            payload = {
                **summary,
                "scope": "downstream",
            }
            signature = json.dumps(payload, ensure_ascii=False, sort_keys=True)
            if self.state.get("_downstream_skills_context_signature") != signature:
                self.state["_downstream_skills_context_signature"] = signature
                self.event_sink("skills_context_updated", payload)
        except Exception:
            pass
        return rendered

    # =========================================================================
    # 数据状态检测
    # =========================================================================
    @log_reason
    def get_data_state(self, reason: str = "") -> Dict[str, Any]:
        """Summarize dataset state and cache it for reuse.

        Args:
            reason: Detailed reasoning and intent for calling this tool.

        Returns:
            A dictionary describing dataset structure, model availability, and analysis readiness.
        """
        if self._data_state is not None:
            return self._data_state
        
        state = {
            "n_cells": int(self.adata.n_obs),
            "n_genes": int(self.adata.n_vars),
            "var_columns": list(self.adata.var.columns),
            "obs_columns": list(self.adata.obs.columns),
            "obsm_keys": list(self.adata.obsm.keys()),
            "layers_keys": list(self.adata.layers.keys()),
            "uns_keys": list(self.adata.uns.keys()),
            
            # 基因名称信息
            "gene_name_column": None,
            "has_real_gene_names": False,
            
            # 预处理状态
            "has_X_latent": "X_latent" in self.adata.obsm.keys(),
            "has_X_pca": "X_pca" in self.adata.obsm.keys(),
            "has_X_umap": "X_umap" in self.adata.obsm.keys(),
            "has_time_processed": "time_point_processed" in self.adata.obs.columns,
            "has_pca_loadings": "PCs" in self.adata.varm.keys() if hasattr(self.adata, 'varm') else False,
            "has_raw": self.adata.raw is not None,
            
            # 模型状态
            "has_model": "all_model" in self.adata.uns.keys(),
            "model_components": [],
            
            # 分析结果状态
            "has_velocity": "velocity_latent" in self.adata.obsm.keys() or "velocity_latent" in self.adata.layers.keys(),
            "has_growth_rate": "growth_rate" in self.adata.obsm.keys() or "growth_rate" in self.adata.obs.columns,
            "has_score": "score_latent" in self.adata.obsm.keys() or "score" in self.adata.obsm.keys(),
            "has_interaction": "interaction_force" in self.adata.layers.keys() or "interaction_force" in self.adata.obsm.keys(),
            
            # 时间点信息
            "time_points": [],
            "time_key": None,
        }
        
        # 检测基因名称列
        gene_names, gene_col = _detect_gene_name_column(self.adata)
        
        state["gene_name_column"] = gene_col if gene_col != 'var_names' else None
        state["has_real_gene_names"] = gene_col != 'var_names'
        
        # 检测模型组件
        if state["has_model"]:
            try:
                model_config = self.adata.uns["all_model"].get("model_config", {})
                state["model_components"] = list(model_config.get("components", []))
            except:
                pass
        
        # 检测时间点 - 确保转换为标准 Python 类型
        if state["has_time_processed"]:
            time_vals = self.adata.obs["time_point_processed"].unique()
            state["time_points"] = [float(t) for t in sorted(time_vals)]
            state["time_key"] = "time_point_processed"
        
        # 推断模型类型
        state["inferred_model_type"] = self._infer_model_type(state)
        state["gene_space_available"] = state["has_pca_loadings"] and state["has_X_latent"]
        state["recommended_analyses"] = self._recommend_analyses(state)
        
        self._data_state = state
        return state

    @log_reason
    def create_plan(
        self,
        plan_text: str = "",
        explanation: str = "",
        instruction: str = "",
        reason: str = "",
    ) -> str:
        """
        Parse and store a downstream analysis plan supplied by the model.
        This tool does not call LLM internally.

        Use `plan_text` as primary payload.
        `instruction` is accepted for backward compatibility and treated as plan text when `plan_text` is empty.
        """
        from ..display import DisplayManager
        
        display = DisplayManager()
        display.print_stage("planning", "Structuring provided downstream plan...")

        plan_source = str(plan_text or "").strip() or str(instruction or "").strip()
        if not plan_source:
            return (
                "Error: `create_plan` now requires `plan_text` (or legacy `instruction`) "
                "containing the full plan text to parse."
            )

        plan_state, parse_err = self.plan_service.set_from_text(
            plan_source,
            explanation=explanation or "Structured from provided downstream create_plan payload",
            source="downstream_create_plan",
        )
        if parse_err:
            return f"Error: failed to parse provided plan_text: {parse_err}"

        assert plan_state is not None
        return f" **Plan Generated**:\n{render_plan_state(plan_state)}\n\nPlease proceed with Step 1."

    @log_reason
    def set_plan_from_text(self, plan_text: str, explanation: str = "", reason: str = "") -> str:
        """Parse and store a structured plan from free text."""
        plan_state, err = self.plan_service.set_from_text(
            plan_text,
            explanation=explanation,
            source="downstream_set_plan_from_text",
        )
        if err:
            return err
        assert plan_state is not None
        return f" Plan stored.\n{render_plan_state(plan_state)}"

    @log_reason
    def update_plan(self, plan: List[Dict[str, Any]], explanation: str = "", reason: str = "") -> str:
        """Replace current downstream plan with a FULL structured plan list.

        Requirements:
        - `plan` must include every step (not partial patch).
        - each item must include non-empty `step` and valid `status`
          (`pending` | `in_progress` | `completed`).
        """
        plan_state, err = self.plan_service.update(
            plan,
            explanation=explanation,
            source="downstream_update_plan",
        )
        if err:
            return (
                f"{err}\n"
                "Hint: update_plan requires the FULL plan list with explicit "
                "`step` + `status` for each item."
            )
        assert plan_state is not None
        return f" Plan updated.\n{render_plan_state(plan_state)}"

    @log_reason
    def get_plan_status(self, reason: str = "") -> str:
        """Show current downstream plan progress."""
        return self.plan_service.render()

    def _extract_marked_code_candidates(self, messages: List[BaseMessage]) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []
        marker_pattern = re.compile(r"(?im)^\\s*#\\s*TOOL_CANDIDATE(?:\\s*:\\s*([a-zA-Z0-9_\\-]+))?")

        for msg in messages:
            if not isinstance(msg, AIMessage):
                continue
            tool_calls = getattr(msg, "tool_calls", None) or []
            for tc in tool_calls:
                if tc.get("name") != "execute_python":
                    continue
                args = tc.get("args") or {}
                code = args.get("code", "")
                if not code:
                    continue

                has_marker = any(tag.lower() in code.lower() for tag in DEFAULT_TOOL_CANDIDATE_TAGS)
                if not has_marker:
                    continue

                hinted_name = None
                m = marker_pattern.search(code)
                if m:
                    hinted_name = m.group(1)
                candidates.append({"code": code, "hinted_name": hinted_name})
        return candidates

    def _extract_json_block(self, text: str) -> Dict[str, Any]:
        text = (text or "").strip()
        if not text:
            return {}
        fenced = re.findall(r"```(?:json)?\\s*(\\{.*?\\})\\s*```", text, flags=re.S)
        for block in fenced:
            try:
                return json.loads(block)
            except Exception:
                continue
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except Exception:
                return {}
        return {}

    def _draft_candidate_from_code(self, code: str, hinted_name: Optional[str], idx: int) -> Dict[str, Any]:
        default_name = hinted_name or f"reusable_tool_{idx + 1}"
        fallback = {
            "name": default_name,
            "description": "Reusable generated tool",
            "function_name": "run_tool",
            "args_schema": {"type": "object", "properties": {}},
            "code": code,
            "usage": "Call with validated keyword arguments.",
            "rationale": "Harvested from execute_python tool call.",
            "minimal_tests": [],
        }
        if not self.llm:
            return fallback

        prompt = f"""
You are extracting a reusable tool from a code snippet.
Return JSON ONLY with keys:
- name
- description
- function_name
- args_schema (JSON schema object)
- code (complete Python function code, include imports if needed)
- usage
- rationale
- minimal_tests (list)

Hard constraints:
1. Data agnostic and reusable.
2. No dataset-specific constants or hardcoded absolute paths.
3. Parameterize all variable inputs.
4. Keep only one primary function.

Candidate snippet:
```python
{code}
```
""".strip()
        try:
            resp = invoke_with_retry(
                self.llm,
                [HumanMessage(content=prompt)],
                logger,
                "downstream.toolkit.draft_candidate",
            )
            parsed = self._extract_json_block(getattr(resp, "content", ""))
            if not parsed:
                return fallback
            merged = dict(fallback)
            merged.update(parsed)
            return merged
        except Exception:
            logger.debug("Failed to draft tool candidate from code", exc_info=True)
            return fallback

    def _static_generalization_check(self, candidate: Dict[str, Any]) -> List[str]:
        errors: List[str] = []
        code = candidate.get("code", "") or ""
        fn_name = candidate.get("function_name", "") or ""

        if not code.strip():
            errors.append("Empty code body.")
            return errors

        if re.search(r"""['\\"]([A-Za-z]:\\\\|/)""", code):
            errors.append("Contains hardcoded absolute path literal.")

        dataset_name = self.adata_path.stem.lower()
        dataset_tokens = [tok for tok in re.split(r"[^a-z0-9]+", dataset_name) if len(tok) >= 4]
        lowered_code = code.lower()
        for tok in dataset_tokens:
            if tok and tok in lowered_code:
                errors.append(f"Contains dataset-specific token: {tok}")
                break

        try:
            tree = ast.parse(code)
        except Exception as e:
            errors.append(f"Code is not valid Python: {e}")
            return errors

        fn_nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
        if not fn_nodes:
            errors.append("No function definition found.")
            return errors

        function_names = {n.name for n in fn_nodes}
        if fn_name and fn_name not in function_names:
            errors.append(f"Declared function_name '{fn_name}' not found in code.")

        for fn in fn_nodes:
            arg_names = {arg.arg for arg in fn.args.args}
            for match in re.findall(r"""obs\[['"]([^'"]+)['"]\]""", ast.get_source_segment(code, fn) or ""):
                if match in GENERIC_OBS_KEYS:
                    continue
                if match in arg_names:
                    continue
                errors.append(f"Hardcoded obs column detected: {match}")
        return errors

    def _build_synthetic_args(self, schema: Dict[str, Any], variant: int) -> Dict[str, Any]:
        properties = schema.get("properties", {}) if isinstance(schema, dict) else {}
        required = set(schema.get("required", [])) if isinstance(schema, dict) else set()

        args: Dict[str, Any] = {}
        for name, prop in properties.items():
            prop = prop or {}
            typ = prop.get("type", "string")
            if variant == 0:
                if typ == "string":
                    value = prop.get("default", f"{name}_a")
                elif typ == "integer":
                    value = int(prop.get("default", 1))
                elif typ == "number":
                    value = float(prop.get("default", 1.5))
                elif typ == "boolean":
                    value = bool(prop.get("default", True))
                elif typ == "array":
                    value = prop.get("default", [1, 2, 3])
                elif typ == "object":
                    value = prop.get("default", {"a": 1})
                else:
                    value = prop.get("default", None)
            else:
                if typ == "string":
                    value = f"{name}_b"
                elif typ == "integer":
                    value = 2
                elif typ == "number":
                    value = 2.5
                elif typ == "boolean":
                    value = False
                elif typ == "array":
                    value = [3, 4, 5]
                elif typ == "object":
                    value = {"b": 2}
                else:
                    value = None
            if name in required or value is not None:
                args[name] = value
        return args

    def _run_synthetic_tests(self, candidate: Dict[str, Any]) -> Tuple[bool, List[str], Optional[str]]:
        code = candidate.get("code", "") or ""
        function_name = candidate.get("function_name", "run_tool")
        if not code:
            return False, ["No code to run synthetic tests."], None

        eval_dir = self.output_dir / "generated_tools" / "_eval"
        eval_dir.mkdir(parents=True, exist_ok=True)
        module_path = eval_dir / f"_candidate_{uuid4().hex}.py"
        module_path.write_text(code, encoding="utf-8")

        spec = {
            "name": candidate.get("name", "generated_candidate"),
            "function_name": function_name,
            "code_path": str(module_path),
        }
        schema = candidate.get("args_schema") or {"type": "object", "properties": {}}
        args1 = self._build_synthetic_args(schema, variant=0)
        args2 = self._build_synthetic_args(schema, variant=1)

        out1 = self.generated_executor.execute(spec, args1)
        if out1.startswith(""):
            return False, [f"Synthetic test #1 failed: {out1}"], str(module_path)

        out2 = self.generated_executor.execute(spec, args2)
        if out2.startswith(""):
            return False, [f"Synthetic test #2 failed: {out2}"], str(module_path)

        return True, [], str(module_path)

    def _run_generalization_gate(self, candidate: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        static_errors = self._static_generalization_check(candidate)
        if static_errors:
            return False, {"static_errors": static_errors, "synthetic_errors": []}

        passed, syn_errors, _ = self._run_synthetic_tests(candidate)
        return passed, {"static_errors": [], "synthetic_errors": syn_errors}

    def harvest_reusable_tools(self, messages: List[BaseMessage], current_turn: int) -> Dict[str, Any]:
        if not (self.shared_state or {}).get("tool_harvest_enabled", False):
            return {"harvested": 0, "queued": 0, "rejected": 0, "reason": "tool_harvest_disabled"}

        blocks = self._extract_marked_code_candidates(messages)
        if not blocks:
            report = {"harvested": 0, "queued": 0, "rejected": 0, "reason": "no_marked_candidates"}
            if self.shared_state is not None:
                self.shared_state["tool_harvest_report"] = report
            return report

        queued_items: List[Dict[str, Any]] = []
        rejected_items: List[Dict[str, Any]] = []

        for idx, block in enumerate(blocks):
            self._emit_event(
                "tool_candidate_proposed",
                {"index": idx, "hinted_name": block.get("hinted_name"), "turn": current_turn},
            )
            candidate = self._draft_candidate_from_code(block["code"], block.get("hinted_name"), idx)
            if self.tool_refiner:
                candidate = self.tool_refiner.refine(candidate)
                self._emit_event(
                    "tool_refined",
                    {"index": idx, "name": candidate.get("name", f"candidate_{idx + 1}")},
                )

            passed, gate_report = self._run_generalization_gate(candidate)
            if not passed:
                rejected_items.append(
                    {
                        "name": candidate.get("name", f"candidate_{idx + 1}"),
                        "gate_report": gate_report,
                    }
                )
                self._emit_event(
                    "tool_gate_failed",
                    {
                        "name": candidate.get("name", f"candidate_{idx + 1}"),
                        "gate_report": gate_report,
                    },
                )
                continue

            queued = self.tool_catalog.queue_candidate(candidate, current_turn=current_turn)
            queued_items.append({"tool_id": queued["tool_id"], "name": queued["name"]})
            self._emit_event(
                "tool_gate_passed",
                {
                    "tool_id": queued["tool_id"],
                    "name": queued["name"],
                    "eligible_turn": queued["eligible_turn"],
                },
            )

        report = {
            "harvested": len(blocks),
            "queued": len(queued_items),
            "rejected": len(rejected_items),
            "queued_items": queued_items,
            "rejected_items": rejected_items,
            "activation_mode": self.tool_catalog.get_activation_mode(),
        }
        if self.shared_state is not None:
            self.shared_state["tool_harvest_report"] = report
            self.shared_state["tool_review_report"] = {
                "queued_items": queued_items,
                "rejected_items": rejected_items,
                "turn": current_turn,
            }
        return report

    def _runtime_state_ref(self) -> Dict[str, Any]:
        if isinstance(self.shared_state, dict):
            return self.shared_state
        return getattr(self, "_local_plan_state", {})

    def _resolve_umap_policy(self, use_rep: Optional[str]) -> Dict[str, Any]:
        state_ref = self._runtime_state_ref()
        mode = str(state_ref.get("umap_param_mode", "auto") or "auto").strip().lower()
        if mode not in {"auto", "manual"}:
            mode = "auto"
        overrides = state_ref.get("umap_params_override", {})
        if not isinstance(overrides, dict):
            overrides = {}
        brief = state_ref.get("current_viz_brief", {})
        if not isinstance(brief, dict):
            brief = {}
        viz_goal = str(brief.get("viz_goal") or state_ref.get("viz_default_goal", "publication") or "publication")
        return build_umap_policy(
            n_obs=int(self.adata.n_obs),
            use_rep=use_rep,
            quality_preset=self.figure_quality_preset,
            viz_goal=viz_goal,
            mode=mode,
            overrides=overrides,
        )

    def _select_representation(self, prefer: Optional[str] = None) -> str:
        """Select a representation key for neighborhood graph computation.

        Args:
            prefer: Preferred representation key (e.g., "X", "X_pca", "X_latent", or "umap" without prefix).

        Returns:
            The chosen representation key to use in ``adata.obsm`` or "X".
        """
        if prefer:
            if prefer in self.adata.obsm:
                return prefer
            if not prefer.startswith("X_") and f"X_{prefer}" in self.adata.obsm:
                return f"X_{prefer}"
            if prefer.upper() == "X":
                return "X"
        if "X_latent" in self.adata.obsm:
            return "X_latent"
        if "X_pca" in self.adata.obsm:
            return "X_pca"
        return "X"

    def _ensure_neighbors(
        self,
        use_rep: Optional[str] = None,
        n_neighbors: int = 30,
        force: bool = False,
        metric: Optional[str] = None,
    ) -> str:
        """Ensure a valid neighbors graph exists for the selected representation.

        Args:
            use_rep: Optional representation key to use; defaults to an inferred representation.
            n_neighbors: Number of neighbors for graph construction.
            force: Whether to recompute neighbors even if present.
            metric: Distance metric for neighbor graph.

        Returns:
            The representation key used to compute neighbors.
        """
        rep = self._select_representation(use_rep)
        desired_n_neighbors = int(max(5, n_neighbors))
        desired_metric = str(metric or "euclidean").strip().lower() or "euclidean"
        needs_neighbors = force

        conn = self.adata.obsp.get("connectivities") if hasattr(self.adata, "obsp") else None
        if "neighbors" not in self.adata.uns or conn is None:
            needs_neighbors = True
        else:
            try:
                if conn.shape[0] != self.adata.n_obs or conn.shape[1] != self.adata.n_obs:
                    needs_neighbors = True
                elif sparse.issparse(conn):
                    if np.any(np.asarray(conn.getnnz(axis=1)) == 0):
                        needs_neighbors = True
                else:
                    row_sums = np.asarray(conn.sum(axis=1)).ravel()
                    if np.any(row_sums == 0):
                        needs_neighbors = True
            except Exception:
                needs_neighbors = True

        if not needs_neighbors:
            params = {}
            try:
                params = dict((self.adata.uns.get("neighbors") or {}).get("params") or {})
            except Exception:
                params = {}
            existing_rep = str(params.get("use_rep") or "X")
            if existing_rep == "None":
                existing_rep = "X"
            existing_metric = str(params.get("metric") or "")
            try:
                existing_n_neighbors = int(params.get("n_neighbors")) if params.get("n_neighbors") is not None else None
            except Exception:
                existing_n_neighbors = None
            if existing_rep != rep:
                needs_neighbors = True
            if existing_n_neighbors is not None and existing_n_neighbors != desired_n_neighbors:
                needs_neighbors = True
            if existing_metric and existing_metric != desired_metric:
                needs_neighbors = True

        if needs_neighbors:
            if rep == "X":
                sc.pp.neighbors(self.adata, n_neighbors=desired_n_neighbors, metric=desired_metric)
            else:
                sc.pp.neighbors(
                    self.adata,
                    n_neighbors=desired_n_neighbors,
                    use_rep=rep,
                    metric=desired_metric,
                )
            self.adata.uns["_cytobridge_neighbors_config"] = {
                "use_rep": rep,
                "n_neighbors": int(desired_n_neighbors),
                "metric": desired_metric,
            }
        return rep

    @staticmethod
    def _normalize_basis_name(name: Optional[str]) -> Optional[str]:
        if name is None:
            return None
        basis = str(name).strip().lower()
        if basis.startswith("x_"):
            basis = basis[2:]
        if basis == "sring":
            basis = "spring"
        return basis or None

    def _has_embedding(self, basis: str) -> bool:
        key = f"X_{basis}"
        if key not in self.adata.obsm:
            return False
        try:
            return int(self.adata.obsm[key].shape[1]) >= 2
        except Exception:
            return False

    def _available_plot_bases(self) -> List[str]:
        bases: List[str] = []
        for key in self.adata.obsm.keys():
            k = str(key)
            if not k.startswith("X_"):
                continue
            basis = k[2:]
            if self._has_embedding(basis):
                bases.append(basis)
        return sorted(set(bases))

    def _select_plot_basis(self, preferred: Optional[str] = None, strict: bool = False) -> str:
        """Select a 2D embedding basis for plotting.

        Args:
            preferred: Preferred basis name ("umap", "pca", "latent", or custom ``X_<basis>``).
            strict: If ``True``, custom basis requests fail fast instead of falling back to UMAP.

        Returns:
            The selected basis name used for plotting.
        """
        basis = self._normalize_basis_name(preferred)
        if basis:
            if basis == "umap":
                return "umap"
            if basis == "pca":
                return "pca"
            if basis == "latent":
                return "latent"
            if basis == "fast":
                return "fast"
            if self._has_embedding(basis):
                return basis
            if strict and basis not in {"umap", "pca", "latent", "fast"}:
                raise ValueError(
                    f"Requested plot basis '{basis}' not found. Available bases: {self._available_plot_bases() or ['(none)']}"
                )
        if "X_umap" in self.adata.obsm:
            return "umap"
        # Auto-prefer UMAP fallback when latent/PCA exists: _ensure_embedding()
        # will compute X_umap from the best available representation.
        if self._has_embedding("latent"):
            return "umap"
        if self._has_embedding("pca"):
            return "umap"
        return "umap"

    def _ensure_embedding(self, preferred: Optional[str] = None, strict: bool = False) -> str:
        """Ensure the requested embedding exists and return its basis name.

        Args:
            preferred: Preferred basis name ("umap", "pca", "latent", or custom ``X_<basis>``).
            strict: If ``True``, custom basis requests fail fast instead of falling back.

        Returns:
            The basis name that is available and ensured.
        """
        requested_basis = self._normalize_basis_name(preferred)
        basis = self._select_plot_basis(preferred, strict=strict)
        if basis == "latent":
            if not self._has_embedding("latent"):
                if strict and requested_basis == "latent":
                    raise ValueError("Requested plot basis 'latent' is unavailable (X_latent missing or <2D).")
                basis = "umap"
        if basis == "fast":
            if not self._has_embedding("fast"):
                if self._has_embedding("latent"):
                    self.adata.obsm["X_fast"] = np.asarray(self.adata.obsm["X_latent"], dtype=np.float32)[:, :2]
                elif self._has_embedding("pca"):
                    self.adata.obsm["X_fast"] = np.asarray(self.adata.obsm["X_pca"], dtype=np.float32)[:, :2]
                elif self._has_embedding("umap"):
                    self.adata.obsm["X_fast"] = np.asarray(self.adata.obsm["X_umap"], dtype=np.float32)[:, :2]
                elif strict and requested_basis == "fast":
                    raise ValueError("Requested plot basis 'fast' is unavailable (no latent/pca/umap to derive X_fast).")
                else:
                    basis = "umap"
        if basis == "pca":
            if "X_pca" not in self.adata.obsm:
                sc.pp.pca(self.adata)
        if basis == "umap":
            rep_for_policy = self._select_representation()
            policy = self._resolve_umap_policy(use_rep=rep_for_policy)
            signature = compact_policy_signature(policy)
            current_cfg = self.adata.uns.get("_cytobridge_umap_config")
            force_recompute = bool(self._runtime_state_ref().get("umap_force_recompute", False))
            need_compute = ("X_umap" not in self.adata.obsm) or force_recompute
            if not need_compute and isinstance(current_cfg, dict):
                # Respect existing UMAP by default; only refresh when explicitly forced.
                need_compute = False
            if need_compute:
                neighbors_cfg = policy.get("neighbors", {}) if isinstance(policy, dict) else {}
                umap_cfg = policy.get("umap", {}) if isinstance(policy, dict) else {}
                rep = str(neighbors_cfg.get("use_rep") or rep_for_policy or "X")
                n_neighbors = int(neighbors_cfg.get("n_neighbors", 30))
                metric = str(neighbors_cfg.get("metric", "euclidean"))
                self._ensure_neighbors(
                    use_rep=rep,
                    n_neighbors=n_neighbors,
                    force=force_recompute,
                    metric=metric,
                )
                sc.tl.umap(
                    self.adata,
                    min_dist=float(umap_cfg.get("min_dist", 0.3)),
                    spread=float(umap_cfg.get("spread", 1.0)),
                    random_state=int(umap_cfg.get("random_state", 0)),
                )
                self.adata.uns["_cytobridge_umap_config"] = signature
                if force_recompute:
                    self._runtime_state_ref()["umap_force_recompute"] = False
            elif "X_umap" in self.adata.obsm and not isinstance(current_cfg, dict):
                # Preserve provenance even for externally supplied UMAP embeddings.
                self.adata.uns["_cytobridge_umap_config"] = {
                    "version": 1,
                    "mode": "external",
                    "use_rep": "",
                    "n_neighbors": -1,
                    "metric": "",
                    "min_dist": -1.0,
                    "spread": -1.0,
                    "random_state": -1,
                }
        if basis not in {"umap", "pca", "latent", "fast"} and not self._has_embedding(basis):
            if strict and requested_basis == basis:
                raise ValueError(
                    f"Requested plot basis '{basis}' is unavailable after embedding checks. "
                    f"Available bases: {self._available_plot_bases() or ['(none)']}"
                )
            basis = "umap"
        return basis

    def _resolve_viz_basis_preference(
        self,
        preferred: Optional[str] = None,
        strict: bool = False,
    ) -> Tuple[Optional[str], bool]:
        basis = self._normalize_basis_name(preferred)
        strict_basis = bool(strict)
        if basis:
            return basis, strict_basis
        runtime_state = self.shared_state if isinstance(self.shared_state, dict) else getattr(self, "_local_plan_state", {})
        brief = runtime_state.get("current_viz_brief") if isinstance(runtime_state, dict) else {}
        if not isinstance(brief, dict):
            return None, strict_basis
        inferred = self._normalize_basis_name(brief.get("basis_preference"))
        if not inferred:
            return None, strict_basis
        return inferred, (strict_basis or bool(brief.get("strict_basis", False)))

    def _get_gene_projection_stats(self, vectors: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Project latent vectors to gene space and compute summary statistics.

        Args:
            vectors: Latent vectors of shape (n_cells, n_dims).

        Returns:
            A tuple of (mean_vector, mean_abs_vector) in gene space, or None if unavailable.
        """
        W = _get_pca_loadings(self.adata, vectors.shape[1])
        if W is None:
            return None
        return _mean_projected_vectors(vectors, W)

    def _project_latent_to_gene(self, vectors: np.ndarray) -> Optional[np.ndarray]:
        """Project latent vectors into gene space using PCA loadings.

        Args:
            vectors: Latent vectors of shape (n_cells, n_dims).

        Returns:
            Projected gene-space array or None if PCA loadings are unavailable.
        """
        W = _get_pca_loadings(self.adata, vectors.shape[1])
        if W is None:
            return None
        n_cells = int(vectors.shape[0])
        n_genes = int(W.shape[0])
        est_bytes = n_cells * n_genes * 4  # float32
        if est_bytes > int(getattr(self, "max_dense_projection_bytes", 768 * 1024 * 1024)):
            logger.info(
                "Skip dense latent->gene projection: estimated %.2f GB exceeds limit.",
                est_bytes / (1024 ** 3),
            )
            return None
        return np.asarray(vectors, dtype=np.float32) @ W.T

    def _ensure_expression_layer(self, target_dim: int) -> Optional[str]:
        """Ensure an expression layer aligned to the target dimension exists.

        Args:
            target_dim: Expected number of features for the expression layer.

        Returns:
            The source of the expression layer ("Ms", "raw", "X_latent", or "X"),
            or None if no compatible layer can be created.
        """
        if "Ms" in self.adata.layers and self.adata.layers["Ms"].shape[1] == target_dim:
            return "Ms"
        gene_adata, gene_source = _get_gene_adata(self.adata)
        if gene_adata is not None and gene_adata.n_vars == target_dim:
            self.adata.layers["Ms"] = gene_adata.X
            return gene_source
        if "X_latent" in self.adata.obsm and self.adata.obsm["X_latent"].shape[1] == target_dim:
            self.adata.layers["Ms"] = self.adata.obsm["X_latent"]
            return "X_latent"
        if self.adata.X is not None and self.adata.X.shape[1] == target_dim:
            self.adata.layers["Ms"] = self.adata.X
            return "X"
        return None

    def _ensure_velocity_layer(self, prefer_gene: bool = False) -> Tuple[Optional[str], str]:
        """Ensure a velocity layer exists and indicate its space.

        Args:
            prefer_gene: Whether to prioritize gene-space velocity when possible.

        Returns:
            A tuple of (space, source_reason) where space is "gene", "latent", or None.
        """
        gene_adata, _ = _get_gene_adata(self.adata)
        gene_dim = gene_adata.n_vars if gene_adata is not None else None

        v_fallback = None
        if "velocity" in self.adata.layers:
            v = self.adata.layers["velocity"]
            v_dim = v.shape[1]
            if gene_dim is not None and v_dim == gene_dim:
                return "gene", "velocity_layer"
            if not prefer_gene:
                return "latent", "velocity_layer"
            v_fallback = v

        v_latent = _get_velocity_latent(self.adata)
        if v_latent is None:
            if v_fallback is not None:
                return "latent", "velocity_layer"
            return None, "velocity_missing"

        if v_latent.shape[1] == self.adata.n_vars:
            self.adata.layers["velocity"] = v_latent
            return "gene", "velocity_latent"

        v_gene = self._project_latent_to_gene(v_latent) if prefer_gene else None
        if v_gene is not None:
            self.adata.layers["velocity"] = v_gene
            return "gene", "velocity_latent_pca"

        # Do NOT write latent velocity into .layers when dimensionality != n_vars.
        # Keep it in obsm only to avoid AnnData shape errors and huge dense allocations.
        self.adata.obsm["_velocity_runtime_latent"] = np.asarray(v_latent, dtype=np.float32)
        return "latent", "velocity_latent"

    def _fast_velocity_graph_via_pca(self, vkey: str = "velocity", n_pcs: int = 50, n_jobs: Optional[int] = None) -> Tuple[bool, str]:
        """Compute scVelo velocity graph in a lightweight latent/PCA space.

        This mirrors the retina notebook approach: use a compact AnnData with
        low-dimensional representation + copied neighbors, then transfer graph
        results back to the main AnnData object.
        """
        try:
            import scvelo as scv
        except Exception as exc:
            return False, f"scvelo unavailable: {exc}"

        # Prefer latent representation when velocity_latent is available.
        X_rep = None
        V_rep = None
        rep_label = ""

        v_latent = _get_velocity_latent(self.adata)
        if "X_latent" in self.adata.obsm and v_latent is not None:
            X_lat = np.asarray(self.adata.obsm["X_latent"], dtype=np.float32)
            V_lat = np.asarray(v_latent, dtype=np.float32)
            if X_lat.shape[1] == V_lat.shape[1]:
                X_rep = X_lat
                V_rep = V_lat
                rep_label = "latent"

        if X_rep is None and "X_pca" in self.adata.obsm and v_latent is not None:
            X_pca = np.asarray(self.adata.obsm["X_pca"], dtype=np.float32)
            V_lat = np.asarray(v_latent, dtype=np.float32)
            target_dim = min(int(n_pcs), X_pca.shape[1], V_lat.shape[1])
            if target_dim >= 2:
                X_rep = X_pca[:, :target_dim]
                V_rep = V_lat[:, :target_dim]
                rep_label = "pca"

        if X_rep is None and "velocity" in self.adata.layers:
            V = np.asarray(self.adata.layers["velocity"], dtype=np.float32)
            if "X_pca" in self.adata.obsm:
                X_pca = np.asarray(self.adata.obsm["X_pca"], dtype=np.float32)
                target_dim = min(int(n_pcs), X_pca.shape[1], V.shape[1])
                if target_dim >= 2:
                    X_rep = X_pca[:, :target_dim]
                    V_rep = V[:, :target_dim]
                    rep_label = "pca"
            elif "X_latent" in self.adata.obsm:
                X_lat = np.asarray(self.adata.obsm["X_latent"], dtype=np.float32)
                target_dim = min(X_lat.shape[1], V.shape[1])
                if target_dim >= 2:
                    X_rep = X_lat[:, :target_dim]
                    V_rep = V[:, :target_dim]
                    rep_label = "latent"

        if X_rep is None or V_rep is None:
            return False, "No compatible low-dimensional representation + velocity vectors found."

        adata_fake = AnnData(X=X_rep)
        adata_fake.obs_names = self.adata.obs_names.copy()
        adata_fake.layers["Ms"] = X_rep
        adata_fake.layers[vkey] = V_rep
        if X_rep.shape[1] >= 2:
            adata_fake.obsm["X_fast"] = X_rep[:, :2]
            self.adata.obsm["X_fast"] = np.asarray(X_rep[:, :2], dtype=np.float32)
        if "X_umap" in self.adata.obsm:
            adata_fake.obsm["X_umap"] = np.asarray(self.adata.obsm["X_umap"], dtype=np.float32)
        if "X_pca" in self.adata.obsm:
            adata_fake.obsm["X_pca"] = np.asarray(self.adata.obsm["X_pca"], dtype=np.float32)

        copied_neighbors = False
        if bool(getattr(self, "fast_velocity_reuse_neighbors", False)):
            try:
                conn = self.adata.obsp.get("connectivities") if hasattr(self.adata, "obsp") else None
                dist = self.adata.obsp.get("distances") if hasattr(self.adata, "obsp") else None
                if "neighbors" in self.adata.uns and conn is not None and dist is not None:
                    if conn.shape[0] == self.adata.n_obs and dist.shape[0] == self.adata.n_obs:
                        adata_fake.uns["neighbors"] = dict(self.adata.uns["neighbors"])
                        adata_fake.obsp["connectivities"] = conn.copy()
                        adata_fake.obsp["distances"] = dist.copy()
                        copied_neighbors = True
            except Exception:
                copied_neighbors = False

        if copied_neighbors and _has_empty_neighbor_rows(adata_fake.obsp.get("connectivities")):
            logger.info("Copied neighbor graph contains empty rows; recomputing neighbors on compact AnnData.")
            copied_neighbors = False

        if not copied_neighbors:
            scv.pp.neighbors(adata_fake, n_neighbors=30, use_rep="X")

        # scVelo checks (distances > 0).sum(1); duplicated points can create zero
        # distances and trigger a false "corrupted graph" error. Nudge non-positive
        # stored distances to a tiny epsilon to keep neighbor topology unchanged.
        if hasattr(adata_fake, "obsp") and "distances" in adata_fake.obsp:
            try:
                dist = adata_fake.obsp["distances"].tocsr(copy=True)
                if dist.nnz > 0:
                    non_pos = dist.data <= 0
                    if np.any(non_pos):
                        dist.data[non_pos] = 1e-12
                        adata_fake.obsp["distances"] = dist
            except Exception:
                pass

        if n_jobs is None:
            n_jobs = _available_cpu_count()

        try:
            scv.tl.velocity_graph(adata_fake, vkey=vkey, n_jobs=int(n_jobs))
        except Exception as exc:
            # Robust fallback: rebuild neighbors once and retry velocity graph.
            logger.warning("velocity_graph failed on first attempt, retry after neighbor recompute: %s", exc)
            adata_fake.uns.pop("neighbors", None)
            if hasattr(adata_fake, "obsp"):
                adata_fake.obsp.pop("connectivities", None)
                adata_fake.obsp.pop("distances", None)
            scv.pp.neighbors(adata_fake, n_neighbors=30, use_rep="X")
            if hasattr(adata_fake, "obsp") and "distances" in adata_fake.obsp:
                try:
                    dist = adata_fake.obsp["distances"].tocsr(copy=True)
                    if dist.nnz > 0:
                        non_pos = dist.data <= 0
                        if np.any(non_pos):
                            dist.data[non_pos] = 1e-12
                            adata_fake.obsp["distances"] = dist
                except Exception:
                    pass
            scv.tl.velocity_graph(adata_fake, vkey=vkey, n_jobs=int(n_jobs))
        for basis in ("umap", "pca", "fast"):
            if f"X_{basis}" in adata_fake.obsm:
                try:
                    scv.tl.velocity_embedding(adata_fake, basis=basis, vkey=vkey)
                except Exception:
                    pass

        for key in [f"{vkey}_graph", f"{vkey}_graph_neg", f"{vkey}_params"]:
            if key in adata_fake.uns:
                self.adata.uns[key] = adata_fake.uns[key]
        for basis in ("umap", "pca", "fast"):
            v_emb_key = f"{vkey}_{basis}"
            if v_emb_key in adata_fake.obsm:
                self.adata.obsm[v_emb_key] = adata_fake.obsm[v_emb_key]

        return True, f"velocity_graph computed in lightweight {rep_label} space"

    def _make_downsampled_adata(self, n_cells: int, method: str = "random", random_state: int = 42) -> AnnData:
        """Create a downsampled AnnData copy.

        Args:
            n_cells: Target number of cells in the downsampled dataset.
            method: Downsampling method: "random" or "stratified".
            random_state: Random seed for reproducibility.

        Returns:
            A new AnnData object with at most ``n_cells`` observations.

        Raises:
            ValueError: If an unsupported method is provided.
        """
        current_n_cells = self.adata.n_obs
        if current_n_cells <= n_cells:
            return self.adata.copy()

        method = str(method).lower()
        if method == "random":
            rng = np.random.default_rng(random_state)
            indices = np.sort(rng.choice(current_n_cells, size=n_cells, replace=False))
            return self.adata[indices, :].copy()

        if method == "stratified":
            stratify_col = None
            if "time_point_processed" in self.adata.obs.columns:
                stratify_col = "time_point_processed"
            elif "Cell type annotation" in self.adata.obs.columns:
                stratify_col = "Cell type annotation"

            if stratify_col is None:
                rng = np.random.default_rng(random_state)
                indices = np.sort(rng.choice(current_n_cells, size=n_cells, replace=False))
                return self.adata[indices, :].copy()

            try:
                from sklearn.model_selection import train_test_split
                indices = np.arange(current_n_cells)
                _, selected_indices = train_test_split(
                    indices,
                    test_size=n_cells / current_n_cells,
                    stratify=self.adata.obs[stratify_col],
                    random_state=random_state
                )
                selected_indices = np.sort(selected_indices)
                return self.adata[selected_indices, :].copy()
            except Exception:
                rng = np.random.default_rng(random_state)
                indices = np.sort(rng.choice(current_n_cells, size=n_cells, replace=False))
                return self.adata[indices, :].copy()

        raise ValueError(f"Unknown downsample method: {method}")
    
    @log_reason
    def downsample_data(self, n_cells: int = 10000, method: str = "random", save_path: Optional[str] = None, reason: str = "") -> str:
        """Downsample the dataset to reduce memory usage.

        Args:
            n_cells: Target number of cells (no downsampling if current count is already lower).
            method: Downsampling method: "random" or "stratified".
            save_path: Reserved for compatibility; not used by this implementation.
            reason: Detailed reasoning and intent for calling this tool.

        Returns:
            A human-readable summary of the downsampling result.
        """
        current_n_cells = self.adata.n_obs
        
        if current_n_cells <= n_cells:
            return f" 当前细胞数 ({current_n_cells}) 已经小于或等于目标数 ({n_cells})，无需降采样"

        logger.info(f"Downsampling from {current_n_cells} to {n_cells} cells using {method} method...")
        try:
            downsampled_adata = self._make_downsampled_adata(n_cells=n_cells, method=method)
        except ValueError as e:
            return f" 未知的降采样方法: {method}。支持的方法: 'random', 'stratified'"
        
        # Update manager with downsampled adata
        old_n_cells = self.adata.n_obs
        self._manager.update(downsampled_adata)
        
        # Reset code executor so it picks up new adata
        self._code_executor = None
        
        # Clear data state cache
        self._data_state = None
        
        result = [
            "=" * 60,
            " 数据降采样完成",
            "=" * 60,
            "",
            f"原始细胞数: {old_n_cells:,}",
            f"降采样后: {self.adata.n_obs:,}",
            f"降采样比例: {self.adata.n_obs / old_n_cells * 100:.1f}%",
            f"方法: {method}",
            "",
            " 后续分析将使用降采样后的数据",
            "=" * 60
        ]
        
        return "\n".join(result)
    
    def _infer_model_type(self, state: Dict[str, Any]) -> str:
        """Infer the model type based on cached state metadata.

        Args:
            state: Cached state dictionary from ``get_data_state``.

        Returns:
            A string label describing the inferred model type.
        """
        components = state.get("model_components", [])
        if "interaction" in components:
            return "cyto_simulation"
        elif "score" in components and "growth" in components:
            return "ruot_or_crufm"
        elif "growth" in components:
            return "unbalanced_ot"
        elif "velocity" in components:
            return "dynamical_ot"
        elif state["has_model"]:
            return "unknown_trained"
        return "not_trained"
    
    def _recommend_analyses(self, state: Dict[str, Any]) -> List[str]:
        """Recommend analyses based on available data and model components.

        Args:
            state: Cached state dictionary from ``get_data_state``.

        Returns:
            A list of recommended analysis messages.
        """
        recommendations = []
        if not state["has_model"]:
            return [" 模型未训练"]
        
        if state["has_velocity"]:
            recommendations.append(" 可以分析速度场和轨迹 (analyze_trajectory_fate)")
        if state["has_growth_rate"]:
            recommendations.append(" 可以分析生长率 (analyze_growth_mass)")
            recommendations.append(" 可以分析驱动基因 (analyze_driver_genes_comprehensive)")
        if state["has_score"]:
            recommendations.append(" 可以分析随机性 (analyze_stochasticity)")
        if state["has_interaction"]:
            recommendations.append(" 可以分析细胞互作 (analyze_interaction)")
        if state.get("gene_space_available"):
            recommendations.append(" 可以做基因层分析 (gene-space)")
        return recommendations

    def _set_gene_names(self):
        '''
        Reset the gene names to adata.var_names
        '''
        gene_names, source_col = _detect_gene_name_column(self.adata)
        # 2. 只有当检测到的列不是当前的 var_names 时才进行修改
        if source_col != 'var_names':
            # # [推荐步骤] 在覆盖前，把原来的 var_names (通常是 ID) 备份保存
            # # 例如保存为 'gene_ids'，防止 ID 信息丢失
            # if 'gene_ids_save' not in self.adata.var.columns:
            #     self.adata.var['gene_ids_save'] = self.adata.var_names

            # # 3. 赋值给 var_names
            self.adata.var_names = gene_names

        self.adata.var_names_make_unique()
        


    # =========================================================================
    # 训练指标访问
    # =========================================================================
    @log_reason
    def get_training_metrics(self, reason: str = "") -> str:
        """Format and return training metrics for display.

        Args:
            reason: Detailed reasoning and intent for calling this tool.

        Returns:
            A human-readable string of training metrics.
        """
        if not self.training_metrics:
            return " 训练指标不可用"
        
        lines = ["=" * 60, " 训练指标", "=" * 60]
        for key, value in self.training_metrics.items():
            if isinstance(value, float):
                lines.append(f"  {key}: {value:.6f}")
            else:
                lines.append(f"  {key}: {value}")
        lines.append("=" * 60)
        return "\n".join(lines)
    
    @log_reason
    def evaluate_model_quality(self, reason: str = "") -> str:
        """Assess model quality using available training metrics.

        Args:
            reason: Detailed reasoning and intent for calling this tool.

        Returns:
            A human-readable assessment of model quality.
        """
        if not self.training_metrics:
            return " 无法评估：训练指标不可用"
        
        lines = ["=" * 60, " 模型质量评估", "=" * 60]
        
        loss = self.training_metrics.get("loss")
        if loss is not None:
            lines.append(f"\n Loss: {loss:.6f}")
            if loss < 0.1:
                lines.append("   优秀 - 模型收敛良好")
            elif loss < 0.5:
                lines.append("  ✓ 良好 - 模型基本收敛")
            else:
                lines.append("   较高 - 可能需要更多训练")
        
        wd = self.training_metrics.get("wasserstein_distance")
        if wd is not None:
            lines.append(f"\n📏 Wasserstein Distance: {wd:.6f}")
        
        lines.append("\n" + "=" * 60)
        return "\n".join(lines)

    # =========================================================================
    # 数据摘要
    # =========================================================================
    @log_reason
    def get_data_summary(self, reason: str = "") -> str:
        """Generate a human-readable summary of dataset and model state.

        Args:
            reason: Detailed reasoning and intent for calling this tool.

        Returns:
            A summary string describing data, preprocessing, and model availability.
        """
        state = self.get_data_state()
        gene_names, gene_col = _detect_gene_name_column(self.adata)
        
        lines = [
            "=" * 60, " 数据摘要", "=" * 60,
            f"细胞数: {state['n_cells']}", f"基因数: {state['n_genes']}",
            "", " 基因名称信息:",
        ]
        
        if state['has_real_gene_names']:
            lines.append(f"   真实基因名称列: '{state['gene_name_column']}'")
            lines.append(f"     示例: {gene_names[:3]}")
        else:
            lines.append(f"   使用 var_names: {list(self.adata.var_names[:3])}")
        
        lines.extend([
            "", " 数据结构:",
            f"  obs 列: {state['obs_columns'][:10]}{'...' if len(state['obs_columns']) > 10 else ''}",
            f"  obsm 键: {state['obsm_keys']}",
            f"  layers: {state['layers_keys']}",
            "", " 预处理状态:",
            f"  X_latent: {'' if state['has_X_latent'] else ''}",
            f"  时间点处理: {'' if state['has_time_processed'] else ''}",
            f"  PCA loadings: {'' if state['has_pca_loadings'] else ''}",
            f"  基因层分析: {'' if state['gene_space_available'] else ''}",
            "", " 模型状态:",
            f"  已训练: {'' if state['has_model'] else ''}",
            f"  模型类型: {state['inferred_model_type']}",
            f"  组件: {state['model_components']}",
            "", " 分析结果:",
            f"  velocity: {'' if state['has_velocity'] else ''}",
            f"  growth_rate: {'' if state['has_growth_rate'] else ''}",
            f"  score: {'' if state['has_score'] else ''}",
            f"  interaction: {'' if state['has_interaction'] else ''}",
            "", " 时间信息:",
            f"  时间键: {state['time_key']}",
            f"  时间点: {state['time_points']}",
            "", " 推荐分析:",
        ])
        
        for rec in state["recommended_analyses"]:
            lines.append(f"  • {rec}")
        
        if self.user_requested_analyses:
            lines.extend(["", " 用户预选的分析:"])
            for analysis in self.user_requested_analyses:
                lines.append(f"  • {analysis}")
        
        lines.append("=" * 60)
        return "\n".join(lines)


    # =========================================================================
    # 轨迹和命运分析 (from downstream_tools)
    # =========================================================================
    @log_reason
    def analyze_trajectory_fate(
        self,
        label_key: Optional[str] = None,
        mode: str = "fast",
        include_gene_projection: bool = False,
        include_sde: bool = False,
        plot_basis: Optional[str] = None,
        strict_basis: bool = False,
        save_dir: Optional[str] = None,
        reason: str = "",
    ) -> str:
        """Plot velocity stream and optional trajectory artifacts with unified core.

        Args:
            label_key: Optional ``obs`` column used to color cells.
            mode: ``fast`` (default) or ``full``.
            include_gene_projection: Whether to export PCA-projected gene velocity summary.
            include_sde: Whether to additionally generate SDE trajectory artifacts.
            plot_basis: Optional preferred plotting basis (e.g., ``spring``, ``umap``, ``pca``).
            strict_basis: If ``True``, fail when requested custom basis is unavailable instead of fallback.
            save_dir: Optional artifact output directory for this call.
            reason: Detailed reasoning and intent for calling this tool.

        Returns:
            A summary string describing generated artifacts.
        """
        resolved_basis, resolved_strict = self._resolve_viz_basis_preference(
            preferred=plot_basis,
            strict=bool(strict_basis),
        )
        guarded = self._invoke_core_in_subprocess(
            method="analyze_trajectory_fate",
            kwargs={
                "label_key": label_key,
                "mode": mode,
                "include_gene_projection": bool(include_gene_projection),
                "include_sde": bool(include_sde),
                "preferred_basis": resolved_basis,
                "strict_basis": bool(resolved_strict),
                "save_dir": save_dir,
                "n_pcs": int(self.fast_velocity_n_pcs),
            },
            timeout=max(600, self.subprocess_guard_timeout),
        )
        if guarded is not None:
            if not guarded.get("ok", False):
                return (
                    " trajectory analysis failed in isolated tool process.\n"
                    f"error: {guarded.get('error', 'unknown')}\n"
                    f"stderr: {str(guarded.get('stderr_tail', ''))[:800]}"
                )
            result = self._result_from_guarded_payload(guarded, "Trajectory/Fate Analysis")
        else:
            try:
                result = self._downstream_core.analyze_trajectory_fate(
                    adata=self.adata,
                    label_key=label_key,
                    mode=mode,
                    include_gene_projection=include_gene_projection,
                    include_sde=include_sde,
                    preferred_basis=resolved_basis,
                    strict_basis=bool(resolved_strict),
                    save_dir=save_dir,
                    n_pcs=int(self.fast_velocity_n_pcs),
                )
            except Exception as exc:
                logger.exception("analyze_trajectory_fate failed")
                return f" trajectory analysis failed: {exc}"
        result.artifacts = self._apply_publication_bundle_to_artifacts(result.artifacts)
        self.analysis_results["trajectory_fate"] = {
            "artifacts": result.artifacts,
            "summary": result.summary,
            "warnings": result.warnings,
            "payload": result.payload,
        }
        return result.render_text()


    # =========================================================================
    # 驱动基因分析 (重构版 - 更深入的分析)
    # =========================================================================
    @log_reason
    def analyze_growth_driver_genes(
        self,
        top_n: int = 20,
        max_genes: int = 5000,
        max_cells: int = 20000,
        use_highly_variable: bool = True,
        random_state: int = 0,
        method: str = "auto",
        batch_size: int = 2048,
        per_time_max_cells: Optional[int] = None,
        save_dir: Optional[str] = None,
        reason: str = "",
    ) -> str:
        """Identify growth drivers with a unified gradient-based pipeline.

        Args:
            top_n: Number of top driver genes to report.
            max_genes: Reserved for compatibility.
            max_cells: Maximum cells used for gradient estimation.
            use_highly_variable: Reserved for compatibility.
            random_state: Random seed for sampling.
            method: Reserved for compatibility.
            batch_size: Batch size for gradient computation.
            per_time_max_cells: Reserved for compatibility.
            save_dir: Optional artifact output directory for this call.
            reason: Detailed reasoning and intent for calling this tool.

        Returns:
            A human-readable report with artifacts and summary.
        """
        _ = max_genes
        _ = use_highly_variable
        _ = method
        _ = per_time_max_cells
        guarded = self._invoke_core_in_subprocess(
            method="analyze_growth_driver_genes",
            kwargs={
                "top_n": int(top_n),
                "max_cells": int(max_cells),
                "batch_size": int(batch_size),
                "random_state": int(random_state),
                "save_dir": save_dir,
            },
            timeout=max(600, self.subprocess_guard_timeout),
        )
        if guarded is not None:
            if not guarded.get("ok", False):
                return (
                    " growth driver analysis failed in isolated tool process.\n"
                    f"error: {guarded.get('error', 'unknown')}\n"
                    f"stderr: {str(guarded.get('stderr_tail', ''))[:800]}"
                )
            result = self._result_from_guarded_payload(guarded, "Growth Driver Analysis")
        else:
            try:
                result = self._downstream_core.analyze_growth_driver_genes(
                    adata=self.adata,
                    top_n=int(top_n),
                    max_cells=int(max_cells),
                    batch_size=int(batch_size),
                    random_state=int(random_state),
                    save_dir=save_dir,
                )
            except Exception as exc:
                logger.exception("analyze_growth_driver_genes failed")
                return f" growth driver analysis failed: {exc}"
        self.analysis_results["growth_drivers"] = {
            "artifacts": result.artifacts,
            "summary": result.summary,
            "warnings": result.warnings,
            "payload": result.payload,
            "top_items": result.payload.get("top", []),
            "method": "gradient",
        }
        return result.render_text()
    
    @log_reason
    def analyze_velocity_driver_genes(
        self,
        top_n: int = 20,
        analysis_space: str = "gene",
        save_dir: Optional[str] = None,
        reason: str = "",
    ) -> str:
        """Identify velocity drivers using unified core implementation.

        Args:
            top_n: Number of top driver genes to report.
            analysis_space: "gene" (default), "latent", or "auto".
            save_dir: Optional artifact output directory for this call.
            reason: Detailed reasoning and intent for calling this tool.

        Returns:
            A human-readable report with top drivers and artifacts.
        """
        guarded = self._invoke_core_in_subprocess(
            method="analyze_velocity_driver_genes",
            kwargs={
                "top_n": int(top_n),
                "analysis_space": analysis_space,
                "save_dir": save_dir,
            },
            timeout=max(300, self.subprocess_guard_timeout),
        )
        if guarded is not None:
            if not guarded.get("ok", False):
                return (
                    " velocity driver analysis failed in isolated tool process.\n"
                    f"error: {guarded.get('error', 'unknown')}\n"
                    f"stderr: {str(guarded.get('stderr_tail', ''))[:800]}"
                )
            result = self._result_from_guarded_payload(guarded, "Velocity Driver Analysis")
        else:
            try:
                result = self._downstream_core.analyze_velocity_driver_genes(
                    adata=self.adata,
                    top_n=int(top_n),
                    analysis_space=analysis_space,
                    save_dir=save_dir,
                )
            except Exception as exc:
                logger.exception("analyze_velocity_driver_genes failed")
                return f" velocity driver analysis failed: {exc}"
        self.analysis_results["velocity_drivers"] = {
            "artifacts": result.artifacts,
            "summary": result.summary,
            "warnings": result.warnings,
            "payload": result.payload,
            "top_items": result.payload.get("top", []),
        }
        return result.render_text()
    
    @log_reason
    def analyze_fate_probabilities(
        self,
        label_key: Optional[str] = None,
        terminal_states: Optional[Any] = None,
        n_states: int = 5,
        top_n: int = 6,
        n_simulations: int = 5,
        n_init_cells: int = 2000,
        n_steps: int = 25,
        classifier_path: Optional[str] = None,
        max_train_cells: int = 80000,
        classifier_epochs: int = 200,
        reason: str = "",
    ) -> str:
        """Estimate fate probabilities via CytoBridge SDE simulation + classifier.

        This implementation does not rely on CellRank. It simulates t0 cells
        forward with the trained CytoBridge dynamics model, classifies terminal
        states, and estimates fate probabilities from repeated simulations.

        Args:
            label_key: Cell label column.
            terminal_states: Optional terminal states to keep (list/tuple/set or comma-separated string).
                When provided, global fate statistics remain unchanged, but the
                focused plot/export will highlight only sampled t0 cells whose
                dominant terminal fate is in this set; other cells are background.
        """
        guarded = self._invoke_tool_method_in_subprocess(
            method="analyze_fate_probabilities",
            kwargs={
                "label_key": label_key,
                "terminal_states": terminal_states,
                "n_states": int(n_states),
                "top_n": int(top_n),
                "n_simulations": int(n_simulations),
                "n_init_cells": int(n_init_cells),
                "n_steps": int(n_steps),
                "classifier_path": classifier_path,
                "max_train_cells": int(max_train_cells),
                "classifier_epochs": int(classifier_epochs),
                "reason": reason,
            },
            timeout=max(1200, self.subprocess_guard_timeout),
        )
        if guarded is not None:
            return self._consume_guarded_tool_result(guarded, "fate probability analysis")

        apply_matplotlib_style(self.figure_quality_preset)
        outdir = _ensure_dir(self.output_dir / "fate_probabilities")
        artifacts: Dict[str, str] = {}

        if "X_latent" not in self.adata.obsm:
            return " 未找到 X_latent，无法进行基于 CytoBridge 的命运概率分析。"

        time_key = "time_point_processed"
        if time_key not in self.adata.obs.columns:
            return f" 未找到 {time_key} 列，无法确定 t0 与终点时间。"

        if label_key is None:
            for cand in ["majorclass", "cell_type", "Cell type annotation", "celltype", "cluster", "leiden"]:
                if cand in self.adata.obs.columns:
                    label_key = cand
                    break
        if label_key is None or label_key not in self.adata.obs.columns:
            return " 未找到可用细胞类型标签列。请指定 label_key。"
        label_values_now = self.adata.obs[label_key].astype(str)
        label_values_now = label_values_now[label_values_now.notna()]
        label_set_now = set(label_values_now.unique().tolist())

        requested_terminal_labels: List[str] = []
        if terminal_states is not None:
            if isinstance(terminal_states, str):
                raw = terminal_states.strip()
                if raw:
                    parsed: Any = None
                    try:
                        parsed = ast.literal_eval(raw)
                    except Exception:
                        parsed = None
                    if isinstance(parsed, (list, tuple, set)):
                        requested_terminal_labels = [str(x).strip() for x in parsed if str(x).strip()]
                    else:
                        requested_terminal_labels = [x.strip() for x in raw.split(",") if x.strip()]
            elif isinstance(terminal_states, (list, tuple, set)):
                requested_terminal_labels = [str(x).strip() for x in terminal_states if str(x).strip()]
            else:
                s = str(terminal_states).strip()
                if s:
                    requested_terminal_labels = [s]

        try:
            from CytoBridge.utils import load_model_from_adata
            from CytoBridge.tl.perturbation import (
                classify_final_states,
                compute_sampling_weights,
                load_mlp_classifier,
                simulate_perturbation_sde,
                train_cell_classifier as cb_train_cell_classifier,
                _resolve_downstream_rollout_defaults,
            )
        except ImportError as e:
            return f" 导入 CytoBridge 命运分析组件失败: {e}"

        time_vals = pd.to_numeric(self.adata.obs[time_key], errors="coerce")
        if time_vals.isna().all():
            return f" {time_key} 无法解析为数值时间。"
        init_time = float(np.nanmin(time_vals.values))
        end_time = float(np.nanmax(time_vals.values))

        init_mask = np.asarray(np.isclose(time_vals.values, init_time))
        init_indices = np.where(init_mask)[0]
        if init_indices.size == 0:
            return f" 在 t0={init_time} 未找到可用初始细胞。"

        n_init = int(max(1, min(int(n_init_cells), int(init_indices.size))))
        n_sim = int(max(1, int(n_simulations)))
        n_steps = int(max(2, int(n_steps)))

        try:
            model = load_model_from_adata(self.adata)
            model.eval()
        except Exception as e:
            return f" 加载 CytoBridge 模型失败: {e}"
        rollout_defaults = _resolve_downstream_rollout_defaults(
            adata=self.adata,
            model=model,
            resolved_config=None,
            n_steps=n_steps,
            init_time=init_time,
            end_time=end_time,
        )
        sigma = float(rollout_defaults["sigma"])

        mlp_model = None
        label_encoder = None
        classify_method = "knn"
        classifier_source = "knn"

        candidate_classifier_paths: List[Path] = []
        if classifier_path:
            candidate_classifier_paths.append(Path(classifier_path))
        saved_clf = self.analysis_results.get("cell_classifier", {}).get("model_path")
        if saved_clf:
            candidate_classifier_paths.append(Path(saved_clf))
        candidate_classifier_paths.append(self.output_dir / "classifiers" / "cell_classifier.pt")
        candidate_classifier_paths.append(outdir / "fate_classifier.pt")

        for p in candidate_classifier_paths:
            try:
                if p.exists():
                    loaded_model, loaded_encoder = load_mlp_classifier(str(p), device=self.device)
                    loaded_classes = set(str(x) for x in loaded_encoder.classes_)
                    # Skip stale/incompatible classifiers trained on another label space.
                    if not loaded_classes.issubset(label_set_now):
                        logger.info(
                            "Skip classifier %s due to label mismatch: classes=%d not subset of current label set=%d",
                            p,
                            len(loaded_classes),
                            len(label_set_now),
                        )
                        continue
                    mlp_model, label_encoder = loaded_model, loaded_encoder
                    classify_method = "mlp"
                    classifier_source = str(p)
                    break
            except Exception as e:
                logger.warning("Failed to load classifier %s: %s", p, e)

        if mlp_model is None:
            try:
                adata_train = self.adata
                if max_train_cells > 0 and self.adata.n_obs > max_train_cells:
                    rng = np.random.default_rng(42)
                    y = self.adata.obs[label_key].astype(str)
                    sampled_idx: List[int] = []
                    n_groups = max(1, int(y.nunique()))
                    per_group = max(50, int(max_train_cells // n_groups))
                    group_indices = self.adata.obs.groupby(label_key, observed=False).indices
                    for idx in group_indices.values():
                        idx_arr = np.asarray(idx, dtype=int)
                        if idx_arr.size > per_group:
                            idx_arr = rng.choice(idx_arr, size=per_group, replace=False)
                        sampled_idx.extend(idx_arr.tolist())
                    if len(sampled_idx) > max_train_cells:
                        sampled_idx = rng.choice(
                            np.asarray(sampled_idx, dtype=int),
                            size=max_train_cells,
                            replace=False,
                        ).tolist()
                    sampled_idx = sorted(set(int(i) for i in sampled_idx))
                    if len(sampled_idx) >= 200:
                        adata_train = self.adata[sampled_idx, :].copy()

                clf_save = outdir / "fate_classifier.pt"
                clf_res = cb_train_cell_classifier(
                    adata=adata_train,
                    label_key=label_key,
                    latent_key="X_latent",
                    hidden_dim=128,
                    epochs=int(max(20, classifier_epochs)),
                    batch_size=512,
                    learning_rate=1e-3,
                    validation_split=0.1,
                    device=self.device,
                    save_path=str(clf_save),
                )
                mlp_model = clf_res["model"]
                label_encoder = clf_res["label_encoder"]
                classify_method = "mlp"
                classifier_source = str(clf_save)
                artifacts["classifier"] = str(clf_save)
            except Exception as e:
                logger.warning("Training MLP classifier failed, fallback to KNN: %s", e)
                classify_method = "knn"
                classifier_source = "knn"

        try:
            x_base = np.asarray(self.adata.obsm["X_latent"], dtype=np.float32)
            sampling_weights = compute_sampling_weights(self.adata)
            sim_results = simulate_perturbation_sde(
                model=model,
                adata=self.adata,
                x_perturbed=x_base,
                n_simulations=n_sim,
                n_init_cells=n_init,
                n_steps=n_steps,
                sigma=sigma,
                init_time=init_time,
                end_time=end_time,
                time_key=time_key,
                cell_type_key=label_key,
                sampling_weights=sampling_weights,
                device=self.device,
                run_unperturbed=False,
            )
        except Exception as e:
            return f" CytoBridge 轨迹模拟失败: {e}"

        try:
            final_pos = np.asarray(sim_results["perturbed_trajectories"][:, -1, :, :], dtype=np.float32)
            final_weights = np.asarray(sim_results["perturbed_weights"][:, -1, :], dtype=np.float32)
            sampled_init_idx = np.asarray(sim_results["init_indices"], dtype=int)

            pred_labels, _ = classify_final_states(
                final_positions=final_pos,
                adata=self.adata,
                label_key=label_key,
                method=classify_method,
                mlp_model=mlp_model,
                label_encoder=label_encoder,
                weights=final_weights,
            )
            pred_labels = np.asarray(pred_labels).reshape(-1)
            init_idx_flat = sampled_init_idx.reshape(-1)
            weights_flat = final_weights.reshape(-1).astype(np.float64)
            init_label_values = self.adata.obs[label_key].astype(str).values
            src_labels = init_label_values[init_idx_flat]
            obs_names_arr = np.asarray(self.adata.obs_names).astype(str)
            src_obs_names = obs_names_arr[init_idx_flat]

            flow_df = pd.DataFrame(
                {
                    "source": src_labels,
                    "target": pred_labels.astype(str),
                    "weight": weights_flat,
                    "cell_index": init_idx_flat.astype(int),
                    "obs_index": src_obs_names,
                }
            )
            flow_df = flow_df[flow_df["weight"] > 0]
            if flow_df.empty:
                return " 模拟结果为空（无有效权重流）。"

            available_terminal_states = sorted(flow_df["target"].astype(str).unique().tolist())
            selected_terminal_states: Optional[List[str]] = None
            unresolved_terminal_states: List[str] = []
            if requested_terminal_labels:
                lookup = {str(x).lower(): str(x) for x in available_terminal_states}
                selected: List[str] = []
                for raw_label in requested_terminal_labels:
                    key = str(raw_label).strip().lower()
                    if not key:
                        continue
                    matched = lookup.get(key)
                    if matched is None:
                        unresolved_terminal_states.append(str(raw_label))
                        continue
                    if matched not in selected:
                        selected.append(matched)
                if not selected:
                    return (
                        " 指定的 terminal_states 未命中任何模拟终态。\n"
                        f"requested={requested_terminal_labels}\n"
                        f"available={available_terminal_states[:30]}"
                    )
                selected_terminal_states = selected

            trans = flow_df.pivot_table(
                index="source",
                columns="target",
                values="weight",
                aggfunc="sum",
                fill_value=0.0,
            )
            trans_prob = trans.div(trans.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
            overall = flow_df.groupby("target")["weight"].sum()
            overall = overall / max(float(overall.sum()), 1e-12)

            fate_csv = outdir / "fate_probabilities.csv"
            overall_df = (
                pd.DataFrame({"state": overall.index.astype(str), "probability": overall.values.astype(float)})
                .sort_values("probability", ascending=False)
                .reset_index(drop=True)
            )
            overall_df.to_csv(fate_csv, index=False)
            artifacts["fate_probabilities"] = str(fate_csv)

            trans_csv = outdir / "fate_transition_matrix_t0_to_terminal.csv"
            trans_prob.to_csv(trans_csv)
            artifacts["transition_matrix"] = str(trans_csv)

            dominant = pd.DataFrame(
                {
                    "initial_state": trans_prob.index.astype(str),
                    "dominant_terminal_state": trans_prob.idxmax(axis=1).astype(str),
                    "dominant_probability": trans_prob.max(axis=1).astype(float),
                }
            )
            zero_row_mask = trans_prob.sum(axis=1).to_numpy() <= 0
            if np.any(zero_row_mask):
                dominant.loc[zero_row_mask, "dominant_terminal_state"] = ""
                dominant.loc[zero_row_mask, "dominant_probability"] = 0.0
            dom_csv = outdir / "fate_dominant_state.csv"
            dominant.to_csv(dom_csv, index=False)
            artifacts["dominant_state"] = str(dom_csv)

            cell_flow = pd.DataFrame(
                {
                    "cell_index": init_idx_flat.astype(int),
                    "obs_index": src_obs_names,
                    "init_label": src_labels.astype(str),
                    "target": pred_labels.astype(str),
                    "weight": weights_flat,
                }
            )
            cell_prob = (
                cell_flow.groupby(["cell_index", "obs_index", "init_label", "target"], as_index=False)["weight"]
                .sum()
                .rename(columns={"weight": "mass"})
            )
            cell_prob["probability"] = cell_prob["mass"] / cell_prob.groupby("cell_index")["mass"].transform("sum")
            cell_prob = cell_prob.drop(columns=["mass"])
            cell_prob_csv = outdir / "fate_probabilities_by_sampled_t0_cell.csv"
            cell_prob.to_csv(cell_prob_csv, index=False)
            artifacts["cell_level_probabilities"] = str(cell_prob_csv)

            # Dominant fate per sampled t0 cell (global definition over all terminal states).
            dom_idx = cell_prob.groupby("cell_index")["probability"].idxmax()
            dom_cell = (
                cell_prob.loc[dom_idx, ["cell_index", "target", "probability"]]
                .rename(
                    columns={
                        "target": "dominant_terminal_state",
                        "probability": "dominant_probability",
                    }
                )
                .reset_index(drop=True)
            )

            # Optional focus view: keep sampled t0 cells whose dominant fate is selected.
            selected_cell_indices: Optional[set] = None
            cell_prob_focus = cell_prob
            if selected_terminal_states:
                selected_cell_indices = set(
                    dom_cell.loc[
                        dom_cell["dominant_terminal_state"].isin(selected_terminal_states),
                        "cell_index",
                    ]
                    .astype(int)
                    .tolist()
                )
                if not selected_cell_indices:
                    return (
                        " 未找到主导终态落在指定 terminal_states 的 t0 细胞。\n"
                        f"requested={requested_terminal_labels}\n"
                        f"matched={selected_terminal_states}"
                    )
                cell_prob_focus = cell_prob[cell_prob["cell_index"].isin(selected_cell_indices)].copy()
                focus_csv = outdir / "fate_probabilities_by_sampled_t0_cell_selected.csv"
                cell_prob_focus.to_csv(focus_csv, index=False)
                artifacts["cell_level_probabilities_selected"] = str(focus_csv)

            cell_index_map = (
                cell_flow.groupby(["cell_index", "obs_index", "init_label"], as_index=False)
                .agg(n_draws=("target", "size"), total_terminal_weight=("weight", "sum"))
            )
            cell_index_map = cell_index_map.merge(dom_cell, on="cell_index", how="left")
            idx_map_csv = outdir / "sampled_t0_cell_index_mapping.csv"
            cell_index_map.to_csv(idx_map_csv, index=False)
            artifacts["cell_index_mapping"] = str(idx_map_csv)
            if selected_cell_indices is not None:
                idx_map_selected_csv = outdir / "sampled_t0_cell_index_mapping_selected.csv"
                cell_index_map[cell_index_map["cell_index"].isin(selected_cell_indices)].to_csv(
                    idx_map_selected_csv,
                    index=False,
                )
                artifacts["cell_index_mapping_selected"] = str(idx_map_selected_csv)

            # UMAP dominant-fate visualization (single categorical map).
            basis = "umap" if "X_umap" in self.adata.obsm else self._ensure_embedding()
            emb_key = f"X_{basis}"
            if emb_key in self.adata.obsm:
                pivot_prob = cell_prob_focus.pivot_table(
                    index="cell_index",
                    columns="target",
                    values="probability",
                    aggfunc="mean",
                    fill_value=0.0,
                )
                dom_by_cell = pivot_prob.idxmax(axis=1).astype(str)
                dom_prob_by_cell = pivot_prob.max(axis=1).astype(np.float32)

                obs_dom = "fate_dominant_sim"
                obs_conf = "fate_dominant_prob_sim"
                vals_dom = np.array([np.nan] * self.adata.n_obs, dtype=object)
                vals_conf = np.full(self.adata.n_obs, np.nan, dtype=np.float32)

                idx_all = dom_by_cell.index.to_numpy(dtype=int)
                valid = (idx_all >= 0) & (idx_all < self.adata.n_obs)
                valid_idx = idx_all[valid]
                if valid_idx.size > 0:
                    vals_dom[valid_idx] = dom_by_cell.iloc[np.where(valid)[0]].to_numpy(dtype=str)
                    vals_conf[valid_idx] = dom_prob_by_cell.iloc[np.where(valid)[0]].to_numpy(dtype=np.float32)

                self.adata.obs[obs_dom] = pd.Categorical(vals_dom)
                self.adata.obs[obs_conf] = vals_conf

                coords = np.asarray(self.adata.obsm[emb_key])
                x = coords[:, 0]
                y = coords[:, 1]
                valid_mask = pd.notna(vals_dom)
                fig, ax = plt.subplots(figsize=(9, 7))
                # Layer 1: background (all cells)
                ax.scatter(
                    x,
                    y,
                    s=4,
                    c="lightgray",
                    alpha=0.45,
                    linewidths=0,
                    rasterized=True,
                )
                # Layer 2: sampled t0 cells with dominant fate colors
                cat_vals = pd.Categorical(vals_dom[valid_mask])
                categories = list(cat_vals.categories)
                n_categories = len(categories)
                color_map: Dict[Any, Any] = {}
                palette = plt.get_cmap("tab20")
                if n_categories <= 20:
                    # Spread selected colors evenly across tab20 slots to maximize contrast.
                    slot_idx = np.linspace(0, 19, n_categories, dtype=int)
                    for cat, idx in zip(categories, slot_idx):
                        color_map[cat] = palette(idx)
                else:
                    # tab20 has 20 anchors; for >20 categories we still cycle.
                    for i, cat in enumerate(categories):
                        color_map[cat] = palette(i % 20)

                for cat in categories:
                    m = valid_mask & (vals_dom == cat)
                    if not np.any(m):
                        continue
                    ax.scatter(
                        x[m],
                        y[m],
                        s=7,
                        c=[color_map[cat]],
                        alpha=0.95,
                        linewidths=0,
                        label=str(cat),
                        rasterized=True,
                    )
                if selected_terminal_states:
                    ax.set_title(
                        "Dominant Fate on UMAP (t0 cells with selected dominant terminal states)"
                    )
                else:
                    ax.set_title("Dominant Fate on UMAP (sampled t0 over full background)")
                ax.set_xlabel(f"{basis.upper()}1")
                ax.set_ylabel(f"{basis.upper()}2")
                if len(cat_vals.categories) > 0:
                    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
                plt.tight_layout()
                suffix = "_selected" if selected_terminal_states else ""
                bundle = save_figure_bundle(
                    fig,
                    base_name=f"fate_dominant_{basis}{suffix}",
                    output_dir=self.figures_dir,
                    preset=self.figure_quality_preset,
                )
                plt.close(fig)
                artifacts.update({f"fate_umap_dominant_{k}": v for k, v in bundle.items()})

            fig, ax = plt.subplots(figsize=(8, 5))
            show_n = min(max(1, int(top_n)), len(overall_df))
            show_df = overall_df.head(show_n)
            ax.bar(show_df["state"], show_df["probability"], color="#2a6f97", alpha=0.9)
            ax.set_ylim(0.0, 1.0)
            ax.set_ylabel("Probability")
            ax.set_title("Terminal Fate Probabilities (CytoBridge Simulation)")
            ax.tick_params(axis="x", rotation=30)
            plt.tight_layout()
            bundle = save_figure_bundle(
                fig,
                base_name="fate_probabilities_overall",
                output_dir=self.figures_dir,
                preset=self.figure_quality_preset,
            )
            plt.close(fig)
            artifacts.update({f"fate_probabilities_plot_{k}": v for k, v in bundle.items()})

            rows = list(trans_prob.index.astype(str))
            cols = list(trans_prob.columns.astype(str))
            fig_w = max(7.0, min(16.0, 0.9 * len(cols) + 4.0))
            fig_h = max(5.0, min(16.0, 0.55 * len(rows) + 3.0))
            fig, ax = plt.subplots(figsize=(fig_w, fig_h))
            im = ax.imshow(trans_prob.values, aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
            ax.set_xticks(np.arange(len(cols)))
            ax.set_xticklabels(cols, rotation=45, ha="right")
            ax.set_yticks(np.arange(len(rows)))
            ax.set_yticklabels(rows)
            ax.set_xlabel("Terminal state")
            ax.set_ylabel(f"Initial state at t0 ({label_key})")
            ax.set_title("Fate Transition Matrix (row-normalized)")
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label("Probability")
            plt.tight_layout()
            bundle = save_figure_bundle(
                fig,
                base_name="fate_transition_heatmap",
                output_dir=self.figures_dir,
                preset=self.figure_quality_preset,
            )
            plt.close(fig)
            artifacts.update({f"transition_heatmap_{k}": v for k, v in bundle.items()})

            summary_json = outdir / "fate_probability_summary.json"
            payload = {
                "mode": "cytobridge_simulation_classifier",
                "label_key": label_key,
                "classifier_method": classify_method,
                "classifier_source": classifier_source,
                "n_total_cells": int(self.adata.n_obs),
                "n_t0_cells": int(init_indices.size),
                "n_init_cells_sampled_per_sim": int(n_init),
                "n_simulations": int(n_sim),
                "n_steps": int(n_steps),
                "sigma": float(sigma),
                "rollout_sigma_source": rollout_defaults.get("sigma_source"),
                "init_time": float(init_time),
                "end_time": float(end_time),
                "terminal_states": [str(s) for s in overall_df["state"].tolist()],
                "requested_terminal_states": requested_terminal_labels,
                "matched_terminal_states": selected_terminal_states or [],
                "unresolved_terminal_states": unresolved_terminal_states,
                "output_schema": {
                    "fate_probabilities.csv": ["state", "probability"],
                    "fate_transition_matrix_t0_to_terminal.csv": [
                        "row=index=initial_state",
                        "col=terminal_state",
                        "value=row-normalized probability",
                    ],
                    "fate_dominant_state.csv": [
                        "initial_state",
                        "dominant_terminal_state",
                        "dominant_probability",
                    ],
                    "fate_probabilities_by_sampled_t0_cell.csv": [
                        "cell_index",
                        "obs_index",
                        "init_label",
                        "target",
                        "probability",
                    ],
                    "sampled_t0_cell_index_mapping.csv": [
                        "cell_index",
                        "obs_index",
                        "init_label",
                        "n_draws",
                        "total_terminal_weight",
                        "dominant_terminal_state",
                        "dominant_probability",
                    ],
                    "fate_probabilities_by_sampled_t0_cell_selected.csv": [
                        "cell_index",
                        "obs_index",
                        "init_label",
                        "target",
                        "probability",
                        "(only when terminal_states is provided; includes cells whose dominant fate is selected)",
                    ],
                    "sampled_t0_cell_index_mapping_selected.csv": [
                        "cell_index",
                        "obs_index",
                        "init_label",
                        "n_draws",
                        "total_terminal_weight",
                        "dominant_terminal_state",
                        "dominant_probability",
                        "(only when terminal_states is provided)",
                    ],
                },
                "artifacts": artifacts,
            }
            with summary_json.open("w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            artifacts["summary"] = str(summary_json)

            self.analysis_results["fate_probabilities"] = {
                "mode": "cytobridge_simulation_classifier",
                "label_key": label_key,
                "classifier_method": classify_method,
                "terminal_states": [str(s) for s in overall_df["state"].tolist()],
                "artifacts": artifacts,
            }

            lines = [
                "=" * 60,
                " 命运概率分析 (CytoBridge Simulation)",
                "=" * 60,
                "",
                f"标签列: {label_key}",
                f"模拟次数: {n_sim}",
                f"每次初始细胞数: {n_init}",
                f"时间步数: {n_steps} (t={init_time} -> {end_time})",
                f"噪声 sigma: {sigma}",
                f"分类器: {classify_method} ({classifier_source})",
                (
                    "终态筛选: "
                    + (
                        f"requested={requested_terminal_labels}, matched={selected_terminal_states}, unresolved={unresolved_terminal_states}"
                        if selected_terminal_states
                        else "未启用（输出为全量终态）"
                    )
                ),
                "",
                " Top terminal fates:",
            ]
            for i, row in overall_df.head(show_n).iterrows():
                lines.append(f"  {i+1}. {row['state']}: {float(row['probability']):.4f}")
            lines.extend(
                [
                    "",
                    " 结果文件:",
                    f"  fate_probabilities: {fate_csv}",
                    f"  transition_matrix: {trans_csv}",
                    f"  dominant_state: {dom_csv}",
                    f"  cell_level_probabilities: {cell_prob_csv}",
                    f"  cell_index_mapping: {idx_map_csv}",
                    f"  cell_level_probabilities_selected: {artifacts.get('cell_level_probabilities_selected', 'N/A')}",
                    f"  cell_index_mapping_selected: {artifacts.get('cell_index_mapping_selected', 'N/A')}",
                    f"  dominant_fate_umap: {artifacts.get('fate_umap_dominant', 'N/A')}",
                    f"  summary_json: {summary_json}",
                    "",
                    "📘 结果文件结构:",
                    "  fate_probabilities.csv: state, probability",
                    "  fate_transition_matrix_t0_to_terminal.csv: 行=initial_state, 列=terminal_state, 值=行归一化概率",
                    "  fate_dominant_state.csv: initial_state, dominant_terminal_state, dominant_probability",
                    "  fate_probabilities_by_sampled_t0_cell.csv: cell_index(位置索引), obs_index(原始obs索引), init_label, target, probability",
                    "  sampled_t0_cell_index_mapping.csv: cell_index, obs_index, init_label, n_draws, total_terminal_weight, dominant_*",
                    "=" * 60,
                ]
            )
            return "\n".join(lines)
        except Exception as e:
            return f" Fate probability 统计失败: {e}"
    def analyze_vg_driver_genes(self, label_key: Optional[str] = None, top_n: int = 20, reason: str = "") -> str:
        """Run growth, velocity and aggregate results.

        Args:
            label_key: Optional ``obs`` column for cell type labels.
            top_n: Number of driver genes to report per analysis.
            reason: Detailed reasoning and intent for calling this tool.

        Returns:
            A consolidated report combining growth, velocity analyses.
        """
        lines = [
            "=" * 60, " 综合驱动基因分析", "=" * 60,
            "", "正在执行多维度驱动基因分析...",
            ""
        ]
        
        # 1. 生长驱动基因
        lines.append("【1/2】分析生长驱动基因...")
        growth_result = self.analyze_growth_driver_genes(top_n=top_n)
        lines.append(growth_result)
        lines.append("")
        
        # 2. 速度驱动基因
        lines.append("【2/2】分析速度驱动基因...")
        velocity_result = self.analyze_velocity_driver_genes(top_n=top_n)
        lines.append(velocity_result)
        lines.append("")
        
        # 综合总结
        lines.extend([
            "驱动基因:",
            "  1. 生长驱动基因 - 控制细胞增殖/凋亡",
            "  2. 速度驱动基因 - 在分化过程中变化剧烈",
            "",
            " 建议:",
            "  - 交叉比对几类基因，找出关键调控因子",
            "  - 结合 GRN 分析，构建调控网络",
            "  - 使用 execute_python 进行更深入的交叉分析",
        ])
        
        return "\n".join(lines)


    # =========================================================================
    # 生长率分析 (from downstream_tools + analysis_toolkit)
    # =========================================================================
    @log_reason
    def analyze_growth_mass(self, save_dir: Optional[str] = None, reason: str = "") -> str:
        """Analyze cell mass distribution and growth rate statistics.

        Args:
            save_dir: Optional artifact output directory for this call.
            reason: Detailed reasoning and intent for calling this tool.

        Returns:
            A summary string with growth mass statistics and artifact paths.
        """
        guarded = self._invoke_core_in_subprocess(
            method="analyze_growth_mass",
            kwargs={"save_dir": save_dir},
            timeout=max(300, self.subprocess_guard_timeout),
        )
        if guarded is not None:
            if not guarded.get("ok", False):
                return (
                    " growth/mass analysis failed in isolated tool process.\n"
                    f"error: {guarded.get('error', 'unknown')}\n"
                    f"stderr: {str(guarded.get('stderr_tail', ''))[:800]}"
                )
            result = self._result_from_guarded_payload(guarded, "Growth/Mass Analysis")
        else:
            try:
                result = self._downstream_core.analyze_growth_mass(
                    adata=self.adata,
                    save_dir=save_dir,
                )
            except Exception as exc:
                logger.exception("analyze_growth_mass failed")
                return f" growth/mass analysis failed: {exc}"
        result.artifacts = self._apply_publication_bundle_to_artifacts(result.artifacts)
        self.analysis_results["growth_mass"] = {
            "artifacts": result.artifacts,
            "summary": result.summary,
            "warnings": result.warnings,
            "payload": result.payload,
        }
        return result.render_text()


    # =========================================================================
    # 随机性分析 (from downstream_tools)
    # =========================================================================
    @log_reason
    def analyze_stochasticity(self, include_gene_projection: bool = False, reason: str = "") -> str:
        """Analyze stochasticity using the score field and gene-space projection.

        Args:
            include_gene_projection: Whether to compute optional gene-space projection (slower).
            reason: Detailed reasoning and intent for calling this tool.

        Returns:
            A summary string with stochasticity statistics and artifact paths.
        """
        guarded = self._invoke_tool_method_in_subprocess(
            method="analyze_stochasticity",
            kwargs={
                "include_gene_projection": bool(include_gene_projection),
                "reason": reason,
            },
            timeout=max(300, self.subprocess_guard_timeout),
        )
        if guarded is not None:
            return self._consume_guarded_tool_result(guarded, "stochasticity analysis")

        apply_matplotlib_style(self.figure_quality_preset)
        outdir = _ensure_dir(self.output_dir / "stochasticity")
        artifacts = {}
        summary_parts = []
        numerical_data = {}
        
        score = _get_score_latent(self.adata)
        
        if score is not None:
            # Compute score magnitude
            score_mag = np.linalg.norm(score, axis=1)
            self.adata.obs["score_magnitude"] = score_mag
            
            # Save numerical statistics
            stats = {
                "mean": float(score_mag.mean()),
                "std": float(score_mag.std()),
                "min": float(score_mag.min()),
                "max": float(score_mag.max()),
                "median": float(np.median(score_mag)),
                "q25": float(np.percentile(score_mag, 25)),
                "q75": float(np.percentile(score_mag, 75))
            }
            numerical_data["score_magnitude_stats"] = stats
            
            summary_parts.append(f" Score场统计: 均值={stats['mean']:.4f}, 标准差={stats['std']:.4f}")
            
            # Plot on embedding
            try:
                basis = self._ensure_embedding()
                fig, ax = plt.subplots(figsize=(8, 6))
                sc.pl.embedding(self.adata, basis=basis, color="score_magnitude", ax=ax, show=False)
                bundle = save_figure_bundle(
                    fig,
                    base_name=f"score_magnitude_{basis}",
                    output_dir=self.figures_dir,
                    preset=self.figure_quality_preset,
                )
                plt.close(fig)
                artifacts.update({f"score_embedding_{k}": v for k, v in bundle.items()})
                summary_parts.append(f"Plotted score magnitude on {basis}")

            except Exception as e:
                logger.warning(f"Score embedding plot failed: {e}")
            
            should_gene_projection = bool(include_gene_projection) or (
                (not self.latent_first_default) and self.adata.n_obs <= 100000
            )
            if should_gene_projection:
                proj_stats = self._get_gene_projection_stats(score)
                if proj_stats is not None:
                    gene_out = _ensure_dir(outdir / "gene_score")
                    mean_vec, mean_abs = proj_stats
                    gene_names, _ = _detect_gene_name_column(self.adata)
                    artifacts.update(
                        _write_gene_summary(
                            gene_out,
                            "score",
                            gene_names,
                            mean_vec,
                            mean_abs,
                            figure_dir=self.figures_dir,
                        )
                    )
                    
                    # Save top genes data for LLM
                    top_n = 20
                    top_idx = np.argsort(mean_abs)[::-1][:min(top_n, len(gene_names))]
                    top_genes_data = [
                        {"gene": str(gene_names[i]), "mean": float(mean_vec[i]), "mean_abs": float(mean_abs[i])}
                        for i in top_idx
                    ]
                    numerical_data["top_genes"] = top_genes_data
                    
                    summary_parts.append(" 计算基因层Score摘要")
            else:
                summary_parts.append(" 默认跳过基因空间投影（latent优先模式）")
            
            # Save JSON data (inside the if block to ensure stats exists)
            json_path = outdir / "stochasticity_data.json"
            with json_path.open('w', encoding='utf-8') as f:
                json.dump(numerical_data, f, indent=2)
            artifacts["data"] = str(json_path)
        else:
            summary_parts.append(" 未找到Score场（模型可能是ODE类型，无随机性项）")
        
        self.analysis_results["stochasticity"] = {
            "artifacts": artifacts, 
            "summary": summary_parts,
            "numerical_data": numerical_data
        }
        
        # Build detailed text report with numerical data
        lines = [
            "=" * 60,
            " 随机性分析结果",
            "=" * 60,
            ""
        ]
        
        if "score_magnitude_stats" in numerical_data:
            stats = numerical_data["score_magnitude_stats"]
            lines.extend([
                " Score场统计:",
                f"  均值: {stats['mean']:.4f}",
                f"  标准差: {stats['std']:.4f}",
                f"  最小值: {stats['min']:.4f}",
                f"  最大值: {stats['max']:.4f}",
                f"  中位数: {stats['median']:.4f}",
                ""
            ])
            
            if "top_genes" in numerical_data:
                lines.extend([
                    f" Top {len(numerical_data['top_genes'])} 随机性相关基因:",
                    ""
                ])
                for i, gene_data in enumerate(numerical_data["top_genes"][:10], 1):
                    lines.append(f"  {i}. {gene_data['gene']}: {gene_data['mean_abs']:.4f}")
                lines.append("")

        lines.extend([
            " 解释:",
            "  - score_magnitude 越大: 随机性/扩散越强，状态更不稳定",
            "  - Top 基因代表随机项在基因空间中的主要方向",
            "",
            " 生成的文件:",
            *[f"  {k}: {v}" for k, v in artifacts.items()],
            "=" * 60
        ])
        
        return "\n".join(lines)

    # =========================================================================
    # 细胞互作分析 (from downstream_tools)
    # =========================================================================
    @log_reason
    def analyze_interaction(
        self,
        include_gene_projection: bool = False,
        include_stream_plot: bool = False,
        reason: str = "",
    ) -> str:
        """Analyze interaction forces and gene-level summaries.

        Args:
            include_gene_projection: Whether to compute optional gene-space projection (slower).
            include_stream_plot: Whether to run CytoBridge interaction stream plotting (slower).
            reason: Detailed reasoning and intent for calling this tool.

        Returns:
            A summary string with interaction statistics and artifact paths.
        """
        guarded = self._invoke_tool_method_in_subprocess(
            method="analyze_interaction",
            kwargs={
                "include_gene_projection": bool(include_gene_projection),
                "include_stream_plot": bool(include_stream_plot),
                "reason": reason,
            },
            timeout=max(300, self.subprocess_guard_timeout),
        )
        if guarded is not None:
            return self._consume_guarded_tool_result(guarded, "interaction analysis")

        outdir = _ensure_dir(self.output_dir / "interaction")
        artifacts = {}
        summary_parts = []
        numerical_data = {}
        
        # Optional stream plotting (disabled by default for fast-first runtime).
        if include_stream_plot:
            try:
                import CytoBridge as cb
                plot_basis = self._ensure_embedding()
                fig_dir = self.figures_dir
                cb.pl.plot.plot_interaction_stream(
                    self.adata, output_path=str(fig_dir),
                    dim_reduction=plot_basis, device=self.device
                )
                if (fig_dir / "Interaction_Force_Stream_Plot.svg").exists():
                    artifacts["interaction_stream"] = str(fig_dir / "Interaction_Force_Stream_Plot.svg")
                    summary_parts.append(" 生成互作流图")
            except Exception as e:
                logger.warning(f"Interaction stream failed: {e}")
        else:
            summary_parts.append(" 默认跳过 interaction stream（fast 模式）")
        
        # Gene-space interaction summary
        interaction_force = _get_interaction_force(self.adata)
        if interaction_force is not None:
            # Compute interaction force magnitude
            force_mag = np.linalg.norm(interaction_force, axis=1)
            self.adata.obs["interaction_force_magnitude"] = force_mag
            
            # Save numerical statistics
            stats = {
                "mean": float(force_mag.mean()),
                "std": float(force_mag.std()),
                "min": float(force_mag.min()),
                "max": float(force_mag.max()),
                "median": float(np.median(force_mag)),
                "q25": float(np.percentile(force_mag, 25)),
                "q75": float(np.percentile(force_mag, 75))
            }
            numerical_data["interaction_force_stats"] = stats
            
            should_gene_projection = bool(include_gene_projection) or (
                (not self.latent_first_default) and self.adata.n_obs <= 100000
            )
            if should_gene_projection:
                proj_stats = self._get_gene_projection_stats(interaction_force)
                if proj_stats is not None:
                    gene_out = _ensure_dir(outdir / "gene_interaction")
                    mean_vec, mean_abs = proj_stats
                    gene_names, _ = _detect_gene_name_column(self.adata)
                    artifacts.update(
                        _write_gene_summary(
                            gene_out,
                            "interaction_force",
                            gene_names,
                            mean_vec,
                            mean_abs,
                            figure_dir=self.figures_dir,
                        )
                    )
                    
                    # Save top genes data for LLM
                    top_n = 20
                    top_idx = np.argsort(mean_abs)[::-1][:min(top_n, len(gene_names))]
                    top_genes_data = [
                        {"gene": str(gene_names[i]), "mean": float(mean_vec[i]), "mean_abs": float(mean_abs[i])}
                        for i in top_idx
                    ]
                    numerical_data["top_genes"] = top_genes_data
                    
                    summary_parts.append(" 计算基因层互作摘要")
            else:
                summary_parts.append(" 默认跳过基因空间投影（latent优先模式）")
            
            # Save JSON data (inside the if block to ensure stats exists)
            json_path = outdir / "interaction_data.json"
            with json_path.open('w', encoding='utf-8') as f:
                json.dump(numerical_data, f, indent=2)
            artifacts["data"] = str(json_path)
        else:
            summary_parts.append(" 未找到互作力数据")
        
        self.analysis_results["interaction"] = {
            "artifacts": artifacts, 
            "summary": summary_parts,
            "numerical_data": numerical_data
        }
        
        # Build detailed text report with numerical data
        lines = [
            "=" * 60,
            " 细胞互作分析结果",
            "=" * 60,
            ""
        ]
        
        if "interaction_force_stats" in numerical_data:
            stats = numerical_data["interaction_force_stats"]
            lines.extend([
                " 互作力统计:",
                f"  均值: {stats['mean']:.4f}",
                f"  标准差: {stats['std']:.4f}",
                f"  最小值: {stats['min']:.4f}",
                f"  最大值: {stats['max']:.4f}",
                f"  中位数: {stats['median']:.4f}",
                ""
            ])
            
            if "top_genes" in numerical_data:
                lines.extend([
                    f" Top {len(numerical_data['top_genes'])} 互作相关基因:",
                    ""
                ])
                for i, gene_data in enumerate(numerical_data["top_genes"][:10], 1):
                    lines.append(f"  {i}. {gene_data['gene']}: {gene_data['mean_abs']:.4f}")
                lines.append("")

        lines.extend([
            " 解释:",
            "  - 互作力幅度越大: 细胞间相互作用越强",
            "  - Top 基因代表互作力在基因空间中的主要方向",
            "",
            " 生成的文件:",
            *[f"  {k}: {v}" for k, v in artifacts.items()],
            "=" * 60
        ])
        
        return "\n".join(lines)


    # =========================================================================
    # 代码执行工具 (from analysis_toolkit)
    # =========================================================================
    @log_reason
    def execute_python(self, code: str, reason: str = "") -> str:
        """Execute Python code and manage generated artifacts.
        IMPORTANT - The execution environment is PRE-LOADED with:
        1. Variables: 
        - 'adata': The current AnnData object (use this directly, DO NOT load new data).
        - 'output_dir': Root output directory path. All intermediate files (JSON, CSV) are in subdirectories here.
        - 'figures_dir': Path to save figures. DO NOT use os.path.dirname(output_dir).
        - 'save_pubfig': Helper for publication-quality export bundle (default png+svg with preset dpi).
        - 'load_result_table': Load existing csv/json/npz from output_dir first (supports glob).
        - 'pick_top_figures': Select top-K figures from metrics metadata for main-text curation.
        - 'build_appendix_gallery': Build appendix HTML fragment from figure metadata.
        - 'make_figure_name': Build stem `<analysis>__<plot_type>__<variant>`.
        2. Libraries: 
        - scanpy as sc, numpy as np, pandas as pd, matplotlib.pyplot as plt, os.
        3. Figure Saving Rules:
        - **MUST save figures explicitly** using `save_pubfig(...)` (preferred) or plt.savefig().
        - **Rerun-first for plotting**: For plotting/redraw tasks, rerun the relevant analysis chain before plotting.
          Existing outputs can be used as reference, not as the primary delivery in redraw turns.
        - **MUST use DESCRIPTIVE filenames** that explain what the figure shows.
          Good: "cd8t_vs_monocyte_fate_comparison.png", "growth_rate_by_celltype.png"
          Bad:  "plot1.png", "figure.png", "output.png"
        - Preferred: `save_pubfig(fig, name=make_figure_name("trajectory","overlay","v1"))` (auto style + bundle export).
        4. Notes:
        - In-place modifications (e.g. sc.pp.neighbors(adata)) are preferred.
        - If you must reassign adata, ensure you assign it back to 'adata'.



        When encourtering lookup errors, set use_raw=False to avoid lookup errors.

        Args:
            code: Python code snippet to execute.
            reason: Detailed reasoning and intent for calling this tool.

        Returns:
            A formatted execution summary including output and artifacts.
        """
        guarded = self._invoke_tool_method_in_subprocess(
            method="execute_python",
            kwargs={"code": code, "reason": reason},
            timeout=max(1200, self.subprocess_guard_timeout),
        )
        if guarded is not None:
            return self._consume_guarded_tool_result(guarded, "python execution")

        if self.code_executor.adata is not self.adata:
             self.code_executor.adata = self.adata
        # Execute code
        result = self.code_executor.execute(code)
        if result.get("success") and result.get("new_adata") is not None:
            logger.info("Execute_python: 同步新的 adata 对象到 Toolkit")
            self.adata = result["new_adata"]
            # ！！！关键！！！
            # 因为数据变了，之前的缓存状态（如 get_data_state 的结果）全部失效，必须清空
            self._data_state = None 
            
            # 可选：如果数据变了，可能需要重新做某些轻量级检查或打印日志
            output_parts = [" 检测到数据对象已更新 (Data Object Updated)"]
        else:
            output_parts = []
        
        self.analysis_history.append({
            "type": "code_execution", "code": code,
            "reason": reason, "success": result["success"],
        })
        if result["success"]:
            output_parts.append(" 执行成功")
            if result["output"]:
                output_parts.append(f"\n 输出:\n{result['output']}")
            if result.get("result") is not None:
                result_str = str(result["result"])
                if len(result_str) > 500:
                    result_str = result_str[:500] + "..."
                output_parts.append(f"\n 返回值: {result_str}")
            
            # Move images to figures/ and normalize paths
            images = result.get("images", [])
            if images:
                figures_dir = self.figures_dir
                figures_dir.mkdir(exist_ok=True)
                
                moved_images = []
                import shutil
                
                for img_name in images:
                    rel_path = Path(str(img_name))
                    rel_posix = rel_path.as_posix()
                    src = self.output_dir / rel_path

                    # Already under figures/: keep as-is.
                    if rel_path.parts and rel_path.parts[0] == "figures":
                        if src.exists():
                            moved_images.append(rel_posix)
                        continue

                    # If src is missing, check fallback basename under figures/.
                    if not src.exists():
                        fallback = figures_dir / rel_path.name
                        if fallback.exists():
                            moved_images.append(f"figures/{rel_path.name}")
                        continue

                    dst = figures_dir / rel_path.name
                    try:
                        shutil.move(str(src), str(dst))
                        moved_images.append(f"figures/{rel_path.name}")
                    except Exception as e:
                        logger.warning(f"Failed to move artifact {img_name}: {e}")
                        # If move failed (e.g. same file), still report it if it exists
                        if dst.exists():
                            moved_images.append(f"figures/{rel_path.name}")

                if moved_images:
                    output_parts.append(f"\n [Generated Artifacts]:")
                    for img in moved_images:
                        output_parts.append(f" - {img}")
        else:
            output_parts.append(" 执行失败")
            output_parts.append(f"\n错误:\n{result['error']}")
        
        return truncate_output("\n".join(output_parts))


    # =========================================================================
    # 可视化工具 (from analysis_toolkit)
    # =========================================================================
    
#     def plot_umap(self, color_by: Optional[str] = None, save_name: str = "umap", reason: str = "") -> str:
#         """Generate and save a UMAP plot.

#         Args:
#             color_by: Optional ``obs`` column to color points.
#             save_name: Base filename for the saved figure (without extension).
#             reason: Detailed reasoning and intent for calling this tool.

#         Returns:
#             A formatted execution summary including the saved figure path.
#         """
#         code = f"""
# import matplotlib.pyplot as plt
# import scanpy as sc
# import numpy as np

# if 'X_umap' not in adata.obsm:
#     if 'X_pca' not in adata.obsm:
#         sc.pp.pca(adata)
#     sc.pp.neighbors(adata)
#     sc.tl.umap(adata)

# fig, ax = plt.subplots(figsize=(8, 6))
# sc.pl.umap(adata, color={repr(color_by)}, ax=ax, show=False)
# plt.tight_layout()
# plt.savefig(f"{{output_dir}}/{save_name}.png", dpi=150, bbox_inches='tight')
# plt.close(fig)
# print(f"UMAP 图已保存: {save_name}.png")
# """

#         return self.execute_python(code, reason=f"绘制 UMAP (color_by={color_by})")

    @log_reason
    def compute_correlation(self, var1: str, var2: str, reason: str = "") -> str:
        """Compute Pearson and Spearman correlation between two variables.

        Args:
            var1: Variable name from ``obs`` or ``obsm`` for the first series.
            var2: Variable name from ``obs`` or ``obsm`` for the second series.
            reason: Detailed reasoning and intent for calling this tool.

        Returns:
            A formatted execution summary including correlation statistics and plot.
        """
        code = f"""
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np

def get_data(name):
    if name in adata.obs.columns:
        return adata.obs[name].values
    elif name in adata.obsm.keys():
        data = adata.obsm[name]
        return data.flatten() if hasattr(data, 'flatten') else data
    raise ValueError(f"'{{name}}' 不在 obs 或 obsm 中")

data1 = get_data('{var1}')
data2 = get_data('{var2}')

pearson_r, pearson_p = stats.pearsonr(data1, data2)
spearman_r, spearman_p = stats.spearmanr(data1, data2)

print(f" 相关性分析: '{var1}' vs '{var2}'")
print("=" * 50)
print(f"Pearson:  r = {{pearson_r:.4f}}, p = {{pearson_p:.2e}}")
print(f"Spearman: ρ = {{spearman_r:.4f}}, p = {{spearman_p:.2e}}")

# 绘图
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(data1, data2, alpha=0.5, s=5)
ax.set_xlabel('{var1}')
ax.set_ylabel('{var2}')
ax.set_title(f'r = {{pearson_r:.3f}}, p = {{pearson_p:.2e}}')
z = np.polyfit(data1, data2, 1)
p = np.poly1d(z)
x_line = np.linspace(data1.min(), data1.max(), 100)
ax.plot(x_line, p(x_line), 'r--', alpha=0.8)
plt.tight_layout()
save_pubfig(fig, name="correlation_{var1}_vs_{var2}")
plt.close(fig)
print(f"相关性图已保存: correlation_{var1}_vs_{var2}.png")
"""

        return self.execute_python(code, reason=f"计算相关性 ({var1} vs {var2})")


    # =========================================================================
    # 数据探索工具 (from analysis_toolkit)
    # =========================================================================
    @log_reason
    def get_obs_column(self, column: str, n: int = 20, reason: str = "") -> str:
        """Inspect an ``obs`` column and summarize its distribution.

        Args:
            column: ``obs`` column name to inspect.
            n: Maximum number of categories to display for categorical columns.
            reason: Detailed reasoning and intent for calling this tool.

        Returns:
            A formatted summary of the column statistics or category counts.
        """
        if column not in self.adata.obs.columns:
            return f" 列 '{column}' 不存在\n可用列: {list(self.adata.obs.columns)}"
        
        col_data = self.adata.obs[column]
        lines = [f" 列信息: {column}", "=" * 40,
                 f"数据类型: {col_data.dtype}", f"唯一值数: {col_data.nunique()}"]
        
        if col_data.dtype in ['object', 'category']:
            lines.append("\n值分布:")
            for val, count in col_data.value_counts().head(n).items():
                lines.append(f"  {val}: {count} ({count/len(col_data)*100:.1f}%)")
        else:
            lines.extend([f"\n统计量:", f"  均值: {col_data.mean():.4f}",
                         f"  标准差: {col_data.std():.4f}", f"  中位数: {col_data.median():.4f}"])
        
        return truncate_output("\n".join(lines))
    
    @log_reason
    def search_genes(self, pattern: str, n: int = 20, reason: str = "") -> str:
        """Search gene names by regex pattern.

        Args:
            pattern: Regular expression pattern for gene name matching.
            n: Maximum number of matched genes to display.
            reason: Detailed reasoning and intent for calling this tool.

        Returns:
            A summary string listing matched genes.
        """
        import re
        gene_names, gene_col = _detect_gene_name_column(self.adata)
        matching = [g for g in gene_names if re.search(pattern, str(g), re.IGNORECASE)]
        
        lines = [f" 搜索结果: '{pattern}'", "=" * 40,
                 f"来源: {gene_col}", f"匹配数: {len(matching)}", f"结果: {matching[:n]}"]
        return "\n".join(lines)
    
    @log_reason
    def get_gene_expression(self, genes: List[str], n_cells: int = 10, reason: str = "") -> str:
        """Summarize expression statistics for selected genes.

        Args:
            genes: List of gene identifiers or names to query.
            n_cells: Reserved for compatibility; current implementation scans all cells.
            reason: Detailed reasoning and intent for calling this tool.

        Returns:
            A formatted summary of expression mean, std, and nonzero fraction.
        """
        gene_names, gene_col = _detect_gene_name_column(self.adata)
        
        # 尝试匹配基因名
        gene_indices = []
        for g in genes:
            if g in self.adata.var_names:
                gene_indices.append((g, g))
            elif gene_col != 'var_names':
                # 在基因名列中搜索
                matches = [i for i, name in enumerate(gene_names) if str(name) == g]
                if matches:
                    gene_indices.append((self.adata.var_names[matches[0]], g))
        
        if not gene_indices:
            return f" 未找到基因: {genes}"
        
        lines = [f" 基因表达信息", "=" * 40]
        
        # 批处理计算，避免内存溢出
        batch_size = 5000  # 每次处理 5000 个细胞
        n_cells_total = self.adata.n_obs
        
        for var_name, display_name in gene_indices:
            # 分批计算统计数据
            expr_sum = 0.0
            expr_sq_sum = 0.0
            nonzero_count = 0
            
            for start_idx in range(0, n_cells_total, batch_size):
                end_idx = min(start_idx + batch_size, n_cells_total)
                
                # 只加载这一批数据
                expr_batch = self.adata[start_idx:end_idx, var_name].X
                if hasattr(expr_batch, 'toarray'):
                    expr_batch = expr_batch.toarray().flatten()
                else:
                    expr_batch = expr_batch.flatten()
                
                # 累积统计
                expr_sum += expr_batch.sum()
                expr_sq_sum += (expr_batch ** 2).sum()
                nonzero_count += (expr_batch > 0).sum()
            
            # 计算最终统计数据
            mean_val = expr_sum / n_cells_total
            variance = (expr_sq_sum / n_cells_total) - (mean_val ** 2)
            std_val = np.sqrt(max(0, variance))  # 避免负数
            nonzero_pct = (nonzero_count / n_cells_total) * 100
            
            lines.extend([
                f"\n{display_name}:",
                f"  均值: {mean_val:.4f}",
                f"  标准差: {std_val:.4f}",
                f"  非零比例: {nonzero_pct:.1f}%"
            ])
        
        return truncate_output("\n".join(lines))

    # =========================================================================
    # 轨迹生成工具 (from analysis_toolkit)
    # =========================================================================
    @log_reason
    def generate_trajectories(self, n_trajectories: int = 50, method: str = "ode", reason: str = "") -> str:
        """Generate cell trajectories using ODE or SDE simulators.

        Args:
            n_trajectories: Number of trajectories to generate.
            method: Simulation method: "ode" or "sde".
            reason: Detailed reasoning and intent for calling this tool.

        Returns:
            A formatted execution summary including saved trajectory files.
        """
        code = f"""
try:
    import CytoBridge as cb
    from CytoBridge.utils import load_model_from_adata
    import torch
    import numpy as np
    
    model = load_model_from_adata(adata)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()
    
    print(f"Model loaded on device: {{device}}")
    
    if '{method}' == 'ode':
        from CytoBridge.tl.downstream import generate_ode_trajectory_bundle
        bundle = generate_ode_trajectory_bundle(
            model=model, adata=adata, output_dir=str(output_dir),
            n_trajectories={n_trajectories}, n_bins=100, device=device
        )
        point_array = np.load(bundle['artifacts']['ode_point'])
        traj_array = np.load(bundle['artifacts']['ode_traj'])
        print(f"ODE trajectory generation completed: {{traj_array.shape}}")
        np.save(output_dir / 'ode_trajectories.npy', traj_array)
        np.save(output_dir / 'ode_points.npy', point_array)
        print(f"Saved to {{output_dir}}")
    else:
        from CytoBridge.tl.downstream import generate_sde_trajectory_bundle
        bundle = generate_sde_trajectory_bundle(
            adata=adata, output_dir=str(output_dir), device=device,
            n_time_steps=100, sample_traj_num={n_trajectories}, init_time=0
        )
        sde_traj = np.load(bundle['artifacts']['sde_traj'])
        print(f"SDE trajectory generation completed: {{sde_traj.shape}}")
        
except ImportError as e:
    print(f"CytoBridge is not installed: {{e}}")
except Exception as e:
    print(f"Trajectory generation failed: {{e}}")
    import traceback
    traceback.print_exc()
"""
        return self.execute_python(code, reason=f"generate {method.upper()} trajectories")

    # =========================================================================
    # GRN 分析 (from downstream_tools)
    # =========================================================================
    @log_reason
    def analyze_grn(
        self,
        max_genes: int = 10,
        max_time_points: int = 5,
        genes: Optional[Any] = None,
        save_dir: Optional[str] = None,
        reason: str = "",
    ) -> str:
        """Infer a gene regulatory network using the velocity Jacobian.

        Args:
            max_genes: Maximum number of genes to include in the GRN.
            max_time_points: Maximum number of time points to analyze.
            genes: Optional list of specific genes to include.
            save_dir: Optional artifact output directory for this call.
            reason: Detailed reasoning and intent for calling this tool.

        Returns:
            A summary string describing GRN results and artifact paths.
        """
        genes_arg = genes
        if isinstance(genes, str):
            import ast
            try:
                parsed = ast.literal_eval(genes)
                if isinstance(parsed, (list, tuple)):
                    genes_arg = [str(x) for x in parsed]
                else:
                    genes_arg = [str(parsed)]
            except (ValueError, SyntaxError):
                genes_arg = [g.strip().strip("'\"") for g in genes.strip("[]").split(",") if g.strip()]
        elif genes is not None:
            genes_arg = [str(x) for x in list(genes)]

        guarded = self._invoke_core_in_subprocess(
            method="analyze_grn",
            kwargs={
                "max_genes": int(max_genes),
                "max_time_points": int(max_time_points),
                "genes": genes_arg,
                "save_dir": save_dir,
            },
            timeout=max(300, self.subprocess_guard_timeout),
        )
        if guarded is not None:
            if not guarded.get("ok", False):
                return (
                    " GRN analysis failed in isolated tool process.\n"
                    f"error: {guarded.get('error', 'unknown')}\n"
                    f"stderr: {str(guarded.get('stderr_tail', ''))[:800]}"
                )
            result = self._result_from_guarded_payload(guarded, "GRN Analysis")
        else:
            try:
                result = self._downstream_core.analyze_grn(
                    adata=self.adata,
                    max_genes=int(max_genes),
                    max_time_points=int(max_time_points),
                    genes=genes_arg,
                    save_dir=save_dir,
                )
            except Exception as exc:
                logger.exception("analyze_grn failed")
                return f" GRN analysis failed: {exc}"
        self.analysis_results["grn"] = {
            "artifacts": result.artifacts,
            "summary": result.summary,
            "warnings": result.warnings,
            "payload": result.payload,
            "grn_data": result.payload.get("grn", {}),
        }
        return result.render_text()
    
    def _compute_velocity_jacobian(self, model: Any, z_np: np.ndarray, t_np: np.ndarray) -> np.ndarray:
        """Compute the velocity-field Jacobian with respect to latent inputs.

        Args:
            model: Trained model with a ``velocity_net`` module.
            z_np: Latent state array of shape (n_cells, n_dims).
            t_np: Time array of shape (n_cells, 1).

        Returns:
            A Jacobian matrix averaged across cells, shape (n_dims, n_dims).
        """
        z_t = torch.tensor(z_np, device=self.device, dtype=torch.float32).requires_grad_(True)
        t_t = torch.tensor(t_np, device=self.device, dtype=torch.float32)
        net_input = torch.cat([z_t, t_t], dim=1)
        v = model.velocity_net(net_input)
        dim = v.shape[1]
        jac = torch.zeros(dim, dim, device=self.device)
        for i in range(dim):
            grad = torch.autograd.grad(
                outputs=v[:, i].sum(), inputs=z_t,
                retain_graph=True, create_graph=False, only_inputs=True
            )[0]
            jac[i, :] = grad.mean(0)
        return jac.detach().cpu().numpy()

    # =========================================================================
    # 报告生成工具 (from analysis_toolkit)
    # =========================================================================
    
    def generate_biological_report(self, question: str, findings: List[str], reason: str = "") -> str:
        """Generate a markdown biological analysis report and save it.

        Args:
            question: Research question or user query to address.
            findings: List of key findings to include in the report.
            reason: Detailed reasoning and intent for calling this tool.

        Returns:
            A report string including the saved report path and report contents.
        """
        state = self.get_data_state()
        
        report_lines = [
            "# CytoBridge 动力学分析报告",
            "",
            "## 1. 数据概况",
            f"- 细胞数: {state['n_cells']}",
            f"- 基因数: {state['n_genes']}",
            f"- 时间点: {state['time_points']}",
            f"- 模型类型: {state['inferred_model_type']}",
            f"- 模型组件: {state['model_components']}",
            f"- 基因层分析: {'可用' if state.get('gene_space_available') else '不可用'}",
            "",
            "## 2. 研究问题",
            f"> {question}",
            "",
            "## 3. 分析方法",
        ]
        
        model_type = state['inferred_model_type']
        if model_type == "unbalanced_ot":
            report_lines.extend([
                "本分析使用 **非平衡最优传输 (Unbalanced OT)** 模型：",
                "- 学习细胞状态转移的速度场 (velocity)",
                "- 估计细胞增殖/凋亡率 (growth rate)",
            ])
        elif model_type == "ruot_or_crufm":
            report_lines.extend([
                "本分析使用 **正则化非平衡最优传输 (RUOT/CRUFM)** 模型：",
                "- 学习速度场 + 生长率 + 随机扩散",
            ])
        
        report_lines.extend([
            "",
            "## 4. 关键发现",
        ])
        
        for i, finding in enumerate(findings, 1):
            report_lines.append(f"{i}. {finding}")
        
        report_lines.extend([
            "",
            "## 5. 生成的图表",
            f"所有图表保存在: `{self.output_dir}`",
            "",
            "---",
            f"*报告生成时间: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
        ])
        
        report_content = "\n".join(report_lines)
        
        report_path = self.output_dir / "analysis_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        return f" 报告已生成: {report_path}\n\n{report_content}"

    # =========================================================================
    # 辅助方法
    # =========================================================================
    
    def get_analysis_history(self, reason: str = "") -> List[Dict[str, Any]]:
        """Return the logged analysis history records.

        Args:
            reason: Detailed reasoning and intent for calling this tool.

        Returns:
            A list of analysis history entries.
        """
        return self.analysis_history
    
    def get_all_results(self, reason: str = "") -> Dict[str, Any]:
        """Return cached analysis results for all analyses.

        Args:
            reason: Detailed reasoning and intent for calling this tool.

        Returns:
            A dictionary of analysis results keyed by analysis name.
        """
        return self.analysis_results
    
    def save_analysis_summary(self, filename: str = "analysis_summary.json", reason: str = "") -> str:
        """Serialize and save a JSON analysis summary.

        Args:
            filename: Output filename for the summary JSON.
            reason: Detailed reasoning and intent for calling this tool.

        Returns:
            The file path to the saved summary JSON.
        """
        # 确保所有数据都可以序列化
        def make_serializable(obj: Any) -> Any:
            """Recursively convert objects to JSON-serializable types.

            Args:
                obj: Python object to convert.

            Returns:
                A JSON-serializable representation of the input object.
            """
            if isinstance(obj, dict):
                return {str(k): make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [make_serializable(item) for item in obj]
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, Path):
                return str(obj)
            else:
                return obj
        
        summary = {
            "adata_path": str(self.adata_path),
            "training_metrics": make_serializable(self.training_metrics),
            "user_requested_analyses": self.user_requested_analyses,
            "data_state": make_serializable(self.get_data_state()),
            "analysis_results": {
                k: {
                    "summary": v.get("summary", []) if isinstance(v.get("summary"), list) else [str(v.get("summary", ""))],
                    "artifacts": make_serializable(v.get("artifacts", {}))
                } 
                for k, v in self.analysis_results.items()
            },
            "n_analyses": len(self.analysis_history),
        }
        
        save_path = self.output_dir / filename
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Analysis summary saved to {save_path}")
        return str(save_path)

    # =========================================================================
    # Cell Type Classifier Training
    # =========================================================================
    @log_reason
    def train_cell_classifier(
        self,
        label_key: Optional[str] = None,
        hidden_dim: int = 128,
        epochs: int = 200,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        validation_split: float = 0.2,
        save_model: bool = True,
        save_filename: Optional[str] = None,
        reason: str = "",
    ) -> str:
        """Train an MLP classifier for cell type prediction in latent space.

        This tool trains a neural network classifier that can predict cell types
        from latent representations. The trained model can be used for more 
        accurate cell fate classification in perturbation analysis.

        Args:
            label_key: Column for cell type labels (auto-detected if None).
            hidden_dim: Hidden layer dimension (default: 128).
            epochs: Number of training epochs (default: 200).
            batch_size: Training batch size (default: 256).
            learning_rate: Learning rate for Adam optimizer (default: 0.001).
            validation_split: Fraction of data for validation (default: 0.2).
            save_model: Whether to save the trained model (default: True).
            save_filename: Optional custom filename (without extension) to save results.
                           Useful to train multiple classifiers for different labels (e.g. 'label_1', 'label_2').
            reason: Detailed reasoning and intent for calling this tool.

        Returns:
            A formatted report with training results, save path, and inference instructions.

        """
        guarded = self._invoke_tool_method_in_subprocess(
            method="train_cell_classifier",
            kwargs={
                "label_key": label_key,
                "hidden_dim": int(hidden_dim),
                "epochs": int(epochs),
                "batch_size": int(batch_size),
                "learning_rate": float(learning_rate),
                "validation_split": float(validation_split),
                "save_model": bool(save_model),
                "save_filename": save_filename,
                "reason": reason,
            },
            timeout=max(1200, self.subprocess_guard_timeout),
        )
        if guarded is not None:
            return self._consume_guarded_tool_result(guarded, "cell classifier training")

        from pathlib import Path
        
        outdir = _ensure_dir(self.output_dir / "classifiers")
        
        # Auto-detect label key
        if label_key is None:
            label_candidates = ['cell_type', 'Cell type annotation', 'celltype', 'cluster', 'leiden']
            for col in label_candidates:
                if col in self.adata.obs.columns:
                    label_key = col
                    break
        
        if label_key is None or label_key not in self.adata.obs.columns:
            return " 未找到细胞类型标签列。请指定 label_key 参数。"
        
        # Check for latent representation
        if 'X_latent' not in self.adata.obsm:
            return " 未找到 X_latent 潜在表示。请先进行 CytoBridge 训练。"
        
        lines = [
            "=" * 60,
            " 细胞类型分类器训练",
            "=" * 60,
            "",
            f" 标签列: {label_key}",
            f" 细胞数: {self.adata.n_obs}",
        ]
        
        # Get class distribution
        class_counts = self.adata.obs[label_key].value_counts()
        lines.append(f" 类别数: {len(class_counts)}")
        lines.append("")
        lines.append("类别分布:")
        for cls, cnt in class_counts.items():
            lines.append(f"  • {cls}: {cnt} ({cnt/self.adata.n_obs*100:.1f}%)")
        lines.append("")
        
        try:
            from CytoBridge.tl.perturbation import train_cell_classifier
            if save_model:
                if save_filename != None:
                    save_path = str(outdir / f"{save_filename}.pt")
                else:
                    save_path = str(outdir / "cell_classifier.pt")
            else:
                save_path = None
            
            lines.append(f" 超参数:")
            lines.append(f"  • hidden_dim: {hidden_dim}")
            lines.append(f"  • epochs: {epochs}")
            lines.append(f"  • batch_size: {batch_size}")
            lines.append(f"  • learning_rate: {learning_rate}")
            lines.append(f"  • validation_split: {validation_split}")
            lines.append("")
            lines.append(" 训练中...")
            
            result = train_cell_classifier(
                adata=self.adata,
                label_key=label_key,
                latent_key='X_latent',
                hidden_dim=hidden_dim,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                validation_split=validation_split,
                device=self.device,
                save_path=save_path,
            )
            
            lines.extend([
                "",
                " 训练完成!",
                f" 训练准确率: {result['train_accuracy']*100:.2f}%",
                f" 验证准确率: {result['val_accuracy']*100:.2f}%",
            ])
            
            if save_path:
                lines.append(f" 模型保存: {save_path}")
            
            # Store in analysis results
            self.analysis_results["cell_classifier"] = {
                "model_path": save_path,
                "label_key": label_key,
                "classes": result['classes'],
                "train_accuracy": result['train_accuracy'],
                "val_accuracy": result['val_accuracy'],
            }
            
            # Generate training curve plot
            try:
                import matplotlib.pyplot as plt
                
                history = result['history']
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                
                epochs_range = range(1, len(history['loss']) + 1)
                
                ax1.plot(epochs_range, history['loss'])
                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('Loss')
                ax1.set_title('Training Loss')
                
                ax2.plot(epochs_range, history['train_acc'], label='Train')
                ax2.plot(epochs_range, history['val_acc'], label='Validation')
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('Accuracy')
                ax2.set_title('Accuracy')
                ax2.legend()
                
                plt.tight_layout()
                fig_path = self.figures_dir / "training_curves.png"
                fig.savefig(fig_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                
                lines.append(f" 训练曲线: {fig_path.name}")
                
            except Exception as e:
                logger.warning(f"绘图失败: {e}")
            
            lines.extend([
                "",
                " 推理指南:",
                "  1. 加载模型:",
                "     ```python",
                "     from CytoBridge.tl.perturbation import load_mlp_classifier",
                f"     model, le = load_mlp_classifier('{save_path}')",
                "     ```",
                "  2. 预测 latent (假设 latent 是 (n, d) tensor):",
                "     ```python",
                "     logits = model(latent_tensor)",
                "     probs = torch.softmax(logits, dim=1)",
                "     preds = le.inverse_transform(logits.argmax(1).cpu().numpy())",
                "     ```",
                "=" * 60
            ])
            
        except ImportError as e:
            lines.append(f"\n 导入失败: {e}")
        except Exception as e:
            import traceback
            lines.append(f"\n 训练失败: {e}")
            logger.error(traceback.format_exc())
        
        return truncate_output("\n".join(lines))

    # =========================================================================
    # Trajectory Dataset Generation
    # =========================================================================
    @log_reason
    def generate_trajectory_dataset(
        self,
        n_cells: Optional[int] = None,
        n_steps: Optional[int] = None,
        init_cell_type: Optional[str] = None,
        label_key: Optional[str] = None,
        init_time_point: Optional[float] = None,
        end_time_point: Optional[float] = None,
        perturbed_genes: Optional[Any] = None,
        perturbed_z_score: List[float] = [0.0],
        save_filename: Optional[str] = None,
        output_space: str = "auto",
        reason: str = "",
    ) -> str:
        """Generate and export cell trajectory dataset for downstream analysis.

        This tool simulates cell trajectories using the trained CytoBridge model
        through the evaluation-compatible rollout kernel. To avoid memory spikes
        on large datasets, it supports auto-switching to latent-only export.

        **CytoBridge Unique Capability**:
        Unlike traditional single-cell analysis packages (Scanpy, Seurat), CytoBridge
        can track INDIVIDUAL cell trajectories over continuous time. This enables:
        - Interpolation between observed time points
        - Single-cell lineage tracking
        - Continuous dynamics modeling (not just pseudo-time)
        - Perturbation effect simulation at single-cell resolution

        **Resolved Config Rule**:
        - Rollout method, sigma, and integration dt are resolved from the
          fitted model's `resolved_config_yaml` in `adata.uns["all_model"]`.
        - This agent-facing tool intentionally does not accept `method` or
          `sigma` overrides. `n_steps` may be changed only to adjust output
          sampling density.

        **Time Explanation**:
        - Times correspond to values in `time_point_processed` column in the data.
        - Default: simulates from earliest to latest time in `time_point_processed`.
        - Each step represents: (end_time_point - init_time_point) / n_steps time units.
        - The LLM can calculate step-to-time mapping using this formula.

        Args:
            n_cells: Number of cells to simulate (default: 1000).
            n_steps: Number of output time steps. Default None uses resolved
                     config/evaluation trajectory step when available.
            init_cell_type: Filter for initial cell type (optional).
            label_key: Key of obs for cell type annotation.
            init_time_point: Starting time from time_point_processed (None = earliest).
            end_time_point: Ending time from time_point_processed (None = latest).
            perturbed_genes: Optional list of genes or comma-separated string to perturb. If no genes are provided, no perturbation will be applied.
            perturbed_z_score: Z-score for gene perturbation (default: [0]). 
                               Meaning: Standard deviations from mean. +/- 5.0 is a strong perturbation.
                               - must match length of perturbed_genes (1-to-1 mapping).
            save_filename: Optional custom filename (without extension) to save results.
                           Useful to simulate multiple scenarios (e.g. 'control', 'ko_geneA').
            output_space: 'auto' (recommended), 'latent', 'both', or 'gene'.
                          - auto: estimate memory, fallback to latent on large runs.
            reason: Detailed reasoning and intent for calling this tool.

        Returns:
            A formatted report with save paths and data loading instructions.

        Data Format:
            - trajectories_latent: (n_steps+1, n_cells, latent_dim) numpy array
            - trajectories_gene: (n_steps+1, n_cells, n_genes) numpy array (if output includes gene)
            - gene_names.txt: List of gene names in order (if output includes gene)
            - trajectory_metadata.json: Configuration and statistics

        Weights Interpretation:
            The 'weights' array (n_steps+1, n_cells) represents effective cell counts:
            - weight = 1.0: Represents 1 cell
            - weight = 2.0: Represents 2 cells (proliferation)
            - sum(weights[t]): Total population size at time t
        """
        guarded = self._invoke_tool_method_in_subprocess(
            method="generate_trajectory_dataset",
            kwargs={
                "n_cells": n_cells,
                "n_steps": n_steps,
                "init_cell_type": init_cell_type,
                "label_key": label_key,
                "init_time_point": init_time_point,
                "end_time_point": end_time_point,
                "perturbed_genes": perturbed_genes,
                "perturbed_z_score": perturbed_z_score,
                "save_filename": save_filename,
                "output_space": output_space,
                "reason": reason,
            },
            timeout=max(1800, self.subprocess_guard_timeout),
        )
        if guarded is not None:
            return self._consume_guarded_tool_result(guarded, "trajectory generation")

        from pathlib import Path
        
        outdir = _ensure_dir(self.output_dir / "trajectories")
        
        # Determine save directory based on filename if provided
        if save_filename:
             # Sanitize filename
             import re
             safe_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', save_filename)
             target_dir = _ensure_dir(outdir / safe_name)
        else:
             target_dir = outdir

        # Parse genes list
        if perturbed_genes is not None and isinstance(perturbed_genes, str):
            import ast
            try:
                # Try parsing as list literal
                perturbed_genes = ast.literal_eval(perturbed_genes)
            except:
                # Fallback to comma separation
                perturbed_genes = [g.strip() for g in perturbed_genes.split(',') if g.strip()]

        # Parse z-score list
        if isinstance(perturbed_z_score, str):
            import ast
            try:
                perturbed_z_score = ast.literal_eval(perturbed_z_score)
            except:
                perturbed_z_score = [float(z) for z in perturbed_z_score.split(',') if z.strip()]

        # Normalize trajectory export space policy
        output_space_req = str(output_space or "auto").strip().lower()
        if output_space_req not in {"auto", "latent", "gene", "both"}:
            output_space_req = "auto"

        # Keep None as "use fitted model resolved config". Use a conservative
        # estimate only for pre-run memory planning.
        try:
            n_steps_for_estimate = int(n_steps) if n_steps is not None else 100
        except Exception:
            n_steps_for_estimate = 100
        n_steps_for_estimate = max(1, n_steps_for_estimate)
        
        lines = [
            "=" * 60,
            "Cell trajectory dataset generation",
            "=" * 60,
            "",
            "CytoBridge model capability:",
            "  - Generate continuous single-cell trajectories",
            "  - Interpolate between observed time points",
            "  - Simulate perturbation effects at single-cell resolution",
            "",
            "Parameters:",
            f"  - cells: {n_cells}",
            f"  - output steps: {n_steps if n_steps is not None else 'auto(resolved_config)'}",
            "  - rollout method/sigma: auto(resolved_config/model)",
            f"  - requested output space: {output_space_req}",
        ]
        
        if perturbed_genes and perturbed_z_score != 0:
            lines.append(f"  - perturbed genes: {perturbed_genes} (z={perturbed_z_score})")
        
        lines.append("")
        
        try:
            from CytoBridge.utils import load_model_from_adata
            from CytoBridge.tl.perturbation import (
                generate_trajectory_dataset,
                perturb_gene_expression,
            )
            
            model = load_model_from_adata(self.adata)
            model.eval()

            # Resolve output space before running simulation. For large runs,
            # full gene-space trajectories can exceed memory quickly.
            resolved_output_space = output_space_req
            if output_space_req == "auto":
                est_n_cells = int(n_cells) if n_cells is not None else int(self.adata.n_obs)
                try:
                    time_key = "time_point_processed" if "time_point_processed" in self.adata.obs.columns else None
                    mask = np.ones(self.adata.n_obs, dtype=bool)
                    if time_key is not None:
                        if init_time_point is None:
                            tvals = pd.to_numeric(self.adata.obs[time_key], errors="coerce")
                            if not np.isnan(tvals.values).all():
                                t0 = float(np.nanmin(tvals.values))
                                mask &= np.isclose(tvals.values, t0)
                        else:
                            tvals = pd.to_numeric(self.adata.obs[time_key], errors="coerce")
                            mask &= np.isclose(tvals.values, float(init_time_point))
                    if init_cell_type is not None and label_key and label_key in self.adata.obs.columns:
                        mask &= (self.adata.obs[label_key].astype(str).values == str(init_cell_type))
                    eligible = int(np.sum(mask))
                    if eligible > 0:
                        est_n_cells = min(est_n_cells, eligible) if n_cells is not None else eligible
                except Exception:
                    pass

                est_gene_bytes = int(n_steps_for_estimate + 1) * max(1, int(est_n_cells)) * int(self.adata.n_vars) * 4
                gene_projection_limit = int(getattr(self, "max_dense_projection_bytes", 768 * 1024 * 1024) * 2)
                resolved_output_space = "latent" if est_gene_bytes > gene_projection_limit else "both"
                lines.append(
                    f"Auto export policy: est_gene_mem={est_gene_bytes/(1024**3):.2f}GB -> {resolved_output_space}"
                )
            else:
                lines.append(f"Effective output space: {resolved_output_space}")
            
            # Handle gene perturbation
            perturbed_latent = None
            if perturbed_genes and perturbed_z_score != 0:
                try:
                    perturbed_latent = perturb_gene_expression(
                        self.adata, perturbed_genes, perturbed_z_score
                    )
                    lines.append("Gene perturbation applied")
                except Exception as e:
                    lines.append(f"Gene perturbation failed: {e}")
            
            lines.append("Generating trajectories...")
            
            result = generate_trajectory_dataset(
                adata=self.adata,
                model=model,
                n_cells=n_cells,
                n_steps=n_steps,
                init_time=init_time_point,
                end_time=end_time_point,
                init_cell_type=init_cell_type,
                cell_type_key=label_key,
                output_space=resolved_output_space,
                perturbed_latent=perturbed_latent,
                device=self.device,
                save_dir=str(target_dir),
            )
            rollout_meta = dict((result or {}).get("metadata") or {})
            lines.extend(
                [
                    f"rollout config: {rollout_meta.get('rollout_config_source', 'unknown')}",
                    f"resolved method/sigma/steps: {rollout_meta.get('method')} / {rollout_meta.get('sigma')} / {rollout_meta.get('n_steps')}",
                    f"source(method/sigma/steps): {rollout_meta.get('rollout_method_source')} / {rollout_meta.get('rollout_sigma_source')} / {rollout_meta.get('rollout_n_steps_source')}",
                ]
            )

            # Keep perturb/control output schema consistent:
            # if trajectories_gene is unexpectedly missing, try to backfill it.
            if (
                isinstance(result, dict)
                and result.get("metadata", {}).get("output_space") in {"both", "gene"}
                and ("trajectories_gene" not in result or result.get("trajectories_gene") is None)
            ):
                traj_latent = result.get("trajectories_latent")
                if isinstance(traj_latent, np.ndarray) and traj_latent.ndim == 3:
                    n_t, n_c, n_d = traj_latent.shape
                    W = _get_pca_loadings(self.adata, n_d)
                    if W is None and self.adata.n_vars == n_d:
                        W = np.eye(n_d, dtype=np.float32)

                    if W is not None:
                        est_bytes = int(n_t) * int(n_c) * int(W.shape[0]) * 4  # float32
                        gene_projection_limit = int(getattr(self, "max_dense_projection_bytes", 768 * 1024 * 1024) * 2)
                        if est_bytes <= gene_projection_limit:
                            traj_flat = np.asarray(traj_latent.reshape(-1, n_d), dtype=np.float32)
                            traj_gene_flat = traj_flat @ np.asarray(W.T, dtype=np.float32)
                            traj_gene = traj_gene_flat.reshape(n_t, n_c, -1)

                            result["trajectories_gene"] = traj_gene
                            result["gene_names"] = list(self.adata.var_names)
                            result["metadata"]["shape_gene"] = traj_gene.shape
                            result["metadata"]["n_genes"] = int(traj_gene.shape[2])
                            lines.append(" 已自动补全 trajectories_gene（与 control 输出结构统一）")

                            # Rewrite saved artifacts to keep files in sync with in-memory result.
                            npz_path = target_dir / "trajectory_dataset.npz"
                            if npz_path.exists():
                                with np.load(npz_path, allow_pickle=True) as old_npz:
                                    save_dict = {k: old_npz[k] for k in old_npz.files}
                                save_dict["trajectories_gene"] = traj_gene
                                np.savez(npz_path, **save_dict)

                            gene_names_path = target_dir / "gene_names.txt"
                            with gene_names_path.open("w", encoding="utf-8") as gf:
                                for g in result["gene_names"]:
                                    gf.write(f"{g}\n")

                            metadata_path = target_dir / "trajectory_metadata.json"
                            if metadata_path.exists():
                                try:
                                    with metadata_path.open("r", encoding="utf-8") as mf:
                                        metadata_obj = json.load(mf)
                                except Exception:
                                    metadata_obj = {}
                                metadata_obj.update(
                                    {
                                        "shape_gene": list(traj_gene.shape),
                                        "n_genes": int(traj_gene.shape[2]),
                                    }
                                )
                                with metadata_path.open("w", encoding="utf-8") as mf:
                                    json.dump(metadata_obj, mf, indent=2, ensure_ascii=False)
                        else:
                            lines.append(
                                f" 轨迹基因投影估算内存约 {est_bytes / (1024**3):.2f} GB，已跳过 trajectories_gene 补全"
                            )
                    else:
                        lines.append(" 未找到可用的 PCA/loadings，无法补全 trajectories_gene")
            
            lines.extend([
                "",
                " 轨迹生成完成!",
                "",
                " 保存文件:",
                f"  • trajectory_dataset.npz",
                f"  • trajectory_metadata.json",
                "",
                " 数据形状:",
                f"  • 潜在空间: {result['metadata']['shape_latent']}",
            ])
            
            if 'shape_gene' in result['metadata']:
                lines.append("  • gene_names.txt")
                lines.append(f"  • 基因空间: {result['metadata']['shape_gene']}")
                lines.append(f"  • 基因数量: {result['metadata']['n_genes']}")
            else:
                lines.append(" 当前仅导出 latent 轨迹；如需少量基因动力学，请调用 extract_trajectory_gene_dynamics")
            
            lines.extend([
                "",
                "=" * 60,
                "接下来，你可以写代码来深入分析产生的轨迹，其中可以追踪单个细胞状态随时间的连续变化",
                " 数据使用说明",
                "=" * 60,
                "",
                "```python",
                "# 加载轨迹数据",
                "from CytoBridge.tl.perturbation import load_trajectory_dataset",
                f"data = load_trajectory_dataset('{target_dir}')",
                "",
                "# 潜在空间轨迹: (n_steps+1, n_cells, latent_dim)",
                "traj_latent = data['trajectories_latent']",
                "# 权重: (n_steps+1, n_cells)",
                "weights = data['weights']",
                "```",
            ])
            if "shape_gene" in result["metadata"]:
                lines.extend([
                    "",
                    "```python",
                    "# 若已导出基因空间轨迹",
                    "traj_gene = data['trajectories_gene']",
                    "gene_names = data['gene_names']",
                    "gene_idx = gene_names.index('YourGene')",
                    "gene_traj = traj_gene[:, :, gene_idx]  # (n_steps+1, n_cells)",
                    "```",
                ])
            else:
                lines.extend([
                    "",
                    "```python",
                    "# 仅导出 latent 时，按需投影少量基因（节省内存）",
                    "# 使用工具: extract_trajectory_gene_dynamics(trajectory_dir=..., genes=[...])",
                    "```",
                ])
            lines.extend([
                "",
                " 细胞类型注释:",
                "```python",
                "# 使用已训练的分类器注释 (需要先运行 train_cell_classifier工具)",
                "from CytoBridge.tl.perturbation import load_mlp_classifier",
                "import torch",
                "model, le = load_mlp_classifier('classifiers/cell_classifier.pt')",
                "final_latent = data['trajectories_latent'][-1]",
                "with torch.no_grad():",
                "    logits = model(torch.tensor(final_latent))",
                "    labels = le.inverse_transform(logits.argmax(1).cpu().numpy())",
                "```",
                "=" * 60,
            ])
            
            # Store result metadata
            self.analysis_results["trajectory_dataset"] = {
                "save_dir": str(target_dir),
                "shape_latent": result['metadata']['shape_latent'],
                "shape_gene": result['metadata'].get('shape_gene'),
                "n_cells": n_cells,
                "n_steps": n_steps,
                "output_space": result["metadata"].get("output_space", resolved_output_space),
            }
            
        except ImportError as e:
            lines.append(f"\n 导入失败: {e}")
        except Exception as e:
            import traceback
            lines.append(f"\n 生成失败: {e}")
            logger.error(traceback.format_exc())
        
        return truncate_output("\n".join(lines))

    @log_reason
    def extract_trajectory_gene_dynamics(
        self,
        genes: Any,
        trajectory_dir: Optional[str] = None,
        save_filename: Optional[str] = None,
        include_cell_level: bool = False,
        reason: str = "",
    ) -> str:
        """Project selected genes from latent trajectories for memory-efficient dynamics analysis.

        This tool is designed for large runs where `generate_trajectory_dataset`
        exports latent-only trajectories. It projects only requested genes back to
        gene space, avoiding full dense `(time, cells, genes)` allocation.

        Args:
            genes: Gene list or comma-separated string.
            trajectory_dir: Trajectory folder containing `trajectory_dataset.npz`.
                If None, use latest `trajectory_dataset` result path.
            save_filename: Optional output filename prefix.
            include_cell_level: If True, save full `(time, cell, selected_gene)` tensor.
            reason: Detailed reasoning and intent for calling this tool.

        Returns:
            A report with projected gene dynamics artifacts.
        """
        guarded = self._invoke_tool_method_in_subprocess(
            method="extract_trajectory_gene_dynamics",
            kwargs={
                "genes": genes,
                "trajectory_dir": trajectory_dir,
                "save_filename": save_filename,
                "include_cell_level": bool(include_cell_level),
                "reason": reason,
            },
            timeout=max(900, self.subprocess_guard_timeout),
        )
        if guarded is not None:
            return self._consume_guarded_tool_result(guarded, "trajectory gene dynamics extraction")

        import re
        from CytoBridge.tl.perturbation import load_trajectory_dataset

        # Parse genes
        if isinstance(genes, str):
            try:
                parsed = ast.literal_eval(genes)
                genes = parsed if isinstance(parsed, list) else [str(parsed)]
            except Exception:
                genes = [g.strip() for g in genes.split(",") if g.strip()]
        if not isinstance(genes, list):
            genes = [str(genes)]
        genes = [str(g).strip() for g in genes if str(g).strip()]
        if len(genes) == 0:
            return " genes 不能为空。"

        # Resolve trajectory directory
        if trajectory_dir is None:
            trajectory_dir = (self.analysis_results.get("trajectory_dataset") or {}).get("save_dir")
        if trajectory_dir is None:
            trajectory_dir = str(self.output_dir / "trajectories")

        traj_dir = Path(trajectory_dir)
        if traj_dir.is_dir() and not (traj_dir / "trajectory_dataset.npz").exists():
            # Try latest subdirectory with trajectory dataset.
            cands = sorted(
                [p for p in traj_dir.glob("**/trajectory_dataset.npz") if p.is_file()],
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if cands:
                traj_dir = cands[0].parent

        if not (traj_dir / "trajectory_dataset.npz").exists():
            return f" 未找到 trajectory_dataset.npz: {traj_dir}"

        data = load_trajectory_dataset(str(traj_dir))
        traj_latent = np.asarray(data.get("trajectories_latent"))
        if traj_latent.ndim != 3:
            return " trajectories_latent 维度异常，预期 (time, cells, latent_dim)。"

        time_points = np.asarray(data.get("time_points", np.arange(traj_latent.shape[0], dtype=float)))
        latent_dim = int(traj_latent.shape[2])

        # Load PCA projection for selected genes only.
        loadings = _get_pca_loadings(self.adata, latent_dim)
        if loadings is None:
            if self.adata.n_vars == latent_dim:
                loadings = np.eye(latent_dim, dtype=np.float32)
            else:
                return " 缺少可用的 PCA loadings，无法从 latent 投影到基因空间。"

        gene_names_all = [str(x) for x in self.adata.var_names]
        idx_map = {g: i for i, g in enumerate(gene_names_all)}
        valid_genes = [g for g in genes if g in idx_map]
        missing_genes = [g for g in genes if g not in idx_map]
        if len(valid_genes) == 0:
            return f" 指定基因均未命中 var_names。missing={missing_genes[:20]}"

        gene_idx = np.asarray([idx_map[g] for g in valid_genes], dtype=int)
        W_sel = np.asarray(loadings[gene_idx, :], dtype=np.float32)  # (G, D)

        # Project only selected genes: (T, C, D) x (G, D)^T -> (T, C, G)
        traj_gene_sel = np.einsum(
            "tcd,gd->tcg",
            np.asarray(traj_latent, dtype=np.float32),
            W_sel,
            optimize=True,
        ).astype(np.float32)
        mean_tc = np.mean(traj_gene_sel, axis=1)  # (T, G)
        std_tc = np.std(traj_gene_sel, axis=1)    # (T, G)

        outdir = _ensure_dir(self.output_dir / "trajectory_gene_dynamics")
        prefix = re.sub(r"[^a-zA-Z0-9_\\-]", "_", save_filename) if save_filename else "selected_genes"
        artifacts: Dict[str, str] = {}

        # Save summary CSV (long format)
        rows = []
        for t_i, t_val in enumerate(time_points):
            for g_i, g_name in enumerate(valid_genes):
                rows.append(
                    {
                        "time_index": int(t_i),
                        "time_point": float(t_val),
                        "gene": g_name,
                        "mean_expression_proxy": float(mean_tc[t_i, g_i]),
                        "std_expression_proxy": float(std_tc[t_i, g_i]),
                    }
                )
        summary_csv = outdir / f"{prefix}_gene_dynamics_summary.csv"
        pd.DataFrame(rows).to_csv(summary_csv, index=False)
        artifacts["summary_csv"] = str(summary_csv)

        summary_npz = outdir / f"{prefix}_gene_dynamics_summary.npz"
        np.savez(
            summary_npz,
            time_points=np.asarray(time_points, dtype=np.float32),
            genes=np.asarray(valid_genes, dtype=object),
            mean_expression_proxy=np.asarray(mean_tc, dtype=np.float32),
            std_expression_proxy=np.asarray(std_tc, dtype=np.float32),
        )
        artifacts["summary_npz"] = str(summary_npz)

        if include_cell_level:
            cell_npz = outdir / f"{prefix}_gene_dynamics_cell_level.npz"
            np.savez(
                cell_npz,
                time_points=np.asarray(time_points, dtype=np.float32),
                genes=np.asarray(valid_genes, dtype=object),
                trajectories_gene_selected=np.asarray(traj_gene_sel, dtype=np.float32),
            )
            artifacts["cell_level_npz"] = str(cell_npz)

        # Plot means over time
        fig, ax = plt.subplots(figsize=(9, 5))
        for g_i, g_name in enumerate(valid_genes):
            y = mean_tc[:, g_i]
            yerr = std_tc[:, g_i]
            ax.plot(time_points, y, label=g_name, linewidth=1.8)
            ax.fill_between(time_points, y - yerr, y + yerr, alpha=0.15)
        ax.set_xlabel("Time")
        ax.set_ylabel("Projected expression proxy")
        ax.set_title("Selected Gene Dynamics from Latent Trajectories")
        if len(valid_genes) <= 15:
            ax.legend(loc="best", frameon=False)
        fig.tight_layout()
        fig_path = self.figures_dir / f"{prefix}_trajectory_gene_dynamics.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        artifacts["dynamics_plot"] = str(fig_path)

        self.analysis_results["trajectory_gene_dynamics"] = {
            "trajectory_dir": str(traj_dir),
            "genes": valid_genes,
            "missing_genes": missing_genes,
            "artifacts": artifacts,
        }

        lines = [
            "=" * 60,
            " 轨迹基因动力学提取完成",
            "=" * 60,
            f"轨迹目录: {traj_dir}",
            f"有效基因数: {len(valid_genes)}",
            f"缺失基因数: {len(missing_genes)}",
            f"时间步: {traj_gene_sel.shape[0]} | 细胞数: {traj_gene_sel.shape[1]}",
            "",
            " [Generated Artifacts]:",
        ]
        for _, p in artifacts.items():
            lines.append(f" - {p}")
        if missing_genes:
            lines.append(f"\n 未命中基因: {missing_genes[:20]}")
        lines.append("=" * 60)
        return truncate_output("\n".join(lines))

    # =========================================================================
    # Cell Lineage Sankey Diagram
    # =========================================================================
    @log_reason
    def plot_cell_lineage_sankey(
        self,
        trajectory_dir: Optional[str] = None,
        classifier_path: Optional[str] = None,
        n_time_points: int = 5,
        color_palette: Optional[List[str]] = None,
        title: str = "细胞谱系演变桑基图",
        save_format: str = "both",
        reason: str = "",
    ) -> str:
        """Generate a Sankey diagram showing cell type transitions across time points.

        This tool visualizes how cells transition between different cell types over time,
        based on trajectory data generated by `generate_trajectory_dataset` and a
        classifier trained by `train_cell_classifier`.

        **Prerequisites**:
        - First run `generate_trajectory_dataset` to generate cell trajectories
        - Then run `train_cell_classifier` to train a cell type classifier

        Args:
            trajectory_dir: Path to trajectory dataset directory. If None, uses the
                           most recent trajectory from `generate_trajectory_dataset`.
            classifier_path: Path to trained classifier (.pt file). If None, uses the
                            most recent classifier from `train_cell_classifier`.
            n_time_points: Number of time points to show in the diagram (default: 5).
                          These are evenly sampled from the trajectory time steps.
            color_palette: Optional list of hex colors for cell types. If None, uses
                          a default vibrant palette.
            title: Title for the Sankey diagram (default: "细胞谱系演变桑基图").
            save_format: Output format. Supported: 'html', 'png', 'svg', 'pdf', 'both' (html+png+svg), 'all'.
            reason: Detailed reasoning and intent for calling this tool.

        Returns:
            A formatted report with the generated Sankey diagram paths and statistics.
        """
        guarded = self._invoke_tool_method_in_subprocess(
            method="plot_cell_lineage_sankey",
            kwargs={
                "trajectory_dir": trajectory_dir,
                "classifier_path": classifier_path,
                "n_time_points": int(n_time_points),
                "color_palette": color_palette,
                "title": title,
                "save_format": save_format,
                "reason": reason,
            },
            timeout=max(900, self.subprocess_guard_timeout),
        )
        if guarded is not None:
            return self._consume_guarded_tool_result(guarded, "sankey plotting")

        from pathlib import Path
        
        outdir = self.figures_dir
        
        lines = [
            "=" * 60,
            " 细胞谱系桑基图生成",
            "=" * 60,
            "",
        ]
        
        # --- Step 1: Resolve trajectory directory ---
        if trajectory_dir is None:
            # Try to get from analysis_results
            traj_info = self.analysis_results.get("trajectory_dataset")
            if traj_info and "save_dir" in traj_info:
                trajectory_dir = traj_info["save_dir"]
            else:
                # Default path
                trajectory_dir = str(self.output_dir / "trajectories")
        
        trajectory_path = Path(trajectory_dir)
        if not trajectory_path.exists():
            return " 未找到轨迹数据。请先运行 generate_trajectory_dataset 工具。"
        
        lines.append(f" 轨迹数据目录: {trajectory_dir}")
        
        # --- Step 2: Resolve classifier path ---
        if classifier_path is None:
            # Try to get from analysis_results
            clf_info = self.analysis_results.get("cell_classifier")
            if clf_info and "model_path" in clf_info:
                classifier_path = clf_info["model_path"]
            else:
                # Default path
                classifier_path = str(self.output_dir / "classifiers" / "cell_classifier.pt")
        
        if not Path(classifier_path).exists():
            return " 未找到分类器模型。请先运行 train_cell_classifier 工具。"
        
        lines.append(f" 分类器路径: {classifier_path}")
        lines.append("")
        
        try:
            import torch
            import numpy as np
            import pandas as pd
            
            # --- Step 3: Load trajectory data ---
            from CytoBridge.tl.perturbation import load_trajectory_dataset, load_mlp_classifier
            
            lines.append(" 加载轨迹数据...")
            traj_data = load_trajectory_dataset(trajectory_dir)
            trajectories_latent = traj_data['trajectories_latent']  # (n_steps+1, n_cells, latent_dim)
            
            n_total_steps = trajectories_latent.shape[0]
            n_cells = trajectories_latent.shape[1]
            latent_dim = trajectories_latent.shape[2]
            
            lines.append(f"  • 总时间步数: {n_total_steps}")
            lines.append(f"  • 细胞数量: {n_cells}")
            lines.append(f"  • 潜在维度: {latent_dim}")
            
            # --- Step 4: Load classifier ---
            lines.append(" 加载分类器...")
            model, label_encoder = load_mlp_classifier(classifier_path)
            model.eval()
            classes = label_encoder.classes_
            n_classes = len(classes)
            lines.append(f"  • 类别数: {n_classes}")
            lines.append(f"  • 类别: {', '.join(classes[:5])}{'...' if n_classes > 5 else ''}")
            
            # --- Step 5: Select time points to plot ---
            # Evenly sample n_time_points from the trajectory
            if n_time_points >= n_total_steps:
                time_indices = list(range(n_total_steps))
            else:
                time_indices = np.linspace(0, n_total_steps - 1, n_time_points, dtype=int).tolist()
            
            lines.append(f"  • 绘图时间点索引: {time_indices}")
            lines.append("")
            
            # --- Step 6: Predict cell types at each time point ---
            lines.append(" 预测细胞类型...")
            predicted_labels_list = []
            
            device = next(model.parameters()).device
            
            for t_idx in time_indices:
                latent_t = trajectories_latent[t_idx]  # (n_cells, latent_dim)
                with torch.no_grad():
                    latent_tensor = torch.tensor(latent_t, dtype=torch.float32).to(device)
                    logits = model(latent_tensor)
                    pred_indices = logits.argmax(dim=1).cpu().numpy()
                    pred_labels = label_encoder.inverse_transform(pred_indices)
                predicted_labels_list.append(pred_labels)
            
            lines.append(f"   预测完成 ({len(predicted_labels_list)} 个时间点)")
            
            # --- Step 7: Build Sankey data ---
            lines.append(" 构建桑基图数据...")
            
            links = []
            num_timepoints = len(predicted_labels_list)
            
            for t in range(num_timepoints - 1):
                df_plot = pd.DataFrame({
                    'source': predicted_labels_list[t],
                    'target': predicted_labels_list[t + 1]
                })
                counts = df_plot.groupby(['source', 'target']).size().reset_index(name='value')
                counts['source'] = counts['source'].astype(str) + f'_T{t + 1}'
                counts['target'] = counts['target'].astype(str) + f'_T{t + 2}'
                links.append(counts)
            
            all_links_df = pd.concat(links, axis=0)
            
            # --- Diagnostic: Verify flow conservation ---
            lines.append(" 流量验证:")
            lines.append(f"  • 每个时间点的细胞数: {n_cells}")
            for t in range(num_timepoints - 1):
                flow_out = links[t]['value'].sum()
                lines.append(f"  • T{t+1}→T{t+2} 总流量: {flow_out}")
            
            # --- Diagnostic: Show label distribution at each time point ---
            lines.append("")
            lines.append(" 各时间点标签分布:")
            for t_idx_pos, labels in enumerate(predicted_labels_list):
                # Convert to string array to handle mixed types
                labels_str = np.array([str(l) if l is not None and not (isinstance(l, float) and np.isnan(l)) else 'nan' for l in labels])
                unique, counts_arr = np.unique(labels_str, return_counts=True)
                nan_count = np.sum(labels_str == 'nan')
                valid_count = len(labels_str) - nan_count
                lines.append(f"  T{t_idx_pos+1}: 有效={valid_count}, nan={nan_count}, 类型数={len(unique)}")
                if len(unique) <= 5:
                    for u, c in zip(unique, counts_arr):
                        lines.append(f"      {u}: {c}")
            
            # Build node mapping
            all_nodes = pd.unique(all_links_df[['source', 'target']].values.ravel('K'))
            node_indices = {node: i for i, node in enumerate(all_nodes)}
            
            all_links_df['source_idx'] = all_links_df['source'].map(node_indices)
            all_links_df['target_idx'] = all_links_df['target'].map(node_indices)
            
            # --- Step 8: Prepare colors ---
            base_types = sorted(list(set([node.split('_')[0] for node in all_nodes])))
            
            if color_palette is None:
                color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                                '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
            
            color_map = {base_type: color_palette[i % len(color_palette)] 
                        for i, base_type in enumerate(base_types)}
            node_colors = [color_map[node.split('_')[0]] for node in all_nodes]
            
            # Helper function: hex to rgba
            def hex_to_rgba(h, alpha=0.4):
                h = h.lstrip('#')
                return f'rgba({",".join(str(int(h[i:i+2], 16)) for i in (0, 2, 4))}, {alpha})'
            
            # Link colors (inherit from source node)
            link_colors = []
            for src_idx in all_links_df['source_idx']:
                source_node_label = all_nodes[src_idx]
                base_type = source_node_label.split('_')[0]
                hex_color = color_map[base_type]
                link_colors.append(hex_to_rgba(hex_color, alpha=0.5))
            
            # --- Step 9: Generate Sankey diagram with Plotly ---
            lines.append(" 生成桑基图...")
            
            try:
                import plotly.graph_objects as go
                
                # Sort nodes by time point, then by cell type for consistent layout
                # This ensures left-to-right time flow
                def node_sort_key(node):
                    parts = node.rsplit('_T', 1)
                    cell_type = parts[0]
                    time_num = int(parts[1]) if len(parts) > 1 else 0
                    return (time_num, cell_type)
                
                sorted_nodes = sorted(all_nodes, key=node_sort_key)
                # Rebuild node indices with sorted order
                node_indices = {node: i for i, node in enumerate(sorted_nodes)}
                all_links_df['source_idx'] = all_links_df['source'].map(node_indices)
                all_links_df['target_idx'] = all_links_df['target'].map(node_indices)
                
                # Rebuild colors with sorted order
                node_colors = [color_map[node.split('_')[0]] for node in sorted_nodes]
                
                # Rebuild link colors with new indices
                link_colors = []
                for _, row in all_links_df.iterrows():
                    source_node_label = sorted_nodes[row['source_idx']]
                    base_type = source_node_label.split('_')[0]
                    hex_color = color_map[base_type]
                    link_colors.append(hex_to_rgba(hex_color, alpha=0.5))
                
                # Calculate x positions (time-based: left to right)
                # and y positions (cell type-based: vertical arrangement within each time)
                x_positions = []
                y_positions = []
                
                # Group nodes by time point
                time_to_nodes = {}
                for node in sorted_nodes:
                    t_num = int(node.rsplit('_T', 1)[1])
                    if t_num not in time_to_nodes:
                        time_to_nodes[t_num] = []
                    time_to_nodes[t_num].append(node)
                
                for node in sorted_nodes:
                    t_num = int(node.rsplit('_T', 1)[1])
                    # x position: based on time point (T1=0.1, T2, ..., Tn=0.9)
                    # Use 0.1 to 0.9 range to ensure nodes are visible and not clipped
                    if num_timepoints == 1:
                        x_pos = 0.5
                    else:
                        x_pos = 0.1 + (t_num - 1) / (num_timepoints - 1) * 0.8
                    x_positions.append(x_pos)
                    
                    # y position: evenly distribute cell types within each time point
                    nodes_at_time = time_to_nodes[t_num]
                    idx_in_time = nodes_at_time.index(node)
                    n_nodes_at_time = len(nodes_at_time)
                    # Distribute from 0.1 to 0.9 to avoid edge clipping
                    if n_nodes_at_time == 1:
                        y_pos = 0.5
                    else:
                        y_pos = 0.1 + (idx_in_time / (n_nodes_at_time - 1)) * 0.8
                    y_positions.append(y_pos)
                
                fig = go.Figure(data=[go.Sankey(
                    arrangement='snap',  # Snap to specified positions
                    node=dict(
                        pad=20,
                        thickness=25,
                        line=dict(color="black", width=0.5),
                        label=sorted_nodes,
                        color=node_colors,
                        x=x_positions,
                        y=y_positions,
                    ),
                    link=dict(
                        source=all_links_df['source_idx'],
                        target=all_links_df['target_idx'],
                        value=all_links_df['value'],
                        color=link_colors
                    )
                )])
                
                fig.update_layout(
                    title_text=title,
                    title_font=dict(size=24),
                    font_family="Helvetica, Arial, sans-serif",
                    font_size=14,
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    width=1600,
                    height=max(900, 600 + num_timepoints * 80),
                    margin=dict(l=20, r=20, t=80, b=20),
                )
                
                artifacts = {}
                static_export_errors: Dict[str, str] = {}

                fmt_raw = str(save_format or "both").strip().lower()
                valid_formats = {"html", "png", "svg", "pdf"}
                if fmt_raw == "both":
                    requested_formats = {"html", "png", "svg"}
                elif fmt_raw == "all":
                    requested_formats = set(valid_formats)
                else:
                    requested_formats = {
                        f.strip() for f in re.split(r"[+,]", fmt_raw) if f.strip()
                    } & valid_formats
                    if not requested_formats:
                        requested_formats = {"html", "png", "svg"}
                        lines.append(
                            "   未识别 save_format，已回退为 publication 默认: html + png + svg"
                        )

                artifacts, static_export_errors = save_plotly_bundle(
                    fig,
                    base_name="cell_lineage_sankey",
                    output_dir=outdir,
                    preset=self.figure_quality_preset,
                    formats=sorted(requested_formats),
                    width=2400,
                    height=max(1200, 700 + num_timepoints * 90),
                    scale=2.0,
                )
                if static_export_errors:
                    self._configure_plotly_chrome()
                    retry_formats = [fmt for fmt in static_export_errors.keys() if fmt in {"png", "svg", "pdf"}]
                    if retry_formats:
                        retry_artifacts, retry_errors = save_plotly_bundle(
                            fig,
                            base_name="cell_lineage_sankey",
                            output_dir=outdir,
                            preset=self.figure_quality_preset,
                            formats=retry_formats,
                            width=2400,
                            height=max(1200, 700 + num_timepoints * 90),
                            scale=2.0,
                        )
                        artifacts.update(retry_artifacts)
                        static_export_errors = retry_errors

                for fmt, path_str in sorted(artifacts.items()):
                    lines.append(f"   {fmt.upper()} 保存: {Path(path_str).name}")
                if static_export_errors:
                    lines.append(
                        "   部分静态导出失败。请安装/配置 Chrome 后重试："
                        " `plotly_get_chrome -y --path ~/.local/share/plotly_chrome`，"
                        " 并设置 `BROWSER_PATH` 指向 chrome 可执行文件。"
                    )
                    for ext, err in static_export_errors.items():
                        lines.append(f"  • {ext.upper()} 失败: {err}")
                
            except ImportError:
                lines.append(" Plotly 未安装，使用 Matplotlib 替代...")
                # Fallback to matplotlib-based visualization
                import matplotlib.pyplot as plt
                apply_matplotlib_style(self.figure_quality_preset)
                
                # Create a simplified transition matrix heatmap instead
                fig, axes = plt.subplots(1, num_timepoints - 1, figsize=(4 * (num_timepoints - 1), 6))
                if num_timepoints == 2:
                    axes = [axes]
                
                for t in range(num_timepoints - 1):
                    ax = axes[t]
                    trans_df = links[t].pivot(index='source', columns='target', values='value').fillna(0)
                    # Clean labels
                    trans_df.index = [s.replace(f'_T{t+1}', '') for s in trans_df.index]
                    trans_df.columns = [s.replace(f'_T{t+2}', '') for s in trans_df.columns]
                    
                    im = ax.imshow(trans_df.values, cmap='Blues', aspect='auto')
                    ax.set_xticks(range(len(trans_df.columns)))
                    ax.set_xticklabels(trans_df.columns, rotation=45, ha='right')
                    ax.set_yticks(range(len(trans_df.index)))
                    ax.set_yticklabels(trans_df.index)
                    ax.set_xlabel(f'T{t+2}')
                    ax.set_ylabel(f'T{t+1}')
                    ax.set_title(f'T{t+1} → T{t+2}')
                    plt.colorbar(im, ax=ax, shrink=0.6)
                
                plt.suptitle(title)
                plt.tight_layout()
                
                fmt_raw = str(save_format or "both").strip().lower()
                valid_formats = {"png", "svg", "pdf"}
                if fmt_raw in {"both", "all", "html"}:
                    requested_static = {"png", "svg"}
                else:
                    requested_static = {
                        f.strip() for f in re.split(r"[+,]", fmt_raw) if f.strip()
                    } & valid_formats
                    if not requested_static:
                        requested_static = {"png", "svg"}
                artifacts = save_figure_bundle(
                    fig,
                    base_name="cell_lineage_transitions",
                    output_dir=outdir,
                    preset=self.figure_quality_preset,
                    formats=sorted(requested_static),
                )
                for fmt, path_str in sorted(artifacts.items()):
                    lines.append(f"   转移热图保存 ({fmt.upper()}): {Path(path_str).name}")
                plt.close(fig)
            
            # --- Step 10: Summary statistics ---
            lines.extend([
                "",
                " 细胞类型转移统计:",
            ])
            
            # Count total transitions per type
            for t in range(num_timepoints - 1):
                source_counts = pd.Series(predicted_labels_list[t]).value_counts()
                lines.append(f"  T{t+1}: {dict(source_counts)}")
            
            # Final time point
            final_counts = pd.Series(predicted_labels_list[-1]).value_counts()
            lines.append(f"  T{num_timepoints}: {dict(final_counts)}")
            
            lines.extend([
                "",
                " 生成的文件:",
                *[f"  • {Path(v).name}" for v in artifacts.values()],
                "",
                " 提示:",
                "  - HTML 文件可以在浏览器中打开，支持交互式查看",
                "  - 节点颜色表示细胞类型，连接透明度表示转移数量",
                "  - 报告插图请优先使用 PNG/SVG（publication-ready）",
                "=" * 60,
            ])
            
            # Store result
            self.analysis_results["sankey_plot"] = {
                "artifacts": artifacts,
                "n_time_points": num_timepoints,
                "cell_types": list(base_types),
                "summary": [
                    f"Cell lineage transitions across {num_timepoints} sampled time points.",
                    f"Cells={n_cells}, cell_types={len(base_types)}.",
                    f"Exported formats: {', '.join(sorted(artifacts.keys())) if artifacts else 'none'}",
                ],
            }
            
        except ImportError as e:
            lines.append(f"\n 导入失败: {e}")
            lines.append("请确保已安装: pip install plotly")
        except Exception as e:
            import traceback
            lines.append(f"\n 生成失败: {e}")
            logger.error(traceback.format_exc())
        
        return truncate_output("\n".join(lines))

    # =========================================================================
    # Gene Perturbation Analysis (ported from DeepRUOT)
    # =========================================================================
    @log_reason
    def analyze_gene_perturbation(
        self,
        genes:List[str],
        z_scores: List[float],
        n_simulations: int = 5,
        n_init_cells: int = 1000,
        n_steps: int = 100,
        init_cell_type: Optional[str] = None,
        init_time_point: Optional[float] = None,
        end_time_point: Optional[float] = None,
        label_key: Optional[str] = None,
        classifier_path: Optional[str] = None,
        reason: str = "",
    ) -> str:
        """Analyze gene perturbation effects on cell fate via SDE simulation.

        This tool simulates how perturbing specific gene expression affects
        cell fate trajectories. It uses the trained CytoBridge model to run
        SDE (stochastic differential equation) simulations and predicts
        the resulting cell type distribution.

        **Time Explanation**: 
        - Times correspond to values in `time_point_processed` column in the data.
        - The simulation runs from `init_time_point` to `end_time_point` over `n_steps`.
        - Each step represents: (end_time_point - init_time_point) / n_steps time units.
        - Default: from earliest to latest time in `time_point_processed`.

        Args:
            genes: List of gene names (or single string) to perturb. 
                   If multiple genes, they are perturbed SIMULTANEOUSLY.
            z_scores: List of z-values. 
                      - Meaning: Standard deviations from mean gene expression. 
                        z > 0 (e.g. +5) = Overexpression. z < 0 (e.g. -5) = Knockdown.
                        Magnitude indicates perturbation strength (5.0 is strong).
                        z=0.0 means no perturbation.
                      - If length matches `genes` (and >1), it defines ONE simulation with 1-to-1 mapping.
                      - Otherwise, it is treated as a list of conditions to SCAN (each applied to all genes).
                      - To scan multiple 1-to-1 configurations, pass a list of lists.
            n_simulations: Number of simulation runs per condition (default: 5).
            n_init_cells: Number of initial cells per simulation (default: 1000).
            n_steps: Number of time steps in trajectory (default: 100).
            sigma: SDE diffusion coefficient (default: 0.05).
            init_cell_type: Optional filter for initial cell type.
            init_time_point: Starting time from time_point_processed (None = earliest).
            end_time_point: Ending time from time_point_processed (None = latest).
            label_key: Column for cell type labels.
            classifier_path: Path to trained classifier (.pt file). If None, auto-detects
                             from output_dir/classifiers/cell_classifier.pt. Falls back to KNN.
            reason: Detailed reasoning and intent for calling this tool.

        Returns:
            A formatted report with perturbation analysis results including
            cell fate proportions for each z-score condition at the end time point and generated plots.
        """
        guarded = self._invoke_tool_method_in_subprocess(
            method="analyze_gene_perturbation",
            kwargs={
                "genes": genes,
                "z_scores": z_scores,
                "n_simulations": int(n_simulations),
                "n_init_cells": int(n_init_cells),
                "n_steps": int(n_steps),
                "init_cell_type": init_cell_type,
                "init_time_point": init_time_point,
                "end_time_point": end_time_point,
                "label_key": label_key,
                "classifier_path": classifier_path,
                "reason": reason,
            },
            timeout=max(1800, self.subprocess_guard_timeout),
        )
        if guarded is not None:
            return self._consume_guarded_tool_result(guarded, "gene perturbation analysis")

        from pathlib import Path
        
        outdir = _ensure_dir(self.output_dir / "perturbation")
        artifacts = {}
        
        # Robust parsing for LLM inputs
        if isinstance(genes, str):
            import ast
            try:
                genes = ast.literal_eval(genes)
            except:
                genes = [g.strip() for g in genes.split(',') if g.strip()]
        
        if isinstance(z_scores, str):
            import ast
            try:
                z_scores = ast.literal_eval(z_scores)
            except:
                # Handle simple comma-separated numbers
                z_scores = [float(z) for z in z_scores.split(',') if z.strip()]

        # Default z-scores
        if z_scores is None:
            z_scores = [-5.0, -2.0, 0.0, 2.0, 5.0]
        
        # Normalize z_scores to list (handle scalar input)
        if isinstance(z_scores, (int, float)):
            z_scores = [float(z_scores)]
        elif isinstance(z_scores, str):
            # Try to parse string representation
            import ast
            try:
                z_scores = ast.literal_eval(z_scores)
                if isinstance(z_scores, (int, float)):
                    z_scores = [float(z_scores)]
            except:
                z_scores = [float(z_scores)]
        
        # Auto-detect label key
        if label_key is None:
            label_candidates = ['cell_type', 'Cell type annotation', 'celltype', 'cluster']
            for col in label_candidates:
                if col in self.adata.obs.columns:
                    label_key = col
                    break
        
        if label_key is None or label_key not in self.adata.obs.columns:
            return " 未找到细胞类型标签列。请指定 label_key 参数。"
        
        # Load model
        try:
            from CytoBridge.utils import load_model_from_adata
            model = load_model_from_adata(self.adata)
            model.eval()
        except Exception as e:
            return f" 模型加载失败: {e}"
        
        # Validate genes
        gene_names = list(self.adata.var_names)
        valid_genes = [g for g in genes if g in gene_names]
        missing_genes = [g for g in genes if g not in gene_names]
        
        # Check if we have gene expression data
        has_gene_expression = self.adata.n_vars > 10 and len(valid_genes) > 0
        
        if not has_gene_expression:
            # Check if we can do latent space perturbation
            if 'X_latent' not in self.adata.obsm:
                return " 数据中没有基因表达信息和latent表示，无法进行扰动分析"
            logger.warning("无基因表达数据，将使用模拟的扰动进行演示")
        
        lines = [
            "=" * 60,
            " 基因扰动分析",
            "=" * 60,
            "",
            f" 目标基因: {genes}",
            f" 有效基因: {valid_genes}" if valid_genes else " 无有效基因",
        ]
        if missing_genes:
            lines.append(f" 缺失基因: {missing_genes}")
        lines.extend([
            f" 测试 z-scores: {z_scores}",
            f" 模拟次数: {n_simulations}",
            f" 初始细胞数: {n_init_cells}",
            f" 细胞类型列: {label_key}",
            ""
        ])
        
        try:
            # Import perturbation functions
            from CytoBridge.tl.perturbation import (
                perturb_gene_expression,
                simulate_perturbation_sde,
                classify_final_states,
                compute_sampling_weights,
                load_mlp_classifier,
                _resolve_downstream_rollout_defaults,
            )
            
            # Try to load trained MLP classifier
            # Use provided path or auto-detect
            if classifier_path is not None:
                clf_path = Path(classifier_path)
            else:
                clf_path = self.output_dir / "classifiers" / "cell_classifier.pt"
            
            mlp_model = None
            label_encoder = None
            classify_method = 'knn'  # default
            
            if clf_path.exists():
                try:
                    mlp_model, label_encoder = load_mlp_classifier(str(clf_path), device=self.device)
                    classify_method = 'mlp'
                    lines.append(f" 使用已训练的 MLP 分类器: {clf_path}")
                except Exception as e:
                    logger.warning(f"Failed to load MLP classifier: {e}, falling back to KNN")
                    lines.append(f" 加载 MLP 分类器失败，使用 KNN 分类")
            else:
                lines.append(f" 未找到训练好的分类器，使用 KNN 分类")
            
            # Compute sampling weights
            weights = compute_sampling_weights(
                self.adata
            )
            
            # Determine time parameters
            time_key = 'time_point_processed' if 'time_point_processed' in self.adata.obs else None
            if time_key and init_time_point is None:
                time_vals = self.adata.obs[time_key]
                # Handle categorical or string types
                if hasattr(time_vals, 'cat') or time_vals.dtype == 'object':
                    try:
                        time_vals = pd.to_numeric(time_vals, errors='coerce')
                    except:
                        time_vals = time_vals.astype(float)
                init_time_point = float(time_vals.min())
                if end_time_point is None:
                    end_time_point = float(time_vals.max())
            rollout_defaults = _resolve_downstream_rollout_defaults(
                adata=self.adata,
                model=model,
                resolved_config=None,
                n_steps=n_steps,
                init_time=float(init_time_point if init_time_point is not None else 0.0),
                end_time=float(end_time_point if end_time_point is not None else n_steps),
            )
            sigma = float(rollout_defaults["sigma"])
            
            all_results = {}
            all_proportions = {}
            unperturbed_proportions = None
            
            # Determine conditions to run
            conditions = []
            
            # Smart detection: 1-to-1 mapping vs Scanning
            is_one_to_one = False
            if isinstance(z_scores, list) and len(genes) > 1 and len(z_scores) == len(genes):
                # Check if it's not a list of lists (which would imply scanning specific configs)
                if not any(isinstance(z, list) for z in z_scores):
                    is_one_to_one = True
            
            if is_one_to_one:
                conditions = [z_scores] # Run once with this config
                lines.append(f" 检测到 1-to-1 映射: 将运行单次模拟 (Genes={genes}, Z={z_scores})")
            else:
                # Scanning mode
                conditions = z_scores
                
            for i, z in enumerate(conditions):
                lines.append(f"\n z-score = {z}:")
                
                # Perturb gene expression
                if has_gene_expression:
                    try:
                        x_perturbed = perturb_gene_expression(self.adata, valid_genes, z)
                    except Exception as e:
                        lines.append(f"   扰动失败: {e}")
                        x_perturbed = np.array(self.adata.obsm['X_latent']).astype(np.float32)
                else:
                    x_perturbed = np.array(self.adata.obsm['X_latent']).astype(np.float32)
                
                # Run simulation
                run_unperturbed = (i == 0)
                try:
                    sim_results = simulate_perturbation_sde(
                        model=model,
                        adata=self.adata,
                        x_perturbed=x_perturbed,
                        n_simulations=n_simulations,
                        n_init_cells=n_init_cells,
                        n_steps=n_steps,
                        sigma=sigma,
                        init_time=init_time_point,
                        end_time=end_time_point,
                        init_cell_type=init_cell_type,
                        time_key=time_key or 'time_point_processed',
                        cell_type_key=label_key,
                        sampling_weights=weights,
                        device=self.device,
                        run_unperturbed=run_unperturbed,
                    )
                    
                    # Store results using string key to handle list z-scores
                    z_key = str(z)
                    all_results[z_key] = sim_results
                    
                    # Classify final states using MLP or KNN
                    # Extract final positions: (n_sims, n_init, dim)
                    final_pos = sim_results['perturbed_trajectories'][:, -1, :, :]
                    # Extract final weights: (n_sims, n_init)
                    final_weights = sim_results['perturbed_weights'][:, -1, :]
                    
                    _, proportions = classify_final_states(
                        final_pos, 
                        self.adata, 
                        label_key=label_key, 
                        method=classify_method,
                        mlp_model=mlp_model,
                        label_encoder=label_encoder,
                        weights=final_weights
                    )
                    all_proportions[z_key] = proportions
                    
                    lines.append(f"   模拟完成 ({n_simulations} runs)")
                    for cell_type, prop in sorted(proportions.items(), key=lambda x: -x[1]):
                        lines.append(f"    • {cell_type}: {prop*100:.1f}%")
                    
                    # Get unperturbed proportions
                    if run_unperturbed and 'unperturbed_trajectories' in sim_results:
                        final_pos_unp = sim_results['unperturbed_trajectories'][:, -1, :, :]
                        _, unp_props = classify_final_states(
                            final_pos_unp, self.adata,
                            label_key=label_key,
                            method=classify_method,
                            mlp_model=mlp_model,
                            label_encoder=label_encoder,
                        )
                        unperturbed_proportions = unp_props
                        
                except Exception as e:
                    lines.append(f"   模拟失败: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    continue
            
            # Generate comparison plot
            if all_proportions:
                try:
                    import matplotlib.pyplot as plt
                    import pandas as pd
                    apply_matplotlib_style(self.figure_quality_preset)
                    
                    # Prepare data for plotting
                    plot_data = []
                    for z, props in all_proportions.items():
                        for cell_type, prop in props.items():
                            plot_data.append({
                                'z_score': z,
                                'Cell Type': cell_type,
                                'Proportion': prop
                            })
                    
                    if unperturbed_proportions:
                        for cell_type, prop in unperturbed_proportions.items():
                            plot_data.append({
                                'z_score': 'Ctrl',
                                'Cell Type': cell_type,
                                'Proportion': prop
                            })
                    
                    df = pd.DataFrame(plot_data)
                    
                    # Create grouped bar chart
                    cell_types = df['Cell Type'].unique()
                    n_types = len(cell_types)
                    
                    if n_types <= 6:
                        fig, axes = plt.subplots(1, n_types, figsize=(4*n_types, 5))
                        if n_types == 1:
                            axes = [axes]
                        
                        for ax, ct in zip(axes, cell_types):
                            ct_data = df[df['Cell Type'] == ct]
                            x_vals = ct_data['z_score'].astype(str)
                            y_vals = ct_data['Proportion']
                            ax.bar(x_vals, y_vals, color='steelblue', alpha=0.7)
                            ax.set_xlabel('z-score')
                            ax.set_ylabel('Proportion')
                            ax.set_title(ct)
                            ax.set_ylim(0, 1)
                        
                        plt.tight_layout()
                        base_name = f"{'_'.join(valid_genes) if valid_genes else 'perturbation'}_fate_ratios"
                        bundle = save_figure_bundle(
                            fig,
                            base_name=base_name,
                            output_dir=self.figures_dir,
                            preset=self.figure_quality_preset,
                        )
                        plt.close(fig)
                        artifacts.update({f"fate_ratios_{k}": v for k, v in bundle.items()})
                        lines.append(f"\n 图表已保存: {', '.join(Path(v).name for v in bundle.values())}")
                    
                except Exception as e:
                    logger.warning(f"绘图失败: {e}")
            
            # Save results
            results_data = {
                'genes': genes,
                'valid_genes': valid_genes,
                'z_scores': z_scores,
                'proportions': {str(k): v for k, v in all_proportions.items()},
                'unperturbed_proportions': unperturbed_proportions,
            }
            
            results_path = outdir / "perturbation_results.json"
            
            def np_encoder(obj):
                if isinstance(obj, (np.integer, np.floating)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return str(obj)

            with open(results_path, 'w') as f:
                json.dump(results_data, f, indent=2, default=np_encoder)
            artifacts['results'] = str(results_path)
            
            lines.extend([
                "",
                " 解释:",
                "  - z 值含义: 偏离训练数据均值的标准差 (Standard Deviations)",
                "  - z > 0 (如 +5.0): 强力上调 (Overexpression)",
                "  - z < 0 (如 -5.0): 强力下调 (Knockdown)",
                "  - z = 0: 正常表达 (对照)",
                "",
                f" 结果保存: {results_path}",
                " [Generated Artifacts]:",
                *[f" - {Path(v).name}" for v in artifacts.values()],
                "=" * 60
            ])
            
            self.analysis_results["perturbation"] = {
                "artifacts": artifacts,
                "proportions": all_proportions,
                "genes": genes,
            }
            
        except ImportError as e:
            lines.append(f"\n 导入扰动模块失败: {e}")
            lines.append("请确保 CytoBridge.tl.perturbation 模块已正确安装")
        except Exception as e:
            import traceback
            lines.append(f"\n 分析失败: {e}")
            logger.error(traceback.format_exc())
        
        return truncate_output("\n".join(lines))

#     def plot_scatter(self, x: str, y: str, color_by: Optional[str] = None, save_name: str = "scatter", reason: str = "") -> str:

#         """Generate and save a scatter plot between two variables.

#         Args:
#             x: Variable name from ``obs`` or ``obsm`` for the x-axis.
#             y: Variable name from ``obs`` or ``obsm`` for the y-axis.
#             color_by: Optional ``obs`` column to color points.
#             save_name: Base filename for the saved figure (without extension).
#             reason: Detailed reasoning and intent for calling this tool.

#         Returns:
#             A formatted execution summary including the saved figure path.
#         """
#         code = f"""
# import matplotlib.pyplot as plt
# import numpy as np

# fig, ax = plt.subplots(figsize=(8, 6))

# # 获取数据
# def get_data(name):
#     if name in adata.obs.columns:
#         return adata.obs[name].values
#     elif name in adata.obsm.keys():
#         data = adata.obsm[name]
#         return data.flatten() if hasattr(data, 'flatten') else data
#     raise ValueError(f"'{{name}}' 不在 obs 或 obsm 中")

# x_data = get_data('{x}')
# y_data = get_data('{y}')

# color_by = '{color_by}' if '{color_by}' else None
# if color_by and color_by in adata.obs.columns:
#     scatter = ax.scatter(x_data, y_data, c=adata.obs[color_by], cmap='viridis', alpha=0.6, s=5)
#     plt.colorbar(scatter, ax=ax, label=color_by)
# else:
#     ax.scatter(x_data, y_data, alpha=0.6, s=5)

# ax.set_xlabel('{x}')
# ax.set_ylabel('{y}')
# ax.set_title('{x} vs {y}')
# plt.tight_layout()
# plt.savefig(f"{{output_dir}}/{save_name}.png", dpi=150, bbox_inches='tight')
# plt.close(fig)
# print(f"散点图已保存: {save_name}.png")
# """

#         return self.execute_python(code, reason=f"绘制散点图 ({x} vs {y})")


# Enforce strict plan tool schemas for downstream agent.
DownstreamAnalysisToolkit.create_plan.__tool_args_schema__ = _DownstreamCreatePlanInput
DownstreamAnalysisToolkit.set_plan_from_text.__tool_args_schema__ = _DownstreamSetPlanFromTextInput
DownstreamAnalysisToolkit.update_plan.__tool_args_schema__ = _DownstreamUpdatePlanInput
