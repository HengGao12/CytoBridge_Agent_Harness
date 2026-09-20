"""Planner and runtime_v2 tool implementations."""
from __future__ import annotations

import logging
import ast
import hashlib
import inspect
import json
import math
import os
import re
import shutil
import subprocess
import sys
import traceback
import yaml
from copy import deepcopy
from datetime import datetime
from contextlib import ExitStack
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from pathlib import Path
from langchain_core.messages import BaseMessage, ToolMessage

from ..schemas import AgentState, UserGoal, PlanDecision, CandidateConfig
from .training_tools import (
    get_workspace_root,
    get_cellcompass_root,
    resolve_training_target,
    materialize_training_config,
    resolve_training_data_for_target,
    execute_training_target,
    flow_matching_memory_preflight,
    validate_flow_matching_coupling_preflight,
    write_adata_h5ad_safe,
    _override_path_parts,
    validate_epoch_override_policy,
    prune_redundant_config_overrides,
)
from .training_isolation import (
    run_training_in_subprocess,
    should_isolate_training_target,
)
from .claim_metric_evaluator import (
    campaign_baseline_metric_adapter_status,
    claim_metric_spec_has_evaluator,
    load_campaign_claim_metric_evaluator,
    load_evaluation_trajectory_artifact,
    merge_evaluation_metric_hooks,
)
from .training_run_manager import (
    TrainingRunBundle,
    bundle_to_manifest_dict,
    create_training_run_bundle,
    finalize_training_outputs,
    should_save_full_trained_adata,
    snapshot_algorithm,
    write_model_artifact,
    write_planner_context,
    write_resolved_config,
    write_run_manifest,
    write_training_log,
)
from .file_tools import (
    list_path as list_path_impl,
    read_file as read_file_impl,
    read_text_file as read_text_file_impl,
    find_files as find_files_impl,
    grep_files as grep_files_impl,
)
from .bohrium_paper_search import get_bohrium_paper_search_status
from .skills_tools import SkillsTools
from .planner_file_tools import (
    DEFAULT_CAMPAIGN_BASELINE_SELECTION_POLICY,
    IMPLEMENTATION_REVIEW_POLICY_VERSION,
    PlannerFileTools,
)
from .research_idea_registry import (
    list_research_ideas as list_research_idea_catalog,
    load_research_idea_record,
)
from .workspace_policy import WorkspacePolicy, build_planner_workspace_policy
from .plan_system import (
    PlanService,
    render_plan_state,
)
from .path_resolver import pick_path_from_message
from .tool_catalog import ToolCatalog, VALID_ACTIVATION_MODES
from ..report_manager import ReportManager
from ..prompt_loader import load_prompt

logger = logging.getLogger(__name__)
SUBAGENT_NEEDS_INPUT_PREFIX = "__NEEDS_USER_INPUT__"
VALID_ALGORITHM_PROPOSAL_REVIEW_MODES = {"always_user_review", "agent_decide", "auto_approve"}
VALID_IDEA_REVIEW_MODES = {"always_user_review", "agent_decide", "auto_approve"}
VALID_MASS_MODELING_SCOPES = {"balanced_only", "models_unbalanced_mass"}
DEFAULT_IMPLEMENTATION_REVIEW_INTERVAL = 0
DEFAULT_VIZ_FORMATS = ("png", "svg")
DEFAULT_MAIN_TEXT_FIGURES = 6
DEFAULT_VIZ_STYLE = "nature_clean"
DEFAULT_CAMPAIGN_BUILTIN_BASELINES = (
    "balanced_ot_cfm",
    "sf2m",
    "dynamical_ot",
    "vgfm",
    "wfrfm",
    "crufm",
    "ruot",
    "unbalanced_ot",
    "cyto_simulation",
)
VIZ_GOAL_KEYWORDS: Dict[str, List[str]] = {
    "publication": [
        "publication",
        "paper",
        "nature",
        "journal",
        "投稿",
        "论文",
        "发表",
        "高质量",
    ],
    "exploratory": [
        "explore",
        "exploratory",
        "debug",
        "quick look",
        "快速看",
        "探索",
        "调试",
    ],
    "presentation": [
        "presentation",
        "slide",
        "talk",
        "汇报",
        "演示",
        "讲稿",
    ],
}
FIGURE_REQUEST_RULES: Dict[str, List[str]] = {
    "sankey": ["sankey", "lineage transition", "cell lineage", "谱系", "转移"],
    "stream": ["scvelo", "velocity stream", "streamplot", "流线图", "速度流"],
    "fate_bar": ["fate probability", "fate bar", "命运概率", "命运比例", "柱状图"],
    "trajectory_overlay": ["trajectory", "pseudotime", "轨迹", "伪时间", "overlay"],
    "perturbation_effect": ["perturbation", "knockout", "overexpression", "扰动", "敲除", "过表达"],
    "growth_distribution": ["growth", "mass", "proliferation", "生长", "质量", "增殖"],
}
GROUPING_RULES: Dict[str, List[str]] = {
    "cell_type": ["cell type", "celltype", "细胞类型", "亚群", "cluster"],
    "time": ["time", "timepoint", "pseudotime", "时间", "时序", "时间点"],
    "condition": ["condition", "batch", "treatment", "对照", "处理", "批次"],
}
VIZ_BASIS_RULES: List[Tuple[str, List[str]]] = [
    (
        "spring",
        [
            "sring",
            "spring layout",
            "spring basis",
            "spring embedding",
            "spring coordinate",
            "spring coordinates",
            "spring坐标",
            "spring坐标系",
            "力导向",
        ],
    ),
    (
        "string",
        [
            "string layout",
            "string basis",
            "string embedding",
            "string coordinate",
            "string coordinates",
            "string坐标",
            "string坐标系",
            "弦图坐标",
        ],
    ),
    ("umap", ["umap"]),
    ("pca", ["pca", "pc1", "pc2"]),
    ("latent", ["latent", "隐空间"]),
    ("fast", ["fast basis", "fast embedding", "x_fast"]),
]
TASK_PROFILE_STAGES = {
    "intake",
    "proposal",
    "authoring",
    "review",
    "training",
    "downstream",
    "report",
}
TASK_PROFILE_PRIMARY_GOALS = {
    "tuning",
    "reproduction",
    "new_algorithm",
    "integration",
    "analysis",
    "idea_development",
}

_TRACE_REPO_ROOT = Path(__file__).resolve().parents[2]
_TRACE_PACKAGE_ROOT = (_TRACE_REPO_ROOT / "CytoBridge-main").resolve()
def _display_source_path(path_like: Any) -> str:
    if not path_like:
        return ""
    try:
        path = Path(path_like).resolve()
    except Exception:
        return str(path_like)
    cwd = Path.cwd().resolve()
    try:
        return str(path.relative_to(cwd))
    except Exception:
        return str(path)


def _source_ref(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}
    try:
        source_file = inspect.getsourcefile(obj) or inspect.getfile(obj)
        _, line_no = inspect.getsourcelines(obj)
    except Exception:
        return {}
    return {
        "file": _display_source_path(source_file),
        "line": int(line_no),
    }


def _callable_display_name(obj: Any) -> str:
    if obj is None:
        return ""
    if inspect.ismethod(obj):
        owner = type(getattr(obj, "__self__", None)).__name__
        return f"{owner}.{obj.__name__}"
    if inspect.isfunction(obj):
        module = getattr(obj, "__module__", "")
        qualname = getattr(obj, "__qualname__", getattr(obj, "__name__", ""))
        return f"{module}.{qualname}" if module else str(qualname)
    if inspect.isclass(obj):
        module = getattr(obj, "__module__", "")
        qualname = getattr(obj, "__qualname__", getattr(obj, "__name__", ""))
        return f"{module}.{qualname}" if module else str(qualname)
    if hasattr(obj, "__class__"):
        return f"{obj.__class__.__module__}.{obj.__class__.__name__}"
    return repr(obj)


def _object_summary(obj: Any, *, label: str, override_kind: str) -> Dict[str, Any]:
    return {
        "label": label,
        "override_kind": override_kind,
        "display_name": _callable_display_name(obj),
        "source": _source_ref(obj),
    }


class _TrainingChainTracer:
    def __init__(self, *, workspace_root: Path, max_events: int = 160):
        self.workspace_root = workspace_root.resolve()
        self.max_events = max(20, int(max_events))
        self.events: List[Dict[str, Any]] = []
        self.truncated = False

    def mark(self, label: str) -> None:
        if len(self.events) >= self.max_events:
            self.truncated = True
            return
        self.events.append({"type": "phase", "label": str(label)})

    def record_call(self, obj: Any, *, label: str = "") -> None:
        if len(self.events) >= self.max_events:
            self.truncated = True
            return
        display_name = label or _callable_display_name(obj)
        source = _source_ref(obj)
        filename = str(source.get("file") or "")
        line = int(source.get("line") or 0)
        last = self.events[-1] if self.events else None
        key = ("call", display_name, filename, line)
        if (
            isinstance(last, dict)
            and last.get("type") == "call"
            and (last.get("display_name"), last.get("file"), last.get("line")) == key[1:]
        ):
            last["count"] = int(last.get("count", 1)) + 1
            return
        self.events.append(
            {
                "type": "call",
                "display_name": display_name,
                "module": str(getattr(obj, "__module__", "") or ""),
                "file": filename,
                "line": line,
                "count": 1,
            }
        )


def _iter_traceable_method_names(obj: Any, preferred: List[str]) -> List[str]:
    if obj is None:
        return []
    cls = obj.__class__
    names: List[str] = []
    seen: Set[str] = set()
    for name in preferred:
        if hasattr(obj, name) and name not in seen:
            names.append(name)
            seen.add(name)
    for name, value in cls.__dict__.items():
        if name in seen or name.startswith("__"):
            continue
        if not callable(value):
            continue
        if name.startswith("_") or name in {
            "prepare",
            "sample_batch",
            "compute_lambda",
            "build_state",
            "sample_pairs",
            "build_pairwise_cost",
            "compute_ot_coupling",
            "compute_conditional_flow",
            "compute_conditional_mass",
            "sample_time",
            "sample_xt",
            "sample_noise_like",
            "compute",
        }:
            names.append(name)
            seen.add(name)
    return names
TASK_PROFILE_FACETS = {
    "reproduction",
    "new_algorithm",
    "tuning",
    "integration",
    "analysis",
    "idea_management",
}
TASK_PROFILE_CHANGE_AXES = {
    "data_contract",
    "solver",
    "coupling",
    "path",
    "mass",
    "loss",
    "evaluation",
    "downstream",
}
TASK_PROFILE_RISK_FLAGS = {
    "scalability_sensitive",
    "semantics_sensitive",
    "requires_baseline",
    "requires_literature",
}

class PlannerTools:
    def __init__(
        self,
        llm: ChatOpenAI,
        state: AgentState,
        event_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ):
        self.llm = llm
        self.state = state  # Reference to the shared state (mutable)
        self.state.setdefault("viz_brief_enabled", True)
        self.state.setdefault("report_figure_policy", "main_only")
        self.state.setdefault("viz_default_goal", "publication")
        self.state.setdefault("current_viz_brief", {})
        self.state.setdefault("training_runs", [])
        self.state.setdefault("latest_training_run_id", "")
        self.state.setdefault("latest_training_run_dir", "")
        self.state.setdefault("latest_training_algorithm_id", "")
        self.state.setdefault("active_algorithm_campaign_id", "")
        self.state.setdefault("active_algorithm_campaign", {})
        self.state.setdefault("planner_loaded_skills", [])
        self.state.setdefault("planner_skills_revision", 0)
        self.state.setdefault(
            "hidden_skills",
            {
                "workflow": [],
                "planner": [],
                "downstream": [],
            },
        )
        self.state.setdefault("hidden_training_algorithms", [])
        self.state.setdefault(
            "planner_skill_policy",
            {
                "enabled": True,
                "domain": "planner",
                "prefer_user_dir": True,
                "inject_max_chars": 24000,
            },
        )
        self.state.setdefault("planner_active_auto_skills", [])
        self.state.setdefault("planner_skill_injection_revision", 0)
        self.state.setdefault("planner_workspace_policy", {})
        self.state.setdefault("planner_algorithm_workspace", "")
        self.state.setdefault("planner_last_patch_summary", {})
        self.state.setdefault("algorithm_proposal_review_mode", "agent_decide")
        self.state.setdefault("idea_review_mode", "agent_decide")
        self.state.setdefault("algorithm_proposals", {})
        self.state.setdefault("latest_algorithm_proposal_id", "")
        self.state.setdefault("research_ideas", {})
        self.state.setdefault("latest_research_idea_id", "")
        self.state.setdefault("active_research_idea_id", "")
        self.state.setdefault("active_idea_registry", {})
        self.state.setdefault("decision_log_summary", [])
        self.state.setdefault("obsolete_results_summary", [])
        self.state.setdefault("active_experiment_registry", {})
        self.state.setdefault("active_proposal_id", "")
        self.state.setdefault("active_workspace_snapshot_id", "")
        self.state.setdefault("active_baseline_run_id", "")
        self.state.setdefault(
            "task_profile",
            {
                "current_stage": "intake",
                "primary_goal": "analysis",
                "facets": {key: key == "analysis" for key in TASK_PROFILE_FACETS},
                "change_axes": {key: False for key in TASK_PROFILE_CHANGE_AXES},
                "risk_flags": {key: False for key in TASK_PROFILE_RISK_FLAGS},
                "notes": "",
            },
        )
        self.event_callback = event_callback
        self.stop_check = None  # Callback to check if stop is requested
        self.plan_service = PlanService(self.state, event_sink=self._emit_event)
        self.skills_tools = SkillsTools(
            self.state,
            domain="planner",
            event_sink=self._emit_event,
            active_key="planner_loaded_skills",
            revision_key="planner_skills_revision",
            policy_key="planner_skill_policy",
        )
        self.planner_file_tools = PlannerFileTools(
            self.state,
            policy_provider=self._get_workspace_policy,
            event_sink=self._emit_event,
        )
        self._planner_history_provider: Optional[Callable[[], List[BaseMessage]]] = None

        output_dir = Path(state.get("output_dir") or "cytobridge_output")
        self.report_manager = ReportManager(output_dir)

    def _emit_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        if self.event_callback:
            try:
                self.event_callback(event_type, payload)
                return
            except Exception:
                logger.debug("PlannerTools event callback failed", exc_info=True)
        run_bundle = None
        manifest = None
        try:
            from ..display import DisplayManager

            DisplayManager()._emit(event_type, payload)
        except Exception:
            logger.debug("PlannerTools display event emit failed", exc_info=True)

    @staticmethod
    def _json_stable(value: Any) -> str:
        try:
            return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        except Exception:
            return str(value)

    @staticmethod
    def _file_fingerprint(path: Path) -> Dict[str, Any]:
        try:
            resolved = path.expanduser().resolve()
            data = resolved.read_bytes()
        except Exception:
            return {}
        return {
            "path": str(resolved),
            "sha256": hashlib.sha256(data).hexdigest(),
            "size": len(data),
        }

    @staticmethod
    def _selected_inference_config_surface(config: Dict[str, Any]) -> Dict[str, Any]:
        cfg = dict(config or {})
        model = cfg.get("model") if isinstance(cfg.get("model"), dict) else {}
        training = cfg.get("training") if isinstance(cfg.get("training"), dict) else {}
        defaults = training.get("defaults") if isinstance(training.get("defaults"), dict) else {}
        return {
            "model": {
                key: deepcopy(model.get(key))
                for key in (
                    "components",
                    "models_unbalanced_mass",
                    "use_mass",
                    "mass_modeling",
                )
                if key in model
            },
            "training_defaults": {
                key: deepcopy(defaults.get(key))
                for key in ("sigma", "dt", "simulation_dt", "inference_dt")
                if key in defaults
            },
            "evaluation": deepcopy(cfg.get("evaluation") or {}),
            "inference": deepcopy(cfg.get("inference") or {}),
            "simulation": deepcopy(cfg.get("simulation") or {}),
            "metric": deepcopy(cfg.get("metric") or {}),
            "metrics": deepcopy(cfg.get("metrics") or {}),
            "custom_metrics": deepcopy(cfg.get("custom_metrics") or {}),
        }

    @staticmethod
    def _callable_dependency_blocks(func: Any) -> Dict[str, Any]:
        if func is None:
            return {}
        label = f"{getattr(func, '__module__', '')}.{getattr(func, '__qualname__', getattr(func, '__name__', repr(func)))}"
        source_file = ""
        fallback_source = ""
        try:
            source_file = str(Path(inspect.getsourcefile(func) or "").expanduser().resolve())
        except Exception:
            source_file = ""
        try:
            fallback_source = inspect.getsource(func)
        except Exception:
            fallback_source = repr(func)
        if not source_file or not Path(source_file).exists():
            return {
                "label": label,
                "source_file": source_file,
                "blocks": {label: fallback_source},
            }
        try:
            file_text = Path(source_file).read_text(encoding="utf-8")
            tree = ast.parse(file_text)
        except Exception:
            return {
                "label": label,
                "source_file": source_file,
                "blocks": {label: fallback_source},
            }

        lines = file_text.splitlines()
        defs: Dict[str, ast.AST] = {}
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                defs[str(node.name)] = node

        seed = str(getattr(func, "__name__", "") or "").strip()
        queue: List[str] = [seed] if seed else []
        seen: Set[str] = set()
        blocks: Dict[str, str] = {}
        while queue and len(seen) < 80:
            name = queue.pop(0)
            if name in seen or name not in defs:
                continue
            seen.add(name)
            node = defs[name]
            start = int(getattr(node, "lineno", 1) or 1)
            end = int(getattr(node, "end_lineno", start) or start)
            blocks[name] = "\n".join(lines[start - 1 : end])
            for child in ast.walk(node):
                if isinstance(child, ast.Name):
                    ref = str(child.id)
                    if ref in defs and ref not in seen and ref not in queue:
                        queue.append(ref)
        if not blocks:
            blocks[label] = fallback_source
        return {
            "label": label,
            "source_file": source_file,
            "blocks": blocks,
        }

    def _build_inference_review_fingerprint(
        self,
        training_target: Any,
        *,
        resolved_config: Optional[Dict[str, Any]] = None,
        purpose: str = "",
    ) -> Dict[str, Any]:
        spec = training_target.spec
        callables = {
            "inference_context_builder": self._callable_dependency_blocks(getattr(spec, "inference_context_builder", None)),
            "simulation_hook": self._callable_dependency_blocks(getattr(spec, "simulation_hook", None)),
            "evaluation_metrics_hook": self._callable_dependency_blocks(getattr(spec, "evaluation_metrics_hook", None)),
        }
        source_paths: List[str] = []
        for item in callables.values():
            path = str((item or {}).get("source_file") or "").strip()
            if path and path not in source_paths:
                source_paths.append(path)
        workspace_root = self.planner_file_tools._proposal_dir(str(spec.algorithm_id or "").strip().lower())
        contract_files = [
            self._file_fingerprint(workspace_root / "PROPOSAL.md"),
            self._file_fingerprint(workspace_root / "PROPOSAL.json"),
        ]
        payload = {
            "version": 2,
            "purpose": str(purpose or "").strip(),
            "algorithm_id": str(spec.algorithm_id or "").strip().lower(),
            "base_config": self._json_stable(spec.base_config),
            "has_inference_context_builder": getattr(spec, "inference_context_builder", None) is not None,
            "has_simulation_hook": spec.simulation_hook is not None,
            "has_evaluation_metrics_hook": getattr(spec, "evaluation_metrics_hook", None) is not None,
            "evaluation_metrics_params": self._json_stable(getattr(spec, "evaluation_metrics_params", {}) or {}),
            "config_surface": self._selected_inference_config_surface(resolved_config or {}),
            "contract_files": [item for item in contract_files if item],
            "callables": callables,
        }
        encoded = self._json_stable(payload).encode("utf-8")
        review_hash = hashlib.sha256(encoded).hexdigest()
        return {
            "review_hash": review_hash,
            "payload": payload,
            "source_paths": source_paths,
            "requires_review": bool(
                training_target.training_mode == "custom"
                and (
                    spec.simulation_hook is not None
                    or getattr(spec, "inference_context_builder", None) is not None
                    or getattr(spec, "evaluation_metrics_hook", None) is not None
                )
            ),
        }

    def _inference_review_state(
        self,
        training_target: Any,
        *,
        resolved_config: Optional[Dict[str, Any]] = None,
        purpose: str = "",
    ) -> Dict[str, Any]:
        fingerprint = self._build_inference_review_fingerprint(
            training_target,
            resolved_config=resolved_config,
            purpose=purpose,
        )
        if not fingerprint.get("requires_review"):
            return {"ok": True, "required": False, "fingerprint": fingerprint}
        algorithm_id = str(training_target.spec.algorithm_id or "").strip().lower()
        record = self.planner_file_tools.get_inference_review_record(algorithm_id)
        ok = (
            str(record.get("status") or "") == "approved"
            and str(record.get("review_hash") or "") == str(fingerprint.get("review_hash") or "")
        )
        return {
            "ok": ok,
            "required": True,
            "record": record,
            "fingerprint": fingerprint,
        }

    def _request_inference_review_action(
        self,
        training_target: Any,
        *,
        resolved_config: Optional[Dict[str, Any]] = None,
        purpose: str = "",
        reason: str = "",
    ) -> Dict[str, Any]:
        state = self._inference_review_state(
            training_target,
            resolved_config=resolved_config,
            purpose=purpose,
        )
        if state.get("ok"):
            return state
        fingerprint = dict(state.get("fingerprint") or {})
        source_paths = list(fingerprint.get("source_paths") or [])
        algorithm_id = str(training_target.spec.algorithm_id or "").strip().lower()
        registry = self.planner_file_tools._bootstrap_algorithm_registry(algorithm_id)
        proposal_id = str(registry.get("active_proposal_id") or "").strip()
        workspace_path = str(self.planner_file_tools._proposal_dir(algorithm_id))
        relevant_paths = list(source_paths)
        for path in [
            workspace_path,
            str(self.planner_file_tools._proposal_dir(algorithm_id) / "PROPOSAL.md"),
            str(Path(__file__).resolve().parents[2] / "CytoBridge-main" / "CytoBridge" / "tl" / "trainer.py"),
            str(Path(__file__).resolve().parents[2] / "CytoBridge-main" / "CytoBridge" / "tl" / "analysis.py"),
            str(Path(__file__).resolve().parents[2] / "CytoBridge-main" / "CytoBridge" / "tl" / "training_algorithm.py"),
        ]:
            if path and path not in relevant_paths:
                relevant_paths.append(path)
        for path in self._collect_inference_review_dataset_paths(resolved_config):
            if path and path not in relevant_paths:
                relevant_paths.append(path)
        self.state["runtime_action"] = {
            "kind": "inference_agent_review",
            "algorithm_id": algorithm_id,
            "proposal_id": proposal_id,
            "purpose": str(purpose or "").strip(),
            "reason": str(reason or "custom inference review required").strip(),
            "review_hash": str(fingerprint.get("review_hash") or ""),
            "fingerprint_payload": dict(fingerprint.get("payload") or {}),
            "relevant_paths": relevant_paths,
            "workspace_path": workspace_path,
        }
        self._emit_event(
            "algorithm_inference_review_requested",
            {
                "algorithm_id": algorithm_id,
                "proposal_id": proposal_id,
                "purpose": str(purpose or "").strip(),
                "review_hash": str(fingerprint.get("review_hash") or ""),
                "reason": str(reason or "").strip(),
                "source_paths": source_paths,
            },
        )
        state["runtime_action_set"] = True
        return state

    def _collect_inference_review_dataset_paths(self, resolved_config: Optional[Dict[str, Any]]) -> List[str]:
        paths: List[str] = []

        def add(value: Any) -> None:
            text = str(value or "").strip()
            if not text or not text.lower().endswith(".h5ad"):
                return
            expanded = str(Path(text).expanduser())
            if expanded not in paths:
                paths.append(expanded)

        def walk(value: Any) -> None:
            if isinstance(value, dict):
                for item in value.values():
                    walk(item)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    walk(item)
            elif isinstance(value, str):
                add(value)

        walk(resolved_config or {})
        for key in ("preprocessed_path", "converted_path", "input_path"):
            add(self.state.get(key))
        return paths

    def _workspace_implementation_fingerprint_files(self, algorithm_id: str) -> List[Dict[str, Any]]:
        algo_id = str(algorithm_id or "").strip().lower()
        workspace = self.planner_file_tools._proposal_dir(algo_id)
        if not workspace.exists():
            return []
        allowed_suffixes = {".py", ".yaml", ".yml", ".json", ".md", ".toml", ".txt"}
        excluded_dirs = {"registry", "__pycache__", ".git", ".pytest_cache", ".mypy_cache"}
        excluded_files = {"risk.md"}
        items: List[Dict[str, Any]] = []
        for path in sorted(workspace.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(workspace)
            if any(part in excluded_dirs for part in rel.parts):
                continue
            if rel.name in excluded_files:
                continue
            if path.suffix.lower() not in allowed_suffixes:
                continue
            try:
                data = path.read_bytes()
            except Exception:
                continue
            items.append(
                {
                    "path": str(rel),
                    "size": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
            )
        return items[:300]

    @staticmethod
    def _campaign_completed_trial_count(campaign: Optional[Dict[str, Any]]) -> int:
        if not isinstance(campaign, dict):
            return 0
        stages = campaign.get("stages") if isinstance(campaign.get("stages"), dict) else {}
        total = 0
        for stage_state in stages.values():
            if isinstance(stage_state, dict):
                try:
                    total += int(stage_state.get("trial_count") or 0)
                except Exception:
                    pass
        return max(0, total)

    def _implementation_map_preflight(self, algorithm_id: str) -> Dict[str, Any]:
        algo_id = str(algorithm_id or "").strip().lower()
        path = self.planner_file_tools._proposal_dir(algo_id) / "IMPLEMENTATION_MAP.md"
        if not algo_id:
            return {"ok": False, "path": "", "blockers": ["algorithm_id is required"], "rows": []}
        if not path.exists():
            return {
                "ok": False,
                "path": str(path),
                "blockers": ["IMPLEMENTATION_MAP.md is missing; fill it before implementation review."],
                "rows": [],
            }
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception as exc:
            return {
                "ok": False,
                "path": str(path),
                "blockers": [f"IMPLEMENTATION_MAP.md could not be read: {exc}"],
                "rows": [],
            }

        required_anchor_baseline = self._implementation_map_required_anchor_baseline_from_text(text)
        required_columns = [
            "Step ID",
            "Pseudocode Step",
            "Implementation File",
            "Status",
            "Semantic Check",
            "Deviation Status",
            "Notes",
        ]
        table_rows: List[Dict[str, str]] = []
        header: List[str] = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line.startswith("|") or not line.endswith("|"):
                continue
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if not header and all(col in cells for col in ("Step ID", "Pseudocode Step", "Implementation File")):
                header = cells
                continue
            if not header:
                continue
            if all(re.fullmatch(r":?-{3,}:?", cell or "") for cell in cells):
                continue
            if len(cells) != len(header):
                continue
            table_rows.append({header[idx]: cells[idx] for idx in range(len(header))})

        blockers: List[str] = []
        warnings: List[str] = []
        missing_columns = [col for col in required_columns if col not in header]
        if missing_columns:
            blockers.append("IMPLEMENTATION_MAP.md mapping table is missing required columns: " + ", ".join(missing_columns))
        if header and "Line Number(s)" not in header:
            warnings.append(
                "IMPLEMENTATION_MAP.md is missing optional Line Number(s) column; file/function references are acceptable."
            )
        if not table_rows:
            blockers.append("IMPLEMENTATION_MAP.md has no completed mapping rows.")
        if not required_anchor_baseline:
            blockers.append(
                "IMPLEMENTATION_MAP.md is missing required Baseline Anchor field. "
                "Add `Required anchor baseline: <nearest_builtin>` before implementation review."
            )

        incomplete_rows: List[str] = []
        semantic_drift_rows: List[str] = []
        approximation_rows: List[str] = []
        for row in table_rows:
            step_id = str(row.get("Step ID") or "").strip() or f"row{len(incomplete_rows) + 1}"
            required_values = [
                row.get("Pseudocode Step"),
                row.get("Implementation File"),
                row.get("Status"),
                row.get("Semantic Check"),
                row.get("Deviation Status"),
            ]
            if any(str(value or "").strip().upper() in {"", "TODO", "TBD"} for value in required_values):
                incomplete_rows.append(step_id)
            status_value = str(row.get("Status") or "").strip().lower()
            if status_value == "pending":
                incomplete_rows.append(step_id)
            deviation = str(row.get("Deviation Status") or "").strip().lower()
            if deviation == "semantic drift":
                semantic_drift_rows.append(step_id)
            if "approx" in deviation:
                approximation_rows.append(step_id)

        if incomplete_rows:
            blockers.append("IMPLEMENTATION_MAP.md still has incomplete/TODO/pending rows: " + ", ".join(sorted(set(incomplete_rows))))
        if semantic_drift_rows:
            blockers.append(
                "IMPLEMENTATION_MAP.md declares semantic drift for rows "
                + ", ".join(sorted(set(semantic_drift_rows)))
                + "; patch/review the proposal or fix the implementation before training."
            )
        if approximation_rows:
            warnings.append(
                "Rows marked as approximation must be checked by implementation_evaluator for semantics preservation: "
                + ", ".join(sorted(set(approximation_rows)))
            )
        return {
            "ok": not blockers,
            "path": str(path),
            "blockers": blockers,
            "warnings": warnings,
            "rows": table_rows[:80],
            "row_count": len(table_rows),
            "approximation_rows": sorted(set(approximation_rows)),
            "required_anchor_baseline": required_anchor_baseline,
        }

    @staticmethod
    def _normalize_baseline_algorithm_name(value: Any) -> str:
        return re.sub(r"[^a-z0-9_.-]+", "-", str(value or "").strip().lower()).strip("-")

    @classmethod
    def _implementation_map_required_anchor_baseline_from_text(cls, text: str) -> str:
        for raw_line in str(text or "").splitlines():
            line = raw_line.strip()
            if not re.search(r"(nearest\s+builtin|anchor\s+baseline|required\s+anchor)", line, flags=re.IGNORECASE):
                continue
            match = re.search(
                r"(?:nearest\s+builtin(?:\s+anchor)?|required\s+anchor\s+baseline|anchor\s+baseline)\s*[:=]\s*`?([A-Za-z0-9_.-]+)`?",
                line,
                flags=re.IGNORECASE,
            )
            if not match:
                continue
            anchor = cls._normalize_baseline_algorithm_name(match.group(1))
            if anchor and anchor not in {"todo", "tbd", "none", "n/a", "na"}:
                return anchor
        return ""

    def _implementation_map_required_anchor_baseline(self, algorithm_id: str) -> str:
        algo_id = str(algorithm_id or "").strip().lower()
        if not algo_id:
            return ""
        path = self.planner_file_tools._proposal_dir(algo_id) / "IMPLEMENTATION_MAP.md"
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return ""
        return self._implementation_map_required_anchor_baseline_from_text(text)

    def _build_implementation_review_fingerprint(
        self,
        algorithm_id: str,
        *,
        proposal_id: str = "",
        campaign: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        algo_id = str(algorithm_id or "").strip().lower()
        if not algo_id:
            return {"review_hash": "", "payload": {}, "source_paths": [], "requires_review": False}
        registry = self.planner_file_tools._bootstrap_algorithm_registry(algo_id)
        target_proposal_id = str(proposal_id or registry.get("active_proposal_id") or "").strip()
        proposal_record = (
            self.planner_file_tools._proposal_record_from_registry(algo_id, registry, target_proposal_id)
            if target_proposal_id
            else {}
        )
        proposal_payload = {
            "proposal_id": target_proposal_id,
            "status": str(proposal_record.get("status") or ""),
            "objective": str(proposal_record.get("objective") or ""),
            "theoretical_core": str(proposal_record.get("theoretical_core") or ""),
            "mathematical_abstraction": str(proposal_record.get("mathematical_abstraction") or ""),
            "algorithm_semantics_table": str(proposal_record.get("algorithm_semantics_table") or ""),
            "implementation_pseudocode": str(proposal_record.get("implementation_pseudocode") or ""),
            "evaluation_plan": str(proposal_record.get("evaluation_plan") or ""),
            "unbalanced_decision": str(proposal_record.get("unbalanced_decision") or ""),
            "stochasticity_decision": str(proposal_record.get("stochasticity_decision") or ""),
            "distribution_recovery_argument": str(proposal_record.get("distribution_recovery_argument") or ""),
            "overengineering_self_check": str(proposal_record.get("overengineering_self_check") or ""),
            "uncertainty_and_risks": str(proposal_record.get("uncertainty_and_risks") or ""),
        }
        workspace_files = self._workspace_implementation_fingerprint_files(algo_id)
        implementation_map_status = self._implementation_map_preflight(algo_id)
        workspace_path = str(self.planner_file_tools._proposal_dir(algo_id))
        source_paths = [workspace_path]
        for key in ("proposal_markdown_registry_path", "registry_path", "proposal_path", "proposal_json_path"):
            path = str(proposal_record.get(key) or "").strip()
            if path and path not in source_paths:
                source_paths.append(path)
        for rel in ("algorithm.py", "config.yaml", "IMPLEMENTATION_MAP.md", "manifest.yaml", "README.md"):
            path = str(self.planner_file_tools._proposal_dir(algo_id) / rel)
            if Path(path).exists() and path not in source_paths:
                source_paths.append(path)
        hash_payload = {
            "version": 3,
            "scope": "proposal_semantics",
            "implementation_review_policy_version": IMPLEMENTATION_REVIEW_POLICY_VERSION,
            "algorithm_id": algo_id,
            "proposal": proposal_payload,
        }
        payload = {
            "version": 3,
            "review_hash_scope": "proposal_semantics",
            "implementation_review_policy_version": IMPLEMENTATION_REVIEW_POLICY_VERSION,
            "hash_payload": hash_payload,
            "algorithm_id": algo_id,
            "proposal": proposal_payload,
            "implementation_map": implementation_map_status,
            # Keep current workspace files in the reviewer payload for context, but
            # do not include them in review_hash. Ordinary code/config tuning should
            # be checked by the periodic review interval, not every trial.
            "workspace_files": workspace_files,
        }
        encoded = self._json_stable(hash_payload).encode("utf-8")
        return {
            "review_hash": hashlib.sha256(encoded).hexdigest(),
            "review_hash_scope": "proposal_semantics",
            "payload": payload,
            "source_paths": source_paths,
            "requires_review": bool(algo_id and target_proposal_id),
        }

    def _implementation_review_state(
        self,
        algorithm_id: str,
        *,
        proposal_id: str = "",
        campaign: Optional[Dict[str, Any]] = None,
        review_interval: int = DEFAULT_IMPLEMENTATION_REVIEW_INTERVAL,
    ) -> Dict[str, Any]:
        algo_id = str(algorithm_id or "").strip().lower()
        fingerprint = self._build_implementation_review_fingerprint(
            algo_id,
            proposal_id=proposal_id,
            campaign=campaign,
        )
        if not fingerprint.get("requires_review"):
            return {"ok": True, "required": False, "fingerprint": fingerprint}
        implementation_map = dict((fingerprint.get("payload") or {}).get("implementation_map") or {})
        record = self.planner_file_tools.get_implementation_review_record(algo_id)
        target_proposal_id = str((fingerprint.get("payload") or {}).get("proposal", {}).get("proposal_id") or "").strip()
        current_count = self._campaign_completed_trial_count(campaign)
        interval = max(0, int(review_interval or 0))
        proposal_matches = str(record.get("proposal_id") or "") == target_proposal_id
        status_ok = str(record.get("status") or "") == "approved"
        record_scope = str(record.get("review_hash_scope") or "").strip()
        hash_matches = str(record.get("review_hash") or "") == str(fingerprint.get("review_hash") or "")
        try:
            record_policy_version = int(record.get("implementation_review_policy_version") or 0)
        except Exception:
            record_policy_version = 0
        policy_matches = record_policy_version == IMPLEMENTATION_REVIEW_POLICY_VERSION
        # Backward compatibility for reviews recorded before implementation review
        # switched to proposal-only hashes. Those old records hashed workspace files;
        # keep them valid until the proposal changes and forces a fresh review.
        if status_ok and proposal_matches and not record_scope and str(record.get("review_hash") or "").strip():
            hash_matches = True
        review_due = not bool(status_ok and hash_matches and proposal_matches and policy_matches)
        if review_due and implementation_map and not bool(implementation_map.get("ok")):
            return {
                "ok": False,
                "required": True,
                "record": record,
                "fingerprint": fingerprint,
                "current_trial_count": current_count,
                "review_interval": interval,
                "implementation_map": implementation_map,
                "runtime_action_set": False,
                "due_reasons": list(implementation_map.get("blockers") or []),
            }
        ok = not review_due
        due_reasons: List[str] = []
        if not status_ok:
            due_reasons.append("no approved implementation review is recorded")
        if status_ok and not proposal_matches:
            due_reasons.append("approved implementation review belongs to a different proposal_id")
        if status_ok and not policy_matches:
            due_reasons.append("implementation review policy changed since the approved implementation review")
        if status_ok and proposal_matches and policy_matches and not hash_matches:
            due_reasons.append("proposal semantics changed since the approved implementation review")
        return {
            "ok": ok,
            "required": True,
            "record": record,
            "fingerprint": fingerprint,
            "current_trial_count": current_count,
            "review_interval": interval,
            "due_reasons": due_reasons,
        }

    def _request_implementation_review_action(
        self,
        algorithm_id: str,
        *,
        proposal_id: str = "",
        campaign: Optional[Dict[str, Any]] = None,
        purpose: str = "",
        reason: str = "",
        review_interval: int = DEFAULT_IMPLEMENTATION_REVIEW_INTERVAL,
    ) -> Dict[str, Any]:
        state = self._implementation_review_state(
            algorithm_id,
            proposal_id=proposal_id,
            campaign=campaign,
            review_interval=review_interval,
        )
        if state.get("ok"):
            return state
        implementation_map = dict(state.get("implementation_map") or {})
        if implementation_map and not bool(implementation_map.get("ok")):
            algo_id = str(algorithm_id or "").strip().lower()
            self._emit_event(
                "algorithm_implementation_map_required",
                {
                    "algorithm_id": algo_id,
                    "proposal_id": str(proposal_id or "").strip(),
                    "path": str(implementation_map.get("path") or ""),
                    "blockers": list(implementation_map.get("blockers") or []),
                },
            )
            return state
        fingerprint = dict(state.get("fingerprint") or {})
        algo_id = str(algorithm_id or "").strip().lower()
        payload = dict(fingerprint.get("payload") or {})
        target_proposal_id = str((payload.get("proposal") or {}).get("proposal_id") or proposal_id or "").strip()
        campaign_id = str((campaign or {}).get("campaign_id") or "").strip()
        current_count = int(state.get("current_trial_count") or 0)
        interval = int(state.get("review_interval") or 0)
        self.state["runtime_action"] = {
            "kind": "implementation_agent_review",
            "algorithm_id": algo_id,
            "proposal_id": target_proposal_id,
            "purpose": str(purpose or "").strip(),
            "reason": str(reason or "proposal-implementation alignment review required").strip(),
            "review_hash": str(fingerprint.get("review_hash") or ""),
            "review_hash_scope": str(fingerprint.get("review_hash_scope") or ""),
            "implementation_review_policy_version": IMPLEMENTATION_REVIEW_POLICY_VERSION,
            "fingerprint_payload": payload,
            "relevant_paths": list(fingerprint.get("source_paths") or []),
            "workspace_path": str(self.planner_file_tools._proposal_dir(algo_id)),
            "campaign_id": campaign_id,
            "campaign_trial_count": current_count,
            "review_interval": interval,
            "review_policy": "initial_or_proposal_semantics_change",
            "due_reasons": list(state.get("due_reasons") or []),
        }
        self._emit_event(
            "algorithm_implementation_review_requested",
            {
                "algorithm_id": algo_id,
                "proposal_id": target_proposal_id,
                "purpose": str(purpose or "").strip(),
                "review_hash": str(fingerprint.get("review_hash") or ""),
                "review_hash_scope": str(fingerprint.get("review_hash_scope") or ""),
                "implementation_review_policy_version": IMPLEMENTATION_REVIEW_POLICY_VERSION,
                "campaign_id": campaign_id,
                "campaign_trial_count": current_count,
                "review_interval": interval,
                "review_policy": "initial_or_proposal_semantics_change",
                "due_reasons": list(state.get("due_reasons") or []),
            },
        )
        state["runtime_action_set"] = True
        return state

    def _ensure_training_review_ready(
        self,
        training_target: Any,
        *,
        stage: str,
        resolved_config: Optional[Dict[str, Any]],
        purpose: str,
        campaign: Optional[Dict[str, Any]] = None,
        proposal_id: str = "",
        resume_after_review: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if getattr(training_target, "training_mode", "") != "custom":
            return {"ok": True, "required": False}

        algo_id = str(training_target.spec.algorithm_id or "").strip().lower()
        purpose_value = str(purpose or "manual_training").strip() or "manual_training"
        campaign_payload = dict(campaign or {}) if isinstance(campaign, dict) else None
        target_proposal_id = str(proposal_id or "").strip()
        if not target_proposal_id and campaign_payload:
            target_proposal_id = str(campaign_payload.get("proposal_id") or "").strip()

        implementation_state = self._request_implementation_review_action(
            algo_id,
            proposal_id=target_proposal_id,
            campaign=campaign_payload,
            purpose=purpose_value,
            reason=(
                "campaign trials require a read-only proposal-implementation alignment review "
                "before producing trusted promote/reject evidence"
                if purpose_value == "campaign"
                else "manual custom training requires proposal-implementation alignment review"
            ),
        )
        if not bool(implementation_state.get("ok")):
            runtime_action = self.state.get("runtime_action")
            if (
                resume_after_review
                and isinstance(runtime_action, dict)
                and runtime_action.get("kind") == "implementation_agent_review"
            ):
                runtime_action["resume_after_review"] = deepcopy(resume_after_review)
            implementation_map = dict(implementation_state.get("implementation_map") or {})
            if implementation_map and not bool(implementation_map.get("ok")):
                return {
                    "ok": False,
                    "status": "blocked",
                    "reason": "implementation_map_required",
                    "message": (
                        "IMPLEMENTATION_MAP.md must be completed before implementation review or trusted training. "
                        "Map each proposal pseudocode step to concrete code/config lines, classify deviations, "
                        "and self-check whether approximations preserve the approved proposal semantics."
                    ),
                    "algorithm_id": algo_id,
                    "proposal_id": target_proposal_id,
                    "campaign_id": str((campaign_payload or {}).get("campaign_id") or ""),
                    "implementation_map_path": str(implementation_map.get("path") or ""),
                    "blockers": list(implementation_map.get("blockers") or []),
                    "warnings": list(implementation_map.get("warnings") or []),
                }
            return {
                "ok": False,
                "status": "blocked",
                "reason": "implementation_review_required",
                "message": (
                    "Implementation review is required before trusted training evidence. "
                    "The runtime will start a read-only implementation evaluator subagent now. "
                    "If it approves, the runtime will automatically resume the deferred training call."
                ),
                "algorithm_id": algo_id,
                "proposal_id": target_proposal_id,
                "campaign_id": str((campaign_payload or {}).get("campaign_id") or ""),
                "review_hash": str((implementation_state.get("fingerprint") or {}).get("review_hash") or ""),
                "due_reasons": list(implementation_state.get("due_reasons") or []),
            }

        inference_state = self._request_inference_review_action(
            training_target,
            resolved_config=resolved_config,
            purpose=purpose_value,
            reason=(
                "campaign promote/reject requires trusted custom inference code"
                if purpose_value == "campaign"
                else "custom evaluation-time inference must be reviewed before trusted training"
            ),
        )
        if not bool(inference_state.get("ok")):
            runtime_action = self.state.get("runtime_action")
            if (
                resume_after_review
                and isinstance(runtime_action, dict)
                and runtime_action.get("kind") == "inference_agent_review"
            ):
                runtime_action["resume_after_review"] = deepcopy(resume_after_review)
            return {
                "ok": False,
                "status": "blocked",
                "reason": "inference_review_required",
                "message": (
                    "Custom inference review is required before trusted training metrics. "
                    "The runtime will start a read-only inference evaluator subagent now. "
                    "If it approves, the runtime will automatically resume the deferred training call."
                ),
                "algorithm_id": algo_id,
                "proposal_id": target_proposal_id,
                "campaign_id": str((campaign_payload or {}).get("campaign_id") or ""),
                "review_hash": str((inference_state.get("fingerprint") or {}).get("review_hash") or ""),
            }

        return {
            "ok": True,
            "required": True,
            "implementation_review": implementation_state,
            "inference_review": inference_state,
        }

    def _emit_python_execution_event(self, phase: str, payload: Dict[str, Any]) -> None:
        self._emit_event(f"item/pythonExecution/{phase}", payload)

    def _resolve_active_python_data_path(self) -> Optional[str]:
        data_path = self.state.get("preprocessed_path") or self.state.get("input_path")
        if self.state.get("final_config") and "path" in self.state["final_config"]:
            data_path = self.state["final_config"]["path"]
        return data_path

    def _ensure_active_code_executor(self, data_path: Optional[str]) -> tuple[Path, bool]:
        from .code_executor import CodeExecutor
        from .adata_manager import AnnDataManager

        manager = AnnDataManager()
        output_dir = Path(self.state.get("output_dir") or "cytobridge_output")
        adata = None
        managed_write_owner = self._active_algorithm_id_for_profile()

        if data_path:
            path_obj = Path(data_path)
            if path_obj.exists() and path_obj.suffix.lower() == ".h5ad":
                if manager.get_path() != str(path_obj.resolve()):
                    manager.bind_path(str(path_obj))
            elif adata is None and path_obj.exists():
                logger.info("python runtime bootstrap mode: using raw input path without preloading adata")

        scope_reused = bool(hasattr(self, "_executor") and self._executor_path == data_path)
        if not scope_reused:
            self._executor = CodeExecutor(
                adata=adata,
                output_dir=output_dir,
                input_path=str(data_path) if data_path else None,
                managed_write_owner=managed_write_owner,
            )
            self._executor_path = data_path
        else:
            # Fresh workers create the executor before they have selected an
            # algorithm. Keep the managed-file guard scoped to the current
            # owner once the active algorithm becomes known, otherwise the
            # guard can mistake other workers' concurrent campaign writes for
            # this worker's execute_python side effects.
            self._executor.managed_write_owner = managed_write_owner or None

        return output_dir, scope_reused

    def _get_current_manager_path(self) -> str:
        from .adata_manager import AnnDataManager
        current = AnnDataManager().get_path()
        return current or "(none)"

    def set_planner_history_provider(self, provider: Optional[Callable[[], List[BaseMessage]]]) -> None:
        """Bind a callable that returns the planner's full chat history."""
        self._planner_history_provider = provider

    def _get_workspace_policy(self) -> WorkspacePolicy:
        policy = build_planner_workspace_policy(self.state, workspace_root=get_workspace_root())
        self.state["planner_workspace_policy"] = policy.to_state_dict()
        return policy

    def render_planner_workspace_context(self) -> str:
        return self._get_workspace_policy().render_prompt_block()

    def render_planner_skill_context(self, turn_text: str) -> str:
        routed_payload: List[Dict[str, Any]] = []
        previous = self.state.get("planner_active_auto_skills") or []
        if previous != routed_payload:
            self.state["planner_skill_injection_revision"] = int(
            self.state.get("planner_skill_injection_revision", 0)
            ) + 1
            self._emit_event("planner_auto_skills_updated", {"skills": routed_payload})
        self.state["planner_active_auto_skills"] = routed_payload
        rendered, summary = self.skills_tools.build_turn_context(
            turn_text="",
            auto_skill_names=[],
        )
        summary_payload = {
            **summary,
            "scope": "planner",
            "auto_routes": routed_payload,
        }
        signature = json.dumps(summary_payload, ensure_ascii=False, sort_keys=True)
        if self.state.get("_planner_skills_context_signature") != signature:
            self.state["_planner_skills_context_signature"] = signature
            self._emit_event("skills_context_updated", summary_payload)
        return rendered

    @staticmethod
    def _preview_text(text: str, max_chars: int = 1200) -> str:
        value = str(text or "")
        if len(value) <= max_chars:
            return value
        return value[:max_chars] + "\n...[truncated]..."

    @staticmethod
    def _json_safe_context_value(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(key): PlannerTools._json_safe_context_value(val) for key, val in value.items()}
        if isinstance(value, (list, tuple)):
            return [PlannerTools._json_safe_context_value(item) for item in value]
        if isinstance(value, set):
            return [
                PlannerTools._json_safe_context_value(item)
                for item in sorted(value, key=lambda candidate: str(candidate))
            ]
        try:
            json.dumps(value)
            return value
        except TypeError:
            return str(value)

    @staticmethod
    def _compact_context_record(record: Dict[str, Any], keys: List[str], max_chars: int = 500) -> Dict[str, Any]:
        compact: Dict[str, Any] = {}
        for key in keys:
            if key not in record:
                continue
            value = PlannerTools._json_safe_context_value(record.get(key))
            if isinstance(value, str) and len(value) > max_chars:
                value = value[:max_chars] + "\n...[truncated]..."
            compact[key] = value
        return compact

    def _emit_file_activity(self, action: str, **payload: Any) -> None:
        self._emit_event(
            "file_activity",
            {
                "scope": "planner",
                "action": action,
                **payload,
            },
        )

    @staticmethod
    def _default_task_profile() -> Dict[str, Any]:
        return {
            "current_stage": "intake",
            "primary_goal": "analysis",
            "facets": {key: key == "analysis" for key in TASK_PROFILE_FACETS},
            "change_axes": {key: False for key in TASK_PROFILE_CHANGE_AXES},
            "risk_flags": {key: False for key in TASK_PROFILE_RISK_FLAGS},
            "notes": "",
        }

    def _normalize_task_profile(self, profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        raw = dict(profile or self.state.get("task_profile") or {})
        normalized = self._default_task_profile()

        current_stage = str(raw.get("current_stage") or "").strip().lower()
        if current_stage:
            normalized["current_stage"] = current_stage

        primary_goal = str(raw.get("primary_goal") or "").strip().lower()
        if primary_goal:
            normalized["primary_goal"] = primary_goal

        facets = raw.get("facets")
        if isinstance(facets, dict):
            for key in TASK_PROFILE_FACETS:
                if key in facets:
                    normalized["facets"][key] = bool(facets[key])

        change_axes = raw.get("change_axes")
        if isinstance(change_axes, dict):
            for key in TASK_PROFILE_CHANGE_AXES:
                if key in change_axes:
                    normalized["change_axes"][key] = bool(change_axes[key])

        risk_flags = raw.get("risk_flags")
        if isinstance(risk_flags, dict):
            for key in TASK_PROFILE_RISK_FLAGS:
                if key in risk_flags:
                    normalized["risk_flags"][key] = bool(risk_flags[key])

        normalized["notes"] = str(raw.get("notes") or "").strip()
        return normalized

    @staticmethod
    def _derive_task_profile_risk_flags(profile: Dict[str, Any]) -> Dict[str, bool]:
        facets = dict(profile.get("facets") or {})
        change_axes = dict(profile.get("change_axes") or {})
        primary_goal = str(profile.get("primary_goal") or "").strip().lower()
        semantics_sensitive = any(
            bool(change_axes.get(key))
            for key in ("solver", "coupling", "path", "mass", "loss", "evaluation")
        )
        scalability_sensitive = bool(primary_goal in {"new_algorithm", "reproduction"}) or any(
            bool(change_axes.get(key))
            for key in ("data_contract", "solver", "coupling", "path", "mass", "loss")
        )
        requires_baseline = bool(primary_goal in {"new_algorithm", "reproduction", "tuning"}) or any(
            bool(facets.get(key))
            for key in ("new_algorithm", "reproduction", "tuning")
        )
        requires_literature = bool(primary_goal in {"new_algorithm", "reproduction", "analysis", "idea_development"}) or any(
            bool(facets.get(key))
            for key in ("new_algorithm", "reproduction", "analysis", "idea_management")
        )
        return {
            "scalability_sensitive": scalability_sensitive,
            "semantics_sensitive": semantics_sensitive,
            "requires_baseline": requires_baseline,
            "requires_literature": requires_literature,
        }

    def _active_algorithm_id_for_profile(self) -> str:
        active_context = dict(self.state.get("active_algorithm_context") or {})
        active_algo = str(active_context.get("algorithm_id") or "").strip().lower()
        if active_algo:
            return active_algo
        workspace_path = str(self.state.get("planner_algorithm_workspace") or "").strip()
        if workspace_path:
            return Path(workspace_path).name.strip().lower()
        active_registry = dict(self.state.get("active_experiment_registry") or {})
        registry_algo = str(active_registry.get("algorithm_id") or "").strip().lower()
        if registry_algo:
            return registry_algo
        latest_proposal_id = str(self.state.get("latest_algorithm_proposal_id") or "").strip().lower()
        if latest_proposal_id:
            return latest_proposal_id
        latest_training_algorithm_id = str(self.state.get("latest_training_algorithm_id") or "").strip().lower()
        if latest_training_algorithm_id:
            return latest_training_algorithm_id
        return ""

    def _approved_algorithm_record(self, algorithm_id: str) -> Dict[str, Any]:
        algo_id = str(algorithm_id or "").strip().lower()
        if not algo_id:
            return {}
        record = dict((self.state.get("algorithm_proposals") or {}).get(algo_id) or {})
        if record:
            return record
        try:
            return self._load_proposal_record(algo_id)
        except Exception:
            return {}

    @staticmethod
    def _sanitize_tool_call_history(messages: List[BaseMessage]) -> List[BaseMessage]:
        """Drop dangling assistant tool_call messages missing subsequent ToolMessage replies."""
        sanitized: List[BaseMessage] = []
        i = 0
        while i < len(messages):
            msg = messages[i]
            tool_calls = getattr(msg, "tool_calls", None)
            if tool_calls:
                ids = {tc.get("id") for tc in tool_calls if tc.get("id")}
                j = i + 1
                matched = set()
                while j < len(messages) and isinstance(messages[j], ToolMessage):
                    tcid = getattr(messages[j], "tool_call_id", None)
                    if tcid:
                        matched.add(tcid)
                    j += 1
                if ids and not ids.issubset(matched):
                    logger.warning("Dropping dangling assistant tool_call block before report generation")
                    i = j
                    continue
            sanitized.append(msg)
            i += 1
        return sanitized

    def _get_planner_history(self) -> List[BaseMessage]:
        if not self._planner_history_provider:
            return []
        try:
            history = self._planner_history_provider() or []
            return list(history)
        except Exception:
            logger.debug("Failed to fetch planner history from provider", exc_info=True)
            return []

    def _build_context_sync_block(self, stage: str, preferred_path: Optional[str]) -> str:
        final_cfg = self.state.get("final_config") or {}
        final_model_path = final_cfg.get("path") if isinstance(final_cfg, dict) else ""
        lines = [
            f"stage={stage}",
            f"preferred_data_path={preferred_path or '(none)'}",
            f"input_path={self.state.get('input_path') or '(none)'}",
            f"converted_path={self.state.get('converted_path') or '(none)'}",
            f"preprocessed_path={self.state.get('preprocessed_path') or '(none)'}",
            f"final_model_path={final_model_path or '(none)'}",
            f"adata_manager_current_path={self._get_current_manager_path()}",
            f"time_key={self.state.get('time_key') or '(none)'}",
            f"label_key={self.state.get('label_key') or '(none)'}",
            f"has_preprocessed={'yes' if bool(self.state.get('preprocessed_path')) else 'no'}",
            f"has_trained_model={'yes' if bool(final_model_path) else 'no'}",
        ]
        return "\n".join(lines)

    def _compose_subagent_instruction(self, stage: str, instruction: str, preferred_path: Optional[str]) -> str:
        user_goal_q = self.state.get("user_goal", {}).get("raw_question", "")
        context_block = self._build_context_sync_block(stage=stage, preferred_path=preferred_path)
        return (
            "[PLANNER_CONTEXT_SYNC]\n"
            f"{context_block}\n"
            "[/PLANNER_CONTEXT_SYNC]\n\n"
            f"Global Goal: {user_goal_q}\n"
            f"Planner Instruction: {instruction}\n\n"
            "Execution rule: treat PLANNER_CONTEXT_SYNC as authoritative state. "
            "If required path/key exists there, use it directly before re-discovering."
        )

    @staticmethod
    def _is_visualization_request(text: str) -> bool:
        s = str(text or "").lower()
        if not s:
            return False
        core = (
            "plot",
            "figure",
            "visual",
            "visualization",
            "render",
            "replot",
            "style",
            "draw",
            "画图",
            "可视化",
            "图",
            "风格",
            "重画",
        )
        return any(k in s for k in core)

    @classmethod
    def _infer_viz_brief(cls, instruction: str, *, default_goal: str = "publication") -> Dict[str, Any]:
        text_raw = str(instruction or "").strip()
        s = text_raw.lower()
        # Visualization requests are now rerun-first by default.
        force_rerun_analysis_for_plot = True

        viz_goal = str(default_goal or "publication").strip().lower()
        for goal, keywords in VIZ_GOAL_KEYWORDS.items():
            if any(k in s for k in keywords):
                viz_goal = goal
                break
        if viz_goal not in {"publication", "exploratory", "presentation"}:
            viz_goal = "publication"

        figure_requests: List[str] = []
        for figure_name, keywords in FIGURE_REQUEST_RULES.items():
            if any(k in s for k in keywords):
                figure_requests.append(figure_name)
        if not figure_requests:
            figure_requests = ["trajectory_overlay"]

        grouping: List[str] = []
        for group_name, keywords in GROUPING_RULES.items():
            if any(k in s for k in keywords):
                grouping.append(group_name)
        if not grouping:
            grouping = ["cell_type", "time"]

        style = (
            DEFAULT_VIZ_STYLE
            if ("nature" in s or "clean" in s or "publication" in s or "论文" in s)
            else "default_publication"
        )

        formats: List[str] = []
        for fmt in ("png", "svg", "pdf", "html"):
            if re.search(rf"(?<![a-z0-9]){fmt}(?![a-z0-9])", s):
                formats.append(fmt)
        if not formats:
            formats = list(DEFAULT_VIZ_FORMATS)
        if any(req in figure_requests for req in ("sankey", "stream")) and "html" not in formats:
            formats.append("html")

        main_text_figures = DEFAULT_MAIN_TEXT_FIGURES
        for m in re.finditer(r"(?:main\s*text|主文|正文)\s*[:=]?\s*(\d+)", s):
            try:
                main_text_figures = max(1, min(30, int(m.group(1))))
                break
            except Exception:
                continue
        if "all in main text" in s or "主文全量" in s:
            main_text_figures = 30

        include_all_in_appendix = False

        basis_hits: List[Tuple[int, str, str]] = []
        for m in re.finditer(r"\bx_([a-z0-9_]+)\b", s):
            basis_hits.append((m.start(), str(m.group(1)).strip(), str(m.group(0)).strip()))
        for basis, keywords in VIZ_BASIS_RULES:
            for kw in keywords:
                idx = s.rfind(str(kw).lower())
                if idx >= 0:
                    basis_hits.append((idx, basis, kw))
        basis_preference = ""
        strict_basis = False
        if basis_hits:
            # Prefer the last-mentioned basis in user text to capture overrides like "not UMAP, use spring".
            basis_hits.sort(key=lambda x: x[0])
            _, basis_preference, _ = basis_hits[-1]
            strict_basis = True

        brief: Dict[str, Any] = {
            "viz_goal": viz_goal,
            "figure_requests": figure_requests,
            "grouping": grouping,
            "style": style,
            "formats": formats,
            "main_text_figures": main_text_figures,
            "include_all_in_appendix": include_all_in_appendix,
            "must_use_existing_results": not force_rerun_analysis_for_plot,
            "force_rerun_analysis_for_plot": force_rerun_analysis_for_plot,
            "notes": text_raw,
        }
        if force_rerun_analysis_for_plot:
            brief["rerun_reason"] = "Visualization request policy: rerun relevant analysis chain before plotting."
        if basis_preference:
            brief["basis_preference"] = basis_preference
            brief["strict_basis"] = strict_basis
        return brief

    def _build_viz_brief_auto_block(self, instruction: str) -> str:
        brief = self._infer_viz_brief(
            instruction,
            default_goal=str(self.state.get("viz_default_goal", "publication") or "publication"),
        )
        self.state["current_viz_brief"] = brief
        return load_prompt(
            "downstream_visualization_brief",
            viz_brief_json=json.dumps(brief, ensure_ascii=False, indent=2),
            user_instruction=str(instruction or "").strip(),
            user_goal=str((self.state.get("user_goal") or {}).get("raw_question", "")).strip(),
        )

    def _augment_downstream_instruction_with_viz(self, instruction: str) -> str:
        if not bool(self.state.get("viz_brief_enabled", True)):
            self.state["current_viz_brief"] = {}
            return str(instruction or "")
        if not self._is_visualization_request(instruction):
            self.state["current_viz_brief"] = {}
            return str(instruction or "")
        block = self._build_viz_brief_auto_block(instruction)
        return f"{str(instruction or '').strip()}\n\n{block}"

    def _augment_downstream_instruction_with_skills(self, instruction: str) -> str:
        return str(instruction or "")

    def _set_planner_needs_input(self, source: str, question: str) -> str:
        clean_question = (question or "").strip() or "Please provide additional information."
        self._emit_event(
            "subagent_needs_input",
            {
                "source": source,
                "question": clean_question,
            },
        )
        return (
            f"{SUBAGENT_NEEDS_INPUT_PREFIX}\n"
            f"source={source}\n"
            f"question={clean_question}"
        )

    def _clear_planner_needs_input(self) -> None:
        if str(self.state.get("planner_phase") or "working") == "needs_input":
            self._emit_event("planner_need_resolved", dict(self.state.get("planner_need") or {}))
        self.state["planner_phase"] = "working"
        self.state["planner_need"] = {}

    def _resolve_pending_path_from_state(self) -> Optional[str]:
        """Resolve pending path from explicit state or latest user question text."""
        pending = self.state.get("pending_input_path")
        if pending:
            return pending

        raw_question = (self.state.get("user_goal") or {}).get("raw_question", "")
        picked = pick_path_from_message(raw_question, cwd=str(Path.cwd()))
        if picked:
            self.state["pending_input_path"] = picked
            return picked
        return None

    def _apply_pending_input_path(self) -> Optional[str]:
        """Switch dataset if a new pending path is available."""
        pending = self._resolve_pending_path_from_state()
        if not pending:
            return None

        pending_path = Path(pending).expanduser().resolve()
        current_raw = self.state.get("input_path")
        current_path = Path(current_raw).expanduser().resolve() if current_raw else None

        self.state["pending_input_path"] = None

        if current_path and current_path == pending_path:
            return None

        if not pending_path.exists():
            msg = f"⚠️ Dataset switch skipped: path does not exist: {pending_path}"
            self.state["data_switch_note"] = msg
            return msg

        from .adata_manager import AnnDataManager

        AnnDataManager().clear()
        self.state["input_path"] = str(pending_path)
        self.state["preprocessed_path"] = ""
        self.state["converted_path"] = None
        self.state["plan_decision"] = {}
        self.state["candidates"] = []
        self.state["execution_plan"] = ""
        self.state["planner_mode_plan_draft"] = ""
        self.state["execution_plan_items"] = []
        self.state["execution_plan_explanation"] = ""
        self.state["execution_plan_updated_at"] = ""
        self.state["downstream_execution_plan"] = ""
        self.state["downstream_plan_draft"] = ""
        self.state["downstream_execution_plan_items"] = []
        self.state["downstream_execution_plan_explanation"] = ""
        self.state["downstream_execution_plan_updated_at"] = ""
        self.state["final_config"] = {}
        self.state["final_metrics"] = {}
        self.state["downstream_results"] = []
        self.state["downstream_summary"] = ""
        self.state["downstream_figures"] = []
        self.state["current_viz_brief"] = {}
        self.state["report_path"] = ""
        self.state["pending_tools"] = []
        self.state["tool_harvest_report"] = {}
        self.state["tool_review_report"] = {}
        self._clear_planner_needs_input()

        switch_note = (
            f"🔄 Dataset switched to: {pending_path}. "
            "Cleared preprocessing/training/downstream cached state."
        )
        self.state["data_switch_note"] = switch_note
        self._emit_event(
            "data_switched",
            {
                "input_path": str(pending_path),
                "note": switch_note,
            },
        )

        return switch_note

    def list_path(
        self,
        path: str = ".",
        recursive: bool = False,
        max_entries: int = 200,
        show_hidden: bool = False,
    ) -> str:
        """List directory/file content for quick path inspection."""
        result = list_path_impl(
            path=path,
            recursive=recursive,
            max_entries=max_entries,
            show_hidden=show_hidden,
        )
        self._emit_file_activity(
            "list_path",
            path=path,
            recursive=recursive,
            max_entries=max_entries,
            show_hidden=show_hidden,
            preview=self._preview_text(result, 800),
        )
        return result

    def read_text_file(
        self,
        path: str,
        start_line: int = 1,
        end_line: Optional[int] = None,
        max_chars: int = 20000,
        encoding: str = "utf-8",
    ) -> str:
        """Read text file content with line range and truncation safeguards."""
        result = read_text_file_impl(
            path=path,
            start_line=start_line,
            end_line=end_line,
            max_chars=max_chars,
            encoding=encoding,
        )
        self._emit_file_activity(
            "read_text_file",
            path=path,
            start_line=start_line,
            end_line=end_line,
            max_chars=max_chars,
            preview=self._preview_text(result),
        )
        return result

    def read_file(
        self,
        file_path: str,
        offset: int = 1,
        limit: Optional[int] = None,
        pages: Optional[str] = None,
        pdf_mode: str = "render",
    ) -> Any:
        """Unified file reader for text, images, and PDFs."""
        multimodal_enabled = bool(self.state.get("enable_multimodal", True))
        suffix = Path(str(file_path or "")).suffix.lower()
        is_image_read = suffix in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"}
        is_pdf_render = suffix == ".pdf" and str(pdf_mode or "render").strip().lower() == "render"
        if not multimodal_enabled and (is_image_read or is_pdf_render):
            if is_pdf_render:
                return (
                    "Error: visual PDF inspection is disabled because multimodal is disabled for this session. "
                    "Use `pdf_mode='text'` for textual extraction or enable multimodal."
                )
            return (
                "Error: image inspection is disabled because multimodal is disabled for this session. "
                "Enable multimodal to read images visually."
            )
        result = read_file_impl(
            file_path=file_path,
            offset=offset,
            limit=limit,
            pages=pages,
            pdf_mode=pdf_mode,
        )
        preview_source = getattr(result, "tool_text", result)
        self._emit_file_activity(
            "read_file",
            path=file_path,
            offset=offset,
            limit=limit,
            pages=pages,
            pdf_mode=pdf_mode,
            preview=self._preview_text(preview_source),
        )
        return result

    def find_files(
        self,
        pattern: str = "*",
        path: str = ".",
        limit: int = 200,
        recursive: bool = True,
        show_hidden: bool = False,
        timeout: int = 15,
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
        self._emit_file_activity(
            "find_files",
            path=path,
            pattern=pattern,
            limit=limit,
            recursive=recursive,
            show_hidden=show_hidden,
            timeout=timeout,
            preview=self._preview_text(result, 1000),
        )
        return result

    def grep_files(
        self,
        pattern: str,
        include: Optional[str] = None,
        path: str = ".",
        limit: int = 100,
        case_sensitive: bool = False,
        timeout: int = 15,
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
        self._emit_file_activity(
            "grep_files",
            path=path,
            pattern=pattern,
            include=include,
            limit=limit,
            case_sensitive=case_sensitive,
            timeout=timeout,
            preview=self._preview_text(result, 1000),
        )
        return result

    def init_training_algorithm_workspace(
        self,
        algorithm_id: str,
        description: str,
        requirements: str = "",
        author: str = "agent",
        overwrite: bool = False,
    ) -> str:
        """Initialize a custom training algorithm workspace under ~/.cellcompass/training_algorithms.

        The workspace includes:
        - manifest.yaml
        - algorithm.py
        - config.yaml (seeded from builtin `vgfm`)
        - README.md
        """
        return self.planner_file_tools.init_training_algorithm_workspace(
            algorithm_id=algorithm_id,
            description=description,
            requirements=requirements,
            author=author,
            overwrite=overwrite,
        )

    def record_decision(
        self,
        algorithm_id: str,
        phase: str,
        decision: str,
        rationale: str,
        status: str = "final",
        alternatives: Optional[List[str]] = None,
        evidence: Optional[List[str]] = None,
    ) -> str:
        entry = self.planner_file_tools.record_decision(
            algorithm_id,
            phase=phase,
            decision=decision,
            alternatives=list(alternatives or []),
            evidence=list(evidence or []),
            rationale=rationale,
            status=status,
            related_artifacts=[],
        )
        return json.dumps(entry, ensure_ascii=False, indent=2)

    def mark_algorithm_failed(
        self,
        algorithm_id: str,
        reason: str,
        evidence: Optional[List[str]] = None,
    ) -> str:
        payload = self.planner_file_tools.mark_algorithm_failed(
            algorithm_id,
            reason=reason,
            evidence=list(evidence or []),
        )
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def mark_result_obsolete(
        self,
        algorithm_id: str,
        artifact_type: str,
        artifact_id: str,
        reason: str,
        replaced_by: str = "",
    ) -> str:
        entry = self.planner_file_tools.mark_result_obsolete(
            algorithm_id,
            artifact_type=artifact_type,
            artifact_id=artifact_id,
            reason=reason,
            replaced_by=replaced_by,
        )
        return json.dumps(entry, ensure_ascii=False, indent=2)

    def list_experiment_history(
        self,
        algorithm_id: str,
        include_obsolete: bool = False,
        limit: int = 10,
        max_field_chars: int = 1200,
    ) -> str:
        return self.planner_file_tools.list_experiment_history(
            algorithm_id,
            include_obsolete=include_obsolete,
            limit=limit,
            max_field_chars=max_field_chars,
        )

    def activate_algorithm_workspace(
        self,
        algorithm_id: str,
        target_snapshot_id: str = "",
        target_proposal_id: str = "",
    ) -> str:
        payload = self.planner_file_tools.activate_algorithm_workspace(
            algorithm_id,
            target_snapshot_id=target_snapshot_id,
            target_proposal_id=target_proposal_id,
        )
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def snapshot_active_algorithm_workspace(self, reason: str = "") -> str:
        snapshot_id = self.planner_file_tools.snapshot_active_algorithm_workspace(reason=reason)
        context = self.planner_file_tools.get_active_algorithm_context()
        payload = {
            "snapshot_id": snapshot_id,
            "active_algorithm_id": str(context.get("algorithm_id") or ""),
            "target_algorithm_id": str(context.get("algorithm_id") or ""),
            "proposal_id": str(context.get("proposal_id") or ""),
            "workspace_path": str(context.get("workspace_path") or ""),
            "dirty_since_snapshot": bool(context.get("dirty_since_snapshot", False)),
            "dirty_paths": list(context.get("dirty_paths") or []),
            "reason": str(reason or "").strip() or "Manual active workspace snapshot",
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def rollback_algorithm_workspace(
        self,
        algorithm_id: str,
        target_snapshot_id: str = "",
        target_proposal_id: str = "",
    ) -> str:
        return self.planner_file_tools.rollback_algorithm_workspace(
            algorithm_id,
            target_snapshot_id=target_snapshot_id,
            target_proposal_id=target_proposal_id,
        )

    def set_active_baseline_run(self, algorithm_id: str, run_id: str, reason: str) -> str:
        payload = self.planner_file_tools.set_active_baseline_run(
            algorithm_id,
            run_id,
            reason,
        )
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def compare_algorithm_runs(
        self,
        algorithm_id: str,
        run_ids: Optional[List[str]] = None,
        include_baseline: bool = True,
    ) -> str:
        return self.planner_file_tools.compare_algorithm_runs(
            algorithm_id,
            run_ids=list(run_ids or []),
            include_baseline=include_baseline,
        )

    def register_algorithm_benchmark_dataset(
        self,
        dataset_id: str,
        source_path: str,
        title: str = "",
        description: str = "",
        tags: Optional[List[str]] = None,
        stage_relevance: Optional[List[str]] = None,
        overwrite: bool = False,
        preprocessing_script_path: str = "",
    ) -> str:
        """Register a prepared .h5ad benchmark and optionally copy its preprocessing script."""
        payload = self.planner_file_tools.register_algorithm_benchmark_dataset(
            dataset_id,
            source_path,
            title=title,
            description=description,
            tags=list(tags or []),
            stage_relevance=list(stage_relevance or []),
            overwrite=bool(overwrite),
            preprocessing_script_path=preprocessing_script_path,
        )
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def register_stage2_simulation_dataset(
        self,
        dataset_id: str,
        source_path: str,
        generator_path: str,
        algorithm_id: str = "",
        proposal_id: str = "",
        simulation_version: str = "",
        claim_metric_name: str = "",
        title: str = "",
        description: str = "",
        tags: Optional[List[str]] = None,
        overwrite: bool = False,
        notes: str = "",
    ) -> str:
        payload = self.planner_file_tools.register_stage2_simulation_dataset(
            dataset_id,
            source_path,
            generator_path,
            algorithm_id=algorithm_id,
            proposal_id=proposal_id,
            simulation_version=simulation_version,
            claim_metric_name=claim_metric_name,
            title=title,
            description=description,
            tags=list(tags or []),
            overwrite=bool(overwrite),
            notes=notes,
        )
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def generate_and_register_stage2_simulation_dataset(
        self,
        dataset_id: str,
        scenario_config_path: str = "",
        scenario_config: Optional[Dict[str, Any]] = None,
        output_id: str = "",
        task_package_name: str = "",
        algorithm_id: str = "",
        proposal_id: str = "",
        simulation_version: str = "",
        claim_metric_name: str = "",
        title: str = "",
        description: str = "",
        tags: Optional[List[str]] = None,
        overwrite: bool = False,
        timeout_seconds: int = 300,
        notes: str = "",
    ) -> str:
        payload = self.planner_file_tools.generate_and_register_stage2_simulation_dataset(
            dataset_id,
            scenario_config_path=scenario_config_path,
            scenario_config=scenario_config if isinstance(scenario_config, dict) else None,
            output_id=output_id,
            task_package_name=task_package_name,
            algorithm_id=algorithm_id,
            proposal_id=proposal_id,
            simulation_version=simulation_version,
            claim_metric_name=claim_metric_name,
            title=title,
            description=description,
            tags=list(tags or []),
            overwrite=bool(overwrite),
            timeout_seconds=int(timeout_seconds or 300),
            notes=notes,
        )
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def list_algorithm_benchmarks(
        self,
        stage: str = "",
        tags: Optional[List[str]] = None,
        ready_only: bool = False,
    ) -> str:
        payload = self.planner_file_tools.list_algorithm_benchmarks(
            stage=stage,
            tags=list(tags or []),
            ready_only=bool(ready_only),
        )
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def get_algorithm_benchmark_dataset(self, dataset_id: str) -> str:
        payload = self.planner_file_tools.get_algorithm_benchmark_dataset(dataset_id)
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def get_algorithm_benchmark_baselines(self, dataset_id: str) -> str:
        payload = self.planner_file_tools.get_algorithm_benchmark_baselines(dataset_id)
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def list_algorithm_benchmark_baselines(self, dataset_id: str) -> str:
        payload = self.planner_file_tools.list_algorithm_benchmark_baselines(dataset_id)
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def record_algorithm_benchmark_baseline(
        self,
        dataset_id: str,
        algorithm_name: str,
        metrics: Dict[str, Any],
        baseline_type: str = "builtin",
        run_id: str = "",
        config_path: str = "",
        notes: str = "",
    ) -> str:
        payload = self.planner_file_tools.record_algorithm_benchmark_baseline(
            dataset_id,
            algorithm_name,
            metrics,
            baseline_type=baseline_type,
            run_id=run_id,
            config_path=config_path,
            notes=notes,
        )
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def make_benchmark_dataset_config(
        self,
        dataset_ids: Optional[List[str]] = None,
        stage: str = "",
        common_config_overrides: Optional[Dict[str, Any]] = None,
        per_dataset_config_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> str:
        payload = self.planner_file_tools.make_benchmark_dataset_config(
            dataset_ids=list(dataset_ids or []),
            stage=stage,
            common_config_overrides=dict(common_config_overrides or {}),
            per_dataset_config_overrides=dict(per_dataset_config_overrides or {}),
        )
        return json.dumps(payload, ensure_ascii=False, indent=2)

    @staticmethod
    def _campaign_claim_metric_run_spec(campaign: Dict[str, Any]) -> Dict[str, Any]:
        spec = dict(campaign.get("claim_metric_spec") or {})
        if not spec:
            return {}
        spec.setdefault("algorithm_id", str(campaign.get("algorithm_id") or "").strip().lower())
        spec.setdefault("campaign_id", str(campaign.get("campaign_id") or ""))
        workspace_path = str(((campaign.get("active_algorithm_context") or {}) or {}).get("workspace_path") or "")
        if workspace_path:
            spec.setdefault("workspace_path", workspace_path)
        return spec

    @staticmethod
    def _claim_metric_requires_campaign_evaluator(stage: str, primary_metric: str) -> bool:
        metric = str(primary_metric or "").strip()
        return (
            str(stage or "").strip() == "stage2_claim_validation"
            and bool(metric)
            and metric not in {"w1_mean", "tmv_mean"}
        )

    @staticmethod
    def _claim_metric_evaluator_identity(info: Any) -> Dict[str, str]:
        payload = info if isinstance(info, dict) else {}
        return {
            "name": str(payload.get("name") or "").strip(),
            "source": str(payload.get("source") or "").strip(),
            "function_name": str(payload.get("function_name") or "").strip(),
            "version": str(payload.get("version") or "").strip(),
            "evaluator_id": str(payload.get("evaluator_id") or "").strip(),
        }

    def _claim_metric_evaluator_infos_match(self, observed: Any, expected: Any) -> bool:
        observed_identity = self._claim_metric_evaluator_identity(observed)
        expected_identity = self._claim_metric_evaluator_identity(expected)
        for key in ("name", "source", "function_name", "version"):
            if observed_identity.get(key) != expected_identity.get(key):
                return False
        expected_id = expected_identity.get("evaluator_id") or ""
        observed_id = observed_identity.get("evaluator_id") or ""
        return not expected_id or observed_id == expected_id

    def _claim_metric_evaluator_matches(
        self,
        metrics: Dict[str, Any],
        expected_info: Dict[str, Any],
        *,
        claim_metric_name: str,
    ) -> bool:
        if not isinstance(metrics, dict) or not isinstance(expected_info, dict) or not expected_info:
            return False
        observed = metrics.get("claim_metric_evaluator")
        if not isinstance(observed, dict):
            return False
        expected_identity = self._claim_metric_evaluator_identity(expected_info)
        if not self._claim_metric_evaluator_infos_match(observed, expected_info):
            return False
        metric_name = str(claim_metric_name or expected_identity.get("name") or "").strip()
        return bool(metric_name and self.planner_file_tools._metric_value(metrics, metric_name) is not None)

    @staticmethod
    def _claim_metric_baseline_adapter_matches(observed_info: Any, expected_info: Any) -> bool:
        observed = observed_info if isinstance(observed_info, dict) else {}
        expected = expected_info if isinstance(expected_info, dict) else {}
        observed_adapter = observed.get("baseline_metric_adapter")
        expected_adapter = expected.get("baseline_metric_adapter")
        if not isinstance(expected_adapter, dict) or not expected_adapter:
            return True
        if not isinstance(observed_adapter, dict) or not observed_adapter:
            return False
        for key in ("baseline_algorithm", "source", "function_name", "version", "supported"):
            if str(observed_adapter.get(key) or "") != str(expected_adapter.get(key) or ""):
                return False
        return True

    def _metric_roles_require_claim_metric(self, metric_roles: Optional[List[str]], claim_metric_name: str) -> bool:
        roles = [
            str(item or "").strip()
            for item in list(metric_roles or ["both"])
            if str(item or "").strip()
        ] or ["both"]
        claim = self.planner_file_tools._normalize_metric_name(claim_metric_name).lower()
        for role in roles:
            role_norm = self.planner_file_tools._normalize_metric_name(role).lower()
            if role_norm in {"both", "claim", "claim_metric"} or (claim and role_norm == claim):
                return True
        return False

    def _attach_campaign_claim_metric_evaluator(self, training_target: Any, claim_metric_spec: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not claim_metric_spec_has_evaluator(claim_metric_spec):
            return {}
        explicit_baseline_algorithm = str(
            (claim_metric_spec or {}).get("_baseline_algorithm_id")
            or (claim_metric_spec or {}).get("baseline_algorithm_id")
            or ""
        ).strip().lower()
        baseline_algorithm = explicit_baseline_algorithm or (
            str(getattr(training_target.spec, "algorithm_id", "") or "").strip().lower()
            if str(getattr(training_target, "training_mode", "") or "") == "builtin"
            else ""
        )
        hook, info = load_campaign_claim_metric_evaluator(
            claim_metric_spec,
            baseline_algorithm=baseline_algorithm,
        )
        if hook is None:
            return {}
        metric_name = str(info.get("name") or (claim_metric_spec or {}).get("name") or "claim_metric")
        training_target.spec.evaluation_metrics_hook = merge_evaluation_metric_hooks(
            training_target.spec.evaluation_metrics_hook,
            hook,
            claim_metric_name=metric_name,
        )
        params = dict(training_target.spec.evaluation_metrics_params or {})
        params["campaign_claim_metric_evaluator"] = dict(info)
        training_target.spec.evaluation_metrics_params = params
        return dict(info)

    def _posthoc_evaluate_claim_metric_for_baseline(
        self,
        metrics: Dict[str, Any],
        *,
        claim_metric_spec: Dict[str, Any],
        stage: str,
        baseline_algorithm: str = "",
        dataset_entry: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not claim_metric_spec_has_evaluator(claim_metric_spec):
            return dict(metrics or {})
        hook, info = load_campaign_claim_metric_evaluator(
            claim_metric_spec,
            baseline_algorithm=str(baseline_algorithm or metrics.get("algorithm_name") or ""),
        )
        if hook is None:
            return dict(metrics or {})

        def _merge_eval_metrics(eval_metrics: Dict[str, Any], *, source: str) -> Dict[str, Any]:
            merged = dict(metrics or {})
            custom = dict(merged.get("custom_metrics") or {})
            if isinstance(eval_metrics.get("custom_metrics"), dict):
                custom.update(dict(eval_metrics.get("custom_metrics") or {}))
            merged["custom_metrics"] = custom
            merged["claim_metric_evaluator"] = dict(info)
            merged["claim_metric_posthoc_evaluated"] = True
            merged["claim_metric_evaluation_source"] = source
            if str(custom.get("claim_metric_support_status") or "") == "unsupported":
                merged["claim_metric_support_status"] = "unsupported"
                merged["claim_metric_unsupported_reason"] = str(custom.get("claim_metric_unsupported_reason") or "")
                merged["claim_metric_baseline_algorithm"] = str(custom.get("claim_metric_baseline_algorithm") or baseline_algorithm or "")
            if eval_metrics.get("custom_metrics_warning"):
                merged["custom_metrics_warning"] = str(eval_metrics.get("custom_metrics_warning"))
            return merged

        def _resolved_config() -> Dict[str, Any]:
            config = metrics.get("config") if isinstance(metrics.get("config"), dict) else {}
            if config:
                return dict(config)
            config_path = str(
                metrics.get("resolved_config_path")
                or ((metrics.get("artifacts") or {}) if isinstance(metrics.get("artifacts"), dict) else {}).get("resolved_config_path")
                or ""
            ).strip()
            if config_path and Path(config_path).is_file():
                with Path(config_path).open("r", encoding="utf-8") as f:
                    loaded = yaml.safe_load(f) or {}
                if isinstance(loaded, dict):
                    return loaded
            return {}

        trajectory_path = str(
            metrics.get("evaluation_trajectory_path")
            or metrics.get("trajectory_path")
            or ((metrics.get("artifacts") or {}) if isinstance(metrics.get("artifacts"), dict) else {}).get("evaluation_trajectory_path")
            or ((metrics.get("artifacts") or {}) if isinstance(metrics.get("artifacts"), dict) else {}).get("trajectory_path")
            or ""
        ).strip()
        if trajectory_path:
            try:
                import anndata as ad  # type: ignore
                import numpy as np
                import torch
                from CytoBridge.tl.fit import resolve_training_data_bundle, resolve_training_device
                from CytoBridge.tl.training_algorithm import EvaluationMetricsContext, EvaluationTimepointResult

                entry = dict(dataset_entry or {})
                adata_path = str(
                    entry.get("adata_path")
                    or metrics.get("input_adata_path")
                    or metrics.get("adata_path")
                    or ""
                ).strip()
                if not adata_path:
                    raise ValueError("trajectory posthoc claim metric requires the benchmark dataset adata_path")
                adata_payload = ad.read_h5ad(Path(adata_path).expanduser())
                config = _resolved_config()
                device = resolve_training_device("cpu")
                training_data = resolve_training_data_bundle(
                    adata_payload,
                    config,
                    stage=stage if stage in {"pilot", "final"} else "final",
                    device=str(device),
                    output_dir=str(Path(trajectory_path).expanduser().parent),
                    metadata={"source": "posthoc_claim_metric_from_saved_trajectory"},
                )
                trajectory = load_evaluation_trajectory_artifact(trajectory_path)
                observed_points = trajectory.observed_points_by_time()
                observed_weights = trajectory.observed_weights_by_time()
                w1_scores = metrics.get("w1_scores") if isinstance(metrics.get("w1_scores"), list) else []
                tmv_scores = metrics.get("tmv_scores") if isinstance(metrics.get("tmv_scores"), list) else []
                timepoint_results = []
                for idx in range(1, min(len(training_data.time_points), len(trajectory.observed_time_indices))):
                    observed = training_data.latent_by_time[idx].detach().cpu().numpy()
                    predicted = np.asarray(observed_points[idx])
                    weights = np.asarray(observed_weights[idx]).reshape(-1)
                    normalized = weights / weights.sum() if weights.sum() > 0 else weights
                    uniform = np.ones(observed.shape[0], dtype=np.float64) / max(1, observed.shape[0])
                    timepoint_results.append(
                        EvaluationTimepointResult(
                            time_index=idx,
                            time_point=float(training_data.time_points[idx]),
                            observed_points=observed,
                            predicted_points=predicted,
                            predicted_weights=weights,
                            normalized_predicted_weights=normalized,
                            observed_uniform_weights=uniform,
                            w1=float(w1_scores[idx - 1]) if idx - 1 < len(w1_scores) else float("nan"),
                            tmv=float(tmv_scores[idx - 1]) if idx - 1 < len(tmv_scores) else float("nan"),
                            trajectory_time_index=int(trajectory.observed_time_indices[idx]),
                            trajectory_time_point=float(trajectory.time_points[trajectory.observed_time_indices[idx]]),
                        )
                    )
                context = EvaluationMetricsContext(
                    adata=adata_payload,
                    training_data=training_data,
                    model=None,
                    config=config,
                    data=list(training_data.latent_by_time),
                    time_points=[float(t) for t in training_data.time_points],
                    builtin_metrics=dict(metrics or {}),
                    metric_params={
                        "campaign_claim_metric_evaluator": dict(info),
                        "baseline_algorithm": str(baseline_algorithm or metrics.get("algorithm_name") or ""),
                        "posthoc_source": "saved_evaluation_trajectory",
                    },
                    simulated_points_by_time=list(observed_points),
                    simulated_weights_by_time=list(observed_weights),
                    evaluation_trajectory=trajectory,
                    trajectory_time_points=list(trajectory.time_points),
                    trajectory_points_by_time=list(trajectory.points_by_time),
                    trajectory_weights_by_time=list(trajectory.weights_by_time),
                    observed_time_indices=list(trajectory.observed_time_indices),
                    timepoint_results=timepoint_results,
                    device=torch.device("cpu"),
                    metadata={
                        "source": "posthoc_claim_metric_from_saved_trajectory",
                        "prediction_contract": "full_t0_to_final_trajectory",
                        "trajectory_source": trajectory.source,
                        "evaluation_trajectory_path": str(trajectory_path),
                    },
                )
                result = hook(context)
                if result is None:
                    result = {}
                if not isinstance(result, dict):
                    raise TypeError(f"claim metric evaluator returned {type(result).__name__}, expected dict or None")
                return _merge_eval_metrics({"custom_metrics": dict(result)}, source="posthoc_trajectory")
            except Exception as exc:
                metrics = dict(metrics or {})
                metrics["claim_metric_trajectory_posthoc_error"] = str(exc)

        artifacts_payload = (metrics.get("artifacts") or {}) if isinstance(metrics.get("artifacts"), dict) else {}
        model_path = str(
            metrics.get("model_artifact_path")
            or artifacts_payload.get("model_artifact_path")
            or metrics.get("trained_model_path")
            or artifacts_payload.get("trained_model_path")
            or ""
        ).strip()
        if not model_path:
            raise ValueError("existing baseline metrics do not include model_artifact_path or trained_model_path")
        model_file = Path(model_path).expanduser()
        if not model_file.is_file():
            raise FileNotFoundError(f"trained baseline model is missing: {model_file}")
        config = metrics.get("config") if isinstance(metrics.get("config"), dict) else {}
        if not config:
            config_path = str(metrics.get("resolved_config_path") or "").strip()
            if config_path and Path(config_path).is_file():
                with Path(config_path).open("r", encoding="utf-8") as f:
                    config = yaml.safe_load(f) or {}
        if not isinstance(config, dict) or not config:
            raise ValueError("cannot posthoc evaluate claim metric without the resolved training config")

        import anndata as ad  # type: ignore
        import torch
        from CytoBridge.tl.fit import resolve_training_data_bundle, resolve_training_device
        from CytoBridge.tl.trainer import TrainingPipeline
        from CytoBridge.utils import load_model_from_adata

        if model_file.suffix.lower() == ".json":
            try:
                manifest = json.loads(model_file.read_text(encoding="utf-8"))
            except Exception:
                manifest = {}
            if str(manifest.get("artifact_type") or "") == "cytobridge_model_artifact":
                reference_path = Path(str(manifest.get("reference_adata_path") or manifest.get("input_adata_path") or "")).expanduser()
                state_path = Path(str(manifest.get("model_state_path") or "")).expanduser()
                if not reference_path.is_file():
                    raise FileNotFoundError(f"model artifact reference AnnData is missing: {reference_path}")
                if not state_path.is_file():
                    raise FileNotFoundError(f"model artifact state file is missing: {state_path}")
                adata_trained = ad.read_h5ad(reference_path)
                state_payload = torch.load(state_path, map_location="cpu", weights_only=False)
                if not isinstance(state_payload, dict) or not isinstance(state_payload.get("all_model"), dict):
                    raise ValueError(f"invalid model artifact state payload: {state_path}")
                adata_trained.uns["all_model"] = state_payload["all_model"]
                if "training_summary" in state_payload:
                    adata_trained.uns["training_summary"] = state_payload["training_summary"]
            else:
                adata_trained = ad.read_h5ad(model_file)
        else:
            adata_trained = ad.read_h5ad(model_file)
        device = resolve_training_device("cpu")
        training_data = resolve_training_data_bundle(
            adata_trained,
            config,
            stage=stage if stage in {"pilot", "final"} else "final",
            device=str(device),
            output_dir=str(Path(model_file).parent),
            metadata={"source": "posthoc_claim_metric_evaluation"},
        )
        model = load_model_from_adata(adata_trained).to(device)
        batch_size = min(min(int(x.shape[0]) for x in training_data.latent_by_time), int((config.get("training") or {}).get("batch_size") or 256))
        trainer = TrainingPipeline(
            model,
            config,
            batch_size,
            device,
            training_data=training_data,
            evaluation_metrics_hook=hook,
            evaluation_metrics_params={"campaign_claim_metric_evaluator": dict(info)},
        )
        eval_metrics = trainer.evaluate(adata_trained)
        return _merge_eval_metrics(eval_metrics, source="posthoc_model")

    def start_algorithm_campaign(
        self,
        algorithm_id: str,
        proposal_id: str = "",
        claim_metric_spec: Optional[Dict[str, Any]] = None,
    ) -> str:
        payload = self.planner_file_tools.start_algorithm_campaign(
            algorithm_id,
            proposal_id=proposal_id,
            claim_metric_spec=claim_metric_spec or {},
        )
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def update_campaign_claim_metric_spec(
        self,
        campaign_id: str = "",
        claim_metric_spec: Optional[Dict[str, Any]] = None,
        reason: str = "",
    ) -> str:
        campaign_id = str(campaign_id or self.state.get("active_algorithm_campaign_id") or "").strip()
        payload = self.planner_file_tools.update_campaign_claim_metric_spec(
            campaign_id,
            claim_metric_spec=dict(claim_metric_spec or {}),
            reason=reason,
        )
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def set_campaign_stage_panel(
        self,
        campaign_id: str = "",
        stage: str = "",
        dataset_config_overrides: Optional[Dict[str, Any]] = None,
        overwrite: bool = False,
    ) -> str:
        campaign_id = str(campaign_id or self.state.get("active_algorithm_campaign_id") or "").strip()
        payload = self.planner_file_tools.set_campaign_stage_panel(
            campaign_id,
            stage=stage,
            dataset_config_overrides=dict(dataset_config_overrides or {}),
            overwrite=bool(overwrite),
        )
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def switch_campaign_stage_panel(
        self,
        campaign_id: str = "",
        stage: str = "",
        dataset_config_overrides: Optional[Dict[str, Any]] = None,
        reason: str = "",
    ) -> str:
        campaign_id = str(campaign_id or self.state.get("active_algorithm_campaign_id") or "").strip()
        payload = self.planner_file_tools.switch_campaign_stage_panel(
            campaign_id,
            stage=stage,
            dataset_config_overrides=dict(dataset_config_overrides or {}),
            reason=reason,
        )
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def refresh_campaign_stage_baselines(
        self,
        campaign_id: str = "",
        stage: str = "",
        baseline_algorithms: Optional[List[str]] = None,
        run_missing: bool = True,
        overwrite_existing: bool = False,
        baseline_type: str = "builtin",
    ) -> str:
        campaign_id = str(campaign_id or self.state.get("active_algorithm_campaign_id") or "").strip()
        campaign = self.planner_file_tools.get_algorithm_campaign_status(campaign_id=campaign_id)
        campaign_status = str(campaign.get("status") or "").strip().lower()
        if campaign_status in {"failed", "locked", "complete", "completed", "aborted", "archived", "inactive"}:
            return json.dumps(
                {
                    "status": "blocked",
                    "reason": "campaign_not_mutable",
                    "campaign_id": campaign_id,
                    "campaign_status": campaign_status,
                    "message": (
                        "Refusing to refresh baselines for a failed/locked historical campaign. "
                        "Continue the active sibling campaign or start a clean campaign instead."
                    ),
                },
                ensure_ascii=False,
                indent=2,
            )
        target_stage = str(stage or campaign.get("current_stage") or "stage1_feasibility").strip()
        stages = campaign.get("stages") if isinstance(campaign.get("stages"), dict) else {}
        stage_state = dict(stages.get(target_stage) or {})
        if not stage_state:
            return json.dumps(
                {"status": "blocked", "reason": "unknown_stage", "campaign_id": campaign_id, "stage": target_stage},
                ensure_ascii=False,
                indent=2,
            )
        stage_panel = stage_state.get("stage_panel") if isinstance(stage_state.get("stage_panel"), dict) else {}
        panel_payload = dict(stage_panel.get("dataset_config_overrides") or {})
        target_dataset_ids = self.planner_file_tools._campaign_target_dataset_ids(panel_payload)
        if not target_dataset_ids:
            return json.dumps(
                {
                    "status": "blocked",
                    "reason": "stage_panel_required",
                    "campaign_id": campaign_id,
                    "stage": target_stage,
                    "message": "Freeze the stage data panel before refreshing baselines.",
                },
                ensure_ascii=False,
                indent=2,
            )

        policy = self.planner_file_tools._campaign_stage_policy(campaign, target_stage)
        primary_metric = str(policy.get("primary_metric") or "w1_mean")
        primary_direction = str(policy.get("primary_direction") or "lower")
        claim_metric_run_spec = self._campaign_claim_metric_run_spec(campaign)
        formal_holdout_claim_spec = self._holdout_time_eval_spec_from_claim_metric(claim_metric_run_spec)
        strict_claim_metric = self._claim_metric_requires_campaign_evaluator(target_stage, primary_metric)
        if strict_claim_metric and not claim_metric_spec_has_evaluator(claim_metric_run_spec):
            return json.dumps(
                {
                    "status": "blocked",
                    "reason": "claim_metric_evaluator_required",
                    "campaign_id": campaign_id,
                    "stage": target_stage,
                    "primary_metric": primary_metric,
                    "message": (
                        "Stage 2 claim metrics must be computed through one campaign evaluator for both "
                        "candidate and baselines. Add claim_metric_spec.evaluator_path before refreshing baselines."
                    ),
                    "next_required_tool": "update_campaign_claim_metric_spec",
                    "required_claim_metric_spec_fields": [
                        "name",
                        "direction",
                        "evaluator_path",
                        "evaluator_function",
                    ],
                    "repair_instruction": (
                        "Call update_campaign_claim_metric_spec with the authoritative evaluator_path. "
                        "Do not patch campaign.json directly and do not pass evaluator_path through "
                        "dataset_config_overrides."
                    ),
                },
                ensure_ascii=False,
                indent=2,
            )
        baseline_kind = str(baseline_type or "builtin").strip().lower()
        if baseline_kind not in {"builtin", "reference"}:
            return json.dumps(
                {"status": "blocked", "reason": "invalid_baseline_type", "baseline_type": baseline_type},
                ensure_ascii=False,
                indent=2,
            )

        requested_algorithms = [
            self._normalize_baseline_algorithm_name(item)
            for item in list(baseline_algorithms or [])
            if str(item or "").strip()
        ]
        requested_algorithms = list(dict.fromkeys([item for item in requested_algorithms if item]))
        campaign_claim_spec = campaign.get("claim_metric_spec") if isinstance(campaign.get("claim_metric_spec"), dict) else {}
        baseline_policy = self.planner_file_tools._normalize_campaign_baseline_selection_policy(
            campaign.get("baseline_selection_policy")
            or campaign_claim_spec.get("baseline_selection_policy")
            or DEFAULT_CAMPAIGN_BASELINE_SELECTION_POLICY
        )
        strict_gate_algorithms: List[str] = []
        if baseline_kind == "builtin" and baseline_policy == "strict_all_builtin":
            strict_gate_algorithms = list(DEFAULT_CAMPAIGN_BUILTIN_BASELINES)
        if requested_algorithms:
            refresh_algorithms = list(requested_algorithms)
        else:
            refresh_algorithms = list(DEFAULT_CAMPAIGN_BUILTIN_BASELINES)
        refresh_algorithms = list(dict.fromkeys([item for item in refresh_algorithms if item]))
        required_anchor_baseline = ""
        anchor_baseline_added = False
        if baseline_kind == "builtin":
            required_anchor_baseline = self._implementation_map_required_anchor_baseline(
                str(campaign.get("algorithm_id") or "")
            )
            if required_anchor_baseline and required_anchor_baseline not in strict_gate_algorithms:
                strict_gate_algorithms.append(required_anchor_baseline)
            if (
                required_anchor_baseline
                and required_anchor_baseline not in refresh_algorithms
                and not (baseline_policy == "strict_all_builtin" and requested_algorithms)
            ):
                refresh_algorithms.append(required_anchor_baseline)
                anchor_baseline_added = True
        summary_algorithms = list(
            dict.fromkeys(
                [
                    item
                    for item in (
                        strict_gate_algorithms
                        if baseline_kind == "builtin" and baseline_policy == "strict_all_builtin"
                        else refresh_algorithms
                    )
                    if item
                ]
            )
        )
        if not refresh_algorithms and not summary_algorithms:
            return json.dumps(
                {"status": "blocked", "reason": "no_baseline_algorithms", "campaign_id": campaign_id, "stage": target_stage},
                ensure_ascii=False,
                indent=2,
            )

        existing_baseline_metrics = (
            dict(stage_state.get("external_baseline_metrics") or {})
            if isinstance(stage_state.get("external_baseline_metrics"), dict)
            else {}
        )
        existing_strict_audit = (
            dict(existing_baseline_metrics.get("strict_baseline_audit") or {})
            if isinstance(existing_baseline_metrics.get("strict_baseline_audit"), dict)
            else {}
        )
        existing_audit_ledger = [
            dict(item)
            for item in list(existing_baseline_metrics.get("strict_baseline_audit_ledger") or [])
            if isinstance(item, dict)
        ]
        existing_missing_records = [
            item for item in existing_audit_ledger if bool(item.get("missing_required_record"))
        ]
        existing_ledger_count = int(existing_strict_audit.get("ledger_count") or len(existing_audit_ledger))
        existing_expected_count = len(strict_gate_algorithms)
        active_best_trial_id = str(stage_state.get("active_best_trial_id") or "")
        baseline_panel_ids = self.planner_file_tools._campaign_target_dataset_ids(existing_baseline_metrics)
        baseline_panel_matches = bool(baseline_panel_ids == target_dataset_ids)
        expected_claim_evaluator: Dict[str, Any] = {}
        if strict_claim_metric:
            try:
                _, expected_claim_evaluator = load_campaign_claim_metric_evaluator(claim_metric_run_spec)
            except Exception:
                expected_claim_evaluator = {}
        comparable_audit_rows = [
            item for item in existing_audit_ledger if bool(item.get("comparable"))
        ]
        rows_with_evaluator = [
            item for item in comparable_audit_rows if isinstance(item.get("claim_metric_evaluator"), dict)
        ]
        if not strict_claim_metric:
            baseline_evaluator_current = True
        elif expected_claim_evaluator and rows_with_evaluator:
            baseline_evaluator_current = bool(
                len(rows_with_evaluator) == len(comparable_audit_rows)
                and all(
                    self._claim_metric_evaluator_infos_match(
                        item.get("claim_metric_evaluator"),
                        expected_claim_evaluator,
                    )
                    for item in rows_with_evaluator
                )
            )
        else:
            selected_claim_baseline = (
                existing_baseline_metrics.get("claim_sota_baseline_metrics")
                if isinstance(existing_baseline_metrics.get("claim_sota_baseline_metrics"), dict)
                else existing_baseline_metrics
            )
            baseline_evaluator_current = bool(
                expected_claim_evaluator
                and self._claim_metric_evaluator_infos_match(
                    selected_claim_baseline.get("claim_metric_evaluator"),
                    expected_claim_evaluator,
                )
            )
        if (
            baseline_kind == "builtin"
            and baseline_policy == "strict_all_builtin"
            and not bool(run_missing)
            and not existing_missing_records
            and existing_expected_count > 0
            and existing_ledger_count == existing_expected_count
            and baseline_panel_matches
            and baseline_evaluator_current
        ):
            nonblocking_failed = [
                item for item in existing_audit_ledger if bool(item.get("nonblocking_failed_record"))
            ]
            return json.dumps(
                {
                    "status": "already_complete",
                    "changed": False,
                    "reason": "strict_baseline_audit_complete_for_frozen_panel",
                    "campaign_id": campaign_id,
                    "stage": target_stage,
                    "active_best_trial_id": active_best_trial_id,
                    "strict_baseline_audit": {
                        "ok": True,
                        "ledger_count": existing_ledger_count,
                        "expected_ledger_count": existing_expected_count,
                        "missing_required_records": [],
                        "nonblocking_failed_algorithms": [
                            str(item.get("algorithm_name") or "")
                            for item in nonblocking_failed
                            if isinstance(item, dict) and str(item.get("algorithm_name") or "").strip()
                        ],
                    },
                    "message": (
                        "The strict builtin ledger is already complete for the frozen panel and current claim evaluator. "
                        "No baseline records were refreshed and campaign state was not mutated. "
                        "Use query_campaign_baseline_metrics for read-only inspection. overwrite_existing is ignored "
                        "when run_missing=false and no persisted missing/stale evidence exists."
                    ),
                },
                ensure_ascii=False,
                indent=2,
            )

        raw_datasets = panel_payload.get("datasets")
        panel_entries = raw_datasets if isinstance(raw_datasets, list) else []
        entries_by_id: Dict[str, Dict[str, Any]] = {}
        for idx, item in enumerate(panel_entries):
            entry = dict(item or {})
            dataset_id = str(entry.get("dataset_id") or entry.get("id") or f"dataset{idx + 1}").strip()
            if dataset_id:
                entries_by_id[dataset_id] = entry

        dataset_entries: List[Dict[str, Any]] = []
        for dataset_id in target_dataset_ids:
            entry = dict(entries_by_id.get(dataset_id) or {})
            if not entry:
                card = self.planner_file_tools.get_algorithm_benchmark_dataset(dataset_id)
                entry = dict((card.get("usage") or {}).get("campaign_dataset_entry") or {})
                entry.setdefault("adata_path", str((card.get("paths") or {}).get("data_path") or ""))
            entry["dataset_id"] = dataset_id
            if not str(entry.get("adata_path") or "").strip():
                card = self.planner_file_tools.get_algorithm_benchmark_dataset(dataset_id)
                entry["adata_path"] = str((card.get("paths") or {}).get("data_path") or "")
            dataset_entries.append(entry)

        common_overrides = panel_payload.get("config_overrides") if isinstance(panel_payload.get("config_overrides"), dict) else {}

        def _merge_overrides(base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
            merged = deepcopy(base)
            for key, value in extra.items():
                if key in {"dataset_id", "id", "adata_path", "target_dataset_ids", "config_overrides", "simulation_version", "simulation_generator"}:
                    continue
                if isinstance(value, dict) and isinstance(merged.get(key), dict):
                    merged[key] = _merge_overrides(dict(merged.get(key) or {}), value)
                else:
                    merged[key] = deepcopy(value)
            return merged

        def _baseline_config_overrides_for_entry(entry: Dict[str, Any], dataset_id: str, algorithm_name: str) -> Dict[str, Any]:
            entry_overrides = entry.get("config_overrides") if isinstance(entry.get("config_overrides"), dict) else entry
            config_overrides = _merge_overrides(dict(common_overrides or {}), dict(entry_overrides or {}))
            if baseline_kind == "builtin":
                benchmark_baseline_config = self.planner_file_tools.get_algorithm_benchmark_baseline_config(
                    dataset_id,
                    algorithm_name,
                    baseline_type=baseline_kind,
                )
                benchmark_overrides = (
                    benchmark_baseline_config.get("config_overrides")
                    if isinstance(benchmark_baseline_config, dict)
                    and isinstance(benchmark_baseline_config.get("config_overrides"), dict)
                    else {}
                )
                if benchmark_overrides:
                    config_overrides = _merge_overrides(config_overrides, dict(benchmark_overrides or {}))
            return config_overrides

        def _apply_holdout_claim_to_metrics(
            metrics: Dict[str, Any],
            report: Dict[str, Any],
            holdout_spec: Dict[str, Any],
        ) -> Dict[str, Any]:
            out = dict(metrics or {})
            out["holdout_time_evaluation"] = dict(report or {})
            normalized = self._normalize_holdout_time_eval_spec(holdout_spec)
            if normalized["attach_to_custom_metrics"] and report.get("mean_w1") is not None:
                custom = out.get("custom_metrics")
                if not isinstance(custom, dict):
                    custom = {}
                custom[normalized["metric_name"]] = float(report["mean_w1"])
                out["custom_metrics"] = custom
                evaluator_info = normalized.get("claim_metric_evaluator")
                if isinstance(evaluator_info, dict) and evaluator_info:
                    out["claim_metric_evaluator"] = dict(evaluator_info)
                    out["claim_metric_posthoc_evaluated"] = True
                    out["claim_metric_evaluation_source"] = "holdout_time_auxiliary_split"
            return out

        def _holdout_claim_unsupported_reason(report: Any) -> str:
            if not isinstance(report, dict) or not report:
                return ""
            if self.planner_file_tools._numeric_value(report.get("mean_w1")) is not None:
                return ""
            status = str(report.get("status") or "").strip().lower()
            reasons: List[str] = []
            report_reason = str(report.get("reason") or "").strip()
            if report_reason:
                reasons.append(report_reason)
            for group in list(report.get("group_reports") or []):
                if not isinstance(group, dict):
                    continue
                aux_dir = Path(str(group.get("auxiliary_run_dir") or "")).expanduser()
                if aux_dir.is_dir() and (aux_dir / "artifacts" / "evaluation_trajectory.npz").is_file():
                    return ""
                group_status = str(group.get("status") or "").strip().lower()
                group_reason = str(group.get("reason") or "").strip()
                if group_reason and group_status in {"failed", "partial", "skipped"}:
                    reasons.append(group_reason)
            reasons = list(dict.fromkeys(reasons))
            if status not in {"failed", "partial", "skipped"} and not reasons:
                return ""
            detail = "; ".join(reasons[:6]) if reasons else f"holdout_status={status}"
            return (
                "holdout_time_w1 baseline unsupported: auxiliary hold-out evaluation "
                f"did not produce a finite mean_w1 ({detail})"
            )

        def _mark_holdout_claim_unsupported(
            metrics: Dict[str, Any],
            report: Any,
        ) -> Dict[str, Any]:
            reason = _holdout_claim_unsupported_reason(report)
            out = dict(metrics or {})
            if not reason:
                return out
            if isinstance(report, dict):
                out["holdout_time_evaluation"] = dict(report)
            out["claim_metric_support_status"] = "unsupported"
            out["claim_metric_unsupported_reason"] = reason
            out["claim_metric_posthoc_error"] = reason
            out["baseline_unusable"] = True
            out["baseline_unusable_reason"] = reason
            return out

        def _repair_holdout_claim_from_saved_group_trajectories(
            metrics: Dict[str, Any],
            holdout_spec: Dict[str, Any],
            full_adata_path: str,
        ) -> Tuple[Dict[str, Any], bool]:
            report = metrics.get("holdout_time_evaluation") if isinstance(metrics, dict) else None
            if not isinstance(report, dict) or self.planner_file_tools._numeric_value(report.get("mean_w1")) is not None:
                return dict(metrics or {}), False
            group_reports = [item for item in list(report.get("group_reports") or []) if isinstance(item, dict)]
            if not group_reports:
                return dict(metrics or {}), False
            full_path = str(full_adata_path or metrics.get("adata_path") or metrics.get("input_adata_path") or "").strip()
            if not full_path or not Path(full_path).expanduser().is_file():
                return dict(metrics or {}), False
            normalized = self._normalize_holdout_time_eval_spec(holdout_spec)
            try:
                import anndata as ad  # type: ignore

                full_adata = ad.read_h5ad(full_path)
                time_key = self._detect_time_key(full_adata, normalized["time_key"])
                if not time_key:
                    return dict(metrics or {}), False
            except Exception:
                logger.debug("Failed to inspect full AnnData for saved holdout trajectory repair.", exc_info=True)
                return dict(metrics or {}), False
            repaired_groups: List[Dict[str, Any]] = []
            repaired_any = False
            for group in group_reports:
                repaired_group = dict(group)
                trajectory_path = str(
                    repaired_group.get("trajectory_path")
                    or repaired_group.get("evaluation_trajectory_path")
                    or ""
                ).strip()
                if not trajectory_path:
                    aux_dir = Path(str(repaired_group.get("auxiliary_run_dir") or "")).expanduser()
                    fallback = aux_dir / "artifacts" / "evaluation_trajectory.npz"
                    if fallback.is_file():
                        trajectory_path = str(fallback)
                heldout = [
                    float(value)
                    for value in list(repaired_group.get("heldout_time_points") or [])
                    if self.planner_file_tools._numeric_value(value) is not None
                ]
                if trajectory_path and heldout and Path(trajectory_path).expanduser().is_file():
                    try:
                        eval_report = self._compute_holdout_w1_from_trajectory(
                            trajectory_path=trajectory_path,
                            full_adata_path=full_path,
                            heldout_times=heldout,
                            time_key=time_key,
                            latent_key=normalized["latent_key"],
                            max_trajectory_time_delta=normalized["max_trajectory_time_delta"],
                        )
                        repaired_group.update(eval_report)
                        repaired_group["trajectory_path"] = trajectory_path
                        repaired_group["repaired_from_saved_auxiliary_trajectory"] = True
                        repaired_any = True
                        group_dir = Path(str(repaired_group.get("group_dir") or "")).expanduser()
                        if group_dir:
                            try:
                                group_dir.mkdir(parents=True, exist_ok=True)
                                (group_dir / "metrics.json").write_text(
                                    json.dumps(repaired_group, ensure_ascii=False, indent=2),
                                    encoding="utf-8",
                                )
                            except Exception:
                                logger.debug("Failed to persist repaired holdout group metrics.", exc_info=True)
                    except Exception:
                        logger.debug("Failed to repair holdout group from saved trajectory.", exc_info=True)
                repaired_groups.append(repaired_group)
            if not repaired_any:
                return dict(metrics or {}), False
            repaired_report = dict(report)
            repaired_report["group_reports"] = repaired_groups
            statuses = {str(item.get("status") or "") for item in repaired_groups}
            repaired_report["status"] = "ok" if statuses == {"ok"} else "partial"
            repaired_report["mean_w1"] = self._mean_finite([item.get("mean_w1") for item in repaired_groups])
            repaired_report["mean_tmv"] = self._mean_finite([item.get("mean_tmv") for item in repaired_groups])
            repaired_report["repaired_from_saved_auxiliary_trajectories"] = True
            repaired_report["reason"] = "" if repaired_report["mean_w1"] is not None else str(repaired_report.get("reason") or "")
            report_path = Path(str(repaired_report.get("report_path") or "")).expanduser()
            if report_path:
                try:
                    report_path.parent.mkdir(parents=True, exist_ok=True)
                    report_path.write_text(json.dumps(repaired_report, ensure_ascii=False, indent=2), encoding="utf-8")
                except Exception:
                    logger.debug("Failed to persist repaired holdout report.", exc_info=True)
            return _apply_holdout_claim_to_metrics(dict(metrics or {}), repaired_report, holdout_spec), True

        def _hydrate_holdout_report_from_saved_metrics(
            metrics: Dict[str, Any],
            existing_record: Dict[str, Any],
        ) -> Dict[str, Any]:
            out = dict(metrics or {})
            if isinstance(out.get("holdout_time_evaluation"), dict):
                return out
            candidate_paths: List[Path] = []
            for raw_path in (
                out.get("metrics_path"),
                (out.get("artifacts") or {}).get("metrics_path") if isinstance(out.get("artifacts"), dict) else "",
                (existing_record.get("artifacts") or {}).get("metrics_path") if isinstance(existing_record.get("artifacts"), dict) else "",
            ):
                raw_text = str(raw_path or "").strip()
                if raw_text:
                    candidate_paths.append(Path(raw_text).expanduser())
            for raw_dir in (
                out.get("run_dir"),
                (out.get("artifacts") or {}).get("run_dir") if isinstance(out.get("artifacts"), dict) else "",
                (existing_record.get("artifacts") or {}).get("run_dir") if isinstance(existing_record.get("artifacts"), dict) else "",
            ):
                raw_text = str(raw_dir or "").strip()
                if raw_text:
                    candidate_paths.append(Path(raw_text).expanduser() / "artifacts" / "metrics.json")
            for path in candidate_paths:
                try:
                    if not path.is_file():
                        continue
                    saved = json.loads(path.read_text(encoding="utf-8"))
                    if not isinstance(saved, dict) or not isinstance(saved.get("holdout_time_evaluation"), dict):
                        continue
                    out["holdout_time_evaluation"] = dict(saved.get("holdout_time_evaluation") or {})
                    if str(saved.get("evaluation_trajectory_path") or "").strip():
                        out.setdefault("evaluation_trajectory_path", str(saved.get("evaluation_trajectory_path") or "").strip())
                    saved_artifacts = saved.get("artifacts")
                    if isinstance(saved_artifacts, dict):
                        artifacts = out.get("artifacts")
                        if not isinstance(artifacts, dict):
                            artifacts = {}
                        for key, value in saved_artifacts.items():
                            artifacts.setdefault(str(key), value)
                        out["artifacts"] = artifacts
                    out["holdout_time_evaluation_hydrated_from"] = str(path)
                    return out
                except Exception:
                    logger.debug("Failed to hydrate saved holdout evaluation from %s", path, exc_info=True)
            return out

        ignored_corrupt_records: List[Dict[str, Any]] = []

        def _find_existing_baseline(dataset_id: str, algorithm_name: str) -> Dict[str, Any]:
            records = self.planner_file_tools.get_algorithm_benchmark_baselines(dataset_id).get("baseline_records") or []
            for record in records:
                if (
                    str((record or {}).get("algorithm_name") or "") == algorithm_name
                    and str((record or {}).get("baseline_type") or "") == baseline_kind
                ):
                    issue = self.planner_file_tools._baseline_identity_mismatch(
                        algorithm_name,
                        dict((record or {}).get("metrics") or {}),
                    )
                    if issue:
                        ignored_corrupt_records.append(
                            {
                                "dataset_id": dataset_id,
                                "algorithm_name": algorithm_name,
                                "record_path": str((record or {}).get("record_path") or ""),
                                "reason": issue,
                            }
                        )
                        continue
                    return dict(record or {})
            return {}

        def _baseline_adapter_config(algorithm_name: str) -> Dict[str, Any]:
            raw_map = (
                claim_metric_run_spec.get("baseline_metric_adapters")
                or claim_metric_run_spec.get("baseline_metric_adaptors")
                or claim_metric_run_spec.get("baseline_adapters")
                or claim_metric_run_spec.get("baseline_adaptors")
                or {}
            )
            if not isinstance(raw_map, dict):
                return {}
            normalized = self._normalize_baseline_algorithm_name(algorithm_name)
            for key in (algorithm_name, normalized, "*", "default"):
                raw = raw_map.get(key)
                if isinstance(raw, dict):
                    return dict(raw)
            return {}

        def _materialize_adapter_metrics(metrics: Dict[str, Any], algorithm_name: str) -> Dict[str, Any]:
            if target_stage != "stage2_claim_validation":
                return dict(metrics or {})
            adapter = _baseline_adapter_config(algorithm_name)
            if not adapter or bool(adapter.get("supported", True)) is False:
                return dict(metrics or {})
            def _has_verified_static_value_evidence(metric_name: str) -> bool:
                verified = any(
                    bool(adapter.get(key))
                    for key in (
                        "verified",
                        "static_value_verified",
                        "static_metric_verified",
                        f"{metric_name}_verified",
                    )
                )
                if not verified:
                    return False
                evidence_keys = (
                    "evidence",
                    "evidence_path",
                    "record_path",
                    "artifact_path",
                    "source_run_id",
                    "source_trial_id",
                    "source",
                    "verified_by",
                    "verification_note",
                )
                return any(str(adapter.get(key) or "").strip() for key in evidence_keys)

            metric_names = [
                str(primary_metric or "").strip(),
                str((claim_metric_run_spec or {}).get("name") or "").strip(),
            ]
            out = dict(metrics or {})
            custom = dict(out.get("custom_metrics") or {})
            materialized: List[str] = []
            unverified: List[str] = []
            for metric_name in dict.fromkeys(name for name in metric_names if name):
                if metric_name in {"w1_mean", "tmv_mean"} or metric_name not in adapter:
                    continue
                value = self.planner_file_tools._numeric_value(adapter.get(metric_name))
                if value is None:
                    continue
                if strict_claim_metric:
                    out["baseline_unusable"] = True
                    out["baseline_unusable_reason"] = (
                        "Stage 2 claim metric uses a numeric baseline_metric_adapter value instead of the "
                        "campaign evaluator; recompute from the saved baseline trajectory/model or declare unsupported."
                    )
                    out["claim_metric_static_adapter_rejected"] = metric_name
                    return out
                if not _has_verified_static_value_evidence(metric_name):
                    unverified.append(metric_name)
                    continue
                custom[metric_name] = value
                out.setdefault(metric_name, value)
                materialized.append(metric_name)
            if materialized:
                out["custom_metrics"] = custom
                out["baseline_metric_adapter_materialized"] = materialized
            if unverified:
                out["claim_metric_adapter_verification_error"] = (
                    "numeric baseline_metric_adapters used as static Stage 2 claim metrics must declare "
                    "`verified: true` and provide evidence/source/run/artifact provenance"
                )
                out["claim_metric_adapter_unverified_metrics"] = unverified
            return out

        def _baseline_rejected_reason(existing: Dict[str, Any], metrics: Dict[str, Any]) -> str:
            for payload in (metrics.get("run_verdict"), existing.get("run_verdict")):
                if not isinstance(payload, dict):
                    continue
                status = str(payload.get("status") or "").strip().lower()
                if status == "rejected":
                    reasons = payload.get("reasons")
                    if isinstance(reasons, list) and reasons:
                        return "; ".join(str(item) for item in reasons if str(item).strip())
                    return "baseline run verdict is rejected"
            return ""

        def _baseline_rejection_is_quality_diagnostic_only(reason: str) -> bool:
            parts = [
                str(item or "").strip()
                for item in str(reason or "").split(";")
                if str(item or "").strip()
            ]
            if not parts:
                return False
            diagnostic_prefixes = (
                "TMV quality gate failed:",
                "TMV recorded as a diagnostic only",
            )
            return all(any(part.startswith(prefix) for prefix in diagnostic_prefixes) for part in parts)

        def _expected_claim_metric_info(algorithm_name: str) -> Dict[str, Any]:
            if not strict_claim_metric or not claim_metric_spec_has_evaluator(claim_metric_run_spec):
                return {}
            try:
                _, info = load_campaign_claim_metric_evaluator(
                    claim_metric_run_spec,
                    baseline_algorithm=algorithm_name,
                )
                return dict(info or {})
            except Exception:
                return {}

        def _claim_metric_provenance_ok(metrics: Dict[str, Any], algorithm_name: str) -> bool:
            if not strict_claim_metric:
                return True
            expected = _expected_claim_metric_info(algorithm_name)
            return self._claim_metric_evaluator_matches(
                metrics,
                expected,
                claim_metric_name=primary_metric,
            )

        def _aggregate_metrics(items: List[Dict[str, Any]], algorithm_name: str, run_ids: List[str]) -> Dict[str, Any]:
            aggregate: Dict[str, Any] = {
                "algorithm_name": algorithm_name,
                "baseline_type": baseline_kind,
                "target_dataset_ids": list(target_dataset_ids),
                "baseline_run_ids": list(run_ids),
                "campaign_dataset_metrics": items,
            }
            w1_scores: List[Any] = []
            tmv_scores: List[Any] = []
            runtime_sec = 0.0
            custom_values: Dict[str, List[float]] = {}
            claim_metric_evaluators: List[Dict[str, Any]] = []
            simulation_versions: List[str] = []
            errors: List[str] = []
            unusable_reasons: List[str] = []
            quality_diagnostic_reasons: List[str] = []
            direct_claim_metric_names = [
                str(primary_metric or "").strip(),
                str(policy.get("secondary_metric") or "").strip(),
                str((claim_metric_run_spec or {}).get("name") or "").strip(),
            ]
            for item in items:
                if str(item.get("error") or "").strip():
                    errors.append(str(item.get("error") or "").strip())
                if bool(item.get("baseline_unusable")):
                    reason = str(item.get("baseline_unusable_reason") or "baseline is not eligible").strip()
                    if reason:
                        unusable_reasons.append(reason)
                quality_reason = str(item.get("baseline_quality_diagnostic_reason") or "").strip()
                if quality_reason:
                    quality_diagnostic_reasons.append(quality_reason)
                run_verdict = item.get("run_verdict")
                if isinstance(run_verdict, dict):
                    verdict_reasons = [
                        str(reason or "").strip()
                        for reason in list(run_verdict.get("reasons") or [])
                        if str(reason or "").strip()
                    ]
                    if (
                        str(run_verdict.get("status") or "").strip().lower() == "rejected"
                        and verdict_reasons
                        and _baseline_rejection_is_quality_diagnostic_only("; ".join(verdict_reasons))
                    ):
                        quality_diagnostic_reasons.extend(verdict_reasons)
                if isinstance(item.get("w1_scores"), list):
                    w1_scores.extend(item.get("w1_scores") or [])
                elif item.get("w1_mean") is not None:
                    w1_scores.append(item.get("w1_mean"))
                if isinstance(item.get("tmv_scores"), list):
                    tmv_scores.extend(item.get("tmv_scores") or [])
                elif item.get("tmv_mean") is not None:
                    tmv_scores.append(item.get("tmv_mean"))
                try:
                    runtime_sec += float(item.get("runtime_sec") or 0.0)
                except Exception:
                    pass
                evaluator_payload = item.get("claim_metric_evaluator")
                if isinstance(evaluator_payload, dict) and evaluator_payload:
                    identity = self._claim_metric_evaluator_identity(evaluator_payload)
                    if identity not in [
                        self._claim_metric_evaluator_identity(existing)
                        for existing in claim_metric_evaluators
                    ]:
                        claim_metric_evaluators.append(dict(evaluator_payload))
                custom = item.get("custom_metrics")
                custom_keys: Set[str] = set()
                if isinstance(custom, dict):
                    for key, value in custom.items():
                        try:
                            custom_values.setdefault(str(key), []).append(float(value))
                            custom_keys.add(str(key))
                        except Exception:
                            continue
                for key in dict.fromkeys(name for name in direct_claim_metric_names if name):
                    if key in {"w1_mean", "tmv_mean"} or key in custom_keys:
                        continue
                    value = self.planner_file_tools._numeric_value(item.get(key))
                    if value is not None:
                        custom_values.setdefault(key, []).append(value)
                simulation_version = str(item.get("simulation_version") or "").strip()
                if simulation_version and simulation_version not in simulation_versions:
                    simulation_versions.append(simulation_version)
            if errors:
                aggregate["error"] = "; ".join(errors)
                aggregate["baseline_unusable"] = True
                unusable_reasons.extend(errors)
            if unusable_reasons:
                aggregate["baseline_unusable"] = True
                aggregate["baseline_unusable_reasons"] = list(dict.fromkeys(unusable_reasons))
            if quality_diagnostic_reasons:
                aggregate["baseline_quality_diagnostic_status"] = "failed"
                aggregate["baseline_quality_diagnostic_reason"] = "; ".join(
                    list(dict.fromkeys(quality_diagnostic_reasons))
                )
            if w1_scores:
                aggregate["w1_scores"] = w1_scores
                try:
                    w1_values = [float(x) for x in w1_scores]
                    aggregate["w1_mean"] = sum(w1_values) / len(w1_values)
                except Exception:
                    pass
            if tmv_scores:
                aggregate["tmv_scores"] = tmv_scores
                try:
                    tmv_values = [float(x) for x in tmv_scores]
                    aggregate["tmv_mean"] = sum(tmv_values) / len(tmv_values)
                    aggregate["tmv_max"] = max(tmv_values)
                except Exception:
                    pass
            if runtime_sec:
                aggregate["runtime_sec"] = runtime_sec
            if custom_values:
                aggregate["custom_metrics"] = {
                    key: (sum(values) / len(values))
                    for key, values in custom_values.items()
                    if values
                }
            if len(claim_metric_evaluators) == 1:
                aggregate["claim_metric_evaluator"] = claim_metric_evaluators[0]
            elif len(claim_metric_evaluators) > 1:
                aggregate["claim_metric_evaluator_mismatch"] = [
                    self._claim_metric_evaluator_identity(item)
                    for item in claim_metric_evaluators
                ]
                aggregate["baseline_unusable"] = True
                reasons = list(aggregate.get("baseline_unusable_reasons") or [])
                reasons.append("baseline datasets were evaluated with different claim metric evaluators")
                aggregate["baseline_unusable_reasons"] = reasons
            if len(simulation_versions) == 1:
                aggregate["simulation_version"] = simulation_versions[0]
            elif simulation_versions:
                aggregate["simulation_versions"] = simulation_versions
            aggregate.update(self.planner_file_tools._consistent_w1_backend_metadata(items))
            return aggregate

        baseline_summaries: List[Dict[str, Any]] = []
        run_records: List[Dict[str, Any]] = []
        missing_records: List[Dict[str, Any]] = []
        refresh_algorithm_set = set(refresh_algorithms)
        for algorithm_name in summary_algorithms:
            allow_run_missing_for_algorithm = bool(run_missing) and algorithm_name in refresh_algorithm_set
            per_dataset_metrics: List[Dict[str, Any]] = []
            run_ids: List[str] = []
            complete = True
            adapter_status = campaign_baseline_metric_adapter_status(claim_metric_run_spec, algorithm_name)
            if (
                target_stage == "stage2_claim_validation"
                and bool(adapter_status.get("declared"))
                and not bool(adapter_status.get("supported", True))
            ):
                for dataset_id in target_dataset_ids:
                    missing_records.append(
                        {
                            "dataset_id": dataset_id,
                            "algorithm_name": algorithm_name,
                            "claim_metric_support_status": "unsupported",
                            "claim_metric_unsupported_reason": str(adapter_status.get("reason") or ""),
                            "baseline_metric_adapter": adapter_status,
                        }
                    )
                continue
            for entry in dataset_entries:
                dataset_id = str(entry.get("dataset_id") or "").strip()
                existing = (
                    {}
                    if overwrite_existing
                    and allow_run_missing_for_algorithm
                    and algorithm_name in refresh_algorithm_set
                    else _find_existing_baseline(dataset_id, algorithm_name)
                )
                existing_metrics = dict(existing.get("metrics") or existing.get("metrics_summary") or {})
                if existing:
                    record_artifacts = (
                        dict(existing.get("artifacts") or {})
                        if isinstance(existing.get("artifacts"), dict)
                        else {}
                    )
                    metric_artifacts = (
                        dict(existing_metrics.get("artifacts") or {})
                        if isinstance(existing_metrics.get("artifacts"), dict)
                        else {}
                    )
                    merged_artifacts = {**record_artifacts, **metric_artifacts}
                    artifact_dir_text = str(merged_artifacts.get("artifact_dir") or "").strip()
                    if artifact_dir_text:
                        artifact_dir = Path(artifact_dir_text).expanduser()
                        conventional_artifacts = {
                            "metrics_path": artifact_dir / "metrics.json",
                            "model_artifact_path": artifact_dir / "model_artifact.json",
                            "model_state_path": artifact_dir / "model_state.pt",
                            "resolved_config_path": artifact_dir / "resolved_config.yaml",
                            "evaluation_trajectory_path": artifact_dir / "evaluation_trajectory.npz",
                            "run_manifest_path": artifact_dir / "run_manifest.json",
                        }
                        for key, candidate_path in conventional_artifacts.items():
                            if not str(merged_artifacts.get(key) or "").strip() and candidate_path.is_file():
                                merged_artifacts[key] = str(candidate_path)
                    if merged_artifacts:
                        existing_metrics["artifacts"] = merged_artifacts
                        for key in (
                            "metrics_path",
                            "model_artifact_path",
                            "model_state_path",
                            "resolved_config_path",
                            "evaluation_trajectory_path",
                            "run_manifest_path",
                        ):
                            if not str(existing_metrics.get(key) or "").strip() and str(
                                merged_artifacts.get(key) or ""
                            ).strip():
                                existing_metrics[key] = str(merged_artifacts.get(key) or "").strip()
                    if str(existing.get("status") or "").strip():
                        existing_metrics.setdefault("status", str(existing.get("status") or "").strip())
                    if str(existing.get("error") or "").strip():
	                        existing_metrics.setdefault("error", str(existing.get("error") or "").strip())
                existing_metrics = _materialize_adapter_metrics(existing_metrics, algorithm_name)
                if formal_holdout_claim_spec:
                    existing_metrics = _hydrate_holdout_report_from_saved_metrics(existing_metrics, existing)
                    existing_metrics = _mark_holdout_claim_unsupported(
                        existing_metrics,
                        existing_metrics.get("holdout_time_evaluation"),
                    )
                primary_raw_available = self.planner_file_tools._metric_value(existing_metrics, primary_metric) is not None
                primary_provenance_ok = _claim_metric_provenance_ok(existing_metrics, algorithm_name)
                primary_available = primary_raw_available and primary_provenance_ok
                rejected_reason = _baseline_rejected_reason(existing, existing_metrics)
                existing_failed = bool(existing) and (
                    str(existing.get("status") or "").strip().lower() == "failed"
                    or bool(str(existing_metrics.get("error") or "").strip())
                    or (bool(rejected_reason) and not _baseline_rejection_is_quality_diagnostic_only(rejected_reason))
                )
                if (
                    existing
                    and not existing_failed
                    and (not primary_available)
                    and formal_holdout_claim_spec
                    and str(existing_metrics.get("claim_metric_support_status") or "") != "unsupported"
                ):
                    holdout_spec_for_baseline = self._holdout_time_eval_spec_from_claim_metric(
                        claim_metric_run_spec,
                        baseline_algorithm=algorithm_name,
                    )
                    existing_metrics, repaired_holdout = _repair_holdout_claim_from_saved_group_trajectories(
                        existing_metrics,
                        holdout_spec_for_baseline,
                        str(entry.get("adata_path") or "") or str(existing_metrics.get("input_adata_path") or ""),
                    )
                    primary_raw_available = self.planner_file_tools._metric_value(existing_metrics, primary_metric) is not None
                    primary_provenance_ok = _claim_metric_provenance_ok(existing_metrics, algorithm_name)
                    primary_available = primary_raw_available and primary_provenance_ok
                    if repaired_holdout and primary_available:
                        self.planner_file_tools.record_algorithm_benchmark_baseline(
                            dataset_id,
                            algorithm_name,
                            existing_metrics,
                            baseline_type=baseline_kind,
                            run_id=str(existing.get("run_id") or existing_metrics.get("run_id") or ""),
                            config_path=str(existing.get("config_path") or existing_metrics.get("resolved_config_path") or ""),
                            notes=(
                                f"Repaired auxiliary hold-out claim metric evaluation for campaign {campaign_id} "
                                f"stage {target_stage} from saved evaluation trajectory artifacts."
                            ),
                        )
                if (
                    existing
                    and not existing_failed
                    and (not primary_available)
                    and formal_holdout_claim_spec
                    and allow_run_missing_for_algorithm
                    and str(existing_metrics.get("claim_metric_support_status") or "") != "unsupported"
                ):
                    holdout_spec_for_baseline = self._holdout_time_eval_spec_from_claim_metric(
                        claim_metric_run_spec,
                        baseline_algorithm=algorithm_name,
                    )
                    try:
                        parent_run_id = str(existing.get("run_id") or existing_metrics.get("run_id") or "").strip()
                        if not parent_run_id:
                            parent_run_id = f"baseline-{algorithm_name}-{dataset_id}"
                        holdout_report = self._run_holdout_time_evaluation(
                            holdout_spec_for_baseline,
                            parent_run_id=parent_run_id,
                            candidate_name=algorithm_name,
                            training_algorithm_id=None,
                            stage=str(policy.get("training_stage") or "pilot"),
                            config_overrides=_baseline_config_overrides_for_entry(entry, dataset_id, algorithm_name),
                            run_label_prefix=f"{campaign_id[:14]}-{target_stage[:8]}-holdout-{algorithm_name[:12]}-{dataset_id[:12]}",
                            adata_path=str(entry.get("adata_path") or "") or str(existing_metrics.get("input_adata_path") or ""),
                            device="",
                            seed=None,
                            decision_reason=(
                                "Auxiliary hold-out timepoint evaluation for a formal campaign Stage 2 "
                                "holdout_time_w1 baseline claim metric."
                            ),
                            campaign_id=campaign_id,
                            dataset_id=dataset_id,
                        )
                        existing_metrics = _apply_holdout_claim_to_metrics(
                            existing_metrics,
                            holdout_report,
                            holdout_spec_for_baseline,
                        )
                        existing_metrics = _mark_holdout_claim_unsupported(existing_metrics, holdout_report)
                        primary_raw_available = self.planner_file_tools._metric_value(existing_metrics, primary_metric) is not None
                        primary_provenance_ok = _claim_metric_provenance_ok(existing_metrics, algorithm_name)
                        primary_available = primary_raw_available and primary_provenance_ok
                        if primary_available:
                            self.planner_file_tools.record_algorithm_benchmark_baseline(
                                dataset_id,
                                algorithm_name,
                                existing_metrics,
                                baseline_type=baseline_kind,
                                run_id=str(existing.get("run_id") or existing_metrics.get("run_id") or ""),
                                config_path=str(existing.get("config_path") or existing_metrics.get("resolved_config_path") or ""),
                                notes=(
                                    f"Auxiliary hold-out claim metric evaluation for campaign {campaign_id} "
                                    f"stage {target_stage}; normal baseline evidence was reused."
                                ),
                            )
                    except Exception as exc:
                        existing_metrics["claim_metric_posthoc_error"] = str(exc)
                if (
                    existing
                    and not existing_failed
                    and (not primary_available)
                    and claim_metric_spec_has_evaluator(claim_metric_run_spec)
                    and not formal_holdout_claim_spec
                    and self.planner_file_tools._metric_value(existing_metrics, "w1_mean") is not None
                ):
                    try:
                        existing_metrics = self._posthoc_evaluate_claim_metric_for_baseline(
                            existing_metrics,
                            claim_metric_spec=claim_metric_run_spec,
                            stage=str(policy.get("training_stage") or "pilot"),
                            baseline_algorithm=algorithm_name,
                            dataset_entry=entry,
                        )
                        primary_raw_available = self.planner_file_tools._metric_value(existing_metrics, primary_metric) is not None
                        primary_provenance_ok = _claim_metric_provenance_ok(existing_metrics, algorithm_name)
                        primary_available = primary_raw_available and primary_provenance_ok
                        if primary_available:
                            self.planner_file_tools.record_algorithm_benchmark_baseline(
                                dataset_id,
                                algorithm_name,
                                existing_metrics,
                                baseline_type=baseline_kind,
                                run_id=str(existing.get("run_id") or existing_metrics.get("run_id") or ""),
                                config_path=str(existing.get("config_path") or existing_metrics.get("resolved_config_path") or ""),
                                notes=(
                                    f"Posthoc claim metric evaluation for campaign {campaign_id} "
                                    f"stage {target_stage}; reused existing trained model."
                                ),
                            )
                    except Exception as exc:
                        existing_metrics["claim_metric_posthoc_error"] = str(exc)
                should_rerun_existing_missing_stage_primary = bool(
                    existing
                    and target_stage == "stage2_claim_validation"
                    and not existing_failed
                    and not primary_available
                    and run_missing
                    and allow_run_missing_for_algorithm
                    and algorithm_name in refresh_algorithm_set
                    and claim_metric_spec_has_evaluator(claim_metric_run_spec)
                )
                if (
                    existing
                    and target_stage == "stage2_claim_validation"
                    and not existing_failed
                    and not primary_available
                    and not should_rerun_existing_missing_stage_primary
                ):
                    missing = {
                        "dataset_id": dataset_id,
                        "algorithm_name": algorithm_name,
                        "reason": "existing_baseline_missing_stage_primary_metric",
                        "primary_metric": primary_metric,
                        "existing_run_id": str(existing.get("run_id") or ""),
                        "record_path": str(existing.get("record_path") or existing.get("path") or ""),
                        "existing_baseline_reused": True,
                        "will_not_retrain_existing_baseline": True,
                    }
                    if not claim_metric_spec_has_evaluator(claim_metric_run_spec):
                        missing["claim_metric_evaluator_missing"] = True
                        missing["message"] = (
                            "The saved baseline has reusable W1/TMV evidence, but this Stage 2 gate needs "
                            f"'{primary_metric}' and the campaign claim_metric_spec has no evaluator_path. "
                            "Provide a method-independent claim metric evaluator/adapters, then refresh again; "
                            "existing baseline trajectories/models will be reused instead of retraining."
                        )
                    elif primary_raw_available and not primary_provenance_ok:
                        missing["claim_metric_evaluator_provenance_missing"] = True
                        missing["message"] = (
                            "The saved baseline has a numeric claim metric, but it was not computed by the "
                            "current campaign claim_metric_spec evaluator. Recompute it from the saved "
                            "evaluation trajectory/model before using it as a Stage 2 baseline."
                        )
                    if str(existing_metrics.get("claim_metric_posthoc_error") or "").strip():
                        missing["claim_metric_posthoc_error"] = str(existing_metrics.get("claim_metric_posthoc_error") or "")
                    if str(existing_metrics.get("claim_metric_support_status") or "") == "unsupported":
                        missing["claim_metric_support_status"] = "unsupported"
                        missing["claim_metric_unsupported_reason"] = str(
                            existing_metrics.get("claim_metric_unsupported_reason") or ""
                        )
                    if str(existing_metrics.get("claim_metric_trajectory_posthoc_error") or "").strip():
                        missing["claim_metric_trajectory_posthoc_error"] = str(
                            existing_metrics.get("claim_metric_trajectory_posthoc_error") or ""
                        )
                    if str(existing_metrics.get("claim_metric_adapter_verification_error") or "").strip():
                        missing["claim_metric_adapter_verification_error"] = str(
                            existing_metrics.get("claim_metric_adapter_verification_error") or ""
                        )
                        missing["claim_metric_adapter_unverified_metrics"] = list(
                            existing_metrics.get("claim_metric_adapter_unverified_metrics") or []
                        )
                    existing_status = str(
                        existing.get("status") or existing_metrics.get("status") or ""
                    ).strip().lower()
                    existing_terminal = bool(existing) and (
                        existing_status
                        in {
                            "complete",
                            "completed",
                            "failed",
                            "rejected",
                            "success",
                            "succeeded",
                            "unusable",
                        }
                        or bool(str(existing.get("error") or existing_metrics.get("error") or "").strip())
                    )
                    if existing_terminal:
                        terminal_reasons = [
                            str(missing.get(key) or "").strip()
                            for key in (
                                "reason",
                                "message",
                                "claim_metric_posthoc_error",
                                "claim_metric_trajectory_posthoc_error",
                                "claim_metric_adapter_verification_error",
                                "claim_metric_unsupported_reason",
                            )
                            if str(missing.get(key) or "").strip()
                        ]
                        existing_metrics["baseline_unusable"] = True
                        existing_metrics["baseline_unusable_reason"] = (
                            "existing_terminal_baseline_missing_current_campaign_claim_metric: "
                            + "; ".join(dict.fromkeys(terminal_reasons))
                        )
                        existing_metrics["claim_metric_missing_reason"] = (
                            "existing_terminal_baseline_missing_current_campaign_claim_metric"
                        )
                        existing_metrics["terminal_baseline_record_reused"] = True
                        existing_metrics["terminal_baseline_record_status"] = existing_status
                        existing_failed = True
                    else:
                        missing_records.append(missing)
                        complete = False
                        continue
                if existing and (existing_failed or target_stage != "stage2_claim_validation" or primary_available):
                    metrics = existing_metrics
                    run_id = str(existing.get("run_id") or "")
                    if rejected_reason and _baseline_rejection_is_quality_diagnostic_only(rejected_reason):
                        source = "existing"
                        metrics["baseline_quality_diagnostic_status"] = "failed"
                        metrics["baseline_quality_diagnostic_reason"] = rejected_reason
                        metrics["baseline_run_verdict_status"] = "rejected"
                    elif rejected_reason:
                        source = "existing_rejected"
                        metrics["baseline_unusable"] = True
                        metrics["baseline_unusable_reason"] = rejected_reason
                        metrics["baseline_run_verdict_status"] = "rejected"
                    elif existing_failed:
                        source = "existing_failed"
                    else:
                        source = (
                            str(existing_metrics.get("claim_metric_evaluation_source") or "posthoc_model")
                            if bool(existing_metrics.get("claim_metric_posthoc_evaluated"))
                            else "existing"
                        )
                elif not allow_run_missing_for_algorithm:
                    missing = {"dataset_id": dataset_id, "algorithm_name": algorithm_name}
                    if run_missing and algorithm_name not in refresh_algorithm_set:
                        missing["reason"] = "baseline_not_requested_for_this_repair_refresh"
                        missing["repair_scope"] = list(refresh_algorithms)
                    corrupt = next(
                        (
                            item
                            for item in ignored_corrupt_records
                            if item.get("dataset_id") == dataset_id and item.get("algorithm_name") == algorithm_name
                        ),
                        {},
                    )
                    if corrupt:
                        missing["corrupt_existing_baseline"] = corrupt.get("reason")
                        missing["corrupt_record_path"] = corrupt.get("record_path")
                    if str(existing_metrics.get("claim_metric_posthoc_error") or "").strip():
                        missing["claim_metric_posthoc_error"] = str(existing_metrics.get("claim_metric_posthoc_error") or "")
                    if str(existing_metrics.get("claim_metric_adapter_verification_error") or "").strip():
                        missing["claim_metric_adapter_verification_error"] = str(
                            existing_metrics.get("claim_metric_adapter_verification_error") or ""
                        )
                        missing["claim_metric_adapter_unverified_metrics"] = list(
                            existing_metrics.get("claim_metric_adapter_unverified_metrics") or []
                        )
                    if str(existing_metrics.get("claim_metric_support_status") or "") == "unsupported":
                        missing["claim_metric_support_status"] = "unsupported"
                        missing["claim_metric_unsupported_reason"] = str(existing_metrics.get("claim_metric_unsupported_reason") or "")
                    missing_records.append(missing)
                    complete = False
                    continue
                else:
                    entry_overrides = entry.get("config_overrides") if isinstance(entry.get("config_overrides"), dict) else entry
                    config_overrides = _merge_overrides(dict(common_overrides or {}), dict(entry_overrides or {}))
                    benchmark_baseline_config = {}
                    if baseline_kind == "builtin":
                        benchmark_baseline_config = self.planner_file_tools.get_algorithm_benchmark_baseline_config(
                            dataset_id,
                            algorithm_name,
                            baseline_type=baseline_kind,
                        )
                        benchmark_overrides = (
                            benchmark_baseline_config.get("config_overrides")
                            if isinstance(benchmark_baseline_config, dict)
                            and isinstance(benchmark_baseline_config.get("config_overrides"), dict)
                            else {}
                        )
                        if benchmark_overrides:
                            config_overrides = _merge_overrides(config_overrides, dict(benchmark_overrides or {}))
                    if claim_metric_spec_has_evaluator(claim_metric_run_spec):
                        config_overrides["__campaign_claim_metric_spec"] = deepcopy(claim_metric_run_spec)
                    before_run_id = str(self.state.get("latest_training_run_id") or "")
                    try:
                        baseline_training_kwargs = {
                            "candidate_name": algorithm_name,
                            "stage": str(policy.get("training_stage") or "pilot"),
                            "config_overrides": config_overrides,
                            "run_label": f"{campaign_id[:14]}-{target_stage[:8]}-baseline-{algorithm_name[:12]}-{dataset_id[:12]}",
                            "adata_path": str(entry.get("adata_path") or "") or None,
                            "decision": "reject",
                            "decision_reason": "Campaign baseline refresh records external reference metrics; it does not update campaign active best.",
                        }
                        monkeypatched_runner = self.__dict__.get("run_training_tool")
                        if callable(monkeypatched_runner):
                            training_result = monkeypatched_runner(**baseline_training_kwargs)
                        else:
                            training_result = self._run_training_tool_impl(
                                **baseline_training_kwargs,
                                review_purpose="manual_training",
                                review_campaign=None,
                                _skip_review_gates=False,
                            )
                    except Exception as exc:
                        missing_records.append(
                            {
                                "dataset_id": dataset_id,
                                "algorithm_name": algorithm_name,
                                "error": f"baseline training raised before producing a run: {exc}",
                            }
                        )
                        complete = False
                        continue
                    after_run_id = str(self.state.get("latest_training_run_id") or "")
                    if not after_run_id or after_run_id == before_run_id:
                        missing_records.append(
                            {
                                "dataset_id": dataset_id,
                                "algorithm_name": algorithm_name,
                                "error": "baseline training did not produce a new run_id",
                                "previous_run_id": before_run_id,
                                "training_result_preview": str(training_result or "")[:1000],
                            }
                        )
                        complete = False
                        continue
                    run_id = after_run_id
                    metrics = self._campaign_latest_training_metrics(run_id)
                    if not metrics:
                        metrics = {"error": str(training_result)}
                    run_state = next(
                        (
                            item
                            for item in list(self.state.get("training_runs") or [])
                            if str((item or {}).get("run_id") or "") == run_id
                        ),
                        {},
                    )
                    run_status = str(run_state.get("status") or metrics.get("status") or "").strip().lower()
                    if run_status in {"running", "pending", "queued", "started"}:
                        missing_records.append(
                            {
                                "dataset_id": dataset_id,
                                "algorithm_name": algorithm_name,
                                "error": "baseline training has no terminal metrics record",
                                "run_id": run_id,
                                "run_status": run_status,
                                "training_result_preview": str(training_result or "")[:1000],
                            }
                        )
                        complete = False
                        continue
                    if formal_holdout_claim_spec and not metrics.get("error"):
                        holdout_spec_for_baseline = self._holdout_time_eval_spec_from_claim_metric(
                            claim_metric_run_spec,
                            baseline_algorithm=algorithm_name,
                        )
                        holdout_report = self._run_holdout_time_evaluation(
                            holdout_spec_for_baseline,
                            parent_run_id=run_id,
                            candidate_name=algorithm_name,
                            training_algorithm_id=None,
                            stage=str(policy.get("training_stage") or "pilot"),
                            config_overrides=config_overrides,
                            run_label_prefix=f"{campaign_id[:14]}-{target_stage[:8]}-holdout-{algorithm_name[:12]}-{dataset_id[:12]}",
                            adata_path=str(entry.get("adata_path") or "") or None,
                            device="",
                            seed=None,
                            decision_reason=(
                                "Auxiliary hold-out timepoint evaluation for a formal campaign Stage 2 "
                                "holdout_time_w1 baseline claim metric."
                            ),
                            campaign_id=campaign_id,
                            dataset_id=dataset_id,
                        )
                        metrics = _apply_holdout_claim_to_metrics(
                            metrics,
                            holdout_report,
                            holdout_spec_for_baseline,
                        )
                        metrics = _mark_holdout_claim_unsupported(metrics, holdout_report)
                        self._attach_holdout_report_to_run_metrics(
                            run_id=run_id,
                            report=holdout_report,
                            holdout_spec=holdout_spec_for_baseline,
                        )
                    recorded = self.planner_file_tools.record_algorithm_benchmark_baseline(
                        dataset_id,
                        algorithm_name,
                        metrics,
                        baseline_type=baseline_kind,
                        run_id=run_id,
                        config_path=str(metrics.get("resolved_config_path") or ""),
                        notes=(
                            f"Auto-refreshed for campaign {campaign_id} stage {target_stage}."
                            + (
                                f" Benchmark baseline config: {benchmark_baseline_config.get('config_path')}."
                                if isinstance(benchmark_baseline_config, dict)
                                and str(benchmark_baseline_config.get("config_path") or "").strip()
                                else ""
                            )
                        ),
                    )
                    source = "ran"
                    run_records.append(
                        {
                            "dataset_id": dataset_id,
                            "algorithm_name": algorithm_name,
                            "run_id": run_id,
                            "record_path": str(recorded.get("record_path") or ""),
                            "benchmark_baseline_config_path": (
                                str(benchmark_baseline_config.get("config_path") or "")
                                if isinstance(benchmark_baseline_config, dict)
                                else ""
                            ),
                        }
                    )
                    run_failed_or_unusable = bool(str(metrics.get("error") or "").strip()) or (
                        str(metrics.get("status") or "").strip().lower() == "failed"
                    )
                    if strict_claim_metric and not run_failed_or_unusable and not _claim_metric_provenance_ok(metrics, algorithm_name):
                        unusable_reason = (
                            "baseline_run_missing_campaign_claim_metric: "
                            "Baseline run completed but did not produce the Stage 2 claim metric through "
                            "the current campaign evaluator."
                        )
                        metrics["baseline_unusable"] = True
                        metrics["baseline_unusable_reason"] = unusable_reason
                        metrics["claim_metric_missing_reason"] = "baseline_run_missing_campaign_claim_metric"
                        if str(metrics.get("custom_metrics_warning") or "").strip():
                            metrics["claim_metric_missing_warning"] = str(metrics.get("custom_metrics_warning") or "")
                        recorded = self.planner_file_tools.record_algorithm_benchmark_baseline(
                            dataset_id,
                            algorithm_name,
                            metrics,
                            baseline_type=baseline_kind,
                            run_id=run_id,
                            config_path=str(metrics.get("resolved_config_path") or ""),
                            notes=(
                                f"Auto-refreshed for campaign {campaign_id} stage {target_stage}; "
                                "recorded as terminal unusable because the run did not produce the "
                                "campaign claim metric."
                            ),
                        )
                if run_id:
                    run_ids.append(run_id)
                metrics = dict(metrics or {})
                metrics["campaign_dataset_id"] = dataset_id
                metrics["algorithm_name"] = algorithm_name
                metrics["baseline_type"] = baseline_kind
                metrics["baseline_source"] = source
                simulation_version = str(entry.get("simulation_version") or "").strip()
                if simulation_version and not str(metrics.get("simulation_version") or "").strip():
                    metrics["simulation_version"] = simulation_version
                per_dataset_metrics.append(metrics)
            if not complete or len(per_dataset_metrics) != len(dataset_entries):
                continue
            aggregate = _aggregate_metrics(per_dataset_metrics, algorithm_name, run_ids)
            summary = self.planner_file_tools._summarize_campaign_metrics(aggregate, policy)
            summary.update(
                {
                    "algorithm_name": algorithm_name,
                    "baseline_type": baseline_kind,
                    "target_dataset_ids": list(target_dataset_ids),
                    "baseline_run_ids": run_ids,
                    "baseline_unusable": bool(aggregate.get("baseline_unusable")),
                    "baseline_unusable_reasons": list(aggregate.get("baseline_unusable_reasons") or []),
                    "baseline_dataset_summaries": [
                        {
                            "dataset_id": item.get("campaign_dataset_id"),
                            "w1_mean": self.planner_file_tools._metric_value(item, "w1_mean"),
                            "tmv_mean": self.planner_file_tools._metric_value(item, "tmv_mean"),
                            "primary_value": self.planner_file_tools._metric_value(item, primary_metric),
                            "source": item.get("baseline_source"),
                            "run_id": item.get("run_id"),
                            "baseline_unusable": bool(item.get("baseline_unusable")),
                            "baseline_unusable_reason": str(item.get("baseline_unusable_reason") or ""),
                        }
                        for item in per_dataset_metrics
                    ],
                }
            )
            baseline_summaries.append(summary)

        def _primary_for(summary: Dict[str, Any]) -> Optional[float]:
            value = self.planner_file_tools._metric_value(summary, primary_metric)
            if value is None:
                value = self.planner_file_tools._numeric_value(summary.get("primary_value"))
            return value

        def _clean_baseline_record(summary: Dict[str, Any]) -> Dict[str, Any]:
            record = dict(summary)
            record.pop("claim_sota_baseline_metrics", None)
            record.pop("w1_sota_baseline_metrics", None)
            return record

        def _metric_for(summary: Dict[str, Any], metric_name: str) -> Optional[float]:
            return self.planner_file_tools._metric_value(summary, metric_name)

        def _select_metric_sota(
            candidates: List[Dict[str, Any]],
            *,
            metric_name: str,
            direction: str,
        ) -> Optional[Dict[str, Any]]:
            comparable = [
                item
                for item in candidates
                if _metric_for(item, metric_name) is not None and not bool(item.get("baseline_unusable"))
            ]
            if not comparable:
                return None
            if direction == "greater":
                return max(comparable, key=lambda item: float(_metric_for(item, metric_name)))
            return min(comparable, key=lambda item: float(_metric_for(item, metric_name)))

        eligible = [
            item
            for item in baseline_summaries
            if _primary_for(item) is not None and not bool(item.get("baseline_unusable"))
        ]
        if target_stage == "stage2_claim_validation":
            eligible = [item for item in eligible if self.planner_file_tools._metric_value(item, "w1_mean") is not None]
        strict_requires_complete_baseline_set = (
            baseline_kind == "builtin"
            and baseline_policy == "strict_all_builtin"
            and target_stage in {"stage2_claim_validation", "stage3_tuning"}
        )
        strict_required_baselines = list(strict_gate_algorithms) if strict_requires_complete_baseline_set else []
        strict_comparable_baselines = [
            str(item.get("algorithm_name") or "")
            for item in eligible
            if str(item.get("algorithm_name") or "")
        ]
        strict_incomplete_baselines: List[Dict[str, Any]] = []
        summaries_by_algorithm = {
            str(item.get("algorithm_name") or ""): item
            for item in baseline_summaries
            if str(item.get("algorithm_name") or "")
        }
        missing_by_algorithm: Dict[str, List[Dict[str, Any]]] = {}
        for record in missing_records:
            if not isinstance(record, dict):
                continue
            name = str(record.get("algorithm_name") or "").strip()
            if name:
                missing_by_algorithm.setdefault(name, []).append(record)
        if strict_requires_complete_baseline_set:
            comparable_set = set(strict_comparable_baselines)
            for algorithm_name in strict_required_baselines:
                if algorithm_name in comparable_set:
                    continue
                summary = summaries_by_algorithm.get(algorithm_name) or {}
                reasons: List[str] = []
                if summary:
                    for reason in list(summary.get("baseline_unusable_reasons") or []):
                        text = str(reason or "").strip()
                        if text:
                            reasons.append(text)
                    if bool(summary.get("baseline_unusable")) and not reasons:
                        reasons.append("baseline marked unusable")
                    if _primary_for(summary) is None:
                        reasons.append(f"missing stage primary metric `{primary_metric}`")
                    if (
                        target_stage == "stage2_claim_validation"
                        and self.planner_file_tools._metric_value(summary, "w1_mean") is None
                    ):
                        reasons.append("missing Stage 2 secondary W1 metric `w1_mean`")
                for missing in missing_by_algorithm.get(algorithm_name, []):
                    for key in (
                        "reason",
                        "error",
                        "message",
                        "claim_metric_unsupported_reason",
                        "claim_metric_posthoc_error",
                        "claim_metric_trajectory_posthoc_error",
                        "claim_metric_adapter_verification_error",
                    ):
                        text = str(missing.get(key) or "").strip()
                        if text:
                            reasons.append(text)
                    if missing.get("claim_metric_support_status") == "unsupported" and not reasons:
                        reasons.append("baseline declared unsupported for this claim metric")
                reasons = list(dict.fromkeys(reasons))
                strict_incomplete_baselines.append(
                    {
                        "algorithm_name": algorithm_name,
                        "status": "incomplete_or_unusable",
                        "reasons": reasons or ["no complete comparable baseline metrics were recorded"],
                        "has_summary": bool(summary),
                        "missing_records": missing_by_algorithm.get(algorithm_name, []),
                    }
                )
        if not eligible:
            if strict_requires_complete_baseline_set:
                selected = {
                    "algorithm_name": "",
                    "baseline_type": baseline_kind,
                    "target_dataset_ids": list(target_dataset_ids),
                    "selection_rule": "strict_audit_no_comparable_baseline",
                    "selection_candidate_count": 0,
                    "selection_note": (
                        "No comparable baseline could be selected. The strict audit ledger below still records "
                        "an explicit pass/fail/missing evidence row for every required builtin baseline."
                    ),
                }
                selection_rule = str(selected["selection_rule"])
            else:
                return json.dumps(
                    {
                        "status": "blocked",
                        "reason": "no_comparable_baseline",
                        "campaign_id": campaign_id,
                        "stage": target_stage,
                        "primary_metric": primary_metric,
                        "target_dataset_ids": target_dataset_ids,
                        "baseline_summaries": baseline_summaries,
                        "missing_records": missing_records,
                        "ignored_corrupt_records": ignored_corrupt_records,
                        "run_records": run_records,
                        "message": (
                            "No complete baseline has the stage primary metric. "
                            "For Stage 2, make sure the claim metric is method-independent and computed for builtin/reference baselines. "
                            "If a baseline truly cannot expose the required observable evidence, declare it unsupported via "
                            "baseline_metric_adapters and use a claim metric with enough comparable supported baselines."
                        ),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
        elif target_stage == "stage1_feasibility":
            sorted_by_weakness = sorted(
                eligible,
                key=lambda item: float(_primary_for(item) or 0.0),
                reverse=True,
            )
            selected = sorted_by_weakness[1] if len(sorted_by_weakness) > 1 else sorted_by_weakness[0]
            selection_rule = (
                "second_weakest_w1_baseline_for_feasibility"
                if len(sorted_by_weakness) > 1
                else "only_w1_baseline_for_feasibility"
            )
            selected = _clean_baseline_record(selected)
            selected["selection_rule"] = selection_rule
            selected["selection_candidate_count"] = len(eligible)
            selected["selection_note"] = (
                "Stage 1 uses the second-worst comparable W1 baseline when possible, "
                "so a single outlier-worst builtin does not make the feasibility gate too loose."
            )
        elif primary_direction == "greater":
            selected = max(eligible, key=lambda item: float(_primary_for(item) or 0.0))
            selection_rule = "strongest_primary_metric_baseline"
            selected = _clean_baseline_record(selected)
            selected["selection_rule"] = selection_rule
        else:
            selected = min(eligible, key=lambda item: float(_primary_for(item) or 1e99))
            selection_rule = "strongest_primary_metric_baseline"
            selected = _clean_baseline_record(selected)
            selected["selection_rule"] = selection_rule
        if target_stage in {"stage2_claim_validation", "stage3_tuning"}:
            claim_metric_name = primary_metric
            claim_direction = primary_direction
            if target_stage == "stage3_tuning":
                claim_metric_name = str(policy.get("secondary_metric") or "").strip()
                claim_direction = str(policy.get("secondary_direction") or "").strip()
            claim_sota = (
                _select_metric_sota(
                    baseline_summaries,
                    metric_name=claim_metric_name,
                    direction=claim_direction,
                )
                if claim_metric_name
                else None
            )
            w1_sota = _select_metric_sota(
                baseline_summaries,
                metric_name="w1_mean",
                direction="lower",
            )
            if claim_sota:
                selected["claim_sota_baseline_metrics"] = _clean_baseline_record(claim_sota)
                selected["claim_sota_selection_rule"] = "strongest_claim_metric_baseline"
            if w1_sota:
                selected["w1_sota_baseline_metrics"] = _clean_baseline_record(w1_sota)
                selected["w1_sota_selection_rule"] = "lowest_w1_baseline"
            if claim_sota or w1_sota:
                selected["selection_rule"] = "independent_claim_and_w1_sota_baselines"
                selected["legacy_primary_selection_rule"] = selection_rule
                selection_rule = selected["selection_rule"]
                selected["selection_note"] = (
                    "Stage 2/3 gates use independent SOTA baselines: the claim metric is compared "
                    "against the strongest claim-metric baseline, while W1 is compared against the "
                    "lowest-W1 baseline. These records may be different algorithms."
                )
            selected["selection_candidate_count"] = len(eligible)
        if strict_requires_complete_baseline_set:
            selected["strict_required_baseline_count"] = len(strict_required_baselines)
        audit_ledger: List[Dict[str, Any]] = []
        comparable_set = set(strict_comparable_baselines)
        claim_sota_algorithm = str((selected.get("claim_sota_baseline_metrics") or {}).get("algorithm_name") or "")
        w1_sota_algorithm = str((selected.get("w1_sota_baseline_metrics") or {}).get("algorithm_name") or "")
        selected_algorithm = str(selected.get("algorithm_name") or "")
        audit_algorithm_names = list(strict_required_baselines or summary_algorithms)
        for algorithm_name in audit_algorithm_names:
            summary = summaries_by_algorithm.get(algorithm_name) or {}
            missing_for_algorithm = missing_by_algorithm.get(algorithm_name, [])
            reasons = [
                str(item or "").strip()
                for item in list(summary.get("baseline_unusable_reasons") or [])
                if str(item or "").strip()
            ]
            if bool(summary.get("baseline_unusable")) and not reasons:
                reasons.append("baseline marked unusable")
            for missing in missing_for_algorithm:
                for key in (
                    "reason",
                    "error",
                    "message",
                    "claim_metric_unsupported_reason",
                    "claim_metric_posthoc_error",
                    "claim_metric_trajectory_posthoc_error",
                    "claim_metric_adapter_verification_error",
                ):
                    text = str((missing or {}).get(key) or "").strip()
                    if text:
                        reasons.append(text)
            status = "pass_comparable" if algorithm_name in comparable_set else "fail_incomplete_or_unusable"
            if not summary and not missing_for_algorithm:
                reasons.append("no baseline record was found")
            has_terminal_record = bool(summary)
            missing_required_record = bool(status != "pass_comparable" and not has_terminal_record)
            nonblocking_failed_record = bool(status != "pass_comparable" and has_terminal_record)
            entry: Dict[str, Any] = {
                "algorithm_name": algorithm_name,
                "baseline_type": baseline_kind,
                "target_dataset_ids": list(target_dataset_ids),
                "audit_status": status,
                "comparable": bool(status == "pass_comparable"),
                "selected_primary": bool(algorithm_name == selected_algorithm),
                "selected_claim_sota": bool(algorithm_name == claim_sota_algorithm),
                "selected_w1_sota": bool(algorithm_name == w1_sota_algorithm),
                "reasons": list(dict.fromkeys(reasons)),
                "missing_record_count": len(missing_for_algorithm),
                "has_baseline_summary": has_terminal_record,
                "has_terminal_baseline_record": has_terminal_record,
                "missing_required_record": missing_required_record,
                "nonblocking_failed_record": nonblocking_failed_record,
            }
            if summary:
                entry.update(
                    {
                        "w1_mean": self.planner_file_tools._metric_value(summary, "w1_mean"),
                        "tmv_mean": self.planner_file_tools._metric_value(summary, "tmv_mean"),
                        "tmv_gate_required": bool(summary.get("tmv_gate_required", False)),
                        "tmv_gate_reason": str(summary.get("tmv_gate_reason") or ""),
                        "primary_metric": primary_metric,
                        "primary_value": _primary_for(summary),
                        "baseline_run_ids": list(summary.get("baseline_run_ids") or []),
                        "baseline_dataset_summaries": list(summary.get("baseline_dataset_summaries") or []),
                    }
                )
                quality_reason = str(summary.get("baseline_quality_diagnostic_reason") or "").strip()
                if quality_reason:
                    entry["baseline_quality_diagnostic_status"] = str(
                        summary.get("baseline_quality_diagnostic_status") or "failed"
                    )
                    entry["baseline_quality_diagnostic_reason"] = quality_reason
                if isinstance(summary.get("claim_metric_evaluator"), dict):
                    entry["claim_metric_evaluator"] = deepcopy(summary.get("claim_metric_evaluator") or {})
                secondary_metric = str(policy.get("secondary_metric") or "").strip()
                if secondary_metric:
                    entry["secondary_metric"] = secondary_metric
                    entry["secondary_value"] = self.planner_file_tools._metric_value(summary, secondary_metric)
                for key, value in summary.items():
                    if str(key).startswith("w1_backend"):
                        entry[str(key)] = deepcopy(value)
            if missing_for_algorithm:
                entry["missing_records"] = deepcopy(missing_for_algorithm)
            audit_ledger.append(entry)
        strict_missing_baselines = [
            item for item in audit_ledger if bool(item.get("missing_required_record"))
        ]
        strict_failed_baselines = [
            item for item in audit_ledger if bool(item.get("nonblocking_failed_record"))
        ]
        selected["strict_baseline_audit_ledger"] = audit_ledger
        selected["strict_baseline_audit"] = {
            "required": bool(strict_requires_complete_baseline_set),
            "baseline_selection_policy": baseline_policy,
            "required_baselines": list(strict_required_baselines or summary_algorithms),
            "comparable_baselines": list(strict_comparable_baselines),
            "candidate_count": len(eligible),
            "ledger_count": len(audit_ledger),
            "failed_count": sum(1 for item in audit_ledger if str(item.get("audit_status") or "") != "pass_comparable"),
            "missing_count": len(strict_missing_baselines),
            "nonblocking_failed_count": len(strict_failed_baselines),
        }
        selected["baseline_refresh_at"] = datetime.utcnow().isoformat() + "Z"
        update = self.planner_file_tools.set_campaign_stage_baseline_metrics(
            campaign_id,
            stage=target_stage,
            baseline_metrics=selected,
            source="refresh_campaign_stage_baselines",
            baseline_summaries=baseline_summaries,
            run_missing=run_missing,
        )
        payload = {
            "status": "blocked" if strict_missing_baselines else "updated",
            "campaign_id": campaign_id,
            "stage": target_stage,
            "target_dataset_ids": target_dataset_ids,
            "baseline_algorithms": summary_algorithms,
            "refresh_baseline_algorithms": refresh_algorithms,
            "requested_baseline_algorithms": requested_algorithms,
            "baseline_selection_policy": baseline_policy,
            "ignored_requested_baseline_algorithms": False,
            "required_anchor_baseline": required_anchor_baseline,
            "anchor_baseline_added": anchor_baseline_added,
            "selected_baseline": selected,
            "selection_rule": selection_rule,
            "baseline_summaries": baseline_summaries,
            "run_records": run_records,
            "missing_records": missing_records,
            "ignored_corrupt_records": ignored_corrupt_records,
            "stage_update": update,
        }
        if strict_requires_complete_baseline_set:
            payload["strict_required_baselines"] = strict_required_baselines
            payload["strict_comparable_baselines"] = strict_comparable_baselines
            payload["strict_incomplete_baselines"] = strict_incomplete_baselines
            payload["strict_missing_baselines"] = strict_missing_baselines
            payload["strict_failed_baselines"] = strict_failed_baselines
        if strict_missing_baselines:
            payload.update(
                {
                    "reason": "strict_required_baseline_missing_records",
                    "message": (
                        "strict_all_builtin Stage 2/3 gates require every fixed builtin baseline to have "
                        "an explicit terminal audit record. Baselines with recorded failed/unusable terminal "
                        "records are retained as non-blocking audit rows, but missing terminal records must "
                        "be repaired or rerun."
                    ),
                    "next_required_action": (
                        "Rerun or repair only the baselines listed in strict_missing_baselines so each has "
                        "an explicit pass or failed/unusable record. Do not tune fixed builtin baselines just "
                        "because their recorded terminal outcome is failed."
                    ),
                }
            )
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def register_campaign_control_baseline(
        self,
        campaign_id: str = "",
        baseline_id: str = "",
        baseline_name: str = "",
        baseline_type: str = "control",
        stage: str = "",
        metrics: Optional[Dict[str, Any]] = None,
        target_dataset_ids: Optional[List[str]] = None,
        metric_roles: Optional[List[str]] = None,
        required_for_gate: bool = True,
        source_run_id: str = "",
        source_trial_id: str = "",
        source_metrics_path: str = "",
        source_config_path: str = "",
        source_artifact_paths: Optional[Dict[str, str]] = None,
        control_code_source: Optional[Dict[str, Any]] = None,
        reason: str = "",
    ) -> str:
        campaign_id = str(campaign_id or self.state.get("active_algorithm_campaign_id") or "").strip()
        payload = self.planner_file_tools.register_campaign_control_baseline(
            campaign_id,
            baseline_id=baseline_id,
            baseline_name=baseline_name,
            baseline_type=baseline_type,
            stage=stage,
            metrics=dict(metrics or {}),
            target_dataset_ids=list(target_dataset_ids or []),
            metric_roles=list(metric_roles or []),
            required_for_gate=bool(required_for_gate),
            source_run_id=source_run_id,
            source_trial_id=source_trial_id,
            source_metrics_path=source_metrics_path,
            source_config_path=source_config_path,
            source_artifact_paths=dict(source_artifact_paths or {}),
            control_code_source=dict(control_code_source or {}),
            reason=reason,
        )
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def update_campaign_control_baseline(
        self,
        campaign_id: str = "",
        baseline_id: str = "",
        stage: str = "",
        action: str = "deactivate",
        metrics: Optional[Dict[str, Any]] = None,
        target_dataset_ids: Optional[List[str]] = None,
        metric_roles: Optional[List[str]] = None,
        required_for_gate: Optional[bool] = None,
        source_run_id: str = "",
        source_trial_id: str = "",
        source_metrics_path: str = "",
        source_config_path: str = "",
        source_artifact_paths: Optional[Dict[str, str]] = None,
        control_code_source: Optional[Dict[str, Any]] = None,
        reason: str = "",
    ) -> str:
        campaign_id = str(campaign_id or self.state.get("active_algorithm_campaign_id") or "").strip()
        payload = self.planner_file_tools.update_campaign_control_baseline(
            campaign_id,
            baseline_id=baseline_id,
            stage=stage,
            action=action,
            metrics=dict(metrics or {}) if metrics is not None else None,
            target_dataset_ids=list(target_dataset_ids or []) if target_dataset_ids is not None else None,
            metric_roles=list(metric_roles or []) if metric_roles is not None else None,
            required_for_gate=required_for_gate,
            source_run_id=source_run_id,
            source_trial_id=source_trial_id,
            source_metrics_path=source_metrics_path,
            source_config_path=source_config_path,
            source_artifact_paths=dict(source_artifact_paths or {}) if source_artifact_paths is not None else None,
            control_code_source=dict(control_code_source or {}) if control_code_source is not None else None,
            reason=reason,
        )
        return json.dumps(payload, ensure_ascii=False, indent=2)

    @staticmethod
    def _merge_control_config_overrides(base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
        merged = deepcopy(base or {})
        for key, value in dict(extra or {}).items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = PlannerTools._merge_control_config_overrides(dict(merged.get(key) or {}), value)
            else:
                merged[key] = deepcopy(value)
        return merged

    def _run_locked_reference_dataset_adapter(
        self,
        *,
        adapter_spec: Dict[str, Any],
        campaign_id: str,
        current_algorithm_id: str,
        reference_algorithm_id: str,
        baseline_id: str,
        stage: str,
        dataset_id: str,
        input_adata_path: str,
    ) -> Dict[str, Any]:
        spec = deepcopy(adapter_spec or {})
        if not spec:
            return {"adapted_adata_path": input_adata_path, "adapter_applied": False}
        kind = str(spec.get("kind") or spec.get("type") or "").strip().lower()
        if kind not in {"python_script", "dataset_adapter_script"}:
            raise ValueError(
                "reference_dataset_adapter.kind must be 'python_script' or 'dataset_adapter_script'. "
                "Do not use config overrides to guess format compatibility."
            )
        script_text = str(spec.get("script") or "").strip()
        script_path_text = str(spec.get("script_path") or "").strip()
        if not script_text and not script_path_text:
            raise ValueError("reference_dataset_adapter requires script_path or inline script.")
        adapter_root = (
            get_cellcompass_root()
            / "algorithm_campaigns"
            / str(current_algorithm_id)
            / str(campaign_id)
            / "locked_reference_adapters"
            / str(baseline_id)
            / str(dataset_id)
        )
        adapter_root.mkdir(parents=True, exist_ok=True)
        if script_text:
            script_path = adapter_root / "reference_dataset_adapter.py"
            script_path.write_text(script_text + ("\n" if not script_text.endswith("\n") else ""), encoding="utf-8")
        else:
            script_path = Path(script_path_text).expanduser().resolve()
        if not script_path.is_file():
            raise FileNotFoundError(f"reference_dataset_adapter script not found: {script_path}")
        output_path = adapter_root / f"{dataset_id}__{reference_algorithm_id}__adapted.h5ad"
        metadata_path = adapter_root / "adapter_context.json"
        context = {
            "campaign_id": campaign_id,
            "current_algorithm_id": current_algorithm_id,
            "reference_algorithm_id": reference_algorithm_id,
            "baseline_id": baseline_id,
            "stage": stage,
            "dataset_id": dataset_id,
            "input_adata_path": input_adata_path,
            "output_adata_path": str(output_path),
            "adapter_reason": str(spec.get("reason") or ""),
        }
        metadata_path.write_text(json.dumps(context, indent=2, ensure_ascii=False), encoding="utf-8")
        env = {
            **os.environ,
            "CYTOBRIDGE_REFERENCE_ADAPTER_CONTEXT_JSON": json.dumps(context, ensure_ascii=False),
        }
        cmd = [
            sys.executable,
            str(script_path),
            "--input-adata",
            input_adata_path,
            "--output-adata",
            str(output_path),
            "--dataset-id",
            dataset_id,
            "--reference-algorithm-id",
            reference_algorithm_id,
            "--current-algorithm-id",
            current_algorithm_id,
            "--context-json",
            str(metadata_path),
        ]
        completed = subprocess.run(
            cmd,
            cwd=str(adapter_root),
            env=env,
            text=True,
            capture_output=True,
            timeout=int(spec.get("timeout_sec") or 900),
            check=False,
        )
        report = {
            "adapter_applied": True,
            "kind": kind,
            "script_path": str(script_path),
            "script_sha256": self._path_sha256(script_path),
            "context_path": str(metadata_path),
            "input_adata_path": input_adata_path,
            "adapted_adata_path": str(output_path),
            "returncode": completed.returncode,
            "stdout": completed.stdout[-4000:],
            "stderr": completed.stderr[-4000:],
        }
        report_path = adapter_root / "adapter_report.json"
        report["report_path"] = str(report_path)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        if completed.returncode != 0:
            raise RuntimeError(
                f"reference_dataset_adapter failed for dataset '{dataset_id}' with exit code {completed.returncode}; "
                f"see {report_path}"
            )
        if not output_path.is_file():
            raise FileNotFoundError(
                f"reference_dataset_adapter did not create output_adata_path for dataset '{dataset_id}': {output_path}"
            )
        return report

    def _prepare_locked_reference_claim_metric_adapter(
        self,
        *,
        adapter_spec: Dict[str, Any],
        campaign_id: str,
        current_algorithm_id: str,
        reference_algorithm_id: str,
        baseline_id: str,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        adapter = deepcopy(adapter_spec or {})
        if not adapter:
            return {}, {}
        kind = str(adapter.get("kind") or adapter.get("type") or "claim_metric_adapter").strip().lower()
        if kind not in {"claim_metric_adapter", "claim_metric_wrapper", "python_script", "dataset_metric_adapter"}:
            raise ValueError(
                "reference_claim_metric_adapter.kind must be 'claim_metric_adapter', 'claim_metric_wrapper', "
                "'python_script', or 'dataset_metric_adapter'."
            )
        adapter_root = (
            get_cellcompass_root()
            / "algorithm_campaigns"
            / str(current_algorithm_id)
            / str(campaign_id)
            / "locked_reference_claim_metric_adapters"
            / str(baseline_id)
            / str(reference_algorithm_id)
        )
        adapter_root.mkdir(parents=True, exist_ok=True)
        script_text = str(adapter.get("script") or "").strip()
        raw_script_path = str(adapter.get("adapter_path") or adapter.get("path") or adapter.get("script_path") or "").strip()
        script_path: Optional[Path] = None
        if script_text:
            script_path = adapter_root / "claim_metric_adapter.py"
            script_path.write_text(script_text + ("\n" if not script_text.endswith("\n") else ""), encoding="utf-8")
            adapter["adapter_path"] = str(script_path)
        elif raw_script_path:
            script_path = Path(raw_script_path).expanduser().resolve()
            if not script_path.is_file():
                raise FileNotFoundError(f"reference_claim_metric_adapter script not found: {script_path}")
            adapter["adapter_path"] = str(script_path)
        elif adapter.get("supported") is False or adapter.get("is_supported") is False:
            pass
        elif not any(key in adapter for key in ("metadata", "metric_params")):
            raise ValueError(
                "reference_claim_metric_adapter requires adapter_path/script_path, inline script, "
                "metadata/metric_params, or supported=false with a reason."
            )
        adapter.setdefault("adapter_function", "adapt_baseline_metric_context")
        adapter.setdefault("baseline_algorithm", self.planner_file_tools._normalize_baseline_identity(reference_algorithm_id))
        report = {
            "adapter_applied": True,
            "kind": kind,
            "reference_algorithm_id": reference_algorithm_id,
            "baseline_id": baseline_id,
            "adapter_path": str(script_path) if script_path is not None else "",
            "adapter_sha256": self._path_sha256(script_path) if script_path is not None else "",
            "function_name": str(adapter.get("adapter_function") or adapter.get("function_name") or ""),
            "version": str(adapter.get("version") or adapter.get("adapter_version") or ""),
            "supported": bool(adapter.get("supported", adapter.get("is_supported", True))),
            "reason": str(adapter.get("reason") or adapter.get("unsupported_reason") or "").strip(),
        }
        return adapter, report

    @staticmethod
    def _path_sha256(path: Path) -> str:
        try:
            return hashlib.sha256(Path(path).read_bytes()).hexdigest()
        except Exception:
            return ""

    def _apply_temporary_control_workspace_patch(
        self,
        *,
        algorithm_id: str,
        patch: str,
    ) -> Tuple[List[Dict[str, Any]], Callable[[], None]]:
        algo_id = str(algorithm_id or "").strip().lower()
        algo_root = (get_cellcompass_root() / "training_algorithms" / algo_id).resolve()
        if not algo_root.is_dir():
            raise FileNotFoundError(f"Training algorithm workspace not found: {algo_root}")
        ops = self.planner_file_tools._parse_patch(str(patch or ""))  # noqa: SLF001
        backups: Dict[Path, Tuple[bool, bytes]] = {}
        touched: List[Dict[str, Any]] = []

        def _target_path(raw_path: str) -> Path:
            raw = str(raw_path or "").strip()
            candidate = Path(raw).expanduser()
            if not candidate.is_absolute():
                candidate = algo_root / raw
            resolved = candidate.resolve()
            try:
                resolved.relative_to(algo_root)
            except ValueError as exc:
                raise ValueError(
                    f"control workspace patch path must stay inside {algo_root}: {raw_path}"
                ) from exc
            return resolved

        for op in ops:
            target = _target_path(op.path)
            if target not in backups:
                backups[target] = (target.exists(), target.read_bytes() if target.exists() else b"")
            before_sha = self._path_sha256(target) if target.exists() else ""
            if op.kind == "add":
                if target.exists():
                    raise ValueError(f"control patch cannot add existing file: {target}")
                target.parent.mkdir(parents=True, exist_ok=True)
                content = "\n".join(line[1:] for line in op.lines)
                if op.lines:
                    content += "\n"
                target.write_text(content, encoding="utf-8")
            elif op.kind == "delete":
                if target.exists():
                    target.unlink()
            else:
                if not target.exists():
                    raise ValueError(f"control patch cannot update missing file: {target}")
                original = target.read_text(encoding="utf-8")
                updated = self.planner_file_tools._apply_update_lines(original, op.lines)  # noqa: SLF001
                target.write_text(updated, encoding="utf-8")
            after_sha = self._path_sha256(target) if target.exists() else ""
            touched.append(
                {
                    "path": str(target),
                    "operation": op.kind,
                    "sha256_before": before_sha,
                    "sha256_after": after_sha,
                }
            )

        def _restore() -> None:
            for path, (existed, content) in reversed(list(backups.items())):
                try:
                    if existed:
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(content)
                    elif path.exists():
                        path.unlink()
                except Exception:
                    logger.exception("Failed to restore control workspace patch path %s", path)

        return touched, _restore

    def run_campaign_control_baseline(
        self,
        campaign_id: str = "",
        baseline_id: str = "",
        baseline_name: str = "",
        baseline_type: str = "ablation",
        stage: str = "",
        control_code_source: Optional[Dict[str, Any]] = None,
        metric_roles: Optional[List[str]] = None,
        required_for_gate: bool = True,
        replace_existing: bool = True,
        dataset_config_overrides: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        reason: str = "",
    ) -> str:
        """Run and register an agent-declared control/ablation baseline.

        The control run uses the same frozen campaign stage panel as the candidate,
        applies only the declared control code source, and never promotes/rejects
        a campaign trial or changes active best.
        """
        campaign_id = str(campaign_id or self.state.get("active_algorithm_campaign_id") or "").strip()
        campaign = self.planner_file_tools.get_algorithm_campaign_status(campaign_id)
        self.planner_file_tools._ensure_campaign_mutation_matches_active_binding(
            campaign,
            requested_campaign_id=campaign_id,
            previously_bound_campaign_id=str(self.state.get("active_algorithm_campaign_id") or "").strip(),
            action="run a control baseline for",
        )
        target_stage = str(stage or campaign.get("current_stage") or "stage1_feasibility").strip()
        if target_stage == "final_regression":
            return json.dumps(
                {
                    "status": "blocked",
                    "reason": "final_regression_confirmation_only",
                    "message": "Control baseline runs are not tuning/final-regression actions; register prior-stage controls before final regression.",
                },
                ensure_ascii=False,
                indent=2,
            )
        algo_id = str(campaign.get("algorithm_id") or "").strip().lower()
        name = str(baseline_name or baseline_id or "").strip()
        if not name:
            raise ValueError("baseline_name or baseline_id is required.")
        safe_id = self.planner_file_tools._normalize_baseline_identity(baseline_id or name)  # noqa: SLF001
        code_source = self.planner_file_tools._normalize_control_code_source(dict(control_code_source or {}))  # noqa: SLF001
        dataset_payload = self.planner_file_tools.resolve_campaign_trial_dataset_payload(
            campaign_id,
            dict(dataset_config_overrides or {}),
        )
        raw_datasets = dataset_payload.get("datasets")
        datasets = raw_datasets if isinstance(raw_datasets, list) and raw_datasets else [None]
        common_overrides = (
            dataset_payload.get("config_overrides")
            if isinstance(dataset_payload.get("config_overrides"), dict)
            else {}
        )
        control_overrides = code_source.get("config_overrides") if isinstance(code_source.get("config_overrides"), dict) else {}
        patch_text = str(code_source.get("patch") or code_source.get("workspace_patch") or "").strip()
        patch_path = str(code_source.get("patch_path") or "").strip()
        if patch_path and not patch_text:
            patch_text = Path(patch_path).expanduser().read_text(encoding="utf-8")
            code_source.setdefault("patch_sha256", hashlib.sha256(patch_text.encode("utf-8")).hexdigest())
        patch_restore: Optional[Callable[[], None]] = None
        patch_touched: List[Dict[str, Any]] = []
        run_ids: List[str] = []
        run_metrics: List[Dict[str, Any]] = []
        config_reports: List[Dict[str, Any]] = []
        result_lines: List[str] = []
        stage_policy = dict((campaign.get("stage_policies") or {}).get(target_stage) or {})
        training_stage = str(stage_policy.get("training_stage") or "pilot")
        claim_metric_run_spec = self._campaign_claim_metric_run_spec(campaign)

        try:
            if patch_text:
                patch_touched, patch_restore = self._apply_temporary_control_workspace_patch(
                    algorithm_id=algo_id,
                    patch=patch_text,
                )
                code_source["patched_workspace_paths"] = patch_touched
            for idx, dataset_entry in enumerate(datasets):
                entry = dict(dataset_entry or {})
                dataset_id = str(entry.get("dataset_id") or entry.get("id") or f"dataset{idx + 1}")
                entry_overrides = entry.get("config_overrides") if isinstance(entry.get("config_overrides"), dict) else entry
                config_overrides = self._merge_control_config_overrides(dict(common_overrides or {}), dict(entry_overrides or {}))
                config_overrides = self._merge_control_config_overrides(config_overrides, dict(control_overrides or {}))
                if claim_metric_spec_has_evaluator(claim_metric_run_spec):
                    config_overrides["__campaign_claim_metric_spec"] = deepcopy(claim_metric_run_spec)
                config_report = self._campaign_effective_config_report(
                    algo_id=algo_id,
                    training_stage=training_stage,
                    dataset_id=dataset_id,
                    adata_path=str(entry.get("adata_path") or "") or str(self.state.get("preprocessed_path") or self.state.get("input_path") or ""),
                    config_overrides=config_overrides,
                )
                before_run_id = str(self.state.get("latest_training_run_id") or "")
                training_result = self._run_training_tool_impl(
                    training_algorithm_id=algo_id,
                    stage=training_stage,
                    config_overrides=config_overrides,
                    run_label=f"control-{safe_id[:18]}-{dataset_id[:12]}",
                    adata_path=str(entry.get("adata_path") or "") or None,
                    seed=seed,
                    decision="provisional",
                    decision_reason=reason or f"Control baseline run for {safe_id}; not a campaign trial.",
                    review_purpose="campaign_control_baseline",
                    review_campaign=campaign,
                    _skip_review_gates=True,
                )
                after_run_id = str(self.state.get("latest_training_run_id") or "")
                run_id = after_run_id if after_run_id and after_run_id != before_run_id else ""
                if run_id:
                    run_ids.append(run_id)
                    config_report["run_id"] = run_id
                metrics = self._campaign_latest_training_metrics(run_id)
                if not metrics and str(training_result).startswith("Training failed"):
                    metrics = {"error": str(training_result)}
                if run_id:
                    metrics["run_id"] = run_id
                metrics["campaign_dataset_id"] = dataset_id
                resolved_config_path = str(
                    metrics.get("resolved_config_path")
                    or ((metrics.get("artifacts") or {}) if isinstance(metrics.get("artifacts"), dict) else {}).get("resolved_config_path")
                    or ""
                ).strip()
                if resolved_config_path:
                    config_report["resolved_config_path"] = resolved_config_path
                    config_report["resolved_config_sha256"] = self._path_sha256(Path(resolved_config_path))
                if config_report:
                    metrics["effective_config_report"] = config_report
                    config_reports.append(config_report)
                run_metrics.append(metrics)
                result_lines.extend([f"[{dataset_id}] {line}" for line in str(training_result).splitlines()[:6]])
        finally:
            if patch_restore is not None:
                patch_restore()

        errors = [str(item.get("error") or "").strip() for item in run_metrics if str(item.get("error") or "").strip()]
        aggregate: Dict[str, Any] = {
            "campaign_run_ids": run_ids,
            "campaign_dataset_metrics": run_metrics,
            "control_baseline": True,
            "control_baseline_id": safe_id,
        }
        if config_reports:
            aggregate["campaign_effective_config_reports"] = config_reports
        w1_scores: List[Any] = []
        tmv_scores: List[Any] = []
        custom_values: Dict[str, List[float]] = {}
        claim_evaluators: List[Dict[str, Any]] = []
        for item in run_metrics:
            if isinstance(item.get("w1_scores"), list):
                w1_scores.extend(item.get("w1_scores") or [])
            elif item.get("w1_mean") is not None:
                w1_scores.append(item.get("w1_mean"))
            if isinstance(item.get("tmv_scores"), list):
                tmv_scores.extend(item.get("tmv_scores") or [])
            custom = item.get("custom_metrics")
            if isinstance(custom, dict):
                for key, value in custom.items():
                    try:
                        custom_values.setdefault(str(key), []).append(float(value))
                    except Exception:
                        continue
            evaluator_payload = item.get("claim_metric_evaluator")
            if isinstance(evaluator_payload, dict) and evaluator_payload:
                identity = self._claim_metric_evaluator_identity(evaluator_payload)
                if identity not in [self._claim_metric_evaluator_identity(existing) for existing in claim_evaluators]:
                    claim_evaluators.append(dict(evaluator_payload))
        if errors:
            return json.dumps(
                {
                    "status": "failed",
                    "reason": "control_baseline_training_failed",
                    "campaign_id": campaign_id,
                    "stage": target_stage,
                    "baseline_id": safe_id,
                    "errors": errors,
                    "training_result_summary": result_lines[:18],
                    "message": "Control baseline was not registered because one or more control runs failed.",
                },
                ensure_ascii=False,
                indent=2,
            )
        w1_metadata = self.planner_file_tools._consistent_w1_backend_metadata(  # noqa: SLF001
            [item for item in run_metrics if isinstance(item, dict)]
        )
        if w1_metadata:
            aggregate.update(w1_metadata)
        if w1_scores:
            aggregate["w1_scores"] = w1_scores
            aggregate["w1_mean"] = sum(float(x) for x in w1_scores) / len(w1_scores)
        if tmv_scores:
            aggregate["tmv_scores"] = tmv_scores
            aggregate["tmv_mean"] = sum(float(x) for x in tmv_scores) / len(tmv_scores)
        if custom_values:
            aggregate["custom_metrics"] = {
                key: sum(values) / len(values)
                for key, values in custom_values.items()
                if values
            }
        if len(claim_evaluators) == 1:
            aggregate["claim_metric_evaluator"] = claim_evaluators[0]
        elif len(claim_evaluators) > 1:
            aggregate["claim_metric_evaluator_mismatch"] = [
                self._claim_metric_evaluator_identity(item)
                for item in claim_evaluators
            ]
        target_dataset_ids = self.planner_file_tools._campaign_target_dataset_ids(dataset_payload)  # noqa: SLF001
        if target_dataset_ids:
            aggregate["target_dataset_ids"] = target_dataset_ids
        source_artifacts = {
            "run_ids": ",".join(run_ids),
            "config_reports": json.dumps(config_reports, ensure_ascii=False),
        }
        if patch_touched:
            source_artifacts["patched_workspace_paths"] = json.dumps(patch_touched, ensure_ascii=False)
        register_kwargs = {
            "baseline_id": safe_id,
            "baseline_name": name,
            "baseline_type": baseline_type,
            "stage": target_stage,
            "metrics": aggregate,
            "target_dataset_ids": target_dataset_ids,
            "metric_roles": list(metric_roles or ["both"]),
            "required_for_gate": bool(required_for_gate),
            "source_run_id": run_ids[0] if run_ids else "",
            "source_metrics_path": str((run_metrics[0] or {}).get("metrics_path") or "") if run_metrics else "",
            "source_config_path": str((run_metrics[0] or {}).get("resolved_config_path") or "") if run_metrics else "",
            "source_artifact_paths": source_artifacts,
            "control_code_source": {
                **dict(code_source or {}),
                "kind": "run_campaign_control_baseline",
                "declared_control_kind": str(code_source.get("kind") or ""),
                "campaign_control_run_ids": run_ids,
            },
            "reason": reason,
        }
        try:
            registration = self.planner_file_tools.register_campaign_control_baseline(campaign_id, **register_kwargs)
        except ValueError as exc:
            if not replace_existing or "already registered" not in str(exc):
                raise
            update_kwargs = {
                "baseline_id": safe_id,
                "stage": target_stage,
                "metrics": aggregate,
                "target_dataset_ids": target_dataset_ids,
                "metric_roles": list(metric_roles or ["both"]),
                "required_for_gate": bool(required_for_gate),
                "source_run_id": run_ids[0] if run_ids else "",
                "source_metrics_path": str((run_metrics[0] or {}).get("metrics_path") or "") if run_metrics else "",
                "source_config_path": str((run_metrics[0] or {}).get("resolved_config_path") or "") if run_metrics else "",
                "source_artifact_paths": source_artifacts,
                "control_code_source": register_kwargs["control_code_source"],
                "reason": reason,
            }
            registration = self.planner_file_tools.update_campaign_control_baseline(
                campaign_id,
                action="replace",
                **update_kwargs,
            )
        payload = {
            "status": "registered",
            "campaign_id": campaign_id,
            "stage": target_stage,
            "baseline_id": safe_id,
            "run_ids": run_ids,
            "metrics": aggregate,
            "registration": registration,
            "training_result_summary": result_lines[:18],
            "message": (
                "Control baseline ran on the frozen stage panel and was registered as external gate evidence. "
                "It did not create, promote, reject, or modify a campaign trial active best."
            ),
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def run_campaign_locked_algorithm_baseline(
        self,
        campaign_id: str = "",
        reference_algorithm_id: str = "",
        reference_campaign_id: str = "",
        baseline_id: str = "",
        baseline_name: str = "",
        baseline_type: str = "locked_algorithm",
        stage: str = "",
        metric_roles: Optional[List[str]] = None,
        required_for_gate: bool = True,
        replace_existing: bool = True,
        dataset_config_overrides: Optional[Dict[str, Any]] = None,
        reference_config_overrides: Optional[Dict[str, Any]] = None,
        reference_dataset_adapter: Optional[Dict[str, Any]] = None,
        reference_claim_metric_adapter: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        reason: str = "",
    ) -> str:
        """Run a completed locked custom algorithm as a same-panel baseline.

        This intentionally reuses the registered-baseline gate path, but the
        comparator source is another final-regression locked algorithm rather
        than a patch/config ablation of the current workspace.
        """
        campaign_id = str(campaign_id or self.state.get("active_algorithm_campaign_id") or "").strip()
        campaign = self.planner_file_tools.get_algorithm_campaign_status(campaign_id)
        self.planner_file_tools._ensure_campaign_mutation_matches_active_binding(
            campaign,
            requested_campaign_id=campaign_id,
            previously_bound_campaign_id=str(self.state.get("active_algorithm_campaign_id") or "").strip(),
            action="run a locked algorithm baseline for",
        )
        target_stage = str(stage or campaign.get("current_stage") or "stage1_feasibility").strip()
        if target_stage == "final_regression":
            return json.dumps(
                {
                    "status": "blocked",
                    "reason": "final_regression_confirmation_only",
                    "message": "Locked algorithm baselines are registered for prior-stage gates, not during final regression.",
                },
                ensure_ascii=False,
                indent=2,
            )

        current_algo_id = str(campaign.get("algorithm_id") or "").strip().lower()
        raw_reference_id = str(reference_algorithm_id or "").strip()
        reference_algo_id = self.planner_file_tools._normalize_baseline_identity(raw_reference_id)  # noqa: SLF001
        if not reference_algo_id:
            raise ValueError("reference_algorithm_id is required.")
        if reference_algo_id == current_algo_id:
            raise ValueError("reference_algorithm_id must name a different completed algorithm, not the active campaign algorithm.")

        state_restore = {
            "active_algorithm_context": deepcopy(self.state.get("active_algorithm_context") or {}),
            "active_experiment_registry": deepcopy(self.state.get("active_experiment_registry") or {}),
            "planner_algorithm_workspace": str(self.state.get("planner_algorithm_workspace") or ""),
            "latest_training_algorithm_id": str(self.state.get("latest_training_algorithm_id") or ""),
            "active_algorithm_campaign_id": str(self.state.get("active_algorithm_campaign_id") or ""),
            "active_algorithm_campaign": deepcopy(self.state.get("active_algorithm_campaign") or {}),
        }

        reference_registry = self.planner_file_tools._bootstrap_algorithm_registry(reference_algo_id)  # noqa: SLF001
        lifecycle_status = str(reference_registry.get("algorithm_lifecycle_status") or "").strip().lower()
        completed_release = (
            reference_registry.get("completed_release")
            if isinstance(reference_registry.get("completed_release"), dict)
            else {}
        )
        completed_campaign_id = str(reference_registry.get("completed_campaign_id") or "").strip()
        if lifecycle_status not in {"complete", "completed", "locked", "final"} or not completed_release:
            raise ValueError(
                f"Reference algorithm '{reference_algo_id}' is not a completed final-regression locked release "
                f"(status={lifecycle_status or 'missing'}, completed_release={'present' if completed_release else 'missing'})."
            )

        requested_reference_campaign = str(reference_campaign_id or "").strip()
        audit_reference_campaign_id = requested_reference_campaign or completed_campaign_id
        reference_campaign: Dict[str, Any] = {}
        if requested_reference_campaign:
            reference_campaign = self.planner_file_tools._load_campaign(requested_reference_campaign)  # noqa: SLF001
            if str(reference_campaign.get("algorithm_id") or "").strip().lower() != reference_algo_id:
                raise ValueError(
                    f"reference_campaign_id '{requested_reference_campaign}' belongs to "
                    f"algorithm '{reference_campaign.get('algorithm_id')}', not '{reference_algo_id}'."
                )
            reference_locked = reference_campaign.get("locked_release")
            if not isinstance(reference_locked, dict) or not reference_locked:
                raise ValueError(f"reference_campaign_id '{requested_reference_campaign}' is not final-regression locked.")
            for key, value in state_restore.items():
                self.state[key] = value

        locked_snapshot_id = str(completed_release.get("snapshot_id") or "").strip()
        active_snapshot_id = str(reference_registry.get("active_workspace_snapshot_id") or "").strip()
        if locked_snapshot_id and active_snapshot_id and locked_snapshot_id != active_snapshot_id:
            raise ValueError(
                f"Reference algorithm '{reference_algo_id}' active workspace snapshot '{active_snapshot_id}' does not "
                f"match locked release snapshot '{locked_snapshot_id}'. Refusing to run a possibly drifted baseline."
            )

        name = str(baseline_name or raw_reference_id or reference_algo_id).strip()
        safe_id = self.planner_file_tools._normalize_baseline_identity(  # noqa: SLF001
            baseline_id or f"{reference_algo_id}_locked_reference"
        )
        dataset_payload = self.planner_file_tools.resolve_campaign_trial_dataset_payload(
            campaign_id,
            dict(dataset_config_overrides or {}),
        )
        raw_datasets = dataset_payload.get("datasets")
        datasets = raw_datasets if isinstance(raw_datasets, list) and raw_datasets else [None]
        common_overrides = (
            dataset_payload.get("config_overrides")
            if isinstance(dataset_payload.get("config_overrides"), dict)
            else {}
        )
        if reference_config_overrides:
            raise ValueError(
                "reference_config_overrides are disabled for locked algorithm baselines because they guess at "
                "format compatibility and can silently retune semantics. Provide reference_dataset_adapter with "
                "a Python script that materializes an adapted h5ad instead."
            )
        adapter_spec = dict(reference_dataset_adapter or {})
        claim_adapter_spec, claim_adapter_report = self._prepare_locked_reference_claim_metric_adapter(
            adapter_spec=dict(reference_claim_metric_adapter or {}),
            campaign_id=campaign_id,
            current_algorithm_id=current_algo_id,
            reference_algorithm_id=reference_algo_id,
            baseline_id=safe_id,
        )
        adapter_reports: List[Dict[str, Any]] = []
        run_ids: List[str] = []
        run_metrics: List[Dict[str, Any]] = []
        config_reports: List[Dict[str, Any]] = []
        result_lines: List[str] = []
        stage_policy = dict((campaign.get("stage_policies") or {}).get(target_stage) or {})
        training_stage = str(stage_policy.get("training_stage") or "pilot")
        claim_metric_run_spec = self._campaign_claim_metric_run_spec(campaign)
        if claim_metric_spec_has_evaluator(claim_metric_run_spec):
            claim_metric_run_spec["_baseline_algorithm_id"] = reference_algo_id
            if claim_adapter_spec:
                raw_adapters = claim_metric_run_spec.get("baseline_metric_adapters")
                adapters = deepcopy(raw_adapters) if isinstance(raw_adapters, dict) else {}
                adapters[reference_algo_id] = deepcopy(claim_adapter_spec)
                if raw_reference_id and raw_reference_id != reference_algo_id:
                    adapters[raw_reference_id] = deepcopy(claim_adapter_spec)
                claim_metric_run_spec["baseline_metric_adapters"] = adapters

        try:
            for idx, dataset_entry in enumerate(datasets):
                entry = dict(dataset_entry or {})
                dataset_id = str(entry.get("dataset_id") or entry.get("id") or f"dataset{idx + 1}")
                entry_overrides = entry.get("config_overrides") if isinstance(entry.get("config_overrides"), dict) else entry
                config_overrides = self._merge_control_config_overrides(dict(common_overrides or {}), dict(entry_overrides or {}))
                if claim_metric_spec_has_evaluator(claim_metric_run_spec):
                    config_overrides["__campaign_claim_metric_spec"] = deepcopy(claim_metric_run_spec)
                input_adata_path = str(entry.get("adata_path") or "") or str(self.state.get("preprocessed_path") or self.state.get("input_path") or "")
                adapter_report = self._run_locked_reference_dataset_adapter(
                    adapter_spec=adapter_spec,
                    campaign_id=campaign_id,
                    current_algorithm_id=current_algo_id,
                    reference_algorithm_id=reference_algo_id,
                    baseline_id=safe_id,
                    stage=target_stage,
                    dataset_id=dataset_id,
                    input_adata_path=input_adata_path,
                )
                if bool(adapter_report.get("adapter_applied")):
                    adapter_reports.append(adapter_report)
                reference_adata_path = str(adapter_report.get("adapted_adata_path") or input_adata_path)
                config_report = self._campaign_effective_config_report(
                    algo_id=reference_algo_id,
                    training_stage=training_stage,
                    dataset_id=dataset_id,
                    adata_path=reference_adata_path,
                    config_overrides=config_overrides,
                )
                before_run_id = str(self.state.get("latest_training_run_id") or "")
                training_result = self._run_training_tool_impl(
                    training_algorithm_id=reference_algo_id,
                    stage=training_stage,
                    config_overrides=config_overrides,
                    run_label=f"lockedref-{safe_id[:16]}-{dataset_id[:12]}",
                    adata_path=reference_adata_path or None,
                    seed=seed,
                    decision="provisional",
                    decision_reason=reason or f"Locked algorithm baseline run for {safe_id}; not a campaign trial.",
                    review_purpose="campaign_locked_algorithm_baseline",
                    review_campaign=campaign,
                    _skip_review_gates=True,
                    _record_experiment_registry=False,
                )
                after_run_id = str(self.state.get("latest_training_run_id") or "")
                run_id = after_run_id if after_run_id and after_run_id != before_run_id else ""
                if run_id:
                    run_ids.append(run_id)
                    config_report["run_id"] = run_id
                metrics = self._campaign_latest_training_metrics(run_id)
                if not metrics and str(training_result).startswith("Training failed"):
                    metrics = {"error": str(training_result)}
                if run_id:
                    metrics["run_id"] = run_id
                metrics["campaign_dataset_id"] = dataset_id
                resolved_config_path = str(
                    metrics.get("resolved_config_path")
                    or ((metrics.get("artifacts") or {}) if isinstance(metrics.get("artifacts"), dict) else {}).get("resolved_config_path")
                    or ""
                ).strip()
                if resolved_config_path:
                    config_report["resolved_config_path"] = resolved_config_path
                    config_report["resolved_config_sha256"] = self._path_sha256(Path(resolved_config_path))
                if config_report:
                    metrics["effective_config_report"] = config_report
                    config_reports.append(config_report)
                run_metrics.append(metrics)
                result_lines.extend([f"[{dataset_id}] {line}" for line in str(training_result).splitlines()[:6]])
        finally:
            for key, value in state_restore.items():
                self.state[key] = value

        errors = [str(item.get("error") or "").strip() for item in run_metrics if str(item.get("error") or "").strip()]
        aggregate: Dict[str, Any] = {
            "campaign_run_ids": run_ids,
            "campaign_dataset_metrics": run_metrics,
            "control_baseline": True,
            "control_baseline_id": safe_id,
            "locked_algorithm_reference": True,
            "reference_algorithm_id": reference_algo_id,
            "reference_campaign_id": audit_reference_campaign_id,
            "reference_locked_release": deepcopy(completed_release),
        }
        if config_reports:
            aggregate["campaign_effective_config_reports"] = config_reports
        if adapter_reports:
            aggregate["reference_dataset_adapter_reports"] = adapter_reports
        if claim_adapter_report:
            aggregate["reference_claim_metric_adapter_report"] = claim_adapter_report
        w1_scores: List[Any] = []
        tmv_scores: List[Any] = []
        custom_values: Dict[str, List[float]] = {}
        claim_evaluators: List[Dict[str, Any]] = []
        for item in run_metrics:
            if isinstance(item.get("w1_scores"), list):
                w1_scores.extend(item.get("w1_scores") or [])
            elif item.get("w1_mean") is not None:
                w1_scores.append(item.get("w1_mean"))
            if isinstance(item.get("tmv_scores"), list):
                tmv_scores.extend(item.get("tmv_scores") or [])
            custom = item.get("custom_metrics")
            if isinstance(custom, dict):
                for key, value in custom.items():
                    try:
                        custom_values.setdefault(str(key), []).append(float(value))
                    except Exception:
                        continue
            evaluator_payload = item.get("claim_metric_evaluator")
            if isinstance(evaluator_payload, dict) and evaluator_payload:
                identity = self._claim_metric_evaluator_identity(evaluator_payload)
                if identity not in [self._claim_metric_evaluator_identity(existing) for existing in claim_evaluators]:
                    claim_evaluators.append(dict(evaluator_payload))
        if errors:
            return json.dumps(
                {
                    "status": "failed",
                    "reason": "locked_algorithm_baseline_training_failed",
                    "campaign_id": campaign_id,
                    "stage": target_stage,
                    "baseline_id": safe_id,
                    "reference_algorithm_id": reference_algo_id,
                    "errors": errors,
                    "training_result_summary": result_lines[:18],
                    "message": "Locked algorithm baseline was not registered because one or more runs failed.",
                },
                ensure_ascii=False,
                indent=2,
            )
        w1_metadata = self.planner_file_tools._consistent_w1_backend_metadata(  # noqa: SLF001
            [item for item in run_metrics if isinstance(item, dict)]
        )
        if w1_metadata:
            aggregate.update(w1_metadata)
        if w1_scores:
            aggregate["w1_scores"] = w1_scores
            aggregate["w1_mean"] = sum(float(x) for x in w1_scores) / len(w1_scores)
        if tmv_scores:
            aggregate["tmv_scores"] = tmv_scores
            aggregate["tmv_mean"] = sum(float(x) for x in tmv_scores) / len(tmv_scores)
        if custom_values:
            aggregate["custom_metrics"] = {
                key: sum(values) / len(values)
                for key, values in custom_values.items()
                if values
            }
        if len(claim_evaluators) == 1:
            aggregate["claim_metric_evaluator"] = claim_evaluators[0]
        elif len(claim_evaluators) > 1:
            aggregate["claim_metric_evaluator_mismatch"] = [
                self._claim_metric_evaluator_identity(item)
                for item in claim_evaluators
            ]
        claim_metric_name = str(
            (claim_metric_run_spec or {}).get("name")
            or (claim_metric_run_spec or {}).get("metric_name")
            or (claim_metric_run_spec or {}).get("primary_metric")
            or "claim_metric"
        ).strip()
        if (
            claim_metric_spec_has_evaluator(claim_metric_run_spec)
            and self._metric_roles_require_claim_metric(metric_roles, claim_metric_name)
        ):
            _, expected_claim_evaluator = load_campaign_claim_metric_evaluator(
                claim_metric_run_spec,
                baseline_algorithm=reference_algo_id,
            )
            if not self._claim_metric_evaluator_matches(
                aggregate,
                expected_claim_evaluator,
                claim_metric_name=claim_metric_name,
            ) or not self._claim_metric_baseline_adapter_matches(
                aggregate.get("claim_metric_evaluator"),
                expected_claim_evaluator,
            ):
                return json.dumps(
                    {
                        "status": "failed",
                        "reason": "locked_algorithm_claim_metric_incompatible",
                        "campaign_id": campaign_id,
                        "stage": target_stage,
                        "baseline_id": safe_id,
                        "reference_algorithm_id": reference_algo_id,
                        "claim_metric_name": claim_metric_name,
                        "observed_claim_metric_evaluator": aggregate.get("claim_metric_evaluator"),
                        "observed_claim_metric_evaluator_mismatch": aggregate.get("claim_metric_evaluator_mismatch"),
                        "message": (
                            "Locked reference baseline was not registered because it did not produce the current "
                            "campaign claim metric with matching claim_metric_evaluator provenance. Use metric_roles=['w1'] "
                            "for a W1-only comparator, or provide explicit reference_dataset_adapter and/or "
                            "reference_claim_metric_adapter wrappers so the old algorithm can be evaluated under the "
                            "current campaign claim metric."
                        ),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
        target_dataset_ids = self.planner_file_tools._campaign_target_dataset_ids(dataset_payload)  # noqa: SLF001
        if target_dataset_ids:
            aggregate["target_dataset_ids"] = target_dataset_ids
        source_artifacts = {
            "run_ids": ",".join(run_ids),
            "config_reports": json.dumps(config_reports, ensure_ascii=False),
            "reference_dataset_adapter_reports": json.dumps(adapter_reports, ensure_ascii=False),
            "reference_claim_metric_adapter_report": json.dumps(claim_adapter_report, ensure_ascii=False),
            "reference_completed_campaign_id": completed_campaign_id,
            "reference_requested_campaign_id": requested_reference_campaign,
            "reference_locked_release": json.dumps(completed_release, ensure_ascii=False),
        }
        register_kwargs = {
            "baseline_id": safe_id,
            "baseline_name": name,
            "baseline_type": baseline_type,
            "stage": target_stage,
            "metrics": aggregate,
            "target_dataset_ids": target_dataset_ids,
            "metric_roles": list(metric_roles or ["both"]),
            "required_for_gate": bool(required_for_gate),
            "source_run_id": run_ids[0] if run_ids else "",
            "source_metrics_path": str((run_metrics[0] or {}).get("metrics_path") or "") if run_metrics else "",
            "source_config_path": str((run_metrics[0] or {}).get("resolved_config_path") or "") if run_metrics else "",
            "source_artifact_paths": source_artifacts,
            "control_code_source": {
                "kind": "locked_algorithm_reference",
                "reference_algorithm_id": reference_algo_id,
                "reference_campaign_id": audit_reference_campaign_id,
                "reference_locked_release": deepcopy(completed_release),
                "reference_dataset_adapter": deepcopy(adapter_spec),
                "reference_dataset_adapter_reports": adapter_reports,
                "reference_claim_metric_adapter": deepcopy(claim_adapter_spec),
                "reference_claim_metric_adapter_report": claim_adapter_report,
                "campaign_locked_algorithm_run_ids": run_ids,
            },
            "reason": reason,
        }
        try:
            registration = self.planner_file_tools.register_campaign_control_baseline(campaign_id, **register_kwargs)
        except ValueError as exc:
            if not replace_existing or "already registered" not in str(exc):
                raise
            update_kwargs = {
                "baseline_id": safe_id,
                "stage": target_stage,
                "metrics": aggregate,
                "target_dataset_ids": target_dataset_ids,
                "metric_roles": list(metric_roles or ["both"]),
                "required_for_gate": bool(required_for_gate),
                "source_run_id": run_ids[0] if run_ids else "",
                "source_metrics_path": str((run_metrics[0] or {}).get("metrics_path") or "") if run_metrics else "",
                "source_config_path": str((run_metrics[0] or {}).get("resolved_config_path") or "") if run_metrics else "",
                "source_artifact_paths": source_artifacts,
                "control_code_source": register_kwargs["control_code_source"],
                "reason": reason,
            }
            registration = self.planner_file_tools.update_campaign_control_baseline(
                campaign_id,
                action="replace",
                **update_kwargs,
            )
        payload = {
            "status": "registered",
            "campaign_id": campaign_id,
            "stage": target_stage,
            "baseline_id": safe_id,
            "reference_algorithm_id": reference_algo_id,
            "reference_campaign_id": audit_reference_campaign_id,
            "run_ids": run_ids,
            "metrics": aggregate,
            "registration": registration,
            "training_result_summary": result_lines[:18],
            "message": (
                "Completed locked algorithm ran on the current frozen stage panel and was registered as external "
                "gate evidence. It did not create, promote, reject, or modify a campaign trial active best."
            ),
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def compute_campaign_claim_metric_for_baselines(
        self,
        campaign_id: str = "",
        stage: str = "stage2_claim_validation",
        baseline_algorithms: Optional[List[str]] = None,
        use_saved_trajectory: bool = True,
        regenerate_trajectory_if_missing: bool = False,
        baseline_type: str = "builtin",
    ) -> str:
        """Explicit alias for Stage 2 baseline claim-metric recomputation.

        The implementation intentionally reuses refresh_campaign_stage_baselines
        so the stage baseline selection and structured blocked states stay in one
        code path. Existing baseline trajectories/models are reused when possible.
        """
        del use_saved_trajectory  # refresh_campaign_stage_baselines already prefers saved artifacts.
        return self.refresh_campaign_stage_baselines(
            campaign_id=campaign_id,
            stage=stage,
            baseline_algorithms=baseline_algorithms or [],
            run_missing=bool(regenerate_trajectory_if_missing),
            overwrite_existing=False,
            baseline_type=baseline_type,
        )

    def query_campaign_baseline_metrics(
        self,
        campaign_id: str = "",
        stage: str = "",
        dataset_id: str = "",
        metric_names: Optional[List[str]] = None,
        output_dir: str = "",
        include_training_run_scan: bool = True,
        include_benchmark_registry: bool = True,
    ) -> str:
        """Read-only evidence query for actual campaign baseline metrics.

        This tool is intentionally separate from refresh_campaign_stage_baselines:
        refresh selects the gate comparator, while this query gives paper writers
        and reviewers all available measured baseline numbers with provenance.
        """
        campaign_id = str(campaign_id or self.state.get("active_algorithm_campaign_id") or "").strip()
        if not campaign_id:
            return json.dumps(
                {
                    "status": "blocked",
                    "reason": "campaign_id_required",
                    "message": "Provide campaign_id or set active_algorithm_campaign_id before querying baseline metrics.",
                },
                ensure_ascii=False,
                indent=2,
            )
        try:
            campaign = self.planner_file_tools.get_algorithm_campaign_status(campaign_id=campaign_id)
        except Exception as exc:
            return json.dumps(
                {"status": "blocked", "reason": "campaign_load_failed", "campaign_id": campaign_id, "error": str(exc)},
                ensure_ascii=False,
                indent=2,
            )

        stages = campaign.get("stages") if isinstance(campaign.get("stages"), dict) else {}
        requested_stage = str(stage or "").strip()
        if requested_stage:
            stage_names = [requested_stage] if requested_stage in stages else []
        else:
            stage_names = [name for name in stages.keys() if str(name or "").strip()]
        if not stage_names:
            return json.dumps(
                {
                    "status": "blocked",
                    "reason": "unknown_stage" if requested_stage else "no_campaign_stages",
                    "campaign_id": campaign_id,
                    "requested_stage": requested_stage,
                    "available_stages": list(stages.keys()),
                },
                ensure_ascii=False,
                indent=2,
            )

        requested_metrics = [str(item or "").strip() for item in list(metric_names or []) if str(item or "").strip()]
        claim_spec = campaign.get("claim_metric_spec") if isinstance(campaign.get("claim_metric_spec"), dict) else {}
        claim_metric_name = str(claim_spec.get("name") or claim_spec.get("metric_name") or "").strip()
        dataset_filter = str(dataset_id or "").strip()
        scan_root = Path(str(output_dir or self.state.get("output_dir") or "cytobridge_output")).expanduser()

        def _metric_value(metrics: Dict[str, Any], metric_name: str) -> Optional[float]:
            return self.planner_file_tools._metric_value(metrics, metric_name)

        def _stage_from_run_id(run_id: str) -> str:
            text = str(run_id or "")
            mapping = {
                "stage1_": "stage1_feasibility",
                "stage1": "stage1_feasibility",
                "stage2_": "stage2_claim_validation",
                "stage2": "stage2_claim_validation",
                "stage3_": "stage3_tuning",
                "stage3": "stage3_tuning",
                "final_": "final_regression",
                "final": "final_regression",
            }
            lower = text.lower()
            for token, stage_name in mapping.items():
                if token in lower:
                    return stage_name
            return str(metrics_stage_from_run_id_fallback(text) or "")

        def metrics_stage_from_run_id_fallback(run_id: str) -> str:
            if "__final__" in run_id:
                return "final_regression"
            return ""

        def _algorithm_from_run_id(run_id: str, metrics: Optional[Dict[str, Any]] = None) -> str:
            payload = metrics if isinstance(metrics, dict) else {}
            for key in ("algorithm_name", "candidate_name", "algorithm_id", "base_config_name"):
                value = str(payload.get(key) or "").strip()
                if value and value not in {"builtin", "custom"}:
                    return self._normalize_baseline_algorithm_name(value)
            match = re.search(r"__(?:pilot|final)__([^_][^/]*)__campaign", str(run_id or ""))
            if match:
                return self._normalize_baseline_algorithm_name(match.group(1))
            match = re.search(r"-baseline-([A-Za-z0-9_.-]+)-", str(run_id or ""))
            if match:
                return self._normalize_baseline_algorithm_name(match.group(1))
            return ""

        def _compact_custom_metrics(custom: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str], int]:
            """Keep table-facing scalar metrics while bounding diagnostic payloads.

            Custom evaluators often attach per-interval arrays, bootstrap samples,
            or other nested diagnostics. Returning those blobs for every baseline
            can consume the entire model context and make the agent repeat the
            query after compaction. Full details remain available at metrics_path.
            """
            compact: Dict[str, Any] = {}
            omitted_keys: List[str] = []
            max_scalar_items = 128
            max_string_chars = 512
            for raw_key, value in custom.items():
                key = str(raw_key)
                if isinstance(value, (bool, int, float)) or value is None:
                    if len(compact) < max_scalar_items:
                        compact[key] = value
                    else:
                        omitted_keys.append(key)
                    continue
                if isinstance(value, str):
                    if len(compact) < max_scalar_items and len(value) <= max_string_chars:
                        compact[key] = value
                    else:
                        omitted_keys.append(key)
                    continue
                omitted_keys.append(key)
            omitted_count = len(omitted_keys)
            return compact, omitted_keys[:32], omitted_count

        def _compact_audit_record(record: Any) -> Dict[str, Any]:
            row = dict(record or {}) if isinstance(record, dict) else {}
            reasons = [str(item)[:400] for item in list(row.get("reasons") or []) if str(item).strip()]
            return {
                "algorithm_name": str(row.get("algorithm_name") or ""),
                "audit_status": str(row.get("audit_status") or ""),
                "reasons": reasons[:3],
            }

        def _compact_metrics(metrics: Dict[str, Any], names: List[str]) -> Dict[str, Any]:
            out: Dict[str, Any] = {}
            for key in (
                "w1_mean",
                "tmv_mean",
                "tmv_max",
                "primary_metric",
                "primary_value",
                "secondary_metric",
                "secondary_value",
                "w1_backend",
                "w1_backend_requested",
                "w1_backend_exact",
                "w1_backend_params",
                "w1_backend_selection_scope",
                "w1_backend_selection_reason",
                "w1_backend_threshold_pair_count",
                "w1_backend_max_pair_count",
            ):
                if key in metrics:
                    out[key] = metrics.get(key)
            custom = metrics.get("custom_metrics")
            if isinstance(custom, dict):
                compact_custom, omitted_keys, omitted_count = _compact_custom_metrics(custom)
                out["custom_metrics"] = compact_custom
                if omitted_count:
                    out["custom_metrics_omitted_detail_count"] = omitted_count
                    out["custom_metrics_omitted_detail_keys"] = omitted_keys
                    out["custom_metrics_detail_note"] = (
                        "Nested or oversized evaluator diagnostics were omitted from this agent-facing summary; "
                        "read metrics_path for the full evidence payload."
                    )
            wanted = list(dict.fromkeys([item for item in names if item]))
            if wanted:
                out["requested_metrics"] = {
                    name: _metric_value(metrics, name)
                    for name in wanted
                    if _metric_value(metrics, name) is not None
                }
            return out

        def _target_dataset_ids(stage_state: Dict[str, Any]) -> List[str]:
            panel = stage_state.get("stage_panel") if isinstance(stage_state.get("stage_panel"), dict) else {}
            panel_payload = dict(panel.get("dataset_config_overrides") or {})
            ids = self.planner_file_tools._campaign_target_dataset_ids(panel_payload)
            if dataset_filter:
                ids = [item for item in ids if item == dataset_filter]
            return ids

        def _load_json_path(path: Path) -> Dict[str, Any]:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return {}
            return payload if isinstance(payload, dict) else {}

        def _scan_training_runs() -> List[Dict[str, Any]]:
            if not include_training_run_scan:
                return []
            training_root = scan_root / "training_runs"
            if not training_root.exists():
                return []
            records: List[Dict[str, Any]] = []
            for metrics_path in sorted(training_root.glob("*/artifacts/metrics.json")):
                metrics = _load_json_path(metrics_path)
                if not metrics:
                    continue
                run_id = str(metrics.get("run_id") or metrics_path.parents[1].name)
                run_id_lower = run_id.lower()
                if "baseline" not in run_id_lower:
                    continue
                algo_name = _algorithm_from_run_id(run_id, metrics)
                stage_name = _stage_from_run_id(run_id)
                record = {
                    "source": "training_run_scan",
                    "algorithm_name": algo_name,
                    "baseline_type": "builtin" if algo_name in DEFAULT_CAMPAIGN_BUILTIN_BASELINES else "reference",
                    "stage": stage_name,
                    "run_id": run_id,
                    "metrics_path": str(metrics_path),
                    "run_dir": str(metrics_path.parents[1]),
                    "metrics": metrics,
                }
                records.append(record)
            return records

        scanned_records = _scan_training_runs()

        def _benchmark_records_for(dataset_ids: List[str]) -> List[Dict[str, Any]]:
            if not include_benchmark_registry:
                return []
            records: List[Dict[str, Any]] = []
            for ds_id in dataset_ids:
                try:
                    payload = self.planner_file_tools.get_algorithm_benchmark_baselines(ds_id)
                except Exception:
                    continue
                for raw in list(payload.get("baseline_records") or []):
                    if not isinstance(raw, dict):
                        continue
                    metrics = dict(raw.get("metrics") or {})
                    metrics_path = str(metrics.get("metrics_path") or raw.get("metrics_path") or "").strip()
                    if metrics_path:
                        from_path = _load_json_path(Path(metrics_path).expanduser())
                        if from_path:
                            metrics = from_path
                    records.append(
                        {
                            "source": "benchmark_registry",
                            "algorithm_name": self._normalize_baseline_algorithm_name(raw.get("algorithm_name")),
                            "baseline_type": str(raw.get("baseline_type") or "builtin"),
                            "stage": str(raw.get("stage") or ""),
                            "dataset_id": ds_id,
                            "run_id": str(raw.get("run_id") or metrics.get("run_id") or ""),
                            "record_path": str(raw.get("record_path") or raw.get("path") or ""),
                            "metrics_path": metrics_path or str(metrics.get("metrics_path") or ""),
                            "config_path": str(raw.get("config_path") or metrics.get("resolved_config_path") or ""),
                            "metrics": metrics,
                        }
                    )
            return records

        def _registered_control_records(stage_state: Dict[str, Any], target_stage: str) -> List[Dict[str, Any]]:
            records: List[Dict[str, Any]] = []
            raw_records = stage_state.get("agent_registered_baselines")
            if not raw_records:
                selected_external = (
                    stage_state.get("external_baseline_metrics")
                    if isinstance(stage_state.get("external_baseline_metrics"), dict)
                    else {}
                )
                raw_records = selected_external.get("agent_registered_baselines") if isinstance(selected_external, dict) else []
            iterable = raw_records.values() if isinstance(raw_records, dict) else list(raw_records or [])
            for raw in iterable:
                if not isinstance(raw, dict):
                    continue
                metrics = dict(raw.get("metrics") or {})
                for key, value in raw.items():
                    if key not in metrics and (
                        isinstance(value, (str, int, float, bool))
                        or value is None
                        or key in {"custom_metrics", "claim_metric_evaluator", "target_dataset_ids"}
                    ):
                        metrics[key] = value
                records.append(
                    {
                        "source": "agent_registered_control_baseline",
                        "algorithm_name": self._normalize_baseline_algorithm_name(
                            raw.get("baseline_id") or raw.get("algorithm_name") or raw.get("baseline_name")
                        ),
                        "baseline_type": str(raw.get("baseline_type") or "control"),
                        "stage": str(raw.get("stage") or target_stage),
                        "dataset_id": ",".join([str(item) for item in list(raw.get("target_dataset_ids") or [])]),
                        "run_id": str(raw.get("source_run_id") or ""),
                        "record_path": "",
                        "metrics_path": str(raw.get("source_metrics_path") or ""),
                        "config_path": str(raw.get("source_config_path") or ""),
                        "status": str(raw.get("status") or "active"),
                        "required_for_gate": bool(raw.get("required_for_gate", True)),
                        "revision": int(raw.get("revision") or 1),
                        "control_code_source": deepcopy(raw.get("control_code_source") or {}),
                        "metrics": metrics,
                    }
                )
            return records

        def _stage_record_allowed(record: Dict[str, Any], target_stage: str, selected_run_ids: Set[str]) -> bool:
            record_stage = str(record.get("stage") or "")
            run_id = str(record.get("run_id") or "")
            if selected_run_ids and run_id in selected_run_ids:
                return True
            if not target_stage:
                return True
            if record_stage == target_stage:
                return True
            # Stage 3 often reuses Stage 2 fixed builtin comparators for SOTA checks.
            if target_stage in {"stage3_tuning", "final_regression"} and record_stage == "stage2_claim_validation":
                return True
            return False

        def _dedupe_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            seen: Set[Tuple[str, str, str]] = set()
            out: List[Dict[str, Any]] = []
            for item in records:
                evidence_id = str(item.get("metrics_path") or item.get("record_path") or item.get("run_id") or "")
                key = ("evidence", evidence_id, "") if evidence_id else (
                    str(item.get("algorithm_name") or ""),
                    str(item.get("run_id") or ""),
                    str(item.get("stage") or ""),
                )
                if key in seen:
                    continue
                seen.add(key)
                out.append(item)
            return out

        def _canonical_baseline_records(
            records: List[Dict[str, Any]],
            selected_run_ids: Set[str],
            metric_query: List[str],
        ) -> Tuple[List[Dict[str, Any]], int]:
            """Return one paper-facing baseline record per algorithm/dataset.

            A campaign can contain multiple baseline attempts for the same
            algorithm. For paper tables the canonical source should be the
            benchmark registry when present, otherwise the selected comparator,
            otherwise the newest local scan record with the relevant metric.
            """
            buckets: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
            for record in records:
                key = (
                    str(record.get("algorithm_name") or ""),
                    str(record.get("dataset_id") or ""),
                )
                buckets.setdefault(key, []).append(record)

            def _priority(record: Dict[str, Any]) -> Tuple[int, int, str]:
                source = str(record.get("source") or "")
                run_id = str(record.get("run_id") or "")
                metric_count = sum(
                    1
                    for name in metric_query
                    if _metric_value(dict(record.get("metrics") or {}), name) is not None
                )
                if selected_run_ids and run_id in selected_run_ids:
                    source_rank = 0
                elif source == "benchmark_registry":
                    source_rank = 1
                else:
                    source_rank = 2
                return (source_rank, -metric_count, run_id)

            kept: List[Dict[str, Any]] = []
            omitted = 0
            for items in buckets.values():
                ordered = sorted(items, key=_priority)
                if ordered:
                    kept.append(ordered[0])
                    omitted += max(0, len(ordered) - 1)
            return kept, omitted

        def _best_by_metric(records: List[Dict[str, Any]], metric: str, direction: str) -> Dict[str, Any]:
            values: List[Tuple[float, Dict[str, Any]]] = []
            for record in records:
                if str(record.get("source") or "") == "agent_registered_control_baseline":
                    if str(record.get("status") or "active").strip().lower() not in {"", "active"}:
                        continue
                    if bool(record.get("required_for_gate", True)) is False:
                        continue
                value = _metric_value(dict(record.get("metrics") or {}), metric)
                if value is None:
                    continue
                values.append((float(value), record))
            if not values:
                return {}
            selected_value, selected_record = (
                max(values, key=lambda item: item[0])
                if str(direction or "lower") == "greater"
                else min(values, key=lambda item: item[0])
            )
            return {
                "metric": metric,
                "direction": str(direction or "lower"),
                "value": selected_value,
                "algorithm_name": selected_record.get("algorithm_name"),
                "run_id": selected_record.get("run_id"),
                "source": selected_record.get("source"),
                "metrics_path": selected_record.get("metrics_path"),
            }

        def _record_has_metric(record: Dict[str, Any], names: List[str]) -> bool:
            metrics = dict(record.get("metrics") or {})
            return any(_metric_value(metrics, name) is not None for name in names if str(name or "").strip())

        stage_payloads: Dict[str, Any] = {}
        for stage_name in stage_names:
            stage_state = dict(stages.get(stage_name) or {})
            policy = self.planner_file_tools._campaign_stage_policy(campaign, stage_name)
            primary_metric = str(policy.get("primary_metric") or "w1_mean")
            secondary_metric = str(policy.get("secondary_metric") or "")
            metric_query = list(
                dict.fromkeys(
                    requested_metrics
                    + [primary_metric, secondary_metric, claim_metric_name, "w1_mean", "tmv_mean", "tmv_max"]
                )
            )
            metric_query = [item for item in metric_query if item]
            selected_external = (
                dict(stage_state.get("external_baseline_metrics") or {})
                if isinstance(stage_state.get("external_baseline_metrics"), dict)
                else {}
            )
            selected_run_ids = {str(item) for item in list(selected_external.get("baseline_run_ids") or []) if str(item).strip()}
            target_ids = _target_dataset_ids(stage_state)
            stage_records = _benchmark_records_for(target_ids)
            stage_records.extend(_registered_control_records(stage_state, stage_name))
            stage_records.extend(
                item
                for item in scanned_records
                if _stage_record_allowed(item, stage_name, selected_run_ids)
            )
            required_record_metrics = [primary_metric]
            if requested_metrics:
                required_record_metrics.extend(requested_metrics)
            if stage_name != "stage2_claim_validation":
                required_record_metrics.extend([secondary_metric, claim_metric_name, "w1_mean", "tmv_mean"])
            required_record_metrics = [item for item in dict.fromkeys(required_record_metrics) if item]
            stage_records = [
                item
                for item in stage_records
                if (
                    selected_run_ids
                    and str(item.get("run_id") or "") in selected_run_ids
                )
                or _record_has_metric(item, required_record_metrics)
            ]
            if len(target_ids) == 1:
                for item in stage_records:
                    if not str(item.get("dataset_id") or "").strip():
                        item["dataset_id"] = target_ids[0]
            stage_records = _dedupe_records(stage_records)
            stage_records, omitted_superseded_count = _canonical_baseline_records(
                stage_records,
                selected_run_ids,
                metric_query,
            )
            concise_records: List[Dict[str, Any]] = []
            for record in stage_records:
                metrics = dict(record.get("metrics") or {})
                concise_records.append(
                    {
                        "algorithm_name": record.get("algorithm_name"),
                        "baseline_type": record.get("baseline_type"),
                        "source": record.get("source"),
                        "stage": record.get("stage"),
                        "dataset_id": record.get("dataset_id", ""),
                        "run_id": record.get("run_id"),
                        "metrics_path": record.get("metrics_path", ""),
                        "record_path": record.get("record_path", ""),
                        "config_path": record.get("config_path", ""),
                        "status": record.get("status", ""),
                        "required_for_gate": record.get("required_for_gate", ""),
                        "revision": record.get("revision", ""),
                        "control_code_source": deepcopy(record.get("control_code_source") or {}),
                        "is_selected_external_baseline": bool(
                            selected_run_ids and str(record.get("run_id") or "") in selected_run_ids
                        )
                        or (
                            bool(selected_external)
                            and self._normalize_baseline_algorithm_name(record.get("algorithm_name"))
                            == self._normalize_baseline_algorithm_name(selected_external.get("algorithm_name"))
                            and record.get("source") == "campaign_selected_external_baseline"
                        ),
                        **_compact_metrics(metrics, metric_query),
                    }
                )
            best_metrics: Dict[str, Any] = {}
            metric_directions = {
                primary_metric: str(policy.get("primary_direction") or "lower"),
                secondary_metric: str(policy.get("secondary_direction") or "lower"),
                claim_metric_name: str(claim_spec.get("direction") or policy.get("secondary_direction") or "lower"),
                "w1_mean": "lower",
                "tmv_mean": "lower",
                "tmv_max": "lower",
            }
            for metric_name in metric_query:
                best = _best_by_metric(stage_records, metric_name, metric_directions.get(metric_name, "lower"))
                if best:
                    best_metrics[metric_name] = best
            gate_evidence = (
                dict(stage_state.get("stage_gate_evidence") or {})
                if isinstance(stage_state.get("stage_gate_evidence"), dict)
                else {}
            )
            gate_checks = (
                dict(gate_evidence.get("checks") or {})
                if isinstance(gate_evidence.get("checks"), dict)
                else {}
            )
            strict_audit = (
                dict(gate_checks.get("strict_baseline_audit") or {})
                if isinstance(gate_checks.get("strict_baseline_audit"), dict)
                else {}
            )
            strict_audit_summary: Dict[str, Any] = {}
            if strict_audit:
                missing_records = list(strict_audit.get("missing_required_records") or [])
                nonblocking_failed = list(strict_audit.get("nonblocking_failed_records") or [])
                audit_complete = bool(strict_audit.get("ok")) and not missing_records
                strict_audit_summary = {
                    "required": bool(strict_audit.get("required")),
                    "ok": bool(strict_audit.get("ok")),
                    "ledger_count": int(strict_audit.get("ledger_count") or 0),
                    "expected_ledger_count": int(strict_audit.get("expected_ledger_count") or 0),
                    "missing_required_records": [_compact_audit_record(item) for item in missing_records],
                    "nonblocking_failed_records": [_compact_audit_record(item) for item in nonblocking_failed],
                    "refresh_recommendation": (
                        "no_refresh_needed"
                        if audit_complete
                        else "repair_only_named_missing_or_stale_records"
                    ),
                    "refresh_note": (
                        "The strict ledger is complete. Do not call refresh_campaign_stage_baselines unless "
                        "a concrete missing/stale record or configuration-path error appears."
                        if audit_complete
                        else "Repair only the explicitly named missing/stale records; terminal failed rows are nonblocking."
                    ),
                }
            stage_payloads[stage_name] = {
                "stage": stage_name,
                "stage_status": stage_state.get("status"),
                "stage_gate_status": stage_state.get("stage_gate_status"),
                "target_dataset_ids": target_ids,
                "primary_metric": primary_metric,
                "primary_direction": str(policy.get("primary_direction") or "lower"),
                "secondary_metric": secondary_metric,
                "secondary_direction": str(policy.get("secondary_direction") or ""),
                "selected_external_baseline": _compact_metrics(selected_external, metric_query) if selected_external else {},
                "selected_external_algorithm": self._normalize_baseline_algorithm_name(selected_external.get("algorithm_name")) if selected_external else "",
                "selected_external_run_ids": list(selected_external.get("baseline_run_ids") or []) if selected_external else [],
                "baseline_record_count": len(concise_records),
                "omitted_superseded_baseline_record_count": omitted_superseded_count,
                "baseline_records": concise_records,
                "best_by_metric": best_metrics,
                "strict_baseline_audit": strict_audit_summary,
            }

        payload = {
            "status": "ok",
            "campaign_id": str(campaign.get("campaign_id") or campaign_id),
            "algorithm_id": str(campaign.get("algorithm_id") or ""),
            "baseline_selection_policy": str(campaign.get("baseline_selection_policy") or ""),
            "claim_metric_spec": claim_spec,
            "requested_stage": requested_stage,
            "requested_metric_names": requested_metrics,
            "output_dir_scanned": str(scan_root),
            "training_run_scan_enabled": bool(include_training_run_scan),
            "benchmark_registry_enabled": bool(include_benchmark_registry),
            "filters_applied": {
                "stage_scoped": True,
                "dataset_scoped": bool(dataset_filter),
                "requires_stage_primary_or_requested_metric": True,
                "stage2_requires_claim_metric_by_default": True,
            },
            "paper_use": (
                "Use baseline_records and selected_external_baseline from this payload as the source of truth for "
                "baseline tables and numeric claims. Do not copy baseline numbers from draft manuscript tables or memory."
            ),
            "stages": stage_payloads,
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def get_algorithm_campaign_status(
        self,
        campaign_id: str = "",
        detail: str = "full",
        recent_trials_limit: int = 3,
    ) -> str:
        payload = self.planner_file_tools.get_algorithm_campaign_status(campaign_id=campaign_id)
        if str(detail or "").strip().lower() not in {"full", "registry", "raw"}:
            payload = self.planner_file_tools.summarize_algorithm_campaign_status(
                payload,
                recent_trials_limit=recent_trials_limit,
            )
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def get_algorithm_campaign_status_summary(
        self,
        campaign_id: str = "",
        recent_trials_limit: int = 3,
    ) -> str:
        """Agent-facing campaign status view. Does not return the raw registry."""
        return self.get_algorithm_campaign_status(
            campaign_id=campaign_id,
            detail="summary",
            recent_trials_limit=recent_trials_limit,
        )

    def start_campaign_trial(self, campaign_id: str = "", base_trial_id: str = "") -> str:
        campaign_id = str(campaign_id or self.state.get("active_algorithm_campaign_id") or "").strip()
        payload = self.planner_file_tools.start_campaign_trial(campaign_id, base_trial_id=base_trial_id)
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def resume_rejected_trial(self, campaign_id: str = "", trial_id: str = "") -> str:
        campaign_id = str(campaign_id or self.state.get("active_algorithm_campaign_id") or "").strip()
        payload = self.planner_file_tools.resume_rejected_trial(campaign_id, trial_id)
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def _campaign_latest_training_metrics(self, run_id: str) -> Dict[str, Any]:
        target_run_id = str(run_id or "").strip()
        if not target_run_id:
            return {}
        for manifest in reversed(list(self.state.get("training_runs") or [])):
            if target_run_id and str((manifest or {}).get("run_id") or "") != target_run_id:
                continue
            metrics_path = str((manifest or {}).get("metrics_path") or "").strip()
            if metrics_path:
                try:
                    return json.loads(Path(metrics_path).read_text(encoding="utf-8"))
                except Exception:
                    pass
            verdict = dict((manifest or {}).get("run_verdict") or {})
            if verdict:
                return {"run_verdict": verdict}
        return {}

    @staticmethod
    def _campaign_dataset_control_keys() -> Set[str]:
        return {
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
            "dataset_trial_config_path",
            "config_path",
        }

    @staticmethod
    def _flatten_config_override_paths(overrides: Dict[str, Any], prefix: str = "") -> List[str]:
        paths: List[str] = []
        for raw_key, value in dict(overrides or {}).items():
            key = str(raw_key)
            path = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                paths.extend(PlannerTools._flatten_config_override_paths(value, path))
            else:
                paths.append(path)
        return paths

    @staticmethod
    def _config_path_exists(config: Any, path: str) -> bool:
        current = config
        try:
            parts = _override_path_parts(path)
        except Exception:
            return False
        for part in parts:
            if isinstance(part, int):
                if not isinstance(current, list) or part < 0 or part >= len(current):
                    return False
                current = current[part]
                continue
            if not isinstance(current, dict) or str(part) not in current:
                return False
            current = current[str(part)]
        return True

    def _campaign_effective_config_report(
        self,
        *,
        algo_id: str,
        training_stage: str,
        dataset_id: str,
        adata_path: str,
        config_overrides: Dict[str, Any],
    ) -> Dict[str, Any]:
        report: Dict[str, Any] = {
            "dataset_id": str(dataset_id or ""),
            "training_stage": str(training_stage or ""),
            "adata_path": str(adata_path or ""),
            "config_overrides": deepcopy(config_overrides or {}),
            "override_validation": {"status": "not_checked", "warnings": []},
        }
        if not config_overrides:
            report["override_validation"] = {"status": "no_overrides", "warnings": []}
            return report
        try:
            training_target = resolve_training_target(
                candidate=None,
                training_algorithm_id=algo_id,
                input_adata_path=str(adata_path or self.state.get("preprocessed_path") or self.state.get("input_path") or ""),
                output_dir=str(self.state.get("output_dir") or "cytobridge_output"),
                stage=training_stage,
                metadata={
                    "planner_state_keys": sorted(self.state.keys()),
                    "purpose": "campaign_config_override_preview",
                },
                search_roots=[get_cellcompass_root() / "training_algorithms"],
                workspace_root=get_workspace_root(),
                config_overrides={},
            )
            base_config = materialize_training_config(
                training_target,
                stage=training_stage,
                checkpoints_dir=Path(self.state.get("output_dir") or "cytobridge_output")
                / ".runtime"
                / "campaign_config_preview",
            )
            if not isinstance(base_config, dict) or not base_config:
                report["override_validation"] = {"status": "base_config_empty_or_unavailable", "warnings": []}
                return report
            warnings = []
            for path in self._flatten_config_override_paths(config_overrides):
                if str(path).startswith("__"):
                    continue
                if not self._config_path_exists(base_config, path):
                    warnings.append(
                        {
                            "path": path,
                            "message": "override creates a path that is not present in the live workspace/base config; verify this is intentional",
                        }
                    )
            report["override_validation"] = {
                "status": "checked",
                "warnings": warnings,
            }
        except Exception as exc:
            report["override_validation"] = {
                "status": "check_failed",
                "warnings": [],
                "error": str(exc),
            }
        return report

    def run_campaign_trial(
        self,
        campaign_id: str = "",
        dataset_config_overrides: Optional[Dict[str, Any]] = None,
        holdout_time_evaluation: Optional[Dict[str, Any]] = None,
    ) -> str:
        requested_campaign_id = str(campaign_id or "").strip()
        bound_campaign_id = str(self.state.get("active_algorithm_campaign_id") or "").strip()
        campaign_id = str(requested_campaign_id or bound_campaign_id).strip()
        if dataset_config_overrides is not None and not isinstance(dataset_config_overrides, dict):
            raise ValueError(
                "dataset_config_overrides must be a JSON object/dict, not a JSON string. "
                "Use make_benchmark_dataset_config(...) and pass the returned "
                "dataset_config_overrides object directly into set_campaign_stage_panel(...) "
                "or run_campaign_trial(...)."
            )
        holdout_spec = holdout_time_evaluation if isinstance(holdout_time_evaluation, dict) else {}
        dataset_payload = dict(dataset_config_overrides or {})
        campaign_for_review = self.planner_file_tools.get_algorithm_campaign_status(campaign_id)
        self.planner_file_tools._ensure_campaign_mutation_matches_active_binding(
            campaign_for_review,
            requested_campaign_id=requested_campaign_id,
            previously_bound_campaign_id=bound_campaign_id,
            action="run a trial for",
        )
        stage_for_review = str(campaign_for_review.get("current_stage") or "stage1_feasibility")
        if stage_for_review == "final_regression":
            return json.dumps(
                {
                    "status": "blocked",
                    "reason": "final_regression_confirmation_only",
                    "campaign_id": campaign_id,
                    "stage": stage_for_review,
                    "message": (
                        "Final regression now inherits the latest active best and locks that frozen release. "
                        "Do not run a new tuning/training trial in final_regression."
                    ),
                    "next_required_tool": "check_campaign_stage_gate",
                    "next_required_args": {"campaign_id": campaign_id, "advance": True},
                },
                ensure_ascii=False,
                indent=2,
            )
        stage_state_for_review = (
            dict((campaign_for_review.get("stages") or {}).get(stage_for_review) or {})
            if isinstance(campaign_for_review.get("stages"), dict)
            else {}
        )
        existing_stage_panel_for_review = (
            stage_state_for_review.get("stage_panel")
            if isinstance(stage_state_for_review.get("stage_panel"), dict)
            else {}
        )
        should_resolve_panel_before_review = bool(dataset_payload) or bool(
            self.planner_file_tools._campaign_target_dataset_ids(
                dict(existing_stage_panel_for_review.get("dataset_config_overrides") or {})
            )
        )
        if should_resolve_panel_before_review:
            dataset_payload = self.planner_file_tools.resolve_campaign_trial_dataset_payload(
                campaign_id,
                dataset_payload,
            )
        top_level_epoch_controls: Dict[str, Any] = {}
        for control_key in ("__allow_epoch_override", "__epoch_override_reason"):
            if control_key in dataset_payload:
                top_level_epoch_controls[control_key] = deepcopy(dataset_payload.get(control_key))
        algo_id_for_review = str(campaign_for_review.get("algorithm_id") or "").strip().lower()
        proposal_id_for_review = str(campaign_for_review.get("proposal_id") or "").strip()
        policy_for_review = dict((campaign_for_review.get("stage_policies") or {}).get(stage_for_review) or {})
        training_stage_for_review = str(policy_for_review.get("training_stage") or "pilot")
        claim_metric_run_spec_for_review = self._campaign_claim_metric_run_spec(campaign_for_review)
        primary_metric_for_review = str(policy_for_review.get("primary_metric") or "").strip()
        if (
            self._claim_metric_requires_campaign_evaluator(stage_for_review, primary_metric_for_review)
            and not claim_metric_spec_has_evaluator(claim_metric_run_spec_for_review)
        ):
            return json.dumps(
                {
                    "status": "blocked",
                    "reason": "claim_metric_evaluator_required",
                    "campaign_id": campaign_id,
                    "stage": stage_for_review,
                    "primary_metric": primary_metric_for_review,
                    "message": (
                        "Stage 2 claim metrics must be computed through campaign claim_metric_spec.evaluator_path "
                        "for both candidate and baselines. Add a method-independent evaluator before running trials."
                    ),
                    "next_required_tool": "update_campaign_claim_metric_spec",
                    "required_claim_metric_spec_fields": [
                        "name",
                        "direction",
                        "evaluator_path",
                        "evaluator_function",
                    ],
                    "repair_instruction": (
                        "Call update_campaign_claim_metric_spec with the authoritative evaluator_path. "
                        "Do not start a replacement campaign, patch campaign.json, or pass evaluator_path "
                        "through dataset_config_overrides."
                    ),
                },
                ensure_ascii=False,
                indent=2,
            )
        raw_datasets_for_review = dataset_payload.get("datasets")
        first_dataset = {}
        if isinstance(raw_datasets_for_review, list) and raw_datasets_for_review:
            first_dataset = dict(raw_datasets_for_review[0] or {})
        common_overrides_for_review = dataset_payload.get("config_overrides") if isinstance(dataset_payload.get("config_overrides"), dict) else {}
        if top_level_epoch_controls:
            common_overrides_for_review = {
                **dict(common_overrides_for_review or {}),
                **top_level_epoch_controls,
            }
        entry_overrides_for_review = first_dataset.get("config_overrides") if isinstance(first_dataset.get("config_overrides"), dict) else first_dataset
        config_overrides_for_review = deepcopy(common_overrides_for_review or {})
        for key, value in dict(entry_overrides_for_review or {}).items():
            if key in self._campaign_dataset_control_keys():
                continue
            config_overrides_for_review[key] = deepcopy(value)
        data_path_for_review = (
            str(first_dataset.get("adata_path") or "").strip()
            or str(self.state.get("preprocessed_path") or self.state.get("input_path") or "").strip()
        )
        if algo_id_for_review:
            training_target_for_review = resolve_training_target(
                candidate=None,
                training_algorithm_id=algo_id_for_review,
                input_adata_path=data_path_for_review,
                output_dir=str(self.state.get("output_dir") or "cytobridge_output"),
                stage=training_stage_for_review,
                metadata={
                    "planner_state_keys": sorted(self.state.keys()),
                    "purpose": "campaign_inference_review",
                },
                search_roots=[get_cellcompass_root() / "training_algorithms"],
                workspace_root=get_workspace_root(),
                config_overrides=config_overrides_for_review,
            )
            review_config = materialize_training_config(
                training_target_for_review,
                stage=training_stage_for_review,
                checkpoints_dir=Path(self.state.get("output_dir") or "cytobridge_output")
                / ".runtime"
                / "inference_review_preflight",
            )
            review_state = self._ensure_training_review_ready(
                training_target_for_review,
                stage=training_stage_for_review,
                resolved_config=review_config,
                campaign=campaign_for_review,
                proposal_id=proposal_id_for_review,
                purpose="campaign",
                resume_after_review={
                    "tool": "run_campaign_trial",
                    "args": {
                        "campaign_id": campaign_id,
                        "dataset_config_overrides": deepcopy(dataset_payload),
                        "holdout_time_evaluation": deepcopy(holdout_spec),
                    },
                },
            )
            if not bool(review_state.get("ok")):
                return json.dumps(review_state, ensure_ascii=False, indent=2)
        if not should_resolve_panel_before_review:
            dataset_payload = self.planner_file_tools.resolve_campaign_trial_dataset_payload(campaign_id, dataset_payload)
        top_level_epoch_controls = {}
        for control_key in ("__allow_epoch_override", "__epoch_override_reason"):
            if control_key in dataset_payload:
                top_level_epoch_controls[control_key] = deepcopy(dataset_payload.get(control_key))
        prepared = self.planner_file_tools.prepare_campaign_trial(
            campaign_id,
            dataset_config_overrides=dataset_payload,
        )
        campaign = dict(prepared.get("campaign") or {})
        algo_id = str(campaign.get("algorithm_id") or "").strip().lower()
        trial_id = str(prepared.get("trial_id") or "")
        training_stage = str(prepared.get("training_stage") or "pilot")
        raw_datasets = dataset_payload.get("datasets")
        datasets = raw_datasets if isinstance(raw_datasets, list) and raw_datasets else [None]
        common_overrides = dataset_payload.get("config_overrides") if isinstance(dataset_payload.get("config_overrides"), dict) else {}
        if top_level_epoch_controls:
            common_overrides = {
                **dict(common_overrides or {}),
                **top_level_epoch_controls,
            }
        run_ids: List[str] = []
        run_metrics: List[Dict[str, Any]] = []
        config_reports: List[Dict[str, Any]] = []
        result_lines: List[str] = []
        current_stage = str(campaign.get("current_stage") or "")
        claim_metric_run_spec = self._campaign_claim_metric_run_spec(campaign)
        formal_holdout_spec = self._holdout_time_eval_spec_from_claim_metric(claim_metric_run_spec)
        holdout_spec = self._merge_holdout_eval_specs(holdout_spec, formal_holdout_spec)
        current_stage_state = (
            dict((campaign.get("stages") or {}).get(current_stage) or {})
            if isinstance(campaign.get("stages"), dict)
            else {}
        )
        stage3_claim_guardrail_ids = {
            str(item or "").strip()
            for item in list(current_stage_state.get("claim_guardrail_dataset_ids") or [])
            if str(item or "").strip()
        }

        def _merge_overrides(base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
            merged = deepcopy(base)
            for key, value in extra.items():
                if key in self._campaign_dataset_control_keys():
                    continue
                if isinstance(value, dict) and isinstance(merged.get(key), dict):
                    merged[key] = _merge_overrides(dict(merged.get(key) or {}), value)
                else:
                    merged[key] = deepcopy(value)
            return merged

        for idx, dataset_entry in enumerate(datasets):
            entry = dict(dataset_entry or {})
            dataset_id = str(entry.get("dataset_id") or entry.get("id") or f"dataset{idx + 1}")
            entry_overrides = entry.get("config_overrides") if isinstance(entry.get("config_overrides"), dict) else entry
            config_overrides = _merge_overrides(dict(common_overrides or {}), dict(entry_overrides or {}))
            if claim_metric_spec_has_evaluator(claim_metric_run_spec):
                config_overrides["__campaign_claim_metric_spec"] = deepcopy(claim_metric_run_spec)
            config_report = self._campaign_effective_config_report(
                algo_id=algo_id,
                training_stage=training_stage,
                dataset_id=dataset_id,
                adata_path=str(entry.get("adata_path") or "") or str(self.state.get("preprocessed_path") or self.state.get("input_path") or ""),
                config_overrides=config_overrides,
            )
            dataset_trial_config_path = str(entry.get("dataset_trial_config_path") or entry.get("config_path") or "").strip()
            if dataset_trial_config_path:
                config_report["dataset_trial_config_path"] = dataset_trial_config_path
            before_run_id = str(self.state.get("latest_training_run_id") or "")
            training_result = self._run_training_tool_impl(
                training_algorithm_id=algo_id,
                stage=training_stage,
                config_overrides=config_overrides,
                run_label=f"{str(campaign.get('campaign_id') or '')[:14]}-{trial_id[:14]}-{dataset_id[:12]}",
                adata_path=str(entry.get("adata_path") or "") or None,
                decision="provisional",
                decision_reason="Campaign controller will automatically promote/reject this trial.",
                review_purpose="campaign",
                review_campaign=campaign,
                _skip_review_gates=True,
            )
            after_run_id = str(self.state.get("latest_training_run_id") or "")
            run_id = after_run_id if after_run_id and after_run_id != before_run_id else ""
            if run_id:
                run_ids.append(run_id)
                config_report["run_id"] = run_id
            metrics = self._campaign_latest_training_metrics(run_id)
            resolved_config_path = str(
                metrics.get("resolved_config_path")
                or ((metrics.get("artifacts") or {}) if isinstance(metrics.get("artifacts"), dict) else {}).get("resolved_config_path")
                or ""
            ).strip()
            if resolved_config_path:
                config_report["resolved_config_path"] = resolved_config_path
                try:
                    resolved_path_obj = Path(resolved_config_path)
                    config_report["resolved_config_exists"] = bool(resolved_path_obj.is_file())
                    if resolved_path_obj.is_file():
                        config_report["resolved_config_sha256"] = hashlib.sha256(
                            resolved_path_obj.read_bytes()
                        ).hexdigest()
                except Exception as exc:
                    config_report["resolved_config_stat_error"] = str(exc)
            training_text = str(training_result or "").strip()
            did_not_launch_prefixes = (
                "Implementation map required",
                "Implementation review required",
                "Inference review required",
                "Error:",
            )
            if not run_id and not metrics and training_text.startswith(did_not_launch_prefixes):
                blocked = self.planner_file_tools.abort_campaign_trial(
                    campaign_id,
                    trial_id=trial_id,
                    reason=(
                        f"Training did not launch for dataset `{dataset_id}`; campaign trial was blocked "
                        "instead of being counted as a rejected algorithm trial."
                    ),
                    training_result=training_text,
                )
                blocked["target_dataset_ids"] = list(prepared.get("target_dataset_ids") or [])
                return json.dumps(blocked, ensure_ascii=False, indent=2)
            if not metrics and str(training_result).startswith("Training failed"):
                metrics = {"error": str(training_result)}
            if run_id:
                metrics["run_id"] = run_id
            metrics["campaign_dataset_id"] = dataset_id
            if self._holdout_time_eval_enabled(holdout_spec) and run_id and not metrics.get("error"):
                holdout_report = self._run_holdout_time_evaluation(
                    holdout_spec,
                    parent_run_id=run_id,
                    candidate_name=None,
                    training_algorithm_id=algo_id,
                    stage=training_stage,
                    config_overrides=config_overrides,
                    run_label_prefix=f"{str(campaign.get('campaign_id') or '')[:14]}-{trial_id[:14]}-{dataset_id[:12]}",
                    adata_path=str(entry.get("adata_path") or "") or str(self.state.get("preprocessed_path") or self.state.get("input_path") or ""),
                    device="",
                    seed=None,
                    decision_reason="Auxiliary hold-out timepoint evaluation after the normal campaign trial.",
                    campaign_id=campaign_id,
                    trial_id=trial_id,
                    dataset_id=dataset_id,
                )
                if holdout_report:
                    metrics["holdout_time_evaluation"] = holdout_report
                    self._attach_holdout_report_to_run_metrics(
                        run_id=run_id,
                        report=holdout_report,
                        holdout_spec=holdout_spec,
                    )
                    normalized_holdout_spec = self._normalize_holdout_time_eval_spec(holdout_spec)
                    if (
                        normalized_holdout_spec["attach_to_custom_metrics"]
                        and holdout_report.get("mean_w1") is not None
                    ):
                        local_custom = metrics.get("custom_metrics")
                        if not isinstance(local_custom, dict):
                            local_custom = {}
                        local_custom[normalized_holdout_spec["metric_name"]] = float(holdout_report["mean_w1"])
                        metrics["custom_metrics"] = local_custom
                        evaluator_info = normalized_holdout_spec.get("claim_metric_evaluator")
                        if isinstance(evaluator_info, dict) and evaluator_info:
                            metrics["claim_metric_evaluator"] = dict(evaluator_info)
                            metrics["claim_metric_posthoc_evaluated"] = True
                            metrics["claim_metric_evaluation_source"] = "holdout_time_auxiliary_split"
            if config_report:
                metrics["effective_config_report"] = config_report
                config_reports.append(config_report)
            simulation_version = str(entry.get("simulation_version") or "").strip()
            if simulation_version:
                metrics["simulation_version"] = simulation_version
            run_metrics.append(metrics)
            result_lines.extend([f"[{dataset_id}] {line}" for line in str(training_result).splitlines()[:6]])

        def _aggregate_metrics(items: List[Dict[str, Any]]) -> Dict[str, Any]:
            aggregate: Dict[str, Any] = {
                "campaign_run_ids": run_ids,
                "campaign_dataset_metrics": items,
            }
            if config_reports:
                aggregate["campaign_effective_config_reports"] = config_reports
            errors = [str(item.get("error") or "").strip() for item in items if str(item.get("error") or "").strip()]
            if errors:
                aggregate["error"] = "; ".join(errors)
            w1_scores: List[Any] = []
            tmv_scores: List[Any] = []
            runtime_sec = 0.0
            custom_values: Dict[str, List[float]] = {}
            claim_metric_evaluators: List[Dict[str, Any]] = []
            timeout_items: List[Dict[str, Any]] = []
            inference_timeout_items: List[Dict[str, Any]] = []
            simulation_versions: List[str] = []
            holdout_reports: List[Dict[str, Any]] = []
            for item in items:
                if isinstance(item.get("w1_scores"), list):
                    w1_scores.extend(item.get("w1_scores") or [])
                if isinstance(item.get("tmv_scores"), list):
                    tmv_scores.extend(item.get("tmv_scores") or [])
                simulation_version = str(item.get("simulation_version") or "").strip()
                if simulation_version and simulation_version not in simulation_versions:
                    simulation_versions.append(simulation_version)
                holdout_payload = item.get("holdout_time_evaluation")
                if isinstance(holdout_payload, dict) and holdout_payload:
                    holdout_reports.append(dict(holdout_payload))
                try:
                    runtime_sec += float(item.get("runtime_sec") or 0.0)
                except Exception:
                    pass
                custom = item.get("custom_metrics")
                evaluator_payload = item.get("claim_metric_evaluator")
                if isinstance(evaluator_payload, dict) and evaluator_payload:
                    identity = self._claim_metric_evaluator_identity(evaluator_payload)
                    if identity not in [
                        self._claim_metric_evaluator_identity(existing)
                        for existing in claim_metric_evaluators
                    ]:
                        claim_metric_evaluators.append(dict(evaluator_payload))
                collect_custom_metric = True
                if current_stage == "stage3_tuning" and stage3_claim_guardrail_ids:
                    collect_custom_metric = str(item.get("campaign_dataset_id") or "").strip() in stage3_claim_guardrail_ids
                if collect_custom_metric and isinstance(custom, dict):
                    for key, value in custom.items():
                        try:
                            custom_values.setdefault(str(key), []).append(float(value))
                        except Exception:
                            continue
                if bool(item.get("training_timed_out")):
                    timeout_items.append(
                        {
                            "campaign_dataset_id": item.get("campaign_dataset_id"),
                            "training_time_budget": item.get("training_time_budget"),
                        }
                    )
                if bool(item.get("inference_timed_out")):
                    inference_timeout_items.append(
                        {
                            "campaign_dataset_id": item.get("campaign_dataset_id"),
                            "inference_time_budget": item.get("inference_time_budget"),
                        }
                    )
            if w1_scores:
                aggregate["w1_scores"] = w1_scores
                try:
                    w1_values = [float(x) for x in w1_scores]
                    aggregate["w1_mean"] = sum(w1_values) / len(w1_values)
                except Exception:
                    pass
            if tmv_scores:
                aggregate["tmv_scores"] = tmv_scores
                try:
                    tmv_values = [float(x) for x in tmv_scores]
                    aggregate["tmv_mean"] = sum(tmv_values) / len(tmv_values)
                    aggregate["tmv_max"] = max(tmv_values)
                except Exception:
                    pass
            if runtime_sec:
                aggregate["runtime_sec"] = runtime_sec
            if timeout_items:
                aggregate["training_timed_out"] = True
                aggregate["timeout_datasets"] = timeout_items
                aggregate["training_time_budget"] = timeout_items[0].get("training_time_budget")
            if inference_timeout_items:
                aggregate["inference_timed_out"] = True
                aggregate["inference_timeout_datasets"] = inference_timeout_items
                aggregate["inference_time_budget"] = inference_timeout_items[0].get("inference_time_budget")
            if custom_values:
                aggregate["custom_metrics"] = {
                    key: (sum(values) / len(values))
                    for key, values in custom_values.items()
                    if values
                }
            if len(claim_metric_evaluators) == 1:
                aggregate["claim_metric_evaluator"] = claim_metric_evaluators[0]
            elif len(claim_metric_evaluators) > 1:
                aggregate["claim_metric_evaluator_mismatch"] = [
                    self._claim_metric_evaluator_identity(item)
                    for item in claim_metric_evaluators
                ]
            if len(simulation_versions) == 1:
                aggregate["simulation_version"] = simulation_versions[0]
            elif simulation_versions:
                aggregate["simulation_versions"] = simulation_versions
            if holdout_reports:
                aggregate["holdout_time_evaluation"] = {
                    "status": (
                        "ok"
                        if all(str(item.get("status") or "") == "ok" for item in holdout_reports)
                        else "partial"
                    ),
                    "dataset_reports": holdout_reports,
                    "mean_w1": self._mean_finite([item.get("mean_w1") for item in holdout_reports]),
                }
            aggregate.update(self.planner_file_tools._consistent_w1_backend_metadata(items))
            return aggregate

        metrics = _aggregate_metrics(run_metrics)
        target_dataset_ids = list(prepared.get("target_dataset_ids") or [])
        if target_dataset_ids:
            metrics["target_dataset_ids"] = target_dataset_ids
        run_id = run_ids[0] if run_ids else ""
        decision_payload = self.planner_file_tools.decide_campaign_trial(
            campaign_id,
            trial_id=trial_id,
            run_id=run_id,
            metrics=metrics,
            training_result="\n".join(result_lines),
        )
        payload = {
            **decision_payload,
            "training_result_summary": result_lines[:18],
            "trial_commit": str(prepared.get("trial_commit") or ""),
            "snapshot_id": str(prepared.get("snapshot_id") or ""),
            "effective_config_reports": config_reports,
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def abort_current_campaign_trial(
        self,
        campaign_id: str = "",
        reason: str = "",
        restore_active_best: bool = True,
    ) -> str:
        campaign_id = str(campaign_id or self.state.get("active_algorithm_campaign_id") or "").strip()
        payload = self.planner_file_tools.abort_current_campaign_trial(
            campaign_id,
            reason=reason,
            restore_active_best=restore_active_best,
        )
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def list_campaign_trials(
        self,
        campaign_id: str = "",
        decision: str = "",
        stage: str = "",
        limit: int = 10,
        offset: int = 0,
    ) -> str:
        campaign_id = str(campaign_id or self.state.get("active_algorithm_campaign_id") or "").strip()
        payload = self.planner_file_tools.list_campaign_trials(
            campaign_id,
            decision=decision,
            stage=stage,
            limit=limit,
            offset=offset,
        )
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def check_campaign_stage_gate(self, campaign_id: str = "", advance: bool = True) -> str:
        campaign_id = str(campaign_id or self.state.get("active_algorithm_campaign_id") or "").strip()
        payload = self.planner_file_tools.check_campaign_stage_gate(campaign_id, advance=bool(advance))
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def list_workspace_tree(self, path: str = "", recursive: bool = True, max_entries: int = 200) -> str:
        """List planner-writable workspace trees such as algorithm workspaces and training runs."""
        return self.planner_file_tools.list_workspace_tree(
            path=path,
            recursive=recursive,
            max_entries=max_entries,
        )

    def read_workspace_file(
        self,
        path: str,
        start_line: int = 1,
        end_line: Optional[int] = None,
        max_chars: int = 20000,
    ) -> str:
        """Read a workspace file under planner-managed allowlisted roots."""
        return self.planner_file_tools.read_workspace_file(
            path=path,
            start_line=start_line,
            end_line=end_line,
            max_chars=max_chars,
        )

    def create_workspace_file(self, path: str, content: str, overwrite: bool = False, reason: str = "") -> str:
        """Create a new file inside planner-managed writable roots."""
        return self.planner_file_tools.create_workspace_file(
            path=path,
            content=content,
            overwrite=overwrite,
            reason=reason,
        )

    def create_algorithm_workspace_artifact(
        self,
        relative_path: str,
        kind: str = "file",
        content: str = "",
        overwrite: bool = False,
        reason: str = "",
        algorithm_id: str = "",
    ) -> str:
        """Create a file or directory under the active algorithm workspace."""
        return self.planner_file_tools.create_algorithm_workspace_artifact(
            relative_path=relative_path,
            kind=kind,
            content=content,
            overwrite=overwrite,
            reason=reason,
            algorithm_id=algorithm_id,
        )

    def replace_workspace_file(self, path: str, content: str, reason: str = "") -> str:
        """Rewrite an entire workspace file.

        Use only when `content` is the complete final file from first line to
        last line. For local edits to functions, YAML stanzas, proposal sections,
        or implementation-map rows, prefer `apply_workspace_patch`.
        """
        return self.planner_file_tools.replace_workspace_file(path=path, content=content, reason=reason)

    def patch_algorithm_config(
        self,
        algorithm_id: str = "",
        updates: Optional[Dict[str, Any]] = None,
        dry_run: bool = False,
        reason: str = "",
        allow_create: bool = False,
        allow_list_replace: bool = False,
        strict_types: bool = True,
        allow_epoch_override: bool = False,
        epoch_override_reason: str = "",
    ) -> str:
        """Safely patch a custom algorithm config.yaml with typed path updates and return a resolved diff."""
        payload = self.planner_file_tools.patch_algorithm_config(
            algorithm_id=algorithm_id,
            updates=updates or {},
            dry_run=bool(dry_run),
            reason=reason,
            allow_create=bool(allow_create),
            allow_list_replace=bool(allow_list_replace),
            strict_types=bool(strict_types),
            allow_epoch_override=bool(allow_epoch_override),
            epoch_override_reason=epoch_override_reason,
        )
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def apply_workspace_patch(self, patch: str, reason: str = "") -> str:
        """
        Apply a patch to planner-managed workspace files.

        Supported formats:
        1) Codex patch:
           *** Begin Patch
           *** Update File: /abs/or/relative/path
           @@
           -old
           +new
           *** End Patch
        2) Unified diff:
           --- /path/file.py
           +++ /path/file.py
           @@ -1,2 +1,2 @@
           -old
           +new

        Patch headers are interpreted as real workspace paths, not git path
        aliases. Do not use `--- a/file.py` / `+++ b/file.py`; use absolute
        paths or paths returned by workspace/status tools.
        """
        try:
            proposal_route = self._proposal_patch_route(patch)
            if proposal_route.get("error"):
                return f"Error: {proposal_route['error']}"
            if proposal_route.get("is_proposal_patch"):
                algo_id = str(proposal_route.get("algorithm_id") or "").strip().lower()
                revision_note = str(reason or "").strip() or "Patch PROPOSAL.md via apply_workspace_patch."
                result = self._create_algorithm_proposal_revision(
                    algorithm_id=algo_id,
                    revision_note=revision_note,
                    proposal_patch=patch,
                )
                if result.startswith("Error:"):
                    return result
                return (
                    "✅ Proposal patch applied as a new proposal revision and review flow.\n"
                    f"- algorithm_id: {algo_id}\n"
                    f"- previous_proposal_id: {proposal_route.get('previous_proposal_id') or '(unknown)'}\n"
                    f"- active_proposal_id: {self.state.get('active_proposal_id') or '(unknown)'}\n\n"
                    f"{result}"
                )
            return self.planner_file_tools.apply_workspace_patch(patch=patch, reason=reason)
        except Exception as e:
            logger.exception("apply_workspace_patch failed")
            return f"Error: failed to apply workspace patch: {e}"

    def _proposal_patch_route(self, patch: str) -> Dict[str, Any]:
        ops = self.planner_file_tools._parse_patch(str(patch or ""))
        if not ops:
            return {"is_proposal_patch": False}
        policy = self.planner_file_tools._policy()
        proposal_targets: List[Tuple[str, Path]] = []
        non_proposal_targets: List[Path] = []
        root = (get_cellcompass_root() / "training_algorithms").resolve()
        for op in ops:
            target = policy.validate_write_path(op.path)
            algo_id = ""
            try:
                rel = target.resolve().relative_to(root)
                if len(rel.parts) == 2 and rel.parts[1] == "PROPOSAL.md":
                    algo_id = str(rel.parts[0]).strip().lower()
            except ValueError:
                algo_id = ""
            if algo_id:
                proposal_targets.append((algo_id, target))
                if op.kind != "update":
                    return {"error": "proposal revisions must update an existing PROPOSAL.md; add/delete are not allowed."}
            else:
                non_proposal_targets.append(target)
        if not proposal_targets:
            return {"is_proposal_patch": False}
        if non_proposal_targets:
            return {
                "error": (
                    "Do not mix PROPOSAL.md revisions with code/config/file edits in one patch. "
                    "Patch PROPOSAL.md first so a new proposal_id and review are created, then patch implementation files separately."
                )
            }
        algo_ids = sorted({algo_id for algo_id, _target in proposal_targets})
        if len(algo_ids) != 1:
            return {"error": "A proposal revision patch may target only one algorithm's PROPOSAL.md."}
        store = self._proposal_store()
        current = store.get(algo_ids[0])
        if not isinstance(current, dict):
            return {"error": f"proposal not found for algorithm_id='{algo_ids[0]}'"}
        return {
            "is_proposal_patch": True,
            "algorithm_id": algo_ids[0],
            "proposal_path": str(proposal_targets[0][1]),
            "previous_proposal_id": str(current.get("proposal_id") or ""),
        }

    def preview_workspace_diff(
        self,
        path: str,
        new_content: Optional[str] = None,
        patch: Optional[str] = None,
        context_lines: int = 3,
    ) -> str:
        """Preview a unified diff for a workspace file without writing changes."""
        return self.planner_file_tools.preview_workspace_diff(
            path=path,
            new_content=new_content,
            patch=patch,
            context_lines=context_lines,
        )

    def _normalize_algorithm_proposal_mode(self) -> str:
        mode = str(self.state.get("algorithm_proposal_review_mode") or "agent_decide").strip().lower()
        if mode not in VALID_ALGORITHM_PROPOSAL_REVIEW_MODES:
            mode = "agent_decide"
            self.state["algorithm_proposal_review_mode"] = mode
        return mode

    def _normalize_idea_review_mode(self) -> str:
        mode = str(self.state.get("idea_review_mode") or "agent_decide").strip().lower()
        if mode not in VALID_IDEA_REVIEW_MODES:
            mode = "agent_decide"
            self.state["idea_review_mode"] = mode
        return mode

    def _research_idea_store(self) -> Dict[str, Any]:
        ideas = self.state.get("research_ideas")
        if not isinstance(ideas, dict):
            ideas = {}
            self.state["research_ideas"] = ideas
        return ideas

    @staticmethod
    def _slugify_research_idea_id(title: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", str(title or "").strip().lower()).strip("-")
        if not slug:
            slug = "research-idea"
        return slug[:80]

    @staticmethod
    def _extract_direction_lines(text: str) -> List[str]:
        value = str(text or "").strip()
        if not value:
            return []
        lines = []
        for raw in value.splitlines():
            line = str(raw or "").strip()
            if not line:
                continue
            if re.match(r"^[-*]\s+", line) or re.match(r"^\d+\.\s+", line):
                lines.append(line)
        if lines:
            return lines
        paragraphs = [chunk.strip() for chunk in re.split(r"\n\s*\n", value) if chunk.strip()]
        return paragraphs

    @staticmethod
    def _looks_like_buzzword_stack(text: str) -> bool:
        value = str(text or "").strip().lower()
        if not value:
            return False
        buzzwords = ["ot", "uot", "flow matching", "schrodinger bridge", "sb", "gnn", "transformer", "diffusion"]
        present = [term for term in buzzwords if term in value]
        return len(present) >= 4 and not any(token in value for token in ("because", "ambiguity", "assumption", "signal", "object"))

    def _load_research_idea_record(self, idea_id: str) -> Dict[str, Any]:
        target = str(idea_id or "").strip().lower()
        if not target:
            return {}
        store = self._research_idea_store()
        record = store.get(target)
        if isinstance(record, dict):
            return record
        record = load_research_idea_record(target)
        if isinstance(record, dict) and record:
            store[target] = record
            return record
        return {}

    def _validate_research_idea_payload(self, payload: Dict[str, str], *, current_idea_id: str = "") -> str:
        required = [
            "title",
            "primary_track",
            "problem_definition",
            "scientific_object",
            "current_method_failure_mode",
            "prior_work",
            "why_this_matters",
            "why_now",
            "falsifiable_success_criteria",
            "non_goals",
            "evidence_basis",
            "feasible_direction_families",
            "feasibility_constraints",
        ]
        missing = [key for key in required if not str(payload.get(key) or "").strip()]
        if missing:
            return f"Error: missing required idea fields: {', '.join(missing)}"
        if str(payload.get("primary_track") or "") not in {"biology-journal", "ml-topconf"}:
            return "Error: primary_track must be one of: biology-journal | ml-topconf"
        alternate_track = str(payload.get("alternate_track") or "").strip()
        if alternate_track and alternate_track not in {"biology-journal", "ml-topconf"}:
            return "Error: alternate_track must be empty or one of: biology-journal | ml-topconf"
        if len(str(payload.get("problem_definition") or "").strip()) < 40:
            return "Error: problem_definition is too short. State a concrete scientific question, not a theme."
        if len(str(payload.get("scientific_object") or "").strip()) < 12:
            return "Error: scientific_object is too short. Name the exact object of inference or explanation."
        if len(str(payload.get("current_method_failure_mode") or "").strip()) < 30:
            return "Error: current_method_failure_mode is too short. State what current methods get wrong or cannot identify."
        if len(str(payload.get("prior_work") or "").strip()) < 40:
            return "Error: prior_work is too short. Summarize what prior methods already achieved and what headroom remains."
        if len(str(payload.get("falsifiable_success_criteria") or "").strip()) < 30:
            return "Error: falsifiable_success_criteria is too short. Define a concrete win condition."
        if len(str(payload.get("evidence_basis") or "").strip()) < 20:
            return "Error: evidence_basis is too short. Approval requires traceable evidence."
        direction_lines = self._extract_direction_lines(payload.get("feasible_direction_families", ""))
        if not (1 <= len(direction_lines) <= 5):
            return "Error: feasible_direction_families must contain 1-5 bounded direction families."
        if any(len(line) < 20 for line in direction_lines):
            return "Error: each feasible direction family must say enough to be meaningful, not just a label."
        if self._looks_like_buzzword_stack(payload.get("feasible_direction_families", "")):
            return "Error: feasible_direction_families looks like buzzword stacking. State objects, assumptions, and risks."
        if self._looks_like_impl_code("\n\n".join(str(v) for v in payload.values())):
            return "Error: research idea content should not contain implementation code."
        scientific_object = str(payload.get("scientific_object") or "").strip().lower()
        if scientific_object in {"trajectory", "prediction", "dynamics"}:
            return "Error: scientific_object is too vague. Specify the exact inferential object."
        if current_idea_id:
            idea_id = str(current_idea_id or "").strip().lower()
            if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{1,95}", idea_id):
                return "Error: idea_id must match [a-z0-9][a-z0-9_-]{1,95}."
        return ""

    def _proposal_store(self) -> Dict[str, Any]:
        proposals = self.state.get("algorithm_proposals")
        if not isinstance(proposals, dict):
            proposals = {}
            self.state["algorithm_proposals"] = proposals
        return proposals

    @staticmethod
    def _proposal_dir(algorithm_id: str) -> Path:
        algo_id = str(algorithm_id or "").strip().lower()
        return get_cellcompass_root() / "training_algorithms" / algo_id

    @classmethod
    def _proposal_markdown_path(cls, algorithm_id: str) -> Path:
        return cls._proposal_dir(algorithm_id) / "PROPOSAL.md"

    @classmethod
    def _proposal_json_path(cls, algorithm_id: str) -> Path:
        return cls._proposal_dir(algorithm_id) / "PROPOSAL.json"

    @classmethod
    def _proposal_risk_path(cls, algorithm_id: str) -> Path:
        return cls._proposal_dir(algorithm_id) / "risk.md"

    @staticmethod
    def _render_risk_markdown(record: Dict[str, Any]) -> str:
        implementation_risks = [
            str(item).strip()
            for item in (record.get("implementation_risks") or [])
            if str(item).strip()
        ]
        risk_assessment = str(record.get("risk_assessment") or "").strip()
        lines: List[str] = []
        lines.append(f"# Proposal Risk Assessment: {record.get('algorithm_id','')}")
        lines.append("")
        lines.append("## Metadata")
        lines.append(f"- Proposal ID: {record.get('proposal_id','') or '(unknown)'}")
        lines.append(f"- Proposal Status: {record.get('status','') or '(unknown)'}")
        lines.append(f"- Review Decision: {record.get('review_decision','') or '(none)'}")
        lines.append(f"- Reviewed At: {record.get('reviewed_at','') or '(not reviewed)'}")
        lines.append("")
        lines.append("## Purpose")
        lines.append(
            "This file is written from the proposal evaluator's implementation-risk assessment. "
            "It is advisory: it should guide implementation, preview diagnostics, and campaign debugging, "
            "but it does not by itself change the proposal approve/revise/reject decision. "
            "Reviewer entries are expected to be ordered by severity, highest first."
        )
        lines.append("")
        lines.append("## Structured Risk Assessment")
        if risk_assessment:
            lines.append(risk_assessment)
        elif implementation_risks:
            lines.append("| Rank | Risk | Possible CytoBridge Manifestation | Diagnostic Ideas |")
            lines.append("| --- | --- | --- | --- |")
            for idx, item in enumerate(implementation_risks, start=1):
                escaped = item.replace("|", "\\|")
                lines.append(
                    f"| {idx} | {escaped} | Not specified by reviewer. Inspect preview/campaign metrics, loss curves, W1/TMV, runtime, and diagnostic custom metrics. | Add a targeted diagnostic before trusting campaign results. |"
                )
        else:
            lines.append("(none yet)")
        lines.append("")
        lines.append("## Reviewer Risk Points")
        if implementation_risks:
            for item in implementation_risks:
                lines.append(f"- {item}")
        else:
            lines.append("- (none)")
        lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _normalize_mass_modeling_scope(value: Any) -> str:
        scope = str(value or "").strip().lower().replace("-", "_")
        aliases = {
            "balanced": "balanced_only",
            "balancedonly": "balanced_only",
            "balanced_only": "balanced_only",
            "no_unbalanced": "balanced_only",
            "no_mass": "balanced_only",
            "unbalanced": "models_unbalanced_mass",
            "unbalanced_mass": "models_unbalanced_mass",
            "models_unbalanced": "models_unbalanced_mass",
            "models_unbalanced_mass": "models_unbalanced_mass",
            "mass_modeling": "models_unbalanced_mass",
        }
        return aliases.get(scope, scope if scope in VALID_MASS_MODELING_SCOPES else "")

    @classmethod
    def _proposal_algorithm_attributes(cls, payload: Dict[str, Any]) -> Dict[str, Any]:
        scope = cls._normalize_mass_modeling_scope(payload.get("mass_modeling_scope"))
        models_unbalanced_mass = scope == "models_unbalanced_mass"
        return {
            "mass_modeling_scope": scope,
            "models_unbalanced_mass": models_unbalanced_mass,
            "tmv_gate_required": models_unbalanced_mass,
            "tmv_gate_reason": "proposal.algorithm_attributes.mass_modeling_scope",
        }

    @staticmethod
    def _normalize_markdown_heading(value: str) -> str:
        text = str(value or "").strip().lower()
        text = re.sub(r"`", "", text)
        text = re.sub(r"[^a-z0-9]+", "_", text)
        return re.sub(r"_+", "_", text).strip("_")

    @classmethod
    def _markdown_sections(cls, markdown: str) -> Dict[str, str]:
        sections: Dict[str, List[str]] = {}
        current = ""
        for line in str(markdown or "").splitlines():
            match = re.match(r"^##\s+(.+?)\s*$", line)
            if match:
                current = cls._normalize_markdown_heading(match.group(1))
                sections.setdefault(current, [])
                continue
            if current:
                sections.setdefault(current, []).append(line)
        return {key: "\n".join(value).strip() for key, value in sections.items()}

    @staticmethod
    def _clean_proposal_markdown_section(value: str) -> str:
        lines = str(value or "").splitlines()
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and lines[0].lstrip().startswith(">"):
            lines.pop(0)
            while lines and not lines[0].strip():
                lines.pop(0)
        text = "\n".join(lines).strip()
        if text in {"(empty)", "(none)", "(not specified)"}:
            return ""
        return text

    @classmethod
    def _proposal_payload_from_markdown(cls, markdown: str, fallback: Dict[str, Any]) -> Dict[str, Any]:
        sections = cls._markdown_sections(markdown)
        payload = dict(fallback or {})
        mapping = {
            "abstract": "abstract",
            "literature_and_package_grounding": "literature_and_package_grounding",
            "literature": "literature_and_package_grounding",
            "references": "references",
            "problem_statement": "problem_statement",
            "problem_mathematical_form": "problem_mathematical_form",
            "mathematical_derivation_to_algorithm_design": "mathematical_derivation_to_algorithm_design",
            "derivation_to_algorithm_design": "mathematical_derivation_to_algorithm_design",
            "novelty_and_contributions": "novelty_and_contributions",
            "novelty_contribution": "novelty_and_contributions",
            "contributions": "novelty_and_contributions",
            "claimed_capability": "claimed_capability",
            "objective": "objective",
            "theoretical_core": "theoretical_core",
            "mathematical_abstraction": "mathematical_abstraction",
            "algorithm_semantics_table": "algorithm_semantics_table",
            "implementation_pseudocode": "implementation_pseudocode",
            "evaluation_plan": "evaluation_plan",
            "expected_evaluation_outcome": "expected_evaluation_outcome",
            "inductive_generalization_argument": "inductive_generalization_argument",
            "mass_modeling_scope": "mass_modeling_scope",
            "unbalanced_decision": "unbalanced_decision",
            "stochasticity_decision": "stochasticity_decision",
            "distribution_recovery_argument": "distribution_recovery_argument",
            "overengineering_self_check": "overengineering_self_check",
            "uncertainty_risks": "uncertainty_and_risks",
            "uncertainty_and_risks": "uncertainty_and_risks",
        }
        for heading, key in mapping.items():
            if heading in sections:
                payload[key] = cls._clean_proposal_markdown_section(sections[heading])
        return payload

    def _proposal_markdown_after_patch(self, algorithm_id: str, patch: str, current_record: Dict[str, Any]) -> str:
        algo_id = str(algorithm_id or "").strip().lower()
        current_path = self._proposal_markdown_path(algo_id)
        if current_path.exists():
            current_markdown = current_path.read_text(encoding="utf-8")
        else:
            current_markdown = self._render_proposal_markdown(current_record)
        ops = self.planner_file_tools._parse_patch(str(patch or ""))
        if not ops:
            raise ValueError("proposal_patch did not contain any patch operations")
        updated = current_markdown
        for op in ops:
            if op.kind != "update":
                raise ValueError("proposal_patch may only update the current PROPOSAL.md; add/delete operations are not allowed")
            raw_path = Path(str(op.path))
            if raw_path.name != "PROPOSAL.md":
                raise ValueError("proposal_patch must target PROPOSAL.md")
            if raw_path.is_absolute() and raw_path.resolve() != current_path.resolve():
                raise ValueError(f"proposal_patch targets {raw_path}, expected {current_path}")
            updated = self.planner_file_tools._apply_update_lines(updated, op.lines)
        return updated

    @staticmethod
    def _render_proposal_markdown(record: Dict[str, Any]) -> str:
        proposal = dict(record.get("proposal") or {})
        attributes = dict(record.get("algorithm_attributes") or proposal.get("algorithm_attributes") or {})
        if not attributes and proposal.get("mass_modeling_scope"):
            scope = PlannerTools._normalize_mass_modeling_scope(proposal.get("mass_modeling_scope"))
            attributes = {
                "mass_modeling_scope": scope,
                "models_unbalanced_mass": scope == "models_unbalanced_mass",
                "tmv_gate_required": scope == "models_unbalanced_mass",
            }
        mass_scope = (
            str(attributes.get("mass_modeling_scope") or proposal.get("mass_modeling_scope") or "").strip()
            or "(unset)"
        )
        implementation_risks = [
            str(item).strip()
            for item in (record.get("implementation_risks") or [])
            if str(item).strip()
        ]
        proposal_warnings = [
            str(item).strip()
            for item in (record.get("proposal_warnings") or [])
            if str(item).strip()
        ]
        risk_assessment = str(record.get("risk_assessment") or "").strip()
        lines: List[str] = []
        lines.append(f"# Algorithm Proposal: {record.get('algorithm_id','')}")
        lines.append("")
        lines.append("## Metadata")
        lines.append(f"- Status: {record.get('status','')}")
        lines.append(f"- Review Mode: {record.get('review_mode','')}")
        lines.append(f"- Primary Idea ID: {record.get('primary_idea_id','') or '(none)'}")
        lines.append(f"- Requires User Review: {bool(record.get('requires_user_review', False))}")
        lines.append(f"- Created At: {record.get('created_at','')}")
        lines.append(f"- Updated At: {record.get('updated_at','')}")
        if record.get("reviewed_at"):
            lines.append(f"- Reviewed At: {record.get('reviewed_at','')}")
        if record.get("review_decision"):
            lines.append(f"- Review Decision: {record.get('review_decision','')}")
        lines.append("")
        if proposal_warnings:
            lines.append("## Proposal Tool Warnings")
            for warning in proposal_warnings:
                lines.append(f"- {warning}")
            lines.append("")
        lines.append("## Abstract")
        lines.append(str(proposal.get("abstract", "")).strip() or "(empty)")
        lines.append("")
        lines.append("## Literature and Package Grounding")
        lines.append(
            "> For genuinely new algorithms, cite at least 10 directly relevant papers or literature notes. "
            "Each citation should include why it matters. RAG snippets alone are not enough; read the original PDF or important sections for papers that shape the method."
        )
        lines.append(str(proposal.get("literature_and_package_grounding", "")).strip() or "(empty)")
        lines.append("")
        lines.append("## References")
        lines.append(
            "> Bibliography/reference list for papers, local literature notes, theory-book ranges, package docs, or other stable sources that grounded the proposal."
        )
        lines.append(str(proposal.get("references", "")).strip() or "(not specified)")
        lines.append("")
        lines.append("## Problem Statement")
        lines.append(str(proposal.get("problem_statement", "")).strip() or "(empty)")
        lines.append("")
        lines.append("## Problem Mathematical Form")
        lines.append(
            "> If the method claims to solve a mathematical problem, formulate the dynamic problem when possible, "
            "for example dynamic OT, Schrödinger bridge, WFR/UOT, mean-field dynamics, or a clearly defined new dynamic objective."
        )
        lines.append(str(proposal.get("problem_mathematical_form", "")).strip() or "(not specified)")
        lines.append("")
        lines.append("## Mathematical Derivation to Algorithm Design")
        lines.append(
            "> For a new algorithm with a mathematical objective, make this section self-contained and do not skip "
            "the bridge from problem to algorithm. Define the variables/measures/weights/controls/couplings/paths; "
            "derive the computable representation or justified relaxation; derive the supervision targets or "
            "optimization objective; derive the inference rule and evaluation trajectory; then explain distribution "
            "recoverability and, when `mass_modeling_scope=models_unbalanced_mass`, mass recoverability through the "
            "learned or biologically meaningful mass mechanism rather than post-hoc correction. Name and "
            "justify every proposal-stage assumption or approximation. Engineering shortcuts belong later in "
            "`IMPLEMENTATION_MAP.md` unless they are part of the method. If the proposal is biological-motivation-first, "
            "a config/reproduction variant, or otherwise not making a mathematical-algorithm innovation claim, state "
            "`Not applicable` and explain why. Use `CytoBridge-main/docs/theory/wfrfm-derivation-example.md` as the "
            "explicitness bar for formal dynamic-problem proposals."
        )
        lines.append(str(proposal.get("mathematical_derivation_to_algorithm_design", "")).strip() or "(not specified)")
        lines.append("")
        lines.append("## Novelty and Contributions")
        lines.append(
            "> For genuinely new algorithms, state the concrete novelty/contribution relative to builtin CytoBridge methods "
            "and directly relevant literature. Separate mathematical contribution, modeling contribution, biological-use contribution, "
            "and engineering contribution when applicable. If there is no algorithmic novelty claim, state `Not applicable` and explain why."
        )
        lines.append(str(proposal.get("novelty_and_contributions", "")).strip() or "(not specified)")
        lines.append("")
        lines.append("## Claimed Capability")
        lines.append(str(proposal.get("claimed_capability", "")).strip() or "(empty)")
        lines.append("")
        lines.append("## Algorithm Attributes")
        lines.append("| Attribute | Value |")
        lines.append("| --- | --- |")
        lines.append(f"| Mass modeling scope | `{mass_scope}` |")
        lines.append(f"| Models unbalanced mass | `{bool(attributes.get('models_unbalanced_mass', False))}` |")
        lines.append(f"| TMV hard gate required | `{bool(attributes.get('tmv_gate_required', False))}` |")
        lines.append("")
        lines.append("## Objective")
        lines.append(str(proposal.get("objective", "")).strip() or "(empty)")
        lines.append("")
        lines.append("## Theoretical Core")
        lines.append(str(proposal.get("theoretical_core", "")).strip() or "(empty)")
        lines.append("")
        lines.append("## Mathematical Abstraction")
        lines.append(str(proposal.get("mathematical_abstraction", "")).strip() or "(empty)")
        lines.append("")
        lines.append("## Algorithm Semantics Table")
        lines.append("> Include a markdown table that maps mathematical objects to runtime layers and distinguishes what must preserve semantics from what implementation details can vary.")
        lines.append("")
        lines.append(str(proposal.get("algorithm_semantics_table", "")).strip() or "(empty)")
        lines.append("")
        lines.append("## Implementation Pseudocode")
        lines.append("> Prefer paper-style pseudocode for traceability: global `Inputs`, `Outputs`, optional `Definitions`, then labeled steps `P1`, `P2`, ... . Each step should include `Inputs`, `Outputs`, `Invariants`, and `Forbidden deviations`, plus key equations or update rules. Formatting is advisory at proposal creation; semantic mappability remains reviewable.")
        lines.append("")
        lines.append(str(proposal.get("implementation_pseudocode", "")).strip() or "(empty)")
        lines.append("")
        lines.append("## Evaluation Plan")
        lines.append(str(proposal.get("evaluation_plan", "")).strip() or "(empty)")
        lines.append("")
        lines.append("## Expected Evaluation Outcome")
        lines.append("> State the scenario where the algorithm should work, the expected result pattern beyond builtin W1/TMV, and why that result would validate the claimed capability.")
        lines.append(str(proposal.get("expected_evaluation_outcome", "")).strip() or "(empty)")
        lines.append("")
        lines.append("## Inductive Generalization Argument")
        lines.append(
            "> State the learned runtime rule for new valid t=0 cells/particles, allowed inference-time inputs/context, "
            "training-only supervision signals, and why inference does not rely on training-cell ids, memorized OT rows, "
            "barcode-specific hardcoding, future snapshots, or target-specific lookup/correction."
        )
        lines.append(str(proposal.get("inductive_generalization_argument", "")).strip() or "(empty)")
        lines.append("")
        lines.append("## Mass Modeling Scope")
        lines.append("> Required machine-readable value: `balanced_only` or `models_unbalanced_mass`. This proposal-stage attribute controls whether TMV is a hard campaign gate.")
        lines.append(str(proposal.get("mass_modeling_scope", "")).strip() or mass_scope)
        lines.append("")
        lines.append("## Unbalanced Decision")
        lines.append(
            "> State explicitly whether unbalanced mass is modeled. If yes, justify the actual mass mechanism: "
            "learned growth field/head, WFR/UOT/SB mass dynamics, proliferation/death process, condition-driven growth prior, "
            "or another biologically/theoretically meaningful route. Do not rely on target-count lookup, time-only count-ratio clocks, "
            "post-hoc rescaling, or TMV repair."
        )
        lines.append(str(proposal.get("unbalanced_decision", "")).strip() or "(empty)")
        lines.append("")
        lines.append("## Stochasticity Decision")
        lines.append(str(proposal.get("stochasticity_decision", "")).strip() or "(empty)")
        lines.append("")
        lines.append("## Distribution Recovery Argument")
        lines.append(
            "> Explain why interval inference can recover the next-time distribution. If `Unbalanced Decision` enables mass modeling, "
            "also explain why the final weighted-particle measure recovers observed cell-count / total-mass change through the approved growth/dynamics mechanism, not post-hoc correction."
        )
        lines.append(str(proposal.get("distribution_recovery_argument", "")).strip() or "(empty)")
        lines.append("")
        lines.append("## Overengineering Self-check")
        lines.append(str(proposal.get("overengineering_self_check", "")).strip() or "(empty)")
        lines.append("")
        lines.append("## Uncertainty & Risks")
        lines.append(str(proposal.get("uncertainty_and_risks", "")).strip() or "(none)")
        lines.append("")
        feedback = str(record.get("review_feedback", "")).strip()
        lines.append("## Review Feedback")
        lines.append(feedback or "(none)")
        lines.append("")
        lines.append("## Evaluator Implementation Risks")
        if implementation_risks:
            for item in implementation_risks:
                lines.append(f"- {item}")
        else:
            lines.append("- (none)")
        lines.append("")
        lines.append("## Evaluator Risk Assessment")
        if risk_assessment:
            lines.append("See `risk.md` for the structured risk assessment.")
        else:
            lines.append("(none)")
        history = list(record.get("review_history") or [])
        lines.append("")
        lines.append("## Review History")
        if history:
            for item in history:
                lines.append(
                    f"- [{item.get('at','')}] decision={item.get('decision','')} "
                    f"feedback={item.get('feedback','') or '(none)'}"
                )
        else:
            lines.append("- (none)")
        lines.append("")
        return "\n".join(lines)

    @classmethod
    def _read_markdown_status(cls, algorithm_id: str) -> str:
        path = cls._proposal_markdown_path(algorithm_id)
        if not path.exists():
            return ""
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            return ""
        for line in text.splitlines():
            if line.strip().lower().startswith("- status:"):
                return line.split(":", 1)[1].strip().lower()
        return ""

    def _persist_proposal_record(self, algorithm_id: str, record: Dict[str, Any]) -> None:
        algo_id = str(algorithm_id or "").strip().lower()
        proposal_dir = self._proposal_dir(algo_id)
        proposal_dir.mkdir(parents=True, exist_ok=True)
        md_path = self._proposal_markdown_path(algo_id)
        json_path = self._proposal_json_path(algo_id)
        risk_path = self._proposal_risk_path(algo_id)
        record["proposal_path"] = str(md_path)
        record["editable_proposal_path"] = str(md_path)
        record["proposal_json_path"] = str(json_path)
        record["risk_path"] = str(risk_path)
        markdown = self._render_proposal_markdown(record)
        md_path.write_text(markdown, encoding="utf-8")
        json_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        risk_path.write_text(self._render_risk_markdown(record), encoding="utf-8")

    @staticmethod
    def _looks_like_impl_code(text: str) -> bool:
        value = str(text or "")
        if "```" in value:
            return True
        patterns = [
            r"(^|\n)\s*def\s+[A-Za-z_]\w*\s*\(",
            r"(^|\n)\s*class\s+[A-Za-z_]\w*",
            r"(^|\n)\s*import\s+[A-Za-z_][\w\.]*",
            r"(^|\n)\s*from\s+[A-Za-z_][\w\.]*\s+import\s+",
        ]
        return any(re.search(pattern, value) for pattern in patterns)

    @staticmethod
    def _looks_like_formal_pseudocode(text: str) -> bool:
        value = str(text or "")
        has_inputs = bool(re.search(r"(^|\n)\s*inputs\s*:", value, flags=re.IGNORECASE))
        has_outputs = bool(re.search(r"(^|\n)\s*outputs\s*:", value, flags=re.IGNORECASE))
        step_matches = list(re.finditer(r"(^|\n)\s*(P\d+)\s*[\.:]", value))
        has_step_ids = bool(step_matches)
        has_equation_like = any(
            token in value
            for token in (" = ", "E[", "||", "∂", "->", "→", ":=", "sum_", "log ", "exp(")
        ) or bool(re.search(r"\b(minimize|solve|sample|integrate|compute|return)\b", value, flags=re.IGNORECASE))
        if not (has_inputs and has_outputs and has_step_ids and has_equation_like):
            return False

        step_blocks: List[str] = []
        for idx, match in enumerate(step_matches):
            start = match.start(2)
            end = step_matches[idx + 1].start(2) if idx + 1 < len(step_matches) else len(value)
            step_blocks.append(value[start:end])
        if not step_blocks:
            return False
        required_patterns = [
            r"(^|\n)\s*inputs\s*:",
            r"(^|\n)\s*outputs\s*:",
            r"(^|\n)\s*(required\s+)?invariants\s*:",
            r"(^|\n)\s*forbidden\s+deviations\s*:",
        ]
        for block in step_blocks:
            if not all(re.search(pattern, block, flags=re.IGNORECASE) for pattern in required_patterns):
                return False
        return True

    @staticmethod
    def _looks_like_algorithm_semantics_table(text: str) -> bool:
        value = str(text or "").strip()
        if "|" not in value:
            return False
        required_terms = [
            "mathematical object",
            "runtime layer",
            "semantic definition",
        ]
        lower = value.lower()
        if not all(term in lower for term in required_terms):
            return False
        rows = [line for line in value.splitlines() if "|" in line and not re.fullmatch(r"\s*\|?[\-\s:|]+\|?\s*", line)]
        return len(rows) >= 3

    def get_algorithm_proposal_template(self) -> str:
        """Return the current standard proposal template and section guidance."""
        return """# CytoBridge Algorithm Proposal Template

Use this template before calling create_algorithm_proposal(...) or when patching PROPOSAL.md with apply_workspace_patch(...).

Core rule: this is a proposal for a learned dynamics model. The trained model must roll the t0 population through a continuous t0-to-final trajectory. W1, TMV, and claim metrics must be computed from the model-generated trajectory and trained model state, not from direct metric prediction, future-snapshot lookup, or post-hoc particle/weight repair. If unbalanced mass is claimed, cell-count/total-mass changes must come from a meaningful growth/dynamics mechanism, not from target-count lookup, time-only count-ratio clocks, or post-hoc rescaling.

## Abstract
Write one compact paragraph: target problem, core algorithm idea, why it should help, and what evidence would validate it.

## Literature and Package Grounding
State what local survey notes, builtin CytoBridge algorithm docs/source, package semantics, theory references, and prior papers were checked.
For a genuinely new algorithm, cite at least 10 directly relevant papers or literature notes with one-line relevance notes. Do not pad this section.
For OT/SB/WFR/UOT/continuity-equation theory, include relevant theory-book chapter/section/page ranges found through search_theory(...), then read those ranges with read_file(...).

## References
List stable citations, local note paths, PDF paths, or theory-book ranges used above.

## Problem Statement
Define the real scientific, mathematical, or modeling gap being solved. Do not only describe implementation mechanics.

## Problem Mathematical Form
Optional but preferred for mathematical/algorithmic proposals.
If present, write a high-level meaningful dynamic problem/objective/constraint system, such as dynamic OT, Schrödinger bridge, WFR/UOT, mean-field dynamics, or a clearly defined new dynamic problem.
Do not merely rewrite the proposed algorithm or loss in notation.
If no clean global form exists, say that explicitly and define the modeling target in precise prose.

## Mathematical Derivation to Algorithm Design
Optional unless the proposal claims a concrete mathematical problem or algorithmic novelty.
If present, make this section self-contained. A technically strong reviewer should be able to follow the chain without opening source code or external papers first.
Derive from Problem Mathematical Form to the concrete solution route without skipped steps:
1. define variables/measures/particles/weights/controls/growth/scores/couplings/paths/constraints/target marginals;
2. show the exact equivalence, relaxation, estimator, path identity, bridge identity, score/flow matching identity, or finite-sample approximation that makes the problem computable;
3. derive the supervision targets or optimization objective, including coupling/path/dynamics choices and losses/objectives when relevant;
4. derive the inference rule and generated evaluation trajectory from the trained model;
5. explain distribution recoverability and, if mass_modeling_scope=models_unbalanced_mass, mass recoverability;
6. name every proposal-stage assumption or approximation, justify it, and keep it minimal.
Engineering shortcuts belong in IMPLEMENTATION_MAP.md later, not in the theory unless they are part of the actual proposed method. Use CytoBridge-main/docs/theory/wfrfm-derivation-example.md as the explicitness bar for a formal dynamic-problem proposal.

## Novelty and Contributions
For a genuinely new algorithm, state concrete contributions relative to builtin CytoBridge methods and directly relevant literature.
Valid contribution types include a new dynamic formulation, estimator/loss, modeling assumption, biological capability, scalability idea, data contract, or evaluation setup.
If this is a reproduction, config variant, or biology-motivation-first method without algorithmic novelty, say so explicitly.

## Claimed Capability
State what capability should improve and what evidence would validate that claim.

## Objective
State the scientific/modeling objective in plain language.

## Theoretical Core
Explain the central theoretical rationale without code details.

## Mathematical Abstraction
Define the method's mathematical objects and relationships at proposal level.

## Algorithm Semantics Table
Provide a markdown table with columns:
| Mathematical Object | Runtime Layer | Semantic Definition |
Each key object should have a clear semantic role.

## Implementation Pseudocode
Write paper-style pseudocode, not real code.
Include global Inputs/Outputs and labeled steps P1/P2/... .
For every step include Inputs, Outputs, Invariants, and Forbidden deviations.
This structure is recommended for reviewer traceability, but create_algorithm_proposal will warn rather than reject on formatting alone.
The implementation reviewer may still check whether code/config can be meaningfully mapped against this section.

## Mass Modeling Scope
Set the tool field `mass_modeling_scope` to exactly one machine-readable value:
- balanced_only: the algorithm deliberately does not model cell-count / total-mass changes; TMV is diagnostic, not a hard gate.
- models_unbalanced_mass: the algorithm claims to model mass/growth; TMV is a hard gate and the proposal must justify mass recovery.

## Unbalanced Decision
Explain why the algorithm is balanced_only or models_unbalanced_mass.
If mass is enabled, explain why weighted-particle inference can recover observed cell-count / total-mass change through a learned growth field/head, WFR/UOT/SB mass mechanism, proliferation/death process, condition-driven growth prior, or another biologically/theoretically meaningful mechanism. Explain why it is not a target-count correction, time-only count-ratio clock, post-hoc rescaling, or TMV repair.

## Stochasticity Decision
State whether stochasticity/noise is enabled and why.

## Distribution Recovery Argument
Explain why inference from t_k to t_{k+1} can recover the observed next-time distribution in the exact-fit/convergence limit.
If mass is enabled, also explain why the final weighted-particle measure recovers observed total mass / cell-count change.

## Expected Evaluation Outcome
Specify the data or simulation scenario where the algorithm should work, the expected result pattern beyond builtin W1/TMV, and why that pattern validates the claimed capability.
Write this so a later campaign agent can choose benchmark datasets or design simulations without rereading the whole theory.

## Inductive Generalization Argument
State the learned runtime rule for new valid t0 cells/particles.
List allowed inference-time inputs/context and training-only supervision signals.
Explain why runtime inference does not rely on training-cell ids, row ids, memorized OT rows, barcode-specific hardcoding unavailable for new cells, future snapshots, or target-specific lookup/correction.

## Evaluation Plan
Builtin W1 and TMV are fixed and must not be redefined.
Custom claim metrics are additive only and must be computed from the generated trajectory and trained model state.
Explain why each metric validates the stated scientific goal.

## Overengineering Self-Check
Explain why the design keeps a simple effective core and does not add unnecessary machinery.

## Uncertainty and Risks
List known assumptions, limits, and likely implementation/training risks. The proposal evaluator will write the official severity-ranked risk.md after review.
"""

    def create_algorithm_proposal(
        self,
        algorithm_id: str,
        abstract: str = "",
        literature_and_package_grounding: str = "",
        literature: str = "",
        references: str = "",
        problem_statement: str = "",
        problem_mathematical_form: str = "",
        mathematical_derivation_to_algorithm_design: str = "",
        novelty_and_contributions: str = "",
        claimed_capability: str = "",
        objective: str = "",
        theoretical_core: str = "",
        mathematical_abstraction: str = "",
        algorithm_semantics_table: str = "",
        implementation_pseudocode: str = "",
        evaluation_plan: str = "",
        expected_evaluation_outcome: str = "",
        inductive_generalization_argument: str = "",
        mass_modeling_scope: str = "",
        unbalanced_decision: str = "",
        stochasticity_decision: str = "",
        distribution_recovery_argument: str = "",
        overengineering_self_check: str = "",
        uncertainty_and_risks: str = "",
        primary_idea_id: str = "",
        requires_user_review: bool = False,
    ) -> str:
        """
        Create/update a high-level algorithm proposal before writing algorithm implementation files.

        Keep proposal content at theory/pseudocode level.
        Do not include real implementation code.
        """
        algo_id = str(algorithm_id or "").strip().lower()
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{1,63}", algo_id):
            return (
                "Error: algorithm_id must match [a-z0-9][a-z0-9_-]{1,63}, "
                f"got: {algorithm_id!r}"
            )
        try:
            self.planner_file_tools._ensure_algorithm_id_mutable(algo_id, action="create_algorithm_proposal")
        except Exception as exc:
            return f"Error: {exc}"

        normalized_mass_scope = self._normalize_mass_modeling_scope(mass_modeling_scope)
        grounding = str(literature_and_package_grounding or "").strip()
        literature_alias = str(literature or "").strip()
        if grounding and literature_alias and literature_alias not in grounding:
            grounding = f"{grounding}\n\nAdditional literature:\n{literature_alias}"
        elif literature_alias:
            grounding = literature_alias
        payload = {
            "abstract": str(abstract or "").strip(),
            "literature_and_package_grounding": grounding,
            "references": str(references or "").strip(),
            "problem_statement": str(problem_statement or "").strip(),
            "problem_mathematical_form": str(problem_mathematical_form or "").strip(),
            "mathematical_derivation_to_algorithm_design": str(mathematical_derivation_to_algorithm_design or "").strip(),
            "novelty_and_contributions": str(novelty_and_contributions or "").strip(),
            "claimed_capability": str(claimed_capability or "").strip(),
            "objective": str(objective or "").strip(),
            "theoretical_core": str(theoretical_core or "").strip(),
            "mathematical_abstraction": str(mathematical_abstraction or "").strip(),
            "algorithm_semantics_table": str(algorithm_semantics_table or "").strip(),
            "implementation_pseudocode": str(implementation_pseudocode or "").strip(),
            "evaluation_plan": str(evaluation_plan or "").strip(),
            "expected_evaluation_outcome": str(expected_evaluation_outcome or "").strip(),
            "inductive_generalization_argument": str(inductive_generalization_argument or "").strip(),
            "mass_modeling_scope": normalized_mass_scope,
            "unbalanced_decision": str(unbalanced_decision or "").strip(),
            "stochasticity_decision": str(stochasticity_decision or "").strip(),
            "distribution_recovery_argument": str(distribution_recovery_argument or "").strip(),
            "overengineering_self_check": str(overengineering_self_check or "").strip(),
            "uncertainty_and_risks": str(uncertainty_and_risks or "").strip(),
        }
        payload["algorithm_attributes"] = self._proposal_algorithm_attributes(payload)
        primary_idea_value = str(primary_idea_id or "").strip().lower()
        if primary_idea_value and not self._load_research_idea_record(primary_idea_value):
            return f"Error: research idea not found for idea_id='{primary_idea_value}'"
        missing = [
            key
            for key, value in payload.items()
            if key not in {
                "uncertainty_and_risks",
                "problem_mathematical_form",
                "mathematical_derivation_to_algorithm_design",
                "novelty_and_contributions",
                "references",
                "algorithm_attributes",
            } and not value
        ]
        if missing:
            return f"Error: missing required proposal fields: {', '.join(missing)}"
        if normalized_mass_scope not in VALID_MASS_MODELING_SCOPES:
            return "Error: mass_modeling_scope must be one of: balanced_only | models_unbalanced_mass"
        if len(payload["abstract"]) < 40:
            return (
                "Error: abstract is too short. "
                "Summarize the core algorithm idea, target problem, and expected evidence in a compact paragraph."
            )
        literature_evidence_text = "\n\n".join(
            part
            for part in [payload["literature_and_package_grounding"], payload.get("references", "")]
            if str(part or "").strip()
        )
        if len(literature_evidence_text) < 200:
            return (
                "Error: literature/literature_and_package_grounding is too short. "
                "Summarize the local survey/package grounding and cite the directly relevant papers or notes; "
                "you may pass this as `literature`, `literature_and_package_grounding`, and/or `references`. "
                "For genuinely new algorithms, include at least 10 relevant citations with one-line relevance notes."
            )
        if len(payload["problem_statement"]) < 40:
            return (
                "Error: problem_statement is too short. "
                "State the specific problem or method gap the algorithm is meant to solve."
            )
        if len(payload["claimed_capability"]) < 40:
            return (
                "Error: claimed_capability is too short. "
                "State what the method claims to solve/improve and what evidence would validate that claim."
            )
        if len(payload["distribution_recovery_argument"]) < 40:
            return (
                "Error: distribution_recovery_argument is too short. "
                "Provide a concrete argument for why one-step/time-step inference can recover the observed next-time distribution."
            )
        if len(payload["algorithm_semantics_table"]) < 80:
            return (
                "Error: algorithm_semantics_table is too short. "
                "Provide a markdown table describing the mathematical objects, runtime layers, and semantic definitions."
            )
        if not self._looks_like_algorithm_semantics_table(payload["algorithm_semantics_table"]):
            return (
                "Error: algorithm_semantics_table must be a markdown table with columns such as "
                "`Mathematical Object`, `Runtime Layer`, and `Semantic Definition`."
            )
        if len(payload["implementation_pseudocode"]) < 60:
            return (
                "Error: implementation_pseudocode is too short. "
                "Provide a concrete paper-style pseudocode plan for the algorithm, but not real code."
            )
        proposal_warnings: List[str] = []
        if not self._looks_like_formal_pseudocode(payload["implementation_pseudocode"]):
            proposal_warnings.append(
                "implementation_pseudocode_format: preferred reviewer-friendly format is global Inputs/Outputs plus labeled "
                "P1/P2/... steps with per-step Inputs, Outputs, Invariants, Forbidden deviations, and key equations or "
                "update rules. This formatting issue is advisory and does not block proposal creation."
            )
        if len(payload["evaluation_plan"]) < 60:
            return (
                "Error: evaluation_plan is too short. "
                "State how builtin W1/TMV are kept fixed and what additive custom metrics will validate the new method."
            )
        if len(payload["expected_evaluation_outcome"]) < 80:
            return (
                "Error: expected_evaluation_outcome is too short. "
                "State the scenario where the algorithm should work, the expected result pattern beyond builtin W1/TMV, "
                "and why those results would validate the claimed capability."
            )
        if len(payload["inductive_generalization_argument"]) < 80:
            return (
                "Error: inductive_generalization_argument is too short. "
                "State the learned runtime rule for new valid t=0 cells/particles, allowed inference-time inputs/context, "
                "training-only supervision signals, and why inference does not rely on cell-id/row-id/barcode-specific or target-specific memorization."
            )
        if len(payload["unbalanced_decision"]) < 20:
            return (
                "Error: unbalanced_decision is too short. "
                "State clearly whether unbalanced mass is enabled or not, and why."
            )
        if len(payload["stochasticity_decision"]) < 20:
            return (
                "Error: stochasticity_decision is too short. "
                "State clearly whether stochasticity/noise is enabled or not, and why."
            )
        if len(payload["overengineering_self_check"]) < 20:
            return (
                "Error: overengineering_self_check is too short. "
                "Explicitly justify simplicity vs complexity tradeoff."
            )
        combined = "\n\n".join(str(value) for value in payload.values() if not isinstance(value, dict))
        if self._looks_like_impl_code(combined):
            return (
                "Error: proposal contains implementation-code details. "
                "Submit conceptual/theoretical/mathematical description and pseudocode only (no code blocks/imports/defs/classes)."
            )

        mode = self._normalize_algorithm_proposal_mode()
        needs_user_review = mode == "always_user_review"
        needs_agent_review = mode == "agent_decide"
        status = "pending_user_review" if needs_user_review else ("pending_agent_review" if needs_agent_review else "approved_auto")
        now = datetime.utcnow().isoformat() + "Z"
        store = self._proposal_store()
        previous = store.get(algo_id) or {}
        proposal_id = self.planner_file_tools._make_registry_id("proposal")
        proposal_record = {
            "algorithm_id": algo_id,
            "proposal_id": proposal_id,
            "status": status,
            "review_mode": mode,
            "requires_user_review": bool(requires_user_review),
            "created_at": now,
            "updated_at": now,
            "reviewed_at": "",
            "review_decision": "",
            "review_feedback": "",
            "implementation_risks": [],
            "risk_assessment": "",
            "review_history": [],
            "supersedes": str((previous.get("proposal_id") if isinstance(previous, dict) else "") or ""),
            "superseded_by": "",
            "primary_idea_id": primary_idea_value,
            "algorithm_attributes": dict(payload["algorithm_attributes"]),
            "proposal_warnings": list(proposal_warnings),
            "proposal": payload,
        }
        self._persist_proposal_record(algo_id, proposal_record)
        proposal_record = self.planner_file_tools.register_proposal_record(
            algo_id,
            proposal_record,
            previous_record=previous if isinstance(previous, dict) else None,
            reason="create_algorithm_proposal",
        )
        store[algo_id] = proposal_record
        active_context = self.planner_file_tools.set_active_algorithm_context(algo_id)
        self.state["latest_algorithm_proposal_id"] = algo_id
        self.state["active_proposal_id"] = str(active_context.get("proposal_id") or proposal_id)
        if primary_idea_value:
            self.link_algorithm_to_idea(
                idea_id=primary_idea_value,
                algorithm_id=algo_id,
                proposal_id=proposal_id,
                summary=f"Algorithm proposal {proposal_id} is now associated with this research idea.",
            )
            self.planner_file_tools.append_research_idea_attempt(
                primary_idea_value,
                attempt_type="attempt_created",
                summary=f"Created algorithm proposal {proposal_id} for algorithm `{algo_id}`.",
                linked_algorithm_id=algo_id,
                proposal_id=proposal_id,
                evidence_refs=[str(proposal_record.get('proposal_json_path') or '')],
                next_step="Review whether the proposal actually addresses the idea's stated scientific object and success criteria.",
                machine_generated=True,
            )

        workspace_dir = self._proposal_dir(algo_id)
        if status in {"approved", "approved_auto"} and workspace_dir.exists():
            visible_files = [p for p in workspace_dir.iterdir() if p.name != "registry"]
            if visible_files:
                snapshot_id = self.planner_file_tools.create_workspace_snapshot(
                    algo_id,
                    reason="Approved proposal revision before authoring",
                    source="proposal_auto_approved",
                    proposal_id=proposal_id,
                    set_active=True,
                    allow_empty=False,
                )
                self.planner_file_tools.record_decision(
                    algo_id,
                    phase="proposal",
                    decision="proposal_approved",
                    alternatives=[],
                    evidence=[],
                    rationale=f"Proposal {proposal_id} is approved and ready for authoring.",
                    status="final",
                    related_artifacts=[
                        {"artifact_type": "proposal", "artifact_id": proposal_id},
                        {"artifact_type": "workspace_snapshot", "artifact_id": snapshot_id},
                    ],
                )

        active_context = self.planner_file_tools.get_active_algorithm_context()
        self._emit_event(
            "algorithm_proposal_submitted",
            {
                "algorithm_id": algo_id,
                "proposal_id": proposal_id,
                "status": status,
                "review_mode": mode,
                "requires_user_review": bool(requires_user_review),
                "primary_idea_id": primary_idea_value,
                "proposal": payload,
                "proposal_warnings": list(proposal_warnings),
                "proposal_path": proposal_record.get("proposal_path", ""),
                "editable_proposal_path": proposal_record.get("editable_proposal_path") or proposal_record.get("proposal_path", ""),
                "proposal_json_path": proposal_record.get("registry_path") or proposal_record.get("proposal_json_path", ""),
                "proposal_markdown_registry_path": proposal_record.get("proposal_markdown_registry_path", ""),
                "active_algorithm_id": str(active_context.get("algorithm_id") or ""),
                "target_algorithm_id": algo_id,
                "snapshot_id": str(active_context.get("active_snapshot_id") or ""),
                "dirty_since_snapshot": bool(active_context.get("dirty_since_snapshot", False)),
                "proposal_markdown": self._render_proposal_markdown(proposal_record),
                "updated_at": now,
            },
        )

        if needs_user_review:
            question = (
                f"Algorithm proposal '{algo_id}' is ready for review. "
                "Please approve, reject, or request revision."
            )
            self.state["planner_phase"] = "needs_input"
            self.state["planner_need"] = {
                "source": "algorithm_proposal",
                "algorithm_id": algo_id,
                "question": question,
                "reason": "algorithm_proposal_pending_review",
            }
            self._emit_event("planner_needs_input", dict(self.state["planner_need"]))
            self._emit_event(
                "algorithm_proposal_review_requested",
                {
                    "algorithm_id": algo_id,
                    "proposal_id": proposal_id,
                    "review_mode": mode,
                    "question": question,
                    "status": status,
                    "primary_idea_id": primary_idea_value,
                    "proposal_path": proposal_record.get("proposal_path", ""),
                    "editable_proposal_path": proposal_record.get("editable_proposal_path") or proposal_record.get("proposal_path", ""),
                    "proposal_json_path": proposal_record.get("registry_path") or proposal_record.get("proposal_json_path", ""),
                    "proposal_markdown_registry_path": proposal_record.get("proposal_markdown_registry_path", ""),
                    "proposal_markdown": self._render_proposal_markdown(proposal_record),
                    "proposal": payload,
                    "proposal_warnings": list(proposal_warnings),
                },
            )
            warnings_suffix = ""
            if proposal_warnings:
                warnings_suffix = "\nWarnings:\n" + "\n".join(f"- {warning}" for warning in proposal_warnings)
            return (
                f"📝 Proposal submitted for `{algo_id}` and is pending user review.\n"
                "Next: ask the user to approve/reject/revise this proposal in the UI "
                "(or via /api/proposals/review)."
                f"{warnings_suffix}"
            )

        if needs_agent_review:
            self.state["runtime_action"] = {
                "kind": "proposal_agent_review",
                "algorithm_id": algo_id,
                "proposal_id": proposal_id,
                "proposal_path": proposal_record.get("proposal_path", ""),
                "editable_proposal_path": proposal_record.get("editable_proposal_path") or proposal_record.get("proposal_path", ""),
                "review_proposal_path": proposal_record.get("proposal_markdown_registry_path") or proposal_record.get("proposal_path", ""),
                "proposal_json_path": proposal_record.get("registry_path") or proposal_record.get("proposal_json_path", ""),
            }
            self._emit_event(
                "algorithm_proposal_agent_review_requested",
                {
                    "algorithm_id": algo_id,
                    "proposal_id": proposal_id,
                    "primary_idea_id": primary_idea_value,
                    "proposal_path": proposal_record.get("proposal_path", ""),
                    "editable_proposal_path": proposal_record.get("editable_proposal_path") or proposal_record.get("proposal_path", ""),
                    "review_proposal_path": proposal_record.get("proposal_markdown_registry_path") or proposal_record.get("proposal_path", ""),
                    "proposal_json_path": proposal_record.get("registry_path") or proposal_record.get("proposal_json_path", ""),
                    "active_algorithm_id": str(active_context.get("algorithm_id") or ""),
                    "target_algorithm_id": algo_id,
                    "snapshot_id": str(active_context.get("active_snapshot_id") or ""),
                    "dirty_since_snapshot": bool(active_context.get("dirty_since_snapshot", False)),
                    "proposal_warnings": list(proposal_warnings),
                },
            )
            warnings_suffix = ""
            if proposal_warnings:
                warnings_suffix = "\nWarnings:\n" + "\n".join(f"- {warning}" for warning in proposal_warnings)
            return (
                f"🧪 Proposal submitted for `{algo_id}` and queued for runtime evaluator review "
                f"(review_mode={mode})."
                f"{warnings_suffix}"
            )

        warnings_suffix = ""
        if proposal_warnings:
            warnings_suffix = "\nWarnings:\n" + "\n".join(f"- {warning}" for warning in proposal_warnings)
        return (
            f"✅ Proposal submitted for `{algo_id}` and auto-approved "
            f"(review_mode={mode}). You may proceed to workspace initialization."
            f"{warnings_suffix}"
        )

    def revise_algorithm_proposal(
        self,
        algorithm_id: str,
        revision_note: str,
        proposal_markdown: str = "",
        proposal_patch: str = "",
        requires_user_review: bool = False,
    ) -> str:
        """
        Create a new proposal revision from either full markdown replacement or
        a local patch, then route the revision through the normal proposal
        review flow.
        """
        result = self._create_algorithm_proposal_revision(
            algorithm_id=algorithm_id,
            revision_note=revision_note,
            proposal_markdown=proposal_markdown,
            proposal_patch=proposal_patch,
            revision_source="revise_algorithm_proposal",
            requires_user_review=requires_user_review,
        )
        if result.startswith("Error:"):
            return result
        mode = "proposal_patch" if str(proposal_patch or "").strip() else "proposal_markdown"
        active_id = str(self.state.get("active_proposal_id") or "").strip() or "(unknown)"
        return (
            "✅ Proposal revision created via revise_algorithm_proposal.\n"
            f"- algorithm_id: {str(algorithm_id or '').strip().lower()}\n"
            f"- revision_mode: {mode}\n"
            f"- active_proposal_id: {active_id}\n\n"
            f"{result}"
        )

    def _create_algorithm_proposal_revision(
        self,
        algorithm_id: str,
        revision_note: str,
        proposal_patch: str = "",
        proposal_markdown: str = "",
        revision_source: str = "",
        abstract: str = "",
        literature_and_package_grounding: str = "",
        literature: str = "",
        references: str = "",
        problem_statement: str = "",
        problem_mathematical_form: str = "",
        mathematical_derivation_to_algorithm_design: str = "",
        novelty_and_contributions: str = "",
        claimed_capability: str = "",
        objective: str = "",
        theoretical_core: str = "",
        mathematical_abstraction: str = "",
        algorithm_semantics_table: str = "",
        implementation_pseudocode: str = "",
        evaluation_plan: str = "",
        expected_evaluation_outcome: str = "",
        inductive_generalization_argument: str = "",
        mass_modeling_scope: str = "",
        unbalanced_decision: str = "",
        stochasticity_decision: str = "",
        distribution_recovery_argument: str = "",
        overengineering_self_check: str = "",
        uncertainty_and_risks: str = "",
        primary_idea_id: str = "",
        requires_user_review: bool = False,
    ) -> str:
        algo_id = str(algorithm_id or "").strip().lower()
        note = str(revision_note or "").strip()
        if not algo_id:
            return "Error: algorithm_id is required."
        if not note:
            return "Error: revision_note is required."
        current = self._proposal_store().get(algo_id)
        if not isinstance(current, dict):
            return f"Error: proposal not found for algorithm_id='{algo_id}'"
        payload = dict(current.get("proposal") or {})
        previous_proposal_id = str(current.get("proposal_id") or "").strip()
        revision_source_value = (
            str(revision_source or "").strip()
            or ("apply_workspace_patch" if str(proposal_patch or "").strip() else "revise_algorithm_proposal")
        )
        revision_mode = "patch" if str(proposal_patch or "").strip() else "full_markdown"
        previous_markdown = ""
        current_path = self._proposal_markdown_path(algo_id)
        if current_path.exists():
            try:
                previous_markdown = current_path.read_text(encoding="utf-8")
            except Exception:
                previous_markdown = ""
        if not previous_markdown:
            previous_markdown = self._render_proposal_markdown(current)
        patched_markdown = ""
        revision_diff = ""
        if str(proposal_markdown or "").strip() and str(proposal_patch or "").strip():
            return "Error: provide only one of proposal_markdown or proposal_patch."
        if str(proposal_markdown or "").strip():
            patched_markdown = str(proposal_markdown or "").strip()
            revision_diff = self.planner_file_tools._render_diff(
                self._proposal_markdown_path(algo_id),
                previous_markdown,
                patched_markdown,
            )
            payload = self._proposal_payload_from_markdown(patched_markdown, payload)
        if str(proposal_patch or "").strip():
            try:
                patched_markdown = self._proposal_markdown_after_patch(algo_id, str(proposal_patch), current)
            except Exception as e:
                return f"Error: failed to apply proposal_patch: {e}"
            revision_diff = self.planner_file_tools._render_diff(
                self._proposal_markdown_path(algo_id),
                previous_markdown,
                patched_markdown,
            )
            payload = self._proposal_payload_from_markdown(patched_markdown, payload)
        overrides = {
            "abstract": abstract,
            "literature_and_package_grounding": literature_and_package_grounding,
            "literature": literature,
            "references": references,
            "problem_statement": problem_statement,
            "problem_mathematical_form": problem_mathematical_form,
            "mathematical_derivation_to_algorithm_design": mathematical_derivation_to_algorithm_design,
            "novelty_and_contributions": novelty_and_contributions,
            "claimed_capability": claimed_capability,
            "objective": objective,
            "theoretical_core": theoretical_core,
            "mathematical_abstraction": mathematical_abstraction,
            "algorithm_semantics_table": algorithm_semantics_table,
            "implementation_pseudocode": implementation_pseudocode,
            "evaluation_plan": evaluation_plan,
            "expected_evaluation_outcome": expected_evaluation_outcome,
            "inductive_generalization_argument": inductive_generalization_argument,
            "mass_modeling_scope": mass_modeling_scope,
            "unbalanced_decision": unbalanced_decision,
            "stochasticity_decision": stochasticity_decision,
            "distribution_recovery_argument": distribution_recovery_argument,
            "overengineering_self_check": overengineering_self_check,
            "uncertainty_and_risks": uncertainty_and_risks,
        }
        for key, value in overrides.items():
            if str(value or "").strip():
                if key == "literature":
                    existing = str(payload.get("literature_and_package_grounding") or "").strip()
                    incoming = str(value).strip()
                    if existing and incoming not in existing:
                        payload["literature_and_package_grounding"] = f"{existing}\n\nAdditional literature:\n{incoming}"
                    elif existing:
                        payload["literature_and_package_grounding"] = existing
                    else:
                        payload["literature_and_package_grounding"] = incoming
                else:
                    payload[key] = str(value).strip()
        result = self.create_algorithm_proposal(
            algorithm_id=algo_id,
            abstract=payload.get("abstract", ""),
            literature_and_package_grounding=payload.get("literature_and_package_grounding", ""),
            references=payload.get("references", ""),
            problem_statement=payload.get("problem_statement", ""),
            problem_mathematical_form=payload.get("problem_mathematical_form", ""),
            mathematical_derivation_to_algorithm_design=payload.get("mathematical_derivation_to_algorithm_design", ""),
            novelty_and_contributions=payload.get("novelty_and_contributions", ""),
            claimed_capability=payload.get("claimed_capability", ""),
            objective=payload.get("objective", ""),
            theoretical_core=payload.get("theoretical_core", ""),
            mathematical_abstraction=payload.get("mathematical_abstraction", ""),
            algorithm_semantics_table=payload.get("algorithm_semantics_table", ""),
            implementation_pseudocode=payload.get("implementation_pseudocode", ""),
            evaluation_plan=payload.get("evaluation_plan", ""),
            expected_evaluation_outcome=payload.get("expected_evaluation_outcome", ""),
            inductive_generalization_argument=payload.get("inductive_generalization_argument", ""),
            mass_modeling_scope=payload.get("mass_modeling_scope", ""),
            unbalanced_decision=payload.get("unbalanced_decision", ""),
            stochasticity_decision=payload.get("stochasticity_decision", ""),
            distribution_recovery_argument=payload.get("distribution_recovery_argument", ""),
            overengineering_self_check=payload.get("overengineering_self_check", ""),
            uncertainty_and_risks=payload.get("uncertainty_and_risks", ""),
            primary_idea_id=str(primary_idea_id or current.get("primary_idea_id") or "").strip().lower(),
            requires_user_review=requires_user_review,
        )
        if result.startswith("✅") or result.startswith("📝") or result.startswith("🧪"):
            new_record = dict((self.state.get("algorithm_proposals") or {}).get(algo_id) or {})
            new_proposal_id = str(new_record.get("proposal_id") or self.state.get("active_proposal_id") or "").strip()
            if (str(proposal_patch or "").strip() or str(proposal_markdown or "").strip()) and new_record:
                revision_meta = {
                    "source": revision_source_value,
                    "revision_mode": revision_mode,
                    "revision_note": note,
                    "previous_proposal_id": previous_proposal_id,
                    "proposal_id": new_proposal_id,
                    "proposal_diff": revision_diff,
                    "created_at": datetime.utcnow().isoformat() + "Z",
                }
                if str(proposal_patch or "").strip():
                    revision_meta["proposal_patch"] = str(proposal_patch or "")
                else:
                    revision_meta["proposal_markdown_replacement"] = True
                new_record["revision_source"] = revision_source_value
                new_record["revision_note"] = note
                new_record["previous_proposal_id"] = previous_proposal_id
                new_record["proposal_revision"] = revision_meta
                self._persist_proposal_record(algo_id, new_record)
                registry_path = str(new_record.get("registry_path") or "").strip()
                if registry_path:
                    try:
                        Path(registry_path).write_text(json.dumps(new_record, ensure_ascii=False, indent=2), encoding="utf-8")
                    except Exception:
                        pass
                self._proposal_store()[algo_id] = new_record
                runtime_action = self.state.get("runtime_action")
                if (
                    isinstance(runtime_action, dict)
                    and runtime_action.get("kind") == "proposal_agent_review"
                    and str(runtime_action.get("proposal_id") or "") == new_proposal_id
                ):
                    runtime_action.update(
                        {
                            "revision_source": revision_source_value,
                            "revision_mode": revision_mode,
                            "revision_note": note,
                            "previous_proposal_id": previous_proposal_id,
                            "proposal_revision_diff": revision_diff,
                        }
                    )
                    if str(proposal_patch or "").strip():
                        runtime_action["proposal_revision_patch"] = str(proposal_patch or "")
                self._emit_event(
                    "algorithm_proposal_revision_created",
                    {
                        "algorithm_id": algo_id,
                        "previous_proposal_id": previous_proposal_id,
                        "proposal_id": new_proposal_id,
                        "revision_source": revision_source_value,
                        "revision_mode": revision_mode,
                        "revision_note": note,
                        "proposal_path": str(self._proposal_markdown_path(algo_id)),
                        "editable_proposal_path": str(self._proposal_markdown_path(algo_id)),
                        "proposal_diff": revision_diff,
                    },
                )
            self.planner_file_tools.record_decision(
                algo_id,
                phase="proposal",
                decision="proposal_patch_revision" if revision_mode == "patch" else "proposal_full_revision",
                alternatives=[],
                evidence=[str(self._proposal_markdown_path(algo_id))],
                rationale=note,
                status="final",
                related_artifacts=[{"artifact_type": "proposal", "artifact_id": str(self.state.get("active_proposal_id") or "")}],
            )
        return result

    def review_algorithm_proposal(
        self,
        algorithm_id: str,
        decision: str,
        reviewer_feedback: str = "",
        proposal_id: str = "",
        implementation_risks: Optional[List[str]] = None,
        risk_assessment: str = "",
    ) -> str:
        """
        Apply a user review decision to an existing algorithm proposal.
        decision: approve | reject | revise
        """
        mode = self._normalize_algorithm_proposal_mode()
        review_source = str(self.state.get("_proposal_review_source") or "").strip().lower()
        if mode == "always_user_review" and review_source not in {"user_api", "user"}:
            return (
                "Error: review_algorithm_proposal is user-only when algorithm_proposal_review_mode=always_user_review. "
                "Use the proposal review UI/API to approve/reject/revise."
            )

        algo_id = str(algorithm_id or "").strip().lower()
        try:
            self.planner_file_tools._ensure_algorithm_id_mutable(algo_id, action="review_algorithm_proposal")
        except Exception as exc:
            return f"Error: {exc}"
        choice = str(decision or "").strip().lower()
        if choice not in {"approve", "reject", "revise"}:
            return "Error: decision must be one of: approve | reject | revise"
        store = self._proposal_store()
        record = store.get(algo_id)
        if not isinstance(record, dict):
            return f"Error: proposal not found for algorithm_id='{algo_id}'"
        current_proposal_id = str(record.get("proposal_id") or "").strip()
        requested_proposal_id = str(proposal_id or "").strip()
        if not requested_proposal_id:
            active_context = self.planner_file_tools.get_active_algorithm_context()
            active_algo = str(active_context.get("algorithm_id") or "").strip().lower()
            active_proposal = str(active_context.get("proposal_id") or "").strip()
            if active_algo != algo_id or not active_proposal:
                return (
                    "Error: proposal_id is required unless reviewing the current active proposal "
                    f"(active_algorithm_id={active_algo or 'unset'})."
                )
            requested_proposal_id = active_proposal
        if requested_proposal_id != current_proposal_id:
            return (
                f"Error: proposal_id mismatch for algorithm '{algo_id}'. "
                f"Requested '{requested_proposal_id}', but the active latest proposal is '{current_proposal_id}'. "
                "Refusing to apply a stale or cross-bound proposal review."
            )
        registry = self.planner_file_tools._bootstrap_algorithm_registry(algo_id)
        immutable_record = self.planner_file_tools._proposal_record_from_registry(
            algo_id,
            registry,
            requested_proposal_id,
        )
        if immutable_record:
            record = dict(record)
            record.update(immutable_record)

        status_map = {
            "approve": "approved",
            "reject": "rejected",
            "revise": "revision_requested",
        }
        new_status = status_map[choice]
        now = datetime.utcnow().isoformat() + "Z"
        record["status"] = new_status
        record["updated_at"] = now
        record["reviewed_at"] = now
        record["review_decision"] = choice
        record["review_feedback"] = str(reviewer_feedback or "").strip()
        record["implementation_risks"] = [
            str(item).strip()
            for item in (implementation_risks or [])
            if str(item).strip()
        ]
        record["risk_assessment"] = str(risk_assessment or "").strip()
        review_history = list(record.get("review_history") or [])
        review_history.append(
            {
                "at": now,
                "decision": choice,
                "feedback": record["review_feedback"],
                "implementation_risks": list(record["implementation_risks"]),
                "risk_assessment": record["risk_assessment"],
            }
        )
        record["review_history"] = review_history
        self._persist_proposal_record(algo_id, record)
        active_before = self.planner_file_tools.get_active_algorithm_context()
        should_update_active_context = choice == "approve" or str(active_before.get("algorithm_id") or "").strip().lower() in {"", algo_id}
        self.planner_file_tools.register_proposal_record(algo_id, record)
        store[algo_id] = record
        if should_update_active_context:
            active_context = self.planner_file_tools.set_active_algorithm_context(algo_id)
        else:
            registry = self.planner_file_tools._bootstrap_algorithm_registry(algo_id)
            active_context = self.planner_file_tools._build_active_algorithm_context(algo_id, registry)
        self.state["latest_algorithm_proposal_id"] = algo_id
        if should_update_active_context:
            self.state["active_proposal_id"] = str(active_context.get("proposal_id") or record.get("proposal_id") or "")

        if str(self.state.get("planner_phase") or "working") == "needs_input":
            reason = str((self.state.get("planner_need") or {}).get("reason") or "")
            need_algo = str((self.state.get("planner_need") or {}).get("algorithm_id") or "").strip().lower()
            if reason == "algorithm_proposal_pending_review" and need_algo == algo_id:
                self._emit_event("planner_need_resolved", dict(self.state.get("planner_need") or {}))
                self.state["planner_phase"] = "working"
                self.state["planner_need"] = {}

        primary_idea_id = str(record.get("primary_idea_id") or "").strip().lower()
        review_event_payload = {
            "algorithm_id": algo_id,
            "proposal_id": str(record.get("proposal_id") or ""),
            "active_algorithm_id": self.planner_file_tools._current_active_algorithm_id(),
            "target_algorithm_id": algo_id,
            "decision": choice,
            "status": new_status,
            "primary_idea_id": primary_idea_id,
            "review_feedback": record.get("review_feedback", ""),
            "implementation_risks": list(record.get("implementation_risks") or []),
            "risk_assessment": record.get("risk_assessment", ""),
            "risk_path": record.get("risk_path", ""),
            "proposal_path": record.get("proposal_path", ""),
            "editable_proposal_path": record.get("editable_proposal_path") or record.get("proposal_path", ""),
            "proposal_json_path": record.get("registry_path") or record.get("proposal_json_path", ""),
            "proposal_markdown_registry_path": record.get("proposal_markdown_registry_path", ""),
            "snapshot_id": str(active_context.get("active_snapshot_id") or ""),
            "dirty_since_snapshot": bool(active_context.get("dirty_since_snapshot", False)),
            "proposal_markdown": self._render_proposal_markdown(record),
            "updated_at": now,
        }
        related_artifacts = [{"artifact_type": "proposal", "artifact_id": str(record.get("proposal_id") or "")}]
        if choice == "approve":
            workspace_dir = self._proposal_dir(algo_id)
            if workspace_dir.exists():
                visible_files = [p for p in workspace_dir.iterdir() if p.name != "registry"]
                if visible_files:
                    snapshot_id = self.planner_file_tools.create_workspace_snapshot(
                        algo_id,
                        reason="Proposal approved for authoring/review",
                        source="proposal_review_approved",
                        proposal_id=str(record.get("proposal_id") or ""),
                        set_active=True,
                        allow_empty=False,
                    )
                    related_artifacts.append({"artifact_type": "workspace_snapshot", "artifact_id": snapshot_id})
                    if should_update_active_context:
                        active_context = self.planner_file_tools.get_active_algorithm_context()
                    else:
                        refreshed_registry = self.planner_file_tools._bootstrap_algorithm_registry(algo_id)
                        active_context = self.planner_file_tools._build_active_algorithm_context(algo_id, refreshed_registry)
                    review_event_payload["snapshot_id"] = str(active_context.get("active_snapshot_id") or snapshot_id)
                    review_event_payload["dirty_since_snapshot"] = bool(active_context.get("dirty_since_snapshot", False))
        self._emit_event("algorithm_proposal_reviewed", review_event_payload)
        if primary_idea_id:
            self.planner_file_tools.append_research_idea_attempt(
                primary_idea_id,
                attempt_type="proposal_reviewed",
                summary=f"Algorithm proposal {record.get('proposal_id') or ''} for `{algo_id}` received decision={choice}.",
                linked_algorithm_id=algo_id,
                proposal_id=str(record.get("proposal_id") or ""),
                evidence_refs=[str(record.get("proposal_json_path") or "")],
                next_step="Decide whether this proposal review changes the idea's overall resolution status.",
                machine_generated=True,
            )
        self.planner_file_tools.record_decision(
            algo_id,
            phase="proposal",
            decision=f"review_{choice}",
            alternatives=[],
            evidence=[],
            rationale=record.get("review_feedback", "") or f"Proposal review decision: {choice}",
            status="final" if choice in {"approve", "reject"} else "provisional",
            related_artifacts=related_artifacts,
        )

        return (
            f"✅ Proposal review recorded for `{algo_id}`: decision={choice}, status={new_status}.\n"
            f"feedback={record.get('review_feedback', '') or '(none)'}"
        )

    def get_algorithm_proposal_status(self, algorithm_id: str = "") -> str:
        """Return algorithm proposal status (single item or summary list)."""
        store = self._proposal_store()
        target = str(algorithm_id or "").strip().lower()
        if target:
            record = store.get(target)
            if not isinstance(record, dict):
                json_path = self._proposal_json_path(target)
                if json_path.exists():
                    try:
                        raw = json.loads(json_path.read_text(encoding="utf-8"))
                        if isinstance(raw, dict):
                            record = raw
                            store[target] = raw
                    except Exception:
                        record = None
                if not isinstance(record, dict):
                    return f"Error: proposal not found for algorithm_id='{target}'"
            record = dict(record)
            record["editable_proposal_path"] = str(record.get("editable_proposal_path") or self._proposal_markdown_path(target))
            record["proposal_patch_instruction"] = (
                "Prefer revise_algorithm_proposal(...) for revisions: pass proposal_markdown for a full rewrite "
                "or proposal_patch for a small section edit. If you must patch PROPOSAL.md directly, re-read "
                "editable_proposal_path immediately before patching and target that root file. Do not patch "
                "proposal_markdown_registry_path; it is an immutable review/archive copy."
            )
            return json.dumps(record, ensure_ascii=False, indent=2)

        if not store:
            return "No algorithm proposals recorded."
        rows: List[str] = []
        for key in sorted(store.keys()):
            record = store.get(key) or {}
            rows.append(
                f"- {key}: status={record.get('status','')}, review_mode={record.get('review_mode','')}, "
                f"primary_idea_id={record.get('primary_idea_id','') or '(none)'}, updated_at={record.get('updated_at','')}"
            )
        return "Algorithm proposal status:\n" + "\n".join(rows)

    def create_research_idea(
        self,
        title: str,
        primary_track: str,
        problem_definition: str,
        scientific_object: str,
        current_method_failure_mode: str,
        prior_work: str,
        why_this_matters: str,
        why_now: str,
        falsifiable_success_criteria: str,
        non_goals: str,
        evidence_basis: str,
        feasible_direction_families: str,
        feasibility_constraints: str,
        idea_id: str = "",
        alternate_track: str = "",
        requires_user_review: bool = False,
    ) -> str:
        payload = {
            "title": str(title or "").strip(),
            "primary_track": str(primary_track or "").strip().lower(),
            "alternate_track": str(alternate_track or "").strip().lower(),
            "problem_definition": str(problem_definition or "").strip(),
            "scientific_object": str(scientific_object or "").strip(),
            "current_method_failure_mode": str(current_method_failure_mode or "").strip(),
            "prior_work": str(prior_work or "").strip(),
            "why_this_matters": str(why_this_matters or "").strip(),
            "why_now": str(why_now or "").strip(),
            "falsifiable_success_criteria": str(falsifiable_success_criteria or "").strip(),
            "non_goals": str(non_goals or "").strip(),
            "evidence_basis": str(evidence_basis or "").strip(),
            "feasible_direction_families": str(feasible_direction_families or "").strip(),
            "feasibility_constraints": str(feasibility_constraints or "").strip(),
        }
        normalized_idea_id = str(idea_id or "").strip().lower()
        if normalized_idea_id and not re.fullmatch(r"[a-z0-9][a-z0-9_-]{1,95}", normalized_idea_id):
            return "Error: idea_id must match [a-z0-9][a-z0-9_-]{1,95}."
        if not normalized_idea_id:
            normalized_idea_id = self._slugify_research_idea_id(payload["title"])
            if self._load_research_idea_record(normalized_idea_id):
                normalized_idea_id = f"{normalized_idea_id}_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ').lower()}"
        elif self._load_research_idea_record(normalized_idea_id):
            return (
                f"Error: research idea '{normalized_idea_id}' already exists. "
                "Use revise_research_idea(...) to create a new revision."
            )
        validation_error = self._validate_research_idea_payload(payload, current_idea_id=normalized_idea_id)
        if validation_error:
            return validation_error

        mode = self._normalize_idea_review_mode()
        needs_user_review = bool(requires_user_review) or mode == "always_user_review"
        needs_agent_review = mode == "agent_decide" and not needs_user_review
        review_status = "pending_user_review" if needs_user_review else ("pending_agent_review" if needs_agent_review else "approved")
        now = datetime.utcnow().isoformat() + "Z"
        active_target = str(self.state.get("active_research_idea_id") or "").strip().lower()
        is_active = not active_target or active_target == normalized_idea_id
        record = {
            "idea_id": normalized_idea_id,
            "title": payload["title"],
            "primary_track": payload["primary_track"],
            "alternate_track": payload["alternate_track"],
            "review_mode": mode,
            "review_status": review_status,
            "portfolio_status": "active",
            "resolution_status": "unresolved",
            "execution_status": "awaiting_review" if review_status.startswith("pending_") else "idle",
            "active": is_active,
            "created_at": now,
            "updated_at": now,
            "reviewed_at": "",
            "review_feedback": "",
            "review_history": [],
            "linked_algorithms": [],
            "attempt_count": 0,
            **payload,
        }
        self.planner_file_tools._persist_research_idea_record(normalized_idea_id, record)
        record = self.planner_file_tools.register_research_idea_record(
            normalized_idea_id,
            record,
            reason="create_research_idea",
        )
        self._research_idea_store()[normalized_idea_id] = record
        self.state["latest_research_idea_id"] = normalized_idea_id
        if is_active:
            self.planner_file_tools.set_research_idea_active(normalized_idea_id)

        idea_markdown = self.planner_file_tools._render_research_idea_markdown(record)
        self._emit_event(
            "research_idea_submitted",
            {
                "idea_id": normalized_idea_id,
                "title": record["title"],
                "status": review_status,
                "review_mode": mode,
                "primary_track": record["primary_track"],
                "alternate_track": record["alternate_track"],
                "idea_path": record.get("idea_path", ""),
                "idea_markdown": idea_markdown,
                "idea": record,
                "updated_at": now,
            },
        )

        if needs_user_review:
            question = (
                f"Research idea '{normalized_idea_id}' is ready for review. "
                "Please approve, reject, or request revision."
            )
            self.state["planner_phase"] = "needs_input"
            self.state["planner_need"] = {
                "source": "research_idea",
                "idea_id": normalized_idea_id,
                "question": question,
                "reason": "research_idea_pending_review",
            }
            self._emit_event("planner_needs_input", dict(self.state["planner_need"]))
            self._emit_event(
                "research_idea_review_requested",
                {
                    "idea_id": normalized_idea_id,
                    "title": record["title"],
                    "status": review_status,
                    "review_mode": mode,
                    "question": question,
                    "idea_path": record.get("idea_path", ""),
                    "idea_markdown": idea_markdown,
                },
            )
            return (
                f"📝 Research idea submitted for `{normalized_idea_id}` and is pending user review.\n"
                "Next: approve/reject/revise this idea in the UI or via /api/ideas/review."
            )

        if needs_agent_review:
            self.state["runtime_action"] = {
                "kind": "research_idea_agent_review",
                "idea_id": normalized_idea_id,
                "title": record["title"],
                "idea_path": record.get("idea_path", ""),
                "idea_json_path": record.get("idea_json_path", ""),
            }
            self._emit_event(
                "research_idea_agent_review_requested",
                {
                    "idea_id": normalized_idea_id,
                    "title": record["title"],
                    "idea_path": record.get("idea_path", ""),
                    "idea_json_path": record.get("idea_json_path", ""),
                },
            )
            return (
                f"🧪 Research idea submitted for `{normalized_idea_id}` and queued for runtime evaluator review "
                f"(review_mode={mode})."
            )

        self.planner_file_tools.record_research_idea_decision(
            normalized_idea_id,
            phase="review",
            decision="auto_approved",
            rationale="Idea review mode is auto_approve.",
            status="final",
            evidence=[record["evidence_basis"]],
            related_artifacts=[{"artifact_type": "idea_revision", "artifact_id": str(record.get("current_revision_id") or "")}],
        )
        return f"✅ Research idea submitted for `{normalized_idea_id}` and auto-approved."

    def revise_research_idea(
        self,
        idea_id: str,
        revision_note: str,
        title: str = "",
        primary_track: str = "",
        alternate_track: str = "",
        problem_definition: str = "",
        scientific_object: str = "",
        current_method_failure_mode: str = "",
        prior_work: str = "",
        why_this_matters: str = "",
        why_now: str = "",
        falsifiable_success_criteria: str = "",
        non_goals: str = "",
        evidence_basis: str = "",
        feasible_direction_families: str = "",
        feasibility_constraints: str = "",
        requires_user_review: bool = False,
    ) -> str:
        target = str(idea_id or "").strip().lower()
        note = str(revision_note or "").strip()
        if not target:
            return "Error: idea_id is required."
        if not note:
            return "Error: revision_note is required."
        current = self._load_research_idea_record(target)
        if not current:
            return f"Error: research idea not found for idea_id='{target}'"
        payload = {
            "title": str(title or current.get("title") or "").strip(),
            "primary_track": str(primary_track or current.get("primary_track") or "").strip().lower(),
            "alternate_track": str(alternate_track or current.get("alternate_track") or "").strip().lower(),
            "problem_definition": str(problem_definition or current.get("problem_definition") or "").strip(),
            "scientific_object": str(scientific_object or current.get("scientific_object") or "").strip(),
            "current_method_failure_mode": str(current_method_failure_mode or current.get("current_method_failure_mode") or "").strip(),
            "prior_work": str(prior_work or current.get("prior_work") or "").strip(),
            "why_this_matters": str(why_this_matters or current.get("why_this_matters") or "").strip(),
            "why_now": str(why_now or current.get("why_now") or "").strip(),
            "falsifiable_success_criteria": str(falsifiable_success_criteria or current.get("falsifiable_success_criteria") or "").strip(),
            "non_goals": str(non_goals or current.get("non_goals") or "").strip(),
            "evidence_basis": str(evidence_basis or current.get("evidence_basis") or "").strip(),
            "feasible_direction_families": str(feasible_direction_families or current.get("feasible_direction_families") or "").strip(),
            "feasibility_constraints": str(feasibility_constraints or current.get("feasibility_constraints") or "").strip(),
        }
        validation_error = self._validate_research_idea_payload(payload, current_idea_id=target)
        if validation_error:
            return validation_error
        mode = self._normalize_idea_review_mode()
        needs_user_review = bool(requires_user_review) or mode == "always_user_review"
        needs_agent_review = mode == "agent_decide" and not needs_user_review
        review_status = "pending_user_review" if needs_user_review else ("pending_agent_review" if needs_agent_review else "approved")
        now = datetime.utcnow().isoformat() + "Z"
        record = {
            **current,
            **payload,
            "review_mode": mode,
            "review_status": review_status,
            "execution_status": "awaiting_review" if review_status.startswith("pending_") else "idle",
            "updated_at": now,
            "reviewed_at": "",
            "review_feedback": "",
            "review_history": [],
            "current_revision_id": self.planner_file_tools._make_registry_id("idea_revision"),
            "supersedes_revision_id": str(current.get("current_revision_id") or ""),
            "superseded_by_revision_id": "",
        }
        self.planner_file_tools._persist_research_idea_record(target, record)
        record = self.planner_file_tools.register_research_idea_record(
            target,
            record,
            previous_record=current,
            reason=note,
        )
        self._research_idea_store()[target] = record
        if bool(record.get("active")):
            self.planner_file_tools.set_research_idea_active(target)

        idea_markdown = self.planner_file_tools._render_research_idea_markdown(record)
        self._emit_event(
            "research_idea_submitted",
            {
                "idea_id": target,
                "title": record["title"],
                "status": review_status,
                "review_mode": mode,
                "primary_track": record["primary_track"],
                "alternate_track": record["alternate_track"],
                "idea_path": record.get("idea_path", ""),
                "idea_markdown": idea_markdown,
                "idea": record,
                "updated_at": now,
                "revision_note": note,
            },
        )
        self.planner_file_tools.record_research_idea_decision(
            target,
            phase="revision",
            decision="revise_research_idea",
            rationale=note,
            status="final",
            evidence=[record["evidence_basis"]],
            related_artifacts=[
                {"artifact_type": "idea_revision", "artifact_id": str(record.get("supersedes_revision_id") or "")},
                {"artifact_type": "idea_revision", "artifact_id": str(record.get("current_revision_id") or "")},
            ],
        )
        if needs_user_review:
            question = (
                f"Research idea '{target}' revision is ready for review. "
                "Please approve, reject, or request another revision."
            )
            self.state["planner_phase"] = "needs_input"
            self.state["planner_need"] = {
                "source": "research_idea",
                "idea_id": target,
                "question": question,
                "reason": "research_idea_pending_review",
            }
            self._emit_event("planner_needs_input", dict(self.state["planner_need"]))
            self._emit_event(
                "research_idea_review_requested",
                {
                    "idea_id": target,
                    "title": record["title"],
                    "status": review_status,
                    "review_mode": mode,
                    "question": question,
                    "idea_path": record.get("idea_path", ""),
                    "idea_markdown": idea_markdown,
                },
            )
            return f"📝 Research idea revision submitted for `{target}` and is pending user review."
        if needs_agent_review:
            self.state["runtime_action"] = {
                "kind": "research_idea_agent_review",
                "idea_id": target,
                "title": record["title"],
                "idea_path": record.get("idea_path", ""),
                "idea_json_path": record.get("idea_json_path", ""),
            }
            self._emit_event(
                "research_idea_agent_review_requested",
                {
                    "idea_id": target,
                    "title": record["title"],
                    "idea_path": record.get("idea_path", ""),
                    "idea_json_path": record.get("idea_json_path", ""),
                },
            )
            return f"🧪 Research idea revision submitted for `{target}` and queued for runtime evaluator review."
        return f"✅ Research idea revision submitted for `{target}` and auto-approved."

    def review_research_idea(
        self,
        idea_id: str,
        decision: str,
        reviewer_feedback: str = "",
    ) -> str:
        mode = self._normalize_idea_review_mode()
        review_source = str(self.state.get("_idea_review_source") or "").strip().lower()
        if mode == "always_user_review" and review_source not in {"user_api", "user"}:
            return (
                "Error: review_research_idea is user-only when idea_review_mode=always_user_review. "
                "Use the idea review UI/API to approve/reject/revise."
            )
        target = str(idea_id or "").strip().lower()
        choice = str(decision or "").strip().lower()
        if choice not in {"approve", "reject", "revise"}:
            return "Error: decision must be one of: approve | reject | revise"
        record = self._load_research_idea_record(target)
        if not record:
            return f"Error: research idea not found for idea_id='{target}'"
        if choice == "approve" and len(str(record.get("evidence_basis") or "").strip()) < 20:
            return "Error: cannot approve a research idea without a non-empty evidence_basis."
        status_map = {"approve": "approved", "reject": "rejected", "revise": "revise"}
        now = datetime.utcnow().isoformat() + "Z"
        record["review_status"] = status_map[choice]
        record["updated_at"] = now
        record["reviewed_at"] = now
        record["review_feedback"] = str(reviewer_feedback or "").strip()
        record["execution_status"] = "idle" if choice == "approve" else "awaiting_review"
        review_history = list(record.get("review_history") or [])
        review_history.append(
            {
                "at": now,
                "review_status": record["review_status"],
                "feedback": record["review_feedback"],
            }
        )
        record["review_history"] = review_history
        self.planner_file_tools._persist_research_idea_record(target, record)
        self.planner_file_tools.register_research_idea_record(
            target,
            record,
            reason="review_research_idea",
        )
        self._research_idea_store()[target] = record

        if str(self.state.get("planner_phase") or "working") == "needs_input":
            need = dict(self.state.get("planner_need") or {})
            if str(need.get("reason") or "") == "research_idea_pending_review" and str(need.get("idea_id") or "").strip().lower() == target:
                self._emit_event("planner_need_resolved", need)
                self.state["planner_phase"] = "working"
                self.state["planner_need"] = {}

        self._emit_event(
            "research_idea_reviewed",
            {
                "idea_id": target,
                "title": record.get("title", ""),
                "decision": choice,
                "status": record["review_status"],
                "review_feedback": record.get("review_feedback", ""),
                "idea_path": record.get("idea_path", ""),
                "idea_markdown": self.planner_file_tools._render_research_idea_markdown(record),
                "updated_at": now,
            },
        )
        self.planner_file_tools.record_research_idea_decision(
            target,
            phase="review",
            decision=f"review_{choice}",
            rationale=record.get("review_feedback", "") or f"Research idea review decision: {choice}",
            status="final" if choice in {"approve", "reject"} else "provisional",
            evidence=[str(record.get("evidence_basis") or "").strip()],
            related_artifacts=[{"artifact_type": "idea_revision", "artifact_id": str(record.get("current_revision_id") or "")}],
        )
        return (
            f"✅ Research idea review recorded for `{target}`: decision={choice}, status={record['review_status']}.\n"
            f"feedback={record.get('review_feedback', '') or '(none)'}"
        )

    def get_research_idea_status(self, idea_id: str = "") -> str:
        target = str(idea_id or "").strip().lower()
        if target:
            record = self._load_research_idea_record(target)
            if not record:
                return f"Error: research idea not found for idea_id='{target}'"
            return json.dumps(record, ensure_ascii=False, indent=2)
        ideas = list_research_idea_catalog()
        if not ideas:
            return "No research ideas recorded."
        return json.dumps({"ideas": ideas}, ensure_ascii=False, indent=2)

    def list_research_ideas(self, track: str = "", include_inactive: bool = True) -> str:
        selected_track = str(track or "").strip().lower()
        payload = []
        for item in list_research_idea_catalog():
            if selected_track and str(item.get("primary_track") or "").strip().lower() != selected_track:
                continue
            if not include_inactive and not bool(item.get("active")):
                continue
            payload.append(item)
        return json.dumps({"ideas": payload}, ensure_ascii=False, indent=2)

    def set_active_research_idea(self, idea_id: str) -> str:
        target = str(idea_id or "").strip().lower()
        if not target:
            return "Error: idea_id is required."
        record = self._load_research_idea_record(target)
        if not record:
            return f"Error: research idea not found for idea_id='{target}'"
        self.planner_file_tools.set_research_idea_active(target)
        self._emit_event(
            "research_idea_active_changed",
            {
                "idea_id": target,
                "title": record.get("title", ""),
                "active": True,
            },
        )
        return f"✅ Active research idea set to `{target}`."

    def update_research_idea_progress(
        self,
        idea_id: str,
        progress_summary: str,
        resolution_status: str = "",
        execution_status: str = "",
        portfolio_status: str = "",
        linked_algorithm_id: str = "",
        proposal_id: str = "",
        run_id: str = "",
        evidence_refs: Optional[List[str]] = None,
        next_step: str = "",
    ) -> str:
        target = str(idea_id or "").strip().lower()
        summary = str(progress_summary or "").strip()
        if not target:
            return "Error: idea_id is required."
        if not summary:
            return "Error: progress_summary is required."
        record = self._load_research_idea_record(target)
        if not record:
            return f"Error: research idea not found for idea_id='{target}'"
        resolution_value = str(resolution_status or "").strip().lower()
        execution_value = str(execution_status or "").strip().lower()
        portfolio_value = str(portfolio_status or "").strip().lower()
        if resolution_value and resolution_value not in {"unresolved", "partially_resolved", "resolved", "invalidated"}:
            return "Error: resolution_status must be one of unresolved | partially_resolved | resolved | invalidated"
        if execution_value and execution_value not in {"idle", "agent_working", "awaiting_review", "blocked"}:
            return "Error: execution_status must be one of idle | agent_working | awaiting_review | blocked"
        if portfolio_value and portfolio_value not in {"active", "parked", "retired"}:
            return "Error: portfolio_status must be one of active | parked | retired"
        if resolution_value:
            record["resolution_status"] = resolution_value
        if execution_value:
            record["execution_status"] = execution_value
        if portfolio_value:
            record["portfolio_status"] = portfolio_value
        if linked_algorithm_id:
            linked = self.planner_file_tools._normalize_linked_algorithms(
                list(record.get("linked_algorithms") or []) + [str(linked_algorithm_id).strip().lower()]
            )
            record["linked_algorithms"] = linked
        record["updated_at"] = datetime.utcnow().isoformat() + "Z"
        self.planner_file_tools._persist_research_idea_record(target, record)
        self.planner_file_tools.register_research_idea_record(target, record, reason="update_research_idea_progress")
        self._research_idea_store()[target] = record
        attempt = self.planner_file_tools.append_research_idea_attempt(
            target,
            attempt_type="progress_update",
            summary=summary,
            linked_algorithm_id=str(linked_algorithm_id or "").strip().lower(),
            proposal_id=str(proposal_id or "").strip(),
            run_id=str(run_id or "").strip(),
            evidence_refs=evidence_refs or [],
            next_step=str(next_step or "").strip(),
            machine_generated=False,
        )
        self._emit_event(
            "research_idea_progress_updated",
            {
                "idea_id": target,
                "title": record.get("title", ""),
                "progress_summary": summary,
                "resolution_status": record.get("resolution_status", ""),
                "execution_status": record.get("execution_status", ""),
                "portfolio_status": record.get("portfolio_status", ""),
                "linked_algorithm_id": str(linked_algorithm_id or "").strip().lower(),
                "proposal_id": str(proposal_id or "").strip(),
                "run_id": str(run_id or "").strip(),
                "evidence_refs": list(evidence_refs or []),
                "next_step": str(next_step or "").strip(),
                "attempt_id": attempt.get("id", ""),
            },
        )
        return f"✅ Research idea `{target}` progress updated."

    def link_algorithm_to_idea(
        self,
        idea_id: str,
        algorithm_id: str,
        proposal_id: str = "",
        summary: str = "",
    ) -> str:
        target = str(idea_id or "").strip().lower()
        algo_id = str(algorithm_id or "").strip().lower()
        if not target:
            return "Error: idea_id is required."
        if not algo_id:
            return "Error: algorithm_id is required."
        record = self._load_research_idea_record(target)
        if not record:
            return f"Error: research idea not found for idea_id='{target}'"
        linked = self.planner_file_tools._normalize_linked_algorithms(list(record.get("linked_algorithms") or []) + [algo_id])
        record["linked_algorithms"] = linked
        record["updated_at"] = datetime.utcnow().isoformat() + "Z"
        self.planner_file_tools._persist_research_idea_record(target, record)
        self.planner_file_tools.register_research_idea_record(target, record, reason="link_algorithm_to_idea")
        self._research_idea_store()[target] = record

        proposal_record = self._proposal_store().get(algo_id)
        if not isinstance(proposal_record, dict):
            proposal_path = self._proposal_json_path(algo_id)
            if proposal_path.exists():
                try:
                    raw = json.loads(proposal_path.read_text(encoding="utf-8"))
                    proposal_record = raw if isinstance(raw, dict) else {}
                except Exception:
                    proposal_record = {}
            else:
                proposal_record = {}
        if isinstance(proposal_record, dict) and proposal_record:
            proposal_record["primary_idea_id"] = target
            self._persist_proposal_record(algo_id, proposal_record)
            self.planner_file_tools.register_proposal_record(algo_id, proposal_record)
            self._proposal_store()[algo_id] = proposal_record
            proposal_id = str(proposal_id or proposal_record.get("proposal_id") or "").strip()

        self._emit_event(
            "research_idea_linked_algorithm",
            {
                "idea_id": target,
                "title": record.get("title", ""),
                "algorithm_id": algo_id,
                "proposal_id": proposal_id,
                "summary": str(summary or "").strip(),
            },
        )
        return f"✅ Linked algorithm `{algo_id}` to research idea `{target}`."

    def _get_algorithm_proposal_approval(self, algorithm_id: str) -> Tuple[bool, str]:
        algo_id = str(algorithm_id or "").strip().lower()
        if not algo_id:
            return False, "missing algorithm_id"
        store = self._proposal_store()
        record = store.get(algo_id)
        if not isinstance(record, dict):
            json_path = self._proposal_json_path(algo_id)
            if json_path.exists():
                try:
                    raw = json.loads(json_path.read_text(encoding="utf-8"))
                    if isinstance(raw, dict):
                        store[algo_id] = raw
                        record = raw
                except Exception:
                    record = None
        if not isinstance(record, dict):
            status_from_md = self._read_markdown_status(algo_id)
            if status_from_md in {"approved", "approved_auto"}:
                return True, ""
            return (
                False,
                f"No proposal found for '{algo_id}'. Call create_algorithm_proposal(...) first.",
            )
        status = str(record.get("status") or "").strip().lower()
        if status in {"approved", "approved_auto"}:
            return True, ""
        return (
            False,
            f"Proposal for '{algo_id}' is not approved (status={status or 'unknown'}). "
            "Apply user review through the proposal UI/API, then retry.",
        )

    def _get_tool_catalog(self) -> ToolCatalog:
        output_dir = Path(self.state.get("output_dir") or "cytobridge_output")
        output_dir.mkdir(parents=True, exist_ok=True)
        return ToolCatalog(self.state, output_dir, event_sink=self._emit_event)

    def set_tool_activation_mode(self, mode: str = "auto_next_turn") -> str:
        """Set generated-tool activation mode: auto_next_turn or manual_review."""
        if mode not in VALID_ACTIVATION_MODES:
            return f"Error: invalid mode '{mode}'. Valid values: {sorted(VALID_ACTIVATION_MODES)}"

        catalog = self._get_tool_catalog()
        catalog.set_activation_mode(mode)
        self._emit_event("tool_activation_mode_updated", {"mode": mode})
        return f"✅ tool_activation_mode set to {mode}"

    def set_tool_harvest_enabled(self, enabled: bool = False) -> str:
        """Enable/disable downstream end-of-turn reusable-tool harvesting."""
        flag = bool(enabled)
        self.state["tool_harvest_enabled"] = flag
        self._emit_event("tool_harvest_mode_updated", {"enabled": flag})
        state_text = "enabled" if flag else "disabled"
        return f"✅ tool_harvest_enabled={flag} ({state_text})"

    def get_tool_catalog_status(self) -> str:
        """Show active/pending tool registry summary for downstream agent."""
        catalog = self._get_tool_catalog()
        return catalog.dump_json()

    def approve_pending_tools(self, tool_ids: Optional[List[str]] = None) -> str:
        """Approve pending generated tools and activate eligible ones."""
        catalog = self._get_tool_catalog()
        approved = catalog.approve_pending(tool_ids=tool_ids)
        current_turn = int(self.state.get("conversation_turn", 0))
        activation = catalog.activate_pending(current_turn=current_turn)
        return (
            f"✅ approved={approved.get('approved', 0)}, "
            f"activated={len(activation.get('activated', []))}"
        )

    def reject_pending_tools(self, tool_ids: Optional[List[str]] = None, reason: str = "") -> str:
        """Reject pending generated tools."""
        catalog = self._get_tool_catalog()
        result = catalog.reject_pending(tool_ids=tool_ids, reason=reason)
        return f"✅ rejected={result.get('rejected', 0)}"

    def disable_generated_tool(self, name_or_id: str) -> str:
        """Disable a generated (non-builtin) tool by name or tool_id."""
        catalog = self._get_tool_catalog()
        result = catalog.disable_generated(name_or_id)
        if not result.get("ok"):
            return f"Error: {result.get('error', 'Unknown error')}"
        return f"✅ Disabled generated tool: {result.get('name')}"

    def _get_plan_state(self) -> Dict[str, Any]:
        return self.plan_service.get()

    def _set_plan_state(self, plan_state: Dict[str, Any]) -> None:
        self.plan_service.set(plan_state, source="planner_tools")

    def set_plan_from_text(self, plan_text: str, explanation: str = "") -> str:
        """Parse free-text plan and store it as structured plan state."""
        plan_state, err = self.plan_service.set_from_text(
            plan_text,
            explanation=explanation,
            source="set_plan_from_text",
        )
        if err:
            return err
        assert plan_state is not None
        return f"✅ Plan stored.\n{render_plan_state(plan_state)}"

    def update_plan(self, plan: List[Dict[str, Any]], explanation: str = "") -> str:
        """Replace current plan with structured items.

        Each item must contain:
        - step: str
        - status: pending | in_progress | completed
        """
        plan_state, err = self.plan_service.update(
            plan,
            explanation=explanation,
            source="update_plan",
        )
        if err:
            return (
                f"{err}\n"
                "Hint: update_plan requires the FULL plan list with explicit `step` + `status` "
                "for every item. Allowed statuses: pending | in_progress | completed."
            )
        assert plan_state is not None
        return f"✅ Plan updated.\n{render_plan_state(plan_state)}"

    def get_plan_status(self) -> str:
        """Return current structured plan status."""
        return self.plan_service.render()

    def list_skills(self, domain: str = "all") -> str:
        """List available skills with descriptions and SKILL.md paths."""
        requested = str(domain or "all").strip().lower()
        valid_domains = ("planner", "workflow", "downstream", "algorithm")
        if requested not in {"all", *valid_domains}:
            return (
                f"Error: invalid domain '{domain}'. "
                "Valid values: all | planner | workflow | downstream | algorithm."
            )

        def tools_for(target_domain: str) -> SkillsTools:
            if target_domain == "planner":
                return self.skills_tools
            return SkillsTools(
                self.state,
                domain=target_domain,
                event_sink=self._emit_event,
            )

        selected_domains = list(valid_domains) if requested == "all" else [requested]
        combined_skills: List[Dict[str, Any]] = []
        for target_domain in selected_domains:
            try:
                payload = json.loads(tools_for(target_domain).list_skills())
            except Exception as exc:
                combined_skills.append(
                    {
                        "name": f"{target_domain}-skills-unavailable",
                        "description": f"Error while listing this skill domain: {exc}",
                        "skill_md_path": "",
                    }
                )
                continue
            for item in list(payload.get("skills") or []):
                if not isinstance(item, dict):
                    continue
                combined_skills.append(
                    {
                        "name": str(item.get("name", "")),
                        "description": str(item.get("description", "")),
                        "skill_md_path": str(item.get("skill_md_path", "")),
                    }
                )

        result = {
            "skills": combined_skills,
        }
        return json.dumps(result, ensure_ascii=False, indent=2)

    def get_current_workflow_context(self) -> str:
        """Return a compact read-only snapshot of the current workflow state."""
        active_context = self.planner_file_tools.get_active_algorithm_context()
        plan_state = self.plan_service.get()
        final_config = self.state.get("final_config") if isinstance(self.state.get("final_config"), dict) else {}
        active_idea_registry = (
            self.state.get("active_idea_registry")
            if isinstance(self.state.get("active_idea_registry"), dict)
            else {}
        )
        training_runs = [run for run in (self.state.get("training_runs") or []) if isinstance(run, dict)]
        downstream_results = [item for item in (self.state.get("downstream_results") or []) if isinstance(item, dict)]
        downstream_figures = [item for item in (self.state.get("downstream_figures") or []) if isinstance(item, dict)]

        payload = {
            "tool": "get_current_workflow_context",
            "read_only": True,
            "note": (
                "This is the current tool-time runtime state. It does not rebuild the system prompt, "
                "but it is safe to call whenever active bindings or paths are uncertain."
            ),
            "workflow": {
                "workflow_phase": str(self.state.get("workflow_phase") or "intake"),
                "phase_status": deepcopy(self.state.get("phase_status") or {}),
                "task_profile": self._normalize_task_profile(),
            },
            "active_algorithm": {
                "algorithm_id": str(active_context.get("algorithm_id") or ""),
                "workspace_path": str(active_context.get("workspace_path") or ""),
                "editable_proposal_path": str(active_context.get("editable_proposal_path") or ""),
                "proposal_id": str(active_context.get("proposal_id") or ""),
                "proposal_status": str(active_context.get("proposal_status") or ""),
                "proposal_claimed_capability": str(active_context.get("proposal_claimed_capability") or ""),
                "expected_evaluation_outcome": str(active_context.get("expected_evaluation_outcome") or ""),
                "algorithm_attributes": deepcopy(active_context.get("algorithm_attributes") or {}),
                "primary_idea_id": str(active_context.get("primary_idea_id") or ""),
                "registry_path": str(active_context.get("registry_path") or ""),
                "active_snapshot_id": str(active_context.get("active_snapshot_id") or ""),
                "active_baseline_run_id": str(active_context.get("active_baseline_run_id") or ""),
                "dirty_since_snapshot": bool(active_context.get("dirty_since_snapshot", False)),
                "dirty_paths": list(active_context.get("dirty_paths") or []),
                "last_mutation_at": str(active_context.get("last_mutation_at") or ""),
                "last_verified_at": str(active_context.get("last_verified_at") or ""),
            },
            "active_algorithm_campaign": deepcopy(self.state.get("active_algorithm_campaign") or {}),
            "active_research_idea": {
                "idea_id": str(self.state.get("active_research_idea_id") or ""),
                "status": str(active_idea_registry.get("status") or ""),
                "registry_path": str(active_idea_registry.get("path") or active_idea_registry.get("registry_path") or ""),
            },
            "plan": {
                "items": list(plan_state.get("items") or []),
                "explanation": str(plan_state.get("explanation") or ""),
                "updated_at": str(plan_state.get("updated_at") or ""),
                "rendered": render_plan_state(plan_state),
            },
            "paths": {
                "input_path": str(self.state.get("input_path") or ""),
                "preprocessed_path": str(self.state.get("preprocessed_path") or ""),
                "output_dir": str(self.state.get("output_dir") or ""),
                "final_config_path": str(final_config.get("path") or ""),
                "active_algorithm_workspace": str(
                    active_context.get("workspace_path") or self.state.get("planner_algorithm_workspace") or ""
                ),
            },
            "external_literature_tools": {
                "bohrium_paper_search": get_bohrium_paper_search_status(),
            },
            "pending": {
                "planner_need": deepcopy(self.state.get("planner_need") or {}),
                "runtime_action": deepcopy(self.state.get("runtime_action") or {}),
            },
            "latest": {
                "latest_algorithm_proposal_id": str(self.state.get("latest_algorithm_proposal_id") or ""),
                "latest_research_idea_id": str(self.state.get("latest_research_idea_id") or ""),
                "latest_training_algorithm_id": str(self.state.get("latest_training_algorithm_id") or ""),
                "active_proposal_id": str(self.state.get("active_proposal_id") or ""),
                "active_workspace_snapshot_id": str(self.state.get("active_workspace_snapshot_id") or ""),
                "active_baseline_run_id": str(self.state.get("active_baseline_run_id") or ""),
            },
            "artifacts": {
                "training_runs_count": len(training_runs),
                "latest_training_run": self._compact_context_record(
                    training_runs[-1],
                    [
                        "run_id",
                        "candidate_name",
                        "algorithm_id",
                        "training_algorithm_id",
                        "stage",
                        "status",
                        "decision",
                        "run_dir",
                        "model_path",
                        "created_at",
                        "completed_at",
                    ],
                ) if training_runs else {},
                "downstream_results_count": len(downstream_results),
                "latest_downstream_result": self._compact_context_record(
                    downstream_results[-1],
                    ["name", "type", "path", "summary", "created_at"],
                ) if downstream_results else {},
                "downstream_figures_count": len(downstream_figures),
                "latest_downstream_figure": self._compact_context_record(
                    downstream_figures[-1],
                    ["path", "caption", "analysis", "created_at"],
                ) if downstream_figures else {},
                "report_path": str(self.state.get("report_path") or ""),
                "final_summary": self._preview_text(str(self.state.get("final_summary") or ""), max_chars=500),
            },
        }
        return json.dumps(self._json_safe_context_value(payload), ensure_ascii=False, indent=2)

    def set_task_profile(
        self,
        current_stage: str,
        primary_goal: str = "analysis",
        facets: Optional[Dict[str, bool]] = None,
        change_axes: Optional[Dict[str, bool]] = None,
        notes: str = "",
    ) -> str:
        """Set the multi-axis task profile used by workflow gates and reviews."""
        stage_value = str(current_stage or "").strip().lower()
        if stage_value not in TASK_PROFILE_STAGES:
            return (
                f"Error: invalid current_stage '{current_stage}'. "
                f"Valid values: {sorted(TASK_PROFILE_STAGES)}"
            )
        goal_value = str(primary_goal or "").strip().lower()
        if goal_value not in TASK_PROFILE_PRIMARY_GOALS:
            return (
                f"Error: invalid primary_goal '{primary_goal}'. "
                f"Valid values: {sorted(TASK_PROFILE_PRIMARY_GOALS)}"
            )

        normalized = self._normalize_task_profile()
        normalized["current_stage"] = stage_value
        normalized["primary_goal"] = goal_value

        if facets is not None:
            if not isinstance(facets, dict):
                return "Error: facets must be a dict[str, bool]."
            unknown = sorted(set(facets.keys()) - TASK_PROFILE_FACETS)
            if unknown:
                return f"Error: unknown facets: {unknown}. Valid facets: {sorted(TASK_PROFILE_FACETS)}"
            normalized["facets"] = {key: False for key in TASK_PROFILE_FACETS}
            normalized["facets"][goal_value] = True
            for key, value in facets.items():
                normalized["facets"][str(key)] = bool(value)
        else:
            normalized["facets"][goal_value] = True

        if change_axes is not None:
            if not isinstance(change_axes, dict):
                return "Error: change_axes must be a dict[str, bool]."
            unknown = sorted(set(change_axes.keys()) - TASK_PROFILE_CHANGE_AXES)
            if unknown:
                return (
                    f"Error: unknown change_axes: {unknown}. "
                    f"Valid axes: {sorted(TASK_PROFILE_CHANGE_AXES)}"
                )
            normalized["change_axes"] = {key: False for key in TASK_PROFILE_CHANGE_AXES}
            for key, value in change_axes.items():
                normalized["change_axes"][str(key)] = bool(value)

        normalized["notes"] = str(notes or "").strip()
        normalized["risk_flags"] = self._derive_task_profile_risk_flags(normalized)
        self.state["task_profile"] = normalized
        self._emit_event("task_profile_updated", normalized)
        return "✅ task_profile updated.\n" + json.dumps(normalized, ensure_ascii=False, indent=2)

    def check_workflow_gate(self, stage: str = "") -> str:
        """Check whether the current workflow state satisfies the minimum gate for a stage."""
        profile = self._normalize_task_profile()
        profile["risk_flags"] = self._derive_task_profile_risk_flags(profile)
        stage_value = str(stage or profile.get("current_stage") or "").strip().lower()
        if not stage_value:
            stage_value = str(self.state.get("workflow_phase") or "intake").strip().lower()
        if stage_value not in TASK_PROFILE_STAGES:
            return (
                f"Error: invalid stage '{stage_value}'. "
                f"Valid values: {sorted(TASK_PROFILE_STAGES)}"
            )

        blockers: List[str] = []
        warnings: List[str] = []
        checks: List[Dict[str, Any]] = []

        def add_check(name: str, ok: bool, detail: str, severity: str = "blocker") -> None:
            checks.append({"name": name, "ok": bool(ok), "detail": detail, "severity": severity})
            if ok:
                return
            if severity == "warning":
                warnings.append(detail)
            else:
                blockers.append(detail)

        risk_flags = dict(profile.get("risk_flags") or {})
        active_algorithm_context = self.planner_file_tools.get_active_algorithm_context()
        algo_id = str(active_algorithm_context.get("algorithm_id") or self._active_algorithm_id_for_profile()).strip().lower()
        proposal_record = self._approved_algorithm_record(algo_id) if algo_id else {}
        workspace_raw = str(active_algorithm_context.get("workspace_path") or self.state.get("planner_algorithm_workspace") or "").strip()
        workspace_path = Path(workspace_raw).expanduser() if workspace_raw else None
        implementation_map_path = workspace_path / "IMPLEMENTATION_MAP.md" if workspace_path else None
        experiment_registry = (
            self.planner_file_tools._bootstrap_algorithm_registry(algo_id)
            if algo_id and workspace_path and workspace_path.exists()
            else {}
        )

        if stage_value == "proposal":
            add_check(
                "change_axes_declared",
                any(bool(v) for v in (profile.get("change_axes") or {}).values()),
                "Proposal gate blocked: declare at least one changed layer in task_profile.change_axes.",
            )
            if risk_flags.get("semantics_sensitive"):
                add_check(
                    "semantics_sensitive_goal",
                    bool(profile.get("facets", {}).get("new_algorithm") or profile.get("facets", {}).get("reproduction")),
                    "Proposal gate blocked: semantics-sensitive work should be marked as reproduction or new_algorithm.",
                )
            if risk_flags.get("requires_literature"):
                add_check(
                    "literature_plan_notice",
                    True,
                    "Literature may be required later, but proposal stage does not require package/source inspection.",
                    severity="warning",
                )
        elif stage_value == "authoring":
            add_check(
                "approved_proposal",
                bool(proposal_record and str(proposal_record.get("status") or "").lower() in {"approved", "approved_auto"}),
                f"Authoring gate blocked: approved proposal required for algorithm '{algo_id or '(unset)'}'.",
            )
            add_check(
                "workspace_initialized",
                bool(workspace_path and workspace_path.exists()),
                "Authoring gate blocked: initialize the algorithm workspace before editing implementation files.",
            )
            if experiment_registry:
                add_check(
                    "active_workspace_snapshot",
                    bool(str(experiment_registry.get("active_workspace_snapshot_id") or "").strip()),
                    "Authoring gate blocked: experiment registry is missing an active workspace snapshot.",
                )
        elif stage_value == "review":
            add_check(
                "approved_proposal",
                bool(proposal_record and str(proposal_record.get("status") or "").lower() in {"approved", "approved_auto"}),
                f"Review gate blocked: approved proposal required for algorithm '{algo_id or '(unset)'}'.",
            )
            add_check(
                "workspace_initialized",
                bool(workspace_path and workspace_path.exists()),
                "Review gate blocked: algorithm workspace is missing.",
            )
            add_check(
                "implementation_map_exists",
                bool(implementation_map_path and implementation_map_path.exists()),
                "Review gate blocked: IMPLEMENTATION_MAP.md is missing.",
            )
            if implementation_map_path and implementation_map_path.exists():
                map_text = implementation_map_path.read_text(encoding="utf-8", errors="ignore")
                mapped = bool(re.search(r"\bP\d+\b", map_text) and re.search(r"\|\s*completed\s*\|", map_text, re.IGNORECASE))
                add_check(
                    "implementation_map_filled",
                    mapped,
                    "Review gate blocked: IMPLEMENTATION_MAP.md exists but does not yet map pseudocode steps to completed implementation rows.",
                )
            if experiment_registry:
                add_check(
                    "active_proposal_bound",
                    bool(str(experiment_registry.get("active_proposal_id") or "").strip()),
                    "Review gate blocked: experiment registry is missing an active proposal id.",
                )
                add_check(
                    "active_snapshot_bound",
                    bool(str(experiment_registry.get("active_workspace_snapshot_id") or "").strip()),
                    "Review gate blocked: experiment registry is missing an active workspace snapshot id.",
                )
            if risk_flags.get("scalability_sensitive"):
                add_check(
                    "scalability_review_notice",
                    True,
                    "Scalability-sensitive work: verify pairwise complexity, chunking, and large-data feasibility before training.",
                    severity="warning",
                )
        elif stage_value == "training":
            data_path = str(self.state.get("preprocessed_path") or self.state.get("input_path") or "").strip()
            add_check(
                "dataset_available",
                bool(data_path and Path(data_path).exists()),
                "Training gate blocked: no valid dataset path is available in preprocessed_path/input_path.",
            )
            if any(bool(profile.get("facets", {}).get(key)) for key in ("new_algorithm", "reproduction")):
                add_check(
                    "review_completed",
                    bool(implementation_map_path and implementation_map_path.exists()),
                    "Training gate blocked: custom/reproduction work should finish review and create IMPLEMENTATION_MAP.md before running training.",
                )
            if risk_flags.get("requires_baseline"):
                if experiment_registry:
                    baseline_id = str(experiment_registry.get("active_baseline_run_id") or "").strip()
                    has_baseline = bool(baseline_id)
                else:
                    has_baseline = any(
                        str(run.get("candidate_name") or "").strip().lower() == "crufm"
                        or str(run.get("algorithm_id") or "").strip().lower() == "crufm"
                        for run in (self.state.get("training_runs") or [])
                        if isinstance(run, dict)
                    )
                add_check(
                    "baseline_available",
                    has_baseline,
                    "Training warning: no recorded crufm baseline run found yet; compare against a baseline before treating results as final.",
                    severity="warning",
                )
        elif stage_value == "downstream":
            final_config = self.state.get("final_config") or {}
            final_metrics = self.state.get("final_metrics") or {}
            trained_model_path = str(final_config.get("path") or "").strip()
            add_check(
                "trained_model_available",
                bool(trained_model_path and Path(trained_model_path).exists()),
                "Downstream gate blocked: no final trained model path is recorded.",
            )
            add_check(
                "final_metrics_available",
                bool(final_metrics),
                "Downstream gate blocked: final_metrics are missing.",
            )
        elif stage_value == "report":
            add_check(
                "downstream_results_available",
                bool(self.state.get("downstream_results")),
                "Report gate blocked: downstream_results are missing.",
            )
            add_check(
                "final_metrics_available",
                bool(self.state.get("final_metrics")),
                "Report gate blocked: final_metrics are missing.",
            )

        gate_result = {
            "stage": stage_value,
            "task_profile": profile,
            "ok": not blockers,
            "blockers": blockers,
            "warnings": warnings,
            "checks": checks,
            "active_algorithm_context": active_algorithm_context,
            "diagnostic_only": True,
        }
        self.state["last_workflow_gate"] = gate_result
        self._emit_event("workflow_gate_checked", gate_result)
        return json.dumps(gate_result, ensure_ascii=False, indent=2)

    @staticmethod
    def _explicit_bool(value: Any) -> Optional[bool]:
        if isinstance(value, bool):
            return value
        return None

    @classmethod
    def _tmv_gate_status_for_metrics(cls, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Return whether TMV is a hard gate using structured runtime metadata only."""
        if not isinstance(metrics, dict):
            return {"required": True, "reason": "missing_metrics_payload"}

        algorithm_attributes = metrics.get("algorithm_attributes")
        if isinstance(algorithm_attributes, dict):
            mass_scope = cls._normalize_mass_modeling_scope(algorithm_attributes.get("mass_modeling_scope"))
            if mass_scope:
                return {
                    "required": mass_scope == "models_unbalanced_mass",
                    "reason": str(
                        algorithm_attributes.get("tmv_gate_reason")
                        or "metrics.algorithm_attributes.mass_modeling_scope"
                    ),
                    "mass_modeling_scope": mass_scope,
                }
            for key in ("tmv_gate_required", "requires_tmv_gate", "mass_modeling_enabled", "models_unbalanced_mass"):
                explicit = cls._explicit_bool(algorithm_attributes.get(key))
                if explicit is not None:
                    return {"required": explicit, "reason": f"metrics.algorithm_attributes.{key}"}

        proposal = metrics.get("proposal")
        if isinstance(proposal, dict):
            proposal_attributes = proposal.get("algorithm_attributes")
            if isinstance(proposal_attributes, dict):
                nested = dict(metrics)
                nested["algorithm_attributes"] = proposal_attributes
                nested_status = cls._tmv_gate_status_for_metrics(nested)
                if nested_status.get("reason"):
                    return nested_status
            mass_scope = cls._normalize_mass_modeling_scope(proposal.get("mass_modeling_scope"))
            if mass_scope:
                return {
                    "required": mass_scope == "models_unbalanced_mass",
                    "reason": "metrics.proposal.mass_modeling_scope",
                    "mass_modeling_scope": mass_scope,
                }

        for key in ("tmv_gate_required", "requires_tmv_gate", "mass_modeling_enabled", "models_unbalanced_mass"):
            explicit = cls._explicit_bool(metrics.get(key))
            if explicit is not None:
                return {"required": explicit, "reason": f"metrics.{key}"}

        config = metrics.get("config")
        if isinstance(config, dict):
            model = config.get("model")
            if isinstance(model, dict):
                components = model.get("components")
                if isinstance(components, (list, tuple, set)):
                    normalized = {str(item).strip().lower() for item in components}
                    return {
                        "required": "growth" in normalized,
                        "reason": "config.model.components",
                        "model_components": sorted(normalized),
                    }

        return {"required": True, "reason": "default_require_tmv_gate_when_mass_modeling_unknown"}

    @classmethod
    def _infer_training_verdict(cls, metrics: Dict[str, Any]) -> Dict[str, Any]:
        verdict = {
            "status": "provisional",
            "reasons": [],
            "likely_cause": "",
            "next_action": "",
            "tmv_pass": None,
            "tmv_gate_required": True,
            "tmv_gate_reason": "",
            "w1_manual_check_required": True,
        }
        if not isinstance(metrics, dict):
            verdict.update(
                {
                    "status": "rejected",
                    "reasons": ["metrics payload is missing or invalid"],
                    "likely_cause": "runtime_or_metrics_error",
                    "next_action": "inspect the training error before any scientific interpretation",
                    "w1_manual_check_required": False,
                }
            )
            return verdict

        tmv_gate = cls._tmv_gate_status_for_metrics(metrics)
        tmv_gate_required = bool(tmv_gate.get("required", True))
        verdict["tmv_gate_required"] = tmv_gate_required
        verdict["tmv_gate_reason"] = str(tmv_gate.get("reason") or "")

        if metrics.get("error"):
            verdict.update(
                {
                    "status": "rejected",
                    "reasons": [str(metrics.get("error"))],
                    "likely_cause": "training_runtime_failure",
                    "next_action": "fix the runtime or numerical issue before comparing algorithms",
                    "w1_manual_check_required": False,
                }
            )
            return verdict

        if bool(metrics.get("training_timed_out")):
            timeout_message = str(metrics.get("training_timeout_message") or "").strip()
            verdict.update(
                {
                    "status": "rejected",
                    "reasons": [timeout_message or "training exceeded the wall-clock budget"],
                    "likely_cause": "training_runtime_budget_timeout",
                    "next_action": "reduce training cost or improve convergence before comparing this run",
                    "w1_manual_check_required": False,
                }
            )
            return verdict

        if bool(metrics.get("inference_timed_out")):
            timeout_message = str(metrics.get("inference_timeout_message") or "").strip()
            verdict.update(
                {
                    "status": "rejected",
                    "reasons": [timeout_message or "inference/evaluation exceeded the wall-clock budget"],
                    "likely_cause": "inference_runtime_budget_timeout",
                    "next_action": "fix or simplify rollout/evaluation code before comparing this run",
                    "w1_manual_check_required": False,
                }
            )
            return verdict

        tmv_scores = metrics.get("tmv_scores")
        if isinstance(tmv_scores, list) and tmv_scores:
            try:
                tmv_values = [float(x) for x in tmv_scores]
                # Keep the run-level verdict aligned with campaign Stage gates.
                # A looser local threshold makes metrics.json say tmv_pass=True
                # while the campaign correctly rejects the same run.
                tmv_threshold = 0.2
                tmv_ok = all(x < tmv_threshold for x in tmv_values)
                verdict["tmv_pass"] = tmv_ok
                if tmv_gate_required and not tmv_ok:
                    verdict["status"] = "rejected"
                    verdict["reasons"].append(
                        f"TMV quality gate failed: at least one TMV is >= {tmv_threshold}"
                    )
                    verdict["likely_cause"] = "training_quality_or_hyperparameter_issue"
                    verdict["next_action"] = (
                        "inspect convergence/stability first, then try hyperparameter adjustment before modifying the algorithm"
                    )
                elif not tmv_gate_required:
                    verdict["reasons"].append(
                        "TMV recorded as a diagnostic only because this run is not configured to model unbalanced mass."
                    )
            except Exception:
                if tmv_gate_required:
                    verdict["reasons"].append("TMV quality gate could not be evaluated automatically")
                else:
                    verdict["reasons"].append("TMV diagnostic could not be evaluated automatically")
        elif tmv_gate_required:
            verdict["reasons"].append("TMV scores are missing")
        else:
            verdict["reasons"].append(
                "TMV hard gate skipped because this run is not configured to model unbalanced mass."
            )

        w1_scores = metrics.get("w1_scores")
        if not (isinstance(w1_scores, list) and w1_scores):
            verdict["reasons"].append("W1 scores are missing")
            if verdict["status"] != "rejected":
                verdict["status"] = "rejected"
                verdict["likely_cause"] = "evaluation_missing"
                verdict["next_action"] = "fix evaluation output before treating the run as complete"
                verdict["w1_manual_check_required"] = False
        else:
            if verdict["status"] != "rejected":
                verdict["status"] = "provisional"
                if not verdict["likely_cause"]:
                    verdict["likely_cause"] = "needs_manual_w1_scale_check"
                if not verdict["next_action"]:
                    verdict["next_action"] = (
                        "verify that W1 is within the latent data scale range; if not, inspect training quality and hyperparameters first"
                    )
        return verdict

    @staticmethod
    def _summarize_backend_preview(preflight_cache: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not isinstance(preflight_cache, dict):
            return {}
        backend = preflight_cache.get("backend")
        if backend is None:
            return {}
        coupling = getattr(backend, "coupling", None)
        path = getattr(backend, "path", None)
        mass = getattr(backend, "mass", None)
        summary: Dict[str, Any] = {
            "backend_class": backend.__class__.__name__,
            "coupling_class": coupling.__class__.__name__ if coupling is not None else "",
            "path_class": path.__class__.__name__ if path is not None else "",
            "mass_class": mass.__class__.__name__ if mass is not None else "",
        }
        for attr in ("solver_kind", "reg_type", "reg", "reg_m", "alpha_regm", "delta", "chunk_size"):
            value = getattr(coupling, attr, None) if coupling is not None else None
            if value is not None:
                summary[attr] = value
        for attr in ("sigma",):
            value = getattr(path, attr, None) if path is not None else None
            if value is not None:
                summary[attr] = value
        return summary

    def preview_training_run(
        self,
        candidate_name: str = None,
        training_algorithm_id: str = None,
        stage: str = "final",
        config_overrides: Optional[Dict[str, Any]] = None,
        adata_path: str = None,
        device: str = None,
        run_smoke_test: bool = True,
    ) -> str:
        """Dry-run training preview with effective call-chain inspection included."""
        return self._inspect_training_chain(
            candidate_name=candidate_name,
            training_algorithm_id=training_algorithm_id,
            stage=stage,
            config_overrides=config_overrides,
            adata_path=adata_path,
            device=device,
            run_smoke_test=run_smoke_test,
        )

    @staticmethod
    def _preview_metric_expected_ranges(metric_params: Dict[str, Any]) -> Dict[str, Dict[str, Optional[float]]]:
        def _range_from_spec(spec: Any) -> Optional[Dict[str, Optional[float]]]:
            if isinstance(spec, dict):
                lo = spec.get("min", spec.get("lower", spec.get("expected_min")))
                hi = spec.get("max", spec.get("upper", spec.get("expected_max")))
            elif isinstance(spec, (list, tuple)) and len(spec) >= 2:
                lo, hi = spec[0], spec[1]
            else:
                return None
            try:
                lo_value = float(lo) if lo is not None else None
                hi_value = float(hi) if hi is not None else None
            except Exception:
                return None
            return {"min": lo_value, "max": hi_value}

        ranges: Dict[str, Dict[str, Optional[float]]] = {
            "w1_scores": {"min": 0.0, "max": None},
            "tmv_scores": {"min": 0.0, "max": None},
        }
        if not isinstance(metric_params, dict):
            return ranges
        for key in ("expected_ranges", "metric_ranges", "custom_metric_ranges"):
            value = metric_params.get(key)
            if isinstance(value, dict):
                for metric_name, spec in value.items():
                    parsed = _range_from_spec(spec)
                    if parsed is not None:
                        ranges[str(metric_name)] = parsed
        metrics_spec = metric_params.get("metrics")
        if isinstance(metrics_spec, dict):
            for metric_name, spec in metrics_spec.items():
                parsed = _range_from_spec(spec)
                if parsed is not None:
                    ranges[str(metric_name)] = parsed
        elif isinstance(metrics_spec, list):
            for item in metrics_spec:
                if not isinstance(item, dict):
                    continue
                metric_name = str(item.get("name") or item.get("metric") or "").strip()
                if not metric_name:
                    continue
                parsed = _range_from_spec(item)
                if parsed is not None:
                    ranges[metric_name] = parsed
        return ranges

    @staticmethod
    def _infer_preview_metric_range(metric_name: str) -> Optional[Dict[str, Optional[float]]]:
        name = str(metric_name or "").strip().lower()
        bounded_unit_tokens = (
            "auc",
            "auroc",
            "auprc",
            "accuracy",
            "balanced_accuracy",
            "f1",
            "precision",
            "recall",
            "specificity",
            "sensitivity",
            "fraction",
            "proportion",
            "probability",
            "nmi",
        )
        if name == "ari" or name.endswith("_ari") or "adjusted_rand" in name:
            return {"min": -1.0, "max": 1.0}
        if any(token in name for token in bounded_unit_tokens):
            return {"min": 0.0, "max": 1.0}
        if any(token in name for token in ("distance", "loss", "error", "mse", "mae", "rmse", "w1", "tmv")):
            return {"min": 0.0, "max": None}
        return None

    @staticmethod
    def _flatten_preview_numeric_values(value: Any) -> List[float]:
        values: List[float] = []
        if isinstance(value, bool) or value is None:
            return values
        if isinstance(value, (int, float)):
            values.append(float(value))
            return values
        if isinstance(value, dict):
            for item in value.values():
                values.extend(PlannerTools._flatten_preview_numeric_values(item))
            return values
        if isinstance(value, (list, tuple)):
            for item in value:
                values.extend(PlannerTools._flatten_preview_numeric_values(item))
            return values
        try:
            import numpy as np

            if isinstance(value, np.ndarray):
                for item in value.reshape(-1).tolist():
                    values.extend(PlannerTools._flatten_preview_numeric_values(item))
                return values
            if isinstance(value, np.generic):
                values.append(float(value.item()))
                return values
        except Exception:
            pass
        return values

    @classmethod
    def _preview_metric_sanity(
        cls,
        metrics: Dict[str, Any],
        *,
        metric_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        expected_ranges = cls._preview_metric_expected_ranges(metric_params)
        metric_values: Dict[str, Any] = {}
        for key in ("w1_scores", "tmv_scores"):
            if key in metrics:
                metric_values[key] = metrics.get(key)
        custom_metrics = metrics.get("custom_metrics")
        if isinstance(custom_metrics, dict):
            for key, value in custom_metrics.items():
                metric_values[str(key)] = value

        warnings: List[str] = []
        checks: Dict[str, Any] = {}
        for metric_name, raw_value in metric_values.items():
            values = cls._flatten_preview_numeric_values(raw_value)
            check: Dict[str, Any] = {
                "count": len(values),
                "min": None,
                "max": None,
                "finite": True,
                "expected_range": None,
                "range_source": "",
                "ok": True,
            }
            if not values:
                check["ok"] = False
                warnings.append(f"{metric_name}: no numeric value returned")
                checks[metric_name] = check
                continue
            finite_values = [v for v in values if math.isfinite(v)]
            check["finite"] = len(finite_values) == len(values)
            if not check["finite"]:
                check["ok"] = False
                warnings.append(f"{metric_name}: contains NaN or inf")
            if finite_values:
                check["min"] = min(finite_values)
                check["max"] = max(finite_values)
            expected = expected_ranges.get(metric_name)
            source = "declared"
            if expected is None:
                expected = cls._infer_preview_metric_range(metric_name)
                source = "inferred" if expected is not None else ""
            if expected is not None:
                lo = expected.get("min")
                hi = expected.get("max")
                check["expected_range"] = {"min": lo, "max": hi}
                check["range_source"] = source
                if finite_values:
                    if lo is not None and min(finite_values) < float(lo) - 1e-8:
                        check["ok"] = False
                        warnings.append(
                            f"{metric_name}: value below expected range "
                            f"(min={min(finite_values):.6g}, expected_min={lo})"
                        )
                    if hi is not None and max(finite_values) > float(hi) + 1e-8:
                        check["ok"] = False
                        warnings.append(
                            f"{metric_name}: value above expected range "
                            f"(max={max(finite_values):.6g}, expected_max={hi})"
                        )
            checks[metric_name] = check
        return {
            "ok": not warnings,
            "warnings": warnings,
            "checks": checks,
            "note": (
                "This is a 1-epoch preview sanity check. Use it to catch broken inference/metric code "
                "and obvious range mismatches, not to judge final metric quality."
            ),
        }

    def _render_training_chain_report(self, payload: Dict[str, Any]) -> str:
        lines: List[str] = []
        lines.append("## Effective Training Chain")
        lines.append(f"- ok: {bool(payload.get('ok'))}")
        if payload.get("error"):
            lines.append(f"- error: {payload.get('error')}")
        lines.append(f"- stage: {payload.get('stage')}")
        lines.append(f"- training_mode: {payload.get('training_mode')}")
        lines.append(f"- algorithm_id: {payload.get('algorithm_id')}")
        lines.append(f"- data_path: {payload.get('data_path')}")
        lines.append(f"- requested_device: {payload.get('requested_device')}")
        if payload.get("preflight_error"):
            lines.append(f"- preflight_error: {payload.get('preflight_error')}")
        warnings = list(payload.get("preflight_warnings") or [])
        if warnings:
            lines.append(f"- preflight_warnings: {len(warnings)}")
        if payload.get("selected_stage_name"):
            lines.append(f"- selected_stage_name: {payload.get('selected_stage_name')}")
        if payload.get("selected_stage_mode"):
            lines.append(f"- selected_stage_mode: {payload.get('selected_stage_mode')}")
        isolation = payload.get("preview_inspection_isolation")
        if isinstance(isolation, dict):
            lines.append(
                "- preview_inspection_isolated: "
                f"{bool(isolation.get('isolated'))} "
                f"(returncode={isolation.get('returncode')}, "
                f"timed_out={bool(isolation.get('timed_out'))}, "
                f"memory_limit_mb={isolation.get('memory_limit_mb')})"
            )
        artifact_paths = payload.get("preview_artifacts") or {}
        if artifact_paths:
            md_path = str(artifact_paths.get("markdown_path") or "")
            json_path = str(artifact_paths.get("json_path") or "")
            if md_path:
                lines.append(f"- preview_markdown: `{md_path}`")
            if json_path:
                lines.append(f"- preview_json: `{json_path}`")
        lines.append("")

        resolved_training_plan = payload.get("resolved_training_plan") or []
        if resolved_training_plan:
            lines.append("### Resolved Training Plan")
            for item in resolved_training_plan:
                lines.append(
                    f"- {item.get('name') or '<unnamed>'}: mode=`{item.get('mode')}`, "
                    f"train_strategy=`{item.get('train_strategy')}`, epochs={item.get('epochs')}"
                )
            lines.append("")

        lines.append("### Effective Objects")
        for item in payload.get("effective_objects", []):
            source = item.get("source") or {}
            source_text = ""
            if source.get("file"):
                source_text = f" (`{source.get('file')}:{source.get('line', 1)}`)"
            lines.append(
                f"- {item.get('label')}: `{item.get('display_name')}` [{item.get('override_kind')}]"
                f"{source_text}"
            )
        lines.append("")

        registry = payload.get("experiment_registry")
        if isinstance(registry, dict):
            lines.append("### Experiment Registry")
            for key in ("registry_path", "active_proposal_id", "active_snapshot_id", "baseline_run_id"):
                value = str(registry.get(key) or "")
                if value:
                    lines.append(f"- {key}: `{value}`")
            obsolete = registry.get("obsolete_warnings") or []
            if obsolete:
                lines.append(f"- obsolete_warnings: {len(obsolete)}")
            lines.append("")

        lines.append("### Observed Runtime Call Chain")
        for event in payload.get("observed_trace", []):
            if event.get("type") == "phase":
                lines.append(f"- [phase] {event.get('label')}")
                continue
            if event.get("type") != "call":
                continue
            repeat = int(event.get("count", 1))
            suffix = f" x{repeat}" if repeat > 1 else ""
            lines.append(
                f"- `{event.get('display_name')}` -> `{event.get('file')}:{event.get('line')}`{suffix}"
            )
        if payload.get("trace_truncated"):
            lines.append("- [trace truncated] call trace exceeded max_events")
        lines.append("")

        smoke = payload.get("one_epoch_smoke_test")
        if isinstance(smoke, dict):
            lines.append("### One-Epoch Inference And Metric Smoke Test")
            lines.append(f"- enabled: {bool(smoke.get('enabled'))}")
            lines.append(f"- ok: {bool(smoke.get('ok'))}")
            if smoke.get("error"):
                lines.append(f"- error: {smoke.get('error')}")
            if smoke.get("outdir"):
                lines.append(f"- outdir: `{smoke.get('outdir')}`")
            if smoke.get("runtime_sec") is not None:
                lines.append(f"- runtime_sec: {smoke.get('runtime_sec')}")
            sanity = smoke.get("metric_sanity") or {}
            if isinstance(sanity, dict):
                lines.append(f"- metric_sanity_ok: {bool(sanity.get('ok'))}")
                warnings = list(sanity.get("warnings") or [])
                if warnings:
                    lines.append("- metric_sanity_warnings:")
                    for warning in warnings[:12]:
                        lines.append(f"  - {warning}")
                lines.append(
                    "- note: 1 epoch is only for inference/metric sanity; do not interpret metric magnitude as final performance."
                )
            metrics_preview = smoke.get("metrics_preview") or {}
            if isinstance(metrics_preview, dict) and (
                bool(metrics_preview.get("training_timed_out"))
                or bool(metrics_preview.get("inference_timed_out"))
            ):
                lines.append(
                    "- blocking_note: timeout means this preview produced no trusted W1/TMV/claim evidence; "
                    "fix the runtime path before campaign trials."
                )
            lines.append("")

        lines.append("### Structured Summary")
        lines.append("```json")
        lines.append(json.dumps(payload, ensure_ascii=False, indent=2))
        lines.append("```")
        return "\n".join(lines)

    def _allocate_training_preview_artifact_paths(self, payload: Dict[str, Any]) -> Dict[str, str]:
        output_dir = Path(self.state.get("output_dir") or "cytobridge_output").expanduser().resolve()
        preview_dir = output_dir / ".runtime" / "previews"
        preview_dir.mkdir(parents=True, exist_ok=True)
        algorithm_token = str(payload.get("algorithm_id") or payload.get("candidate_name") or "preview").strip() or "preview"
        safe_token = re.sub(r"[^A-Za-z0-9_.-]+", "-", algorithm_token)
        stage_token = str(payload.get("stage") or "final").strip() or "final"
        stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        base_name = f"{safe_token}_{stage_token}_{stamp}"
        return {
            "markdown_path": str(preview_dir / f"{base_name}.preview.md"),
            "json_path": str(preview_dir / f"{base_name}.preview.json"),
        }

    @staticmethod
    def _write_training_preview_artifacts(
        *,
        payload: Dict[str, Any],
        report_markdown: str,
    ) -> None:
        artifact_paths = payload.get("preview_artifacts") or {}
        md_path = Path(str(artifact_paths.get("markdown_path") or "")).expanduser()
        json_path = Path(str(artifact_paths.get("json_path") or "")).expanduser()
        if not md_path or not json_path:
            raise ValueError("preview_artifacts paths must be populated before writing preview files")
        json_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        md_path.write_text(report_markdown, encoding="utf-8")

    def _inspect_training_chain(
        self,
        candidate_name: str = None,
        training_algorithm_id: str = None,
        stage: str = "final",
        config_overrides: Optional[Dict[str, Any]] = None,
        adata_path: str = None,
        device: str = None,
        sample_batch_size: int = 32,
        max_events: int = 160,
        run_smoke_test: bool = True,
    ) -> str:
        """Inspect the effective dry-run training chain by tracing real calls and source refs."""
        if stage not in {"pilot", "final"}:
            return f"Error: stage must be 'pilot' or 'final', got '{stage}'"
        if bool(candidate_name) == bool(training_algorithm_id):
            return "Error: Provide exactly one of candidate_name or training_algorithm_id."
        if config_overrides is not None and not isinstance(config_overrides, dict):
            return "Error: config_overrides must be a dict."
        if int(sample_batch_size) <= 0:
            return "Error: sample_batch_size must be > 0."
        if int(max_events) <= 0:
            return "Error: max_events must be > 0."

        normalized_overrides: Dict[str, Any] = deepcopy(config_overrides or {})
        allow_epoch_override = bool(normalized_overrides.pop("__allow_epoch_override", False))
        epoch_override_reason = str(normalized_overrides.pop("__epoch_override_reason", "")).strip()

        from .adata_manager import AnnDataManager
        import torch
        from unittest.mock import patch
        from CytoBridge.tl.fit import resolve_training_device, fit as cb_fit
        from CytoBridge.tl.methods import neural_ode_step
        from CytoBridge.tl.models import DynamicalModel
        from CytoBridge.tl.trainer import TrainingPipeline
        from CytoBridge.tl.training_algorithm import FlowMatchingBuildContext, ModelBuildContext
        from CytoBridge.tl.flow_matching_backends import (
            build_flow_matching_backend,
            default_flow_matching_backend_builder,
        )

        switch_note = self._apply_pending_input_path()
        user_goal = UserGoal(**self.state["user_goal"])
        requested_device = str(device).strip() if str(device or "").strip() else str(user_goal.device)
        explicit_adata_path = str(adata_path or "").strip()
        data_path = explicit_adata_path or self.state.get("preprocessed_path") or self.state.get("input_path")
        if not data_path:
            msg = "Error: No dataset path available. Run preprocessing or provide a dataset path first."
            return f"{switch_note}\n{msg}" if switch_note else msg

        try:
            manager = AnnDataManager()
            adata = manager.load(data_path, force_reload=True)
        except Exception as exc:
            return f"Error loading data from {data_path}: {exc}"

        VALID_CONFIGS = [
            "dynamical_ot",
            "unbalanced_ot",
            "ruot",
            "crufm",
            "balanced_ot_cfm",
            "sf2m",
            "vgfm",
            "wfrfm",
            "cyto_simulation",
        ]
        chosen_candidate: Optional[CandidateConfig] = None
        if candidate_name:
            if self.state.get("candidates"):
                candidates: List[CandidateConfig] = []
                for raw_candidate in self.state["candidates"]:
                    if isinstance(raw_candidate, CandidateConfig):
                        candidates.append(raw_candidate)
                    elif isinstance(raw_candidate, dict):
                        name = (
                            raw_candidate.get("name")
                            or raw_candidate.get("family")
                            or raw_candidate.get("model_family")
                            or raw_candidate.get("candidate_name")
                            or raw_candidate.get("config_name")
                        )
                        if not str(name or "").strip():
                            continue
                        overrides = raw_candidate.get("overrides")
                        if not isinstance(overrides, dict):
                            overrides = raw_candidate.get("config_overrides")
                        candidates.append(
                            CandidateConfig(
                                name=str(name).strip(),
                                overrides=overrides if isinstance(overrides, dict) else {},
                                rationale=str(raw_candidate.get("rationale") or raw_candidate.get("reasoning") or ""),
                            )
                        )
                chosen_candidate = next((c for c in candidates if c.name == candidate_name), None)
            if chosen_candidate is None:
                if candidate_name in VALID_CONFIGS:
                    chosen_candidate = CandidateConfig(
                        name=candidate_name,
                        overrides={},
                        rationale="User specified directly",
                    )
                else:
                    return f"Error: '{candidate_name}' is not a valid config. Options: {VALID_CONFIGS}"

        workspace_root = get_workspace_root()
        base_target = resolve_training_target(
            candidate=chosen_candidate,
            training_algorithm_id=training_algorithm_id,
            input_adata_path=data_path,
            output_dir=str(self.state.get("output_dir") or "cytobridge_output"),
            stage=stage,
            metadata={
                "user_goal": user_goal.model_dump(),
                "requested_device": requested_device,
                "planner_state_keys": sorted(self.state.keys()),
                "preview_only": True,
                "inspect_chain": True,
                "epoch_policy_base": True,
            },
            search_roots=[get_cellcompass_root() / "training_algorithms"] if training_algorithm_id else None,
            workspace_root=workspace_root,
            config_overrides={},
        )
        base_config = materialize_training_config(
            base_target,
            stage=stage,
            checkpoints_dir=Path(self.state.get("output_dir") or "cytobridge_output")
            / ".runtime"
            / "epoch_policy_preview",
        )
        normalized_overrides, _pruned_config_noops = prune_redundant_config_overrides(
            base_config=base_config,
            overrides=normalized_overrides,
        )
        epoch_blocks = validate_epoch_override_policy(
            base_config=base_config,
            overrides=normalized_overrides,
            allow_epoch_override=allow_epoch_override,
            epoch_override_reason=epoch_override_reason,
        )
        if epoch_blocks:
            return (
                "Error: epoch overrides are blocked by default to avoid undertrained/unstable comparisons. "
                "Redundant default/same-value epoch declarations are allowed; effective epoch changes require "
                "`config_overrides.__allow_epoch_override=true` with non-empty "
                "`config_overrides.__epoch_override_reason`. "
                f"Blocked: {json.dumps(epoch_blocks, ensure_ascii=False)}"
            )

        def _resolve_training_stages(config: Dict[str, Any]) -> List[Dict[str, Any]]:
            training = config.get("training") if isinstance(config, dict) else None
            defaults = training.get("defaults") if isinstance(training, dict) else None
            plan = training.get("plan") if isinstance(training, dict) else None
            if not isinstance(plan, list):
                return []
            resolved: List[Dict[str, Any]] = []
            for stage_cfg in plan:
                if not isinstance(stage_cfg, dict):
                    continue
                merged = dict(defaults or {})
                merged.update(stage_cfg)
                resolved.append(merged)
            return resolved

        def _pick_stage_by_mode(stages: List[Dict[str, Any]], mode: str) -> Optional[Dict[str, Any]]:
            target_mode = str(mode or "").lower()
            for stage_cfg in stages:
                if str(stage_cfg.get("mode", "")).lower() == target_mode:
                    return stage_cfg
            return None

        def _train_strategy_flags(stage_params: Dict[str, Any]) -> Tuple[bool, bool, bool]:
            train_strategy = str(stage_params.get("train_strategy", "s")).lower()
            return ("v" in train_strategy, "g" in train_strategy, "s" in train_strategy)

        def _patch_call(stack: ExitStack, owner: Any, attr_name: str, *, label: Optional[str] = None) -> None:
            if owner is None or not hasattr(owner, attr_name):
                return
            original = getattr(owner, attr_name)
            source_obj = getattr(original, "__func__", original)
            display_label = label or _callable_display_name(original)

            def wrapped(*args, __original=original, __source_obj=source_obj, __display_label=display_label, **kwargs):
                tracer.record_call(__source_obj, label=__display_label)
                return __original(*args, **kwargs)

            stack.enter_context(patch.object(owner, attr_name, wrapped))

        def _safe_shape(value: Any) -> Any:
            shape = getattr(value, "shape", None)
            if shape is None:
                return None
            try:
                return list(map(int, shape))
            except Exception:
                return None

        def _preview_value(value: Any, *, depth: int = 0) -> Any:
            if value is None or isinstance(value, (bool, int, float, str)):
                return value
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, torch.Tensor):
                return {
                    "type": "tensor",
                    "shape": _safe_shape(value),
                    "dtype": str(value.dtype),
                    "device": str(value.device),
                }
            shape = getattr(value, "shape", None)
            if shape is not None:
                return {
                    "type": type(value).__name__,
                    "shape": _safe_shape(value),
                }
            if depth >= 2:
                return repr(value)
            if isinstance(value, dict):
                items = list(value.items())[:20]
                return {str(k): _preview_value(v, depth=depth + 1) for k, v in items}
            if isinstance(value, (list, tuple)):
                return [_preview_value(v, depth=depth + 1) for v in list(value)[:20]]
            return repr(value)

        def _build_preview_model(runtime_device: torch.device, stage_params: Dict[str, Any]) -> torch.nn.Module:
            latent_dim = int(training_data.latent_by_time[0].shape[1])
            if training_target.spec.model_builder is None:
                tracer.mark("build_dynamical_model")
                tracer.record_call(DynamicalModel, label="DynamicalModel")
                return DynamicalModel(latent_dim, resolved_config["model"])
            tracer.mark("build_custom_model")
            tracer.record_call(training_target.spec.model_builder, label="model_builder")
            model = training_target.spec.model_builder(
                ModelBuildContext(
                    algorithm_id=training_target.spec.algorithm_id,
                    resolved_config=deepcopy(resolved_config),
                    latent_dim=latent_dim,
                    training_data=training_data,
                    device=runtime_device,
                    stage=stage,
                    metadata={
                        "preview_only": True,
                        "inspect_chain": True,
                        "selected_stage_name": str(stage_params.get("name") or ""),
                        "selected_stage_mode": str(stage_params.get("mode") or ""),
                    },
                )
            )
            if not isinstance(model, torch.nn.Module):
                raise TypeError(
                    f"model_builder must return torch.nn.Module, got {type(model).__name__}"
                )
            return model

        tracer = _TrainingChainTracer(workspace_root=workspace_root, max_events=max_events)
        payload: Dict[str, Any] = {
            "ok": False,
            "stage": stage,
            "candidate_name": candidate_name or "",
            "training_algorithm_id": training_algorithm_id or "",
            "data_path": data_path,
            "requested_device": requested_device,
            "effective_objects": [],
            "observed_trace": [],
            "trace_truncated": False,
            "preflight_error": "",
            "preflight_warnings": [],
            "preview_warning": "",
            "sample_batch_error": "",
            "epoch_override_reason": epoch_override_reason,
            "config_overrides": normalized_overrides,
            "pruned_redundant_config_overrides": _pruned_config_noops,
            "one_epoch_smoke_test": {
                "enabled": bool(run_smoke_test),
                "ok": False,
                "skipped": "",
                "note": (
                    "1 epoch is only for inference and metric sanity. "
                    "Do not interpret metric magnitude as final algorithm quality."
                ),
            },
        }
        preview_run_id = datetime.utcnow().strftime("preview_%Y%m%dT%H%M%S%fZ")
        payload["preview_id"] = preview_run_id
        preview_progress_id = (
            f"preview_training:{training_algorithm_id or candidate_name or 'training'}:{stage}:{preview_run_id}"
        )
        payload["progress_id"] = preview_progress_id

        def _preview_progress(message: str, progress: float) -> None:
            try:
                value = max(0.0, min(1.0, float(progress)))
            except Exception:
                value = 0.0
            self._emit_event(
                "training_progress",
                {
                    "id": preview_progress_id,
                    "message": message,
                    "progress": value,
                },
            )

        def _make_runtime_bundle(kind: str, algorithm_id: str) -> TrainingRunBundle:
            root = (
                Path(self.state.get("output_dir") or "cytobridge_output").expanduser().resolve()
                / ".runtime"
                / kind
            )
            root.mkdir(parents=True, exist_ok=True)
            safe_algo = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(algorithm_id or "preview"))
            safe_stage = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(stage or "pilot"))
            run_dir = root / f"{safe_algo}_{safe_stage}_{preview_run_id}"
            run_dir.mkdir(parents=True, exist_ok=False)
            bundle = TrainingRunBundle(
                run_id=run_dir.name,
                run_dir=run_dir,
                logs_dir=run_dir / "logs",
                artifacts_dir=run_dir / "artifacts",
                checkpoints_dir=run_dir / "checkpoints",
                algorithm_snapshot_dir=run_dir / "algorithm_snapshot",
                run_manifest_path=run_dir / "run_manifest.json",
                resolved_config_path=run_dir / "resolved_config.yaml",
                training_log_path=run_dir / "logs" / "training.log",
                planner_context_path=run_dir / "logs" / "planner_context.json",
                trained_model_path=run_dir / "artifacts" / "trained_model.h5ad",
                model_artifact_path=run_dir / "artifacts" / "model_artifact.json",
                model_state_path=run_dir / "artifacts" / "model_state.pt",
                metrics_path=run_dir / "artifacts" / "metrics.json",
            )
            bundle.logs_dir.mkdir(parents=True, exist_ok=True)
            bundle.artifacts_dir.mkdir(parents=True, exist_ok=True)
            bundle.checkpoints_dir.mkdir(parents=True, exist_ok=True)
            bundle.algorithm_snapshot_dir.mkdir(parents=True, exist_ok=True)
            bundle.training_log_path.write_text("", encoding="utf-8")
            return bundle

        def _cleanup_preview_smoke_h5ad(bundle: Optional[TrainingRunBundle]) -> Dict[str, Any]:
            """Preview smoke outputs are scratch; keep metrics/logs but drop large h5ad copies."""
            if bundle is None:
                return {}
            removed: List[Dict[str, Any]] = []
            total_bytes = 0
            try:
                candidates = sorted(Path(bundle.run_dir).rglob("*.h5ad"))
            except Exception:
                candidates = []
            for path in candidates:
                try:
                    if not path.is_file():
                        continue
                    size = int(path.stat().st_size)
                    path.unlink()
                    total_bytes += size
                    removed.append({"path": str(path), "bytes": size})
                except Exception as exc:
                    removed.append({"path": str(path), "error": str(exc)})
            if not removed:
                return {}
            return {
                "policy": (
                    "preview smoke h5ad files are scratch artifacts; metrics, logs, "
                    "resolved config, checkpoints metadata, and preview reports are retained"
                ),
                "removed_count": len([item for item in removed if "bytes" in item]),
                "removed_bytes": total_bytes,
                "items": removed,
            }

        def _attach_custom_preview_registry(algorithm_id: str) -> None:
            if not str(algorithm_id or "").strip():
                return
            try:
                obsolete_entries: List[Dict[str, Any]] = []
                obsolete_path = self.planner_file_tools._registry_obsolete_path(str(algorithm_id))
                if obsolete_path.exists():
                    for line in obsolete_path.read_text(encoding="utf-8").splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            item = json.loads(line)
                        except Exception:
                            continue
                        if isinstance(item, dict):
                            obsolete_entries.append(item)
                registry = self.planner_file_tools._bootstrap_algorithm_registry(str(algorithm_id))
                payload["experiment_registry"] = {
                    "registry_path": str(self.planner_file_tools._registry_index_path(str(algorithm_id))),
                    "active_proposal_id": str(registry.get("active_proposal_id") or ""),
                    "active_snapshot_id": str(registry.get("active_workspace_snapshot_id") or ""),
                    "baseline_run_id": str(registry.get("active_baseline_run_id") or ""),
                    "obsolete_warnings": obsolete_entries,
                }
            except Exception as exc:
                payload["experiment_registry_warning"] = str(exc)

        def _finish_preview_payload() -> str:
            payload["observed_trace"] = tracer.events
            payload["trace_truncated"] = tracer.truncated
            payload["ok"] = (
                not bool(payload.get("error"))
                and not bool(payload.get("preflight_error"))
                and (not run_smoke_test or bool((payload.get("one_epoch_smoke_test") or {}).get("ok")))
            )
            payload["preview_artifacts"] = self._allocate_training_preview_artifact_paths(payload)
            if str(payload.get("training_mode") or "") == "custom" and str(payload.get("algorithm_id") or ""):
                preview_registry_entry = self.planner_file_tools.record_preview_artifacts(
                    str(payload.get("algorithm_id")),
                    preview_artifacts=payload["preview_artifacts"],
                    stage=stage,
                    data_path=data_path,
                    requested_device=requested_device,
                    ok=payload["ok"],
                )
                payload.setdefault("experiment_registry", {})
                payload["experiment_registry"]["latest_preview"] = preview_registry_entry
            report = self._render_training_chain_report(payload)
            self._write_training_preview_artifacts(payload=payload, report_markdown=report)
            self.state["last_training_preview"] = payload
            self.state["last_training_preview_markdown_path"] = payload["preview_artifacts"].get("markdown_path", "")
            self.state["last_training_preview_json_path"] = payload["preview_artifacts"].get("json_path", "")
            self._emit_event("training_run_previewed", payload)
            self._emit_event("training_chain_inspected", payload)
            if payload["ok"]:
                _preview_progress("Preview completed", 1.0)
            else:
                _preview_progress("Preview completed with warnings or errors", 1.0)
            return f"{switch_note}\n{report}" if switch_note else report

        _preview_progress("Preview: resolving training target", 0.03)

        if training_algorithm_id and should_isolate_training_target("custom"):
            payload["training_mode"] = "custom"
            payload["algorithm_id"] = str(training_algorithm_id)
            payload["base_config_name"] = ""
            payload["effective_objects"] = [
                {
                    "label": "preview_inspection",
                    "display_name": "cytobridge_agent.tools.training_isolation_worker",
                    "override_kind": "isolated_subprocess",
                    "source": {},
                }
            ]
            _attach_custom_preview_registry(str(training_algorithm_id))
            try:
                _preview_progress("Preview: running isolated chain inspection", 0.22)
                tracer.mark("isolated_preview_inspection_subprocess")
                inspection_bundle = _make_runtime_bundle("preview_inspect", str(training_algorithm_id))
                inspection_metrics, inspection_error, inspection_isolation = run_training_in_subprocess(
                    adata_path=str(data_path),
                    stage=stage,
                    device=requested_device,
                    bundle=inspection_bundle,
                    candidate_name="",
                    training_algorithm_id=str(training_algorithm_id or ""),
                    output_dir=str(self.state.get("output_dir") or "cytobridge_output"),
                    workspace_root=str(workspace_root),
                    config_overrides=normalized_overrides,
                    claim_metric_spec={},
                    max_epochs=1,
                    purpose="preview_inspect",
                )
                payload["preview_inspection_isolation"] = inspection_isolation
                inspection = dict((inspection_metrics or {}).get("preview_inspection") or {})
                if inspection:
                    payload["training_mode"] = str(inspection.get("training_mode") or payload.get("training_mode") or "")
                    payload["algorithm_id"] = str(inspection.get("algorithm_id") or payload.get("algorithm_id") or "")
                    payload["base_config_name"] = str(inspection.get("base_config_name") or "")
                    payload["resolved_training_plan"] = list(inspection.get("resolved_training_plan") or [])
                    payload["selected_stage_name"] = str(inspection.get("selected_stage_name") or "")
                    payload["selected_stage_mode"] = str(inspection.get("selected_stage_mode") or "")
                    payload["preview_inspection"] = inspection
                    payload.setdefault("preflight_warnings", []).extend(
                        [str(item) for item in list(inspection.get("preflight_warnings") or [])]
                    )
                if inspection_error:
                    payload["preflight_error"] = (
                        str(inspection.get("preflight_error") or "").strip()
                        or f"Isolated preview inspection failed: {inspection_error}"
                    )
                elif inspection.get("preflight_error"):
                    payload["preflight_error"] = str(inspection.get("preflight_error") or "")
            except Exception as exc:
                payload["preflight_error"] = f"Isolated preview inspection failed before result capture: {exc}"
                payload["preview_inspection_traceback"] = traceback.format_exc()

            _preview_progress("Preview: chain inspection finished", 0.50)
            smoke_payload = dict(payload.get("one_epoch_smoke_test") or {})
            smoke_bundle = None
            if not run_smoke_test:
                smoke_payload["skipped"] = "disabled by tool argument"
                _preview_progress("Preview: smoke test skipped", 0.90)
            elif payload.get("preflight_error"):
                smoke_payload["skipped"] = "skipped because isolated dry-run preflight failed"
                _preview_progress("Preview: smoke test skipped after preflight error", 0.90)
            else:
                try:
                    _preview_progress("Preview: running isolated one-epoch smoke training", 0.68)
                    tracer.mark("isolated_one_epoch_smoke_subprocess")
                    smoke_bundle = _make_runtime_bundle("preview_smoke", str(training_algorithm_id))
                    smoke_metrics, smoke_error, smoke_isolation = run_training_in_subprocess(
                        adata_path=str(data_path),
                        stage=stage,
                        device=requested_device,
                        bundle=smoke_bundle,
                        candidate_name="",
                        training_algorithm_id=str(training_algorithm_id or ""),
                        output_dir=str(self.state.get("output_dir") or "cytobridge_output"),
                        workspace_root=str(workspace_root),
                        config_overrides=normalized_overrides,
                        claim_metric_spec={},
                        max_epochs=1,
                        purpose="preview",
                    )
                    smoke_payload["training_isolation"] = smoke_isolation
                    _preview_progress("Preview: checking smoke metrics", 0.92)
                    smoke_metrics = dict(smoke_metrics or {})
                    metric_sanity = self._preview_metric_sanity(smoke_metrics, metric_params={})
                    selected_metric_keys = (
                        "w1_scores",
                        "tmv_scores",
                        "custom_metrics",
                        "custom_metrics_warning",
                        "evaluation_warning",
                        "runtime_sec",
                        "training_timed_out",
                        "training_timeout_message",
                        "inference_timed_out",
                        "inference_timeout_message",
                        "last_completed_epoch",
                        "completed_epochs",
                        "final_train_loss",
                        "best_train_loss",
                        "last_epoch_train_loss",
                        "last_training_stage_name",
                        "last_training_stage_mode",
                        "requested_device",
                        "resolved_device",
                    )
                    metrics_preview = {
                        key: smoke_metrics.get(key)
                        for key in selected_metric_keys
                        if key in smoke_metrics
                    }
                    if isinstance(smoke_metrics.get("training_time_budget"), dict):
                        metrics_preview["training_time_budget"] = smoke_metrics.get("training_time_budget")
                    if isinstance(smoke_metrics.get("inference_time_budget"), dict):
                        metrics_preview["inference_time_budget"] = smoke_metrics.get("inference_time_budget")
                    smoke_payload.update(
                        {
                            "ok": not bool(smoke_error) and bool(metric_sanity.get("ok")),
                            "error": smoke_error or "",
                            "outdir": str(smoke_bundle.run_dir),
                            "runtime_sec": smoke_metrics.get("runtime_sec"),
                            "metrics_preview": metrics_preview,
                            "metric_sanity": metric_sanity,
                            "max_epochs": 1,
                        }
                    )
                except Exception as exc:
                    smoke_payload.update(
                        {
                            "ok": False,
                            "error": str(exc),
                            "traceback": traceback.format_exc(),
                        }
                    )
                    _preview_progress("Preview smoke failed", 0.95)
            cleanup = _cleanup_preview_smoke_h5ad(smoke_bundle)
            if cleanup:
                smoke_payload["storage_cleanup"] = cleanup
            payload["one_epoch_smoke_test"] = smoke_payload
            return _finish_preview_payload()

        training_target = None
        resolved_config: Dict[str, Any] = {}
        training_data = None
        preflight_cache = None
        try:
            tracer.mark("resolve_training_target")
            tracer.record_call(resolve_training_target, label="resolve_training_target")
            training_target = resolve_training_target(
                candidate=chosen_candidate,
                training_algorithm_id=training_algorithm_id,
                input_adata_path=data_path,
                output_dir=str(self.state.get("output_dir") or "cytobridge_output"),
                stage=stage,
                metadata={
                    "user_goal": user_goal.model_dump(),
                    "requested_device": requested_device,
                    "planner_state_keys": sorted(self.state.keys()),
                    "preview_only": True,
                    "inspect_chain": True,
                },
                search_roots=[get_cellcompass_root() / "training_algorithms"] if training_algorithm_id else None,
                workspace_root=workspace_root,
                config_overrides=normalized_overrides,
            )
            _preview_progress("Preview: materializing training config", 0.10)
            tracer.mark("materialize_training_config")
            tracer.record_call(materialize_training_config, label="materialize_training_config")
            preview_ckpt_root = (
                Path(self.state.get("output_dir") or "cytobridge_output").expanduser().resolve()
                / ".runtime"
                / "preview_ckpts"
            )
            preview_ckpt_root.mkdir(parents=True, exist_ok=True)
            preview_ckpt_dir = preview_ckpt_root / f"{training_target.spec.algorithm_id}_{stage}"
            resolved_config = materialize_training_config(
                training_target,
                stage=stage,
                checkpoints_dir=preview_ckpt_dir,
            )
            _preview_progress("Preview: resolving training data", 0.16)
            tracer.mark("resolve_training_data_for_target")
            tracer.record_call(resolve_training_data_for_target, label="resolve_training_data_for_target")
            training_data = resolve_training_data_for_target(
                adata=adata,
                target=training_target,
                resolved_config=resolved_config,
                stage=stage,
                device=requested_device,
                outdir=None,
            )

            resolved_stages = _resolve_training_stages(resolved_config)
            payload["resolved_training_plan"] = [
                {
                    "name": str(stage_cfg.get("name") or ""),
                    "mode": str(stage_cfg.get("mode") or ""),
                    "train_strategy": str(stage_cfg.get("train_strategy") or ""),
                    "epochs": stage_cfg.get("epochs"),
                }
                for stage_cfg in resolved_stages
            ]
            flow_matching_stage = _pick_stage_by_mode(resolved_stages, "flow_matching")
            neural_ode_stage = _pick_stage_by_mode(resolved_stages, "neural_ode")
            if flow_matching_stage is not None:
                stage_params = flow_matching_stage
                stage_index = int(resolved_stages.index(stage_params))
                payload["selected_stage_name"] = str(stage_params.get("name") or "")
                payload["selected_stage_mode"] = "flow_matching"
                runtime_device = resolve_training_device(requested_device)
                effective_batch_size = min(
                    int(sample_batch_size),
                    min(max(1, int(arr.shape[0])) for arr in training_data.latent_by_time),
                )
                preview_model = None
                if training_target.spec.model_builder is not None or training_target.spec.stage_runner is not None:
                    preview_model = _build_preview_model(runtime_device, stage_params)
                if training_target.spec.stage_runner is not None:
                    _preview_progress("Preview: dry-running custom stage", 0.28)
                    tracer.mark("build_training_pipeline")
                    tracer.record_call(TrainingPipeline, label="TrainingPipeline")
                    trainer = TrainingPipeline(
                        preview_model or _build_preview_model(runtime_device, stage_params),
                        resolved_config,
                        effective_batch_size,
                        runtime_device,
                        training_data=training_data,
                        progress_callback=None,
                        flow_matching_backend_builder=training_target.spec.flow_matching_backend_builder,
                        flow_matching_loss_hook=training_target.spec.flow_matching_loss_hook,
                        evaluation_metrics_hook=training_target.spec.evaluation_metrics_hook,
                        evaluation_metrics_params=training_target.spec.evaluation_metrics_params,
                        stage_runner=training_target.spec.stage_runner,
                        inference_context_builder=training_target.spec.inference_context_builder,
                        simulation_hook=training_target.spec.simulation_hook,
                    )
                    with ExitStack() as stack:
                        _patch_call(stack, trainer, "run_custom_stage", label="TrainingPipeline.run_custom_stage")
                        if preview_model is not None and hasattr(preview_model, "forward"):
                            _patch_call(stack, preview_model, "forward", label="model.forward")
                        try:
                            tracer.mark("trainer.run_custom_stage")
                            tracer.record_call(training_target.spec.stage_runner, label="stage_runner")
                            result = trainer.run_custom_stage(stage_params, stage_index, preview_only=True)
                        except NotImplementedError as exc:
                            payload["preview_warning"] = (
                                f"Custom stage_runner does not support preview_only dry-run: {exc}"
                            )
                        except Exception as exc:
                            payload["preflight_error"] = f"Custom stage_runner preview failed: {exc}"
                        else:
                            payload["custom_stage_preview"] = {
                                "stage_summary": _preview_value(getattr(result, "stage_summary", {})),
                                "logs": _preview_value(getattr(result, "logs", {})),
                                "artifacts": _preview_value(getattr(result, "artifacts", {})),
                                "model_outputs": _preview_value(getattr(result, "model_outputs", {})),
                            }
                    preflight_cache = {
                        "trainer": trainer,
                        "model": preview_model,
                        "stage_name": stage_params.get("name"),
                        "stage_params": deepcopy(stage_params),
                        "stage_mode": "flow_matching",
                        "custom_stage_runner": True,
                    }
                else:
                    _preview_progress("Preview: checking flow-matching backend", 0.28)
                    regress_v, regress_g, regress_score = _train_strategy_flags(stage_params)
                    build_context = FlowMatchingBuildContext(
                        stage_params=stage_params,
                        training_data=training_data,
                        device=runtime_device,
                        regress_v=regress_v,
                        regress_g=regress_g,
                        regress_score=regress_score,
                        model=preview_model,
                        metadata={
                            "resolved_config": resolved_config,
                            "preview_only": True,
                            "inspect_chain": True,
                            "training_mode": training_target.training_mode,
                            "algorithm_id": training_target.spec.algorithm_id,
                        },
                    )
                    backend_builder = training_target.spec.flow_matching_backend_builder or default_flow_matching_backend_builder
                    tracer.mark("build_flow_matching_backend")
                    tracer.record_call(build_flow_matching_backend, label="build_flow_matching_backend")
                    tracer.record_call(backend_builder, label="effective backend builder")
                    backend = build_flow_matching_backend(backend_builder, build_context=build_context)
                    memory_warning = flow_matching_memory_preflight(
                        training_data=training_data,
                        target=training_target,
                        backend=backend,
                    )
                    if memory_warning:
                        payload.setdefault("preflight_warnings", []).append(memory_warning)
                    X = [x.float().cpu().detach().numpy() for x in training_data.latent_by_time]
                    time_tensor = torch.tensor(list(training_data.time_points), dtype=torch.float32, device=runtime_device)

                    with ExitStack() as stack:
                        for obj, names in [
                            (backend, ["prepare", "sample_batch", "compute_lambda"]),
                            (getattr(backend, "coupling", None), ["build_state", "sample_pairs", "build_pairwise_cost", "compute_ot_coupling"]),
                            (getattr(backend, "path", None), ["sample_time", "sample_noise_like", "sample_xt", "compute_conditional_flow", "compute_conditional_mass", "compute_lambda"]),
                            (getattr(backend, "mass", None), ["compute"]),
                        ]:
                            for name in _iter_traceable_method_names(obj, names):
                                _patch_call(stack, obj, name)

                        if not payload["preflight_error"]:
                            tracer.mark("backend.prepare")
                            state = backend.prepare(X, time_tensor, runtime_device)
                            plans = list(getattr(state, "plans", []) or [])
                            payload["prepared_plan_shapes"] = [_safe_shape(plan) for plan in plans]
                            if len(plans) != len(training_data.time_points) - 1:
                                payload["preflight_error"] = (
                                    f"Coupling plan count mismatch: expected={len(training_data.time_points)-1}, got={len(plans)}."
                                )
                            else:
                                problems: List[str] = []
                                for i, plan in enumerate(plans):
                                    expected_shape = (X[i].shape[0], X[i + 1].shape[0])
                                    try:
                                        pair_batch = backend.coupling.sample_pairs(
                                            state,
                                            X,
                                            i,
                                            min(int(sample_batch_size), max(1, expected_shape[0])),
                                            runtime_device,
                                        )
                                    except Exception as exc:
                                        problems.append(f"t{i}->{i+1}: sample_pairs failed: {exc}")
                                        continue
                                    if pair_batch is None or int(getattr(pair_batch.x0, "shape", [0])[0]) == 0:
                                        problems.append(f"t{i}->{i+1}: sample_pairs returned empty batch")
                                        continue
                                    if getattr(plan, "shape", None) != expected_shape:
                                        payload.setdefault("preflight_warnings", []).append(
                                            f"t{i}->{i+1}: coupling plan shape {getattr(plan, 'shape', None)} does not "
                                            f"match dense shape {expected_shape}; accepted because runtime sample_pairs "
                                            "returned a non-empty batch."
                                        )
                                        continue
                                    try:
                                        total_mass = float(plan.sum())
                                    except Exception:
                                        total_mass = 0.0
                                    if total_mass <= 1e-9:
                                        problems.append(f"t{i}->{i+1}: dense plan total mass too small ({total_mass:.3e})")
                                if problems:
                                    payload["preflight_error"] = "; ".join(problems[:8])

                        preflight_cache = {
                            "backend": backend,
                            "model": preview_model,
                            "stage_name": stage_params.get("name"),
                            "stage_params": deepcopy(stage_params),
                            "stage_mode": "flow_matching",
                        }

                        if not payload["preflight_error"]:
                            tracer.mark("backend.sample_batch")
                            batch = backend.sample_batch(X, time_tensor, effective_batch_size, runtime_device)
                            payload["sample_batch_size"] = int(effective_batch_size)
                            payload["sample_batch_shape"] = {
                                "xt": list(map(int, batch.xt.shape)),
                                "ut": list(map(int, batch.ut.shape)),
                                "gt": list(map(int, batch.gt.shape)),
                                "loss_weights": list(map(int, batch.loss_weights.shape)),
                            }
                            if getattr(batch, "extras", None):
                                payload["sample_batch_extras"] = _preview_value(batch.extras)
                            if int(batch.t_local.numel()) > 0:
                                tracer.mark("backend.compute_lambda")
                                _ = backend.compute_lambda(batch.t_local[: min(8, int(batch.t_local.shape[0]))])
            elif neural_ode_stage is not None:
                _preview_progress("Preview: checking neural ODE stage", 0.28)
                stage_params = neural_ode_stage
                stage_index = int(resolved_stages.index(stage_params))
                payload["selected_stage_name"] = str(stage_params.get("name") or "")
                payload["selected_stage_mode"] = "neural_ode"
                runtime_device = resolve_training_device(requested_device)
                time_points = list(training_data.time_points)
                if len(time_points) < 2:
                    payload["preflight_error"] = f"Need at least 2 time points for neural_ode preview, got {len(time_points)}."
                else:
                    batch_size_cfg = ((resolved_config.get("training") or {}).get("batch_size"))
                    effective_batch_size = int(batch_size_cfg) if isinstance(batch_size_cfg, int) and batch_size_cfg > 0 else min(
                        min(int(x.shape[0]) for x in training_data.latent_by_time),
                        256,
                    )
                    model = _build_preview_model(runtime_device, stage_params)
                    tracer.mark("build_training_pipeline")
                    tracer.record_call(TrainingPipeline, label="TrainingPipeline")
                    trainer = TrainingPipeline(
                        model,
                        resolved_config,
                        effective_batch_size,
                        runtime_device,
                        training_data=training_data,
                        progress_callback=None,
                        flow_matching_backend_builder=training_target.spec.flow_matching_backend_builder,
                        flow_matching_loss_hook=training_target.spec.flow_matching_loss_hook,
                        evaluation_metrics_hook=training_target.spec.evaluation_metrics_hook,
                        evaluation_metrics_params=training_target.spec.evaluation_metrics_params,
                        stage_runner=training_target.spec.stage_runner,
                        inference_context_builder=training_target.spec.inference_context_builder,
                        simulation_hook=training_target.spec.simulation_hook,
                    )
                    with ExitStack() as stack:
                        if training_target.spec.stage_runner is not None:
                            _patch_call(stack, trainer, "run_custom_stage", label="TrainingPipeline.run_custom_stage")
                            if hasattr(model, "forward"):
                                _patch_call(stack, model, "forward", label="model.forward")
                            try:
                                tracer.mark("trainer.run_custom_stage")
                                tracer.record_call(training_target.spec.stage_runner, label="stage_runner")
                                result = trainer.run_custom_stage(stage_params, stage_index, preview_only=True)
                            except NotImplementedError as exc:
                                payload["preview_warning"] = (
                                    f"Custom stage_runner does not support preview_only dry-run: {exc}"
                                )
                            except Exception as exc:
                                payload["preflight_error"] = f"Custom stage_runner preview failed: {exc}"
                            else:
                                payload["custom_stage_preview"] = {
                                    "stage_summary": _preview_value(getattr(result, "stage_summary", {})),
                                    "logs": _preview_value(getattr(result, "logs", {})),
                                    "artifacts": _preview_value(getattr(result, "artifacts", {})),
                                    "model_outputs": _preview_value(getattr(result, "model_outputs", {})),
                                }
                        else:
                            _patch_call(stack, trainer, "_setup_stage", label="TrainingPipeline._setup_stage")
                            _patch_call(stack, trainer, "train_neural_ode_epoch", label="TrainingPipeline.train_neural_ode_epoch")
                            _patch_call(stack, trainer.ode_func, "forward", label="ODEFunc.forward")
                            tracer.mark("trainer._setup_stage")
                            trainer._setup_stage(stage_params)
                            tracer.mark("neural_ode_step")
                            tracer.record_call(neural_ode_step, label="neural_ode_step")
                            x0_full = training_data.latent_by_time[0]
                            take_n = min(int(sample_batch_size), max(1, int(x0_full.shape[0])))
                            x0 = x0_full[:take_n].to(runtime_device)
                            lnw0 = torch.log(torch.ones(take_n, 1, device=runtime_device) / float(take_n))
                            t0 = float(time_points[0])
                            t1 = float(time_points[1])
                            x1, lnw1, e1 = neural_ode_step(trainer.ode_func, x0, lnw0, t0, t1, runtime_device)
                            payload["sample_batch_size"] = int(take_n)
                            payload["neural_ode_step_shape"] = {
                                "x0": list(map(int, x0.shape)),
                                "x1": list(map(int, x1.shape)),
                                "lnw1": list(map(int, lnw1.shape)),
                                "e1": list(map(int, e1.shape)),
                            }
                            payload["neural_ode_stage_summary"] = {
                                "method": "euler",
                                "train_strategy": str(stage_params.get("train_strategy") or ""),
                                "OT_loss": stage_params.get("OT_loss"),
                                "global_mass": bool(stage_params.get("global_mass", False)),
                                "lambda_ot": stage_params.get("lambda_ot"),
                                "lambda_mass": stage_params.get("lambda_mass"),
                                "lambda_energy": stage_params.get("lambda_energy"),
                            }
                    preflight_cache = {
                        "trainer": trainer,
                        "model": model,
                        "stage_name": stage_params.get("name"),
                        "stage_params": deepcopy(stage_params),
                        "stage_mode": "neural_ode",
                        "custom_stage_runner": bool(training_target.spec.stage_runner is not None),
                    }
            else:
                payload["preflight_error"] = "No supported training stage found in resolved config."
        except Exception as exc:
            payload["error"] = str(exc)

        _preview_progress("Preview: chain inspection finished", 0.50)
        payload["observed_trace"] = tracer.events
        payload["trace_truncated"] = tracer.truncated
        if training_target is None:
            payload["preview_artifacts"] = self._allocate_training_preview_artifact_paths(payload)
            report = self._render_training_chain_report(payload)
            self._write_training_preview_artifacts(payload=payload, report_markdown=report)
            self.state["last_training_preview"] = payload
            self.state["last_training_preview_markdown_path"] = payload["preview_artifacts"].get("markdown_path", "")
            self.state["last_training_preview_json_path"] = payload["preview_artifacts"].get("json_path", "")
            self._emit_event("training_run_previewed", payload)
            self._emit_event("training_chain_inspected", payload)
            _preview_progress("Preview failed before training target resolution", 1.0)
            return report

        payload["training_mode"] = training_target.training_mode
        payload["algorithm_id"] = training_target.spec.algorithm_id
        payload["base_config_name"] = training_target.base_config_name
        try:
            review_state = self._inference_review_state(
                training_target,
                resolved_config=resolved_config,
                purpose="preview",
            )
            payload["inference_review"] = {
                "required": bool(review_state.get("required")),
                "ok": bool(review_state.get("ok")),
                "review_hash": str((review_state.get("fingerprint") or {}).get("review_hash") or ""),
                "status": str((review_state.get("record") or {}).get("status") or ""),
                "metrics_trusted_for_campaign": bool(review_state.get("ok")) or not bool(review_state.get("required")),
            }
        except Exception as exc:
            payload["inference_review"] = {
                "required": False,
                "ok": False,
                "warning": str(exc),
                "metrics_trusted_for_campaign": False,
            }
        payload["resolved_config_preview"] = {
            "seed": resolved_config.get("seed"),
            "training_batch_size": ((resolved_config.get("training") or {}).get("batch_size")),
            "ckpt_dir": resolved_config.get("ckpt_dir"),
        }
        if training_data is not None:
            payload["training_data_summary"] = {
                "time_points": [str(tp) for tp in list(training_data.time_points)],
                "num_time_points": len(list(training_data.time_points)),
                "latent_shapes": [list(map(int, x.shape)) for x in list(training_data.latent_by_time)],
            }

        effective_objects: List[Dict[str, Any]] = []
        effective_objects.append(
            _object_summary(
                training_target.spec.training_data_builder or resolve_training_data_for_target,
                label="training_data_builder",
                override_kind="custom" if training_target.spec.training_data_builder is not None else "default",
            )
        )
        effective_objects.append(
            _object_summary(
                training_target.spec.model_builder or DynamicalModel,
                label="model_builder",
                override_kind="custom" if training_target.spec.model_builder is not None else "default",
            )
        )
        if training_target.spec.stage_runner is not None:
            effective_objects.append(
                _object_summary(
                    training_target.spec.stage_runner,
                    label="stage_runner",
                    override_kind="custom",
                )
            )
        if training_target.spec.inference_context_builder is not None:
            effective_objects.append(
                _object_summary(
                    training_target.spec.inference_context_builder,
                    label="inference_context_builder",
                    override_kind="custom",
                )
            )
        if training_target.spec.simulation_hook is not None:
            effective_objects.append(
                _object_summary(
                    training_target.spec.simulation_hook,
                    label="simulation_hook",
                    override_kind="custom",
                )
            )
        selected_stage_mode = str(payload.get("selected_stage_mode") or "")
        if selected_stage_mode == "flow_matching":
            if not (isinstance(preflight_cache, dict) and preflight_cache.get("custom_stage_runner")):
                backend_builder = training_target.spec.flow_matching_backend_builder or default_flow_matching_backend_builder
                effective_objects.append(
                    _object_summary(
                        backend_builder,
                        label="flow_matching_backend_builder",
                        override_kind="custom" if training_target.spec.flow_matching_backend_builder is not None else "default",
                    )
                )
            if isinstance(preflight_cache, dict) and preflight_cache.get("backend") is not None:
                backend = preflight_cache["backend"]
                effective_objects.append(_object_summary(backend.__class__, label="backend_class", override_kind="effective"))
                effective_objects.append(_object_summary(backend.prepare, label="backend.prepare", override_kind="effective"))
                effective_objects.append(_object_summary(backend.sample_batch, label="backend.sample_batch", override_kind="effective"))
                effective_objects.append(_object_summary(backend.compute_lambda, label="backend.compute_lambda", override_kind="effective"))
                coupling = getattr(backend, "coupling", None)
                path = getattr(backend, "path", None)
                mass = getattr(backend, "mass", None)
                if coupling is not None:
                    effective_objects.append(_object_summary(coupling.__class__, label="coupling_class", override_kind="effective"))
                    effective_objects.append(_object_summary(coupling.sample_pairs, label="coupling.sample_pairs", override_kind="effective"))
                    if hasattr(coupling, "build_pairwise_cost"):
                        effective_objects.append(_object_summary(getattr(coupling, "build_pairwise_cost"), label="coupling.build_pairwise_cost", override_kind="effective"))
                    if hasattr(coupling, "compute_ot_coupling"):
                        effective_objects.append(_object_summary(getattr(coupling, "compute_ot_coupling"), label="coupling.compute_ot_coupling", override_kind="effective"))
                if path is not None:
                    effective_objects.append(_object_summary(path.__class__, label="path_class", override_kind="effective"))
                    if hasattr(path, "compute_conditional_mass"):
                        effective_objects.append(_object_summary(getattr(path, "compute_conditional_mass"), label="path.compute_conditional_mass", override_kind="effective"))
                    if hasattr(path, "compute_conditional_flow"):
                        effective_objects.append(_object_summary(getattr(path, "compute_conditional_flow"), label="path.compute_conditional_flow", override_kind="effective"))
                if mass is not None:
                    effective_objects.append(_object_summary(mass.__class__, label="mass_class", override_kind="effective"))
                    if hasattr(mass, "compute"):
                        effective_objects.append(_object_summary(getattr(mass, "compute"), label="mass.compute", override_kind="effective"))
            if isinstance(preflight_cache, dict) and preflight_cache.get("model") is not None:
                model = preflight_cache.get("model")
                effective_objects.append(_object_summary(model.__class__, label="model_class", override_kind="effective"))
                if hasattr(model, "forward"):
                    effective_objects.append(_object_summary(model.forward, label="model.forward", override_kind="effective"))
            if isinstance(preflight_cache, dict) and preflight_cache.get("trainer") is not None:
                trainer = preflight_cache["trainer"]
                effective_objects.append(_object_summary(trainer.__class__, label="trainer_class", override_kind="effective"))
                effective_objects.append(_object_summary(trainer.run_custom_stage, label="trainer.run_custom_stage", override_kind="effective"))
        elif selected_stage_mode == "neural_ode":
            if isinstance(preflight_cache, dict) and preflight_cache.get("trainer") is not None:
                trainer = preflight_cache["trainer"]
                model = preflight_cache.get("model")
                if model is not None:
                    effective_objects.append(_object_summary(model.__class__, label="model_class", override_kind="effective"))
                    effective_objects.append(_object_summary(model.forward, label="model.forward", override_kind="effective"))
                effective_objects.append(_object_summary(trainer.__class__, label="trainer_class", override_kind="effective"))
                if preflight_cache.get("custom_stage_runner"):
                    effective_objects.append(_object_summary(trainer.run_custom_stage, label="trainer.run_custom_stage", override_kind="effective"))
                else:
                    effective_objects.append(_object_summary(trainer._setup_stage, label="trainer._setup_stage", override_kind="effective"))
                    effective_objects.append(_object_summary(trainer.run_neural_ode_stage, label="trainer.run_neural_ode_stage", override_kind="effective"))
                    effective_objects.append(_object_summary(trainer.train_neural_ode_epoch, label="trainer.train_neural_ode_epoch", override_kind="effective"))
                    effective_objects.append(_object_summary(neural_ode_step, label="neural_ode_step", override_kind="package"))
                    effective_objects.append(_object_summary(trainer.ode_func.forward, label="ODEFunc.forward", override_kind="effective"))
        if training_target.spec.flow_matching_loss_hook is not None:
            effective_objects.append(
                _object_summary(
                    training_target.spec.flow_matching_loss_hook,
                    label="flow_matching_loss_hook",
                    override_kind="custom",
                )
            )
        if training_target.spec.evaluation_metrics_hook is not None:
            effective_objects.append(
                _object_summary(
                    training_target.spec.evaluation_metrics_hook,
                    label="evaluation_metrics_hook",
                    override_kind="custom",
                )
            )
        effective_objects.append(_object_summary(cb_fit, label="package.fit", override_kind="package"))
        effective_objects.append(_object_summary(TrainingPipeline.train, label="TrainingPipeline.train", override_kind="package"))
        effective_objects.append(_object_summary(TrainingPipeline.evaluate, label="TrainingPipeline.evaluate", override_kind="package"))
        payload["effective_objects"] = effective_objects

        if training_target.training_mode == "custom":
            obsolete_entries: List[Dict[str, Any]] = []
            obsolete_path = self.planner_file_tools._registry_obsolete_path(training_target.spec.algorithm_id)
            if obsolete_path.exists():
                for line in obsolete_path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except Exception:
                        continue
                    if isinstance(item, dict):
                        obsolete_entries.append(item)
            registry = self.planner_file_tools._bootstrap_algorithm_registry(training_target.spec.algorithm_id)
            payload["experiment_registry"] = {
                "registry_path": str(self.planner_file_tools._registry_index_path(training_target.spec.algorithm_id)),
                "active_proposal_id": str(registry.get("active_proposal_id") or ""),
                "active_snapshot_id": str(registry.get("active_workspace_snapshot_id") or ""),
                "baseline_run_id": str(registry.get("active_baseline_run_id") or ""),
                "obsolete_warnings": obsolete_entries,
            }

        smoke_payload = dict(payload.get("one_epoch_smoke_test") or {})
        smoke_bundle = None
        if not run_smoke_test:
            smoke_payload["skipped"] = "disabled by tool argument"
            _preview_progress("Preview: smoke test skipped", 0.90)
        elif payload.get("error"):
            smoke_payload["skipped"] = "skipped because preview setup failed"
            _preview_progress("Preview: smoke test skipped after setup error", 0.90)
        elif payload.get("preflight_error"):
            smoke_payload["skipped"] = "skipped because dry-run preflight failed"
            _preview_progress("Preview: smoke test skipped after preflight error", 0.90)
        else:
            try:
                _preview_progress("Preview: starting one-epoch smoke training", 0.58)
                tracer.mark("one_epoch_smoke_execute_training_target")
                tracer.record_call(execute_training_target, label="execute_training_target(one_epoch_smoke)")
                smoke_root = (
                    Path(self.state.get("output_dir") or "cytobridge_output").expanduser().resolve()
                    / ".runtime"
                    / "preview_smoke"
                )
                smoke_root.mkdir(parents=True, exist_ok=True)
                safe_algo = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(training_target.spec.algorithm_id or "preview"))
                smoke_dir = smoke_root / f"{safe_algo}_{stage}_{datetime.now().strftime('%Y%m%dT%H%M%S')}"
                smoke_dir.mkdir(parents=True, exist_ok=False)
                smoke_bundle = TrainingRunBundle(
                    run_id=smoke_dir.name,
                    run_dir=smoke_dir,
                    logs_dir=smoke_dir / "logs",
                    artifacts_dir=smoke_dir / "artifacts",
                    checkpoints_dir=smoke_dir / "checkpoints",
                    algorithm_snapshot_dir=smoke_dir / "algorithm_snapshot",
                    run_manifest_path=smoke_dir / "run_manifest.json",
                    resolved_config_path=smoke_dir / "resolved_config.yaml",
                    training_log_path=smoke_dir / "logs" / "training.log",
                    planner_context_path=smoke_dir / "logs" / "planner_context.json",
                    trained_model_path=smoke_dir / "artifacts" / "trained_model.h5ad",
                    model_artifact_path=smoke_dir / "artifacts" / "model_artifact.json",
                    model_state_path=smoke_dir / "artifacts" / "model_state.pt",
                    metrics_path=smoke_dir / "artifacts" / "metrics.json",
                )
                smoke_bundle.logs_dir.mkdir(parents=True, exist_ok=True)
                smoke_bundle.artifacts_dir.mkdir(parents=True, exist_ok=True)
                smoke_bundle.checkpoints_dir.mkdir(parents=True, exist_ok=True)
                smoke_bundle.algorithm_snapshot_dir.mkdir(parents=True, exist_ok=True)
                smoke_bundle.training_log_path.write_text("", encoding="utf-8")
                if should_isolate_training_target(training_target.training_mode):
                    _preview_progress("Preview: running isolated one-epoch smoke training", 0.68)
                    smoke_metrics, smoke_error, smoke_isolation = run_training_in_subprocess(
                        adata_path=str(data_path),
                        stage=stage,
                        device=requested_device,
                        bundle=smoke_bundle,
                        candidate_name=chosen_candidate.name if chosen_candidate else "",
                        training_algorithm_id=str(training_algorithm_id or ""),
                        output_dir=str(self.state.get("output_dir") or "cytobridge_output"),
                        workspace_root=str(workspace_root),
                        config_overrides=normalized_overrides,
                        claim_metric_spec={},
                        max_epochs=1,
                        purpose="preview",
                    )
                    smoke_payload["training_isolation"] = smoke_isolation
                else:
                    def smoke_progress_callback(message: str, progress: float) -> None:
                        try:
                            local_progress = max(0.0, min(1.0, float(progress)))
                        except Exception:
                            local_progress = 0.0
                        _preview_progress(
                            f"Preview smoke: {message}",
                            0.62 + (0.28 * local_progress),
                        )

                    smoke_adata, smoke_metrics, smoke_error = execute_training_target(
                        adata=adata,
                        target=training_target,
                        stage=stage,
                        device=requested_device,
                        max_epochs=1,
                        outdir=smoke_dir,
                        seed=42,
                        progress_callback=smoke_progress_callback,
                    )
                    del smoke_adata
                _preview_progress("Preview: checking smoke metrics", 0.92)
                smoke_metrics = dict(smoke_metrics or {})
                metric_sanity = self._preview_metric_sanity(
                    smoke_metrics,
                    metric_params=dict(training_target.spec.evaluation_metrics_params or {}),
                )
                selected_metric_keys = (
                    "w1_scores",
                    "tmv_scores",
                    "custom_metrics",
                    "custom_metrics_warning",
                    "evaluation_warning",
                    "runtime_sec",
                    "training_timed_out",
                    "training_timeout_message",
                    "inference_timed_out",
                    "inference_timeout_message",
                    "last_completed_epoch",
                    "completed_epochs",
                    "final_train_loss",
                    "best_train_loss",
                    "last_epoch_train_loss",
                    "last_training_stage_name",
                    "last_training_stage_mode",
                    "requested_device",
                    "resolved_device",
                )
                metrics_preview = {
                    key: smoke_metrics.get(key)
                    for key in selected_metric_keys
                    if key in smoke_metrics
                }
                if isinstance(smoke_metrics.get("training_time_budget"), dict):
                    metrics_preview["training_time_budget"] = smoke_metrics.get("training_time_budget")
                if isinstance(smoke_metrics.get("inference_time_budget"), dict):
                    metrics_preview["inference_time_budget"] = smoke_metrics.get("inference_time_budget")
                smoke_payload.update(
                    {
                        "ok": not bool(smoke_error) and bool(metric_sanity.get("ok")),
                        "error": smoke_error or "",
                        "outdir": str(smoke_dir),
                        "runtime_sec": smoke_metrics.get("runtime_sec"),
                        "metrics_preview": metrics_preview,
                        "metric_sanity": metric_sanity,
                        "max_epochs": 1,
                    }
                )
            except Exception as exc:
                smoke_payload.update(
                    {
                        "ok": False,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
                _preview_progress("Preview smoke failed", 0.95)
        cleanup = _cleanup_preview_smoke_h5ad(smoke_bundle)
        if cleanup:
            smoke_payload["storage_cleanup"] = cleanup
        payload["one_epoch_smoke_test"] = smoke_payload
        payload["observed_trace"] = tracer.events
        payload["trace_truncated"] = tracer.truncated

        payload["ok"] = (
            not bool(payload.get("error"))
            and not bool(payload.get("preflight_error"))
            and (not run_smoke_test or bool((payload.get("one_epoch_smoke_test") or {}).get("ok")))
        )
        payload["preview_artifacts"] = self._allocate_training_preview_artifact_paths(payload)
        if training_target.training_mode == "custom":
            preview_registry_entry = self.planner_file_tools.record_preview_artifacts(
                training_target.spec.algorithm_id,
                preview_artifacts=payload["preview_artifacts"],
                stage=stage,
                data_path=data_path,
                requested_device=requested_device,
                ok=payload["ok"],
            )
            payload.setdefault("experiment_registry", {})
            payload["experiment_registry"]["latest_preview"] = preview_registry_entry
        report = self._render_training_chain_report(payload)
        self._write_training_preview_artifacts(payload=payload, report_markdown=report)
        self.state["last_training_preview"] = payload
        self.state["last_training_preview_markdown_path"] = payload["preview_artifacts"].get("markdown_path", "")
        self.state["last_training_preview_json_path"] = payload["preview_artifacts"].get("json_path", "")
        self._emit_event("training_run_previewed", payload)
        self._emit_event("training_chain_inspected", payload)
        if payload["ok"]:
            _preview_progress("Preview completed", 1.0)
        else:
            _preview_progress("Preview completed with warnings or errors", 1.0)
        return f"{switch_note}\n{report}" if switch_note else report

    def restore_from_snapshot(self, agent_snapshots: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Legacy multi-agent restore is no longer supported."""
        del agent_snapshots
        return {"restored": {}, "degraded": False, "reasons": []}

    def execute_python(
        self,
        code: str,
        reason: str = "",
        timeout: int = 300,
        timeout_mode: str = "isolated",
    ) -> str:
        """
        Execute Python code for quick data exploration or validation.
        Use this for simple checks like verifying sub-agent outputs, 
        exploring data characteristics, or generating quick plots.
        The adata has already been loaded for you, DO NOT reload the adata yourself.
        The CytoBridge package is preloaded as `cb` when available.
        By default this runs in an isolated subprocess with strict timeout enforcement.
        AnnData changes are synchronized back when possible, but temporary Python
        variables are not persisted. For reusable code, save a .py file under
        output_dir/scripts and run it with run_saved_python_script(...).
        
        Args:
            code: Python code to execute. Available variables: adata, sc, ad, sp, np, pd, plt, os, Path, input_path, output_dir, figures_dir, scripts_dir, save_pubfig, load_result_table, pick_top_figures, build_appendix_gallery, make_figure_name, and cb (CytoBridge package, when import succeeds).
            reason: Brief explanation of why you're running this code.
            timeout: Max execution time in seconds. Defaults to 300.
            timeout_mode: Timeout strategy. Use `isolated` for strict subprocess timeout enforcement
                (default), `shared` only for small stateful interactive checks, or `auto` to choose based on runtime.
        
        Returns:
            Output from the code execution (stdout, result, or error).
        """
        switch_note = self._apply_pending_input_path()
        data_path = self._resolve_active_python_data_path()

        exec_id = f"pythonexec_{datetime.now().strftime('%Y%m%d%H%M%S')}_{Path(str(data_path or 'runtime')).stem}_{datetime.now().microsecond}"
        timeout = max(1, int(timeout))
        timeout_mode = str(timeout_mode or "isolated").strip().lower()
        if timeout_mode not in {"shared", "isolated", "auto"}:
            timeout_mode = "isolated"

        try:
            output_dir, scope_reused = self._ensure_active_code_executor(data_path)

            self._emit_python_execution_event(
                "started",
                {
                    "id": exec_id,
                    "toolName": "execute_python",
                    "reason": reason or "",
                    "timeout": timeout,
                    "timeoutMode": timeout_mode,
                    "status": "inProgress",
                    "code": code,
                    "dataPath": str(data_path) if data_path else "",
                    "scopeReused": scope_reused,
                    "outputDir": str(output_dir),
                },
            )

            def _stream_python_output(stream_name: str, chunk: str) -> None:
                if not chunk:
                    return
                self._emit_python_execution_event(
                    "outputDelta",
                    {
                        "id": exec_id,
                        "stream": stream_name,
                        "delta": chunk,
                    },
                )

            result_dict = self._executor.execute(
                code,
                timeout=timeout,
                timeout_mode=timeout_mode,
                stream_callback=_stream_python_output,
                stop_check=self.stop_check,
            )
            result = result_dict.get("output", "No output")
            if result_dict.get("converted_path"):
                self.state["converted_path"] = result_dict["converted_path"]
            if switch_note:
                result = f"{switch_note}\n{result}"

            completion_status = "completed"
            if result_dict.get("interrupted"):
                completion_status = "interrupted"
            elif result_dict.get("timed_out"):
                completion_status = "timedOut"
            elif not result_dict.get("success"):
                completion_status = "failed"

            self._emit_python_execution_event(
                "completed",
                {
                    "id": exec_id,
                    "toolName": "execute_python",
                    "reason": reason or "",
                    "timeout": timeout,
                    "timeoutMode": result_dict.get("effective_mode", timeout_mode),
                    "status": completion_status,
                    "stdout": result_dict.get("stdout", ""),
                    "stderr": result_dict.get("stderr", ""),
                    "error": result_dict.get("error", ""),
                    "traceback": result_dict.get("traceback", ""),
                    "durationMs": int(result_dict.get("duration_ms", 0) or 0),
                    "adataChanged": bool(result_dict.get("adata_changed")),
                    "scopeReused": scope_reused,
                    "dataPath": str(data_path) if data_path else "",
                    "timeoutEnforced": bool(result_dict.get("timeout_enforced")),
                    "switchNote": switch_note or "",
                },
            )
            return result
        except BaseException as e:
            error_msg = f"Code execution error: {e}"
            self._emit_python_execution_event(
                "completed",
                {
                    "id": exec_id,
                    "toolName": "execute_python",
                    "reason": reason or "",
                    "timeout": timeout,
                    "timeoutMode": timeout_mode,
                    "status": "failed",
                    "stdout": "",
                    "stderr": "",
                    "error": error_msg,
                    "traceback": traceback.format_exc(),
                    "durationMs": 0,
                    "adataChanged": False,
                    "scopeReused": scope_reused if 'scope_reused' in locals() else False,
                    "dataPath": str(data_path) if data_path else "",
                    "timeoutEnforced": False,
                    "switchNote": switch_note or "",
                },
            )
            return error_msg

    def run_saved_python_script(
        self,
        script_path: str,
        reason: str = "",
        timeout: int = 300,
        timeout_mode: str = "isolated",
    ) -> str:
        """
        Execute a saved Python script under <output_dir>/scripts in true script mode.

        Relative paths are resolved against <output_dir>/scripts.
        """
        output_dir = Path(self.state.get("output_dir") or "cytobridge_output").expanduser().resolve()
        scripts_dir = (output_dir / "scripts").resolve()
        scripts_dir.mkdir(parents=True, exist_ok=True)

        raw_input = str(script_path or "").strip()
        if not raw_input:
            return "Error: script_path is required."
        raw = Path(raw_input).expanduser()
        target = raw.resolve() if raw.is_absolute() else (scripts_dir / raw).resolve()

        try:
            target.relative_to(scripts_dir)
        except ValueError:
            return f"Error: saved scripts must live under {scripts_dir}"

        policy = self.planner_file_tools._policy()
        target = policy.validate_read_path(str(target))
        if not target.exists():
            return f"Error: script file does not exist: {target}"
        if target.is_dir():
            return f"Error: script path is a directory, not a Python file: {target}"
        if target.suffix.lower() != ".py":
            return f"Error: saved script must be a .py file: {target}"

        run_reason = str(reason or "").strip() or f"Execute saved script {target.name}"
        switch_note = self._apply_pending_input_path()
        data_path = self._resolve_active_python_data_path()
        exec_id = f"pyscript_{datetime.now().strftime('%Y%m%d%H%M%S')}_{target.stem}_{datetime.now().microsecond}"
        timeout = max(1, int(timeout))
        timeout_mode = str(timeout_mode or "isolated").strip().lower()
        if timeout_mode not in {"shared", "isolated", "auto"}:
            timeout_mode = "isolated"

        try:
            output_dir, scope_reused = self._ensure_active_code_executor(data_path)
            self._emit_python_execution_event(
                "started",
                {
                    "id": exec_id,
                    "toolName": "run_saved_python_script",
                    "reason": run_reason,
                    "timeout": timeout,
                    "timeoutMode": timeout_mode,
                    "status": "inProgress",
                    "scriptPath": str(target),
                    "dataPath": str(data_path) if data_path else "",
                    "scopeReused": scope_reused,
                    "outputDir": str(output_dir),
                },
            )

            def _stream_python_output(stream_name: str, chunk: str) -> None:
                if not chunk:
                    return
                self._emit_python_execution_event(
                    "outputDelta",
                    {
                        "id": exec_id,
                        "stream": stream_name,
                        "delta": chunk,
                    },
                )

            result_dict = self._executor.execute_script_file(
                str(target),
                timeout=timeout,
                timeout_mode=timeout_mode,
                stream_callback=_stream_python_output,
                stop_check=self.stop_check,
            )
            result = result_dict.get("output", "No output")
            if result_dict.get("converted_path"):
                self.state["converted_path"] = result_dict["converted_path"]
            if switch_note:
                result = f"{switch_note}\n{result}"

            completion_status = "completed"
            if result_dict.get("interrupted"):
                completion_status = "interrupted"
            elif result_dict.get("timed_out"):
                completion_status = "timedOut"
            elif not result_dict.get("success"):
                completion_status = "failed"

            self._emit_python_execution_event(
                "completed",
                {
                    "id": exec_id,
                    "toolName": "run_saved_python_script",
                    "reason": run_reason,
                    "timeout": timeout,
                    "timeoutMode": result_dict.get("effective_mode", timeout_mode),
                    "status": completion_status,
                    "stdout": result_dict.get("stdout", ""),
                    "stderr": result_dict.get("stderr", ""),
                    "error": result_dict.get("error", ""),
                    "traceback": result_dict.get("traceback", ""),
                    "durationMs": int(result_dict.get("duration_ms", 0) or 0),
                    "adataChanged": bool(result_dict.get("adata_changed")),
                    "scopeReused": scope_reused,
                    "dataPath": str(data_path) if data_path else "",
                    "timeoutEnforced": bool(result_dict.get("timeout_enforced")),
                    "switchNote": switch_note or "",
                    "scriptPath": str(target),
                },
            )
            return result
        except BaseException as e:
            error_msg = f"Saved script execution error: {e}"
            self._emit_python_execution_event(
                "completed",
                {
                    "id": exec_id,
                    "toolName": "run_saved_python_script",
                    "reason": run_reason,
                    "timeout": timeout,
                    "timeoutMode": timeout_mode,
                    "status": "failed",
                    "stdout": "",
                    "stderr": "",
                    "error": error_msg,
                    "traceback": traceback.format_exc(),
                    "durationMs": 0,
                    "adataChanged": False,
                    "scopeReused": scope_reused if 'scope_reused' in locals() else False,
                    "dataPath": str(data_path) if data_path else "",
                    "timeoutEnforced": False,
                    "switchNote": switch_note or "",
                    "scriptPath": str(target),
                },
            )
            return error_msg

    @staticmethod
    def _holdout_time_eval_enabled(spec: Optional[Dict[str, Any]]) -> bool:
        if not isinstance(spec, dict):
            return False
        raw = spec.get("enabled", False)
        if isinstance(raw, str):
            return raw.strip().lower() in {"1", "true", "yes", "on"}
        return bool(raw)

    @staticmethod
    def _mean_finite(values: List[Any]) -> Optional[float]:
        finite: List[float] = []
        for value in values:
            try:
                number = float(value)
            except Exception:
                continue
            if math.isfinite(number):
                finite.append(number)
        if not finite:
            return None
        return float(sum(finite) / len(finite))

    @staticmethod
    def _safe_time_slug(values: List[float]) -> str:
        parts = []
        for value in values:
            text = ("%g" % float(value)).replace("-", "m").replace(".", "p")
            parts.append(text)
        return "_".join(parts) or "none"

    def _normalize_holdout_time_eval_spec(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        mode = str(spec.get("mode") or "single").strip().lower()
        if mode not in {"single", "sequential", "simultaneous"}:
            mode = "single"
        raw_times = (
            spec.get("time_points")
            if "time_points" in spec
            else spec.get("holdout_time_points", spec.get("holdout_times", []))
        )
        if raw_times is None:
            raw_times = []
        if not isinstance(raw_times, list):
            raw_times = [raw_times]
        time_points: List[float] = []
        for value in raw_times:
            try:
                time_points.append(float(value))
            except Exception:
                continue
        deduped: List[float] = []
        for value in time_points:
            if not any(abs(value - existing) <= 1e-8 for existing in deduped):
                deduped.append(value)
        if mode == "sequential":
            groups = [[value] for value in deduped]
        elif mode == "simultaneous":
            groups = [deduped] if deduped else []
        else:
            groups = [[deduped[0]]] if deduped else []
        return {
            "enabled": self._holdout_time_eval_enabled(spec),
            "mode": mode,
            "time_points": deduped,
            "groups": groups,
            "time_key": str(spec.get("time_key") or "").strip(),
            "latent_key": str(spec.get("latent_key") or "X_latent").strip() or "X_latent",
            "allow_initial_time": bool(spec.get("allow_initial_time", False)),
            "attach_to_custom_metrics": bool(
                spec.get("attach_to_custom_metrics", spec.get("use_as_claim_metric", False))
            ),
            "metric_name": str(spec.get("metric_name") or "holdout_time_w1").strip() or "holdout_time_w1",
            "max_trajectory_time_delta": spec.get("max_trajectory_time_delta"),
            "claim_metric_evaluator": (
                dict(spec.get("claim_metric_evaluator"))
                if isinstance(spec.get("claim_metric_evaluator"), dict)
                else {}
            ),
        }

    @staticmethod
    def _is_holdout_time_w1_claim_metric_spec(spec: Any) -> bool:
        if not isinstance(spec, dict):
            return False
        raw = str(
            spec.get("evaluator_path")
            or spec.get("claim_metric_evaluator_path")
            or spec.get("metric_evaluator_path")
            or spec.get("campaign_metric_evaluator_path")
            or ""
        ).strip().lower()
        return raw in {
            "builtin:holdout_time_w1",
            "cytobridge:holdout_time_w1",
            "holdout_time_w1",
        }

    def _holdout_time_eval_spec_from_claim_metric(
        self,
        claim_metric_spec: Any,
        *,
        baseline_algorithm: str = "",
    ) -> Dict[str, Any]:
        if not self._is_holdout_time_w1_claim_metric_spec(claim_metric_spec):
            return {}
        spec = dict(claim_metric_spec or {})
        holdout = dict(spec.get("holdout_time_evaluation") or {})
        for key in (
            "mode",
            "time_points",
            "holdout_time_points",
            "holdout_times",
            "time_key",
            "latent_key",
            "allow_initial_time",
            "max_trajectory_time_delta",
        ):
            if key in spec and key not in holdout:
                holdout[key] = deepcopy(spec.get(key))
        holdout["enabled"] = True
        holdout["attach_to_custom_metrics"] = True
        holdout["metric_name"] = str(spec.get("name") or spec.get("metric_name") or "holdout_time_w1").strip()
        try:
            _, info = load_campaign_claim_metric_evaluator(
                spec,
                baseline_algorithm=str(baseline_algorithm or ""),
            )
            if info:
                holdout["claim_metric_evaluator"] = dict(info)
        except Exception:
            pass
        return holdout

    def _merge_holdout_eval_specs(self, explicit: Any, formal: Dict[str, Any]) -> Dict[str, Any]:
        explicit_spec = dict(explicit or {}) if isinstance(explicit, dict) else {}
        formal_spec = dict(formal or {}) if isinstance(formal, dict) else {}
        if not formal_spec:
            return explicit_spec
        merged = deepcopy(formal_spec)
        for key, value in explicit_spec.items():
            if key in {"attach_to_custom_metrics", "use_as_claim_metric", "metric_name", "claim_metric_evaluator"}:
                continue
            merged[key] = deepcopy(value)
        merged["enabled"] = True
        merged["attach_to_custom_metrics"] = True
        return merged

    @staticmethod
    def _detect_time_key(adata: Any, requested: str = "") -> Optional[str]:
        candidates = [requested] if requested else []
        candidates.extend(["time_point_processed", "time_point", "time", "t"])
        seen: Set[str] = set()
        for key in candidates:
            key = str(key or "").strip()
            if not key or key in seen:
                continue
            seen.add(key)
            try:
                if key in adata.obs:
                    return key
            except Exception:
                continue
        return None

    @staticmethod
    def _observed_matrix_for_holdout(adata: Any, latent_key: str) -> Any:
        if latent_key and hasattr(adata, "obsm") and latent_key in adata.obsm:
            return adata.obsm[latent_key]
        if hasattr(adata, "X"):
            return adata.X
        raise ValueError(f"AnnData has neither obsm[{latent_key!r}] nor X for hold-out W1 evaluation.")

    def _compute_holdout_w1_from_trajectory(
        self,
        *,
        trajectory_path: str,
        full_adata_path: str,
        heldout_times: List[float],
        time_key: str,
        latent_key: str,
        max_trajectory_time_delta: Any = None,
    ) -> Dict[str, Any]:
        import numpy as np
        import anndata as ad
        from CytoBridge.tl.w1_backend import (
            compute_w1_distance,
            select_w1_backend_policy,
            w1_backend_metric_metadata,
        )

        trajectory = load_evaluation_trajectory_artifact(trajectory_path)
        full_adata = ad.read_h5ad(full_adata_path)
        if time_key not in full_adata.obs:
            raise ValueError(f"hold-out time key {time_key!r} is not present in full AnnData obs.")
        raw_time_values = np.asarray(full_adata.obs[time_key], dtype=float)
        matrix = self._observed_matrix_for_holdout(full_adata, latent_key)
        if not isinstance(matrix, np.ndarray):
            matrix = matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)
        trajectory_times = [float(t) for t in list(trajectory.time_points or [])]
        step = trajectory.step
        try:
            allowed_delta = float(max_trajectory_time_delta)
        except Exception:
            allowed_delta = max(1e-6, abs(float(step or 0.0)) * 0.51) if step is not None else 1e-6
        predicted_initial_mass = float(np.asarray(trajectory.weights_by_time[0]).reshape(-1).sum())
        initial_count = int(np.isclose(raw_time_values, float(min(raw_time_values))).sum())
        results: List[Dict[str, Any]] = []
        prepared_pairs: List[Dict[str, Any]] = []
        for heldout_time in heldout_times:
            target = float(heldout_time)
            mask = np.isclose(raw_time_values, target)
            observed = np.asarray(matrix[mask], dtype=float)
            if observed.shape[0] == 0:
                results.append(
                    {
                        "time_point": target,
                        "status": "skipped",
                        "reason": "no_observed_cells_at_time",
                    }
                )
                continue
            if not trajectory_times:
                results.append(
                    {
                        "time_point": target,
                        "status": "failed",
                        "reason": "empty_evaluation_trajectory",
                    }
                )
                continue
            nearest_idx = min(range(len(trajectory_times)), key=lambda idx: abs(trajectory_times[idx] - target))
            nearest_time = float(trajectory_times[nearest_idx])
            delta = abs(nearest_time - target)
            if delta > allowed_delta:
                results.append(
                    {
                        "time_point": target,
                        "status": "unsupported",
                        "reason": "heldout_time_not_covered_by_auxiliary_rollout",
                        "nearest_trajectory_time": nearest_time,
                        "time_delta": float(delta),
                        "max_trajectory_time_delta": float(allowed_delta),
                    }
                )
                continue
            predicted = np.asarray(trajectory.points_by_time[nearest_idx], dtype=float)
            raw_weights = np.asarray(trajectory.weights_by_time[nearest_idx], dtype=float).reshape(-1)
            if predicted.shape[0] == 0 or raw_weights.shape[0] != predicted.shape[0] or raw_weights.sum() <= 0:
                results.append(
                    {
                        "time_point": target,
                        "status": "failed",
                        "reason": "invalid_predicted_distribution",
                    }
                )
                continue
            pred_weights = raw_weights / raw_weights.sum()
            observed_weights = np.ones(observed.shape[0], dtype=float) / float(observed.shape[0])
            prepared_pairs.append(
                {
                    "target": target,
                    "observed": observed,
                    "predicted": predicted,
                    "observed_weights": observed_weights,
                    "pred_weights": pred_weights,
                    "raw_weights": raw_weights,
                    "nearest_time": nearest_time,
                    "delta": delta,
                }
            )

        w1_backend_policy = select_w1_backend_policy(
            [(item["observed"].shape[0], item["predicted"].shape[0]) for item in prepared_pairs],
            selection_scope="holdout_time_panel",
        )
        for item in prepared_pairs:
            target = float(item["target"])
            observed = item["observed"]
            predicted = item["predicted"]
            raw_weights = item["raw_weights"]
            w1 = compute_w1_distance(
                observed,
                predicted,
                observed_weights=item["observed_weights"],
                predicted_weights=item["pred_weights"],
                backend_policy=w1_backend_policy,
            )
            tmv = None
            if predicted_initial_mass > 0 and initial_count > 0:
                predicted_relative_mass = float(raw_weights.sum()) / predicted_initial_mass
                observed_relative_mass = float(observed.shape[0]) / float(initial_count)
                tmv = float(abs(predicted_relative_mass - observed_relative_mass))
            results.append(
                {
                    "time_point": target,
                    "status": "ok",
                    "w1": w1,
                    "tmv": tmv,
                    "observed_cells": int(observed.shape[0]),
                    "predicted_particles": int(predicted.shape[0]),
                    "trajectory_time": float(item["nearest_time"]),
                    "time_delta": float(item["delta"]),
                    "w1_backend": w1_backend_policy.get("w1_backend"),
                }
            )
        payload = {
            "status": "ok" if all(item.get("status") == "ok" for item in results) else "partial",
            "timepoint_results": results,
            "mean_w1": self._mean_finite([item.get("w1") for item in results]),
            "mean_tmv": self._mean_finite([item.get("tmv") for item in results]),
            "trajectory_path": str(trajectory_path),
        }
        payload.update(w1_backend_metric_metadata(w1_backend_policy))
        return payload

    def _run_holdout_time_evaluation(
        self,
        spec: Dict[str, Any],
        *,
        parent_run_id: str,
        candidate_name: Optional[str],
        training_algorithm_id: Optional[str],
        stage: str,
        config_overrides: Dict[str, Any],
        run_label_prefix: str,
        adata_path: str,
        device: str,
        seed: Optional[int],
        decision_reason: str,
        campaign_id: str = "",
        trial_id: str = "",
        dataset_id: str = "",
    ) -> Dict[str, Any]:
        import numpy as np
        import anndata as ad

        normalized = self._normalize_holdout_time_eval_spec(spec)
        output_dir = Path(self.state.get("output_dir") or "cytobridge_output").expanduser().resolve()
        parent_run_dir = output_dir / "training_runs" / str(parent_run_id)
        report_dir = parent_run_dir / "holdout_time_evaluation"
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / "holdout_time_evaluation.json"
        report: Dict[str, Any] = {
            "status": "skipped",
            "reason": "",
            "schema_version": "1",
            "mode": normalized["mode"],
            "time_points": normalized["time_points"],
            "groups": normalized["groups"],
            "parent_run_id": str(parent_run_id),
            "campaign_id": str(campaign_id or ""),
            "trial_id": str(trial_id or ""),
            "dataset_id": str(dataset_id or ""),
            "report_path": str(report_path),
            "note": (
                "Auxiliary hold-out time evaluation is default-off. The normal full-data run is kept as "
                "the campaign/manual training run; these auxiliary runs are diagnostics unless explicitly "
                "attached to custom_metrics."
            ),
        }
        if not normalized["enabled"]:
            return report
        if not normalized["groups"]:
            report.update({"status": "failed", "reason": "no_holdout_time_points"})
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            return report
        full_path = str(adata_path or self.state.get("preprocessed_path") or self.state.get("input_path") or "").strip()
        if not full_path or not Path(full_path).expanduser().is_file():
            report.update({"status": "failed", "reason": f"full AnnData path missing: {full_path}"})
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            return report
        latest_snapshot = {
            "latest_training_run_id": self.state.get("latest_training_run_id"),
            "latest_training_run_dir": self.state.get("latest_training_run_dir"),
            "latest_training_algorithm_id": self.state.get("latest_training_algorithm_id"),
            "final_config": deepcopy(self.state.get("final_config")),
            "final_metrics": deepcopy(self.state.get("final_metrics")),
        }
        group_reports: List[Dict[str, Any]] = []
        try:
            full_adata = ad.read_h5ad(full_path)
            time_key = self._detect_time_key(full_adata, normalized["time_key"])
            if not time_key:
                raise ValueError("Could not detect a numeric time key in obs; set holdout_time_evaluation.time_key.")
            time_values = np.asarray(full_adata.obs[time_key], dtype=float)
            available_times = sorted({float(round(float(v), 10)) for v in time_values})
            initial_time = float(available_times[0]) if available_times else 0.0
            report["time_key"] = time_key
            report["latent_key"] = normalized["latent_key"]
            report["available_time_points"] = available_times
            for group_index, group in enumerate(normalized["groups"], start=1):
                heldout = [float(x) for x in group]
                group_slug = self._safe_time_slug(heldout)
                group_dir = report_dir / f"group{group_index}_{group_slug}"
                group_dir.mkdir(parents=True, exist_ok=True)
                group_report: Dict[str, Any] = {
                    "group_index": group_index,
                    "heldout_time_points": heldout,
                    "status": "pending",
                    "group_dir": str(group_dir),
                }
                if (
                    not normalized["allow_initial_time"]
                    and any(abs(float(t) - initial_time) <= 1e-8 for t in heldout)
                ):
                    group_report.update(
                        {
                            "status": "skipped",
                            "reason": "initial_time_holdout_requires_allow_initial_time",
                        }
                    )
                    group_reports.append(group_report)
                    continue
                holdout_mask = np.zeros(time_values.shape[0], dtype=bool)
                missing_times: List[float] = []
                for heldout_time in heldout:
                    mask = np.isclose(time_values, heldout_time)
                    if not bool(mask.any()):
                        missing_times.append(float(heldout_time))
                    holdout_mask |= mask
                if missing_times:
                    group_report.update({"status": "skipped", "reason": "missing_time_points", "missing": missing_times})
                    group_reports.append(group_report)
                    continue
                train_mask = ~holdout_mask
                train_times = sorted({float(round(float(v), 10)) for v in time_values[train_mask]})
                if len(train_times) < 2:
                    group_report.update({"status": "skipped", "reason": "train_split_has_fewer_than_two_timepoints"})
                    group_reports.append(group_report)
                    continue
                train_path = group_dir / "train_without_holdout.h5ad"
                write_adata_h5ad_safe(full_adata[train_mask].copy(), train_path)
                before_nested_run_id = str(self.state.get("latest_training_run_id") or "")
                nested_result = self._run_training_tool_impl(
                    candidate_name=candidate_name,
                    training_algorithm_id=training_algorithm_id,
                    stage=stage,
                    config_overrides=deepcopy(config_overrides or {}),
                    run_label=f"{str(run_label_prefix or parent_run_id)[:28]}-holdout-{group_slug}",
                    adata_path=str(train_path),
                    device=device or "",
                    seed=seed,
                    decision="provisional",
                    decision_reason=decision_reason,
                    review_purpose="holdout_time_evaluation",
                    review_campaign=None,
                    _skip_review_gates=True,
                )
                nested_run_id = str(self.state.get("latest_training_run_id") or "")
                if not nested_run_id or nested_run_id == before_nested_run_id:
                    group_report.update(
                        {
                            "status": "failed",
                            "reason": "auxiliary_training_did_not_create_run",
                            "training_result": str(nested_result)[:2000],
                        }
                    )
                    group_reports.append(group_report)
                    continue
                nested_metrics = self._campaign_latest_training_metrics(nested_run_id)
                trajectory_path = str(
                    nested_metrics.get("evaluation_trajectory_path")
                    or nested_metrics.get("trajectory_path")
                    or ((nested_metrics.get("artifacts") or {}) if isinstance(nested_metrics.get("artifacts"), dict) else {}).get("evaluation_trajectory_path")
                    or ((nested_metrics.get("artifacts") or {}) if isinstance(nested_metrics.get("artifacts"), dict) else {}).get("trajectory_path")
                    or ""
                ).strip()
                if not trajectory_path:
                    fallback_path = output_dir / "training_runs" / nested_run_id / "artifacts" / "evaluation_trajectory.npz"
                    if fallback_path.is_file():
                        trajectory_path = str(fallback_path)
                        nested_metrics["evaluation_trajectory_path"] = trajectory_path
                        artifacts = nested_metrics.get("artifacts")
                        if not isinstance(artifacts, dict):
                            artifacts = {}
                        artifacts["evaluation_trajectory_path"] = trajectory_path
                        nested_metrics["artifacts"] = artifacts
                        metrics_path = fallback_path.parent / "metrics.json"
                        try:
                            if metrics_path.is_file():
                                saved_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                                if isinstance(saved_metrics, dict):
                                    saved_metrics.setdefault("evaluation_trajectory_path", trajectory_path)
                                    saved_metrics.setdefault("trajectory_path", trajectory_path)
                                    saved_artifacts = saved_metrics.get("artifacts")
                                    if not isinstance(saved_artifacts, dict):
                                        saved_artifacts = {}
                                    saved_artifacts["evaluation_trajectory_path"] = trajectory_path
                                    saved_metrics["artifacts"] = saved_artifacts
                                    metrics_path.write_text(
                                        json.dumps(saved_metrics, ensure_ascii=False, indent=2),
                                        encoding="utf-8",
                                    )
                        except Exception:
                            logger.debug(
                                "Failed to backfill evaluation trajectory path for run %s",
                                nested_run_id,
                                exc_info=True,
                            )
                group_report.update(
                    {
                        "auxiliary_run_id": nested_run_id,
                        "auxiliary_run_dir": str(output_dir / "training_runs" / nested_run_id),
                        "train_without_holdout_path": str(train_path),
                        "train_time_points": train_times,
                        "training_result_summary": str(nested_result).splitlines()[:8],
                    }
                )
                if not trajectory_path:
                    group_report.update(
                        {
                            "status": "failed",
                            "reason": "auxiliary_run_missing_evaluation_trajectory",
                            "metrics_path": nested_metrics.get("metrics_path"),
                        }
                    )
                    group_reports.append(group_report)
                    continue
                eval_report = self._compute_holdout_w1_from_trajectory(
                    trajectory_path=trajectory_path,
                    full_adata_path=full_path,
                    heldout_times=heldout,
                    time_key=time_key,
                    latent_key=normalized["latent_key"],
                    max_trajectory_time_delta=normalized["max_trajectory_time_delta"],
                )
                group_report.update(eval_report)
                (group_dir / "metrics.json").write_text(
                    json.dumps(group_report, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                group_reports.append(group_report)
        except Exception as exc:
            report.update({"status": "failed", "reason": str(exc), "traceback": traceback.format_exc()})
        finally:
            for key, value in latest_snapshot.items():
                self.state[key] = value
        if group_reports:
            report["group_reports"] = group_reports
            statuses = {str(item.get("status") or "") for item in group_reports}
            report["status"] = "ok" if statuses == {"ok"} else "partial"
            report["mean_w1"] = self._mean_finite([item.get("mean_w1") for item in group_reports])
            report["mean_tmv"] = self._mean_finite([item.get("mean_tmv") for item in group_reports])
            report["reason"] = ""
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return report

    def _attach_holdout_report_to_run_metrics(
        self,
        *,
        run_id: str,
        report: Dict[str, Any],
        holdout_spec: Dict[str, Any],
    ) -> None:
        output_dir = Path(self.state.get("output_dir") or "cytobridge_output").expanduser().resolve()
        run_dir = output_dir / "training_runs" / str(run_id)
        metrics_path = run_dir / "artifacts" / "metrics.json"
        manifest_path = run_dir / "run_manifest.json"
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else {}
            metrics["holdout_time_evaluation"] = report
            normalized = self._normalize_holdout_time_eval_spec(holdout_spec)
            if normalized["attach_to_custom_metrics"] and report.get("mean_w1") is not None:
                custom = metrics.get("custom_metrics")
                if not isinstance(custom, dict):
                    custom = {}
                custom[normalized["metric_name"]] = float(report["mean_w1"])
                metrics["custom_metrics"] = custom
                evaluator_info = normalized.get("claim_metric_evaluator")
                if isinstance(evaluator_info, dict) and evaluator_info:
                    metrics["claim_metric_evaluator"] = dict(evaluator_info)
                    metrics["claim_metric_posthoc_evaluated"] = True
                    metrics["claim_metric_evaluation_source"] = "holdout_time_auxiliary_split"
            metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            logger.debug("Failed to attach holdout report to metrics for run %s", run_id, exc_info=True)
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
            manifest["holdout_time_evaluation_path"] = str(report.get("report_path") or "")
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            logger.debug("Failed to attach holdout report to manifest for run %s", run_id, exc_info=True)

    def run_training_tool(
        self,
        candidate_name: str = None,
        training_algorithm_id: str = None,
        config_overrides: Optional[Dict[str, Any]] = None,
        run_label: str = None,
        adata_path: str = None,
        device: str = None,
        seed: Optional[int] = None,
        decision: str = "provisional",
        decision_reason: str = "",
        holdout_time_evaluation: Optional[Dict[str, Any]] = None,
    ) -> str:
        result = self._run_training_tool_impl(
            candidate_name=candidate_name,
            training_algorithm_id=training_algorithm_id,
            stage="final",
            config_overrides=config_overrides,
            run_label=run_label,
            adata_path=adata_path,
            device=device,
            seed=seed,
            decision=decision,
            decision_reason=decision_reason,
            review_purpose="manual_training",
            review_campaign=None,
            _skip_review_gates=False,
        )
        holdout_spec = holdout_time_evaluation if isinstance(holdout_time_evaluation, dict) else {}
        parent_run_id = str(self.state.get("latest_training_run_id") or "").strip()
        if self._holdout_time_eval_enabled(holdout_spec) and parent_run_id and not str(result or "").startswith(("Error:", "Training failed")):
            report = self._run_holdout_time_evaluation(
                holdout_spec,
                parent_run_id=parent_run_id,
                candidate_name=candidate_name,
                training_algorithm_id=training_algorithm_id,
                stage="final",
                config_overrides=config_overrides or {},
                run_label_prefix=str(run_label or parent_run_id[:18]),
                adata_path=adata_path or str(self.state.get("preprocessed_path") or self.state.get("input_path") or ""),
                device=device or "",
                seed=seed,
                decision_reason="Auxiliary hold-out timepoint evaluation after the normal manual training run.",
            )
            if report:
                self._attach_holdout_report_to_run_metrics(
                    run_id=parent_run_id,
                    report=report,
                    holdout_spec=holdout_spec,
                )
                result = (
                    f"{result}\n\nHold-out time evaluation: status={report.get('status')} "
                    f"mean_w1={report.get('mean_w1')} report={report.get('report_path')}"
                )
        return result

    def _run_training_tool_impl(
        self,
        candidate_name: str = None,
        training_algorithm_id: str = None,
        stage: str = "final",
        config_overrides: Optional[Dict[str, Any]] = None,
        run_label: str = None,
        adata_path: str = None,
        device: str = None,
        seed: Optional[int] = None,
        decision: str = "provisional",
        decision_reason: str = "",
        review_purpose: str = "manual_training",
        review_campaign: Optional[Dict[str, Any]] = None,
        _skip_review_gates: bool = False,
        _record_experiment_registry: bool = True,
    ) -> str:
        """
        Execute builtin or custom training through a standardized run bundle.

        Args:
            candidate_name: Builtin candidate/config name. Mutually exclusive with training_algorithm_id.
            training_algorithm_id: Custom training algorithm id from ~/.cellcompass/training_algorithms.
            stage: "pilot" or "final".
            config_overrides: Config overrides merged on top of builtin/custom spec.
                Supports dot-notation and nested dict merges.
                Epoch edits are blocked by default unless explicitly allowed via:
                {"__allow_epoch_override": true, "__epoch_override_reason": "...", ...}
            run_label: Optional label appended to the generated run_id.
            adata_path: Optional .h5ad path to use for this training run.
                If omitted, falls back to state preprocessed_path/input_path.
            device: Optional device override for this run (for example "cuda" or "cpu").
                If omitted, uses the session-level user_goal.device.
            seed: Optional training random seed. If omitted, defaults to 42.
            decision: For custom algorithms, record this run as promote | reject | provisional.
            decision_reason: Optional rationale for the experiment-management decision.
        """
        from ..display import DisplayManager

        display = DisplayManager()
        target_label = training_algorithm_id or candidate_name or "default"
        display.print_stage("training", f"Training CytoBridge model ({target_label}, stage={stage})...")
        switch_note = self._apply_pending_input_path()
        self._clear_planner_needs_input()

        logger.info(
            "Planner running training (candidate=%s, algorithm=%s, stage=%s)...",
            candidate_name,
            training_algorithm_id,
            stage,
        )

        if stage not in {"pilot", "final"}:
            return f"Error: stage must be 'pilot' or 'final', got '{stage}'"
        if bool(candidate_name) == bool(training_algorithm_id):
            return "Error: Provide exactly one of candidate_name or training_algorithm_id."
        decision_value = str(decision or "provisional").strip().lower()
        if decision_value not in {"promote", "reject", "provisional"}:
            return "Error: decision must be one of promote | reject | provisional."
        decision_reason_value = str(decision_reason or "").strip()
        review_purpose_value = str(review_purpose or "manual_training").strip() or "manual_training"
        campaign_managed_run = review_purpose_value == "campaign"
        review_campaign_payload = dict(review_campaign or {}) if isinstance(review_campaign, dict) else None
        record_experiment_registry = bool(_record_experiment_registry)

        if config_overrides is not None and not isinstance(config_overrides, dict):
            return "Error: config_overrides must be a dict."
        try:
            training_seed = 42 if seed is None else int(seed)
        except Exception:
            return f"Error: seed must be an integer, got {seed!r}"

        normalized_overrides: Dict[str, Any] = deepcopy(config_overrides or {})
        campaign_claim_metric_spec = normalized_overrides.pop("__campaign_claim_metric_spec", None)
        campaign_claim_metric_info: Dict[str, Any] = {}
        allow_epoch_override = bool(normalized_overrides.pop("__allow_epoch_override", False))
        epoch_override_reason = str(normalized_overrides.pop("__epoch_override_reason", "")).strip()

        target_algorithm_context: Dict[str, Any] = {}
        if training_algorithm_id:
            try:
                self.planner_file_tools._ensure_algorithm_id_mutable(
                    str(training_algorithm_id).strip().lower(),
                    action="run_training",
                )
            except Exception as exc:
                return f"Error: {exc}"
            verification = self.planner_file_tools.verify_algorithm_context(training_algorithm_id, stage="training")
            target_algorithm_context = dict(verification.get("context") or {})
            if not bool(verification.get("ok")):
                msg = "; ".join(str(item) for item in (verification.get("blockers") or [])) or "algorithm context verification failed"
                self._emit_event(
                    "algorithm_context_gate_blocked",
                    {
                        "algorithm_id": str(training_algorithm_id),
                        "active_algorithm_id": str((self.state.get("active_algorithm_context") or {}).get("algorithm_id") or ""),
                        "target_algorithm_id": str(training_algorithm_id).strip().lower(),
                        "action": "run_training",
                        "stage": "training",
                        "reason": msg,
                        "context": target_algorithm_context,
                    },
                )
                return f"Error: {msg}"

        VALID_CONFIGS = [
            "dynamical_ot",
            "unbalanced_ot",
            "ruot",
            "crufm",
            "balanced_ot_cfm",
            "sf2m",
            "vgfm",
            "wfrfm",
            "cyto_simulation",
        ]
        user_goal = UserGoal(**self.state["user_goal"])
        requested_device = str(device).strip() if str(device or "").strip() else str(user_goal.device)
        run_bundle = None
        manifest = None

        output_dir = Path(self.state.get("output_dir") or "cytobridge_output")
        explicit_adata_path = str(adata_path or "").strip()
        data_path = explicit_adata_path or self.state.get("preprocessed_path") or self.state.get("input_path")
        if not data_path:
            msg = "Error: No dataset path available. Run preprocessing or provide a dataset path first."
            return f"{switch_note}\n{msg}" if switch_note else msg
        if not Path(str(data_path)).expanduser().exists():
            return f"Error loading data from {data_path}: file does not exist"
        adata = None

        try:
            chosen_candidate: Optional[CandidateConfig] = None
            if candidate_name:
                if self.state.get("candidates"):
                    candidates: List[CandidateConfig] = []
                    for raw_candidate in self.state["candidates"]:
                        if isinstance(raw_candidate, CandidateConfig):
                            candidates.append(raw_candidate)
                        elif isinstance(raw_candidate, dict):
                            # Backward compatibility: older state may store candidate records
                            # with keys like `family` instead of `name`.
                            name = (
                                raw_candidate.get("name")
                                or raw_candidate.get("family")
                                or raw_candidate.get("model_family")
                                or raw_candidate.get("candidate_name")
                                or raw_candidate.get("config_name")
                            )
                            overrides = raw_candidate.get("overrides")
                            if not isinstance(overrides, dict):
                                overrides = raw_candidate.get("config_overrides")
                            if not isinstance(overrides, dict):
                                overrides = {}
                            rationale = str(
                                raw_candidate.get("rationale")
                                or raw_candidate.get("reasoning")
                                or raw_candidate.get("theory_support")
                                or "Recovered from legacy candidate state"
                            )
                            if not str(name or "").strip():
                                logger.warning(
                                    "Skipping candidate dict missing name/family keys during training selection: %s",
                                    raw_candidate,
                                )
                                continue
                            try:
                                candidates.append(
                                    CandidateConfig(
                                        name=str(name).strip(),
                                        overrides=overrides,
                                        rationale=rationale,
                                    )
                                )
                            except Exception:
                                logger.warning(
                                    "Skipping invalid candidate dict during training selection: %s",
                                    raw_candidate,
                                    exc_info=True,
                                )
                                continue
                        elif isinstance(raw_candidate, str):
                            candidates.append(
                                CandidateConfig(
                                    name=raw_candidate,
                                    overrides={},
                                    rationale="Recovered from string candidate state",
                                )
                            )
                        else:
                            logger.warning(
                                "Ignoring unsupported candidate entry of type %s during training selection",
                                type(raw_candidate).__name__,
                            )
                    chosen_candidate = next((c for c in candidates if c.name == candidate_name), None)
                if chosen_candidate is None:
                    if candidate_name in VALID_CONFIGS:
                        logger.info("Creating ad-hoc builtin config for '%s'", candidate_name)
                        chosen_candidate = CandidateConfig(
                            name=candidate_name,
                            overrides={},
                            rationale="User specified directly",
                        )
                    else:
                        return f"Error: '{candidate_name}' is not a valid config. Options: {VALID_CONFIGS}"
            elif training_algorithm_id:
                logger.info("Using custom training algorithm '%s'", training_algorithm_id)

            workspace_root = get_workspace_root()
            base_target = resolve_training_target(
                candidate=chosen_candidate,
                training_algorithm_id=training_algorithm_id,
                input_adata_path=data_path,
                output_dir=str(output_dir),
                stage=stage,
                metadata={
                    "user_goal": user_goal.model_dump(),
                    "requested_device": requested_device,
                    "planner_state_keys": sorted(self.state.keys()),
                    "epoch_policy_base": True,
                },
                search_roots=[get_cellcompass_root() / "training_algorithms"] if training_algorithm_id else None,
                workspace_root=workspace_root,
                config_overrides={},
            )
            base_config = materialize_training_config(
                base_target,
                stage=stage,
                checkpoints_dir=output_dir / ".runtime" / "epoch_policy_preflight",
            )
            normalized_overrides, pruned_config_noops = prune_redundant_config_overrides(
                base_config=base_config,
                overrides=normalized_overrides,
            )
            epoch_blocks = validate_epoch_override_policy(
                base_config=base_config,
                overrides=normalized_overrides,
                allow_epoch_override=allow_epoch_override,
                epoch_override_reason=epoch_override_reason,
            )
            if epoch_blocks:
                return (
                    "Error: epoch overrides are blocked by default to avoid undertrained/unstable comparisons. "
                    "Redundant default/same-value epoch declarations are allowed; effective epoch changes require "
                    "`config_overrides.__allow_epoch_override=true` with non-empty "
                    "`config_overrides.__epoch_override_reason`. "
                    f"Blocked: {json.dumps(epoch_blocks, ensure_ascii=False)}"
                )

            training_target = resolve_training_target(
                candidate=chosen_candidate,
                training_algorithm_id=training_algorithm_id,
                input_adata_path=data_path,
                output_dir=str(output_dir),
                stage=stage,
                metadata={
                    "user_goal": user_goal.model_dump(),
                    "requested_device": requested_device,
                    "planner_state_keys": sorted(self.state.keys()),
                },
                search_roots=[get_cellcompass_root() / "training_algorithms"] if training_algorithm_id else None,
                workspace_root=workspace_root,
                config_overrides=normalized_overrides,
            )
            if isinstance(campaign_claim_metric_spec, dict):
                campaign_claim_metric_info = self._attach_campaign_claim_metric_evaluator(
                    training_target,
                    campaign_claim_metric_spec,
                )

            if training_target.training_mode == "custom" and not _skip_review_gates:
                review_config = materialize_training_config(
                    training_target,
                    stage=stage,
                    checkpoints_dir=output_dir / ".runtime" / "inference_review_preflight",
                )
                review_state = self._ensure_training_review_ready(
                    training_target,
                    stage=stage,
                    resolved_config=review_config,
                    purpose=review_purpose_value,
                    campaign=review_campaign_payload,
                    proposal_id=str(target_algorithm_context.get("proposal_id") or ""),
                )
                if not bool(review_state.get("ok")):
                    reason = str(review_state.get("reason") or "review_required")
                    message = str(review_state.get("message") or "Training review is required before this custom run.")
                    if reason == "implementation_map_required":
                        blockers = "; ".join(str(item) for item in list(review_state.get("blockers") or []))
                        return f"Implementation map required before this custom training run can be trusted. {message}{' Blockers: ' + blockers if blockers else ''}"
                    if reason == "implementation_review_required":
                        return (
                            "Implementation review required before this custom training run can be trusted. "
                            "The runtime will start a read-only implementation evaluator subagent now. "
                            "After it approves, call the training tool again."
                        )
                    if reason == "inference_review_required":
                        return (
                            "Inference review required before this custom training run can be trusted. "
                            "The runtime will start a read-only inference evaluator subagent now. "
                            "After it approves, call the training tool again."
                        )
                    return f"Error: {message}"

            isolate_execution = should_isolate_training_target(training_target.training_mode)
            if not isolate_execution:
                try:
                    from .adata_manager import AnnDataManager

                    manager = AnnDataManager()
                    # Standardized training should start from the on-disk dataset selected by
                    # workflow state / adata_path, not from a potentially mutated in-memory cache.
                    adata = manager.load(data_path, force_reload=True)
                    logger.info(
                        "run_training loaded adata from disk: path=%s shape=(%s, %s)",
                        data_path,
                        getattr(adata, "n_obs", "?"),
                        getattr(adata, "n_vars", "?"),
                    )
                except Exception as e:
                    return f"Error loading data from {data_path}: {e}"

            run_bundle = create_training_run_bundle(
                output_dir=output_dir,
                stage=stage,
                name=training_target.display_name,
                run_label=run_label,
            )
            planner_context = {
                "candidate_name": candidate_name,
                "training_algorithm_id": training_algorithm_id,
                "stage": stage,
                "config_overrides": normalized_overrides or {},
                "pruned_redundant_config_overrides": pruned_config_noops,
                "data_path": data_path,
                "adata_path": explicit_adata_path or None,
                "requested_device": requested_device,
                "training_seed": int(training_seed),
                "output_dir": str(output_dir),
            }
            if campaign_claim_metric_info:
                planner_context["campaign_claim_metric_evaluator"] = campaign_claim_metric_info
            if target_algorithm_context:
                planner_context.update(
                    {
                        "active_algorithm_id": str((self.state.get("active_algorithm_context") or {}).get("algorithm_id") or ""),
                        "target_algorithm_id": str(target_algorithm_context.get("algorithm_id") or ""),
                        "proposal_id": str(target_algorithm_context.get("proposal_id") or ""),
                        "snapshot_id": str(target_algorithm_context.get("active_snapshot_id") or ""),
                        "dirty_since_snapshot": bool(target_algorithm_context.get("dirty_since_snapshot", False)),
                    }
                )
            write_planner_context(run_bundle, planner_context)
            snapshot_algorithm(
                run_bundle,
                training_target.loaded_algorithm.root_dir if training_target.loaded_algorithm else None,
            )
            resolved_config = materialize_training_config(
                training_target,
                stage=stage,
                checkpoints_dir=run_bundle.checkpoints_dir,
            )
            write_resolved_config(run_bundle, resolved_config)

            manifest = {
                **bundle_to_manifest_dict(run_bundle),
                "status": "running",
                "stage": stage,
                "training_mode": training_target.training_mode,
                "candidate_name": chosen_candidate.name if chosen_candidate else None,
                "algorithm_id": training_target.spec.algorithm_id,
                "base_config_ref": str(training_target.spec.base_config),
                "base_config_name": training_target.base_config_name,
                "base_config_dir": (
                    str(training_target.base_config_dir) if training_target.base_config_dir else None
                ),
                "algorithm_source": (
                    training_target.loaded_algorithm.source if training_target.loaded_algorithm else None
                ),
                "algorithm_root": (
                    str(training_target.loaded_algorithm.root_dir)
                    if training_target.loaded_algorithm
                    else None
                ),
                "requested_device": requested_device,
                "training_seed": int(training_seed),
                "resolved_config_path": str(run_bundle.resolved_config_path),
                "active_algorithm_id": str((self.state.get("active_algorithm_context") or {}).get("algorithm_id") or ""),
                "target_algorithm_id": str(target_algorithm_context.get("algorithm_id") or training_target.spec.algorithm_id),
                "proposal_id": str(target_algorithm_context.get("proposal_id") or ""),
                "snapshot_id": str(target_algorithm_context.get("active_snapshot_id") or ""),
                "dirty_since_snapshot": bool(target_algorithm_context.get("dirty_since_snapshot", False)),
                "created_at": datetime.utcnow().isoformat() + "Z",
            }
            if campaign_claim_metric_info:
                manifest["campaign_claim_metric_evaluator"] = campaign_claim_metric_info
            write_run_manifest(run_bundle, manifest)

            progress_id = f"training:{run_bundle.run_id}"

            def training_progress_callback(message: str, progress: float):
                display.print_progress(message, progress, progress_id=progress_id)
                write_training_log(run_bundle, f"[{progress:.3f}] {message}")

            adata_trained = None
            if isolate_execution:
                metrics, error, isolation = run_training_in_subprocess(
                    adata_path=str(data_path),
                    stage=stage,
                    device=requested_device,
                    bundle=run_bundle,
                    candidate_name=chosen_candidate.name if chosen_candidate else "",
                    training_algorithm_id=str(training_algorithm_id or ""),
                    output_dir=str(output_dir),
                    workspace_root=str(workspace_root),
                    config_overrides=normalized_overrides,
                    claim_metric_spec=campaign_claim_metric_spec if isinstance(campaign_claim_metric_spec, dict) else None,
                    seed=training_seed,
                    purpose="training",
                )
                write_training_log(
                    run_bundle,
                    (
                        "[isolation] subprocess returncode="
                        f"{isolation.get('returncode')} elapsed={isolation.get('elapsed_sec')}s "
                        f"timeout={isolation.get('timed_out')} memory_limit_mb={isolation.get('memory_limit_mb')}"
                    ),
                )
            else:
                adata_trained, metrics, error = execute_training_target(
                    adata=adata,
                    target=training_target,
                    stage=stage,
                    device=requested_device,
                    outdir=run_bundle.checkpoints_dir,
                    seed=training_seed,
                    progress_callback=training_progress_callback,
                )

            if error:
                failure_verdict = self._infer_training_verdict({"error": error})
                metrics["run_id"] = run_bundle.run_id
                metrics["run_dir"] = str(run_bundle.run_dir)
                metrics["metrics_path"] = str(run_bundle.metrics_path)
                metrics["resolved_config_path"] = str(run_bundle.resolved_config_path)
                metrics["algorithm_snapshot_path"] = str(run_bundle.algorithm_snapshot_dir)
                if campaign_claim_metric_info:
                    metrics["claim_metric_evaluator"] = campaign_claim_metric_info
                metrics["run_verdict"] = deepcopy(failure_verdict)
                failure_manifest = dict(manifest)
                failure_manifest["status"] = "failed"
                failure_manifest["error"] = error
                failure_manifest["metrics_path"] = str(run_bundle.metrics_path)
                failure_manifest["failed_at"] = datetime.utcnow().isoformat() + "Z"
                failure_manifest["run_verdict"] = deepcopy(failure_verdict)
                finalize_training_outputs(
                    run_bundle,
                    metrics=metrics,
                    run_manifest=failure_manifest,
                )
                if training_target.training_mode == "custom" and record_experiment_registry:
                    run_record = self.planner_file_tools.register_training_run(
                        training_target.spec.algorithm_id,
                        run_manifest=failure_manifest,
                        metrics=metrics,
                        data_path=str(data_path),
                        decision="reject",
                        reason=decision_reason_value or "Training failed before producing a valid model.",
                    )
                    metrics["experiment_run_record"] = run_record
                    failure_manifest["experiment_run_record"] = {
                        "registry_path": run_record.get("registry_path"),
                        "decision": run_record.get("decision"),
                    }
                    finalize_training_outputs(
                        run_bundle,
                        metrics=metrics,
                        run_manifest=failure_manifest,
                    )
                self.state["training_runs"].append(failure_manifest)
                self.state["latest_training_run_id"] = run_bundle.run_id
                self.state["latest_training_run_dir"] = str(run_bundle.run_dir)
                self.state["latest_training_algorithm_id"] = training_target.spec.algorithm_id
                return f"Training failed: {error}"

            if adata_trained is not None:
                write_model_artifact(
                    run_bundle,
                    adata_trained=adata_trained,
                    input_adata_path=data_path,
                    resolved_config_path=run_bundle.resolved_config_path,
                    metrics_path=run_bundle.metrics_path,
                )
                if should_save_full_trained_adata():
                    write_adata_h5ad_safe(adata_trained, run_bundle.trained_model_path)
                    try:
                        artifact_manifest = json.loads(run_bundle.model_artifact_path.read_text(encoding="utf-8"))
                        artifact_manifest["legacy_trained_model_h5ad_path"] = str(run_bundle.trained_model_path)
                        run_bundle.model_artifact_path.write_text(
                            json.dumps(artifact_manifest, indent=2, ensure_ascii=False),
                            encoding="utf-8",
                        )
                    except Exception:
                        pass
            elif not (run_bundle.model_artifact_path.exists() or run_bundle.trained_model_path.exists()):
                raise FileNotFoundError(
                    f"Isolated training completed but did not write a model artifact: {run_bundle.model_artifact_path}"
                )
            model_path_for_consumers = (
                run_bundle.model_artifact_path
                if run_bundle.model_artifact_path.exists()
                else run_bundle.trained_model_path
            )
            metrics["run_id"] = run_bundle.run_id
            metrics["run_dir"] = str(run_bundle.run_dir)
            metrics["model_artifact_path"] = str(run_bundle.model_artifact_path)
            metrics["model_state_path"] = str(run_bundle.model_state_path)
            metrics["trained_model_path"] = str(run_bundle.trained_model_path) if run_bundle.trained_model_path.exists() else ""
            metrics["metrics_path"] = str(run_bundle.metrics_path)
            metrics["resolved_config_path"] = str(run_bundle.resolved_config_path)
            metrics["algorithm_snapshot_path"] = str(run_bundle.algorithm_snapshot_dir)
            metrics["input_adata_path"] = str(data_path)
            metrics["adata_path"] = explicit_adata_path or str(data_path)
            if campaign_claim_metric_info:
                metrics["claim_metric_evaluator"] = campaign_claim_metric_info

            completed_manifest = dict(manifest)
            completed_manifest.update(
                {
                    "status": "completed",
                    "trained_model_path": str(run_bundle.trained_model_path) if run_bundle.trained_model_path.exists() else "",
                    "model_artifact_path": str(run_bundle.model_artifact_path),
                    "model_state_path": str(run_bundle.model_state_path),
                    "metrics_path": str(run_bundle.metrics_path),
                    "completed_at": datetime.utcnow().isoformat() + "Z",
                }
            )
            if training_target.training_mode == "custom":
                try:
                    algorithm_attributes = self.planner_file_tools._campaign_proposal_algorithm_attributes(  # noqa: SLF001
                        {
                            "algorithm_id": training_target.spec.algorithm_id,
                            "proposal_id": completed_manifest.get("proposal_id"),
                        }
                    )
                except Exception:
                    algorithm_attributes = {}
                if algorithm_attributes:
                    metrics["algorithm_attributes"] = dict(algorithm_attributes)
                    completed_manifest["algorithm_attributes"] = dict(algorithm_attributes)
            target_summary = (
                f"candidate={chosen_candidate.name}"
                if chosen_candidate
                else f"algorithm={training_algorithm_id}"
            )
            loss_summary = ""
            final_train_loss = metrics.get("final_train_loss")
            best_train_loss = metrics.get("best_train_loss")
            last_stage_name = metrics.get("last_training_stage_name")
            last_stage_mode = metrics.get("last_training_stage_mode")
            if final_train_loss is not None or best_train_loss is not None:
                stage_bits = []
                if last_stage_name:
                    stage_bits.append(f"name={last_stage_name}")
                if last_stage_mode:
                    stage_bits.append(f"mode={last_stage_mode}")
                stage_prefix = f" ({', '.join(stage_bits)})" if stage_bits else ""
                loss_summary = (
                    f"Training loss{stage_prefix}: "
                    f"saved={final_train_loss}, best={best_train_loss}\n"
                )
            timeout_summary = ""
            if bool(metrics.get("training_timed_out")):
                time_budget = metrics.get("training_time_budget")
                if isinstance(time_budget, dict):
                    timeout_summary = (
                        "Training timeout: "
                        f"budget={time_budget.get('budget_sec')}s, "
                        f"elapsed={time_budget.get('elapsed_sec')}s, "
                        f"completed_epochs={time_budget.get('completed_epochs')}, "
                        f"last_stage={time_budget.get('last_stage_name')}, "
                        f"last_epoch={time_budget.get('last_epoch')}/{time_budget.get('last_total_epochs')}.\n"
                    )
                else:
                    timeout_summary = "Training timeout: wall-clock budget was reached before all configured epochs completed.\n"
            if bool(metrics.get("inference_timed_out")):
                inference_budget = metrics.get("inference_time_budget")
                if isinstance(inference_budget, dict):
                    timeout_summary += (
                        "Inference timeout: "
                        f"budget={inference_budget.get('budget_sec')}s, "
                        f"elapsed={inference_budget.get('elapsed_sec')}s, "
                        f"effective_gap_cells={inference_budget.get('effective_cell_count')}, "
                        f"gaps={inference_budget.get('gap_count')}.\n"
                    )
                else:
                    timeout_summary += "Inference timeout: wall-clock budget was reached before evaluation completed.\n"
            custom_lifecycle_status = "complete"
            custom_note = ""
            if training_target.training_mode == "custom":
                lifecycle_warning = ""
                if not campaign_managed_run:
                    try:
                        registry = self.planner_file_tools._bootstrap_algorithm_registry(  # noqa: SLF001
                            training_target.spec.algorithm_id
                        )
                        custom_lifecycle_status = str(registry.get("algorithm_lifecycle_status") or "developing")
                        lifecycle_reason = str(registry.get("algorithm_lifecycle_status_reason") or "").strip()
                    except Exception:
                        custom_lifecycle_status = "developing"
                        lifecycle_reason = ""
                    if custom_lifecycle_status != "complete":
                        lifecycle_warning = (
                            "\nAlgorithm lifecycle warning: this custom algorithm is still "
                            f"`{custom_lifecycle_status}`"
                            f"{' (' + lifecycle_reason + ')' if lifecycle_reason else ''}. "
                            "A direct `run_training(...)` result is acceptable for debugging, but it is not the "
                            "formal algorithm result. To complete development and produce results suitable for "
                            "user-facing reporting or downstream analysis, run the full campaign lifecycle through "
                            "`start_algorithm_campaign(...)`, `run_campaign_trial(...)`, stage gates, and a "
                            "`final_regression` locked release. See the algorithm-orchestrator and campaign-tuning skills.\n"
                        )
                custom_note = (
                    lifecycle_warning +
                    "\nCustom algorithm note: do not judge this run from a single metric alone. "
                    "Compare against a baseline, typically builtin `crufm`, and check whether the "
                    "algorithm actually solves its intended problem instead of only optimizing one score.\n"
                )
            custom_metrics_summary = ""
            custom_metrics = metrics.get("custom_metrics")
            if isinstance(custom_metrics, dict) and custom_metrics:
                custom_metrics_summary = f"Custom Metrics: {custom_metrics}\n"
            quality_gate_lines: List[str] = []
            w1_scores = metrics.get("w1_scores")
            tmv_scores = metrics.get("tmv_scores")
            tmv_gate = self._tmv_gate_status_for_metrics(metrics)
            tmv_gate_required = bool(tmv_gate.get("required", True))
            if isinstance(tmv_scores, list) and tmv_scores:
                try:
                    tmv_values = [float(x) for x in tmv_scores]
                    mean_tmv = sum(tmv_values) / len(tmv_values)
                    tmv_ok = all(x < 0.3 for x in tmv_values)
                    if tmv_gate_required:
                        quality_gate_lines.append(
                            f"Training quality gate: TMV {'PASS' if tmv_ok else 'FAIL'} "
                            f"(mean={mean_tmv:.4f}; target: every TMV < 0.3)."
                        )
                    else:
                        quality_gate_lines.append(
                            f"Training quality gate: TMV diagnostic only "
                            f"(mean={mean_tmv:.4f}; no hard TMV gate because this run does not model unbalanced mass)."
                        )
                except Exception:
                    if tmv_gate_required:
                        quality_gate_lines.append(
                            "Training quality gate: could not evaluate TMV threshold automatically."
                        )
                    else:
                        quality_gate_lines.append(
                            "Training quality gate: could not evaluate optional TMV diagnostic."
                        )
            elif tmv_gate_required:
                quality_gate_lines.append("Training quality gate: TMV scores are missing.")
            else:
                quality_gate_lines.append(
                    "Training quality gate: TMV hard gate skipped because this run does not model unbalanced mass."
                )
            if isinstance(w1_scores, list) and w1_scores:
                quality_gate_lines.append(
                    "Training quality gate: manually verify that W1 stays within the latent data scale range."
                )
            quality_gate_lines.append(
                "If the quality gate fails, first inspect whether training is undertrained or numerically unstable, "
                "then try hyperparameter adjustment. Modify the algorithm only if the issue is unlikely to be caused by hyperparameters."
            )
            run_verdict = self._infer_training_verdict(metrics)
            metrics["run_verdict"] = deepcopy(run_verdict)
            completed_manifest["run_verdict"] = deepcopy(run_verdict)
            registry_decision = decision_value
            experiment_decision = "campaign_managed" if campaign_managed_run else decision_value
            if training_target.training_mode == "custom" and run_verdict.get("status") == "rejected":
                registry_decision = "reject"
                experiment_decision = "reject"
            completed_manifest["experiment_decision"] = experiment_decision
            completed_manifest["experiment_decision_reason"] = decision_reason_value
            if campaign_managed_run:
                metrics["run_decision_mode"] = "campaign_managed"
                metrics["manual_run_decision"] = "not_applicable"
                metrics["campaign_trial_decision"] = "pending"
                completed_manifest["run_decision_mode"] = "campaign_managed"
                completed_manifest["manual_run_decision"] = "not_applicable"
                completed_manifest["campaign_trial_decision"] = "pending"
            finalize_training_outputs(
                run_bundle,
                metrics=metrics,
                run_manifest=completed_manifest,
            )

            self.state["training_runs"].append(completed_manifest)
            self.state["latest_training_run_id"] = run_bundle.run_id
            self.state["latest_training_run_dir"] = str(run_bundle.run_dir)
            self.state["latest_training_algorithm_id"] = training_target.spec.algorithm_id
            experiment_note = ""
            if training_target.training_mode == "custom" and record_experiment_registry:
                run_record = self.planner_file_tools.register_training_run(
                    training_target.spec.algorithm_id,
                    run_manifest=completed_manifest,
                    metrics=metrics,
                    data_path=str(data_path),
                    decision=registry_decision,
                    reason=decision_reason_value or f"Training run recorded as {experiment_decision}.",
                )
                metrics["experiment_run_record"] = run_record
                completed_manifest["experiment_run_record"] = {
                    "registry_path": run_record.get("registry_path"),
                    "decision": run_record.get("decision"),
                }
                experiment_note = (
                    f"Experiment decision: {experiment_decision}"
                    f"{' | reason=' + decision_reason_value if decision_reason_value else ''}\n"
                    f"Experiment registry: {run_record.get('registry_path')}\n"
                )
            manual_developing_custom_run = (
                training_target.training_mode == "custom"
                and not campaign_managed_run
                and custom_lifecycle_status != "complete"
            )
            verdict_status = str(run_verdict.get("status") or "").strip().lower()
            run_quality_rejected = verdict_status == "rejected"
            if stage == "final" and not manual_developing_custom_run and not run_quality_rejected:
                self.state["final_config"] = {
                    "path": str(model_path_for_consumers),
                    "run_id": run_bundle.run_id,
                    "run_dir": str(run_bundle.run_dir),
                }
                self.state["final_metrics"] = metrics
            quality_gate_summary = "\n".join(quality_gate_lines) + "\n"
            if manual_developing_custom_run:
                next_step = (
                    "NEXT STEP: treat this as debug evidence only; use campaign tools through "
                    "final_regression locked release before reporting final algorithm results or downstream analysis."
                )
            else:
                if run_quality_rejected:
                    next_step = (
                        "NEXT STEP: DO NOT use this run for downstream analysis, final file delivery, "
                        "or report claims. Treat the run as failed quality evidence, inspect "
                        "`artifacts/metrics.json`, `resolved_config.yaml`, and `logs/training.log`, "
                        "then retrain with corrected data/time keys or adjusted hyperparameters/model family."
                    )
                else:
                    next_step = (
                        "NEXT STEP: continue with downstream analysis or report authoring in the single-agent workflow."
                        if stage == "final"
                        else "NEXT STEP: Review the pilot run metrics or launch a final training run."
                    )
            training_status_header = (
                "🔴 TRAINING FINISHED BUT QUALITY GATE REJECTED.\n"
                if run_quality_rejected
                else "✅ TRAINING COMPLETED SUCCESSFULLY.\n"
            )
            rejected_block = ""
            if run_quality_rejected:
                rejected_block = (
                    "🔴 DO NOT PROCEED WITH THIS MODEL.\n"
                    "The saved artifacts exist, but `run_verdict.status` is `rejected`; "
                    "this means the trained dynamics are not acceptable evidence. "
                    "A public/schema verifier can still pass from these files, but that does not make "
                    "the model scientifically usable.\n"
                )
            return (
                f"{(switch_note + chr(10)) if switch_note else ''}{training_status_header}"
                f"Run ID: {run_bundle.run_id}\n"
                f"Target: {target_summary}\n"
                f"Stage: {stage}\n"
                f"Saved to: {model_path_for_consumers}\n"
                f"Run dir: {run_bundle.run_dir}\n"
                f"Metrics: W1={metrics.get('w1_scores')}, TMV={metrics.get('tmv_scores')}\n"
                f"Run verdict: {run_verdict.get('status')} | likely_cause={run_verdict.get('likely_cause')}\n"
                f"{rejected_block}"
                f"{experiment_note}"
                f"{custom_metrics_summary}"
                f"{loss_summary}"
                f"{timeout_summary}"
                f"{quality_gate_summary}"
                f"{custom_note}"
                f"\n🔴 CRITICAL: DO NOT call `run_training_tool` again. "
                f"{next_step}"
            )

        except Exception as e:
            tb = traceback.format_exc()
            if run_bundle is not None:
                failure_verdict = self._infer_training_verdict({"error": str(e)})
                failure_manifest = dict(manifest or bundle_to_manifest_dict(run_bundle))
                failure_manifest["status"] = "failed"
                failure_manifest["error"] = str(e)
                failure_manifest["traceback"] = tb
                failure_manifest["run_verdict"] = deepcopy(failure_verdict)
                try:
                    finalize_training_outputs(
                        run_bundle,
                        metrics={"error": str(e), "traceback": tb, "run_verdict": deepcopy(failure_verdict)},
                        run_manifest=failure_manifest,
                    )
                    if training_algorithm_id and record_experiment_registry:
                        self.planner_file_tools.register_training_run(
                            str(training_algorithm_id),
                            run_manifest=failure_manifest,
                            metrics={"error": str(e), "traceback": tb, "run_verdict": deepcopy(failure_verdict)},
                            data_path=str(data_path),
                            decision="reject",
                            reason=decision_reason_value or "Training raised an exception and the run is invalid.",
                        )
                except Exception:
                    logger.debug("Failed to finalize errored training run bundle", exc_info=True)
            logger.error("Training execution error: %s", e, exc_info=True)
            return f"Training execution error: {e}\nTraceback:\n{tb}"

    def create_plan(self, plan_text: str = "", explanation: str = "", instruction: str = "") -> str:
        """
        Parse and store a detailed execution plan supplied by the model.
        This tool does not call LLM internally.

        Use `plan_text` as the primary payload.
        `instruction` is accepted for backward compatibility and treated as plan text when `plan_text` is empty.
        """
        from ..display import DisplayManager

        display = DisplayManager()
        display.print_stage("planning", "Structuring provided plan...")
        logger.info("create_plan tool called with provided plan text")

        plan_source = str(plan_text or "").strip() or str(instruction or "").strip()
        if not plan_source:
            return (
                "Error: `create_plan` now requires `plan_text` (or legacy `instruction`) "
                "containing the full plan text to parse."
            )

        self.state["planner_mode_plan_draft"] = plan_source
        plan_state, parse_err = self.plan_service.set_from_text(
            plan_source,
            explanation=explanation or "Structured from provided create_plan payload",
            source="create_plan",
        )
        if parse_err:
            return f"Error: failed to parse provided plan_text: {parse_err}"

        assert plan_state is not None
        return f"✅ Plan generated and structured.\n{render_plan_state(plan_state)}"

    def get_tools(self) -> List[StructuredTool]:
        tools = [
            StructuredTool.from_function(
                func=self.create_plan,
                name="create_plan",
                description="Store a pre-written plan text as structured execution steps (no internal LLM call)."
            ),
            StructuredTool.from_function(
                func=self.set_plan_from_text,
                name="set_plan_from_text",
                description="Parse free-text plan into structured steps with statuses."
            ),
            StructuredTool.from_function(
                func=self.update_plan,
                name="update_plan",
                description="Update structured execution plan. Plan items require step + status (pending/in_progress/completed)."
            ),
            StructuredTool.from_function(
                func=self.get_plan_status,
                name="get_plan_status",
                description="Get current execution plan status and progress."
            ),
            StructuredTool.from_function(
                func=self.list_skills,
                name="list_skills",
                description="List available skills with descriptions and SKILL.md paths. Optional domain: all | planner | workflow | downstream | algorithm. Use this when the relevant skill path is uncertain, then read the chosen SKILL.md with read_file before following it."
            ),
            StructuredTool.from_function(
                func=self.run_training_tool,
                name="run_training_tool",
                description="Execute a manual/debug training run with the standardized backend. Provide candidate_name or training_algorithm_id. Optional adata_path overrides the dataset used for this run; otherwise training uses the current workflow dataset path (usually preprocessed_path). Pass sparse config_overrides: tune one parameter with one leaf path such as training.plan[0].lr, not a copied training/model config block; redundant copied values are pruned before training. Raw adata changes and extra modality assembly belong in training_data_builder(...); backend builders receive only build_context.training_data. For custom algorithms this does not complete development; final user-facing/downstream results must come from a campaign final_regression locked release. Effective epoch changes are blocked by default unless explicitly allowed with reason in config_overrides."
            ),
            StructuredTool.from_function(
                func=self.execute_python,
                name="execute_python",
                description="Execute Python code for quick data exploration or validation. Use for simple checks, verifying sub-agent outputs, or quick plots. Provide 'code' and optionally 'reason' and 'timeout' (seconds, default 300). If the timeout is exceeded, execution is interrupted and the tool returns a timeout message."
            ),
            StructuredTool.from_function(
                func=self.list_path,
                name="list_path",
                description="List a file or directory path. Useful to inspect user-provided data locations before loading."
            ),
            StructuredTool.from_function(
                func=self.read_file,
                name="read_file",
                description="Unified file reader for text, images, and PDFs. Uses line-based offset/limit for text; for PDFs choose pdf_mode='text' for extracted text or pdf_mode='render' for page images, with optional pages like 1-3,5."
            ),
            StructuredTool.from_function(
                func=self.find_files,
                name="find_files",
                description="Find files by glob pattern under a path. Good for discovering docs and configs."
            ),
            StructuredTool.from_function(
                func=self.grep_files,
                name="grep_files",
                description="Search text pattern in files and return path:line snippets."
            ),
            StructuredTool.from_function(
                func=self.set_task_profile,
                name="set_task_profile",
                description="Set a multi-axis task profile for the current work. Use this instead of a single task_mode when the task mixes reproduction, tuning, integration, or new algorithm design."
            ),
            StructuredTool.from_function(
                func=self.check_workflow_gate,
                name="check_workflow_gate",
                description="Check whether the current state satisfies the minimum gate for a stage such as proposal, authoring, review, training, downstream, or report."
            ),
            StructuredTool.from_function(
                func=self.preview_training_run,
                name="preview_training_run",
                description="Training preview with full-data chain inspection plus an optional 1-epoch inference/evaluation smoke test. Inspection may run backend setup/build_state/sample_pairs before any epoch logs; a timeout there means setup scalability or contract failure, not slow one-epoch training. Do not treat 1-epoch metric magnitude as final performance."
            ),
            StructuredTool.from_function(
                func=self.get_algorithm_proposal_template,
                name="get_algorithm_proposal_template",
                description=(
                    "Read-only helper that returns the current standard CytoBridge algorithm proposal template "
                    "and per-section writing guidance. Use before create_algorithm_proposal(...) or "
                    "revise_algorithm_proposal(...)."
                ),
            ),
            StructuredTool.from_function(
                func=self.create_algorithm_proposal,
                name="create_algorithm_proposal",
                description=(
                    "Create/update a conceptual algorithm proposal before implementation. "
                    "Proposal must focus on theory/math/pseudocode level (no real code) "
                    "and include abstract, literature_and_package_grounding, problem statement, claimed capability, "
                    "novelty_and_contributions for genuinely new algorithms, mathematical_derivation_to_algorithm_design "
                    "when the method claims a mathematical objective or algorithmic novelty, "
                    "machine-readable mass_modeling_scope, distribution recovery argument, implementation pseudocode "
                    "(formal paper-style format preferred but format-only issues warn rather than block), evaluation plan, "
                    "expected evaluation outcome, and overengineering self-check. "
                    "If unbalanced mass is claimed, cell-count/total-mass changes must come from a meaningful growth/dynamics mechanism, not post-hoc correction or target-count clocks. "
                    "For genuinely new algorithms, first use search_theory for mathematical foundations and search_literature for algorithm papers/baselines, then read relevant page/section ranges with read_file; literature_and_package_grounding should cite at least 10 directly relevant papers/notes plus theory-book page or chapter/section/page ranges when formal OT/SB/WFR/UOT/continuity-equation theory is used."
                ),
            ),
            StructuredTool.from_function(
                func=self.revise_algorithm_proposal,
                name="revise_algorithm_proposal",
                description=(
                    "Create a new revision of an existing algorithm proposal and trigger the normal proposal review flow. "
                    "Use proposal_markdown for full proposal rewrites and proposal_patch for small section edits; provide exactly one. "
                    "This preserves proposal_id history, registry artifacts, risk artifacts, and review semantics."
                ),
            ),
            StructuredTool.from_function(
                func=self.review_algorithm_proposal,
                name="review_algorithm_proposal",
                description="Record user review decision for the exact proposal_id of an algorithm proposal. decision: approve | reject | revise.",
            ),
            StructuredTool.from_function(
                func=self.get_algorithm_proposal_status,
                name="get_algorithm_proposal_status",
                description=(
                    "Inspect algorithm proposal status for one algorithm_id or all proposals. "
                    "For revisions, prefer revise_algorithm_proposal(...); editable_proposal_path is available for direct root PROPOSAL.md patches when needed."
                ),
            ),
            StructuredTool.from_function(
                func=self.create_research_idea,
                name="create_research_idea",
                description="Create a persistent research-idea asset that defines one concrete scientific question, prior-work basis, evidence basis, and 1-5 feasible direction families without committing to a full algorithm.",
            ),
            StructuredTool.from_function(
                func=self.revise_research_idea,
                name="revise_research_idea",
                description="Create a new revision of an existing research idea while preserving unspecified fields from the latest revision.",
            ),
            StructuredTool.from_function(
                func=self.review_research_idea,
                name="review_research_idea",
                description="Record user review decision for a research idea. decision: approve | reject | revise.",
            ),
            StructuredTool.from_function(
                func=self.get_research_idea_status,
                name="get_research_idea_status",
                description="Inspect research idea status for one idea_id or all recorded ideas.",
            ),
            StructuredTool.from_function(
                func=self.list_research_ideas,
                name="list_research_ideas",
                description="List recorded research ideas with optional track/inactive filtering.",
            ),
            StructuredTool.from_function(
                func=self.set_active_research_idea,
                name="set_active_research_idea",
                description="Make one research idea the active portfolio item for the current session.",
            ),
            StructuredTool.from_function(
                func=self.update_research_idea_progress,
                name="update_research_idea_progress",
                description="Update research-idea progress, attach evidence/next steps, and explicitly change resolution_status when justified.",
            ),
            StructuredTool.from_function(
                func=self.link_algorithm_to_idea,
                name="link_algorithm_to_idea",
                description="Weakly associate an algorithm with a research idea and, when a proposal exists, persist the proposal's primary_idea_id as well.",
            ),
            StructuredTool.from_function(
                func=self.init_training_algorithm_workspace,
                name="init_training_algorithm_workspace",
                description="Create a custom training algorithm workspace under ~/.cellcompass/training_algorithms with manifest.yaml, algorithm.py, config.yaml, and README.md. config.yaml is seeded from builtin vgfm. Requires an approved algorithm proposal first."
            ),
            StructuredTool.from_function(
                func=self.record_decision,
                name="record_decision",
                description="Record a structured experiment decision for a custom algorithm workspace."
            ),
            StructuredTool.from_function(
                func=self.mark_algorithm_failed,
                name="mark_algorithm_failed",
                description=(
                    "Mark a developing custom algorithm lifecycle as failed when evidence shows this direction cannot satisfy "
                    "the user goal. This is not task completion; after using it, revise/repair the algorithm, revise the proposal, "
                    "or design a replacement algorithm."
                )
            ),
            StructuredTool.from_function(
                func=self.mark_result_obsolete,
                name="mark_result_obsolete",
                description="Mark a proposal or run as obsolete so it is no longer treated as valid evidence."
            ),
            StructuredTool.from_function(
                func=self.list_experiment_history,
                name="list_experiment_history",
                description="List registry-backed proposal, snapshot, run, decision, and obsolete history for a custom algorithm."
            ),
            StructuredTool.from_function(
                func=self.snapshot_active_algorithm_workspace,
                name="snapshot_active_algorithm_workspace",
                description="Create a clean registry snapshot of the currently active algorithm workspace and clear its dirty flag."
            ),
            StructuredTool.from_function(
                func=self.rollback_algorithm_workspace,
                name="rollback_algorithm_workspace",
                description="Restore a custom algorithm workspace from a recorded workspace snapshot or proposal-linked snapshot."
            ),
            StructuredTool.from_function(
                func=self.set_active_baseline_run,
                name="set_active_baseline_run",
                description="Promote a registered custom-algorithm run to the active baseline for its experiment track."
            ),
            StructuredTool.from_function(
                func=self.compare_algorithm_runs,
                name="compare_algorithm_runs",
                description="Compare registered custom-algorithm runs and report deltas versus the active baseline when available."
            ),
            StructuredTool.from_function(
                func=self.list_algorithm_benchmarks,
                name="list_algorithm_benchmarks",
                description="List prepared algorithm benchmark datasets under ~/.cellcompass/algorithm_benchmarks. Use before campaign tuning to choose a real, comparable frozen stage panel."
            ),
            StructuredTool.from_function(
                func=self.get_algorithm_benchmark_dataset,
                name="get_algorithm_benchmark_dataset",
                description="Inspect one benchmark dataset card, including data_path, available fields, README, and campaign dataset_config_overrides example."
            ),
            StructuredTool.from_function(
                func=self.get_algorithm_benchmark_baselines,
                name="get_algorithm_benchmark_baselines",
                description="Read builtin/reference W1/TMV baseline records and leaderboard for one benchmark dataset."
            ),
            StructuredTool.from_function(
                func=self.list_algorithm_benchmark_baselines,
                name="list_algorithm_benchmark_baselines",
                description="List concise builtin/reference baseline metrics and local benchmark-managed model/config/log artifact paths for one dataset. Use this before debugging against baseline checkpoints."
            ),
            StructuredTool.from_function(
                func=self.record_algorithm_benchmark_baseline,
                name="record_algorithm_benchmark_baseline",
                description="Record builtin/reference W1/TMV baseline metrics for one benchmark dataset and update its leaderboard. Do not use for custom claim metrics."
            ),
            StructuredTool.from_function(
                func=self.make_benchmark_dataset_config,
                name="make_benchmark_dataset_config",
                description="Build the dataset_config_overrides payload expected by run_campaign_trial from registered benchmark dataset ids. Automatically includes benchmark-managed dataset trial configs from trial_configs/default.yaml or trial_configs/<stage>.yaml when present, then applies common_config_overrides and per_dataset_config_overrides. Pass sparse config overrides: when tuning one hyperparameter, include only that leaf path/value, not a copied training/model config block. The stage freezes dataset ids, not per-dataset config values."
            ),
            StructuredTool.from_function(
                func=self.register_algorithm_benchmark_dataset,
                name="register_algorithm_benchmark_dataset",
                description="Admin tool: validate and register a prepared benchmark .h5ad. It does not preprocess; missing obs['time_point_processed'] or obsm['X_latent'] returns invalid_contract with preprocessing instructions."
            ),
            StructuredTool.from_function(
                func=self.register_stage2_simulation_dataset,
                name="register_stage2_simulation_dataset",
                description="Register an agent-designed Stage 2 claim-validation simulation dataset with a frozen generator hash and simulation_version. Use this instead of plain benchmark registration for claim simulations."
            ),
            StructuredTool.from_function(
                func=self.generate_and_register_stage2_simulation_dataset,
                name="generate_and_register_stage2_simulation_dataset",
                description="Generate a DynBench/BoolODE-style Stage 2 simulation from a claim-directed scenario config, freeze the generator/config, register the generated train.h5ad as a benchmark dataset, and return dataset_config_overrides for run_campaign_trial. Prefer real benchmark datasets when they can test the claim."
            ),
            StructuredTool.from_function(
                func=self.start_algorithm_campaign,
                name="start_algorithm_campaign",
                description="Start an autoresearch-style custom algorithm tuning campaign with git archive, staged policies, and automatic trial promote/reject. New campaigns default to the server-configured baseline policy, normally strict_all_builtin: builtin baselines are fixed comparators and gates audit the full fixed builtin set; explicit shorter builtin baseline lists on refresh only scope repair runs for named missing/stale baselines. The older permissive mode is an operator/server setting, not an agent-tunable campaign shortcut. For a custom Stage 2 claim metric, claim_metric_spec must include evaluator_path so refresh_campaign_stage_baselines can compute the same metric from saved candidate/baseline trajectories; name/direction alone is not comparable. claim_metric_spec may also include tmv_gate_required and baseline_metric_adapters for baseline output standardization or explicit unsupported declarations."
            ),
            StructuredTool.from_function(
                func=self.update_campaign_claim_metric_spec,
                name="update_campaign_claim_metric_spec",
                description="Repair the authoritative claim_metric_spec for an existing campaign. Use this when Stage 2 returns claim_metric_evaluator_required because evaluator_path is missing. This is the only sanctioned way to add or correct claim_metric_spec.evaluator_path on an active campaign; do not patch campaign.json directly, pass evaluator_path through dataset_config_overrides, or start a duplicate campaign just to repair the evaluator."
            ),
            StructuredTool.from_function(
                func=self.set_campaign_stage_panel,
                name="set_campaign_stage_panel",
                description="Freeze the dataset ids for one campaign stage before trials. Build the payload with make_benchmark_dataset_config(...), pass its dataset_config_overrides here, then run_campaign_trial(...) can omit the payload or tune sparse leaf config overrides for the same ids. This locks dataset ids, not per-dataset config values."
            ),
            StructuredTool.from_function(
                func=self.refresh_campaign_stage_baselines,
                name="refresh_campaign_stage_baselines",
                description="Refresh builtin/reference baselines for the frozen campaign stage panel, auto-select comparable stage baselines, and write them into the stage gate when required evidence is complete. Under the default strict_all_builtin campaign policy, gates audit all default builtin baselines as fixed comparators; explicit shorter baseline_algorithms only scope repair runs for named missing/stale baselines and do not weaken the gate. Permissive mode is an operator/server-selected compatibility mode. Stage 2/3 gates keep independent SOTA records: claim metrics compare to the strongest claim-metric baseline, while W1 compares to the lowest-W1 baseline. For Stage 2 custom claim metrics, candidate and baseline values must come from the same campaign claim_metric_spec evaluator and carry matching claim_metric_evaluator provenance; static direct fields and numeric adapters are not accepted as comparable evidence. For builtin refreshes, includes the Required anchor baseline declared in IMPLEMENTATION_MAP.md when needed. Reuses existing records/trajectories/models when possible; run_missing only trains truly missing baselines and does not retrain an existing Stage 2 baseline just because claim_metric_spec lacks evaluator_path."
            ),
            StructuredTool.from_function(
                func=self.run_campaign_control_baseline,
                name="run_campaign_control_baseline",
                description="Run an agent-declared ablation/shuffle/no-lag control on the same frozen campaign stage panel using either sparse config_overrides or a temporary workspace_patch, aggregate metrics, and register or replace the control baseline. This never promotes/rejects a campaign trial and never changes active best."
            ),
            StructuredTool.from_function(
                func=self.run_campaign_locked_algorithm_baseline,
                name="run_campaign_locked_algorithm_baseline",
                description="Run a completed final-regression locked custom algorithm as an audited reference baseline on the current campaign's frozen stage panel, then register or replace it as gate evidence. For h5ad format incompatibility, provide reference_dataset_adapter as an explicit Python script that materializes an adapted h5ad. For claim-metric context incompatibility, provide reference_claim_metric_adapter defining adapt_baseline_metric_context(context); it wraps the current campaign evaluator and must not replace it. reference_config_overrides are rejected to avoid guess-based retuning."
            ),
            StructuredTool.from_function(
                func=self.update_campaign_control_baseline,
                name="update_campaign_control_baseline",
                description="Deactivate, reactivate, update, or replace a previously registered control/ablation baseline with revision history. Gate checks ignore inactive controls."
            ),
            StructuredTool.from_function(
                func=self.compute_campaign_claim_metric_for_baselines,
                name="compute_campaign_claim_metric_for_baselines",
                description="Stage 2 helper: compute the campaign claim metric for builtin/reference baselines on the frozen panel using saved trajectories/models when possible, then update the comparable stage baseline. Returns structured missing/unsupported records."
            ),
            StructuredTool.from_function(
                func=self.query_campaign_baseline_metrics,
                name="query_campaign_baseline_metrics",
                description=(
                    "Read-only evidence query for actual measured campaign baseline metrics. "
                    "Use before writing or reviewing paper baseline/result tables: it reports compact scalar builtin/reference "
                    "baseline metrics, W1 backend provenance, strict-ledger completeness and repair guidance from campaign "
                    "selected comparators, benchmark registry records, and local baseline training_run metrics.json files. "
                    "Nested evaluator diagnostics stay in the returned metrics_path rather than entering agent context. "
                    "This does not train, refresh, tune, or mutate campaign state; paper numbers should cite this output or "
                    "the returned metrics_path, not draft manuscript tables."
                )
            ),
            StructuredTool.from_function(
                func=self.get_algorithm_campaign_status_summary,
                name="get_algorithm_campaign_status",
                description=(
                    "Return concise campaign status with current stage, gate status, active best, blocked reason, "
                    "next action, and recent trials. This agent-facing tool does not return the raw campaign registry."
                )
            ),
            StructuredTool.from_function(
                func=self.start_campaign_trial,
                name="start_campaign_trial",
                description="Ensure the campaign has one open trial. If a trial is already open, returns that same trial; otherwise starts from the current stage active best or from a rejected trial as a non-active working base."
            ),
            StructuredTool.from_function(
                func=self.run_campaign_trial,
                name="run_campaign_trial",
                description="Archive the current algorithm workspace, read the live workspace config.yaml, run the frozen stage panel or provided make_benchmark_dataset_config(...) payload with dataset-specific config overrides, record effective config/resolved_config paths per dataset, and automatically promote/reject the trial. For tuning, pass sparse config overrides: if changing one hyperparameter, pass only that leaf path/value such as {'config_overrides': {'training.plan[0].lr': 0.002}} or per-dataset equivalent, not a copied training/model config block. Redundant copied defaults are pruned before training, but sparse overrides are expected. On promote, a single consistent resolved_config.yaml is synced back to workspace config.yaml; distinct dataset-specific resolved configs are not collapsed into the global config. The agent does not pass a run decision."
            ),
            StructuredTool.from_function(
                func=self.resume_rejected_trial,
                name="resume_rejected_trial",
                description="Resume a rejected trial as a working base without changing the campaign active best unless the new trial later promotes."
            ),
            StructuredTool.from_function(
                func=self.list_campaign_trials,
                name="list_campaign_trials",
                description=(
                    "List concise paginated campaign trial summaries, optionally filtering by decision or stage. "
                    "Defaults to the 10 most recent matching trials and never returns raw full trial payloads."
                )
            ),
            StructuredTool.from_function(
                func=self.check_campaign_stage_gate,
                name="check_campaign_stage_gate",
                description="Check whether the current stage active best passes its lifecycle gate and persist stage_gate_evidence in the campaign registry. Stage 3 has an optional SOTA/Pareto early-completion gate: W1 and the validated claim metric should be non-worse than the strongest comparable external baseline, with claim-metric regression capped by the configured tolerance. If the Stage 3 budget is exhausted without that optional gate, final regression is still allowed. Use advance=false to inspect readiness without moving stages; use advance=true to advance or lock after a pass."
            ),
            StructuredTool.from_function(
                func=self.list_workspace_tree,
                name="list_workspace_tree",
                description="List the planner-managed algorithm workspace or another allowlisted workspace path."
            ),
            StructuredTool.from_function(
                func=self.read_workspace_file,
                name="read_workspace_file",
                description="Read an allowlisted workspace file used for custom algorithm development or run inspection."
            ),
            StructuredTool.from_function(
                func=self.create_workspace_file,
                name="create_workspace_file",
                description="Create a new text file inside planner-managed writable roots."
            ),
            StructuredTool.from_function(
                func=self.create_algorithm_workspace_artifact,
                name="create_algorithm_workspace_artifact",
                description=(
                    "Create a file or directory under the active algorithm workspace using a relative path. "
                    "Use this for diagnostics scripts and outputs such as diagnostics/check_risk.py or "
                    "diagnostics/risk_summary.json. Absolute paths, '..', inactive algorithms, and unapproved "
                    "proposal workspaces are rejected."
                )
            ),
            StructuredTool.from_function(
                func=self.replace_workspace_file,
                name="replace_workspace_file",
                description="Replace the full contents of an allowlisted workspace file."
            ),
            StructuredTool.from_function(
                func=self.patch_algorithm_config,
                name="patch_algorithm_config",
                description=(
                    "Safely patch the active custom algorithm config.yaml with typed leaf path updates and return a resolved unified diff. "
                    "Writes by default so campaign promote/reject archives the real config file; set dry_run=True only for preview. "
                    "Use this as a shortcut for simple scalar hyperparameter edits; use apply_workspace_patch for complex config edits."
                )
            ),
            StructuredTool.from_function(
                func=self.apply_workspace_patch,
                name="apply_workspace_patch",
                description=(
                    "Apply patch edits to allowlisted workspace files only (package source paths are blocked). "
                    "Supported patch formats: Codex patch (`*** Begin Patch ... *** End Patch`) "
                    "and unified diff (`---/+++` with `@@` hunks). Patch headers are real paths, not git path aliases: "
                    "use absolute paths or tool-returned workspace paths; do not use `--- a/file` / `+++ b/file`. "
                    "For proposal revisions, prefer revise_algorithm_proposal(...), which supports full markdown replacement "
                    "and local patches while preserving review semantics. Direct PROPOSAL.md patches still create a new proposal_id "
                    "and trigger proposal review, but must target editable_proposal_path and must not mix proposal edits with code/config edits."
                )
            ),
            StructuredTool.from_function(
                func=self.preview_workspace_diff,
                name="preview_workspace_diff",
                description="Preview a unified diff for an allowlisted workspace file without writing changes."
            ),
            StructuredTool.from_function(
                func=self.set_tool_activation_mode,
                name="set_tool_activation_mode",
                description="Set delayed activation mode for generated downstream tools. Values: auto_next_turn or manual_review."
            ),
            StructuredTool.from_function(
                func=self.set_tool_harvest_enabled,
                name="set_tool_harvest_enabled",
                description="Enable or disable downstream reusable-tool harvesting at end of each turn."
            ),
            StructuredTool.from_function(
                func=self.get_tool_catalog_status,
                name="get_tool_catalog_status",
                description="Inspect downstream tool catalog including active builtin/generated tools and pending candidates."
            ),
            StructuredTool.from_function(
                func=self.approve_pending_tools,
                name="approve_pending_tools",
                description="Approve pending generated tools and activate currently eligible ones."
            ),
            StructuredTool.from_function(
                func=self.reject_pending_tools,
                name="reject_pending_tools",
                description="Reject pending generated tools."
            ),
            StructuredTool.from_function(
                func=self.disable_generated_tool,
                name="disable_generated_tool",
                description="Disable an already active generated tool by name or tool_id."
            ),
        ]
        if self._normalize_algorithm_proposal_mode() == "always_user_review":
            tools = [t for t in tools if t.name != "review_algorithm_proposal"]
        if self._normalize_idea_review_mode() == "always_user_review":
            tools = [t for t in tools if t.name != "review_research_idea"]
        return tools
