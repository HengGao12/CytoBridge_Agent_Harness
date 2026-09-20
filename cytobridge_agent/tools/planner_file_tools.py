from __future__ import annotations

import difflib
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import yaml

from ..algorithm_ownership import AlgorithmOwnershipError, AlgorithmOwnershipRegistry
from .file_tools import ReadResult, list_path as list_path_impl, read_file as read_file_impl
from .research_idea_registry import (
    research_idea_dir,
    research_idea_json_path,
    research_idea_markdown_path,
    research_idea_registry_dir,
    research_idea_registry_index_path,
    research_idea_revisions_dir,
    research_idea_root,
)
from .workspace_policy import WorkspacePolicy
from .training_tools import (
    _override_path_parts,
    get_cellcompass_root,
    load_config as load_training_config,
    validate_epoch_override_policy,
)


@dataclass
class PatchOperation:
    kind: str
    path: str
    lines: List[str]


REGISTRY_TRACKED_WORKSPACE_FILES = (
    "algorithm.py",
    "config.yaml",
    "manifest.yaml",
    "PROPOSAL.md",
    "PROPOSAL.json",
    "risk.md",
    "IMPLEMENTATION_MAP.md",
    "README.md",
)

CAMPAIGN_STAGE_ORDER = (
    "stage1_feasibility",
    "stage2_claim_validation",
    "stage3_tuning",
    "final_regression",
)

CAMPAIGN_STAGE_MAX_TRIALS = {
    "stage1_feasibility": 10,
    "stage2_claim_validation": 50,
    "stage3_tuning": 30,
    "final_regression": 1,
}
CAMPAIGN_BASELINE_SELECTION_POLICIES = {"strict_all_builtin", "permissive"}
DEFAULT_CAMPAIGN_BASELINE_SELECTION_POLICY = "strict_all_builtin"
LOCKED_STAGE_GATE_SPEC_ENV = "CYTOBRIDGE_LOCKED_STAGE_GATE_SPEC_PATH"
LOCKED_STAGE_GATE_INLINE_ENV = "CYTOBRIDGE_LOCKED_STAGE_GATE_SPEC_JSON"
IMPLEMENTATION_REVIEW_POLICY_VERSION = 4
IMPLEMENTATION_RISK_SECTION_START = "<!-- CYTOBRIDGE_IMPLEMENTATION_REVIEW_RISKS_START -->"
IMPLEMENTATION_RISK_SECTION_END = "<!-- CYTOBRIDGE_IMPLEMENTATION_REVIEW_RISKS_END -->"


class PlannerFileTools:
    def __init__(
        self,
        state: Dict[str, Any],
        policy_provider: Callable[[], WorkspacePolicy],
        event_sink: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> None:
        self.state = state
        self.policy_provider = policy_provider
        self.event_sink = event_sink

    def _emit(self, event_type: str, payload: Dict[str, Any]) -> None:
        if not self.event_sink:
            return
        try:
            self.event_sink(event_type, payload)
        except Exception:
            pass

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _policy(self) -> WorkspacePolicy:
        policy = self.policy_provider()
        self.state["planner_workspace_policy"] = policy.to_state_dict()
        return policy

    def _algorithm_ownership_registry(self) -> AlgorithmOwnershipRegistry:
        return AlgorithmOwnershipRegistry(get_cellcompass_root() / "algorithm_ownership.json")

    def _algorithm_owner_session_id(self) -> str:
        session_id = (
            str(self.state.get("session_id") or "").strip()
            or str(self.state.get("thread_id") or "").strip()
            or str(os.getenv("CYTOBRIDGE_SESSION_ID") or "").strip()
        )
        if session_id:
            return session_id
        return f"process:{os.getpid()}"

    def _ensure_algorithm_write_owner(
        self,
        algorithm_id: str,
        *,
        action: str,
        path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        algo_id = str(algorithm_id or "").strip().lower()
        if not algo_id:
            return {}
        session_id = self._algorithm_owner_session_id()
        try:
            record = self._algorithm_ownership_registry().acquire(
                algorithm_id=algo_id,
                session_id=session_id,
                action=action,
            )
        except AlgorithmOwnershipError as exc:
            reason = str(exc)
            self._emit(
                "algorithm_ownership_blocked",
                {
                    "algorithm_id": algo_id,
                    "session_id": session_id,
                    "action": action,
                    "path": str(path) if path else "",
                    "reason": reason,
                },
            )
            raise ValueError(reason) from exc
        self.state["active_algorithm_ownership"] = {
            "algorithm_id": algo_id,
            "owner_session_id": str(record.get("owner_session_id") or ""),
            "last_action": str(record.get("last_action") or ""),
            "updated_at": str(record.get("updated_at") or ""),
            "registry_path": str(self._algorithm_ownership_registry().path),
        }
        self._emit(
            "algorithm_ownership_acquired",
            {
                "algorithm_id": algo_id,
                "session_id": session_id,
                "action": action,
                "path": str(path) if path else "",
                "ownership_path": str(self._algorithm_ownership_registry().path),
            },
        )
        return record

    def _proposal_store(self) -> Dict[str, Any]:
        proposals = self.state.get("algorithm_proposals")
        if not isinstance(proposals, dict):
            proposals = {}
            self.state["algorithm_proposals"] = proposals
        return proposals

    def _research_idea_store(self) -> Dict[str, Any]:
        ideas = self.state.get("research_ideas")
        if not isinstance(ideas, dict):
            ideas = {}
            self.state["research_ideas"] = ideas
        return ideas

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

    @classmethod
    def _registry_dir(cls, algorithm_id: str) -> Path:
        return cls._proposal_dir(algorithm_id) / "registry"

    @classmethod
    def _registry_index_path(cls, algorithm_id: str) -> Path:
        return cls._registry_dir(algorithm_id) / "algorithm_registry.json"

    @classmethod
    def _registry_proposals_dir(cls, algorithm_id: str) -> Path:
        return cls._registry_dir(algorithm_id) / "proposals"

    @classmethod
    def _registry_runs_dir(cls, algorithm_id: str) -> Path:
        return cls._registry_dir(algorithm_id) / "runs"

    @classmethod
    def _registry_snapshots_dir(cls, algorithm_id: str) -> Path:
        return cls._registry_dir(algorithm_id) / "workspace_snapshots"

    @classmethod
    def _registry_decisions_path(cls, algorithm_id: str) -> Path:
        return cls._registry_dir(algorithm_id) / "decisions.jsonl"

    @classmethod
    def _registry_obsolete_path(cls, algorithm_id: str) -> Path:
        return cls._registry_dir(algorithm_id) / "obsolete.jsonl"

    @staticmethod
    def _research_idea_dir(idea_id: str) -> Path:
        return research_idea_dir(idea_id)

    @staticmethod
    def _research_idea_root() -> Path:
        return research_idea_root()

    @classmethod
    def _research_idea_markdown_path(cls, idea_id: str) -> Path:
        return research_idea_markdown_path(idea_id)

    @classmethod
    def _research_idea_json_path(cls, idea_id: str) -> Path:
        return research_idea_json_path(idea_id)

    @classmethod
    def _research_idea_registry_dir(cls, idea_id: str) -> Path:
        return research_idea_registry_dir(idea_id)

    @classmethod
    def _research_idea_registry_index_path(cls, idea_id: str) -> Path:
        return research_idea_registry_index_path(idea_id)

    @classmethod
    def _research_idea_revisions_dir(cls, idea_id: str) -> Path:
        return research_idea_revisions_dir(idea_id)

    @classmethod
    def _research_idea_attempts_path(cls, idea_id: str) -> Path:
        return cls._research_idea_registry_dir(idea_id) / "attempts.jsonl"

    @classmethod
    def _research_idea_decisions_path(cls, idea_id: str) -> Path:
        return cls._research_idea_registry_dir(idea_id) / "decisions.jsonl"

    @classmethod
    def _research_idea_obsolete_path(cls, idea_id: str) -> Path:
        return cls._research_idea_registry_dir(idea_id) / "obsolete.jsonl"

    @staticmethod
    def _normalize_linked_algorithms(values: Any) -> List[str]:
        normalized: List[str] = []
        for item in (values or []):
            value = str(item or "").strip().lower()
            if value and value not in normalized:
                normalized.append(value)
        return normalized

    @staticmethod
    def _normalize_revision_summary(record: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "revision_id": str(record.get("current_revision_id") or "").strip(),
            "title": str(record.get("title") or "").strip(),
            "review_status": str(record.get("review_status") or "").strip(),
            "portfolio_status": str(record.get("portfolio_status") or "").strip(),
            "resolution_status": str(record.get("resolution_status") or "").strip(),
            "execution_status": str(record.get("execution_status") or "").strip(),
            "created_at": str(record.get("created_at") or "").strip(),
            "updated_at": str(record.get("updated_at") or "").strip(),
            "reviewed_at": str(record.get("reviewed_at") or "").strip(),
            "registry_path": str(record.get("current_revision_path") or "").strip(),
            "superseded_by_revision_id": str(record.get("superseded_by_revision_id") or "").strip(),
        }

    @staticmethod
    def _read_json(path: Path, default: Any) -> Any:
        if not path.exists():
            return default
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return default

    @staticmethod
    def _write_json(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(payload, ensure_ascii=False, indent=2)
        tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        tmp_path.write_text(text, encoding="utf-8")
        os.replace(tmp_path, path)

    @staticmethod
    def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    @staticmethod
    def _make_registry_id(prefix: str) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        safe_prefix = re.sub(r"[^a-z0-9_-]+", "-", str(prefix or "").strip().lower()) or "item"
        return f"{safe_prefix}_{timestamp}"

    @staticmethod
    def _metric_mean(values: Any) -> Optional[float]:
        if not isinstance(values, list) or not values:
            return None
        try:
            arr = [float(x) for x in values]
        except Exception:
            return None
        return sum(arr) / len(arr) if arr else None

    @classmethod
    def _algorithm_benchmarks_root(cls) -> Path:
        return get_cellcompass_root() / "algorithm_benchmarks"

    @classmethod
    def _algorithm_benchmark_datasets_dir(cls) -> Path:
        return cls._algorithm_benchmarks_root() / "datasets"

    @classmethod
    def _algorithm_benchmark_dataset_dir(cls, dataset_id: str) -> Path:
        return cls._algorithm_benchmark_datasets_dir() / cls._safe_benchmark_dataset_id(dataset_id)

    @classmethod
    def _algorithm_benchmark_registry_path(cls) -> Path:
        return cls._algorithm_benchmarks_root() / "registry.json"

    @classmethod
    def _algorithm_benchmark_simulation_generators_dir(cls) -> Path:
        return cls._algorithm_benchmarks_root() / "simulation_generators"

    @staticmethod
    def _safe_benchmark_dataset_id(dataset_id: str) -> str:
        safe = re.sub(r"[^a-z0-9_-]+", "-", str(dataset_id or "").strip().lower()).strip("-")
        if not safe:
            raise ValueError("dataset_id is required.")
        return safe

    @staticmethod
    def _safe_simulation_version(value: str) -> str:
        safe = re.sub(r"[^a-z0-9_.-]+", "-", str(value or "").strip().lower()).strip("-")
        if not safe:
            raise ValueError("simulation_version is required.")
        return safe

    @staticmethod
    def _file_sha256(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    @classmethod
    def _path_sha256(cls, path: Path) -> str:
        if path.is_file():
            return cls._file_sha256(path)
        if not path.is_dir():
            raise FileNotFoundError(f"Path does not exist: {path}")
        h = hashlib.sha256()
        for child in sorted(item for item in path.rglob("*") if item.is_file()):
            rel = child.relative_to(path).as_posix()
            h.update(rel.encode("utf-8"))
            h.update(b"\0")
            h.update(cls._file_sha256(child).encode("utf-8"))
            h.update(b"\0")
        return h.hexdigest()

    @staticmethod
    def _copy_path_to_dir(source: Path, target_dir: Path) -> str:
        target_dir.mkdir(parents=True, exist_ok=True)
        if source.is_file():
            target = target_dir / source.name
            shutil.copy2(source, target)
            return str(target)
        target = target_dir / "source"
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target)
        return str(target)

    def ensure_algorithm_benchmark_registry(self) -> Dict[str, Any]:
        root = self._algorithm_benchmarks_root()
        datasets_dir = self._algorithm_benchmark_datasets_dir()
        simulation_generators_dir = self._algorithm_benchmark_simulation_generators_dir()
        root.mkdir(parents=True, exist_ok=True)
        datasets_dir.mkdir(parents=True, exist_ok=True)
        simulation_generators_dir.mkdir(parents=True, exist_ok=True)
        registry_path = self._algorithm_benchmark_registry_path()
        registry = self._read_json(registry_path, {})
        if not isinstance(registry, dict):
            registry = {}
        now = self._now_iso()
        changed = False

        def _set_default(key: str, value: Any) -> None:
            nonlocal changed
            if key not in registry:
                registry[key] = value
                changed = True

        _set_default("schema_version", 1)
        _set_default("created_at", now)
        if str(registry.get("benchmark_root") or "") != str(root):
            registry["benchmark_root"] = str(root)
            changed = True
        if str(registry.get("datasets_dir") or "") != str(datasets_dir):
            registry["datasets_dir"] = str(datasets_dir)
            changed = True
        if str(registry.get("simulation_generators_dir") or "") != str(simulation_generators_dir):
            registry["simulation_generators_dir"] = str(simulation_generators_dir)
            changed = True
        _set_default("metric_scope", "builtin_global_metrics_only")
        _set_default(
            "metric_policy",
            {
                "global_metrics": ["W1", "TMV"],
                "custom_claim_metrics": "campaign_local_only",
                "notes": (
                    "Global benchmark cards store comparable builtin/reference metrics only. "
                    "Agent-proposed claim metrics belong in campaign registries."
                ),
            },
        )
        datasets = registry.get("datasets")
        if not isinstance(datasets, dict):
            datasets = {}
            registry["datasets"] = datasets
            changed = True
        simulation_generators = registry.get("simulation_generators")
        if not isinstance(simulation_generators, dict):
            simulation_generators = {}
            registry["simulation_generators"] = simulation_generators
            changed = True
        if changed or not registry_path.exists():
            registry["updated_at"] = now
            self._write_json(registry_path, registry)
        self.state["algorithm_benchmark_registry"] = {
            "registry_path": str(registry_path),
            "benchmark_root": str(root),
            "dataset_count": len(datasets),
            "simulation_generator_count": len(simulation_generators),
        }
        return registry

    def _benchmark_dataset_readme(self, card: Dict[str, Any]) -> str:
        profile = dict(card.get("data_profile") or {})
        paths = dict(card.get("paths") or {})
        provenance = dict(card.get("provenance") or {})
        preprocessing_script = dict(provenance.get("preprocessing_script") or {})
        cells_by_time = profile.get("cells_by_time") if isinstance(profile.get("cells_by_time"), dict) else {}
        time_lines = "\n".join(f"- {key}: {value}" for key, value in cells_by_time.items()) or "- Not available"
        tags = ", ".join(str(x) for x in (card.get("tags") or [])) or "none"
        stages = ", ".join(str(x) for x in (card.get("stage_relevance") or [])) or "unspecified"
        transforms = card.get("transformations") if isinstance(card.get("transformations"), list) else []
        transform_lines = "\n".join(f"- {item}" for item in transforms) or "- None"
        preprocessing_lines = "- Not recorded"
        if preprocessing_script:
            preprocessing_lines = (
                f"- original_path: `{preprocessing_script.get('original_path')}`\n"
                f"- copied_path: `{preprocessing_script.get('copied_path')}`\n"
                f"- sha256: `{preprocessing_script.get('sha256')}`"
            )
        simulation = card.get("simulation") if isinstance(card.get("simulation"), dict) else {}
        simulation_section = ""
        if simulation:
            simulation_section = (
                "\n## Stage 2 Simulation\n\n"
                f"- simulation_version: `{simulation.get('simulation_version')}`\n"
                f"- generator_sha256: `{simulation.get('generator_sha256')}`\n"
                f"- data_sha256: `{simulation.get('data_sha256')}`\n"
                f"- generator_copy_path: `{simulation.get('generator_copy_path')}`\n"
                f"- claim_metric_name: `{simulation.get('claim_metric_name') or ''}`\n"
                f"- notes: {simulation.get('notes') or 'None'}\n"
            )
        return (
            f"# {card.get('title') or card.get('dataset_id')}\n\n"
            f"- dataset_id: `{card.get('dataset_id')}`\n"
            f"- contract_status: `{card.get('contract_status')}`\n"
            f"- data_path: `{paths.get('data_path')}`\n"
            f"- source_path: `{card.get('source_path')}`\n"
            f"- tags: {tags}\n"
            f"- stage_relevance: {stages}\n\n"
            "## Description\n\n"
            f"{card.get('description') or 'No description provided.'}\n\n"
            "## Data Contract\n\n"
            f"- cells: {profile.get('n_obs')}\n"
            f"- features: {profile.get('n_vars')}\n"
            f"- has obs['time_point_processed']: {profile.get('has_time_point_processed')}\n"
            f"- has obsm['X_latent']: {profile.get('has_X_latent')}\n"
            f"- latent_dim: {profile.get('latent_dim')}\n\n"
            "## Cells By Time\n\n"
            f"{time_lines}\n\n"
            "## Preparation Notes\n\n"
            f"{transform_lines}\n\n"
            "## Preprocessing Script Provenance\n\n"
            f"{preprocessing_lines}\n\n"
            f"{simulation_section}"
            "## Benchmark Policy\n\n"
            "- Store comparable builtin/reference metrics here: W1, TMV, runtime, memory.\n"
            "- Keep algorithm-specific claim metrics in campaign registries, not in the global benchmark card.\n"
            "- Dataset-specific custom algorithm configs may be stored under `algorithm_configs/` and archived by campaigns.\n"
        )

    def register_algorithm_benchmark_dataset(
        self,
        dataset_id: str,
        source_path: str,
        *,
        title: str = "",
        description: str = "",
        tags: Optional[List[str]] = None,
        stage_relevance: Optional[List[str]] = None,
        overwrite: bool = False,
        preprocessing_script_path: str = "",
    ) -> Dict[str, Any]:
        """Register a prepared benchmark dataset and optionally archive its preprocessing script."""
        safe_id = self._safe_benchmark_dataset_id(dataset_id)
        source = Path(source_path).expanduser().resolve()
        if not source.exists():
            raise FileNotFoundError(f"Benchmark source_path does not exist: {source}")
        if source.suffix.lower() != ".h5ad":
            raise ValueError(f"Benchmark datasets must currently be .h5ad files, got: {source}")
        preprocessing_script: Optional[Path] = None
        if str(preprocessing_script_path or "").strip():
            preprocessing_script = Path(preprocessing_script_path).expanduser().resolve()
            if not preprocessing_script.exists():
                raise FileNotFoundError(f"preprocessing_script_path does not exist: {preprocessing_script}")
            if not preprocessing_script.is_file():
                raise ValueError(f"preprocessing_script_path must be a file, got: {preprocessing_script}")

        registry = self.ensure_algorithm_benchmark_registry()
        dataset_dir = self._algorithm_benchmark_dataset_dir(safe_id)
        dataset_json_path = dataset_dir / "dataset.json"
        existing = self._read_json(dataset_json_path, {}) if dataset_json_path.exists() else {}
        if dataset_json_path.exists() and not overwrite:
            return {
                "status": "exists",
                "dataset_id": safe_id,
                "dataset_path": str(dataset_json_path),
                "message": "Dataset already registered. Pass overwrite=true to refresh the data card.",
                "dataset": existing,
            }

        try:
            import anndata as ad  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on runtime env
            raise RuntimeError("Registering benchmark .h5ad files requires anndata.") from exc

        adata = ad.read_h5ad(source)
        obs_columns = [str(x) for x in list(adata.obs.columns)]
        obsm_keys = [str(x) for x in list(adata.obsm.keys())]
        cells_by_time: Dict[str, int] = {}
        time_points: List[str] = []
        if "time_point_processed" in adata.obs:
            counts = adata.obs["time_point_processed"].astype(str).value_counts().sort_index()
            cells_by_time = {str(k): int(v) for k, v in counts.items()}
            time_points = list(cells_by_time.keys())
        latent_shape = tuple(getattr(adata.obsm.get("X_latent"), "shape", ())) if "X_latent" in adata.obsm else ()
        data_profile = {
            "n_obs": int(adata.n_obs),
            "n_vars": int(adata.n_vars),
            "obs_columns": obs_columns,
            "obsm_keys": obsm_keys,
            "has_time_point_processed": "time_point_processed" in adata.obs,
            "has_X_latent": "X_latent" in adata.obsm,
            "latent_dim": int(latent_shape[1]) if len(latent_shape) == 2 else None,
            "time_points": time_points,
            "cells_by_time": cells_by_time,
        }
        missing_fields: List[str] = []
        if not data_profile["has_time_point_processed"]:
            missing_fields.append("adata.obs['time_point_processed']")
        if not data_profile["has_X_latent"]:
            missing_fields.append("adata.obsm['X_latent']")
        if missing_fields:
            payload = {
                "status": "invalid_contract",
                "dataset_id": safe_id,
                "source_path": str(source),
                "missing_fields": missing_fields,
                "data_profile": data_profile,
                "message": (
                    "Benchmark registration did not modify or copy this dataset because it does not satisfy "
                    "the CytoBridge training data contract. Read the preprocessing skill, create a prepared "
                    ".h5ad with obs['time_point_processed'] and obsm['X_latent'], then call this tool again."
                ),
                "next_steps": [
                    "Read ~/.cellcompass/skills/workflow/preprocessing-execution/SKILL.md.",
                    "Read ~/.cellcompass/skills/algorithm/benchmark-dataset-registration/SKILL.md.",
                    "Prepare and persist a new .h5ad that contains the required fields.",
                    "Retry register_algorithm_benchmark_dataset(...) with that prepared .h5ad path.",
                ],
            }
            self._emit("algorithm_benchmark_dataset_rejected", payload)
            return payload

        dataset_dir.mkdir(parents=True, exist_ok=True)
        for subdir in ("builtin_configs", "algorithm_configs", "builtin_baselines", "baseline_artifacts"):
            (dataset_dir / subdir).mkdir(parents=True, exist_ok=True)

        now = self._now_iso()
        data_path = dataset_dir / "data.h5ad"
        if source != data_path.resolve():
            shutil.copy2(source, data_path)
        preprocessing_record: Dict[str, Any] = {}
        preprocessing_scripts_dir = dataset_dir / "scripts" / "preprocessing"
        if preprocessing_script is not None:
            preprocessing_scripts_dir.mkdir(parents=True, exist_ok=True)
            safe_script_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", preprocessing_script.name).strip("._")
            if not safe_script_name:
                safe_script_name = "preprocessing.py"
            script_copy_path = preprocessing_scripts_dir / safe_script_name
            if script_copy_path.resolve() != preprocessing_script.resolve():
                shutil.copy2(preprocessing_script, script_copy_path)
            preprocessing_record = {
                "original_path": str(preprocessing_script),
                "copied_path": str(script_copy_path),
                "sha256": self._file_sha256(preprocessing_script),
                "registered_at": now,
            }
        transformations: List[str] = []
        contract_ready = True
        leaderboard_path = dataset_dir / "leaderboard.json"
        leaderboard = self._read_json(leaderboard_path, {})
        if not isinstance(leaderboard, dict):
            leaderboard = {}
        leaderboard.setdefault("schema_version", 1)
        leaderboard.setdefault("dataset_id", safe_id)
        leaderboard.setdefault("metric_scope", "builtin_global_metrics_only")
        leaderboard.setdefault("entries", [])
        leaderboard["updated_at"] = now
        self._write_json(leaderboard_path, leaderboard)

        card = {
            "schema_version": 1,
            "dataset_id": safe_id,
            "title": title.strip() or safe_id.replace("-", " ").replace("_", " ").title(),
            "description": description.strip(),
            "source_path": str(source),
            "source_sha256": self._file_sha256(source),
            "registered_at": str((existing or {}).get("registered_at") or now),
            "updated_at": now,
            "tags": [str(x).strip() for x in (tags or []) if str(x).strip()],
            "stage_relevance": [str(x).strip() for x in (stage_relevance or []) if str(x).strip()],
            "contract_status": "ready" if contract_ready else "needs_preprocessed_fields",
            "data_profile": data_profile,
            "transformations": transformations,
            "provenance": {
                "preprocessing_script": preprocessing_record,
            },
            "paths": {
                "dataset_dir": str(dataset_dir),
                "data_path": str(data_path),
                "readme_path": str(dataset_dir / "README.md"),
                "preprocessing_scripts_dir": str(preprocessing_scripts_dir),
                "preprocessing_script_path": str(preprocessing_record.get("copied_path") or ""),
                "builtin_configs_dir": str(dataset_dir / "builtin_configs"),
                "trial_configs_dir": str(dataset_dir / "trial_configs"),
                "algorithm_configs_dir": str(dataset_dir / "algorithm_configs"),
                "builtin_baselines_dir": str(dataset_dir / "builtin_baselines"),
                "baseline_artifacts_dir": str(dataset_dir / "baseline_artifacts"),
                "leaderboard_path": str(leaderboard_path),
            },
            "baseline_status": {
                "builtin_metrics_available": bool(leaderboard.get("entries")),
                "recommended_builtin_algorithms": [
                    "crufm",
                    "balanced_ot_cfm",
                    "sf2m",
                    "vgfm",
                    "wfrfm",
                    "dynamical_ot",
                    "unbalanced_ot",
                    "ruot",
                    "cyto_simulation",
                ],
                "notes": "Run and record builtin baselines before using this dataset as a stage gate reference.",
            },
            "usage": {
                "campaign_dataset_entry": {
                    "dataset_id": safe_id,
                    "adata_path": str(data_path),
                    "config_overrides": {},
                },
                "global_metrics": ["W1", "TMV"],
            },
        }
        self._write_json(dataset_json_path, card)
        (dataset_dir / "README.md").write_text(self._benchmark_dataset_readme(card), encoding="utf-8")

        dataset_summary = {
            "dataset_id": safe_id,
            "title": card["title"],
            "contract_status": card["contract_status"],
            "data_path": str(data_path),
            "dataset_json_path": str(dataset_json_path),
            "tags": card["tags"],
            "stage_relevance": card["stage_relevance"],
            "n_obs": data_profile["n_obs"],
            "n_vars": data_profile["n_vars"],
            "time_points": data_profile["time_points"],
            "builtin_metrics_available": bool(leaderboard.get("entries")),
            "preprocessing_script_path": str(preprocessing_record.get("copied_path") or ""),
            "updated_at": now,
        }
        registry.setdefault("datasets", {})[safe_id] = dataset_summary
        registry["updated_at"] = now
        self._write_json(self._algorithm_benchmark_registry_path(), registry)
        self.state["algorithm_benchmark_registry"] = {
            "registry_path": str(self._algorithm_benchmark_registry_path()),
            "benchmark_root": str(self._algorithm_benchmarks_root()),
            "dataset_count": len(registry.get("datasets") or {}),
        }
        self._emit(
            "algorithm_benchmark_dataset_registered",
            {
                "dataset_id": safe_id,
                "data_path": str(data_path),
                "preprocessing_script_path": str(preprocessing_record.get("copied_path") or ""),
                "contract_status": card["contract_status"],
                "n_obs": data_profile["n_obs"],
                "n_vars": data_profile["n_vars"],
            },
        )
        return {
            "status": "registered",
            "dataset_id": safe_id,
            "dataset": card,
            "registry_path": str(self._algorithm_benchmark_registry_path()),
        }

    def register_stage2_simulation_dataset(
        self,
        dataset_id: str,
        source_path: str,
        generator_path: str,
        *,
        algorithm_id: str = "",
        proposal_id: str = "",
        simulation_version: str = "",
        claim_metric_name: str = "",
        title: str = "",
        description: str = "",
        tags: Optional[List[str]] = None,
        overwrite: bool = False,
        notes: str = "",
    ) -> Dict[str, Any]:
        """Register a prepared Stage 2 simulation dataset with a frozen generator fingerprint."""
        safe_id = self._safe_benchmark_dataset_id(dataset_id)
        source = Path(source_path).expanduser().resolve()
        generator = Path(generator_path).expanduser().resolve()
        if not source.exists():
            raise FileNotFoundError(f"Simulation dataset source_path does not exist: {source}")
        if not generator.exists():
            raise FileNotFoundError(f"Simulation generator_path does not exist: {generator}")

        data_sha = self._file_sha256(source)
        generator_sha = self._path_sha256(generator)
        combined = hashlib.sha256(f"{safe_id}\0{data_sha}\0{generator_sha}".encode("utf-8")).hexdigest()
        version = (
            self._safe_simulation_version(simulation_version)
            if str(simulation_version or "").strip()
            else self._safe_simulation_version(f"{safe_id}_{combined[:12]}")
        )

        registry = self.ensure_algorithm_benchmark_registry()
        existing_generator = dict((registry.get("simulation_generators") or {}).get(version) or {})
        if existing_generator:
            existing_combined = str(existing_generator.get("combined_sha256") or "")
            if existing_combined and existing_combined != combined:
                raise ValueError(
                    f"simulation_version '{version}' is already bound to a different generator/data hash. "
                    "Use a new simulation_version when the generator or generated dataset changes."
                )

        merged_tags = []
        for item in list(tags or []) + ["simulation", "stage2", "claim"]:
            value = str(item or "").strip()
            if value and value not in merged_tags:
                merged_tags.append(value)
        registered = self.register_algorithm_benchmark_dataset(
            safe_id,
            str(source),
            title=title or safe_id.replace("-", " ").replace("_", " ").title(),
            description=description
            or (
                "Agent-designed Stage 2 claim-validation simulation dataset. "
                "Use only with matching simulation_version baselines."
            ),
            tags=merged_tags,
            stage_relevance=["stage2_claim_validation"],
            overwrite=overwrite,
        )
        if registered.get("status") == "invalid_contract":
            return registered
        if registered.get("status") == "exists" and not overwrite:
            card = dict(registered.get("dataset") or {})
            current_version = str((card.get("simulation") or {}).get("simulation_version") or "")
            if current_version == version:
                return {
                    "status": "exists",
                    "dataset_id": safe_id,
                    "simulation_version": version,
                    "dataset": card,
                    "message": "Stage 2 simulation dataset already registered with this simulation_version.",
                }
            return {
                "status": "exists",
                "dataset_id": safe_id,
                "simulation_version": version,
                "message": (
                    "Dataset already exists without this Stage 2 simulation binding. "
                    "Pass overwrite=true only if you intentionally want to refresh the dataset card; "
                    "use a new simulation_version if the generator or generated data changed."
                ),
                "dataset": card,
            }

        dataset_dir = self._algorithm_benchmark_dataset_dir(safe_id)
        dataset_json_path = dataset_dir / "dataset.json"
        card = self._read_json(dataset_json_path, {})
        if not isinstance(card, dict) or not card:
            raise RuntimeError(f"Registered dataset card was not written: {dataset_json_path}")

        generator_dir = dataset_dir / "simulation_generator"
        generator_copy_path = self._copy_path_to_dir(generator, generator_dir)
        generator_record_path = self._algorithm_benchmark_simulation_generators_dir() / f"{version}.json"
        now = self._now_iso()
        simulation_record = {
            "schema_version": 1,
            "simulation_version": version,
            "dataset_id": safe_id,
            "algorithm_id": str(algorithm_id or "").strip().lower(),
            "proposal_id": str(proposal_id or "").strip(),
            "claim_metric_name": str(claim_metric_name or "").strip(),
            "source_path": str(source),
            "data_sha256": data_sha,
            "generator_path": str(generator),
            "generator_type": "directory" if generator.is_dir() else "file",
            "generator_sha256": generator_sha,
            "combined_sha256": combined,
            "generator_copy_path": generator_copy_path,
            "record_path": str(generator_record_path),
            "notes": str(notes or "").strip(),
            "registered_at": now,
            "updated_at": now,
            "baseline_policy": (
                "Stage 2 external baseline metrics must carry the same simulation_version. "
                "If the generator or generated data changes, register a new simulation_version and rerun baselines."
            ),
        }
        card["simulation"] = simulation_record
        card["tags"] = merged_tags
        card["stage_relevance"] = ["stage2_claim_validation"]
        card["updated_at"] = now
        paths = card.setdefault("paths", {})
        paths["simulation_generator_dir"] = str(generator_dir)
        usage = card.setdefault("usage", {})
        campaign_entry = dict(usage.get("campaign_dataset_entry") or {})
        campaign_entry.update(
            {
                "dataset_id": safe_id,
                "adata_path": str((card.get("paths") or {}).get("data_path") or ""),
                "config_overrides": dict(campaign_entry.get("config_overrides") or {}),
                "simulation_version": version,
                "simulation_generator": {
                    "generator_sha256": generator_sha,
                    "combined_sha256": combined,
                    "generator_copy_path": generator_copy_path,
                },
            }
        )
        usage["campaign_dataset_entry"] = campaign_entry
        self._write_json(dataset_json_path, card)
        (dataset_dir / "README.md").write_text(self._benchmark_dataset_readme(card), encoding="utf-8")

        registry = self.ensure_algorithm_benchmark_registry()
        summary_entry = dict((registry.get("datasets") or {}).get(safe_id) or {})
        summary_entry.update(
            {
                "dataset_id": safe_id,
                "tags": merged_tags,
                "stage_relevance": ["stage2_claim_validation"],
                "simulation_version": version,
                "simulation_generator_sha256": generator_sha,
                "updated_at": now,
            }
        )
        registry.setdefault("datasets", {})[safe_id] = summary_entry
        registry.setdefault("simulation_generators", {})[version] = {
            "simulation_version": version,
            "dataset_id": safe_id,
            "algorithm_id": simulation_record["algorithm_id"],
            "proposal_id": simulation_record["proposal_id"],
            "claim_metric_name": simulation_record["claim_metric_name"],
            "generator_sha256": generator_sha,
            "data_sha256": data_sha,
            "combined_sha256": combined,
            "record_path": str(generator_record_path),
            "updated_at": now,
        }
        registry["updated_at"] = now
        self._write_json(generator_record_path, simulation_record)
        self._write_json(self._algorithm_benchmark_registry_path(), registry)
        self._emit(
            "stage2_simulation_dataset_registered",
            {
                "dataset_id": safe_id,
                "simulation_version": version,
                "generator_sha256": generator_sha,
                "data_sha256": data_sha,
            },
        )
        return {
            "status": "registered",
            "dataset_id": safe_id,
            "simulation_version": version,
            "simulation": simulation_record,
            "dataset": card,
            "registry_path": str(self._algorithm_benchmark_registry_path()),
            "generator_record_path": str(generator_record_path),
            "usage": {
                "claim_metric_spec_fields": {
                    "simulation_dataset_id": safe_id,
                    "simulation_version": version,
                    "name": str(claim_metric_name or "claim_metric").strip(),
                },
                "stage2_baseline_requirement": (
                    "Add the same simulation_version to stage_baselines.stage2_claim_validation "
                    "after rerunning builtin/reference baselines on this generated dataset."
                ),
            },
        }

    def generate_and_register_stage2_simulation_dataset(
        self,
        dataset_id: str,
        *,
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
    ) -> Dict[str, Any]:
        """Generate a DynBench/BoolODE scenario and register its train.h5ad for Stage 2."""
        safe_id = self._safe_benchmark_dataset_id(dataset_id)
        config_payload = scenario_config if isinstance(scenario_config, dict) else None
        config_path = Path(scenario_config_path).expanduser().resolve() if str(scenario_config_path or "").strip() else None
        if config_payload and config_path:
            raise ValueError("Pass either scenario_config or scenario_config_path, not both.")
        if not config_payload and not config_path:
            raise ValueError(
                "A Stage 2 simulation must be claim-directed. Pass scenario_config or scenario_config_path "
                "after checking that existing real benchmark datasets cannot test the claim."
            )
        if config_path and not config_path.exists():
            raise FileNotFoundError(f"scenario_config_path does not exist: {config_path}")

        repo_root = Path(__file__).resolve().parents[2]
        simulator_dir = repo_root / "benchmark" / "dynbench" / "simulators"
        scenario_factory = simulator_dir / "scenario_factory.py"
        if not scenario_factory.exists():
            raise FileNotFoundError(f"DynBench scenario factory was not found: {scenario_factory}")

        output_key = self._safe_simulation_version(output_id or simulation_version or safe_id)
        output_dir = (get_cellcompass_root() / "generated_simulations" / output_key).resolve()
        if output_dir.exists() and any(output_dir.iterdir()):
            if not overwrite:
                raise FileExistsError(
                    f"Generated simulation output already exists: {output_dir}. "
                    "Pass overwrite=true or choose a new output_id/simulation_version."
                )
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        frozen_config_path = output_dir / "scenario_config.json"
        if config_payload:
            self._write_json(frozen_config_path, config_payload)
            effective_config_path = frozen_config_path
        else:
            assert config_path is not None
            shutil.copy2(config_path, frozen_config_path)
            effective_config_path = frozen_config_path

        cmd = [
            sys.executable,
            str(scenario_factory),
            "--config",
            str(effective_config_path),
            "--output-dir",
            str(output_dir),
            "--quiet",
        ]
        completed = subprocess.run(
            cmd,
            cwd=str(repo_root),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=max(30, int(timeout_seconds or 300)),
            check=False,
        )
        run_log = {
            "command": cmd,
            "cwd": str(repo_root),
            "returncode": completed.returncode,
            "stdout": completed.stdout[-20000:],
            "stderr": completed.stderr[-20000:],
        }
        self._write_json(output_dir / "generation_run.json", run_log)
        if completed.returncode != 0:
            return {
                "status": "generation_failed",
                "dataset_id": safe_id,
                "output_dir": str(output_dir),
                "returncode": completed.returncode,
                "stdout_tail": completed.stdout[-4000:],
                "stderr_tail": completed.stderr[-4000:],
                "message": "DynBench scenario generation failed before benchmark registration.",
            }

        task_root = output_dir / "task_packages"
        candidates = sorted(task_root.glob("*/train.h5ad")) if task_root.exists() else []
        if task_package_name:
            requested = task_root / task_package_name / "train.h5ad"
            if not requested.exists():
                raise FileNotFoundError(
                    f"Requested task_package_name did not produce train.h5ad: {requested}. "
                    f"Available candidates: {[str(x.parent.name) for x in candidates]}"
                )
            train_h5ad = requested
        elif len(candidates) == 1:
            train_h5ad = candidates[0]
        elif not candidates:
            raise FileNotFoundError(f"No generated train.h5ad was found under {task_root}.")
        else:
            raise ValueError(
                "The scenario config generated multiple task packages. Pass task_package_name to choose one: "
                + ", ".join(str(path.parent.name) for path in candidates)
            )

        prepared_h5ad = output_dir / "prepared_train.h5ad"
        prepared_report = self._prepare_stage2_simulation_h5ad_for_benchmark(train_h5ad, prepared_h5ad)

        generator_bundle = output_dir / "frozen_generator"
        generator_bundle.mkdir(parents=True, exist_ok=True)
        for name in ("scenario_factory.py", "boolode_sim.py", "scenario_registry.py"):
            source = simulator_dir / name
            if source.exists():
                shutil.copy2(source, generator_bundle / name)
        shutil.copy2(effective_config_path, generator_bundle / "scenario_config.json")
        generator_manifest = {
            "schema_version": 1,
            "generator": "dynbench_boolode",
            "repo_root": str(repo_root),
            "scenario_factory": str(scenario_factory),
            "source_config_path": str(config_path) if config_path else "",
            "frozen_config_path": str(effective_config_path),
            "output_dir": str(output_dir),
            "task_package_name": train_h5ad.parent.name,
            "train_h5ad": str(train_h5ad),
            "prepared_train_h5ad": str(prepared_h5ad),
            "preparation_report": prepared_report,
            "created_at": self._now_iso(),
            "usage_policy": (
                "Use this simulation only when existing real benchmark datasets cannot test the approved claim. "
                "Ground truth artifacts are for evaluation and diagnostics, not training-time leakage."
            ),
        }
        self._write_json(generator_bundle / "generator_manifest.json", generator_manifest)

        register_payload = self.register_stage2_simulation_dataset(
            safe_id,
            str(prepared_h5ad),
            str(generator_bundle),
            algorithm_id=algorithm_id,
            proposal_id=proposal_id,
            simulation_version=simulation_version,
            claim_metric_name=claim_metric_name,
            title=title or safe_id.replace("-", " ").replace("_", " ").title(),
            description=description
            or (
                "Generated DynBench/BoolODE-style Stage 2 claim-validation simulation. "
                "Prefer real benchmark evidence when real data can support the claim metric."
            ),
            tags=list(tags or []) + ["dynbench", "boolode"],
            overwrite=overwrite,
            notes=notes,
        )
        dataset_config: Optional[Dict[str, Any]] = None
        if str(register_payload.get("status") or "") in {"registered", "exists"}:
            try:
                dataset_config = self.make_benchmark_dataset_config(
                    dataset_ids=[safe_id],
                    stage="stage2_claim_validation",
                )
            except Exception:
                dataset_config = None

        payload = {
            "status": register_payload.get("status", "registered"),
            "dataset_id": safe_id,
            "simulation_version": register_payload.get("simulation_version", ""),
            "generated_output_dir": str(output_dir),
            "task_package_dir": str(train_h5ad.parent),
            "train_h5ad": str(train_h5ad),
            "prepared_train_h5ad": str(prepared_h5ad),
            "preparation_report": prepared_report,
            "frozen_generator_dir": str(generator_bundle),
            "ground_truth_dirs": [str(path) for path in sorted(output_dir.rglob("ground_truth")) if path.is_dir()],
            "registration": register_payload,
            "dataset_config_overrides": dataset_config,
            "next_steps": [
                "Use dataset_config_overrides with run_campaign_trial(...) for Stage 2 claim validation.",
                "Run comparable builtin/reference baselines on the same simulation_version before claiming Stage 2 success.",
                "Prefer registered real benchmark data instead if it can evaluate the claim metric fairly.",
            ],
        }
        self._emit(
            "stage2_simulation_dataset_generated_and_registered",
            {
                "dataset_id": safe_id,
                "simulation_version": payload.get("simulation_version", ""),
                "output_dir": str(output_dir),
                "train_h5ad": str(train_h5ad),
            },
        )
        return payload

    @staticmethod
    def _prepare_stage2_simulation_h5ad_for_benchmark(source_path: Path, output_path: Path) -> Dict[str, Any]:
        """Add the CytoBridge benchmark fields expected by registration without changing X."""
        try:
            import anndata as ad  # type: ignore
            import numpy as np  # type: ignore
        except Exception as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError("Preparing generated simulation .h5ad requires anndata and numpy.") from exc

        adata = ad.read_h5ad(source_path)
        transformations: List[str] = []
        if "time_point_processed" not in adata.obs:
            if "time_point" in adata.obs:
                adata.obs["time_point_processed"] = adata.obs["time_point"].astype(str)
                transformations.append("obs['time_point_processed'] copied from obs['time_point'].")
            elif "time_bin" in adata.obs:
                adata.obs["time_point_processed"] = adata.obs["time_bin"].astype(str)
                transformations.append("obs['time_point_processed'] copied from obs['time_bin'].")
            elif "time" in adata.obs:
                adata.obs["time_point_processed"] = adata.obs["time"].astype(str)
                transformations.append("obs['time_point_processed'] copied from obs['time'].")
            else:
                raise ValueError(
                    "Generated simulation lacks obs['time_point_processed'] and no obs['time_point'], "
                    "obs['time_bin'], or obs['time'] fallback is available."
                )

        if "X_latent" not in adata.obsm:
            matrix = adata.X
            if hasattr(matrix, "toarray"):
                matrix = matrix.toarray()
            matrix = np.asarray(matrix, dtype=np.float32)
            if matrix.ndim != 2 or matrix.shape[0] != adata.n_obs:
                raise ValueError("Generated simulation X matrix is not a valid 2D cell-by-feature matrix.")
            if matrix.shape[1] <= 32:
                latent = matrix.copy()
                transformations.append("obsm['X_latent'] copied from X because feature dimension is <= 32.")
            else:
                from sklearn.decomposition import PCA  # type: ignore

                n_components = min(32, matrix.shape[0], matrix.shape[1])
                latent = PCA(n_components=n_components, random_state=0).fit_transform(matrix).astype(np.float32)
                transformations.append(
                    f"obsm['X_latent'] computed by PCA with n_components={n_components}."
                )
            adata.obsm["X_latent"] = latent

        prep = dict(adata.uns.get("cytobridge_stage2_simulation_preparation") or {})
        prep.update(
            {
                "schema_version": 1,
                "source_path": str(source_path),
                "output_path": str(output_path),
                "transformations": transformations,
                "contract_fields": ["obs['time_point_processed']", "obsm['X_latent']"],
            }
        )
        adata.uns["cytobridge_stage2_simulation_preparation"] = prep
        output_path.parent.mkdir(parents=True, exist_ok=True)
        adata.write_h5ad(output_path)
        return {
            "source_path": str(source_path),
            "prepared_path": str(output_path),
            "n_obs": int(adata.n_obs),
            "n_vars": int(adata.n_vars),
            "transformations": transformations,
            "has_time_point_processed": "time_point_processed" in adata.obs,
            "has_X_latent": "X_latent" in adata.obsm,
        }

    def list_algorithm_benchmarks(
        self,
        *,
        stage: str = "",
        tags: Optional[List[str]] = None,
        ready_only: bool = False,
    ) -> Dict[str, Any]:
        registry = self.ensure_algorithm_benchmark_registry()
        requested_stage = str(stage or "").strip()
        requested_tags = {str(x).strip() for x in (tags or []) if str(x).strip()}
        datasets: List[Dict[str, Any]] = []
        for dataset_id, summary in sorted((registry.get("datasets") or {}).items()):
            item = dict(summary or {})
            if ready_only and str(item.get("contract_status") or "") != "ready":
                continue
            stage_values = {str(x) for x in (item.get("stage_relevance") or [])}
            tag_values = {str(x) for x in (item.get("tags") or [])}
            if requested_stage and requested_stage not in stage_values:
                continue
            if requested_tags and not requested_tags.intersection(tag_values):
                continue
            datasets.append(item)
        return {
            "benchmark_root": str(self._algorithm_benchmarks_root()),
            "registry_path": str(self._algorithm_benchmark_registry_path()),
            "count": len(datasets),
            "datasets": datasets,
            "metric_policy": registry.get("metric_policy") or {},
        }

    def get_algorithm_benchmark_dataset(self, dataset_id: str) -> Dict[str, Any]:
        safe_id = self._safe_benchmark_dataset_id(dataset_id)
        dataset_path = self._algorithm_benchmark_dataset_dir(safe_id) / "dataset.json"
        card = self._read_json(dataset_path, {})
        if not isinstance(card, dict) or not card:
            raise FileNotFoundError(f"Benchmark dataset '{safe_id}' is not registered.")
        raw_readme_path = str((card.get("paths") or {}).get("readme_path") or "").strip()
        readme_path = Path(raw_readme_path) if raw_readme_path else None
        readme = readme_path.read_text(encoding="utf-8") if readme_path and readme_path.is_file() else ""
        payload = dict(card)
        payload["readme"] = readme
        payload["dataset_config_overrides_example"] = {
            "datasets": [dict((card.get("usage") or {}).get("campaign_dataset_entry") or {})],
            "config_overrides": {},
        }
        return payload

    def get_algorithm_benchmark_baselines(self, dataset_id: str) -> Dict[str, Any]:
        card = self.get_algorithm_benchmark_dataset(dataset_id)
        paths = dict(card.get("paths") or {})
        raw_baselines_dir = str(paths.get("builtin_baselines_dir") or "").strip()
        raw_leaderboard_path = str(paths.get("leaderboard_path") or "").strip()
        raw_artifacts_dir = str(paths.get("baseline_artifacts_dir") or "").strip()
        if not raw_artifacts_dir:
            raw_artifacts_dir = str(self._algorithm_benchmark_dataset_dir(str(card.get("dataset_id") or dataset_id)) / "baseline_artifacts")
        baselines_dir = Path(raw_baselines_dir) if raw_baselines_dir else Path()
        leaderboard_path = Path(raw_leaderboard_path) if raw_leaderboard_path else Path()
        baseline_records: List[Dict[str, Any]] = []
        if raw_baselines_dir and baselines_dir.exists():
            for path in sorted(baselines_dir.glob("*.json")):
                record = self._read_json(path, {})
                if isinstance(record, dict) and record:
                    enriched = {"path": str(path), **record}
                    identity_issue = self._baseline_identity_mismatch(
                        str(enriched.get("algorithm_name") or ""),
                        dict(enriched.get("metrics") or {}),
                    )
                    if identity_issue:
                        enriched["corrupt_identity"] = True
                        enriched["identity_warning"] = identity_issue
                    baseline_records.append(enriched)
        leaderboard = self._read_json(leaderboard_path, {}) if raw_leaderboard_path and leaderboard_path.exists() else {}
        return {
            "dataset_id": card.get("dataset_id"),
            "builtin_baselines_dir": str(baselines_dir),
            "baseline_artifacts_dir": raw_artifacts_dir,
            "leaderboard_path": str(leaderboard_path),
            "leaderboard": leaderboard if isinstance(leaderboard, dict) else {},
            "baseline_records": baseline_records,
            "baseline_count": len(baseline_records),
            "metric_scope": "builtin_global_metrics_only",
            "message": (
                "No builtin baselines are recorded yet; this dataset can be used for smoke/feasibility trials, "
                "but stage gates need reference W1/TMV baselines."
                if not baseline_records and not ((leaderboard or {}).get("entries") if isinstance(leaderboard, dict) else None)
                else ""
            ),
        }

    def get_algorithm_benchmark_baseline_config(
        self,
        dataset_id: str,
        algorithm_name: str,
        *,
        baseline_type: str = "builtin",
    ) -> Dict[str, Any]:
        """Return dataset-scoped baseline config overrides, if the benchmark provides them."""
        safe_id = self._safe_benchmark_dataset_id(dataset_id)
        algo_name = re.sub(r"[^a-z0-9_.-]+", "-", str(algorithm_name or "").strip().lower()).strip("-")
        baseline_kind = str(baseline_type or "builtin").strip().lower()
        if not safe_id or not algo_name:
            return {"status": "missing", "reason": "dataset_id_and_algorithm_name_required"}
        card = self.get_algorithm_benchmark_dataset(safe_id)
        paths = dict(card.get("paths") or {})
        raw_config_dir = str(paths.get("builtin_configs_dir") or "").strip()
        config_dir = Path(raw_config_dir) if raw_config_dir else self._algorithm_benchmark_dataset_dir(safe_id) / "builtin_configs"
        candidates: List[Path] = []
        stems = [f"{baseline_kind}_{algo_name}", algo_name]
        for stem in stems:
            for suffix in (".yaml", ".yml", ".json"):
                candidates.append(config_dir / f"{stem}{suffix}")
        for path in candidates:
            if not path.is_file():
                continue
            try:
                if path.suffix.lower() == ".json":
                    raw = self._read_json(path, {})
                else:
                    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
                    raw = loaded if isinstance(loaded, dict) else {}
            except Exception as exc:
                return {
                    "status": "invalid",
                    "dataset_id": safe_id,
                    "algorithm_name": algo_name,
                    "baseline_type": baseline_kind,
                    "config_path": str(path),
                    "error": str(exc),
                }
            if not isinstance(raw, dict) or not raw:
                continue
            declared_algorithm = str(raw.get("algorithm_name") or raw.get("algorithm_id") or "").strip()
            if declared_algorithm and self._normalize_baseline_identity(declared_algorithm) != algo_name:
                continue
            declared_type = str(raw.get("baseline_type") or "").strip().lower()
            if declared_type and declared_type != baseline_kind:
                continue
            if isinstance(raw.get("config_overrides"), dict):
                overrides = deepcopy(raw.get("config_overrides") or {})
            else:
                metadata_keys = {
                    "schema_version",
                    "dataset_id",
                    "algorithm_name",
                    "algorithm_id",
                    "baseline_type",
                    "config_name",
                    "description",
                    "notes",
                    "created_at",
                    "updated_at",
                }
                overrides = {
                    str(key): deepcopy(value)
                    for key, value in raw.items()
                    if str(key) not in metadata_keys
                }
            if not overrides:
                continue
            return {
                "status": "found",
                "dataset_id": safe_id,
                "algorithm_name": algo_name,
                "baseline_type": baseline_kind,
                "config_path": str(path),
                "config_overrides": overrides,
            }
        return {
            "status": "missing",
            "dataset_id": safe_id,
            "algorithm_name": algo_name,
            "baseline_type": baseline_kind,
            "searched_dir": str(config_dir),
        }

    def get_algorithm_benchmark_dataset_trial_config(
        self,
        dataset_id: str,
        *,
        stage: str = "",
    ) -> Dict[str, Any]:
        """Return dataset-scoped campaign trial config overrides, if available."""
        safe_id = self._safe_benchmark_dataset_id(dataset_id)
        if not safe_id:
            return {"status": "missing", "reason": "dataset_id_required"}
        dataset_path = self._algorithm_benchmark_dataset_dir(safe_id) / "dataset.json"
        if not dataset_path.is_file():
            return {
                "status": "missing",
                "dataset_id": safe_id,
                "reason": "benchmark_dataset_not_registered",
            }
        card = self.get_algorithm_benchmark_dataset(safe_id)
        paths = dict(card.get("paths") or {})
        raw_config_dir = str(
            paths.get("trial_configs_dir")
            or paths.get("campaign_configs_dir")
            or ""
        ).strip()
        config_dir = Path(raw_config_dir) if raw_config_dir else self._algorithm_benchmark_dataset_dir(safe_id) / "trial_configs"
        stage_value = str(stage or "").strip()
        stems: List[str] = []
        if stage_value:
            stems.extend([stage_value, f"campaign_{stage_value}", f"trial_{stage_value}"])
        stems.extend(["default", "campaign", "trial"])
        candidates: List[Path] = []
        for stem in stems:
            for suffix in (".yaml", ".yml", ".json"):
                candidates.append(config_dir / f"{stem}{suffix}")
        for path in candidates:
            if not path.is_file():
                continue
            try:
                if path.suffix.lower() == ".json":
                    raw = self._read_json(path, {})
                else:
                    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
                    raw = loaded if isinstance(loaded, dict) else {}
            except Exception as exc:
                return {
                    "status": "invalid",
                    "dataset_id": safe_id,
                    "stage": stage_value,
                    "config_path": str(path),
                    "error": str(exc),
                }
            if not isinstance(raw, dict) or not raw:
                continue
            declared_dataset = str(raw.get("dataset_id") or "").strip()
            if declared_dataset and self._safe_benchmark_dataset_id(declared_dataset) != safe_id:
                continue
            declared_stage = str(raw.get("stage") or raw.get("campaign_stage") or "").strip()
            if declared_stage and stage_value and declared_stage != stage_value:
                continue
            if isinstance(raw.get("config_overrides"), dict):
                overrides = deepcopy(raw.get("config_overrides") or {})
            else:
                metadata_keys = {
                    "schema_version",
                    "dataset_id",
                    "stage",
                    "campaign_stage",
                    "config_name",
                    "description",
                    "notes",
                    "created_at",
                    "updated_at",
                }
                overrides = {
                    str(key): deepcopy(value)
                    for key, value in raw.items()
                    if str(key) not in metadata_keys
                }
            if not overrides:
                continue
            return {
                "status": "found",
                "dataset_id": safe_id,
                "stage": stage_value,
                "config_path": str(path),
                "config_overrides": overrides,
            }
        return {
            "status": "missing",
            "dataset_id": safe_id,
            "stage": stage_value,
            "searched_dir": str(config_dir),
        }

    def list_algorithm_benchmark_baselines(self, dataset_id: str) -> Dict[str, Any]:
        payload = self.get_algorithm_benchmark_baselines(dataset_id)
        rows: List[Dict[str, Any]] = []
        for record in list(payload.get("baseline_records") or []):
            metrics_summary = dict((record or {}).get("metrics_summary") or {})
            metrics_payload = dict((record or {}).get("metrics") or {})
            artifacts = dict((record or {}).get("artifacts") or {})
            source_artifacts = dict((record or {}).get("source_artifacts") or {})
            trained_model_path = str(artifacts.get("trained_model_path") or "").strip()
            model_artifact_path = str(artifacts.get("model_artifact_path") or "").strip()
            ckpt_path = trained_model_path or model_artifact_path
            error_reason = str((record or {}).get("error") or metrics_payload.get("error") or "")
            status = str((record or {}).get("status") or ("failed" if error_reason else "completed"))
            row = {
                "algorithm_name": str((record or {}).get("algorithm_name") or ""),
                "baseline_type": str((record or {}).get("baseline_type") or ""),
                "status": status,
                "error": error_reason,
                "w1_mean": metrics_summary.get("w1_mean"),
                "tmv_mean": metrics_summary.get("tmv_mean"),
                "tmv_max": metrics_summary.get("tmv_max"),
                "runtime_sec": metrics_summary.get("runtime_sec"),
                "memory_peak_mb": metrics_summary.get("memory_peak_mb"),
                "run_id": str((record or {}).get("run_id") or ""),
                "record_path": str((record or {}).get("path") or ""),
                "artifact_dir": str(artifacts.get("artifact_dir") or ""),
                "trained_model_path": trained_model_path,
                "model_artifact_path": model_artifact_path,
                "model_state_path": str(artifacts.get("model_state_path") or ""),
                "ckpt_path": ckpt_path,
                "metrics_path": str(artifacts.get("metrics_path") or ""),
                "resolved_config_path": str(artifacts.get("resolved_config_path") or ""),
                "evaluation_trajectory_path": str(artifacts.get("evaluation_trajectory_path") or ""),
                "run_manifest_path": str(artifacts.get("run_manifest_path") or ""),
                "training_log_path": str(artifacts.get("training_log_path") or ""),
                "planner_context_path": str(artifacts.get("planner_context_path") or ""),
                "source_run_dir": str(source_artifacts.get("run_dir") or artifacts.get("run_dir") or ""),
                "corrupt_identity": bool((record or {}).get("corrupt_identity")),
                "identity_warning": str((record or {}).get("identity_warning") or ""),
            }
            for source in (record or {}, metrics_summary, metrics_payload):
                if not isinstance(source, dict):
                    continue
                for key, value in source.items():
                    if str(key).startswith("w1_backend") and key not in row:
                        row[str(key)] = deepcopy(value)
            row["artifacts_available"] = {
                "trained_model": bool(row["trained_model_path"] and Path(row["trained_model_path"]).exists()),
                "metrics": bool(row["metrics_path"] and Path(row["metrics_path"]).exists()),
                "resolved_config": bool(row["resolved_config_path"] and Path(row["resolved_config_path"]).exists()),
                "evaluation_trajectory": bool(row["evaluation_trajectory_path"] and Path(row["evaluation_trajectory_path"]).exists()),
                "training_log": bool(row["training_log_path"] and Path(row["training_log_path"]).exists()),
            }
            rows.append(row)
        rows.sort(key=lambda item: float(item.get("w1_mean") if item.get("w1_mean") is not None else 1e99))
        return {
            "dataset_id": payload.get("dataset_id"),
            "baseline_count": len(rows),
            "metric_scope": payload.get("metric_scope"),
            "baseline_artifacts_dir": payload.get("baseline_artifacts_dir"),
            "leaderboard_path": payload.get("leaderboard_path"),
            "baselines": rows,
            "message": payload.get("message") or "",
        }

    def _copy_or_link_baseline_artifact(self, source_path: str, destination_path: Path) -> str:
        source = Path(str(source_path or "")).expanduser()
        if not str(source_path or "").strip() or not source.is_file():
            return ""
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            if destination_path.exists():
                try:
                    if destination_path.stat().st_size == source.stat().st_size:
                        return str(destination_path)
                except Exception:
                    pass
                destination_path.unlink()
            try:
                os.link(source, destination_path)
            except OSError:
                shutil.copy2(source, destination_path)
            return str(destination_path)
        except Exception:
            return ""

    def _archive_baseline_artifacts(
        self,
        safe_id: str,
        baseline_kind: str,
        algo_name: str,
        artifacts: Dict[str, str],
    ) -> Dict[str, str]:
        dataset_dir = self._algorithm_benchmark_dataset_dir(safe_id)
        artifact_dir = dataset_dir / "baseline_artifacts" / f"{baseline_kind}_{algo_name}"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        archived: Dict[str, str] = {"artifact_dir": str(artifact_dir)}
        copy_specs = {
            "metrics_path": artifact_dir / "metrics.json",
            "trained_model_path": artifact_dir / "trained_model.h5ad",
            "model_artifact_path": artifact_dir / "model_artifact.json",
            "model_state_path": artifact_dir / "model_state.pt",
            "resolved_config_path": artifact_dir / "resolved_config.yaml",
            "evaluation_trajectory_path": artifact_dir / "evaluation_trajectory.npz",
        }
        for key, destination in copy_specs.items():
            copied = self._copy_or_link_baseline_artifact(str(artifacts.get(key) or ""), destination)
            if copied:
                archived[key] = copied
            elif str(artifacts.get(key) or "").strip():
                archived[key] = str(artifacts.get(key) or "").strip()

        run_dir = Path(str(artifacts.get("run_dir") or "")).expanduser()
        if run_dir.is_dir():
            extra_specs = {
                "run_manifest_path": run_dir / "run_manifest.json",
                "training_log_path": run_dir / "logs" / "training.log",
                "stdout_stderr_log_path": run_dir / "logs" / "stdout_stderr.log",
                "planner_context_path": run_dir / "logs" / "planner_context.json",
            }
            for key, source in extra_specs.items():
                copied = self._copy_or_link_baseline_artifact(str(source), artifact_dir / source.name)
                if copied:
                    archived[key] = copied
            archived["run_dir"] = str(run_dir)
        return archived

    @staticmethod
    def _normalize_baseline_identity(value: Any) -> str:
        raw = str(value or "").strip()
        if not raw:
            return ""
        if raw.lower().endswith((".yaml", ".yml", ".json")) or "/" in raw or "\\" in raw:
            raw = Path(raw).stem
        return re.sub(r"[^a-z0-9_.-]+", "-", raw.lower()).strip("-")

    @classmethod
    def _baseline_identity_mismatch(cls, algorithm_name: str, metrics: Dict[str, Any]) -> str:
        expected = cls._normalize_baseline_identity(algorithm_name)
        if not expected or not isinstance(metrics, dict):
            return ""
        explicit: Dict[str, str] = {}
        for key in ("algorithm_id", "base_config_name", "candidate_name", "base_config_ref"):
            normalized = cls._normalize_baseline_identity(metrics.get(key))
            if normalized:
                explicit[key] = normalized
        if not explicit:
            return ""
        mismatched = {key: value for key, value in explicit.items() if value != expected}
        if not mismatched:
            return ""
        return (
            f"baseline identity mismatch for '{expected}': "
            + ", ".join(f"{key}={value}" for key, value in sorted(mismatched.items()))
        )

    def record_algorithm_benchmark_baseline(
        self,
        dataset_id: str,
        algorithm_name: str,
        metrics: Dict[str, Any],
        *,
        baseline_type: str = "builtin",
        run_id: str = "",
        config_path: str = "",
        notes: str = "",
    ) -> Dict[str, Any]:
        safe_id = self._safe_benchmark_dataset_id(dataset_id)
        algo_name = re.sub(r"[^a-z0-9_.-]+", "-", str(algorithm_name or "").strip().lower()).strip("-")
        if not algo_name:
            raise ValueError("algorithm_name is required.")
        baseline_kind = str(baseline_type or "builtin").strip().lower()
        if baseline_kind not in {"builtin", "reference"}:
            raise ValueError("baseline_type must be 'builtin' or 'reference'; custom claim metrics stay in campaign registries.")
        if not isinstance(metrics, dict) or not metrics:
            raise ValueError("metrics must be a non-empty dict containing W1/TMV evidence.")
        identity_issue = self._baseline_identity_mismatch(algo_name, metrics)
        if baseline_kind == "builtin" and identity_issue:
            raise ValueError(identity_issue)

        card = self.get_algorithm_benchmark_dataset(safe_id)
        paths = dict(card.get("paths") or {})
        baselines_dir = Path(str(paths.get("builtin_baselines_dir") or ""))
        leaderboard_path = Path(str(paths.get("leaderboard_path") or ""))
        dataset_json_path = self._algorithm_benchmark_dataset_dir(safe_id) / "dataset.json"
        baselines_dir.mkdir(parents=True, exist_ok=True)

        def _number(value: Any) -> Optional[float]:
            try:
                if value is None:
                    return None
                return float(value)
            except Exception:
                return None

        w1_scores = metrics.get("w1_scores") if isinstance(metrics.get("w1_scores"), list) else []
        tmv_scores = metrics.get("tmv_scores") if isinstance(metrics.get("tmv_scores"), list) else []
        w1_values = [_number(x) for x in w1_scores]
        w1_values = [x for x in w1_values if x is not None]
        tmv_values = [_number(x) for x in tmv_scores]
        tmv_values = [x for x in tmv_values if x is not None]
        w1_mean = _number(metrics.get("w1_mean"))
        if w1_mean is None:
            w1_mean = self._metric_mean(w1_scores)
        tmv_mean = _number(metrics.get("tmv_mean"))
        if tmv_mean is None and tmv_values:
            tmv_mean = sum(tmv_values) / len(tmv_values)
        tmv_max = _number(metrics.get("tmv_max"))
        if tmv_max is None and tmv_values:
            tmv_max = max(tmv_values)
        summary = {
            "w1_mean": w1_mean,
            "tmv_mean": tmv_mean,
            "tmv_max": tmv_max,
            "runtime_sec": _number(metrics.get("runtime_sec")),
            "memory_peak_mb": _number(metrics.get("memory_peak_mb") or metrics.get("peak_memory_mb")),
        }
        for key, value in metrics.items():
            if str(key).startswith("w1_backend"):
                summary[str(key)] = deepcopy(value)
        error_reason = str(metrics.get("error") or "").strip()
        record_status = "failed" if error_reason else str(metrics.get("status") or "completed").strip().lower()
        if record_status not in {"completed", "failed"}:
            record_status = "completed"
        record_path = baselines_dir / f"{baseline_kind}_{algo_name}.json"

        def _has_numeric_evidence(payload: Dict[str, Any]) -> bool:
            if not isinstance(payload, dict):
                return False
            metric_sources = [
                payload,
                payload.get("metrics_summary") if isinstance(payload.get("metrics_summary"), dict) else {},
                payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {},
            ]
            for source in metric_sources:
                for key in ("w1_mean", "tmv_mean", "tmv_max"):
                    if _number(source.get(key)) is not None:
                        return True
            return False

        missing_numeric_evidence = record_status == "completed" and not _has_numeric_evidence(summary)
        if missing_numeric_evidence:
            existing = self._read_json(record_path, {}) if record_path.exists() else {}
            if (
                isinstance(existing, dict)
                and str(existing.get("status") or "").strip().lower() == "completed"
                and _has_numeric_evidence(existing)
            ):
                return {
                    "status": "skipped_incomplete_record_preserved",
                    "dataset_id": safe_id,
                    "algorithm_name": algo_name,
                    "baseline_type": baseline_kind,
                    "record_path": str(record_path),
                    "leaderboard_path": str(leaderboard_path),
                    "baseline_status": str(existing.get("status") or "completed"),
                    "error": (
                        "Refused to overwrite an existing completed baseline with a completed record "
                        "that lacks W1/TMV numeric evidence."
                    ),
                    "metrics_summary": dict(existing.get("metrics_summary") or {}),
                    "artifacts": dict(existing.get("artifacts") or {}),
                    "preserved_existing": True,
                }
            record_status = "failed"
            error_reason = "Completed baseline record lacked W1/TMV numeric evidence."
        artifacts = {
            "run_dir": str(metrics.get("run_dir") or "").strip(),
            "metrics_path": str(metrics.get("metrics_path") or "").strip(),
            "trained_model_path": str(metrics.get("trained_model_path") or "").strip(),
            "model_artifact_path": str(metrics.get("model_artifact_path") or "").strip(),
            "model_state_path": str(metrics.get("model_state_path") or "").strip(),
            "resolved_config_path": str(metrics.get("resolved_config_path") or config_path or "").strip(),
            "evaluation_trajectory_path": str(
                metrics.get("evaluation_trajectory_path")
                or metrics.get("trajectory_path")
                or ""
            ).strip(),
        }
        existing_artifacts = metrics.get("artifacts")
        if isinstance(existing_artifacts, dict):
            for key in ("run_dir", "metrics_path", "trained_model_path", "model_artifact_path", "model_state_path", "resolved_config_path", "evaluation_trajectory_path"):
                if not artifacts.get(key) and str(existing_artifacts.get(key) or "").strip():
                    artifacts[key] = str(existing_artifacts.get(key) or "").strip()
            if not artifacts.get("evaluation_trajectory_path") and str(existing_artifacts.get("trajectory_path") or "").strip():
                artifacts["evaluation_trajectory_path"] = str(existing_artifacts.get("trajectory_path") or "").strip()
        artifacts = {key: value for key, value in artifacts.items() if value}
        archived_artifacts = self._archive_baseline_artifacts(safe_id, baseline_kind, algo_name, artifacts)
        now = self._now_iso()
        record = {
            "schema_version": 1,
            "dataset_id": safe_id,
            "algorithm_name": algo_name,
            "baseline_type": baseline_kind,
            "status": record_status,
            "error": error_reason,
            "metric_scope": "builtin_global_metrics_only",
            "metrics_summary": summary,
            "w1_mean": summary.get("w1_mean"),
            "tmv_mean": summary.get("tmv_mean"),
            "tmv_max": summary.get("tmv_max"),
            "runtime_sec": summary.get("runtime_sec"),
            "memory_peak_mb": summary.get("memory_peak_mb"),
            "metrics": metrics,
            "artifacts": archived_artifacts or artifacts,
            "source_artifacts": artifacts,
            "run_id": str(run_id or "").strip(),
            "config_path": str(config_path or "").strip(),
            "notes": str(notes or "").strip(),
            "recorded_at": now,
            "updated_at": now,
        }
        record["record_path"] = str(record_path)
        for key, value in summary.items():
            if str(key).startswith("w1_backend"):
                record[str(key)] = deepcopy(value)
        self._write_json(record_path, record)

        leaderboard = self._read_json(leaderboard_path, {})
        if not isinstance(leaderboard, dict):
            leaderboard = {}
        entries = [
            dict(item)
            for item in (leaderboard.get("entries") or [])
            if not (
                str((item or {}).get("algorithm_name") or "") == algo_name
                and str((item or {}).get("baseline_type") or "") == baseline_kind
            )
        ]
        entries.append(
            {
                "dataset_id": safe_id,
                "algorithm_name": algo_name,
                "baseline_type": baseline_kind,
                "status": record_status,
                "error": error_reason,
                "metrics_summary": summary,
                "w1_mean": summary.get("w1_mean"),
                "tmv_mean": summary.get("tmv_mean"),
                "tmv_max": summary.get("tmv_max"),
                "runtime_sec": summary.get("runtime_sec"),
                "memory_peak_mb": summary.get("memory_peak_mb"),
                "artifacts": archived_artifacts or artifacts,
                "run_id": str(run_id or "").strip(),
                "record_path": str(record_path),
                "updated_at": now,
                **{
                    str(key): deepcopy(value)
                    for key, value in summary.items()
                    if str(key).startswith("w1_backend")
                },
            }
        )
        entries.sort(key=lambda item: (item.get("metrics_summary") or {}).get("w1_mean") is None)
        entries.sort(key=lambda item: float((item.get("metrics_summary") or {}).get("w1_mean") or 1e99))
        leaderboard.update(
            {
                "schema_version": 1,
                "dataset_id": safe_id,
                "metric_scope": "builtin_global_metrics_only",
                "entries": entries,
                "updated_at": now,
            }
        )
        self._write_json(leaderboard_path, leaderboard)

        card["baseline_status"] = {
            **dict(card.get("baseline_status") or {}),
            "builtin_metrics_available": True,
            "last_recorded_baseline": {
                "algorithm_name": algo_name,
                "baseline_type": baseline_kind,
                "record_path": str(record_path),
                "status": record_status,
                "updated_at": now,
            },
        }
        paths = card.setdefault("paths", {})
        paths.setdefault("baseline_artifacts_dir", str(self._algorithm_benchmark_dataset_dir(safe_id) / "baseline_artifacts"))
        card["updated_at"] = now
        self._write_json(dataset_json_path, card)

        registry = self.ensure_algorithm_benchmark_registry()
        summary_entry = dict((registry.get("datasets") or {}).get(safe_id) or {})
        summary_entry["builtin_metrics_available"] = True
        summary_entry["updated_at"] = now
        registry.setdefault("datasets", {})[safe_id] = summary_entry
        registry["updated_at"] = now
        self._write_json(self._algorithm_benchmark_registry_path(), registry)
        self._emit(
            "algorithm_benchmark_baseline_recorded",
            {
                "dataset_id": safe_id,
                "algorithm_name": algo_name,
                "baseline_type": baseline_kind,
                "record_path": str(record_path),
                "status": record_status,
                "error": error_reason,
                "w1_mean": w1_mean,
                "tmv_max": tmv_max,
            },
        )
        return {
            "status": "recorded",
            "dataset_id": safe_id,
            "algorithm_name": algo_name,
            "baseline_type": baseline_kind,
            "record_path": str(record_path),
            "leaderboard_path": str(leaderboard_path),
            "baseline_status": record_status,
            "error": error_reason,
            "metrics_summary": summary,
            "artifacts": archived_artifacts or artifacts,
        }

    def make_benchmark_dataset_config(
        self,
        dataset_ids: Optional[List[str]] = None,
        *,
        stage: str = "",
        common_config_overrides: Optional[Dict[str, Any]] = None,
        per_dataset_config_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        requested = [self._safe_benchmark_dataset_id(x) for x in (dataset_ids or []) if str(x or "").strip()]
        if not requested:
            listing = self.list_algorithm_benchmarks(stage=stage, ready_only=True)
            requested = [str(item.get("dataset_id") or "") for item in listing.get("datasets") or []]
        if not requested:
            raise ValueError("No benchmark dataset ids were provided and no ready datasets matched the requested stage.")
        per_dataset = per_dataset_config_overrides if isinstance(per_dataset_config_overrides, dict) else {}
        datasets: List[Dict[str, Any]] = []
        for dataset_id in requested:
            card = self.get_algorithm_benchmark_dataset(dataset_id)
            entry = dict((card.get("usage") or {}).get("campaign_dataset_entry") or {})
            if not entry:
                entry = {
                    "dataset_id": dataset_id,
                    "adata_path": str((card.get("paths") or {}).get("data_path") or ""),
                    "config_overrides": {},
                }
            simulation = card.get("simulation") if isinstance(card.get("simulation"), dict) else {}
            if simulation:
                entry["simulation_version"] = str(simulation.get("simulation_version") or "")
                entry["simulation_generator"] = {
                    "generator_sha256": str(simulation.get("generator_sha256") or ""),
                    "combined_sha256": str(simulation.get("combined_sha256") or ""),
                    "generator_copy_path": str(simulation.get("generator_copy_path") or ""),
                }
            dataset_trial_config = self.get_algorithm_benchmark_dataset_trial_config(dataset_id, stage=stage)
            base_overrides = (
                dataset_trial_config.get("config_overrides")
                if isinstance(dataset_trial_config, dict)
                and isinstance(dataset_trial_config.get("config_overrides"), dict)
                else {}
            )
            entry_overrides = (
                entry.get("config_overrides")
                if isinstance(entry.get("config_overrides"), dict)
                else {}
            )
            merged_overrides = self._merge_campaign_config_overrides(
                dict(entry_overrides or {}),
                dict(base_overrides or {}),
            )
            entry["config_overrides"] = self._merge_campaign_config_overrides(
                dict(merged_overrides or {}),
                dict(per_dataset.get(dataset_id) or {}),
            )
            if str(dataset_trial_config.get("status") or "") == "found":
                entry["dataset_trial_config_path"] = str(dataset_trial_config.get("config_path") or "")
            datasets.append(entry)
        payload = {
            "datasets": datasets,
            "config_overrides": dict(common_config_overrides or {}),
            "target_dataset_ids": requested,
        }
        return {
            "dataset_config_overrides": payload,
            "usage": "Pass dataset_config_overrides to run_campaign_trial(...).",
            "dataset_ids": requested,
            "stage": str(stage or ""),
        }

    @staticmethod
    def _campaign_target_dataset_ids(dataset_config_overrides: Optional[Dict[str, Any]]) -> List[str]:
        payload = dataset_config_overrides if isinstance(dataset_config_overrides, dict) else {}
        explicit = payload.get("target_dataset_ids")
        if isinstance(explicit, list):
            values = explicit
        elif isinstance(payload.get("dataset_ids"), list):
            values = payload.get("dataset_ids")
        elif isinstance(payload.get("stage_panel_dataset_ids"), list):
            values = payload.get("stage_panel_dataset_ids")
        else:
            datasets = payload.get("datasets")
            values = []
            if isinstance(datasets, list):
                for idx, item in enumerate(datasets):
                    entry = item if isinstance(item, dict) else {}
                    value = (
                        entry.get("dataset_id")
                        or entry.get("id")
                        or (f"dataset{idx + 1}" if entry else "")
                    )
                    values.append(value)
        normalized: List[str] = []
        for item in values:
            value = str(item or "").strip()
            if value and value not in normalized:
                normalized.append(value)
        return normalized

    @staticmethod
    def _merge_campaign_config_overrides(base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
        merged = deepcopy(base)
        for key, value in dict(extra or {}).items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = PlannerFileTools._merge_campaign_config_overrides(dict(merged.get(key) or {}), value)
            elif isinstance(value, list) and isinstance(merged.get(key), list):
                merged_list = deepcopy(merged.get(key) or [])
                for idx, item in enumerate(value):
                    if (
                        idx < len(merged_list)
                        and isinstance(merged_list[idx], dict)
                        and isinstance(item, dict)
                    ):
                        merged_list[idx] = PlannerFileTools._merge_campaign_config_overrides(
                            dict(merged_list[idx] or {}),
                            item,
                        )
                    elif idx < len(merged_list):
                        merged_list[idx] = deepcopy(item)
                    else:
                        merged_list.append(deepcopy(item))
                merged[key] = merged_list
            else:
                merged[key] = deepcopy(value)
        return merged

    def _normalize_campaign_dataset_payload(
        self,
        dataset_config_overrides: Optional[Dict[str, Any]],
        *,
        existing_payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if dataset_config_overrides is not None and not isinstance(dataset_config_overrides, dict):
            raise ValueError(
                "dataset_config_overrides must be a JSON object/dict, not a JSON string. "
                "Use make_benchmark_dataset_config(...) and pass the returned "
                "dataset_config_overrides object directly into set_campaign_stage_panel(...) "
                "or run_campaign_trial(...)."
            )
        raw = dict(dataset_config_overrides or {})
        if isinstance(raw.get("dataset_config_overrides"), dict):
            nested = dict(raw.get("dataset_config_overrides") or {})
            for key, value in raw.items():
                if key in {"dataset_config_overrides", "usage", "stage", "dataset_ids"}:
                    continue
                nested[key] = value
            raw = nested

        control_keys = {
            "datasets",
            "target_dataset_ids",
            "dataset_ids",
            "stage_panel_dataset_ids",
            "config_overrides",
            "common_config_overrides",
            "per_dataset_config_overrides",
        }

        payload = deepcopy(existing_payload or {})
        if any(key in raw for key in ("datasets", "target_dataset_ids", "dataset_ids", "stage_panel_dataset_ids")):
            for key in ("target_dataset_ids", "dataset_ids", "stage_panel_dataset_ids"):
                payload.pop(key, None)
        for key in ("datasets", "target_dataset_ids", "dataset_ids", "stage_panel_dataset_ids"):
            if key in raw:
                payload[key] = deepcopy(raw.get(key))

        existing_common = payload.get("config_overrides") if isinstance(payload.get("config_overrides"), dict) else {}
        alias_common = (
            raw.get("common_config_overrides")
            if isinstance(raw.get("common_config_overrides"), dict)
            else {}
        )
        direct_common = {key: value for key, value in raw.items() if key not in control_keys}
        canonical_common = raw.get("config_overrides") if isinstance(raw.get("config_overrides"), dict) else {}
        common = self._merge_campaign_config_overrides(dict(existing_common or {}), dict(alias_common or {}))
        common = self._merge_campaign_config_overrides(dict(common or {}), dict(direct_common or {}))
        common = self._merge_campaign_config_overrides(dict(common or {}), dict(canonical_common or {}))
        payload["config_overrides"] = common

        per_dataset = (
            raw.get("per_dataset_config_overrides")
            if isinstance(raw.get("per_dataset_config_overrides"), dict)
            else {}
        )
        datasets = payload.get("datasets")
        if isinstance(datasets, list):
            normalized_datasets: List[Dict[str, Any]] = []
            for idx, item in enumerate(datasets):
                entry = dict(item or {})
                dataset_id = str(entry.get("dataset_id") or entry.get("id") or f"dataset{idx + 1}").strip()
                entry_overrides = entry.get("config_overrides") if isinstance(entry.get("config_overrides"), dict) else {}
                dataset_delta = per_dataset.get(dataset_id) if isinstance(per_dataset.get(dataset_id), dict) else {}
                entry["config_overrides"] = self._merge_campaign_config_overrides(
                    dict(entry_overrides or {}),
                    dict(dataset_delta or {}),
                )
                normalized_datasets.append(entry)
            payload["datasets"] = normalized_datasets
        elif per_dataset:
            payload["datasets"] = [
                {"dataset_id": str(dataset_id), "config_overrides": dict(overrides or {})}
                for dataset_id, overrides in per_dataset.items()
                if str(dataset_id or "").strip() and isinstance(overrides, dict)
            ]

        target_ids = self._campaign_target_dataset_ids(payload)
        if target_ids:
            payload["target_dataset_ids"] = target_ids
        return payload

    def _apply_dataset_trial_configs_to_payload(
        self,
        payload: Dict[str, Any],
        *,
        stage: str,
    ) -> Dict[str, Any]:
        updated = deepcopy(payload or {})
        datasets = updated.get("datasets")
        if not isinstance(datasets, list):
            return updated
        normalized: List[Dict[str, Any]] = []
        for idx, item in enumerate(datasets):
            entry = dict(item or {})
            dataset_id = str(entry.get("dataset_id") or entry.get("id") or f"dataset{idx + 1}").strip()
            if not dataset_id:
                normalized.append(entry)
                continue
            trial_config = self.get_algorithm_benchmark_dataset_trial_config(dataset_id, stage=stage)
            trial_overrides = (
                trial_config.get("config_overrides")
                if isinstance(trial_config, dict)
                and isinstance(trial_config.get("config_overrides"), dict)
                else {}
            )
            if trial_overrides:
                explicit_overrides = (
                    entry.get("config_overrides")
                    if isinstance(entry.get("config_overrides"), dict)
                    else {}
                )
                entry["config_overrides"] = self._merge_campaign_config_overrides(
                    dict(trial_overrides or {}),
                    dict(explicit_overrides or {}),
                )
                entry.setdefault("dataset_trial_config_path", str(trial_config.get("config_path") or ""))
            normalized.append(entry)
        updated["datasets"] = normalized
        return updated

    def _stage_panel_from_payload(
        self,
        dataset_config_overrides: Dict[str, Any],
        *,
        source: str,
    ) -> Dict[str, Any]:
        target_ids = self._campaign_target_dataset_ids(dataset_config_overrides)
        if not target_ids:
            raise ValueError("Stage panel requires dataset_config_overrides.datasets or target_dataset_ids.")
        payload = self._stage_panel_payload_for_storage(dataset_config_overrides)
        payload["target_dataset_ids"] = target_ids
        return {
            "target_dataset_ids": target_ids,
            "dataset_config_overrides": payload,
            "config_override_persistence": "trial_only",
            "source": source,
            "frozen_at": self._now_iso(),
        }

    @staticmethod
    def _stage_panel_payload_for_storage(dataset_config_overrides: Dict[str, Any]) -> Dict[str, Any]:
        """Return the dataset-panel portion that should persist across trials.

        Stage panels freeze the comparable data panel: dataset ids, data paths,
        simulation/version metadata, and benchmark trial config pointers. They
        must not freeze ordinary tuning overrides. Persisting config_overrides
        makes later edits to the live algorithm config.yaml look ineffective
        because stale panel overrides are merged on top of the workspace config.
        """

        payload = deepcopy(dataset_config_overrides or {})
        for key in ("config_overrides", "common_config_overrides", "per_dataset_config_overrides"):
            payload.pop(key, None)
        datasets = payload.get("datasets")
        if isinstance(datasets, list):
            cleaned: List[Dict[str, Any]] = []
            for item in datasets:
                if not isinstance(item, dict):
                    cleaned.append(item)
                    continue
                entry = deepcopy(item)
                entry.pop("config_overrides", None)
                trial_config_path = str(entry.get("dataset_trial_config_path") or "").strip()
                if trial_config_path:
                    trial_config_file = Path(trial_config_path).expanduser()
                    if trial_config_file.is_file():
                        try:
                            if trial_config_file.suffix.lower() == ".json":
                                loaded = json.loads(trial_config_file.read_text(encoding="utf-8"))
                            else:
                                loaded = yaml.safe_load(trial_config_file.read_text(encoding="utf-8"))
                            trial_config_overrides = (
                                loaded.get("config_overrides")
                                if isinstance(loaded, dict) and isinstance(loaded.get("config_overrides"), dict)
                                else {}
                            )
                            if trial_config_overrides:
                                entry["config_overrides"] = deepcopy(trial_config_overrides)
                        except Exception:
                            # A malformed benchmark trial config will still be surfaced when
                            # the live payload is resolved. Do not persist agent tuning
                            # overrides just because the sidecar config could not be read here.
                            pass
                cleaned.append(entry)
            payload["datasets"] = cleaned
        return payload

    @staticmethod
    def _stage_panel_simulation_versions(stage_panel: Dict[str, Any]) -> List[str]:
        payload = stage_panel.get("dataset_config_overrides") if isinstance(stage_panel.get("dataset_config_overrides"), dict) else {}
        versions: List[str] = []
        for entry in list(payload.get("datasets") or []):
            if not isinstance(entry, dict):
                continue
            value = str(entry.get("simulation_version") or "").strip()
            if value and value not in versions:
                versions.append(value)
        return versions

    @staticmethod
    def _missing_required_panel_ids(panel_ids: List[str], required_ids: List[str]) -> List[str]:
        present = {str(item or "").strip() for item in panel_ids if str(item or "").strip()}
        missing: List[str] = []
        for item in required_ids:
            value = str(item or "").strip()
            if value and value not in present and value not in missing:
                missing.append(value)
        return missing

    def _required_stage_panel_ids(self, stage: str, stage_state: Dict[str, Any]) -> List[str]:
        if str(stage or "") != "stage3_tuning":
            return []
        raw = stage_state.get("required_dataset_ids")
        if not isinstance(raw, list):
            raw = stage_state.get("claim_guardrail_dataset_ids")
        return [
            str(item or "").strip()
            for item in list(raw or [])
            if str(item or "").strip()
        ]

    def _resolve_campaign_stage_panel_payload(
        self,
        campaign: Dict[str, Any],
        stage_state: Dict[str, Any],
        stage: str,
        dataset_config_overrides: Optional[Dict[str, Any]],
        *,
        freeze_if_new: bool,
    ) -> Dict[str, Any]:
        existing_panel = stage_state.get("stage_panel") if isinstance(stage_state.get("stage_panel"), dict) else {}
        existing_payload = (
            dict(existing_panel.get("dataset_config_overrides") or {})
            if existing_panel
            else {}
        )
        provided = self._normalize_campaign_dataset_payload(
            dataset_config_overrides,
            existing_payload=existing_payload if existing_panel else None,
        )
        provided = self._apply_dataset_trial_configs_to_payload(provided, stage=stage)
        if not provided and existing_panel:
            return existing_payload
        target_ids = self._campaign_target_dataset_ids(provided)
        if existing_panel:
            panel_ids = [
                str(item or "").strip()
                for item in list(existing_panel.get("target_dataset_ids") or [])
                if str(item or "").strip()
            ]
            if target_ids and target_ids != panel_ids:
                raise ValueError(
                    f"Stage '{stage}' panel is already frozen to {panel_ids}; "
                    f"received {target_ids}. Use switch_campaign_stage_panel(...) if you intentionally "
                    "want to change the current stage data panel without resetting the stage trial count."
                )
            if target_ids:
                provided["target_dataset_ids"] = target_ids
            return provided or existing_payload
        if target_ids:
            provided["target_dataset_ids"] = target_ids
            missing_required = self._missing_required_panel_ids(
                target_ids,
                self._required_stage_panel_ids(stage, stage_state),
            )
            if missing_required:
                raise ValueError(
                    f"Stage '{stage}' panel must include inherited Stage 2 guardrail dataset ids: {missing_required}."
                )
            if freeze_if_new:
                stage_state["stage_panel"] = self._stage_panel_from_payload(provided, source="first_trial")
                campaign["updated_at"] = self._now_iso()
        return provided

    def set_campaign_stage_panel(
        self,
        campaign_id: str,
        *,
        stage: str = "",
        dataset_config_overrides: Optional[Dict[str, Any]] = None,
        overwrite: bool = False,
    ) -> Dict[str, Any]:
        campaign = self._load_campaign(campaign_id)
        self._ensure_campaign_mutable_status(campaign, action="set stage panel for")
        self._ensure_algorithm_id_mutable(
            str(campaign.get("algorithm_id") or "").strip().lower(),
            action="set_campaign_stage_panel",
        )
        target_stage = str(stage or campaign.get("current_stage") or CAMPAIGN_STAGE_ORDER[0]).strip()
        stages = campaign.setdefault("stages", {})
        if target_stage not in stages:
            raise ValueError(f"Unknown campaign stage: {target_stage}")
        stage_state = stages.setdefault(target_stage, {})
        if int(stage_state.get("trial_count") or 0) > 0:
            raise ValueError(
                f"Stage '{target_stage}' already has completed trials; use switch_campaign_stage_panel(...) "
                "if you intentionally want to change the data panel mid-stage without resetting trial count."
            )
        payload = self._normalize_campaign_dataset_payload(dataset_config_overrides)
        payload = self._apply_dataset_trial_configs_to_payload(payload, stage=target_stage)
        panel = self._stage_panel_from_payload(payload, source="manual")
        missing_required = self._missing_required_panel_ids(
            list(panel.get("target_dataset_ids") or []),
            self._required_stage_panel_ids(target_stage, stage_state),
        )
        if missing_required:
            raise ValueError(
                f"Stage '{target_stage}' panel must include inherited Stage 2 guardrail dataset ids: {missing_required}."
            )
        existing = stage_state.get("stage_panel") if isinstance(stage_state.get("stage_panel"), dict) else {}
        current_trial_id = str(campaign.get("current_trial_id") or "").strip()
        if existing and current_trial_id and target_stage == str(campaign.get("current_stage") or CAMPAIGN_STAGE_ORDER[0]).strip():
            existing_ids = list(existing.get("target_dataset_ids") or [])
            if overwrite or existing_ids != list(panel.get("target_dataset_ids") or []):
                raise ValueError(
                    f"Stage '{target_stage}' panel is already frozen to {existing_ids} while trial "
                    f"'{current_trial_id}' is open; finish the open trial before changing panel config, "
                    "and start a new campaign/stage if dataset ids need to change."
                )
        if existing and not overwrite:
            existing_ids = list(existing.get("target_dataset_ids") or [])
            if existing_ids == panel["target_dataset_ids"]:
                return {
                    "status": "exists",
                    "campaign_id": str(campaign.get("campaign_id") or ""),
                    "stage": target_stage,
                    "stage_panel": existing,
                    "message": "Stage panel is already frozen to these dataset ids.",
                }
            raise ValueError(
                f"Stage '{target_stage}' panel is already frozen to {existing_ids}; pass overwrite=true only before any trial, "
                "or use switch_campaign_stage_panel(...) after trials already exist."
            )
        stage_state["stage_panel"] = panel
        stages[target_stage] = stage_state
        campaign["stages"] = stages
        campaign["updated_at"] = self._now_iso()
        self._save_campaign(campaign)
        payload_out = {
            "status": "registered",
            "campaign_id": str(campaign.get("campaign_id") or ""),
            "algorithm_id": str(campaign.get("algorithm_id") or ""),
            "stage": target_stage,
            "target_dataset_ids": list(panel.get("target_dataset_ids") or []),
            "stage_panel": panel,
        }
        self._emit("algorithm_campaign_stage_panel_set", payload_out)
        return payload_out

    def switch_campaign_stage_panel(
        self,
        campaign_id: str,
        *,
        stage: str = "",
        dataset_config_overrides: Optional[Dict[str, Any]] = None,
        reason: str = "",
    ) -> Dict[str, Any]:
        campaign = self._load_campaign(campaign_id)
        self._ensure_campaign_mutable_status(campaign, action="switch stage panel for")
        target_stage = str(stage or campaign.get("current_stage") or CAMPAIGN_STAGE_ORDER[0]).strip()
        stages = campaign.setdefault("stages", {})
        if target_stage not in stages:
            raise ValueError(f"Unknown campaign stage: {target_stage}")
        current_stage = str(campaign.get("current_stage") or CAMPAIGN_STAGE_ORDER[0]).strip()
        current_trial_id = str(campaign.get("current_trial_id") or "").strip()
        if current_trial_id and target_stage == current_stage:
            raise ValueError(
                f"Cannot switch stage panel while current trial '{current_trial_id}' is open. "
                "Finish it or call abort_current_campaign_trial(...) for stale interrupted state first."
            )

        stage_state = stages.setdefault(target_stage, {})
        payload = self._normalize_campaign_dataset_payload(dataset_config_overrides)
        payload = self._apply_dataset_trial_configs_to_payload(payload, stage=target_stage)
        panel = self._stage_panel_from_payload(payload, source="manual_switch")
        new_ids = list(panel.get("target_dataset_ids") or [])
        missing_required = self._missing_required_panel_ids(
            new_ids,
            self._required_stage_panel_ids(target_stage, stage_state),
        )
        if missing_required:
            raise ValueError(
                f"Stage '{target_stage}' panel must include inherited Stage 2 guardrail dataset ids: {missing_required}."
            )

        existing = stage_state.get("stage_panel") if isinstance(stage_state.get("stage_panel"), dict) else {}
        old_ids = [
            str(item or "").strip()
            for item in list(existing.get("target_dataset_ids") or [])
            if str(item or "").strip()
        ]
        if old_ids == new_ids:
            return {
                "status": "unchanged",
                "campaign_id": str(campaign.get("campaign_id") or ""),
                "algorithm_id": str(campaign.get("algorithm_id") or ""),
                "stage": target_stage,
                "target_dataset_ids": new_ids,
                "message": "Stage panel already uses these dataset ids; no gate/best reset was applied.",
                "trial_count_preserved": int(stage_state.get("trial_count") or 0),
            }

        now = self._now_iso()
        reason_text = str(reason or "").strip() or "agent intentionally switched stage data panel"
        previous_best = {
            "active_best_trial_id": str(stage_state.get("active_best_trial_id") or ""),
            "active_best_snapshot_id": str(stage_state.get("active_best_snapshot_id") or ""),
            "active_best_commit": str(stage_state.get("active_best_commit") or ""),
            "active_best_run_ids": list(stage_state.get("active_best_run_ids") or []),
            "active_best_metrics_summary": dict(stage_state.get("active_best_metrics_summary") or {}),
            "external_baseline_metrics": dict(stage_state.get("external_baseline_metrics") or {}),
            "gate_ready": bool(stage_state.get("gate_ready", False)),
            "last_gate_check": dict(stage_state.get("last_gate_check") or {}),
        }
        history = list(stage_state.get("panel_switch_history") or [])
        history.append(
            {
                "switched_at": now,
                "reason": reason_text,
                "from_target_dataset_ids": old_ids,
                "to_target_dataset_ids": new_ids,
                "trial_count_preserved": int(stage_state.get("trial_count") or 0),
                "previous_active_best_trial_id": previous_best["active_best_trial_id"],
                "previous_active_best_snapshot_id": previous_best["active_best_snapshot_id"],
            }
        )
        stage_state["panel_switch_history"] = history[-20:]
        stage_state["previous_panel_active_best"] = previous_best
        stage_state["stage_panel"] = {
            **panel,
            "previous_target_dataset_ids": old_ids,
            "switched_at": now,
            "reason": reason_text,
        }
        # Trial counters and promote/reject counts are intentionally preserved as stage budget.
        for key in (
            "active_best_trial_id",
            "active_best_commit",
            "active_best_snapshot_id",
            "active_best_metrics_summary",
            "gate_passed_at",
            "gate_passed_trial_id",
            "stage_gate_evidence",
            "agent_registered_baselines",
        ):
            stage_state.pop(key, None)
        stage_state["active_best_run_ids"] = []
        stage_state["external_baseline_metrics"] = {}
        stage_state["gate_ready"] = False
        stage_state["last_gate_check"] = {
            "ok": False,
            "checked_at": now,
            "advance_requested": False,
            "next_stage": "",
            "blockers": [
                "stage panel was switched; run_campaign_trial(...) on the new panel and refresh_campaign_stage_baselines(...) before gate check"
            ],
        }
        stage_state["status"] = "active" if target_stage == current_stage else str(stage_state.get("status") or "pending")
        stages[target_stage] = stage_state
        campaign["stages"] = stages
        if target_stage == "stage2_claim_validation":
            campaign["algorithm_validated"] = False
            campaign["validated_at"] = ""
        self._save_campaign(campaign)
        payload_out = {
            "status": "switched",
            "campaign_id": str(campaign.get("campaign_id") or ""),
            "algorithm_id": str(campaign.get("algorithm_id") or ""),
            "stage": target_stage,
            "from_target_dataset_ids": old_ids,
            "target_dataset_ids": new_ids,
            "trial_count_preserved": int(stage_state.get("trial_count") or 0),
            "promote_count_preserved": int(stage_state.get("promote_count") or 0),
            "reject_count_preserved": int(stage_state.get("reject_count") or 0),
            "active_best_cleared": True,
            "gate_evidence_invalidated": True,
            "external_baseline_metrics_cleared": True,
            "next_required_action": (
                "Run run_campaign_trial(...) on the new panel. Then refresh_campaign_stage_baselines(...) "
                "so gate criteria use the new target_dataset_ids."
            ),
            "reason": reason_text,
            "stage_panel": stage_state["stage_panel"],
        }
        self._emit("algorithm_campaign_stage_panel_switched", payload_out)
        return payload_out

    def resolve_campaign_trial_dataset_payload(
        self,
        campaign_id: str,
        dataset_config_overrides: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        campaign = self._load_campaign(campaign_id)
        stage = str(campaign.get("current_stage") or CAMPAIGN_STAGE_ORDER[0])
        stages = campaign.setdefault("stages", {})
        stage_state = stages.setdefault(stage, {})
        payload = self._resolve_campaign_stage_panel_payload(
            campaign,
            stage_state,
            stage,
            dataset_config_overrides,
            freeze_if_new=True,
        )
        if not self._campaign_target_dataset_ids(payload):
            raise ValueError(
                f"Campaign stage '{stage}' has no frozen dataset panel and no valid "
                "dataset_config_overrides payload. First call make_benchmark_dataset_config(...) "
                "for the intended benchmark dataset ids and pass its dataset_config_overrides "
                "to set_campaign_stage_panel(...), or pass that object directly to "
                "run_campaign_trial(...)."
            )
        stages[stage] = stage_state
        campaign["stages"] = stages
        self._save_campaign(campaign)
        return payload

    def set_campaign_stage_baseline_metrics(
        self,
        campaign_id: str,
        *,
        stage: str = "",
        baseline_metrics: Optional[Dict[str, Any]] = None,
        source: str = "",
        baseline_summaries: Optional[List[Dict[str, Any]]] = None,
        run_missing: bool = True,
    ) -> Dict[str, Any]:
        campaign = self._load_campaign(campaign_id)
        self._ensure_campaign_mutable_status(campaign, action="set baseline metrics for")
        target_stage = str(stage or campaign.get("current_stage") or CAMPAIGN_STAGE_ORDER[0]).strip()
        stages = campaign.setdefault("stages", {})
        if target_stage not in stages:
            raise ValueError(f"Unknown campaign stage: {target_stage}")
        metrics = dict(baseline_metrics or {})
        if not metrics:
            raise ValueError("baseline_metrics must be a non-empty dict.")
        stage_state = stages.setdefault(target_stage, {})
        registered_baselines = list(stage_state.get("agent_registered_baselines") or [])
        if registered_baselines and not metrics.get("agent_registered_baselines"):
            metrics["agent_registered_baselines"] = deepcopy(registered_baselines)
        stage_state["external_baseline_metrics"] = metrics
        stage_state["external_baseline_source"] = {
            "source": str(source or "refresh_campaign_stage_baselines"),
            "updated_at": self._now_iso(),
            "run_missing": bool(run_missing),
            "baseline_algorithm": str(metrics.get("algorithm_name") or ""),
            "target_dataset_ids": list(metrics.get("target_dataset_ids") or []),
            "candidate_count": len(list(baseline_summaries or [])),
        }
        stages[target_stage] = stage_state
        campaign["stages"] = stages
        campaign["updated_at"] = self._now_iso()
        self._save_campaign(campaign)
        payload = {
            "status": "updated",
            "campaign_id": str(campaign.get("campaign_id") or ""),
            "algorithm_id": str(campaign.get("algorithm_id") or ""),
            "stage": target_stage,
            "external_baseline_metrics": metrics,
            "baseline_summaries": list(baseline_summaries or []),
        }
        self._emit("algorithm_campaign_stage_baseline_refreshed", payload)
        return payload

    def register_campaign_control_baseline(
        self,
        campaign_id: str,
        *,
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
    ) -> Dict[str, Any]:
        campaign = self._load_campaign(campaign_id)
        algo_id = str(campaign.get("algorithm_id") or "").strip().lower()
        self._ensure_algorithm_id_mutable(algo_id, action="register_campaign_control_baseline")
        target_stage = str(stage or campaign.get("current_stage") or CAMPAIGN_STAGE_ORDER[0]).strip()
        stages = campaign.setdefault("stages", {})
        if target_stage not in stages:
            raise ValueError(f"Unknown campaign stage: {target_stage}")
        raw_metrics = metrics if isinstance(metrics, dict) else {}
        if not raw_metrics:
            raise ValueError("metrics must be a non-empty dict produced by a real control/ablation baseline run.")
        stage_state = stages.setdefault(target_stage, {})
        stage_panel = stage_state.get("stage_panel") if isinstance(stage_state.get("stage_panel"), dict) else {}
        panel_ids = [
            str(item or "").strip()
            for item in list(stage_panel.get("target_dataset_ids") or [])
            if str(item or "").strip()
        ]
        explicit_ids = [str(item or "").strip() for item in list(target_dataset_ids or []) if str(item or "").strip()]
        metric_ids = self._campaign_target_dataset_ids(raw_metrics)
        baseline_ids = explicit_ids or metric_ids or panel_ids
        if not baseline_ids:
            raise ValueError("target_dataset_ids are required so the control baseline is comparable to the frozen stage panel.")
        if panel_ids and baseline_ids != panel_ids:
            raise ValueError(
                f"control baseline panel does not match frozen stage panel "
                f"(baseline={baseline_ids}, stage_panel={panel_ids})"
            )
        name = str(baseline_name or "").strip()
        if not name:
            raise ValueError("baseline_name is required.")
        safe_id = self._normalize_baseline_identity(baseline_id or name)
        if not safe_id:
            raise ValueError("baseline_id could not be normalized.")
        existing = [
            dict(item or {})
            for item in list(stage_state.get("agent_registered_baselines") or [])
            if isinstance(item, dict)
        ]
        if any(str(item.get("baseline_id") or "") == safe_id for item in existing):
            raise ValueError(
                f"control baseline '{safe_id}' is already registered; "
                "use update_campaign_control_baseline(...) to deactivate, reactivate, or replace it with an audited revision."
            )
        code_source = self._normalize_control_code_source(control_code_source)
        policy = self._campaign_stage_policy(campaign, target_stage)
        summary = self._summarize_campaign_metrics(
            {**deepcopy(raw_metrics), "target_dataset_ids": baseline_ids},
            policy,
        )
        for key, value in raw_metrics.items():
            if key not in summary and (
                isinstance(value, (str, int, float, bool))
                or value is None
                or key in {"claim_metric_evaluator", "custom_metrics", "w1_backend", "w1_backend_policy"}
            ):
                summary[key] = deepcopy(value)
        summary["target_dataset_ids"] = baseline_ids
        summary["algorithm_name"] = safe_id
        summary["baseline_name"] = name
        summary["baseline_type"] = str(baseline_type or "control").strip().lower() or "control"
        summary["baseline_kind"] = "agent_registered_control"
        summary["selection_note"] = "agent-registered frozen control/ablation baseline"
        summary["required_for_gate"] = bool(required_for_gate)
        roles = [
            (
                str(item or "").strip().lower()
                if str(item or "").strip().lower() in {"claim", "w1", "both"}
                else self._normalize_metric_name(item)
            )
            for item in list(metric_roles or [])
            if str(item or "").strip()
        ]
        if not roles:
            roles = ["both"]
        now = self._now_iso()
        record = {
            **summary,
            "schema_version": 1,
            "baseline_id": safe_id,
            "baseline_name": name,
            "baseline_type": summary["baseline_type"],
            "baseline_kind": "agent_registered_control",
            "stage": target_stage,
            "campaign_id": str(campaign.get("campaign_id") or ""),
            "algorithm_id": algo_id,
            "metric_roles": roles,
            "status": "active",
            "required_for_gate": bool(required_for_gate),
            "frozen": True,
            "revision": 1,
            "revision_history": [],
            "revision_policy": "mutable_with_audit",
            "registered_at": now,
            "updated_at": now,
            "reason": str(reason or "").strip(),
            "source_run_id": str(source_run_id or "").strip(),
            "source_trial_id": str(source_trial_id or "").strip(),
            "source_metrics_path": str(source_metrics_path or raw_metrics.get("metrics_path") or "").strip(),
            "source_config_path": str(source_config_path or raw_metrics.get("resolved_config_path") or "").strip(),
            "source_artifact_paths": dict(source_artifact_paths or {}),
            "control_code_source": code_source,
            "metrics": deepcopy(raw_metrics),
        }
        w1_blocker = self._required_control_w1_provenance_blocker(
            record,
            metric_roles=roles,
            required_for_gate=bool(required_for_gate),
        )
        if w1_blocker:
            raise ValueError(w1_blocker)
        existing.append(record)
        stage_state["agent_registered_baselines"] = existing
        external = dict(stage_state.get("external_baseline_metrics") or {})
        external.setdefault("target_dataset_ids", baseline_ids)
        external["agent_registered_baselines"] = deepcopy(existing)
        stage_state["external_baseline_metrics"] = external
        stages[target_stage] = stage_state
        campaign["stages"] = stages
        self._save_campaign(campaign)
        payload = {
            "status": "registered",
            "campaign_id": str(campaign.get("campaign_id") or ""),
            "algorithm_id": algo_id,
            "stage": target_stage,
            "baseline_id": safe_id,
            "baseline_name": name,
            "baseline_type": record["baseline_type"],
            "metric_roles": roles,
            "required_for_gate": bool(required_for_gate),
            "target_dataset_ids": baseline_ids,
            "control_code_source": code_source,
            "metrics_summary": self._compact_campaign_metrics(record, include_per_dataset=False),
            "message": (
                "Control baseline registered as audited gate evidence. Gate checks compare against the strongest "
                "comparable builtin/reference or active registered control baseline for each metric. If this "
                "control was mis-coded, use update_campaign_control_baseline(...) to deactivate or replace it."
            ),
        }
        self._emit("algorithm_campaign_control_baseline_registered", payload)
        return payload

    @staticmethod
    def _compact_control_revision(record: Dict[str, Any]) -> Dict[str, Any]:
        keep = {
            "baseline_id",
            "baseline_name",
            "baseline_type",
            "status",
            "required_for_gate",
            "metric_roles",
            "target_dataset_ids",
            "source_run_id",
            "source_trial_id",
            "source_metrics_path",
            "source_config_path",
            "source_artifact_paths",
            "control_code_source",
            "metrics",
            "revision",
            "updated_at",
            "reason",
        }
        return {key: deepcopy(value) for key, value in dict(record or {}).items() if key in keep}

    @staticmethod
    def _file_sha256_if_exists(path_value: Any) -> str:
        path_text = str(path_value or "").strip()
        if not path_text:
            return ""
        try:
            path = Path(path_text).expanduser()
            if path.is_file():
                return hashlib.sha256(path.read_bytes()).hexdigest()
        except Exception:
            return ""
        return ""

    def _normalize_control_code_source(self, value: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not isinstance(value, dict) or not value:
            raise ValueError(
                "control_code_source is required. Provide the code/config source for this control baseline, "
                "for example {'kind':'config_overrides','config_overrides': {...}} or "
                "{'kind':'workspace_patch','patch_path':'...', 'patch_sha256':'...'}."
            )
        source = deepcopy(value)
        kind = str(source.get("kind") or source.get("type") or "").strip().lower()
        if not kind:
            if isinstance(source.get("config_overrides"), dict):
                kind = "config_overrides"
            elif str(source.get("patch") or source.get("workspace_patch") or source.get("patch_path") or "").strip():
                kind = "workspace_patch"
            else:
                kind = "external_code_source"
        source["kind"] = kind
        if kind not in {
            "config_overrides",
            "workspace_patch",
            "patch",
            "script",
            "workspace_snapshot",
            "external_code_source",
            "run_campaign_control_baseline",
            "locked_algorithm_reference",
        }:
            raise ValueError(
                "control_code_source.kind must be one of config_overrides, workspace_patch, patch, "
                "script, workspace_snapshot, external_code_source, run_campaign_control_baseline, "
                "or locked_algorithm_reference."
            )
        if kind == "config_overrides" and not isinstance(source.get("config_overrides"), dict):
            raise ValueError("control_code_source.kind='config_overrides' requires a config_overrides object.")
        patch_text = str(source.get("patch") or source.get("workspace_patch") or "").strip()
        patch_path = str(source.get("patch_path") or "").strip()
        if kind in {"workspace_patch", "patch"} and not (patch_text or patch_path):
            raise ValueError("control_code_source.kind='workspace_patch' requires patch text or patch_path.")
        for key in ("patch_path", "source_code_path", "config_path", "script_path"):
            sha_key = f"{key}_sha256"
            if source.get(key) and not source.get(sha_key):
                digest = self._file_sha256_if_exists(source.get(key))
                if digest:
                    source[sha_key] = digest
        if patch_text and not source.get("patch_sha256"):
            source["patch_sha256"] = hashlib.sha256(patch_text.encode("utf-8")).hexdigest()
        return source

    def update_campaign_control_baseline(
        self,
        campaign_id: str,
        *,
        baseline_id: str,
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
    ) -> Dict[str, Any]:
        campaign = self._load_campaign(campaign_id)
        algo_id = str(campaign.get("algorithm_id") or "").strip().lower()
        self._ensure_algorithm_id_mutable(algo_id, action="update_campaign_control_baseline")
        target_stage = str(stage or campaign.get("current_stage") or CAMPAIGN_STAGE_ORDER[0]).strip()
        stages = campaign.setdefault("stages", {})
        if target_stage not in stages:
            raise ValueError(f"Unknown campaign stage: {target_stage}")
        stage_state = stages.setdefault(target_stage, {})
        safe_id = self._normalize_baseline_identity(baseline_id)
        existing = [
            dict(item or {})
            for item in list(stage_state.get("agent_registered_baselines") or [])
            if isinstance(item, dict)
        ]
        idx = next((i for i, item in enumerate(existing) if str(item.get("baseline_id") or "") == safe_id), -1)
        if idx < 0:
            raise ValueError(f"control baseline '{safe_id}' is not registered for stage '{target_stage}'.")
        action_value = str(action or "").strip().lower().replace("-", "_")
        if action_value in {"delete", "remove", "disable"}:
            action_value = "deactivate"
        if action_value not in {"deactivate", "reactivate", "replace", "update"}:
            raise ValueError("action must be deactivate, reactivate, update, or replace.")
        record = dict(existing[idx])
        previous = self._compact_control_revision(record)
        history = list(record.get("revision_history") or [])
        history.append(
            {
                "action": action_value,
                "reason": str(reason or "").strip(),
                "record": previous,
                "updated_at": self._now_iso(),
            }
        )
        record["revision_history"] = history[-25:]
        record["revision"] = int(record.get("revision") or 1) + 1
        record["updated_at"] = self._now_iso()
        if reason:
            record["reason"] = str(reason or "").strip()

        if action_value == "deactivate":
            record["status"] = "inactive"
            record["required_for_gate"] = False
            record["deactivated_at"] = record["updated_at"]
            record["deactivation_reason"] = str(reason or "").strip()
        else:
            record["status"] = "active"
            if required_for_gate is not None:
                record["required_for_gate"] = bool(required_for_gate)
            elif action_value in {"reactivate", "replace"}:
                record["required_for_gate"] = True
            if metric_roles is not None:
                roles = [
                    (
                        str(item or "").strip().lower()
                        if str(item or "").strip().lower() in {"claim", "w1", "both"}
                        else self._normalize_metric_name(item)
                    )
                    for item in list(metric_roles or [])
                    if str(item or "").strip()
                ]
                record["metric_roles"] = roles or ["both"]
            if control_code_source is not None:
                record["control_code_source"] = self._normalize_control_code_source(control_code_source)
            if source_run_id:
                record["source_run_id"] = str(source_run_id or "").strip()
            if source_trial_id:
                record["source_trial_id"] = str(source_trial_id or "").strip()
            if source_metrics_path:
                record["source_metrics_path"] = str(source_metrics_path or "").strip()
            if source_config_path:
                record["source_config_path"] = str(source_config_path or "").strip()
            if source_artifact_paths is not None:
                record["source_artifact_paths"] = dict(source_artifact_paths or {})
            raw_metrics = metrics if isinstance(metrics, dict) else {}
            if raw_metrics:
                stage_panel = stage_state.get("stage_panel") if isinstance(stage_state.get("stage_panel"), dict) else {}
                panel_ids = [
                    str(item or "").strip()
                    for item in list(stage_panel.get("target_dataset_ids") or [])
                    if str(item or "").strip()
                ]
                explicit_ids = [str(item or "").strip() for item in list(target_dataset_ids or []) if str(item or "").strip()]
                metric_ids = self._campaign_target_dataset_ids(raw_metrics)
                baseline_ids = explicit_ids or metric_ids or list(record.get("target_dataset_ids") or []) or panel_ids
                if not baseline_ids:
                    raise ValueError("target_dataset_ids are required when replacing control baseline metrics.")
                if panel_ids and baseline_ids != panel_ids:
                    raise ValueError(
                        f"control baseline panel does not match frozen stage panel "
                        f"(baseline={baseline_ids}, stage_panel={panel_ids})"
                    )
                policy = self._campaign_stage_policy(campaign, target_stage)
                summary = self._summarize_campaign_metrics(
                    {**deepcopy(raw_metrics), "target_dataset_ids": baseline_ids},
                    policy,
                )
                for key, value in raw_metrics.items():
                    if key not in summary and (
                        isinstance(value, (str, int, float, bool))
                        or value is None
                        or key in {"claim_metric_evaluator", "custom_metrics", "w1_backend", "w1_backend_policy"}
                    ):
                        summary[key] = deepcopy(value)
                for key, value in summary.items():
                    record[key] = deepcopy(value)
                record["target_dataset_ids"] = baseline_ids
                record["metrics"] = deepcopy(raw_metrics)
                record["algorithm_name"] = safe_id
                record["baseline_id"] = safe_id
                record["baseline_kind"] = "agent_registered_control"
                record["selection_note"] = "agent-registered revised control/ablation baseline"
            roles = record.get("metric_roles") if isinstance(record.get("metric_roles"), list) else None
            w1_blocker = self._required_control_w1_provenance_blocker(
                record,
                metric_roles=roles,
                required_for_gate=bool(record.get("required_for_gate", True)),
            )
            if w1_blocker:
                raise ValueError(w1_blocker)
        existing[idx] = record
        stage_state["agent_registered_baselines"] = existing
        external = dict(stage_state.get("external_baseline_metrics") or {})
        external["agent_registered_baselines"] = deepcopy(existing)
        if target_dataset_ids:
            external.setdefault("target_dataset_ids", [str(item) for item in target_dataset_ids])
        stage_state["external_baseline_metrics"] = external
        stages[target_stage] = stage_state
        campaign["stages"] = stages
        campaign["updated_at"] = self._now_iso()
        self._save_campaign(campaign)
        payload = {
            "status": "updated",
            "action": action_value,
            "campaign_id": str(campaign.get("campaign_id") or ""),
            "algorithm_id": algo_id,
            "stage": target_stage,
            "baseline_id": safe_id,
            "baseline_status": str(record.get("status") or "active"),
            "required_for_gate": bool(record.get("required_for_gate", True)),
            "revision": int(record.get("revision") or 1),
            "metrics_summary": self._compact_campaign_metrics(record, include_per_dataset=False),
            "message": (
                "Control baseline revision recorded. Active gate checks ignore inactive controls and include "
                "active controls marked required_for_gate=true."
            ),
        }
        self._emit("algorithm_campaign_control_baseline_updated", payload)
        return payload

    @classmethod
    def _campaigns_root(cls) -> Path:
        return get_cellcompass_root() / "algorithm_campaigns"

    @classmethod
    def _campaign_algorithm_dir(cls, algorithm_id: str) -> Path:
        return cls._campaigns_root() / str(algorithm_id or "").strip().lower()

    @classmethod
    def _campaign_dir(cls, algorithm_id: str, campaign_id: str) -> Path:
        return cls._campaign_algorithm_dir(algorithm_id) / str(campaign_id or "").strip()

    @classmethod
    def _campaign_index_path(cls, algorithm_id: str, campaign_id: str) -> Path:
        return cls._campaign_dir(algorithm_id, campaign_id) / "campaign.json"

    @classmethod
    def _campaign_trials_dir(cls, algorithm_id: str, campaign_id: str) -> Path:
        return cls._campaign_dir(algorithm_id, campaign_id) / "trials"

    @staticmethod
    def _normalize_metric_direction(value: Any, default: str = "lower") -> str:
        direction = str(value or default).strip().lower().replace("-", "_").replace(" ", "_")
        if direction in {
            "maximize",
            "max",
            "higher",
            "greater",
            "higher_is_better",
            "greater_is_better",
            "larger_is_better",
            "maximize_is_better",
        }:
            return "greater"
        if direction in {
            "minimize",
            "min",
            "lower",
            "less",
            "lower_is_better",
            "less_is_better",
            "smaller_is_better",
            "minimize_is_better",
        }:
            return "lower"
        return default

    @staticmethod
    def _normalize_metric_name(value: Any) -> str:
        raw = str(value or "").strip()
        normalized = raw.lower().replace("-", "_").replace(" ", "_")
        if normalized in {
            "w1",
            "w1_score",
            "w1_scores",
            "w1_mean",
            "wasserstein",
            "wasserstein_1",
            "wasserstein1",
        }:
            return "w1_mean"
        if normalized in {"tmv", "tmv_score", "tmv_scores", "tmv_mean"}:
            return "tmv_mean"
        return raw

    @staticmethod
    def _numeric_value(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except Exception:
            return None

    def _claim_metric_from_spec(self, claim_metric_spec: Any) -> Dict[str, Any]:
        if isinstance(claim_metric_spec, dict):
            spec = dict(claim_metric_spec)
        else:
            spec = {"name": str(claim_metric_spec or "").strip()}
        name = self._normalize_metric_name(
            spec.get("name")
            or spec.get("metric_name")
            or spec.get("primary_metric")
            or "claim_metric"
        )
        return {
            **spec,
            "name": name,
            "direction": self._normalize_metric_direction(spec.get("direction"), default="greater"),
        }

    @staticmethod
    def _claim_metric_spec_has_evaluator_path(claim_metric_spec: Any) -> bool:
        spec = claim_metric_spec if isinstance(claim_metric_spec, dict) else {}
        return bool(str(spec.get("evaluator_path") or "").strip())

    def _validate_campaign_claim_metric_evaluator(
        self,
        claim_metric_spec: Any,
        stage_policies: Optional[Dict[str, Dict[str, Any]]] = None,
        *,
        action: str,
    ) -> None:
        claim_spec = self._claim_metric_from_spec(claim_metric_spec or {})
        policies = stage_policies if isinstance(stage_policies, dict) else self._default_campaign_stage_policies(claim_spec)
        stage2_policy = dict(policies.get("stage2_claim_validation") or {})
        primary_metric = str(stage2_policy.get("primary_metric") or "").strip()
        if (
            self._is_stage2_claim_metric("stage2_claim_validation", primary_metric)
            and not self._claim_metric_spec_has_evaluator_path(claim_spec)
        ):
            if action == "start_algorithm_campaign":
                repair = (
                    "Pass claim_metric_spec.evaluator_path to start_algorithm_campaign. "
                    "If the campaign already exists, repair it with update_campaign_claim_metric_spec(...)."
                )
            else:
                repair = (
                    "Repair the active campaign with update_campaign_claim_metric_spec(...)."
                )
            raise ValueError(
                "Stage 2 custom claim metric requires claim_metric_spec.evaluator_path before campaign execution. "
                f"{repair} Do not patch campaign.json or pass evaluator_path through dataset_config_overrides."
            )

    @staticmethod
    def _normalize_campaign_baseline_selection_policy(value: Any) -> str:
        policy = str(value or DEFAULT_CAMPAIGN_BASELINE_SELECTION_POLICY).strip().lower().replace("-", "_")
        aliases = {
            "strict": "strict_all_builtin",
            "all_builtin": "strict_all_builtin",
            "all_builtins": "strict_all_builtin",
            "fixed_builtin": "strict_all_builtin",
            "fixed_builtins": "strict_all_builtin",
            "legacy": "permissive",
            "lenient": "permissive",
            "wide": "permissive",
        }
        policy = aliases.get(policy, policy)
        if policy not in CAMPAIGN_BASELINE_SELECTION_POLICIES:
            return DEFAULT_CAMPAIGN_BASELINE_SELECTION_POLICY
        return policy

    @classmethod
    def _configured_campaign_baseline_selection_policy(cls) -> str:
        return cls._normalize_campaign_baseline_selection_policy(
            os.environ.get("CYTOBRIDGE_CAMPAIGN_BASELINE_SELECTION_POLICY")
        )

    def _default_campaign_stage_policies(self, claim_metric_spec: Any) -> Dict[str, Dict[str, Any]]:
        claim = self._claim_metric_from_spec(claim_metric_spec)
        claim_metric = str(claim.get("name") or "claim_metric")
        claim_direction = str(claim.get("direction") or "greater")
        secondary_tolerance = 0.2
        return {
            "stage1_feasibility": {
                "primary_metric": "w1_mean",
                "primary_direction": "lower",
                "secondary_metric": "",
                "secondary_direction": "",
                "secondary_tolerance": secondary_tolerance,
                "secondary_abs_tolerance": 1e-8,
                "min_delta": 0.0,
                "abs_min_delta": 0.0,
                "tmv_max": 0.2,
                "tmv_gate_required": None,
                "tmv_gate_mode": "mass_modeling_only",
                "gate_vs_baseline_multiplier": 1.2,
                "training_stage": "pilot",
                "max_trials": CAMPAIGN_STAGE_MAX_TRIALS["stage1_feasibility"],
                "min_trials_before_advance": 0,
            },
            "stage2_claim_validation": {
                "primary_metric": claim_metric,
                "primary_direction": claim_direction,
                "secondary_metric": "w1_mean",
                "secondary_direction": "lower",
                "secondary_tolerance": secondary_tolerance,
                "secondary_abs_tolerance": 1e-8,
                "min_delta": 0.0,
                "abs_min_delta": 0.0,
                "tmv_max": 0.2,
                "tmv_gate_required": None,
                "tmv_gate_mode": "mass_modeling_only",
                "w1_vs_baseline_multiplier": 1.5,
                "w1_vs_baseline_multiplier_real_high": 1.2,
                "w1_vs_baseline_multiplier_real_mid": 1.3,
                "w1_dynamic_baseline_threshold_high": 3.0,
                "w1_dynamic_baseline_threshold_mid": 1.0,
                "gate_vs_baseline_min_delta": 0.1,
                "gate_vs_baseline_abs_min_delta": 0.0,
                "training_stage": "pilot",
                "max_trials": CAMPAIGN_STAGE_MAX_TRIALS["stage2_claim_validation"],
                "min_trials_before_advance": 0,
            },
            "stage3_tuning": {
                "primary_metric": "w1_mean",
                "primary_direction": "lower",
                "secondary_metric": claim_metric,
                "secondary_direction": claim_direction,
                "secondary_tolerance": 0.05,
                "secondary_abs_tolerance": 1e-8,
                "min_delta": 0.0,
                "abs_min_delta": 0.0,
                "tmv_max": 0.2,
                "tmv_gate_required": None,
                "tmv_gate_mode": "mass_modeling_only",
                "gate_vs_baseline_multiplier": 1.0,
                "optional_sota_gate_enabled": True,
                "optional_sota_gate_budget_exhaustion_allows_advance": True,
                "training_stage": "pilot",
                "max_trials": CAMPAIGN_STAGE_MAX_TRIALS["stage3_tuning"],
                "min_trials_before_advance": 10,
            },
            "final_regression": {
                "primary_metric": "w1_mean",
                "primary_direction": "lower",
                "secondary_metric": claim_metric,
                "secondary_direction": claim_direction,
                "secondary_tolerance": 0.05,
                "secondary_abs_tolerance": 1e-8,
                "min_delta": 0.0,
                "abs_min_delta": 0.0,
                "tmv_max": 0.2,
                "tmv_gate_required": None,
                "tmv_gate_mode": "mass_modeling_only",
                "gate_vs_baseline_multiplier": 1.0,
                "training_stage": "final",
                "config_only_tuning": False,
                "confirmation_only": True,
                "inherits_active_best": True,
                "max_trials": CAMPAIGN_STAGE_MAX_TRIALS["final_regression"],
                "min_trials_before_advance": 0,
            },
        }

    def _campaign_proposal_algorithm_attributes(self, campaign: Dict[str, Any]) -> Dict[str, Any]:
        algo_id = str(campaign.get("algorithm_id") or "").strip().lower()
        proposal_id = str(campaign.get("proposal_id") or "").strip()
        if not algo_id or not proposal_id:
            return {}
        try:
            registry = self._load_algorithm_registry(algo_id)
            proposal_record = self._proposal_record_from_registry(algo_id, registry, proposal_id)
        except Exception:
            return {}
        proposal_payload = (
            proposal_record.get("proposal")
            if isinstance(proposal_record.get("proposal"), dict)
            else {}
        )
        attributes = (
            proposal_record.get("algorithm_attributes")
            if isinstance(proposal_record.get("algorithm_attributes"), dict)
            else proposal_payload.get("algorithm_attributes")
            if isinstance(proposal_payload.get("algorithm_attributes"), dict)
            else {}
        )
        if attributes:
            return dict(attributes)
        mass_scope = str(proposal_payload.get("mass_modeling_scope") or "").strip()
        if not mass_scope:
            return {}
        models_unbalanced = mass_scope == "models_unbalanced_mass"
        return {
            "mass_modeling_scope": mass_scope,
            "models_unbalanced_mass": models_unbalanced,
            "tmv_gate_required": models_unbalanced,
            "tmv_gate_reason": "proposal.algorithm_attributes.mass_modeling_scope",
        }

    def _sync_campaign_tmv_policy_from_proposal(self, campaign: Dict[str, Any]) -> None:
        attributes = self._campaign_proposal_algorithm_attributes(campaign)
        if not attributes:
            return
        mass_scope = str(attributes.get("mass_modeling_scope") or "").strip()
        if not mass_scope:
            return
        required = bool(attributes.get("tmv_gate_required", attributes.get("models_unbalanced_mass", False)))
        reason = str(attributes.get("tmv_gate_reason") or "proposal.algorithm_attributes.mass_modeling_scope")
        claim_spec = campaign.get("claim_metric_spec") if isinstance(campaign.get("claim_metric_spec"), dict) else {}
        claim_spec = dict(claim_spec)
        claim_spec["mass_modeling_scope"] = mass_scope
        claim_spec["models_unbalanced_mass"] = bool(attributes.get("models_unbalanced_mass", required))
        claim_spec["tmv_gate_required"] = required
        claim_spec["tmv_gate_reason"] = reason
        campaign["claim_metric_spec"] = claim_spec
        campaign["tmv_policy_synced_from_proposal_id"] = str(campaign.get("proposal_id") or "")
        policies = campaign.get("stage_policies") if isinstance(campaign.get("stage_policies"), dict) else {}
        campaign["stage_policies"] = policies
        for stage in CAMPAIGN_STAGE_ORDER:
            policy = policies.get(stage) if isinstance(policies.get(stage), dict) else {}
            policy["tmv_gate_required"] = required
            policy["tmv_gate_reason"] = reason
            policies[stage] = policy

    def _ensure_campaign_stage_policy_defaults(self, campaign: Dict[str, Any]) -> Dict[str, Any]:
        self._sync_campaign_tmv_policy_from_proposal(campaign)
        defaults = self._default_campaign_stage_policies(campaign.get("claim_metric_spec") or {})
        policies = campaign.setdefault("stage_policies", {})
        if not isinstance(policies, dict):
            policies = {}
            campaign["stage_policies"] = policies
        for stage, default_policy in defaults.items():
            policy = policies.setdefault(stage, {})
            if not isinstance(policy, dict):
                policy = {}
                policies[stage] = policy
            for key, value in default_policy.items():
                policy.setdefault(key, value)
            if stage == "stage2_claim_validation":
                try:
                    if float(policy.get("w1_vs_baseline_multiplier", 1.2)) == 1.2:
                        policy["w1_vs_baseline_multiplier"] = 1.5
                except Exception:
                    policy["w1_vs_baseline_multiplier"] = 1.5
                policy.setdefault("w1_vs_baseline_multiplier_real_high", 1.2)
                policy.setdefault("w1_vs_baseline_multiplier_real_mid", 1.3)
                policy.setdefault("w1_dynamic_baseline_threshold_high", 3.0)
                policy.setdefault("w1_dynamic_baseline_threshold_mid", 1.0)
            elif stage == "final_regression":
                policy["max_trials"] = CAMPAIGN_STAGE_MAX_TRIALS["final_regression"]
                policy["config_only_tuning"] = False
                policy["confirmation_only"] = True
                policy["inherits_active_best"] = True
                policy["secondary_tolerance"] = 0.05
        return campaign

    def _campaign_stage_policy(self, campaign: Dict[str, Any], stage: str) -> Dict[str, Any]:
        self._ensure_campaign_stage_policy_defaults(campaign)
        return dict((campaign.get("stage_policies") or {}).get(stage) or {})

    @staticmethod
    def _campaign_stage_budget(stage_state: Dict[str, Any], policy: Dict[str, Any]) -> Dict[str, Any]:
        max_trials = int(policy.get("max_trials") or 0)
        trial_count = int(stage_state.get("trial_count") or 0)
        remaining = max(0, max_trials - trial_count) if max_trials > 0 else None
        return {
            "trial_count": trial_count,
            "max_trials": max_trials,
            "remaining_trials": remaining,
            "budget_exhausted": bool(max_trials > 0 and trial_count >= max_trials),
        }

    @staticmethod
    def _effective_stage2_w1_multiplier(policy: Dict[str, Any], baseline_w1: float) -> Tuple[float, str]:
        """Return a scale-aware W1 guardrail multiplier for Stage 2.

        A fixed 1.5x guardrail is useful on low-scale simulation benchmarks but
        too permissive on real latent-space datasets where baseline W1 can be
        several units. The cap tightens as the baseline W1 scale grows while
        preserving the policy multiplier as an upper bound.
        """

        configured = float(policy.get("w1_vs_baseline_multiplier", 1.5) or 1.5)
        high_threshold = float(policy.get("w1_dynamic_baseline_threshold_high", 3.0) or 3.0)
        mid_threshold = float(policy.get("w1_dynamic_baseline_threshold_mid", 1.0) or 1.0)
        high_multiplier = float(policy.get("w1_vs_baseline_multiplier_real_high", 1.2) or 1.2)
        mid_multiplier = float(policy.get("w1_vs_baseline_multiplier_real_mid", 1.3) or 1.3)
        baseline = float(baseline_w1)
        if baseline >= high_threshold:
            return min(configured, high_multiplier), "high_baseline_w1_scale"
        if baseline >= mid_threshold:
            return min(configured, mid_multiplier), "mid_baseline_w1_scale"
        return configured, "low_baseline_w1_scale"

    def _metric_value(self, metrics: Dict[str, Any], metric_name: str) -> Optional[float]:
        name = self._normalize_metric_name(metric_name)
        if not name:
            return None
        if name == "w1_mean":
            value = metrics.get("w1_mean")
            return self._numeric_value(value) if value is not None else self._metric_mean(metrics.get("w1_scores"))
        if name == "tmv_mean":
            value = metrics.get("tmv_mean")
            return self._numeric_value(value) if value is not None else self._metric_mean(metrics.get("tmv_scores"))
        direct = self._numeric_value(metrics.get(name))
        if direct is not None:
            return direct
        custom = metrics.get("custom_metrics")
        if isinstance(custom, dict):
            custom_value = self._numeric_value(custom.get(name))
            if custom_value is not None:
                return custom_value
            metric_name = str(custom.get("metric_name") or custom.get("name") or "").strip()
            if metric_name == name:
                value = self._numeric_value(custom.get("value"))
                if value is not None:
                    return value
                details = custom.get("details")
                if isinstance(details, dict):
                    detail_value = self._numeric_value(details.get(name))
                    if detail_value is not None:
                        return detail_value
        return None

    @staticmethod
    def _coerce_locked_gate_requirements(raw: Any) -> List[Dict[str, Any]]:
        if isinstance(raw, dict) and isinstance(raw.get("aggregate"), list):
            raw = raw.get("aggregate")
        if isinstance(raw, list):
            requirements = []
            for item in raw:
                if not isinstance(item, dict):
                    continue
                normalized = dict(item)
                op = str(normalized.get("op") or normalized.get("operator") or "").strip()
                if "value" in normalized:
                    if op in {">", ">="}:
                        normalized.setdefault("min", normalized.get("value"))
                    elif op in {"<", "<="}:
                        normalized.setdefault("max", normalized.get("value"))
                    elif op in {"=", "=="}:
                        normalized.setdefault("equals", normalized.get("value"))
                requirements.append(normalized)
            return requirements
        if isinstance(raw, dict):
            requirements: List[Dict[str, Any]] = []
            for metric, rule in raw.items():
                if metric in {"description", "require_native_multi_dataset_panel", "require_exact_target_dataset_ids", "forbid_single_anchor_nested_all_case_scorer"}:
                    continue
                if isinstance(rule, dict):
                    requirements.append({"metric": metric, **dict(rule)})
                else:
                    requirements.append({"metric": metric, "min": rule})
            return requirements
        return []

    @classmethod
    def _load_locked_stage_gate_spec(cls) -> Dict[str, Any]:
        inline = str(os.environ.get(LOCKED_STAGE_GATE_INLINE_ENV) or "").strip()
        path_raw = str(os.environ.get(LOCKED_STAGE_GATE_SPEC_ENV) or "").strip()
        if inline:
            return json.loads(inline)
        if not path_raw:
            return {}
        path = Path(path_raw).expanduser()
        if not path.is_file():
            return {
                "_locked_gate_error": f"{LOCKED_STAGE_GATE_SPEC_ENV} points to a missing file: {path}",
                "_locked_gate_source": str(path),
            }
        try:
            text = path.read_text(encoding="utf-8")
            if path.suffix.lower() in {".yaml", ".yml"}:
                loaded = yaml.safe_load(text)
            else:
                loaded = json.loads(text)
        except Exception as exc:  # noqa: BLE001
            return {
                "_locked_gate_error": f"failed to parse locked stage gate spec {path}: {exc}",
                "_locked_gate_source": str(path),
            }
        if not isinstance(loaded, dict):
            return {
                "_locked_gate_error": f"locked stage gate spec must be a mapping: {path}",
                "_locked_gate_source": str(path),
            }
        loaded.setdefault("_locked_gate_source", str(path))
        return loaded

    def _locked_stage_gate_for_campaign_stage(self, campaign: Dict[str, Any], stage: str) -> Dict[str, Any]:
        spec = self._load_locked_stage_gate_spec()
        if not spec:
            return {}
        if spec.get("_locked_gate_error"):
            return spec
        scope = spec.get("scope") if isinstance(spec.get("scope"), dict) else {}
        algorithm_id = str(scope.get("algorithm_id") or "").strip().lower()
        if algorithm_id and algorithm_id != str(campaign.get("algorithm_id") or "").strip().lower():
            return {}
        campaign_id = str(scope.get("campaign_id") or "").strip()
        if campaign_id and campaign_id != str(campaign.get("campaign_id") or "").strip():
            return {}
        stages = spec.get("stages") if isinstance(spec.get("stages"), dict) else {}
        stage_spec = stages.get(stage) if isinstance(stages.get(stage), dict) else {}
        if not stage_spec and stage in spec and isinstance(spec.get(stage), dict):
            stage_spec = dict(spec.get(stage) or {})
        if not stage_spec:
            spec_stage = str(spec.get("stage") or "").strip().lower()
            stage_aliases = {
                "stage1": "stage1_feasibility",
                "stage2": "stage2_claim_validation",
                "stage3": "stage3_tuning",
                "final": "final_regression",
                "final_regression": "final_regression",
            }
            if stage_aliases.get(spec_stage, spec_stage) == stage:
                stage_spec = dict(spec)
        if not stage_spec:
            return {}
        return {
            "gate_id": str(spec.get("gate_id") or "locked_stage_gate"),
            "source": str(spec.get("_locked_gate_source") or "env_inline"),
            "stage": stage,
            **dict(stage_spec),
        }

    def _stage2_locked_gate_floor_for_stage3(self, campaign: Dict[str, Any]) -> Dict[str, Any]:
        """Return Stage 2 absolute locked-gate requirements as Stage 3 floors.

        Stage 3 is allowed to tune W1, but it must not promote or finish with a
        candidate that falls below explicit absolute score thresholds that were
        required for Stage 2 validation.
        """

        stage2_gate = self._locked_stage_gate_for_campaign_stage(campaign, "stage2_claim_validation")
        if stage2_gate:
            inherited = dict(stage2_gate)
            inherited["gate_id"] = f"{str(stage2_gate.get('gate_id') or 'locked_stage_gate')}.stage3_floor"
            inherited["stage"] = "stage3_tuning"
            inherited["inherited_from_stage"] = "stage2_claim_validation"
            inherited["note"] = "Stage 3 must preserve explicit absolute floors from Stage 2 validation."
            return inherited

        stages = campaign.get("stages") if isinstance(campaign.get("stages"), dict) else {}
        stage2 = stages.get("stage2_claim_validation") if isinstance(stages.get("stage2_claim_validation"), dict) else {}
        evidence = stage2.get("stage_gate_evidence") if isinstance(stage2.get("stage_gate_evidence"), dict) else {}
        checks = evidence.get("checks") if isinstance(evidence.get("checks"), dict) else {}
        locked = checks.get("locked_stage_gate") if isinstance(checks.get("locked_stage_gate"), dict) else {}
        if not locked or locked.get("error"):
            return {}

        requirements: List[Dict[str, Any]] = []
        expected_ids: List[str] = []
        for item in list(locked.get("requirements") or []):
            if not isinstance(item, dict):
                continue
            metric = self._normalize_metric_name(item.get("metric") or item.get("name") or "")
            if metric == "target_dataset_ids":
                expected_ids = [
                    str(value or "").strip()
                    for value in list(item.get("expected") or [])
                    if str(value or "").strip()
                ]
                continue
            if not metric:
                continue
            min_value = self._numeric_value(item.get("min"))
            max_value = self._numeric_value(item.get("max"))
            equals_value = self._numeric_value(item.get("equals"))
            requirement: Dict[str, Any] = {"metric": metric}
            direction = self._normalize_metric_direction(item.get("direction"), default="")
            if direction:
                requirement["direction"] = direction
            if min_value is not None:
                requirement["min"] = min_value
            elif max_value is not None:
                requirement["max"] = max_value
            elif equals_value is not None:
                requirement["equals"] = equals_value
            else:
                continue
            requirements.append(requirement)

        if not expected_ids:
            stage2_panel = stage2.get("stage_panel") if isinstance(stage2.get("stage_panel"), dict) else {}
            expected_ids = [
                str(value or "").strip()
                for value in list(stage2_panel.get("target_dataset_ids") or [])
                if str(value or "").strip()
            ]
        if not requirements:
            return {}
        return {
            "gate_id": "persisted_stage2_locked_gate.stage3_floor",
            "source": str(locked.get("source") or "stage2_stage_gate_evidence"),
            "stage": "stage3_tuning",
            "inherited_from_stage": "stage2_claim_validation",
            "requirements": requirements,
            "target_dataset_ids": expected_ids,
            "note": "Stage 3 must preserve explicit absolute floors from Stage 2 validation.",
        }

    def _evaluate_stage3_inherited_stage2_floor(
        self,
        *,
        campaign: Dict[str, Any],
        stage: str,
        active_metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        if str(stage or "").strip() != "stage3_tuning":
            return {}
        gate_spec = self._stage2_locked_gate_floor_for_stage3(campaign)
        if not gate_spec:
            return {}
        check = self._evaluate_locked_stage_gate(gate_spec=gate_spec, active_metrics=active_metrics)
        check["inherited_from_stage"] = "stage2_claim_validation"
        check["note"] = "Stage 3 cannot regress below explicit Stage 2 locked-gate floors."
        return check

    def _stage2_claim_metric_regression_floor_for_stage(
        self,
        campaign: Dict[str, Any],
        stage: str,
    ) -> Dict[str, Any]:
        target_stage = str(stage or "").strip()
        if target_stage not in {"stage3_tuning", "final_regression"}:
            return {}
        stages = campaign.get("stages") if isinstance(campaign.get("stages"), dict) else {}
        stage2 = stages.get("stage2_claim_validation") if isinstance(stages.get("stage2_claim_validation"), dict) else {}
        evidence = stage2.get("stage_gate_evidence") if isinstance(stage2.get("stage_gate_evidence"), dict) else {}
        stage2_metrics = (
            evidence.get("active_best_metrics_summary")
            if isinstance(evidence.get("active_best_metrics_summary"), dict)
            else {}
        )
        if not stage2_metrics:
            active_trial_id = str(stage2.get("active_best_trial_id") or "").strip()
            trials = campaign.get("trials") if isinstance(campaign.get("trials"), dict) else {}
            active_trial = trials.get(active_trial_id) if isinstance(trials.get(active_trial_id), dict) else {}
            stage2_metrics = (
                active_trial.get("metrics_summary")
                if isinstance(active_trial.get("metrics_summary"), dict)
                else {}
            )
        if not stage2_metrics:
            stage2_metrics = (
                stage2.get("active_best_metrics_summary")
                if isinstance(stage2.get("active_best_metrics_summary"), dict)
                else {}
            )
        if not stage2_metrics:
            return {}

        stage2_policy = (
            evidence.get("policy")
            if isinstance(evidence.get("policy"), dict)
            else self._campaign_stage_policy(campaign, "stage2_claim_validation")
        )
        metric = self._normalize_metric_name(stage2_policy.get("primary_metric") or "")
        if not metric or metric in {"w1_mean", "tmv_mean"}:
            return {}
        direction = self._normalize_metric_direction(stage2_policy.get("primary_direction"), default="greater")
        stage3_policy = self._campaign_stage_policy(campaign, "stage3_tuning")
        tolerance = float(stage3_policy.get("secondary_tolerance", 0.05) or 0.05)
        abs_tolerance = float(stage3_policy.get("secondary_abs_tolerance", 1e-8) or 1e-8)
        stage2_value = self._metric_value(stage2_metrics, metric)
        if stage2_value is None and self._normalize_metric_name(stage2_metrics.get("primary_metric") or "") == metric:
            stage2_value = self._numeric_value(stage2_metrics.get("primary_value"))
        if stage2_value is None:
            return {}

        if direction == "lower":
            threshold = stage2_value * (1.0 + max(0.0, tolerance)) + max(0.0, abs_tolerance)
            requirement = {
                "metric": metric,
                "direction": "lower",
                "max": threshold,
                "stage2_value": stage2_value,
                "relative_tolerance": tolerance,
                "absolute_tolerance": abs_tolerance,
            }
        else:
            threshold = stage2_value * (1.0 - max(0.0, tolerance)) - max(0.0, abs_tolerance)
            requirement = {
                "metric": metric,
                "direction": "greater",
                "min": threshold,
                "stage2_value": stage2_value,
                "relative_tolerance": tolerance,
                "absolute_tolerance": abs_tolerance,
            }

        stage2_panel = stage2.get("stage_panel") if isinstance(stage2.get("stage_panel"), dict) else {}
        expected_ids = [
            str(value or "").strip()
            for value in list(stage2_panel.get("target_dataset_ids") or [])
            if str(value or "").strip()
        ]
        if not expected_ids:
            expected_ids = self._campaign_target_dataset_ids(stage2_metrics)
        return {
            "gate_id": "stage2_claim_metric_regression_floor",
            "source": "stage2_claim_validation_active_best",
            "stage": target_stage,
            "inherited_from_stage": "stage2_claim_validation",
            "requirements": [requirement],
            "target_dataset_ids": expected_ids,
            "note": (
                "Stage 3 and final regression must preserve the Stage 2 validated claim metric "
                "within the fixed Stage 2 active-best tolerance; this floor does not move with "
                "later Stage 3 active-best regressions."
            ),
        }

    def _evaluate_stage2_claim_metric_regression_floor(
        self,
        *,
        campaign: Dict[str, Any],
        stage: str,
        active_metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        gate_spec = self._stage2_claim_metric_regression_floor_for_stage(campaign, stage)
        if not gate_spec:
            return {}
        check = self._evaluate_locked_stage_gate(gate_spec=gate_spec, active_metrics=active_metrics)
        check["inherited_from_stage"] = "stage2_claim_validation"
        check["note"] = str(gate_spec.get("note") or "")
        return check

    def _evaluate_locked_stage_gate(
        self,
        *,
        gate_spec: Dict[str, Any],
        active_metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        if gate_spec.get("_locked_gate_error"):
            return {
                "ok": False,
                "error": str(gate_spec.get("_locked_gate_error") or ""),
                "source": str(gate_spec.get("_locked_gate_source") or ""),
                "requirements": [],
            }
        raw_requirements = gate_spec.get("requirements")
        if raw_requirements is None:
            raw_requirements = gate_spec.get("metrics")
        requirements = self._coerce_locked_gate_requirements(raw_requirements)
        checks: List[Dict[str, Any]] = []
        blockers: List[str] = []
        expected_ids = [
            str(item or "").strip()
            for item in list(gate_spec.get("target_dataset_ids") or [])
            if str(item or "").strip()
        ]
        observed_ids = self._campaign_target_dataset_ids(active_metrics)
        if expected_ids:
            panel_ok = observed_ids == expected_ids
            if not panel_ok:
                blockers.append(
                    "locked stage gate failed: active best target_dataset_ids do not match locked panel "
                    f"(expected={expected_ids}, observed={observed_ids})"
                )
            checks.append(
                {
                    "metric": "target_dataset_ids",
                    "value": observed_ids,
                    "expected": expected_ids,
                    "ok": bool(panel_ok),
                    "reason": "requires exact locked target_dataset_ids",
                }
            )
        for item in requirements:
            metric = self._normalize_metric_name(item.get("metric") or item.get("name") or "")
            value = self._metric_value(active_metrics, metric)
            direction = self._normalize_metric_direction(item.get("direction"), default="")
            min_value = self._numeric_value(item.get("min") if "min" in item else item.get("min_value"))
            max_value = self._numeric_value(item.get("max") if "max" in item else item.get("max_value"))
            equals_value = self._numeric_value(item.get("equals") if "equals" in item else item.get("eq"))
            ok = True
            reason = ""
            if not metric:
                ok = False
                reason = "missing_metric_name"
            elif value is None:
                ok = False
                reason = "metric_missing"
            elif equals_value is not None:
                tolerance = self._numeric_value(item.get("tolerance"))
                tol = 1e-12 if tolerance is None else max(0.0, tolerance)
                ok = abs(value - equals_value) <= tol
                reason = f"requires {metric} == {equals_value}"
            elif min_value is not None:
                ok = value >= min_value if direction != "lower" else value <= min_value
                op = ">=" if direction != "lower" else "<="
                reason = f"requires {metric} {op} {min_value}"
            elif max_value is not None:
                ok = value <= max_value if direction != "greater" else value >= max_value
                op = "<=" if direction != "greater" else ">="
                reason = f"requires {metric} {op} {max_value}"
            else:
                ok = False
                reason = "missing_threshold"
            check = {
                "metric": metric,
                "value": value,
                "min": min_value,
                "max": max_value,
                "equals": equals_value,
                "direction": direction,
                "ok": bool(ok),
                "reason": reason,
            }
            checks.append(check)
            if not ok:
                blockers.append(f"locked stage gate failed: {reason} (observed={value})")
        if not requirements:
            blockers.append("locked stage gate has no requirements")
        return {
            "ok": not blockers,
            "gate_id": str(gate_spec.get("gate_id") or "locked_stage_gate"),
            "source": str(gate_spec.get("source") or ""),
            "requirements": checks,
            "blockers": blockers,
        }

    @staticmethod
    def _is_stage2_claim_metric(stage: str, metric_name: str) -> bool:
        metric = PlannerFileTools._normalize_metric_name(metric_name)
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

    def _claim_metric_provenance_blockers(
        self,
        *,
        stage: str,
        claim_metric: str,
        candidate_metrics: Dict[str, Any],
        baseline_metrics: Optional[Dict[str, Any]] = None,
        expected_evaluator_info: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        if not self._is_stage2_claim_metric(stage, claim_metric):
            return []
        blockers: List[str] = []
        candidate_info = (
            candidate_metrics.get("claim_metric_evaluator")
            if isinstance(candidate_metrics, dict)
            else None
        )
        if not isinstance(candidate_info, dict) or not candidate_info:
            blockers.append(
                "stage2 active best claim metric lacks campaign claim_metric_evaluator provenance"
            )
        elif self._metric_value(candidate_metrics, claim_metric) is None:
            blockers.append(f"stage2 active best is missing claim metric `{claim_metric}`")
        elif (
            str(self._claim_metric_evaluator_identity(expected_evaluator_info).get("evaluator_id") or "")
            and str(self._claim_metric_evaluator_identity(candidate_info).get("evaluator_id") or "")
            != str(self._claim_metric_evaluator_identity(expected_evaluator_info).get("evaluator_id") or "")
        ):
            blockers.append(
                "stage2 active best claim metric evaluator does not match the current campaign claim_metric_spec"
            )
        if baseline_metrics is not None:
            baseline_info = (
                baseline_metrics.get("claim_metric_evaluator")
                if isinstance(baseline_metrics, dict)
                else None
            )
            if not isinstance(baseline_info, dict) or not baseline_info:
                blockers.append(
                    "stage2 external baseline claim metric lacks campaign claim_metric_evaluator provenance"
                )
            elif self._metric_value(baseline_metrics, claim_metric) is None:
                blockers.append(f"stage2 external baseline is missing claim metric `{claim_metric}`")
            elif (
                str(self._claim_metric_evaluator_identity(expected_evaluator_info).get("evaluator_id") or "")
                and str(self._claim_metric_evaluator_identity(baseline_info).get("evaluator_id") or "")
                != str(self._claim_metric_evaluator_identity(expected_evaluator_info).get("evaluator_id") or "")
            ):
                blockers.append(
                    "stage2 external baseline claim metric evaluator does not match the current campaign claim_metric_spec"
                )
            elif isinstance(candidate_info, dict) and candidate_info:
                if not self._claim_metric_evaluator_infos_match(candidate_info, baseline_info):
                    candidate_identity = self._claim_metric_evaluator_identity(candidate_info)
                    baseline_identity = self._claim_metric_evaluator_identity(baseline_info)
                    blockers.append(
                        "stage2 active best and external baseline used different claim metric evaluators "
                        f"(active={candidate_identity}, baseline={baseline_identity})"
                    )
        return blockers

    @staticmethod
    def _w1_backend_identity(metrics: Any) -> Dict[str, Any]:
        payload = metrics if isinstance(metrics, dict) else {}
        backend = str(payload.get("w1_backend") or "").strip()
        if not backend:
            return {}
        params = payload.get("w1_backend_params")
        return {
            "backend": backend,
            "exact": bool(payload.get("w1_backend_exact", False)),
            "params": dict(params) if isinstance(params, dict) else {},
        }

    def _consistent_w1_backend_metadata(self, metric_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Return W1 backend provenance only when every available item agrees."""
        metadata_records: List[Dict[str, Any]] = []
        identities: List[Dict[str, Any]] = []
        for item in metric_items:
            if not isinstance(item, dict):
                continue
            metadata = {
                str(key): deepcopy(value)
                for key, value in item.items()
                if str(key).startswith("w1_backend")
            }
            identity = self._w1_backend_identity(metadata)
            if not identity:
                continue
            metadata_records.append(metadata)
            if identity not in identities:
                identities.append(identity)
        if not metadata_records:
            return {}
        if len(identities) > 1:
            return {
                "w1_backend_panel_inconsistent": True,
                "w1_backend_panel_identities": identities,
            }
        return metadata_records[0]

    def _training_run_metrics_for_campaign_run_id(self, algo_id: str, run_id: str) -> Dict[str, Any]:
        safe_algo = str(algo_id or "").strip().lower()
        safe_run = str(run_id or "").strip()
        if not safe_algo or not safe_run:
            return {}
        manifest_path = get_cellcompass_root() / "training_algorithms" / safe_algo / "registry" / "runs" / f"{safe_run}.json"
        manifest = self._read_json(manifest_path, {})
        if not isinstance(manifest, dict):
            return {}
        metrics_path = Path(str(manifest.get("metrics_path") or "")).expanduser()
        if metrics_path.is_file():
            metrics = self._read_json(metrics_path, {})
            if isinstance(metrics, dict):
                return metrics
        return {}

    def _rehydrate_w1_backend_metadata_from_run_ids(self, algo_id: str, run_ids: List[str]) -> Dict[str, Any]:
        metrics = [
            self._training_run_metrics_for_campaign_run_id(algo_id, str(run_id or ""))
            for run_id in list(run_ids or [])
            if str(run_id or "").strip()
        ]
        return self._consistent_w1_backend_metadata([item for item in metrics if item])

    def _rehydrate_w1_backend_metadata_from_baseline_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(record, dict):
            return {}
        direct_metadata = self._consistent_w1_backend_metadata(
            [
                item
                for item in (
                    record,
                    record.get("metrics_summary") if isinstance(record.get("metrics_summary"), dict) else {},
                    record.get("metrics") if isinstance(record.get("metrics"), dict) else {},
                )
                if isinstance(item, dict)
            ]
        )
        if self._w1_backend_identity(direct_metadata):
            return direct_metadata
        algorithm_name = str(record.get("algorithm_name") or "").strip()
        baseline_type = str(record.get("baseline_type") or "builtin").strip()
        run_ids = {str(item or "").strip() for item in list(record.get("baseline_run_ids") or []) if str(item or "").strip()}
        dataset_ids = self._campaign_target_dataset_ids(record)
        exact_metrics_items: List[Dict[str, Any]] = []
        fallback_metrics_items: List[Dict[str, Any]] = []
        for dataset_id in dataset_ids:
            try:
                payload = self.get_algorithm_benchmark_baselines(dataset_id)
            except FileNotFoundError:
                continue
            for candidate in list(payload.get("baseline_records") or []):
                if not isinstance(candidate, dict):
                    continue
                if algorithm_name and str(candidate.get("algorithm_name") or "").strip() != algorithm_name:
                    continue
                if baseline_type and str(candidate.get("baseline_type") or "").strip() != baseline_type:
                    continue
                candidate_run_id = str(candidate.get("run_id") or "").strip()
                metrics_payload = candidate.get("metrics") if isinstance(candidate.get("metrics"), dict) else {}
                metrics_summary = candidate.get("metrics_summary") if isinstance(candidate.get("metrics_summary"), dict) else {}
                source_metrics: Dict[str, Any] = {}
                for source in (candidate, metrics_summary, metrics_payload):
                    if not isinstance(source, dict):
                        continue
                    for key, value in source.items():
                        if str(key).startswith("w1_backend") or key in {"w1_mean", "w1_scores"}:
                            source_metrics.setdefault(str(key), deepcopy(value))
                if not source_metrics:
                    continue
                if run_ids and candidate_run_id in run_ids:
                    exact_metrics_items.append(source_metrics)
                else:
                    fallback_metrics_items.append(source_metrics)
        if exact_metrics_items:
            return self._consistent_w1_backend_metadata(exact_metrics_items)
        return self._consistent_w1_backend_metadata(fallback_metrics_items)

    def _w1_backend_provenance_blockers(
        self,
        *,
        candidate_metrics: Dict[str, Any],
        baseline_metrics: Optional[Dict[str, Any]],
        metric_name: str = "w1_mean",
    ) -> List[str]:
        metric = self._normalize_metric_name(metric_name)
        if metric != "w1_mean" or not isinstance(baseline_metrics, dict):
            return []
        candidate_identity = self._w1_backend_identity(candidate_metrics)
        baseline_identity = self._w1_backend_identity(baseline_metrics)
        if not candidate_identity and not baseline_identity:
            return [
                "active best and external baseline both lack W1 backend provenance; "
                "rerun comparable metrics on the same frozen panel before using W1 as a gate"
            ]
        if candidate_identity and not baseline_identity:
            return [
                "external baseline lacks W1 backend provenance; refresh baselines on the frozen panel "
                "before comparing W1"
            ]
        if baseline_identity and not candidate_identity:
            return [
                "active best lacks W1 backend provenance; rerun the active best on the frozen panel "
                "before comparing W1"
            ]
        if candidate_identity != baseline_identity:
            return [
                "active best and external baseline used different W1 backends "
                f"(active={candidate_identity}, baseline={baseline_identity}); refresh comparable metrics"
            ]
        return []

    @staticmethod
    def _prefixed_w1_backend_metadata(metrics: Optional[Dict[str, Any]], prefix: str) -> Dict[str, Any]:
        if not isinstance(metrics, dict):
            return {}
        return {
            f"{prefix}_{key}": deepcopy(value)
            for key, value in metrics.items()
            if str(key).startswith("w1_backend")
        }

    def _required_control_w1_provenance_blocker(
        self,
        record: Dict[str, Any],
        *,
        metric_roles: Optional[List[str]] = None,
        required_for_gate: bool = True,
    ) -> str:
        if not required_for_gate or not isinstance(record, dict):
            return ""
        roles = [
            str(item or "").strip().lower()
            for item in list(metric_roles or record.get("metric_roles") or [])
            if str(item or "").strip()
        ]
        supports_w1 = not roles or "both" in roles or "w1" in roles or "w1_mean" in roles
        if not supports_w1:
            return ""
        if self._metric_value(record, "w1_mean") is None:
            return ""
        if self._w1_backend_identity(record):
            return ""
        raw_metrics = record.get("metrics") if isinstance(record.get("metrics"), dict) else {}
        if self._w1_backend_identity(raw_metrics):
            return ""
        if isinstance(raw_metrics, dict) and isinstance(raw_metrics.get("campaign_dataset_metrics"), list):
            nested_metadata = self._consistent_w1_backend_metadata(
                [item for item in raw_metrics.get("campaign_dataset_metrics") or [] if isinstance(item, dict)]
            )
            if self._w1_backend_identity(nested_metadata):
                return ""
        baseline_id = str(record.get("baseline_id") or record.get("algorithm_name") or "control baseline")
        return (
            f"required W1-bearing control baseline '{baseline_id}' has W1 metrics but lacks "
            "w1_backend/w1_backend_exact/w1_backend_params provenance; rerun or replace the control "
            "with metrics from the same frozen panel before using it as gate evidence"
        )

    def _strict_baseline_audit_gate_check(
        self,
        *,
        campaign: Dict[str, Any],
        stage: str,
        baseline: Dict[str, Any],
        active_metrics: Dict[str, Any],
        policy: Dict[str, Any],
    ) -> Dict[str, Any]:
        baseline_policy = str(campaign.get("baseline_selection_policy") or "").strip()
        required = baseline_policy == "strict_all_builtin" and stage in {"stage2_claim_validation", "stage3_tuning"}
        ledger = [
            dict(item or {})
            for item in list((baseline or {}).get("strict_baseline_audit_ledger") or [])
            if isinstance(item, dict)
        ]
        candidate_count = int((baseline or {}).get("selection_candidate_count") or 0)
        strict_audit_meta = (baseline or {}).get("strict_baseline_audit")
        required_baselines = [
            str(item or "").strip()
            for item in list((strict_audit_meta or {}).get("required_baselines") or [])
            if str(item or "").strip()
        ] if isinstance(strict_audit_meta, dict) else []
        expected_ledger_count = max(candidate_count, len(required_baselines))
        check: Dict[str, Any] = {
            "required": bool(required),
            "baseline_selection_policy": baseline_policy,
            "selection_rule": str((baseline or {}).get("selection_rule") or ""),
            "selection_candidate_count": candidate_count,
            "strict_required_baseline_count": len(required_baselines),
            "expected_ledger_count": expected_ledger_count,
            "ledger_count": len(ledger),
            "ok": True,
            "blockers": [],
            "records": [],
            "missing_required_records": [],
            "nonblocking_failed_records": [],
        }
        if not required:
            return check
        if not baseline:
            check["ok"] = False
            check["blockers"].append("strict baseline audit requires external baseline metrics")
            return check
        if expected_ledger_count > 0 and not ledger:
            check["ok"] = False
            check["blockers"].append(
                "strict baseline audit ledger is missing; refresh strict baselines so every required baseline has an explicit pass/fail evidence record"
            )
            return check
        if expected_ledger_count > 0 and len(ledger) < expected_ledger_count:
            check["ok"] = False
            check["blockers"].append(
                f"strict baseline audit ledger is incomplete: ledger_count={len(ledger)} expected_count={expected_ledger_count}"
            )
        if not ledger:
            return check

        if stage == "stage3_tuning":
            primary_metric = str(policy.get("primary_metric") or "w1_mean")
            primary_direction = str(policy.get("primary_direction") or "lower")
            claim_metric = str(policy.get("secondary_metric") or "").strip()
            claim_direction = str(policy.get("secondary_direction") or "").strip()
        else:
            primary_metric = str(policy.get("primary_metric") or "").strip()
            primary_direction = str(policy.get("primary_direction") or "lower")
            claim_metric = primary_metric
            claim_direction = primary_direction
        candidate_primary = self._metric_value(active_metrics, primary_metric)
        candidate_w1 = self._metric_value(active_metrics, "w1_mean")
        candidate_claim = self._metric_value(active_metrics, claim_metric) if claim_metric else None

        def _passes(candidate: Optional[float], baseline_value: Optional[float], direction: str) -> Optional[bool]:
            if candidate is None or baseline_value is None or not direction:
                return None
            return bool(candidate >= baseline_value if direction == "greater" else candidate <= baseline_value)

        for item in ledger:
            status = str(item.get("audit_status") or "").strip()
            algorithm_name = str(item.get("algorithm_name") or "").strip()
            reasons = [str(reason or "").strip() for reason in list(item.get("reasons") or []) if str(reason or "").strip()]
            baseline_primary = self._metric_value(item, primary_metric)
            if baseline_primary is None:
                baseline_primary = self._numeric_value(item.get("primary_value"))
            baseline_w1 = self._metric_value(item, "w1_mean")
            baseline_claim = self._metric_value(item, claim_metric) if claim_metric else None
            if baseline_claim is None and claim_metric == primary_metric:
                baseline_claim = baseline_primary
            if baseline_claim is None:
                baseline_claim = self._numeric_value(item.get("secondary_value"))
            record: Dict[str, Any] = {
                "algorithm_name": algorithm_name,
                "audit_status": status,
                "comparable": bool(item.get("comparable")),
                "selected_primary": bool(item.get("selected_primary")),
                "selected_claim_sota": bool(item.get("selected_claim_sota")),
                "selected_w1_sota": bool(item.get("selected_w1_sota")),
                "reasons": reasons,
                "baseline_run_ids": list(item.get("baseline_run_ids") or []),
                "primary_metric": primary_metric,
                "candidate_primary": candidate_primary,
                "baseline_primary": baseline_primary,
                "primary_pass": _passes(candidate_primary, baseline_primary, primary_direction),
                "candidate_w1": candidate_w1,
                "baseline_w1": baseline_w1,
                "w1_pass": _passes(candidate_w1, baseline_w1, "lower"),
                "has_terminal_baseline_record": bool(
                    item.get("has_terminal_baseline_record")
                    or item.get("has_baseline_summary")
                    or item.get("baseline_run_ids")
                    or item.get("baseline_dataset_summaries")
                ),
                "missing_required_record": bool(item.get("missing_required_record")),
                "nonblocking_failed_record": bool(item.get("nonblocking_failed_record")),
            }
            if claim_metric:
                record.update(
                    {
                        "claim_metric": claim_metric,
                        "candidate_claim": candidate_claim,
                        "baseline_claim": baseline_claim,
                        "claim_pass": _passes(candidate_claim, baseline_claim, claim_direction),
                    }
                )
            for key, value in item.items():
                if str(key).startswith("w1_backend"):
                    record[str(key)] = deepcopy(value)
            if status != "pass_comparable":
                legacy_missing = (
                    not bool(record.get("has_terminal_baseline_record"))
                    or any("no baseline record was found" in reason for reason in reasons)
                )
                missing_required_record = bool(record.get("missing_required_record") or legacy_missing)
                record["missing_required_record"] = missing_required_record
                record["nonblocking_failed_record"] = not missing_required_record
                if missing_required_record:
                    check["ok"] = False
                    check["missing_required_records"].append(record)
                    check["blockers"].append(
                        f"strict required baseline `{algorithm_name or 'unknown'}` has no terminal audit record: {', '.join(reasons) or status}"
                    )
                else:
                    check["nonblocking_failed_records"].append(record)
            check["records"].append(record)
        check["blockers"] = list(dict.fromkeys(check["blockers"]))
        return check

    def _summarize_campaign_metrics(self, metrics: Dict[str, Any], policy: Dict[str, Any]) -> Dict[str, Any]:
        tmv_gate = self._tmv_gate_status(metrics=metrics, policy=policy)
        summary = {
            "w1_mean": self._metric_value(metrics, "w1_mean"),
            "tmv_mean": self._metric_value(metrics, "tmv_mean"),
            "tmv_gate_required": bool(tmv_gate.get("required", True)),
            "tmv_gate_reason": str(tmv_gate.get("reason") or ""),
            "primary_metric": str(policy.get("primary_metric") or ""),
            "primary_direction": str(policy.get("primary_direction") or ""),
            "primary_value": self._metric_value(metrics, str(policy.get("primary_metric") or "")),
            "secondary_metric": str(policy.get("secondary_metric") or ""),
            "secondary_direction": str(policy.get("secondary_direction") or ""),
            "secondary_value": self._metric_value(metrics, str(policy.get("secondary_metric") or "")),
        }
        custom = metrics.get("custom_metrics")
        if isinstance(custom, dict):
            summary["custom_metrics"] = dict(custom)
        evaluator = metrics.get("claim_metric_evaluator")
        if isinstance(evaluator, dict) and evaluator:
            summary["claim_metric_evaluator"] = dict(evaluator)
        evaluator_mismatch = metrics.get("claim_metric_evaluator_mismatch")
        if isinstance(evaluator_mismatch, list) and evaluator_mismatch:
            summary["claim_metric_evaluator_mismatch"] = list(evaluator_mismatch)
        for key in (
            "baseline_quality_diagnostic_status",
            "baseline_quality_diagnostic_reason",
            "baseline_run_verdict_status",
        ):
            if str(metrics.get(key) or "").strip():
                summary[key] = str(metrics.get(key) or "").strip()
        for key, value in metrics.items():
            if str(key).startswith("w1_backend"):
                summary[str(key)] = deepcopy(value)
        if not self._w1_backend_identity(summary):
            panel_w1_metadata = self._consistent_w1_backend_metadata(
                [
                    item
                    for item in list(metrics.get("campaign_dataset_metrics") or [])
                    if isinstance(item, dict)
                ]
            )
            if panel_w1_metadata:
                summary.update(panel_w1_metadata)
        simulation_version = str(metrics.get("simulation_version") or "").strip()
        if simulation_version:
            summary["simulation_version"] = simulation_version
        simulation_versions = [
            str(item or "").strip()
            for item in list(metrics.get("simulation_versions") or [])
            if str(item or "").strip()
        ]
        if simulation_versions:
            summary["simulation_versions"] = simulation_versions
        target_dataset_ids = self._campaign_target_dataset_ids(metrics)
        if target_dataset_ids:
            summary["target_dataset_ids"] = target_dataset_ids
        dataset_metrics = metrics.get("campaign_dataset_metrics")
        if isinstance(dataset_metrics, list) and dataset_metrics:
            per_dataset: Dict[str, Dict[str, Any]] = {}
            primary_metric = str(policy.get("primary_metric") or "")
            secondary_metric = str(policy.get("secondary_metric") or "")
            for idx, item in enumerate(dataset_metrics):
                if not isinstance(item, dict):
                    continue
                dataset_id = str(
                    item.get("campaign_dataset_id")
                    or item.get("dataset_id")
                    or item.get("id")
                    or f"dataset{idx + 1}"
                ).strip()
                if not dataset_id:
                    continue
                row: Dict[str, Any] = {
                    "dataset_id": dataset_id,
                    "w1_mean": self._metric_value(item, "w1_mean"),
                    "tmv_mean": self._metric_value(item, "tmv_mean"),
                    "tmv_max": self._numeric_value(item.get("tmv_max")),
                    "primary_value": self._metric_value(item, primary_metric),
                    "secondary_value": self._metric_value(item, secondary_metric),
                    "run_id": str(item.get("run_id") or ""),
                    "error": str(item.get("error") or ""),
                }
                if isinstance(item.get("custom_metrics"), dict):
                    row["custom_metrics"] = dict(item.get("custom_metrics") or {})
                if isinstance(item.get("w1_scores"), list):
                    row["w1_scores"] = list(item.get("w1_scores") or [])
                if isinstance(item.get("tmv_scores"), list):
                    row["tmv_scores"] = list(item.get("tmv_scores") or [])
                for key, value in item.items():
                    if str(key).startswith("w1_backend"):
                        row[str(key)] = deepcopy(value)
                if bool(item.get("training_timed_out")):
                    row["training_timed_out"] = True
                    if isinstance(item.get("training_time_budget"), dict):
                        row["training_time_budget"] = dict(item.get("training_time_budget") or {})
                if bool(item.get("inference_timed_out")):
                    row["inference_timed_out"] = True
                    if isinstance(item.get("inference_time_budget"), dict):
                        row["inference_time_budget"] = dict(item.get("inference_time_budget") or {})
                w1_scores = row.get("w1_scores") if isinstance(row.get("w1_scores"), list) else []
                tmv_scores = row.get("tmv_scores") if isinstance(row.get("tmv_scores"), list) else []
                gap_count = max(len(w1_scores), len(tmv_scores))
                if gap_count:
                    row["per_time_gap"] = [
                        {
                            "gap_index": gap_idx,
                            "w1": w1_scores[gap_idx] if gap_idx < len(w1_scores) else None,
                            "tmv": tmv_scores[gap_idx] if gap_idx < len(tmv_scores) else None,
                        }
                        for gap_idx in range(gap_count)
                    ]
                per_dataset[dataset_id] = row
            if per_dataset:
                summary["per_dataset"] = per_dataset
        if bool(metrics.get("training_timed_out")):
            summary["training_timed_out"] = True
        if isinstance(metrics.get("training_time_budget"), dict):
            summary["training_time_budget"] = dict(metrics.get("training_time_budget") or {})
        if bool(metrics.get("inference_timed_out")):
            summary["inference_timed_out"] = True
        if isinstance(metrics.get("inference_time_budget"), dict):
            summary["inference_time_budget"] = dict(metrics.get("inference_time_budget") or {})
        return summary

    def _campaign_target_dataset_blockers(self, summary: Dict[str, Any]) -> List[str]:
        target_ids = [
            str(item or "").strip()
            for item in list(summary.get("target_dataset_ids") or [])
            if str(item or "").strip()
        ]
        if not target_ids:
            return []
        per_dataset = summary.get("per_dataset")
        if not isinstance(per_dataset, dict) or not per_dataset:
            if len(target_ids) > 1:
                return ["target dataset metrics are missing for the stage panel"]
            return []

        blockers: List[str] = []
        for dataset_id in target_ids:
            row = per_dataset.get(dataset_id)
            if not isinstance(row, dict):
                blockers.append(f"target dataset metrics are missing: {dataset_id}")
                continue
            error = str(row.get("error") or "").strip()
            if error:
                blockers.append(f"target dataset '{dataset_id}' returned an error: {error}")
            if bool(row.get("training_timed_out")):
                blockers.append(f"target dataset '{dataset_id}' exceeded the training wall-clock budget")
            if bool(row.get("inference_timed_out")):
                blockers.append(f"target dataset '{dataset_id}' exceeded the inference/evaluation wall-clock budget")

            has_numeric_metric = any(
                self._numeric_value(row.get(key)) is not None
                for key in ("primary_value", "secondary_value", "w1_mean", "tmv_mean")
            )
            custom_metrics = row.get("custom_metrics")
            has_custom_metric = isinstance(custom_metrics, dict) and any(
                self._numeric_value(value) is not None for value in custom_metrics.values()
            )
            has_score_vector = any(
                isinstance(row.get(key), list) and bool(row.get(key))
                for key in ("w1_scores", "tmv_scores")
            )
            if not has_numeric_metric and not has_custom_metric and not has_score_vector:
                blockers.append(f"target dataset '{dataset_id}' has no usable metrics")
        return blockers

    @staticmethod
    def _explicit_bool(value: Any) -> Optional[bool]:
        if isinstance(value, bool):
            return value
        return None

    def _tmv_gate_status(
        self,
        *,
        metrics: Optional[Dict[str, Any]] = None,
        policy: Optional[Dict[str, Any]] = None,
        campaign: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Decide whether TMV is a hard gate using explicit structured metadata.

        We intentionally do not parse free-text proposal wording here. If a custom
        algorithm needs to override the automatic model-component rule, pass a
        boolean `tmv_gate_required` in the campaign claim_metric_spec or policy.
        """
        policy_payload = policy if isinstance(policy, dict) else {}
        campaign_payload = campaign if isinstance(campaign, dict) else {}
        claim_spec = campaign_payload.get("claim_metric_spec") if isinstance(campaign_payload.get("claim_metric_spec"), dict) else {}
        metrics_payload = metrics if isinstance(metrics, dict) else {}

        for source_name, payload in (
            ("policy", policy_payload),
            ("campaign", campaign_payload),
            ("claim_metric_spec", claim_spec),
            ("metrics", metrics_payload),
        ):
            for key in ("tmv_gate_required", "requires_tmv_gate", "mass_modeling_enabled", "models_unbalanced_mass"):
                explicit = self._explicit_bool(payload.get(key))
                if explicit is not None:
                    return {"required": explicit, "reason": f"{source_name}.{key}"}

        dataset_metrics = metrics_payload.get("campaign_dataset_metrics")
        if isinstance(dataset_metrics, list) and dataset_metrics:
            child_statuses = [
                self._tmv_gate_status(metrics=dict(item), policy=policy_payload, campaign=campaign_payload)
                for item in dataset_metrics
                if isinstance(item, dict)
            ]
            if child_statuses:
                if any(bool(item.get("required", True)) for item in child_statuses):
                    return {"required": True, "reason": "campaign_dataset_metrics.any_requires_tmv_gate"}
                return {"required": False, "reason": "campaign_dataset_metrics.all_skip_tmv_gate"}

        config = metrics_payload.get("config")
        if isinstance(config, dict):
            model = config.get("model")
            if isinstance(model, dict):
                components = model.get("components")
                if isinstance(components, (list, tuple, set)):
                    normalized = {str(item).strip().lower() for item in components}
                    return {
                        "required": "growth" in normalized,
                        "reason": "metrics.config.model.components",
                        "model_components": sorted(normalized),
                    }

        return {"required": True, "reason": "default_require_tmv_gate_when_mass_modeling_unknown"}

    def _tmv_hard_pass(self, metrics: Dict[str, Any], tmv_max: float) -> bool:
        tmv_scores = metrics.get("tmv_scores")
        if isinstance(tmv_scores, list) and tmv_scores:
            values: List[float] = []
            for item in tmv_scores:
                value = self._numeric_value(item)
                if value is None:
                    return False
                values.append(value)
            return all(value < float(tmv_max) for value in values)
        tmv_mean = self._metric_value(metrics, "tmv_mean")
        return tmv_mean is not None and tmv_mean < float(tmv_max)

    @staticmethod
    def _metric_improves(candidate: float, baseline: float, *, direction: str, min_delta: float, abs_min_delta: float) -> bool:
        rel = max(0.0, float(min_delta or 0.0))
        abs_delta = max(0.0, float(abs_min_delta or 0.0))
        if direction == "greater":
            return candidate > baseline * (1.0 + rel) + abs_delta
        return candidate < baseline * (1.0 - rel) - abs_delta

    @staticmethod
    def _nested_baseline_record(baseline: Dict[str, Any], key: str) -> Dict[str, Any]:
        record = baseline.get(key) if isinstance(baseline, dict) else None
        return dict(record) if isinstance(record, dict) and record else {}

    def _registered_baseline_records_for_metric(
        self,
        baseline: Dict[str, Any],
        metric_name: str,
        *,
        claim_metric: str = "",
    ) -> List[Dict[str, Any]]:
        if not isinstance(baseline, dict):
            return []
        metric = self._normalize_metric_name(metric_name)
        claim = self._normalize_metric_name(claim_metric)
        raw_records = baseline.get("agent_registered_baselines")
        if isinstance(raw_records, dict):
            iterable = list(raw_records.values())
        elif isinstance(raw_records, list):
            iterable = raw_records
        else:
            iterable = []
        records: List[Dict[str, Any]] = []
        for raw in iterable:
            if not isinstance(raw, dict):
                continue
            if str(raw.get("status") or "active").strip().lower() not in {"", "active"}:
                continue
            if bool(raw.get("required_for_gate", True)) is False:
                continue
            candidate = dict(raw)
            raw_metrics = raw.get("metrics") if isinstance(raw.get("metrics"), dict) else {}
            for key, value in raw_metrics.items():
                candidate.setdefault(key, deepcopy(value))
            if isinstance(raw_metrics.get("custom_metrics"), dict):
                custom = candidate.get("custom_metrics") if isinstance(candidate.get("custom_metrics"), dict) else {}
                candidate["custom_metrics"] = {**dict(raw_metrics.get("custom_metrics") or {}), **dict(custom or {})}
            roles = [
                str(item or "").strip().lower()
                for item in list(candidate.get("metric_roles") or [])
                if str(item or "").strip()
            ]
            supports_metric = not roles or "both" in roles or metric in roles
            if metric == "w1_mean":
                supports_metric = supports_metric or "w1" in roles
            if claim and metric == claim:
                supports_metric = supports_metric or "claim" in roles
            if not supports_metric:
                continue
            if self._metric_value(candidate, metric) is None:
                continue
            records.append(candidate)
        return records

    def _baseline_record_for_metric(
        self,
        baseline: Dict[str, Any],
        metric_name: str,
        *,
        claim_metric: str = "",
        direction: str = "",
    ) -> Dict[str, Any]:
        metric = self._normalize_metric_name(metric_name)
        claim = self._normalize_metric_name(claim_metric)
        candidates: List[Dict[str, Any]] = []
        if metric == "w1_mean":
            base = self._nested_baseline_record(baseline, "w1_sota_baseline_metrics") or dict(baseline or {})
        elif claim and metric == claim:
            base = self._nested_baseline_record(baseline, "claim_sota_baseline_metrics") or dict(baseline or {})
        else:
            base = dict(baseline or {})
        if self._metric_value(base, metric) is not None:
            candidates.append(base)
        candidates.extend(
            self._registered_baseline_records_for_metric(
                baseline,
                metric,
                claim_metric=claim,
            )
        )
        if not candidates:
            return base
        metric_direction = self._normalize_metric_direction(
            direction,
            default="lower" if metric in {"w1_mean", "tmv_mean"} else "greater",
        )
        reverse = metric_direction == "greater"
        candidates.sort(
            key=lambda item: self._metric_value(item, metric)
            if self._metric_value(item, metric) is not None
            else (-1e300 if reverse else 1e300),
            reverse=reverse,
        )
        selected = dict(candidates[0])
        selected["selected_from_agent_registered_baselines"] = (
            str(selected.get("baseline_kind") or "") == "agent_registered_control"
        )
        return selected

    @staticmethod
    def _secondary_within_tolerance(candidate: Optional[float], baseline: Optional[float], *, direction: str, tolerance: float, abs_tolerance: float) -> bool:
        if candidate is None or baseline is None or not direction:
            return True
        tol = max(0.0, float(tolerance or 0.0))
        abs_tol = max(0.0, float(abs_tolerance or 0.0))
        if direction == "greater":
            return candidate >= baseline * (1.0 - tol) - abs_tol
        return candidate <= baseline * (1.0 + tol) + abs_tol

    def _save_campaign(self, campaign: Dict[str, Any]) -> None:
        self._ensure_campaign_stage_policy_defaults(campaign)
        for derived_key in (
            "stage_statuses",
            "current_stage_status",
            "stage_internal_status",
            "stage_gate_status",
            "user_facing_status",
            "next_required_action",
        ):
            campaign.pop(derived_key, None)
        algo_id = str(campaign.get("algorithm_id") or "").strip().lower()
        campaign_id = str(campaign.get("campaign_id") or "").strip()
        self._ensure_algorithm_write_owner(algo_id, action="save_campaign")
        campaign["updated_at"] = self._now_iso()
        self._write_json(self._campaign_index_path(algo_id, campaign_id), campaign)
        self._sync_campaign_summary_to_state(campaign)

    def _sync_campaign_summary_to_state(self, campaign: Dict[str, Any]) -> None:
        algo_id = str(campaign.get("algorithm_id") or "").strip().lower()
        campaign_id = str(campaign.get("campaign_id") or "").strip()
        if not algo_id or not campaign_id:
            return
        stage = str(campaign.get("current_stage") or "")
        stage_state = (
            (campaign.get("stages") or {}).get(stage)
            if isinstance(campaign.get("stages"), dict)
            else {}
        )
        stage_state = stage_state if isinstance(stage_state, dict) else {}
        policy = self._campaign_stage_policy(campaign, stage) if stage else {}
        budget = self._campaign_stage_budget(stage_state, policy) if stage else {}
        status_view = self._campaign_stage_status_view(campaign, stage) if stage else {}
        self.state["active_algorithm_campaign_id"] = campaign_id
        self.state["active_algorithm_campaign"] = {
            "campaign_id": campaign_id,
            "algorithm_id": algo_id,
            "status": str(campaign.get("status") or ""),
            "current_stage": stage,
            "current_trial_id": str(campaign.get("current_trial_id") or ""),
            "active_best_trial_id": str(stage_state.get("active_best_trial_id") or ""),
            "gate_ready": bool(stage_state.get("gate_ready", False)),
            "last_gate_check": dict(stage_state.get("last_gate_check") or {}),
            "trial_count": int(budget.get("trial_count") or stage_state.get("trial_count") or 0),
            "promote_count": int(stage_state.get("promote_count") or 0),
            "reject_count": int(stage_state.get("reject_count") or 0),
            "max_trials": budget.get("max_trials"),
            "remaining_trials": budget.get("remaining_trials"),
            "budget_exhausted": bool(budget.get("budget_exhausted", False)),
            "algorithm_validated": bool(campaign.get("algorithm_validated", False)),
            "registry_path": str(self._campaign_index_path(algo_id, campaign_id)),
            "stage_internal_status": str(status_view.get("stage_internal_status") or ""),
            "stage_gate_status": str(status_view.get("stage_gate_status") or ""),
            "user_facing_status": str(status_view.get("user_facing_status") or ""),
            "next_required_action": str(status_view.get("next_required_action") or ""),
        }

    def _campaign_stage_status_view(self, campaign: Dict[str, Any], stage: str) -> Dict[str, Any]:
        stage_name = str(stage or "").strip()
        stages = campaign.get("stages") if isinstance(campaign.get("stages"), dict) else {}
        stage_state = dict(stages.get(stage_name) or {}) if stage_name else {}
        policy = self._campaign_stage_policy(campaign, stage_name) if stage_name else {}
        budget = self._campaign_stage_budget(stage_state, policy) if stage_name else {}
        active_best_trial_id = str(stage_state.get("active_best_trial_id") or "").strip()
        current_trial_id = str(campaign.get("current_trial_id") or "").strip()
        current_stage = str(campaign.get("current_stage") or "").strip()
        status = str(stage_state.get("status") or "").strip() or ("active" if stage_name == current_stage else "pending")
        baseline = dict(stage_state.get("external_baseline_metrics") or {})
        last_gate = dict(stage_state.get("last_gate_check") or {})
        blockers = [str(item) for item in list(last_gate.get("blockers") or []) if str(item).strip()]
        stage_has_external_gate = stage_name != "stage3_tuning"

        if current_trial_id and stage_name == current_stage:
            internal_status = "trial_open"
        elif active_best_trial_id:
            internal_status = "active_best_promoted"
        elif status == "failed_needs_revision":
            internal_status = "failed_needs_revision"
        else:
            internal_status = "no_active_best"

        gate_ok = bool(last_gate.get("ok")) or bool(stage_state.get("gate_ready"))
        stage_gate_evidence = (
            stage_state.get("stage_gate_evidence")
            if isinstance(stage_state.get("stage_gate_evidence"), dict)
            else {}
        )
        stage3_optional_gate = {}
        if stage_name == "stage3_tuning":
            stage3_optional_gate = dict(
                ((stage_gate_evidence.get("checks") or {}).get("stage3_optional_sota_gate") or {})
                if isinstance(stage_gate_evidence.get("checks"), dict)
                else {}
            )
        if stage_name == "stage3_tuning" and blockers:
            gate_status = "blocked_min_tuning_trials" if any("completed trial" in item for item in blockers) else "blocked_failed_checks"
        elif stage_name == "stage3_tuning" and gate_ok and bool(stage3_optional_gate.get("ok")):
            gate_status = "passed_optional_sota_gate"
        elif stage_name == "stage3_tuning" and gate_ok and bool(stage3_optional_gate.get("budget_exhausted")):
            gate_status = "budget_exhausted_optional_gate_not_met"
        elif stage_name == "stage3_tuning":
            gate_status = "not_required_optional_tuning"
        elif gate_ok:
            gate_status = "passed"
        elif not active_best_trial_id:
            gate_status = "blocked_no_active_best"
        elif stage_has_external_gate and not baseline:
            gate_status = "blocked_missing_external_baseline"
        elif blockers:
            blocker_text = " | ".join(blockers).lower()
            if "missing primary metric" in blocker_text or "claim" in blocker_text:
                gate_status = "blocked_missing_baseline_claim_metric"
            elif "panel" in blocker_text:
                gate_status = "blocked_panel_mismatch"
            else:
                gate_status = "blocked_failed_checks"
        else:
            gate_status = "not_checked"

        primary_metric = str(policy.get("primary_metric") or "")
        if current_trial_id and stage_name == current_stage:
            next_action = f"finish or run open campaign trial `{current_trial_id}`"
        elif not active_best_trial_id:
            next_action = "run_campaign_trial(...) to create the first successful active-best trial"
        elif gate_status == "blocked_missing_external_baseline":
            if stage_name == "stage2_claim_validation" and primary_metric not in {"", "w1_mean"}:
                next_action = (
                    "compute claim_metric for external baselines on the frozen stage panel "
                    "using refresh_campaign_stage_baselines(...) or compute_campaign_claim_metric_for_baselines(...)"
                )
            else:
                next_action = "refresh_campaign_stage_baselines(...) for the frozen stage panel"
        elif gate_status == "blocked_missing_baseline_claim_metric":
            next_action = (
                "provide claim_metric_spec.evaluator_path/adapters if missing, then recompute baseline claim metrics"
            )
        elif gate_status == "blocked_panel_mismatch":
            next_action = "refresh or reset baselines so their target_dataset_ids match the frozen stage panel"
        elif gate_status == "blocked_min_tuning_trials":
            next_action = "complete the minimum Stage 3 tuning trials on the frozen panel before moving to final regression"
        elif gate_status == "passed_optional_sota_gate" and stage_name == current_stage:
            next_action = "Stage 3 optional SOTA gate passed; call check_campaign_stage_gate(..., advance=true) to move to final regression"
        elif gate_status == "budget_exhausted_optional_gate_not_met" and stage_name == current_stage:
            next_action = "Stage 3 budget is exhausted; proceed to final_regression from the active best"
        elif gate_status == "passed" and stage_name == current_stage:
            next_action = "call check_campaign_stage_gate(..., advance=true) to advance or lock when ready"
        elif stage_name == "stage3_tuning":
            next_action = "continue optional tuning or proceed to final_regression from the active best"
        else:
            next_action = ""

        if gate_status == "passed":
            user_status = f"{stage_name}: active-best trial promoted and formal stage gate passed."
        elif stage_name == "stage3_tuning" and gate_status == "passed_optional_sota_gate":
            user_status = f"{stage_name}: optional SOTA gate passed; active best is ready for final regression."
        elif stage_name == "stage3_tuning" and gate_status == "budget_exhausted_optional_gate_not_met":
            user_status = f"{stage_name}: tuning budget exhausted without optional SOTA gate; algorithm remains validated and can proceed to final regression."
        elif stage_name == "stage3_tuning" and gate_status == "not_required_optional_tuning":
            user_status = f"{stage_name}: optional tuning stage; active best is enough to continue when tuning budget is no longer useful."
        elif stage_name == "stage3_tuning":
            reason = blockers[0] if blockers else gate_status.replace("_", " ")
            user_status = f"{stage_name}: optional tuning stage, but final-regression advance is blocked for now ({reason})."
        elif internal_status == "active_best_promoted" and gate_status.startswith("blocked"):
            reason = blockers[0] if blockers else gate_status.replace("_", " ")
            user_status = (
                f"{stage_name}: trial promoted internally, but formal stage gate is blocked ({reason})."
            )
        elif internal_status == "trial_open":
            user_status = f"{stage_name}: trial is open and must finish before gate status is meaningful."
        else:
            user_status = f"{stage_name}: no promoted active-best trial yet."

        return {
            "stage": stage_name,
            "stage_status": status,
            "stage_internal_status": internal_status,
            "stage_gate_status": gate_status,
            "user_facing_status": user_status,
            "next_required_action": next_action,
            "active_best_trial_id": active_best_trial_id,
            "current_trial_id": current_trial_id if stage_name == current_stage else "",
            "gate_ready": bool(stage_state.get("gate_ready", False)),
            "last_gate_check": last_gate,
            "blockers": blockers,
            "primary_metric": primary_metric,
            "target_dataset_ids": list(
                (stage_state.get("stage_panel") or {}).get("target_dataset_ids") or []
            )
            if isinstance(stage_state.get("stage_panel"), dict)
            else [],
            **budget,
        }

    def _load_campaign(self, campaign_id: str = "") -> Dict[str, Any]:
        target = str(campaign_id or self.state.get("active_algorithm_campaign_id") or "").strip()
        if not target:
            raise ValueError("campaign_id is required because no active algorithm campaign is bound.")
        active = dict(self.state.get("active_algorithm_campaign") or {})
        active_algo = str(active.get("algorithm_id") or "").strip().lower()
        if active_algo:
            path = self._campaign_index_path(active_algo, target)
            if path.exists():
                payload = self._read_json(path, {})
                if isinstance(payload, dict) and payload.get("campaign_id") == target:
                    payload = self._ensure_campaign_stage_policy_defaults(payload)
                    self._sync_campaign_summary_to_state(payload)
                    return payload
        for path in self._campaigns_root().glob(f"*/{target}/campaign.json"):
            payload = self._read_json(path, {})
            if isinstance(payload, dict) and payload.get("campaign_id") == target:
                payload = self._ensure_campaign_stage_policy_defaults(payload)
                self._sync_campaign_summary_to_state(payload)
                return payload
        raise FileNotFoundError(f"Campaign '{target}' was not found under {self._campaigns_root()}.")

    def _ensure_campaign_mutation_matches_active_binding(
        self,
        requested_campaign: Dict[str, Any],
        *,
        requested_campaign_id: str,
        previously_bound_campaign_id: str,
        action: str,
    ) -> None:
        """Prevent stale sibling campaigns from receiving new trial/gate evidence."""
        requested_id = str(requested_campaign.get("campaign_id") or requested_campaign_id or "").strip()
        bound_id = str(previously_bound_campaign_id or "").strip()
        if not requested_id or not bound_id or requested_id == bound_id:
            return
        bound = self._load_campaign(bound_id)
        bound_status = str(bound.get("status") or "").strip()
        bound_algo = str(bound.get("algorithm_id") or "").strip().lower()
        requested_algo = str(requested_campaign.get("algorithm_id") or "").strip().lower()
        if bound_status in {"active", "running"} and (not requested_algo or requested_algo == bound_algo):
            raise ValueError(
                f"Refusing to {action} campaign '{requested_id}' because this session is already "
                f"bound to active campaign '{bound_id}' for algorithm '{bound_algo}'. This usually "
                "means a stale campaign id from an older session or review resume was reused. "
                "Continue the active campaign by omitting campaign_id, or explicitly abort/archive "
                "the stale active campaign before switching."
            )

    @staticmethod
    def _ensure_campaign_mutable_status(campaign: Dict[str, Any], *, action: str) -> None:
        status = str((campaign or {}).get("status") or "").strip().lower()
        if status in {"failed", "locked", "complete", "completed", "aborted", "archived", "inactive"}:
            campaign_id = str((campaign or {}).get("campaign_id") or "").strip()
            algo_id = str((campaign or {}).get("algorithm_id") or "").strip().lower()
            raise ValueError(
                f"Refusing to {action} campaign '{campaign_id}' for algorithm '{algo_id}' because "
                f"campaign status is '{status}'. Start or continue an active sibling campaign instead "
                "of mutating failed/locked historical evidence."
            )

    def _more_advanced_active_campaign(self, campaign: Dict[str, Any]) -> Dict[str, Any]:
        """Find a sibling active campaign that should be continued instead."""
        algo_id = str(campaign.get("algorithm_id") or "").strip().lower()
        campaign_id = str(campaign.get("campaign_id") or "").strip()
        proposal_id = str(campaign.get("proposal_id") or "").strip()
        if not algo_id or not campaign_id:
            return {}

        def _progress_key(item: Dict[str, Any]) -> tuple:
            stage = str(item.get("current_stage") or CAMPAIGN_STAGE_ORDER[0])
            try:
                stage_index = CAMPAIGN_STAGE_ORDER.index(stage)
            except ValueError:
                stage_index = -1
            stages = item.get("stages") if isinstance(item.get("stages"), dict) else {}
            stage_state = stages.get(stage) if isinstance(stages.get(stage), dict) else {}
            trials = item.get("trials") if isinstance(item.get("trials"), dict) else {}
            active_best = 1 if str(stage_state.get("active_best_trial_id") or "").strip() else 0
            gate_ready = 1 if bool(stage_state.get("gate_ready")) else 0
            return (
                1 if bool(item.get("algorithm_validated")) else 0,
                stage_index,
                gate_ready,
                active_best,
                len(trials),
            )

        current_key = _progress_key(campaign)
        best: Dict[str, Any] = {}
        best_key: tuple = current_key
        root = self._campaign_algorithm_dir(algo_id)
        if not root.exists():
            return {}
        for path in root.glob("*/campaign.json"):
            other = self._read_json(path, {})
            if not isinstance(other, dict):
                continue
            other_id = str(other.get("campaign_id") or "").strip()
            if not other_id or other_id == campaign_id:
                continue
            if str(other.get("status") or "").strip() != "active":
                continue
            other_proposal = str(other.get("proposal_id") or "").strip()
            if proposal_id and other_proposal and other_proposal != proposal_id:
                continue
            other_key = _progress_key(other)
            if other_key > best_key:
                best = other
                best_key = other_key
        return best

    def _empty_campaign_stages_from_existing(self, campaign: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        old_stages = campaign.get("stages") if isinstance(campaign.get("stages"), dict) else {}
        stages: Dict[str, Dict[str, Any]] = {}
        for idx, stage in enumerate(CAMPAIGN_STAGE_ORDER):
            previous = old_stages.get(stage) if isinstance(old_stages.get(stage), dict) else {}
            stages[stage] = {
                "stage": stage,
                "status": "active" if idx == 0 else "pending",
                "active_best_trial_id": "",
                "active_best_commit": "",
                "active_best_snapshot_id": "",
                "active_best_run_ids": [],
                "trial_count": 0,
                "promote_count": 0,
                "reject_count": 0,
                "external_baseline_metrics": dict(previous.get("external_baseline_metrics") or {}),
            }
        return stages

    def _reset_active_campaigns_for_new_proposal(
        self,
        algorithm_id: str,
        *,
        previous_proposal_id: str,
        proposal_id: str,
        reason: str = "",
    ) -> List[Dict[str, Any]]:
        algo_id = str(algorithm_id or "").strip().lower()
        previous_id = str(previous_proposal_id or "").strip()
        next_id = str(proposal_id or "").strip()
        if not algo_id or not previous_id or not next_id or previous_id == next_id:
            return []

        root = self._campaign_algorithm_dir(algo_id)
        if not root.exists():
            return []

        reset_payloads: List[Dict[str, Any]] = []
        for index_path in sorted(root.glob("*/campaign.json")):
            campaign = self._read_json(index_path, {})
            if not isinstance(campaign, dict):
                continue
            if str(campaign.get("status") or "").strip().lower() == "locked":
                continue
            campaign_proposal = str(campaign.get("proposal_id") or "").strip()
            if campaign_proposal and campaign_proposal != previous_id:
                continue

            old_stage = str(campaign.get("current_stage") or CAMPAIGN_STAGE_ORDER[0])
            old_stages = dict(campaign.get("stages") or {})
            old_stage_state = dict(old_stages.get(old_stage) or {})
            old_trial_id = str(campaign.get("current_trial_id") or "").strip()
            trials = campaign.setdefault("trials", {})
            if old_trial_id and isinstance(trials.get(old_trial_id), dict):
                trials[old_trial_id]["status"] = "abandoned_due_to_proposal_revision"
                trials[old_trial_id]["decision"] = "abandoned"
                trials[old_trial_id]["updated_at"] = self._now_iso()
                trials[old_trial_id]["decision_reasons"] = [
                    f"proposal revised from {previous_id} to {next_id}; campaign reset to stage1"
                ]

            reset_entry = {
                "reset_at": self._now_iso(),
                "reason": str(reason or "").strip() or "proposal revision changed algorithm semantics",
                "previous_proposal_id": previous_id,
                "proposal_id": next_id,
                "previous_stage": old_stage,
                "previous_active_best_trial_id": str(old_stage_state.get("active_best_trial_id") or ""),
                "previous_active_best_snapshot_id": str(old_stage_state.get("active_best_snapshot_id") or ""),
                "abandoned_trial_id": old_trial_id,
            }
            history = list(campaign.get("reset_history") or [])
            history.append(reset_entry)
            campaign["reset_history"] = history[-20:]
            campaign["proposal_id"] = next_id
            campaign["status"] = "active"
            for failure_key in (
                "failed_at",
                "failure_reason",
                "failure_stage",
                "failure_trial_id",
                "failed_trial_id",
                "failure_report_path",
            ):
                campaign.pop(failure_key, None)
            if str(campaign.get("algorithm_lifecycle_status") or "").strip().lower() == "failed":
                campaign["algorithm_lifecycle_status"] = "developing"
            campaign["current_stage"] = CAMPAIGN_STAGE_ORDER[0]
            campaign["current_trial_id"] = ""
            campaign["stages"] = self._empty_campaign_stages_from_existing(campaign)
            campaign["trials"] = trials
            campaign["algorithm_validated"] = False
            campaign["validated_at"] = ""
            self._save_campaign(campaign)

            payload = {
                "campaign_id": str(campaign.get("campaign_id") or ""),
                "algorithm_id": algo_id,
                "previous_proposal_id": previous_id,
                "proposal_id": next_id,
                "previous_stage": old_stage,
                "stage": CAMPAIGN_STAGE_ORDER[0],
                "abandoned_trial_id": old_trial_id,
                "reason": reset_entry["reason"],
            }
            self._emit("algorithm_campaign_reset", payload)
            reset_payloads.append(payload)
        return reset_payloads

    def _ensure_campaign_archive(self, campaign: Dict[str, Any]) -> None:
        archive = dict(campaign.get("archive") or {})
        archive_path = Path(str(archive.get("archive_path") or "")).expanduser()
        worktree_path = Path(str(archive.get("worktree_path") or "")).expanduser()
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        worktree_path.mkdir(parents=True, exist_ok=True)
        if not (archive_path / "HEAD").exists():
            subprocess.run(
                ["git", "init", "--bare", str(archive_path)],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

    def _archive_campaign_workspace(
        self,
        campaign: Dict[str, Any],
        *,
        trial_id: str,
        message: str,
        dataset_config_overrides: Optional[Dict[str, Any]] = None,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> str:
        self._ensure_campaign_archive(campaign)
        algo_id = str(campaign.get("algorithm_id") or "").strip().lower()
        archive = dict(campaign.get("archive") or {})
        archive_path = Path(str(archive.get("archive_path") or "")).expanduser()
        worktree_path = Path(str(archive.get("worktree_path") or "")).expanduser()
        if worktree_path.exists():
            shutil.rmtree(worktree_path)
        worktree_path.mkdir(parents=True, exist_ok=True)
        workspace_dir = worktree_path / "workspace"
        workspace_dir.mkdir(parents=True, exist_ok=True)
        for src in self._tracked_workspace_paths(algo_id):
            if src.exists():
                shutil.copy2(src, workspace_dir / src.name)
        meta = {
            "campaign_id": str(campaign.get("campaign_id") or ""),
            "algorithm_id": algo_id,
            "trial_id": str(trial_id or ""),
            "stage": str(campaign.get("current_stage") or ""),
            "created_at": self._now_iso(),
            **dict(extra_meta or {}),
        }
        self._write_json(worktree_path / "trial_meta.json", meta)
        self._write_json(worktree_path / "dataset_config_overrides.json", dict(dataset_config_overrides or {}))
        git_base = [
            "git",
            "-c",
            "user.name=CytoBridge Campaign",
            "-c",
            "user.email=campaign@cellcompass.local",
            f"--git-dir={archive_path}",
            f"--work-tree={worktree_path}",
        ]
        subprocess.run(git_base + ["add", "-A"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        subprocess.run(
            git_base + ["commit", "--allow-empty", "-m", str(message or f"campaign trial {trial_id}")],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        rev = subprocess.run(
            git_base + ["rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return rev.stdout.strip()

    def _tracked_workspace_paths(self, algorithm_id: str) -> List[Path]:
        algo_dir = self._proposal_dir(algorithm_id)
        return [algo_dir / name for name in REGISTRY_TRACKED_WORKSPACE_FILES]

    @staticmethod
    def _empty_active_algorithm_context() -> Dict[str, Any]:
        return {
            "algorithm_id": "",
            "algorithm_lifecycle_status": "developing",
            "algorithm_lifecycle_status_reason": "",
            "completed_campaign_id": "",
            "workspace_path": "",
            "editable_proposal_path": "",
            "proposal_id": "",
            "proposal_status": "",
            "proposal_claimed_capability": "",
            "expected_evaluation_outcome": "",
            "algorithm_attributes": {},
            "primary_idea_id": "",
            "registry_path": "",
            "active_snapshot_id": "",
            "dirty_since_snapshot": False,
            "dirty_paths": [],
            "last_mutation_at": "",
            "last_verified_at": "",
        }

    @staticmethod
    def _default_workspace_dirty() -> Dict[str, Any]:
        return {
            "dirty_since_snapshot": False,
            "dirty_paths": [],
            "last_mutation_at": "",
            "reason": "",
        }

    def _current_active_algorithm_id(self) -> str:
        ctx = self.state.get("active_algorithm_context")
        if isinstance(ctx, dict):
            active = str(ctx.get("algorithm_id") or "").strip().lower()
            if active:
                return active
        return ""

    def _is_current_active_algorithm(self, algorithm_id: str) -> bool:
        algo_id = str(algorithm_id or "").strip().lower()
        return bool(algo_id and self._current_active_algorithm_id() == algo_id)

    def _proposal_record_from_registry(self, algorithm_id: str, registry: Dict[str, Any], proposal_id: str) -> Dict[str, Any]:
        algo_id = str(algorithm_id or "").strip().lower()
        target_id = str(proposal_id or "").strip()
        if not target_id:
            return {}
        summary = dict((registry.get("proposals") or {}).get(target_id) or {})
        registry_path = str(summary.get("registry_path") or "").strip()
        if registry_path:
            loaded = self._read_json(Path(registry_path), {})
            if isinstance(loaded, dict) and str(loaded.get("proposal_id") or "").strip() == target_id:
                record = dict(summary)
                record.update(loaded)
                record["editable_proposal_path"] = str(record.get("editable_proposal_path") or self._proposal_dir(algo_id) / "PROPOSAL.md")
                return record
        current = self._get_proposal_record(algo_id)
        if isinstance(current, dict) and str(current.get("proposal_id") or "").strip() == target_id:
            record = dict(summary)
            record.update(current)
            record["editable_proposal_path"] = str(record.get("editable_proposal_path") or self._proposal_dir(algo_id) / "PROPOSAL.md")
            return record
        summary["editable_proposal_path"] = str(summary.get("editable_proposal_path") or self._proposal_dir(algo_id) / "PROPOSAL.md")
        return summary

    def _build_active_algorithm_context(self, algorithm_id: str, registry: Dict[str, Any]) -> Dict[str, Any]:
        algo_id = str(algorithm_id or "").strip().lower()
        if not algo_id:
            return self._empty_active_algorithm_context()
        proposal_id = str(registry.get("active_proposal_id") or "").strip()
        active_proposal_summary = self._proposal_record_from_registry(algo_id, registry, proposal_id)
        proposal_payload = (
            active_proposal_summary.get("proposal")
            if isinstance(active_proposal_summary.get("proposal"), dict)
            else {}
        )
        algorithm_attributes = (
            active_proposal_summary.get("algorithm_attributes")
            if isinstance(active_proposal_summary.get("algorithm_attributes"), dict)
            else proposal_payload.get("algorithm_attributes")
            if isinstance(proposal_payload.get("algorithm_attributes"), dict)
            else {}
        )
        dirty = dict(registry.get("workspace_dirty") or self._default_workspace_dirty())
        snapshot_id = str(registry.get("active_workspace_snapshot_id") or "").strip()
        return {
            "algorithm_id": algo_id,
            "workspace_path": str(self._proposal_dir(algo_id)),
            "editable_proposal_path": str(active_proposal_summary.get("editable_proposal_path") or self._proposal_dir(algo_id) / "PROPOSAL.md"),
            "proposal_id": proposal_id,
            "proposal_status": str(active_proposal_summary.get("status") or "").strip(),
            "proposal_claimed_capability": str(proposal_payload.get("claimed_capability") or "").strip(),
            "expected_evaluation_outcome": str(proposal_payload.get("expected_evaluation_outcome") or "").strip(),
            "algorithm_attributes": dict(algorithm_attributes or {}),
            "primary_idea_id": str(active_proposal_summary.get("primary_idea_id") or "").strip().lower(),
            "registry_path": str(self._registry_index_path(algo_id)),
            "active_snapshot_id": snapshot_id,
            "active_workspace_snapshot_id": snapshot_id,
            "active_baseline_run_id": str(registry.get("active_baseline_run_id") or ""),
            "algorithm_lifecycle_status": str(registry.get("algorithm_lifecycle_status") or "developing"),
            "algorithm_lifecycle_status_reason": str(registry.get("algorithm_lifecycle_status_reason") or ""),
            "completed_campaign_id": str(registry.get("completed_campaign_id") or ""),
            "dirty_since_snapshot": bool(dirty.get("dirty_since_snapshot")),
            "dirty_paths": [str(path) for path in (dirty.get("dirty_paths") or []) if str(path).strip()],
            "last_mutation_at": str(dirty.get("last_mutation_at") or ""),
            "last_verified_at": str(registry.get("last_verified_at") or ""),
        }

    def _sync_registry_summary_to_state(
        self,
        algorithm_id: str,
        registry: Dict[str, Any],
        *,
        update_active_context: bool = False,
    ) -> None:
        algo_id = str(algorithm_id or "").strip().lower()
        if not algo_id:
            return
        if not update_active_context and not self._is_current_active_algorithm(algo_id):
            return
        active_context = self._build_active_algorithm_context(algo_id, registry)
        self.state["active_experiment_registry"] = {
            "algorithm_id": algo_id,
            "path": str(self._registry_index_path(algo_id)),
            "active_proposal_id": str(registry.get("active_proposal_id") or ""),
            "active_primary_idea_id": str(active_context.get("primary_idea_id") or ""),
            "active_workspace_snapshot_id": str(registry.get("active_workspace_snapshot_id") or ""),
            "active_baseline_run_id": str(registry.get("active_baseline_run_id") or ""),
            "latest_accepted_run_id": str(registry.get("latest_accepted_run_id") or ""),
            "algorithm_lifecycle_status": str(registry.get("algorithm_lifecycle_status") or "developing"),
            "algorithm_lifecycle_status_reason": str(registry.get("algorithm_lifecycle_status_reason") or ""),
            "completed_campaign_id": str(registry.get("completed_campaign_id") or ""),
            "latest_review_verdict": str(registry.get("latest_review_verdict") or ""),
            "latest_preview": dict(registry.get("latest_preview") or {}),
            "updated_at": str(registry.get("updated_at") or ""),
        }
        self.state["active_algorithm_context"] = active_context
        self.state["planner_algorithm_workspace"] = str(active_context.get("workspace_path") or "")
        self.state["active_proposal_id"] = str(registry.get("active_proposal_id") or "")
        self.state["active_workspace_snapshot_id"] = str(registry.get("active_workspace_snapshot_id") or "")
        self.state["active_baseline_run_id"] = str(registry.get("active_baseline_run_id") or "")
        if str(registry.get("active_proposal_id") or "").strip():
            self.state["latest_algorithm_proposal_id"] = algo_id
        recent_decisions = list(registry.get("recent_decisions") or [])
        recent_obsolete = list(registry.get("recent_obsolete") or [])
        self.state["decision_log_summary"] = recent_decisions[-10:]
        self.state["obsolete_results_summary"] = recent_obsolete[-10:]

    def set_active_algorithm_context(self, algorithm_id: str, registry: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        algo_id = str(algorithm_id or "").strip().lower()
        if not algo_id:
            ctx = self._empty_active_algorithm_context()
            self.state["active_algorithm_context"] = ctx
            return ctx
        target_registry = registry if isinstance(registry, dict) else self._bootstrap_algorithm_registry(algo_id)
        self._sync_registry_summary_to_state(algo_id, target_registry, update_active_context=True)
        return dict(self.state.get("active_algorithm_context") or self._empty_active_algorithm_context())

    def get_active_algorithm_context(self) -> Dict[str, Any]:
        ctx = self.state.get("active_algorithm_context")
        if isinstance(ctx, dict) and str(ctx.get("algorithm_id") or "").strip():
            algo_id = str(ctx.get("algorithm_id") or "").strip().lower()
            if self._proposal_dir(algo_id).exists():
                registry = self._bootstrap_algorithm_registry(algo_id)
                return self.set_active_algorithm_context(algo_id, registry)
            normalized = self._empty_active_algorithm_context()
            normalized.update(ctx)
            self.state["active_algorithm_context"] = normalized
            return normalized

        candidates: List[str] = []
        workspace_path = str(self.state.get("planner_algorithm_workspace") or "").strip()
        if workspace_path:
            candidates.append(Path(workspace_path).name.strip().lower())
        active_registry = dict(self.state.get("active_experiment_registry") or {})
        candidates.append(str(active_registry.get("algorithm_id") or "").strip().lower())
        candidates.append(str(self.state.get("latest_algorithm_proposal_id") or "").strip().lower())
        candidates.append(str(self.state.get("latest_training_algorithm_id") or "").strip().lower())
        for candidate in candidates:
            if candidate and self._proposal_dir(candidate).exists():
                registry = self._bootstrap_algorithm_registry(candidate)
                return self.set_active_algorithm_context(candidate, registry)

        empty = self._empty_active_algorithm_context()
        self.state["active_algorithm_context"] = empty
        return empty

    def resolve_algorithm_context(self, algorithm_id: str) -> Dict[str, Any]:
        algo_id = str(algorithm_id or "").strip().lower()
        if not algo_id:
            return self.get_active_algorithm_context()
        registry = self._bootstrap_algorithm_registry(algo_id)
        return self._build_active_algorithm_context(algo_id, registry)

    def verify_algorithm_context(self, algorithm_id: str, stage: str) -> Dict[str, Any]:
        algo_id = str(algorithm_id or "").strip().lower()
        stage_value = str(stage or "").strip().lower()
        context = self.resolve_algorithm_context(algo_id)
        blockers: List[str] = []
        workspace_path = Path(str(context.get("workspace_path") or "")).expanduser()
        if not algo_id:
            blockers.append("algorithm_id is required")
        if not workspace_path.exists():
            blockers.append(f"algorithm workspace does not exist: {workspace_path}")
        if stage_value in {"authoring", "review", "training"}:
            status = str(context.get("proposal_status") or "").strip().lower()
            if status not in {"approved", "approved_auto"}:
                blockers.append(f"approved proposal required for '{algo_id}' (status={status or 'missing'})")
        if stage_value in {"review", "training"}:
            if not str(context.get("active_snapshot_id") or "").strip():
                blockers.append(f"clean workspace snapshot required for '{algo_id}' before {stage_value}")
            if bool(context.get("dirty_since_snapshot")):
                dirty_paths = list(context.get("dirty_paths") or [])
                suffix = f": {', '.join(dirty_paths[:6])}" if dirty_paths else ""
                blockers.append(f"workspace for '{algo_id}' has unsnapshotted edits{suffix}")

        ok = not blockers
        result = {
            "ok": ok,
            "stage": stage_value,
            "algorithm_id": algo_id,
            "active_algorithm_id": self._current_active_algorithm_id(),
            "target_algorithm_id": algo_id,
            "context": context,
            "blockers": blockers,
        }
        if ok and algo_id:
            registry = self._bootstrap_algorithm_registry(algo_id)
            registry["last_verified_at"] = self._now_iso()
            self._save_algorithm_registry(algo_id, registry, sync_active_context=self._is_current_active_algorithm(algo_id))
            result["context"] = self._build_active_algorithm_context(algo_id, registry)
            self._emit("algorithm_context_verified", result)
        elif not ok:
            self._emit("algorithm_context_gate_blocked", result)
        return result

    def snapshot_active_algorithm_workspace(self, reason: str) -> str:
        ctx = self.get_active_algorithm_context()
        algo_id = str(ctx.get("algorithm_id") or "").strip().lower()
        if not algo_id:
            raise ValueError("No active algorithm context. Call activate_algorithm_workspace(...) first.")
        return self.create_workspace_snapshot(
            algo_id,
            reason=str(reason or "").strip() or "Manual active workspace snapshot",
            source="active_workspace_snapshot",
            proposal_id=str(ctx.get("proposal_id") or ""),
            set_active=True,
            allow_empty=False,
        )

    def _sync_research_idea_summary_to_state(
        self,
        idea_id: str,
        registry: Dict[str, Any],
        *,
        record: Optional[Dict[str, Any]] = None,
    ) -> None:
        target = str(idea_id or "").strip().lower()
        if not target:
            return
        current_record = dict(record or self._research_idea_store().get(target) or {})
        if not current_record:
            current_record = self._read_json(self._research_idea_json_path(target), {})
        if not isinstance(current_record, dict):
            current_record = {}
        self.state["active_idea_registry"] = {
            "idea_id": target,
            "path": str(self._research_idea_registry_index_path(target)),
            "current_revision_id": str(registry.get("current_revision_id") or ""),
            "review_status": str(registry.get("last_review_status") or current_record.get("review_status") or ""),
            "portfolio_status": str(registry.get("portfolio_status") or current_record.get("portfolio_status") or ""),
            "resolution_status": str(registry.get("resolution_status") or current_record.get("resolution_status") or ""),
            "execution_status": str(registry.get("execution_status") or current_record.get("execution_status") or ""),
            "linked_algorithms": list(registry.get("linked_algorithms") or []),
            "attempt_count": int(registry.get("attempt_count") or 0),
            "updated_at": str(registry.get("updated_at") or ""),
        }
        self.state["latest_research_idea_id"] = target
        if bool(current_record.get("active")) or str(self.state.get("active_research_idea_id") or "").strip().lower() == target:
            self.state["active_research_idea_id"] = target

    def _default_research_idea_registry(self, idea_id: str) -> Dict[str, Any]:
        now = self._now_iso()
        return {
            "registry_version": 1,
            "idea_id": str(idea_id or "").strip().lower(),
            "created_at": now,
            "updated_at": now,
            "current_revision_id": "",
            "last_review_status": "",
            "portfolio_status": "active",
            "resolution_status": "unresolved",
            "execution_status": "idle",
            "linked_algorithms": [],
            "attempt_count": 0,
            "revisions": {},
            "recent_attempts": [],
            "recent_decisions": [],
            "recent_obsolete": [],
        }

    def _load_research_idea_registry(self, idea_id: str) -> Dict[str, Any]:
        target = str(idea_id or "").strip().lower()
        registry = self._read_json(
            self._research_idea_registry_index_path(target),
            self._default_research_idea_registry(target),
        )
        if not isinstance(registry, dict):
            registry = self._default_research_idea_registry(target)
        registry.setdefault("registry_version", 1)
        registry.setdefault("idea_id", target)
        registry.setdefault("created_at", self._now_iso())
        registry.setdefault("updated_at", self._now_iso())
        registry.setdefault("current_revision_id", "")
        registry.setdefault("last_review_status", "")
        registry.setdefault("portfolio_status", "active")
        registry.setdefault("resolution_status", "unresolved")
        registry.setdefault("execution_status", "idle")
        registry.setdefault("linked_algorithms", [])
        registry.setdefault("attempt_count", 0)
        registry.setdefault("revisions", {})
        registry.setdefault("recent_attempts", [])
        registry.setdefault("recent_decisions", [])
        registry.setdefault("recent_obsolete", [])
        return registry

    def _save_research_idea_registry(
        self,
        idea_id: str,
        registry: Dict[str, Any],
        *,
        record: Optional[Dict[str, Any]] = None,
    ) -> None:
        target = str(idea_id or "").strip().lower()
        registry["idea_id"] = target
        registry["updated_at"] = self._now_iso()
        self._write_json(self._research_idea_registry_index_path(target), registry)
        self._sync_research_idea_summary_to_state(target, registry, record=record)

    def _bootstrap_research_idea_registry(self, idea_id: str) -> Dict[str, Any]:
        target = str(idea_id or "").strip().lower()
        idea_dir = self._research_idea_dir(target)
        idea_dir.mkdir(parents=True, exist_ok=True)
        self._research_idea_revisions_dir(target).mkdir(parents=True, exist_ok=True)
        self._research_idea_attempts_path(target).touch(exist_ok=True)
        self._research_idea_decisions_path(target).touch(exist_ok=True)
        self._research_idea_obsolete_path(target).touch(exist_ok=True)
        registry = self._load_research_idea_registry(target)
        self._save_research_idea_registry(target, registry)
        return registry

    def _default_algorithm_registry(self, algorithm_id: str) -> Dict[str, Any]:
        now = self._now_iso()
        return {
            "registry_version": 1,
            "algorithm_id": str(algorithm_id or "").strip().lower(),
            "created_at": now,
            "updated_at": now,
            "active_proposal_id": "",
            "active_workspace_snapshot_id": "",
            "active_baseline_run_id": "",
            "latest_accepted_run_id": "",
            "algorithm_lifecycle_status": "developing",
            "algorithm_lifecycle_status_reason": "workspace initialized; final_regression locked release not yet recorded",
            "completed_campaign_id": "",
            "completed_release": {},
            "latest_review_verdict": "",
            "latest_preview": {},
            "workspace_dirty": self._default_workspace_dirty(),
            "last_verified_at": "",
            "proposals": {},
            "workspace_snapshots": {},
            "runs": {},
            "recent_previews": [],
            "recent_decisions": [],
            "recent_obsolete": [],
        }

    def _load_algorithm_registry(self, algorithm_id: str) -> Dict[str, Any]:
        algo_id = str(algorithm_id or "").strip().lower()
        index_path = self._registry_index_path(algo_id)
        registry = self._read_json(index_path, self._default_algorithm_registry(algo_id))
        if not isinstance(registry, dict):
            registry = self._default_algorithm_registry(algo_id)
        registry.setdefault("registry_version", 1)
        registry.setdefault("algorithm_id", algo_id)
        registry.setdefault("created_at", self._now_iso())
        registry.setdefault("updated_at", self._now_iso())
        registry.setdefault("active_proposal_id", "")
        registry.setdefault("active_workspace_snapshot_id", "")
        registry.setdefault("active_baseline_run_id", "")
        registry.setdefault("latest_accepted_run_id", "")
        registry.setdefault("algorithm_lifecycle_status", "developing")
        registry.setdefault("algorithm_lifecycle_status_reason", "")
        registry.setdefault("completed_campaign_id", "")
        registry.setdefault("completed_release", {})
        registry.setdefault("latest_review_verdict", "")
        registry.setdefault("latest_preview", {})
        registry.setdefault("workspace_dirty", self._default_workspace_dirty())
        registry.setdefault("last_verified_at", "")
        registry.setdefault("proposals", {})
        registry.setdefault("workspace_snapshots", {})
        registry.setdefault("runs", {})
        registry.setdefault("recent_previews", [])
        registry.setdefault("recent_decisions", [])
        registry.setdefault("recent_obsolete", [])
        return registry

    def _save_algorithm_registry(
        self,
        algorithm_id: str,
        registry: Dict[str, Any],
        *,
        sync_active_context: bool = False,
    ) -> None:
        algo_id = str(algorithm_id or "").strip().lower()
        registry["algorithm_id"] = algo_id
        registry["updated_at"] = self._now_iso()
        self._write_json(self._registry_index_path(algo_id), registry)
        self._sync_registry_summary_to_state(
            algo_id,
            registry,
            update_active_context=sync_active_context or self._is_current_active_algorithm(algo_id),
        )

    def _bootstrap_algorithm_registry(self, algorithm_id: str) -> Dict[str, Any]:
        algo_id = str(algorithm_id or "").strip().lower()
        algo_dir = self._proposal_dir(algo_id)
        algo_dir.mkdir(parents=True, exist_ok=True)
        registry_dir = self._registry_dir(algo_id)
        self._registry_proposals_dir(algo_id).mkdir(parents=True, exist_ok=True)
        self._registry_runs_dir(algo_id).mkdir(parents=True, exist_ok=True)
        self._registry_snapshots_dir(algo_id).mkdir(parents=True, exist_ok=True)
        self._registry_decisions_path(algo_id).touch(exist_ok=True)
        self._registry_obsolete_path(algo_id).touch(exist_ok=True)
        registry = self._load_algorithm_registry(algo_id)
        changed = False

        current_record = self._get_proposal_record(algo_id)
        if isinstance(current_record, dict):
            proposal_id = str(current_record.get("proposal_id") or "").strip()
            if not proposal_id:
                proposal_id = self._make_registry_id("proposal")
                current_record["proposal_id"] = proposal_id
                changed = True
            proposal_path = self._registry_proposals_dir(algo_id) / f"{proposal_id}.json"
            if not proposal_path.exists():
                self._write_json(proposal_path, current_record)
            proposals = registry.setdefault("proposals", {})
            if proposal_id not in proposals:
                proposals[proposal_id] = {
                    "proposal_id": proposal_id,
                    "status": str(current_record.get("status") or ""),
                    "created_at": str(current_record.get("created_at") or ""),
                    "updated_at": str(current_record.get("updated_at") or ""),
                    "primary_idea_id": str(current_record.get("primary_idea_id") or ""),
                    "proposal_path": str(current_record.get("proposal_path") or self._proposal_markdown_path(algo_id)),
                    "proposal_json_path": str(current_record.get("proposal_json_path") or self._proposal_json_path(algo_id)),
                    "registry_path": str(proposal_path),
                    "superseded_by": str(current_record.get("superseded_by") or ""),
                }
                changed = True
            else:
                summary = dict(proposals.get(proposal_id) or {})
                summary_updates = {
                    "proposal_id": proposal_id,
                    "status": str(current_record.get("status") or ""),
                    "created_at": str(current_record.get("created_at") or summary.get("created_at") or ""),
                    "updated_at": str(current_record.get("updated_at") or summary.get("updated_at") or ""),
                    "primary_idea_id": str(current_record.get("primary_idea_id") or ""),
                    "proposal_path": str(current_record.get("proposal_path") or self._proposal_markdown_path(algo_id)),
                    "proposal_json_path": str(current_record.get("proposal_json_path") or self._proposal_json_path(algo_id)),
                    "registry_path": str(proposal_path),
                    "superseded_by": str(current_record.get("superseded_by") or ""),
                }
                if any(summary.get(key) != value for key, value in summary_updates.items()):
                    summary.update(summary_updates)
                    proposals[proposal_id] = summary
                    changed = True
            if not registry.get("active_proposal_id"):
                registry["active_proposal_id"] = proposal_id
                changed = True

        if registry_dir.exists() and not registry.get("active_workspace_snapshot_id"):
            snapshot_id = self.create_workspace_snapshot(
                algo_id,
                reason="registry bootstrap snapshot",
                source="registry_bootstrap",
                proposal_id=str(registry.get("active_proposal_id") or ""),
                set_active=True,
                allow_empty=True,
            )
            if snapshot_id:
                registry = self._load_algorithm_registry(algo_id)
                changed = True

        if changed:
            self._save_algorithm_registry(algo_id, registry)
        else:
            self._sync_registry_summary_to_state(algo_id, registry)
        return registry

    def start_algorithm_campaign(
        self,
        algorithm_id: str,
        *,
        proposal_id: str = "",
        claim_metric_spec: Optional[Dict[str, Any]] = None,
        baseline_selection_policy: str = DEFAULT_CAMPAIGN_BASELINE_SELECTION_POLICY,
    ) -> Dict[str, Any]:
        algo_id = str(algorithm_id or "").strip().lower()
        if not algo_id:
            raise ValueError("algorithm_id is required.")
        if not self._proposal_dir(algo_id).exists():
            raise FileNotFoundError(f"Algorithm workspace does not exist: {self._proposal_dir(algo_id)}")
        registry = self._bootstrap_algorithm_registry(algo_id)
        self._ensure_algorithm_id_mutable(algo_id, action="start_algorithm_campaign")
        target_proposal_id = str(proposal_id or registry.get("active_proposal_id") or "").strip()
        proposal_record: Dict[str, Any] = {}
        if target_proposal_id:
            proposal_record = self._proposal_record_from_registry(algo_id, registry, target_proposal_id)
            status = str(proposal_record.get("status") or "").strip().lower()
            if status not in {"approved", "approved_auto"}:
                raise ValueError(
                    f"Approved proposal is required before campaign start (proposal_id={target_proposal_id}, status={status or 'missing'})."
                )
        claim_spec = self._claim_metric_from_spec(claim_metric_spec or {})
        baseline_policy = self._normalize_campaign_baseline_selection_policy(
            baseline_selection_policy or self._configured_campaign_baseline_selection_policy()
        )
        claim_spec["baseline_selection_policy"] = baseline_policy
        sim_dataset_value = (
            claim_spec.get("simulation_dataset_id")
            or claim_spec.get("stage2_simulation_dataset_id")
            or claim_spec.get("simulation_dataset")
        )
        if not sim_dataset_value and isinstance(claim_spec.get("simulation_dataset_ids"), list) and claim_spec.get("simulation_dataset_ids"):
            sim_dataset_value = claim_spec.get("simulation_dataset_ids")[0]
        if sim_dataset_value and not str(claim_spec.get("simulation_version") or "").strip():
            try:
                sim_card = self.get_algorithm_benchmark_dataset(str(sim_dataset_value))
                sim_info = sim_card.get("simulation") if isinstance(sim_card.get("simulation"), dict) else {}
                if sim_info:
                    claim_spec.setdefault("simulation_dataset_id", str(sim_card.get("dataset_id") or sim_dataset_value))
                    claim_spec.setdefault("simulation_version", str(sim_info.get("simulation_version") or ""))
                    claim_spec.setdefault("simulation_generator_sha256", str(sim_info.get("generator_sha256") or ""))
                    claim_spec.setdefault("simulation_combined_sha256", str(sim_info.get("combined_sha256") or ""))
            except Exception:
                pass
        proposal_payload = proposal_record.get("proposal") if isinstance(proposal_record.get("proposal"), dict) else {}
        algorithm_attributes = (
            proposal_record.get("algorithm_attributes")
            if isinstance(proposal_record.get("algorithm_attributes"), dict)
            else proposal_payload.get("algorithm_attributes")
            if isinstance(proposal_payload.get("algorithm_attributes"), dict)
            else {}
        )
        if not algorithm_attributes and isinstance(proposal_payload, dict):
            mass_scope = str(proposal_payload.get("mass_modeling_scope") or "").strip()
            if mass_scope:
                models_unbalanced = mass_scope == "models_unbalanced_mass"
                algorithm_attributes = {
                    "mass_modeling_scope": mass_scope,
                    "models_unbalanced_mass": models_unbalanced,
                    "tmv_gate_required": models_unbalanced,
                    "tmv_gate_reason": "proposal.algorithm_attributes.mass_modeling_scope",
                }
        if algorithm_attributes and not any(
            isinstance(claim_spec.get(key), bool)
            for key in ("tmv_gate_required", "requires_tmv_gate", "mass_modeling_enabled", "models_unbalanced_mass")
        ):
            models_unbalanced = bool(algorithm_attributes.get("models_unbalanced_mass", False))
            claim_spec.setdefault("mass_modeling_scope", str(algorithm_attributes.get("mass_modeling_scope") or ""))
            claim_spec.setdefault("models_unbalanced_mass", models_unbalanced)
            claim_spec.setdefault("tmv_gate_required", bool(algorithm_attributes.get("tmv_gate_required", models_unbalanced)))
            claim_spec.setdefault("tmv_gate_reason", str(algorithm_attributes.get("tmv_gate_reason") or "proposal.algorithm_attributes"))
        stage_policies = self._default_campaign_stage_policies(claim_spec)
        if isinstance(claim_spec.get("tmv_gate_required"), bool):
            for policy in stage_policies.values():
                policy["tmv_gate_required"] = bool(claim_spec.get("tmv_gate_required"))
                policy["tmv_gate_reason"] = str(claim_spec.get("tmv_gate_reason") or "proposal.algorithm_attributes")
        self._validate_campaign_claim_metric_evaluator(
            claim_spec,
            stage_policies,
            action="start_algorithm_campaign",
        )
        external_baselines = {}
        if isinstance(claim_metric_spec, dict):
            external_baselines = dict(
                claim_metric_spec.get("external_baselines")
                or claim_metric_spec.get("stage_baselines")
                or {}
            )
        campaign_id = self._make_registry_id("campaign")
        campaign_dir = self._campaign_dir(algo_id, campaign_id)
        campaign_dir.mkdir(parents=True, exist_ok=False)
        self._campaign_trials_dir(algo_id, campaign_id).mkdir(parents=True, exist_ok=True)
        archive_path = campaign_dir / "archive.git"
        worktree_path = campaign_dir / "archive_worktree"
        now = self._now_iso()
        stages = {}
        stage_panels = claim_spec.get("stage_panels") if isinstance(claim_spec.get("stage_panels"), dict) else {}
        for idx, stage in enumerate(CAMPAIGN_STAGE_ORDER):
            stage_state = {
                "stage": stage,
                "status": "active" if idx == 0 else "pending",
                "active_best_trial_id": "",
                "active_best_commit": "",
                "active_best_snapshot_id": "",
                "active_best_run_ids": [],
                "trial_count": 0,
                "promote_count": 0,
                "reject_count": 0,
                "external_baseline_metrics": dict(external_baselines.get(stage) or {}),
            }
            stage_panel_spec = stage_panels.get(stage) if isinstance(stage_panels, dict) else None
            if stage_panel_spec:
                if isinstance(stage_panel_spec, dict) and "datasets" in stage_panel_spec:
                    stage_panel_payload = dict(stage_panel_spec)
                elif isinstance(stage_panel_spec, dict) and "dataset_config_overrides" in stage_panel_spec:
                    stage_panel_payload = dict(stage_panel_spec.get("dataset_config_overrides") or {})
                elif isinstance(stage_panel_spec, list):
                    stage_panel_payload = {
                        "datasets": [
                            {"dataset_id": str(item or "").strip()}
                            for item in stage_panel_spec
                            if str(item or "").strip()
                        ],
                    }
                else:
                    stage_panel_payload = {}
                if stage_panel_payload:
                    stage_panel_payload = self._apply_dataset_trial_configs_to_payload(
                        stage_panel_payload,
                        stage=stage,
                    )
                    stage_panel_payload["target_dataset_ids"] = self._campaign_target_dataset_ids(stage_panel_payload)
                    stage_state["stage_panel"] = self._stage_panel_from_payload(
                        stage_panel_payload,
                        source="campaign_start",
                    )
            stages[stage] = stage_state
        campaign = {
            "campaign_version": 1,
            "campaign_id": campaign_id,
            "algorithm_id": algo_id,
            "proposal_id": target_proposal_id,
            "status": "active",
            "current_stage": CAMPAIGN_STAGE_ORDER[0],
            "stage_order": list(CAMPAIGN_STAGE_ORDER),
            "claim_metric_spec": claim_spec,
            "baseline_selection_policy": baseline_policy,
            "algorithm_attributes": dict(algorithm_attributes or {}),
            "stage_policies": stage_policies,
            "created_at": now,
            "updated_at": now,
            "archive": {
                "backend": "git",
                "archive_path": str(archive_path),
                "worktree_path": str(worktree_path),
            },
            "stages": stages,
            "current_trial_id": "",
            "trials": {},
            "locked_release": {},
        }
        init_commit = self._archive_campaign_workspace(
            campaign,
            trial_id="campaign_init",
            message=f"Initialize campaign {campaign_id}",
            extra_meta={"event": "campaign_init"},
        )
        campaign["initial_commit"] = init_commit
        self._save_campaign(campaign)
        registry["algorithm_lifecycle_status"] = "developing"
        registry["algorithm_lifecycle_status_reason"] = f"campaign {campaign_id} active; final_regression locked release not yet recorded"
        registry["completed_campaign_id"] = ""
        registry["completed_release"] = {}
        self._save_algorithm_registry(algo_id, registry, sync_active_context=self._is_current_active_algorithm(algo_id))
        self._emit(
            "algorithm_campaign_started",
            {
                "campaign_id": campaign_id,
                "algorithm_id": algo_id,
                "proposal_id": target_proposal_id,
                "stage": campaign["current_stage"],
                "archive_path": str(archive_path),
                "active_best_trial_id": "",
            },
        )
        return campaign

    def update_campaign_claim_metric_spec(
        self,
        campaign_id: str = "",
        *,
        claim_metric_spec: Optional[Dict[str, Any]] = None,
        reason: str = "",
    ) -> Dict[str, Any]:
        campaign = self._load_campaign(campaign_id)
        self._ensure_campaign_mutable_status(campaign, action="update claim metric spec for")
        self._ensure_algorithm_id_mutable(
            str(campaign.get("algorithm_id") or "").strip().lower(),
            action="update_campaign_claim_metric_spec",
        )
        incoming = claim_metric_spec if isinstance(claim_metric_spec, dict) else {}
        if not incoming:
            raise ValueError("claim_metric_spec must be a non-empty JSON object.")
        existing = campaign.get("claim_metric_spec") if isinstance(campaign.get("claim_metric_spec"), dict) else {}
        previous_spec = deepcopy(existing)
        merged = self._claim_metric_from_spec({**dict(existing or {}), **dict(incoming)})
        baseline_policy = self._normalize_campaign_baseline_selection_policy(
            merged.get("baseline_selection_policy")
            or campaign.get("baseline_selection_policy")
            or existing.get("baseline_selection_policy")
        )
        merged["baseline_selection_policy"] = baseline_policy
        for key in (
            "stage_baselines",
            "external_baselines",
            "stage_panels",
            "mass_modeling_scope",
            "models_unbalanced_mass",
            "tmv_gate_required",
            "tmv_gate_reason",
            "baseline_metric_adapters",
        ):
            if key not in merged and key in existing:
                merged[key] = deepcopy(existing.get(key))

        stage_policies = campaign.get("stage_policies") if isinstance(campaign.get("stage_policies"), dict) else {}
        defaults = self._default_campaign_stage_policies(merged)
        for stage, default_policy in defaults.items():
            policy = stage_policies.setdefault(stage, {})
            if stage == "stage2_claim_validation":
                policy["primary_metric"] = default_policy["primary_metric"]
                policy["primary_direction"] = default_policy["primary_direction"]
            elif stage in {"stage3_tuning", "final_regression"}:
                policy["secondary_metric"] = default_policy["secondary_metric"]
                policy["secondary_direction"] = default_policy["secondary_direction"]
            for key, value in default_policy.items():
                policy.setdefault(key, value)

        self._validate_campaign_claim_metric_evaluator(
            merged,
            stage_policies,
            action="update_campaign_claim_metric_spec",
        )
        campaign["claim_metric_spec"] = merged
        campaign["baseline_selection_policy"] = baseline_policy
        campaign["stage_policies"] = stage_policies
        campaign.setdefault("campaign_repairs", []).append(
            {
                "timestamp": self._now_iso(),
                "action": "update_campaign_claim_metric_spec",
                "reason": str(reason or "").strip(),
                "previous_claim_metric_spec": previous_spec,
                "claim_metric_spec": deepcopy(merged),
            }
        )
        self._save_campaign(campaign)
        self._emit(
            "campaign_claim_metric_spec_updated",
            {
                "campaign_id": str(campaign.get("campaign_id") or ""),
                "algorithm_id": str(campaign.get("algorithm_id") or ""),
                "current_stage": str(campaign.get("current_stage") or ""),
                "claim_metric": str(merged.get("name") or ""),
                "evaluator_path": str(merged.get("evaluator_path") or ""),
            },
        )
        return {
            "status": "updated",
            "campaign_id": str(campaign.get("campaign_id") or ""),
            "algorithm_id": str(campaign.get("algorithm_id") or ""),
            "current_stage": str(campaign.get("current_stage") or ""),
            "claim_metric_spec": merged,
            "next_steps": [
                "Refresh Stage 2 baselines on the frozen panel.",
                "Run campaign trials; do not patch campaign.json directly.",
            ],
        }

    def get_algorithm_campaign_status(self, campaign_id: str = "") -> Dict[str, Any]:
        campaign = self._load_campaign(campaign_id)
        self._save_campaign(campaign)
        algo_id = str(campaign.get("algorithm_id") or "").strip().lower()
        registry = self._bootstrap_algorithm_registry(algo_id) if algo_id else {}
        campaign["algorithm_lifecycle_status"] = str(registry.get("algorithm_lifecycle_status") or "developing")
        campaign["algorithm_lifecycle_status_reason"] = str(registry.get("algorithm_lifecycle_status_reason") or "")
        campaign["completed_campaign_id"] = str(registry.get("completed_campaign_id") or "")
        stages = campaign.get("stages") if isinstance(campaign.get("stages"), dict) else {}
        stage_statuses = {
            stage: self._campaign_stage_status_view(campaign, stage)
            for stage in list(stages.keys() or [])
        }
        current_stage = str(campaign.get("current_stage") or "")
        current_view = dict(stage_statuses.get(current_stage) or {})
        campaign["stage_statuses"] = stage_statuses
        campaign["current_stage_status"] = current_view
        for key in ("stage_internal_status", "stage_gate_status", "user_facing_status", "next_required_action"):
            if key in current_view:
                campaign[key] = current_view[key]
        return campaign

    @staticmethod
    def _compact_campaign_metrics(
        metrics: Dict[str, Any],
        *,
        include_per_dataset: bool = True,
    ) -> Dict[str, Any]:
        if not isinstance(metrics, dict):
            return {}
        keep_keys = (
            "w1_mean",
            "tmv_mean",
            "tmv_max",
            "tmv_gate_required",
            "tmv_gate_reason",
            "primary_metric",
            "primary_direction",
            "primary_value",
            "secondary_metric",
            "secondary_direction",
            "secondary_value",
            "training_timed_out",
            "inference_timed_out",
            "simulation_version",
            "algorithm_name",
            "baseline_type",
            "selection_rule",
            "selection_note",
            "baseline_refresh_at",
            "error",
        )
        compact = {key: metrics.get(key) for key in keep_keys if key in metrics}
        for key, value in metrics.items():
            if str(key).startswith("w1_backend"):
                compact[str(key)] = deepcopy(value)
        if isinstance(metrics.get("target_dataset_ids"), list):
            compact["target_dataset_ids"] = list(metrics.get("target_dataset_ids") or [])
        if isinstance(metrics.get("run_ids"), list):
            compact["run_ids"] = list(metrics.get("run_ids") or [])
        if isinstance(metrics.get("baseline_run_ids"), list):
            compact["baseline_run_ids"] = list(metrics.get("baseline_run_ids") or [])
        custom_metrics = metrics.get("custom_metrics")
        if isinstance(custom_metrics, dict) and custom_metrics:
            compact["custom_metrics"] = {
                str(key): value
                for key, value in custom_metrics.items()
                if isinstance(value, (str, int, float, bool)) or value is None
            }
        per_dataset = metrics.get("per_dataset")
        if include_per_dataset and isinstance(per_dataset, dict) and per_dataset:
            compact_per_dataset: Dict[str, Dict[str, Any]] = {}
            for dataset_id, row in per_dataset.items():
                if not isinstance(row, dict):
                    continue
                compact_row = {
                    key: row.get(key)
                    for key in (
                        "dataset_id",
                        "w1_mean",
                        "tmv_mean",
                        "tmv_max",
                        "primary_value",
                        "secondary_value",
                        "run_id",
                        "error",
                        "training_timed_out",
                        "inference_timed_out",
                    )
                    if key in row
                }
                row_custom = row.get("custom_metrics")
                if isinstance(row_custom, dict) and row_custom:
                    compact_row["custom_metrics"] = {
                        str(key): value
                        for key, value in row_custom.items()
                        if isinstance(value, (str, int, float, bool)) or value is None
                    }
                compact_per_dataset[str(dataset_id)] = compact_row
            if compact_per_dataset:
                compact["per_dataset"] = compact_per_dataset
        return compact

    def _compact_campaign_trial(
        self,
        trial: Dict[str, Any],
        *,
        include_per_dataset: bool = True,
        decision_reasons_limit: int = 5,
    ) -> Dict[str, Any]:
        if not isinstance(trial, dict):
            return {}
        compact = {
            "trial_id": str(trial.get("trial_id") or ""),
            "stage": str(trial.get("stage") or ""),
            "status": str(trial.get("status") or ""),
            "decision": str(trial.get("decision") or ""),
            "created_at": str(trial.get("created_at") or ""),
            "updated_at": str(trial.get("updated_at") or ""),
            "completed_at": str(trial.get("completed_at") or ""),
            "snapshot_id": str(trial.get("snapshot_id") or ""),
            "restored_snapshot_id": str(trial.get("restored_snapshot_id") or ""),
            "trial_commit": str(trial.get("trial_commit") or ""),
        }
        if isinstance(trial.get("run_ids"), list):
            compact["run_ids"] = list(trial.get("run_ids") or [])
        if isinstance(trial.get("target_dataset_ids"), list):
            compact["target_dataset_ids"] = list(trial.get("target_dataset_ids") or [])
        reasons = [str(item) for item in list(trial.get("decision_reasons") or []) if str(item).strip()]
        if reasons:
            compact["decision_reasons"] = reasons[: max(0, int(decision_reasons_limit))]
        metrics = self._compact_campaign_metrics(
            dict(trial.get("metrics_summary") or {}),
            include_per_dataset=include_per_dataset,
        )
        if metrics:
            compact["metrics_summary"] = metrics
        return compact

    def summarize_algorithm_campaign_status(
        self,
        campaign: Dict[str, Any],
        *,
        recent_trials_limit: int = 3,
    ) -> Dict[str, Any]:
        campaign = dict(campaign or {})
        stages = campaign.get("stages") if isinstance(campaign.get("stages"), dict) else {}
        stage_statuses = campaign.get("stage_statuses") if isinstance(campaign.get("stage_statuses"), dict) else {}
        trials = campaign.get("trials") if isinstance(campaign.get("trials"), dict) else {}
        try:
            recent_limit = max(0, min(int(recent_trials_limit), 20))
        except Exception:
            recent_limit = 5

        stage_summaries: Dict[str, Dict[str, Any]] = {}
        for stage, stage_state_raw in stages.items():
            stage_state = dict(stage_state_raw or {}) if isinstance(stage_state_raw, dict) else {}
            status_view = dict(stage_statuses.get(stage) or {})
            stage_panel = stage_state.get("stage_panel") if isinstance(stage_state.get("stage_panel"), dict) else {}
            external_baseline = self._compact_campaign_metrics(
                dict(stage_state.get("external_baseline_metrics") or {}),
                include_per_dataset=False,
            )
            stage_summaries[str(stage)] = {
                "stage": str(stage),
                "status": str(stage_state.get("status") or ""),
                "stage_internal_status": str(status_view.get("stage_internal_status") or ""),
                "stage_gate_status": str(status_view.get("stage_gate_status") or ""),
                "user_facing_status": str(status_view.get("user_facing_status") or ""),
                "next_required_action": str(status_view.get("next_required_action") or ""),
                "trial_count": int(stage_state.get("trial_count") or 0),
                "promote_count": int(stage_state.get("promote_count") or 0),
                "reject_count": int(stage_state.get("reject_count") or 0),
                "max_trials": status_view.get("max_trials"),
                "remaining_trials": status_view.get("remaining_trials"),
                "budget_exhausted": bool(status_view.get("budget_exhausted", False)),
                "active_best_trial_id": str(stage_state.get("active_best_trial_id") or ""),
                "gate_ready": bool(stage_state.get("gate_ready", False)),
                "target_dataset_ids": list(stage_panel.get("target_dataset_ids") or []),
                "stage_panel_source": str(stage_panel.get("source") or ""),
                "stage_panel_frozen_at": str(stage_panel.get("frozen_at") or ""),
                "external_baseline_metrics": external_baseline,
            }
            blockers = status_view.get("blockers")
            if isinstance(blockers, list) and blockers:
                stage_summaries[str(stage)]["blockers"] = [str(item) for item in blockers if str(item).strip()]

        sorted_trials = sorted(
            [dict(item or {}) for item in trials.values() if isinstance(item, dict)],
            key=lambda item: str(item.get("updated_at") or item.get("completed_at") or item.get("created_at") or ""),
            reverse=True,
        )
        recent_trials = [self._compact_campaign_trial(item) for item in sorted_trials[:recent_limit]]
        current_stage = str(campaign.get("current_stage") or "")
        current_stage_state = dict(stages.get(current_stage) or {}) if current_stage else {}
        active_best_id = str(current_stage_state.get("active_best_trial_id") or "")
        active_best_trial = self._compact_campaign_trial(dict(trials.get(active_best_id) or {})) if active_best_id else {}
        current_trial_id = str(campaign.get("current_trial_id") or "")
        current_trial = self._compact_campaign_trial(dict(trials.get(current_trial_id) or {})) if current_trial_id else {}

        return {
            "view": "summary",
            "message": "Concise campaign status. Use list_campaign_trials(...) or narrower campaign tools for details instead of requesting the raw registry.",
            "campaign_id": str(campaign.get("campaign_id") or ""),
            "algorithm_id": str(campaign.get("algorithm_id") or ""),
            "proposal_id": str(campaign.get("proposal_id") or ""),
            "status": str(campaign.get("status") or ""),
            "current_stage": current_stage,
            "current_trial_id": current_trial_id,
            "algorithm_validated": bool(campaign.get("algorithm_validated", False)),
            "validated_at": str(campaign.get("validated_at") or ""),
            "locked_release": dict(campaign.get("locked_release") or {}),
            "stage_internal_status": str(campaign.get("stage_internal_status") or ""),
            "stage_gate_status": str(campaign.get("stage_gate_status") or ""),
            "user_facing_status": str(campaign.get("user_facing_status") or ""),
            "next_required_action": str(campaign.get("next_required_action") or ""),
            "stage_summaries": stage_summaries,
            "active_best_trial": active_best_trial,
            "current_trial": current_trial,
            "recent_trials": recent_trials,
            "trial_count_total": len(trials),
            "recent_trials_limit": recent_limit,
            "archive": {
                "archive_path": str((campaign.get("archive") or {}).get("archive_path") or "")
                if isinstance(campaign.get("archive"), dict)
                else "",
                "worktree_path": str((campaign.get("archive") or {}).get("worktree_path") or "")
                if isinstance(campaign.get("archive"), dict)
                else "",
            },
        }

    def start_campaign_trial(
        self,
        campaign_id: str,
        *,
        base_trial_id: str = "",
        restore_base_snapshot: bool = True,
    ) -> Dict[str, Any]:
        previously_bound_campaign_id = str(self.state.get("active_algorithm_campaign_id") or "").strip()
        campaign = self._load_campaign(campaign_id)
        self._ensure_campaign_mutation_matches_active_binding(
            campaign,
            requested_campaign_id=campaign_id,
            previously_bound_campaign_id=previously_bound_campaign_id,
            action="start a trial for",
        )
        more_advanced = self._more_advanced_active_campaign(campaign)
        if more_advanced:
            requested_id = str(campaign.get("campaign_id") or "")
            preferred_id = str(more_advanced.get("campaign_id") or "")
            preferred_stage = str(more_advanced.get("current_stage") or "")
            requested_stage = str(campaign.get("current_stage") or "")
            raise ValueError(
                f"Campaign '{requested_id}' is not the most advanced active campaign for "
                f"algorithm '{campaign.get('algorithm_id')}'. A sibling active campaign "
                f"'{preferred_id}' is already at stage '{preferred_stage}' while the requested "
                f"campaign is at stage '{requested_stage}'. Continue the more advanced campaign "
                "or explicitly abort/archive stale duplicate campaign state before starting a new trial."
            )
        if str(campaign.get("status") or "") == "locked":
            raise ValueError(f"Campaign '{campaign.get('campaign_id')}' is locked; start a new campaign for further tuning.")
        algo_id = str(campaign.get("algorithm_id") or "").strip().lower()
        self._ensure_algorithm_id_mutable(algo_id, action="start_campaign_trial")
        stage = str(campaign.get("current_stage") or CAMPAIGN_STAGE_ORDER[0])
        stages = campaign.setdefault("stages", {})
        stage_state = stages.setdefault(stage, {})
        policy = self._campaign_stage_policy(campaign, stage)
        budget = self._campaign_stage_budget(stage_state, policy)
        if budget["budget_exhausted"]:
            raise ValueError(
                f"Stage '{stage}' trial budget is exhausted "
                f"({budget['trial_count']}/{budget['max_trials']}). "
                "Check the stage gate and advance to the next stage if it passes; "
                "otherwise patch/review the proposal or start a new campaign."
            )
        current_trial_id = str(campaign.get("current_trial_id") or "").strip()
        trials = campaign.setdefault("trials", {})
        if current_trial_id:
            current_trial = dict(trials.get(current_trial_id) or {})
            if current_trial:
                current_trial.setdefault("start_status", "existing_open_trial")
                current_trial.setdefault(
                    "message",
                    f"Campaign already has an open trial `{current_trial_id}`; reuse it instead of starting another.",
                )
                return current_trial
            raise ValueError(
                f"Campaign '{campaign.get('campaign_id')}' has dangling current_trial_id "
                f"'{current_trial_id}'. Check campaign registry before starting another trial."
            )
        base_id = str(base_trial_id or "").strip()
        working_base_snapshot = ""
        working_base_commit = ""
        restored_snapshot = ""
        if base_id:
            base_trial = dict(trials.get(base_id) or {})
            if not base_trial:
                raise ValueError(f"Trial '{base_id}' was not found in campaign '{campaign.get('campaign_id')}'.")
            if str(base_trial.get("decision") or "") != "reject":
                raise ValueError("Only rejected trials can be resumed as a non-active working base.")
            working_base_snapshot = str(base_trial.get("snapshot_id") or "")
            working_base_commit = str(base_trial.get("trial_commit") or "")
        else:
            working_base_snapshot = str(stage_state.get("active_best_snapshot_id") or "")
            working_base_commit = str(stage_state.get("active_best_commit") or "")

        self.set_active_algorithm_context(algo_id)
        if working_base_snapshot and restore_base_snapshot:
            self.rollback_algorithm_workspace(algo_id, target_snapshot_id=working_base_snapshot)
            restored_snapshot = working_base_snapshot

        trial_id = self._make_registry_id("trial")
        trial = {
            "trial_id": trial_id,
            "campaign_id": str(campaign.get("campaign_id") or ""),
            "algorithm_id": algo_id,
            "stage": stage,
            "status": "editing",
            "decision": "",
            "base_trial_id": base_id,
            "working_base_snapshot_id": working_base_snapshot,
            "working_base_commit": working_base_commit,
            "restored_snapshot_id": restored_snapshot,
            "started_from_current_workspace": bool(not restore_base_snapshot),
            "created_at": self._now_iso(),
            "updated_at": self._now_iso(),
            "snapshot_id": "",
            "trial_commit": "",
            "run_ids": [],
            "metrics_summary": {},
            "decision_reasons": [],
        }
        trials[trial_id] = trial
        campaign["current_trial_id"] = trial_id
        self._save_campaign(campaign)
        self._emit(
            "algorithm_campaign_trial_started",
            {
                "campaign_id": campaign.get("campaign_id"),
                "algorithm_id": algo_id,
                "stage": stage,
                "trial_id": trial_id,
                "base_trial_id": base_id,
                "restored_snapshot_id": restored_snapshot,
                "started_from_current_workspace": bool(trial.get("started_from_current_workspace")),
                "active_best_trial_id": stage_state.get("active_best_trial_id", ""),
                "trial_count": budget["trial_count"],
                "max_trials": budget["max_trials"],
                "remaining_trials": budget["remaining_trials"],
            },
        )
        return trial

    def resume_rejected_trial(self, campaign_id: str, trial_id: str) -> Dict[str, Any]:
        return self.start_campaign_trial(campaign_id, base_trial_id=trial_id)

    def prepare_campaign_trial(
        self,
        campaign_id: str,
        *,
        dataset_config_overrides: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        campaign = self._load_campaign(campaign_id)
        self._ensure_campaign_mutable_status(campaign, action="prepare trial for")
        self._ensure_algorithm_id_mutable(
            str(campaign.get("algorithm_id") or "").strip().lower(),
            action="prepare_campaign_trial",
        )
        trial_id = str(campaign.get("current_trial_id") or "").strip()
        if not trial_id:
            trial = self.start_campaign_trial(
                str(campaign.get("campaign_id") or ""),
                restore_base_snapshot=False,
            )
            campaign = self._load_campaign(campaign_id)
            trial_id = str(trial.get("trial_id") or "")
        algo_id = str(campaign.get("algorithm_id") or "").strip().lower()
        stage = str(campaign.get("current_stage") or CAMPAIGN_STAGE_ORDER[0])
        policy = self._campaign_stage_policy(campaign, stage)
        stages = campaign.setdefault("stages", {})
        stage_state = stages.setdefault(stage, {})
        trials = campaign.setdefault("trials", {})
        trial = dict(trials.get(trial_id) or {})
        if str(trial.get("status") or "") not in {"editing", "prepared", "blocked_before_training"}:
            raise ValueError(f"Current trial '{trial_id}' is not editable/preparable (status={trial.get('status')}).")
        resolved_dataset_config = self._resolve_campaign_stage_panel_payload(
            campaign,
            stage_state,
            stage,
            dataset_config_overrides,
            freeze_if_new=True,
        )
        target_dataset_ids = self._campaign_target_dataset_ids(resolved_dataset_config)
        self.set_active_algorithm_context(algo_id)
        snapshot_id = self.create_workspace_snapshot(
            algo_id,
            reason=f"campaign {campaign.get('campaign_id')} trial {trial_id}",
            source="campaign_trial",
            proposal_id=str(campaign.get("proposal_id") or ""),
            set_active=True,
            allow_empty=False,
        )
        commit = self._archive_campaign_workspace(
            campaign,
            trial_id=trial_id,
            message=f"{stage} {trial_id}",
            dataset_config_overrides=resolved_dataset_config,
            extra_meta={
                "event": "trial_prepared",
                "snapshot_id": snapshot_id,
                "base_trial_id": str(trial.get("base_trial_id") or ""),
                "target_dataset_ids": target_dataset_ids,
            },
        )
        trial.update(
            {
                "status": "running",
                "snapshot_id": snapshot_id,
                "trial_commit": commit,
                "dataset_config_overrides": resolved_dataset_config,
                "target_dataset_ids": target_dataset_ids,
                "updated_at": self._now_iso(),
            }
        )
        trials[trial_id] = trial
        stages[stage] = stage_state
        campaign["stages"] = stages
        campaign["trials"] = trials
        self._save_campaign(campaign)
        return {
            "campaign": campaign,
            "trial": trial,
            "trial_id": trial_id,
            "snapshot_id": snapshot_id,
            "trial_commit": commit,
            "stage": stage,
            "training_stage": str(policy.get("training_stage") or "pilot"),
            "target_dataset_ids": target_dataset_ids,
        }

    def _update_registered_run_for_campaign(
        self,
        algorithm_id: str,
        *,
        run_id: str,
        decision: str,
        reason: str,
        campaign_id: str,
        trial_id: str,
        snapshot_id: str = "",
    ) -> None:
        if not run_id:
            return
        algo_id = str(algorithm_id or "").strip().lower()
        registry = self._bootstrap_algorithm_registry(algo_id)
        runs = registry.setdefault("runs", {})
        summary = dict(runs.get(run_id) or {})
        run_path = self._registry_runs_dir(algo_id) / f"{run_id}.json"
        record = self._read_json(run_path, summary)
        if not isinstance(record, dict):
            record = summary
        for target in (summary, record):
            target["decision"] = decision
            target["decision_reason"] = reason
            target["run_decision_mode"] = "campaign_managed"
            target["manual_run_decision"] = "not_applicable"
            target["campaign_trial_decision"] = decision
            target["campaign_id"] = campaign_id
            target["trial_id"] = trial_id
            if snapshot_id:
                target["snapshot_id"] = snapshot_id
                target["active_workspace_snapshot_id"] = snapshot_id
        runs[run_id] = summary
        if decision == "promote":
            registry["active_baseline_run_id"] = run_id
            registry["latest_accepted_run_id"] = run_id
            if snapshot_id:
                registry["active_workspace_snapshot_id"] = snapshot_id
        registry["runs"] = runs
        self._save_algorithm_registry(algo_id, registry, sync_active_context=self._is_current_active_algorithm(algo_id))
        self._write_json(run_path, record)

    def abort_campaign_trial(
        self,
        campaign_id: str,
        *,
        trial_id: str,
        reason: str,
        training_result: str = "",
    ) -> Dict[str, Any]:
        campaign = self._load_campaign(campaign_id)
        algo_id = str(campaign.get("algorithm_id") or "").strip().lower()
        self._ensure_algorithm_id_mutable(algo_id, action="abort_campaign_trial")
        stage = str(campaign.get("current_stage") or CAMPAIGN_STAGE_ORDER[0])
        trials = campaign.setdefault("trials", {})
        trial = dict(trials.get(trial_id) or {})
        if not trial:
            raise ValueError(f"Trial '{trial_id}' not found in campaign '{campaign_id}'.")
        reason_text = str(reason or "campaign trial blocked before training launched").strip()
        result_text = str(training_result or "").strip()
        trial.update(
            {
                "status": "blocked_before_training",
                "decision": "",
                "decision_reasons": [reason_text],
                "training_result_summary": result_text.splitlines()[:12] if result_text else [],
                "updated_at": self._now_iso(),
            }
        )
        trials[trial_id] = trial
        if str(campaign.get("current_trial_id") or "") == trial_id:
            campaign["current_trial_id"] = ""
        campaign["trials"] = trials
        self._save_campaign(campaign)
        self._write_json(self._campaign_trials_dir(algo_id, str(campaign.get("campaign_id") or "")) / f"{trial_id}.json", trial)
        payload = {
            "status": "blocked",
            "reason": "training_not_launched",
            "message": reason_text,
            "campaign_id": str(campaign.get("campaign_id") or ""),
            "algorithm_id": algo_id,
            "stage": stage,
            "trial_id": trial_id,
            "training_result_summary": trial["training_result_summary"],
        }
        self._emit("algorithm_campaign_trial_blocked", payload)
        return payload

    def abort_current_campaign_trial(
        self,
        campaign_id: str,
        *,
        reason: str = "",
        restore_active_best: bool = True,
    ) -> Dict[str, Any]:
        campaign = self._load_campaign(campaign_id)
        campaign_id_value = str(campaign.get("campaign_id") or "").strip()
        algo_id = str(campaign.get("algorithm_id") or "").strip().lower()
        stage = str(campaign.get("current_stage") or CAMPAIGN_STAGE_ORDER[0])
        trial_id = str(campaign.get("current_trial_id") or "").strip()
        if not trial_id:
            return {
                "ok": True,
                "status": "no_open_trial",
                "message": "Campaign has no current open trial to abort.",
                "campaign_id": campaign_id_value,
                "algorithm_id": algo_id,
                "stage": stage,
                "trial_id": "",
                "current_trial_id": "",
                "restored_active_best": False,
            }

        trials = campaign.setdefault("trials", {})
        trial = dict(trials.get(trial_id) or {})
        if not trial:
            campaign["current_trial_id"] = ""
            self._save_campaign(campaign)
            payload = {
                "ok": True,
                "status": "dangling_current_trial_cleared",
                "message": f"Dangling current_trial_id `{trial_id}` was cleared.",
                "campaign_id": campaign_id_value,
                "algorithm_id": algo_id,
                "stage": stage,
                "trial_id": trial_id,
                "current_trial_id": "",
                "restored_active_best": False,
            }
            self._emit("algorithm_campaign_trial_aborted", payload)
            return payload

        original_status = str(trial.get("status") or "").strip()
        terminal_statuses = {
            "completed",
            "aborted",
            "abandoned",
            "abandoned_due_to_proposal_revision",
            "blocked_before_training",
        }
        if original_status in terminal_statuses:
            campaign["current_trial_id"] = ""
            self._save_campaign(campaign)
            payload = {
                "ok": True,
                "status": "terminal_current_trial_cleared",
                "message": (
                    f"Current trial `{trial_id}` already has terminal status "
                    f"`{original_status}`; cleared current_trial_id."
                ),
                "campaign_id": campaign_id_value,
                "algorithm_id": algo_id,
                "stage": stage,
                "trial_id": trial_id,
                "original_status": original_status,
                "current_trial_id": "",
                "restored_active_best": False,
            }
            self._emit("algorithm_campaign_trial_aborted", payload)
            return payload

        reason_text = str(reason or "").strip() or (
            "Operator aborted an open campaign trial so the campaign can recover from a stale/incomplete run."
        )
        now = self._now_iso()
        previous_reasons = [
            str(item)
            for item in list(trial.get("decision_reasons") or [])
            if str(item).strip()
        ]
        previous_reasons.append(reason_text)
        trial.update(
            {
                "status": "aborted",
                "decision": "aborted",
                "aborted_at": now,
                "updated_at": now,
                "decision_reasons": previous_reasons,
                "abort_reason": reason_text,
                "original_status": original_status,
            }
        )

        restored_active_best = False
        restore_error = ""
        restore_message = ""
        active_best_snapshot = ""
        if bool(restore_active_best):
            stage_state = (
                dict((campaign.get("stages") or {}).get(stage) or {})
                if isinstance(campaign.get("stages"), dict)
                else {}
            )
            active_best_snapshot = str(stage_state.get("active_best_snapshot_id") or "").strip()
            if active_best_snapshot:
                try:
                    restore_message = self.rollback_algorithm_workspace(
                        algo_id,
                        target_snapshot_id=active_best_snapshot,
                    )
                    trial["restored_snapshot_id"] = active_best_snapshot
                    restored_active_best = True
                except Exception as exc:  # noqa: BLE001
                    restore_error = str(exc)
                    trial["restore_error"] = restore_error

        trials[trial_id] = trial
        campaign["trials"] = trials
        campaign["current_trial_id"] = ""
        self._save_campaign(campaign)
        self._write_json(
            self._campaign_trials_dir(algo_id, campaign_id_value) / f"{trial_id}.json",
            trial,
        )
        payload = {
            "ok": True,
            "status": "aborted",
            "message": (
                f"Aborted current campaign trial `{trial_id}` "
                f"(previous status={original_status}) and cleared current_trial_id."
            ),
            "campaign_id": campaign_id_value,
            "algorithm_id": algo_id,
            "stage": stage,
            "trial_id": trial_id,
            "original_status": original_status,
            "current_trial_id": "",
            "reason": reason_text,
            "restored_active_best": restored_active_best,
            "restored_snapshot_id": active_best_snapshot if restored_active_best else "",
            "restore_error": restore_error,
            "restore_message": restore_message,
        }
        self._emit("algorithm_campaign_trial_aborted", payload)
        return payload

    @staticmethod
    def _collect_resolved_config_paths(payload: Any) -> List[str]:
        paths: List[str] = []

        def visit(value: Any) -> None:
            if isinstance(value, dict):
                for key in ("resolved_config_path",):
                    raw_path = str(value.get(key) or "").strip()
                    if raw_path and raw_path not in paths:
                        paths.append(raw_path)
                artifacts = value.get("artifacts")
                if isinstance(artifacts, dict):
                    raw_path = str(artifacts.get("resolved_config_path") or "").strip()
                    if raw_path and raw_path not in paths:
                        paths.append(raw_path)
                for nested in value.values():
                    visit(nested)
            elif isinstance(value, list):
                for item in value:
                    visit(item)

        visit(payload)
        return paths

    @staticmethod
    def _normalized_yaml_text(payload: Dict[str, Any]) -> str:
        return yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)

    def _sync_promoted_resolved_config_to_workspace(
        self,
        algorithm_id: str,
        *,
        campaign_id: str,
        trial_id: str,
        metrics_payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        algo_id = str(algorithm_id or "").strip().lower()
        config_path = self._proposal_dir(algo_id) / "config.yaml" if algo_id else Path("config.yaml")
        result: Dict[str, Any] = {
            "status": "skipped_no_resolved_config",
            "source_resolved_config_paths": [],
            "workspace_config_path": str(config_path),
            "snapshot_id": "",
        }
        if not algo_id:
            result["status"] = "skipped_missing_algorithm_id"
            return result

        dataset_metrics = (
            metrics_payload.get("campaign_dataset_metrics")
            if isinstance(metrics_payload.get("campaign_dataset_metrics"), list)
            else []
        )
        dataset_metric_count = len(dataset_metrics)
        dataset_metrics_with_config = 0
        for item in dataset_metrics:
            if not isinstance(item, dict):
                continue
            item_paths = self._collect_resolved_config_paths(item)
            if any(Path(raw_path).expanduser().is_file() for raw_path in item_paths):
                dataset_metrics_with_config += 1
        if dataset_metric_count > 1 and dataset_metrics_with_config not in {0, dataset_metric_count}:
            result.update(
                {
                    "status": "skipped_partial_resolved_config_paths",
                    "reason": (
                        "Promoted multi-dataset trial exposed resolved_config.yaml for only some datasets. "
                        "Not writing a partial dataset config into the global workspace config.yaml."
                    ),
                    "dataset_metric_count": dataset_metric_count,
                    "dataset_metrics_with_resolved_config": dataset_metrics_with_config,
                }
            )
            return result

        existing_paths: List[Path] = []
        for raw_path in self._collect_resolved_config_paths(metrics_payload):
            path = Path(raw_path).expanduser()
            if path.is_file() and path not in existing_paths:
                existing_paths.append(path)
        result["source_resolved_config_paths"] = [str(path) for path in existing_paths]
        if not existing_paths:
            return result

        normalized_to_payload: Dict[str, Dict[str, Any]] = {}
        path_by_normalized: Dict[str, str] = {}
        for path in existing_paths:
            try:
                loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
            except Exception as exc:
                result.update(
                    {
                        "status": "skipped_invalid_resolved_config",
                        "invalid_resolved_config_path": str(path),
                        "error": str(exc),
                    }
                )
                return result
            if not isinstance(loaded, dict) or not loaded:
                result.update(
                    {
                        "status": "skipped_empty_or_non_mapping_resolved_config",
                        "invalid_resolved_config_path": str(path),
                    }
                )
                return result
            normalized = self._normalized_yaml_text(loaded)
            normalized_to_payload[normalized] = loaded
            path_by_normalized.setdefault(normalized, str(path))

        if len(normalized_to_payload) != 1:
            result.update(
                {
                    "status": "skipped_multiple_distinct_resolved_configs",
                    "reason": (
                        "Promoted trial used multiple dataset-specific resolved configs. "
                        "Not writing one dataset's config into the global workspace config.yaml."
                    ),
                    "distinct_config_count": len(normalized_to_payload),
                }
            )
            return result

        resolved_text = next(iter(normalized_to_payload.keys()))
        source_path = path_by_normalized.get(resolved_text, "")
        result.update(
            {
                "source_resolved_config_path": source_path,
                "resolved_config_sha256": hashlib.sha256(resolved_text.encode("utf-8")).hexdigest(),
            }
        )
        if self._read_text(config_path) == resolved_text:
            result["status"] = "already_current"
            return result

        self._write_text(config_path, resolved_text)
        proposal_id = str(self._load_algorithm_registry(algo_id).get("active_proposal_id") or "")
        snapshot_id = self.create_workspace_snapshot(
            algo_id,
            reason=f"campaign {campaign_id} promoted trial {trial_id} resolved config",
            source="campaign_promoted_resolved_config",
            proposal_id=proposal_id,
            set_active=True,
            allow_empty=False,
        )
        result.update(
            {
                "status": "synced",
                "snapshot_id": snapshot_id,
                "message": "Workspace config.yaml was updated from the promoted run resolved_config.yaml.",
            }
        )
        return result

    def decide_campaign_trial(
        self,
        campaign_id: str,
        *,
        trial_id: str,
        run_id: str = "",
        metrics: Optional[Dict[str, Any]] = None,
        training_result: str = "",
    ) -> Dict[str, Any]:
        campaign = self._load_campaign(campaign_id)
        algo_id = str(campaign.get("algorithm_id") or "").strip().lower()
        self._ensure_algorithm_id_mutable(algo_id, action="decide_campaign_trial")
        stage = str(campaign.get("current_stage") or CAMPAIGN_STAGE_ORDER[0])
        policy = self._campaign_stage_policy(campaign, stage)
        stages = campaign.setdefault("stages", {})
        stage_state = stages.setdefault(stage, {})
        trials = campaign.setdefault("trials", {})
        trial = dict(trials.get(trial_id) or {})
        if not trial:
            raise ValueError(f"Trial '{trial_id}' not found in campaign '{campaign_id}'.")
        metrics_payload = dict(metrics or {})
        summary = self._summarize_campaign_metrics(metrics_payload, policy)
        reasons: List[str] = []
        hard_pass = True
        tmv_gate = self._tmv_gate_status(metrics=metrics_payload, policy=policy, campaign=campaign)
        summary["tmv_gate_required"] = bool(tmv_gate.get("required", True))
        summary["tmv_gate_reason"] = str(tmv_gate.get("reason") or "")
        if str(metrics_payload.get("error") or "").strip() or str(training_result or "").startswith("Training failed"):
            hard_pass = False
            reasons.append("training failed or returned an error")
        if bool(metrics_payload.get("training_timed_out")):
            hard_pass = False
            timeout_budget = metrics_payload.get("training_time_budget")
            if isinstance(timeout_budget, dict):
                reasons.append(
                    "training exceeded the wall-clock budget "
                    f"({timeout_budget.get('elapsed_sec')}s/{timeout_budget.get('budget_sec')}s)"
                )
            else:
                reasons.append("training exceeded the wall-clock budget")
        if bool(metrics_payload.get("inference_timed_out")):
            hard_pass = False
            timeout_budget = metrics_payload.get("inference_time_budget")
            if isinstance(timeout_budget, dict):
                reasons.append(
                    "inference/evaluation exceeded the wall-clock budget "
                    f"({timeout_budget.get('elapsed_sec')}s/{timeout_budget.get('budget_sec')}s)"
                )
            else:
                reasons.append("inference/evaluation exceeded the wall-clock budget")
        if bool(tmv_gate.get("required", True)) and not self._tmv_hard_pass(metrics_payload, float(policy.get("tmv_max", 0.2))):
            hard_pass = False
            reasons.append(f"TMV hard gate failed: every time point must be < {policy.get('tmv_max', 0.2)}")
        elif not bool(tmv_gate.get("required", True)):
            reasons.append("TMV hard gate skipped because the algorithm/run is not configured to model unbalanced mass")
        primary_value = self._numeric_value(summary.get("primary_value"))
        if primary_value is None:
            hard_pass = False
            reasons.append(f"primary metric is missing: {policy.get('primary_metric')}")
        provenance_blockers = self._claim_metric_provenance_blockers(
            stage=stage,
            claim_metric=str(policy.get("primary_metric") or ""),
            candidate_metrics=summary,
        )
        if provenance_blockers:
            hard_pass = False
            reasons.extend(provenance_blockers)
        dataset_blockers = self._campaign_target_dataset_blockers(summary)
        if dataset_blockers:
            hard_pass = False
            reasons.extend(dataset_blockers)
        stage3_floor_check = self._evaluate_stage3_inherited_stage2_floor(
            campaign=campaign,
            stage=stage,
            active_metrics=summary,
        )
        if stage3_floor_check:
            summary["stage3_inherited_stage2_locked_gate_floor"] = stage3_floor_check
            if not bool(stage3_floor_check.get("ok", False)):
                hard_pass = False
                reasons.extend(
                    [
                        f"Stage 3 inherited Stage 2 locked-gate floor failed: {str(item)}"
                        for item in list(stage3_floor_check.get("blockers") or [])
                    ]
                )
        stage2_claim_floor_check = self._evaluate_stage2_claim_metric_regression_floor(
            campaign=campaign,
            stage=stage,
            active_metrics=summary,
        )
        if stage2_claim_floor_check:
            summary["stage2_claim_metric_regression_floor"] = stage2_claim_floor_check
            if not bool(stage2_claim_floor_check.get("ok", False)):
                hard_pass = False
                reasons.extend(
                    [
                        f"Stage 2 claim metric regression floor failed: {str(item)}"
                        for item in list(stage2_claim_floor_check.get("blockers") or [])
                    ]
                )

        active_best_trial_id = str(stage_state.get("active_best_trial_id") or "")
        decision = "reject"
        if hard_pass and not active_best_trial_id:
            decision = "promote"
            reasons.append("first successful trial in this stage becomes active best")
        elif hard_pass:
            best_trial = dict(trials.get(active_best_trial_id) or {})
            best_summary = dict(stage_state.get("active_best_metrics_summary") or best_trial.get("metrics_summary") or {})
            best_primary = self._numeric_value(best_summary.get("primary_value"))
            candidate_secondary = self._numeric_value(summary.get("secondary_value"))
            best_secondary = self._numeric_value(best_summary.get("secondary_value"))
            primary_ok = (
                best_primary is not None
                and primary_value is not None
                and self._metric_improves(
                    primary_value,
                    best_primary,
                    direction=str(policy.get("primary_direction") or "lower"),
                    min_delta=float(policy.get("min_delta", 0.0) or 0.0),
                    abs_min_delta=float(policy.get("abs_min_delta", 0.0) or 0.0),
                )
            )
            secondary_ok = self._secondary_within_tolerance(
                candidate_secondary,
                best_secondary,
                direction=str(policy.get("secondary_direction") or ""),
                tolerance=float(policy.get("secondary_tolerance", 0.2) or 0.2),
                abs_tolerance=float(policy.get("secondary_abs_tolerance", 1e-8) or 1e-8),
            )
            if primary_ok and secondary_ok:
                decision = "promote"
                reasons.append("primary metric improved over self active best and secondary metric stayed within tolerance")
            else:
                if not primary_ok:
                    reasons.append("primary metric did not improve over self active best")
                if not secondary_ok:
                    reasons.append("secondary metric exceeded allowed regression tolerance")

        restored_snapshot_id = ""
        snapshot_id = str(trial.get("snapshot_id") or "")
        commit = str(trial.get("trial_commit") or "")
        run_ids = [item for item in list(trial.get("run_ids") or []) if str(item).strip()]
        for item in list(metrics_payload.get("campaign_run_ids") or []):
            value = str(item or "").strip()
            if value and value not in run_ids:
                run_ids.append(value)
        if run_id and run_id not in run_ids:
            run_ids.append(run_id)
        promoted_config_sync: Dict[str, Any] = {}
        active_snapshot_id = snapshot_id
        if decision == "promote":
            promoted_config_sync = self._sync_promoted_resolved_config_to_workspace(
                algo_id,
                campaign_id=str(campaign.get("campaign_id") or ""),
                trial_id=trial_id,
                metrics_payload=metrics_payload,
            )
            active_snapshot_id = str(promoted_config_sync.get("snapshot_id") or snapshot_id)
            stage_state["status"] = "active"
            stage_state["active_best_trial_id"] = trial_id
            stage_state["active_best_commit"] = commit
            stage_state["active_best_snapshot_id"] = active_snapshot_id
            stage_state["active_best_run_ids"] = run_ids
            stage_state["active_best_metrics_summary"] = summary
            stage_state["promote_count"] = int(stage_state.get("promote_count") or 0) + 1
            if active_snapshot_id:
                registry = self._bootstrap_algorithm_registry(algo_id)
                registry["active_workspace_snapshot_id"] = active_snapshot_id
                if str(registry.get("algorithm_lifecycle_status") or "developing") == "failed":
                    registry["algorithm_lifecycle_status"] = "developing"
                    registry["algorithm_lifecycle_status_reason"] = (
                        f"campaign {campaign.get('campaign_id')} has a promoted trial in {stage}; "
                        "final_regression locked release not yet recorded"
                    )
                    registry["completed_campaign_id"] = ""
                if run_id:
                    registry["active_baseline_run_id"] = run_id
                    registry["latest_accepted_run_id"] = run_id
                self._save_algorithm_registry(algo_id, registry, sync_active_context=self._is_current_active_algorithm(algo_id))
        else:
            stage_state["reject_count"] = int(stage_state.get("reject_count") or 0) + 1
            restore_target = str(stage_state.get("active_best_snapshot_id") or trial.get("working_base_snapshot_id") or "")
            if restore_target:
                self.rollback_algorithm_workspace(algo_id, target_snapshot_id=restore_target)
                restored_snapshot_id = restore_target
            elif not active_best_trial_id:
                stage_state["status"] = "failed_needs_revision"
                registry = self._bootstrap_algorithm_registry(algo_id)
                if str(registry.get("algorithm_lifecycle_status") or "developing") != "complete":
                    registry["algorithm_lifecycle_status"] = "failed"
                    registry["algorithm_lifecycle_status_reason"] = (
                        f"campaign {campaign.get('campaign_id')} stage {stage} has no active best after rejected trial {trial_id}"
                    )
                    self._save_algorithm_registry(
                        algo_id,
                        registry,
                        sync_active_context=self._is_current_active_algorithm(algo_id),
                    )
        stage_state["trial_count"] = int(stage_state.get("trial_count") or 0) + 1
        budget = self._campaign_stage_budget(stage_state, policy)
        stage_state["budget_exhausted"] = bool(budget["budget_exhausted"])
        stages[stage] = stage_state
        for registered_run_id in run_ids:
            self._update_registered_run_for_campaign(
                algo_id,
                run_id=registered_run_id,
                decision=decision,
                reason="; ".join(reasons),
                campaign_id=str(campaign.get("campaign_id") or ""),
                trial_id=trial_id,
                snapshot_id=active_snapshot_id,
            )
        trial.update(
            {
                "status": "completed",
                "decision": decision,
                "run_ids": run_ids,
                "metrics_summary": summary,
                "decision_reasons": reasons,
                "restored_snapshot_id": restored_snapshot_id,
                "promoted_config_sync": promoted_config_sync,
                "active_best_snapshot_id": active_snapshot_id if decision == "promote" else "",
                "completed_at": self._now_iso(),
                "updated_at": self._now_iso(),
            }
        )
        trials[trial_id] = trial
        campaign["stages"] = stages
        campaign["trials"] = trials
        campaign["current_trial_id"] = ""
        self._save_campaign(campaign)
        self._write_json(self._campaign_trials_dir(algo_id, str(campaign.get("campaign_id") or "")) / f"{trial_id}.json", trial)
        self.record_decision(
            algo_id,
            phase="campaign",
            decision=f"campaign_trial_{decision}",
            alternatives=[],
            evidence=[str(self._campaign_index_path(algo_id, str(campaign.get("campaign_id") or "")))],
            rationale="; ".join(reasons),
            status="final",
            related_artifacts=[
                {"artifact_type": "campaign", "artifact_id": str(campaign.get("campaign_id") or "")},
                {"artifact_type": "campaign_trial", "artifact_id": trial_id},
            ],
        )
        payload = {
            "campaign_id": str(campaign.get("campaign_id") or ""),
            "algorithm_id": algo_id,
            "stage": stage,
            "trial_id": trial_id,
            "decision": decision,
            "active_best_trial_id": str(stage_state.get("active_best_trial_id") or ""),
            "target_dataset_ids": list((trial.get("dataset_config_overrides") or {}).get("target_dataset_ids") or []),
            "metrics_summary": summary,
            "restored_snapshot_id": restored_snapshot_id,
            "run_ids": run_ids,
            "decision_reasons": reasons,
            "promoted_config_sync": promoted_config_sync,
            "trial_count": budget["trial_count"],
            "max_trials": budget["max_trials"],
            "remaining_trials": budget["remaining_trials"],
            "budget_exhausted": budget["budget_exhausted"],
        }
        self._emit("algorithm_campaign_trial_decided", payload)
        return payload

    def list_campaign_trials(
        self,
        campaign_id: str,
        *,
        decision: str = "",
        stage: str = "",
        limit: int = 10,
        offset: int = 0,
    ) -> Dict[str, Any]:
        campaign = self._load_campaign(campaign_id)
        decision_value = str(decision or "").strip().lower()
        stage_value = str(stage or "").strip()
        trials = [dict(item or {}) for item in list((campaign.get("trials") or {}).values()) if isinstance(item, dict)]
        if decision_value:
            trials = [trial for trial in trials if str((trial or {}).get("decision") or "") == decision_value]
        if stage_value:
            trials = [trial for trial in trials if str((trial or {}).get("stage") or "") == stage_value]
        trials.sort(
            key=lambda item: str(item.get("updated_at") or item.get("completed_at") or item.get("created_at") or ""),
            reverse=True,
        )
        try:
            bounded_limit = max(1, min(int(limit), 50))
        except Exception:
            bounded_limit = 10
        try:
            bounded_offset = max(0, int(offset))
        except Exception:
            bounded_offset = 0
        page = trials[bounded_offset : bounded_offset + bounded_limit]
        return {
            "view": "trial_list_summary",
            "message": "Concise paginated trial summaries. Use a narrower single-trial detail tool for deep inspection instead of loading the raw campaign registry.",
            "campaign_id": campaign.get("campaign_id"),
            "algorithm_id": campaign.get("algorithm_id"),
            "current_stage": campaign.get("current_stage"),
            "decision_filter": decision_value,
            "stage_filter": stage_value,
            "count": len(trials),
            "returned_count": len(page),
            "limit": bounded_limit,
            "offset": bounded_offset,
            "has_more": bounded_offset + bounded_limit < len(trials),
            "next_offset": bounded_offset + bounded_limit if bounded_offset + bounded_limit < len(trials) else None,
            "trials": [
                self._compact_campaign_trial(
                    trial,
                    include_per_dataset=False,
                    decision_reasons_limit=3,
                )
                for trial in page
            ],
        }

    def check_campaign_stage_gate(self, campaign_id: str, *, advance: bool = True) -> Dict[str, Any]:
        campaign = self._load_campaign(campaign_id)
        algo_id = str(campaign.get("algorithm_id") or "").strip().lower()
        if advance:
            self._ensure_campaign_mutable_status(campaign, action="advance gate for")
            self._ensure_algorithm_id_mutable(algo_id, action="check_campaign_stage_gate")
        stage = str(campaign.get("current_stage") or CAMPAIGN_STAGE_ORDER[0])
        stages = campaign.setdefault("stages", {})
        stage_state = stages.setdefault(stage, {})
        policy = self._campaign_stage_policy(campaign, stage)
        budget = self._campaign_stage_budget(stage_state, policy)
        trials = campaign.setdefault("trials", {})
        active_trial_id = str(stage_state.get("active_best_trial_id") or "")
        active_trial = dict(trials.get(active_trial_id) or {})
        active_metrics = dict(active_trial.get("metrics_summary") or {})
        baseline = dict(stage_state.get("external_baseline_metrics") or {})
        if active_trial_id and active_metrics and not self._w1_backend_identity(active_metrics):
            active_run_ids = [
                str(item or "").strip()
                for item in list(active_trial.get("run_ids") or stage_state.get("active_best_run_ids") or [])
                if str(item or "").strip()
            ]
            rehydrated = self._rehydrate_w1_backend_metadata_from_run_ids(algo_id, active_run_ids)
            if rehydrated:
                active_metrics.update(rehydrated)
                active_trial["metrics_summary"] = active_metrics
                trials[active_trial_id] = active_trial
                stage_state["active_best_metrics_summary"] = active_metrics
        if baseline:
            if not self._w1_backend_identity(baseline):
                rehydrated = self._rehydrate_w1_backend_metadata_from_baseline_record(baseline)
                if rehydrated:
                    baseline.update(rehydrated)
            for nested_key in ("claim_sota_baseline_metrics", "w1_sota_baseline_metrics"):
                nested = dict(baseline.get(nested_key) or {})
                if nested and not self._w1_backend_identity(nested):
                    rehydrated = self._rehydrate_w1_backend_metadata_from_baseline_record(nested)
                    if rehydrated:
                        nested.update(rehydrated)
                        baseline[nested_key] = nested
            stage_state["external_baseline_metrics"] = baseline
        stage_claim_metric = (
            str(policy.get("primary_metric") or "").strip()
            if stage == "stage2_claim_validation"
            else str(policy.get("secondary_metric") or "").strip()
        )
        claim_baseline = self._baseline_record_for_metric(
            baseline,
            stage_claim_metric,
            claim_metric=stage_claim_metric,
            direction=str(
                policy.get("primary_direction")
                if stage == "stage2_claim_validation"
                else policy.get("secondary_direction")
                or ""
            ),
        )
        w1_sota_baseline = self._baseline_record_for_metric(
            baseline,
            "w1_mean",
            claim_metric=stage_claim_metric,
            direction="lower",
        )
        blockers: List[str] = []
        checks: Dict[str, Any] = {}
        if not active_trial_id:
            blockers.append("stage has no active best trial")
        min_trials_before_advance = int(policy.get("min_trials_before_advance") or 0)
        min_trials_satisfied = int(budget.get("trial_count") or 0) >= min_trials_before_advance
        if (
            stage != "stage3_tuning"
            and min_trials_before_advance > 0
            and not min_trials_satisfied
        ):
            blockers.append(
                f"stage requires at least {min_trials_before_advance} completed trial(s) before advancing; "
                f"current={int(budget.get('trial_count') or 0)}"
            )
        checks["trial_budget"] = {
            "trial_count": budget["trial_count"],
            "max_trials": budget["max_trials"],
            "remaining_trials": budget["remaining_trials"],
            "budget_exhausted": budget["budget_exhausted"],
            "min_trials_before_advance": min_trials_before_advance,
            "min_trials_satisfied": min_trials_satisfied,
        }
        strict_baseline_audit = self._strict_baseline_audit_gate_check(
            campaign=campaign,
            stage=stage,
            baseline=baseline,
            active_metrics=active_metrics,
            policy=policy,
        )
        if bool(strict_baseline_audit.get("required")):
            checks["strict_baseline_audit"] = strict_baseline_audit
            if not bool(strict_baseline_audit.get("ok", False)):
                blockers.extend([str(item) for item in list(strict_baseline_audit.get("blockers") or [])])
        stage_has_external_gate = stage not in {"stage3_tuning", "final_regression"}
        if stage_has_external_gate and not baseline:
            blockers.append("external baseline metrics are required for stage pass/fail")
        stage_panel = stage_state.get("stage_panel") if isinstance(stage_state.get("stage_panel"), dict) else {}
        stage_panel_ids = [
            str(item or "").strip()
            for item in list(stage_panel.get("target_dataset_ids") or [])
            if str(item or "").strip()
        ]
        active_panel_ids = self._campaign_target_dataset_ids(active_metrics)
        if not active_panel_ids and stage_panel_ids:
            active_panel_ids = stage_panel_ids
        baseline_panel_ids = self._campaign_target_dataset_ids(baseline)
        baseline_role_panel_ids: Dict[str, List[str]] = {}
        for role_name, role_baseline in (
            ("claim_sota_baseline", claim_baseline),
            ("w1_sota_baseline", w1_sota_baseline),
        ):
            if not role_baseline or role_baseline == baseline:
                continue
            role_panel_ids = self._campaign_target_dataset_ids(role_baseline)
            if role_panel_ids:
                baseline_role_panel_ids[role_name] = role_panel_ids
                if stage_has_external_gate and active_panel_ids and role_panel_ids != active_panel_ids:
                    blockers.append(
                        f"{role_name} panel does not match active best panel "
                        f"(active={active_panel_ids}, baseline={role_panel_ids})"
                    )
        if stage_has_external_gate and active_panel_ids:
            if not baseline_panel_ids:
                blockers.append("external baseline metrics must include target_dataset_ids matching the frozen stage panel")
            elif baseline_panel_ids != active_panel_ids:
                blockers.append(
                    "external baseline panel does not match active best panel "
                    f"(active={active_panel_ids}, baseline={baseline_panel_ids})"
                )
            checks["stage_panel"] = {
                "target_dataset_ids": active_panel_ids,
                "baseline_target_dataset_ids": baseline_panel_ids,
                "baseline_role_target_dataset_ids": baseline_role_panel_ids,
                "ok": bool(baseline_panel_ids == active_panel_ids),
            }
        if stage == "stage2_claim_validation":
            claim_metric = str(policy.get("primary_metric") or "").strip()
            expected_claim_evaluator: Dict[str, Any] = {}
            try:
                from .claim_metric_evaluator import load_campaign_claim_metric_evaluator

                _, expected_claim_evaluator = load_campaign_claim_metric_evaluator(
                    campaign.get("claim_metric_spec") or {}
                )
            except Exception as exc:
                blockers.append(f"current campaign claim metric evaluator could not be loaded: {exc}")
            provenance_blockers = self._claim_metric_provenance_blockers(
                stage=stage,
                claim_metric=claim_metric,
                candidate_metrics=active_metrics,
                baseline_metrics=claim_baseline,
                expected_evaluator_info=expected_claim_evaluator,
            )
            if provenance_blockers:
                blockers.extend(provenance_blockers)
            checks["claim_metric_provenance"] = {
                "claim_metric": claim_metric,
                "expected_evaluator": self._claim_metric_evaluator_identity(
                    expected_claim_evaluator
                ),
                "active_evaluator": self._claim_metric_evaluator_identity(
                    active_metrics.get("claim_metric_evaluator")
                ),
                "baseline_evaluator": self._claim_metric_evaluator_identity(
                    claim_baseline.get("claim_metric_evaluator")
                ),
                "baseline_algorithm": str(claim_baseline.get("algorithm_name") or ""),
                "baseline_run_ids": list(claim_baseline.get("baseline_run_ids") or []),
                "ok": not bool(provenance_blockers),
            }
            stage_panel_sims = self._stage_panel_simulation_versions(stage_panel)
            expected_sim = stage_panel_sims[0] if len(stage_panel_sims) == 1 else ""
            if len(stage_panel_sims) > 1:
                blockers.append("stage2 stage panel mixes simulation versions; use one frozen generator version per gate")
            if not expected_sim:
                expected_sim = str((campaign.get("claim_metric_spec") or {}).get("simulation_version") or "").strip()
            active_sim = str(active_metrics.get("simulation_version") or "").strip()
            active_sims = [
                str(item or "").strip()
                for item in list(active_metrics.get("simulation_versions") or [])
                if str(item or "").strip()
            ]
            if not expected_sim and active_sim:
                expected_sim = active_sim
            if not expected_sim and len(set(active_sims)) == 1:
                expected_sim = active_sims[0]
            baseline_sim_records: List[tuple[str, Dict[str, Any]]] = [
                ("claim_sota_baseline", claim_baseline or baseline)
            ]
            if w1_sota_baseline and w1_sota_baseline != (claim_baseline or baseline):
                baseline_sim_records.append(("w1_sota_baseline", w1_sota_baseline))
            if expected_sim:
                if active_sim and active_sim != expected_sim:
                    blockers.append("stage2 active best simulation_version does not match the campaign frozen generator version")
                if active_sims and any(item != expected_sim for item in active_sims):
                    blockers.append("stage2 active best mixes simulation versions; use one frozen generator version per gate")
                if not active_sim and not active_sims:
                    blockers.append("stage2 active best is missing simulation_version for the frozen generator version")
                for role_name, role_baseline in baseline_sim_records:
                    baseline_sim = str(role_baseline.get("simulation_version") or "").strip()
                    if not baseline_sim:
                        blockers.append(
                            f"stage2 {role_name} is missing simulation_version; rerun baselines for the frozen generator version"
                        )
                    elif baseline_sim != expected_sim:
                        blockers.append(
                            f"stage2 {role_name} simulation_version mismatch; rerun baselines for the frozen generator version"
                        )
        tmv_gate = self._tmv_gate_status(metrics=active_metrics, policy=policy, campaign=campaign)
        checks["tmv_gate_policy"] = {
            "required": bool(tmv_gate.get("required", True)),
            "reason": str(tmv_gate.get("reason") or ""),
            "tmv_max": policy.get("tmv_max", 0.2),
            "note": "TMV is a hard gate only for algorithms/runs configured to model unbalanced mass.",
        }
        if stage == "stage3_tuning":
            inherited_floor_check = self._evaluate_stage3_inherited_stage2_floor(
                campaign=campaign,
                stage=stage,
                active_metrics=active_metrics,
            )
            if inherited_floor_check:
                checks["stage3_inherited_stage2_locked_gate_floor"] = inherited_floor_check
                if not bool(inherited_floor_check.get("ok", False)):
                    blockers.extend([str(item) for item in list(inherited_floor_check.get("blockers") or [])])
            claim_floor_check = self._evaluate_stage2_claim_metric_regression_floor(
                campaign=campaign,
                stage=stage,
                active_metrics=active_metrics,
            )
            if claim_floor_check:
                checks["stage2_claim_metric_regression_floor"] = claim_floor_check
                if not bool(claim_floor_check.get("ok", False)):
                    blockers.extend([str(item) for item in list(claim_floor_check.get("blockers") or [])])
            optional_sota_gate = {
                "enabled": bool(policy.get("optional_sota_gate_enabled", True)),
                "pass_fail_gate": False,
                "budget_exhaustion_allows_advance": bool(
                    policy.get("optional_sota_gate_budget_exhaustion_allows_advance", True)
                ),
                "ok": False,
                "budget_exhausted": bool(budget.get("budget_exhausted")),
                "note": (
                    "Stage 3 can finish early only when the active best is SOTA/non-worse than the external baseline on both "
                    "W1 and the validated claim metric. If the Stage 3 budget is exhausted before this optional gate passes, "
                    "the algorithm is still allowed to proceed to final regression because Stage 2 already validated the method, "
                    "provided any explicit Stage 2 absolute locked-gate floors are still satisfied."
                ),
            }
            if bool(optional_sota_gate["enabled"]):
                primary_metric = str(policy.get("primary_metric") or "w1_mean")
                secondary_metric = str(policy.get("secondary_metric") or "")
                primary_direction = str(policy.get("primary_direction") or "lower")
                secondary_direction = str(policy.get("secondary_direction") or "")
                candidate_primary = self._metric_value(active_metrics, primary_metric)
                primary_baseline_record = self._baseline_record_for_metric(
                    baseline,
                    primary_metric,
                    claim_metric=secondary_metric,
                    direction=primary_direction,
                )
                baseline_primary = self._metric_value(primary_baseline_record, primary_metric)
                primary_w1_backend_blockers = self._w1_backend_provenance_blockers(
                    candidate_metrics=active_metrics,
                    baseline_metrics=primary_baseline_record,
                    metric_name=primary_metric,
                )
                candidate_secondary = self._metric_value(active_metrics, secondary_metric)
                secondary_baseline_record = self._baseline_record_for_metric(
                    baseline,
                    secondary_metric,
                    claim_metric=secondary_metric,
                    direction=secondary_direction,
                )
                baseline_secondary = self._metric_value(secondary_baseline_record, secondary_metric)
                primary_ok = (
                    candidate_primary is not None
                    and baseline_primary is not None
                    and not primary_w1_backend_blockers
                    and (
                        candidate_primary >= baseline_primary
                        if primary_direction == "greater"
                        else candidate_primary <= baseline_primary
                    )
                )
                secondary_ok = (
                    candidate_secondary is not None
                    and baseline_secondary is not None
                    and (
                        candidate_secondary >= baseline_secondary
                        if secondary_direction == "greater"
                        else candidate_secondary <= baseline_secondary
                    )
                )
                optional_sota_gate.update(
                    {
                        "primary_metric": primary_metric,
                        "primary_direction": primary_direction,
                        "primary_candidate": candidate_primary,
                        "primary_sota_baseline": baseline_primary,
                        "primary_baseline_algorithm": str(primary_baseline_record.get("algorithm_name") or ""),
                        "primary_baseline_run_ids": list(primary_baseline_record.get("baseline_run_ids") or []),
                        "primary_w1_backend_provenance_blockers": list(primary_w1_backend_blockers),
                        "primary_ok": bool(primary_ok),
                        "claim_metric": secondary_metric,
                        "claim_direction": secondary_direction,
                        "claim_candidate": candidate_secondary,
                        "claim_sota_baseline": baseline_secondary,
                        "claim_baseline_algorithm": str(secondary_baseline_record.get("algorithm_name") or ""),
                        "claim_baseline_run_ids": list(secondary_baseline_record.get("baseline_run_ids") or []),
                        "claim_ok": bool(secondary_ok),
                        "ok": bool(primary_ok and secondary_ok),
                    }
                )
                missing = []
                if not baseline:
                    missing.append("external baseline metrics")
                if candidate_primary is None:
                    missing.append(f"active best primary metric `{primary_metric}`")
                if baseline_primary is None:
                    missing.append(f"SOTA/baseline primary metric `{primary_metric}`")
                if primary_w1_backend_blockers:
                    missing.extend(primary_w1_backend_blockers)
                if secondary_metric and candidate_secondary is None:
                    missing.append(f"active best claim metric `{secondary_metric}`")
                if secondary_metric and baseline_secondary is None:
                    missing.append(f"SOTA/baseline claim metric `{secondary_metric}`")
                if missing:
                    optional_sota_gate["missing"] = missing
            else:
                optional_sota_gate["ok"] = True
            checks["stage3_optional_sota_gate"] = optional_sota_gate
            checks["stage3_tuning_policy"] = {
                "pass_fail_gate": False,
                "ok_if_active_best_exists": True,
                "min_trials_before_advance": min_trials_before_advance,
                "claim_metric_regression_tolerance": float(policy.get("secondary_tolerance", 0.05) or 0.05),
                "note": (
                    "Stage 3 is optional self-tuning after Stage 2 validation; it does not decide whether the algorithm works, "
                    "but it is not optional effort. W1 is the universal distribution-fit metric and lower is better; tune W1 "
                    "aggressively while preserving the validated claim metric and any explicit Stage 2 locked-gate score floors. "
                    "Before the budget is exhausted, final-regression advance expects the optional SOTA gate to pass; once budget "
                    "is exhausted, advance is allowed without marking the algorithm failed only if the inherited score floors hold."
                ),
            }
            optional_gate_passed = bool(optional_sota_gate.get("ok"))
            budget_allows_advance = bool(optional_sota_gate.get("budget_exhaustion_allows_advance")) and bool(
                budget.get("budget_exhausted")
            )
            if not optional_gate_passed and not budget_allows_advance:
                if min_trials_before_advance > 0 and not min_trials_satisfied:
                    blockers.append(
                        f"stage requires either the optional SOTA gate or at least {min_trials_before_advance} completed trial(s) "
                        f"before budget-exhaustion advance; current={int(budget.get('trial_count') or 0)}"
                    )
                for missing_item in list(optional_sota_gate.get("missing") or []):
                    blockers.append(f"stage3 optional SOTA gate missing: {missing_item}")
                blockers.append("stage3 optional SOTA gate is not met and tuning budget is not exhausted")
        elif stage == "final_regression":
            claim_floor_check = self._evaluate_stage2_claim_metric_regression_floor(
                campaign=campaign,
                stage=stage,
                active_metrics=active_metrics,
            )
            if claim_floor_check:
                checks["stage2_claim_metric_regression_floor"] = claim_floor_check
                if not bool(claim_floor_check.get("ok", False)):
                    blockers.extend([str(item) for item in list(claim_floor_check.get("blockers") or [])])
            checks["final_regression_policy"] = {
                "pass_fail_gate": False,
                "ok_if_active_best_exists": True,
                "confirmation_only": True,
                "note": (
                    "Final regression is a frozen-semantics confirmation/reporting step. It inherits the latest active best "
                    "and locks the release; it does not tune a new candidate or impose an external baseline gate."
                ),
            }
        else:
            primary_metric = str(policy.get("primary_metric") or "")
            primary_direction = str(policy.get("primary_direction") or "lower")
            candidate_primary = self._numeric_value(active_metrics.get("primary_value"))
            primary_baseline_record = self._baseline_record_for_metric(
                baseline,
                primary_metric,
                claim_metric=primary_metric if stage == "stage2_claim_validation" else "",
                direction=primary_direction,
            )
            baseline_primary = self._metric_value(primary_baseline_record, primary_metric)
            if candidate_primary is None:
                blockers.append(f"active best is missing primary metric: {primary_metric}")
            if baseline_primary is None:
                blockers.append(f"external baseline is missing primary metric: {primary_metric}")
            blockers.extend(
                self._w1_backend_provenance_blockers(
                    candidate_metrics=active_metrics,
                    baseline_metrics=primary_baseline_record,
                    metric_name=primary_metric,
                )
            )
            if not blockers and candidate_primary is not None and baseline_primary is not None:
                if stage == "stage1_feasibility":
                    ok_primary = candidate_primary <= baseline_primary * float(policy.get("gate_vs_baseline_multiplier", 1.2))
                elif stage == "stage2_claim_validation":
                    ok_primary = self._metric_improves(
                        candidate_primary,
                        baseline_primary,
                        direction=primary_direction,
                        min_delta=float(policy.get("gate_vs_baseline_min_delta", 0.1) or 0.0),
                        abs_min_delta=float(policy.get("gate_vs_baseline_abs_min_delta", 0.0) or 0.0),
                    )
                else:
                    multiplier = float(policy.get("gate_vs_baseline_multiplier", 1.0))
                    ok_primary = (
                        candidate_primary >= baseline_primary * multiplier
                        if primary_direction == "greater"
                        else candidate_primary <= baseline_primary * multiplier
                    )
                checks["primary_vs_external_baseline"] = {
                    "metric": primary_metric,
                    "direction": primary_direction,
                    "candidate": candidate_primary,
                    "baseline": baseline_primary,
                    "baseline_algorithm": str(primary_baseline_record.get("algorithm_name") or ""),
                    "baseline_run_ids": list(primary_baseline_record.get("baseline_run_ids") or []),
                    "ok": ok_primary,
                }
                checks["primary_vs_external_baseline"].update(
                    self._prefixed_w1_backend_metadata(active_metrics, "candidate")
                )
                checks["primary_vs_external_baseline"].update(
                    self._prefixed_w1_backend_metadata(primary_baseline_record, "baseline")
                )
                if not ok_primary:
                    blockers.append("active best does not satisfy the stage primary metric gate versus external baseline")
        if stage == "stage2_claim_validation":
            w1_candidate = self._numeric_value(active_metrics.get("w1_mean"))
            w1_baseline = self._metric_value(w1_sota_baseline, "w1_mean")
            w1_backend_blockers = self._w1_backend_provenance_blockers(
                candidate_metrics=active_metrics,
                baseline_metrics=w1_sota_baseline,
                metric_name="w1_mean",
            )
            blockers.extend(w1_backend_blockers)
            if w1_candidate is None or w1_baseline is None:
                blockers.append("stage2 gate requires W1 on active best and baseline")
            else:
                effective_multiplier, multiplier_reason = self._effective_stage2_w1_multiplier(policy, w1_baseline)
                w1_ok = w1_candidate <= w1_baseline * effective_multiplier
                checks["w1_secondary_vs_external_baseline"] = {
                    "candidate": w1_candidate,
                    "baseline": w1_baseline,
                    "baseline_algorithm": str(w1_sota_baseline.get("algorithm_name") or ""),
                    "baseline_run_ids": list(w1_sota_baseline.get("baseline_run_ids") or []),
                    "configured_multiplier": float(policy.get("w1_vs_baseline_multiplier", 1.5) or 1.5),
                    "effective_multiplier": effective_multiplier,
                    "multiplier_reason": multiplier_reason,
                    "allowed_w1": w1_baseline * effective_multiplier,
                    "w1_backend_provenance_blockers": list(w1_backend_blockers),
                    "ok": bool(w1_ok and not w1_backend_blockers),
                }
                checks["w1_secondary_vs_external_baseline"].update(
                    self._prefixed_w1_backend_metadata(active_metrics, "candidate")
                )
                checks["w1_secondary_vs_external_baseline"].update(
                    self._prefixed_w1_backend_metadata(w1_sota_baseline, "baseline")
                )
                if not w1_ok:
                    blockers.append("stage2 W1 regression versus external baseline exceeds tolerance")
        locked_gate_spec = self._locked_stage_gate_for_campaign_stage(campaign, stage)
        if locked_gate_spec:
            locked_gate_check = self._evaluate_locked_stage_gate(
                gate_spec=locked_gate_spec,
                active_metrics=active_metrics,
            )
            checks["locked_stage_gate"] = locked_gate_check
            if not bool(locked_gate_check.get("ok", False)):
                blockers.extend([str(item) for item in list(locked_gate_check.get("blockers") or [])])
        ok = not blockers
        next_stage = ""
        if ok:
            order = list(campaign.get("stage_order") or CAMPAIGN_STAGE_ORDER)
            try:
                idx = order.index(stage)
            except ValueError:
                idx = len(order) - 1
            if stage != "final_regression" and idx < len(order) - 1:
                next_stage = str(order[idx + 1])
        stage_state["last_gate_check"] = {
            "ok": ok,
            "checked_at": self._now_iso(),
            "advance_requested": bool(advance),
            "next_stage": next_stage,
            "blockers": list(blockers),
        }
        stage_state["stage_gate_evidence"] = {
            "ok": ok,
            "checked_at": stage_state["last_gate_check"]["checked_at"],
            "advance_requested": bool(advance),
            "stage": stage,
            "active_best_trial_id": active_trial_id,
            "active_best_snapshot_id": str(stage_state.get("active_best_snapshot_id") or ""),
            "active_best_commit": str(stage_state.get("active_best_commit") or ""),
            "active_best_run_ids": list(stage_state.get("active_best_run_ids") or []),
            "active_best_metrics_summary": active_metrics,
            "external_baseline_metrics": baseline,
            "stage_panel": stage_panel,
            "policy": policy,
            "checks": checks,
            "blockers": list(blockers),
            "next_stage": next_stage,
        }
        if ok:
            stage_state["gate_ready"] = True
            stage_state["gate_passed_at"] = stage_state["last_gate_check"]["checked_at"]
            stage_state["gate_passed_trial_id"] = active_trial_id
            if stage == "stage3_tuning":
                stage3_sota_gate = dict(checks.get("stage3_optional_sota_gate") or {})
                stage_state["stage3_optional_sota_gate_passed"] = bool(stage3_sota_gate.get("ok"))
                stage_state["stage3_advanced_after_budget_exhaustion"] = bool(
                    not stage3_sota_gate.get("ok") and budget.get("budget_exhausted")
                )
                if stage3_sota_gate.get("ok"):
                    stage_state["stage3_optional_sota_gate_passed_at"] = stage_state["last_gate_check"]["checked_at"]
            if stage == "stage2_claim_validation":
                stage_state["algorithm_validated"] = True
                campaign["algorithm_validated"] = True
                campaign["validated_at"] = self._now_iso()
            if not advance:
                stages[stage] = stage_state
            elif stage == "final_regression" or not next_stage:
                stage_state["status"] = "locked"
                campaign["status"] = "locked"
                locked_release = {
                    "trial_id": active_trial_id,
                    "snapshot_id": str(stage_state.get("active_best_snapshot_id") or ""),
                    "commit": str(stage_state.get("active_best_commit") or ""),
                    "run_ids": list(stage_state.get("active_best_run_ids") or []),
                    "locked_at": self._now_iso(),
                }
                campaign["locked_release"] = locked_release
                registry = self._bootstrap_algorithm_registry(algo_id)
                registry["algorithm_lifecycle_status"] = "complete"
                registry["algorithm_lifecycle_status_reason"] = (
                    f"final_regression locked release for campaign {campaign.get('campaign_id')}"
                )
                registry["completed_campaign_id"] = str(campaign.get("campaign_id") or "")
                registry["completed_release"] = dict(locked_release)
                if active_trial_id:
                    registry["latest_completed_trial_id"] = active_trial_id
                active_run_ids = [str(item or "").strip() for item in list(locked_release.get("run_ids") or []) if str(item or "").strip()]
                if active_run_ids:
                    registry["latest_accepted_run_id"] = active_run_ids[-1]
                    registry["active_baseline_run_id"] = active_run_ids[-1]
                snapshot_id = str(locked_release.get("snapshot_id") or "")
                if snapshot_id:
                    registry["active_workspace_snapshot_id"] = snapshot_id
                self._save_algorithm_registry(
                    algo_id,
                    registry,
                    sync_active_context=self._is_current_active_algorithm(algo_id),
                )
            else:
                stage_state["status"] = "passed"
                campaign["current_stage"] = next_stage
                next_state = stages.setdefault(next_stage, {})
                next_state["status"] = "active"
                if next_stage in {"stage3_tuning", "final_regression"} and not str(next_state.get("active_best_trial_id") or "").strip():
                    next_state["active_best_trial_id"] = active_trial_id
                    next_state["active_best_commit"] = str(stage_state.get("active_best_commit") or "")
                    next_state["active_best_snapshot_id"] = str(stage_state.get("active_best_snapshot_id") or "")
                    next_state["active_best_run_ids"] = list(stage_state.get("active_best_run_ids") or [])
                    next_policy = self._campaign_stage_policy(campaign, next_stage)
                    next_state["active_best_metrics_summary"] = self._summarize_campaign_metrics(
                        active_metrics,
                        next_policy,
                    )
                    next_state["seeded_from_stage"] = stage
                    next_state["seeded_at"] = self._now_iso()
                    if next_stage == "stage3_tuning" and baseline and not next_state.get("external_baseline_metrics"):
                        next_state["external_baseline_metrics"] = deepcopy(baseline)
                        inherited_source = (
                            stage_state.get("external_baseline_source")
                            if isinstance(stage_state.get("external_baseline_source"), dict)
                            else {}
                        )
                        next_state["external_baseline_source"] = {
                            **dict(inherited_source or {}),
                            "source": "inherited_from_stage2_claim_validation",
                            "inherited_from_stage": stage,
                            "inherited_at": self._now_iso(),
                        }
                    if next_stage == "final_regression":
                        next_state["confirmation_only"] = True
                        next_state["note"] = (
                            "Final regression inherits the latest active best from the previous stage and "
                            "locks that frozen release; it is not a tuning stage."
                        )
                    stage2_panel = stage_state.get("stage_panel") if isinstance(stage_state.get("stage_panel"), dict) else {}
                    stage2_panel_ids = [
                        str(item or "").strip()
                        for item in list(stage2_panel.get("target_dataset_ids") or active_panel_ids or [])
                        if str(item or "").strip()
                    ]
                    if next_stage == "stage3_tuning" and stage2_panel_ids:
                        next_state["claim_guardrail_dataset_ids"] = stage2_panel_ids
                        next_state["required_dataset_ids"] = stage2_panel_ids
                        inherited_payload = dict(stage2_panel.get("dataset_config_overrides") or {})
                        if inherited_payload:
                            existing_stage3_panel = (
                                next_state.get("stage_panel")
                                if isinstance(next_state.get("stage_panel"), dict)
                                else {}
                            )
                            if existing_stage3_panel:
                                merged_payload = dict(existing_stage3_panel.get("dataset_config_overrides") or {})
                                existing_ids = self._campaign_target_dataset_ids(merged_payload)
                                missing_stage2_ids = self._missing_required_panel_ids(existing_ids, stage2_panel_ids)
                                if missing_stage2_ids:
                                    datasets = list(merged_payload.get("datasets") or [])
                                    stage2_datasets = {
                                        str((item or {}).get("dataset_id") or (item or {}).get("id") or "").strip(): dict(item or {})
                                        for item in list(inherited_payload.get("datasets") or [])
                                        if isinstance(item, dict)
                                    }
                                    for dataset_id in missing_stage2_ids:
                                        datasets.append(stage2_datasets.get(dataset_id) or {"dataset_id": dataset_id})
                                    merged_payload["datasets"] = datasets
                                    merged_payload["target_dataset_ids"] = existing_ids + missing_stage2_ids
                                    next_state["stage_panel"] = {
                                        **existing_stage3_panel,
                                        "target_dataset_ids": existing_ids + missing_stage2_ids,
                                        "dataset_config_overrides": merged_payload,
                                        "source": "stage3_panel_plus_inherited_stage2_guardrail",
                                        "updated_at": self._now_iso(),
                                        "note": "Stage 2 claim-validation panel was automatically added to Stage 3 as the claim-regression guardrail.",
                                    }
                            else:
                                inherited_payload["target_dataset_ids"] = stage2_panel_ids
                                next_state["stage_panel"] = {
                                    "target_dataset_ids": stage2_panel_ids,
                                    "dataset_config_overrides": inherited_payload,
                                    "source": "inherited_stage2_guardrail",
                                    "frozen_at": self._now_iso(),
                                    "note": "Stage 2 claim-validation panel is automatically carried into Stage 3 as the claim-regression guardrail.",
                                }
        else:
            stage_state["gate_ready"] = False
        campaign["stages"] = stages
        self._save_campaign(campaign)
        payload = {
            "campaign_id": str(campaign.get("campaign_id") or ""),
            "algorithm_id": algo_id,
            "stage": stage,
            "ok": ok,
            "advanced": bool(ok and advance),
            "advance_requested": bool(advance),
            "blockers": blockers,
            "checks": checks,
            "active_best_trial_id": active_trial_id,
            "next_stage": next_stage,
            "campaign_status": str(campaign.get("status") or ""),
            "algorithm_validated": bool(campaign.get("algorithm_validated", False)),
            "trial_count": budget["trial_count"],
            "max_trials": budget["max_trials"],
            "remaining_trials": budget["remaining_trials"],
            "budget_exhausted": budget["budget_exhausted"],
        }
        self._emit("algorithm_campaign_stage_gate_checked", payload)
        return payload

    def activate_algorithm_workspace(
        self,
        algorithm_id: str,
        *,
        target_snapshot_id: str = "",
        target_proposal_id: str = "",
    ) -> Dict[str, Any]:
        algo_id = str(algorithm_id or "").strip().lower()
        if not algo_id:
            raise ValueError("algorithm_id is required.")
        algo_dir = self._proposal_dir(algo_id)
        if not algo_dir.exists():
            raise FileNotFoundError(f"Algorithm workspace does not exist: {algo_dir}")

        registry = self._bootstrap_algorithm_registry(algo_id)
        self._ensure_algorithm_id_mutable(algo_id, action="activate_algorithm_workspace")
        changed = False
        snapshot_id = str(target_snapshot_id or "").strip()
        proposal_id = str(target_proposal_id or "").strip()
        snapshots = dict(registry.get("workspace_snapshots") or {})
        proposals = dict(registry.get("proposals") or {})

        if proposal_id and proposal_id not in proposals:
            raise ValueError(f"Proposal '{proposal_id}' not found for algorithm '{algo_id}'.")

        if proposal_id and not snapshot_id:
            candidates = [
                snap_id
                for snap_id, meta in snapshots.items()
                if str((meta or {}).get("proposal_id") or "") == proposal_id
            ]
            if candidates:
                snapshot_id = sorted(candidates)[-1]

        if snapshot_id:
            if snapshot_id not in snapshots:
                raise ValueError(f"Workspace snapshot '{snapshot_id}' not found for algorithm '{algo_id}'.")
            if str(registry.get("active_workspace_snapshot_id") or "") != snapshot_id:
                registry["active_workspace_snapshot_id"] = snapshot_id
                changed = True
            meta = dict(snapshots.get(snapshot_id) or {})
            if not proposal_id and meta.get("proposal_id"):
                proposal_id = str(meta.get("proposal_id") or "").strip()

        if proposal_id and str(registry.get("active_proposal_id") or "") != proposal_id:
            registry["active_proposal_id"] = proposal_id
            changed = True

        if changed:
            self._save_algorithm_registry(algo_id, registry, sync_active_context=True)
        else:
            self._sync_registry_summary_to_state(algo_id, registry, update_active_context=True)

        active_context = dict(self.state.get("active_algorithm_context") or {})
        self.state["latest_training_algorithm_id"] = algo_id
        self.state["latest_algorithm_proposal_id"] = algo_id if str(registry.get("active_proposal_id") or "").strip() else ""

        payload = {
            "algorithm_id": algo_id,
            "active_algorithm_id": algo_id,
            "workspace_path": str(algo_dir),
            "active_proposal_id": str(registry.get("active_proposal_id") or ""),
            "active_workspace_snapshot_id": str(registry.get("active_workspace_snapshot_id") or ""),
            "active_baseline_run_id": str(registry.get("active_baseline_run_id") or ""),
            "proposal_id": str(active_context.get("proposal_id") or registry.get("active_proposal_id") or ""),
            "snapshot_id": str(active_context.get("active_snapshot_id") or registry.get("active_workspace_snapshot_id") or ""),
            "dirty_since_snapshot": bool(active_context.get("dirty_since_snapshot", False)),
        }
        self._emit("planner_algorithm_workspace_activated", payload)
        return payload

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

    @staticmethod
    def _render_research_idea_markdown(record: Dict[str, Any]) -> str:
        lines: List[str] = [
            f"# Research Idea: {record.get('title') or record.get('idea_id') or '(untitled)'}",
            "",
            "## Metadata",
            f"- Idea ID: {record.get('idea_id','')}",
            f"- Current Revision ID: {record.get('current_revision_id','')}",
            f"- Primary Track: {record.get('primary_track','')}",
            f"- Alternate Track: {record.get('alternate_track','') or '(none)'}",
            f"- Review Mode: {record.get('review_mode','')}",
            f"- Review Status: {record.get('review_status','')}",
            f"- Portfolio Status: {record.get('portfolio_status','')}",
            f"- Resolution Status: {record.get('resolution_status','')}",
            f"- Execution Status: {record.get('execution_status','')}",
            f"- Active: {bool(record.get('active', False))}",
            f"- Created At: {record.get('created_at','')}",
            f"- Updated At: {record.get('updated_at','')}",
        ]
        if record.get("reviewed_at"):
            lines.append(f"- Reviewed At: {record.get('reviewed_at','')}")
        if record.get("supersedes_revision_id"):
            lines.append(f"- Supersedes Revision: {record.get('supersedes_revision_id','')}")
        if record.get("superseded_by_revision_id"):
            lines.append(f"- Superseded By Revision: {record.get('superseded_by_revision_id','')}")
        linked_algorithms = PlannerFileTools._normalize_linked_algorithms(record.get("linked_algorithms") or [])
        lines.append(f"- Linked Algorithms: {', '.join(linked_algorithms) if linked_algorithms else '(none)'}")
        lines.append("")
        sections = [
            ("Problem Definition", "problem_definition"),
            ("Scientific Object", "scientific_object"),
            ("Current-Method Failure Mode", "current_method_failure_mode"),
            ("Prior Work / Research Basis", "prior_work"),
            ("Why This Matters", "why_this_matters"),
            ("Why Now", "why_now"),
            ("Falsifiable Success Criteria", "falsifiable_success_criteria"),
            ("Non-goals / What Not To Claim", "non_goals"),
            ("Evidence Basis", "evidence_basis"),
            ("Feasible Direction Families", "feasible_direction_families"),
            ("Feasibility Constraints / Impossible Conditions", "feasibility_constraints"),
        ]
        for title, key in sections:
            lines.append(f"## {title}")
            lines.append(str(record.get(key, "")).strip() or "(empty)")
            lines.append("")
        feedback = str(record.get("review_feedback", "")).strip()
        lines.append("## Review Feedback")
        lines.append(feedback or "(none)")
        lines.append("")
        history = list(record.get("review_history") or [])
        lines.append("## Review History")
        if history:
            for item in history:
                lines.append(
                    f"- [{item.get('at','')}] status={item.get('review_status','')} "
                    f"feedback={item.get('feedback','') or '(none)'}"
                )
        else:
            lines.append("- (none)")
        lines.append("")
        return "\n".join(lines)

    def _persist_research_idea_record(self, idea_id: str, record: Dict[str, Any]) -> None:
        target = str(idea_id or "").strip().lower()
        idea_dir = self._research_idea_dir(target)
        idea_dir.mkdir(parents=True, exist_ok=True)
        md_path = self._research_idea_markdown_path(target)
        json_path = self._research_idea_json_path(target)
        record["idea_path"] = str(md_path)
        record["idea_json_path"] = str(json_path)
        md_path.write_text(self._render_research_idea_markdown(record), encoding="utf-8")
        json_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    def register_research_idea_record(
        self,
        idea_id: str,
        record: Dict[str, Any],
        *,
        previous_record: Optional[Dict[str, Any]] = None,
        reason: str = "",
    ) -> Dict[str, Any]:
        target = str(idea_id or "").strip().lower()
        registry = self._bootstrap_research_idea_registry(target)
        revision_id = str(record.get("current_revision_id") or "").strip() or self._make_registry_id("idea_revision")
        record["current_revision_id"] = revision_id
        revision_path = self._research_idea_revisions_dir(target) / f"{revision_id}.json"
        record["current_revision_path"] = str(revision_path)

        previous = dict(previous_record or {})
        previous_revision_id = str(previous.get("current_revision_id") or "").strip()
        if previous_revision_id and previous_revision_id != revision_id:
            prev_path = self._research_idea_revisions_dir(target) / f"{previous_revision_id}.json"
            prev_record = self._read_json(prev_path, previous)
            if isinstance(prev_record, dict):
                prev_record["review_status"] = "superseded"
                prev_record["superseded_by_revision_id"] = revision_id
                prev_record["updated_at"] = self._now_iso()
                self._write_json(prev_path, prev_record)
            revisions = registry.setdefault("revisions", {})
            prev_summary = dict(revisions.get(previous_revision_id) or {})
            prev_summary.update(
                {
                    "revision_id": previous_revision_id,
                    "review_status": "superseded",
                    "superseded_by_revision_id": revision_id,
                    "updated_at": self._now_iso(),
                }
            )
            revisions[previous_revision_id] = prev_summary
            self.mark_research_idea_obsolete(
                target,
                artifact_type="revision",
                artifact_id=previous_revision_id,
                reason=str(reason or "").strip() or f"Superseded by {revision_id}",
                replaced_by=revision_id,
            )

        self._write_json(revision_path, record)
        registry.setdefault("revisions", {})[revision_id] = self._normalize_revision_summary(record)
        registry["current_revision_id"] = revision_id
        registry["last_review_status"] = str(record.get("review_status") or "")
        registry["portfolio_status"] = str(record.get("portfolio_status") or "active")
        registry["resolution_status"] = str(record.get("resolution_status") or "unresolved")
        registry["execution_status"] = str(record.get("execution_status") or "idle")
        linked_algorithms = self._normalize_linked_algorithms(record.get("linked_algorithms") or registry.get("linked_algorithms") or [])
        record["linked_algorithms"] = linked_algorithms
        registry["linked_algorithms"] = linked_algorithms
        record["attempt_count"] = int(record.get("attempt_count") or registry.get("attempt_count") or 0)
        registry["attempt_count"] = int(record["attempt_count"])
        self._save_research_idea_registry(target, registry, record=record)
        return record

    def record_research_idea_decision(
        self,
        idea_id: str,
        *,
        phase: str,
        decision: str,
        rationale: str,
        status: str,
        evidence: Optional[List[str]] = None,
        related_artifacts: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        target = str(idea_id or "").strip().lower()
        entry = {
            "id": self._make_registry_id("idea_decision"),
            "idea_id": target,
            "phase": str(phase or "").strip(),
            "decision": str(decision or "").strip(),
            "rationale": str(rationale or "").strip(),
            "status": str(status or "final").strip() or "final",
            "evidence": list(evidence or []),
            "related_artifacts": list(related_artifacts or []),
            "at": self._now_iso(),
        }
        self._append_jsonl(self._research_idea_decisions_path(target), entry)
        registry = self._bootstrap_research_idea_registry(target)
        recent = list(registry.get("recent_decisions") or [])
        recent.append(entry)
        registry["recent_decisions"] = recent[-20:]
        self._save_research_idea_registry(target, registry)
        self._emit("research_idea_decision_recorded", entry)
        return entry

    def mark_research_idea_obsolete(
        self,
        idea_id: str,
        *,
        artifact_type: str,
        artifact_id: str,
        reason: str,
        replaced_by: str = "",
    ) -> Dict[str, Any]:
        target = str(idea_id or "").strip().lower()
        entry = {
            "id": self._make_registry_id("idea_obsolete"),
            "idea_id": target,
            "artifact_type": str(artifact_type or "").strip(),
            "artifact_id": str(artifact_id or "").strip(),
            "reason": str(reason or "").strip(),
            "replaced_by": str(replaced_by or "").strip(),
            "at": self._now_iso(),
        }
        self._append_jsonl(self._research_idea_obsolete_path(target), entry)
        registry = self._bootstrap_research_idea_registry(target)
        recent = list(registry.get("recent_obsolete") or [])
        recent.append(entry)
        registry["recent_obsolete"] = recent[-20:]
        self._save_research_idea_registry(target, registry)
        return entry

    def append_research_idea_attempt(
        self,
        idea_id: str,
        *,
        attempt_type: str,
        summary: str,
        linked_algorithm_id: str = "",
        proposal_id: str = "",
        run_id: str = "",
        evidence_refs: Optional[List[str]] = None,
        next_step: str = "",
        machine_generated: bool = False,
    ) -> Dict[str, Any]:
        target = str(idea_id or "").strip().lower()
        registry = self._bootstrap_research_idea_registry(target)
        entry = {
            "id": self._make_registry_id("idea_attempt"),
            "idea_id": target,
            "attempt_type": str(attempt_type or "").strip(),
            "summary": str(summary or "").strip(),
            "linked_algorithm_id": str(linked_algorithm_id or "").strip().lower(),
            "proposal_id": str(proposal_id or "").strip(),
            "run_id": str(run_id or "").strip(),
            "evidence_refs": [str(item).strip() for item in (evidence_refs or []) if str(item).strip()],
            "next_step": str(next_step or "").strip(),
            "machine_generated": bool(machine_generated),
            "at": self._now_iso(),
        }
        self._append_jsonl(self._research_idea_attempts_path(target), entry)
        recent = list(registry.get("recent_attempts") or [])
        recent.append(entry)
        registry["recent_attempts"] = recent[-20:]
        registry["attempt_count"] = int(registry.get("attempt_count") or 0) + 1
        linked_algorithms = self._normalize_linked_algorithms(
            list(registry.get("linked_algorithms") or [])
            + ([entry["linked_algorithm_id"]] if entry["linked_algorithm_id"] else [])
        )
        registry["linked_algorithms"] = linked_algorithms
        store = self._research_idea_store()
        record = dict(store.get(target) or self._read_json(self._research_idea_json_path(target), {}))
        if record:
            record["linked_algorithms"] = linked_algorithms
            record["attempt_count"] = int(registry["attempt_count"])
            record["updated_at"] = self._now_iso()
            store[target] = record
            self._persist_research_idea_record(target, record)
        self._save_research_idea_registry(target, registry, record=record if record else None)
        self._emit("research_idea_attempt_recorded", entry)
        return entry

    def set_research_idea_active(self, idea_id: str) -> Dict[str, Any]:
        target = str(idea_id or "").strip().lower()
        root = self._research_idea_root()
        if not target:
            raise ValueError("idea_id is required.")
        if not (root / target).exists():
            raise FileNotFoundError(f"Research idea does not exist: {target}")
        ideas = self._research_idea_store()
        for child in sorted(root.iterdir()) if root.exists() else []:
            if not child.is_dir():
                continue
            child_id = child.name.strip().lower()
            record = dict(ideas.get(child_id) or self._read_json(self._research_idea_json_path(child_id), {}))
            if not record:
                continue
            is_active = child_id == target
            if bool(record.get("active")) == is_active:
                if is_active:
                    ideas[child_id] = record
                continue
            record["active"] = is_active
            record["updated_at"] = self._now_iso()
            ideas[child_id] = record
            self._persist_research_idea_record(child_id, record)
            registry = self._load_research_idea_registry(child_id)
            self._save_research_idea_registry(child_id, registry, record=record)
        self.state["active_research_idea_id"] = target
        target_record = dict(ideas.get(target) or self._read_json(self._research_idea_json_path(target), {}))
        target_registry = self._load_research_idea_registry(target)
        self._sync_research_idea_summary_to_state(target, target_registry, record=target_record)
        return target_record

    def _get_proposal_record(self, algorithm_id: str) -> Optional[Dict[str, Any]]:
        algo_id = str(algorithm_id or "").strip().lower()
        if not algo_id:
            return None
        proposals = self._proposal_store()
        json_path = self._proposal_json_path(algo_id)
        raw: Any = None
        if json_path.exists():
            try:
                raw = json.loads(json_path.read_text(encoding="utf-8"))
            except Exception:
                raw = None
        if isinstance(raw, dict):
            # Disk proposal files are the durable source of truth. Session state
            # can be stale after an interrupted turn/resume, so never let an old
            # cached pending record override an approved PROPOSAL.json.
            record = self._reconcile_root_and_registry_proposal_record(algo_id, raw)
            proposals[algo_id] = record
            return record
        record = proposals.get(algo_id)
        if isinstance(record, dict):
            return record
        return None

    _REVIEWED_PROPOSAL_STATUSES = {
        "approved",
        "approved_auto",
        "rejected",
        "revision_requested",
    }

    @classmethod
    def _proposal_record_review_rank(cls, record: Dict[str, Any]) -> int:
        status = str(record.get("status") or "").strip().lower()
        decision = str(record.get("review_decision") or "").strip().lower()
        reviewed_at = str(record.get("reviewed_at") or "").strip()
        if status in cls._REVIEWED_PROPOSAL_STATUSES:
            return 3
        if decision in {"approve", "reject", "revise"} or reviewed_at:
            return 2
        if status.startswith("pending"):
            return 1
        return 0

    def _load_registry_proposal_record(self, algorithm_id: str, proposal_id: str) -> Dict[str, Any]:
        algo_id = str(algorithm_id or "").strip().lower()
        target_id = str(proposal_id or "").strip()
        if not algo_id or not target_id:
            return {}
        registry = self._load_algorithm_registry(algo_id)
        summary = dict((registry.get("proposals") or {}).get(target_id) or {})
        registry_path = str(summary.get("registry_path") or "").strip()
        if not registry_path:
            registry_path = str(self._registry_proposals_dir(algo_id) / f"{target_id}.json")
        loaded = self._read_json(Path(registry_path), {})
        if isinstance(loaded, dict) and str(loaded.get("proposal_id") or "").strip() == target_id:
            record = dict(summary)
            record.update(loaded)
            return record
        return summary

    def _reconcile_root_and_registry_proposal_record(
        self,
        algorithm_id: str,
        root_record: Dict[str, Any],
    ) -> Dict[str, Any]:
        algo_id = str(algorithm_id or "").strip().lower()
        proposal_id = str(root_record.get("proposal_id") or "").strip()
        if not algo_id or not proposal_id:
            return root_record
        registry_record = self._load_registry_proposal_record(algo_id, proposal_id)
        if not registry_record:
            return root_record
        if str(registry_record.get("proposal_id") or "").strip() != proposal_id:
            return root_record

        root_rank = self._proposal_record_review_rank(root_record)
        registry_rank = self._proposal_record_review_rank(registry_record)
        selected = root_record
        if registry_rank > root_rank:
            selected = registry_record
        elif registry_rank == root_rank and registry_rank > 0:
            root_updated = str(root_record.get("updated_at") or "")
            registry_updated = str(registry_record.get("updated_at") or "")
            if registry_updated and registry_updated > root_updated:
                selected = registry_record

        if selected is root_record:
            return root_record

        healed = dict(selected)
        healed["proposal_path"] = str(healed.get("proposal_path") or self._proposal_markdown_path(algo_id))
        healed["proposal_json_path"] = str(healed.get("proposal_json_path") or self._proposal_json_path(algo_id))
        self._write_json(self._proposal_json_path(algo_id), healed)
        self._sync_proposal_markdown_metadata_status(algo_id, healed)
        return healed

    def _sync_proposal_markdown_metadata_status(self, algorithm_id: str, record: Dict[str, Any]) -> None:
        md_path = self._proposal_markdown_path(algorithm_id)
        if not md_path.exists():
            return
        status = str(record.get("status") or "").strip()
        updated_at = str(record.get("updated_at") or "").strip()
        reviewed_at = str(record.get("reviewed_at") or "").strip()
        review_decision = str(record.get("review_decision") or "").strip()
        try:
            lines = md_path.read_text(encoding="utf-8").splitlines()
        except Exception:
            return
        replacements = {
            "- Status:": f"- Status: {status}" if status else "",
            "- Updated At:": f"- Updated At: {updated_at}" if updated_at else "",
        }
        if reviewed_at:
            replacements["- Reviewed At:"] = f"- Reviewed At: {reviewed_at}"
        if review_decision:
            replacements["- Review Decision:"] = f"- Review Decision: {review_decision}"
        changed = False
        seen = set()
        for idx, line in enumerate(lines):
            stripped = line.strip()
            for prefix, replacement in replacements.items():
                if stripped.startswith(prefix) and replacement:
                    if line != replacement:
                        lines[idx] = replacement
                        changed = True
                    seen.add(prefix)
                    break
        if reviewed_at and "- Reviewed At:" not in seen:
            try:
                updated_idx = next(idx for idx, line in enumerate(lines) if line.strip().startswith("- Updated At:"))
                lines.insert(updated_idx + 1, f"- Reviewed At: {reviewed_at}")
                changed = True
            except StopIteration:
                pass
        if review_decision and "- Review Decision:" not in seen:
            try:
                updated_idx = next(idx for idx, line in enumerate(lines) if line.strip().startswith("- Updated At:"))
                lines.insert(updated_idx + 1, f"- Review Decision: {review_decision}")
                changed = True
            except StopIteration:
                pass
        if changed:
            md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    @staticmethod
    def _algorithm_root() -> Path:
        return get_cellcompass_root() / "training_algorithms"

    @classmethod
    def _algorithm_id_from_path(cls, path: Path) -> Optional[str]:
        root = cls._algorithm_root().resolve()
        try:
            rel = path.resolve().relative_to(root)
        except Exception:
            return None
        if not rel.parts:
            return None
        return str(rel.parts[0]).strip().lower()

    def _ensure_algorithm_id_proposal_approved(self, algorithm_id: str, action: str, path: Optional[Path] = None) -> None:
        algo_id = str(algorithm_id or "").strip().lower()
        if not algo_id:
            return
        self._bootstrap_algorithm_registry(algo_id)
        record = self._get_proposal_record(algo_id)
        status = str((record or {}).get("status") or "").strip().lower()
        if not status:
            status = self._read_markdown_status(algo_id)
        if status in {"approved", "approved_auto"}:
            return
        reason = (
            f"Algorithm '{algo_id}' proposal is not approved "
            f"(status={status or 'missing'}). Create/review proposal first."
        )
        self._emit(
            "algorithm_proposal_gate_blocked",
            {
                "algorithm_id": algo_id,
                "action": action,
                "path": str(path) if path else "",
                "reason": reason,
            },
        )
        raise ValueError(reason)

    @staticmethod
    def _algorithm_registry_is_final_locked(registry: Dict[str, Any]) -> bool:
        status = str(registry.get("algorithm_lifecycle_status") or "").strip().lower()
        completed_campaign = str(registry.get("completed_campaign_id") or "").strip()
        completed_release = registry.get("completed_release")
        has_completed_release = isinstance(completed_release, dict) and bool(completed_release)
        return status in {"complete", "completed", "locked", "final"} or bool(completed_campaign) or has_completed_release

    def _ensure_algorithm_id_mutable(self, algorithm_id: str, action: str, path: Optional[Path] = None) -> None:
        algo_id = str(algorithm_id or "").strip().lower()
        if not algo_id:
            return
        registry = self._load_algorithm_registry(algo_id)
        if self._algorithm_registry_is_final_locked(registry):
            completed_campaign = str(registry.get("completed_campaign_id") or "").strip()
            reason = (
                f"Algorithm '{algo_id}' is final-regression locked/read-only "
                f"(status={registry.get('algorithm_lifecycle_status') or 'complete'}, "
                f"completed_campaign_id={completed_campaign or 'recorded'}). "
                "Do not mutate completed algorithms; create a new algorithm_id/workspace for new data or new development."
            )
            self._emit(
                "algorithm_lifecycle_lock_blocked",
                {
                    "algorithm_id": algo_id,
                    "action": action,
                    "path": str(path) if path else "",
                    "algorithm_lifecycle_status": str(registry.get("algorithm_lifecycle_status") or ""),
                    "completed_campaign_id": completed_campaign,
                    "reason": reason,
                },
            )
            raise ValueError(reason)
        self._ensure_algorithm_write_owner(algo_id, action=action, path=path)

    def _ensure_algorithm_path_mutable(self, path: Path, action: str) -> None:
        algo_id = self._algorithm_id_from_path(path)
        if algo_id:
            self._ensure_algorithm_id_mutable(algo_id, action=action, path=path)

    def _ensure_algorithm_path_proposal_approved(self, path: Path, action: str) -> None:
        algo_id = self._algorithm_id_from_path(path)
        if algo_id:
            self._ensure_algorithm_id_proposal_approved(algo_id, action=action, path=path)

    def _ensure_algorithm_path_active_for_write(self, path: Path, action: str) -> None:
        target_algo = self._algorithm_id_from_path(path)
        if not target_algo:
            return
        self._ensure_algorithm_id_mutable(target_algo, action=action, path=path)
        active_context = self.get_active_algorithm_context()
        active_algo = str(active_context.get("algorithm_id") or "").strip().lower()
        if active_algo == target_algo:
            return
        reason = (
            f"Workspace write target belongs to algorithm '{target_algo}', but the active authoring "
            f"context is '{active_algo or '(unset)'}'. Call activate_algorithm_workspace('{target_algo}') "
            "before editing that algorithm workspace."
        )
        self._emit(
            "algorithm_context_gate_blocked",
            {
                "stage": "authoring",
                "algorithm_id": target_algo,
                "active_algorithm_id": active_algo,
                "target_algorithm_id": target_algo,
                "action": action,
                "path": str(path),
                "reason": reason,
            },
        )
        raise ValueError(reason)

    def _algorithm_context_payload_for_paths(self, paths: List[Path]) -> Dict[str, Any]:
        active_context = self.get_active_algorithm_context()
        target_ids: List[str] = []
        for path in paths:
            target = self._algorithm_id_from_path(path)
            if target and target not in target_ids:
                target_ids.append(target)
        return {
            "active_algorithm_id": str(active_context.get("algorithm_id") or ""),
            "target_algorithm_id": target_ids[0] if len(target_ids) == 1 else "",
            "target_algorithm_ids": target_ids,
            "proposal_id": str(active_context.get("proposal_id") or ""),
            "snapshot_id": str(active_context.get("active_snapshot_id") or ""),
            "dirty_since_snapshot": bool(active_context.get("dirty_since_snapshot", False)),
        }

    def _mark_algorithm_paths_dirty(self, paths: List[Path], reason: str = "") -> Dict[str, Any]:
        target_ids: List[str] = []
        for path in paths:
            target = self._algorithm_id_from_path(path)
            if target and target not in target_ids:
                target_ids.append(target)
        if len(target_ids) != 1:
            return self._algorithm_context_payload_for_paths(paths)
        algo_id = target_ids[0]
        registry = self._bootstrap_algorithm_registry(algo_id)
        dirty = dict(registry.get("workspace_dirty") or self._default_workspace_dirty())
        existing = [str(item) for item in (dirty.get("dirty_paths") or []) if str(item).strip()]
        for path in paths:
            text_path = str(path)
            if text_path not in existing:
                existing.append(text_path)
        dirty["dirty_since_snapshot"] = True
        dirty["dirty_paths"] = existing[-50:]
        dirty["last_mutation_at"] = self._now_iso()
        dirty["reason"] = str(reason or "").strip()
        registry["workspace_dirty"] = dirty
        self._save_algorithm_registry(algo_id, registry, sync_active_context=self._is_current_active_algorithm(algo_id))
        return self._algorithm_context_payload_for_paths(paths)

    def get_inference_review_record(self, algorithm_id: str) -> Dict[str, Any]:
        algo_id = str(algorithm_id or "").strip().lower()
        if not algo_id:
            return {}
        registry = self._bootstrap_algorithm_registry(algo_id)
        record = registry.get("inference_review")
        return dict(record) if isinstance(record, dict) else {}

    def get_implementation_review_record(self, algorithm_id: str) -> Dict[str, Any]:
        algo_id = str(algorithm_id or "").strip().lower()
        if not algo_id:
            return {}
        registry = self._bootstrap_algorithm_registry(algo_id)
        record = registry.get("implementation_review")
        return dict(record) if isinstance(record, dict) else {}

    @staticmethod
    def _render_review_list(items: List[str]) -> List[str]:
        return [f"- {item}" for item in items] if items else ["- (none)"]

    @classmethod
    def _render_implementation_review_risk_section(cls, record: Dict[str, Any]) -> str:
        blocking_issues = [str(item).strip() for item in (record.get("blocking_issues") or []) if str(item).strip()]
        advisory_risks = [str(item).strip() for item in (record.get("advisory_risks") or []) if str(item).strip()]
        efficiency_recommendations = [
            str(item).strip()
            for item in (record.get("efficiency_recommendations") or [])
            if str(item).strip()
        ]
        generalization_shortcut_risks = [
            str(item).strip()
            for item in (record.get("generalization_shortcut_risks") or [])
            if str(item).strip()
        ]
        risk_assessment = str(record.get("risk_assessment") or "").strip()
        lines: List[str] = [
            IMPLEMENTATION_RISK_SECTION_START,
            "## Implementation Reviewer Risk Assessment",
            "",
            "This section is maintained from the latest `implementation_evaluator` review. "
            "It complements the proposal reviewer risks above: proposal risks focus on theoretical/design concerns, "
            "while implementation-review risks focus on concrete code/config, scalability, runtime, and optimization concerns.",
            "",
            "### Metadata",
            f"- Proposal ID: {record.get('proposal_id','') or '(unknown)'}",
            f"- Review Decision: {record.get('decision','') or '(unknown)'}",
            f"- Review Status: {record.get('status','') or '(unknown)'}",
            f"- Review Hash: {record.get('review_hash','') or '(unknown)'}",
            f"- Reviewed At: {record.get('reviewed_at','') or '(not reviewed)'}",
            "",
            "### Blocking Implementation Issues",
            *cls._render_review_list(blocking_issues),
            "",
            "### Advisory Implementation Risks",
            *cls._render_review_list(advisory_risks),
            "",
            "### Efficiency Recommendations",
            *cls._render_review_list(efficiency_recommendations),
            "",
            "### Generalization Shortcut Risks",
            *cls._render_review_list(generalization_shortcut_risks),
            "",
        ]
        if risk_assessment:
            lines.extend(["### Structured Implementation Risk Notes", "", risk_assessment, ""])
        reviewer_feedback = str(record.get("reviewer_feedback") or "").strip()
        if reviewer_feedback:
            lines.extend(["### Reviewer Feedback", "", reviewer_feedback, ""])
        lines.append(IMPLEMENTATION_RISK_SECTION_END)
        lines.append("")
        return "\n".join(lines)

    @classmethod
    def _merge_implementation_review_risk_section(cls, existing: str, section: str, *, algorithm_id: str) -> str:
        text = str(existing or "").rstrip()
        if not text:
            text = (
                f"# Proposal Risk Assessment: {algorithm_id}\n\n"
                "## Structured Risk Assessment\n"
                "(none yet)\n"
            )
        start = text.find(IMPLEMENTATION_RISK_SECTION_START)
        end = text.find(IMPLEMENTATION_RISK_SECTION_END)
        if start >= 0 and end >= start:
            end += len(IMPLEMENTATION_RISK_SECTION_END)
            return f"{text[:start].rstrip()}\n\n{section.strip()}\n{text[end:].strip()}\n".rstrip() + "\n"
        return f"{text}\n\n{section.strip()}\n"

    def _update_implementation_review_risk_markdown(self, algorithm_id: str, record: Dict[str, Any]) -> str:
        risk_path = self._proposal_risk_path(algorithm_id)
        risk_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            existing = risk_path.read_text(encoding="utf-8") if risk_path.exists() else ""
        except Exception:
            existing = ""
        section = self._render_implementation_review_risk_section(record)
        risk_path.write_text(
            self._merge_implementation_review_risk_section(existing, section, algorithm_id=algorithm_id),
            encoding="utf-8",
        )
        return str(risk_path)

    def record_implementation_review(
        self,
        algorithm_id: str,
        *,
        review_hash: str,
        proposal_id: str = "",
        decision: str,
        reviewer_feedback: str = "",
        findings: Optional[List[str]] = None,
        blocking_issues: Optional[List[str]] = None,
        advisory_risks: Optional[List[str]] = None,
        efficiency_recommendations: Optional[List[str]] = None,
        generalization_shortcut_risks: Optional[List[str]] = None,
        risk_assessment: str = "",
        subagent_id: str = "",
        reviewed_paths: Optional[List[str]] = None,
        review_summary: str = "",
        campaign_id: str = "",
        campaign_trial_count: int = 0,
        review_interval: int = 0,
        review_hash_scope: str = "proposal_semantics",
        implementation_review_policy_version: int = IMPLEMENTATION_REVIEW_POLICY_VERSION,
    ) -> Dict[str, Any]:
        algo_id = str(algorithm_id or "").strip().lower()
        if not algo_id:
            raise ValueError("algorithm_id is required.")
        normalized_decision = str(decision or "").strip().lower()
        if normalized_decision not in {"approve", "revise", "reject"}:
            raise ValueError("decision must be one of approve | revise | reject.")
        registry = self._bootstrap_algorithm_registry(algo_id)
        now = self._now_iso()
        record = {
            "algorithm_id": algo_id,
            "proposal_id": str(proposal_id or registry.get("active_proposal_id") or "").strip(),
            "review_hash": str(review_hash or "").strip(),
            "review_hash_scope": str(review_hash_scope or "proposal_semantics").strip(),
            "implementation_review_policy_version": max(0, int(implementation_review_policy_version or 0)),
            "status": "approved" if normalized_decision == "approve" else "blocked",
            "decision": normalized_decision,
            "reviewer_feedback": str(reviewer_feedback or "").strip(),
            "summary": str(review_summary or "").strip(),
            "findings": [str(item).strip() for item in (findings or []) if str(item).strip()],
            "blocking_issues": [str(item).strip() for item in (blocking_issues or []) if str(item).strip()],
            "advisory_risks": [str(item).strip() for item in (advisory_risks or []) if str(item).strip()],
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
            "subagent_id": str(subagent_id or "").strip(),
            "reviewed_paths": [str(item).strip() for item in (reviewed_paths or []) if str(item).strip()],
            "campaign_id": str(campaign_id or "").strip(),
            "campaign_trial_count": max(0, int(campaign_trial_count or 0)),
            "review_interval": max(0, int(review_interval or 0)),
            "reviewed_at": now,
        }
        record["implementation_risk_path"] = self._update_implementation_review_risk_markdown(algo_id, record)
        history = list(registry.get("implementation_review_history") or [])
        history.append(record)
        registry["implementation_review"] = record
        registry["implementation_review_history"] = history[-50:]
        self._save_algorithm_registry(algo_id, registry, sync_active_context=self._is_current_active_algorithm(algo_id))
        self._emit(
            "algorithm_implementation_review_recorded",
            {
                "algorithm_id": algo_id,
                "proposal_id": record["proposal_id"],
                "decision": normalized_decision,
                "status": record["status"],
                "review_hash": record["review_hash"],
                "review_hash_scope": record["review_hash_scope"],
                "subagent_id": record["subagent_id"],
                "campaign_id": record["campaign_id"],
                "campaign_trial_count": record["campaign_trial_count"],
                "implementation_risk_path": record["implementation_risk_path"],
            },
        )
        return record

    def record_inference_review(
        self,
        algorithm_id: str,
        *,
        review_hash: str,
        decision: str,
        reviewer_feedback: str = "",
        findings: Optional[List[str]] = None,
        blocking_issues: Optional[List[str]] = None,
        advisory_risks: Optional[List[str]] = None,
        subagent_id: str = "",
        reviewed_paths: Optional[List[str]] = None,
        review_summary: str = "",
    ) -> Dict[str, Any]:
        algo_id = str(algorithm_id or "").strip().lower()
        if not algo_id:
            raise ValueError("algorithm_id is required.")
        normalized_decision = str(decision or "").strip().lower()
        if normalized_decision not in {"approve", "revise", "reject"}:
            raise ValueError("decision must be one of approve | revise | reject.")
        registry = self._bootstrap_algorithm_registry(algo_id)
        now = self._now_iso()
        record = {
            "algorithm_id": algo_id,
            "review_hash": str(review_hash or "").strip(),
            "status": "approved" if normalized_decision == "approve" else "blocked",
            "decision": normalized_decision,
            "reviewer_feedback": str(reviewer_feedback or "").strip(),
            "summary": str(review_summary or "").strip(),
            "findings": [str(item).strip() for item in (findings or []) if str(item).strip()],
            "blocking_issues": [str(item).strip() for item in (blocking_issues or []) if str(item).strip()],
            "advisory_risks": [str(item).strip() for item in (advisory_risks or []) if str(item).strip()],
            "subagent_id": str(subagent_id or "").strip(),
            "reviewed_paths": [str(item).strip() for item in (reviewed_paths or []) if str(item).strip()],
            "reviewed_at": now,
        }
        history = list(registry.get("inference_review_history") or [])
        history.append(record)
        registry["inference_review"] = record
        registry["inference_review_history"] = history[-50:]
        self._save_algorithm_registry(algo_id, registry, sync_active_context=self._is_current_active_algorithm(algo_id))
        self._emit(
            "algorithm_inference_review_recorded",
            {
                "algorithm_id": algo_id,
                "decision": normalized_decision,
                "status": record["status"],
                "review_hash": record["review_hash"],
                "subagent_id": record["subagent_id"],
            },
        )
        return record

    def _record_patch_summary(
        self,
        action: str,
        paths: List[Path],
        reason: str = "",
        *,
        mark_dirty: bool = True,
    ) -> None:
        context_payload = (
            self._mark_algorithm_paths_dirty(paths, reason=reason)
            if mark_dirty
            else self._algorithm_context_payload_for_paths(paths)
        )
        summary = {
            "action": action,
            "paths": [str(path) for path in paths],
            "reason": reason,
            "updated_at": self._now_iso(),
            **context_payload,
        }
        self.state["planner_last_patch_summary"] = summary
        self._emit("planner_workspace_updated", summary)

    def create_workspace_snapshot(
        self,
        algorithm_id: str,
        *,
        reason: str,
        source: str,
        proposal_id: str = "",
        set_active: bool = True,
        allow_empty: bool = False,
    ) -> str:
        algo_id = str(algorithm_id or "").strip().lower()
        if not algo_id:
            return ""
        registry = self._load_algorithm_registry(algo_id)
        self._ensure_algorithm_id_mutable(algo_id, action="create_workspace_snapshot")
        proposal_value = str(proposal_id or registry.get("active_proposal_id") or "").strip()
        snapshot_id = self._make_registry_id("snapshot")
        snapshot_dir = self._registry_snapshots_dir(algo_id) / snapshot_id
        snapshot_dir.mkdir(parents=True, exist_ok=False)
        copied: List[str] = []
        for src in self._tracked_workspace_paths(algo_id):
            if not src.exists():
                continue
            shutil.copy2(src, snapshot_dir / src.name)
            copied.append(src.name)
        if not copied and not allow_empty:
            shutil.rmtree(snapshot_dir, ignore_errors=True)
            raise ValueError(f"No workspace files available to snapshot for algorithm '{algo_id}'.")

        snapshot_meta = {
            "snapshot_id": snapshot_id,
            "algorithm_id": algo_id,
            "proposal_id": proposal_value,
            "source": str(source or "").strip(),
            "reason": str(reason or "").strip(),
            "created_at": self._now_iso(),
            "path": str(snapshot_dir),
            "files": copied,
        }
        self._write_json(snapshot_dir / "snapshot_meta.json", snapshot_meta)
        snapshots = registry.setdefault("workspace_snapshots", {})
        snapshots[snapshot_id] = snapshot_meta
        if set_active:
            registry["active_workspace_snapshot_id"] = snapshot_id
            registry["workspace_dirty"] = self._default_workspace_dirty()
        self._save_algorithm_registry(algo_id, registry, sync_active_context=self._is_current_active_algorithm(algo_id))
        active_context = self._build_active_algorithm_context(algo_id, registry)
        self._emit(
            "algorithm_workspace_snapshot_created",
            {
                "algorithm_id": algo_id,
                "active_algorithm_id": self._current_active_algorithm_id(),
                "target_algorithm_id": algo_id,
                "snapshot_id": snapshot_id,
                "proposal_id": proposal_value,
                "source": source,
                "reason": reason,
                "path": str(snapshot_dir),
                "dirty_since_snapshot": bool(active_context.get("dirty_since_snapshot", False)),
            },
        )
        return snapshot_id

    def register_proposal_record(
        self,
        algorithm_id: str,
        record: Dict[str, Any],
        *,
        previous_record: Optional[Dict[str, Any]] = None,
        reason: str = "",
    ) -> Dict[str, Any]:
        algo_id = str(algorithm_id or "").strip().lower()
        registry = self._bootstrap_algorithm_registry(algo_id)
        self._ensure_algorithm_id_mutable(algo_id, action="register_proposal_record")
        proposal_id = str(record.get("proposal_id") or "").strip() or self._make_registry_id("proposal")
        record["proposal_id"] = proposal_id
        registry_path = self._registry_proposals_dir(algo_id) / f"{proposal_id}.json"
        registry_markdown_path = self._registry_proposals_dir(algo_id) / f"{proposal_id}.md"
        registry_risk_path = self._registry_proposals_dir(algo_id) / f"{proposal_id}.risk.md"
        record["registry_path"] = str(registry_path)
        record["proposal_markdown_registry_path"] = str(registry_markdown_path)
        record["risk_registry_path"] = str(registry_risk_path)

        previous = dict(previous_record or {})
        previous_id = str(previous.get("proposal_id") or "").strip()
        if previous_id and previous_id != proposal_id:
            prev_registry_path = self._registry_proposals_dir(algo_id) / f"{previous_id}.json"
            prev_record = self._read_json(prev_registry_path, previous)
            if isinstance(prev_record, dict):
                prev_record["status"] = "superseded"
                prev_record["superseded_by"] = proposal_id
                prev_record["updated_at"] = self._now_iso()
                self._write_json(prev_registry_path, prev_record)
            proposals = registry.setdefault("proposals", {})
            prev_summary = dict(proposals.get(previous_id) or {})
            prev_summary.update(
                {
                    "proposal_id": previous_id,
                    "status": "superseded",
                    "superseded_by": proposal_id,
                    "updated_at": self._now_iso(),
                }
            )
            proposals[previous_id] = prev_summary
            self.record_decision(
                algo_id,
                phase="proposal",
                decision="proposal_superseded",
                alternatives=[],
                evidence=[],
                rationale=str(reason or "").strip() or f"Superseded by {proposal_id}",
                status="final",
                related_artifacts=[
                    {"artifact_type": "proposal", "artifact_id": previous_id},
                    {"artifact_type": "proposal", "artifact_id": proposal_id},
                ],
                update_registry=False,
            )
            self._reset_active_campaigns_for_new_proposal(
                algo_id,
                previous_proposal_id=previous_id,
                proposal_id=proposal_id,
                reason=str(reason or "").strip() or f"Proposal {previous_id} superseded by {proposal_id}",
            )

        source_markdown_path = Path(str(record.get("proposal_path") or self._proposal_markdown_path(algo_id)))
        if source_markdown_path.exists():
            try:
                shutil.copy2(source_markdown_path, registry_markdown_path)
            except Exception:
                pass
        source_risk_path = Path(str(record.get("risk_path") or ""))
        if source_risk_path.exists():
            try:
                shutil.copy2(source_risk_path, registry_risk_path)
            except Exception:
                pass
        self._write_json(registry_path, record)
        proposals = registry.setdefault("proposals", {})
        proposals[proposal_id] = {
            "proposal_id": proposal_id,
            "status": str(record.get("status") or ""),
            "created_at": str(record.get("created_at") or ""),
            "updated_at": str(record.get("updated_at") or ""),
            "primary_idea_id": str(record.get("primary_idea_id") or ""),
            "proposal_path": str(record.get("proposal_path") or self._proposal_markdown_path(algo_id)),
            "editable_proposal_path": str(record.get("editable_proposal_path") or record.get("proposal_path") or self._proposal_markdown_path(algo_id)),
            "proposal_json_path": str(record.get("proposal_json_path") or self._proposal_json_path(algo_id)),
            "registry_path": str(registry_path),
            "proposal_markdown_registry_path": str(registry_markdown_path),
            "risk_path": str(record.get("risk_path") or ""),
            "risk_registry_path": str(registry_risk_path),
            "algorithm_attributes": dict(record.get("algorithm_attributes") or {}),
            "superseded_by": str(record.get("superseded_by") or ""),
        }
        registry["active_proposal_id"] = proposal_id
        if previous_id and previous_id != proposal_id:
            registry["algorithm_lifecycle_status"] = "developing"
            registry["algorithm_lifecycle_status_reason"] = f"proposal revised from {previous_id} to {proposal_id}; final_regression locked release required"
            registry["completed_campaign_id"] = ""
            registry["completed_release"] = {}
        registry["latest_review_verdict"] = str(record.get("review_decision") or record.get("status") or "")
        self._save_algorithm_registry(algo_id, registry)
        return record

    def record_decision(
        self,
        algorithm_id: str,
        *,
        phase: str,
        decision: str,
        alternatives: List[str],
        evidence: List[str],
        rationale: str,
        status: str,
        related_artifacts: Optional[List[Dict[str, Any]]] = None,
        update_registry: bool = True,
    ) -> Dict[str, Any]:
        algo_id = str(algorithm_id or "").strip().lower()
        self._ensure_algorithm_id_mutable(algo_id, action="record_decision")
        entry = {
            "id": self._make_registry_id("decision"),
            "algorithm_id": algo_id,
            "phase": str(phase or "").strip(),
            "decision": str(decision or "").strip(),
            "alternatives": list(alternatives or []),
            "evidence": list(evidence or []),
            "rationale": str(rationale or "").strip(),
            "status": str(status or "final").strip() or "final",
            "related_artifacts": list(related_artifacts or []),
            "at": self._now_iso(),
        }
        self._append_jsonl(self._registry_decisions_path(algo_id), entry)
        if update_registry:
            registry = self._bootstrap_algorithm_registry(algo_id)
            recent = list(registry.get("recent_decisions") or [])
            recent.append(entry)
            registry["recent_decisions"] = recent[-20:]
            self._save_algorithm_registry(algo_id, registry)
        self._emit("experiment_decision_recorded", entry)
        return entry

    def mark_algorithm_failed(
        self,
        algorithm_id: str,
        *,
        reason: str,
        evidence: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        algo_id = str(algorithm_id or "").strip().lower()
        if not algo_id:
            raise ValueError("algorithm_id is required")
        failure_reason = str(reason or "").strip()
        if not failure_reason:
            raise ValueError("reason is required")
        evidence_items = [str(item or "").strip() for item in list(evidence or []) if str(item or "").strip()]
        registry = self._bootstrap_algorithm_registry(algo_id)
        self._ensure_algorithm_id_mutable(algo_id, action="mark_algorithm_failed")
        current_status = str(registry.get("algorithm_lifecycle_status") or "developing").strip() or "developing"
        if current_status == "complete":
            raise ValueError(
                "algorithm lifecycle is already complete via final_regression locked release; "
                "revise the proposal or start a new campaign instead of marking the completed release failed"
            )
        registry["algorithm_lifecycle_status"] = "failed"
        registry["algorithm_lifecycle_status_reason"] = failure_reason
        registry["failed_at"] = self._now_iso()
        registry["failure_evidence"] = evidence_items
        registry["completed_campaign_id"] = ""
        registry["completed_release"] = {}
        self._save_algorithm_registry(
            algo_id,
            registry,
            sync_active_context=self._is_current_active_algorithm(algo_id),
        )
        closed_campaign_ids: List[str] = []
        for index_path in sorted(self._campaign_algorithm_dir(algo_id).glob("*/campaign.json")):
            campaign = self._read_json(index_path, {})
            if not isinstance(campaign, dict):
                continue
            if str(campaign.get("algorithm_id") or "").strip().lower() != algo_id:
                continue
            campaign_status = str(campaign.get("status") or "").strip().lower()
            if campaign_status in {"failed", "locked", "complete", "completed", "aborted"}:
                continue
            campaign_id = str(campaign.get("campaign_id") or "").strip()
            campaign["status"] = "failed"
            campaign["failed_at"] = registry["failed_at"]
            campaign["failure_reason"] = failure_reason
            open_trial_id = str(campaign.get("current_trial_id") or "").strip()
            if open_trial_id:
                campaign["failed_open_trial_id"] = open_trial_id
                campaign["current_trial_id"] = ""
            current_stage = str(campaign.get("current_stage") or "").strip()
            stages = campaign.get("stages") if isinstance(campaign.get("stages"), dict) else {}
            stage_state = stages.get(current_stage) if current_stage else None
            if isinstance(stage_state, dict):
                stage_state["status"] = "failed_needs_revision"
                stage_state["failed_at"] = registry["failed_at"]
                stage_state["failure_reason"] = failure_reason
                stages[current_stage] = stage_state
                campaign["stages"] = stages
            self._save_campaign(campaign)
            if campaign_id:
                closed_campaign_ids.append(campaign_id)
        decision = self.record_decision(
            algo_id,
            phase="lifecycle",
            decision="algorithm_failed",
            alternatives=[],
            evidence=evidence_items,
            rationale=failure_reason,
            status="final",
            related_artifacts=[],
            update_registry=True,
        )
        payload = {
            "algorithm_id": algo_id,
            "algorithm_lifecycle_status": "failed",
            "algorithm_lifecycle_status_reason": failure_reason,
            "failed_at": registry["failed_at"],
            "failure_evidence": evidence_items,
            "decision_id": decision.get("id"),
            "closed_campaign_ids": closed_campaign_ids,
            "next_required_action": (
                "Do not stop as complete. Revise/repair this algorithm, revise its proposal if needed, "
                "or design a replacement algorithm that satisfies the user goal."
            ),
        }
        self._emit("algorithm_lifecycle_status_changed", payload)
        return payload

    def mark_result_obsolete(
        self,
        algorithm_id: str,
        *,
        artifact_type: str,
        artifact_id: str,
        reason: str,
        replaced_by: str = "",
    ) -> Dict[str, Any]:
        algo_id = str(algorithm_id or "").strip().lower()
        self._ensure_algorithm_id_mutable(algo_id, action="mark_result_obsolete")
        entry = {
            "id": self._make_registry_id("obsolete"),
            "algorithm_id": algo_id,
            "artifact_type": str(artifact_type or "").strip(),
            "artifact_id": str(artifact_id or "").strip(),
            "reason": str(reason or "").strip(),
            "replaced_by": str(replaced_by or "").strip(),
            "at": self._now_iso(),
        }
        self._append_jsonl(self._registry_obsolete_path(algo_id), entry)
        registry = self._bootstrap_algorithm_registry(algo_id)
        if entry["artifact_type"] == "run":
            run_summary = dict((registry.get("runs") or {}).get(entry["artifact_id"]) or {})
            if run_summary:
                run_summary["obsolete"] = True
                run_summary["obsolete_reason"] = entry["reason"]
                run_summary["replaced_by"] = entry["replaced_by"]
                registry.setdefault("runs", {})[entry["artifact_id"]] = run_summary
        elif entry["artifact_type"] == "proposal":
            proposal_summary = dict((registry.get("proposals") or {}).get(entry["artifact_id"]) or {})
            if proposal_summary:
                proposal_summary["obsolete"] = True
                proposal_summary["obsolete_reason"] = entry["reason"]
                proposal_summary["replaced_by"] = entry["replaced_by"]
                registry.setdefault("proposals", {})[entry["artifact_id"]] = proposal_summary
        recent = list(registry.get("recent_obsolete") or [])
        recent.append(entry)
        registry["recent_obsolete"] = recent[-20:]
        self._save_algorithm_registry(algo_id, registry)
        self._emit("experiment_result_marked_obsolete", entry)
        return entry

    def set_active_baseline_run(self, algorithm_id: str, run_id: str, reason: str) -> Dict[str, Any]:
        algo_id = str(algorithm_id or "").strip().lower()
        target_run_id = str(run_id or "").strip()
        registry = self._bootstrap_algorithm_registry(algo_id)
        self._ensure_algorithm_id_mutable(algo_id, action="set_active_baseline_run")
        runs = registry.get("runs") or {}
        run_summary = dict(runs.get(target_run_id) or {})
        if not run_summary:
            raise ValueError(f"Run '{target_run_id}' is not registered for algorithm '{algo_id}'.")
        if run_summary.get("obsolete"):
            raise ValueError(f"Run '{target_run_id}' is obsolete and cannot be set as baseline.")
        run_summary["decision"] = "promote"
        runs[target_run_id] = run_summary
        registry["runs"] = runs
        registry["active_baseline_run_id"] = target_run_id
        registry["latest_accepted_run_id"] = target_run_id
        self._save_algorithm_registry(algo_id, registry)
        self.record_decision(
            algo_id,
            phase="training",
            decision="set_active_baseline_run",
            alternatives=[],
            evidence=[],
            rationale=reason,
            status="final",
            related_artifacts=[{"artifact_type": "run", "artifact_id": target_run_id}],
        )
        return run_summary

    def record_preview_artifacts(
        self,
        algorithm_id: str,
        *,
        preview_artifacts: Dict[str, Any],
        stage: str,
        data_path: str,
        requested_device: str = "",
        ok: Optional[bool] = None,
    ) -> Dict[str, Any]:
        algo_id = str(algorithm_id or "").strip().lower()
        registry = self._bootstrap_algorithm_registry(algo_id)
        self._ensure_algorithm_id_mutable(algo_id, action="record_preview_artifacts")
        entry = {
            "preview_id": self._make_registry_id("preview"),
            "algorithm_id": algo_id,
            "stage": str(stage or "").strip() or "final",
            "data_path": str(data_path or "").strip(),
            "requested_device": str(requested_device or "").strip(),
            "markdown_path": str((preview_artifacts or {}).get("markdown_path") or ""),
            "json_path": str((preview_artifacts or {}).get("json_path") or ""),
            "ok": bool(ok) if ok is not None else None,
            "active_proposal_id": str(registry.get("active_proposal_id") or ""),
            "active_workspace_snapshot_id": str(registry.get("active_workspace_snapshot_id") or ""),
            "active_baseline_run_id": str(registry.get("active_baseline_run_id") or ""),
            "at": self._now_iso(),
        }
        registry["latest_preview"] = dict(entry)
        recent = list(registry.get("recent_previews") or [])
        recent.append(entry)
        registry["recent_previews"] = recent[-20:]
        self._save_algorithm_registry(algo_id, registry)
        self._emit("experiment_preview_recorded", entry)
        return entry

    def register_training_run(
        self,
        algorithm_id: str,
        *,
        run_manifest: Dict[str, Any],
        metrics: Dict[str, Any],
        data_path: str,
        decision: str = "provisional",
        reason: str = "",
    ) -> Dict[str, Any]:
        algo_id = str(algorithm_id or "").strip().lower()
        registry = self._bootstrap_algorithm_registry(algo_id)
        self._ensure_algorithm_id_mutable(algo_id, action="register_training_run")
        run_id = str(run_manifest.get("run_id") or metrics.get("run_id") or "").strip()
        if not run_id:
            raise ValueError("Cannot register training run without run_id.")
        decision_value = str(decision or "provisional").strip().lower() or "provisional"
        if decision_value not in {"promote", "reject", "provisional"}:
            raise ValueError("decision must be one of promote, reject, provisional")
        run_record = {
            "run_id": run_id,
            "algorithm_id": algo_id,
            "active_algorithm_id": self._current_active_algorithm_id(),
            "target_algorithm_id": algo_id,
            "decision": decision_value,
            "decision_reason": str(reason or "").strip(),
            "status": str(run_manifest.get("status") or ""),
            "stage": str(run_manifest.get("stage") or ""),
            "training_mode": str(run_manifest.get("training_mode") or ""),
            "candidate_name": run_manifest.get("candidate_name"),
            "data_path": str(data_path or ""),
            "data_track_id": f"{algo_id}::{str(data_path or '').strip()}",
            "active_proposal_id": str(registry.get("active_proposal_id") or ""),
            "proposal_id": str(registry.get("active_proposal_id") or ""),
            "primary_idea_id": "",
            "active_workspace_snapshot_id": str(registry.get("active_workspace_snapshot_id") or ""),
            "snapshot_id": str(registry.get("active_workspace_snapshot_id") or ""),
            "dirty_since_snapshot": bool((registry.get("workspace_dirty") or {}).get("dirty_since_snapshot")),
            "run_manifest_path": str(run_manifest.get("run_manifest_path") or ""),
            "run_dir": str(run_manifest.get("run_dir") or metrics.get("run_dir") or ""),
            "trained_model_path": str(run_manifest.get("trained_model_path") or metrics.get("trained_model_path") or ""),
            "metrics_path": str(run_manifest.get("metrics_path") or metrics.get("metrics_path") or ""),
            "resolved_config_path": str(run_manifest.get("resolved_config_path") or metrics.get("resolved_config_path") or ""),
            "algorithm_snapshot_path": str(metrics.get("algorithm_snapshot_path") or ""),
            "created_at": str(run_manifest.get("created_at") or ""),
            "completed_at": str(run_manifest.get("completed_at") or ""),
            "run_verdict": dict(metrics.get("run_verdict") or {}),
            "tmv_mean": self._metric_mean(metrics.get("tmv_scores")),
            "w1_mean": self._metric_mean(metrics.get("w1_scores")),
            "obsolete": False,
        }
        run_path = self._registry_runs_dir(algo_id) / f"{run_id}.json"
        run_record["registry_path"] = str(run_path)
        self._write_json(run_path, run_record)
        registry.setdefault("runs", {})[run_id] = {
            key: run_record.get(key)
            for key in (
                "run_id",
                "decision",
                "decision_reason",
                "status",
                "stage",
                "training_mode",
                "candidate_name",
                "data_track_id",
                "active_proposal_id",
                "active_workspace_snapshot_id",
                "active_algorithm_id",
                "target_algorithm_id",
                "proposal_id",
                "snapshot_id",
                "dirty_since_snapshot",
                "tmv_mean",
                "w1_mean",
                "created_at",
                "completed_at",
                "registry_path",
                "primary_idea_id",
                "obsolete",
            )
        }
        active_proposal_id = str(registry.get("active_proposal_id") or "").strip()
        if active_proposal_id:
            proposal_summary = dict((registry.get("proposals") or {}).get(active_proposal_id) or {})
            run_record["primary_idea_id"] = str(proposal_summary.get("primary_idea_id") or "").strip().lower()
            registry["runs"][run_id]["primary_idea_id"] = run_record["primary_idea_id"]
        if decision_value == "promote":
            promote_snapshot_id = self.create_workspace_snapshot(
                algo_id,
                reason=reason or f"Promoted run {run_id}",
                source="promote_baseline",
                proposal_id=str(registry.get("active_proposal_id") or ""),
                set_active=True,
                allow_empty=False,
            )
            run_record["promote_snapshot_id"] = promote_snapshot_id
            registry["runs"][run_id]["promote_snapshot_id"] = promote_snapshot_id
            registry["active_workspace_snapshot_id"] = promote_snapshot_id
            registry["active_baseline_run_id"] = run_id
            registry["latest_accepted_run_id"] = run_id
            self.record_decision(
                algo_id,
                phase="training",
                decision="promote_run",
                alternatives=[],
                evidence=[],
                rationale=reason or f"Promoted run {run_id} to active baseline.",
                status="final",
                related_artifacts=[{"artifact_type": "run", "artifact_id": run_id}],
                update_registry=False,
            )
        elif decision_value in {"reject", "provisional"}:
            self.record_decision(
                algo_id,
                phase="training",
                decision=f"{decision_value}_run",
                alternatives=[],
                evidence=[],
                rationale=reason or f"Marked run {run_id} as {decision_value}.",
                status="final" if decision_value == "reject" else "provisional",
                related_artifacts=[{"artifact_type": "run", "artifact_id": run_id}],
                update_registry=False,
            )
        self._save_algorithm_registry(algo_id, registry)
        self._write_json(run_path, run_record)
        primary_idea_id = str(run_record.get("primary_idea_id") or "").strip().lower()
        if primary_idea_id:
            self.append_research_idea_attempt(
                primary_idea_id,
                attempt_type="run_evaluated",
                summary=reason or f"Registered training run {run_id} with decision={decision_value}.",
                linked_algorithm_id=algo_id,
                proposal_id=active_proposal_id,
                run_id=run_id,
                evidence_refs=[str(run_record.get("registry_path") or "")],
                next_step="Compare against the idea's falsifiable success criteria before changing resolution status.",
                machine_generated=True,
            )
        self._emit("experiment_training_run_registered", run_record)
        return run_record

    @staticmethod
    def _truncate_history_value(value: Any, *, max_chars: int, max_list_items: int = 8, depth: int = 0) -> Any:
        if depth >= 4:
            text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
            if len(text) <= max_chars:
                return value
            return text[:max_chars] + f"\n...[truncated {len(text) - max_chars} chars]..."
        if isinstance(value, str):
            if len(value) <= max_chars:
                return value
            return value[:max_chars] + f"\n...[truncated {len(value) - max_chars} chars]..."
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {
                str(key): PlannerFileTools._truncate_history_value(
                    val,
                    max_chars=max_chars,
                    max_list_items=max_list_items,
                    depth=depth + 1,
                )
                for key, val in value.items()
            }
        if isinstance(value, (list, tuple)):
            items = [
                PlannerFileTools._truncate_history_value(
                    item,
                    max_chars=max_chars,
                    max_list_items=max_list_items,
                    depth=depth + 1,
                )
                for item in list(value)[:max_list_items]
            ]
            if len(value) > max_list_items:
                items.append({"omitted_items": len(value) - max_list_items})
            return items
        return value

    @staticmethod
    def _recent_registry_items(items: List[Dict[str, Any]], *, limit: int) -> List[Dict[str, Any]]:
        def sort_key(item: Dict[str, Any]) -> str:
            for key in (
                "updated_at",
                "reviewed_at",
                "created_at",
                "registered_at",
                "started_at",
                "completed_at",
                "timestamp",
                "proposal_id",
                "snapshot_id",
                "run_id",
                "decision_id",
            ):
                value = str((item or {}).get(key) or "")
                if value:
                    return value
            return json.dumps(item, ensure_ascii=False, sort_keys=True, default=str)

        return list(sorted(items, key=sort_key, reverse=True)[:limit])

    @staticmethod
    def _summarize_history_record(record: Dict[str, Any], *, keys: List[str], max_chars: int) -> Dict[str, Any]:
        compact: Dict[str, Any] = {}
        for key in keys:
            if key not in record:
                continue
            compact[key] = PlannerFileTools._truncate_history_value(record.get(key), max_chars=max_chars)
        return compact

    def list_experiment_history(
        self,
        algorithm_id: str,
        include_obsolete: bool = False,
        limit: int = 10,
        max_field_chars: int = 1200,
    ) -> str:
        algo_id = str(algorithm_id or "").strip().lower()
        limit = max(1, min(50, int(limit or 10)))
        max_field_chars = max(200, min(4000, int(max_field_chars or 1200)))
        registry_path = self._registry_index_path(algo_id) if algo_id else Path()
        if not algo_id or not registry_path.exists():
            payload = {
                "view": "experiment_history_summary",
                "algorithm_id": algo_id,
                "registry_path": str(registry_path) if algo_id else "",
                "exists": False,
                "active_proposal_id": "",
                "active_workspace_snapshot_id": "",
                "active_baseline_run_id": "",
                "latest_accepted_run_id": "",
                "algorithm_lifecycle_status": "",
                "algorithm_lifecycle_status_reason": "No experiment registry exists for this algorithm_id.",
                "completed_campaign_id": "",
                "latest_review_verdict": "",
                "counts": {
                    "proposals": 0,
                    "workspace_snapshots": 0,
                    "runs": 0,
                    "recent_decisions": 0,
                    "recent_obsolete": 0,
                },
                "limit": limit,
                "max_field_chars": max_field_chars,
                "truncated": False,
                "omitted": {
                    "proposals": 0,
                    "workspace_snapshots": 0,
                    "runs": 0,
                    "recent_decisions": 0,
                    "recent_obsolete": 0,
                },
                "recent_proposals": [],
                "recent_workspace_snapshots": [],
                "recent_runs": [],
                "recent_decisions": [],
                "recent_obsolete": [],
                "full_history_access": {
                    "read_file": "",
                    "note": "No registry exists; read-only history lookup did not create one.",
                },
            }
            return json.dumps(payload, ensure_ascii=False, indent=2)

        registry = self._load_algorithm_registry(algo_id)
        proposals = list((registry.get("proposals") or {}).values())
        runs = list((registry.get("runs") or {}).values())
        if not include_obsolete:
            proposals = [item for item in proposals if not bool(item.get("obsolete"))]
            runs = [item for item in runs if not bool(item.get("obsolete"))]
        snapshots = list((registry.get("workspace_snapshots") or {}).values())
        decisions = list(registry.get("recent_decisions") or [])
        obsolete = list(registry.get("recent_obsolete") or [])
        recent_proposals = self._recent_registry_items(proposals, limit=limit)
        recent_runs = self._recent_registry_items(runs, limit=limit)
        recent_snapshots = self._recent_registry_items(snapshots, limit=limit)
        recent_decisions = self._recent_registry_items(decisions, limit=limit)
        recent_obsolete = self._recent_registry_items(obsolete, limit=limit)
        payload = {
            "view": "experiment_history_summary",
            "algorithm_id": algo_id,
            "registry_path": str(self._registry_index_path(algo_id)),
            "active_proposal_id": registry.get("active_proposal_id"),
            "active_workspace_snapshot_id": registry.get("active_workspace_snapshot_id"),
            "active_baseline_run_id": registry.get("active_baseline_run_id"),
            "latest_accepted_run_id": registry.get("latest_accepted_run_id"),
            "algorithm_lifecycle_status": registry.get("algorithm_lifecycle_status") or "developing",
            "algorithm_lifecycle_status_reason": registry.get("algorithm_lifecycle_status_reason") or "",
            "completed_campaign_id": registry.get("completed_campaign_id") or "",
            "latest_review_verdict": registry.get("latest_review_verdict"),
            "counts": {
                "proposals": len(proposals),
                "workspace_snapshots": len(snapshots),
                "runs": len(runs),
                "recent_decisions": len(decisions),
                "recent_obsolete": len(obsolete),
            },
            "limit": limit,
            "max_field_chars": max_field_chars,
            "truncated": any(
                len(collection) > limit
                for collection in (proposals, snapshots, runs, decisions, obsolete)
            ),
            "omitted": {
                "proposals": max(0, len(proposals) - len(recent_proposals)),
                "workspace_snapshots": max(0, len(snapshots) - len(recent_snapshots)),
                "runs": max(0, len(runs) - len(recent_runs)),
                "recent_decisions": max(0, len(decisions) - len(recent_decisions)),
                "recent_obsolete": max(0, len(obsolete) - len(recent_obsolete)),
            },
            "recent_proposals": [
                self._summarize_history_record(
                    item,
                    keys=[
                        "proposal_id",
                        "algorithm_id",
                        "status",
                        "review_status",
                        "decision",
                        "created_at",
                        "updated_at",
                        "reviewed_at",
                        "mass_modeling_scope",
                        "workspace_path",
                        "registry_path",
                        "obsolete",
                        "obsolete_reason",
                        "summary",
                        "abstract",
                    ],
                    max_chars=max_field_chars,
                )
                for item in recent_proposals
            ],
            "recent_workspace_snapshots": [
                self._summarize_history_record(
                    item,
                    keys=[
                        "snapshot_id",
                        "algorithm_id",
                        "proposal_id",
                        "created_at",
                        "reason",
                        "path",
                        "dirty_paths",
                        "file_count",
                    ],
                    max_chars=max_field_chars,
                )
                for item in recent_snapshots
            ],
            "recent_runs": [
                self._summarize_history_record(
                    item,
                    keys=[
                        "run_id",
                        "algorithm_id",
                        "proposal_id",
                        "snapshot_id",
                        "stage",
                        "status",
                        "decision",
                        "experiment_decision",
                        "campaign_trial_decision",
                        "created_at",
                        "started_at",
                        "completed_at",
                        "metrics_summary",
                        "metrics",
                        "output_dir",
                        "config_path",
                        "log_path",
                        "error",
                    ],
                    max_chars=max_field_chars,
                )
                for item in recent_runs
            ],
            "recent_decisions": [
                self._summarize_history_record(
                    item,
                    keys=[
                        "decision_id",
                        "phase",
                        "decision",
                        "status",
                        "created_at",
                        "timestamp",
                        "rationale",
                        "evidence",
                        "related_artifacts",
                    ],
                    max_chars=max_field_chars,
                )
                for item in recent_decisions
            ],
            "recent_obsolete": [
                self._summarize_history_record(
                    item,
                    keys=[
                        "artifact_type",
                        "artifact_id",
                        "reason",
                        "replaced_by",
                        "created_at",
                        "timestamp",
                    ],
                    max_chars=max_field_chars,
                )
                for item in recent_obsolete
            ],
            "full_history_access": {
                "read_file": str(self._registry_index_path(algo_id)),
                "note": "This tool intentionally returns a concise summary to avoid putting full registry payloads into model context.",
            },
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def rollback_algorithm_workspace(
        self,
        algorithm_id: str,
        *,
        target_snapshot_id: str = "",
        target_proposal_id: str = "",
    ) -> str:
        algo_id = str(algorithm_id or "").strip().lower()
        registry = self._bootstrap_algorithm_registry(algo_id)
        snapshot_id = str(target_snapshot_id or "").strip()
        proposal_id = str(target_proposal_id or "").strip()
        snapshots = registry.get("workspace_snapshots") or {}
        if proposal_id and not snapshot_id:
            candidates = [
                snap_id
                for snap_id, meta in snapshots.items()
                if str((meta or {}).get("proposal_id") or "") == proposal_id
            ]
            if not candidates:
                raise ValueError(f"No workspace snapshot found for proposal '{proposal_id}'.")
            snapshot_id = sorted(candidates)[-1]
        if not snapshot_id:
            snapshot_id = str(registry.get("active_workspace_snapshot_id") or "").strip()
        meta = dict(snapshots.get(snapshot_id) or {})
        if not meta:
            raise ValueError(f"Snapshot '{snapshot_id}' not found for algorithm '{algo_id}'.")
        snapshot_dir = Path(str(meta.get("path") or "")).expanduser().resolve()
        if not snapshot_dir.exists():
            raise FileNotFoundError(f"Snapshot directory is missing: {snapshot_dir}")
        algo_dir = self._proposal_dir(algo_id)
        self._ensure_algorithm_path_active_for_write(algo_dir / "algorithm.py", action="rollback_algorithm_workspace")
        restored_paths: List[Path] = []
        for name in REGISTRY_TRACKED_WORKSPACE_FILES:
            src = snapshot_dir / name
            dst = algo_dir / name
            if src.exists():
                shutil.copy2(src, dst)
                restored_paths.append(dst)
            elif dst.exists():
                dst.unlink()
                restored_paths.append(dst)
        registry["active_workspace_snapshot_id"] = snapshot_id
        if proposal_id:
            registry["active_proposal_id"] = proposal_id
        elif meta.get("proposal_id"):
            registry["active_proposal_id"] = str(meta.get("proposal_id"))
        registry["workspace_dirty"] = self._default_workspace_dirty()
        self._save_algorithm_registry(algo_id, registry, sync_active_context=self._is_current_active_algorithm(algo_id))
        self.state["planner_algorithm_workspace"] = str(algo_dir)
        self.state["latest_training_algorithm_id"] = algo_id
        self.state["latest_algorithm_proposal_id"] = algo_id if str(registry.get("active_proposal_id") or "").strip() else ""
        self.record_decision(
            algo_id,
            phase="authoring",
            decision="rollback_workspace_snapshot",
            alternatives=[],
            evidence=[],
            rationale=f"Restored workspace snapshot {snapshot_id}.",
            status="final",
            related_artifacts=[
                {"artifact_type": "workspace_snapshot", "artifact_id": snapshot_id},
                {"artifact_type": "proposal", "artifact_id": str(registry.get('active_proposal_id') or '')},
            ],
        )
        self._record_patch_summary(
            "rollback_algorithm_workspace",
            restored_paths,
            reason=f"restore snapshot {snapshot_id}",
            mark_dirty=False,
        )
        return (
            f"✅ Restored algorithm workspace for `{algo_id}` from snapshot `{snapshot_id}`.\n"
            f"Active proposal: {registry.get('active_proposal_id') or '(unchanged)'}"
        )

    def compare_algorithm_runs(
        self,
        algorithm_id: str,
        *,
        run_ids: Optional[List[str]] = None,
        include_baseline: bool = True,
    ) -> str:
        algo_id = str(algorithm_id or "").strip().lower()
        registry = self._bootstrap_algorithm_registry(algo_id)
        runs = registry.get("runs") or {}
        chosen_ids = [str(item).strip() for item in (run_ids or []) if str(item).strip()]
        baseline_id = str(registry.get("active_baseline_run_id") or "").strip()
        if include_baseline and baseline_id and baseline_id not in chosen_ids:
            chosen_ids.insert(0, baseline_id)
        if not chosen_ids:
            chosen_ids = [
                str(run_id)
                for run_id, summary in runs.items()
                if not bool((summary or {}).get("obsolete"))
            ]
        compared: List[Dict[str, Any]] = []
        baseline_record: Optional[Dict[str, Any]] = None
        for run_id in chosen_ids:
            summary = dict(runs.get(run_id) or {})
            if not summary or bool(summary.get("obsolete")):
                continue
            record = self._read_json(self._registry_runs_dir(algo_id) / f"{run_id}.json", summary)
            if not isinstance(record, dict):
                continue
            if run_id == baseline_id:
                baseline_record = record
            compared.append(record)
        payload = {
            "algorithm_id": algo_id,
            "baseline_run_id": baseline_id or None,
            "runs": [],
        }
        baseline_w1 = baseline_record.get("w1_mean") if isinstance(baseline_record, dict) else None
        baseline_tmv = baseline_record.get("tmv_mean") if isinstance(baseline_record, dict) else None
        for record in compared:
            item = {
                "run_id": record.get("run_id"),
                "decision": record.get("decision"),
                "status": record.get("status"),
                "stage": record.get("stage"),
                "w1_mean": record.get("w1_mean"),
                "tmv_mean": record.get("tmv_mean"),
                "run_verdict": record.get("run_verdict"),
            }
            if baseline_record and record.get("run_id") != baseline_record.get("run_id"):
                try:
                    item["delta_vs_baseline_w1"] = (
                        float(record.get("w1_mean")) - float(baseline_w1)
                        if record.get("w1_mean") is not None and baseline_w1 is not None
                        else None
                    )
                except Exception:
                    item["delta_vs_baseline_w1"] = None
                try:
                    item["delta_vs_baseline_tmv"] = (
                        float(record.get("tmv_mean")) - float(baseline_tmv)
                        if record.get("tmv_mean") is not None and baseline_tmv is not None
                        else None
                    )
                except Exception:
                    item["delta_vs_baseline_tmv"] = None
            payload["runs"].append(item)
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def _maybe_emit_report_generated(self, path: Path) -> None:
        policy = self._policy()
        if not policy._is_within(path, policy.output_root):
            return
        if path.parent != policy.output_root:
            return
        if path.suffix.lower() not in {".html", ".md"}:
            return
        if not any(path.name.startswith(prefix) for prefix in ("report",)):
            return
        self.state["report_path"] = str(path)
        self.state["final_summary"] = f"Report generated at {path}"
        self._emit(
            "report_generated",
            {
                "path": str(path),
                "filename": path.name,
                "url": f"/output/{path.name}",
            },
        )

    @staticmethod
    def _read_text(path: Path) -> str:
        return path.read_text(encoding="utf-8") if path.exists() else ""

    @staticmethod
    def _write_text(path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    @staticmethod
    def _render_diff(path: Path, before: str, after: str, context_lines: int = 3) -> str:
        diff = difflib.unified_diff(
            before.splitlines(),
            after.splitlines(),
            fromfile=str(path),
            tofile=str(path),
            lineterm="",
            n=context_lines,
        )
        rendered = "\n".join(diff)
        return rendered or f"No changes for {path}"

    @staticmethod
    def _config_value_type(value: Any) -> str:
        if isinstance(value, bool):
            return "bool"
        if isinstance(value, int) and not isinstance(value, bool):
            return "int"
        if isinstance(value, float):
            return "float"
        if isinstance(value, str):
            return "str"
        if isinstance(value, list):
            return "list"
        if isinstance(value, dict):
            return "dict"
        if value is None:
            return "null"
        return type(value).__name__

    @staticmethod
    def _config_values_type_compatible(existing: Any, value: Any) -> bool:
        if existing is None:
            return True
        if isinstance(existing, bool):
            return isinstance(value, bool)
        if isinstance(existing, (int, float)) and not isinstance(existing, bool):
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        if isinstance(existing, str):
            return isinstance(value, str)
        if isinstance(existing, list):
            return isinstance(value, list)
        if isinstance(existing, dict):
            return isinstance(value, dict)
        return isinstance(value, type(existing))

    def _validate_config_patch_value(
        self,
        *,
        path: str,
        exists: bool,
        existing: Any,
        value: Any,
        allow_list_replace: bool,
        strict_types: bool,
    ) -> None:
        if exists and isinstance(existing, (list, dict)) and not allow_list_replace:
            raise ValueError(
                f"{path}: refusing to replace whole {self._config_value_type(existing)} value. "
                "Patch a leaf path such as training.plan[0].lr, or set allow_list_replace=true intentionally."
            )
        if isinstance(value, (list, dict)) and not allow_list_replace:
            raise ValueError(
                f"{path}: refusing to write a whole {self._config_value_type(value)} value by default. "
                "Use leaf paths for typed config patches, or set allow_list_replace=true intentionally."
            )
        if exists and strict_types and not self._config_values_type_compatible(existing, value):
            raise ValueError(
                f"{path}: type mismatch, existing {self._config_value_type(existing)} "
                f"but new value is {self._config_value_type(value)}."
            )

    def _set_config_path_value(
        self,
        config: Dict[str, Any],
        *,
        path: str,
        value: Any,
        allow_create: bool,
        allow_list_replace: bool,
        strict_types: bool,
    ) -> Dict[str, Any]:
        parts = _override_path_parts(path)
        if not parts:
            raise ValueError("empty config path")
        cursor: Any = config
        for depth, part in enumerate(parts[:-1]):
            next_part = parts[depth + 1]
            if isinstance(cursor, dict):
                if part not in cursor:
                    if not allow_create:
                        raise ValueError(f"{path}: missing key at {'.'.join(parts[: depth + 1])}")
                    cursor[part] = [] if str(next_part).isdigit() else {}
                cursor = cursor[part]
                continue
            if isinstance(cursor, list):
                if not str(part).isdigit():
                    raise ValueError(f"{path}: expected numeric list index at {'.'.join(parts[: depth + 1])}")
                index = int(part)
                if index < 0 or index >= len(cursor):
                    raise ValueError(
                        f"{path}: list index {index} out of range at {'.'.join(parts[: depth + 1])}"
                    )
                cursor = cursor[index]
                continue
            raise ValueError(
                f"{path}: cannot traverse through scalar {self._config_value_type(cursor)} "
                f"at {'.'.join(parts[: depth + 1])}"
            )

        final = parts[-1]
        if isinstance(cursor, dict):
            exists = final in cursor
            if not exists and not allow_create:
                raise ValueError(f"{path}: missing final key {final}")
            before = cursor.get(final)
            self._validate_config_patch_value(
                path=path,
                exists=exists,
                existing=before,
                value=value,
                allow_list_replace=allow_list_replace,
                strict_types=strict_types,
            )
            cursor[final] = value
            return {
                "path": path,
                "resolved_path": ".".join(parts),
                "before": before if exists else None,
                "after": value,
                "created": not exists,
            }
        if isinstance(cursor, list):
            if not str(final).isdigit():
                raise ValueError(f"{path}: expected numeric final list index")
            index = int(final)
            if index < 0 or index >= len(cursor):
                raise ValueError(f"{path}: list index {index} out of range at final segment")
            before = cursor[index]
            self._validate_config_patch_value(
                path=path,
                exists=True,
                existing=before,
                value=value,
                allow_list_replace=allow_list_replace,
                strict_types=strict_types,
            )
            cursor[index] = value
            return {
                "path": path,
                "resolved_path": ".".join(parts),
                "before": before,
                "after": value,
                "created": False,
            }
        raise ValueError(f"{path}: parent is scalar {self._config_value_type(cursor)}")

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
    ) -> Dict[str, Any]:
        """Safely patch a custom algorithm workspace config.yaml and return the resolved diff."""
        active_context = self.get_active_algorithm_context()
        algo_id = str(algorithm_id or active_context.get("algorithm_id") or "").strip().lower()
        if not algo_id:
            return {
                "ok": False,
                "status": "error",
                "error": "algorithm_id is required when no active algorithm context is set.",
                "next_action": "Call activate_algorithm_workspace(...) or pass algorithm_id explicitly.",
            }
        normalized_updates = dict(updates or {})
        if not normalized_updates:
            return {
                "ok": False,
                "status": "error",
                "algorithm_id": algo_id,
                "error": "updates must contain at least one path -> value entry.",
                "example": {"training.plan[0].lr": 0.001},
            }

        config_path = self._proposal_dir(algo_id) / "config.yaml"
        policy = self._policy()
        try:
            policy.validate_write_path(str(config_path))
            self._ensure_algorithm_path_active_for_write(config_path, action="patch_algorithm_config")
            self._ensure_algorithm_path_proposal_approved(config_path, action="patch_algorithm_config")
        except Exception as exc:
            return {
                "ok": False,
                "status": "blocked",
                "algorithm_id": algo_id,
                "config_path": str(config_path),
                "error": str(exc),
            }
        if not config_path.exists():
            return {
                "ok": False,
                "status": "error",
                "algorithm_id": algo_id,
                "config_path": str(config_path),
                "error": "config.yaml does not exist. Initialize the training algorithm workspace first.",
            }

        before_text = self._read_text(config_path)
        try:
            config = yaml.safe_load(before_text) or {}
        except Exception as exc:
            return {
                "ok": False,
                "status": "error",
                "algorithm_id": algo_id,
                "config_path": str(config_path),
                "error": f"config.yaml is not valid YAML: {exc}",
            }
        if not isinstance(config, dict):
            return {
                "ok": False,
                "status": "error",
                "algorithm_id": algo_id,
                "config_path": str(config_path),
                "error": "config.yaml root must be a mapping.",
            }

        epoch_blocks = validate_epoch_override_policy(
            base_config=config,
            overrides=normalized_updates,
            allow_epoch_override=bool(allow_epoch_override),
            epoch_override_reason=str(epoch_override_reason or ""),
        )
        if epoch_blocks:
            return {
                "ok": False,
                "status": "blocked",
                "algorithm_id": algo_id,
                "config_path": str(config_path),
                "dry_run": bool(dry_run),
                "applied_updates": [],
                "blocked_updates": epoch_blocks,
                "message": (
                    "No config changes were written because at least one effective epoch change was blocked. "
                    "Same-value/default epoch declarations are allowed; changing epochs requires "
                    "allow_epoch_override=true and epoch_override_reason."
                ),
                "safe_path_format": "Use typed leaf paths such as training.plan[0].lr or training.plan.0.lr.",
            }

        candidate = deepcopy(config)
        applied: List[Dict[str, Any]] = []
        blocked: List[Dict[str, Any]] = []
        for raw_path, value in normalized_updates.items():
            path = str(raw_path or "").strip()
            if not path:
                blocked.append({"path": str(raw_path), "reason": "empty path"})
                continue
            try:
                applied.append(
                    self._set_config_path_value(
                        candidate,
                        path=path,
                        value=value,
                        allow_create=bool(allow_create),
                        allow_list_replace=bool(allow_list_replace),
                        strict_types=bool(strict_types),
                    )
                )
            except Exception as exc:
                blocked.append({"path": path, "reason": str(exc)})

        if blocked:
            return {
                "ok": False,
                "status": "blocked",
                "algorithm_id": algo_id,
                "config_path": str(config_path),
                "dry_run": bool(dry_run),
                "applied_updates": [],
                "blocked_updates": blocked,
                "message": "No config changes were written because at least one update was blocked.",
                "safe_path_format": "Use typed leaf paths such as training.plan[0].lr or training.plan.0.lr.",
            }

        after_text = yaml.safe_dump(candidate, sort_keys=False, allow_unicode=True)
        diff = self._render_diff(config_path, before_text, after_text)
        context_payload = self._algorithm_context_payload_for_paths([config_path])
        payload = {
            "ok": True,
            "status": "dry_run" if dry_run else "applied",
            "algorithm_id": algo_id,
            "config_path": str(config_path),
            "dry_run": bool(dry_run),
            "applied_updates": applied,
            "blocked_updates": [],
            "diff": diff,
            "changed": before_text != after_text,
            **context_payload,
        }
        if dry_run:
            return payload

        if before_text != after_text:
            self._write_text(config_path, after_text)
            self._record_patch_summary("patch_algorithm_config", [config_path], reason=reason)
            context_payload = self._algorithm_context_payload_for_paths([config_path])
        payload.update(context_payload)
        self._emit(
            "workspace_file_written",
            {
                "scope": "planner",
                "mode": "config_patch",
                "path": str(config_path),
                "reason": reason,
                "diff": diff,
                "applied_updates": applied,
                "blocked_updates": [],
                **context_payload,
            },
        )
        return payload

    @staticmethod
    def _find_sequence(lines: List[str], needle: List[str], start: int = 0) -> int:
        if not needle:
            return start
        stop = len(lines) - len(needle) + 1
        for idx in range(max(start, 0), max(stop, 0)):
            if lines[idx : idx + len(needle)] == needle:
                return idx
        return -1

    @staticmethod
    def _find_unique_line_fragment(lines: List[str], fragment: str, start: int = 0) -> int:
        if not fragment:
            return -1
        matches = [idx for idx in range(max(start, 0), len(lines)) if fragment in lines[idx]]
        if len(matches) == 1:
            return matches[0]
        if not matches and start > 0:
            matches = [idx for idx, line in enumerate(lines) if fragment in line]
            if len(matches) == 1:
                return matches[0]
        return -1

    @classmethod
    def _apply_update_lines(cls, original: str, patch_lines: List[str]) -> str:
        source = original.splitlines()
        cursor = 0
        i = 0
        while i < len(patch_lines):
            if patch_lines[i].startswith("@@"):
                i += 1
                continue
            hunk: List[str] = []
            while i < len(patch_lines) and not patch_lines[i].startswith("@@"):
                hunk.append(patch_lines[i])
                i += 1
            before = [line[1:] for line in hunk if line[:1] in {" ", "-"}]
            after = [line[1:] for line in hunk if line[:1] in {" ", "+"}]
            pos = cls._find_sequence(source, before, start=cursor)
            if pos < 0:
                pos = cls._find_sequence(source, before, start=0)
            if pos < 0:
                removed = [line[1:] for line in hunk if line.startswith("-")]
                added = [line[1:] for line in hunk if line.startswith("+")]
                context = [line[1:] for line in hunk if line.startswith(" ")]
                if len(removed) == 1 and len(added) == 1 and not context:
                    fragment_pos = cls._find_unique_line_fragment(source, removed[0], start=cursor)
                    if fragment_pos >= 0:
                        source[fragment_pos] = source[fragment_pos].replace(removed[0], added[0], 1)
                        cursor = fragment_pos + 1
                        continue
                raise ValueError("Failed to apply patch hunk: context not found")
            source = source[:pos] + after + source[pos + len(before) :]
            cursor = pos + len(after)
        text = "\n".join(source)
        if original.endswith("\n"):
            return text + "\n"
        return text

    @classmethod
    def _parse_patch(cls, patch: str) -> List[PatchOperation]:
        lines = patch.splitlines()
        if not lines:
            raise ValueError("Patch cannot be empty")
        # Accept both Codex-style patches and unified diff patches.
        first = lines[0].strip()
        if first == "*** Begin Patch":
            return cls._parse_codex_patch(lines)
        if first.startswith("--- ") or first.startswith("diff --git "):
            return cls._parse_unified_diff(lines)
        raise ValueError(
            "Unsupported patch format. Use either Codex patch "
            "(`*** Begin Patch ... *** End Patch`) or unified diff "
            "(`---` / `+++` with `@@` hunks)."
        )

    @classmethod
    def _parse_codex_patch(cls, lines: List[str]) -> List[PatchOperation]:
        ops: List[PatchOperation] = []
        i = 1
        while i < len(lines):
            line = lines[i]
            if line.strip() == "*** End Patch":
                return ops
            if line.startswith("*** Add File: "):
                path = line.split(": ", 1)[1].strip()
                i += 1
                body: List[str] = []
                while i < len(lines) and not lines[i].startswith("*** "):
                    if not lines[i].startswith("+"):
                        raise ValueError("Add File entries require '+' lines only")
                    body.append(lines[i])
                    i += 1
                ops.append(PatchOperation("add", path, body))
                continue
            if line.startswith("*** Update File: "):
                path = line.split(": ", 1)[1].strip()
                i += 1
                body = []
                while i < len(lines) and not lines[i].startswith("*** "):
                    body.append(lines[i])
                    i += 1
                ops.append(PatchOperation("update", path, body))
                continue
            if line.startswith("*** Delete File: "):
                path = line.split(": ", 1)[1].strip()
                ops.append(PatchOperation("delete", path, []))
                i += 1
                continue
            raise ValueError(f"Unsupported patch directive: {line}")
        raise ValueError("Patch missing '*** End Patch'")

    @staticmethod
    def _normalize_unified_path(raw: str) -> str:
        path = str(raw or "").strip()
        # Drop optional timestamp suffix from unified diffs.
        if "\t" in path:
            path = path.split("\t", 1)[0].strip()
        # Git unified diff prefixes.
        if path.startswith("a/") or path.startswith("b/"):
            path = path[2:]
        return path

    @classmethod
    def _parse_unified_diff(cls, lines: List[str]) -> List[PatchOperation]:
        ops: List[PatchOperation] = []
        i = 0
        n = len(lines)
        while i < n:
            # Optional header line from git-style patches.
            if lines[i].startswith("diff --git "):
                i += 1
                continue
            if not lines[i].startswith("--- "):
                i += 1
                continue
            old_path = cls._normalize_unified_path(lines[i][4:])
            i += 1
            if i >= n or not lines[i].startswith("+++ "):
                raise ValueError("Malformed unified diff: missing '+++' after '---'")
            new_path = cls._normalize_unified_path(lines[i][4:])
            i += 1

            body: List[str] = []
            while i < n and not lines[i].startswith("--- ") and not lines[i].startswith("diff --git "):
                line = lines[i]
                # Ignore metadata marker in unified patches.
                if line.startswith("\\ No newline at end of file"):
                    i += 1
                    continue
                # Keep only hunk lines + headers that updater understands.
                if line.startswith("@@") or line[:1] in {" ", "+", "-"}:
                    body.append(line)
                i += 1

            if old_path == "/dev/null":
                path = new_path
                kind = "add"
            elif new_path == "/dev/null":
                path = old_path
                kind = "delete"
            else:
                path = new_path
                kind = "update"
            if not path:
                raise ValueError("Malformed unified diff: empty target path")
            ops.append(PatchOperation(kind=kind, path=path, lines=body))

        if not ops:
            raise ValueError("No file operations found in unified diff patch")
        return ops

    def init_training_algorithm_workspace(
        self,
        algorithm_id: str,
        description: str,
        requirements: str = "",
        author: str = "agent",
        overwrite: bool = False,
    ) -> str:
        algo_id = str(algorithm_id or "").strip().lower()
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{1,63}", algo_id):
            return (
                "Error: algorithm_id must match [a-z0-9][a-z0-9_-]{1,63}. "
                f"Got: {algorithm_id!r}"
            )
        cellcompass_root = get_cellcompass_root()
        algo_dir = cellcompass_root / "training_algorithms" / algo_id
        policy = self._policy()
        policy.validate_write_path(str(algo_dir / "manifest.yaml"))
        self._ensure_algorithm_path_active_for_write(algo_dir / "manifest.yaml", action="init_training_algorithm_workspace")
        self._ensure_algorithm_id_proposal_approved(
            algo_id,
            action="init_training_algorithm_workspace",
            path=algo_dir / "manifest.yaml",
        )
        if algo_dir.exists() and not overwrite:
            existing_entries = [p.name for p in algo_dir.iterdir()]
            allowed_preinit = {"PROPOSAL.md", "PROPOSAL.json", "risk.md", "registry"}
            blocking_entries = [name for name in existing_entries if name not in allowed_preinit]
            if blocking_entries:
                return f"Error: algorithm workspace already exists: {algo_dir}"
        algo_dir.mkdir(parents=True, exist_ok=True)
        base_config_seed = "vgfm"
        try:
            seed_config = load_training_config(base_config_seed)
        except Exception as e:
            return f"Error: invalid base_config '{base_config_seed}': {e}"
        # Template guard: ensure training.plan entries always carry `mode`.
        # For custom algorithm workspaces we default missing stage mode to `flow_matching`.
        training_cfg = seed_config.get("training") if isinstance(seed_config, dict) else None
        plan_cfg = training_cfg.get("plan") if isinstance(training_cfg, dict) else None
        if isinstance(plan_cfg, list):
            for stage_cfg in plan_cfg:
                if isinstance(stage_cfg, dict):
                    stage_cfg.setdefault("mode", "flow_matching")

        requirements_text = str(requirements or "").strip() or (
            "Specify data prerequisites and preprocessing expectations for this algorithm. "
            "Example: requires biologically ordered time_point_processed and valid X_latent. "
            "Custom algorithms must document how they scale beyond toy data."
        )
        local_config_ref = "./config.yaml"
        manifest = {
            "algorithm_id": algo_id,
            "api_version": 1,
            "entrypoint": "algorithm.py",
            "entry_function": "build_training_algorithm",
            "description": str(description or "").strip(),
            "requirements": requirements_text,
            "base_config": local_config_ref,
            "base_config_seed": base_config_seed,
            "minibatch_required": False,
            "minibatch_policy": "scale_dependent",
            "scalability_required": True,
            "default_chunk_size": 1000,
            "tags": ["flow-matching", "custom"],
            "author": str(author or "agent").strip() or "agent",
        }
        algorithm_py = f'''from __future__ import annotations

import numpy as np
import torch

from CytoBridge.tl.flow_matching_backends import (
    ChunkedTransportCouplingStrategy,
    CostBasedPairwiseOTCouplingStrategy,
    FlowMatchingBackend,
    LinearDeterministicConditionalPath,
    PairwiseCost,
    RegularizedUnbalancedConditionalPath,
    UOTMassStrategy,
)
from CytoBridge.tl.models import DynamicalModel
from CytoBridge.tl.training_algorithm import (
    EvaluationMetricsContext,
    FlowMatchingBuildContext,
    FlowMatchingLossContext,
    FlowMatchingLossResult,
    InferenceContext,
    InferenceContextBuilderContext,
    ModelBuildContext,
    ModelLoadContext,
    ModelSerializeContext,
    SimulationContext,
    SimulationResult,
    StageRunnerContext,
    StageRunnerResult,
    TrainingAlgorithmSpec,
)


class CustomCoupling(ChunkedTransportCouplingStrategy):
    """
    Scale-aware custom entry point: override `build_pairwise_cost_block(...)`.

    `chunk_size=1000` means an adjacent gap with about 1000 source cells and
    1000 target cells yields one `1000 x 1000` cost/coupling matrix and is
    solved as one full block. That is fine for tiny/small benchmarks such as the
    2D simulation. Larger real-data gaps, including Weinreb-scale gaps, are
    split into bounded blocks; the package handles per-block OT/UOT/WFR solving,
    cached subplans, and pair sampling. Keep this lifecycle unless the approved
    proposal needs global cross-time coordination or a different solver
    convention.
    """

    def __init__(
        self,
        *,
        chunk_size: int = 1000,
        alpha_regm: float = 1.0,
        reg_strategy: str = "per_time",
        reg: float | None = None,
        reg_m: float | None = None,
        auto_reg_device: str = "cpu",
    ):
        super().__init__(
            solver_mode="uot",
            chunk_size=int(chunk_size),
            alpha_regm=float(alpha_regm),
            reg_strategy=str(reg_strategy),
            reg=reg,
            reg_m=reg_m,
            auto_reg_device=auto_reg_device,
        )

    def describe_metadata(self) -> dict:
        return {{
            "kind": "custom_pairwise_ot",
            "chunk_size": self.chunk_size,
            "alpha_regm": self.alpha_regm,
            "reg_strategy": self.reg_strategy,
            "fixed_reg": self.fixed_reg,
            "fixed_reg_m": self.fixed_reg_m,
            "auto_reg_device": self.auto_reg_device,
        }}

    def build_pairwise_cost_block(
        self,
        x0_block: np.ndarray,
        x1_block: np.ndarray,
        *,
        source_indices: np.ndarray,
        target_indices: np.ndarray,
        time_idx: int,
        t0: float,
        t1: float,
        device: torch.device,
    ) -> PairwiseCost:
        del t0, t1, source_indices, target_indices
        # Replace this body with the approved pairwise geometry for one bounded
        # source/target block. Keep tensor work on `device` when possible; the
        # package will copy the bounded result back for cached sampling.
        x0_t = torch.as_tensor(x0_block, dtype=torch.float32, device=device)
        x1_t = torch.as_tensor(x1_block, dtype=torch.float32, device=device)
        cost_matrix = torch.cdist(x0_t, x1_t).pow(2)
        return PairwiseCost(
            cost_matrix=cost_matrix,
            metadata={{"time_idx": time_idx}},
        )


def build_training_algorithm(context):
    """Return a TrainingAlgorithmSpec for {{context.algorithm_id}}."""

    def build_flow_matching_backend(build_context: FlowMatchingBuildContext) -> FlowMatchingBackend:
        stage_params = build_context.stage_params
        device = build_context.device
        training_data = build_context.training_data
        # Use training_data.extra_modalities_by_time when the algorithm needs
        # additional aligned modalities, but keep X_latent as the transcriptomic
        # backbone and time_point_processed as the canonical time field.
        _extra_modalities = training_data.extra_modalities_by_time
        chunk_size = int(stage_params.get("chunk_size", 1000))
        coupling_config = (stage_params.get("flow_matching") or {{}}).get("coupling") or {{}}
        coupling = CustomCoupling(
            chunk_size=int(coupling_config.get("chunk_size", chunk_size)),
            alpha_regm=float(coupling_config.get("alpha_regm", stage_params.get("alpha_regm", 1.0))),
            reg_strategy=str(coupling_config.get("reg_strategy", stage_params.get("reg_strategy", "per_time"))),
            reg=coupling_config.get("reg", stage_params.get("reg")),
            reg_m=coupling_config.get("reg_m", stage_params.get("reg_m")),
            auto_reg_device=str(coupling_config.get("auto_reg_device", stage_params.get("auto_reg_device", "cpu"))),
        )
        del _extra_modalities
        path_config = (stage_params.get("flow_matching") or {{}}).get("path") or {{}}
        sigma = float(path_config.get("sigma", stage_params.get("sigma", 0.0)))
        path = (
            LinearDeterministicConditionalPath(sigma=0.0)
            if sigma <= 0.0
            else RegularizedUnbalancedConditionalPath(sigma=sigma)
        )
        return FlowMatchingBackend(
            path=path,
            coupling=coupling,
            mass=UOTMassStrategy(),
        )

    # Optional advanced hook:
    # def training_data_builder(build_context):
    #     ... return TrainingDataBundle(...)

    # Optional additive hook:
    # def flow_matching_loss_hook(loss_context: FlowMatchingLossContext):
    #     # Builtin model heads take one concatenated tensor in [x, t] order.
    #     # Prefer loss_context.x_input / loss_context.t_input when splitting is needed.
    #     extra = 0.01 * loss_context.x_input.pow(2).mean()
    #     return FlowMatchingLossResult(extra_loss=extra, logs={{"aux_loss": extra.detach()}})
    #
    # Optional component replacement hook:
    # def flow_matching_loss_hook(loss_context: FlowMatchingLossContext):
    #     # Replace only the builtin velocity component; growth/score stay builtin.
    #     v_pred = loss_context.model.velocity_net(loss_context.net_input)
    #     custom_v_loss = torch.mean(loss_context.batch.loss_weights * (v_pred - loss_context.batch.ut) ** 2)
    #     return FlowMatchingLossResult(replace_velocity_loss=custom_v_loss)

    # Optional additive evaluation hook:
    # def evaluation_metrics_hook(eval_context: EvaluationMetricsContext):
    #     # Builtin W1/TMV are already computed by the package.
    #     # Reuse eval_context.simulated_points_by_time /
    #     # eval_context.simulated_weights_by_time / eval_context.timepoint_results
    #     # instead of re-running trajectory simulation.
    #     # Additional user-defined parameters are available in
    #     # eval_context.metric_params.
    #     # Return only additive custom metrics here.
    #     return {{"my_custom_metric": 0.0}}

    # Optional generic model hook:
    # def model_builder(model_context: ModelBuildContext) -> torch.nn.Module:
    #     model = DynamicalModel(
    #         model_context.latent_dim,
    #         model_context.resolved_config["model"],
    #     )
    #     # If the approved algorithm has a trainable extra module, attach it to
    #     # the returned model and make optimizer ownership explicit. The backend
    #     # can then retrieve this exact module through build_context.model.
    #     #
    #     # model.add_module("custom_path_net", torch.nn.Sequential(
    #     #     torch.nn.Linear(model_context.latent_dim + 1, 32),
    #     #     torch.nn.Tanh(),
    #     #     torch.nn.Linear(32, model_context.latent_dim),
    #     # ))
    #     # model.cytobridge_component_modules = {{"velocity": ["custom_path_net"]}}
    #     return model

    # Optional model serialization hooks when state_dict alone is not enough:
    # def serialize_model(serialize_context: ModelSerializeContext) -> dict[str, object]:
    #     return {{"notes": "extra payload for reload"}}
    #
    # def deserialize_model(load_context: ModelLoadContext) -> torch.nn.Module:
    #     model = DynamicalModel(load_context.latent_dim, load_context.resolved_config["model"])
    #     model.load_state_dict(load_context.model_state_dict)
    #     return model

    # Optional full stage override for stages you own. Return None for ordinary
    # flow_matching/neural_ode stages that should use the package default runner.
    # def stage_runner(stage_context: StageRunnerContext) -> StageRunnerResult:
    #     if stage_context.stage_params.get("mode") in {{"flow_matching", "neural_ode"}}:
    #         return None
    #     if stage_context.preview_only:
    #         return StageRunnerResult(stage_summary={{"preview": True}})
    #     # Own forward/loss/backward logic here.
    #     return StageRunnerResult(stage_summary={{"status": "completed"}})

    # Optional inference-context builder for custom simulation_hook.
    # Keep this flexible: payload can be algorithm-specific, but provenance
    # should show that inference inputs come from t=0 cells, known exogenous
    # conditions, constants/priors, or model state, not future observed truth.
    # def inference_context_builder(infer_context: InferenceContextBuilderContext) -> InferenceContext:
    #     return InferenceContext(
    #         payload={{"x0": infer_context.initial_data, "time_points": infer_context.time_points}},
    #         visibility={{"x0": "t0_inference", "time_points": "exogenous_inference"}},
    #         provenance={{"time_filter": "t0 only", "future_rows_used": False}},
    #     )

    # Optional evaluation-time predictor/simulator:
    # def simulation_hook(sim_context: SimulationContext) -> SimulationResult:
    #     # If inference_context_builder is provided, use
    #     # sim_context.inference_context.payload for algorithm-specific inputs.
    #     # Return one full t0-to-final trajectory matching
    #     # sim_context.trajectory_time_points. Builtin W1/TMV and claim metrics
    #     # are computed from slices of this sealed trajectory artifact.
    #     return SimulationResult(
    #         predicted_points_by_time=[],
    #         predicted_weights_by_time=[],
    #         trajectory_time_points=list(sim_context.trajectory_time_points),
    #     )

    return TrainingAlgorithmSpec(
        algorithm_id=context.algorithm_id,
        base_config="{local_config_ref}",
        config_overrides={{}},
        flow_matching_backend_builder=build_flow_matching_backend,
        model_builder=None,
        serialize_model=None,
        deserialize_model=None,
        stage_runner=None,
        inference_context_builder=None,
        simulation_hook=None,
        evaluation_metrics_hook=None,
        evaluation_metrics_params={{}},
        notes="{str(description or '').strip()}",
    )
'''
        readme = (
            f"# {algo_id}\n\n"
            f"{str(description or '').strip()}\n\n"
            f"> Runtime catalog fields are read from `manifest.yaml: description` and `manifest.yaml: requirements`.\n\n"
            f"## Requirements\n"
            f"{requirements_text}\n\n"
            f"## Config Baseline\n"
            f"- Seed source: `{base_config_seed}` (fixed by workspace initializer)\n"
            f"- Workspace config: `config.yaml`\n"
            f"- Manifest base_config: `{local_config_ref}`\n\n"
            f"## Training Plan Mode\n"
            f"- Builtin template is seeded for `flow_matching`, but custom algorithms may also use `neural_ode` stages.\n"
            f"- If you introduce nonstandard per-stage optimization semantics, implement `stage_runner(...)` instead of forcing the logic into additive hooks.\n\n"
            f"## Generic Runtime Hooks\n"
            f"- `model_builder(...)` lets the algorithm replace the builtin `DynamicalModel`.\n"
            f"- Proposal-required trainable modules must be model-owned or explicitly selected into the optimizer; backend/path/coupling-local `nn.Module` objects are not trained by default.\n"
            f"- `serialize_model(...)` / `deserialize_model(...)` are only needed when reload requires extra payload beyond config + state_dict.\n"
            f"- `stage_runner(...)` is the full training-stage override for stages you own; return `None` for ordinary `flow_matching`/`neural_ode` stages that should continue through the package default runner.\n"
            f"- `simulation_hook(...)` is the evaluation-time prediction override; builtin `W1/TMV` definitions stay fixed.\n\n"
            f"## Default Coupling API\n"
            f"- The generated `CustomCoupling` already uses `ChunkedTransportCouplingStrategy`.\n"
            f"- Customize `CustomCoupling.build_pairwise_cost_block(...)` for one bounded source/target block when only pairwise geometry changes.\n"
            f"- With `chunk_size: 1000`, an adjacent gap with about 1000 source cells and 1000 target cells gives one `1000 x 1000` cost/coupling matrix; do not add unnecessary mini-batch complexity for tiny/small simulations.\n"
            f"- Use `CostBasedPairwiseOTCouplingStrategy.build_pairwise_cost(...)` only when a full adjacent-pair cost matrix is acceptable.\n"
            f"- Drop down to a custom `CouplingStrategy.build_state(...)` / `sample_pairs(...)` only when coupling needs global cross-time coordination or streaming/sparse state not covered by the chunked-cost API.\n"
            f"- `build_flow_matching_backend(build_context)` receives `build_context.training_data`.\n"
            f"- Add `training_data_builder` only when you need extra aligned modalities.\n"
            f"- Add `flow_matching_loss_hook` when you need additive differentiable loss or an explicit replacement for one builtin component loss (`replace_velocity_loss`, `replace_growth_loss`, `replace_score_loss`). Builtin model heads take `loss_context.net_input` in `[x, t]` order; prefer `loss_context.x_input` and `loss_context.t_input` instead of manually slicing when possible.\n\n"
            f"- Add `evaluation_metrics_hook` only when you need additive custom metrics.\n"
            f"- Put custom metric hyperparameters or external evaluation knobs in `evaluation_metrics_params`.\n"
            f"- Do not modify builtin `W1` or `TMV`; custom metrics must be appended under `custom_metrics`.\n\n"
            f"## Scalability Requirement\n"
            f"- Scalability is scale-dependent, not a checkbox. Full cost is acceptable for small adjacent gaps whose cost/coupling matrix is about `1000 x 1000`; large real-data gaps such as Weinreb need bounded-memory chunking/sparse/streaming behavior.\n"
            f"- `use_mini_batch=True` or a `chunk_size` value does not prove large-data scalability unless full cost/mask/kernel/plan state is avoided before chunking.\n"
            f"- If a real benchmark has large adjacent time pairs and the method only changes pairwise geometry, use `ChunkedTransportCouplingStrategy` before writing custom sampler code.\n"
            f"- If the chunked-cost API is insufficient, avoid full dense cost/mask/plan state by implementing a streaming, sparse, landmark, or coreset coupling path.\n"
            f"- Document the actual memory path in `IMPLEMENTATION_MAP.md`; do not rely on a mini-batch flag as the evidence.\n\n"
            f"## Files\n"
            f"- manifest.yaml\n"
            f"- algorithm.py\n"
            f"- config.yaml\n"
            f"- IMPLEMENTATION_MAP.md\n"
            f"\n## Entry point\n"
            f"- build_training_algorithm(context) -> TrainingAlgorithmSpec\n"
        )
        implementation_map = (
            f"# Implementation Map: {algo_id}\n\n"
            f"Fill this file before training.\n\n"
            f"Purpose:\n"
            f"- map each approved proposal pseudocode step to concrete code lines\n"
            f"- verify that the implementation preserves the intended proposal semantics\n"
            f"- prove that review covered the real implementation rather than a verbal summary\n\n"
            f"## Instructions\n"
            f"1. Copy the approved pseudocode steps from `PROPOSAL.md` exactly, preserving the step ids `P1`, `P2`, ...\n"
            f"2. For each step, record the implementation file, package default, or hook/config path. Line numbers are helpful for review but not required.\n"
            f"3. In `Semantic Check`, state which invariant from the proposal is being preserved.\n"
            f"4. In `Deviation Status`, classify the implementation as `exact`, `acceptable approximation`, or `semantic drift`.\n"
            f"5. If a step is delegated to package default behavior, say so explicitly in `Notes`.\n"
            f"6. Add a scalability row that states whether the hot path constructs full adjacent-pair dense cost/mask/plan arrays on realistic data.\n"
            f"7. If the proposal adds any trainable module beyond the standard heads, add a trainable-ownership row showing where the module is attached to the model and how the optimizer selects it.\n"
            f"8. Fill `Required anchor baseline` with the nearest builtin algorithm used as the mandatory engineering/baseline anchor. This builtin is always included when campaign baselines are refreshed; other baselines may still use the default set or explicit user choices.\n"
            f"9. If the proposal defines a custom claim metric, fill `Claim Metric Contract` with prediction/truth/grouping/timepoint/label sources, truth-semantics evidence, and code lines. Do not use campaign evidence until this is auditable. Column names or category counts alone do not prove terminal fate, descendant fate, lineage endpoint, growth, perturbation, or other biological truth semantics.\n"
            f"10. Review is not complete until all steps have concrete mappings, non-TODO status, scalability evidence, required anchor baseline, optimizer-ownership evidence for proposal-required trainable modules, and claim-metric contract evidence when applicable.\n\n"
            f"## Baseline Anchor\n\n"
            f"- Required anchor baseline: TODO\n"
            f"- Why this is the nearest builtin: TODO\n"
            f"- Components reused or intentionally changed: TODO\n\n"
            f"## Claim Metric Contract\n\n"
            f"| Metric | Proposal Claim Object | Prediction Source | Truth / Observed Source | Truth Semantics Evidence | Grouping Key | Timepoints | Label Source | Leakage Controls | Code Line(s) | Status |\n"
            f"| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
            f"| TODO or N/A | TODO | TODO | TODO | TODO: cite proposal/dataset contract; column names alone are insufficient | TODO | TODO | TODO | TODO | TODO | pending |\n\n"
            f"## Mapping Table\n\n"
            f"| Step ID | Pseudocode Step | Implementation File | Line Number(s) | Status | Semantic Check | Deviation Status | Notes |\n"
            f"| --- | --- | --- | --- | --- | --- | --- | --- |\n"
            f"| P1 | TODO | TODO | TODO | pending | TODO | TODO | TODO |\n"
            f"| P2 | TODO | TODO | TODO | pending | TODO | TODO | TODO |\n"
            f"| S1 | Scalability / memory path | TODO | TODO | pending | No full dense adjacent-pair state on target-scale data, or proposal explicitly limits scale | TODO | TODO |\n"
            f"| T1 | Trainable module ownership | TODO | TODO | pending | Proposal-required trainable modules are model-owned or otherwise optimizer-selected | TODO | Name module, model attachment path, trainer selection mechanism, and gradient/optimizer smoke evidence |\n"
        )
        policy.validate_write_path(str(algo_dir / "config.yaml"))
        (algo_dir / "manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True), encoding="utf-8")
        (algo_dir / "algorithm.py").write_text(algorithm_py, encoding="utf-8")
        (algo_dir / "config.yaml").write_text(yaml.safe_dump(seed_config, sort_keys=False, allow_unicode=True), encoding="utf-8")
        (algo_dir / "README.md").write_text(readme, encoding="utf-8")
        (algo_dir / "IMPLEMENTATION_MAP.md").write_text(implementation_map, encoding="utf-8")
        self.state["planner_algorithm_workspace"] = str(algo_dir)
        registry = self._bootstrap_algorithm_registry(algo_id)
        proposal_id = str(registry.get("active_proposal_id") or "")
        snapshot_id = self.create_workspace_snapshot(
            algo_id,
            reason="Initial workspace scaffold",
            source="workspace_init",
            proposal_id=proposal_id,
            set_active=True,
            allow_empty=False,
        )
        self.set_active_algorithm_context(algo_id)
        self.record_decision(
            algo_id,
            phase="authoring",
            decision="init_training_algorithm_workspace",
            alternatives=[],
            evidence=[],
            rationale="Initialized custom algorithm workspace from builtin vgfm seed config.",
            status="final",
            related_artifacts=[
                {"artifact_type": "proposal", "artifact_id": proposal_id},
                {"artifact_type": "workspace_snapshot", "artifact_id": snapshot_id},
            ],
        )
        self._record_patch_summary(
            "init_training_algorithm_workspace",
            [algo_dir / "manifest.yaml", algo_dir / "algorithm.py", algo_dir / "config.yaml", algo_dir / "README.md", algo_dir / "IMPLEMENTATION_MAP.md"],
            mark_dirty=False,
        )
        active_context = self.get_active_algorithm_context()
        self._emit(
            "planner_algorithm_workspace_initialized",
            {
                "algorithm_id": algo_id,
                "active_algorithm_id": str(active_context.get("algorithm_id") or ""),
                "target_algorithm_id": algo_id,
                "path": str(algo_dir),
                "base_config_seed": base_config_seed,
                "initial_snapshot_id": snapshot_id,
                "proposal_id": str(active_context.get("proposal_id") or ""),
                "snapshot_id": str(active_context.get("active_snapshot_id") or snapshot_id),
                "dirty_since_snapshot": bool(active_context.get("dirty_since_snapshot", False)),
            },
        )
        return (
            f"✅ Initialized training algorithm workspace: {algo_dir}\n"
            f"- Seed config: {base_config_seed}\n"
            f"- Generated: manifest.yaml, algorithm.py, config.yaml, README.md, IMPLEMENTATION_MAP.md\n"
            f"- Initial snapshot: {snapshot_id}"
        )

    def list_workspace_tree(self, path: str = "", recursive: bool = True, max_entries: int = 200) -> str:
        policy = self._policy()
        if str(path or "").strip():
            target = policy.validate_read_path(path)
        else:
            base = self.state.get("planner_algorithm_workspace") or str(get_cellcompass_root() / "training_algorithms")
            target = policy.validate_read_path(str(base))
        result = list_path_impl(str(target), recursive=recursive, max_entries=max_entries, show_hidden=True)
        self._emit(
            "workspace_tree_listed",
            {
                "scope": "planner",
                "path": str(target),
                "recursive": bool(recursive),
                "max_entries": int(max_entries),
            },
        )
        return result

    def read_workspace_file(
        self,
        path: str,
        start_line: int = 1,
        end_line: Optional[int] = None,
        max_chars: int = 20000,
    ) -> str:
        policy = self._policy()
        target = policy.validate_read_path(path)
        limit = None
        if end_line is not None:
            limit = end_line - start_line + 1
        result = read_file_impl(
            file_path=str(target),
            offset=start_line,
            limit=limit,
            max_chars=max_chars,
            default_text_limit=None,
        )
        if isinstance(result, ReadResult):
            result_text = result.tool_text
        else:
            result_text = result
        self._emit(
            "workspace_file_read",
            {
                "scope": "planner",
                "path": str(target),
                "start_line": int(start_line),
                "end_line": int(end_line) if end_line is not None else None,
                "max_chars": int(max_chars),
                "preview": result_text[:4000],
            },
        )
        return result_text

    def create_workspace_file(self, path: str, content: str, overwrite: bool = False, reason: str = "") -> str:
        policy = self._policy()
        target = policy.validate_write_path(path)
        self._ensure_algorithm_path_active_for_write(target, action="create_workspace_file")
        self._ensure_algorithm_path_proposal_approved(target, action="create_workspace_file")
        if target.exists() and not overwrite:
            return f"Error: file already exists: {target}"
        before = self._read_text(target)
        self._write_text(target, content)
        self._record_patch_summary("create_workspace_file", [target], reason=reason)
        context_payload = self._algorithm_context_payload_for_paths([target])
        self._emit(
            "workspace_file_written",
            {
                "scope": "planner",
                "mode": "create",
                "path": str(target),
                "reason": reason,
                "diff": self._render_diff(target, before, content),
                **context_payload,
            },
        )
        self._maybe_emit_report_generated(target)
        return f"✅ Wrote file: {target}"

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
        requested_algo = str(algorithm_id or "").strip().lower()
        active_context = self.get_active_algorithm_context()
        active_algo = str(active_context.get("algorithm_id") or "").strip().lower()
        algo_id = requested_algo or active_algo
        if not algo_id:
            return "Error: no active algorithm workspace is set. Activate or initialize an algorithm workspace first."
        rel_text = str(relative_path or "").strip()
        if not rel_text:
            return "Error: relative_path is required."
        rel_path = Path(rel_text)
        if not rel_path.parts or rel_path == Path(".") or rel_path.is_absolute() or any(part in {"..", ""} for part in rel_path.parts):
            return "Error: relative_path must be a clean path relative to the algorithm workspace; absolute paths and '..' are not allowed."
        normalized_kind = str(kind or "file").strip().lower()
        if normalized_kind not in {"file", "directory"}:
            return "Error: kind must be 'file' or 'directory'."

        workspace_dir = self._proposal_dir(algo_id)
        target = self._policy().validate_write_path(str(workspace_dir / rel_path))
        self._ensure_algorithm_path_active_for_write(target, action="create_algorithm_workspace_artifact")
        self._ensure_algorithm_path_proposal_approved(target, action="create_algorithm_workspace_artifact")
        if not workspace_dir.exists():
            return f"Error: algorithm workspace does not exist: {workspace_dir}"

        if normalized_kind == "directory":
            if target.exists() and not target.is_dir():
                return f"Error: path exists and is not a directory: {target}"
            created = not target.exists()
            target.mkdir(parents=True, exist_ok=True)
            self._record_patch_summary("create_algorithm_workspace_directory", [target], reason=reason)
            context_payload = self._algorithm_context_payload_for_paths([target])
            self._emit(
                "workspace_directory_created",
                {
                    "scope": "planner",
                    "mode": "mkdir",
                    "path": str(target),
                    "relative_path": str(rel_path),
                    "reason": reason,
                    "created": created,
                    "note": "Empty directories are not preserved by git archives until they contain at least one file.",
                    **context_payload,
                },
            )
            return (
                f"✅ Created directory: {target}"
                if created
                else f"✅ Directory already exists: {target}"
            )

        if target.exists() and target.is_dir():
            return f"Error: path exists and is a directory: {target}"
        if target.exists() and not overwrite:
            return f"Error: file already exists: {target}"
        before = self._read_text(target)
        self._write_text(target, content)
        self._record_patch_summary("create_algorithm_workspace_file", [target], reason=reason)
        context_payload = self._algorithm_context_payload_for_paths([target])
        self._emit(
            "workspace_file_written",
            {
                "scope": "planner",
                "mode": "create_algorithm_artifact",
                "path": str(target),
                "relative_path": str(rel_path),
                "reason": reason,
                "diff": self._render_diff(target, before, content),
                **context_payload,
            },
        )
        self._maybe_emit_report_generated(target)
        return f"✅ Wrote file: {target}"

    def replace_workspace_file(self, path: str, content: str, reason: str = "") -> str:
        policy = self._policy()
        target = policy.validate_write_path(path)
        self._ensure_algorithm_path_active_for_write(target, action="replace_workspace_file")
        self._ensure_algorithm_path_proposal_approved(target, action="replace_workspace_file")
        before = self._read_text(target)
        self._write_text(target, content)
        self._record_patch_summary("replace_workspace_file", [target], reason=reason)
        context_payload = self._algorithm_context_payload_for_paths([target])
        self._emit(
            "workspace_file_written",
            {
                "scope": "planner",
                "mode": "replace",
                "path": str(target),
                "reason": reason,
                "diff": self._render_diff(target, before, content),
                **context_payload,
            },
        )
        self._maybe_emit_report_generated(target)
        return f"✅ Replaced file: {target}"

    def preview_workspace_diff(
        self,
        path: str,
        new_content: Optional[str] = None,
        patch: Optional[str] = None,
        context_lines: int = 3,
    ) -> str:
        if bool(new_content is not None) == bool(patch is not None):
            return "Error: provide exactly one of new_content or patch"
        policy = self._policy()
        target = policy.validate_write_path(path)
        before = self._read_text(target)
        if new_content is not None:
            after = new_content
        else:
            ops = self._parse_patch(str(patch or ""))
            matched = [op for op in ops if policy.resolve_user_path(op.path) == target]
            if not matched:
                return f"Error: patch does not include target file: {target}"
            after = before
            for op in matched:
                if op.kind == "add":
                    after = "\n".join(line[1:] for line in op.lines) + ("\n" if op.lines else "")
                elif op.kind == "delete":
                    after = ""
                else:
                    after = self._apply_update_lines(after, op.lines)
        diff = self._render_diff(target, before, after, context_lines=context_lines)
        self._emit(
            "workspace_diff_previewed",
            {
                "scope": "planner",
                "path": str(target),
                "context_lines": int(context_lines),
                "diff": diff,
            },
        )
        return diff

    def apply_workspace_patch(self, patch: str, reason: str = "") -> str:
        policy = self._policy()
        ops = self._parse_patch(str(patch or ""))
        updated_paths: List[Path] = []
        diffs: List[Dict[str, Any]] = []
        for op in ops:
            target = policy.validate_write_path(op.path)
            self._ensure_algorithm_path_active_for_write(target, action="apply_workspace_patch")
            self._ensure_algorithm_path_proposal_approved(target, action="apply_workspace_patch")
            updated_paths.append(target)
        for op in ops:
            target = policy.validate_write_path(op.path)
            if op.kind == "add":
                if target.exists():
                    return f"Error: cannot add existing file: {target}"
                content = "\n".join(line[1:] for line in op.lines)
                if op.lines:
                    content += "\n"
                before = ""
                self._write_text(target, content)
                diffs.append({"path": str(target), "kind": "add", "diff": self._render_diff(target, before, content)})
            elif op.kind == "delete":
                if target.exists():
                    before = self._read_text(target)
                    target.unlink()
                    diffs.append({"path": str(target), "kind": "delete", "diff": self._render_diff(target, before, "")})
            else:
                if not target.exists():
                    return f"Error: cannot update missing file: {target}"
                original = self._read_text(target)
                updated = self._apply_update_lines(original, op.lines)
                self._write_text(target, updated)
                diffs.append({"path": str(target), "kind": "update", "diff": self._render_diff(target, original, updated)})
        self._record_patch_summary("apply_workspace_patch", updated_paths, reason=reason)
        context_payload = self._algorithm_context_payload_for_paths(updated_paths)
        self._emit(
            "workspace_patch_applied",
            {
                "scope": "planner",
                "reason": reason,
                "paths": [str(path) for path in updated_paths],
                "diffs": diffs,
                **context_payload,
            },
        )
        for target in updated_paths:
            self._maybe_emit_report_generated(target)
        return "✅ Applied workspace patch to:\n" + "\n".join(f"- {path}" for path in updated_paths)
