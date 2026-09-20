from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import yaml

from CytoBridge.tl.training_algorithm import TrainingAlgorithmContext, TrainingAlgorithmSpec
from CytoBridge.utils.config import load_config as cb_load_config


REQUIRED_MANIFEST_FIELDS = {
    "algorithm_id",
    "api_version",
    "entrypoint",
    "entry_function",
    "description",
    "requirements",
    "base_config",
}


@dataclass
class ValidationResult:
    valid: bool
    errors: list[str] = field(default_factory=list)
    manifest: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingAlgorithmRecord:
    algorithm_id: str
    root_dir: Path
    manifest_path: Path
    algorithm_path: Path
    readme_path: Optional[Path]
    manifest: dict[str, Any]
    description: str
    requirements: str
    source: str


@dataclass
class LoadedTrainingAlgorithm(TrainingAlgorithmRecord):
    entry_function_name: str
    builder: Callable[[TrainingAlgorithmContext], TrainingAlgorithmSpec]

    def build_spec(self, context: TrainingAlgorithmContext) -> TrainingAlgorithmSpec:
        try:
            spec = self.builder(context)
        except TypeError as exc:
            message = str(exc)
            legacy_markers = (
                "flow_matching_backend",
                "flow_matching_backend_factory",
                "fit_override",
            )
            if any(marker in message for marker in legacy_markers):
                raise TypeError(
                    f"{self.algorithm_id}: old custom training API is no longer supported. "
                    "Please migrate to TrainingAlgorithmSpec(training_data_builder=..., "
                    "flow_matching_backend_builder=..., flow_matching_loss_hook=..., "
                    "evaluation_metrics_hook=..., model_builder=..., stage_runner=..., simulation_hook=...). "
                    f"Original error: {exc}"
                ) from exc
            raise
        if not isinstance(spec, TrainingAlgorithmSpec):
            raise TypeError(
                f"{self.algorithm_id}: build_training_algorithm must return "
                f"TrainingAlgorithmSpec, got {type(spec).__name__}"
            )
        if spec.algorithm_id != self.algorithm_id:
            raise ValueError(
                f"{self.algorithm_id}: TrainingAlgorithmSpec.algorithm_id must match manifest algorithm_id"
            )
        return spec


DEFAULT_REQUIREMENTS_TEXT = (
    "General temporal single-cell data. "
    "Requires preprocessing contract readiness (time_point_processed and X_latent)."
)


def _normalize_requirements(raw: Any) -> str:
    if isinstance(raw, str):
        text = raw.strip()
        return text or DEFAULT_REQUIREMENTS_TEXT
    if isinstance(raw, (list, tuple, set)):
        items = [str(x).strip() for x in raw if str(x).strip()]
        return "; ".join(items) if items else DEFAULT_REQUIREMENTS_TEXT
    if isinstance(raw, dict):
        try:
            text = json.dumps(raw, ensure_ascii=False)
            return text if text else DEFAULT_REQUIREMENTS_TEXT
        except Exception:
            return DEFAULT_REQUIREMENTS_TEXT
    if raw is None:
        return DEFAULT_REQUIREMENTS_TEXT
    text = str(raw).strip()
    return text or DEFAULT_REQUIREMENTS_TEXT


def default_training_algorithm_roots(workspace_root: Optional[Path] = None) -> list[Path]:
    _ = workspace_root  # kept for API compatibility
    return [Path.home() / ".cellcompass" / "training_algorithms"]


def _detect_source(root: Path, workspace_root: Optional[Path]) -> str:
    if workspace_root is None:
        return "user-global"
    try:
        root.resolve().relative_to(Path(workspace_root).resolve())
        return "project-local"
    except ValueError:
        return "user-global"


def _iter_algorithm_dirs(search_roots: list[Path]) -> list[tuple[Path, Path]]:
    seen: set[Path] = set()
    entries: list[tuple[Path, Path]] = []
    for root in search_roots:
        root = Path(root).expanduser().resolve()
        if root in seen or not root.exists():
            continue
        seen.add(root)
        for child in sorted(root.iterdir()):
            if child.is_dir():
                entries.append((root, child))
    return entries


def validate_training_algorithm(path: Path) -> ValidationResult:
    path = Path(path).expanduser().resolve()
    errors: list[str] = []
    manifest_path = path / "manifest.yaml"
    algorithm_path = path / "algorithm.py"

    manifest: dict[str, Any] = {}
    if not manifest_path.exists():
        errors.append(f"Missing manifest: {manifest_path}")
    else:
        try:
            with manifest_path.open("r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f) or {}
            if not isinstance(loaded, dict):
                errors.append("manifest.yaml must contain a YAML object")
            else:
                manifest = loaded
        except Exception as e:
            errors.append(f"Failed to parse manifest.yaml: {e}")

    if not algorithm_path.exists():
        errors.append(f"Missing algorithm entrypoint: {algorithm_path}")

    if manifest:
        if "requirements" not in manifest:
            manifest["requirements"] = DEFAULT_REQUIREMENTS_TEXT
        missing = REQUIRED_MANIFEST_FIELDS - set(manifest.keys())
        if missing:
            errors.append(f"manifest.yaml missing required fields: {sorted(missing)}")
        if manifest.get("algorithm_id") != path.name:
            errors.append(
                f"algorithm_id '{manifest.get('algorithm_id')}' must match directory name '{path.name}'"
            )
        if manifest.get("api_version") != 1:
            errors.append("api_version must be 1")
        if manifest.get("entrypoint") != "algorithm.py":
            errors.append("entrypoint must be 'algorithm.py' in v1")
        if manifest.get("entry_function") != "build_training_algorithm":
            errors.append("entry_function must be 'build_training_algorithm' in v1")

    return ValidationResult(valid=not errors, errors=errors, manifest=manifest)


def list_training_algorithms(
    search_roots: list[Path],
    workspace_root: Optional[Path] = None,
) -> list[TrainingAlgorithmRecord]:
    records: list[TrainingAlgorithmRecord] = []
    seen_ids: set[str] = set()
    for root, algo_dir in _iter_algorithm_dirs(search_roots):
        validation = validate_training_algorithm(algo_dir)
        if not validation.valid:
            continue
        algorithm_id = validation.manifest["algorithm_id"]
        if algorithm_id in seen_ids:
            continue
        seen_ids.add(algorithm_id)
        records.append(
            TrainingAlgorithmRecord(
                algorithm_id=algorithm_id,
                root_dir=algo_dir,
                manifest_path=algo_dir / "manifest.yaml",
                algorithm_path=algo_dir / "algorithm.py",
                readme_path=(algo_dir / "README.md") if (algo_dir / "README.md").exists() else None,
                manifest=validation.manifest,
                description=str(validation.manifest.get("description") or "").strip(),
                requirements=_normalize_requirements(validation.manifest.get("requirements")),
                source=_detect_source(root, workspace_root),
            )
        )
    return records


def _candidate_campaign_roots(records: list[TrainingAlgorithmRecord]) -> list[Path]:
    roots: list[Path] = []
    seen: set[Path] = set()
    for record in records:
        try:
            training_root = record.root_dir.parent.resolve()
            cellcompass_root = training_root.parent.resolve()
        except Exception:
            continue
        candidate = cellcompass_root / "algorithm_campaigns"
        if candidate in seen:
            continue
        seen.add(candidate)
        roots.append(candidate)
    return roots


def _final_locked_campaign_sort_key(campaign: dict[str, Any]) -> str:
    locked_release = campaign.get("locked_release")
    if not isinstance(locked_release, dict):
        locked_release = {}
    return str(
        locked_release.get("locked_at")
        or campaign.get("locked_at")
        or campaign.get("completed_at")
        or campaign.get("updated_at")
        or campaign.get("created_at")
        or campaign.get("campaign_id")
        or ""
    )


def _is_final_locked_campaign(campaign: dict[str, Any]) -> bool:
    locked_release = campaign.get("locked_release")
    return (
        str(campaign.get("status") or "").strip().lower() == "locked"
        and str(campaign.get("current_stage") or "").strip().lower() == "final_regression"
        and isinstance(locked_release, dict)
        and bool(locked_release)
    )


def _latest_final_locked_campaigns(
    records: list[TrainingAlgorithmRecord],
) -> dict[str, dict[str, Any]]:
    """Return latest final-regression locked campaign metadata per algorithm."""

    algorithm_ids = {record.algorithm_id for record in records}
    latest: dict[str, dict[str, Any]] = {}
    for root in _candidate_campaign_roots(records):
        if not root.exists():
            continue
        for algorithm_id in algorithm_ids:
            algo_campaign_root = root / algorithm_id
            if not algo_campaign_root.exists():
                continue
            for campaign_path in algo_campaign_root.glob("campaign_*/campaign.json"):
                try:
                    with campaign_path.open("r", encoding="utf-8") as f:
                        campaign = json.load(f) or {}
                except Exception:
                    continue
                if not isinstance(campaign, dict) or not _is_final_locked_campaign(campaign):
                    continue
                if str(campaign.get("algorithm_id") or "").strip() != algorithm_id:
                    continue
                sort_key = _final_locked_campaign_sort_key(campaign)
                existing = latest.get(algorithm_id)
                if existing is None or sort_key > str(existing.get("_sort_key") or ""):
                    locked_release = campaign.get("locked_release")
                    if not isinstance(locked_release, dict):
                        locked_release = {}
                    latest[algorithm_id] = {
                        "algorithm_id": algorithm_id,
                        "campaign_id": str(campaign.get("campaign_id") or campaign_path.parent.name),
                        "trial_id": str(locked_release.get("trial_id") or ""),
                        "locked_at": str(
                            locked_release.get("locked_at")
                            or campaign.get("locked_at")
                            or campaign.get("updated_at")
                            or ""
                        ),
                        "path": str(campaign_path),
                        "_sort_key": sort_key,
                    }
    return latest


def _load_module_from_file(module_path: Path):
    module_hash = hashlib.sha1(str(module_path).encode("utf-8")).hexdigest()[:12]
    module_name = f"cytobridge_training_algorithm_{module_hash}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import training algorithm module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_training_algorithm(
    algorithm_id: str,
    search_roots: list[Path],
    workspace_root: Optional[Path] = None,
) -> LoadedTrainingAlgorithm:
    target = str(algorithm_id or "").strip()
    invalid_match_errors: list[str] = []
    for root, algo_dir in _iter_algorithm_dirs(search_roots):
        validation = validate_training_algorithm(algo_dir)
        manifest_id = str(validation.manifest.get("algorithm_id") or "").strip()
        is_target = algo_dir.name == target or manifest_id == target
        if not is_target:
            continue
        if not validation.valid:
            joined = "; ".join(validation.errors)
            invalid_match_errors.append(f"{algo_dir}: {joined}")
            continue
        if manifest_id != target:
            continue

        record = TrainingAlgorithmRecord(
            algorithm_id=manifest_id,
            root_dir=algo_dir,
            manifest_path=algo_dir / "manifest.yaml",
            algorithm_path=algo_dir / "algorithm.py",
            readme_path=(algo_dir / "README.md") if (algo_dir / "README.md").exists() else None,
            manifest=validation.manifest,
            description=str(validation.manifest.get("description") or "").strip(),
            requirements=_normalize_requirements(validation.manifest.get("requirements")),
            source=_detect_source(root, workspace_root),
        )
        module = _load_module_from_file(record.algorithm_path)
        builder = getattr(module, record.manifest["entry_function"], None)
        if builder is None or not callable(builder):
            raise AttributeError(
                f"{algorithm_id}: missing callable entry function {record.manifest['entry_function']}"
            )
        return LoadedTrainingAlgorithm(
            algorithm_id=record.algorithm_id,
            root_dir=record.root_dir,
            manifest_path=record.manifest_path,
            algorithm_path=record.algorithm_path,
            readme_path=record.readme_path,
            manifest=record.manifest,
            description=record.description,
            requirements=record.requirements,
            source=record.source,
            entry_function_name=record.manifest["entry_function"],
            builder=builder,
        )
    if invalid_match_errors:
        raise ValueError(
            f"Training algorithm '{target}' exists but has invalid manifest/entrypoint: "
            + " | ".join(invalid_match_errors)
        )
    raise FileNotFoundError(f"Training algorithm '{algorithm_id}' was not found in {search_roots}")


def resolve_base_config(
    base_config: str | dict[str, Any],
    *,
    config_dir: Optional[Path] = None,
) -> tuple[dict[str, Any], Optional[str]]:
    if isinstance(base_config, dict):
        return dict(base_config), None
    raw_ref = str(base_config or "").strip()
    if not raw_ref:
        raise ValueError("base_config must be a non-empty string or dict")

    candidate_path = Path(raw_ref).expanduser()
    if config_dir is not None and not candidate_path.is_absolute():
        candidate_path = (Path(config_dir).expanduser().resolve() / candidate_path).resolve()
    if candidate_path.is_file():
        return cb_load_config(str(candidate_path)), str(candidate_path)

    if raw_ref.lower().endswith((".yaml", ".yml")):
        stem = Path(raw_ref).stem
        try:
            return cb_load_config(stem), stem
        except Exception:
            pass

    if config_dir is not None and (raw_ref.startswith(".") or "/" in raw_ref or "\\" in raw_ref):
        raise FileNotFoundError(
            f"base_config path '{raw_ref}' resolved to '{candidate_path}', but the file does not exist"
        )

    return cb_load_config(raw_ref), raw_ref


def render_training_algorithm_catalog_context(
    workspace_root: Optional[Path] = None,
    max_items: int = 10,
    max_desc_chars: int = 180,
    max_requirements_chars: int = 220,
    hidden_algorithm_ids: Optional[set[str]] = None,
) -> str:
    roots = default_training_algorithm_roots(workspace_root)
    records = list_training_algorithms(roots, workspace_root=workspace_root)
    hidden = {str(x or "").strip().lower() for x in (hidden_algorithm_ids or set()) if str(x or "").strip()}
    if hidden:
        records = [rec for rec in records if str(rec.algorithm_id).strip().lower() not in hidden]
    locked_campaigns = _latest_final_locked_campaigns(records)
    records = [rec for rec in records if rec.algorithm_id in locked_campaigns]
    records.sort(
        key=lambda rec: str(locked_campaigns.get(rec.algorithm_id, {}).get("_sort_key") or ""),
        reverse=True,
    )
    lines = [
        "Custom Training Algorithms Catalog (latest final-regression locked releases only):",
        "- Only final-regression locked custom algorithms are listed here.",
        "- Proposal-only, developing, failed, parked, and pending-paper algorithms are intentionally excluded.",
        "- Description source: manifest.yaml -> description",
        "- Requirements source: manifest.yaml -> requirements",
        "- README.md is free-form and optional for documentation.",
    ]
    if not records:
        lines.append("- (no final-regression locked custom algorithms found under ~/.cellcompass/training_algorithms)")
        return "\n".join(lines)
    for rec in records[: max(1, int(max_items))]:
        desc = str(rec.description or "").strip() or "(no description)"
        if len(desc) > max_desc_chars:
            desc = desc[: max_desc_chars - 3] + "..."
        requirements = str(rec.requirements or "").strip() or DEFAULT_REQUIREMENTS_TEXT
        if len(requirements) > max_requirements_chars:
            requirements = requirements[: max_requirements_chars - 3] + "..."
        lock_meta = locked_campaigns.get(rec.algorithm_id, {})
        lock_bits = []
        if lock_meta.get("locked_at"):
            lock_bits.append(f"locked_at={lock_meta['locked_at']}")
        if lock_meta.get("campaign_id"):
            lock_bits.append(f"campaign={lock_meta['campaign_id']}")
        if lock_meta.get("trial_id"):
            lock_bits.append(f"trial={lock_meta['trial_id']}")
        lock_text = "; ".join(lock_bits)
        lines.append(
            f"- {rec.algorithm_id}: {desc} | requirements={requirements} "
            f"| final_lock={lock_text or 'recorded'} "
            f"(path={rec.root_dir}, source={rec.source})"
        )
    if len(records) > max_items:
        lines.append(f"- ... and {len(records) - max_items} more final-regression locked custom algorithm(s).")
    return "\n".join(lines)
