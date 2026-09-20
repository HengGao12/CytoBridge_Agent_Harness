from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml

try:
    import numpy as np
except Exception:  # pragma: no cover - optional at import time
    np = None


@dataclass
class TrainingRunBundle:
    run_id: str
    run_dir: Path
    logs_dir: Path
    artifacts_dir: Path
    checkpoints_dir: Path
    algorithm_snapshot_dir: Path
    run_manifest_path: Path
    resolved_config_path: Path
    training_log_path: Path
    planner_context_path: Path
    trained_model_path: Path
    model_artifact_path: Path
    model_state_path: Path
    metrics_path: Path


def _slugify(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in value.strip())
    collapsed = "-".join(part for part in cleaned.split("-") if part)
    return collapsed or "run"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if np is not None:
        if isinstance(value, np.ndarray):
            return _json_safe(value.tolist())
        if isinstance(value, np.generic):
            return _json_safe(value.item())
    return value


def generate_run_id(stage: str, name: str, run_label: Optional[str] = None) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    parts = [timestamp, _slugify(stage), _slugify(name)]
    if run_label:
        parts.append(_slugify(run_label))
    return "__".join(parts)


def create_training_run_bundle(
    output_dir: Path,
    stage: str,
    name: str,
    run_label: Optional[str] = None,
) -> TrainingRunBundle:
    base = Path(output_dir).expanduser().resolve() / "training_runs"
    base.mkdir(parents=True, exist_ok=True)
    run_id = generate_run_id(stage=stage, name=name, run_label=run_label)
    run_dir = base / run_id
    if run_dir.exists():
        raise FileExistsError(f"Training run directory already exists: {run_dir}")
    logs_dir = run_dir / "logs"
    artifacts_dir = run_dir / "artifacts"
    checkpoints_dir = artifacts_dir / "checkpoints"
    algorithm_snapshot_dir = run_dir / "algorithm_snapshot"
    logs_dir.mkdir(parents=True, exist_ok=False)
    checkpoints_dir.mkdir(parents=True, exist_ok=False)
    algorithm_snapshot_dir.mkdir(parents=True, exist_ok=False)
    bundle = TrainingRunBundle(
        run_id=run_id,
        run_dir=run_dir,
        logs_dir=logs_dir,
        artifacts_dir=artifacts_dir,
        checkpoints_dir=checkpoints_dir,
        algorithm_snapshot_dir=algorithm_snapshot_dir,
        run_manifest_path=run_dir / "run_manifest.json",
        resolved_config_path=run_dir / "resolved_config.yaml",
        training_log_path=logs_dir / "training.log",
        planner_context_path=logs_dir / "planner_context.json",
        trained_model_path=artifacts_dir / "trained_model.h5ad",
        model_artifact_path=artifacts_dir / "model_artifact.json",
        model_state_path=artifacts_dir / "model_state.pt",
        metrics_path=artifacts_dir / "metrics.json",
    )
    bundle.training_log_path.write_text("", encoding="utf-8")
    return bundle


def write_training_log(bundle: TrainingRunBundle, line: str) -> None:
    with bundle.training_log_path.open("a", encoding="utf-8") as f:
        f.write(line.rstrip() + "\n")


def write_resolved_config(bundle: TrainingRunBundle, resolved_config: dict[str, Any]) -> None:
    with bundle.resolved_config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(resolved_config, f, sort_keys=False, allow_unicode=True)


def write_planner_context(bundle: TrainingRunBundle, planner_context: Optional[dict[str, Any]]) -> None:
    with bundle.planner_context_path.open("w", encoding="utf-8") as f:
        json.dump(_json_safe(planner_context or {}), f, indent=2, ensure_ascii=False)


def snapshot_algorithm(bundle: TrainingRunBundle, algorithm_root: Optional[Path]) -> Optional[Path]:
    if algorithm_root is None:
        return None
    algorithm_root = Path(algorithm_root).expanduser().resolve()
    if not algorithm_root.exists():
        raise FileNotFoundError(f"Algorithm root does not exist: {algorithm_root}")
    for child in algorithm_root.iterdir():
        target = bundle.algorithm_snapshot_dir / child.name
        if child.is_dir():
            shutil.copytree(child, target)
        else:
            shutil.copy2(child, target)
    return bundle.algorithm_snapshot_dir


def write_run_manifest(bundle: TrainingRunBundle, payload: dict[str, Any]) -> None:
    with bundle.run_manifest_path.open("w", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, indent=2, ensure_ascii=False)


def finalize_training_outputs(
    bundle: TrainingRunBundle,
    *,
    metrics: dict[str, Any],
    run_manifest: dict[str, Any],
) -> None:
    with bundle.metrics_path.open("w", encoding="utf-8") as f:
        json.dump(_json_safe(metrics), f, indent=2, ensure_ascii=False)
    write_run_manifest(bundle, run_manifest)


def bundle_to_manifest_dict(bundle: TrainingRunBundle) -> dict[str, str]:
    payload = asdict(bundle)
    return {key: str(value) if isinstance(value, Path) else value for key, value in payload.items()}


def should_save_full_trained_adata() -> bool:
    """Return whether new training runs should persist a full trained AnnData copy.

    The default is intentionally false: large datasets can make every trial write
    multi-GB `trained_model.h5ad` and `checkpoints/adata.h5ad` duplicates. Set
    CYTOBRIDGE_SAVE_TRAINED_H5AD=1 only for legacy workflows that require a
    standalone trained AnnData file.
    """

    raw = str(os.environ.get("CYTOBRIDGE_SAVE_TRAINED_H5AD", "")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def write_model_artifact(
    bundle: TrainingRunBundle,
    *,
    adata_trained: Any,
    input_adata_path: str | Path,
    resolved_config_path: str | Path,
    metrics_path: str | Path | None = None,
) -> dict[str, str]:
    """Persist a compact model artifact without copying the full AnnData matrix."""

    import torch

    all_model = dict((getattr(adata_trained, "uns", {}) or {}).get("all_model") or {})
    if not all_model:
        raise ValueError("Cannot write model artifact: trained AnnData is missing uns['all_model'].")
    training_summary = (getattr(adata_trained, "uns", {}) or {}).get("training_summary", {})
    payload = {
        "all_model": all_model,
        "training_summary": training_summary,
    }
    bundle.model_state_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, bundle.model_state_path)

    manifest = {
        "artifact_type": "cytobridge_model_artifact",
        "schema_version": "1",
        "model_state_path": str(bundle.model_state_path),
        "reference_adata_path": str(Path(input_adata_path).expanduser()),
        "input_adata_path": str(Path(input_adata_path).expanduser()),
        "resolved_config_path": str(Path(resolved_config_path).expanduser()),
        "metrics_path": str(Path(metrics_path).expanduser()) if metrics_path else "",
        "legacy_trained_model_h5ad_path": str(bundle.trained_model_path) if bundle.trained_model_path.exists() else "",
    }
    with bundle.model_artifact_path.open("w", encoding="utf-8") as f:
        json.dump(_json_safe(manifest), f, indent=2, ensure_ascii=False)
    return manifest
