"""Worker process entrypoint for isolated CytoBridge training."""
from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import resource
except Exception:  # pragma: no cover - non-Unix fallback
    resource = None  # type: ignore[assignment]

from ..schemas import CandidateConfig
from .claim_metric_evaluator import (
    claim_metric_spec_has_evaluator,
    load_campaign_claim_metric_evaluator,
    merge_evaluation_metric_hooks,
)
from .training_run_manager import (
    TrainingRunBundle,
    should_save_full_trained_adata,
    write_model_artifact,
    write_training_log,
)
from .training_tools import (
    execute_training_target,
    get_cellcompass_root,
    materialize_training_config,
    resolve_training_target,
    resolve_training_data_for_target,
    validate_flow_matching_coupling_preflight,
    write_adata_h5ad_safe,
)


def _set_memory_limit(memory_limit_mb: Optional[float]) -> None:
    if resource is None:
        return
    try:
        limit = float(memory_limit_mb or 0)
    except Exception:
        limit = 0.0
    if limit <= 0:
        return
    bytes_limit = int(limit * 1024 * 1024)
    try:
        resource.setrlimit(resource.RLIMIT_AS, (bytes_limit, bytes_limit))
    except Exception:
        pass


def _bundle_from_payload(payload: Dict[str, Any]) -> TrainingRunBundle:
    raw = dict(payload.get("bundle") or {})
    return TrainingRunBundle(
        run_id=str(raw["run_id"]),
        run_dir=Path(str(raw["run_dir"])),
        logs_dir=Path(str(raw["logs_dir"])),
        artifacts_dir=Path(str(raw["artifacts_dir"])),
        checkpoints_dir=Path(str(raw["checkpoints_dir"])),
        algorithm_snapshot_dir=Path(str(raw["algorithm_snapshot_dir"])),
        run_manifest_path=Path(str(raw["run_manifest_path"])),
        resolved_config_path=Path(str(raw["resolved_config_path"])),
        training_log_path=Path(str(raw["training_log_path"])),
        planner_context_path=Path(str(raw["planner_context_path"])),
        trained_model_path=Path(str(raw["trained_model_path"])),
        model_artifact_path=Path(str(raw.get("model_artifact_path") or Path(str(raw["artifacts_dir"])) / "model_artifact.json")),
        model_state_path=Path(str(raw.get("model_state_path") or Path(str(raw["artifacts_dir"])) / "model_state.pt")),
        metrics_path=Path(str(raw["metrics_path"])),
    )


def _attach_claim_metric_evaluator(training_target: Any, spec: Dict[str, Any]) -> Dict[str, Any]:
    if not claim_metric_spec_has_evaluator(spec):
        return {}
    explicit_baseline_algorithm = str(
        (spec or {}).get("_baseline_algorithm_id")
        or (spec or {}).get("baseline_algorithm_id")
        or ""
    ).strip().lower()
    baseline_algorithm = explicit_baseline_algorithm or (
        str(getattr(training_target.spec, "algorithm_id", "") or "").strip().lower()
        if str(getattr(training_target, "training_mode", "") or "") == "builtin"
        else ""
    )
    hook, info = load_campaign_claim_metric_evaluator(spec, baseline_algorithm=baseline_algorithm)
    if hook is None:
        return {}
    metric_name = str(info.get("name") or spec.get("name") or "claim_metric")
    training_target.spec.evaluation_metrics_hook = merge_evaluation_metric_hooks(
        training_target.spec.evaluation_metrics_hook,
        hook,
        claim_metric_name=metric_name,
    )
    params = dict(training_target.spec.evaluation_metrics_params or {})
    params["campaign_claim_metric_evaluator"] = dict(info)
    training_target.spec.evaluation_metrics_params = params
    return dict(info)


def _resolve_worker_target(payload: Dict[str, Any], adata_path: str) -> Any:
    candidate_name = str(payload.get("candidate_name") or "").strip()
    training_algorithm_id = str(payload.get("training_algorithm_id") or "").strip()
    config_overrides = dict(payload.get("config_overrides") or {})
    candidate = None
    if candidate_name:
        candidate = CandidateConfig(
            name=candidate_name,
            overrides={},
            rationale="Reconstructed inside isolated training worker",
        )
    workspace_root = Path(str(payload.get("workspace_root") or Path.cwd())).expanduser().resolve()
    search_roots = [get_cellcompass_root() / "training_algorithms"] if training_algorithm_id else None
    return resolve_training_target(
        candidate=candidate,
        training_algorithm_id=training_algorithm_id or None,
        input_adata_path=adata_path,
        output_dir=str(payload.get("output_dir") or ""),
        stage=str(payload.get("stage") or "pilot"),
        metadata={
            "isolated_worker": True,
            "purpose": str(payload.get("purpose") or "training"),
        },
        search_roots=search_roots,
        workspace_root=workspace_root,
        config_overrides=config_overrides,
    )


def _resolve_training_plan(config: Dict[str, Any]) -> list[Dict[str, Any]]:
    training = config.get("training") if isinstance(config, dict) else None
    defaults = training.get("defaults") if isinstance(training, dict) else None
    plan = training.get("plan") if isinstance(training, dict) else None
    if not isinstance(plan, list):
        return []
    resolved: list[Dict[str, Any]] = []
    for stage_cfg in plan:
        if not isinstance(stage_cfg, dict):
            continue
        merged = dict(defaults or {})
        merged.update(stage_cfg)
        resolved.append(
            {
                "name": str(merged.get("name") or ""),
                "mode": str(merged.get("mode") or ""),
                "train_strategy": str(merged.get("train_strategy") or ""),
                "epochs": merged.get("epochs"),
            }
        )
    return resolved


def _run_preview_inspection(payload: Dict[str, Any]) -> Dict[str, Any]:
    _set_memory_limit(payload.get("memory_limit_mb"))
    bundle = _bundle_from_payload(payload)
    adata_path = str(payload.get("adata_path") or "")
    stage = str(payload.get("stage") or "pilot")
    device = str(payload.get("device") or "cpu")

    import scanpy as sc

    adata = sc.read_h5ad(adata_path)
    training_target = _resolve_worker_target(payload, adata_path)
    resolved_config = materialize_training_config(
        training_target,
        stage=stage,
        checkpoints_dir=bundle.checkpoints_dir,
        max_epochs=1,
    )
    training_data = resolve_training_data_for_target(
        adata=adata,
        target=training_target,
        resolved_config=resolved_config,
        stage=stage,
        device=device,
        outdir=bundle.checkpoints_dir,
    )
    preflight_error, preflight_cache = validate_flow_matching_coupling_preflight(
        training_data=training_data,
        target=training_target,
        resolved_config=resolved_config,
        device=device,
    )
    flow_matching_stage = None
    for stage_cfg in _resolve_training_plan(resolved_config):
        if str(stage_cfg.get("mode") or "").lower() == "flow_matching":
            flow_matching_stage = stage_cfg
            break
    warnings = list((preflight_cache or {}).get("preflight_warnings") or [])
    metrics = {
        "preview_inspection": {
            "ok": not bool(preflight_error),
            "preflight_error": preflight_error or "",
            "preflight_warnings": warnings,
            "resolved_training_plan": _resolve_training_plan(resolved_config),
            "selected_stage_name": str((flow_matching_stage or {}).get("name") or ""),
            "selected_stage_mode": str((flow_matching_stage or {}).get("mode") or ""),
            "training_mode": training_target.training_mode,
            "algorithm_id": training_target.spec.algorithm_id,
            "base_config_name": training_target.base_config_name,
            "requested_device": str(payload.get("requested_device") or payload.get("device") or ""),
            "resolved_device": device,
            "isolation_note": (
                "Custom algorithm preview inspection ran in a subprocess so OOM/runtime failures "
                "do not kill the planner process."
            ),
        }
    }
    return {
        "ok": not bool(preflight_error),
        "error": preflight_error or None,
        "metrics": metrics,
        "trained_model_path": "",
    }


def _run(payload: Dict[str, Any]) -> Dict[str, Any]:
    if str(payload.get("purpose") or "").strip().lower() == "preview_inspect":
        return _run_preview_inspection(payload)

    _set_memory_limit(payload.get("memory_limit_mb"))
    bundle = _bundle_from_payload(payload)
    adata_path = str(payload.get("adata_path") or "")
    stage = str(payload.get("stage") or "pilot")
    device = str(payload.get("device") or "cpu")
    claim_metric_spec = dict(payload.get("claim_metric_spec") or {})

    import scanpy as sc

    adata = sc.read_h5ad(adata_path)
    training_target = _resolve_worker_target(payload, adata_path)
    claim_info = _attach_claim_metric_evaluator(training_target, claim_metric_spec)

    def progress_callback(message: str, progress: float) -> None:
        write_training_log(bundle, f"[{progress:.3f}] {message}")

    adata_trained, metrics, error = execute_training_target(
        adata=adata,
        target=training_target,
        stage=stage,
        device=device,
        max_epochs=payload.get("max_epochs"),
        outdir=bundle.checkpoints_dir,
        seed=int(payload.get("seed") or 42),
        progress_callback=progress_callback,
    )
    metrics = dict(metrics or {})
    if claim_info:
        metrics["claim_metric_evaluator"] = claim_info
    model_artifact_path = ""
    if not error:
        manifest = write_model_artifact(
            bundle,
            adata_trained=adata_trained,
            input_adata_path=adata_path,
            resolved_config_path=bundle.resolved_config_path,
            metrics_path=bundle.metrics_path,
        )
        model_artifact_path = str(bundle.model_artifact_path)
        if should_save_full_trained_adata():
            write_adata_h5ad_safe(adata_trained, bundle.trained_model_path)
            manifest["legacy_trained_model_h5ad_path"] = str(bundle.trained_model_path)
            with bundle.model_artifact_path.open("w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)
    return {
        "ok": not bool(error),
        "error": error,
        "metrics": metrics,
        "trained_model_path": str(bundle.trained_model_path) if (not error and bundle.trained_model_path.exists()) else "",
        "model_artifact_path": model_artifact_path,
    }


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("usage: python -m cytobridge_agent.tools.training_isolation_worker PAYLOAD_JSON RESULT_JSON", file=sys.stderr)
        return 2
    payload_path = Path(argv[1])
    result_path = Path(argv[2])
    try:
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        result = _run(payload)
    except BaseException as exc:  # child must convert even MemoryError into JSON if possible
        result = {
            "ok": False,
            "error": str(exc),
            "metrics": {
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        }
    try:
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        print(json.dumps(result, ensure_ascii=False), file=sys.stderr)
        return 1
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
