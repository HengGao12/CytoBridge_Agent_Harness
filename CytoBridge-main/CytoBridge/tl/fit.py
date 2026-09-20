from __future__ import annotations

import json
import os
import pathlib
from copy import deepcopy
from typing import Any, Dict, Optional

import numpy as np
import scanpy as sc
import torch
import yaml

from CytoBridge.tl.models import DynamicalModel
from CytoBridge.tl.trainer import TrainingPipeline
from CytoBridge.tl.training_algorithm import (
    ModelBuildContext,
    ModelSerializeContext,
    TrainingDataBuilderContext,
    TrainingDataBundle,
)
from CytoBridge.utils.config import load_config

TIME_KEY = "time_point_processed"
LATENT_KEY = "X_latent"


def _should_save_trained_adata(resolved_config: Dict[str, Any]) -> bool:
    runtime_cfg = resolved_config.get("runtime") if isinstance(resolved_config.get("runtime"), dict) else {}
    if "save_trained_adata" in runtime_cfg:
        return bool(runtime_cfg.get("save_trained_adata"))
    if "save_trained_adata" in resolved_config:
        return bool(resolved_config.get("save_trained_adata"))
    raw = str(os.environ.get("CYTOBRIDGE_SAVE_TRAINED_H5AD", "")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _row_count(value: Any) -> Optional[int]:
    if isinstance(value, torch.Tensor):
        return int(value.shape[0]) if value.ndim >= 1 else None
    if hasattr(value, "shape") and getattr(value, "ndim", 0) >= 1:
        return int(value.shape[0])
    if isinstance(value, (list, tuple)):
        return len(value)
    return None


def _to_float_time(value: Any) -> float:
    if isinstance(value, np.generic):
        return float(value.item())
    return float(value)


def resolve_training_device(device: str | torch.device) -> torch.device:
    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        print("[WARN] Requested CUDA for training but CUDA is unavailable; falling back to CPU.")
        return torch.device("cpu")
    return target


def validate_training_data_bundle(bundle: TrainingDataBundle) -> TrainingDataBundle:
    if not isinstance(bundle, TrainingDataBundle):
        raise TypeError(
            f"training_data_builder must return TrainingDataBundle, got {type(bundle).__name__}"
        )

    adata = bundle.adata
    if adata is None:
        raise ValueError("TrainingDataBundle.adata must be provided.")
    if TIME_KEY not in adata.obs:
        raise ValueError(f"TrainingDataBundle.adata must contain obs['{TIME_KEY}'].")
    if LATENT_KEY not in adata.obsm:
        raise ValueError(f"TrainingDataBundle.adata must contain obsm['{LATENT_KEY}'].")

    n_time = len(bundle.time_points)
    if n_time == 0:
        raise ValueError("TrainingDataBundle.time_points must be non-empty.")
    if len(bundle.latent_by_time) != n_time:
        raise ValueError(
            "TrainingDataBundle.latent_by_time length must equal len(time_points)."
        )
    if len(bundle.obs_indices_by_time) != n_time:
        raise ValueError(
            "TrainingDataBundle.obs_indices_by_time length must equal len(time_points)."
        )

    for i, (tp, latent, indices) in enumerate(
        zip(bundle.time_points, bundle.latent_by_time, bundle.obs_indices_by_time)
    ):
        if not isinstance(latent, torch.Tensor):
            raise TypeError(
                f"TrainingDataBundle.latent_by_time[{i}] must be torch.Tensor, "
                f"got {type(latent).__name__}"
            )
        if latent.ndim != 2:
            raise ValueError(
                f"TrainingDataBundle.latent_by_time[{i}] must be 2D, got shape={tuple(latent.shape)}"
            )
        row_count = int(latent.shape[0])
        index_count = len(indices)
        if row_count != index_count:
            raise ValueError(
                f"TrainingDataBundle.obs_indices_by_time[{i}] has {index_count} rows, "
                f"but latent_by_time[{i}] has {row_count} rows."
            )
        subset_times = np.asarray(adata.obs.iloc[list(indices)][TIME_KEY].values)
        if subset_times.size == 0:
            raise ValueError(f"TrainingDataBundle.obs_indices_by_time[{i}] is empty.")
        if not np.allclose(subset_times.astype(float), float(tp)):
            raise ValueError(
                f"TrainingDataBundle.obs_indices_by_time[{i}] does not align to time_point_processed={tp}."
            )

    if not isinstance(bundle.extra_modalities_by_time, dict):
        raise TypeError("TrainingDataBundle.extra_modalities_by_time must be a dict.")

    for key, values in bundle.extra_modalities_by_time.items():
        if not isinstance(values, list):
            raise TypeError(
                f"TrainingDataBundle.extra_modalities_by_time['{key}'] must be a list aligned to time_points."
            )
        if len(values) != n_time:
            raise ValueError(
                f"TrainingDataBundle.extra_modalities_by_time['{key}'] must have "
                f"{n_time} entries, got {len(values)}."
            )
        for i, (value, latent) in enumerate(zip(values, bundle.latent_by_time)):
            rows = _row_count(value)
            if rows is None:
                raise ValueError(
                    f"Extra modality '{key}' at time index {i} must be row-aligned array/tensor/list-like."
                )
            if rows != int(latent.shape[0]):
                raise ValueError(
                    f"Extra modality '{key}' at time index {i} has {rows} rows but "
                    f"latent_by_time[{i}] has {int(latent.shape[0])} rows."
                )

    return bundle


def build_default_training_data(
    adata: sc.AnnData,
    resolved_config: Dict[str, Any],
    *,
    stage: str,
    device: str,
    output_dir: Optional[str],
    metadata: Optional[dict[str, Any]] = None,
) -> TrainingDataBundle:
    del resolved_config, stage, output_dir

    if TIME_KEY not in adata.obs:
        raise ValueError(f"adata.obs['{TIME_KEY}'] is required.")
    if LATENT_KEY not in adata.obsm:
        raise ValueError(f"adata.obsm['{LATENT_KEY}'] is required.")

    torch_device = resolve_training_device(device)
    time_points = sorted(_to_float_time(t) for t in adata.obs[TIME_KEY].unique())
    latent_by_time = []
    obs_indices_by_time = []

    for t in time_points:
        mask = np.asarray(adata.obs[TIME_KEY].values, dtype=float) == float(t)
        indices = np.where(mask)[0]
        subset_latent = np.asarray(adata.obsm[LATENT_KEY][indices], dtype=np.float32)
        latent_by_time.append(torch.tensor(subset_latent, dtype=torch.float32, device=torch_device))
        obs_indices_by_time.append(indices)

    bundle = TrainingDataBundle(
        adata=adata,
        time_points=time_points,
        latent_by_time=latent_by_time,
        obs_indices_by_time=obs_indices_by_time,
        extra_modalities_by_time={},
        metadata={
            "time_key": TIME_KEY,
            "latent_key": LATENT_KEY,
            **dict(metadata or {}),
        },
    )
    return validate_training_data_bundle(bundle)


def resolve_training_data_bundle(
    adata: sc.AnnData,
    resolved_config: Dict[str, Any],
    *,
    stage: str,
    device: str,
    output_dir: Optional[str],
    training_data_builder=None,
    metadata: Optional[dict[str, Any]] = None,
) -> TrainingDataBundle:
    if training_data_builder is None:
        return build_default_training_data(
            adata,
            resolved_config,
            stage=stage,
            device=device,
            output_dir=output_dir,
            metadata=metadata,
        )

    context = TrainingDataBuilderContext(
        adata=adata,
        resolved_config=deepcopy(resolved_config),
        stage=stage,
        device=device,
        output_dir=output_dir,
        metadata=dict(metadata or {}),
    )
    bundle = training_data_builder(context)
    return validate_training_data_bundle(bundle)


def _make_training_summary_h5_safe(summary: Any) -> Any:
    if not isinstance(summary, dict):
        return summary
    safe_summary = deepcopy(summary)
    stages = safe_summary.get("stages")
    if isinstance(stages, list):
        safe_summary["stages"] = json.dumps(stages, ensure_ascii=False)
        safe_summary["stage_count"] = len(stages)
    return safe_summary


def _build_model_capabilities(model: torch.nn.Module) -> dict[str, Any]:
    components = list(getattr(model, "components", []) or [])
    return {
        "components": components,
        "has_velocity_net": hasattr(model, "velocity_net"),
        "has_growth_net": hasattr(model, "growth_net"),
        "has_score_net": hasattr(model, "score_net"),
        "has_interaction_net": hasattr(model, "interaction_net"),
    }


def _serialize_state_dict_numpy(model: torch.nn.Module) -> dict[str, Any]:
    return {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}


def _build_runtime_model(
    *,
    training_algorithm_id: Optional[str],
    resolved_config: Dict[str, Any],
    training_data: TrainingDataBundle,
    device: torch.device,
    stage: str,
    model_builder=None,
) -> torch.nn.Module:
    latent_dim = int(training_data.latent_by_time[0].shape[1])
    if model_builder is None:
        return DynamicalModel(latent_dim, resolved_config["model"])
    context = ModelBuildContext(
        algorithm_id=str(training_algorithm_id or "builtin"),
        resolved_config=deepcopy(resolved_config),
        latent_dim=latent_dim,
        training_data=training_data,
        device=device,
        stage=stage,
        metadata={"source": "cb.tl.fit"},
    )
    model = model_builder(context)
    if not isinstance(model, torch.nn.Module):
        raise TypeError(
            f"model_builder must return torch.nn.Module, got {type(model).__name__}"
        )
    return model


def _serialize_all_model(
    *,
    model: torch.nn.Module,
    resolved_config: Dict[str, Any],
    training_data: TrainingDataBundle,
    stage: str,
    training_algorithm_id: Optional[str],
    serialize_model=None,
) -> dict[str, Any]:
    provider = "training_algorithm" if training_algorithm_id else "builtin"
    latent_dim = int(training_data.latent_by_time[0].shape[1])
    model_payload: dict[str, Any] = {}
    if serialize_model is not None:
        serialize_context = ModelSerializeContext(
            algorithm_id=str(training_algorithm_id or "builtin"),
            model=model,
            resolved_config=deepcopy(resolved_config),
            latent_dim=latent_dim,
            training_data=training_data,
            stage=stage,
            metadata={"source": "cb.tl.fit"},
        )
        payload = serialize_model(serialize_context)
        if payload is None:
            model_payload = {}
        elif not isinstance(payload, dict):
            raise TypeError(
                f"serialize_model must return dict[str, Any] or None, got {type(payload).__name__}"
            )
        else:
            model_payload = deepcopy(payload)

    return {
        "schema_version": 2,
        "provider": provider,
        "algorithm_id": str(training_algorithm_id) if training_algorithm_id else None,
        "stage": str(stage),
        "model_config": deepcopy(resolved_config.get("model", {})),
        "training_config": {
            "defaults": deepcopy((resolved_config.get("training") or {}).get("defaults", {})),
            "plan": json.dumps(((resolved_config.get("training") or {}).get("plan") or [])),
        },
        "resolved_config_yaml": yaml.safe_dump(
            resolved_config,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        ),
        "model_state_dict": _serialize_state_dict_numpy(model),
        "model_payload": model_payload,
        "capabilities": _build_model_capabilities(model),
    }


def fit(
    adata: sc.AnnData,
    config: Dict[str, Any] | str,
    batch_size: int | None = None,
    device: str = "cuda",
    stage: str = "final",
    progress_callback=None,
    training_data_builder=None,
    flow_matching_backend_builder=None,
    flow_matching_loss_hook=None,
    evaluation_metrics_hook=None,
    evaluation_metrics_params=None,
    training_algorithm_id: str | None = None,
    model_builder=None,
    serialize_model=None,
    stage_runner=None,
    inference_context_builder=None,
    simulation_hook=None,
    prepared_flow_matching_backend=None,
    prepared_flow_matching_stage_name: str | None = None,
    prepared_flow_matching_stage_params: Dict[str, Any] | None = None,
) -> sc.AnnData:
    """
    Train a CytoBridge model on the standard runtime contracts.

    Required runtime contracts:
    - `adata.obs["time_point_processed"]`
    - `adata.obsm["X_latent"]`

    Custom algorithms may augment training inputs via `training_data_builder`,
    but should not replace the canonical time or transcriptomic keys.

    Builtin evaluation metrics (`w1_scores`, `tmv_scores`) are always computed
    by the package. Custom algorithms may append additive metrics via
    `evaluation_metrics_hook`, but may not replace builtin metric definitions.
    """

    resolved_config = load_config(config)
    device_t = resolve_training_device(device)
    device_str = str(device_t)
    output_dir = str(resolved_config.get("ckpt_dir", "./cytobridge_output"))
    training_data = resolve_training_data_bundle(
        adata,
        resolved_config,
        stage=stage,
        device=device_str,
        output_dir=output_dir,
        training_data_builder=training_data_builder,
        metadata={"source": "cb.tl.fit"},
    )

    if batch_size is None:
        batch_size = min(min(x.shape[0] for x in training_data.latent_by_time), 256)

    model = _build_runtime_model(
        training_algorithm_id=training_algorithm_id,
        resolved_config=resolved_config,
        training_data=training_data,
        device=device_t,
        stage=stage,
        model_builder=model_builder,
    )
    trainer = TrainingPipeline(
        model,
        resolved_config,
        batch_size,
        device_t,
        training_data=training_data,
        progress_callback=progress_callback,
        flow_matching_backend_builder=flow_matching_backend_builder,
        flow_matching_loss_hook=flow_matching_loss_hook,
        evaluation_metrics_hook=evaluation_metrics_hook,
        evaluation_metrics_params=evaluation_metrics_params,
        stage_runner=stage_runner,
        inference_context_builder=inference_context_builder,
        simulation_hook=simulation_hook,
        prepared_flow_matching_backend=prepared_flow_matching_backend,
        prepared_flow_matching_stage_name=prepared_flow_matching_stage_name,
        prepared_flow_matching_stage_params=prepared_flow_matching_stage_params,
    )
    model = trainer.train()

    trained_adata = training_data.adata
    all_times = torch.tensor(
        np.asarray(trained_adata.obs[TIME_KEY].values, dtype=np.float32),
        dtype=torch.float32,
        device=device_t,
    ).unsqueeze(1)
    all_data = torch.tensor(
        np.asarray(trained_adata.obsm[LATENT_KEY], dtype=np.float32),
        dtype=torch.float32,
        device=device_t,
    )
    net_input = torch.cat([all_data, all_times], dim=1)

    if hasattr(model, "velocity_net"):
        velocity = model.velocity_net(net_input)
        trained_adata.obsm["velocity_latent"] = velocity.detach().cpu().numpy()

    if hasattr(model, "growth_net"):
        growth = model.growth_net(net_input)
        # Store the predicted growth rate g(t, x) = d/dt log w, not absolute mass.
        trained_adata.obsm["growth_rate"] = growth.detach().cpu().numpy()

    if hasattr(model, "score_net"):
        score = model.score_net(net_input)
        trained_adata.obsm["score_latent"] = score.detach().cpu().numpy()

    trained_adata.uns["all_model"] = _serialize_all_model(
        model=model,
        resolved_config=resolved_config,
        training_data=training_data,
        stage=stage,
        training_algorithm_id=training_algorithm_id,
        serialize_model=serialize_model,
    )
    trained_adata.uns["training_summary"] = getattr(trainer, "training_summary", {})

    ckpt_dir = pathlib.Path(output_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = ckpt_dir / "config.yaml"

    if _should_save_trained_adata(resolved_config):
        h5ad_path = ckpt_dir / "adata.h5ad"
        original_training_summary = trained_adata.uns.get("training_summary")
        if "training_summary" in trained_adata.uns:
            trained_adata.uns["training_summary"] = _make_training_summary_h5_safe(original_training_summary)
        try:
            trained_adata.write_h5ad(h5ad_path)
        finally:
            if "training_summary" in trained_adata.uns:
                trained_adata.uns["training_summary"] = original_training_summary
    with yaml_path.open("w", encoding="utf-8") as yf:
        yaml.safe_dump(resolved_config, yf, default_flow_style=False, allow_unicode=True)

    print(f"Model checkpoints saved -> {ckpt_dir}")

    try:
        metrics = trainer.evaluate(trained_adata)
    except Exception as e:
        print(f"[WARN] trainer.evaluate failed, fallback to placeholder metrics: {e}")
        metrics = {
            "w1_scores": [],
            "tmv_scores": [],
            "custom_metrics": {},
            "evaluation_warning": str(e),
        }
    trained_adata.uns["training_summary"] = getattr(trainer, "training_summary", {})
    trained_adata.uns["evaluation_metrics"] = metrics

    return trained_adata
