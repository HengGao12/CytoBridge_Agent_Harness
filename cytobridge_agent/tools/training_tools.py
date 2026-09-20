"""
Training tools for CytoBridge Agent.

Provides builtin and custom training orchestration around `cb.tl.fit(...)`.
"""
from __future__ import annotations

import json
import logging
import numbers
import os
import sys
import time
import traceback
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from anndata import AnnData
from langchain_core.tools import tool

# Add CytoBridge to path
_CB_PATH = Path(__file__).resolve().parents[2] / "CytoBridge-main"
if _CB_PATH.exists() and str(_CB_PATH) not in sys.path:
    sys.path.insert(0, str(_CB_PATH))

from CytoBridge.tl.fit import resolve_training_data_bundle, resolve_training_device
from CytoBridge.tl.flow_matching_backends import (
    ChunkedCostPairwiseOTCouplingStrategy,
    ChunkedTransportCouplingStrategy,
    CostBasedPairwiseOTCouplingStrategy,
    build_flow_matching_backend,
    default_flow_matching_backend_builder,
)
from CytoBridge.tl.training_algorithm import (
    TrainingAlgorithmContext,
    TrainingAlgorithmSpec,
    FlowMatchingBuildContext,
    ModelBuildContext,
    TrainingDataBundle,
)
from CytoBridge.tl.models import DynamicalModel
from CytoBridge.utils.config import load_config as cb_load_config

from ..schemas import CandidateConfig, PilotResult
from .training_algorithm_registry import (
    LoadedTrainingAlgorithm,
    default_training_algorithm_roots,
    load_training_algorithm,
)

logger = logging.getLogger(__name__)

DEFAULT_CUSTOM_DENSE_PREFLIGHT_MAX_MB = 2048.0
DEFAULT_CUSTOM_DENSE_PREFLIGHT_MAX_PAIR_ENTRIES = 50_000_000


def _override_path_parts(key: Any) -> List[str]:
    parts: List[str] = []
    for dot_part in str(key).split("."):
        if not dot_part:
            continue
        token = ""
        idx = 0
        while idx < len(dot_part):
            char = dot_part[idx]
            if char == "[":
                if token:
                    parts.append(token)
                    token = ""
                end = dot_part.find("]", idx + 1)
                if end < 0:
                    token += dot_part[idx:]
                    break
                bracket_value = dot_part[idx + 1 : end].strip()
                if bracket_value:
                    parts.append(bracket_value)
                idx = end + 1
                continue
            token += char
            idx += 1
        if token:
            parts.append(token)
    return parts


def _json_safe_metric_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe_metric_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_metric_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_safe_metric_value(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return _json_safe_metric_value(value.item())
    return value


def _coerce_metric_sequence(value: Any) -> Optional[List[float]]:
    value = _json_safe_metric_value(value)
    if value is None:
        return None
    if isinstance(value, numbers.Real):
        return [float(value)]
    if not isinstance(value, list):
        return None
    out: List[float] = []
    for item in value:
        if not isinstance(item, numbers.Real):
            return None
        out.append(float(item))
    return out


@dataclass
class PreparedTrainingTarget:
    training_mode: str  # builtin | custom
    display_name: str
    spec: TrainingAlgorithmSpec
    base_config_name: Optional[str]
    base_config_dir: Optional[Path] = None
    loaded_algorithm: Optional[LoadedTrainingAlgorithm] = None


def get_workspace_root() -> Path:
    return _CB_PATH.parent.resolve()


def get_cellcompass_root() -> Path:
    """Return the user-level CellCompass home directory."""
    return (Path.home() / ".cellcompass").resolve()


def get_config_path(model_family: str) -> Path:
    """Get path to the YAML config for a model family."""
    config_dir = _CB_PATH / "CytoBridge" / "configs"

    mapping = {
        "dynamical_ot": "dynamical_ot.yaml",
        "unbalanced_ot": "unbalanced_ot.yaml",
        "ruot": "ruot.yaml",
        "crufm": "crufm.yaml",
        "balanced_ot_cfm": "balanced_ot_cfm.yaml",
        "sf2m": "sf2m.yaml",
        "vgfm": "vgfm.yaml",
        "wfrfm": "wfrfm.yaml",
        "cyto_simulation": "cyto_simulation.yaml",
        "cyto_interaction": "cyto_simulation.yaml",
    }

    filename = mapping.get(model_family, f"{model_family}.yaml")
    return config_dir / filename


def load_config(config_ref: str | Dict[str, Any]) -> Dict[str, Any]:
    """Load a config dict from builtin name, file path, or an existing dict."""
    return deepcopy(cb_load_config(config_ref))


def deep_merge_dict(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge patch into base.

    Dicts merge recursively. Lists merge by index when both sides contain dict
    entries, which lets dataset-specific training.plan overrides update only
    fields such as lr/delta without deleting required stage keys like name/mode.
    """
    out = deepcopy(base)
    for key, value in (patch or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge_dict(out[key], value)
        elif isinstance(value, list) and isinstance(out.get(key), list):
            merged_list = deepcopy(out[key])
            for idx, patch_item in enumerate(value):
                if (
                    idx < len(merged_list)
                    and isinstance(merged_list[idx], dict)
                    and isinstance(patch_item, dict)
                ):
                    merged_list[idx] = deep_merge_dict(merged_list[idx], patch_item)
                elif idx < len(merged_list):
                    merged_list[idx] = deepcopy(patch_item)
                else:
                    merged_list.append(deepcopy(patch_item))
            out[key] = merged_list
        else:
            out[key] = deepcopy(value)
    return out


def apply_overrides(config: Dict[str, Any], overrides: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Apply overrides to a config dict.

    Supports:
    1) dot/bracket-notation keys, e.g. {"training.plan[0].epochs": 50}
    2) nested dict merge, e.g. {"training": {"plan": [...]}}

    Example:
        apply_overrides(cfg, {"training.plan[0].epochs": 50})
    """
    result = deepcopy(config)
    if not overrides:
        return result

    for key, value in overrides.items():
        parts = _override_path_parts(key)
        if not parts:
            continue
        # Nested dict override: deep-merge instead of replacing whole subtree.
        # This avoids accidental loss of required keys such as training.defaults.
        if len(parts) == 1 and isinstance(value, dict):
            existing = result.get(parts[0], {})
            if isinstance(existing, dict):
                result[parts[0]] = deep_merge_dict(existing, value)
            else:
                result[parts[0]] = deepcopy(value)
            continue

        target = result
        for part in parts[:-1]:
            if isinstance(target, list):
                target = target[int(part)]
            elif part.isdigit():
                target = target[int(part)]
            else:
                if part not in target:
                    target[part] = {}
                target = target[part]

        final_key = parts[-1]
        if isinstance(target, list) or final_key.isdigit():
            target[int(final_key)] = value
        else:
            target[final_key] = value

    return result


def _get_config_path_value(config: Dict[str, Any], path: str) -> Tuple[bool, Any]:
    current: Any = config
    try:
        parts = _override_path_parts(path)
    except Exception:
        return False, None
    for part in parts:
        if isinstance(current, list):
            try:
                idx = int(part)
            except Exception:
                return False, None
            if idx < 0 or idx >= len(current):
                return False, None
            current = current[idx]
            continue
        if isinstance(current, dict):
            key = str(part)
            if key not in current:
                return False, None
            current = current[key]
            continue
        return False, None
    return True, current


def _join_override_path(prefix: str, key: Any) -> str:
    parts: List[str] = []
    if prefix:
        parts.extend(_override_path_parts(prefix))
    parts.extend(_override_path_parts(key))
    return ".".join(str(part) for part in parts)


def _flatten_epoch_override_paths(node: Any, prefix: str = "") -> List[str]:
    paths: List[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            path = _join_override_path(prefix, key)
            parts = _override_path_parts(path)
            if parts and parts[-1] == "epochs" and not isinstance(value, (dict, list)):
                paths.append(path)
            if isinstance(value, (dict, list)):
                paths.extend(_flatten_epoch_override_paths(value, path))
    elif isinstance(node, list):
        for idx, value in enumerate(node):
            path = f"{prefix}.{idx}" if prefix else str(idx)
            if isinstance(value, (dict, list)):
                paths.extend(_flatten_epoch_override_paths(value, path))
    return paths


def _coerce_epoch_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        as_float = float(value)
        return int(as_float) if as_float.is_integer() else None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            as_float = float(stripped)
        except Exception:
            return None
        return int(as_float) if as_float.is_integer() else None
    return None


def validate_epoch_override_policy(
    *,
    base_config: Dict[str, Any],
    overrides: Optional[Dict[str, Any]],
    allow_epoch_override: bool = False,
    epoch_override_reason: str = "",
    default_safe_epochs: int = 3000,
) -> List[Dict[str, Any]]:
    """Return blocked epoch overrides after comparing against the effective base config.

    The guard is meant to catch undertrained or otherwise changed epoch schedules.
    It should not block a redundant declaration of the package/default 3000-epoch
    flow-matching schedule, or a patch that leaves an existing epoch value unchanged.
    """
    if not overrides:
        return []
    candidate = apply_overrides(base_config, overrides)
    blocked: List[Dict[str, Any]] = []
    for path in _flatten_epoch_override_paths(overrides):
        old_exists, old_value = _get_config_path_value(base_config, path)
        new_exists, new_value = _get_config_path_value(candidate, path)
        if not new_exists:
            continue
        old_epoch = _coerce_epoch_int(old_value)
        new_epoch = _coerce_epoch_int(new_value)
        unchanged = old_exists and old_epoch is not None and new_epoch == old_epoch
        redundant_default = (not old_exists) and new_epoch == int(default_safe_epochs)
        if unchanged or redundant_default:
            continue
        if not allow_epoch_override:
            blocked.append(
                {
                    "path": path,
                    "old_value": old_value if old_exists else None,
                    "new_value": new_value,
                    "reason": (
                        "epoch override changes the effective training schedule. "
                        "Keep the package/default value, or set allow_epoch_override=true "
                        "with epoch_override_reason explaining convergence evidence."
                    ),
                }
            )
        elif not str(epoch_override_reason or "").strip():
            blocked.append(
                {
                    "path": path,
                    "old_value": old_value if old_exists else None,
                    "new_value": new_value,
                    "reason": "epoch_override_reason is required for effective epoch changes.",
                }
            )
    return blocked


def prune_redundant_config_overrides(
    *,
    base_config: Dict[str, Any],
    overrides: Optional[Dict[str, Any]],
    default_safe_epochs: int = 3000,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Convert broad config copies into sparse leaf overrides and drop no-ops."""
    if not overrides:
        return {}, []
    sparse: Dict[str, Any] = {}
    removed: List[Dict[str, Any]] = []

    def _values_equal(left: Any, right: Any) -> bool:
        if isinstance(left, numbers.Real) and isinstance(right, numbers.Real):
            return float(left) == float(right)
        return left == right

    def _walk(node: Any, prefix: str = "") -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if str(key).startswith("__"):
                    sparse[str(key)] = deepcopy(value)
                    continue
                path = _join_override_path(prefix, key)
                if isinstance(value, dict):
                    _walk(value, path)
                    continue
                if isinstance(value, list):
                    exists, old = _get_config_path_value(base_config, path)
                    if isinstance(old, list) and len(old) == len(value):
                        for idx, item in enumerate(value):
                            child_path = f"{path}.{idx}"
                            if isinstance(item, (dict, list)):
                                _walk(item, child_path)
                            else:
                                old_exists, old_value = _get_config_path_value(base_config, child_path)
                                if old_exists and _values_equal(old_value, item):
                                    removed.append({"path": child_path, "value": item, "reason": "same_as_base"})
                                else:
                                    sparse[child_path] = deepcopy(item)
                    elif exists and _values_equal(old, value):
                        removed.append({"path": path, "value": value, "reason": "same_as_base"})
                    else:
                        sparse[path] = deepcopy(value)
                    continue

                exists, old_value = _get_config_path_value(base_config, path)
                new_epoch = _coerce_epoch_int(value)
                if (
                    not exists
                    and _override_path_parts(path)[-1:] == ["epochs"]
                    and new_epoch == int(default_safe_epochs)
                ):
                    removed.append({"path": path, "value": value, "reason": "redundant_default_epoch"})
                elif exists and _values_equal(old_value, value):
                    removed.append({"path": path, "value": value, "reason": "same_as_base"})
                else:
                    sparse[path] = deepcopy(value)
        elif isinstance(node, list):
            exists, old = _get_config_path_value(base_config, prefix)
            if exists and _values_equal(old, node):
                removed.append({"path": prefix, "value": node, "reason": "same_as_base"})
            else:
                sparse[prefix] = deepcopy(node)

    _walk(overrides)
    return sparse, removed


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_training_device_str(device: str | torch.device) -> str:
    return str(resolve_training_device(device))


def resolve_base_config(
    base_config: str | Dict[str, Any],
    *,
    config_dir: Optional[Path] = None,
) -> tuple[Dict[str, Any], Optional[str]]:
    if isinstance(base_config, dict):
        return deepcopy(base_config), None
    raw_ref = str(base_config or "").strip()
    if not raw_ref:
        raise ValueError("base_config must be a non-empty string or dict")

    candidate_path = Path(raw_ref).expanduser()
    if config_dir is not None and not candidate_path.is_absolute():
        candidate_path = (Path(config_dir).expanduser().resolve() / candidate_path).resolve()
    if candidate_path.is_file():
        return load_config(str(candidate_path)), str(candidate_path)

    # Compatibility: allow "crufm.yaml" form to map to builtin "crufm".
    if raw_ref.lower().endswith((".yaml", ".yml")):
        stem = Path(raw_ref).stem
        try:
            return load_config(stem), stem
        except Exception:
            pass

    if config_dir is not None and (
        raw_ref.startswith(".")
        or "/" in raw_ref
        or "\\" in raw_ref
    ):
        raise FileNotFoundError(
            f"base_config path '{raw_ref}' resolved to '{candidate_path}', but the file does not exist"
        )

    return load_config(raw_ref), raw_ref


def create_builtin_training_target(candidate: CandidateConfig) -> PreparedTrainingTarget:
    spec = TrainingAlgorithmSpec(
        algorithm_id=candidate.name,
        base_config=candidate.name,
        config_overrides=deepcopy(candidate.overrides),
        notes=candidate.rationale or None,
    )
    return PreparedTrainingTarget(
        training_mode="builtin",
        display_name=candidate.name,
        spec=spec,
        base_config_name=candidate.name,
        base_config_dir=None,
    )


def create_custom_training_target(
    *,
    algorithm_id: str,
    input_adata_path: str,
    output_dir: str,
    stage: str,
    metadata: Optional[Dict[str, Any]] = None,
    search_roots: Optional[list[Path]] = None,
    workspace_root: Optional[Path] = None,
    config_overrides: Optional[Dict[str, Any]] = None,
) -> PreparedTrainingTarget:
    workspace_root = Path(workspace_root or get_workspace_root()).resolve()
    search_roots = search_roots or default_training_algorithm_roots(workspace_root)
    loaded = load_training_algorithm(
        algorithm_id,
        search_roots=search_roots,
        workspace_root=workspace_root,
    )
    base_config = loaded.manifest["base_config"]
    resolved_base_config, base_config_name = resolve_base_config(base_config, config_dir=loaded.root_dir)
    context = TrainingAlgorithmContext(
        algorithm_id=algorithm_id,
        input_adata_path=str(input_adata_path),
        output_dir=str(output_dir),
        stage=stage,
        base_config_name=base_config_name,
        resolved_base_config=deepcopy(resolved_base_config),
        metadata=deepcopy(metadata or {}),
    )
    spec = loaded.build_spec(context)
    merged_overrides = deepcopy(spec.config_overrides)
    if config_overrides:
        merged_overrides = deep_merge_dict(merged_overrides, config_overrides)
    spec.config_overrides = merged_overrides
    return PreparedTrainingTarget(
        training_mode="custom",
        display_name=algorithm_id,
        spec=spec,
        base_config_name=base_config_name,
        base_config_dir=loaded.root_dir,
        loaded_algorithm=loaded,
    )


def resolve_training_target(
    *,
    candidate: Optional[CandidateConfig] = None,
    training_algorithm_id: Optional[str] = None,
    input_adata_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    stage: str = "final",
    metadata: Optional[Dict[str, Any]] = None,
    search_roots: Optional[list[Path]] = None,
    workspace_root: Optional[Path] = None,
    config_overrides: Optional[Dict[str, Any]] = None,
) -> PreparedTrainingTarget:
    if bool(candidate) == bool(training_algorithm_id):
        raise ValueError("Exactly one of candidate or training_algorithm_id must be provided")
    if candidate is not None:
        target = create_builtin_training_target(candidate)
        if config_overrides:
            target.spec.config_overrides.update(config_overrides)
        return target
    return create_custom_training_target(
        algorithm_id=str(training_algorithm_id),
        input_adata_path=str(input_adata_path or ""),
        output_dir=str(output_dir or ""),
        stage=stage,
        metadata=metadata,
        search_roots=search_roots,
        workspace_root=workspace_root,
        config_overrides=config_overrides,
    )


def materialize_training_config(
    target: PreparedTrainingTarget,
    *,
    stage: str,
    checkpoints_dir: Optional[Path] = None,
    epochs_scale: float = 1.0,
    max_epochs: Optional[int] = None,
) -> Dict[str, Any]:
    config, _ = resolve_base_config(target.spec.base_config, config_dir=target.base_config_dir)
    config = apply_overrides(config, target.spec.config_overrides)

    if epochs_scale != 1.0 and "training" in config and "plan" in config["training"]:
        for stage_cfg in config["training"]["plan"]:
            if "epochs" in stage_cfg:
                stage_cfg["epochs"] = max(1, int(stage_cfg["epochs"] * epochs_scale))

    if max_epochs is not None and "training" in config and "plan" in config["training"]:
        cap = max(1, int(max_epochs))
        for stage_cfg in config["training"]["plan"]:
            if "epochs" in stage_cfg:
                stage_cfg["epochs"] = min(cap, max(1, int(stage_cfg["epochs"])))

    if checkpoints_dir is not None:
        config["ckpt_dir"] = str(Path(checkpoints_dir).expanduser().resolve())
    return config


def resolve_training_data_for_target(
    *,
    adata: AnnData,
    target: PreparedTrainingTarget,
    resolved_config: Dict[str, Any],
    stage: str,
    device: str,
    outdir: Optional[Path],
) -> TrainingDataBundle:
    return resolve_training_data_bundle(
        adata,
        resolved_config,
        stage=stage,
        device=device,
        output_dir=str(outdir) if outdir is not None else str(resolved_config.get("ckpt_dir", "")),
        training_data_builder=target.spec.training_data_builder,
        metadata={
            "training_mode": target.training_mode,
            "algorithm_id": target.spec.algorithm_id,
            "base_config_name": target.base_config_name,
        },
    )


# Downsampling for pilot training
def downsample_by_time_ratio(
    adata: AnnData,
    time_key: str,
    sample_ratio: float,
    seed: int = 42,
    min_cells_per_time: int = 2,
) -> AnnData:
    """
    Downsample each time point by a ratio instead of a fixed absolute count.

    Args:
        adata: Input AnnData.
        time_key: obs column containing processed time points.
        sample_ratio: Ratio in (0, 1]. 1.0 means no downsampling.
        seed: Random seed.
        min_cells_per_time: Minimum retained cells per time point.
    """
    if not (0.0 < float(sample_ratio) <= 1.0):
        raise ValueError(f"sample_ratio must be in (0, 1], got {sample_ratio}")
    np.random.seed(seed)

    indices = []
    time_points = sorted(adata.obs[time_key].unique())

    for tp in time_points:
        mask = adata.obs[time_key] == tp
        tp_indices = np.where(mask)[0]

        if len(tp_indices) < min_cells_per_time:
            logger.warning(
                "Skipping time point %s: only %s cells (< %s)",
                tp,
                len(tp_indices),
                min_cells_per_time,
            )
            continue

        target_n = max(min_cells_per_time, int(round(len(tp_indices) * float(sample_ratio))))
        target_n = min(len(tp_indices), target_n)
        if target_n < len(tp_indices):
            sampled = np.random.choice(tp_indices, target_n, replace=False)
        else:
            sampled = tp_indices

        indices.extend(sampled.tolist())

    if len(indices) == 0:
        raise ValueError(f"No valid time points found (need ≥{min_cells_per_time} cells per TP)")

    result = adata[sorted(indices)].copy()
    remaining_tps = sorted(result.obs[time_key].unique())
    logger.info(
        "Downsampled by ratio=%.4f: %s → %s time points, %s → %s cells",
        float(sample_ratio),
        len(time_points),
        len(remaining_tps),
        adata.n_obs,
        result.n_obs,
    )
    return result


def train_model(
    adata: AnnData,
    config: Dict[str, Any],
    device: str = "cuda",
    stage: str = "final",
    outdir: Optional[Path] = None,
    progress_callback=None,
    training_algorithm_id: Optional[str] = None,
    training_data_builder=None,
    flow_matching_backend_builder=None,
    flow_matching_loss_hook=None,
    evaluation_metrics_hook=None,
    evaluation_metrics_params=None,
    model_builder=None,
    serialize_model=None,
    stage_runner=None,
    inference_context_builder=None,
    simulation_hook=None,
    prepared_flow_matching_backend=None,
    prepared_flow_matching_stage_name: Optional[str] = None,
    prepared_flow_matching_stage_params: Optional[Dict[str, Any]] = None,
) -> Tuple[AnnData, float, Optional[str]]:
    """
    Train a CytoBridge model using cb.tl.fit().
    """
    try:
        import CytoBridge as cb

        start_time = time.time()
        config = deepcopy(config)
        runtime_device = _resolve_training_device_str(device)
        if outdir:
            outdir = Path(outdir)
            outdir.mkdir(parents=True, exist_ok=True)
            config["ckpt_dir"] = str(outdir)
        else:
            config["ckpt_dir"] = "./cytobridge_output"

        batch_size = config.get("training", {}).get("batch_size", None)
        adata_trained = cb.tl.fit(
            adata,
            config=config,
            batch_size=batch_size,
            device=runtime_device,
            stage=stage,
            progress_callback=progress_callback,
            training_algorithm_id=training_algorithm_id,
            training_data_builder=training_data_builder,
            flow_matching_backend_builder=flow_matching_backend_builder,
            flow_matching_loss_hook=flow_matching_loss_hook,
            evaluation_metrics_hook=evaluation_metrics_hook,
            evaluation_metrics_params=evaluation_metrics_params,
            model_builder=model_builder,
            serialize_model=serialize_model,
            stage_runner=stage_runner,
            inference_context_builder=inference_context_builder,
            simulation_hook=simulation_hook,
            prepared_flow_matching_backend=prepared_flow_matching_backend,
            prepared_flow_matching_stage_name=prepared_flow_matching_stage_name,
            prepared_flow_matching_stage_params=prepared_flow_matching_stage_params,
        )
        runtime = time.time() - start_time
        return adata_trained, runtime, None
    except RuntimeError as e:
        error_msg = str(e)
        tb = traceback.format_exc()
        if "out of memory" in error_msg.lower() or "cuda" in error_msg.lower():
            return adata, 0.0, f"OOM: {error_msg[:200]}\nTraceback:\n{tb}"
        if "nan" in error_msg.lower():
            return adata, 0.0, f"NaN: {error_msg[:200]}\nTraceback:\n{tb}"
        return adata, 0.0, f"Runtime: {error_msg[:200]}\nTraceback:\n{tb}"
    except Exception as e:
        return adata, 0.0, f"Error: {str(e)[:200]}\nTraceback:\n{traceback.format_exc()}"


def _pick_flow_matching_stage(config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    training = config.get("training") if isinstance(config, dict) else None
    plan = training.get("plan") if isinstance(training, dict) else None
    if not isinstance(plan, list):
        return None
    for stage in plan:
        if isinstance(stage, dict) and str(stage.get("mode", "")).lower() == "flow_matching":
            return stage
    return None


def _train_strategy_flags(stage_params: Dict[str, Any]) -> tuple[bool, bool, bool]:
    train_strategy = str(stage_params.get("train_strategy", "s")).lower()
    return ("v" in train_strategy, "g" in train_strategy, "s" in train_strategy)


def _is_mini_batch_capable_coupling(coupling: Any) -> tuple[bool, Optional[int], str]:
    if coupling is None:
        return False, None, "missing coupling strategy"
    if hasattr(coupling, "use_mini_batch_uot"):
        enabled = bool(getattr(coupling, "use_mini_batch_uot", False))
        chunk_size = getattr(coupling, "chunk_size", None)
        if enabled:
            return True, int(chunk_size) if isinstance(chunk_size, int) else None, "use_mini_batch_uot"
    if hasattr(coupling, "use_mini_batch_balanced"):
        enabled = bool(getattr(coupling, "use_mini_batch_balanced", False))
        chunk_size = getattr(coupling, "chunk_size", None)
        if enabled:
            return True, int(chunk_size) if isinstance(chunk_size, int) else None, "use_mini_batch_balanced"
    if hasattr(coupling, "supports_minibatch"):
        enabled = bool(getattr(coupling, "supports_minibatch", False))
        chunk_size = getattr(coupling, "chunk_size", None) or getattr(coupling, "mini_batch_chunk_size", None)
        return enabled, int(chunk_size) if isinstance(chunk_size, int) else None, "supports_minibatch"
    return False, None, "no mini-batch capability marker"


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return float(default)
    try:
        value = float(raw)
    except Exception:
        return float(default)
    return value if value > 0 else float(default)


def _available_memory_mb() -> Optional[float]:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                parts = line.split()
                if len(parts) >= 2:
                    return float(parts[1]) / 1024.0
    except Exception:
        return None
    return None


def _training_pairwise_profile(training_data: TrainingDataBundle) -> Dict[str, Any]:
    shapes: List[Dict[str, Any]] = []
    max_pair_entries = 0
    total_pair_entries = 0
    for idx in range(max(0, len(training_data.latent_by_time) - 1)):
        left = training_data.latent_by_time[idx]
        right = training_data.latent_by_time[idx + 1]
        n0 = int(getattr(left, "shape", [0])[0])
        n1 = int(getattr(right, "shape", [0])[0])
        dim = int(getattr(left, "shape", [0, 0])[1]) if len(getattr(left, "shape", [])) > 1 else 0
        entries = int(n0 * n1)
        max_pair_entries = max(max_pair_entries, entries)
        total_pair_entries += entries
        shapes.append(
            {
                "time_idx": idx,
                "n0": n0,
                "n1": n1,
                "latent_dim": dim,
                "pair_entries": entries,
                "single_float32_matrix_mb": entries * 4.0 / (1024.0 * 1024.0),
            }
        )
    return {
        "time_pair_shapes": shapes,
        "max_pair_entries": int(max_pair_entries),
        "total_pair_entries": int(total_pair_entries),
    }


def flow_matching_memory_preflight(
    *,
    training_data: TrainingDataBundle,
    target: PreparedTrainingTarget,
    backend: Any,
) -> Optional[str]:
    """
    Return a non-blocking memory/scalability warning for custom FM coupling.

    This intentionally does not block preview/training. Static inspection is
    only allowed to flag obvious full pairwise cost helpers; it must not infer
    correctness or scalability from whether a custom class overrides
    build_state/sample_pairs. Actual OOM and timeout failures are handled by the
    isolated training runtime.
    """
    if str(target.training_mode or "") != "custom":
        return None
    if os.getenv("CYTOBRIDGE_DISABLE_CUSTOM_MEMORY_PREFLIGHT", "").strip().lower() in {"1", "true", "yes", "on"}:
        return None

    coupling = getattr(backend, "coupling", None)
    profile = _training_pairwise_profile(training_data)
    max_pair_entries = int(profile.get("max_pair_entries") or 0)
    if max_pair_entries <= 0:
        return None

    coupling_class = coupling.__class__ if coupling is not None else None
    coupling_name = coupling_class.__name__ if coupling_class is not None else "unknown"
    is_cost_based = isinstance(coupling, CostBasedPairwiseOTCouplingStrategy)
    is_chunked_cost_based = isinstance(coupling, (ChunkedTransportCouplingStrategy, ChunkedCostPairwiseOTCouplingStrategy))

    if is_chunked_cost_based:
        return None

    # Only warn for helpers whose contract explicitly returns a full pairwise
    # cost matrix. Custom build_state/sample_pairs implementations may be
    # streaming or sparse, so they are validated by runtime sample_pairs and
    # isolation rather than static heuristics.
    if not is_cost_based:
        return None

    multiplier = 4.0
    estimated_dense_mb = max_pair_entries * 4.0 * multiplier / (1024.0 * 1024.0)
    configured_cap_mb = _env_float(
        "CYTOBRIDGE_CUSTOM_FM_PREFLIGHT_MAX_DENSE_MB",
        DEFAULT_CUSTOM_DENSE_PREFLIGHT_MAX_MB,
    )
    available_mb = _available_memory_mb()
    if available_mb is not None:
        configured_cap_mb = min(configured_cap_mb, max(512.0, available_mb * 0.35))
    max_entries_cap = int(
        _env_float(
            "CYTOBRIDGE_CUSTOM_FM_PREFLIGHT_MAX_PAIR_ENTRIES",
            float(DEFAULT_CUSTOM_DENSE_PREFLIGHT_MAX_PAIR_ENTRIES),
        )
    )

    if estimated_dense_mb <= configured_cap_mb and max_pair_entries <= max_entries_cap:
        return None

    worst_pair = max(
        list(profile.get("time_pair_shapes") or []),
        key=lambda item: int(item.get("pair_entries") or 0),
        default={},
    )
    return (
        "Memory/scalability warning: custom flow-matching coupling may be expensive on this dataset. "
        f"Coupling `{coupling_name}` uses the full pairwise cost helper "
        f"(cost_based={is_cost_based}, chunked_cost_based={is_chunked_cost_based}). "
        f"Worst adjacent time pair t{worst_pair.get('time_idx')} has "
        f"{worst_pair.get('n0')} x {worst_pair.get('n1')} cells "
        f"({max_pair_entries:,} pair entries; one float32 matrix is "
        f"{float(worst_pair.get('single_float32_matrix_mb') or 0.0):.1f} MB). "
        f"Static estimated dense working set is {estimated_dense_mb:.1f} MB, above the warning cap "
        f"{configured_cap_mb:.1f} MB or pair-entry warning cap {max_entries_cap:,}. "
        "This is advisory only and never a mini-batch blocker. If runtime is slow or OOM occurs, "
        "refactor toward the unified chunked/streaming coupling API, avoid storing full "
        "cost/plan/bias matrices, or test first on a smaller benchmark panel."
    )


def validate_flow_matching_coupling_preflight(
    *,
    training_data: TrainingDataBundle,
    target: PreparedTrainingTarget,
    resolved_config: Dict[str, Any],
    device: str = "cuda",
    sample_batch_size: int = 64,
) -> tuple[Optional[str], Optional[Dict[str, Any]]]:
    """
    Validate coupling quality before training so bad custom couplings fail fast
    with actionable messages for the agent.
    """
    if target.spec.stage_runner is not None:
        return None, None
    stage_params = _pick_flow_matching_stage(resolved_config)
    if stage_params is None:
        return None, None
    runtime_device = resolve_training_device(device)
    adata = training_data.adata
    time_points = list(training_data.time_points)
    if len(time_points) < 2:
        return f"Preflight failed: need at least 2 time points, got {len(time_points)}.", None
    X = [x.float().cpu().detach().numpy() for x in training_data.latent_by_time]
    time_tensor = torch.tensor(time_points, dtype=torch.float32, device=runtime_device)
    regress_v, regress_g, regress_score = _train_strategy_flags(stage_params)
    model = None
    try:
        latent_dim = int(training_data.latent_by_time[0].shape[1])
        if target.spec.model_builder is not None:
            model = target.spec.model_builder(
                ModelBuildContext(
                    algorithm_id=target.spec.algorithm_id,
                    resolved_config=deepcopy(resolved_config),
                    latent_dim=latent_dim,
                    training_data=training_data,
                    device=runtime_device,
                    stage="preflight",
                    metadata={
                        "resolved_config": resolved_config,
                        "preflight": True,
                        "training_mode": target.training_mode,
                    },
                )
            )
        else:
            model = DynamicalModel(latent_dim, resolved_config["model"])
        if isinstance(model, torch.nn.Module):
            model = model.to(runtime_device)
    except Exception as exc:
        return f"Preflight failed during model construction: {exc}", None
    build_context = FlowMatchingBuildContext(
        stage_params=stage_params,
        training_data=training_data,
        device=runtime_device,
        regress_v=regress_v,
        regress_g=regress_g,
        regress_score=regress_score,
        model=model,
        metadata={
            "resolved_config": resolved_config,
            "preflight": True,
            "training_mode": target.training_mode,
            "algorithm_id": target.spec.algorithm_id,
        },
    )
    backend_builder = target.spec.flow_matching_backend_builder or default_flow_matching_backend_builder
    backend = build_flow_matching_backend(backend_builder, build_context=build_context)
    preflight_warnings: List[str] = []

    if target.training_mode == "custom":
        mini_batch_ok, chunk_size, marker = _is_mini_batch_capable_coupling(getattr(backend, "coupling", None))
        if not mini_batch_ok:
            preflight_warnings.append(
                "Mini-batch/scalability warning: custom algorithm coupling did not expose a recognized "
                "mini-batch capability marker. This does not block training because markers can be stale "
                "or incomplete; use the actual runtime result as the source of truth. For large datasets, "
                "consider following "
                "`CytoBridge-main/CytoBridge/tl/flow_matching_backends.py` "
                "(for example `CostBasedPairwiseOTCouplingStrategy.build_pairwise_cost(...)`; "
                "`ChunkedTransportCouplingStrategy.build_pairwise_cost_block(...)`; "
                "see also `UnbalancedOTCouplingStrategy` / `BalancedOTCouplingStrategy`) "
                "and `CytoBridge-main/docs/runtime/flow-matching/README.md` mini-batch contract. "
                f"Detected marker: {marker}."
            )
        if mini_batch_ok and (chunk_size is None or chunk_size <= 0):
            preflight_warnings.append(
                "Mini-batch/scalability warning: custom algorithm mini-batch marker is present but "
                "chunk_size is missing/invalid. This is advisory only; if the implementation handles "
                "chunking through another path, training may still be valid. Consider following "
                "`CytoBridge-main/CytoBridge/tl/flow_matching_backends.py` "
                "and setting a positive chunk_size when applicable."
            )

    memory_warning = flow_matching_memory_preflight(
        training_data=training_data,
        target=target,
        backend=backend,
    )
    if memory_warning:
        preflight_warnings.append(memory_warning)

    try:
        state = backend.prepare(X, time_tensor, runtime_device)
    except Exception as exc:
        return f"Preflight failed during backend.prepare: {exc}", None

    plans = list(getattr(state, "plans", []) or [])
    if len(plans) != len(time_points) - 1:
        return (
            f"Preflight failed: coupling plan count mismatch. "
            f"expected={len(time_points)-1}, got={len(plans)}."
        ), None

    def _fmt_float(value: Any) -> str:
        if value is None:
            return "None"
        try:
            value_f = float(value)
        except Exception:
            return str(value)
        if not np.isfinite(value_f):
            return str(value_f)
        return f"{value_f:.3e}"

    def _summarize_plan_store(time_idx: int) -> str:
        try:
            store = state.plan_store(time_idx)
        except Exception as exc:
            return f"plan_store_unavailable={exc}"
        metadata = getattr(store, "metadata", {}) or {}
        pieces: List[str] = [
            f"source_n={getattr(store, 'source_n', '?')}",
            f"target_n={getattr(store, 'target_n', '?')}",
            f"dense={bool(getattr(store, 'dense_plan', None) is not None)}",
            f"chunked={bool(getattr(store, 'sub_plans', None))}",
            f"sparse_edges={bool(getattr(store, 'edge_src', None) is not None)}",
            f"reg={_fmt_float(metadata.get('reg'))}",
            f"reg_m={_fmt_float(metadata.get('reg_m'))}",
            f"solver={metadata.get('solver_mode', metadata.get('storage', 'unknown'))}",
            f"n_chunks={metadata.get('n_chunks', len(getattr(store, 'sub_plans', []) or []))}",
        ]
        dense_plan = getattr(store, "dense_plan", None)
        if dense_plan is not None:
            arr = np.asarray(dense_plan)
            pieces.extend(
                [
                    f"dense_mass={_fmt_float(arr.sum())}",
                    f"dense_positive={int(np.count_nonzero(arr > 0))}",
                ]
            )
        sub_plans = [np.asarray(p) for p in (getattr(store, "sub_plans", []) or [])]
        if sub_plans:
            masses = np.asarray([float(np.asarray(p).sum()) for p in sub_plans], dtype=float)
            positives = np.asarray([int(np.count_nonzero(np.asarray(p) > 0)) for p in sub_plans], dtype=int)
            pieces.extend(
                [
                    f"subplan_mass_min={_fmt_float(np.min(masses))}",
                    f"subplan_mass_median={_fmt_float(np.median(masses))}",
                    f"subplan_mass_max={_fmt_float(np.max(masses))}",
                    f"positive_subplans={int(np.count_nonzero(masses > 1e-12))}/{len(sub_plans)}",
                    f"positive_entries_median={_fmt_float(np.median(positives))}",
                ]
            )
        edge_weight = getattr(store, "edge_weight", None)
        if edge_weight is not None:
            weights = np.asarray(edge_weight)
            pieces.extend(
                [
                    f"edge_mass={_fmt_float(weights.sum())}",
                    f"n_edges={int(weights.size)}",
                    f"positive_edges={int(np.count_nonzero(weights > 0))}",
                ]
            )
        block_metadata = metadata.get("block_metadata")
        if isinstance(block_metadata, list) and block_metadata:
            first = block_metadata[0] if isinstance(block_metadata[0], dict) else {}
            for key in ("cost_min", "cost_max", "cost_median", "cost_mean", "reg", "reg_m", "plan_mass"):
                if key in first:
                    pieces.append(f"first_block_{key}={_fmt_float(first.get(key))}")
        return ", ".join(pieces)

    problems: List[str] = []
    for i, plan in enumerate(plans):
        expected_shape = (X[i].shape[0], X[i + 1].shape[0])
        try:
            pair_batch = backend.coupling.sample_pairs(
                state,
                X,
                i,
                min(sample_batch_size, max(1, expected_shape[0])),
                runtime_device,
            )
        except Exception as exc:
            problems.append(f"t{i}->{i+1}: sample_pairs failed: {exc}; {_summarize_plan_store(i)}")
            continue
        if pair_batch is None or int(getattr(pair_batch.x0, "shape", [0])[0]) == 0:
            problems.append(f"t{i}->{i+1}: sample_pairs returned empty batch; {_summarize_plan_store(i)}")
            continue
        if not isinstance(plan, np.ndarray):
            preflight_warnings.append(
                f"t{i}->{i+1}: coupling plan is {type(plan).__name__}, not np.ndarray; "
                "accepted because runtime sample_pairs returned a non-empty batch."
            )
            continue
        if plan.shape != expected_shape:
            preflight_warnings.append(
                f"t{i}->{i+1}: coupling plan shape {plan.shape} does not match dense shape {expected_shape}; "
                "accepted because runtime sample_pairs returned a non-empty batch. This is expected for "
                "chunked/metadata-backed couplings that do not materialize the full plan."
            )
            continue
        if not np.isfinite(plan).all():
            problems.append(f"t{i}->{i+1}: dense plan contains NaN/inf")
            continue
        total_mass = float(plan.sum())
        if total_mass <= 1e-9:
            problems.append(f"t{i}->{i+1}: dense plan total mass too small ({total_mass:.3e})")

    if problems:
        joined = "; ".join(problems[:8])
        return (
            "Preflight failed: invalid coupling detected before training. "
            "Please revise algorithm coupling/path/regularization. "
            f"Details: {joined}"
        ), None
    return None, {
        "backend": backend,
        "stage_name": stage_params.get("name"),
        "stage_params": deepcopy(stage_params),
        "preflight_warnings": preflight_warnings,
    }


def evaluate_model(
    adata: AnnData,
    config: Dict[str, Any],
    device: str = "cuda",
) -> Tuple[Optional[List[float]], Optional[List[float]], Dict[str, Any], Optional[str]]:
    try:
        del config, device
        if "evaluation_metrics" in adata.uns.keys():
            metrics = adata.uns["evaluation_metrics"]
            w1 = _coerce_metric_sequence(metrics.get("w1_scores"))
            tmv = _coerce_metric_sequence(metrics.get("tmv_scores"))
            custom_metrics = _json_safe_metric_value(metrics.get("custom_metrics"))
            if not isinstance(custom_metrics, dict):
                custom_metrics = {}
            if w1 is None or tmv is None:
                return None, None, custom_metrics, "evaluation_metrics exists but w1_scores/tmv_scores are invalid"
            if len(w1) == 0 or len(tmv) == 0:
                warning = metrics.get("evaluation_warning", "evaluation returned empty metric lists")
                return None, None, custom_metrics, str(warning)
            return w1, tmv, custom_metrics, None
        if "velocity_latent" not in adata.obsm.keys():
            return None, None, {}, "Training did not produce velocity_latent"
        velocity = adata.obsm["velocity_latent"]
        if not np.isfinite(velocity).all():
            return None, None, {}, "velocity_latent contains NaN/inf"
        return None, None, {}, "evaluation_metrics not found in adata.uns"
    except Exception as e:
        logger.warning("Evaluation failed: %s", e)
        return None, None, {}, str(e)[:200]


def _copy_evaluation_metric_metadata(metrics: Dict[str, Any], raw_eval_metrics: Any) -> None:
    """Preserve evaluator provenance needed by campaign gates.

    The package-level trainer records W1 backend identity next to the builtin
    scores. Campaign gates must know whether W1 values were exact, Sinkhorn, or
    another approximation; otherwise an approximate candidate can be compared to
    an exact baseline on a different numeric scale.
    """

    if not isinstance(raw_eval_metrics, dict):
        return
    for key, value in raw_eval_metrics.items():
        name = str(key)
        if name.startswith("w1_backend") or name in {
            "evaluation_prediction_contract",
            "evaluation_trajectory_source",
            "evaluation_trajectory_step",
            "evaluation_sigma",
            "evaluation_trajectory_time_points",
            "evaluation_observed_time_indices",
            "evaluation_trajectory_path",
            "trajectory_path",
            "artifacts",
        }:
            metrics[name] = _json_safe_metric_value(value)


def extract_training_summary(adata: AnnData) -> Dict[str, Any]:
    summary = adata.uns.get("training_summary") if hasattr(adata, "uns") else None
    if not isinstance(summary, dict):
        return {}
    stages = summary.get("stages")
    time_budget = summary.get("time_budget") if isinstance(summary.get("time_budget"), dict) else {}
    inference_time_budget = (
        summary.get("inference_time_budget")
        if isinstance(summary.get("inference_time_budget"), dict)
        else {}
    )
    budget_summary = {}
    if time_budget:
        budget_summary["time_budget"] = dict(time_budget)
    if inference_time_budget:
        budget_summary["inference_time_budget"] = dict(inference_time_budget)
    if not isinstance(stages, list):
        return budget_summary
    clean_stages: List[Dict[str, Any]] = []
    for stage in stages:
        if not isinstance(stage, dict):
            continue
        clean_stages.append(dict(stage))
    if not clean_stages:
        return {"stages": [], **budget_summary} if budget_summary else {"stages": []}
    last_stage = clean_stages[-1]
    result = {
        "stages": clean_stages,
        "last_stage": last_stage,
        "final_saved_loss": last_stage.get("saved_loss"),
        "final_best_loss": last_stage.get("best_loss"),
        "final_last_epoch_loss": last_stage.get("last_epoch_loss"),
    }
    if time_budget:
        result["time_budget"] = dict(time_budget)
    if inference_time_budget:
        result["inference_time_budget"] = dict(inference_time_budget)
    return result


def _make_training_summary_h5_safe(summary: Any) -> Any:
    if not isinstance(summary, dict):
        return summary
    safe_summary = deepcopy(summary)
    stages = safe_summary.get("stages")
    if isinstance(stages, list):
        safe_summary["stages"] = json.dumps(stages, ensure_ascii=False)
        safe_summary["stage_count"] = len(stages)
    return safe_summary


def write_adata_h5ad_safe(adata: AnnData, path: str | Path) -> None:
    """Write AnnData while temporarily converting rich training_summary to HDF5-safe form."""
    original_training_summary = adata.uns.get("training_summary") if hasattr(adata, "uns") else None
    if hasattr(adata, "uns") and "training_summary" in adata.uns:
        adata.uns["training_summary"] = _make_training_summary_h5_safe(original_training_summary)
    try:
        adata.write_h5ad(str(path))
    finally:
        if hasattr(adata, "uns") and "training_summary" in adata.uns:
            adata.uns["training_summary"] = original_training_summary


def execute_training_target(
    adata: AnnData,
    target: PreparedTrainingTarget,
    *,
    stage: str = "final",
    device: str = "cuda",
    epochs_scale: float = 1.0,
    max_epochs: Optional[int] = None,
    outdir: Optional[Path] = None,
    seed: int = 42,
    progress_callback=None,
) -> Tuple[AnnData, Dict[str, Any], Optional[str]]:
    set_seed(seed)
    if stage not in {"pilot", "final"}:
        raise ValueError(f"Unsupported stage: {stage}")
    requested_device = str(device)
    runtime_device = _resolve_training_device_str(device)

    # Stage no longer applies implicit pilot-specific data reduction.
    # Agent should prepare pilot dataset explicitly (e.g., ratio subsampling)
    # before calling training.
    adata_input = adata

    resolved_config = materialize_training_config(
        target,
        stage=stage,
        checkpoints_dir=outdir,
        epochs_scale=epochs_scale,
        max_epochs=max_epochs,
    )
    try:
        training_data = resolve_training_data_for_target(
            adata=adata_input,
            target=target,
            resolved_config=resolved_config,
            stage=stage,
            device=device,
            outdir=outdir,
        )
    except Exception as exc:
        error_text = f"Training data preparation failed: {exc}"
        return adata_input, {
            "error": error_text,
            "runtime_sec": 0.0,
            "config": resolved_config,
            "stage": stage,
            "training_mode": target.training_mode,
            "algorithm_id": target.spec.algorithm_id,
            "base_config_name": target.base_config_name,
            "requested_device": requested_device,
            "resolved_device": runtime_device,
        }, error_text

    preflight_error, preflight_cache = validate_flow_matching_coupling_preflight(
        training_data=training_data,
        target=target,
        resolved_config=resolved_config,
        device=runtime_device,
    )
    if preflight_error:
        return adata_input, {
            "error": preflight_error,
            "runtime_sec": 0.0,
            "config": resolved_config,
            "stage": stage,
            "training_mode": target.training_mode,
            "algorithm_id": target.spec.algorithm_id,
            "base_config_name": target.base_config_name,
            "requested_device": requested_device,
            "resolved_device": runtime_device,
        }, preflight_error

    adata_trained, runtime, error = train_model(
        adata_input,
        resolved_config,
        device=runtime_device,
        stage=stage,
        outdir=outdir,
        progress_callback=progress_callback,
        training_algorithm_id=(
            target.spec.algorithm_id if target.training_mode == "custom" else None
        ),
        training_data_builder=target.spec.training_data_builder,
        flow_matching_backend_builder=target.spec.flow_matching_backend_builder,
        flow_matching_loss_hook=target.spec.flow_matching_loss_hook,
        evaluation_metrics_hook=target.spec.evaluation_metrics_hook,
        evaluation_metrics_params=target.spec.evaluation_metrics_params,
        model_builder=target.spec.model_builder,
        serialize_model=target.spec.serialize_model,
        stage_runner=target.spec.stage_runner,
        inference_context_builder=target.spec.inference_context_builder,
        simulation_hook=target.spec.simulation_hook,
        prepared_flow_matching_backend=(preflight_cache or {}).get("backend"),
        prepared_flow_matching_stage_name=(preflight_cache or {}).get("stage_name"),
        prepared_flow_matching_stage_params=(preflight_cache or {}).get("stage_params"),
    )

    if error:
        return adata_input, {
            "error": error,
            "runtime_sec": runtime,
            "config": resolved_config,
            "stage": stage,
            "training_mode": target.training_mode,
            "algorithm_id": target.spec.algorithm_id,
            "base_config_name": target.base_config_name,
            "requested_device": requested_device,
            "resolved_device": runtime_device,
        }, error

    w1, tmv, custom_metrics, eval_error = evaluate_model(adata_trained, resolved_config, device=device)
    metrics = {
        "w1_scores": w1,
        "tmv_scores": tmv,
        "custom_metrics": custom_metrics,
        "runtime_sec": runtime,
        "config": resolved_config,
        "stage": stage,
        "training_mode": target.training_mode,
        "algorithm_id": target.spec.algorithm_id,
        "base_config_name": target.base_config_name,
        "notes": target.spec.notes,
        "requested_device": requested_device,
        "resolved_device": runtime_device,
    }
    preflight_warnings = (preflight_cache or {}).get("preflight_warnings")
    if isinstance(preflight_warnings, list) and preflight_warnings:
        metrics["preflight_warnings"] = list(preflight_warnings)
    training_summary = extract_training_summary(adata_trained)
    if training_summary:
        metrics["training_summary"] = training_summary
        metrics["final_train_loss"] = training_summary.get("final_saved_loss")
        metrics["best_train_loss"] = training_summary.get("final_best_loss")
        metrics["last_epoch_train_loss"] = training_summary.get("final_last_epoch_loss")
        time_budget = training_summary.get("time_budget")
        if isinstance(time_budget, dict):
            metrics["training_time_budget"] = dict(time_budget)
            metrics["training_timed_out"] = bool(time_budget.get("timed_out"))
            metrics["training_timeout_message"] = str(time_budget.get("message") or "")
            metrics["last_completed_epoch"] = int(time_budget.get("last_epoch") or 0)
            metrics["completed_epochs"] = int(time_budget.get("completed_epochs") or 0)
        inference_time_budget = training_summary.get("inference_time_budget")
        if isinstance(inference_time_budget, dict):
            metrics["inference_time_budget"] = dict(inference_time_budget)
            metrics["inference_timed_out"] = bool(inference_time_budget.get("timed_out"))
            metrics["inference_timeout_message"] = str(inference_time_budget.get("message") or "")
        last_stage = training_summary.get("last_stage") or {}
        if isinstance(last_stage, dict):
            metrics["last_training_stage_name"] = last_stage.get("name")
            metrics["last_training_stage_mode"] = last_stage.get("mode")
    if target.loaded_algorithm is not None:
        metrics["algorithm_source"] = target.loaded_algorithm.source
        metrics["algorithm_root"] = str(target.loaded_algorithm.root_dir)
    raw_eval_metrics = adata_trained.uns.get("evaluation_metrics") if hasattr(adata_trained, "uns") else None
    if isinstance(raw_eval_metrics, dict):
        _copy_evaluation_metric_metadata(metrics, raw_eval_metrics)
        if raw_eval_metrics.get("custom_metrics_warning"):
            metrics["custom_metrics_warning"] = str(raw_eval_metrics.get("custom_metrics_warning"))
        if isinstance(raw_eval_metrics.get("inference_time_budget"), dict):
            metrics["inference_time_budget"] = dict(raw_eval_metrics.get("inference_time_budget") or {})
        if raw_eval_metrics.get("inference_timed_out") is not None:
            metrics["inference_timed_out"] = bool(raw_eval_metrics.get("inference_timed_out"))
        if raw_eval_metrics.get("inference_timeout_message"):
            metrics["inference_timeout_message"] = str(raw_eval_metrics.get("inference_timeout_message") or "")
    if eval_error:
        metrics["evaluation_warning"] = eval_error
    return adata_trained, metrics, None


def run_pilot_training(
    adata: AnnData,
    candidates: List[CandidateConfig],
    device: str = "cuda",
    max_epochs: int = 100,
    outdir: Optional[Path] = None,
    seed: int = 42,
) -> List[PilotResult]:
    results = []
    for i, candidate in enumerate(candidates):
        logger.info("Pilot training candidate %s: %s", i, candidate.name)
        try:
            target = create_builtin_training_target(candidate)
            candidate_outdir = outdir / f"candidate_{i}" if outdir else None
            adata_trained, metrics, error = execute_training_target(
                adata=adata,
                target=target,
                stage="pilot",
                device=device,
                max_epochs=max_epochs,
                outdir=candidate_outdir,
                seed=seed,
            )
            if error:
                error_type = "oom" if "OOM" in error else "nan" if "NaN" in error else "other"
                results.append(
                    PilotResult(
                        candidate_index=i,
                        candidate_name=candidate.name,
                        success=False,
                        error_summary=error,
                        error_type=error_type,
                    )
                )
                continue

            del adata_trained
            results.append(
                PilotResult(
                    candidate_index=i,
                    candidate_name=candidate.name,
                    success=True,
                    w1_scores=metrics.get("w1_scores"),
                    tmv_scores=metrics.get("tmv_scores"),
                    runtime_sec=metrics.get("runtime_sec"),
                    error_summary=metrics.get("evaluation_warning"),
                )
            )
        except Exception as e:
            results.append(
                PilotResult(
                    candidate_index=i,
                    candidate_name=candidate.name,
                    success=False,
                    error_summary=str(e)[:200],
                    error_type="other",
                )
            )
    return results


def run_final_training(
    adata: AnnData,
    candidate: Optional[CandidateConfig] = None,
    *,
    training_target: Optional[PreparedTrainingTarget] = None,
    device: str = "cuda",
    epochs_scale: float = 1.0,
    outdir: Optional[Path] = None,
    seed: int = 42,
    progress_callback=None,
) -> Tuple[AnnData, Dict[str, Any], Optional[str]]:
    if training_target is None:
        if candidate is None:
            raise ValueError("Either candidate or training_target must be provided")
        training_target = create_builtin_training_target(candidate)
    return execute_training_target(
        adata=adata,
        target=training_target,
        stage="final",
        device=device,
        epochs_scale=epochs_scale,
        outdir=outdir,
        seed=seed,
        progress_callback=progress_callback,
    )


@tool
def PilotTrainTool(
    input_path: str,
    candidates_json: str,
    output_dir: str,
    device: str = "cuda",
    max_epochs: int = 100,
) -> List[dict]:
    """
    Run pilot training for multiple candidate configurations.
    """
    import scanpy as sc

    adata = sc.read_h5ad(input_path)
    candidates = [CandidateConfig(**c) for c in json.loads(candidates_json)]
    results = run_pilot_training(
        adata,
        candidates,
        device=device,
        max_epochs=max_epochs,
        outdir=Path(output_dir),
    )
    return [r.model_dump() for r in results]


@tool
def FinalTrainTool(
    input_path: str,
    candidate_json: str,
    output_dir: str,
    device: str = "cuda",
    epochs_scale: float = 1.0,
) -> dict:
    """
    Run final training with the chosen candidate.
    """
    import scanpy as sc
    import torch

    from .training_run_manager import should_save_full_trained_adata

    adata = sc.read_h5ad(input_path)
    candidate = CandidateConfig(**json.loads(candidate_json))

    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    adata_trained, metrics, error = run_final_training(
        adata,
        candidate,
        device=device,
        epochs_scale=epochs_scale,
        outdir=outdir,
    )
    if error is not None:
        return {
            "success": False,
            "h5ad_path": "",
            "model_artifact_path": "",
            "model_state_path": "",
            "metrics": metrics,
            "error": error,
        }

    model_state_path = outdir / "model_state.pt"
    model_artifact_path = outdir / "model_artifact.json"
    torch.save(
        {
            "all_model": dict((adata_trained.uns or {}).get("all_model") or {}),
            "training_summary": (adata_trained.uns or {}).get("training_summary", {}),
        },
        model_state_path,
    )
    model_artifact_path.write_text(
        json.dumps(
            {
                "artifact_type": "cytobridge_model_artifact",
                "schema_version": "1",
                "model_state_path": str(model_state_path),
                "reference_adata_path": str(Path(input_path).expanduser()),
                "input_adata_path": str(Path(input_path).expanduser()),
                "resolved_config_path": str(outdir / "config.yaml"),
                "metrics_path": "",
                "legacy_trained_model_h5ad_path": str(outdir / "trained_model.h5ad") if should_save_full_trained_adata() else "",
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    h5ad_path = outdir / "trained_model.h5ad"
    if should_save_full_trained_adata():
        write_adata_h5ad_safe(adata_trained, h5ad_path)

    return {
        "success": error is None,
        "h5ad_path": str(h5ad_path) if h5ad_path.exists() else "",
        "model_artifact_path": str(model_artifact_path),
        "model_state_path": str(model_state_path),
        "metrics": metrics,
        "error": error,
    }


@tool
def EvaluateTool(
    input_path: str,
    config_path: str,
    device: str = "cuda",
) -> dict:
    """
    Evaluate a trained model.
    """
    import scanpy as sc

    adata = sc.read_h5ad(input_path)
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    w1, tmv, custom_metrics, error = evaluate_model(adata, config, device=device)
    metrics = {
        "w1_scores": w1,
        "tmv_scores": tmv,
        "custom_metrics": custom_metrics,
        "error": error,
    }
    raw_eval_metrics = adata.uns.get("evaluation_metrics") if hasattr(adata, "uns") else None
    _copy_evaluation_metric_metadata(metrics, raw_eval_metrics)
    return metrics
