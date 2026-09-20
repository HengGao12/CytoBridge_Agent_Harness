"""Trajectory downstream wrappers with evaluation-kernel semantics."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import numpy as np
import torch

from .contracts import LATENT_KEY, TIME_KEY, ensure_contract


def _unique_sorted(values: Iterable[float]) -> list[float]:
    return [float(v) for v in sorted({round(float(v), 10) for v in values})]


def _build_time_grid(start: float, end: float, n_steps: int, observed: Sequence[float]) -> list[float]:
    if end < start:
        raise ValueError("trajectory end time must be >= start time")
    n_steps = max(1, int(n_steps))
    grid = np.linspace(float(start), float(end), n_steps + 1, dtype=float).tolist()
    return _unique_sorted([*grid, *[float(t) for t in observed if start <= float(t) <= end]])


def _time_indices(trajectory_time_points: Sequence[float], observed: Sequence[float]) -> list[int]:
    indices: list[int] = []
    for target in observed:
        best_idx = min(
            range(len(trajectory_time_points)),
            key=lambda idx: abs(float(trajectory_time_points[idx]) - float(target)),
        )
        if abs(float(trajectory_time_points[best_idx]) - float(target)) > 1e-6:
            raise ValueError(f"trajectory grid does not contain observed time {target}")
        indices.append(int(best_idx))
    return indices


def _select_initial_indices(adata, *, init_time: float, n_cells: int) -> np.ndarray:
    time_values = np.asarray(adata.obs[TIME_KEY].values, dtype=float)
    init_idx = np.where(np.isclose(time_values, float(init_time)))[0]
    if init_idx.size == 0:
        raise ValueError(f"No cells found at initial time {init_time}")
    n_cells = max(1, int(n_cells))
    if init_idx.size <= n_cells:
        return init_idx
    # Deterministic subsampling keeps downstream scripts reproducible.
    chosen_pos = np.linspace(0, init_idx.size - 1, n_cells, dtype=int)
    return init_idx[chosen_pos]


def _integration_dt(time_points: Sequence[float]) -> float:
    gaps = [
        float(time_points[i + 1]) - float(time_points[i])
        for i in range(len(time_points) - 1)
        if float(time_points[i + 1]) > float(time_points[i])
    ]
    if not gaps:
        return 0.01
    return max(min(gaps) / 10.0, 1e-4)


def _nested_get(mapping: Any, path: Sequence[str], default: Any = None) -> Any:
    cur = mapping
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _coerce_float(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _last_training_plan_value(config: dict[str, Any], key: str) -> Any:
    plan = _nested_get(config, ("training", "plan"), [])
    if isinstance(plan, list):
        for stage in reversed(plan):
            if isinstance(stage, dict) and stage.get(key) is not None:
                return stage.get(key)
    defaults = _nested_get(config, ("training", "defaults"), {})
    return defaults.get(key) if isinstance(defaults, dict) else None


def _resolved_config_from_adata_safe(adata) -> tuple[dict[str, Any], str]:
    try:
        from CytoBridge.utils import resolved_config_from_adata

        loaded = resolved_config_from_adata(adata)
        if isinstance(loaded, dict) and loaded:
            return loaded, "adata.uns['all_model'].resolved_config_yaml"
    except Exception:
        pass
    return {}, "unavailable"


def _resolve_steps_from_config(
    *,
    config: dict[str, Any],
    n_steps: int | None,
    start: float,
    end: float,
) -> tuple[int, str]:
    if n_steps is not None:
        return max(1, int(n_steps)), "argument"
    trajectory_step = (
        _coerce_float(_nested_get(config, ("evaluation", "trajectory_step")))
        or _coerce_float(_nested_get(config, ("evaluation", "trajectory_dt")))
        or _coerce_float(_nested_get(config, ("evaluation", "prediction_trajectory_step")))
    )
    if trajectory_step and trajectory_step > 0 and end > start:
        return max(1, int(math.ceil((end - start) / trajectory_step))), "resolved_config.evaluation.trajectory_step"
    return 100, "fallback_default"


def _resolve_sigma_from_config(
    *,
    config: dict[str, Any],
    fallback: float,
) -> tuple[float, str]:
    resolved = (
        _coerce_float(_nested_get(config, ("evaluation", "sigma")))
        or _coerce_float(_nested_get(config, ("evaluation", "trajectory_sigma")))
        or _coerce_float(_nested_get(config, ("evaluation", "simulation_sigma")))
        or _coerce_float(_last_training_plan_value(config, "sigma"))
    )
    if resolved is not None:
        return float(resolved), "resolved_config"
    return float(fallback), "fallback_default"


def _save_evaluation_trajectory_npz(
    *,
    path: Path,
    trajectory_time_points: Sequence[float],
    observed_time_points: Sequence[float],
    observed_time_indices: Sequence[int],
    points_by_time: np.ndarray,
    weights_by_time: np.ndarray,
    source: str,
    metadata: dict[str, Any],
) -> str:
    arrays: dict[str, Any] = {
        "time_points": np.asarray(trajectory_time_points, dtype=np.float64),
        "observed_time_points": np.asarray(observed_time_points, dtype=np.float64),
        "observed_time_indices": np.asarray(observed_time_indices, dtype=np.int64),
    }
    point_keys: list[str] = []
    weight_keys: list[str] = []
    for idx, (points, weights) in enumerate(zip(points_by_time, weights_by_time)):
        point_key = f"points_{idx:06d}"
        weight_key = f"weights_{idx:06d}"
        arrays[point_key] = np.asarray(points)
        arrays[weight_key] = np.asarray(weights).reshape(-1)
        point_keys.append(point_key)
        weight_keys.append(weight_key)
    manifest = {
        "schema_version": 1,
        "format": "cytobridge_evaluation_trajectory_npz",
        "source": str(source),
        "time_points": [float(t) for t in trajectory_time_points],
        "observed_time_points": [float(t) for t in observed_time_points],
        "observed_time_indices": [int(i) for i in observed_time_indices],
        "point_keys": point_keys,
        "weight_keys": weight_keys,
        "metadata": dict(metadata),
    }
    arrays["manifest_json"] = np.asarray(json.dumps(manifest, ensure_ascii=False))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return str(path)


def _run_evaluation_kernel_trajectory(
    *,
    model: Any,
    adata,
    output_dir: str,
    n_cells: int,
    n_steps: int | None,
    sigma: float,
    init_time: float | None,
    end_time: float | None,
    device: str,
    source: str,
) -> Dict[str, Any]:
    from CytoBridge.tl.analysis import simulate_trajectory

    ensure_contract(adata, [LATENT_KEY, TIME_KEY], where=source)
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    observed_times = _unique_sorted(np.asarray(adata.obs[TIME_KEY].values, dtype=float))
    if len(observed_times) < 2:
        raise ValueError(f"Need at least two unique '{TIME_KEY}' values for trajectory generation.")
    start = float(observed_times[0] if init_time is None else init_time)
    end = float(observed_times[-1] if end_time is None else end_time)
    config, config_source = _resolved_config_from_adata_safe(adata)
    n_steps, n_steps_source = _resolve_steps_from_config(config=config, n_steps=n_steps, start=start, end=end)
    trajectory_time_points = _build_time_grid(start, end, int(n_steps), observed_times)
    observed_in_range = [float(t) for t in observed_times if start <= float(t) <= end]
    observed_time_indices = _time_indices(trajectory_time_points, observed_in_range)

    device_obj = torch.device(device if str(device).startswith("cuda") and torch.cuda.is_available() else "cpu")
    model = model.to(device_obj)
    model.eval()
    x_latent = np.asarray(adata.obsm[LATENT_KEY], dtype=np.float32)
    init_idx = _select_initial_indices(adata, init_time=start, n_cells=int(n_cells))
    x0 = torch.as_tensor(x_latent[init_idx], dtype=torch.float32, device=device_obj)
    dt = _integration_dt(trajectory_time_points)

    points, weights = simulate_trajectory(
        adata=adata,
        model=model,
        x0=x0,
        sigma=float(sigma),
        time=trajectory_time_points,
        dt=dt,
        device=device_obj,
    )
    points = np.asarray(points, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32).squeeze(-1)

    observed_points = points[observed_time_indices]
    observed_weights = weights[observed_time_indices]
    np.save(outdir / "ode_traj.npy", points)
    np.save(outdir / "ode_point.npy", observed_points)
    np.save(outdir / "ode_traj_lnw.npy", np.log(np.clip(weights, 1e-30, None)))
    np.save(outdir / "ode_point_lnw.npy", np.log(np.clip(observed_weights, 1e-30, None)))
    np.save(outdir / "trajectory_weights.npy", weights)
    np.save(outdir / "trajectory_time_points.npy", np.asarray(trajectory_time_points, dtype=np.float64))
    np.save(outdir / "observed_time_indices.npy", np.asarray(observed_time_indices, dtype=np.int64))
    evaluation_path = _save_evaluation_trajectory_npz(
        path=outdir / "evaluation_trajectory.npz",
        trajectory_time_points=trajectory_time_points,
        observed_time_points=observed_in_range,
        observed_time_indices=observed_time_indices,
        points_by_time=points,
        weights_by_time=weights,
        source=source,
        metadata={
            "kernel": "CytoBridge.tl.analysis.simulate_trajectory",
            "contract": "TrainingPipeline.evaluate-compatible dense t0-to-final trajectory",
            "sigma": float(sigma),
            "dt": float(dt),
            "init_time": float(start),
            "end_time": float(end),
            "init_indices_count": int(len(init_idx)),
            "resolved_config_source": config_source,
            "n_steps_source": n_steps_source,
        },
    )

    return {
        "space_used": "latent",
        "projection_backend": "none",
        "artifacts": {
            "ode_point": str(outdir / "ode_point.npy"),
            "ode_traj": str(outdir / "ode_traj.npy"),
            "ode_point_lnw": str(outdir / "ode_point_lnw.npy"),
            "ode_traj_lnw": str(outdir / "ode_traj_lnw.npy"),
            "trajectory_weights": str(outdir / "trajectory_weights.npy"),
            "trajectory_time_points": str(outdir / "trajectory_time_points.npy"),
            "observed_time_indices": str(outdir / "observed_time_indices.npy"),
            "evaluation_trajectory": evaluation_path,
        },
        "warnings": [],
        "kernel": "TrainingPipeline.evaluate-compatible simulate_trajectory",
        "source": source,
        "point_shape": list(observed_points.shape),
        "traj_shape": list(points.shape),
        "weight_shape": list(weights.shape),
        "trajectory_time_points": [float(t) for t in trajectory_time_points],
        "observed_time_points": [float(t) for t in observed_in_range],
        "observed_time_indices": [int(i) for i in observed_time_indices],
    }


def generate_ode_trajectory_bundle(
    model: Any,
    adata,
    output_dir: str,
    n_trajectories: int = 50,
    n_bins: int | None = None,
    device: str = "cuda",
    split_true: bool = False,
) -> Dict[str, Any]:
    result = _run_evaluation_kernel_trajectory(
        model=model,
        adata=adata,
        output_dir=output_dir,
        n_cells=n_trajectories,
        n_steps=n_bins,
        sigma=0.0,
        init_time=None,
        end_time=None,
        device=device,
        source="tl.downstream.generate_ode_trajectory_bundle",
    )
    if split_true:
        result["warnings"].append(
            "split_true is ignored by the evaluation-kernel trajectory path; "
            "the bundle now uses TrainingPipeline.evaluate-compatible simulate_trajectory."
        )
    return result


def generate_sde_trajectory_bundle(
    adata,
    output_dir: str,
    n_time_steps: int | None = None,
    sample_traj_num: int = 1000,
    init_time: int = 0,
    device: str = "cuda",
    split_true: bool = False,
) -> Dict[str, Any]:
    from CytoBridge.utils import load_model_from_adata

    model = load_model_from_adata(adata)
    observed_times = _unique_sorted(np.asarray(adata.obs[TIME_KEY].values, dtype=float))
    if int(init_time) == 0 and observed_times and not np.isclose(float(init_time), observed_times[0]):
        resolved_init_time: float | None = None
    else:
        resolved_init_time = float(init_time)
    config, _config_source = _resolved_config_from_adata_safe(adata)
    resolved_sigma, sigma_source = _resolve_sigma_from_config(config=config, fallback=1.0)
    result = _run_evaluation_kernel_trajectory(
        model=model,
        adata=adata,
        output_dir=output_dir,
        n_cells=sample_traj_num,
        n_steps=n_time_steps,
        sigma=float(resolved_sigma),
        init_time=resolved_init_time,
        end_time=None,
        device=device,
        source="tl.downstream.generate_sde_trajectory_bundle",
    )
    result["sigma_source"] = sigma_source
    # Preserve legacy artifact names used by plotting and older reports.
    outdir = Path(output_dir)
    traj = np.load(result["artifacts"]["ode_traj"])
    point = np.load(result["artifacts"]["ode_point"])
    weight = np.load(result["artifacts"]["trajectory_weights"])
    np.save(outdir / "sde_trajec.npy", traj)
    np.save(outdir / "sde_point.npy", point)
    np.save(outdir / "sde_weight.npy", weight)
    result["artifacts"].update(
        {
            "sde_traj": str(outdir / "sde_trajec.npy"),
            "sde_point": str(outdir / "sde_point.npy"),
            "sde_weight": str(outdir / "sde_weight.npy"),
        }
    )
    result["traj_shape"] = list(np.load(result["artifacts"]["sde_traj"]).shape)
    result["point_shape"] = list(np.load(result["artifacts"]["sde_point"]).shape)
    result["weight_shape"] = list(np.load(result["artifacts"]["sde_weight"]).shape)
    if split_true:
        result["warnings"].append(
            "split_true is ignored by the evaluation-kernel trajectory path; "
            "the bundle now uses TrainingPipeline.evaluate-compatible simulate_trajectory."
        )
    return result
