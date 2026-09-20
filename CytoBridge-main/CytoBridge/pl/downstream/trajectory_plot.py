"""Trajectory plotting bundle wrappers."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


def plot_ode_trajectories_bundle(
    adata,
    output_dir: str,
    color_key: str = None,
    model: Any = None,
    device: str = "cuda",
    n_trajectories: int = 50,
    n_bins: int | None = None,
) -> Dict[str, object]:
    """Plot deterministic model-generated trajectory ensembles.

    This is the structured downstream entrypoint for deterministic/ODE-style
    rollout figures. It delegates to the package plotting implementation so
    downstream scripts do not fall back to ad hoc centroid or mean-path plots.
    """
    from CytoBridge.pl.plot import plot_ode_trajectories

    if model is None:
        from CytoBridge.utils import load_model_from_adata

        model = load_model_from_adata(adata)

    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    plot_path = plot_ode_trajectories(
        adata=adata,
        model=model,
        output_path=str(outdir),
        n_trajectories=int(n_trajectories),
        n_bins=n_bins,
        device=device,
    )
    artifacts = {}
    for file_name in [
        "ODE_Trajectories_Plot.png",
        "ode_traj.npy",
        "ode_point.npy",
        "trajectory_time_points.npy",
        "evaluation_trajectory.npz",
    ]:
        p = outdir / file_name
        if p.exists():
            artifacts[file_name] = str(p)
    if plot_path:
        artifacts.setdefault("plot", str(plot_path))
    return {
        "space_used": "latent",
        "projection_backend": "package_projection",
        "artifacts": artifacts,
        "warnings": [],
    }


def plot_sde_trajectories_bundle(adata, output_dir: str, color_key: str = None) -> Dict[str, object]:
    from CytoBridge.pl.plot import plot_sde_trajectories

    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    plot_sde_trajectories(
        adata=adata,
        output_path=str(outdir),
        color_key=color_key,
    )
    artifacts = {}
    for file_name in ["SDE_Trajectories_Plot.png"]:
        p = outdir / file_name
        if p.exists():
            artifacts[file_name] = str(p)
    return {
        "space_used": "latent",
        "projection_backend": "none",
        "artifacts": artifacts,
        "warnings": [],
    }
