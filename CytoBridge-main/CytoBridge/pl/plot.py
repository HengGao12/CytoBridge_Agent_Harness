"""Clean plotting APIs for CytoBridge.

This module replaces the previous monolithic plotting implementation with a
stable, minimal, and testable set of functions while preserving commonly used
function names.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
from anndata import AnnData


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _as_file_path(output_path: Optional[str], default_dir: str, default_name: str) -> Path:
    if not output_path:
        return _ensure_dir(Path(default_dir)) / default_name
    p = Path(output_path).expanduser().resolve()
    if p.suffix:
        _ensure_dir(p.parent)
        return p
    return _ensure_dir(p) / default_name


def _as_dir(output_path: Optional[str], default_dir: str = "figures") -> Path:
    if not output_path:
        return _ensure_dir(Path(default_dir).resolve())
    p = Path(output_path).expanduser().resolve()
    if p.suffix:
        return _ensure_dir(p.parent)
    return _ensure_dir(p)


def _pick_basis(adata: AnnData, dim_reduction: str = "none") -> Tuple[str, np.ndarray]:
    req = str(dim_reduction or "none").lower()
    if req in ("umap", "x_umap"):
        if "X_umap" not in adata.obsm:
            sc.pp.neighbors(adata)
            sc.tl.umap(adata)
        return "umap", np.asarray(adata.obsm["X_umap"], dtype=float)[:, :2]
    if req in ("pca", "x_pca"):
        if "X_pca" not in adata.obsm:
            if "X_latent" in adata.obsm:
                adata.obsm["X_pca"] = np.asarray(adata.obsm["X_latent"], dtype=float)
            else:
                sc.tl.pca(adata)
        return "pca", np.asarray(adata.obsm["X_pca"], dtype=float)[:, :2]

    if "X_umap" in adata.obsm:
        return "umap", np.asarray(adata.obsm["X_umap"], dtype=float)[:, :2]
    if "X_pca" in adata.obsm:
        return "pca", np.asarray(adata.obsm["X_pca"], dtype=float)[:, :2]
    if "X_latent" in adata.obsm:
        x = np.asarray(adata.obsm["X_latent"], dtype=float)
        if x.shape[1] >= 2:
            return "latent", x[:, :2]
    x = np.asarray(adata.X, dtype=float)
    if x.ndim != 2 or x.shape[1] < 2:
        raise ValueError("No 2D embedding can be resolved from adata")
    return "X", x[:, :2]


def _get_vec(adata: AnnData, key: str, fallback_obsm: Optional[str] = None) -> Optional[np.ndarray]:
    if key in adata.layers:
        return np.asarray(adata.layers[key], dtype=float)
    if key in adata.obsm:
        return np.asarray(adata.obsm[key], dtype=float)
    if fallback_obsm and fallback_obsm in adata.obsm:
        return np.asarray(adata.obsm[fallback_obsm], dtype=float)
    return None


def _projection_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError("expected 2D array")
    if x.shape[1] <= 2:
        return x[:, :2]
    # PCA via SVD
    x0 = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x0, full_matrices=False)
    comp = vt[:2].T
    return x0 @ comp


def _color_array(adata: AnnData, key: Optional[str]) -> Optional[np.ndarray]:
    if not key:
        return None
    if key not in adata.obs.columns:
        return None
    col = adata.obs[key]
    if np.issubdtype(col.dtype, np.number):
        return np.asarray(col, dtype=float)
    # categorical fallback to codes
    return np.asarray(col.astype("category").cat.codes, dtype=float)


def plot_growth(adata: AnnData, dim_reduction: str = "umap", output_path: Optional[str] = None):
    """Plot growth rate on embedding.

    Returns the saved figure path.
    """
    if "growth_rate" in adata.obsm:
        growth = np.asarray(adata.obsm["growth_rate"]).reshape(-1)
    elif "growth_rate" in adata.obs.columns:
        growth = np.asarray(adata.obs["growth_rate"], dtype=float)
    else:
        raise ValueError("Growth rate not found in adata.obsm or adata.obs")

    _, emb = _pick_basis(adata, dim_reduction=dim_reduction)
    p = _as_file_path(output_path, "figures", "growth_rate.png")

    fig, ax = plt.subplots(figsize=(7, 5))
    sca = ax.scatter(emb[:, 0], emb[:, 1], c=growth, cmap="RdYlBu_r", s=8, alpha=0.85)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Predicted Growth Rate")
    cbar = fig.colorbar(sca, ax=ax)
    cbar.set_label("growth_rate")
    fig.tight_layout()
    fig.savefig(p, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(p)


def plot_ode(X, point_array, traj_array, save_dir):
    """Compatibility helper for legacy ODE plotting calls."""
    p = _as_file_path(str(save_dir), "figures", "ode_trajectories.png")
    fig, ax = plt.subplots(figsize=(8, 6))

    # scatter observed points by time
    for t, arr in enumerate(X):
        arr = np.asarray(arr)
        if arr.ndim == 2 and arr.shape[1] >= 2:
            ax.scatter(arr[:, 0], arr[:, 1], s=10, alpha=0.25, label=f"T{t}")

    traj = np.asarray(traj_array)
    if traj.ndim == 3 and traj.shape[2] >= 2:
        for j in range(traj.shape[1]):
            cur = traj[:, j, :2]
            mask = np.all(np.isfinite(cur), axis=1)
            cur = cur[mask]
            if cur.shape[0] >= 2:
                ax.plot(cur[:, 0], cur[:, 1], color="red", alpha=0.6, linewidth=1)

    pts = np.asarray(point_array)
    if pts.ndim == 3 and pts.shape[2] >= 2:
        pts2 = pts.reshape(-1, pts.shape[2])[:, :2]
        ax.scatter(pts2[:, 0], pts2[:, 1], s=8, alpha=0.5, color="black")

    ax.set_title("ODE trajectories")
    ax.set_xlabel("dim1")
    ax.set_ylabel("dim2")
    fig.tight_layout()
    fig.savefig(p, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(p)


def plot_ode_trajectories(
    adata: AnnData,
    model: Any,
    output_path: str = "figures",
    n_trajectories: int = 50,
    n_bins: int = 100,
    device: str = "cuda",
    split_true: bool = False,
):
    """Generate and plot ODE trajectories."""
    from CytoBridge.tl.downstream.trajectory import generate_ode_trajectory_bundle

    outdir = _as_dir(output_path)
    res = generate_ode_trajectory_bundle(
        model=model,
        adata=adata,
        output_dir=str(outdir),
        n_trajectories=int(n_trajectories),
        n_bins=int(n_bins),
        device=device,
        split_true=split_true,
    )

    point = np.load(res["artifacts"]["ode_point"])
    traj = np.load(res["artifacts"]["ode_traj"])

    if "X_latent" in adata.obsm:
        base = np.asarray(adata.obsm["X_latent"], dtype=float)
    else:
        base = np.asarray(adata.X, dtype=float)
    base2 = _projection_2d(base)

    # project simulated trajectory to same 2D basis using linear regression on latent if possible
    traj2 = traj[:, :, :2] if traj.shape[-1] >= 2 else np.pad(traj, ((0, 0), (0, 0), (0, 2 - traj.shape[-1])))
    point2 = point[:, :, :2] if point.shape[-1] >= 2 else np.pad(point, ((0, 0), (0, 0), (0, 2 - point.shape[-1])))

    p = outdir / "ODE_Trajectories_Plot.png"
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(base2[:, 0], base2[:, 1], s=5, alpha=0.15, color="gray")
    for j in range(traj2.shape[1]):
        cur = traj2[:, j, :]
        mask = np.all(np.isfinite(cur), axis=1)
        cur = cur[mask]
        if cur.shape[0] >= 2:
            ax.plot(cur[:, 0], cur[:, 1], color="red", linewidth=0.9, alpha=0.65)
    ax.scatter(point2.reshape(-1, 2)[:, 0], point2.reshape(-1, 2)[:, 1], s=8, alpha=0.45, color="black")
    ax.set_title("ODE trajectories")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(p, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(p)


def plot_sde_trajectories(
    adata: AnnData,
    output_path: Optional[str] = None,
    color_key: Optional[str] = None,
    device: str = "cuda",
    dim_reduction: str = "none",
    n_trajectories: int = 50,
):
    """Generate/plot SDE trajectories on 2D embedding."""
    from CytoBridge.tl.downstream.trajectory import generate_sde_trajectory_bundle

    outdir = _as_dir(output_path)
    res = generate_sde_trajectory_bundle(
        adata=adata,
        output_dir=str(outdir),
        n_time_steps=40,
        sample_traj_num=max(100, int(n_trajectories) * 8),
        init_time=0,
        device=device,
    )

    sde_traj = np.load(res["artifacts"]["sde_traj"])
    # shape: [steps, n_cells, d]
    traj = np.asarray(sde_traj)

    _, emb = _pick_basis(adata, dim_reduction=dim_reduction)
    col = _color_array(adata, color_key)

    p = outdir / "SDE_Trajectories_Plot.png"
    fig, ax = plt.subplots(figsize=(8, 6))
    if col is None:
        ax.scatter(emb[:, 0], emb[:, 1], s=5, alpha=0.15, color="gray")
    else:
        ax.scatter(emb[:, 0], emb[:, 1], c=col, s=5, alpha=0.25, cmap="tab20")

    if traj.ndim == 3 and traj.shape[-1] >= 2:
        m = min(traj.shape[1], max(20, int(n_trajectories)))
        idx = np.linspace(0, traj.shape[1] - 1, num=m, dtype=int)
        for j in idx:
            cur = traj[:, j, :2]
            mask = np.all(np.isfinite(cur), axis=1)
            cur = cur[mask]
            if cur.shape[0] >= 2:
                ax.plot(cur[:, 0], cur[:, 1], color="red", linewidth=0.8, alpha=0.55)

    ax.set_title("SDE trajectories")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(p, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(p)


def plot_velocity_stream(
    adata: AnnData,
    model: Any,
    output_path: Optional[str],
    dim_reduction: str = "none",
    device: str = "cuda",
    color_key: Optional[str] = None,
):
    """Compute velocity graph and save stream plots.

    Produces:
    - Velocity_Stream_Plot.png
    - Velocity_cluster_Stream_Plot.png (if ``color_key`` is valid)
    """
    from CytoBridge.tl.downstream.velocity import compute_velocity_bundle, build_velocity_graph_bundle
    from CytoBridge.pl.downstream.velocity_plot import plot_velocity_stream_bundle

    outdir = _as_dir(output_path)
    compute_velocity_bundle(adata=adata, model=model, device=device, output_dir=str(outdir / "velocity"))
    build_velocity_graph_bundle(adata=adata, output_dir=str(outdir / "velocity_graph"), n_pcs=50, n_neighbors=30)

    base = plot_velocity_stream_bundle(
        adata=adata,
        model=model,
        output_dir=str(outdir),
        dim_reduction=dim_reduction,
        device=device,
        color_key=None,
        vkey="velocity",
    )
    src = base.get("artifacts", {}).get("velocity_stream")
    if src:
        Path(src).rename(outdir / "Velocity_Stream_Plot.png")

    if color_key and color_key in adata.obs.columns:
        col = plot_velocity_stream_bundle(
            adata=adata,
            model=model,
            output_dir=str(outdir),
            dim_reduction=dim_reduction,
            device=device,
            color_key=color_key,
            vkey="velocity",
        )
        src2 = col.get("artifacts", {}).get("velocity_stream")
        if src2:
            Path(src2).rename(outdir / "Velocity_cluster_Stream_Plot.png")

    return adata


def plot_interaction_stream(adata: AnnData, output_path: Optional[str], dim_reduction: str = "none", device: str = "cuda"):
    """Plot interaction force field as quiver on embedding."""
    outdir = _as_dir(output_path)
    _, emb = _pick_basis(adata, dim_reduction)
    force = _get_vec(adata, "interaction_force", fallback_obsm="interaction_force")
    if force is None:
        raise ValueError("interaction_force not found in adata.layers/obsm")

    vec2 = _projection_2d(force)
    p = outdir / "Interaction_Stream_Plot.png"
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(emb[:, 0], emb[:, 1], s=4, alpha=0.15, color="gray")
    step = max(1, emb.shape[0] // 2000)
    ax.quiver(emb[::step, 0], emb[::step, 1], vec2[::step, 0], vec2[::step, 1], angles="xy", scale_units="xy", scale=1.0, alpha=0.6)
    ax.set_title("Interaction force field")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(p, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(p)


def plot_score_stream(adata: AnnData, model: Any, output_path: Optional[str], dim_reduction: str = "none", device: str = "cuda"):
    """Plot score field as quiver on embedding."""
    outdir = _as_dir(output_path)
    _, emb = _pick_basis(adata, dim_reduction)
    score = _get_vec(adata, "score_latent", fallback_obsm="score")
    if score is None:
        raise ValueError("score_latent/score not found in adata.obsm")

    vec2 = _projection_2d(score)
    p = outdir / "Score_Stream_Plot.png"
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(emb[:, 0], emb[:, 1], s=4, alpha=0.15, color="gray")
    step = max(1, emb.shape[0] // 2000)
    ax.quiver(emb[::step, 0], emb[::step, 1], vec2[::step, 0], vec2[::step, 1], angles="xy", scale_units="xy", scale=1.0, alpha=0.6, color="tab:blue")
    ax.set_title("Score field")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(p, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(p)


def plot_combined_velocity_stream(adata: AnnData, model: Any, output_path: Optional[str], dim_reduction: str = "none", device: str = "cuda"):
    """Plot combined (velocity + score) vector field on embedding."""
    outdir = _as_dir(output_path)
    _, emb = _pick_basis(adata, dim_reduction)
    vel = _get_vec(adata, "velocity", fallback_obsm="velocity_latent")
    score = _get_vec(adata, "score_latent", fallback_obsm="score")
    if vel is None and score is None:
        raise ValueError("Neither velocity nor score vectors are available")

    if vel is None:
        comb = np.asarray(score)
    elif score is None:
        comb = np.asarray(vel)
    else:
        v = np.asarray(vel)
        s = np.asarray(score)
        k = min(v.shape[1], s.shape[1])
        comb = v[:, :k] + s[:, :k]

    vec2 = _projection_2d(comb)
    p = outdir / "All_Velocity_Stream.png"
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(emb[:, 0], emb[:, 1], s=4, alpha=0.15, color="gray")
    step = max(1, emb.shape[0] // 2000)
    ax.quiver(emb[::step, 0], emb[::step, 1], vec2[::step, 0], vec2[::step, 1], angles="xy", scale_units="xy", scale=1.0, alpha=0.6, color="tab:red")
    ax.set_title("Combined vector field")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(p, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(p)


def plot_interaction_potential(model: Any, d: int = 1, num_points: int = 40, output_path: Optional[str] = None, device: str = "cuda"):
    """Visualize pairwise interaction potential on a 1D grid."""
    out = _as_file_path(output_path, "figures", "interaction_potential.png")

    if not hasattr(model, "interaction_net"):
        raise ValueError("model has no interaction_net")
    net = model.interaction_net

    z = np.linspace(-1.5, 1.5, num_points)
    zz1, zz2 = np.meshgrid(z, z)
    dlt = zz1 - zz2

    import torch
    with torch.no_grad():
        x = torch.tensor(dlt.reshape(-1, 1), dtype=torch.float32, device=device)
        y = net(x).detach().cpu().numpy().reshape(num_points, num_points)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(y, cmap="viridis", origin="lower")
    ax.set_title("Interaction potential")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(out)


def plot_interaction_potential_epoch(model: Any, d: int = 1, num_points: int = 40, output_path: Optional[str] = None, device: str = "cuda"):
    """Compatibility alias for trainer debug plotting."""
    return plot_interaction_potential(model=model, d=d, num_points=num_points, output_path=output_path, device=device)


def plot_landscape(adata: AnnData, model: Any, output_path: Optional[str] = None, dim_reduction: str = "none", device: str = "cuda"):
    """Compatibility helper: currently maps to combined velocity field plot."""
    return plot_combined_velocity_stream(adata=adata, model=model, output_path=output_path, dim_reduction=dim_reduction, device=device)


def analyze_terminal_states(
    adata: AnnData,
    classified_type: str = "cell_type",
    terminal_states: Sequence[str] = ("type1", "type2"),
    output_path: Optional[str] = None,
):
    """Analyze terminal states using nearest-neighbor vote in embedding.

    Saves a JSON summary and returns the path.
    """
    out = _as_file_path(output_path, "figures", "terminal_state_summary.json")
    if classified_type not in adata.obs.columns:
        raise ValueError(f"{classified_type} not found in adata.obs")

    key = str(classified_type)
    obs = adata.obs[key].astype(str)
    term = set(str(x) for x in terminal_states)
    n_total = int(adata.n_obs)
    n_terminal = int(np.sum(obs.isin(term)))

    payload = {
        "classified_type": key,
        "terminal_states": list(term),
        "n_total": n_total,
        "n_terminal": n_terminal,
        "terminal_fraction": float(n_terminal / max(n_total, 1)),
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(out)


def visualize_trajectory_classification(
    sde_point: np.ndarray,
    predicted_labels_list: Sequence[Sequence[Any]],
    output_path: Optional[str],
    adata: Optional[AnnData] = None,
    time_points: Optional[Iterable[Any]] = None,
):
    """Visualize classified SDE trajectory points by time."""
    outdir = _as_dir(output_path)
    pts = np.asarray(sde_point)
    if pts.ndim != 3 or pts.shape[2] < 2:
        raise ValueError("sde_point should have shape [time, cells, dim>=2]")

    T = pts.shape[0]
    tp_list = list(time_points) if time_points is not None else None
    for t in range(T):
        lab = np.asarray(predicted_labels_list[t]) if t < len(predicted_labels_list) else None
        cur = pts[t, :, :2]
        fig, ax = plt.subplots(figsize=(7, 5))
        if lab is None or len(lab) != cur.shape[0]:
            ax.scatter(cur[:, 0], cur[:, 1], s=6, alpha=0.65)
        else:
            # categorical labels -> codes
            _, inv = np.unique(lab.astype(str), return_inverse=True)
            ax.scatter(cur[:, 0], cur[:, 1], c=inv, s=6, alpha=0.7, cmap="tab20")
        tlabel = tp_list[t] if tp_list is not None and t < len(tp_list) else t
        ax.set_title(f"trajectory classification t={tlabel}")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(outdir / f"trajectory_time_{t}.png", dpi=160, bbox_inches="tight")
        plt.close(fig)


# ------------------------------------------------------------------------------
# Structured downstream wrappers (compatibility layer)
# ------------------------------------------------------------------------------
def plot_velocity_stream_bundle(
    adata,
    model,
    output_dir,
    dim_reduction="none",
    device="cuda",
    color_key=None,
):
    from .downstream.velocity_plot import plot_velocity_stream_bundle as _impl

    return _impl(
        adata=adata,
        model=model,
        output_dir=output_dir,
        dim_reduction=dim_reduction,
        device=device,
        color_key=color_key,
    )


def plot_sde_trajectories_bundle(adata, output_dir, color_key=None):
    from .downstream.trajectory_plot import plot_sde_trajectories_bundle as _impl

    return _impl(adata=adata, output_dir=output_dir, color_key=color_key)


def plot_ode_trajectories_bundle(adata, output_dir, color_key=None, model=None, device="cuda", n_trajectories=50, n_bins=None):
    from .downstream.trajectory_plot import plot_ode_trajectories_bundle as _impl

    return _impl(
        adata=adata,
        output_dir=output_dir,
        color_key=color_key,
        model=model,
        device=device,
        n_trajectories=n_trajectories,
        n_bins=n_bins,
    )


__all__ = [
    "plot_growth",
    "plot_ode",
    "plot_ode_trajectories",
    "plot_sde_trajectories",
    "plot_velocity_stream",
    "plot_interaction_stream",
    "plot_score_stream",
    "plot_combined_velocity_stream",
    "plot_interaction_potential",
    "plot_interaction_potential_epoch",
    "plot_landscape",
    "analyze_terminal_states",
    "visualize_trajectory_classification",
    "plot_velocity_stream_bundle",
    "plot_ode_trajectories_bundle",
    "plot_sde_trajectories_bundle",
]
