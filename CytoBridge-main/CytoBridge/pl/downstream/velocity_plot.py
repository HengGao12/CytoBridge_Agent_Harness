"""Velocity plotting helpers (package-native)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _ensure_dir(output_dir: str) -> Path:
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    return out


def _basis_candidates(dim_reduction: str) -> List[str]:
    out: List[str] = []
    if dim_reduction and dim_reduction != "none":
        out.append(str(dim_reduction))
    for b in ["umap", "pca", "fast"]:
        if b not in out:
            out.append(b)
    return out


def _maybe_synthesize_fast(adata, vkey: str, warnings: List[str]) -> None:
    """Synthesize X_fast when velocity_fast exists but coordinates are missing."""
    if f"{vkey}_fast" not in adata.obsm or "X_fast" in adata.obsm:
        return

    if "X_latent" in adata.obsm and np.asarray(adata.obsm["X_latent"]).shape[1] >= 2:
        adata.obsm["X_fast"] = np.asarray(adata.obsm["X_latent"], dtype=np.float32)[:, :2]
    elif "X_pca" in adata.obsm and np.asarray(adata.obsm["X_pca"]).shape[1] >= 2:
        adata.obsm["X_fast"] = np.asarray(adata.obsm["X_pca"], dtype=np.float32)[:, :2]
    else:
        v_fast = np.asarray(adata.obsm[f"{vkey}_fast"], dtype=np.float32)
        if v_fast.ndim == 2 and v_fast.shape[1] >= 2:
            adata.obsm["X_fast"] = v_fast[:, :2]

    if "X_fast" in adata.obsm:
        warnings.append("X_fast missing; synthesized fallback coordinates.")


def _find_stream_basis(adata, vkey: str, candidates: List[str]) -> Optional[str]:
    for b in candidates:
        if f"X_{b}" in adata.obsm and f"{vkey}_{b}" in adata.obsm:
            return b
    return None


def _resolve_fallback_fields(
    adata,
    vkey: str,
    preferred_basis: Optional[str],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[str]]:
    basis_order: List[str] = []
    if preferred_basis:
        basis_order.append(preferred_basis)
    for b in ["umap", "pca", "fast", "latent"]:
        if b not in basis_order:
            basis_order.append(b)

    for b in basis_order:
        emb_key = f"X_{b}" if b != "latent" else "X_latent"
        vec_key = f"{vkey}_{b}" if b != "latent" else f"{vkey}_latent"
        emb = adata.obsm.get(emb_key)
        vec = adata.obsm.get(vec_key)
        if emb is not None and np.asarray(emb).ndim == 2 and np.asarray(emb).shape[1] >= 2:
            emb_np = np.asarray(emb, dtype=np.float32)[:, :2]
            if vec is not None and np.asarray(vec).ndim == 2 and np.asarray(vec).shape[1] >= 2:
                vec_np = np.asarray(vec, dtype=np.float32)[:, :2]
                if vec_np.shape[0] == emb_np.shape[0]:
                    return emb_np, vec_np, b
            return emb_np, None, b
    return None, None, None


def _plot_fallback_velocity_view(
    adata,
    output_dir: Path,
    color_key: Optional[str],
    vkey: str,
    preferred_basis: Optional[str],
) -> Dict[str, Any]:
    warnings: List[str] = []
    artifacts: Dict[str, str] = {}
    emb, vec, basis = _resolve_fallback_fields(adata, vkey=vkey, preferred_basis=preferred_basis)
    if emb is None:
        return {
            "artifacts": {},
            "warnings": ["fallback plotting unavailable: no 2D embedding found"],
            "plot_mode": "none",
            "basis": None,
        }

    c = None
    if color_key:
        if color_key in adata.obs.columns:
            col = adata.obs[color_key]
            try:
                c = np.asarray(col, dtype=float)
            except Exception:
                c = np.asarray(col.astype("category").cat.codes, dtype=float)
                warnings.append(f"color_key '{color_key}' is non-numeric; using category codes.")
        else:
            warnings.append(f"color_key '{color_key}' not found; fallback to default color.")

    fig, ax = plt.subplots(figsize=(10, 8))
    if c is None:
        ax.scatter(emb[:, 0], emb[:, 1], s=6, alpha=0.35, color="gray")
    else:
        ax.scatter(emb[:, 0], emb[:, 1], c=c, s=6, alpha=0.35, cmap="tab20")

    if vec is not None:
        step = max(1, emb.shape[0] // 2000)
        ax.quiver(
            emb[::step, 0],
            emb[::step, 1],
            vec[::step, 0],
            vec[::step, 1],
            angles="xy",
            scale_units="xy",
            scale=1.0,
            alpha=0.6,
            width=0.002,
        )
        ax.set_title(f"Velocity fallback quiver ({basis})")
        plot_mode = "fallback_quiver"
        out = output_dir / f"velocity_fallback_quiver_{basis}.png"
    else:
        ax.set_title(f"Velocity fallback trajectory view ({basis})")
        plot_mode = "fallback_trajectory"
        out = output_dir / f"velocity_fallback_trajectory_{basis}.png"

    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    artifacts["velocity_fallback"] = str(out)
    return {
        "artifacts": artifacts,
        "warnings": warnings,
        "plot_mode": plot_mode,
        "basis": basis,
    }


def plot_velocity_stream_bundle(
    adata,
    model=None,
    output_dir: str = ".",
    dim_reduction: str = "none",
    device: str = "cuda",
    color_key: Optional[str] = None,
    vkey: str = "velocity",
) -> Dict[str, Any]:
    """Plot velocity stream from prepared velocity graph/embeddings.

    Hard requirements:
    - scvelo is available
    - velocity graph exists in ``adata.uns``
    - at least one matching pair ``X_{basis}`` + ``{vkey}_{basis}``

    This function is intentionally strict: any unmet requirement or stream
    plotting failure raises immediately instead of degrading to fallback plots.
    """
    artifacts: Dict[str, str] = {}
    warnings: List[str] = []
    outdir = _ensure_dir(output_dir)
    plot_mode = "stream"

    candidates = _basis_candidates(dim_reduction=dim_reduction)
    basis = _find_stream_basis(adata=adata, vkey=vkey, candidates=candidates)

    try:
        import scvelo as scv
    except Exception as e:
        raise RuntimeError(f"scvelo unavailable: {e}") from e

    missing_requirements: List[str] = []
    if f"{vkey}_graph" not in adata.uns:
        missing_requirements.append(f"{vkey}_graph_missing")
    if basis is None:
        missing_requirements.append("matching_embedding_velocity_pair_missing")

    if missing_requirements:
        raise RuntimeError(
            "velocity stream requirements not satisfied: "
            + ", ".join(missing_requirements)
        )

    fig, ax = plt.subplots(figsize=(10, 8))
    color = color_key if (color_key and color_key in adata.obs.columns) else None
    if color_key and color is None:
        warnings.append(f"color_key '{color_key}' not found, fallback to default")

    try:
        scv.pl.velocity_embedding_stream(
            adata,
            basis=basis,
            vkey=vkey,
            color=color,
            ax=ax,
            show=False,
        )
        p = outdir / f"velocity_stream_{basis}.png"
        fig.savefig(p, dpi=180, bbox_inches="tight")
        artifacts["velocity_stream"] = str(p)
        plt.close(fig)
    except Exception as e:
        plt.close(fig)
        raise RuntimeError(f"velocity stream plotting failed on basis '{basis}': {e}") from e

    return {
        "space_used": "latent",
        "projection_backend": "none",
        "artifacts": artifacts,
        "warnings": warnings,
        "basis": basis,
        "missing_requirements": [],
        "fallback_used": False,
        "plot_mode": plot_mode,
    }
