"""Growth downstream analysis (package-native unified implementation)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from anndata import AnnData

from CytoBridge.utils import load_model_from_adata
from .contracts import (
    GROWTH_KEY,
    LATENT_KEY,
    PCA_LOADINGS_KEY,
    TIME_KEY,
    ensure_contract,
)


def _resolve_device(device: str) -> str:
    d = str(device or "cpu")
    if d.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return d


def _ensure_dir(path: Optional[str]) -> Optional[Path]:
    if not path:
        return None
    out = Path(path).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    return out


def _growth_vec(adata: AnnData, key: str = "growth_rate") -> Optional[np.ndarray]:
    if key in adata.obsm:
        return np.asarray(adata.obsm[key], dtype=np.float32).reshape(-1)
    if key in adata.obs.columns:
        return np.asarray(pd.to_numeric(adata.obs[key], errors="coerce"), dtype=np.float32)
    return None


def _pca_loadings(adata: AnnData, latent_dim: int) -> Optional[np.ndarray]:
    pcs = adata.varm.get("PCs", None) if hasattr(adata, "varm") else None
    if pcs is None:
        return None
    pcs_np = np.asarray(pcs, dtype=np.float32)
    if pcs_np.ndim != 2 or pcs_np.shape[1] < latent_dim:
        return None
    return pcs_np[:, :latent_dim]


def _to_dense_np(x: Any) -> np.ndarray:
    if hasattr(x, "toarray"):
        return np.asarray(x.toarray())
    return np.asarray(x)


def _identity_gene_loadings(adata: AnnData, latent_dim: int) -> Optional[np.ndarray]:
    """Return an identity latent->gene map when no dimensionality reduction was used."""
    if int(latent_dim) != int(adata.n_vars):
        return None
    x_lat = adata.obsm.get(LATENT_KEY)
    if x_lat is None:
        return None
    x_lat_np = np.asarray(x_lat)
    if x_lat_np.ndim != 2 or x_lat_np.shape[1] != int(adata.n_vars):
        return None
    n = min(64, int(adata.n_obs))
    if n <= 0:
        return None
    idx = np.linspace(0, int(adata.n_obs) - 1, n, dtype=int)
    x_np = _to_dense_np(adata.X[idx])
    if x_np.shape != x_lat_np[idx].shape:
        return None
    if not np.allclose(x_np, x_lat_np[idx], rtol=1e-4, atol=1e-5, equal_nan=False):
        return None
    return np.eye(int(adata.n_vars), dtype=np.float32)


def _gene_loadings(adata: AnnData, latent_dim: int) -> Tuple[Optional[np.ndarray], str]:
    pcs = _pca_loadings(adata, latent_dim)
    if pcs is not None:
        return pcs, "pca"
    identity = _identity_gene_loadings(adata, latent_dim)
    if identity is not None:
        return identity, "identity_no_reduction"
    return None, "none"


def summarize_growth_bundle(
    adata: AnnData,
    key: str = "growth_rate",
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    warnings = []
    artifacts: Dict[str, str] = {}

    ensure_contract(adata, [LATENT_KEY, TIME_KEY], where="summarize_growth_bundle")
    growth = _growth_vec(adata, key=key)
    if growth is None:
        warnings.append(f"missing growth key: {key}")

    time_vals = pd.to_numeric(adata.obs[TIME_KEY], errors="coerce")
    counts = pd.Series(time_vals[~time_vals.isna()]).value_counts().sort_index()

    stats = {
        "time_key": TIME_KEY,
        "time_points": [float(x) for x in counts.index.tolist()],
        "cell_counts": [int(x) for x in counts.values.tolist()],
        "mass_cv": float(np.std(counts.values) / max(np.mean(counts.values), 1e-8)) if len(counts) > 1 else 0.0,
    }

    if growth is not None:
        g = growth[np.isfinite(growth)]
        if g.size > 0:
            stats.update(
                {
                    "growth_mean": float(np.mean(g)),
                    "growth_std": float(np.std(g)),
                    "growth_min": float(np.min(g)),
                    "growth_max": float(np.max(g)),
                    "growth_positive": int(np.sum(g > 0)),
                    "growth_negative": int(np.sum(g < 0)),
                    "growth_n": int(g.size),
                }
            )
        else:
            warnings.append("growth values are all non-finite")

    outdir = _ensure_dir(output_dir)
    if outdir is not None:
        p = outdir / "growth_summary.json"
        p.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
        artifacts["growth_summary"] = str(p)

    return {
        "space_used": "latent",
        "projection_backend": "none",
        "artifacts": artifacts,
        "warnings": warnings,
        "stats": stats,
    }


def _sample_idx_by_time(adata: AnnData, max_cells: int, random_state: int = 0) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    n = adata.n_obs
    if n <= max_cells:
        return np.arange(n, dtype=int)

    ensure_contract(adata, [TIME_KEY], where="_sample_idx_by_time")
    time_vals = pd.to_numeric(adata.obs[TIME_KEY], errors="coerce")
    uniq = np.unique(time_vals[~np.isnan(time_vals)])
    if uniq.size == 0:
        return np.sort(rng.choice(n, size=max_cells, replace=False))

    per_t = max(20, int(max_cells // max(1, uniq.size)))
    pick = []
    for t in uniq:
        idx = np.where(np.isclose(time_vals.values, t))[0]
        if idx.size == 0:
            continue
        k = min(per_t, idx.size)
        pick.extend(rng.choice(idx, size=k, replace=False).tolist())

    if len(pick) < max_cells:
        remain = np.setdiff1d(np.arange(n, dtype=int), np.asarray(pick, dtype=int), assume_unique=False)
        extra = min(max_cells - len(pick), remain.size)
        if extra > 0:
            pick.extend(rng.choice(remain, size=extra, replace=False).tolist())

    return np.sort(np.asarray(pick[:max_cells], dtype=int))


def summarize_growth_drivers_bundle(
    adata: AnnData,
    output_dir: Optional[str] = None,
    analysis_space: str = "auto",
    top_n: int = 20,
    max_cells: int = 20000,
    batch_size: int = 2048,
    random_state: int = 0,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Compute growth drivers via growth-net sensitivity to latent state.

    Scores use ``abs(mean(gradient))`` rather than ``mean(abs(gradient))``.
    This emphasizes genes with a consistent net growth effect across sampled
    cells instead of genes with large but canceling local sensitivities.

    Projection rule:
    - PCA available: return gene-space scores.
    - Otherwise: return latent-space scores.

    ``analysis_space`` matches the velocity driver API:
    - ``"gene"``: require gene-space projection, falling back to latent with a warning;
    - ``"latent"``: return latent driver scores directly;
    - ``"auto"``: use gene projection when available, otherwise latent.
    """
    warnings = []
    artifacts: Dict[str, str] = {}

    ensure_contract(adata, [LATENT_KEY, TIME_KEY], where="summarize_growth_drivers_bundle")

    runtime_device = _resolve_device(device)
    model = load_model_from_adata(adata)
    if not hasattr(model, "growth_net"):
        return {
            "space_used": "latent",
            "projection_backend": "none",
            "artifacts": {},
            "warnings": ["model has no growth_net"],
        }

    model = model.to(runtime_device)
    model.eval()

    x = np.asarray(adata.obsm[LATENT_KEY], dtype=np.float32)
    t = np.asarray(pd.to_numeric(adata.obs[TIME_KEY], errors="coerce"), dtype=np.float32).reshape(-1, 1)
    sel = _sample_idx_by_time(adata, max_cells=max_cells, random_state=random_state)
    x_sel = x[sel]
    t_sel = t[sel]

    mode = str(analysis_space or "auto").strip().lower()
    if mode not in {"latent", "gene", "auto"}:
        warnings.append(f"invalid analysis_space={analysis_space!r}; using auto")
        mode = "auto"

    use_gene_projection = mode == "gene"
    if mode == "auto":
        use_gene_projection = _gene_loadings(adata, x.shape[1])[0] is not None

    w_pca, projection_backend = _gene_loadings(adata, x.shape[1]) if use_gene_projection else (None, "none")
    if mode == "gene" and w_pca is None:
        warnings.append(f"{PCA_LOADINGS_KEY} missing; fallback to latent growth drivers")
        use_gene_projection = False

    # For gene projection, cap batch size by projected chunk memory to avoid OOM.
    bs = int(max(1, batch_size))
    if use_gene_projection:
        n_genes = int(w_pca.shape[0])
        target_bytes = 128 * 1024 * 1024  # ~128 MB for chunk projection buffer
        bs_cap = max(32, int(target_bytes // max(4, 4 * n_genes)))
        if bs > bs_cap:
            bs = bs_cap
            warnings.append(f"batch_size capped to {bs} for memory-safe gene projection")

    latent_grad_sum = np.zeros((x.shape[1],), dtype=np.float64)
    gene_grad_sum = np.zeros((int(w_pca.shape[0]),), dtype=np.float64) if use_gene_projection else None
    total_rows = 0
    for s in range(0, x_sel.shape[0], bs):
        e = min(s + bs, x_sel.shape[0])
        z = torch.tensor(x_sel[s:e], dtype=torch.float32, device=runtime_device, requires_grad=True)
        tt = torch.tensor(t_sel[s:e], dtype=torch.float32, device=runtime_device)
        inp = torch.cat([z, tt], dim=1)
        g = model.growth_net(inp).reshape(-1)
        grad = torch.autograd.grad(g.sum(), z, retain_graph=False, create_graph=False)[0]
        grad_np = grad.detach().cpu().numpy().astype(np.float32, copy=False)
        latent_grad_sum += grad_np.sum(axis=0, dtype=np.float64)
        if use_gene_projection:
            # Streamed projection: avoid constructing full (n_cells x n_genes) dense matrix.
            proj = grad_np @ w_pca.T
            gene_grad_sum += proj.sum(axis=0, dtype=np.float64)
        total_rows += grad_np.shape[0]

    if total_rows <= 0:
        raise ValueError("No sampled cells available for growth driver scoring")

    latent_scores = np.abs(latent_grad_sum / float(total_rows)).astype(np.float32, copy=False)

    if use_gene_projection and gene_grad_sum is not None:
        gene_scores = np.abs(gene_grad_sum / float(total_rows)).astype(np.float32, copy=False)
        names = [str(x) for x in adata.var_names]
        scores = gene_scores
        space_used = "gene"
    else:
        names = [f"Latent_{i}" for i in range(latent_scores.shape[0])]
        scores = latent_scores
        space_used = "latent"
        projection_backend = "none"
        if mode == "auto":
            warnings.append(f"{PCA_LOADINGS_KEY} missing; returned latent growth drivers")

    order = np.argsort(scores)[::-1]
    rows = []
    for rank, idx in enumerate(order.tolist(), start=1):
        rows.append(
            {
                "rank": rank,
                "feature_index": int(idx),
                "feature_name": names[idx] if idx < len(names) else f"f_{idx}",
                "driver_score": float(scores[idx]),
            }
        )

    outdir = _ensure_dir(output_dir)
    if outdir is not None:
        csv_path = outdir / "growth_driver_scores.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        artifacts["growth_driver_scores"] = str(csv_path)
        idx_path = outdir / "growth_driver_sampled_cells.csv"
        pd.DataFrame(
            {
                "sample_order": np.arange(sel.size, dtype=int),
                "obs_position_index": sel,
                "obs_name": adata.obs_names.values[sel],
            }
        ).to_csv(idx_path, index=False)
        artifacts["sampled_cells"] = str(idx_path)

    return {
        "space_used": space_used,
        "projection_backend": projection_backend,
        "artifacts": artifacts,
        "warnings": warnings,
        "top": rows[: max(1, int(top_n))],
        "all": rows,
    }


def summarize_growth_jacobian_drivers_bundle(
    adata: AnnData,
    output_dir: Optional[str] = None,
    top_n: int = 20,
    max_cells: int = 20000,
    batch_size: int = 2048,
    random_state: int = 0,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Summarize growth drivers with Jacobian sensitivity as primary score.

    For scalar growth output, Jacobian sensitivity corresponds to the absolute
    gradient magnitude of growth with respect to latent/gene features.
    """
    base = summarize_growth_drivers_bundle(
        adata=adata,
        output_dir=output_dir,
        top_n=top_n,
        max_cells=max_cells,
        batch_size=batch_size,
        random_state=random_state,
        device=device,
    )

    warnings = list(base.get("warnings", []))
    artifacts: Dict[str, str] = dict(base.get("artifacts", {}))
    rows = []
    for row in base.get("all", []):
        score = float(row.get("driver_score", 0.0))
        rows.append(
            {
                "rank": int(row.get("rank", len(rows) + 1)),
                "feature_index": int(row.get("feature_index", -1)),
                "feature_name": str(row.get("feature_name", "unknown")),
                "jacobian_sensitivity": score,
                "driver_score": score,
                "auxiliary_metric": score,
                "auxiliary_metric_name": "abs_mean_growth_gradient",
            }
        )

    outdir = _ensure_dir(output_dir)
    if outdir is not None and rows:
        csv_path = outdir / "growth_jacobian_driver_scores.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        artifacts["growth_jacobian_driver_scores"] = str(csv_path)

        meta = {
            "space_used": base.get("space_used", "latent"),
            "projection_backend": base.get("projection_backend", "none"),
            "top_n": int(max(1, top_n)),
            "warnings": warnings,
        }
        meta_path = outdir / "growth_jacobian_driver_summary.json"
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        artifacts["growth_jacobian_driver_summary"] = str(meta_path)

    return {
        "space_used": base.get("space_used", "latent"),
        "projection_backend": base.get("projection_backend", "none"),
        "artifacts": artifacts,
        "warnings": warnings,
        "top": rows[: max(1, int(top_n))],
        "all": rows,
    }
