"""GRN downstream analysis (package-native unified implementation)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from anndata import AnnData

from CytoBridge.utils import load_model_from_adata
from .contracts import LATENT_KEY, PCA_LOADINGS_KEY, TIME_KEY, ensure_contract


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
    """Return an identity latent->gene map when X_latent is the expression matrix."""
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


def _gene_loadings(adata: AnnData, latent_dim: int) -> tuple[Optional[np.ndarray], str]:
    pcs = _pca_loadings(adata, latent_dim)
    if pcs is not None:
        return pcs, "pca"
    identity = _identity_gene_loadings(adata, latent_dim)
    if identity is not None:
        return identity, "identity_no_reduction"
    return None, "none"


def _velocity_jacobian_mean(model: Any, z_np: np.ndarray, t_np: np.ndarray, device: str) -> np.ndarray:
    z = torch.tensor(z_np, dtype=torch.float32, device=device, requires_grad=True)
    t = torch.tensor(t_np, dtype=torch.float32, device=device)
    inp = torch.cat([z, t], dim=1)
    v = model.velocity_net(inp)
    dim = v.shape[1]
    jac = torch.zeros((dim, dim), dtype=torch.float32, device=device)
    for i in range(dim):
        grad = torch.autograd.grad(v[:, i].sum(), z, retain_graph=True, create_graph=False)[0]
        jac[i, :] = grad.mean(dim=0)
    return jac.detach().cpu().numpy()


def analyze_grn_bundle(
    adata: AnnData,
    output_dir: Optional[str] = None,
    max_genes: int = 10,
    max_time_points: int = 5,
    genes: Optional[Sequence[str]] = None,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Analyze GRN from velocity Jacobian with PCA-only gene projection contract."""
    warnings = []
    artifacts: Dict[str, str] = {}

    ensure_contract(adata, [LATENT_KEY, TIME_KEY], where="analyze_grn_bundle")

    runtime_device = _resolve_device(device)
    model = load_model_from_adata(adata)
    if not hasattr(model, "velocity_net"):
        return {
            "space_used": "latent",
            "projection_backend": "none",
            "artifacts": {},
            "warnings": ["model has no velocity_net"],
        }
    model = model.to(runtime_device)
    model.eval()

    x_lat = np.asarray(adata.obsm[LATENT_KEY], dtype=np.float32)
    t_vals = pd.to_numeric(adata.obs[TIME_KEY], errors="coerce")
    uniq = np.sort(np.unique(t_vals[~np.isnan(t_vals)]))[: int(max(1, max_time_points))]

    W, projection_backend = _gene_loadings(adata, x_lat.shape[1])
    if W is None:
        space_used = "latent"
        projection_backend = "none"
        warnings.append(f"{PCA_LOADINGS_KEY} missing; returning latent Jacobian submatrix")
        select_idx = np.arange(min(max_genes, x_lat.shape[1]), dtype=int)
        select_names = [f"Latent_{i}" for i in select_idx.tolist()]
    else:
        space_used = "gene"
        names = [str(x) for x in adata.var_names]
        if genes:
            idx_map = {str(n): i for i, n in enumerate(names)}
            idx = [idx_map[g] for g in genes if str(g) in idx_map]
            if len(idx) == 0:
                idx = list(range(min(max_genes, len(names))))
                warnings.append("requested genes not found; fallback to leading genes")
            select_idx = np.asarray(idx[:max_genes], dtype=int)
        else:
            if "highly_variable" in adata.var.columns and np.any(adata.var["highly_variable"].values):
                hv = np.where(adata.var["highly_variable"].values)[0]
                select_idx = np.asarray(hv[:max_genes], dtype=int)
            else:
                select_idx = np.arange(min(max_genes, adata.n_vars), dtype=int)
        select_names = [names[i] if i < len(names) else f"gene_{i}" for i in select_idx.tolist()]

    rng = np.random.default_rng(0)
    payload: Dict[str, Any] = {}
    for t in uniq:
        idx = np.where(np.isclose(t_vals.values, t))[0]
        if idx.size == 0:
            continue
        k = min(128, idx.size)
        pick = rng.choice(idx, size=k, replace=False)
        z = x_lat[pick]
        tt = np.full((k, 1), float(t), dtype=np.float32)
        jac_lat = _velocity_jacobian_mean(model, z, tt, device=runtime_device)

        if space_used == "gene" and W is not None:
            W_sel = W[select_idx, :]
            mat = W_sel @ jac_lat @ W_sel.T
        else:
            mat = jac_lat[np.ix_(select_idx, select_idx)]

        payload[f"t{float(t):.4g}"] = {
            "time_point": float(t),
            "features": select_names,
            "matrix": mat.tolist(),
        }

    outdir = _ensure_dir(output_dir)
    if outdir is not None:
        p = outdir / "grn_data.json"
        p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        artifacts["grn_data"] = str(p)

    return {
        "space_used": space_used,
        "projection_backend": projection_backend,
        "artifacts": artifacts,
        "warnings": warnings,
        "features": select_names,
        "grn": payload,
    }


def estimate_velocity_jacobian_bundle(
    adata: AnnData,
    model: Any = None,
    n_cells: int = 128,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Backwards-compatible Jacobian summary helper."""
    if model is None:
        model = load_model_from_adata(adata)
    ensure_contract(adata, [LATENT_KEY, TIME_KEY], where="estimate_velocity_jacobian_bundle")

    runtime_device = _resolve_device(device)
    model = model.to(runtime_device)
    model.eval()

    x = np.asarray(adata.obsm[LATENT_KEY], dtype=np.float32)
    t = np.asarray(pd.to_numeric(adata.obs[TIME_KEY], errors="coerce"), dtype=np.float32).reshape(-1, 1)
    idx = np.random.choice(x.shape[0], size=min(n_cells, x.shape[0]), replace=False)
    jac = _velocity_jacobian_mean(model, x[idx], t[idx], device=runtime_device)
    row_l2 = np.linalg.norm(jac, axis=1)
    return {
        "space_used": "latent",
        "projection_backend": "none",
        "artifacts": {},
        "warnings": [],
        "jacobian_row_l2_mean": row_l2.tolist(),
        "jacobian": jac.tolist(),
    }
