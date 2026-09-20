"""Velocity downstream analysis (package-native unified implementation)."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from anndata import AnnData

from CytoBridge.utils import load_model_from_adata
from .contracts import (
    LATENT_KEY,
    PCA_LOADINGS_KEY,
    TIME_KEY,
    VELOCITY_LATENT_KEY,
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


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)


def _available_cpu_count(default: int = 4) -> int:
    """Return CPU count available to current process (respecting cpuset affinity)."""
    try:
        affinity = os.sched_getaffinity(0)
        if affinity:
            return max(1, len(affinity))
    except Exception:
        pass
    return max(1, int(os.cpu_count() or default))


def _pca_loadings(adata: AnnData, latent_dim: int) -> Optional[np.ndarray]:
    pcs = adata.varm.get("PCs", None) if hasattr(adata, "varm") else None
    if pcs is None:
        return None
    pcs_np = _to_numpy(pcs)
    if pcs_np.ndim != 2 or pcs_np.shape[1] < latent_dim:
        return None
    return pcs_np[:, :latent_dim].astype(np.float32)


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


def _gene_loadings(adata: AnnData, latent_dim: int) -> Tuple[Optional[np.ndarray], str]:
    pcs = _pca_loadings(adata, latent_dim)
    if pcs is not None:
        return pcs, "pca"
    identity = _identity_gene_loadings(adata, latent_dim)
    if identity is not None:
        return identity, "identity_no_reduction"
    return None, "none"


def _sample_idx_by_time(adata: AnnData, max_cells: int, random_state: int = 0) -> np.ndarray:
    import pandas as pd

    rng = np.random.default_rng(random_state)
    n = int(adata.n_obs)
    if n <= max_cells:
        return np.arange(n, dtype=int)

    time_vals = np.asarray(pd.to_numeric(adata.obs[TIME_KEY], errors="coerce"), dtype=np.float32)
    finite_mask = np.isfinite(time_vals)
    uniq = np.unique(time_vals[finite_mask])
    if uniq.size == 0:
        return np.sort(rng.choice(n, size=max_cells, replace=False))

    per_t = max(16, int(max_cells // max(1, uniq.size)))
    picked = []
    for t in uniq:
        idx = np.where(np.isclose(time_vals, t))[0]
        if idx.size == 0:
            continue
        k = min(per_t, idx.size)
        picked.extend(rng.choice(idx, size=k, replace=False).tolist())

    if len(picked) < max_cells:
        remain = np.setdiff1d(np.arange(n, dtype=int), np.asarray(picked, dtype=int), assume_unique=False)
        extra = min(max_cells - len(picked), remain.size)
        if extra > 0:
            picked.extend(rng.choice(remain, size=extra, replace=False).tolist())

    return np.sort(np.asarray(picked[:max_cells], dtype=int))


def _normalize_basis_name(name: Optional[str]) -> Optional[str]:
    if name is None:
        return None
    basis = str(name).strip().lower()
    if basis.startswith("x_"):
        basis = basis[2:]
    if basis == "sring":
        basis = "spring"
    return basis or None


def compute_velocity_bundle(
    adata: AnnData,
    model: Optional[Any] = None,
    device: str = "cuda",
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute latent velocity and optional gene velocity.

    Contract:
    - Always writes ``adata.obsm['velocity_latent']``.
    - Writes gene velocity when PCA loadings or identity no-reduction mapping is available.
    """
    warnings = []
    artifacts: Dict[str, str] = {}

    ensure_contract(adata, [LATENT_KEY, TIME_KEY], where="compute_velocity_bundle")

    runtime_device = _resolve_device(device)
    if model is None:
        model = load_model_from_adata(adata)
    model = model.to(runtime_device)
    model.eval()

    all_times = torch.tensor(np.asarray(adata.obs[TIME_KEY])).reshape(-1, 1).float().to(runtime_device)
    all_data = torch.tensor(np.asarray(adata.obsm[LATENT_KEY], dtype=np.float32)).float().to(runtime_device)
    net_input = torch.cat([all_data, all_times], dim=1)

    with torch.no_grad():
        velocity_latent = model.velocity_net(net_input)
    velocity_latent_np = velocity_latent.detach().cpu().numpy().astype(np.float32)
    adata.obsm[VELOCITY_LATENT_KEY] = velocity_latent_np

    outdir = _ensure_dir(output_dir)
    if outdir is not None:
        p = outdir / "velocity_latent.npy"
        np.save(p, velocity_latent_np)
        artifacts["velocity_latent"] = str(p)

    space_used = "latent"
    projection_backend = "none"

    W, backend = _gene_loadings(adata, velocity_latent_np.shape[1])
    if W is not None:
        v_gene = velocity_latent_np @ W.T
        adata.layers["velocity"] = v_gene.astype(np.float32)
        space_used = "gene"
        projection_backend = backend
        if outdir is not None:
            p = outdir / "velocity_gene.npy"
            np.save(p, adata.layers["velocity"])
            artifacts["velocity_gene"] = str(p)
    else:
        warnings.append(f"{PCA_LOADINGS_KEY} unavailable or incompatible; keep latent velocity only")

    if outdir is not None:
        meta = {
            "space_used": space_used,
            "projection_backend": projection_backend,
            "n_obs": int(adata.n_obs),
            "latent_dim": int(velocity_latent_np.shape[1]),
            "warnings": warnings,
        }
        m = outdir / "velocity_summary.json"
        m.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        artifacts["velocity_summary"] = str(m)

    return {
        "space_used": space_used,
        "projection_backend": projection_backend,
        "artifacts": artifacts,
        "warnings": warnings,
    }


def _prepare_fake_for_velocity_graph(
    adata: AnnData,
    n_pcs: int = 50,
    preferred_basis: Optional[str] = None,
    strict_preferred: bool = False,
) -> Tuple[Optional[AnnData], Optional[str], str]:
    from anndata import AnnData as _AnnData

    v_lat = adata.obsm.get(VELOCITY_LATENT_KEY)
    if v_lat is None:
        return None, None, f"{VELOCITY_LATENT_KEY} missing"

    V = np.asarray(v_lat, dtype=np.float32)
    X = None
    rep_name = "latent"

    if LATENT_KEY in adata.obsm:
        X_lat = np.asarray(adata.obsm[LATENT_KEY], dtype=np.float32)
        if X_lat.shape[1] == V.shape[1]:
            X = X_lat
            rep_name = "latent"

    if X is None and "X_pca" in adata.obsm:
        X_pca = np.asarray(adata.obsm["X_pca"], dtype=np.float32)
        k = int(max(2, min(n_pcs, X_pca.shape[1], V.shape[1])))
        X = X_pca[:, :k]
        V = V[:, :k]
        rep_name = "pca"

    if X is None:
        X = V
        rep_name = "velocity"

    if X.shape[1] < 2:
        return None, None, "representation dimension < 2"

    fake = _AnnData(X=X)
    fake.obs_names = adata.obs_names.copy()
    fake.layers["Ms"] = X
    fake.layers["velocity"] = V

    pref = _normalize_basis_name(preferred_basis)
    basis_candidates = []
    if pref:
        basis_candidates.append(pref)
    for b in ("umap", "pca", "latent", "fast"):
        if b not in basis_candidates:
            basis_candidates.append(b)

    basis = None
    requested_pref = pref
    for candidate in basis_candidates:
        if candidate == "umap":
            if "X_umap" in adata.obsm and np.asarray(adata.obsm["X_umap"]).shape[1] >= 2:
                fake.obsm["X_umap"] = np.asarray(adata.obsm["X_umap"], dtype=np.float32)[:, :2]
                basis = "umap"
                break
        elif candidate == "pca":
            if "X_pca" in adata.obsm and np.asarray(adata.obsm["X_pca"]).shape[1] >= 2:
                fake.obsm["X_pca"] = np.asarray(adata.obsm["X_pca"], dtype=np.float32)[:, :2]
                basis = "pca"
                break
        elif candidate in {"latent", "fast"}:
            if "X_fast" in adata.obsm and np.asarray(adata.obsm["X_fast"]).shape[1] >= 2:
                fake.obsm["X_fast"] = np.asarray(adata.obsm["X_fast"], dtype=np.float32)[:, :2]
            elif "X_latent" in adata.obsm and np.asarray(adata.obsm["X_latent"]).shape[1] >= 2:
                fake.obsm["X_fast"] = np.asarray(adata.obsm["X_latent"], dtype=np.float32)[:, :2]
            elif "X_pca" in adata.obsm and np.asarray(adata.obsm["X_pca"]).shape[1] >= 2:
                fake.obsm["X_fast"] = np.asarray(adata.obsm["X_pca"], dtype=np.float32)[:, :2]
            else:
                fake.obsm["X_fast"] = np.asarray(X, dtype=np.float32)[:, :2]
            basis = "fast"
            break
        else:
            custom_key = f"X_{candidate}"
            custom = adata.obsm.get(custom_key)
            if custom is not None and np.asarray(custom).shape[1] >= 2:
                fake.obsm[custom_key] = np.asarray(custom, dtype=np.float32)[:, :2]
                basis = candidate
                break

    if basis is None:
        if strict_preferred and requested_pref:
            available = []
            for k in adata.obsm.keys():
                key = str(k)
                if not key.startswith("X_"):
                    continue
                arr = np.asarray(adata.obsm[key])
                if arr.ndim == 2 and arr.shape[1] >= 2:
                    available.append(key[2:])
            available = sorted(set(available))
            return None, None, f"preferred basis '{requested_pref}' unavailable; available={available or ['(none)']}"
        fake.obsm["X_fast"] = np.asarray(X, dtype=np.float32)[:, :2]
        basis = "fast"

    return fake, basis, rep_name


def build_velocity_graph_bundle(
    adata: AnnData,
    output_dir: Optional[str] = None,
    n_pcs: int = 50,
    n_neighbors: int = 30,
    n_jobs: Optional[int] = None,
    reuse_neighbors: bool = True,
    vkey: str = "velocity",
    preferred_basis: Optional[str] = None,
    strict_preferred: bool = False,
) -> Dict[str, Any]:
    """Build scVelo velocity graph in lightweight space and sync results back to ``adata``."""
    warnings = []
    artifacts: Dict[str, str] = {}

    try:
        import scvelo as scv
    except Exception as e:
        return {
            "space_used": "latent",
            "projection_backend": "none",
            "artifacts": {},
            "warnings": [f"scvelo unavailable: {e}"],
        }

    ensure_contract(adata, [LATENT_KEY, TIME_KEY, VELOCITY_LATENT_KEY], where="build_velocity_graph_bundle")
    fake, basis, rep_name = _prepare_fake_for_velocity_graph(
        adata,
        n_pcs=n_pcs,
        preferred_basis=preferred_basis,
        strict_preferred=bool(strict_preferred),
    )
    if fake is None or basis is None:
        return {
            "space_used": "latent",
            "projection_backend": "none",
            "artifacts": {},
            "warnings": [rep_name or "failed to prepare fake adata"],
        }

    k = int(max(5, min(n_neighbors, fake.n_obs - 1)))
    copied_neighbors = False
    if reuse_neighbors:
        try:
            conn = adata.obsp.get("connectivities") if hasattr(adata, "obsp") else None
            dist = adata.obsp.get("distances") if hasattr(adata, "obsp") else None
            if "neighbors" in adata.uns and conn is not None and dist is not None:
                if conn.shape[0] == fake.n_obs and dist.shape[0] == fake.n_obs:
                    fake.uns["neighbors"] = dict(adata.uns["neighbors"])
                    fake.obsp["connectivities"] = conn.copy()
                    fake.obsp["distances"] = dist.copy()
                    copied_neighbors = True
        except Exception:
            copied_neighbors = False

    if not copied_neighbors:
        try:
            scv.pp.neighbors(fake, n_neighbors=k, use_rep="X")
        except Exception as e:
            return {
                "space_used": "latent",
                "projection_backend": "none",
                "artifacts": {},
                "warnings": [f"neighbors failed: {e}"],
            }

    if hasattr(fake, "obsp") and "distances" in fake.obsp:
        try:
            dist = fake.obsp["distances"].tocsr(copy=True)
            if dist.nnz > 0:
                mask = dist.data <= 0
                if np.any(mask):
                    dist.data[mask] = 1e-12
                    fake.obsp["distances"] = dist
        except Exception:
            pass

    if n_jobs is None:
        n_jobs = _available_cpu_count()

    try:
        scv.tl.velocity_graph(fake, vkey=vkey, n_jobs=int(n_jobs))
    except Exception as e:
        # Retry once with fresh neighbors when reused graph is incompatible.
        try:
            fake.uns.pop("neighbors", None)
            if hasattr(fake, "obsp"):
                fake.obsp.pop("connectivities", None)
                fake.obsp.pop("distances", None)
            scv.pp.neighbors(fake, n_neighbors=k, use_rep="X")
            scv.tl.velocity_graph(fake, vkey=vkey, n_jobs=int(n_jobs))
            warnings.append(f"velocity graph retry after neighbor rebuild: {e}")
        except Exception as e2:
            return {
                "space_used": "latent",
                "projection_backend": "none",
                "artifacts": {},
                "warnings": [f"velocity graph failed: {e2}"],
            }

    try:
        scv.tl.velocity_embedding(fake, basis=basis, vkey=vkey)
    except Exception as e:
        return {
            "space_used": "latent",
            "projection_backend": "none",
            "artifacts": {},
            "warnings": [f"velocity embedding failed: {e}"],
        }

    for key in (f"{vkey}_graph", f"{vkey}_graph_neg", f"{vkey}_params"):
        if key in fake.uns:
            adata.uns[key] = fake.uns[key]

    # Sync both the embedding vector field and its coordinate basis back to
    # the original adata so plotting can always find a matching X_{basis}.
    emb_key = f"{vkey}_{basis}"
    basis_key = f"X_{basis}"
    if basis_key in fake.obsm:
        adata.obsm[basis_key] = fake.obsm[basis_key]
    if emb_key in fake.obsm:
        adata.obsm[emb_key] = fake.obsm[emb_key]

    outdir = _ensure_dir(output_dir)
    if outdir is not None:
        meta = {
            "basis": basis,
            "representation": rep_name,
            "n_obs": int(adata.n_obs),
            "n_neighbors": int(k),
        }
        p = outdir / "velocity_graph_summary.json"
        p.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        artifacts["velocity_graph_summary"] = str(p)

    return {
        "space_used": "latent",
        "projection_backend": "none",
        "artifacts": artifacts,
        "warnings": warnings,
        "basis": basis,
        "representation": rep_name,
    }


def summarize_velocity_drivers_bundle(
    adata: AnnData,
    output_dir: Optional[str] = None,
    analysis_space: str = "gene",
    top_n: int = 20,
) -> Dict[str, Any]:
    """Summarize velocity drivers in latent or PCA-projected gene space."""
    warnings = []
    artifacts: Dict[str, str] = {}

    ensure_contract(adata, [LATENT_KEY, TIME_KEY, VELOCITY_LATENT_KEY], where="summarize_velocity_drivers_bundle")
    v_lat = adata.obsm.get(VELOCITY_LATENT_KEY)
    if v_lat is None:
        raise KeyError(f"{VELOCITY_LATENT_KEY} missing")

    V = np.asarray(v_lat, dtype=np.float32)
    mode = str(analysis_space or "gene").strip().lower()
    if mode not in {"latent", "gene", "auto"}:
        mode = "gene"

    use_gene = mode == "gene"
    if mode == "auto":
        use_gene = _gene_loadings(adata, V.shape[1])[0] is not None

    W, backend = _gene_loadings(adata, V.shape[1]) if use_gene else (None, "none")
    if use_gene and W is None:
        warnings.append("PCA loadings missing; fallback to latent")

    if use_gene and W is not None:
        n_obs = int(V.shape[0])
        n_genes = int(W.shape[0])
        target_bytes = 128 * 1024 * 1024  # ~128 MB projected chunk
        bs = max(256, int(target_bytes // max(4, 4 * n_genes)))
        if bs < n_obs:
            warnings.append(f"streamed gene projection enabled (batch_size={bs})")

        sum_v = np.zeros((n_genes,), dtype=np.float64)
        sum_abs = np.zeros((n_genes,), dtype=np.float64)
        for s in range(0, n_obs, bs):
            e = min(s + bs, n_obs)
            proj = V[s:e] @ W.T
            sum_v += proj.sum(axis=0, dtype=np.float64)
            sum_abs += np.abs(proj).sum(axis=0, dtype=np.float64)

        mean_v = (sum_v / float(n_obs)).astype(np.float32, copy=False)
        mean_abs = (sum_abs / float(n_obs)).astype(np.float32, copy=False)
        names = [str(x) for x in adata.var_names]
        space_used = "gene"
        projection_backend = backend
    else:
        X = V
        names = [f"Latent_{i}" for i in range(X.shape[1])]
        space_used = "latent"
        projection_backend = "none"
        mean_v = np.mean(X, axis=0)
        mean_abs = np.mean(np.abs(X), axis=0)
    order = np.argsort(mean_abs)[::-1]

    rows = []
    for rank, idx in enumerate(order.tolist(), start=1):
        rows.append(
            {
                "rank": rank,
                "feature_index": int(idx),
                "feature_name": names[idx] if idx < len(names) else f"f_{idx}",
                "mean_velocity": float(mean_v[idx]),
                "mean_abs_velocity": float(mean_abs[idx]),
            }
        )

    outdir = _ensure_dir(output_dir)
    if outdir is not None:
        import pandas as pd

        p = outdir / "velocity_driver_scores.csv"
        pd.DataFrame(rows).to_csv(p, index=False)
        artifacts["velocity_driver_scores"] = str(p)

    return {
        "space_used": space_used,
        "projection_backend": projection_backend,
        "artifacts": artifacts,
        "warnings": warnings,
        "top": rows[: max(1, int(top_n))],
        "all": rows,
    }


def summarize_velocity_jacobian_drivers_bundle(
    adata: AnnData,
    model: Optional[Any] = None,
    output_dir: Optional[str] = None,
    analysis_space: str = "gene",
    top_n: int = 20,
    n_cells: int = 512,
    random_state: int = 0,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Summarize velocity drivers with Jacobian sensitivity as primary score.

    Scoring:
    - latent mode: ``jacobian_sensitivity = ||J_i,:||_2``
    - gene mode (PCA available): ``jacobian_sensitivity = ||W_g @ J||_2``

    Auxiliary explanation field:
    - ``mean_abs_velocity`` in the same analysis space.
    """
    warnings = []
    artifacts: Dict[str, str] = {}

    ensure_contract(adata, [LATENT_KEY, TIME_KEY], where="summarize_velocity_jacobian_drivers_bundle")

    runtime_device = _resolve_device(device)
    if model is None:
        model = load_model_from_adata(adata)
    if not hasattr(model, "velocity_net"):
        return {
            "space_used": "latent",
            "projection_backend": "none",
            "artifacts": {},
            "warnings": ["model has no velocity_net"],
            "top": [],
            "all": [],
        }
    model = model.to(runtime_device)
    model.eval()

    import pandas as pd

    x = np.asarray(adata.obsm[LATENT_KEY], dtype=np.float32)
    t_vals = np.asarray(pd.to_numeric(adata.obs[TIME_KEY], errors="coerce"), dtype=np.float32)
    if not np.all(np.isfinite(t_vals)):
        finite = t_vals[np.isfinite(t_vals)]
        fill = float(np.median(finite)) if finite.size > 0 else 0.0
        t_vals = np.where(np.isfinite(t_vals), t_vals, fill).astype(np.float32, copy=False)
        warnings.append("non-numeric time values coerced for Jacobian sampling")
    t = t_vals.reshape(-1, 1)
    sel = _sample_idx_by_time(adata, max_cells=max(32, int(n_cells)), random_state=int(random_state))
    x_sel = x[sel]
    t_sel = t[sel]

    z = torch.tensor(x_sel, dtype=torch.float32, device=runtime_device, requires_grad=True)
    tt = torch.tensor(t_sel, dtype=torch.float32, device=runtime_device)
    inp = torch.cat([z, tt], dim=1)
    vel = model.velocity_net(inp)
    vel_np = vel.detach().cpu().numpy().astype(np.float32, copy=False)

    latent_dim = int(vel.shape[1])
    jac = torch.zeros((latent_dim, latent_dim), dtype=torch.float32, device=runtime_device)
    for i in range(latent_dim):
        grad = torch.autograd.grad(
            vel[:, i].sum(),
            z,
            retain_graph=(i < latent_dim - 1),
            create_graph=False,
        )[0]
        jac[i, :] = grad.mean(dim=0)
    jac_np = jac.detach().cpu().numpy().astype(np.float32, copy=False)

    mode = str(analysis_space or "gene").strip().lower()
    if mode not in {"latent", "gene", "auto"}:
        mode = "gene"

    use_gene = mode == "gene"
    if mode == "auto":
        use_gene = _gene_loadings(adata, latent_dim)[0] is not None

    W, backend = _gene_loadings(adata, latent_dim) if use_gene else (None, "none")
    if use_gene and W is None:
        warnings.append(f"{PCA_LOADINGS_KEY} missing; fallback to latent Jacobian ranking")

    if use_gene and W is not None:
        jacobian_gene = W @ jac_np  # (n_genes, latent_dim)
        sensitivity = np.linalg.norm(jacobian_gene, axis=1).astype(np.float32, copy=False)
        v_gene = vel_np @ W.T  # (n_cells, n_genes)
        mean_abs_velocity = np.mean(np.abs(v_gene), axis=0).astype(np.float32, copy=False)
        names = [str(x) for x in adata.var_names]
        space_used = "gene"
        projection_backend = backend
    else:
        sensitivity = np.linalg.norm(jac_np, axis=1).astype(np.float32, copy=False)
        mean_abs_velocity = np.mean(np.abs(vel_np), axis=0).astype(np.float32, copy=False)
        names = [f"Latent_{i}" for i in range(latent_dim)]
        space_used = "latent"
        projection_backend = "none"

    order = np.argsort(sensitivity)[::-1]
    rows = []
    for rank, idx in enumerate(order.tolist(), start=1):
        rows.append(
            {
                "rank": rank,
                "feature_index": int(idx),
                "feature_name": names[idx] if idx < len(names) else f"f_{idx}",
                "jacobian_sensitivity": float(sensitivity[idx]),
                "driver_score": float(sensitivity[idx]),
                "mean_abs_velocity": float(mean_abs_velocity[idx]),
            }
        )

    outdir = _ensure_dir(output_dir)
    if outdir is not None:
        import pandas as pd

        scores_path = outdir / "velocity_jacobian_driver_scores.csv"
        pd.DataFrame(rows).to_csv(scores_path, index=False)
        artifacts["velocity_jacobian_driver_scores"] = str(scores_path)

        jac_path = outdir / "velocity_jacobian_mean.npy"
        np.save(jac_path, jac_np.astype(np.float32, copy=False))
        artifacts["velocity_jacobian_mean"] = str(jac_path)

        idx_path = outdir / "velocity_jacobian_sampled_cells.csv"
        pd.DataFrame(
            {
                "sample_order": np.arange(sel.size, dtype=int),
                "obs_position_index": sel,
                "obs_name": adata.obs_names.values[sel],
            }
        ).to_csv(idx_path, index=False)
        artifacts["sampled_cells"] = str(idx_path)

        meta = {
            "space_used": space_used,
            "projection_backend": projection_backend,
            "n_cells_sampled": int(sel.size),
            "latent_dim": int(latent_dim),
            "top_n": int(max(1, top_n)),
            "warnings": warnings,
        }
        meta_path = outdir / "velocity_jacobian_driver_summary.json"
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        artifacts["velocity_jacobian_driver_summary"] = str(meta_path)

    return {
        "space_used": space_used,
        "projection_backend": projection_backend,
        "artifacts": artifacts,
        "warnings": warnings,
        "n_cells_sampled": int(sel.size),
        "top": rows[: max(1, int(top_n))],
        "all": rows,
    }
