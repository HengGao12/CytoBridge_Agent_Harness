"""Strict downstream data contract for CytoBridge.

Downstream analysis assumes preprocessing/training have already normalized data
into a fixed schema. We intentionally keep this contract strict to avoid
per-tool ambiguity and branching.
"""
from __future__ import annotations

from typing import Iterable, List

from anndata import AnnData

TIME_KEY = "time_point_processed"
LATENT_KEY = "X_latent"
VELOCITY_LATENT_KEY = "velocity_latent"
GROWTH_KEY = "growth_rate"
PCA_LOADINGS_KEY = "PCs"


def missing_contract_fields(adata: AnnData, required: Iterable[str]) -> List[str]:
    missing: List[str] = []
    for key in required:
        if key == TIME_KEY and key not in adata.obs.columns:
            missing.append(key)
        elif key == LATENT_KEY and key not in adata.obsm:
            missing.append(key)
        elif key == VELOCITY_LATENT_KEY and (key not in adata.obsm and key not in adata.layers):
            missing.append(key)
        elif key == GROWTH_KEY and (key not in adata.obsm and key not in adata.obs.columns):
            missing.append(key)
        elif key == PCA_LOADINGS_KEY and (not hasattr(adata, "varm") or key not in adata.varm):
            missing.append(key)
    return missing


def ensure_contract(adata: AnnData, required: Iterable[str], where: str = "downstream") -> None:
    missing = missing_contract_fields(adata, required)
    if missing:
        raise KeyError(
            f"{where}: missing required contract field(s): {', '.join(missing)}. "
            f"Expected strict downstream contract with keys like '{LATENT_KEY}' and '{TIME_KEY}'."
        )
