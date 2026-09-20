"""Growth plotting helpers (package-native)."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import matplotlib.pyplot as plt
import numpy as np


def _ensure_dir(output_dir: str) -> Path:
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    return out


def plot_growth_bundle(adata, output_dir: str, key: str = "growth_rate") -> Dict[str, object]:
    outdir = _ensure_dir(output_dir)

    if key in adata.obsm:
        arr = np.asarray(adata.obsm[key]).reshape(-1)
    elif key in adata.obs:
        arr = np.asarray(adata.obs[key]).reshape(-1)
    else:
        return {
            "space_used": "latent",
            "projection_backend": "none",
            "artifacts": {},
            "warnings": [f"Growth key not found: {key}"],
        }

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(arr[np.isfinite(arr)], bins=50)
    ax.set_title("Growth distribution")
    ax.set_xlabel(key)
    ax.set_ylabel("count")
    path = outdir / "growth_distribution.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    return {
        "space_used": "latent",
        "projection_backend": "none",
        "artifacts": {"growth_distribution": str(path)},
        "warnings": [],
    }


def plot_driver_scores_bundle(
    rows: Sequence[dict],
    output_dir: str,
    file_name: str,
    value_key: str,
    title: str,
    top_n: int = 20,
) -> Dict[str, object]:
    outdir = _ensure_dir(output_dir)
    top = list(rows[: max(1, int(top_n))])

    if len(top) == 0:
        return {
            "space_used": "latent",
            "projection_backend": "none",
            "artifacts": {},
            "warnings": ["no rows to plot"],
        }

    labels = [str(r.get("feature_name", r.get("feature", "unknown"))) for r in top][::-1]
    vals = [float(r.get(value_key, 0.0)) for r in top][::-1]

    fig, ax = plt.subplots(figsize=(10, max(4, len(top) * 0.35)))
    ax.barh(np.arange(len(top)), vals, alpha=0.85)
    ax.set_yticks(np.arange(len(top)))
    ax.set_yticklabels(labels)
    ax.set_title(title)
    ax.set_xlabel(value_key)
    fig.tight_layout()

    p = outdir / file_name
    fig.savefig(p, dpi=180, bbox_inches="tight")
    plt.close(fig)

    return {
        "space_used": "latent",
        "projection_backend": "none",
        "artifacts": {"driver_plot": str(p)},
        "warnings": [],
    }
