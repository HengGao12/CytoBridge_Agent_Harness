"""GRN plotting helpers (package-native)."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping

import matplotlib.pyplot as plt
import numpy as np


def _ensure_dir(output_dir: str) -> Path:
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    return out


def plot_grn_bundle(matrix: np.ndarray, output_dir: str, file_name: str = "grn_heatmap.png") -> Dict[str, object]:
    outdir = _ensure_dir(output_dir)
    mat = np.asarray(matrix)
    if mat.ndim != 2:
        return {
            "space_used": "latent",
            "projection_backend": "none",
            "artifacts": {},
            "warnings": ["GRN matrix must be 2D."],
        }

    vmax = float(np.max(np.abs(mat))) if mat.size else 1.0
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(mat, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("GRN / Jacobian heatmap")
    path = outdir / file_name
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return {
        "space_used": "latent",
        "projection_backend": "none",
        "artifacts": {"grn_heatmap": str(path)},
        "warnings": [],
    }


def plot_grn_timeseries_bundle(grn_payload: Mapping[str, dict], output_dir: str, prefix: str = "grn") -> Dict[str, object]:
    outdir = _ensure_dir(output_dir)
    artifacts: Dict[str, str] = {}
    warnings = []

    for key, payload in (grn_payload or {}).items():
        mat = np.asarray(payload.get("matrix", []), dtype=float)
        if mat.ndim != 2 or mat.size == 0:
            warnings.append(f"skip {key}: invalid matrix")
            continue
        out = plot_grn_bundle(mat, str(outdir), file_name=f"{prefix}_{key}.png")
        p = out.get("artifacts", {}).get("grn_heatmap")
        if p:
            artifacts[key] = p

    return {
        "space_used": "latent",
        "projection_backend": "none",
        "artifacts": artifacts,
        "warnings": warnings,
    }
