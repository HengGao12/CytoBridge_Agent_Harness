from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _robust_bounds(values: np.ndarray, q_low: float = 1.0, q_high: float = 99.0) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 1.0
    lo = float(np.percentile(finite, q_low))
    hi = float(np.percentile(finite, q_high))
    if hi <= lo:
        lo = float(np.min(finite))
        hi = float(np.max(finite))
    if hi <= lo:
        hi = lo + 1e-12
    return lo, hi


def plot_spatial_scalar_triptych(
    panels: Sequence[pd.DataFrame],
    *,
    x: str,
    y: str,
    value: str,
    panel_title: str,
    cmap: str = "viridis",
    point_size: float = 3.0,
    foreground_alpha: float = 0.95,
    background_color: str = "#edf2f4",
    background_alpha: float = 0.35,
    shared_scale: bool = False,
    invert_y_flags: Sequence[bool] | None = None,
    output_path: str | Path | None = None,
):
    """Spatial scalar triptych with quiet grey context + colored foreground."""

    n = len(panels)
    if n < 2:
        raise ValueError("Use at least two panels for a triptych/strip layout.")

    arrays = [panel[value].to_numpy(dtype=float) for panel in panels]
    if shared_scale:
        lo, hi = _robust_bounds(np.concatenate(arrays))
        bounds = [(lo, hi)] * n
    else:
        bounds = [_robust_bounds(arr) for arr in arrays]

    fig, axes = plt.subplots(1, n, figsize=(3.4 * n + 0.6, 3.8), dpi=300, facecolor="white")
    axes = np.atleast_1d(axes)

    last_sc = None
    for idx, (ax, panel, (lo, hi)) in enumerate(zip(axes, panels, bounds)):
        xv = panel[x].to_numpy(dtype=float)
        yv = panel[y].to_numpy(dtype=float)
        vv = panel[value].to_numpy(dtype=float)

        ax.scatter(
            xv,
            yv,
            c=background_color,
            s=max(0.5, point_size * 0.9),
            alpha=background_alpha,
            edgecolors="none",
            rasterized=len(panel) > 30000,
        )
        last_sc = ax.scatter(
            xv,
            yv,
            c=np.clip(vv, lo, hi),
            s=point_size,
            alpha=foreground_alpha,
            cmap=cmap,
            vmin=lo,
            vmax=hi,
            edgecolors="none",
            rasterized=len(panel) > 30000,
        )
        ax.set_aspect("equal")
        ax.set_axis_off()
        if invert_y_flags is not None and invert_y_flags[idx]:
            ax.invert_yaxis()
        ax.set_title(str(panel[panel_title].iloc[0]), fontsize=11, pad=6)

    cbar = fig.colorbar(last_sc, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
    cbar.outline.set_visible(False)
    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    return fig
