from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_embedding_trajectory_overlay(
    background: pd.DataFrame,
    trajectories: Sequence[np.ndarray],
    *,
    x: str,
    y: str,
    color: str | None = None,
    background_color: str = "#d9d9d9",
    background_alpha: float = 0.45,
    point_size: float = 3.0,
    trajectory_color: str = "#2c7fb8",
    trajectory_alpha: float = 0.14,
    trajectory_lw: float = 0.8,
    output_path: str | Path | None = None,
):
    """Generic embedding + generated-trajectory overlay template.

    This was missing from the library and is useful when a learned model generates paths
    that should be shown on top of a fixed embedding basis.
    """

    fig, ax = plt.subplots(figsize=(4.8, 4.6), dpi=300, facecolor="white")
    ax.set_facecolor("white")

    if color is None:
        ax.scatter(
            background[x],
            background[y],
            c=background_color,
            s=point_size,
            alpha=background_alpha,
            edgecolors="none",
            rasterized=len(background) > 30000,
        )
    else:
        vals = background[color]
        ax.scatter(
            background[x],
            background[y],
            c=vals,
            s=point_size,
            alpha=background_alpha,
            edgecolors="none",
            rasterized=len(background) > 30000,
        )

    for traj in trajectories:
        traj = np.asarray(traj, dtype=float)
        if traj.ndim != 2 or traj.shape[1] != 2:
            raise ValueError("Each trajectory must have shape (n_steps, 2).")
        ax.plot(
            traj[:, 0],
            traj[:, 1],
            color=trajectory_color,
            alpha=trajectory_alpha,
            linewidth=trajectory_lw,
        )

    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    return fig
