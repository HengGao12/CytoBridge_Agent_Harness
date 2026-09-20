"""Generic velocity stream panel shell.

This is the reusable grammar extracted from scDiffEq-style manifold + stream plots.
It does not compute velocity; it only standardizes rendering once 2D coordinates and
2D vector fields are available.

Style decisions preserved from the source grammar:
- quiet manifold background
- dark, readable stream layer
- optional scalar-colored vectors
- no frame, equal aspect
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_velocity_stream_panel(
    background_df: pd.DataFrame,
    x: str,
    y: str,
    u: np.ndarray,
    v: np.ndarray,
    scalar: Optional[np.ndarray] = None,
    background_color: str = "#D8D8D8",
    stream_color: str = "#1F2937",
    scalar_cmap: str = "viridis",
    ax: Optional[plt.Axes] = None,
):
    if ax is None:
        _, ax = plt.subplots(figsize=(4.0, 4.0), dpi=200)

    ax.scatter(
        background_df[x],
        background_df[y],
        s=10,
        c=background_color,
        alpha=0.35,
        edgecolor="none",
        rasterized=True,
        zorder=1,
    )

    xvals = background_df[x].to_numpy()
    yvals = background_df[y].to_numpy()

    if scalar is None:
        ax.quiver(
            xvals,
            yvals,
            u,
            v,
            angles="xy",
            scale_units="xy",
            scale=1,
            color=stream_color,
            width=0.002,
            alpha=0.6,
            zorder=3,
        )
        mappable = None
    else:
        mappable = ax.quiver(
            xvals,
            yvals,
            u,
            v,
            scalar,
            angles="xy",
            scale_units="xy",
            scale=1,
            cmap=scalar_cmap,
            width=0.002,
            alpha=0.8,
            zorder=3,
        )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_aspect("equal", adjustable="box")
    return ax, mappable


__all__ = ["plot_velocity_stream_panel"]
