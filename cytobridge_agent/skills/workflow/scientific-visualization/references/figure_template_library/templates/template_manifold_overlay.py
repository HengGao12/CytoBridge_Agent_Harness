"""Generic manifold overlay panel.

Inspired by scDiffEq simulated-over-UMAP panels and moslin coupling overlays.

Style decisions preserved from the source grammar:
- quiet grey background manifold
- stronger overlay layer
- optional white start markers with black outline
- hidden axes with equal aspect
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_manifold_overlay(
    background_df: pd.DataFrame,
    x: str,
    y: str,
    background_color: str = "#D6D6D6",
    background_alpha: float = 0.35,
    background_size: float = 8.0,
    overlay_df: Optional[pd.DataFrame] = None,
    overlay_color: Optional[str] = None,
    overlay_value: Optional[str] = None,
    overlay_cmap: str = "viridis",
    overlay_size: float = 14.0,
    overlay_alpha: float = 0.8,
    start_df: Optional[pd.DataFrame] = None,
    ax: Optional[plt.Axes] = None,
):
    if ax is None:
        _, ax = plt.subplots(figsize=(4.0, 4.0), dpi=200)

    ax.scatter(
        background_df[x],
        background_df[y],
        s=background_size,
        c=background_color,
        alpha=background_alpha,
        edgecolor="none",
        rasterized=True,
        zorder=1,
    )

    mappable = None
    if overlay_df is not None and len(overlay_df) > 0:
        kwargs = dict(
            s=overlay_size,
            alpha=overlay_alpha,
            edgecolor="none",
            rasterized=True,
            zorder=3,
        )
        if overlay_value is not None:
            mappable = ax.scatter(
                overlay_df[x],
                overlay_df[y],
                c=overlay_df[overlay_value],
                cmap=overlay_cmap,
                **kwargs,
            )
        else:
            mappable = ax.scatter(
                overlay_df[x],
                overlay_df[y],
                c=overlay_color or "#1D4ED8",
                **kwargs,
            )

    if start_df is not None and len(start_df) > 0:
        centers = start_df[[x, y]].to_numpy()
        ax.scatter(
            centers[:, 0],
            centers[:, 1],
            s=45,
            facecolor="white",
            edgecolor="black",
            linewidth=0.8,
            zorder=4,
        )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_aspect("equal", adjustable="box")
    return ax, mappable


__all__ = ["plot_manifold_overlay"]
