from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import pandas as pd


def plot_embedding_time_overlay(
    background: pd.DataFrame,
    simulated: pd.DataFrame,
    *,
    x: str,
    y: str,
    time_col: str,
    background_groupby: str | None = None,
    background_palette: Mapping[str, str] | None = None,
    lineage: pd.DataFrame | None = None,
    lineage_time_col: str | None = None,
    lineage_palette: Mapping[object, str] | None = None,
    start_df: pd.DataFrame | None = None,
    background_color: str = "#d6d6d6",
    background_alpha: float = 0.32,
    background_size: float = 3.0,
    simulated_size: float = 8.0,
    simulated_alpha: float = 0.5,
    start_outer_color: str = "black",
    start_inner_color: str = "dodgerblue",
    start_outer_size: float = 60.0,
    start_inner_size: float = 30.0,
    simulated_cmap: str = "viridis",
    figsize: tuple[float, float] = (4.2, 4.1),
    output_path: str | Path | None = None,
    rasterized_threshold: int = 30000,
):
    """Overlay time-colored simulated cells on a fixed embedding basis.

    Inspired by scDiffEq Figure 4ABC:
    - fixed manifold background
    - simulated points projected into that basis
    - time encoded by a continuous colormap
    - optional observed lineage overlay
    - optional start marker
    """

    fig, ax = plt.subplots(figsize=figsize, dpi=300, facecolor="white")
    ax.set_facecolor("white")

    if background_groupby is None:
        ax.scatter(
            background[x],
            background[y],
            c=background_color,
            s=background_size,
            alpha=background_alpha,
            edgecolors="none",
            zorder=20,
            rasterized=len(background) > rasterized_threshold,
        )
    else:
        for group, group_df in background.groupby(background_groupby, sort=False):
            color = (
                background_palette.get(group, background_color)
                if background_palette is not None
                else background_color
            )
            ax.scatter(
                group_df[x],
                group_df[y],
                c=color,
                s=background_size,
                alpha=background_alpha,
                edgecolors="none",
                zorder=20,
                rasterized=len(group_df) > rasterized_threshold,
            )

    sim_sorted = simulated.sort_values(time_col)
    ax.scatter(
        sim_sorted[x],
        sim_sorted[y],
        c=sim_sorted[time_col],
        s=simulated_size,
        alpha=simulated_alpha,
        cmap=simulated_cmap,
        edgecolors="none",
        zorder=120,
        rasterized=len(sim_sorted) > rasterized_threshold,
    )

    if lineage is not None:
        if lineage_time_col is None:
            raise ValueError("lineage_time_col is required when lineage is provided.")
        for group, group_df in lineage.groupby(lineage_time_col, sort=False):
            color = (
                lineage_palette.get(group, "#333333")
                if lineage_palette is not None
                else "#333333"
            )
            ax.scatter(
                group_df[x],
                group_df[y],
                c="black",
                s=simulated_size * 2.0,
                alpha=0.95,
                edgecolors="none",
                zorder=220,
            )
            ax.scatter(
                group_df[x],
                group_df[y],
                c=color,
                s=simulated_size,
                alpha=0.95,
                edgecolors="none",
                zorder=221,
            )

    if start_df is not None and len(start_df) > 0:
        ax.scatter(
            start_df[x],
            start_df[y],
            c=start_outer_color,
            s=start_outer_size,
            edgecolors="none",
            zorder=260,
        )
        ax.scatter(
            start_df[x],
            start_df[y],
            c=start_inner_color,
            s=start_inner_size,
            edgecolors="none",
            zorder=261,
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
