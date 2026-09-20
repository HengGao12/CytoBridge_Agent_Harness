from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _scatter_by_group(
    ax,
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    groupby: str | None,
    palette: Mapping[str, str] | None,
    default_color: str,
    size: float,
    alpha: float,
    zorder: float,
    rasterized_threshold: int,
):
    if groupby is None:
        ax.scatter(
            df[x],
            df[y],
            s=size,
            c=default_color,
            alpha=alpha,
            edgecolors="none",
            zorder=zorder,
            rasterized=len(df) > rasterized_threshold,
        )
        return

    for group, group_df in df.groupby(groupby, sort=False):
        color = palette.get(group, default_color) if palette is not None else default_color
        ax.scatter(
            group_df[x],
            group_df[y],
            s=size,
            c=color,
            alpha=alpha,
            edgecolors="none",
            zorder=zorder,
            rasterized=len(group_df) > rasterized_threshold,
        )


def _draw_contour(
    ax,
    df: pd.DataFrame | None,
    *,
    x: str,
    y: str,
    color: str,
    linewidth: float,
    levels: int,
):
    if df is None or len(df) < 10:
        return
    sns.kdeplot(
        data=df,
        x=x,
        y=y,
        ax=ax,
        color=color,
        fill=False,
        levels=levels,
        linewidths=linewidth,
        zorder=250,
        warn_singular=False,
    )


def plot_perturbation_embedding_compare(
    background: pd.DataFrame,
    control: pd.DataFrame,
    perturbed: pd.DataFrame,
    *,
    x: str,
    y: str,
    background_groupby: str | None = None,
    overlay_groupby: str | None = None,
    background_palette: Mapping[str, str] | None = None,
    overlay_palette: Mapping[str, str] | None = None,
    control_contour: pd.DataFrame | None = None,
    perturbed_contour: pd.DataFrame | None = None,
    background_color: str = "#d0d0d0",
    background_alpha: float = 0.35,
    background_size: float = 3.0,
    overlay_size: float = 6.0,
    overlay_alpha: float = 0.9,
    contour_color: str = "black",
    contour_linewidth: float = 0.8,
    contour_levels: int = 5,
    titles: tuple[str, str] = ("Unperturbed", "Perturbed"),
    figsize: tuple[float, float] = (6.2, 3.1),
    output_path: str | Path | None = None,
    rasterized_threshold: int = 30000,
):
    """Side-by-side embedding comparison for perturbation outputs.

    Inspired by scDiffEq Figure 3AB:
    project control and perturbed cells onto one fixed embedding basis, keep the
    manifold background quiet, then overlay the two foreground conditions with optional
    contour highlights.
    """

    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=300, facecolor="white")

    for ax, foreground, contour_df, title in zip(
        axes,
        [control, perturbed],
        [control_contour, perturbed_contour],
        titles,
        strict=True,
    ):
        ax.set_facecolor("white")
        _scatter_by_group(
            ax,
            background,
            x=x,
            y=y,
            groupby=background_groupby,
            palette=background_palette,
            default_color=background_color,
            size=background_size,
            alpha=background_alpha,
            zorder=20,
            rasterized_threshold=rasterized_threshold,
        )
        _scatter_by_group(
            ax,
            foreground,
            x=x,
            y=y,
            groupby=overlay_groupby,
            palette=overlay_palette if overlay_palette is not None else background_palette,
            default_color="#3b6fb6",
            size=overlay_size,
            alpha=overlay_alpha,
            zorder=120,
            rasterized_threshold=rasterized_threshold,
        )
        _draw_contour(
            ax,
            contour_df,
            x=x,
            y=y,
            color=contour_color,
            linewidth=contour_linewidth,
            levels=contour_levels,
        )
        ax.set_title(title, fontsize=9)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.subplots_adjust(wspace=0.08)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight", facecolor="white")

    return fig
