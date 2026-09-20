"""Generic stylish box+jitter panel.

Inspired by scDiffEq manuscript benchmark panels and the packaged StyledBoxPlot helper.
This version is dataset-agnostic and only assumes a tidy dataframe.

Style decisions preserved from the source grammar:
- translucent background box
- crisp foreground outline
- jittered foreground points
- quiet y-grid only
"""

from __future__ import annotations

from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_stylish_box_jitter(
    df: pd.DataFrame,
    x: str,
    y: str,
    order: Optional[Iterable[str]] = None,
    palette: Optional[list[str]] = None,
    width: float = 0.7,
    jitter: float = 0.12,
    point_size: float = 18,
    point_alpha: float = 0.65,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(4.0, 2.8), dpi=200)

    data = df.copy()
    if order is None:
        order = list(pd.unique(data[x]))
    order = list(order)

    if palette is None:
        palette = ["#2A6F97", "#468FAF", "#61A5C2", "#89C2D9", "#014F86"][: len(order)]
    if len(palette) < len(order):
        raise ValueError("palette shorter than number of groups")

    grouped = [data.loc[data[x] == group, y].dropna().to_numpy() for group in order]
    positions = np.arange(1, len(order) + 1)

    # Background box
    bp_bg = ax.boxplot(
        grouped,
        positions=positions,
        widths=width,
        showfliers=False,
        patch_artist=True,
        showmeans=True,
        meanline=True,
        zorder=1,
    )
    for i, box in enumerate(bp_bg["boxes"]):
        box.set_facecolor(palette[i])
        box.set_alpha(0.18)
        box.set_edgecolor("none")
    for mean in bp_bg["means"]:
        mean.set_color("#333333")
        mean.set_linewidth(1.0)
    for key in ("whiskers", "caps", "medians"):
        for item in bp_bg[key]:
            item.set_visible(False)

    # Foreground outline box
    bp_fg = ax.boxplot(
        grouped,
        positions=positions,
        widths=width,
        showfliers=False,
        patch_artist=True,
        zorder=2,
    )
    repeated_colors = np.repeat(np.array(palette), 2, axis=0)
    for i, box in enumerate(bp_fg["boxes"]):
        box.set_facecolor("none")
        box.set_edgecolor(palette[i])
        box.set_linewidth(1.2)
    for i, whisker in enumerate(bp_fg["whiskers"]):
        whisker.set_color(repeated_colors[i])
        whisker.set_linewidth(1.0)
    for i, cap in enumerate(bp_fg["caps"]):
        cap.set_color(repeated_colors[i])
        cap.set_linewidth(1.0)
    for median in bp_fg["medians"]:
        median.set_visible(False)

    # Jittered points
    rng = np.random.default_rng(0)
    for pos, values, color in zip(positions, grouped, palette):
        if len(values) == 0:
            continue
        xvals = pos + rng.uniform(-jitter, jitter, size=len(values))
        ax.scatter(
            xvals,
            values,
            s=point_size,
            color=color,
            alpha=point_alpha,
            edgecolor="none",
            zorder=3,
            rasterized=True,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(order, rotation=35, ha="right")
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.15, zorder=0)
    ax.set_axisbelow(True)
    return ax


__all__ = ["plot_stylish_box_jitter"]
