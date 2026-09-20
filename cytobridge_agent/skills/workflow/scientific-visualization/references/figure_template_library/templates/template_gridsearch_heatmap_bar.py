"""Generic sweep-summary templates.

Inspired by moslin gridsearch_heatmap/gridsearch_bar helpers.

Style decisions preserved from the source grammar:
- heatmap with explicit best-cell outline
- grouped benchmark bars with quiet y-grid
- clean white background and minimal framing
"""

from __future__ import annotations

from typing import Optional

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_sweep_heatmap(
    df: pd.DataFrame,
    row: str,
    col: str,
    value: str,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    cmap: str = "coolwarm",
):
    if ax is None:
        _, ax = plt.subplots(figsize=(3.6, 3.0), dpi=200)

    pivot_df = df.pivot(index=row, columns=col, values=value)
    sns.heatmap(pivot_df, cmap=cmap, annot=True, fmt=".2f", ax=ax, cbar=True)

    best_idx = pivot_df.stack().idxmin()
    y_idx = list(pivot_df.index).index(best_idx[0])
    x_idx = list(pivot_df.columns).index(best_idx[1])
    rect = patches.Rectangle((x_idx, y_idx), 1, 1, linewidth=1.5, edgecolor="black", facecolor="none")
    ax.add_patch(rect)
    if title:
        ax.set_title(title)
    ax.set_facecolor("white")
    return ax


def plot_grouped_benchmark_bar(
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: str,
    palette: Optional[list[str]] = None,
    ax: Optional[plt.Axes] = None,
):
    if ax is None:
        _, ax = plt.subplots(figsize=(4.4, 2.8), dpi=200)

    sns.barplot(data=df, x=x, y=y, hue=hue, palette=palette, ax=ax)
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.15)
    return ax


__all__ = ["plot_sweep_heatmap", "plot_grouped_benchmark_bar"]
