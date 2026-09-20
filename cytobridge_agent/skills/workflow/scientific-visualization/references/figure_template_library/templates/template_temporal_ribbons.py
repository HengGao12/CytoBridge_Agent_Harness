"""Generic temporal ribbon panel.

Inspired by scDiffEq temporal composition figures.

Style decisions preserved from the source grammar:
- stacked fills without borders
- quiet horizontal grid
- simple left-to-right temporal readout
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_temporal_ribbons(
    df: pd.DataFrame,
    time_col: str,
    category_col: str,
    value_col: str,
    palette: dict[str, str],
    ax: Optional[plt.Axes] = None,
):
    if ax is None:
        _, ax = plt.subplots(figsize=(4.8, 2.8), dpi=200)

    pivot = (
        df.pivot(index=time_col, columns=category_col, values=value_col)
        .sort_index()
        .fillna(0.0)
    )
    times = pivot.index.to_numpy()
    baseline = np.zeros(len(times), dtype=float)

    for category in pivot.columns:
        values = pivot[category].to_numpy()
        ax.fill_between(
            times,
            baseline,
            baseline + values,
            color=palette.get(category, "#999999"),
            alpha=0.9,
            linewidth=0.0,
            label=category,
        )
        baseline = baseline + values

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_facecolor("white")
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.15)
    return ax


__all__ = ["plot_temporal_ribbons"]
