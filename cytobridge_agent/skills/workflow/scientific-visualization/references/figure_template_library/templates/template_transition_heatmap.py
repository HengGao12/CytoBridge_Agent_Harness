"""Generic annotated transition heatmap.

Inspired by moscot's cell_transition plotting API.
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_transition_heatmap(
    matrix: pd.DataFrame,
    cmap: str = "viridis",
    annotate: bool = True,
    fmt: str = ".2f",
    figsize: tuple[float, float] = (4.0, 3.6),
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=figsize, dpi=200)

    sns.heatmap(
        matrix,
        cmap=cmap,
        annot=annotate,
        fmt=fmt,
        linewidths=0.0,
        linecolor="none",
        cbar=True,
        ax=ax,
    )
    ax.set_facecolor("white")
    ax.tick_params(top=False, bottom=False, left=False, right=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return ax


__all__ = ["plot_transition_heatmap"]

