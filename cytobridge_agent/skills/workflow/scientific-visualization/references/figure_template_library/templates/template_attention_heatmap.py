from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_attention_heatmap(
    matrix: pd.DataFrame | np.ndarray,
    *,
    xticklabels=None,
    yticklabels=None,
    mode: str = "positive",
    sig_mask: np.ndarray | None = None,
    title: str | None = None,
    cbar_label: str | None = None,
    output_path: str | Path | None = None,
):
    """Square communication/attention heatmap inspired by cytobridge-downstream.

    mode:
    - "positive": positive-only matrix, uses a quiet blue sequential palette
    - "asymmetry": signed matrix, uses a diverging palette centered at zero
    """

    if isinstance(matrix, pd.DataFrame):
        values = matrix.to_numpy(dtype=float)
        if xticklabels is None:
            xticklabels = list(matrix.columns)
        if yticklabels is None:
            yticklabels = list(matrix.index)
    else:
        values = np.asarray(matrix, dtype=float)

    sns.set_theme(
        style="whitegrid",
        font_scale=1.0,
        rc={
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "axes.labelcolor": "black",
            "text.color": "black",
        },
    )
    fig, ax = plt.subplots(figsize=(6.8, 5.6), dpi=300)

    if mode == "positive":
        cmap = sns.color_palette("PuBu", as_cmap=True)
        center = None
        cbar_label = cbar_label or "Value"
    elif mode == "asymmetry":
        cmap = sns.diverging_palette(250, 10, as_cmap=True)
        center = 0
        cbar_label = cbar_label or "Asymmetry"
    else:
        raise ValueError("mode must be 'positive' or 'asymmetry'")

    hm = sns.heatmap(
        values,
        xticklabels=xticklabels,
        yticklabels=yticklabels,
        cmap=cmap,
        center=center,
        square=True,
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": cbar_label},
        ax=ax,
    )

    if sig_mask is not None:
        yy, xx = np.where(np.asarray(sig_mask))
        for i, j in zip(yy, xx):
            ax.text(j + 0.5, i + 0.5, "•", ha="center", va="center", fontsize=9, color="k")

    if title:
        ax.set_title(title)
    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    return fig
