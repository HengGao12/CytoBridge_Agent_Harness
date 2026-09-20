from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_bubble_summary(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    size: str,
    color: str,
    xlabel: str | None = None,
    ylabel: str | None = None,
    size_label: str = "Group size",
    color_label: str | None = None,
    cmap: str = "plasma",
    size_min: float = 24.0,
    size_max: float = 360.0,
    output_path: str | Path | None = None,
):
    """Generic bubble-summary scatter inspired by cytobridge-downstream."""

    fig, ax = plt.subplots(figsize=(7.0, 4.8), dpi=300)
    fig.patch.set_facecolor("white")

    raw_size = df[size].to_numpy(dtype=float)
    if raw_size.size == 0:
        raise ValueError("Empty dataframe.")
    lo, hi = float(np.min(raw_size)), float(np.max(raw_size))
    if hi <= lo:
        sizes = np.full_like(raw_size, (size_min + size_max) / 2.0)
    else:
        sizes = size_min + (raw_size - lo) / (hi - lo) * (size_max - size_min)

    sc = ax.scatter(
        df[x],
        df[y],
        s=sizes,
        c=df[color],
        cmap=cmap,
        alpha=0.82,
        edgecolors="white",
        linewidths=0.4,
    )
    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or y)
    ax.grid(False)

    size_examples = np.unique(np.round(np.percentile(raw_size, [0, 50, 100])).astype(int))
    size_examples = size_examples[size_examples > 0]
    handles = []
    labels = []
    for v in size_examples:
        if hi <= lo:
            s = (size_min + size_max) / 2.0
        else:
            s = size_min + (float(v) - lo) / (hi - lo) * (size_max - size_min)
        handles.append(
            ax.scatter([], [], s=s, color="#777777", alpha=0.9, edgecolors="white", linewidths=0.4)
        )
        labels.append(str(v))
    if handles:
        leg = ax.legend(
            handles,
            labels,
            title=size_label,
            frameon=False,
            loc="upper right",
            bbox_to_anchor=(1.02, 1.0),
        )
        ax.add_artist(leg)

    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label(color_label or color)
    cbar.outline.set_visible(False)

    fig.tight_layout()
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    return fig
