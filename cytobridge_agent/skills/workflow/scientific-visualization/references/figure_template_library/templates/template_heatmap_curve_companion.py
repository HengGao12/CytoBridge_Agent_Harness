from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec


def plot_heatmap_curve_companion(
    heatmap_matrix: np.ndarray,
    *,
    x_labels: Sequence[str],
    group_slices: Sequence[tuple[str, int, int]],
    curve_x: np.ndarray,
    curve_means: dict[str, np.ndarray],
    curve_stds: dict[str, np.ndarray] | None = None,
    heatmap_cmap: str = "RdBu_r",
    heatmap_vmin: float = -2.0,
    heatmap_vmax: float = 2.0,
    output_path: str | Path | None = None,
):
    """Heatmap + trajectory-grid template inspired by cytobridge-downstream."""

    n_groups = len(group_slices)
    curve_cols = 2 if n_groups <= 4 else 3
    curve_rows = int(np.ceil(n_groups / curve_cols))

    fig = plt.figure(figsize=(14, 7.5), dpi=300)
    gs = GridSpec(1, 2, width_ratios=[1.45, 1.0], wspace=0.12)
    ax_hm = fig.add_subplot(gs[0, 0])
    im = ax_hm.imshow(
        np.asarray(heatmap_matrix, dtype=float),
        aspect="auto",
        cmap=heatmap_cmap,
        vmin=heatmap_vmin,
        vmax=heatmap_vmax,
    )

    for idx, (name, start, end) in enumerate(group_slices):
        if idx > 0:
            ax_hm.axhline(start - 0.5, color="black", linestyle="--", linewidth=1.2)
        middle = (start + end) / 2.0
        ax_hm.text(
            len(x_labels) - 1.2,
            middle,
            name,
            ha="right",
            va="center",
            fontsize=10,
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.8},
        )
    ax_hm.set_yticks([])
    ax_hm.set_xticks(np.arange(len(x_labels)))
    ax_hm.set_xticklabels(x_labels, rotation=45, fontsize=9)
    cbar = fig.colorbar(im, ax=ax_hm, shrink=0.92, pad=0.02)
    cbar.outline.set_visible(False)

    curve_grid = GridSpec(
        curve_rows,
        curve_cols,
        left=0.60,
        right=0.97,
        bottom=0.10,
        top=0.95,
        wspace=0.28,
        hspace=0.35,
    )
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, max(1, n_groups)))

    all_mean = np.concatenate([np.asarray(curve_means[name], dtype=float) for name, _, _ in group_slices])
    y_pad = 0.1 * max(1e-6, float(all_mean.max() - all_mean.min()))
    y_min = float(all_mean.min() - y_pad)
    y_max = float(all_mean.max() + y_pad)

    for idx, (name, _, _) in enumerate(group_slices):
        row, col = divmod(idx, curve_cols)
        ax = fig.add_subplot(curve_grid[row, col])
        mean = np.asarray(curve_means[name], dtype=float)
        std = np.zeros_like(mean) if curve_stds is None else np.asarray(curve_stds[name], dtype=float)
        ax.plot(curve_x, mean, color=colors[idx], linewidth=2)
        ax.fill_between(curve_x, mean - std, mean + std, color=colors[idx], alpha=0.22)
        ax.axhline(0, color="#999999", linestyle="--", linewidth=0.8)
        ax.set_title(name, fontsize=10, fontweight="bold")
        ax.set_ylim(y_min, y_max)
        ax.grid(True, axis="y", alpha=0.18)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    return fig
