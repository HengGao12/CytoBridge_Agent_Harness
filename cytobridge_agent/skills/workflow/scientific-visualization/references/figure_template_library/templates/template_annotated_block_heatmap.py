from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import numpy as np
import pandas as pd
import seaborn as sns


def _group_boundaries(groups):
    groups = list(groups)
    bounds = []
    for i in range(1, len(groups)):
        if groups[i] != groups[i - 1]:
            bounds.append(i)
    return bounds


def plot_annotated_block_heatmap(
    matrix: pd.DataFrame,
    *,
    row_groups=None,
    col_groups=None,
    cmap: str = "Spectral_r",
    center: float | None = None,
    show_row_names: bool = False,
    show_col_names: bool = False,
    group_palette: dict | None = None,
    output_path: str | Path | None = None,
):
    """Annotated block heatmap inspired by iPOP Aging ComplexHeatmap panels."""

    values = matrix.to_numpy(dtype=float)
    nrows, ncols = values.shape
    row_groups = list(row_groups) if row_groups is not None else [""] * nrows
    col_groups = list(col_groups) if col_groups is not None else [""] * ncols
    row_bounds = _group_boundaries(row_groups)
    col_bounds = _group_boundaries(col_groups)

    if group_palette is None:
        levels = pd.unique(pd.Series(row_groups + col_groups))
        palette = sns.color_palette("Set2", n_colors=max(3, len(levels)))
        group_palette = {k: palette[i] for i, k in enumerate(levels)}

    fig = plt.figure(figsize=(7.4, 6.2), dpi=300, facecolor="white")
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[0.18, 1.0],
        height_ratios=[0.12, 1.0],
        wspace=0.02,
        hspace=0.02,
    )
    ax_top = fig.add_subplot(gs[0, 1])
    ax_left = fig.add_subplot(gs[1, 0])
    ax = fig.add_subplot(gs[1, 1])

    sns.heatmap(
        values,
        cmap=cmap,
        center=center,
        ax=ax,
        cbar=True,
        xticklabels=matrix.columns if show_col_names else False,
        yticklabels=matrix.index if show_row_names else False,
        linewidths=0,
        linecolor="white",
    )
    for b in row_bounds:
        ax.hlines(b, *ax.get_xlim(), colors="white", linewidth=1.2)
    for b in col_bounds:
        ax.vlines(b, *ax.get_ylim(), colors="white", linewidth=1.2)

    top_rgba = np.array([[to_rgba(group_palette[g]) for g in col_groups]])
    ax_top.imshow(top_rgba, aspect="auto")
    ax_top.set_xticks([])
    ax_top.set_yticks([])
    for b in col_bounds:
        ax_top.vlines(b - 0.5, -0.5, 0.5, colors="white", linewidth=1.2)
    for spine in ax_top.spines.values():
        spine.set_visible(False)

    left_rgba = np.array([[to_rgba(group_palette[g]) for g in row_groups]]).transpose(1, 0, 2)
    ax_left.imshow(left_rgba, aspect="auto")
    ax_left.set_xticks([])
    ax_left.set_yticks([])
    for b in row_bounds:
        ax_left.hlines(b - 0.5, -0.5, 0.5, colors="white", linewidth=1.2)
    for spine in ax_left.spines.values():
        spine.set_visible(False)

    fig.tight_layout()
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    return fig
