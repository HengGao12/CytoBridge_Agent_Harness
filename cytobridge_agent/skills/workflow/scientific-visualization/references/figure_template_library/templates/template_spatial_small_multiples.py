from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


def plot_spatial_small_multiples(
    panels: Sequence[pd.DataFrame],
    *,
    x: str,
    y: str,
    label: str,
    panel_title: str,
    color_map: dict[str, str] | None = None,
    point_size: float = 2.0,
    point_alpha: float = 0.85,
    rotate_90ccw: bool = False,
    vertical: bool = True,
    legend_title: str = "Cell types",
    output_path: str | Path | None = None,
):
    """Generic spatial small-multiples template inspired by cytobridge-downstream.

    Expected input:
    - one dataframe per panel
    - each dataframe contains coordinates plus one categorical label column
    - panel_title column contains the per-panel caption
    """

    labels = []
    for panel in panels:
        labels.extend(panel[label].astype(str).tolist())
    unique_labels = sorted(set(labels))

    if color_map is None:
        palette = plt.get_cmap("tab20")(np.linspace(0, 1, max(1, len(unique_labels))))
        color_map = {
            lab: "#{:02x}{:02x}{:02x}".format(
                int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
            )
            for lab, rgb in zip(unique_labels, palette)
        }

    n = len(panels)
    if vertical:
        fig, axes = plt.subplots(n, 1, figsize=(8.5, max(2.8 * n, 3.2)), dpi=300)
    else:
        fig, axes = plt.subplots(1, n, figsize=(max(2.8 * n, 3.2), 4.0), dpi=300)
    axes = np.atleast_1d(axes)
    fig.patch.set_facecolor("white")

    for ax, panel in zip(axes, panels):
        coords = panel[[x, y]].to_numpy(dtype=float)
        if rotate_90ccw:
            coords = np.c_[-coords[:, 1], coords[:, 0]]
        colors = [color_map[str(v)] for v in panel[label].astype(str)]
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=colors,
            s=point_size,
            alpha=point_alpha,
            edgecolors="none",
            rasterized=len(panel) > 30000,
        )
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(str(panel[panel_title].iloc[0]), fontsize=11, pad=4)

    handles = [
        Patch(facecolor=color_map[lab], edgecolor="none", label=lab)
        for lab in unique_labels
    ]
    fig.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        title=legend_title,
        fontsize=8,
        title_fontsize=9,
        ncol=1 if len(handles) <= 14 else 2,
    )
    fig.tight_layout(rect=[0, 0, 0.88, 1])

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    return fig
