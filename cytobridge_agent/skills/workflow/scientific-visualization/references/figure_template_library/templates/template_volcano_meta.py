"""Generic volcano panel.

Inspired by scDiffEq perturbation-screen meta-analysis plots.

Style decisions preserved from the source grammar:
- quiet grey nonsignificant cloud
- symmetric red/blue significant groups
- threshold lines
- sparse manual labels only
"""

from __future__ import annotations

from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def prepare_volcano_groups(
    df: pd.DataFrame,
    effect_col: str,
    pval_col: str,
    alpha: float = 0.05,
) -> pd.DataFrame:
    out = df.copy()
    out["x"] = out[effect_col]
    out["y"] = -np.log10(np.clip(out[pval_col], 1e-300, None))
    out["significant"] = out[pval_col] < alpha
    out["positive"] = out[effect_col] > 0
    out["group"] = np.select(
        [
            out["significant"] & out["positive"],
            out["significant"] & ~out["positive"],
            ~out["significant"] & out["positive"],
        ],
        ["sig.pos", "sig.neg", "insig.pos"],
        default="insig.neg",
    )
    return out


def plot_volcano_meta(
    df: pd.DataFrame,
    effect_col: str,
    pval_col: str,
    label_col: Optional[str] = None,
    highlight: Optional[Iterable[str]] = None,
    alpha: float = 0.05,
    ax: Optional[plt.Axes] = None,
):
    if ax is None:
        _, ax = plt.subplots(figsize=(4.0, 4.0), dpi=200)

    plot_df = prepare_volcano_groups(df, effect_col=effect_col, pval_col=pval_col, alpha=alpha)
    color_map = {
        "insig.pos": "lightgrey",
        "insig.neg": "lightgrey",
        "sig.pos": "crimson",
        "sig.neg": "navy",
    }

    for group, sub in plot_df.groupby("group"):
        ax.scatter(
            sub["x"],
            sub["y"],
            s=14 if "sig" in group else 10,
            c=color_map[group],
            alpha=0.25 if "insig" in group else 0.85,
            edgecolor="none",
            rasterized=True,
            zorder=2 if "sig" in group else 1,
        )

    ax.axhline(-np.log10(alpha), color="#4B5563", ls="--", lw=0.8)
    ax.axvline(0, color="#9CA3AF", lw=0.8)

    if highlight is not None and label_col is not None:
        highlight = set(highlight)
        sub = plot_df.loc[plot_df[label_col].isin(highlight)]
        for _, row in sub.iterrows():
            color = "crimson" if row["x"] > 0 else "navy"
            ax.scatter(row["x"], row["y"], s=28, c="white", edgecolor="none", zorder=3)
            ax.scatter(row["x"], row["y"], s=18, c=color, edgecolor="none", zorder=4)
            ax.text(row["x"], row["y"], str(row[label_col]), fontsize=6, color=color)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_facecolor("white")
    ax.grid(alpha=0.15)
    ax.set_axisbelow(True)
    return ax


__all__ = ["prepare_volcano_groups", "plot_volcano_meta"]
