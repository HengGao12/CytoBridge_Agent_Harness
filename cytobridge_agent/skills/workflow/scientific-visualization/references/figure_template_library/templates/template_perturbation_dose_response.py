from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_perturbation_dose_response(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    hue: str | None = None,
    order=None,
    palette=None,
    show_points: bool = True,
    output_path: str | Path | None = None,
):
    """Generic perturbation dose-response panel.

    Intended for small designed perturbation families such as TF/module z-scans.
    """

    fig, ax = plt.subplots(figsize=(5.4, 3.4), dpi=300, facecolor="white")
    ax.set_facecolor("white")

    sns.boxplot(
        data=df,
        x=x,
        y=y,
        hue=hue,
        order=order,
        palette=palette,
        width=0.72,
        linewidth=1.0,
        fliersize=0,
        ax=ax,
    )
    if show_points:
        sns.stripplot(
            data=df,
            x=x,
            y=y,
            hue=hue,
            order=order,
            palette=palette,
            dodge=hue is not None,
            size=2.6,
            alpha=0.45,
            linewidth=0,
            ax=ax,
        )

    ax.axhline(0, color="#bdbdbd", linewidth=0.9, zorder=0)
    ax.grid(axis="y", alpha=0.18, linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if ax.legend_ is not None:
        handles, labels = ax.get_legend_handles_labels()
        n = len(labels) // 2 if show_points and hue is not None else len(labels)
        ax.legend(handles[:n], labels[:n], frameon=False, fontsize=8)

    fig.tight_layout()
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    return fig
