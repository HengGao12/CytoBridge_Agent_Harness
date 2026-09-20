from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_rich_scatter_facet(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    line_group: str,
    color: str | None = None,
    facet: str | None = None,
    palette=None,
    add_smoother: bool = False,
    baseline: float | None = 0.0,
    output_path: str | Path | None = None,
):
    """Scatter/line/facet summary inspired by iPOP Aging ggplot panels."""

    sns.set_theme(style="white", rc={"figure.facecolor": "white", "axes.facecolor": "white"})
    if facet is None:
        fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=300, facecolor="white")
        sns.lineplot(
            data=df,
            x=x,
            y=y,
            units=line_group,
            estimator=None,
            hue=color,
            palette=palette,
            linewidth=0.85,
            alpha=0.45,
            legend=color is not None,
            ax=ax,
        )
        if add_smoother and color is not None:
            for level, sub in df.groupby(color):
                sns.regplot(
                    data=sub,
                    x=x,
                    y=y,
                    lowess=True,
                    scatter=False,
                    truncate=False,
                    color=(palette[level] if isinstance(palette, dict) and level in palette else None),
                    line_kws={"linewidth": 2.0, "alpha": 0.95},
                    ax=ax,
                )
        elif add_smoother:
            sns.regplot(
                data=df,
                x=x,
                y=y,
                lowess=True,
                scatter=False,
                truncate=False,
                color="#222222",
                line_kws={"linewidth": 2.0, "alpha": 0.95},
                ax=ax,
            )
        axes = [ax]
        fig_out = fig
    else:
        g = sns.FacetGrid(
            df,
            col=facet,
            hue=color,
            palette=palette,
            sharey=False,
            sharex=True,
            despine=True,
            col_wrap=4,
            height=3.0,
            aspect=1.15,
        )
        g.map_dataframe(
            sns.lineplot,
            x=x,
            y=y,
            units=line_group,
            estimator=None,
            linewidth=0.75,
            alpha=0.42,
        )
        if color is not None:
            g.add_legend(frameon=False)
        fig_out = g.fig
        axes = list(g.axes.flat)

    for ax in axes:
        if baseline is not None:
            ax.axhline(baseline, color="#9e9e9e", linewidth=0.8, zorder=0)
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig_out.tight_layout()
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig_out.savefig(output_path, bbox_inches="tight", facecolor="white")
    return fig_out
