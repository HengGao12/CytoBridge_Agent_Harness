from __future__ import annotations

from pathlib import Path


def plot_cellrank_aggregate_fate_probabilities(
    adata,
    *,
    mode: str = "bar",
    cluster_key: str = "clusters",
    lineages=None,
    output_path: str | Path | None = None,
    **kwargs,
):
    """Thin wrapper for native CellRank aggregate fate probabilities."""

    import cellrank as cr
    import matplotlib.pyplot as plt

    fig = cr.pl.aggregate_fate_probabilities(
        adata,
        mode=mode,
        cluster_key=cluster_key,
        lineages=lineages,
        show=False,
        **kwargs,
    )
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.gcf().savefig(output_path, bbox_inches="tight", facecolor="white")
    return fig
