from __future__ import annotations

from pathlib import Path


def plot_squidpy_nhood_enrichment(
    adata,
    *,
    cluster_key: str,
    output_path: str | Path | None = None,
    **kwargs,
):
    """Thin wrapper for native squidpy neighborhood enrichment plots."""

    import matplotlib.pyplot as plt
    import squidpy as sq

    fig = sq.pl.nhood_enrichment(
        adata,
        cluster_key=cluster_key,
        show=False,
        **kwargs,
    )
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.gcf().savefig(output_path, bbox_inches="tight", facecolor="white")
    return fig
