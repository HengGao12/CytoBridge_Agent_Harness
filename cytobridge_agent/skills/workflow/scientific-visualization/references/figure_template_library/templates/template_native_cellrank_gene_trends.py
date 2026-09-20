from __future__ import annotations

from pathlib import Path


def plot_cellrank_gene_trends(
    adata,
    model,
    genes,
    *,
    time_key: str,
    lineages=None,
    output_path: str | Path | None = None,
    **kwargs,
):
    """Thin wrapper for native CellRank gene-trend plots."""

    import cellrank as cr
    import matplotlib.pyplot as plt

    fig = cr.pl.gene_trends(
        adata,
        model=model,
        genes=genes,
        time_key=time_key,
        lineages=lineages,
        show=False,
        **kwargs,
    )
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.gcf().savefig(output_path, bbox_inches="tight", facecolor="white")
    return fig
