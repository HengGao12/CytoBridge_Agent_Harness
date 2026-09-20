from __future__ import annotations

from pathlib import Path


def plot_scanpy_embedding(
    adata,
    *,
    basis: str = "umap",
    color=None,
    frameon: bool = False,
    legend_loc: str | None = None,
    ncols: int = 1,
    wspace: float = 0.25,
    output_path: str | Path | None = None,
):
    """Thin wrapper for native scanpy embedding plots.

    Intended for cases where native scanpy plotting is already the right grammar.
    """

    import matplotlib.pyplot as plt
    import scanpy as sc

    sc.settings.set_figure_params(facecolor="white")
    fig = sc.pl.embedding(
        adata,
        basis=basis,
        color=color,
        frameon=frameon,
        legend_loc=legend_loc,
        ncols=ncols,
        wspace=wspace,
        return_fig=True,
        show=False,
    )
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    return fig
