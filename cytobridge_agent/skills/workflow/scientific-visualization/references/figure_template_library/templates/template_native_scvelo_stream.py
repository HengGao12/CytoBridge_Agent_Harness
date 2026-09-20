from __future__ import annotations

from pathlib import Path


def plot_scvelo_stream(
    adata,
    *,
    basis: str = "umap",
    color=None,
    legend_loc: str | None = "right margin",
    output_path: str | Path | None = None,
    **kwargs,
):
    """Thin wrapper for native scvelo stream plots."""

    import scvelo as scv

    fig = scv.pl.velocity_embedding_stream(
        adata,
        basis=basis,
        color=color,
        legend_loc=legend_loc,
        show=False,
        return_fig=True,
        **kwargs,
    )
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    return fig
