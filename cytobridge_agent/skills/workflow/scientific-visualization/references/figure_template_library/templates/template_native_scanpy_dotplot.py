from __future__ import annotations

from pathlib import Path


def plot_scanpy_dotplot(
    adata,
    *,
    var_names,
    groupby: str,
    output_path: str | Path | None = None,
    **kwargs,
):
    """Thin wrapper for native Scanpy dotplot."""

    import matplotlib.pyplot as plt
    import scanpy as sc

    sc.pl.dotplot(
        adata,
        var_names=var_names,
        groupby=groupby,
        show=False,
        **kwargs,
    )
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.gcf().savefig(output_path, bbox_inches="tight", facecolor="white")
    return plt.gcf()
