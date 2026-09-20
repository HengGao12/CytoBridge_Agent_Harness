from __future__ import annotations

from pathlib import Path


def plot_squidpy_spatial_scatter(
    adata,
    *,
    color=None,
    img: bool | None = None,
    output_path: str | Path | None = None,
    **kwargs,
):
    """Thin wrapper for native Squidpy spatial scatter / image overlay panels."""

    import matplotlib.pyplot as plt
    import squidpy as sq

    sq.pl.spatial_scatter(
        adata,
        color=color,
        img=img,
        show=False,
        **kwargs,
    )
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.gcf().savefig(output_path, bbox_inches="tight", facecolor="white")
    return plt.gcf()
