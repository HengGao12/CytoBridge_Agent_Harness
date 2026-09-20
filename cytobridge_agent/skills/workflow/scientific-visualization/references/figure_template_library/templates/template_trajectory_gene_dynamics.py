"""
Template: Along-Trajectory Gene Dynamics Heatmap
=================================================
Show how gene expression changes along a generated trajectory, displayed as a
z-scored heatmap with genes on the y-axis and pseudo-time on the x-axis.

Inspired by: CREST main figure Panel E (lineage tracing project).

Required inputs
---------------
- expr_matrix : ndarray (n_genes, n_steps) — mean expression per gene per
  trajectory time step.  Should already be subset to the cells/genes of
  interest and averaged across cells.
- gene_names : list[str] — gene names (length = n_genes).

Optional inputs
---------------
- cmap : str — colormap.  Default "RdBu_r".
- vmin, vmax : float — color limits.  Default ±2 (z-score).
- figsize : tuple — figure size.  Default (6, 4).
- title : str — panel title.
- xlabel : str — x-axis label.  Default "Trajectory pseudo-time".

Style rules (from style_playbook.md)
-------------------------------------
- Diverging colormap centered at 0.
- Gene names in italic, ≥ 7pt.
- x-axis label descriptive.
- Colorbar compact, labeled "z-score".
"""

import numpy as np
import matplotlib.pyplot as plt


def zscore_columns(mat):
    """Z-score each column (gene) independently, then return transposed."""
    mu = mat.mean(axis=0, keepdims=True)
    sd = mat.std(axis=0, keepdims=True)
    sd[sd < 1e-12] = 1.0
    return ((mat - mu) / sd).T


def plot_trajectory_gene_dynamics(
    expr_matrix,
    gene_names,
    cmap="RdBu_r",
    vmin=-2.0,
    vmax=2.0,
    figsize=(6, 4),
    title="Gene expression along trajectory",
    xlabel="Trajectory pseudo-time",
    dpi=300,
    save_path=None,
):
    """
    Parameters
    ----------
    expr_matrix : ndarray (n_steps, n_genes) — rows=time, cols=genes.
        Will be z-scored per gene before plotting.
    gene_names : list[str]
    cmap, vmin, vmax, figsize, title, xlabel, dpi, save_path
    """
    z = zscore_columns(expr_matrix)  # (n_genes, n_steps)
    n_genes, n_steps = z.shape

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    im = ax.imshow(z, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation="bilinear", origin="upper")

    ax.set_yticks(range(n_genes))
    ax.set_yticklabels(gene_names, fontsize=7, fontstyle="italic")
    ax.set_xlabel(xlabel, fontsize=8)

    n_ticks = min(5, n_steps)
    tick_pos = np.linspace(0, n_steps - 1, n_ticks).astype(int)
    tick_labels = [f"{t / (n_steps - 1):.1f}" for t in tick_pos]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels, fontsize=7)

    ax.set_title(title, fontsize=9, fontweight="bold", pad=6)

    cbar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("z-score", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    if save_path:
        for ext in (".png", ".pdf"):
            fig.savefig(save_path + ext, dpi=dpi, bbox_inches="tight", facecolor="white")
    return fig, ax
