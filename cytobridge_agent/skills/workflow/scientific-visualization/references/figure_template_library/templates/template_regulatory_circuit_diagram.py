"""
Template: Regulatory Circuit Diagram
=====================================
Node-arrow diagram showing directed influence between gene modules.  Nodes
represent modules (colored circles with self-reinforcement values), arrows
represent inter-module influence (green = activation, red = suppression,
thickness ∝ magnitude).

Inspired by: CREST main figure Panel F (lineage tracing project), derived from
Jacobian perturbation analysis.

Required inputs
---------------
- modules : list[str] — module names.
- positions : dict[str, tuple(float, float)] — (x, y) position per module.
- influence_matrix : dict[(str, str), float] — influence from module A to
  module B.  Keys are (source, target) tuples.  Diagonal entries are
  self-reinforcement.
- title : str — panel title (e.g., "BM Early").

Optional inputs
---------------
- module_colors : dict[str, str] — color per module.
- node_radius : float — circle radius.  Default 0.12.
- activation_color : str — arrow color for positive influence.  Default green.
- suppression_color : str — arrow color for negative influence.  Default red.
- influence_threshold : float — minimum |influence| to draw an arrow.
  Default 0.001.
- vmax : float — influence value that maps to max arrow thickness.
  Default 0.03.
- max_lw : float — maximum arrow line width.  Default 3.5.
- show_values : bool — show numeric values on arrows.  Default True.
- figsize : tuple — figure size.  Default (4, 4).

Style rules (from style_playbook.md)
-------------------------------------
- Straight arrows (connectionstyle="arc3,rad=0").
- Green = activation, Red = suppression.
- Minimum thickness threshold to reduce clutter.
- Self-reinforcement value displayed inside the node.
- No axes, no frame. White background.
- Small multiples should share node positions exactly.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def plot_regulatory_circuit(
    modules,
    positions,
    influence_matrix,
    title="",
    module_colors=None,
    node_radius=0.12,
    activation_color="#2ca02c",
    suppression_color="#d62728",
    influence_threshold=0.001,
    vmax=0.03,
    max_lw=3.5,
    show_values=True,
    value_fontsize=5.5,
    figsize=(4, 4),
    dpi=300,
    ax=None,
):
    """Draw a single circuit diagram on the given axes (or create new figure)."""
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        created_fig = True
    else:
        fig = ax.figure

    if module_colors is None:
        default = ["#d62728", "#1f77b4", "#9467bd", "#2ca02c", "#ff7f0e"]
        module_colors = {m: default[i % len(default)] for i, m in enumerate(modules)}

    for mod in modules:
        x, y = positions[mod]
        circ = Circle((x, y), node_radius, fc=module_colors[mod], ec="white",
                       lw=1.5, alpha=0.85, zorder=5)
        ax.add_patch(circ)
        ax.text(x, y + 0.01, mod.replace("_", "\n"), ha="center", va="center",
                fontsize=6.5, fontweight="bold", color="white", zorder=6)

        diag = influence_matrix.get((mod, mod), 0.0)
        if abs(diag) > 1e-6:
            ax.text(x, y - node_radius + 0.03, f"{diag:+.3f}", ha="center",
                    va="top", fontsize=5, color="white", zorder=6)

    for src in modules:
        for tgt in modules:
            if src == tgt:
                continue
            v = influence_matrix.get((src, tgt), 0.0)
            if abs(v) < influence_threshold:
                continue

            lw = min(max_lw, max_lw * abs(v) / vmax)
            if lw < 0.15:
                continue
            colour = activation_color if v > 0 else suppression_color
            alpha = min(0.95, 0.4 + 0.55 * abs(v) / vmax)

            x0, y0 = positions[src]
            x1, y1 = positions[tgt]
            dx, dy = x1 - x0, y1 - y0
            dist = np.hypot(dx, dy)
            if dist < 1e-6:
                continue
            ux, uy = dx / dist, dy / dist
            shrink = node_radius + 0.015

            ax.annotate(
                "", xy=(x1 - ux * shrink, y1 - uy * shrink),
                xytext=(x0 + ux * shrink, y0 + uy * shrink),
                arrowprops=dict(
                    arrowstyle="->", color=colour, lw=lw, alpha=alpha,
                    shrinkA=0, shrinkB=0,
                    connectionstyle="arc3,rad=0",
                ),
                zorder=4,
            )

            if show_values:
                mx = (x0 + x1) / 2 + uy * 0.04
                my = (y0 + y1) / 2 - ux * 0.04
                ax.text(mx, my, f"{v:+.3f}", fontsize=value_fontsize,
                        ha="center", va="center", color=colour, alpha=0.9, zorder=7)

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal")
    ax.axis("off")

    if title:
        ax.set_title(title, fontsize=9, fontweight="bold", pad=6)

    return fig, ax


def plot_circuit_grid(
    modules,
    positions,
    conditions,
    module_colors=None,
    legend_text=None,
    figsize=(10, 10),
    dpi=300,
    save_path=None,
    **kwargs,
):
    """
    Draw a 2×2 grid of circuit diagrams for multiple conditions.

    Parameters
    ----------
    conditions : list of (influence_matrix, title) tuples.
    legend_text : list[str] or None — lines of legend text to place below.
    """
    n = len(conditions)
    ncols = min(n, 2)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=dpi)
    if n == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)

    for idx, (inf_mat, title) in enumerate(conditions):
        r, c = divmod(idx, ncols)
        plot_regulatory_circuit(
            modules, positions, inf_mat, title=title,
            module_colors=module_colors, ax=axes[r, c], **kwargs,
        )

    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].axis("off")

    if legend_text:
        legend_y = 0.02
        for i, line in enumerate(legend_text):
            fig.text(0.5, legend_y + (len(legend_text) - 1 - i) * 0.025, line,
                     ha="center", fontsize=7, fontstyle="italic")

    fig.tight_layout(rect=[0, 0.05 if legend_text else 0, 1, 1])

    if save_path:
        for ext in (".png", ".pdf"):
            fig.savefig(save_path + ext, dpi=dpi, bbox_inches="tight", facecolor="white")
    return fig, axes
