from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd


def plot_labeled_network_summary(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    *,
    node_id: str = "node",
    x: str = "x",
    y: str = "y",
    label: str = "label",
    color: str = "color",
    size: str = "size",
    source: str = "source",
    target: str = "target",
    edge_weight: str | None = None,
    palette: dict | None = None,
    output_path: str | Path | None = None,
):
    """Label-heavy summary network inspired by iPOP Aging cluster-summary graphs."""

    g = nx.Graph()
    for _, row in nodes.iterrows():
        g.add_node(row[node_id], **row.to_dict())
    for _, row in edges.iterrows():
        g.add_edge(row[source], row[target], **row.to_dict())
    pos = {row[node_id]: (row[x], row[y]) for _, row in nodes.iterrows()}

    fig, ax = plt.subplots(figsize=(7.4, 6.4), dpi=300, facecolor="white")
    ax.set_facecolor("white")

    widths = None
    if edge_weight is not None and edge_weight in edges.columns:
        vals = edges[edge_weight].astype(float)
        lo, hi = float(vals.min()), float(vals.max())
        if hi > lo:
            widths = 0.5 + 2.5 * (vals - lo) / (hi - lo)
        else:
            widths = [1.4] * len(vals)
    nx.draw_networkx_edges(
        g,
        pos,
        edge_color="#9e9e9e",
        width=widths if widths is not None else 1.2,
        alpha=0.7,
        ax=ax,
    )

    node_colors = [palette[v] if palette and v in palette else "#4c78a8" for v in nodes[color]]
    node_sizes = nodes[size].astype(float).to_numpy() * 18.0
    ax.scatter(nodes[x], nodes[y], s=node_sizes, c=node_colors, edgecolors="white", linewidths=0.8, zorder=3)

    for _, row in nodes.iterrows():
        ax.text(
            row[x],
            row[y],
            str(row[label]),
            fontsize=8.5,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.88),
            zorder=4,
        )

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    return fig
