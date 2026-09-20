from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd


def plot_bipartite_pathway_network(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    *,
    node_id: str = "node",
    x: str = "x",
    y: str = "y",
    label: str = "label",
    fill: str = "fill",
    size: str = "size",
    shape: str = "shape",
    source: str = "source",
    target: str = "target",
    edge_color: str | None = None,
    fill_palette: dict | None = None,
    edge_palette: dict | None = None,
    output_path: str | Path | None = None,
):
    """Bipartite/pathway network inspired by iPOP Aging ggraph panels."""

    g = nx.Graph()
    for _, row in nodes.iterrows():
        g.add_node(row[node_id], **row.to_dict())
    for _, row in edges.iterrows():
        g.add_edge(row[source], row[target], **row.to_dict())

    pos = {row[node_id]: (row[x], row[y]) for _, row in nodes.iterrows()}
    fig, ax = plt.subplots(figsize=(8.6, 5.0), dpi=300, facecolor="white")
    ax.set_facecolor("white")

    if edge_color is not None:
        for level, sub in edges.groupby(edge_color):
            nx.draw_networkx_edges(
                g,
                pos,
                edgelist=[(r[source], r[target]) for _, r in sub.iterrows()],
                edge_color=(edge_palette[level] if edge_palette and level in edge_palette else "#9e9e9e"),
                width=1.0,
                alpha=0.65,
                ax=ax,
            )
    else:
        nx.draw_networkx_edges(g, pos, edge_color="#9e9e9e", width=1.0, alpha=0.65, ax=ax)

    marker_map = {"circle": "o", "square": "s", "diamond": "D", "triangle": "^"}
    for level, sub in nodes.groupby(shape):
        ax.scatter(
            sub[x],
            sub[y],
            s=sub[size].astype(float) * 18.0,
            c=[fill_palette[v] if fill_palette and v in fill_palette else "#bdbdbd" for v in sub[fill]],
            marker=marker_map.get(level, "o"),
            edgecolors="white",
            linewidths=0.7,
            alpha=0.95,
            zorder=3,
        )

    for _, row in nodes.iterrows():
        if label in row and pd.notna(row[label]):
            ax.text(row[x], row[y], str(row[label]), fontsize=8, ha="left", va="center")

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
