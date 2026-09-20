#!/usr/bin/env python3
from __future__ import annotations

import matplotlib as mpl


BG = "#FFFFFF"
INK = "#201B17"
GRID = "#E4E7EB"


def apply_global_style() -> None:
    """Apply a stable manuscript default.

    This is intentionally conservative. The goal is not to impose a house style for every
    figure, but to prevent random typography and frame decisions across panels.
    """

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.4,
            "axes.titlesize": 9.4,
            "axes.titleweight": "bold",
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.8,
            "legend.fontsize": 7.2,
            "axes.facecolor": BG,
            "figure.facecolor": BG,
            "savefig.facecolor": BG,
            "savefig.edgecolor": BG,
            "axes.edgecolor": "#444444",
            "axes.linewidth": 1.0,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def finish_axes_manifold(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_visible(False)


def finish_axes_stat(ax, grid_axis: str = "y") -> None:
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def finish_axes_time(ax) -> None:
    ax.grid(axis="both", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def add_panel_letter(fig, ax, letter: str, dx: float = 0.026, dy: float = 0.01, color: str = INK) -> None:
    bbox = ax.get_position()
    fig.text(bbox.x0 - dx, bbox.y1 + dy, letter, fontsize=11.5, fontweight="bold", color=color)


def _collapse_titles(ax, title_loc: str) -> None:
    left = ax.get_title(loc="left")
    center = ax.get_title(loc="center")
    right = ax.get_title(loc="right")
    title = left or center or right
    ax.set_title("", loc="left")
    ax.set_title("", loc="center")
    ax.set_title("", loc="right")
    if title:
        ax.set_title(title, loc=title_loc, pad=4, fontweight="bold")


def beautify_manifold_panel(ax, title_loc: str = "left") -> None:
    _collapse_titles(ax, title_loc)
    finish_axes_manifold(ax)


def beautify_distribution_panel(ax, horizontal: bool = False) -> None:
    _collapse_titles(ax, "left")
    finish_axes_stat(ax, grid_axis="x" if horizontal else "y")


def beautify_time_panel(ax) -> None:
    _collapse_titles(ax, "left")
    finish_axes_time(ax)


def beautify_bar_panel(ax, horizontal: bool = False) -> None:
    _collapse_titles(ax, "left")
    finish_axes_stat(ax, grid_axis="x" if horizontal else "y")
