#!/usr/bin/env python3
"""More opinionated style helpers for manuscript result pages.

This module sits above `style_defaults.py`. The defaults prevent random styling mistakes; these
presets aim to make page-level rendering more consistent for main figures and supplement figures.
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.legend import Legend

from style_defaults import GRID, INK, apply_global_style


def apply_result_page_style(
    *,
    page_type: str = "main",
    density: str = "normal",
) -> None:
    """Apply a stronger manuscript preset.

    Parameters
    ----------
    page_type
        `main`, `supp`, `dashboard`, or `landscape`
    density
        `normal` or `dense`
    """

    apply_global_style()

    dense = density == "dense"
    if page_type == "main":
        mpl.rcParams.update(
            {
                "font.size": 8.5,
                "axes.titlesize": 9.6,
                "axes.labelsize": 8.6,
                "xtick.labelsize": 7.5,
                "ytick.labelsize": 7.5,
                "legend.fontsize": 7.2,
                "axes.linewidth": 0.95,
                "lines.linewidth": 1.55,
                "savefig.pad_inches": 0.03,
            }
        )
    elif page_type == "supp":
        mpl.rcParams.update(
            {
                "font.size": 8.2,
                "axes.titlesize": 9.0,
                "axes.labelsize": 8.2,
                "xtick.labelsize": 7.2,
                "ytick.labelsize": 7.2,
                "legend.fontsize": 7.0,
                "savefig.pad_inches": 0.03,
            }
        )
    elif page_type == "dashboard":
        mpl.rcParams.update(
            {
                "font.size": 7.9,
                "axes.titlesize": 8.8,
                "axes.labelsize": 7.9,
                "xtick.labelsize": 6.8,
                "ytick.labelsize": 6.8,
                "legend.fontsize": 6.7,
                "lines.linewidth": 1.3,
                "axes.linewidth": 0.85,
                "savefig.pad_inches": 0.025,
            }
        )
    elif page_type == "landscape":
        mpl.rcParams.update(
            {
                "font.size": 8.3,
                "axes.titlesize": 9.1,
                "axes.labelsize": 8.3,
                "xtick.labelsize": 7.2,
                "ytick.labelsize": 7.2,
            }
        )
    else:
        raise ValueError(f"Unknown page_type: {page_type}")

    if dense:
        mpl.rcParams.update(
            {
                "font.size": mpl.rcParams["font.size"] - 0.2,
                "axes.titlesize": mpl.rcParams["axes.titlesize"] - 0.2,
                "xtick.labelsize": mpl.rcParams["xtick.labelsize"] - 0.2,
                "ytick.labelsize": mpl.rcParams["ytick.labelsize"] - 0.2,
            }
        )


def make_side_colorbar_axis(
    fig: plt.Figure,
    ax: plt.Axes,
    *,
    side: str = "right",
    pad: float = 0.008,
    thickness: float = 0.012,
    shrink: float = 1.0,
):
    """Create a page-stable colorbar axis next to an existing axes."""

    bbox = ax.get_position()
    height = bbox.height * shrink
    y0 = bbox.y0 + 0.5 * (bbox.height - height)
    if side == "right":
        cax = fig.add_axes([bbox.x1 + pad, y0, thickness, height])
    elif side == "left":
        cax = fig.add_axes([bbox.x0 - pad - thickness, y0, thickness, height])
    else:
        raise ValueError("side must be 'right' or 'left'")
    return cax


def attach_side_colorbar(
    fig: plt.Figure,
    mappable,
    ax: plt.Axes,
    *,
    label: str | None = None,
    side: str = "right",
    pad: float = 0.008,
    thickness: float = 0.012,
    shrink: float = 1.0,
):
    cax = make_side_colorbar_axis(
        fig,
        ax,
        side=side,
        pad=pad,
        thickness=thickness,
        shrink=shrink,
    )
    cb = fig.colorbar(mappable, cax=cax)
    cb.outline.set_linewidth(0.6)
    cb.ax.tick_params(labelsize=7.0, width=0.6, length=2.5)
    if label:
        cb.set_label(label, fontsize=8.0)
    return cb


def add_panel_letters(
    fig: plt.Figure,
    axes: Mapping[str, plt.Axes],
    *,
    labels: Sequence[str] | None = None,
    dx: float = 0.022,
    dy: float = 0.01,
) -> None:
    if labels is None:
        labels = [chr(ord("A") + i) for i in range(len(axes))]
    for label, ax in zip(labels, axes.values()):
        bbox = ax.get_position()
        fig.text(
            bbox.x0 - dx,
            bbox.y1 + dy,
            label,
            fontsize=11.2,
            fontweight="bold",
            color=INK,
        )


def set_axis_title(ax: plt.Axes, title: str, *, loc: str = "left", pad: float = 4.0) -> None:
    ax.set_title(title, loc=loc, pad=pad, fontweight="bold", color=INK)


def quiet_legend(legend: Legend | None, *, text_color: str = INK) -> None:
    if legend is None:
        return
    legend.set_frame_on(False)
    for txt in legend.get_texts():
        txt.set_color(text_color)
    title = legend.get_title()
    if title is not None:
        title.set_color(text_color)


def finish_page(
    fig: plt.Figure,
    *,
    top: float = 0.965,
    bottom: float = 0.06,
    left: float = 0.06,
    right: float = 0.97,
) -> None:
    """Apply a stable final subplot region for manually composed pages."""

    fig.subplots_adjust(top=top, bottom=bottom, left=left, right=right)


def add_reference_line(ax: plt.Axes, *, axis: str = "y", value: float = 0.0, color: str = "#8A8A8A") -> None:
    if axis == "y":
        ax.axhline(value, color=color, linewidth=0.9, linestyle="--", zorder=0)
    elif axis == "x":
        ax.axvline(value, color=color, linewidth=0.9, linestyle="--", zorder=0)
    else:
        raise ValueError("axis must be 'x' or 'y'")


def strengthen_grid(ax: plt.Axes, *, axis: str = "y") -> None:
    ax.grid(axis=axis, color=GRID, linewidth=0.85)
    ax.set_axisbelow(True)


__all__ = [
    "apply_result_page_style",
    "make_side_colorbar_axis",
    "attach_side_colorbar",
    "add_panel_letters",
    "set_axis_title",
    "quiet_legend",
    "finish_page",
    "add_reference_line",
    "strengthen_grid",
]
