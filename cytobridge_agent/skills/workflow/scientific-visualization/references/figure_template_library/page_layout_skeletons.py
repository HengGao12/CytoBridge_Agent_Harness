#!/usr/bin/env python3
"""Reusable page-layout skeletons for manuscript result figures.

These helpers encode page geometry only. They do not assume any data object, plotting library, or
scientific question. The goal is to make the page skeleton reusable while keeping panel rendering
separate.
"""

from __future__ import annotations

from typing import Dict, Tuple

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def _new_page(figsize: Tuple[float, float] = (8.3, 11.0), constrained: bool = False):
    fig = plt.figure(figsize=figsize, constrained_layout=constrained, facecolor="white")
    return fig


def make_asymmetric_hero_page(
    figsize: Tuple[float, float] = (8.3, 11.0),
    hero_width: float = 1.55,
    bottom_height: float = 0.9,
) -> tuple[plt.Figure, Dict[str, plt.Axes]]:
    """Large left hero, compact right companions, lower support band."""

    fig = _new_page(figsize)
    gs = GridSpec(
        2,
        4,
        figure=fig,
        width_ratios=[hero_width, hero_width, 1.0, 1.0],
        height_ratios=[1.0, bottom_height],
        wspace=0.25,
        hspace=0.24,
    )
    axes = {
        "hero": fig.add_subplot(gs[0, :2]),
        "companion_top": fig.add_subplot(gs[0, 2]),
        "companion_right": fig.add_subplot(gs[0, 3]),
        "support_left": fig.add_subplot(gs[1, 0]),
        "support_mid": fig.add_subplot(gs[1, 1:3]),
        "support_right": fig.add_subplot(gs[1, 3]),
    }
    return fig, axes


def make_compare_band_page(
    n_top: int = 4,
    figsize: Tuple[float, float] = (8.3, 10.6),
    bottom_height: float = 0.88,
) -> tuple[plt.Figure, Dict[str, plt.Axes]]:
    """Shared-basis compare band on top, broad proof band below."""

    if n_top < 3:
        raise ValueError("n_top must be >= 3 for a compare-band page.")

    fig = _new_page(figsize)
    gs = GridSpec(
        2,
        n_top,
        figure=fig,
        height_ratios=[1.0, bottom_height],
        hspace=0.26,
        wspace=0.18,
    )
    axes: Dict[str, plt.Axes] = {}
    for i in range(n_top):
        axes[f"compare_{i+1}"] = fig.add_subplot(gs[0, i])
    split = max(1, n_top - 1)
    axes["proof_wide"] = fig.add_subplot(gs[1, :split])
    axes["proof_companion"] = fig.add_subplot(gs[1, split:])
    return fig, axes


def make_matrix_hero_page(
    figsize: Tuple[float, float] = (8.3, 10.8),
    hero_width: float = 1.4,
) -> tuple[plt.Figure, Dict[str, plt.Axes]]:
    """Matrix-led page with one hero matrix and smaller companions."""

    fig = _new_page(figsize)
    gs = GridSpec(
        2,
        3,
        figure=fig,
        width_ratios=[hero_width, hero_width, 1.0],
        height_ratios=[1.0, 0.92],
        hspace=0.24,
        wspace=0.20,
    )
    axes = {
        "matrix_hero": fig.add_subplot(gs[0, :2]),
        "summary_top": fig.add_subplot(gs[0, 2]),
        "matrix_support": fig.add_subplot(gs[1, :2]),
        "summary_bottom": fig.add_subplot(gs[1, 2]),
    }
    return fig, axes


def make_spatial_wall_anchor_page(
    figsize: Tuple[float, float] = (8.3, 10.8),
) -> tuple[plt.Figure, Dict[str, plt.Axes]]:
    """Five same-basis maps plus one non-spatial anchor."""

    fig = _new_page(figsize)
    gs = GridSpec(2, 3, figure=fig, hspace=0.18, wspace=0.16)
    axes = {
        "map_1": fig.add_subplot(gs[0, 0]),
        "map_2": fig.add_subplot(gs[0, 1]),
        "map_3": fig.add_subplot(gs[0, 2]),
        "map_4": fig.add_subplot(gs[1, 0]),
        "map_5": fig.add_subplot(gs[1, 1]),
        "anchor": fig.add_subplot(gs[1, 2]),
    }
    return fig, axes


def make_multiscale_interleave_page(
    figsize: Tuple[float, float] = (8.3, 11.0),
) -> tuple[plt.Figure, Dict[str, plt.Axes]]:
    """Top mixed hero band, middle support band, lower closure band."""

    fig = _new_page(figsize)
    gs = GridSpec(
        3,
        4,
        figure=fig,
        height_ratios=[1.0, 0.95, 0.78],
        hspace=0.28,
        wspace=0.22,
    )
    axes = {
        "hero_left": fig.add_subplot(gs[0, :2]),
        "hero_right_top": fig.add_subplot(gs[0, 2]),
        "hero_right_bottom": fig.add_subplot(gs[0, 3]),
        "mid_left": fig.add_subplot(gs[1, 0]),
        "mid_center": fig.add_subplot(gs[1, 1:3]),
        "mid_right": fig.add_subplot(gs[1, 3]),
        "closure_left": fig.add_subplot(gs[2, :2]),
        "closure_right": fig.add_subplot(gs[2, 2:]),
    }
    return fig, axes


def add_debug_slot_labels(axes: Dict[str, plt.Axes], fontsize: float = 9.0) -> None:
    """Mark slot names onto empty skeleton axes when prototyping page geometry."""

    for name, ax in axes.items():
        ax.text(
            0.5,
            0.5,
            name,
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=fontsize,
            color="#666666",
            fontweight="bold",
        )


__all__ = [
    "make_asymmetric_hero_page",
    "make_compare_band_page",
    "make_matrix_hero_page",
    "make_spatial_wall_anchor_page",
    "make_multiscale_interleave_page",
    "add_debug_slot_labels",
]
