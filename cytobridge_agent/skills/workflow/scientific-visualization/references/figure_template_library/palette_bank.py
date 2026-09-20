#!/usr/bin/env python3
"""Shared palette bank for manuscript figures.

This module centralizes palette classes that repeatedly appear across the figure-template
library. The goal is not to freeze one visual identity forever; the goal is to avoid ad hoc
color choices and to preserve source-backed palette logic:

- quiet context greys for background geometry
- compact categorical atlas palettes
- two-pole perturbation palettes
- stable method-comparison accents
- sequential and diverging palettes for scalar and matrix panels
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap


QUIET_GREY_BG = "#D7DDE2"
QUIET_GREY_FG = "#A8B2BC"
INK = "#201B17"


def _take(colors: Sequence[str], n: int) -> List[str]:
    if n <= len(colors):
        return list(colors[:n])
    out = list(colors)
    while len(out) < n:
        out.extend(colors)
    return out[:n]


def quiet_context_palette() -> Dict[str, str]:
    """Background/foreground greys for manifold and spatial context."""

    return {
        "background": QUIET_GREY_BG,
        "background_dark": QUIET_GREY_FG,
        "outline": "#7B848D",
        "ink": INK,
    }


def atlas_categorical_palette(n: int = 8) -> List[str]:
    """Compact categorical palette for atlas states or lineages.

    Muted but distinct, designed for medium-cardinality biological categories.
    """

    base = [
        "#C65B4B",  # muted warm red
        "#4A6FA5",  # steel blue
        "#D39B2A",  # ochre
        "#4A8F6C",  # green
        "#8A6BBE",  # restrained violet
        "#C97A56",  # terracotta
        "#5E8EA0",  # teal-blue
        "#9A7A52",  # brown
        "#D86B91",  # muted magenta
        "#6C7C4A",  # olive
    ]
    return _take(base, n)


def compare_method_palette(labels: Iterable[str] | None = None) -> Dict[str, str]:
    """Stable accents for method-comparison panels.

    The mapping prefers `Ours` as the strongest accent and keeps baselines distinct but quieter.
    Unknown labels fall back to neutral greys or cycle accents if needed.
    """

    mapping = {
        "Clone": "#1F1F1F",
        "Ground truth": "#1F1F1F",
        "Truth": "#1F1F1F",
        "Ours": "#C44E38",
        "DeepRUOT": "#C44E38",
        "FM": "#C44E38",
        "CellRank": "#4A6FA5",
        "scDiffEq": "#4A8F6C",
        "VGFM": "#C97A56",
        "TIGON": "#8A6BBE",
        "OT-CFM": "#5E8EA0",
        "Moslin": "#A46A2A",
        "Moscot": "#6A7FAE",
        "Baseline": "#7B848D",
        "Other": "#7B848D",
    }
    if labels is None:
        return mapping
    out: Dict[str, str] = {}
    fallback = atlas_categorical_palette(12)
    j = 0
    for label in labels:
        if label in mapping:
            out[label] = mapping[label]
        else:
            out[label] = fallback[j % len(fallback)]
            j += 1
    return out


def perturbation_diverging_palette() -> Dict[str, object]:
    """Signed palette for perturbation panels."""

    positive = ["#F4C7B8", "#E7967C", "#D96C58", "#C44E38"]
    negative = ["#BFD1EA", "#8FB1DA", "#5E8FC7", "#3E6FA9"]
    return {
        "positive_steps": positive,
        "negative_steps": negative,
        "positive_core": positive[-1],
        "negative_core": negative[-1],
        "control": "#D7DDE2",
        "neutral": "#9AA4AE",
        "cmap": LinearSegmentedColormap.from_list(
            "perturbation_diverging",
            [negative[-1], "#F7F7F7", positive[-1]],
        ),
    }


def sequential_scalar_palette(name: str = "warm_signal"):
    """Sequential colormaps for scalar overlays."""

    banks = {
        "warm_signal": LinearSegmentedColormap.from_list(
            "warm_signal",
            ["#F8F5F1", "#E8C89A", "#C97A56", "#8E3B2E"],
        ),
        "cool_signal": LinearSegmentedColormap.from_list(
            "cool_signal",
            ["#F4F7FB", "#B8D0EA", "#5E8FC7", "#234A73"],
        ),
        "green_signal": LinearSegmentedColormap.from_list(
            "green_signal",
            ["#F5F8F4", "#BED7C5", "#6EA07D", "#2E5E43"],
        ),
        "grey_signal": LinearSegmentedColormap.from_list(
            "grey_signal",
            ["#FBFBFB", "#D6DDE2", "#93A0AB", "#4D5963"],
        ),
    }
    if name not in banks:
        raise KeyError(f"Unknown sequential palette: {name}")
    return banks[name]


def diverging_matrix_palette(name: str = "soft_balance"):
    """Diverging colormaps for centered matrices or signed heatmaps."""

    banks = {
        "soft_balance": LinearSegmentedColormap.from_list(
            "soft_balance",
            ["#2E5E8A", "#B8D0EA", "#F7F7F7", "#E8B9AE", "#A43E33"],
        ),
        "cool_warm_quiet": LinearSegmentedColormap.from_list(
            "cool_warm_quiet",
            ["#31597D", "#A9C2DE", "#FAFAFA", "#E5C1B7", "#A14A3B"],
        ),
        "teal_ochre": LinearSegmentedColormap.from_list(
            "teal_ochre",
            ["#2D6E73", "#A8D0D0", "#F7F6F1", "#E9C68B", "#9B6A1B"],
        ),
    }
    if name not in banks:
        raise KeyError(f"Unknown diverging palette: {name}")
    return banks[name]


def attention_matrix_palette(mode: str = "magnitude"):
    """Palettes for communication/attention matrices."""

    if mode == "magnitude":
        return sequential_scalar_palette("cool_signal")
    if mode == "asymmetry":
        return diverging_matrix_palette("soft_balance")
    raise KeyError(f"Unknown attention palette mode: {mode}")


def benchmark_palette() -> Dict[str, str]:
    """Compact palette for benchmark bars/boxes where method families matter."""

    return {
        "ours": "#C44E38",
        "baseline_blue": "#4A6FA5",
        "baseline_green": "#4A8F6C",
        "baseline_teal": "#5E8EA0",
        "baseline_orange": "#C97A56",
        "baseline_purple": "#8A6BBE",
        "neutral": "#8D98A3",
        "truth": "#1F1F1F",
    }


def gene_program_pole_palette() -> Dict[str, str]:
    """Two-pole palette for opposing biological programs."""

    return {
        "program_a": "#C44E38",
        "program_a_light": "#E8B2A6",
        "program_b": "#3E6FA9",
        "program_b_light": "#BED1EA",
        "buffer": "#9CA7B1",
        "neutral": "#D7DDE2",
    }


def to_hex_list(cmap, n: int = 5) -> List[str]:
    return [mpl.colors.to_hex(cmap(i / max(n - 1, 1))) for i in range(n)]


__all__ = [
    "QUIET_GREY_BG",
    "QUIET_GREY_FG",
    "INK",
    "quiet_context_palette",
    "atlas_categorical_palette",
    "compare_method_palette",
    "perturbation_diverging_palette",
    "sequential_scalar_palette",
    "diverging_matrix_palette",
    "attention_matrix_palette",
    "benchmark_palette",
    "gene_program_pole_palette",
    "to_hex_list",
]
