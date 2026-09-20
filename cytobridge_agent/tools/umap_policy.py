"""UMAP parameter policy helpers.

This module centralizes adaptive UMAP/neighbors defaults so downstream code
does not hardcode a single parameter set across all datasets.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional


def _clamp_int(value: Any, low: int, high: int, default: int) -> int:
    try:
        iv = int(value)
    except Exception:
        iv = int(default)
    return max(low, min(high, iv))


def _clamp_float(value: Any, low: float, high: float, default: float) -> float:
    try:
        fv = float(value)
    except Exception:
        fv = float(default)
    return max(low, min(high, fv))


def auto_n_neighbors(n_obs: int) -> int:
    """Adaptive neighbors count based on dataset size."""
    n = int(max(1, n_obs))
    # 1k->25, 10k->30, 100k->35, 1M->40 (clamped)
    raw = 10.0 + 5.0 * math.log10(max(10, n))
    return _clamp_int(round(raw), low=12, high=80, default=30)


def default_min_dist(viz_goal: str, quality_preset: str) -> float:
    goal = str(viz_goal or "publication").strip().lower()
    quality = str(quality_preset or "publication").strip().lower()
    if goal == "exploratory":
        return 0.5
    if goal == "presentation":
        return 0.2
    if quality == "fast":
        return 0.45
    return 0.3


def default_metric(use_rep: Optional[str]) -> str:
    rep = str(use_rep or "").strip()
    if rep in {"X_latent", "X_pca"}:
        return "cosine"
    return "euclidean"


def build_umap_policy(
    *,
    n_obs: int,
    use_rep: Optional[str],
    quality_preset: str = "publication",
    viz_goal: str = "publication",
    mode: str = "auto",
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build normalized neighbors+UMAP config."""
    normalized_mode = str(mode or "auto").strip().lower()
    if normalized_mode not in {"auto", "manual"}:
        normalized_mode = "auto"

    over = dict(overrides or {})
    if "use_rep" in over and over.get("use_rep") is not None:
        rep = str(over.get("use_rep")).strip()
    else:
        rep = str(use_rep or "").strip() or None

    base_n = 30 if normalized_mode == "manual" else auto_n_neighbors(int(max(1, n_obs)))
    base_min_dist = 0.3 if normalized_mode == "manual" else default_min_dist(viz_goal, quality_preset)
    base_spread = 1.0
    base_random_state = 0
    base_metric = default_metric(rep)

    n_neighbors = _clamp_int(over.get("n_neighbors", base_n), low=5, high=200, default=base_n)
    min_dist = _clamp_float(over.get("min_dist", base_min_dist), low=0.0, high=0.99, default=base_min_dist)
    spread = _clamp_float(over.get("spread", base_spread), low=0.1, high=10.0, default=base_spread)
    random_state = _clamp_int(over.get("random_state", base_random_state), low=0, high=2**31 - 1, default=base_random_state)
    metric = str(over.get("metric", base_metric) or base_metric).strip().lower()
    if not metric:
        metric = base_metric

    return {
        "version": 1,
        "mode": normalized_mode,
        "quality_preset": str(quality_preset or "publication"),
        "viz_goal": str(viz_goal or "publication"),
        "neighbors": {
            "use_rep": rep,
            "n_neighbors": int(n_neighbors),
            "metric": metric,
        },
        "umap": {
            "min_dist": float(min_dist),
            "spread": float(spread),
            "random_state": int(random_state),
        },
        "overrides": over,
    }


def compact_policy_signature(policy: Dict[str, Any]) -> Dict[str, Any]:
    """Return deterministic subset for equality checks and provenance."""
    ncfg = (policy or {}).get("neighbors", {}) if isinstance(policy, dict) else {}
    ucfg = (policy or {}).get("umap", {}) if isinstance(policy, dict) else {}
    return {
        "version": int((policy or {}).get("version", 1)),
        "mode": str((policy or {}).get("mode", "auto")),
        "use_rep": str(ncfg.get("use_rep") or ""),
        "n_neighbors": int(ncfg.get("n_neighbors", 30)),
        "metric": str(ncfg.get("metric", "euclidean")),
        "min_dist": float(ucfg.get("min_dist", 0.3)),
        "spread": float(ucfg.get("spread", 1.0)),
        "random_state": int(ucfg.get("random_state", 0)),
    }
