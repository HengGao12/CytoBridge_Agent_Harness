"""Shared W1 backend selection and computation utilities."""

from __future__ import annotations

import math
import os
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import ot


DEFAULT_W1_AUTO_PAIR_THRESHOLD = 16_000_000
DEFAULT_SLICED_W1_PROJECTIONS = 200
DEFAULT_SLICED_W1_SEED = 1729
DEFAULT_GEOMLOSS_P = 1
DEFAULT_GEOMLOSS_BLUR = 1.0


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return float(default)
    try:
        value = float(raw)
    except Exception:
        return float(default)
    return value if math.isfinite(value) and value > 0 else float(default)


def _env_int(name: str, default: int) -> int:
    value = _env_float(name, float(default))
    try:
        return int(value)
    except Exception:
        return int(default)


def _geomloss_online_available() -> bool:
    try:
        import pykeops  # noqa: F401
        from geomloss import SamplesLoss  # noqa: F401
    except Exception:
        return False
    return True


def select_w1_backend_policy(
    pair_sizes: Iterable[Tuple[int, int]],
    *,
    requested_backend: Optional[str] = None,
    pair_threshold: Optional[float] = None,
    selection_scope: str = "evaluation_panel",
) -> Dict[str, Any]:
    """Select one W1 backend for an entire evaluation panel.

    `pair_sizes` should contain every observed/predicted distribution pair that
    will be evaluated in the panel. Backend selection is intentionally panel-wide:
    if any gap is large, every gap in the same panel uses the same backend.
    """

    normalized_pairs: List[Tuple[int, int]] = []
    for left, right in pair_sizes:
        try:
            n_left = max(0, int(left))
            n_right = max(0, int(right))
        except Exception:
            continue
        normalized_pairs.append((n_left, n_right))
    pair_counts = [int(left) * int(right) for left, right in normalized_pairs]
    max_pair_count = max(pair_counts) if pair_counts else 0
    threshold = float(
        pair_threshold
        if pair_threshold is not None
        else _env_float("CYTOBRIDGE_W1_AUTO_PAIR_THRESHOLD", DEFAULT_W1_AUTO_PAIR_THRESHOLD)
    )
    requested = (requested_backend or os.getenv("CYTOBRIDGE_W1_BACKEND") or "auto").strip().lower()
    requested = {
        "pot": "exact",
        "emd": "exact",
        "geomloss": "geomloss_sinkhorn_online",
        "sinkhorn": "geomloss_sinkhorn_online",
        "sliced": "sliced_w1",
    }.get(requested, requested)
    if requested not in {"auto", "exact", "geomloss_sinkhorn_online", "sliced_w1"}:
        requested = "auto"

    reason = ""
    backend = requested
    exact = backend == "exact"
    if requested == "auto":
        if max_pair_count <= threshold:
            backend = "exact"
            exact = True
            reason = "max_pair_count_within_threshold"
        elif _geomloss_online_available():
            backend = "geomloss_sinkhorn_online"
            exact = False
            reason = "max_pair_count_exceeds_threshold"
        else:
            backend = "sliced_w1"
            exact = False
            reason = "max_pair_count_exceeds_threshold_geomloss_unavailable"
    else:
        reason = "explicit_backend"
        exact = backend == "exact"

    params: Dict[str, Any] = {}
    if backend == "geomloss_sinkhorn_online":
        params = {
            "loss": "sinkhorn",
            "p": _env_int("CYTOBRIDGE_W1_GEOMLOSS_P", DEFAULT_GEOMLOSS_P),
            "blur": _env_float("CYTOBRIDGE_W1_GEOMLOSS_BLUR", DEFAULT_GEOMLOSS_BLUR),
            "backend": "online",
        }
    elif backend == "sliced_w1":
        params = {
            "n_projections": _env_int("CYTOBRIDGE_W1_SLICED_PROJECTIONS", DEFAULT_SLICED_W1_PROJECTIONS),
            "seed": _env_int("CYTOBRIDGE_W1_SLICED_SEED", DEFAULT_SLICED_W1_SEED),
        }

    return {
        "w1_backend": backend,
        "w1_backend_requested": requested,
        "w1_backend_exact": bool(exact),
        "w1_backend_selection_scope": selection_scope,
        "w1_backend_selection_reason": reason,
        "w1_backend_threshold_pair_count": threshold,
        "w1_backend_max_pair_count": int(max_pair_count),
        "w1_backend_pair_counts": [int(value) for value in pair_counts],
        "w1_backend_pair_sizes": [[int(left), int(right)] for left, right in normalized_pairs],
        "w1_backend_params": params,
    }


def w1_backend_metric_metadata(policy: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in dict(policy or {}).items()
        if str(key).startswith("w1_backend")
    }


def _normalize_weights(weights: Optional[Sequence[float]], n: int) -> np.ndarray:
    if weights is None:
        return np.ones(n, dtype=np.float64) / float(max(n, 1))
    arr = np.asarray(weights, dtype=np.float64).reshape(-1)
    if arr.shape[0] != n:
        raise ValueError(f"Weight length {arr.shape[0]} does not match point count {n}.")
    total = float(arr.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Weights must have positive finite total mass.")
    return arr / total


def _exact_w1(
    observed: np.ndarray,
    predicted: np.ndarray,
    observed_weights: np.ndarray,
    predicted_weights: np.ndarray,
) -> float:
    cost_matrix = ot.dist(observed, predicted, metric="euclidean")
    return float(ot.emd2(observed_weights, predicted_weights, cost_matrix, numItermax=1e7))


def _geomloss_sinkhorn_w1(
    observed: np.ndarray,
    predicted: np.ndarray,
    observed_weights: np.ndarray,
    predicted_weights: np.ndarray,
    *,
    params: Mapping[str, Any],
    device: Optional[str] = None,
) -> float:
    import torch
    from geomloss import SamplesLoss

    requested_device = str(device or "cuda")
    torch_device = torch.device("cuda" if requested_device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    x = torch.as_tensor(observed, dtype=torch.float32, device=torch_device)
    y = torch.as_tensor(predicted, dtype=torch.float32, device=torch_device)
    a = torch.as_tensor(observed_weights, dtype=torch.float32, device=torch_device)
    b = torch.as_tensor(predicted_weights, dtype=torch.float32, device=torch_device)
    loss = SamplesLoss(
        loss="sinkhorn",
        p=int(params.get("p", DEFAULT_GEOMLOSS_P)),
        blur=float(params.get("blur", DEFAULT_GEOMLOSS_BLUR)),
        backend="online",
    )
    value = float(loss(a, x, b, y).detach().cpu().item())
    return max(0.0, value)


def _weighted_1d_w1(x: np.ndarray, wx: np.ndarray, y: np.ndarray, wy: np.ndarray) -> float:
    order_x = np.argsort(x)
    order_y = np.argsort(y)
    xs = x[order_x]
    ys = y[order_y]
    wxs = wx[order_x].astype(np.float64, copy=True)
    wys = wy[order_y].astype(np.float64, copy=True)
    i = 0
    j = 0
    total = 0.0
    while i < xs.shape[0] and j < ys.shape[0]:
        mass = min(wxs[i], wys[j])
        total += mass * abs(float(xs[i]) - float(ys[j]))
        wxs[i] -= mass
        wys[j] -= mass
        if wxs[i] <= 1e-15:
            i += 1
        if wys[j] <= 1e-15:
            j += 1
    return float(total)


def _sliced_w1(
    observed: np.ndarray,
    predicted: np.ndarray,
    observed_weights: np.ndarray,
    predicted_weights: np.ndarray,
    *,
    params: Mapping[str, Any],
) -> float:
    n_projections = max(1, int(params.get("n_projections", DEFAULT_SLICED_W1_PROJECTIONS)))
    seed = int(params.get("seed", DEFAULT_SLICED_W1_SEED))
    rng = np.random.default_rng(seed)
    dim = int(observed.shape[1])
    values: List[float] = []
    for _ in range(n_projections):
        direction = rng.normal(size=dim)
        norm = float(np.linalg.norm(direction))
        if not np.isfinite(norm) or norm <= 0:
            continue
        direction = direction / norm
        values.append(
            _weighted_1d_w1(
                observed @ direction,
                observed_weights,
                predicted @ direction,
                predicted_weights,
            )
        )
    return float(np.mean(values)) if values else float("nan")


def compute_w1_distance(
    observed: Any,
    predicted: Any,
    *,
    observed_weights: Optional[Sequence[float]] = None,
    predicted_weights: Optional[Sequence[float]] = None,
    backend_policy: Optional[Mapping[str, Any]] = None,
    device: Optional[str] = None,
) -> float:
    observed_arr = np.asarray(observed, dtype=np.float64)
    predicted_arr = np.asarray(predicted, dtype=np.float64)
    if observed_arr.ndim != 2 or predicted_arr.ndim != 2:
        raise ValueError("W1 inputs must be two-dimensional point arrays.")
    if observed_arr.shape[1] != predicted_arr.shape[1]:
        raise ValueError("Observed and predicted point arrays must have the same feature dimension.")
    obs_w = _normalize_weights(observed_weights, observed_arr.shape[0])
    pred_w = _normalize_weights(predicted_weights, predicted_arr.shape[0])
    policy = dict(backend_policy or select_w1_backend_policy([(observed_arr.shape[0], predicted_arr.shape[0])]))
    backend = str(policy.get("w1_backend") or "exact")
    if backend == "exact":
        return _exact_w1(observed_arr, predicted_arr, obs_w, pred_w)
    if backend == "geomloss_sinkhorn_online":
        return _geomloss_sinkhorn_w1(
            observed_arr,
            predicted_arr,
            obs_w,
            pred_w,
            params=policy.get("w1_backend_params") or {},
            device=device,
        )
    if backend == "sliced_w1":
        return _sliced_w1(
            observed_arr,
            predicted_arr,
            obs_w,
            pred_w,
            params=policy.get("w1_backend_params") or {},
        )
    raise ValueError(f"Unsupported W1 backend: {backend!r}")
