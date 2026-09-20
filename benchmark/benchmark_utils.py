"""
Shared utilities for the benchmark pipeline.

Provides:
- YAML task-card loading
- PCA projection helpers
- W1 / W2 computation (weighted and uniform)
- Run directory management
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

logger = logging.getLogger(__name__)

BENCHMARK_ROOT = Path(__file__).resolve().parent
DATA_DIR = BENCHMARK_ROOT / "data"
RESULTS_DIR = BENCHMARK_ROOT / "results"
CONFIGS_DIR = BENCHMARK_ROOT / "configs"


# ---------------------------------------------------------------------------
# YAML task-card helpers
# ---------------------------------------------------------------------------

def load_task_card(config_path: str | Path) -> Dict[str, Any]:
    """Load and validate a benchmark task-card YAML."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Task card not found: {config_path}")
    with open(config_path, "r") as f:
        card = yaml.safe_load(f)
    _validate_task_card(card)
    return card


def _validate_task_card(card: Dict[str, Any]) -> None:
    """Minimal validation of required fields."""
    required = ["dataset_id", "source_h5ad", "time_key", "folds"]
    for key in required:
        if key not in card:
            raise ValueError(f"Task card missing required field: '{key}'")
    for fold in card["folds"]:
        for fk in ["fold_id", "held_out_timepoint", "training_timepoints"]:
            if fk not in fold:
                raise ValueError(
                    f"Fold entry missing required field: '{fk}'"
                )


# ---------------------------------------------------------------------------
# Run directory management
# ---------------------------------------------------------------------------

def get_fold_data_dir(dataset_id: str, fold_id: str) -> Path:
    """Return the directory where prepared fold h5ad files live."""
    return DATA_DIR / dataset_id / fold_id


def get_run_dir(
    agent_id: str, dataset_id: str, fold_id: str, seed: int
) -> Path:
    """Return the output directory for a specific benchmark run."""
    return RESULTS_DIR / agent_id / dataset_id / fold_id / f"run_seed{seed}"


def save_run_manifest(
    run_dir: Path, manifest: Dict[str, Any]
) -> Path:
    """Save a run manifest JSON and return the path."""
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "run_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    return manifest_path


def load_run_manifest(run_dir: Path) -> Dict[str, Any]:
    """Load a run manifest JSON."""
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest at {manifest_path}")
    with open(manifest_path, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# PCA projection
# ---------------------------------------------------------------------------

def fit_pca(X: np.ndarray, n_components: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit PCA on X (cells × genes) and return (mean, components).
    Uses sklearn randomized SVD for fast computation on large dense matrices.
    """
    from sklearn.decomposition import PCA
    if n_components > min(X.shape):
        n_components = min(X.shape)
        
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=42)
    pca.fit(X)
    return pca.mean_, pca.components_


def project_pca(
    X: np.ndarray, mean: np.ndarray, components: np.ndarray
) -> np.ndarray:
    """Project X into the PCA space defined by (mean, components)."""
    return (X - mean) @ components.T


# ---------------------------------------------------------------------------
# W1 / W2 computation
# ---------------------------------------------------------------------------

def compute_wasserstein(
    X_pred: np.ndarray,
    X_true: np.ndarray,
    weights_pred: Optional[np.ndarray] = None,
    weights_true: Optional[np.ndarray] = None,
    p: int = 1,
) -> float:
    """
    Compute the p-Wasserstein distance between two point clouds.

    If weights are provided, use weighted OT. Otherwise uniform.

    Parameters
    ----------
    X_pred : (n, d) predicted distribution in PCA space
    X_true : (m, d) ground-truth distribution in PCA space
    weights_pred : (n,) per-particle weights, or None for uniform
    weights_true : (m,) per-particle weights, or None for uniform
    p : 1 for W1, 2 for W2

    Returns
    -------
    float : Wasserstein distance
    """
    try:
        import ot  # POT library
    except ImportError:
        raise ImportError(
            "POT (Python Optimal Transport) is required: pip install POT"
        )

    n, m = len(X_pred), len(X_true)

    # Weights: normalize to sum to 1
    if weights_pred is None:
        a = np.ones(n) / n
    else:
        a = np.asarray(weights_pred, dtype=np.float64)
        a = a / a.sum()

    if weights_true is None:
        b = np.ones(m) / m
    else:
        b = np.asarray(weights_true, dtype=np.float64)
        b = b / b.sum()

    # Cost matrix
    if p == 1:
        M = ot.dist(X_pred, X_true, metric="euclidean")
    elif p == 2:
        M = ot.dist(X_pred, X_true, metric="sqeuclidean")
    else:
        raise ValueError(f"Only p=1 or p=2 supported, got {p}")

    # Solve OT
    w_dist = ot.emd2(a, b, M)

    # For W2, take sqrt since cost was squared
    if p == 2:
        w_dist = np.sqrt(max(w_dist, 0.0))

    return float(w_dist)


def compute_w1_w2(
    X_pred: np.ndarray,
    X_true: np.ndarray,
    weights_pred: Optional[np.ndarray] = None,
    weights_true: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute both W1 and W2, return as dict."""
    w1 = compute_wasserstein(X_pred, X_true, weights_pred, weights_true, p=1)
    w2 = compute_wasserstein(X_pred, X_true, weights_pred, weights_true, p=2)
    return {"W1": w1, "W2": w2, "weighted": weights_pred is not None}


# ---------------------------------------------------------------------------
# Metric I/O
# ---------------------------------------------------------------------------

def save_metrics(run_dir: Path, metrics: Dict[str, Any]) -> Path:
    """Save evaluation metrics to JSON."""
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "metrics.json"
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    return path


def load_metrics(run_dir: Path) -> Dict[str, Any]:
    """Load evaluation metrics from JSON."""
    path = run_dir / "metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"No metrics at {path}")
    with open(path, "r") as f:
        return json.load(f)


def collect_all_metrics(
    agent_ids: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Walk the results directory and collect all metrics.json files.
    Returns a list of dicts, each augmented with agent/dataset/fold/seed info.
    """
    all_metrics = []
    if not RESULTS_DIR.exists():
        return all_metrics

    for agent_dir in sorted(RESULTS_DIR.iterdir()):
        if not agent_dir.is_dir():
            continue
        if agent_ids and agent_dir.name not in agent_ids:
            continue
        for dataset_dir in sorted(agent_dir.iterdir()):
            if not dataset_dir.is_dir():
                continue
            for fold_dir in sorted(dataset_dir.iterdir()):
                if not fold_dir.is_dir():
                    continue
                for run_dir in sorted(fold_dir.iterdir()):
                    if not run_dir.is_dir():
                        continue
                    metrics_path = run_dir / "metrics.json"
                    if metrics_path.exists():
                        try:
                            m = json.loads(metrics_path.read_text())
                            m["agent_id"] = agent_dir.name
                            m["dataset_id"] = dataset_dir.name
                            m["fold_id"] = fold_dir.name
                            m["run_id"] = run_dir.name
                            all_metrics.append(m)
                        except Exception as e:
                            logger.warning(f"Failed to read {metrics_path}: {e}")
    return all_metrics
