#!/usr/bin/env python
"""
benchmark_eval.py — Evaluate benchmark predictions using W1/W2 metrics.

All agents must output gene expression in .X (normalized + log1p space).
Evaluation uses a canonical HVG → PCA pipeline for fair comparison.

Usage:
    # Evaluate all runs for a dataset
    python -m benchmark.benchmark_eval \\
        --config benchmark/configs/weinreb_2020.yaml \\
        --agent cytobridge

    # Evaluate a single fold/seed
    python -m benchmark.benchmark_eval \\
        --config benchmark/configs/weinreb_2020.yaml \\
        --agent cytobridge \\
        --fold middle_holdout \\
        --seed 42
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import scanpy as sc

from .benchmark_utils import (
    load_task_card,
    get_fold_data_dir,
    get_run_dir,
    load_run_manifest,
    save_metrics,
    fit_pca,
    project_pca,
    compute_w1_w2,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _to_dense(X) -> np.ndarray:
    """Convert sparse matrix or array to dense numpy."""
    if hasattr(X, "toarray"):
        return X.toarray()
    return np.asarray(X)


def evaluate_run(
    run_dir: Path,
    fold_data_dir: Path,
    n_pca: int = 50,
    n_hvg: int = 2000,
) -> dict:
    """
    Evaluate a single benchmark run using a UNIFIED pipeline.

    All agents must output gene expression in .X (normalized + log1p space).
    Even agents working in latent/PCA space must reconstruct back to gene
    expression for fair, comparable evaluation.

    Pipeline:
      1. Preprocess train: normalize_total(1e4) → log1p → select top HVGs
      2. Preprocess ground truth: normalize_total(1e4) → log1p → subset to HVGs
      3. Prediction (already normalized+log1p by agent): subset to HVGs
      4. Fit PCA(50) on preprocessed train HVGs
      5. Project both pred and ground truth → compute W1/W2

    Returns a dict with W1, W2, success, and metadata.
    """
    heldout_path = fold_data_dir / "heldout.h5ad"
    train_path = fold_data_dir / "train.h5ad"

    if not heldout_path.exists():
        raise FileNotFoundError(f"Heldout data not found: {heldout_path}")

    # Load manifest to find predicted path
    manifest = load_run_manifest(run_dir)
    predicted_path = manifest.get("predicted_h5ad")

    if not predicted_path or not Path(predicted_path).exists():
        candidates = list(run_dir.rglob("predicted_heldout.h5ad"))
        if candidates:
            predicted_path = str(candidates[0])
        else:
            logger.warning(f"No prediction found in {run_dir}")
            return {
                "success": False,
                "error": "No predicted_heldout.h5ad found",
                "W1": None, "W2": None, "weighted": False,
            }

    logger.info(f"  Loading prediction: {predicted_path}")
    logger.info(f"  Loading heldout: {heldout_path}")

    try:
        adata_pred = sc.read_h5ad(predicted_path)
        adata_true = sc.read_h5ad(heldout_path)
        adata_train = sc.read_h5ad(train_path)
    except Exception as e:
        logger.error(f"  Failed to load h5ad: {e}")
        return {
            "success": False,
            "error": f"Failed to load h5ad: {e}",
            "W1": None, "W2": None, "weighted": False,
        }

    # --- Require gene expression in .X ---
    if adata_pred.X is None or adata_pred.shape[1] < 100:
        logger.error(
            f"  Prediction must have .X with gene expression (got {adata_pred.shape[1]} genes). "
            f"Even if you work in latent space, reconstruct back to gene expression."
        )
        return {
            "success": False,
            "error": "Prediction .X missing or too few genes",
            "W1": None, "W2": None, "weighted": False,
        }

    # --- Check for per-cell weights ---
    weights_pred = None
    raw_weight_sum = None
    for wname in ["weight", "weights", "mass"]:
        if wname in adata_pred.obs.columns:
            weights_pred = adata_pred.obs[wname].values.astype(np.float64)
            raw_weight_sum = float(np.sum(weights_pred))
            logger.info(f"  Found '{wname}' in pred.obs (raw sum={raw_weight_sum:.6f})")
            # Normalize for W1/W2 (POT requires weights sum to 1)
            if raw_weight_sum > 0:
                weights_pred = weights_pred / raw_weight_sum
            logger.info(f"  Using weighted W1/W2 (normalized sum={weights_pred.sum():.6f})")
            break

    # ===================================================================
    # Step 1: Select HVGs on raw train data, then preprocess
    # ===================================================================
    logger.info("  Selecting HVGs on raw train data (seurat_v3)")
    actual_n_hvg = min(n_hvg, adata_train.shape[1])
    adata_train_hvg = adata_train.copy()
    sc.pp.highly_variable_genes(adata_train_hvg, n_top_genes=actual_n_hvg, flavor="seurat_v3")
    hvg_mask = adata_train_hvg.var["highly_variable"]
    hvg_genes = list(adata_train_hvg.var_names[hvg_mask])
    logger.info(f"  Selected {len(hvg_genes)} HVGs from train")

    # Preprocess train: normalize + log1p (on all genes, then subset to HVGs)
    logger.info("  Preprocessing train: normalize_total(1e4) → log1p")
    adata_train_pp = adata_train.copy()
    sc.pp.normalize_total(adata_train_pp, target_sum=1e4)
    sc.pp.log1p(adata_train_pp)

    # ===================================================================
    # Step 2: Preprocess ground truth — normalize + log1p
    # ===================================================================
    adata_true_pp = adata_true.copy()
    sc.pp.normalize_total(adata_true_pp, target_sum=1e4)
    sc.pp.log1p(adata_true_pp)

    # ===================================================================
    # Step 3: Align to HVGs (common across pred and train HVGs)
    # ===================================================================
    pred_genes = set(adata_pred.var_names)
    common_hvg = [g for g in hvg_genes if g in pred_genes]
    n_common_hvg = len(common_hvg)

    if n_common_hvg < 100:
        logger.error(
            f"  Only {n_common_hvg} HVGs found in prediction "
            f"(out of {len(hvg_genes)} train HVGs). Cannot evaluate."
        )
        return {
            "success": False,
            "error": f"Too few common HVGs: {n_common_hvg}",
            "W1": None, "W2": None, "weighted": False,
            "n_common_hvg": n_common_hvg,
        }

    if n_common_hvg < len(hvg_genes):
        logger.info(f"  Using {n_common_hvg}/{len(hvg_genes)} HVGs (some not in prediction)")

    # Heuristic: check if prediction is in normalized+log1p space
    X_pred_sample = _to_dense(adata_pred[:min(100, adata_pred.shape[0]), common_hvg].X)
    max_val = float(np.max(X_pred_sample))
    if max_val > 15:
        logger.warning(
            f"  Prediction max expression = {max_val:.1f} on HVGs — "
            f"expected normalized+log1p values (< ~12). "
            f"Results may be unreliable."
        )

    # ===================================================================
    # Step 4: Fit PCA on train HVGs → project both pred and ground truth
    # ===================================================================
    X_train = _to_dense(adata_train_pp[:, common_hvg].X)
    n_components = min(n_pca, X_train.shape[0] - 1, len(common_hvg))
    logger.info(f"  Fitting canonical PCA({n_components}) on train HVGs ({X_train.shape})")
    mean, components = fit_pca(X_train, n_components)

    X_pred = _to_dense(adata_pred[:, common_hvg].X)
    X_true = _to_dense(adata_true_pp[:, common_hvg].X)
    Z_pred = project_pca(X_pred, mean, components)
    Z_true = project_pca(X_true, mean, components)

    eval_space = "hvg_pca"

    # --- Compute W1/W2 ---
    logger.info(
        f"  Computing W1/W2 in {eval_space} space: "
        f"pred={Z_pred.shape[0]} cells, true={Z_true.shape[0]} cells, dim={Z_pred.shape[1]}"
    )
    metrics = compute_w1_w2(Z_pred, Z_true, weights_pred=weights_pred)
    metrics["success"] = True
    metrics["eval_space"] = eval_space
    metrics["n_pred_cells"] = int(Z_pred.shape[0])
    metrics["n_true_cells"] = int(Z_true.shape[0])
    # Effective mass: captures growth/death prediction from weights
    # If agent saved raw (unnormalized) weights, sum(weights) reflects predicted mass change
    # n_pred_effective_mass = n_cells * sum(raw_weights) for unnormalized
    # mass_ratio = effective_mass / n_true_cells (ideal ≈ 1.0)
    if raw_weight_sum is not None and raw_weight_sum != 1.0:
        # Raw weights available — use them for effective mass
        metrics["n_pred_effective_mass"] = float(Z_pred.shape[0] * raw_weight_sum)
    else:
        # Weights were pre-normalized or absent — effective mass = n_pred_cells
        metrics["n_pred_effective_mass"] = float(Z_pred.shape[0])
    metrics["mass_ratio"] = metrics["n_pred_effective_mass"] / metrics["n_true_cells"]
    metrics["n_pca_components"] = n_components
    metrics["n_hvg"] = len(hvg_genes)
    metrics["n_common_hvg"] = n_common_hvg

    logger.info(f"  W1={metrics['W1']:.4f}  W2={metrics['W2']:.4f}  weighted={metrics['weighted']}")

    return metrics


def evaluate_dataset(
    config_path: str,
    agent_id: str = "cytobridge",
    fold_filter: Optional[str] = None,
    seed_filter: Optional[int] = None,
    n_pca: int = 50,
) -> None:
    """Evaluate all runs for a dataset config."""
    card = load_task_card(config_path)
    dataset_id = card["dataset_id"]

    folds = card["folds"]
    if fold_filter:
        folds = [f for f in folds if f["fold_id"] == fold_filter]

    for fold in folds:
        fold_id = fold["fold_id"]
        fold_data_dir = get_fold_data_dir(dataset_id, fold_id)

        # Find run directories
        runs_base = get_run_dir(agent_id, dataset_id, fold_id, 0).parent
        if not runs_base.exists():
            logger.warning(f"No runs found for {agent_id}/{dataset_id}/{fold_id}")
            continue

        for run_dir in sorted(runs_base.iterdir()):
            if not run_dir.is_dir() or not run_dir.name.startswith("run_seed"):
                continue

            # Extract seed from dir name
            seed_str = run_dir.name.replace("run_seed", "")
            try:
                seed = int(seed_str)
            except ValueError:
                continue

            if seed_filter is not None and seed != seed_filter:
                continue

            logger.info(f"\nEvaluating: {agent_id}/{dataset_id}/{fold_id}/seed={seed}")

            metrics = evaluate_run(run_dir, fold_data_dir, n_pca=n_pca)

            # Add context
            metrics["agent_id"] = agent_id
            metrics["dataset_id"] = dataset_id
            metrics["fold_id"] = fold_id
            metrics["seed"] = seed
            metrics["held_out_timepoint"] = fold["held_out_timepoint"]

            save_metrics(run_dir, metrics)
            logger.info(f"  Saved metrics to {run_dir / 'metrics.json'}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate benchmark predictions")
    parser.add_argument("--config", required=True, help="Task-card YAML config")
    parser.add_argument("--agent", default="cytobridge", help="Agent ID to evaluate")
    parser.add_argument("--fold", default=None, help="Evaluate only this fold")
    parser.add_argument("--seed", type=int, default=None, help="Evaluate only this seed")
    parser.add_argument("--n-pca", type=int, default=50, help="Number of PCA components (default: 50)")

    args = parser.parse_args()

    try:
        evaluate_dataset(
            config_path=args.config,
            agent_id=args.agent,
            fold_filter=args.fold,
            seed_filter=args.seed,
            n_pca=args.n_pca,
        )
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
