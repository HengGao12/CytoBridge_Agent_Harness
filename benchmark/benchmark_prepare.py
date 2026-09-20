#!/usr/bin/env python
"""
benchmark_prepare.py — Prepare leave-one-time-point-out folds.

For each fold defined in a task-card YAML, splits an h5ad dataset into:
  - train.h5ad  (all time points except the held-out)
  - heldout.h5ad (only the held-out time point)

Usage:
    python -m benchmark.benchmark_prepare --config benchmark/configs/weinreb_2020.yaml
    python -m benchmark.benchmark_prepare --config benchmark/configs/weinreb_2020.yaml --fold middle_holdout
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import scanpy as sc

from .benchmark_utils import load_task_card, get_fold_data_dir, BENCHMARK_ROOT

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def prepare_fold(
    adata,
    time_key: str,
    fold: dict,
    dataset_id: str,
) -> Path:
    """
    Split adata into train / heldout for a single fold.

    Returns the fold data directory.
    """
    fold_id = fold["fold_id"]
    held_out = fold["held_out_timepoint"]
    training_tps = fold["training_timepoints"]

    fold_dir = get_fold_data_dir(dataset_id, fold_id)
    fold_dir.mkdir(parents=True, exist_ok=True)

    # Validate that held_out and training_tps cover the data
    all_tps = set(adata.obs[time_key].astype(str).unique())
    expected = set([str(held_out)] + [str(t) for t in training_tps])
    if not expected.issubset(all_tps):
        missing = expected - all_tps
        logger.warning(
            f"Some time points in fold '{fold_id}' not found in data: {missing}. "
            f"Available: {all_tps}"
        )

    # Split
    mask_held = adata.obs[time_key].astype(str) == str(held_out)
    mask_train = adata.obs[time_key].astype(str).isin([str(t) for t in training_tps])

    adata_heldout = adata[mask_held].copy()
    adata_train = adata[mask_train].copy()

    n_held = adata_heldout.n_obs
    n_train = adata_train.n_obs

    if n_held == 0:
        raise ValueError(
            f"Fold '{fold_id}': held-out time point '{held_out}' has 0 cells"
        )
    if n_train == 0:
        raise ValueError(
            f"Fold '{fold_id}': training set has 0 cells"
        )

    # Save
    train_path = fold_dir / "train.h5ad"
    heldout_path = fold_dir / "heldout.h5ad"

    adata_train.write_h5ad(train_path)
    adata_heldout.write_h5ad(heldout_path)

    logger.info(
        f"  [{fold_id}] train={n_train} cells, heldout={n_held} cells "
        f"(held_out='{held_out}')"
    )

    return fold_dir


def prepare_dataset(config_path: str | Path, fold_filter: str | None = None) -> None:
    """Prepare all folds for a dataset from a task card."""
    card = load_task_card(config_path)
    dataset_id = card["dataset_id"]
    time_key = card["time_key"]

    # Resolve source h5ad path (relative to benchmark/ or absolute)
    source = card["source_h5ad"]
    source_path = Path(source)
    if not source_path.is_absolute():
        # Try relative to the repo root (parent of benchmark/)
        source_path = BENCHMARK_ROOT.parent / source
    if not source_path.exists():
        raise FileNotFoundError(f"Source h5ad not found: {source_path}")

    logger.info(f"Loading {source_path} ...")
    adata = sc.read_h5ad(source_path)
    logger.info(f"Loaded: {adata.n_obs} cells, {adata.n_vars} genes")

    # Show available time points
    tps = sorted(adata.obs[time_key].astype(str).unique())
    logger.info(f"Time points ({time_key}): {tps}")

    folds = card["folds"]
    if fold_filter:
        folds = [f for f in folds if f["fold_id"] == fold_filter]
        if not folds:
            raise ValueError(f"Fold '{fold_filter}' not found in task card")

    logger.info(f"Preparing {len(folds)} fold(s) for dataset '{dataset_id}'...")

    for fold in folds:
        prepare_fold(adata, time_key, fold, dataset_id)

    logger.info("Done.")


def main():
    parser = argparse.ArgumentParser(description="Prepare benchmark folds")
    parser.add_argument(
        "--config", required=True,
        help="Path to task-card YAML config",
    )
    parser.add_argument(
        "--fold", default=None,
        help="Prepare only this fold (default: all folds)",
    )
    args = parser.parse_args()

    try:
        prepare_dataset(args.config, args.fold)
    except Exception as e:
        logger.error(f"Failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
