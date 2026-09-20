#!/usr/bin/env python
"""
Evaluate an externally-produced prediction (from Codex App, Antigravity, etc.)

Usage:
    python -m benchmark.eval_external \
        --prediction path/to/predicted_heldout.h5ad \
        --config benchmark/configs/weinreb_2020.yaml \
        --fold middle_holdout \
        --agent codex \
        --seed 42 \
        --runtime 180.5

This copies the prediction into the standard results directory and
runs the same evaluation pipeline as the automated runners.
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import time
from pathlib import Path

from .benchmark_utils import load_task_card, get_fold_data_dir, get_run_dir, save_run_manifest
from .benchmark_eval import evaluate_run

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Evaluate external agent prediction")
    parser.add_argument("--prediction", required=True, help="Path to predicted_heldout.h5ad")
    parser.add_argument("--config", required=True, help="Task-card YAML")
    parser.add_argument("--fold", required=True, help="Fold ID")
    parser.add_argument("--agent", required=True, help="Agent name (e.g. codex, antigravity)")
    parser.add_argument("--seed", type=int, default=42, help="Seed used")
    parser.add_argument("--runtime", type=float, default=None, help="Agent runtime in seconds (optional)")
    parser.add_argument("--notes", default="", help="Any notes about the run")
    args = parser.parse_args()

    pred_path = Path(args.prediction)
    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction not found: {pred_path}")

    card = load_task_card(args.config)
    fold = next((f for f in card["folds"] if f["fold_id"] == args.fold), None)
    if not fold:
        raise ValueError(f"Fold '{args.fold}' not found")

    # Auto-read run_log.json for runtime/method if present
    pkg_dir = pred_path.parent
    run_log = {}
    run_log_path = pkg_dir / "run_log.json"
    if run_log_path.exists():
        with open(run_log_path) as f:
            run_log = json.load(f)
        logger.info(f"Found run_log.json: runtime={run_log.get('runtime_sec')}s, method={run_log.get('method')}")

    runtime = args.runtime or run_log.get("runtime_sec")
    method = run_log.get("method", "external_semi_auto")

    # Set up results directory
    run_dir = get_run_dir(args.agent, card["dataset_id"], args.fold, args.seed)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Copy prediction + supporting files into results dir
    dest = run_dir / "predicted_heldout.h5ad"
    if pred_path.resolve() != dest.resolve():
        logger.info(f"Copying prediction to {dest}")
        shutil.copy2(pred_path, dest)
    # Look for supporting files in the prediction dir and its parent (for output/ subdir layout)
    search_dirs = [pkg_dir, pkg_dir.parent]
    for fname in ["predict.py", "run_log.json", "report.md"]:
        for sdir in search_dirs:
            src = sdir / fname
            if src.exists() and src.resolve() != (run_dir / fname).resolve():
                shutil.copy2(src, run_dir / fname)
                logger.info(f"Copied {fname} from {sdir}")
                break

    # Save manifest
    manifest = {
        "dataset_id": card["dataset_id"],
        "fold_id": args.fold,
        "held_out_timepoint": fold["held_out_timepoint"],
        "training_timepoints": fold["training_timepoints"],
        "agent_id": args.agent,
        "seed": args.seed,
        "success": True,
        "runtime_sec": runtime,
        "predicted_h5ad": str(dest),
        "method": method,
        "notes": args.notes,
    }
    save_run_manifest(run_dir, manifest)

    # Evaluate
    fold_data_dir = get_fold_data_dir(card["dataset_id"], args.fold)
    logger.info(f"Evaluating {args.agent}/{card['dataset_id']}/{args.fold}/seed={args.seed}")
    metrics = evaluate_run(run_dir, fold_data_dir)

    # Save metrics
    metrics["agent_id"] = args.agent
    metrics["dataset_id"] = card["dataset_id"]
    metrics["fold_id"] = args.fold
    metrics["seed"] = args.seed
    metrics["held_out_timepoint"] = fold["held_out_timepoint"]
    if args.runtime:
        metrics["runtime_sec"] = args.runtime

    metrics_path = run_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Agent: {args.agent}")
    print(f"W1: {metrics.get('W1', 'N/A')}")
    print(f"W2: {metrics.get('W2', 'N/A')}")
    print(f"Eval space: {metrics.get('eval_space', 'N/A')}")
    if args.runtime:
        print(f"Runtime: {args.runtime:.1f}s")
    print(f"Saved to: {metrics_path}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
