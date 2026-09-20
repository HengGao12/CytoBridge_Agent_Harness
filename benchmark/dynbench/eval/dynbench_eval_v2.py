"""
DynBench v2 Evaluation Pipeline

6 equally weighted metrics, all requiring dynamical understanding:
  M1: Velocity Field Accuracy
  M2: Growth Driver Recovery
  M3: Holdout Distribution
  M4: Per-cell Fate Probability
  M5: Perturbation Response
  M6: Driver Gene Identification

GT sources:
  M1, M2, M4, M5, M6: BoolODE simulator (analytical)
  M3: Holdout cells from simulation (independent)

Current metric contract:
  - M3 uses package-evaluate-style POT exact W1 (`ot.dist` + `ot.emd2`) with
    noise-calibrated anchors; composition is reported as a diagnostic, not
    mixed into the score.
  - M6 reports strict TP diagnostics in addition to its primary score.
  - Output validation is strict and per-metric: no evaluator-side repair,
    fallback, or schema guessing. A missing or schema-invalid artifact zeros
    only the metric that depends on it; independent metrics still score.
"""

import json
import logging
import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import average_precision_score
from benchmark.dynbench.eval.fate_classifier import (
    train_fate_classifier, load_fate_classifier, classify_cells,
)
from typing import Dict, List, Optional
from collections import Counter

logger = logging.getLogger(__name__)


REQUIRED_OUTPUT_FILES = (
    "velocity_field.csv",
    "growth_rates.csv",
    "holdout_prediction.csv",
    "per_cell_fate.json",
    "perturbation_results.json",
    "driver_genes.json",
)


# ─── Weights ─────────────────────────────────────────────────────────────────

METRIC_WEIGHTS = {
    "M1_velocity":      1/6,
    "M2_growth":        1/6,
    "M3_distribution":  1/6,
    "M4_fate":          1/6,
    "M5_perturbation":  1/6,
    "M6_grn":           1/6,
}


# ─── M1: Velocity Field Accuracy ─────────────────────────────────────────────

def score_M1_velocity(
    agent_velocity: np.ndarray,
    gt_velocity: np.ndarray,
    metadata: pd.DataFrame,
    group_cols: List[str] = ['time_bin', 'fate_true'],
) -> Dict:
    """
    Compare agent's predicted velocity vs GT velocity from BoolODE.
    
    Uses per-cluster (time_bin × fate) average cosine similarity.
    
    Args:
        agent_velocity: (n_cells, n_genes) predicted velocity
        gt_velocity: (n_cells, n_genes) ground truth velocity
        metadata: DataFrame with group columns
        
    Returns:
        Dict with score and per-group details
    """
    groups = metadata.groupby(group_cols)
    cos_scores = []
    details = []
    
    for group_key, group_df in groups:
        idx = group_df.index
        if len(idx) < 5:
            continue
        
        # Map index to position array indices
        pos = np.array([metadata.index.get_loc(i) for i in idx])
        
        v_agent = agent_velocity[pos].mean(axis=0)
        v_gt = gt_velocity[pos].mean(axis=0)
        
        # Cosine similarity
        norm_agent = np.linalg.norm(v_agent)
        norm_gt = np.linalg.norm(v_gt)
        
        if norm_agent < 1e-10 or norm_gt < 1e-10:
            cos_sim = 0.0
        else:
            cos_sim = float(np.dot(v_agent, v_gt) / (norm_agent * norm_gt))
        
        cos_sim = max(0.0, cos_sim)  # negative → 0
        cos_scores.append(cos_sim)
        details.append({
            'group': str(group_key),
            'n_cells': len(idx),
            'cosine_similarity': cos_sim,
        })
    
    score = float(np.mean(cos_scores)) if cos_scores else 0.0
    
    return {
        'metric': 'M1_velocity',
        'score': score,
        'n_groups': len(cos_scores),
        'details': details,
    }


# ─── M2: Growth Driver Recovery ──────────────────────────────────────────────

def score_M2_growth(
    agent_growth_drivers: Dict[str, float] = None,
    gene_names: List[str] = None,
    gt_growth_driver_genes: List[str] = None,
) -> Dict:
    """
    Compare agent's growth-driver ranking against BoolODE growth-driver genes.

    Per-cell growth-rate values are model-specific internal quantities whose
    scale and semantics are not guaranteed to be comparable across algorithms.
    The benchmark still validates growth_rates.csv as an output contract, but
    M2 scores only growth-driver recovery.
    """
    driver_auprc = 0.0  # NO FALLBACK: missing driver data = score 0
    if agent_growth_drivers and gene_names and gt_growth_driver_genes:
        gt_labels = np.array([1 if g in gt_growth_driver_genes else 0 for g in gene_names])
        agent_scores = np.array([abs(agent_growth_drivers.get(g, 0.0)) for g in gene_names])
        if gt_labels.sum() > 0 and gt_labels.sum() < len(gt_labels):
            driver_auprc = average_precision_score(gt_labels, agent_scores)
        else:
            driver_auprc = 1.0 if gt_labels.sum() == len(gt_labels) else 0.0

    return {
        'metric': 'M2_growth',
        'score': float(driver_auprc),
        'driver_auprc': float(driver_auprc),
        'score_formula': 'growth_driver_auprc',
        'growth_rate_values_used_for_score': False,
    }


# ─── M3: Holdout Distribution ────────────────────────────────────────────────
# Package-evaluate-style POT exact W1 with noise-calibrated anchors.


def _normalize_weights(weights: np.ndarray, n: int, *, name: str) -> np.ndarray:
    if weights is None:
        return np.ones(n, dtype=np.float64) / n
    values = np.asarray(weights, dtype=np.float64).reshape(-1)
    if values.shape[0] != n:
        raise ValueError(f"{name} length {values.shape[0]} does not match point count {n}")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains non-finite values")
    total = float(values.sum())
    if total <= 0:
        raise ValueError(f"{name} has non-positive total mass")
    return values / total


def _pot_w1_distance(
    X_pred: np.ndarray,
    X_true: np.ndarray,
    weights_pred: np.ndarray = None,
    weights_true: np.ndarray = None,
    *,
    max_samples: int = 2000,
    seed: int = 42,
) -> float:
    """Exact W1 using the same POT call pattern as TrainingPipeline.evaluate."""
    import ot

    rng = np.random.RandomState(seed)
    X_pred = np.asarray(X_pred, dtype=np.float64)
    X_true = np.asarray(X_true, dtype=np.float64)
    n = int(X_pred.shape[0])
    m = int(X_true.shape[0])
    if n == 0 or m == 0:
        raise ValueError("Cannot compute W1 for empty point clouds")
    if not np.isfinite(X_pred).all() or not np.isfinite(X_true).all():
        raise ValueError("M3 point clouds contain non-finite values")

    pred_weights = None if weights_pred is None else np.asarray(weights_pred, dtype=np.float64).reshape(-1)
    true_weights = None if weights_true is None else np.asarray(weights_true, dtype=np.float64).reshape(-1)

    if n > max_samples:
        idx = rng.choice(n, max_samples, replace=False)
        X_pred = X_pred[idx]
        if pred_weights is not None:
            pred_weights = pred_weights[idx]
        n = max_samples
    if m > max_samples:
        idx = rng.choice(m, max_samples, replace=False)
        X_true = X_true[idx]
        if true_weights is not None:
            true_weights = true_weights[idx]
        m = max_samples

    pred_weights = _normalize_weights(pred_weights, n, name="pred_weights")
    true_weights = _normalize_weights(true_weights, m, name="true_weights")
    cost_matrix = ot.dist(X_true, X_pred, metric="euclidean")
    return float(ot.emd2(true_weights, pred_weights, cost_matrix, numItermax=int(1e7)))


def compute_anchors(
    holdout_expr: np.ndarray,
    holdout_labels: np.ndarray,
    holdout_time: np.ndarray = None,
    *,
    n_bootstrap: int = 100,
    max_samples: int = 2000,
    seed: int = 42,
) -> Dict:
    """
    Precompute noise-calibrated anchors for M3 scoring.
    
    These anchors define the normalization range:
      d_noise:   self-matching noise floor (bootstrap resampling)
      d_mismatch: cross-group divergence (different time_bin × fate)
      d_random:   label-shuffled reference (diagnostic only)
    
    Args:
        holdout_expr: (n_cells, n_genes) GT holdout expression
        holdout_labels: (n_cells,) fate labels
        holdout_time: (n_cells,) time values; if None, uses time_bin from labels context
        n_bootstrap: number of bootstrap iterations for d_noise
        max_samples: max cells per subsample
        seed: random seed
    
    Returns:
        Dict with d_noise, d_mismatch, d_random, and per-group details
    """
    rng = np.random.RandomState(seed)
    n_cells = holdout_expr.shape[0]
    
    # --- d_noise: bootstrap self-matching ---
    # For each (time_bin, fate) group, resample and compute exact W1 between
    # bootstrap samples, take median across groups and iterations
    noise_vals = []
    
    # Build groups: (time_bin, fate) pairs
    if holdout_time is not None:
        # Discretize time into bins
        time_vals = np.asarray(holdout_time, dtype=float)
        unique_times = np.unique(time_vals)
        # Group by time_bin × fate
        groups = {}
        for i in range(n_cells):
            t_bin = unique_times[np.argmin(np.abs(unique_times - time_vals[i]))]
            key = (str(t_bin), str(holdout_labels[i]))
            if key not in groups:
                groups[key] = []
            groups[key].append(i)
    else:
        groups = {}
        for i in range(n_cells):
            key = ('all', str(holdout_labels[i]))
            if key not in groups:
                groups[key] = []
            groups[key].append(i)
    
    # Merge small groups (fewer than 5 cells) with adjacent time points
    merged_groups = {}
    sorted_keys = sorted(groups.keys())
    for key in sorted_keys:
        cells = groups[key]
        if len(cells) >= 5:
            merged_groups[key] = cells
        else:
            # Try to merge with adjacent time bin of same fate
            time_str, fate = key
            merged = False
            for other_key in sorted_keys:
                if other_key == key:
                    continue
                other_time, other_fate = other_key
                if other_fate == fate and abs(float(time_str) - float(other_time)) < 1.0:
                    merged_groups.setdefault(other_key, []).extend(cells)
                    merged = True
                    break
            if not merged:
                # Create a "merged" key
                merge_key = ('merged', fate)
                merged_groups.setdefault(merge_key, []).extend(cells)
    
    for grp_key, grp_cells in merged_groups.items():
        if len(grp_cells) < 5:
            continue
        grp_cells = np.array(grp_cells)
        for _ in range(min(n_bootstrap, 30)):  # 30 reps per group is usually enough
            n_grp = len(grp_cells)
            idx_a = rng.choice(n_grp, size=n_grp, replace=True)
            idx_b = rng.choice(n_grp, size=n_grp, replace=True)
            w1 = _pot_w1_distance(
                holdout_expr[grp_cells[idx_a]],
                holdout_expr[grp_cells[idx_b]],
                max_samples=max_samples,
                seed=rng.randint(0, 100000),
            )
            noise_vals.append(w1)
    
    d_noise = float(np.median(noise_vals)) if noise_vals else 0.0
    d_noise_std = float(np.std(noise_vals)) if noise_vals else 0.0
    
    # --- d_mismatch: exact W1 between different (time_bin, fate) groups ---
    mismatch_vals = []
    grp_keys = sorted(merged_groups.keys())
    for i in range(len(grp_keys)):
        for j in range(i + 1, len(grp_keys)):
            cells_a = np.array(merged_groups[grp_keys[i]])
            cells_b = np.array(merged_groups[grp_keys[j]])
            # Subsample to balance
            n_a = min(len(cells_a), max_samples)
            n_b = min(len(cells_b), max_samples)
            idx_a = rng.choice(len(cells_a), n_a, replace=False)
            idx_b = rng.choice(len(cells_b), n_b, replace=False)
            w1 = _pot_w1_distance(
                holdout_expr[cells_a[idx_a]],
                holdout_expr[cells_b[idx_b]],
                max_samples=max_samples,
                seed=rng.randint(0, 100000),
            )
            mismatch_vals.append(w1)
    
    d_mismatch = float(np.median(mismatch_vals)) if mismatch_vals else 1.0
    
    # --- d_random: label-shuffled reference ---
    rng2 = np.random.RandomState(seed + 999)
    random_vals = []
    for _ in range(20):
        idx_a = rng2.choice(n_cells, min(n_cells, max_samples), replace=False)
        idx_b = rng2.choice(n_cells, min(n_cells, max_samples), replace=False)
        w1 = _pot_w1_distance(
            holdout_expr[idx_a], holdout_expr[idx_b],
            max_samples=max_samples,
            seed=rng2.randint(0, 100000),
        )
        random_vals.append(w1)
    d_random = float(np.median(random_vals)) if random_vals else d_mismatch
    
    return {
        'd_noise': d_noise,
        'd_noise_std': d_noise_std,
        'd_mismatch': d_mismatch,
        'd_random': d_random,
        'n_groups': len(merged_groups),
        'n_bootstrap_total': len(noise_vals),
        'n_mismatch_pairs': len(mismatch_vals),
    }


def score_M3_distribution(
    pred_expr: np.ndarray,
    holdout_expr: np.ndarray,
    train_expr: np.ndarray,
    train_labels: np.ndarray,
    holdout_labels: np.ndarray,
    pred_time: np.ndarray = None,
    holdout_time: np.ndarray = None,
    train_time: np.ndarray = None,
    fate_clf=None,
    pred_weights: np.ndarray = None,
    holdout_weights: np.ndarray = None,
    anchors: Dict = None,
) -> Dict:
    """
    Hold-one-out distribution quality using exact W1
    with noise-calibrated anchoring.
    
    The formula normalizes the predicted W1 against
    a noise floor (d_noise) and a mismatch ceiling (d_mismatch):
    
      ratio = (d_pred - d_noise) / (d_mismatch - d_noise)
      M3_score = smooth(ratio)  # [0,1] range, no negatives, no cliff
    
    Primary score:
      A) exact W1 distance (noise-calibrated)

    Diagnostics:
      B) Cell type composition JSD (weighted)
    
    Args:
        anchors: precomputed dict from compute_anchors(). If None, computed on-the-fly.
    """
    n_pred = pred_expr.shape[0]
    n_holdout = holdout_expr.shape[0]
    
    # Default to uniform weights if not provided
    if pred_weights is None:
        pred_weights = np.ones(n_pred) / n_pred
    else:
        pred_weights = np.asarray(pred_weights, dtype=float)
        pred_weights = pred_weights / (pred_weights.sum() + 1e-10)
    
    if holdout_weights is None:
        holdout_weights = np.ones(n_holdout) / n_holdout
    else:
        holdout_weights = np.asarray(holdout_weights, dtype=float)
        holdout_weights = holdout_weights / (holdout_weights.sum() + 1e-10)
    
    # Compute or use precomputed anchors
    if anchors is None:
        anchors = compute_anchors(
            holdout_expr, holdout_labels, holdout_time,
            n_bootstrap=100, max_samples=2000, seed=42,
        )
    
    d_noise = anchors['d_noise']
    d_mismatch = anchors['d_mismatch']
    d_random = anchors['d_random']
    
    # Sub-metric A: exact W1, matching TrainingPipeline.evaluate's POT call.
    d_pred = _pot_w1_distance(
        pred_expr, holdout_expr,
        weights_pred=pred_weights,
        weights_true=holdout_weights,
        max_samples=2000, seed=42,
    )
    
    # Noise-calibrated scoring formula.
    denom = d_mismatch - d_noise
    if denom > 1e-10:
        ratio = (d_pred - d_noise) / denom
    else:
        ratio = d_pred / (d_noise + 1e-10)

    # Smooth mapping: [0, ∞) → [0, 1], no negatives, no cliff.
    # A very good prediction can land below the finite-sample noise floor
    # (ratio < 0); it should receive full credit, not a score above 1.
    scoring_ratio = max(0.0, float(ratio))
    # ratio ≤ 1: linear 1→0 (between noise floor and mismatch ceiling)
    # ratio > 1: exponential decay toward 0 (worse than mismatch)
    if scoring_ratio <= 1.0:
        w1_score = float(1.0 - scoring_ratio)
    else:
        w1_score = float(np.exp(-3.0 * (scoring_ratio - 1.0)) * 0.01)
    w1_score = float(np.clip(w1_score, 0.0, 1.0))
    
    # Diagnostic: weighted cell type composition JSD.
    if fate_clf is None and train_time is not None:
        fate_clf = train_fate_classifier(train_expr, train_labels, train_time)
    
    if pred_time is None:
        pred_time = np.full(n_pred,
                           holdout_time.mean() if holdout_time is not None else 5.0)
    
    pred_labels = classify_cells(fate_clf, pred_expr, pred_time)
    
    all_fates = sorted(set(list(holdout_labels) + list(pred_labels)))
    
    pred_dist = np.zeros(len(all_fates))
    for i, f in enumerate(all_fates):
        mask = np.array([l == f for l in pred_labels])
        pred_dist[i] = pred_weights[mask].sum()
    
    gt_dist = np.zeros(len(all_fates))
    for i, f in enumerate(all_fates):
        mask = np.array([l == f for l in holdout_labels])
        gt_dist[i] = holdout_weights[mask].sum()
    
    pred_dist = pred_dist / (pred_dist.sum() + 1e-10)
    gt_dist = gt_dist / (gt_dist.sum() + 1e-10)
    
    comp_jsd = float(jensenshannon(pred_dist, gt_dist))
    comp_score = 1.0 - comp_jsd
    
    # Final score is the noise-calibrated W1 score.
    score = w1_score
    
    return {
        'metric': 'M3_distribution',
        'score': float(score),
        'w1_distance': float(d_pred),
        'w1_score': float(w1_score),
        'distance_backend': 'pot.emd2_euclidean',
        'd_noise': float(d_noise),
        'd_mismatch': float(d_mismatch),
        'd_random': float(d_random),
        'noise_calibrated_ratio': float(ratio),
        'composition_jsd': comp_jsd,
        'composition_score': float(comp_score),
        'pred_fate_proportions': {f: float(p) for f, p in zip(all_fates, pred_dist)},
        'gt_fate_proportions': {f: float(g) for f, g in zip(all_fates, gt_dist)},
        'weighted': bool(pred_weights.std() > 1e-8 or holdout_weights.std() > 1e-8),
        'anchor_details': {
            'd_noise': float(d_noise),
            'd_noise_std': float(anchors.get('d_noise_std', 0)),
            'd_mismatch': float(d_mismatch),
            'd_random': float(d_random),
            'n_groups': anchors.get('n_groups', 0),
        },
    }


# ─── M4: Per-cell Fate Probability ───────────────────────────────────────────

def score_M4_fate(
    agent_fate_probs: Dict[str, Dict[str, float]],
    gt_fate_probs: Dict[str, Dict[str, float]],
    target_fate: str = None,
) -> Dict:
    """
    Compare per-cell fate using top-1 terminal-state agreement.

    Primary score:
      - For each common cell, take the argmax terminal state from the agent and GT
      - Score = fraction of cells where the top-1 terminal state matches

    Probability calibration diagnostics are intentionally not part of M4.
    """
    common_cells = sorted(set(agent_fate_probs.keys()) & set(gt_fate_probs.keys()))

    if not common_cells:
        return {'metric': 'M4_fate', 'score': 0.0, 'error': 'No common cells'}

    full_fates = sorted(set(
        f
        for c in common_cells
        for f in set(gt_fate_probs[c].keys()) | set(agent_fate_probs[c].keys())
    ))
    all_fates = list(full_fates)
    if target_fate is not None:
        all_fates = [target_fate]

    def _top1_label(prob_dict: Dict[str, float], allowed_fates: List[str]) -> str:
        if not allowed_fates:
            return ""
        best_fate = allowed_fates[0]
        best_val = float(prob_dict.get(best_fate, 0.0))
        for fate in allowed_fates[1:]:
            val = float(prob_dict.get(fate, 0.0))
            if val > best_val:
                best_fate = fate
                best_val = val
        return best_fate

    if target_fate is None:
        gt_top1 = [_top1_label(gt_fate_probs[c], full_fates) for c in common_cells]
        agent_top1 = [_top1_label(agent_fate_probs[c], full_fates) for c in common_cells]
        matches = [int(a == g) for a, g in zip(agent_top1, gt_top1)]
        top1_score = float(np.mean(matches)) if matches else 0.0
        gt_counter = Counter(gt_top1)
        agent_counter = Counter(agent_top1)
        confusion = {}
        for gt_fate in all_fates:
            row = {}
            for agent_fate in all_fates:
                count = sum(
                    1
                    for a, g in zip(agent_top1, gt_top1)
                    if g == gt_fate and a == agent_fate
                )
                if count:
                    row[agent_fate] = int(count)
            confusion[gt_fate] = row
        mode = "top1_agreement"
    else:
        gt_top1 = [_top1_label(gt_fate_probs[c], full_fates) == target_fate for c in common_cells]
        agent_top1 = [_top1_label(agent_fate_probs[c], full_fates) == target_fate for c in common_cells]
        matches = [int(a == g) for a, g in zip(agent_top1, gt_top1)]
        top1_score = float(np.mean(matches)) if matches else 0.0
        gt_counter = Counter({"target": int(sum(gt_top1)), "other": int(len(gt_top1) - sum(gt_top1))})
        agent_counter = Counter({"target": int(sum(agent_top1)), "other": int(len(agent_top1) - sum(agent_top1))})
        confusion = {
            "target": {
                "target": int(sum(1 for a, g in zip(agent_top1, gt_top1) if g and a)),
                "other": int(sum(1 for a, g in zip(agent_top1, gt_top1) if g and not a)),
            },
            "other": {
                "target": int(sum(1 for a, g in zip(agent_top1, gt_top1) if (not g) and a)),
                "other": int(sum(1 for a, g in zip(agent_top1, gt_top1) if (not g) and (not a))),
            },
        }
        mode = "top1_target_membership_agreement"

    return {
        'metric': 'M4_fate',
        'score': top1_score,
        'evaluation_mode': mode,
        'top1_agreement': top1_score,
        'n_cells': len(common_cells),
        'n_fates': len(all_fates),
        'gt_top1_counts': {k: int(v) for k, v in gt_counter.items()},
        'agent_top1_counts': {k: int(v) for k, v in agent_counter.items()},
        'top1_confusion': confusion,
    }


# ─── M5: Perturbation Response ───────────────────────────────────────────────

def score_M5_perturbation(
    agent_perturbation: List[Dict],
    gt_perturbation: List[Dict],
) -> Dict:
    """
    Compare perturbation responses.
    
    Args:
        agent_perturbation: list of {"gene_name": ..., "delta": {fate: delta}}
        gt_perturbation: list of {"gene_name": ..., "delta": {fate: delta}}
    """
    threshold = 0.01

    # Match by gene name
    gt_by_gene = {p['gene_name']: p for p in gt_perturbation}

    tp = 0
    fp = 0
    fn = 0
    tn = 0
    sign_matches = 0
    gt_active_pairs = 0
    details = []

    for agent_p in agent_perturbation:
        gene = agent_p.get('gene_name', agent_p.get('gene_idx', '?'))
        if gene not in gt_by_gene:
            continue

        gt_p = gt_by_gene[gene]

        for fate in set(list(agent_p['delta'].keys()) + list(gt_p['delta'].keys())):
            agent_d = agent_p['delta'].get(fate, 0.0)
            gt_d = gt_p['delta'].get(fate, 0.0)

            pred_active = abs(agent_d) >= threshold
            gt_active = abs(gt_d) >= threshold

            if gt_active and pred_active:
                tp += 1
            elif (not gt_active) and pred_active:
                fp += 1
            elif gt_active and (not pred_active):
                fn += 1
            else:
                tn += 1

            sign_match = None
            if gt_active:
                gt_active_pairs += 1
                sign_match = np.sign(agent_d) == np.sign(gt_d)
                if sign_match:
                    sign_matches += 1

            details.append({
                'gene': gene,
                'fate': fate,
                'agent_delta': float(agent_d),
                'gt_delta': float(gt_d),
                'pred_active': bool(pred_active),
                'gt_active': bool(gt_active),
                'sign_match': None if sign_match is None else bool(sign_match),
            })

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    if precision + recall > 0:
        detection_f1 = 2 * precision * recall / (precision + recall)
    else:
        detection_f1 = 0.0

    sign_accuracy = sign_matches / max(gt_active_pairs, 1)
    score = 0.5 * detection_f1 + 0.5 * sign_accuracy

    return {
        'metric': 'M5_perturbation',
        'score': float(score),
        'evaluation_mode': '0.5_detection_F1_plus_0.5_sign_accuracy',
        'threshold': float(threshold),
        'detection_F1': float(detection_f1),
        'sign_accuracy': float(sign_accuracy),
        'active_detection_precision': float(precision),
        'active_detection_recall': float(recall),
        'n_gt_active_pairs': int(gt_active_pairs),
        'n_total_pairs': int(tp + fp + fn + tn),
        'active_detection_confusion': {
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn),
            'tn': int(tn),
        },
        'details': details,
    }


# ─── M6: Driver Gene Identification ──────────────────────────────────────────

def score_M6_grn(
    agent_grn_edges: List[Dict],
    gt_grn_edges: List[Dict],
    gene_names: List[str],
) -> Dict:
    """
    GRN edge prediction: does the agent correctly identify regulatory edges?
    
    Two sub-metrics (50/50):
    A) Edge existence AUPRC: does the agent correctly identify which gene pairs have edges?
    B) Direction accuracy: among true edges, does the agent correctly predict activate/inhibit?
    
    Agent format: [{"source": "Gene_1", "target": "Gene_2", "score": 0.85}, ...]
       - Positive score = activate, negative score = inhibit
    GT format: [{"from": "Gene_1", "to": "Gene_2", "direction": "activate"}, ...]
    """
    n_genes = len(gene_names)
    
    # Build GT edge set and direction map
    gt_edge_set = set()
    gt_direction = {}  # (src, tgt) -> +1 (activate) or -1 (inhibit)
    for edge in gt_grn_edges:
        key = (edge['from'], edge['to'])
        gt_edge_set.add(key)
        gt_direction[key] = 1.0 if edge.get('direction', 'activate') == 'activate' else -1.0
    
    # Build agent score map (signed: positive=activate, negative=inhibit)
    agent_score_map = {}
    for edge in agent_grn_edges:
        src = edge.get('source', edge.get('from', ''))
        tgt = edge.get('target', edge.get('to', ''))
        score = float(edge.get('score', edge.get('weight', 0.0)))
        agent_score_map[(src, tgt)] = score
    
    # --- Sub-metric A: Edge existence AUPRC ---
    gt_labels = []
    agent_abs_scores = []
    for src in gene_names:
        for tgt in gene_names:
            gt_labels.append(1 if (src, tgt) in gt_edge_set else 0)
            agent_abs_scores.append(abs(agent_score_map.get((src, tgt), 0.0)))
    
    gt_labels = np.array(gt_labels)
    agent_abs_scores = np.array(agent_abs_scores)
    
    if gt_labels.sum() > 0 and gt_labels.sum() < len(gt_labels):
        auprc = average_precision_score(gt_labels, agent_abs_scores)
        from sklearn.metrics import roc_auc_score
        auroc = roc_auc_score(gt_labels, agent_abs_scores)
    else:
        auprc = 0.0
        auroc = 0.5
    
    # --- Sub-metric B: Direction accuracy (on true edges) ---
    correct_dir = 0
    total_dir = 0
    for key in gt_edge_set:
        if key in agent_score_map:
            total_dir += 1
            agent_sign = np.sign(agent_score_map[key])
            gt_sign = gt_direction[key]
            if agent_sign == gt_sign:
                correct_dir += 1
    
    dir_accuracy = correct_dir / total_dir if total_dir > 0 else 0.0
    
    # --- Diagnostic: Strict TP (edge exists AND direction correct) ---
    # An edge is a strict TP only if the agent predicts the edge AND gets
    # the direction right. This is a harder criterion than AUPRC alone.
    strict_tp = 0
    strict_fp_edges = 0  # agent predicts edge but either wrong direction or false positive
    n_pos = int(gt_labels.sum())  # total GT edges (positive samples)
    for src in gene_names:
        for tgt in gene_names:
            key = (src, tgt)
            agent_score = agent_score_map.get(key, 0.0)
            agent_has_edge = abs(agent_score) > 1e-6
            gt_has_edge = key in gt_edge_set
            if agent_has_edge and gt_has_edge:
                # Edge exists in both — check direction
                agent_sign = np.sign(agent_score)
                gt_sign = gt_direction[key]
                if agent_sign == gt_sign:
                    strict_tp += 1
                # else: direction mismatch (counted in dir stats)
            elif agent_has_edge and not gt_has_edge:
                strict_fp_edges += 1
    strict_fn_edges = n_pos - strict_tp  # GT edges missed or wrong direction
    strict_precision = strict_tp / max(strict_tp + strict_fp_edges, 1)
    strict_recall = strict_tp / max(strict_tp + strict_fn_edges, 1)
    strict_f1 = (2 * strict_precision * strict_recall /
                 max(strict_precision + strict_recall, 1e-10))

    # Random baseline for context
    n_total = len(gt_labels)
    random_baseline = n_pos / n_total if n_total > 0 else 0.0
    
    # Score = 50% AUPRC + 50% direction accuracy
    score = 0.5 * auprc + 0.5 * dir_accuracy
    
    return {
        'metric': 'M6_grn',
        'score': float(score),
        'edge_auprc': float(auprc),
        'edge_auroc': float(auroc),
        'direction_accuracy': float(dir_accuracy),
        'score_formula': '0.5 * auprc + 0.5 * direction_accuracy',
        'strict_tp': int(strict_tp),
        'strict_fp_edges': int(strict_fp_edges),
        'strict_fn_edges': int(strict_fn_edges),
        'strict_precision': float(strict_precision),
        'strict_recall': float(strict_recall),
        'strict_f1': float(strict_f1),
        'random_baseline': float(random_baseline),
        'n_gt_edges': n_pos,
        'n_possible_edges': n_total,
        'n_agent_edges': len(agent_grn_edges),
        'n_direction_evaluated': total_dir,
    }


# ─── DynBench contract helpers ──────────────────────────────────────────────

def _resolve_task_dir(gt_dir: str) -> str:
    scenario_name = os.path.basename(os.path.normpath(gt_dir))
    return os.path.join(gt_dir, '..', '..', 'task_packages', scenario_name)


def _to_dense(X) -> np.ndarray:
    if hasattr(X, "toarray"):
        return X.toarray()
    return np.asarray(X)


def _train_obs_names_preserve_gt_identity(
    train_adata,
    meta: pd.DataFrame,
    check_cols: Optional[List[str]] = None,
) -> bool:
    """Return True only when train.h5ad obs_names can be trusted as GT ids.

    Some DynBench task packages subset the full simulation and then relabel
    obs_names sequentially (e.g. ``cell_4069`` in train.h5ad is no longer the
    original ``cell_4069`` from full_metadata.csv). In that case, aligning GT by
    obs_names silently corrupts M1/M2. We therefore verify that shared ids also
    agree on stable metadata fields such as ``time_bin``/``time`` before using
    id-based alignment.
    """
    if train_adata is None:
        return False

    obs_ids = train_adata.obs_names.astype(str).tolist()
    if len(obs_ids) == 0:
        return False
    if any(cid not in meta.index for cid in obs_ids):
        return False

    cols = check_cols or [c for c in ["time_bin", "time"] if c in train_adata.obs.columns and c in meta.columns]
    if not cols:
        return True

    obs_df = train_adata.obs.loc[:, cols].copy()
    obs_df.index = obs_df.index.astype(str)
    meta_df = meta.loc[obs_df.index, cols].copy()

    for col in cols:
        obs_col = pd.to_numeric(obs_df[col], errors="coerce")
        meta_col = pd.to_numeric(meta_df[col], errors="coerce")

        if obs_col.notna().all() and meta_col.notna().all():
            mismatch = ~np.isclose(obs_col.values.astype(float), meta_col.values.astype(float))
        else:
            mismatch = obs_df[col].astype(str).values != meta_df[col].astype(str).values

        if mismatch.any():
            return False

    return True


def _load_dynbench_contract(gt_dir: str, meta: pd.DataFrame) -> Dict:
    task_dir = _resolve_task_dir(gt_dir)
    train_path = os.path.join(task_dir, 'train.h5ad')
    sim_config_path = os.path.join(gt_dir, 'simulation_config.json')
    prediction_targets_path = os.path.join(task_dir, 'prediction_targets.json')

    sim_cfg = {}
    if os.path.exists(sim_config_path):
        with open(sim_config_path) as f:
            sim_cfg = json.load(f)

    prediction_cfg = {}
    if os.path.exists(prediction_targets_path):
        with open(prediction_targets_path) as f:
            prediction_cfg = json.load(f)

    train_adata = None
    if os.path.exists(train_path):
        import anndata
        train_adata = anndata.read_h5ad(train_path)

    all_bins = sorted(pd.to_numeric(meta['time_bin'], errors='coerce').dropna().astype(int).unique().tolist())

    holdout_bins = sim_cfg.get('holdout_bins')
    if holdout_bins is None:
        if train_adata is not None and 'time_bin' in train_adata.obs:
            train_bins_detected = sorted(
                pd.to_numeric(train_adata.obs['time_bin'], errors='coerce')
                .dropna()
                .astype(int)
                .unique()
                .tolist()
            )
            holdout_bins = [b for b in all_bins if b not in train_bins_detected]
        else:
            holdout_bins = [2, 3]
    holdout_bins = [int(b) for b in holdout_bins]

    train_bins = sim_cfg.get('train_bins')
    if train_bins is None:
        if train_adata is not None and 'time_bin' in train_adata.obs:
            train_bins = sorted(
                pd.to_numeric(train_adata.obs['time_bin'], errors='coerce')
                .dropna()
                .astype(int)
                .unique()
                .tolist()
            )
        else:
            train_bins = [b for b in all_bins if b not in holdout_bins]
    train_bins = [int(b) for b in train_bins]

    prediction_target_bins = prediction_cfg.get(
        'prediction_target_bins',
        sim_cfg.get('prediction_target_bins', holdout_bins),
    )
    prediction_target_bins = [int(b) for b in prediction_target_bins]

    prediction_eval_source = prediction_cfg.get(
        'prediction_eval_source',
        sim_cfg.get('prediction_eval_source', 'log1p_counts_ground_truth'),
    )
    prediction_gt_weight_policy = prediction_cfg.get(
        'prediction_gt_weight_policy',
        sim_cfg.get('prediction_gt_weight_policy', 'uniform_empirical'),
    )
    prediction_agent_weight_policy = prediction_cfg.get(
        'prediction_agent_weight_policy',
        sim_cfg.get('prediction_agent_weight_policy', 'optional_output_weight'),
    )

    return {
        'task_dir': task_dir,
        'train_adata': train_adata,
        'holdout_bins': holdout_bins,
        'train_bins': train_bins,
        'prediction_target_bins': prediction_target_bins,
        'prediction_eval_source': prediction_eval_source,
        'prediction_gt_weight_policy': prediction_gt_weight_policy,
        'prediction_agent_weight_policy': prediction_agent_weight_policy,
        'sim_cfg': sim_cfg,
    }


# ─── Strict Output Validation ────────────────────────────────────────────────

def _read_required_json(agent_output_dir: str, fname: str):
    path = os.path.join(agent_output_dir, fname)
    if not os.path.exists(path):
        raise ValueError(f"Missing required DynBench output: {fname}")
    if os.path.getsize(path) == 0:
        raise ValueError(f"Empty required DynBench output: {fname}")
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as exc:
        raise ValueError(f"Invalid JSON in required DynBench output {fname}: {exc}") from exc


def _read_required_csv(agent_output_dir: str, fname: str) -> pd.DataFrame:
    path = os.path.join(agent_output_dir, fname)
    if not os.path.exists(path):
        raise ValueError(f"Missing required DynBench output: {fname}")
    if os.path.getsize(path) == 0:
        raise ValueError(f"Empty required DynBench output: {fname}")
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise ValueError(f"Invalid CSV in required DynBench output {fname}: {exc}") from exc
    if len(df) == 0:
        raise ValueError(f"Required DynBench output has zero rows: {fname}")
    if any(str(c).startswith("Unnamed:") for c in df.columns):
        raise ValueError(
            f"{fname} contains an implicit pandas index column. "
            "Write CSV outputs with index=False and the exact benchmark schema."
        )
    return df


def _metric_failure(metric: str, exc: Exception) -> Dict:
    return {
        "metric": metric,
        "score": 0.0,
        "error": str(exc),
        "validation_failed": True,
    }


def _validate_velocity_field(agent_output_dir: str, gene_names: List[str], n_train: Optional[int]) -> pd.DataFrame:
    velocity_df = _read_required_csv(agent_output_dir, "velocity_field.csv")
    velocity_cols = [f"velocity_{g}" for g in gene_names]
    if list(velocity_df.columns) != velocity_cols:
        raise ValueError(
            "velocity_field.csv schema mismatch: expected columns "
            f"{velocity_cols[:5]}{'...' if len(velocity_cols) > 5 else ''}, "
            f"got {list(velocity_df.columns)[:5]}{'...' if len(velocity_df.columns) > 5 else ''}"
        )
    if n_train is not None and len(velocity_df) != n_train:
        raise ValueError(f"velocity_field.csv row count {len(velocity_df)} != train cells {n_train}")
    if not np.isfinite(velocity_df[velocity_cols].to_numpy(dtype=float)).all():
        raise ValueError("velocity_field.csv contains non-finite values")
    return velocity_df


def _validate_growth_rates(agent_output_dir: str, n_train: Optional[int]) -> pd.DataFrame:
    growth_df = _read_required_csv(agent_output_dir, "growth_rates.csv")
    if list(growth_df.columns) != ["growth_rate"]:
        raise ValueError(f"growth_rates.csv must contain exactly ['growth_rate'], got {list(growth_df.columns)}")
    if n_train is not None and len(growth_df) != n_train:
        raise ValueError(f"growth_rates.csv row count {len(growth_df)} != train cells {n_train}")
    if not np.isfinite(growth_df[["growth_rate"]].to_numpy(dtype=float)).all():
        raise ValueError("growth_rates.csv contains non-finite values")
    return growth_df


def _validate_holdout_prediction(agent_output_dir: str, gene_names: List[str]) -> pd.DataFrame:
    prediction_df = _read_required_csv(agent_output_dir, "holdout_prediction.csv")
    allowed_prediction_cols = set(gene_names) | {"time", "weight"}
    missing_gene_cols = [g for g in gene_names if g not in prediction_df.columns]
    extra_prediction_cols = [c for c in prediction_df.columns if c not in allowed_prediction_cols]
    if missing_gene_cols or extra_prediction_cols:
        raise ValueError(
            "holdout_prediction.csv schema mismatch: "
            f"missing gene columns={missing_gene_cols[:5]}, extra columns={extra_prediction_cols[:5]}"
        )
    if "time" not in prediction_df.columns:
        raise ValueError("holdout_prediction.csv must include a time column")
    numeric_cols = gene_names + ["time"] + (["weight"] if "weight" in prediction_df.columns else [])
    if not np.isfinite(prediction_df[numeric_cols].to_numpy(dtype=float)).all():
        raise ValueError("holdout_prediction.csv contains non-finite values")
    return prediction_df


def _validate_fate_probs(agent_output_dir: str) -> Dict:
    fate = _read_required_json(agent_output_dir, "per_cell_fate.json")
    if not isinstance(fate, dict) or not fate:
        raise ValueError("per_cell_fate.json must be a non-empty object mapping cell ids to fate probabilities")
    for cell_id, probs in fate.items():
        if not isinstance(cell_id, str) or not isinstance(probs, dict) or not probs:
            raise ValueError("per_cell_fate.json entries must be {cell_id: {fate: probability}}")
        for fate_name, prob in probs.items():
            if not isinstance(fate_name, str):
                raise ValueError("per_cell_fate.json fate names must be strings")
            try:
                p = float(prob)
            except Exception as exc:
                raise ValueError(f"per_cell_fate.json has non-numeric probability for {cell_id}/{fate_name}") from exc
            if not np.isfinite(p) or p < 0:
                raise ValueError(f"per_cell_fate.json has invalid probability for {cell_id}/{fate_name}: {prob}")
    return fate


def _validate_perturbation_results(agent_output_dir: str, gene_names: List[str]) -> List[Dict]:
    perturbation = _read_required_json(agent_output_dir, "perturbation_results.json")
    if not isinstance(perturbation, list) or not perturbation:
        raise ValueError("perturbation_results.json must be a non-empty list")
    pert_genes = []
    for row in perturbation:
        if not isinstance(row, dict) or "gene_name" not in row or "delta" not in row:
            raise ValueError("perturbation_results.json entries must contain gene_name and delta")
        if row["gene_name"] not in gene_names:
            raise ValueError(f"perturbation_results.json contains unknown gene: {row['gene_name']}")
        if not isinstance(row["delta"], dict) or not row["delta"]:
            raise ValueError("perturbation_results.json delta must be a non-empty fate-to-effect object")
        for fate_name, delta in row["delta"].items():
            if not isinstance(fate_name, str):
                raise ValueError("perturbation_results.json delta fate names must be strings")
            try:
                value = float(delta)
            except Exception as exc:
                raise ValueError(f"perturbation_results.json has non-numeric delta for {row['gene_name']}/{fate_name}") from exc
            if not np.isfinite(value):
                raise ValueError(f"perturbation_results.json has non-finite delta for {row['gene_name']}/{fate_name}: {delta}")
        pert_genes.append(row["gene_name"])
    missing_pert_genes = [g for g in gene_names if g not in set(pert_genes)]
    if missing_pert_genes:
        raise ValueError(f"perturbation_results.json is missing genes: {missing_pert_genes[:5]}")
    return perturbation


def _read_driver_genes(agent_output_dir: str) -> Dict:
    drivers = _read_required_json(agent_output_dir, "driver_genes.json")
    if not isinstance(drivers, dict):
        raise ValueError("driver_genes.json must be an object")
    return drivers


def _validate_growth_drivers(drivers: Dict, gene_names: List[str]) -> Dict[str, float]:
    growth_drivers = drivers.get("growth_drivers")
    if not isinstance(growth_drivers, dict):
        raise ValueError("driver_genes.json must contain growth_drivers object")
    missing_growth_driver_genes = [g for g in gene_names if g not in growth_drivers]
    if missing_growth_driver_genes:
        raise ValueError(f"driver_genes.json growth_drivers is missing genes: {missing_growth_driver_genes[:5]}")
    for gene, score in growth_drivers.items():
        if gene not in gene_names:
            raise ValueError(f"driver_genes.json growth_drivers contains unknown gene: {gene}")
        try:
            value = float(score)
        except Exception as exc:
            raise ValueError(f"driver_genes.json growth driver score is non-numeric for {gene}") from exc
        if not np.isfinite(value):
            raise ValueError(f"driver_genes.json growth driver score is not finite for {gene}: {score}")
    return growth_drivers


def _validate_grn_edges(drivers: Dict, gene_names: List[str]) -> List[Dict]:
    grn_edges = drivers.get("grn_edges")
    if not isinstance(grn_edges, list):
        raise ValueError("driver_genes.json must contain grn_edges list")
    for edge in grn_edges:
        if not isinstance(edge, dict) or "source" not in edge or "target" not in edge or "score" not in edge:
            raise ValueError("driver_genes.json grn_edges entries must contain source, target, and score")
        if edge["source"] not in gene_names or edge["target"] not in gene_names:
            raise ValueError(f"driver_genes.json grn edge contains unknown gene: {edge}")
        try:
            score = float(edge["score"])
        except Exception as exc:
            raise ValueError(f"driver_genes.json grn edge score is non-numeric: {edge}") from exc
        if not np.isfinite(score):
            raise ValueError(f"driver_genes.json grn edge score is not finite: {edge}")
    return grn_edges


def validate_outputs_strict(
    agent_output_dir: str,
    gene_names: List[str],
    train_adata,
) -> Dict[str, object]:
    """Validate all DynBench outputs without repair, fallback, or schema guessing."""
    missing = [fname for fname in REQUIRED_OUTPUT_FILES if not os.path.exists(os.path.join(agent_output_dir, fname))]
    if missing:
        raise ValueError(f"Missing required DynBench outputs: {missing}")

    n_train = int(train_adata.n_obs) if train_adata is not None else None
    drivers = _read_driver_genes(agent_output_dir)
    _validate_growth_drivers(drivers, gene_names)
    _validate_grn_edges(drivers, gene_names)

    return {
        "velocity_field.csv": _validate_velocity_field(agent_output_dir, gene_names, n_train),
        "growth_rates.csv": _validate_growth_rates(agent_output_dir, n_train),
        "holdout_prediction.csv": _validate_holdout_prediction(agent_output_dir, gene_names),
        "per_cell_fate.json": _validate_fate_probs(agent_output_dir),
        "perturbation_results.json": _validate_perturbation_results(agent_output_dir, gene_names),
        "driver_genes.json": drivers,
    }

# ─── Main Evaluation ─────────────────────────────────────────────────────────

def evaluate_all(
    agent_output_dir: str,
    gt_dir: str,
    verbose: bool = True,
) -> Dict:
    """
    Run all 6 metrics and compute weighted total score.
    
    Expected agent outputs in agent_output_dir:
        - velocity_field.csv       (for M1)
        - growth_rates.csv         (for M2)
        - holdout_prediction.csv   (for M3)
        - per_cell_fate.json       (for M4)
        - perturbation_results.json (for M5)
        - driver_genes.json        (for M6)
    
    Expected GT files in gt_dir:
        - velocity_ground_truth.csv
        - full_metadata.csv
        - counts_cells_x_genes.csv
        - percell_fate_ground_truth.json
        - perturbation_ground_truth.json
        - grn_ground_truth.json
    """
    results = {}
    
    # Load GT data
    meta = pd.read_csv(os.path.join(gt_dir, 'full_metadata.csv'), index_col=0)
    expr = pd.read_csv(os.path.join(gt_dir, 'counts_cells_x_genes.csv'), index_col=0)
    vel_gt = pd.read_csv(os.path.join(gt_dir, 'velocity_ground_truth.csv'), index_col=0)
    
    gene_names = list(expr.columns)
    
    with open(os.path.join(gt_dir, 'perturbation_ground_truth.json')) as f:
        pert_gt = json.load(f)
    with open(os.path.join(gt_dir, 'percell_fate_ground_truth.json')) as f:
        fate_gt = json.load(f)
    with open(os.path.join(gt_dir, 'grn_ground_truth.json')) as f:
        grn_gt = json.load(f)

    contract = _load_dynbench_contract(gt_dir, meta)
    task_dir = contract['task_dir']
    train_adata = contract['train_adata']
    holdout_bins = contract['holdout_bins']
    train_bins = contract['train_bins']
    prediction_target_bins = contract['prediction_target_bins']
    prediction_eval_source = contract['prediction_eval_source']
    prediction_gt_weight_policy = contract['prediction_gt_weight_policy']
    prediction_agent_weight_policy = contract['prediction_agent_weight_policy']

    if train_adata is not None:
        ids_preserve_identity = _train_obs_names_preserve_gt_identity(train_adata, meta)
        if ids_preserve_identity:
            train_ids = [
                cid for cid in train_adata.obs_names.astype(str).tolist()
                if cid in meta.index and cid in vel_gt.index
            ]
            meta_train = meta.loc[train_ids].copy()
            vel_gt_train = vel_gt.loc[train_ids].copy()
        else:
            train_ids = None
            train_mask = meta['time_bin'].isin(train_bins)
            meta_train = meta[train_mask].copy()
            vel_gt_train = vel_gt[train_mask].copy()
    else:
        train_ids = None
        train_mask = meta['time_bin'].isin(train_bins)
        meta_train = meta[train_mask].copy()
        vel_gt_train = vel_gt[train_mask].copy()

    if 'fate_true' not in meta_train.columns and 'cell_type' in meta_train.columns:
        meta_train['fate_true'] = meta_train['cell_type']

    meta_train = meta_train.reset_index(drop=True)
    vel_gt_train = vel_gt_train.reset_index(drop=True)
    n_train = int(train_adata.n_obs) if train_adata is not None else len(meta_train)

    drivers_cache = None

    def _drivers() -> Dict:
        nonlocal drivers_cache
        if drivers_cache is None:
            drivers_cache = _read_driver_genes(agent_output_dir)
        return drivers_cache
    
    # ── M1: Velocity ──
    try:
        vel_df = _validate_velocity_field(agent_output_dir, gene_names, n_train)
        agent_vel = vel_df[[f"velocity_{g}" for g in gene_names]].values
        gt_vel = vel_gt_train.values
        if len(agent_vel) != len(gt_vel):
            raise ValueError(f"velocity_field.csv row count {len(agent_vel)} != GT train rows {len(gt_vel)}")
        results['M1_velocity'] = score_M1_velocity(agent_vel, gt_vel, meta_train)
    except Exception as exc:
        results['M1_velocity'] = _metric_failure('M1_velocity', exc)
    
    # ── M2: Growth ──
    try:
        growth_df = _validate_growth_rates(agent_output_dir, n_train)
        agent_growth = growth_df['growth_rate'].values
        if len(agent_growth) != len(meta_train):
            raise ValueError(f"growth_rates.csv row count {len(agent_growth)} != GT train rows {len(meta_train)}")
        agent_growth_drivers_m2 = _validate_growth_drivers(_drivers(), gene_names)
        results['M2_growth'] = score_M2_growth(
            agent_growth_drivers=agent_growth_drivers_m2,
            gene_names=gene_names,
            gt_growth_driver_genes=grn_gt.get('growth_drivers', grn_gt.get('growth_driver_genes')),
        )
    except Exception as exc:
        results['M2_growth'] = _metric_failure('M2_growth', exc)
    
    # ── M3: Distribution ──
    # Load holdout and prediction
    try:
        pred_expr_df = _validate_holdout_prediction(agent_output_dir, gene_names)
        pred_expr_vals = pred_expr_df[gene_names].values

        if prediction_eval_source == 'train_h5ad_X' and train_adata is not None:
            train_time_bins = pd.to_numeric(train_adata.obs['time_bin'], errors='coerce').astype('Int64')
            target_mask = np.asarray(train_time_bins.isin(prediction_target_bins).values, dtype=bool)
            background_mask = ~target_mask
            if background_mask.sum() == 0:
                background_mask = np.asarray(train_time_bins.isin(train_bins).values, dtype=bool)

            label_col = 'fate_true' if 'fate_true' in train_adata.obs.columns else 'cell_type'
            holdout_expr_vals = _to_dense(train_adata.X[target_mask]).astype(np.float32)
            train_expr_vals = _to_dense(train_adata.X[background_mask]).astype(np.float32)
            holdout_labels = train_adata.obs.loc[target_mask, label_col].values
            train_labels = train_adata.obs.loc[background_mask, label_col].values
            holdout_time = pd.to_numeric(train_adata.obs.loc[target_mask, 'time'], errors='coerce').values.astype(np.float32)
            train_time = pd.to_numeric(train_adata.obs.loc[background_mask, 'time'], errors='coerce').values.astype(np.float32)
            holdout_weights = None
            if prediction_gt_weight_policy == 'metadata_weight' and 'weight' in train_adata.obs.columns:
                holdout_weights = pd.to_numeric(train_adata.obs.loc[target_mask, 'weight'], errors='coerce').fillna(0.0).values.astype(np.float64)
        else:
            holdout_mask = meta['time_bin'].isin(prediction_target_bins)
            train_mask = meta['time_bin'].isin(train_bins)
            holdout_expr_vals = np.log1p(expr.loc[holdout_mask, gene_names].values)
            train_expr_vals = np.log1p(expr.loc[train_mask, gene_names].values)
            train_labels = meta.loc[train_mask, 'fate_true'].values
            holdout_labels = meta.loc[holdout_mask, 'fate_true'].values
            train_time = meta.loc[train_mask, 'time'].values.astype(np.float32)
            holdout_time = meta.loc[holdout_mask, 'time'].values.astype(np.float32)
            holdout_weights = None
            if prediction_gt_weight_policy == 'metadata_weight' and 'weight' in meta.columns:
                holdout_weights = meta.loc[holdout_mask, 'weight'].values.astype(np.float64)

        pred_time = pred_expr_df['time'].values.astype(np.float32)

        fate_clf_path = os.path.join(task_dir, 'fate_classifier.pkl')
        fate_clf = None
        if os.path.exists(fate_clf_path):
            fate_clf = load_fate_classifier(fate_clf_path)

        pred_weights = None
        if prediction_agent_weight_policy != 'ignore_output_weight' and 'weight' in pred_expr_df.columns:
            pred_weights = pred_expr_df['weight'].values.astype(np.float64)

        # Precompute anchors once for all M3 scoring.
        anchors = compute_anchors(
            holdout_expr_vals, holdout_labels, holdout_time,
            n_bootstrap=100, max_samples=2000, seed=42,
        )

        results['M3_distribution'] = score_M3_distribution(
            pred_expr_vals, holdout_expr_vals,
            train_expr_vals, train_labels, holdout_labels,
            pred_time=pred_time,
            holdout_time=holdout_time,
            train_time=train_time,
            fate_clf=fate_clf,
            pred_weights=pred_weights,
            holdout_weights=holdout_weights,
            anchors=anchors,
        )
        results['M3_distribution']['gt_weight_policy'] = prediction_gt_weight_policy
        results['M3_distribution']['agent_weight_policy'] = prediction_agent_weight_policy
    except Exception as exc:
        results['M3_distribution'] = _metric_failure('M3_distribution', exc)
    
    # ── M4: Fate ──
    try:
        agent_fate = _validate_fate_probs(agent_output_dir)
        results['M4_fate'] = score_M4_fate(agent_fate, fate_gt)
    except Exception as exc:
        results['M4_fate'] = _metric_failure('M4_fate', exc)
    
    # ── M5: Perturbation ──
    try:
        agent_pert = _validate_perturbation_results(agent_output_dir, gene_names)
        results['M5_perturbation'] = score_M5_perturbation(agent_pert, pert_gt)
    except Exception as exc:
        results['M5_perturbation'] = _metric_failure('M5_perturbation', exc)
    
    # ── M6: GRN Edge Prediction ──
    try:
        agent_grn = _validate_grn_edges(_drivers(), gene_names)
        results['M6_grn'] = score_M6_grn(agent_grn, grn_gt['edges'], gene_names)
    except Exception as exc:
        results['M6_grn'] = _metric_failure('M6_grn', exc)
    
    # ── Weighted Total ──
    total_score = sum(
        METRIC_WEIGHTS[key] * results[key]['score']
        for key in METRIC_WEIGHTS
    )
    
    summary = {
        'total_score': float(total_score),
        'weights': METRIC_WEIGHTS,
        'per_metric': {k: results[k] for k in METRIC_WEIGHTS},
        'validation_mode': 'per_metric_strict_no_repair',
        'failed_metrics': [
            key for key in METRIC_WEIGHTS
            if results[key].get('validation_failed')
        ],
    }
    
    if verbose:
        print("\n" + "=" * 60)
        print("DynBench v2 Evaluation Results")
        print("=" * 60)
        for key, weight in METRIC_WEIGHTS.items():
            r = results[key]
            s = r['score']
            err = r.get('error', '')
            marker = '✅' if s > 0.3 else '⚠️' if s > 0.0 else '❌'
            line = f"  {marker} {key:25s}  score={s:.3f}  weight={weight:.0%}"
            if err:
                line += f"  ({err})"
            print(line)
            
            # Sub-metric details
            if key == 'M2_growth' and 'driver_auprc' in r:
                print(f"      ↳ driver_auprc={r['driver_auprc']:.3f}")
            if key == 'M6_grn' and 'edge_auprc' in r:
                print(f"      ↳ edge_auprc={r['edge_auprc']:.3f}  direction={r['direction_accuracy']:.3f}  edges={r['n_gt_edges']}/{r['n_possible_edges']}  baseline={r['random_baseline']:.3f}")
                if 'strict_f1' in r:
                    print(f"      ↳ strict TP={r['strict_tp']}  strict_F1={r['strict_f1']:.3f}  strict_prec={r['strict_precision']:.3f}  strict_recall={r['strict_recall']:.3f}")
            if key == 'M3_distribution' and 'w1_score' in r:
                print(f"      ↳ W1={r['w1_score']:.3f}  composition={r['composition_score']:.3f}")
                print(f"      ↳ d_pred={r['w1_distance']:.4f}  d_noise={r['d_noise']:.4f}  d_mismatch={r['d_mismatch']:.4f}")
                pred_p = r.get('pred_fate_proportions', {})
                gt_p = r.get('gt_fate_proportions', {})
                if pred_p:
                    print(f"      ↳ pred: {', '.join(f'{k}={v:.1%}' for k,v in pred_p.items())}")
                    print(f"      ↳ gt:   {', '.join(f'{k}={v:.1%}' for k,v in gt_p.items())}")
            if key == 'M5_perturbation':
                if 'detection_F1' in r:
                    print(f"      ↳ detection_F1={r['detection_F1']:.3f}  sign_accuracy={r['sign_accuracy']:.3f}")
        
        print(f"\n  {'TOTAL':25s}  score={total_score:.3f}")
        print("=" * 60)
    
    return summary


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3:
        evaluate_all(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python dynbench_eval_v2.py <agent_output_dir> <gt_dir>")
