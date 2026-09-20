"""
DynBench Track B — Snapshot Builder & Ground Truth Extraction

Takes raw scMultiSim outputs → produces:
  1. Sanitized train.h5ad for agents (sparse snapshots, no truth columns)
  2. ground_truth.json with structured answers for evaluation
  3. holdout_cells.h5ad for M4 (distribution recovery) scoring
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd

# ─── Scenario configurations ─────────────────────────────────────────────────

SCENARIO_CONFIGS = {
    "S1": {
        "title": "Simple Bifurcation",
        "difficulty": "★★☆",
        "n_bins": 6,
        # Keep early dense bins (0,1,2) for dynamics + final (5).
        # Holdout: bin 3,4 = interpolation tests (between observed t=2 and t=3).
        "keep_bins": [0, 1, 2, 5],
        "grn_regulators": ["Gene_1", "Gene_2"],
        # Perturbation GT: Gene_1 drives Fate 4_1, Gene_2 drives Fate 4_2 (toggle switch)
        "perturbation_targets": [
            {"gene": "Gene_1", "direction": "knockout",
             "expected_shift_sign": {"4_1": -1, "4_2": 1}},
            {"gene": "Gene_2", "direction": "knockout",
             "expected_shift_sign": {"4_1": 1, "4_2": -1}},
        ],
    },
    "S2": {
        "title": "Asymmetric Growth + Transient State",
        "difficulty": "★★★",
        "n_bins": 6,
        # Keep early dense (0,1,2) + final (5).
        # Holdout: bin 3,4 = interpolation tests.
        "keep_bins": [0, 1, 2, 5],
        "grn_regulators": ["Gene_1", "Gene_2", "Gene_3"],
        # Gene_1: growth/progenitor, Gene_2: FateA TF, Gene_3: FateB TF
        "perturbation_targets": [
            {"gene": "Gene_2", "direction": "knockout",
             "expected_shift_sign": {"4_1": -1, "4_2": 1}},
            {"gene": "Gene_3", "direction": "knockout",
             "expected_shift_sign": {"4_1": 1, "4_2": -1}},
        ],
    },
    "S3": {
        "title": "Complex Tree + Dynamic GRN",
        "difficulty": "★★★★",
        "n_bins": 6,
        # Keep early dense (0,1,2) + final (5).
        # Holdout: bin 3,4 = interpolation tests.
        "keep_bins": [0, 1, 2, 5],
        "grn_regulators": None,  # extracted from grn_input.json
        "perturbation_targets": None,  # auto-generate from GRN edges
    },
}


def bin_pseudotime(
    pseudotime: np.ndarray, n_bins: int = 6
) -> np.ndarray:
    """Assign cells to pseudotime bins via equal-quantile binning."""
    bin_edges = np.quantile(pseudotime, np.linspace(0, 1, n_bins + 1))
    # Handle edge cases: ensure unique edges
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < n_bins + 1:
        bin_edges = np.linspace(
            pseudotime.min(), pseudotime.max() + 1e-8, n_bins + 1
        )
    bin_labels = np.digitize(pseudotime, bin_edges[1:-1])  # 0 to n_bins-1
    return bin_labels


def build_snapshot_adata(
    counts_path: str | Path,
    metadata_path: str | Path,
    scenario_id: str,
    output_dir: str | Path,
    grn_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Build sanitized train.h5ad + ground_truth.json from raw scMultiSim outputs.

    Args:
        counts_path: Path to counts_cells_x_genes.csv
        metadata_path: Path to full_metadata.csv
        scenario_id: "S1", "S2", or "S3"
        output_dir: Where to write train.h5ad and ground_truth.json
        grn_path: Path to grn_input.json (for extracting regulator list)

    Returns:
        Summary dict with file paths and key statistics.
    """
    config = SCENARIO_CONFIGS[scenario_id]
    n_bins = config["n_bins"]
    keep_bins = config["keep_bins"]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ─── Load data ────────────────────────────────────────────────────────
    counts = pd.read_csv(counts_path, index_col=0)
    metadata = pd.read_csv(metadata_path, index_col=0)

    # Ensure aligned indices
    shared_cells = [c for c in counts.index if c in metadata.index]
    counts = counts.loc[shared_cells]
    metadata = metadata.loc[shared_cells]

    # ─── Bin pseudotime ───────────────────────────────────────────────────
    pseudotime = metadata["pseudotime"].values.astype(float)
    bin_labels = bin_pseudotime(pseudotime, n_bins)
    metadata["time_bin"] = bin_labels

    # ─── Split into agent data and holdout ────────────────────────────────
    agent_mask = metadata["time_bin"].isin(keep_bins)
    holdout_mask = ~agent_mask

    agent_metadata = metadata[agent_mask].copy()
    holdout_metadata = metadata[holdout_mask].copy()

    # Relabel agent bins as clean time points: 0.0, 1.0, 2.0
    bin_to_time = {b: float(i) for i, b in enumerate(keep_bins)}
    agent_metadata["Time point"] = agent_metadata["time_bin"].map(bin_to_time)

    agent_counts = counts.loc[agent_metadata.index]
    holdout_counts = counts.loc[holdout_metadata.index]

    # Include cell_type labels from simulation — this is standard biological
    # prior (like FACS sorting or marker-based annotation), not data leakage.
    fate_col = "fate_true" if "fate_true" in agent_metadata.columns else None
    obs_dict = {"Time point": agent_metadata["Time point"].values}
    if fate_col:
        obs_dict["cell_type"] = agent_metadata[fate_col].values.astype(str)
    agent_obs = pd.DataFrame(
        obs_dict,
        index=agent_metadata.index.astype(str),
    )
    agent_var = pd.DataFrame(index=counts.columns)

    agent_X = agent_counts.values.astype(np.float32)
    agent_adata = ad.AnnData(X=agent_X, obs=agent_obs, var=agent_var)
    agent_adata.obs_names_make_unique()
    agent_adata.var_names_make_unique()

    agent_h5ad_path = output_dir / "train.h5ad"
    agent_adata.write_h5ad(agent_h5ad_path)

    # ─── Build holdout AnnData (for M4 scoring) ──────────────────────────
    holdout_obs = pd.DataFrame(
        {
            "time_bin": holdout_metadata["time_bin"].values,
            "fate_true": holdout_metadata.get("fate_true", "unknown").values,
        },
        index=holdout_metadata.index.astype(str),
    )
    holdout_var = pd.DataFrame(index=counts.columns)
    holdout_X = holdout_counts.values.astype(np.float32)
    holdout_adata = ad.AnnData(X=holdout_X, obs=holdout_obs, var=holdout_var)
    holdout_h5ad_path = output_dir / "holdout_cells.h5ad"
    holdout_adata.write_h5ad(holdout_h5ad_path)

    # ─── Extract ground truth ─────────────────────────────────────────────
    ground_truth = _extract_ground_truth(
        metadata, counts, config, scenario_id, grn_path
    )

    gt_path = output_dir / "ground_truth.json"
    with open(gt_path, "w") as f:
        json.dump(ground_truth, f, indent=2, ensure_ascii=False)

    # ─── Copy GRN ground truth for eval (M5) ─────────────────────────────
    if grn_path and Path(grn_path).exists():
        import shutil
        shutil.copy2(grn_path, output_dir / "grn_input.json")

    # ─── Generate perturbation_targets.json for M1 ────────────────────────
    perturbation_targets = config.get("perturbation_targets")
    if perturbation_targets is None and grn_path and Path(grn_path).exists():
        # Auto-generate from GRN: KO each regulator gene
        with open(grn_path) as f:
            grn_data = json.load(f)
        regs = set()
        for edge in grn_data.get("edges", []):
            regs.add(edge.get("from", ""))
        perturbation_targets = [
            {"gene": g, "direction": "knockout", "expected_shift_sign": {}}
            for g in sorted(regs) if g
        ]
    if perturbation_targets:
        pt_path = output_dir / "perturbation_targets.json"
        with open(pt_path, "w") as f:
            json.dump(perturbation_targets, f, indent=2, ensure_ascii=False)

    # ─── Generate prediction_targets.json for the agent ───────────────────
    # Maps holdout bins → agent time coordinates so the agent knows exactly
    # which intermediate time points to predict at.
    holdout_bins = [b for b in range(n_bins) if b not in keep_bins]
    pred_targets = []
    for b in holdout_bins:
        lower_bins = [k for k in keep_bins if k <= b]
        upper_bins = [k for k in keep_bins if k >= b]
        if not lower_bins:
            continue  # before first observed bin — skip
        lower = max(lower_bins)
        lower_idx = keep_bins.index(lower)
        if upper_bins:
            # Interpolation: between two observed bins
            upper = min(upper_bins)
            upper_idx = keep_bins.index(upper)
            if lower == upper:
                t = float(lower_idx)
            else:
                frac = (b - lower) / (upper - lower)
                t = round(float(lower_idx) + frac * (upper_idx - lower_idx), 2)
        else:
            # Extrapolation: beyond the last observed bin
            # Use same spacing as the last interval
            if lower_idx > 0:
                prev_bin = keep_bins[lower_idx - 1]
                bin_step = (float(lower_idx) - float(lower_idx - 1)) / (lower - prev_bin)
            else:
                bin_step = 1.0
            t = round(float(lower_idx) + (b - lower) * bin_step, 2)
        pred_targets.append(t)

    targets_config = {
        "prediction_time_points": pred_targets,
        "observed_time_points": sorted(
            agent_metadata["Time point"].unique().tolist()
        ),
        "note": "Predict the cell distribution at each prediction_time_point. "
                "These are held-out intermediate times between the observed times.",
    }
    targets_path = output_dir / "prediction_targets.json"
    with open(targets_path, "w") as f:
        json.dump(targets_config, f, indent=2, ensure_ascii=False)

    # ─── Summary ──────────────────────────────────────────────────────────
    summary = {
        "scenario_id": scenario_id,
        "title": config["title"],
        "difficulty": config["difficulty"],
        "n_bins": n_bins,
        "keep_bins": keep_bins,
        "holdout_bins": holdout_bins,
        "prediction_time_points": pred_targets,
        "agent_cells": int(agent_mask.sum()),
        "holdout_cells": int(holdout_mask.sum()),
        "total_cells": len(metadata),
        "n_genes": counts.shape[1],
        "agent_time_points": sorted(agent_metadata["Time point"].unique().tolist()),
        "files": {
            "train_h5ad": str(agent_h5ad_path),
            "holdout_h5ad": str(holdout_h5ad_path),
            "ground_truth": str(gt_path),
            "prediction_targets": str(targets_path),
        },
    }

    summary_path = output_dir / "build_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"  {scenario_id}: {config['title']} ({config['difficulty']})")
    print(f"{'='*60}")
    print(f"  Agent cells : {summary['agent_cells']} at bins {keep_bins}")
    print(f"  Holdout cells: {summary['holdout_cells']}")
    print(f"  Genes       : {summary['n_genes']}")
    print(f"  Time points : {summary['agent_time_points']}")
    print(f"  Prediction targets: {pred_targets}")
    print(f"  train.h5ad  : {agent_h5ad_path}")
    print(f"  ground_truth: {gt_path}")
    print(f"{'='*60}\n")

    return summary


def _extract_ground_truth(
    metadata: pd.DataFrame,
    counts: pd.DataFrame,
    config: dict,
    scenario_id: str,
    grn_path: str | Path | None,
) -> dict[str, Any]:
    """Extract structured ground truth from simulation metadata."""
    gt: dict[str, Any] = {
        "scenario_id": scenario_id,
        "title": config["title"],
    }

    fate_col = "fate_true" if "fate_true" in metadata.columns else "pop"
    if fate_col not in metadata.columns:
        fate_col = None

    # ─── Fate ratios (for M3: Fate Probability) ──────────────────────────
    if fate_col:
        fate_counts = Counter(metadata[fate_col].values)
        total = sum(fate_counts.values())
        gt["fate_ratio"] = {
            str(k): round(v / total, 4) for k, v in sorted(fate_counts.items())
        }
        gt["n_fates"] = len(fate_counts)

        # Fate ratios at earliest bin only (what progenitors become)
        earliest_bin = min(config["keep_bins"])
        early_mask = metadata["time_bin"] == earliest_bin
        if early_mask.sum() > 0:
            early_fates = Counter(metadata.loc[early_mask, fate_col].values)
            early_total = sum(early_fates.values())
            gt["progenitor_fate_ratio"] = {
                str(k): round(v / early_total, 4)
                for k, v in sorted(early_fates.items())
            }

    # ─── Growth asymmetry (for M2) ───────────────────────────────────────
    if fate_col:
        # Compare cell counts at ALL bins across fates
        bin_fate_counts: dict[int, dict[str, int]] = {}
        for b in range(config["n_bins"]):
            mask = metadata["time_bin"] == b
            if mask.sum() > 0:
                fc = Counter(metadata.loc[mask, fate_col].values)
                bin_fate_counts[b] = {str(k): v for k, v in fc.items()}

        gt["bin_fate_counts"] = {
            str(k): v for k, v in bin_fate_counts.items()
            if k in config["keep_bins"]
        }

        # Growth analysis: compute proportions of TERMINAL fates at the latest
        # observed bin. scMultiSim assigns different pop labels at different
        # tree depths, so we focus on fates present at the latest bin.
        latest = max(config["keep_bins"])
        late_fc = bin_fate_counts.get(latest, {})
        late_total = sum(late_fc.values())

        if late_total > 0:
            # Terminal fate proportions
            terminal_proportions: dict[str, float] = {
                f: round(c / late_total, 4) for f, c in sorted(late_fc.items())
            }
            gt["terminal_fate_proportions"] = terminal_proportions

            # Growth ratio = largest fraction / smallest fraction
            fates_sorted = sorted(
                terminal_proportions.items(), key=lambda x: x[1], reverse=True
            )
            fastest_fate, fastest_prop = fates_sorted[0]
            slowest_fate, slowest_prop = fates_sorted[-1]

            gt["faster_branch"] = fastest_fate
            gt["slower_branch"] = slowest_fate
            gt["growth_ratio"] = round(
                fastest_prop / max(slowest_prop, 0.001), 3
            )
            gt["growth_rates_by_fate"] = terminal_proportions

    # ─── GRN regulators + direct GRN targets (for M1) ─────────────────────
    # Only genes that are part of the GRN regulatory network (cause, not effect).
    # NOT DE marker genes — those are consequences, not drivers.
    driver_genes: list[str] = []

    # (a) GRN input regulators from config or grn_input.json
    grn_data: dict = {}
    if grn_path and Path(grn_path).exists():
        with open(grn_path) as f:
            grn_data = json.load(f)

    if config["grn_regulators"]:
        driver_genes.extend(config["grn_regulators"])
    elif grn_data.get("regulators"):
        driver_genes.extend(grn_data["regulators"])

    # (b) Direct GRN targets: genes directly regulated by the GRN edges.
    #     These are dynamically relevant (part of the regulatory circuit).
    if grn_data.get("edges"):
        for edge in grn_data["edges"]:
            src = edge.get("from", "")
            tgt = edge.get("to", "")
            if src:
                driver_genes.append(src)
            if tgt:
                driver_genes.append(tgt)

    # Deduplicate while preserving order (regulators first)
    seen: set[str] = set()
    unique_drivers: list[str] = []
    for g in driver_genes:
        if g not in seen:
            seen.add(g)
            unique_drivers.append(g)
    gt["grn_driver_genes"] = unique_drivers

    # ─── Holdout bins ─────────────────────────────────────────────────────
    gt["holdout_bins"] = [
        b for b in range(config["n_bins"]) if b not in config["keep_bins"]
    ]

    return gt


# ─── CLI entry point ──────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="DynBench: Build sparse snapshots from scMultiSim outputs"
    )
    parser.add_argument(
        "--scenario", "-s", required=True, choices=["S1", "S2", "S3"],
        help="Scenario ID",
    )
    parser.add_argument(
        "--counts", required=True,
        help="Path to counts_cells_x_genes.csv",
    )
    parser.add_argument(
        "--metadata", required=True,
        help="Path to full_metadata.csv",
    )
    parser.add_argument(
        "--grn", default=None,
        help="Path to grn_input.json (optional, for GRN ground truth)",
    )
    parser.add_argument(
        "--output", "-o", required=True,
        help="Output directory for task package",
    )

    args = parser.parse_args()

    summary = build_snapshot_adata(
        counts_path=args.counts,
        metadata_path=args.metadata,
        scenario_id=args.scenario,
        output_dir=args.output,
        grn_path=args.grn,
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
