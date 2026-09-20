"""
DynBench v2 Snapshot Builder

Takes BoolODE simulation output and builds a task package:
  - train.h5ad:  training data (selected time bins)
  - ground_truth.json: all GT fields
  - TASK.md: task description for agent
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import anndata as ad
from scipy.sparse import csr_matrix

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from benchmark.dynbench.eval.fate_classifier import train_fate_classifier


def build_task_package(
    gt_dir: str,
    output_dir: str,
    train_bins: list = [0, 1, 4, 5],
    holdout_bins: list = [2, 3],
    perturbation_genes: list = None,
    scenario_name: str = "S1",
):
    """
    Build a task package from BoolODE GT data.
    
    Args:
        gt_dir: path to ground_truth/S1_v2/
        output_dir: path to task_packages/S1_v2/
        train_bins: which time bins are given to the agent
        holdout_bins: which are held out for evaluation
        perturbation_genes: genes to perturb (default: all)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load full data
    meta = pd.read_csv(os.path.join(gt_dir, 'full_metadata.csv'), index_col=0)
    expr = pd.read_csv(os.path.join(gt_dir, 'counts_cells_x_genes.csv'), index_col=0)
    gene_names = list(expr.columns)
    
    with open(os.path.join(gt_dir, 'simulation_config.json')) as f:
        sim_config = json.load(f)
    with open(os.path.join(gt_dir, 'perturbation_ground_truth.json')) as f:
        pert_gt = json.load(f)
    with open(os.path.join(gt_dir, 'percell_fate_ground_truth.json')) as f:
        fate_gt = json.load(f)
    with open(os.path.join(gt_dir, 'grn_ground_truth.json')) as f:
        grn_gt = json.load(f)
    
    # ── 1. Build train.h5ad ──
    print("Building train.h5ad...")
    train_mask = meta['time_bin'].isin(train_bins)
    train_meta = meta.loc[train_mask].copy()
    train_expr = expr.loc[train_mask].copy()
    
    # Create AnnData
    adata = ad.AnnData(
        X=csr_matrix(train_expr.values.astype(np.float32)),
        obs=pd.DataFrame({
            'time_bin': train_meta['time_bin'].values,
            'time': train_meta['time'].values,
            'cell_type': train_meta['fate_true'].values,
            'weight': train_meta['weight'].values if 'weight' in train_meta.columns else 1.0,
        }, index=train_meta.index),
        var=pd.DataFrame(index=gene_names),
    )
    adata.write_h5ad(os.path.join(output_dir, 'train.h5ad'))
    print(f"  train.h5ad: {adata.shape[0]} cells x {adata.shape[1]} genes")
    print(f"  Time bins: {sorted(adata.obs['time_bin'].unique())}")
    print(f"  Cell types: {dict(adata.obs['cell_type'].value_counts())}")
    if 'weight' in adata.obs.columns:
        for tb in sorted(adata.obs['time_bin'].unique()):
            w = adata.obs.loc[adata.obs['time_bin'] == tb, 'weight']
            print(f"    bin={tb}: n={len(w)}, total_mass={w.sum():.0f}")
    
    # ── 1b. Train MLP fate classifier on ALL time points (gold standard) ──
    # The classifier is an oracle tool — it sees all data including holdout
    # so it can correctly classify cells at any developmental stage.
    print("Training MLP fate classifier (gold standard, all timepoints)...")
    clf_save_path = os.path.join(output_dir, 'fate_classifier.pkl')
    fate_clf = train_fate_classifier(
        expr.values.astype(np.float32),        # ALL cells, all time points
        meta['fate_true'].values,              # ALL labels
        meta['time'].values.astype(np.float32), # ALL times
        save_path=clf_save_path,
    )
    
    # ── 2. Build ground_truth.json ──
    print("Building ground_truth.json...")
    
    gt_json = {
        "scenario": scenario_name,
        "simulator": "BoolODE-style (Hill function ODE + SDE noise + Poisson growth)",
        "n_genes": len(gene_names),
        "gene_names": gene_names,
        "train_bins": train_bins,
        "holdout_bins": holdout_bins,
        "snapshot_times": sim_config['snapshot_times'],
        
        # M1: velocity GT file path
        "velocity_gt_file": os.path.join(gt_dir, 'velocity_ground_truth.csv'),
        
        # M2: growth GT in metadata
        "growth_gt_file": os.path.join(gt_dir, 'full_metadata.csv'),
        
        # M3: holdout cells
        "holdout": {
            "bins": holdout_bins,
            "n_cells": int(meta['time_bin'].isin(holdout_bins).sum()),
            "expression_file": os.path.join(gt_dir, 'counts_cells_x_genes.csv'),
            "metadata_file": os.path.join(gt_dir, 'full_metadata.csv'),
        },
        
        # M4: per-cell fate
        "percell_fate_gt_file": os.path.join(gt_dir, 'percell_fate_ground_truth.json'),
        
        # M5: perturbation
        "perturbation": pert_gt,
        
        # M6: GRN edges (for driver gene evaluation)
        "grn": grn_gt,
        
        # Perturbation targets for agent
        "perturbation_targets": [p['gene_name'] for p in pert_gt],
        
        # Unified MLP fate classifier
        "fate_classifier_path": clf_save_path,
    }
    
    with open(os.path.join(output_dir, 'ground_truth.json'), 'w') as f:
        json.dump(gt_json, f, indent=2, default=str)
    
    # ── 3. Build TASK.md (only if not already present) ──
    task_md_path = os.path.join(output_dir, 'TASK.md')
    if os.path.exists(task_md_path):
        print(f"TASK.md already exists, skipping auto-generation. Edit manually at: {task_md_path}")
    else:
        print("Building TASK.md...")
        task_md = f"""# DynBench v2 - {scenario_name}: Multi-timepoint Prediction Task

## Objective
You are given single-cell gene expression data from a multi-timepoint system.
Your task is to build or use an analysis approach that captures per-cell dynamics,
growth behavior, cell fate probabilities, and perturbation responses.

## Data
- `train.h5ad`: Gene expression data at {len(train_bins)} time points
  - `adata.obs['time_bin']`: Time bin index
  - `adata.obs['time']`: Actual time value
  - `adata.obs['cell_type']`: Cell type / fate labels
  - Genes: {', '.join(gene_names)}
  - Holdout time bins {holdout_bins} are NOT provided

## What You Must Produce

## Completion Contract
- This task is complete only when the required output files are actually written to the requested output directory.
- Narrative summaries, plans, or "done" messages do not count as completion unless the files are present and readable.
- Historical benchmark outputs found elsewhere in the workspace are reference-only and do not satisfy this task.
- Do not claim success by citing an old batch directory; the required files must come from the current requested output directory / current run.
- If you use a temporary staging area such as `_workspace`, copy the validated outputs into the requested output directory before stopping.
- Before stopping, verify that each required file exists, is non-empty, and has the expected basic schema.

## Minimal Execution Template
1. Read `train.h5ad`, `prediction_targets.json`, and `fate_classifier.pkl`.
2. Perform only the minimum modeling needed to produce the required benchmark files.
3. Write the required output files.
4. Verify existence, non-zero size, and basic schema.
5. If any file is missing or invalid, rerun the export path for that artifact. Do not fill placeholders, statistical shortcuts, or fallback approximations.
6. Stop once the file contract is satisfied.

## Public Sanity Checks
- Use only the task-package inputs, the public file contract, and your own generated outputs.
- Do not read hidden evaluator data or ground-truth answers.
- Sanity checks should focus on existence, shape, required keys/columns, and obvious degeneracies in your own outputs.

## Representation Contract
- This benchmark is defined directly in the measured gene state space.
- Treat the provided gene expression coordinates as the model state by default.
- `gene_space_equals_latent = true`
- Set `X_latent = X` and do not apply an extra PCA compression unless the task explicitly requires a different embedding.

### 1. Holdout Distribution Prediction
Predict the cell distribution at the holdout time points.
- Output: `output/holdout_prediction.csv` — gene expression matrix plus a `time` column
- Include columns: {', '.join(gene_names)}, time
- Write with `index=False`; implicit pandas index columns are invalid.

### 2. Velocity Field
Predict the velocity (rate of change) of gene expression for each cell.  
The velocity should represent dx/dt at each cell's position.
- Output: `output/velocity_field.csv` — (n_cells x {len(gene_names)}) velocity matrix
- Columns: {', '.join([f'velocity_{g}' for g in gene_names])}
- Rows should match the cells in train.h5ad (same order)
- Write with `index=False`; implicit pandas index columns are invalid.

### 3. Growth Rates
Predict the per-cell growth rate (positive = proliferation, negative = death).
- Output: `output/growth_rates.csv` — (n_cells x 1) with column `growth_rate`
- Rows should match the cells in train.h5ad (same order)
- Write with `index=False`; implicit pandas index columns are invalid.

### 4. Per-cell Fate Probability
For each cell at t=0, predict the probability of reaching each terminal fate.
Run forward simulation multiple times with stochasticity and count outcomes.
- Output: `output/per_cell_fate.json`
- Format: {{"cell_0": {{"Fate_A": 0.7, "Fate_B": 0.3}}, ...}}
- Only include cells from time_bin=0

### 5. Perturbation Analysis
For each perturbation target gene, run the benchmark-canonical perturbation protocol:
- Work directly in the measured gene state space (`gene_space_equals_latent = true`)
- Use the time_bin=0 cells as the root-cell population
- Build a control branch with `control_z_score = 0.0`
- Build a perturbed branch with `perturbed_z_score = -5.0`
- Apply the intervention directly to the target gene coordinate; do not add PCA/reconstruction steps
- Run forward simulation and compare terminal fate proportions
- Output: `output/perturbation_results.json`
- Format: [{{"gene_name": "Gene_1", "control": {{"Fate_A": 0.6, "Fate_B": 0.4}}, "perturbed": {{"Fate_A": 0.3, "Fate_B": 0.7}}, "delta": {{"Fate_A": -0.3, "Fate_B": 0.3}}}}, ...]
- Perturbation targets: {', '.join([p['gene_name'] for p in pert_gt])}

### 6. Driver Gene Analysis
Identify which genes drive velocity direction and growth rate.
- Compute Jacobian ∂velocity/∂gene and ∂growth/∂gene
- Output: `output/driver_genes.json`
- Format: {{"grn_edges": [{{"source": "Gene_1", "target": "Gene_2", "score": 0.8}}], "growth_drivers": {{"Gene_1": 0.3, ...}}}}

## Notes
- The underlying system is a gene regulatory network with ODE dynamics + stochastic noise
- {gene_names[0]} and {gene_names[1]} are in a toggle switch (mutual inhibition)
- {gene_names[2]} is a downstream gene activated by both
- Growth rate depends on {gene_names[2]} expression
- Use the CytoBridge model's velocity_net, growth_net, and score_net for analysis
"""
        with open(task_md_path, 'w') as f:
            f.write(task_md)
    
    print(f"\nTask package created in: {output_dir}")
    for f_name in os.listdir(output_dir):
        size = os.path.getsize(os.path.join(output_dir, f_name))
        print(f"  {f_name} ({size//1024}KB)")


if __name__ == "__main__":
    gt_dir = os.path.join(os.path.dirname(__file__), '..', 'ground_truth', 'S1_v2')
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'task_packages', 'S1_v2')
    build_task_package(gt_dir, output_dir)
