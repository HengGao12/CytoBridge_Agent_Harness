"""
DynBench v2 - Scenario S3: 10-Gene Complex Tree with 4 Fates

GRN topology (hierarchical bifurcation):
  Layer 1 (toggle switch 1):
    Gene_1 ↔ Gene_2  (mutual inhibition + self-activation)
    → First bifurcation: Gene_1-high vs Gene_2-high

  Layer 2 (toggle switch 2, independent):
    Gene_3 ↔ Gene_4  (mutual inhibition + self-activation)
    → Second bifurcation (independent of layer 1)

  Reporters:
    Gene_5: activated by Gene_1 (marker for Gene_1-high lineage)
    Gene_6: activated by Gene_2 (marker for Gene_2-high lineage)
    Gene_7: activated by Gene_3 (marker for Gene_3-high sub-lineage)
    Gene_8: activated by Gene_4 (marker for Gene_4-high sub-lineage)

  Late markers (confounding — both increase with time):
    Gene_9:  activated by Gene_5 and Gene_6 (downstream of ALL lineages)
    Gene_10: activated by Gene_7 and Gene_8 (downstream of ALL lineages)

4 Terminal Fates:
  B: Gene_1↑ Gene_3↑ (Gene_5↑ Gene_7↑)
  C: Gene_1↑ Gene_4↑ (Gene_5↑ Gene_8↑)
  D: Gene_2↑ Gene_3↑ (Gene_6↑ Gene_7↑)
  E: Gene_2↑ Gene_4↑ (Gene_6↑ Gene_8↑)

Growth:
  growth_rate = sigmoid(mean(Gene_5, Gene_6))
  Gene_5 and Gene_6 are the TRUE growth drivers (reporter genes)
  Gene_9 and Gene_10 are confounding (similar time trend, NOT growth drivers)
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
import pickle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from benchmark.dynbench.simulators.boolode_sim import (
    ScenarioConfig, GRNEdge, GrowthConfig,
    simulate, simulate_perturbation, compute_percell_fate_probability,
)


def build_S3_config(
    n_cells: int = 2000,
    t_end: float = 12.0,
    sigma: float = 0.12,
    growth_enabled: bool = True,
    seed: int = 42,
) -> ScenarioConfig:
    """
    Build the S3 complex tree scenario.

    GRN: 10 genes, 16 regulatory edges, hierarchical bifurcation.
    """
    grn_edges = [
        # ── Toggle switch 1: Gene_1 ↔ Gene_2 ──
        GRNEdge(source=0, target=0, effect=1.0, K=0.5, n=3.0),     # Gene_1 self-activates
        GRNEdge(source=1, target=0, effect=-1.0, K=0.5, n=3.0),    # Gene_2 represses Gene_1
        GRNEdge(source=1, target=1, effect=1.0, K=0.5, n=3.0),     # Gene_2 self-activates
        GRNEdge(source=0, target=1, effect=-1.0, K=0.5, n=3.0),    # Gene_1 represses Gene_2

        # ── Toggle switch 2: Gene_3 ↔ Gene_4 (driven by layer 1) ──
        GRNEdge(source=2, target=2, effect=1.0, K=0.5, n=3.0),     # Gene_3 self-activates
        GRNEdge(source=3, target=2, effect=-1.0, K=0.5, n=3.0),    # Gene_4 represses Gene_3
        GRNEdge(source=3, target=3, effect=1.0, K=0.5, n=3.0),     # Gene_4 self-activates
        GRNEdge(source=2, target=3, effect=-1.0, K=0.5, n=3.0),    # Gene_3 represses Gene_4

        # No cross-layer activation: toggle switches bifurcate independently
        # This creates balanced 4 fates

        # ── Reporters ──
        GRNEdge(source=0, target=4, effect=1.0, K=0.3, n=2.0),     # Gene_1 → Gene_5
        GRNEdge(source=1, target=5, effect=1.0, K=0.3, n=2.0),     # Gene_2 → Gene_6
        GRNEdge(source=2, target=6, effect=1.0, K=0.3, n=2.0),     # Gene_3 → Gene_7
        GRNEdge(source=3, target=7, effect=1.0, K=0.3, n=2.0),     # Gene_4 → Gene_8

        # ── Late markers (confounding: both activated by all lineages) ──
        GRNEdge(source=4, target=8, effect=1.0, K=0.3, n=2.0),     # Gene_5 → Gene_9
        GRNEdge(source=5, target=8, effect=1.0, K=0.3, n=2.0),     # Gene_6 → Gene_9
        GRNEdge(source=6, target=9, effect=1.0, K=0.3, n=2.0),     # Gene_7 → Gene_10
        GRNEdge(source=7, target=9, effect=1.0, K=0.3, n=2.0),     # Gene_8 → Gene_10
    ]

    # Per-gene parameters
    production_rates = np.array([
        1.0, 1.0,       # TFs layer 1
        0.9, 0.9,       # TFs layer 2
        0.7, 0.7,       # reporters 1
        0.7, 0.7,       # reporters 2
        0.6, 0.6,       # functional
    ])
    degradation_rates = np.array([
        0.3, 0.3,       # TFs layer 1
        0.35, 0.35,     # TFs layer 2
        0.4, 0.4,       # reporters 1
        0.4, 0.4,       # reporters 2
        0.5, 0.5,       # functional
    ])

    # Initial condition: near the unstable center
    initial_mean = np.array([0.5, 0.5, 0.5, 0.5, 0.3, 0.3, 0.3, 0.3, 0.2, 0.2])

    # 8 time bins, will hold out 2 intermediate for evaluation
    snapshot_times = [0.0, 1.5, 3.0, 4.5, 6.0, 8.0, 10.0, 12.0]

    # Growth: Gene_5 (idx=4) + Gene_6 (idx=5) drive growth
    # Gene_9/Gene_10 are confounding (similar time profile but NOT growth drivers)
    growth = GrowthConfig(
        enabled=growth_enabled,
        growth_gene_idx=[4, 5],  # Gene_5 and Gene_6
        base_rate=-0.02,
        amplitude=0.06,
        threshold=0.3,
        beta=5.0,
    )

    return ScenarioConfig(
        name="S3_complex_tree",
        n_genes=10,
        n_cells=n_cells,
        grn_edges=grn_edges,
        production_rates=production_rates,
        degradation_rates=degradation_rates,
        t_start=0.0,
        t_end=t_end,
        dt=0.1,
        sigma=sigma,
        snapshot_times=snapshot_times,
        initial_mean=initial_mean,
        initial_std=0.12,
        growth=growth,
        seed=seed,
    )


def assign_fates_S3(expr: np.ndarray, t: float) -> np.ndarray:
    """
    Assign fate labels based on expression at a given time.

    Hierarchical:
      t < 2.0: Progenitor
      t < 5.0: Gene_1 > Gene_2 → "branch_1", else → "branch_2" (intermediate)
      t >= 5.0: Full 4-fate assignment based on both toggle switches
    """
    n = expr.shape[0]
    if t < 2.0:
        return np.array(["A"] * n)  # A = progenitor-like early state

    # Layer 1: Gene_1 vs Gene_2
    layer1 = expr[:, 0] > expr[:, 1]  # True = Gene_1-high

    if t < 5.0:
        # Intermediate: 2 branches
        fates = np.where(layer1, "B", "C")  # simplified intermediate
        return fates

    # Layer 2: Gene_3 vs Gene_4
    layer2 = expr[:, 2] > expr[:, 3]  # True = Gene_3-high

    fates = np.empty(n, dtype='U1')
    fates[(layer1) & (layer2)] = "B"     # Gene_1↑ Gene_3↑
    fates[(layer1) & (~layer2)] = "C"    # Gene_1↑ Gene_4↑
    fates[(~layer1) & (layer2)] = "D"    # Gene_2↑ Gene_3↑
    fates[(~layer1) & (~layer2)] = "E"   # Gene_2↑ Gene_4↑

    return fates


def generate_S3(output_dir: str, verbose: bool = True):
    """Generate S3 data + all ground truth artifacts."""

    os.makedirs(output_dir, exist_ok=True)

    config = build_S3_config()
    gene_names = [f"Gene_{i+1}" for i in range(config.n_genes)]

    if verbose:
        print("=" * 60)
        print("DynBench v2 - S3: 10-Gene Complex Tree (4 Fates)")
        print("=" * 60)
        print(f"Cells: {config.n_cells}")
        print(f"Genes: {config.n_genes}")
        print(f"Time: [{config.t_start}, {config.t_end}]")
        print(f"Snapshots: {config.snapshot_times}")
        print(f"Sigma (noise): {config.sigma}")
        print(f"Growth enabled: {config.growth.enabled}")
        print(f"Seed: {config.seed}")
        print()

    # Final-time fate function for simulator callbacks
    def fate_fn_final(x):
        return assign_fates_S3(x, config.t_end)

    # ── 1. Run simulation ──
    print("Step 1: Running BoolODE simulation...")
    result = simulate(config, verbose=verbose, fate_fn=fate_fn_final)

    # ── 2. Save snapshots ──
    print("\nStep 2: Saving snapshot data...")

    all_cells = []
    all_meta = []
    cell_id = 0

    for t_idx, t in enumerate(sorted(result.snapshots.keys())):
        expr = result.snapshots[t]
        vel = result.velocity_gt[t]
        growth = result.growth_gt[t]
        weights = result.cell_weights[t]
        fates = assign_fates_S3(expr, t)

        if verbose:
            fate_counts = {f: (fates == f).sum() for f in np.unique(fates)}
            print(f"  t={t:.1f}: {expr.shape[0]} cells, fates={fate_counts}")

        for i in range(expr.shape[0]):
            all_cells.append({
                'cell_id': f"cell_{cell_id}",
                'time_bin': t_idx,
                'time': t,
                **{gene_names[g]: expr[i, g] for g in range(config.n_genes)},
            })
            all_meta.append({
                'cell_id': f"cell_{cell_id}",
                'time_bin': t_idx,
                'time': t,
                'fate_true': fates[i],
                'growth_rate': growth[i],
                'weight': float(weights[i]),
                **{f"velocity_{gene_names[g]}": vel[i, g] for g in range(config.n_genes)},
            })
            cell_id += 1

    df_expr = pd.DataFrame(all_cells).set_index('cell_id')
    df_meta = pd.DataFrame(all_meta).set_index('cell_id')

    # Save expression
    expr_cols = gene_names
    df_expr[expr_cols].to_csv(os.path.join(output_dir, 'counts_cells_x_genes.csv'))
    df_meta.to_csv(os.path.join(output_dir, 'full_metadata.csv'))

    vel_cols = [f"velocity_{g}" for g in gene_names]
    df_meta[vel_cols].to_csv(os.path.join(output_dir, 'velocity_ground_truth.csv'))

    print(f"  Saved {len(df_expr)} cells x {len(expr_cols)} genes")

    # ── 3. GRN ground truth ──
    print("\nStep 3: Saving GRN ground truth...")
    grn_export = {
        "edges": [
            {
                "from": gene_names[e.source],
                "to": gene_names[e.target],
                "effect": e.effect,
                "direction": "activate" if e.effect > 0 else "inhibit",
                "K": e.K,
                "n": e.n,
            }
            for e in config.grn_edges
        ],
        "regulators": list(set(
            [gene_names[e.source] for e in config.grn_edges] +
            [gene_names[e.target] for e in config.grn_edges]
        )),
        "growth_driver_genes": [gene_names[i] for i in config.growth.growth_gene_idx],
    }
    with open(os.path.join(output_dir, 'grn_ground_truth.json'), 'w') as f:
        json.dump(grn_export, f, indent=2)

    t0 = sorted(result.snapshots.keys())[0]
    initial_cells = result.snapshots[t0]

    # ── 4. Perturbation GT ──
    print("\nStep 4: Computing perturbation ground truth...")
    perturbation_results = []
    for gene_idx in range(config.n_genes):
        print(f"  Perturbing Gene_{gene_idx+1}...")
        pert = simulate_perturbation(
            config,
            gene_idx,
            n_runs=5,
            verbose=False,
            fate_fn=fate_fn_final,
            initial_cells=initial_cells,
            reference_cells=initial_cells,
            control_z_score=0.0,
            perturbed_z_score=-5.0,
        )
        pert["gene_name"] = gene_names[gene_idx]
        perturbation_results.append(pert)

    with open(os.path.join(output_dir, 'perturbation_ground_truth.json'), 'w') as f:
        json.dump(perturbation_results, f, indent=2, default=str)

    # ── 5. Per-cell fate probability GT ──
    print("\nStep 5: Computing per-cell fate probabilities...")

    fate_probs, fate_names = compute_percell_fate_probability(
        config, initial_cells, n_runs=15, fate_fn=fate_fn_final
    )

    # Override fate names to use abstract labels
    # The simulator uses Fate_A/Fate_B internally, remap
    fate_prob_export = {}
    for i in range(initial_cells.shape[0]):
        cell_id_str = f"cell_{i}"
        fate_prob_export[cell_id_str] = {
            fate_names[j]: float(fate_probs[i, j])
            for j in range(len(fate_names))
        }

    with open(os.path.join(output_dir, 'percell_fate_ground_truth.json'), 'w') as f:
        json.dump(fate_prob_export, f, indent=2)

    for j, fn in enumerate(fate_names):
        print(f"  Mean P({fn}) = {fate_probs[:, j].mean():.3f}")

    # ── 6. Save config ──
    config_export = {
        "name": config.name,
        "n_genes": config.n_genes,
        "n_cells": config.n_cells,
        "n_fates": 4,
        "t_start": config.t_start,
        "t_end": config.t_end,
        "dt": config.dt,
        "sigma": config.sigma,
        "snapshot_times": config.snapshot_times,
        "seed": config.seed,
        "growth_enabled": config.growth.enabled,
        "growth_genes": [gene_names[i] for i in config.growth.growth_gene_idx] if isinstance(config.growth.growth_gene_idx, list) else [gene_names[config.growth.growth_gene_idx]],
        "gene_space_equals_latent": True,
        "perturbation_protocol": {
            "intervention_space": "gene_space",
            "control_z_score": 0.0,
            "perturbed_z_score": -5.0,
        },
        "simulator": "BoolODE-style (Hill function ODE + SDE noise + Poisson growth)",
    }
    with open(os.path.join(output_dir, 'simulation_config.json'), 'w') as f:
        json.dump(config_export, f, indent=2)

    # ── 7. Build task package ──
    print("\nStep 6: Building task package...")
    task_dir = os.path.join(os.path.dirname(output_dir), '..', 'task_packages', 'S3')
    os.makedirs(task_dir, exist_ok=True)

    # Training data: hold out time bins 2,3 (t=3.0, 4.5)
    train_bins = [0, 1, 4, 5, 6, 7]
    holdout_bins = [2, 3]

    train_mask = df_meta['time_bin'].isin(train_bins)
    train_expr = df_expr.loc[train_mask, expr_cols]
    train_meta = df_meta.loc[train_mask, ['time_bin', 'time', 'fate_true']]
    train_meta = train_meta.rename(columns={'fate_true': 'cell_type'})

    # Create h5ad
    import anndata
    adata = anndata.AnnData(
        X=train_expr.values.astype(np.float32),
        obs=train_meta.reset_index(drop=True),
        var=pd.DataFrame(index=expr_cols),
    )
    adata.obs.index = [f"cell_{i}" for i in range(len(adata))]
    # Apply log1p normalization
    adata.X = np.log1p(adata.X)
    adata.write_h5ad(os.path.join(task_dir, 'train.h5ad'))
    print(f"  train.h5ad: {adata.shape}")

    # Train fate classifier on ALL data (oracle)
    print("  Training fate classifier...")
    all_expr_vals = df_expr[expr_cols].values
    all_times = df_meta['time'].values
    all_fates = df_meta['fate_true'].values

    X_clf = np.hstack([np.log1p(all_expr_vals), all_times.reshape(-1, 1)])
    clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=100, random_state=42)
    clf.fit(X_clf, all_fates)
    print(f"  Classifier accuracy: {clf.score(X_clf, all_fates):.4f}")
    print(f"  Classes: {clf.classes_.tolist()}")

    with open(os.path.join(task_dir, 'fate_classifier.pkl'), 'wb') as f:
        pickle.dump(clf, f)

    # ── Summary ──
    print("\n" + "=" * 60)
    print("S3 Generation Complete!")
    print(f"  Ground truth: {output_dir}")
    print(f"  Task package: {task_dir}")
    print(f"  Total cells: {len(df_expr)}")
    print(f"  Training cells: {train_mask.sum()}")
    print(f"  Holdout bins: {holdout_bins}")
    print(f"  Fate classes: {sorted(df_meta['fate_true'].unique())}")
    print(f"  Classifier accuracy: {clf.score(X_clf, all_fates):.4f}")
    for f_name in os.listdir(output_dir):
        size = os.path.getsize(os.path.join(output_dir, f_name))
        print(f"    GT: {f_name} ({size // 1024}KB)")
    for f_name in os.listdir(task_dir):
        size = os.path.getsize(os.path.join(task_dir, f_name))
        print(f"    Task: {f_name} ({size // 1024}KB)")
    print("=" * 60)

    return result, config


if __name__ == "__main__":
    output_dir = os.path.join(
        os.path.dirname(__file__), '..', 'ground_truth', 'S3'
    )
    generate_S3(output_dir)
