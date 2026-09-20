"""
DynBench v2 - Scenario S1: 3-Gene Toggle Switch with Growth

GRN topology (following TIGON / BoolODE):
  Gene_1 → self-activation, represses Gene_2  → drives Fate_A
  Gene_2 → self-activation, represses Gene_1  → drives Fate_B
  Gene_3 → activated by both Gene_1 and Gene_2 → housekeeping / growth driver

Growth (TIGON-style):
  growth_rate = sigmoid(Gene_3) → cells with higher Gene_3 grow faster
  Wrapped as Poisson birth/death on top of BoolODE SDE

Scenarios:
  S1a: Symmetric growth (equal birth rate in both fates)
  S1b: Asymmetric growth (Fate_A grows 2x faster than Fate_B)
"""

import sys
import os
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from benchmark.dynbench.simulators.boolode_sim import (
    ScenarioConfig, GRNEdge, GrowthConfig,
    simulate, simulate_perturbation, compute_percell_fate_probability,
)


def build_S1_config(
    n_cells: int = 2000,
    t_end: float = 8.0,
    sigma: float = 0.15,
    growth_enabled: bool = True,
    seed: int = 42,
) -> ScenarioConfig:
    """
    Build the S1 toggle switch scenario.
    
    GRN:
      Gene_0 --[activate]--> Gene_0 (self-activation)
      Gene_0 --[repress]---> Gene_1
      Gene_1 --[activate]--> Gene_1 (self-activation)
      Gene_1 --[repress]---> Gene_0
      Gene_0 --[activate]--> Gene_2
      Gene_1 --[activate]--> Gene_2
    """
    grn_edges = [
        # Toggle switch: mutual inhibition with self-activation
        GRNEdge(source=0, target=0, effect=1.0, K=0.5, n=3.0),    # Gene_0 self-activates
        GRNEdge(source=1, target=0, effect=-1.0, K=0.5, n=3.0),   # Gene_1 represses Gene_0
        GRNEdge(source=1, target=1, effect=1.0, K=0.5, n=3.0),    # Gene_1 self-activates
        GRNEdge(source=0, target=1, effect=-1.0, K=0.5, n=3.0),   # Gene_0 represses Gene_1
        # Gene_2: activated by both (downstream reporter)
        GRNEdge(source=0, target=2, effect=1.0, K=0.3, n=2.0),    # Gene_0 activates Gene_2
        GRNEdge(source=1, target=2, effect=1.0, K=0.3, n=2.0),    # Gene_1 activates Gene_2
    ]
    
    # Parameters
    production_rates = np.array([1.0, 1.0, 0.8])    # max production
    degradation_rates = np.array([0.3, 0.3, 0.5])   # degradation
    
    # Initial condition: near the unstable fixed point (both genes ~equal)
    initial_mean = np.array([0.5, 0.5, 0.4])
    
    # Snapshot times: 6 time points (will hold out some for evaluation)
    snapshot_times = [0.0, 1.5, 3.0, 4.5, 6.0, 8.0]
    
    # Growth: Gene_2 drives growth
    growth = GrowthConfig(
        enabled=growth_enabled,
        growth_gene_idx=2,
        base_rate=-0.05,
        amplitude=0.2,
        threshold=0.5,
        beta=5.0,
    )
    
    return ScenarioConfig(
        name="S1_toggle_switch",
        n_genes=3,
        n_cells=n_cells,
        grn_edges=grn_edges,
        production_rates=production_rates,
        degradation_rates=degradation_rates,
        t_start=0.0,
        t_end=t_end,
        dt=0.005,
        sigma=sigma,
        snapshot_times=snapshot_times,
        initial_mean=initial_mean,
        initial_std=0.15,
        growth=growth,
        seed=seed,
    )


def generate_S1(output_dir: str, verbose: bool = True):
    """Generate S1 data + all ground truth artifacts."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    config = build_S1_config()
    
    if verbose:
        print("=" * 60)
        print("DynBench v2 - S1: 3-Gene Toggle Switch")
        print("=" * 60)
        print(f"Cells: {config.n_cells}")
        print(f"Genes: {config.n_genes}")
        print(f"Time: [{config.t_start}, {config.t_end}]")
        print(f"Snapshots: {config.snapshot_times}")
        print(f"Sigma (noise): {config.sigma}")
        print(f"Growth enabled: {config.growth.enabled}")
        print(f"Seed: {config.seed}")
        print()
    
    # ── 1. Run simulation ──
    print("Step 1: Running BoolODE simulation...")
    result = simulate(config, verbose=verbose)
    
    # ── 2. Save snapshots as CSV/h5ad ──
    print("\nStep 2: Saving snapshot data...")
    gene_names = [f"Gene_{i+1}" for i in range(config.n_genes)]
    
    all_cells = []
    all_meta = []
    cell_id = 0
    
    for t_idx, t in enumerate(sorted(result.snapshots.keys())):
        expr = result.snapshots[t]
        vel = result.velocity_gt[t]
        growth = result.growth_gt[t]
        weights = result.cell_weights[t]
        fates = assign_fates_at_time(expr, t, config)
        
        # Population now naturally varies due to birth/death events in simulator
        # Weight = exp(cumulative_growth) tracks effective mass for OT
        total_mass = weights.sum()
        
        if verbose:
            print(f"  t={t:.1f}: {len(weights)} cells, weight_sum={total_mass:.0f}, "
                  f"weight range=[{weights.min():.2f}, {weights.max():.2f}]")
        
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
    
    # Save full metadata
    df_meta.to_csv(os.path.join(output_dir, 'full_metadata.csv'))
    
    # Save velocity GT
    vel_cols = [f"velocity_{g}" for g in gene_names]
    df_meta[vel_cols].to_csv(os.path.join(output_dir, 'velocity_ground_truth.csv'))
    
    print(f"  Saved {len(df_expr)} cells x {len(expr_cols)} genes")
    
    # ── 3. Save GRN ground truth ──
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
    }
    with open(os.path.join(output_dir, 'grn_ground_truth.json'), 'w') as f:
        json.dump(grn_export, f, indent=2)
    
    # Use cells from the first snapshot (t=0) as initial conditions
    t0 = sorted(result.snapshots.keys())[0]
    initial_cells = result.snapshots[t0]

    # ── 4. Perturbation GT ──
    print("\nStep 4: Computing perturbation ground truth...")
    perturbation_results = []
    for gene_idx in range(min(config.n_genes, 3)):
        print(f"  Perturbing Gene_{gene_idx+1}...")
        pert = simulate_perturbation(
            config,
            gene_idx,
            n_runs=10,
            verbose=verbose,
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
        config, initial_cells, n_runs=30
    )
    
    fate_prob_export = {}
    for i in range(initial_cells.shape[0]):
        cell_id = f"cell_{i}"
        fate_prob_export[cell_id] = {
            fate_names[j]: float(fate_probs[i, j])
            for j in range(len(fate_names))
        }
    
    with open(os.path.join(output_dir, 'percell_fate_ground_truth.json'), 'w') as f:
        json.dump(fate_prob_export, f, indent=2)
    
    print(f"  Mean P(Fate_A) = {fate_probs[:, 0].mean():.3f}")
    print(f"  Mean P(Fate_B) = {fate_probs[:, 1].mean():.3f}" if fate_probs.shape[1] > 1 else "")
    
    # ── 6. Save config for reproducibility ──
    config_export = {
        "name": config.name,
        "n_genes": config.n_genes,
        "n_cells": config.n_cells,
        "t_start": config.t_start,
        "t_end": config.t_end,
        "dt": config.dt,
        "sigma": config.sigma,
        "snapshot_times": config.snapshot_times,
        "seed": config.seed,
        "growth_enabled": config.growth.enabled,
        "growth_gene": gene_names[config.growth.growth_gene_idx],
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
    
    # ── Summary ──
    print("\n" + "=" * 60)
    print("S1 Generation Complete!")
    print(f"  Output: {output_dir}")
    print(f"  Files:")
    for f in os.listdir(output_dir):
        size = os.path.getsize(os.path.join(output_dir, f))
        print(f"    {f} ({size//1024}KB)")
    print("=" * 60)
    
    return result, config


def assign_fates_at_time(expr: np.ndarray, t: float, config: ScenarioConfig) -> np.ndarray:
    """Assign fate labels based on expression at a given time."""
    if t < 2.0:
        # Early: all progenitors
        return np.array(["Progenitor"] * expr.shape[0])
    else:
        # After bifurcation: Gene_0 > Gene_1 → Fate_A, else Fate_B
        return np.where(expr[:, 0] > expr[:, 1], "Fate_A", "Fate_B")


if __name__ == "__main__":
    output_dir = os.path.join(
        os.path.dirname(__file__), '..', 'ground_truth', 'S1_v2'
    )
    generate_S1(output_dir)
