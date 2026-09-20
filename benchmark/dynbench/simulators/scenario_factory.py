"""
DynBench Scenario Factory — Batch benchmark generation.

Two modes:
  1. Preset mode:  Fixed difficulty presets (easy/medium/hard/extreme)
  2. Config mode:  Agent-designed JSON configs with diverse structural parameters

Usage:
    # Preset mode — same structure, different seeds
    python scenario_factory.py --difficulty medium --seeds 42

    # Config mode — diverse structures from JSON
    python scenario_factory.py --config scenario_configs.json
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import pickle
import sys
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier

sys.path.insert(0, os.path.dirname(__file__))
from boolode_sim import (
    ScenarioConfig, GRNEdge, GrowthConfig, SimulationResult,
    simulate, simulate_perturbation, compute_percell_fate_probability,
    compute_growth_rate,
)

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
#  Difficulty Presets
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class DifficultyPreset:
    """High-level difficulty specification."""
    name: str
    n_fates: int                   # 2, 4, 8, 16
    n_reporters_per_tf: int        # reporters per TF gene
    n_confounders: int             # late marker genes
    n_noise_genes: int             # uncorrelated noise genes
    noise_sigma: float             # SDE noise level
    n_time_bins: int               # total time bins
    holdout_bins: List[int]        # bins to hold out
    n_init_cells: int              # initial cell count
    growth_type: str               # 'single', 'multi', 'interaction'
    t_end: float = 10.0

    @property
    def n_toggles(self) -> int:
        return math.ceil(math.log2(self.n_fates))

    @property
    def n_tfs(self) -> int:
        return self.n_toggles * 2

    @property
    def n_reporters(self) -> int:
        return self.n_tfs * self.n_reporters_per_tf

    @property
    def n_genes_total(self) -> int:
        return self.n_tfs + self.n_reporters + self.n_confounders + self.n_noise_genes


PRESETS = {
    'easy': DifficultyPreset(
        name='easy', n_fates=2,
        n_reporters_per_tf=1, n_confounders=1, n_noise_genes=0,
        noise_sigma=0.10, n_time_bins=6, holdout_bins=[2, 3],
        n_init_cells=1500, growth_type='single',
    ),
    'medium': DifficultyPreset(
        name='medium', n_fates=4,
        n_reporters_per_tf=1, n_confounders=2, n_noise_genes=0,
        noise_sigma=0.12, n_time_bins=8, holdout_bins=[2, 3],
        n_init_cells=2000, growth_type='multi', t_end=12.0,
    ),
    'hard': DifficultyPreset(
        name='hard', n_fates=8,
        n_reporters_per_tf=1, n_confounders=4, n_noise_genes=2,
        noise_sigma=0.15, n_time_bins=10, holdout_bins=[3, 4, 5],
        n_init_cells=2500, growth_type='interaction', t_end=14.0,
    ),
    'extreme': DifficultyPreset(
        name='extreme', n_fates=16,
        n_reporters_per_tf=1, n_confounders=6, n_noise_genes=8,
        noise_sigma=0.18, n_time_bins=12, holdout_bins=[4, 5, 6],
        n_init_cells=3000, growth_type='interaction', t_end=18.0,
    ),
}


TOPOLOGY_PROCEDURAL = "procedural"
TOPOLOGY_TOGGLE_SWITCH_3GENE = "toggle_switch_3gene"
TOPOLOGY_LLM_SPEC = "llm_spec"


# ══════════════════════════════════════════════════════════════════════════════
#  V2 Deterministic Seed Derivation
# ══════════════════════════════════════════════════════════════════════════════

def deterministic_seed_from_config(config: dict) -> int:
    """
    Derive a deterministic internal seed from config parameters.
    Same config always produces the same data, regardless of caller.
    """
    # Build a canonical string from all deterministic config parameters
    seed_keys = [
        'n_fates', 'n_reporters_per_tf', 'n_confounders', 'n_noise_genes',
        'growth_type', 'noise_sigma', 'n_time_bins', 'n_init_cells', 't_end',
    ]
    parts = []
    for k in sorted(seed_keys):
        if k in config:
            parts.append(f"{k}={config[k]}")
    # Include topology spec if present
    topo_spec = config.get('topology_spec')
    if isinstance(topo_spec, dict):
        for k in sorted(topo_spec.keys()):
            parts.append(f"topo.{k}={topo_spec[k]}")
    canonical = "|".join(parts)
    h = hashlib.sha256(canonical.encode()).hexdigest()
    return int(h[:8], 16)


# ══════════════════════════════════════════════════════════════════════════════
#  GRN Generator
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class GRNBlueprint:
    """Complete GRN specification with gene roles."""
    edges: List[GRNEdge]
    gene_names: List[str]
    gene_roles: Dict[int, str]           # idx -> 'tf', 'reporter', 'confounder', 'noise'
    tf_indices: List[int]
    reporter_indices: List[int]
    confounder_indices: List[int]
    noise_indices: List[int]
    toggle_pairs: List[Tuple[int, int]]  # pairs of mutually inhibiting TFs
    reporter_parent: Dict[int, int]      # reporter_idx -> parent_tf_idx
    fate_labels: List[str]               # e.g. ['A', 'B', 'C', 'D', 'E']
    growth_driver_indices: List[int]     # true growth driver gene indices
    n_genes: int


def generate_grn(preset: DifficultyPreset, rng: np.random.RandomState) -> GRNBlueprint:
    """
    Procedurally generate a GRN following the layered architecture:
      Layer 1: Toggle switches (TFs) — fate decisions
      Layer 2: Reporters — signal readouts from TFs
      Layer 3: Confounders — late markers with high time-correlation
      Layer 4: Noise genes — uncorrelated random walk
    """
    edges = []
    gene_roles = {}
    reporter_parent = {}
    gene_idx = 0

    # ── Layer 1: Toggle switches ──
    tf_indices = []
    toggle_pairs = []

    for t in range(preset.n_toggles):
        g_a = gene_idx
        g_b = gene_idx + 1
        gene_idx += 2

        tf_indices.extend([g_a, g_b])
        toggle_pairs.append((g_a, g_b))
        gene_roles[g_a] = 'tf'
        gene_roles[g_b] = 'tf'

        # Mutual inhibition (with slight randomness for variety)
        K = 0.5 + rng.uniform(-0.05, 0.05)
        n_hill = 3.0 + rng.uniform(-0.3, 0.3)

        edges.append(GRNEdge(source=g_a, target=g_a, effect=1.0, K=K, n=n_hill))   # self-activation
        edges.append(GRNEdge(source=g_b, target=g_a, effect=-1.0, K=K, n=n_hill))   # mutual inhibition
        edges.append(GRNEdge(source=g_b, target=g_b, effect=1.0, K=K, n=n_hill))   # self-activation
        edges.append(GRNEdge(source=g_a, target=g_b, effect=-1.0, K=K, n=n_hill))   # mutual inhibition

    # ── Layer 2: Reporters (one per TF) ──
    reporter_indices = []
    for tf_idx in tf_indices:
        for _ in range(preset.n_reporters_per_tf):
            r_idx = gene_idx
            gene_idx += 1
            reporter_indices.append(r_idx)
            gene_roles[r_idx] = 'reporter'
            reporter_parent[r_idx] = tf_idx

            K = 0.3 + rng.uniform(-0.05, 0.05)
            edges.append(GRNEdge(source=tf_idx, target=r_idx, effect=1.0, K=K, n=2.0))

    # ── Growth drivers: reporters from FIRST toggle ──
    # (first 2 reporters come from tf_indices[0] and tf_indices[1])
    first_toggle_reporters = reporter_indices[:2]  # one from each TF of toggle 1
    # Fallback: when n_reporters_per_tf=0, use TFs themselves as growth drivers
    if not first_toggle_reporters:
        first_toggle_reporters = list(tf_indices[:2])

    if preset.growth_type == 'single':
        growth_drivers = [first_toggle_reporters[0]]
    else:  # 'multi' or 'interaction'
        growth_drivers = first_toggle_reporters[:2]

    # ── Layer 3: Confounders (late markers) ──
    confounder_indices = []
    for c in range(preset.n_confounders):
        c_idx = gene_idx
        gene_idx += 1
        confounder_indices.append(c_idx)
        gene_roles[c_idx] = 'confounder'

        # Confounders integrate 2+ reporters from DIFFERENT toggles
        n_parents = min(2 + c, len(reporter_indices))
        # Mix reporters from different toggles for higher time-correlation
        if n_parents > 0:
            parent_reporters = list(rng.choice(reporter_indices, size=n_parents, replace=False))
        else:
            parent_reporters = []
        for p in parent_reporters:
            # Lower K = easier to activate = rises earlier = higher time-corr
            K = 0.20 + rng.uniform(-0.03, 0.03)
            edges.append(GRNEdge(source=p, target=c_idx, effect=1.0, K=K, n=2.0))

    # ── Layer 4: Noise genes (no regulation) ──
    noise_indices = []
    for _ in range(preset.n_noise_genes):
        n_idx = gene_idx
        gene_idx += 1
        noise_indices.append(n_idx)
        gene_roles[n_idx] = 'noise'

    # ── Gene names & fates ──
    n_genes = gene_idx
    gene_names = [f"Gene_{i+1}" for i in range(n_genes)]

    fate_labels = ['A']  # progenitor
    for i in range(preset.n_fates):
        fate_labels.append(chr(ord('B') + i))

    return GRNBlueprint(
        edges=edges,
        gene_names=gene_names,
        gene_roles=gene_roles,
        tf_indices=tf_indices,
        reporter_indices=reporter_indices,
        confounder_indices=confounder_indices,
        noise_indices=noise_indices,
        toggle_pairs=toggle_pairs,
        reporter_parent=reporter_parent,
        fate_labels=fate_labels,
        growth_driver_indices=growth_drivers,
        n_genes=n_genes,
    )


def generate_toggle_switch_3gene_grn(rng: np.random.RandomState) -> GRNBlueprint:
    """Generate the compact 3-gene toggle-switch topology used for sanity tasks."""
    edges = [
        GRNEdge(source=0, target=0, effect=1.0, K=0.5 + rng.uniform(-0.03, 0.03), n=3.0),
        GRNEdge(source=1, target=0, effect=-1.0, K=0.5 + rng.uniform(-0.03, 0.03), n=3.0),
        GRNEdge(source=1, target=1, effect=1.0, K=0.5 + rng.uniform(-0.03, 0.03), n=3.0),
        GRNEdge(source=0, target=1, effect=-1.0, K=0.5 + rng.uniform(-0.03, 0.03), n=3.0),
        GRNEdge(source=0, target=2, effect=1.0, K=0.3 + rng.uniform(-0.03, 0.03), n=2.0),
        GRNEdge(source=1, target=2, effect=1.0, K=0.3 + rng.uniform(-0.03, 0.03), n=2.0),
    ]
    return GRNBlueprint(
        edges=edges,
        gene_names=["Gene_1", "Gene_2", "Gene_3"],
        gene_roles={0: "tf", 1: "tf", 2: "reporter"},
        tf_indices=[0, 1],
        reporter_indices=[2],
        confounder_indices=[],
        noise_indices=[],
        toggle_pairs=[(0, 1)],
        reporter_parent={2: 0},
        fate_labels=["A", "B", "C"],
        growth_driver_indices=[2],
        n_genes=3,
    )


def _copy_grn(grn: GRNBlueprint) -> GRNBlueprint:
    return GRNBlueprint(
        edges=[deepcopy(edge) for edge in grn.edges],
        gene_names=list(grn.gene_names),
        gene_roles=dict(grn.gene_roles),
        tf_indices=list(grn.tf_indices),
        reporter_indices=list(grn.reporter_indices),
        confounder_indices=list(grn.confounder_indices),
        noise_indices=list(grn.noise_indices),
        toggle_pairs=list(grn.toggle_pairs),
        reporter_parent=dict(grn.reporter_parent),
        fate_labels=list(grn.fate_labels),
        growth_driver_indices=list(grn.growth_driver_indices),
        n_genes=grn.n_genes,
    )


def generate_grn_from_topology_spec(
    preset: DifficultyPreset,
    rng: np.random.RandomState,
    topology_spec: dict | None,
) -> GRNBlueprint:
    spec = topology_spec or {}
    family = str(spec.get("topology_family") or "branching_tree").strip().lower()
    if family == "compact_toggle":
        return generate_toggle_switch_3gene_grn(rng)

    base = generate_grn(preset, rng)
    grn = _copy_grn(base)
    feedback_strength = float(spec.get("feedback_strength", 0.0))
    coupling = float(spec.get("cross_branch_coupling", 0.0))
    asymmetry = float(spec.get("asymmetry_strength", 0.0))
    density_bias = str(spec.get("density_bias", "medium")).strip().lower()

    if family in {"branching_with_feedback", "competitive_multistable"} and grn.reporter_indices:
        for reporter_idx, parent_tf_idx in grn.reporter_parent.items():
            grn.edges.append(
                GRNEdge(
                    source=reporter_idx,
                    target=parent_tf_idx,
                    effect=max(0.15, 0.6 * max(feedback_strength, 0.2)),
                    K=0.35 + rng.uniform(-0.04, 0.04),
                    n=2.0,
                )
            )

    if family in {"branching_with_feedback", "competitive_multistable"} and len(grn.tf_indices) >= 4:
        step = 1 if density_bias == "high" else 2
        for i in range(0, len(grn.tf_indices) - step, step):
            src = grn.tf_indices[i]
            tgt = grn.tf_indices[(i + step) % len(grn.tf_indices)]
            grn.edges.append(
                GRNEdge(
                    source=src,
                    target=tgt,
                    effect=-max(0.1, 0.5 * max(coupling, 0.2)),
                    K=0.4 + rng.uniform(-0.05, 0.05),
                    n=2.5,
                )
            )

    if family == "asymmetric_tree" and grn.reporter_indices:
        for idx, reporter_idx in enumerate(grn.reporter_indices):
            parent_tf_idx = grn.reporter_parent[reporter_idx]
            scale = 1.0 + asymmetry * (0.3 if idx % 2 == 0 else -0.2)
            grn.edges.append(
                GRNEdge(
                    source=parent_tf_idx,
                    target=reporter_idx,
                    effect=max(0.3, 0.8 * scale),
                    K=0.28 + rng.uniform(-0.03, 0.03),
                    n=2.0,
                )
            )

    if density_bias == "high" and len(grn.reporter_indices) >= 2:
        sampled_reporters = grn.reporter_indices[: min(4, len(grn.reporter_indices))]
        for reporter_idx in sampled_reporters:
            regulator = int(rng.choice(grn.tf_indices))
            grn.edges.append(
                GRNEdge(
                    source=regulator,
                    target=reporter_idx,
                    effect=0.35 + rng.uniform(-0.08, 0.08),
                    K=0.32 + rng.uniform(-0.04, 0.04),
                    n=2.0,
                )
            )

    return grn


# ══════════════════════════════════════════════════════════════════════════════
#  Fate Assignment (general, for any number of toggles)
# ══════════════════════════════════════════════════════════════════════════════

def make_fate_fn(grn: GRNBlueprint, preset: DifficultyPreset):
    """Create a fate assignment function for any toggle topology."""
    toggle_pairs = grn.toggle_pairs
    n_toggles = len(toggle_pairs)
    t_early = preset.t_end * 0.15   # progenitor cutoff
    t_mid = preset.t_end * 0.4      # full fate assignment cutoff

    def fate_fn(expr: np.ndarray, t: float = None) -> np.ndarray:
        n = expr.shape[0]
        if t is not None and t < t_early:
            return np.array(["A"] * n)

        # Compute binary state for each toggle: True = first gene high
        bits = []
        for g_a, g_b in toggle_pairs:
            bits.append(expr[:, g_a] > expr[:, g_b])

        if t is not None and t < t_mid and n_toggles > 1:
            # Intermediate: only first toggle resolved
            return np.where(bits[0], "B", "C")

        # Full combinatorial fate assignment
        fates = np.empty(n, dtype='U1')
        for fate_i in range(2**n_toggles):
            # Each fate = unique combination of toggle states
            mask = np.ones(n, dtype=bool)
            for bit_idx in range(n_toggles):
                bit_val = bool((fate_i >> bit_idx) & 1)
                mask &= (bits[bit_idx] == bit_val)
            fates[mask] = chr(ord('B') + fate_i)

        return fates

    return fate_fn


# ══════════════════════════════════════════════════════════════════════════════
#  Build ScenarioConfig from GRN Blueprint
# ══════════════════════════════════════════════════════════════════════════════

def build_config(preset: DifficultyPreset, grn: GRNBlueprint, seed: int) -> ScenarioConfig:
    """Convert GRN blueprint + preset into simulation-ready config."""
    n_genes = grn.n_genes

    # Per-gene production & degradation rates
    production = np.ones(n_genes) * 1.0
    degradation = np.ones(n_genes) * 0.3

    for idx in grn.tf_indices:
        production[idx] = 1.2
        degradation[idx] = 0.3

    for idx in grn.reporter_indices:
        production[idx] = 0.8
        degradation[idx] = 0.25

    for idx in grn.confounder_indices:
        # KEY: slow dynamics → late activation → HIGH time-correlation
        production[idx] = 0.6
        degradation[idx] = 0.12

    for idx in grn.noise_indices:
        production[idx] = 0.5
        degradation[idx] = 0.3

    # Growth config
    growth_idx = grn.growth_driver_indices
    growth = GrowthConfig(
        enabled=True,
        growth_gene_idx=growth_idx if len(growth_idx) > 1 else growth_idx[0],
        base_rate=-0.02,
        amplitude=0.06,
        threshold=0.3,
        beta=5.0,
    )

    # Snapshot times
    snapshot_times = np.linspace(0, preset.t_end, preset.n_time_bins).tolist()

    # Initial condition: near center for toggle genes, lower for reporters
    initial_mean = np.ones(n_genes) * 0.5
    for idx in grn.reporter_indices:
        initial_mean[idx] = 0.3
    for idx in grn.confounder_indices:
        initial_mean[idx] = 0.2
    for idx in grn.noise_indices:
        initial_mean[idx] = 0.5

    return ScenarioConfig(
        name=f"S_{preset.name}_seed{seed}",
        n_genes=n_genes,
        n_cells=preset.n_init_cells,
        grn_edges=grn.edges,
        production_rates=production,
        degradation_rates=degradation,
        t_start=0.0,
        t_end=preset.t_end,
        dt=0.1,
        sigma=preset.noise_sigma,
        snapshot_times=snapshot_times,
        initial_mean=initial_mean,
        initial_std=0.12,
        growth=growth,
        seed=seed,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Anti-Heuristic Validation
# ══════════════════════════════════════════════════════════════════════════════

def validate_anti_heuristic(
    expr_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    grn: GRNBlueprint,
) -> Dict:
    """Check confounders have higher time-correlation than real growth drivers."""
    from scipy.stats import pearsonr

    time = meta_df['time'].values
    gene_names = grn.gene_names

    corrs = {}
    for g in gene_names:
        if g in expr_df.columns:
            r, _ = pearsonr(expr_df[g].values, time)
            corrs[g] = r

    drivers = [grn.gene_names[i] for i in grn.growth_driver_indices]
    confounders = [grn.gene_names[i] for i in grn.confounder_indices]

    driver_corrs = {g: corrs.get(g, 0) for g in drivers}
    confounder_corrs = {g: corrs.get(g, 0) for g in confounders}

    max_driver = max(driver_corrs.values()) if driver_corrs else 0
    max_confounder = max(confounder_corrs.values()) if confounder_corrs else 0

    passed = max_confounder > max_driver if confounder_corrs else True

    return {
        'anti_heuristic_passed': passed,
        'max_driver_time_corr': round(max_driver, 4),
        'max_confounder_time_corr': round(max_confounder, 4),
        'driver_corrs': {k: round(v, 4) for k, v in driver_corrs.items()},
        'confounder_corrs': {k: round(v, 4) for k, v in confounder_corrs.items()},
        'all_time_corrs': {k: round(v, 4) for k, v in sorted(corrs.items(), key=lambda x: -x[1])},
        'growth_drivers': drivers,
        'confounders': confounders,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  TASK.md Generator (no-leakage template)
# ══════════════════════════════════════════════════════════════════════════════

def generate_task_md(grn: GRNBlueprint, preset: DifficultyPreset) -> str:
    """Generate TASK.md matching the S3 template exactly — no method hints."""
    gene_names = grn.gene_names
    n_genes = grn.n_genes
    fate_labels = [f for f in grn.fate_labels if f != 'A']
    all_fates_str = ', '.join(grn.fate_labels)
    gene_range = f"Gene_1 through Gene_{n_genes}"
    gene_list_full = ', '.join(gene_names)

    all_bins = list(range(preset.n_time_bins))
    train_bins = [b for b in all_bins if b not in preset.holdout_bins]
    train_bins_str = ', '.join(str(b) for b in train_bins)

    dt = preset.t_end / (preset.n_time_bins - 1)

    # Observed bins string: "0, 1, 4, 5, 6, 7 are observed"
    observed_str = ', '.join(str(b) for b in train_bins) + ' are observed'

    # Time mapping: "bin 0->t=0.0, bin 1->t=1.5, ..."
    time_parts = []
    for b in train_bins:
        t = b * dt
        time_parts.append(f"bin {b}\u2192t={t:.1f}")
    time_mapping = ', '.join(time_parts)

    # Holdout description: "Time bins 2 (t=3.0) and 3 (t=4.5) are held out"
    holdout_parts = []
    for b in preset.holdout_bins:
        t = b * dt
        holdout_parts.append(f"{b} (t={t:.1f})")
    holdout_desc = ' and '.join(holdout_parts)

    # Holdout times for prediction: "t=3.0 and t=4.5"
    holdout_times = [f"t={b * dt:.1f}" for b in preset.holdout_bins]
    holdout_times_str = ' and '.join(holdout_times)

    # Time values for holdout column description
    holdout_time_vals = ' or '.join(f"{b * dt:.1f}" for b in preset.holdout_bins)

    # Random example values (NOT aligned with GT)
    rng = np.random.RandomState(12345)
    fate_ex_parts = []
    for f in fate_labels:
        fate_ex_parts.append(f'"{f}": {rng.uniform(0.05, 0.4):.2f}')
    fate_ex1 = ', '.join(fate_ex_parts)
    fate_ex_parts2 = []
    for f in fate_labels:
        fate_ex_parts2.append(f'"{f}": {rng.uniform(0.05, 0.4):.2f}')
    fate_ex2 = ', '.join(fate_ex_parts2)
    pert_ex_parts1 = []
    for f in fate_labels:
        pert_ex_parts1.append(f'"{f}": {rng.uniform(-0.15, 0.15):.2f}')
    pert_ex1 = ', '.join(pert_ex_parts1)
    pert_ex_parts2 = []
    for f in fate_labels:
        pert_ex_parts2.append(f'"{f}": {rng.uniform(-0.15, 0.15):.2f}')
    pert_ex2 = ', '.join(pert_ex_parts2)

    fates_str = ', '.join(fate_labels)
    n_cell_types = len(grn.fate_labels)

    return f"""# DynBench v2 - Multi-timepoint Prediction Task

## Objective
You are given multi-timepoint scRNA-seq data from a system with {n_genes} measured genes and {n_cell_types} cell types.
Your task is to build or use an analysis approach from `train.h5ad`, run downstream analyses in the measured gene state space, and write exactly six deliverable files to the requested output directory:
`holdout_prediction.csv`, `velocity_field.csv`, `growth_rates.csv`, `per_cell_fate.json`, `perturbation_results.json`, and `driver_genes.json`.
The deliverables should describe held-out intermediate cell states, per-cell dynamics, growth/mass behavior, root-cell fate behavior, perturbation responses, and candidate regulatory/growth drivers for this run.

## Data

### Input Files
- **`train.h5ad`**: Single-cell gene expression data at {len(train_bins)} observed time points
  - `adata.X`: Gene expression matrix (normalized, log1p)
  - `adata.obs['time']`: Continuous time value for each cell
  - `adata.obs['time_bin']`: Integer time bin index ({observed_str})
  - `adata.obs['fate_true']`: Canonical cell type labels ({all_fates_str})
  - `adata.obs['cell_type']`: Alias of `fate_true` kept for backward compatibility
  - `adata.var_names`: {gene_range}
  - Time mapping: {time_mapping}
  - **Note**: Time bins {holdout_desc} are held out

- **`fate_classifier.pkl`**: Pre-trained fate classifier (sklearn MLPClassifier)
  - Input features: `np.hstack([expression_matrix, time.reshape(-1, 1)])` \u2014 shape (n_cells, {n_genes + 1})
  - Feature order: [{gene_list_full}, time]
  - Classes: {all_fates_str}
  - **You MUST use this classifier** for all fate assignment tasks
  - This classifier assigns the fate/state label of the current input state; it is not a future-fate predictor by itself.
  - Example usage:
    ```python
    import pickle, numpy as np
    clf = pickle.load(open('fate_classifier.pkl', 'rb'))
    features = np.hstack([expr_matrix, time_col.reshape(-1, 1)])
    labels = clf.predict(features)       # discrete labels
    probs = clf.predict_proba(features)  # probability per class
    ```

## Representation Contract

- This benchmark is defined directly in the measured gene state space.
- `gene_space_equals_latent = true`
- If your runtime requires `X_latent`, set `X_latent = X`.
- Do **not** run an extra PCA or reduce the state dimensionality.
- Holdout prediction, fate analysis, and perturbation should all be executed in this measured gene space.

## Perturbation Contract

- Root cells are the progenitor cells at `time_bin = 0`.
- Benchmark perturbation is a gene-space z-score intervention in the current measured-gene state space.
- For each target gene:
  - control branch: `z_score = 0.0`
  - perturbed branch: `z_score = -5.0`
- In this low-dimensional benchmark, apply the intervention directly on the current state coordinates; do **not** project through PCA.
- Report one `control`, one `perturbed`, and one `delta` per gene.

## Downstream Analysis Outputs

Run the following analyses and save results to the output directory.

## Completion Contract

- This is a file-delivery benchmark, not a narrative benchmark.
- A run counts as complete only when all required output files are materialized in the requested output directory.
- Explanations, plans, or "step completed" messages do not count as completion by themselves.
- If you discover older benchmark outputs elsewhere in the workspace, treat them as historical/reference artifacts only; they do not satisfy this request.
- Do not claim completion by pointing to a previous batch directory. Completion requires files produced for the current requested output directory / current run.
- If your runtime uses a temporary staging directory such as `_workspace`, treat it as temporary only; the final contract is still the requested output directory.
- Before stopping, verify that each required output file exists, is non-empty, and matches the public format/schema described below.
- If `verify_dynbench_outputs.py` is present in your workspace, run it before final delivery. This public verifier checks only visible task files and output schema; it does not read hidden answers, judge scientific correctness, or repair files. Fix any reported output-file failure and rerun it before stopping.

## Minimal Execution Template

Use the shortest path that satisfies the benchmark contract:
1. Read `train.h5ad`, `prediction_targets.json`, and `fate_classifier.pkl`.
2. Perform only the minimum modeling needed to produce the required benchmark outputs.
3. Write all six required output files.
4. Verify existence, non-zero size, and basic schema/shape for each file; preferably use `verify_dynbench_outputs.py` when available.
5. If any file is missing or invalid, rerun the export path for that artifact. Do not fill placeholders, statistical shortcuts, or fallback approximations.
6. Stop once the contract is satisfied.

## Public Sanity Checks

- Sanity checks must only use the provided task-package inputs, the public output contract, and your own generated outputs.
- Do not read hidden answer files, non-public reference outputs, or implementation files outside the provided task workspace.
- Good sanity checks include:
  - file existence and non-zero size
  - expected row counts for training-cell outputs
  - required columns / JSON keys
  - per-cell fate probabilities are non-negative and approximately sum to 1
  - perturbation deltas are not all identically zero
  - fate outputs are not trivially collapsed into a single terminal fate for nearly every root cell

---

### Analysis 1: Predict held-out cell distribution at intermediate time points
**Output file: `holdout_prediction.csv`**

Predict what cells look like at the held-out time points ({holdout_times_str}) that were NOT in training data.
Read `prediction_targets.json` in the same task package for the canonical target-bin and target-time metadata.
Write this CSV with `index=False`; implicit pandas index columns are invalid.

| Column | Description |
|--------|-------------|
| {gene_names[0]} ... {gene_names[-1]} | Predicted expression |
| time | Time value ({holdout_time_vals}) for each predicted cell |
| weight | Optional rollout/sample weight. Include this column if your prediction is a weighted particle rollout. |

- Generate a realistic number of cells per holdout time point
- Expression values should be in the same scale as training data
- Preserve weighted rollouts: if your simulation produces particle weights, keep them in a `weight` column instead of silently equalizing all samples.

---

### Analysis 2: Velocity field \u2014 trajectory analysis
**Output file: `velocity_field.csv`**

Predict the velocity of gene expression for each training cell.
Rows must match training cells in `train.h5ad` (same order, same count).
Write this CSV with `index=False`; implicit pandas index columns are invalid.

| Column | Description |
|--------|-------------|
| velocity_{gene_names[0]} ... velocity_{gene_names[-1]} | velocity for each gene |

---

### Analysis 3: Growth and mass dynamics
**Output file: `growth_rates.csv`**

Predict the per-cell growth/death rate for each training cell.
Positive values indicate proliferation, negative values indicate cell death.
Rows must match training cells in `train.h5ad` (same order, same count).
Write this CSV with `index=False`; implicit pandas index columns are invalid.

| Column | Description |
|--------|-------------|
| growth_rate | Per-cell growth rate |

---

### Analysis 4: Trajectory and fate probability
**Output file: `per_cell_fate.json`**

For each progenitor cell (time_bin=0), predict its fate probability.

```json
{{{{
  "<cell_id_1>": {{{{{fate_ex1}}}}},
  "<cell_id_2>": {{{{{fate_ex2}}}}}
}}}}
```

- Include exactly the cells from time_bin=0
- Cell IDs must match the cell names in `train.h5ad`
- Report probabilities for fates {fates_str} (not A, since A is the progenitor state)

---

### Analysis 5: In silico perturbation analysis \u2014 gene knockdown
**Output file: `perturbation_results.json`**

Run perturbation analysis: for each gene ({gene_range}), apply the benchmark perturbation protocol in progenitor cells:
- matched control branch with `z_score = 0.0`
- perturbed branch with `z_score = -5.0`

Then report the fate shift (delta) compared to the matched control.

```json
[
  {{{{
    "gene_name": "Gene_X",
    "control": {{{{{fate_ex1}}}}},
    "perturbed": {{{{{fate_ex2}}}}},
    "delta": {{{{{pert_ex1}}}}}
  }}}},
  {{{{
    "gene_name": "Gene_Y",
    "control": {{{{{fate_ex1}}}}},
    "perturbed": {{{{{fate_ex2}}}}},
    "delta": {{{{{pert_ex2}}}}}
  }}}}
]
```

---

### Analysis 6: Gene regulatory network and driver identification
**Output file: `driver_genes.json`**

Report a model-derived gene regulatory network (GRN) and growth driver genes for this run.

**6a) GRN edges**: For each pair of genes, predict whether a regulatory edge exists and its direction.
- Use positive scores for activation and negative scores for inhibition
- Higher absolute scores = higher confidence
- Direction is regulator to regulated gene: `source` is the regulator and `target` is the regulated gene.

**6b) Growth drivers**: Report per-gene importance scores for growth/proliferation (higher = more influential).

```json
{{{{
  "grn_edges": [
    {{{{"source": "Gene_X", "target": "Gene_Y", "score": 0.63}}}},
    {{{{"source": "Gene_Z", "target": "Gene_Y", "score": -0.41}}}},
    {{{{"source": "Gene_X", "target": "Gene_W", "score": 0.29}}}}
  ],
  "growth_drivers": {{{{"Gene_X": 0.37, "Gene_Y": 0.12, "Gene_Z": 0.58, "Gene_W": 0.44}}}}
}}}}
```

- `grn_edges`: List all predicted regulatory edges with confidence scores
  - `source`: the regulator gene
  - `target`: the regulated gene
  - `score`: positive = activation, negative = inhibition, magnitude = confidence
- `growth_drivers`: Per-gene importance for growth dynamics (all genes, higher = more important)

---

## Simulation Constraints
When running forward simulations (optional):
- **Max simulation runs**: 5 per condition
- **Max initial cells**: 1000 per run (subsample if needed)
- **Max time steps**: 100
"""


def build_prediction_target_spec(
    snapshot_times: List[float],
    train_bins: List[int],
    prediction_target_bins: List[int],
    prediction_eval_source: str = "log1p_counts_ground_truth",
    split_mode: str = "canonical_holdout",
    prediction_gt_weight_policy: str = "uniform_empirical",
    prediction_agent_weight_policy: str = "optional_output_weight",
) -> Dict:
    """Create benchmark prediction metadata used by both agent and evaluator."""
    observed_bins = sorted(int(b) for b in train_bins)
    target_bins = sorted(int(b) for b in prediction_target_bins)
    observed_times = [float(snapshot_times[b]) for b in observed_bins]
    target_times = [float(snapshot_times[b]) for b in target_bins]
    return {
        "split_mode": split_mode,
        "train_bins": observed_bins,
        "prediction_target_bins": target_bins,
        "prediction_time_points": target_times,
        "prediction_target_times": target_times,
        "prediction_eval_source": prediction_eval_source,
        "prediction_gt_weight_policy": prediction_gt_weight_policy,
        "prediction_agent_weight_policy": prediction_agent_weight_policy,
        "observed_bins": observed_bins,
        "observed_time_points": observed_times,
        "note": (
            "Use prediction_target_bins/prediction_time_points as the benchmark target set. "
            "If your runtime uses an internal processed time axis, derive that mapping from the "
            "observed training metadata and keep the raw target times for export only. "
            "GT holdout cells are treated as a uniform empirical distribution unless the "
            "task metadata explicitly requests non-uniform GT weights."
        ),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Main Pipeline: generate one scenario end-to-end
# ══════════════════════════════════════════════════════════════════════════════

def generate_scenario(
    difficulty: str = 'medium',
    seed: int = 42,
    output_base: Path = None,
    build_standalone: bool = False,
    verbose: bool = True,
    preset_override: DifficultyPreset = None,
    scenario_name_override: str = None,
    topology: str = TOPOLOGY_PROCEDURAL,
    topology_spec: dict | None = None,
) -> Dict:
    """
    Generate one complete DynBench scenario.

    Returns validation report dict.
    """
    if output_base is None:
        output_base = Path(__file__).resolve().parent.parent

    preset = preset_override if preset_override else PRESETS[difficulty]
    rng = np.random.RandomState(seed)
    scenario_name = scenario_name_override or f"S_{difficulty}_seed{seed}"

    if verbose:
        print("=" * 60)
        print(f"DynBench Factory: {scenario_name}")
        print(f"  Difficulty: {difficulty}")
        print(f"  Genes: {preset.n_genes_total} ({preset.n_tfs} TFs + {preset.n_reporters} reporters + {preset.n_confounders} confounders + {preset.n_noise_genes} noise)")
        print(f"  Fates: {preset.n_fates} (from {preset.n_toggles} toggle switches)")
        print(f"  Growth type: {preset.growth_type}")
        print(f"  Seed: {seed}")
        print("=" * 60)

    # ── 1. Generate GRN ──
    if topology == TOPOLOGY_TOGGLE_SWITCH_3GENE:
        grn = generate_toggle_switch_3gene_grn(rng)
    elif topology == TOPOLOGY_LLM_SPEC:
        grn = generate_grn_from_topology_spec(preset, rng, topology_spec)
    elif topology == TOPOLOGY_PROCEDURAL:
        grn = generate_grn(preset, rng)
    else:
        raise ValueError(
            f"Unknown topology '{topology}'. Expected one of: "
            f"{TOPOLOGY_PROCEDURAL}, {TOPOLOGY_TOGGLE_SWITCH_3GENE}, {TOPOLOGY_LLM_SPEC}."
        )
    if verbose:
        print(f"\n1. GRN: {len(grn.edges)} edges")
        print(f"   Drivers: {[grn.gene_names[i] for i in grn.growth_driver_indices]}")
        print(f"   Confounders: {[grn.gene_names[i] for i in grn.confounder_indices]}")

    # ── 2. Build simulation config ──
    config = build_config(preset, grn, seed)
    fate_fn = make_fate_fn(grn, preset)

    def fate_fn_final(x):
        return fate_fn(x, config.t_end)

    # ── 3. Run simulation ──
    if verbose:
        print(f"\n2. Running BoolODE simulation ({config.n_cells} cells, t=[0, {config.t_end}])...")
    result = simulate(config, verbose=verbose, fate_fn=fate_fn_final)

    # ── 4. Process snapshots into DataFrames ──
    if verbose:
        print(f"\n3. Processing snapshots...")
    gene_names = grn.gene_names

    all_cells = []
    all_meta = []
    cell_id = 0

    for t_idx, t in enumerate(sorted(result.snapshots.keys())):
        expr = result.snapshots[t]
        vel = result.velocity_gt[t]
        growth = result.growth_gt[t]
        weights = result.cell_weights[t]
        fates_at_t = fate_fn(expr, t)

        if verbose:
            fate_counts = {f: int((fates_at_t == f).sum()) for f in np.unique(fates_at_t)}
            print(f"   t={t:.1f} (bin {t_idx}): {expr.shape[0]} cells, fates={fate_counts}")

        for i in range(expr.shape[0]):
            all_cells.append({
                'cell_id': f"cell_{cell_id}",
                'time_bin': t_idx, 'time': t,
                **{gene_names[g]: expr[i, g] for g in range(grn.n_genes)},
            })
            all_meta.append({
                'cell_id': f"cell_{cell_id}",
                'time_bin': t_idx, 'time': t,
                'fate_true': fates_at_t[i],
                'growth_rate': growth[i],
                'weight': float(weights[i]),
                **{f"velocity_{gene_names[g]}": vel[i, g] for g in range(grn.n_genes)},
            })
            cell_id += 1

    df_expr = pd.DataFrame(all_cells).set_index('cell_id')
    df_meta = pd.DataFrame(all_meta).set_index('cell_id')

    # ── 5. Validate anti-heuristic ──
    validation = validate_anti_heuristic(df_expr, df_meta, grn)
    status = "✅ PASSED" if validation['anti_heuristic_passed'] else "❌ FAILED"
    if verbose:
        print(f"\n4. Anti-heuristic validation: {status}")
        print(f"   Driver max corr:     {validation['max_driver_time_corr']:.4f}")
        print(f"   Confounder max corr: {validation['max_confounder_time_corr']:.4f}")

    # ── 6. Save ground truth ──
    gt_dir = output_base / "ground_truth" / scenario_name
    gt_dir.mkdir(parents=True, exist_ok=True)

    df_expr[gene_names].to_csv(gt_dir / "counts_cells_x_genes.csv")
    df_meta.to_csv(gt_dir / "full_metadata.csv")

    vel_cols = [f"velocity_{g}" for g in gene_names]
    df_meta[vel_cols].to_csv(gt_dir / "velocity_ground_truth.csv")

    # GRN ground truth
    grn_gt = {
        'edges': [{'from': gene_names[e.source], 'to': gene_names[e.target],
                    'direction': 'activate' if e.effect > 0 else 'inhibit'}
                  for e in grn.edges],
        'growth_drivers': [gene_names[i] for i in grn.growth_driver_indices],
        'n_genes': grn.n_genes,
        'n_true_edges': len(grn.edges),
    }
    with open(gt_dir / "grn_ground_truth.json", 'w') as f:
        json.dump(grn_gt, f, indent=2)

    train_bins = [b for b in range(preset.n_time_bins) if b not in preset.holdout_bins]
    prediction_spec = build_prediction_target_spec(
        snapshot_times=config.snapshot_times,
        train_bins=train_bins,
        prediction_target_bins=preset.holdout_bins,
        prediction_eval_source="log1p_counts_ground_truth",
        split_mode="canonical_holdout",
    )

    # Simulation config
    sim_config = {
        'scenario': scenario_name, 'difficulty': difficulty, 'seed': seed,
        'topology': topology,
        'growth_drivers': [gene_names[i] for i in grn.growth_driver_indices],
        'gene_roles': {gene_names[k]: v for k, v in grn.gene_roles.items()},
        'toggle_pairs': [[gene_names[a], gene_names[b]] for a, b in grn.toggle_pairs],
        'n_fates': preset.n_fates, 'n_genes': grn.n_genes,
        'holdout_bins': preset.holdout_bins,
        'train_bins': prediction_spec['train_bins'],
        'prediction_target_bins': prediction_spec['prediction_target_bins'],
        'prediction_target_times': prediction_spec['prediction_target_times'],
        'prediction_eval_source': prediction_spec['prediction_eval_source'],
        'prediction_gt_weight_policy': prediction_spec['prediction_gt_weight_policy'],
        'prediction_agent_weight_policy': prediction_spec['prediction_agent_weight_policy'],
        'evaluation_contract': {
            'split_mode': prediction_spec['split_mode'],
            'velocity_scope': 'train_h5ad_cells',
            'growth_scope': 'train_h5ad_cells',
            'prediction_scope': 'prediction_target_bins',
            'prediction_gt_weight_policy': prediction_spec['prediction_gt_weight_policy'],
            'prediction_agent_weight_policy': prediction_spec['prediction_agent_weight_policy'],
        },
        'gene_space_equals_latent': True,
        'perturbation_protocol': {
            'intervention_space': 'gene_space',
            'control_z_score': 0.0,
            'perturbed_z_score': -5.0,
        },
    }
    with open(gt_dir / "simulation_config.json", 'w') as f:
        json.dump(sim_config, f, indent=2)

    # Perturbation GT via SDE re-simulation
    gt_dir.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"\n5. Running perturbation GT (SDE)...")
    t0_cells = df_expr[df_meta['time_bin'] == 0][gene_names].values.astype(np.float64)
    pert_results = []
    for g_idx in range(grn.n_genes):
        pert_res = simulate_perturbation(
            config,
            gene_idx=g_idx,
            n_runs=30,
            fate_fn=fate_fn_final,
            initial_cells=t0_cells,
            reference_cells=t0_cells,
            control_z_score=0.0,
            perturbed_z_score=-5.0,
        )
        pert_res['gene_name'] = gene_names[g_idx]
        pert_results.append(pert_res)
        if verbose:
            print(f"   Perturbation {gene_names[g_idx]} done")
    with open(gt_dir / "perturbation_ground_truth.json", 'w') as f:
        json.dump(pert_results, f, indent=2, default=str)

    # Per-cell fate GT
    gt_dir.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"   Running per-cell fate probabilities...")
    t0_cells = df_expr[df_meta['time_bin'] == 0][gene_names].values.astype(np.float64)
    t0_ids = df_expr[df_meta['time_bin'] == 0].index.tolist()
    fate_prob_matrix, fate_labels_out = compute_percell_fate_probability(
        config, initial_cells=t0_cells, n_runs=50, fate_fn=fate_fn_final,
    )
    # Convert to {cell_id: {fate: prob}} dict
    fate_probs_dict = {}
    for i, cid in enumerate(t0_ids):
        fate_probs_dict[cid] = {fl: float(fate_prob_matrix[i, j])
                                for j, fl in enumerate(fate_labels_out)}
    with open(gt_dir / "percell_fate_ground_truth.json", 'w') as f:
        json.dump(fate_probs_dict, f, indent=2)

    # ── 7. Build task package ──
    if verbose:
        print(f"\n6. Building task package...")
    task_dir = output_base / "task_packages" / scenario_name
    task_dir.mkdir(parents=True, exist_ok=True)

    # Train AnnData (exclude holdout bins)
    train_mask = df_meta['time_bin'].isin(train_bins)

    import anndata
    train_expr = df_expr.loc[train_mask, gene_names].values.astype(np.float32)
    train_obs = df_meta.loc[train_mask, ['time', 'time_bin', 'fate_true']].copy()
    train_obs['cell_type'] = train_obs['fate_true'].astype(str)
    train_adata = anndata.AnnData(
        X=np.log1p(train_expr),
        obs=train_obs,
        var=pd.DataFrame(index=gene_names),
    )
    train_adata.obs_names = df_meta.index[train_mask].tolist()
    train_adata.write_h5ad(str(task_dir / "train.h5ad"))

    # Oracle fate classifier
    all_log = np.log1p(df_expr[gene_names].values.astype(np.float32))
    X_clf = np.column_stack([all_log, df_meta['time'].values])
    y_clf = df_meta['fate_true'].values
    clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=seed)
    clf.fit(X_clf, y_clf)
    acc = clf.score(X_clf, y_clf)
    if verbose:
        print(f"   Classifier accuracy: {acc:.4f}")
    with open(task_dir / "fate_classifier.pkl", 'wb') as f:
        pickle.dump(clf, f)

    # TASK.md
    (task_dir / "TASK.md").write_text(generate_task_md(grn, preset))
    with open(task_dir / "prediction_targets.json", 'w') as f:
        json.dump(prediction_spec, f, indent=2)

    # ── 8. Validation report ──
    report = {
        'scenario': scenario_name,
        'difficulty': difficulty,
        'seed': seed,
        'n_genes': grn.n_genes,
        'n_fates': preset.n_fates,
        'n_cells': len(df_meta),
        'n_edges': len(grn.edges),
        'classifier_accuracy': round(acc, 4),
        'validation': validation,
        'paths': {
            'ground_truth': str(gt_dir),
            'task_package': str(task_dir),
        },
    }
    with open(gt_dir / "validation_report.json", 'w') as f:
        json.dump(report, f, indent=2, default=str)

    # ── 9. Ground truth check: run "perfect agent" through eval ──
    if verbose:
        print(f"\n7. Ground truth check (perfect agent → eval)...")
    gt_check = validate_ground_truth(gt_dir, task_dir, preset, verbose=verbose)
    report['gt_check'] = gt_check
    with open(gt_dir / "validation_report.json", 'w') as f:
        json.dump(report, f, indent=2, default=str)
    if not gt_check.get('passed', False):
        raise RuntimeError(f"DynBench ground-truth validation failed: {gt_check}")

    if verbose:
        print(f"\n✅ Done! Scenario: {scenario_name}")
        print(f"   GT:   {gt_dir}")
        print(f"   Task: {task_dir}")

    return report


# ══════════════════════════════════════════════════════════════════════════════
#  Ground Truth Validation: run "perfect agent" through eval pipeline
# ══════════════════════════════════════════════════════════════════════════════

def validate_ground_truth(
    gt_dir: Path,
    task_dir: Path,
    preset: DifficultyPreset,
    verbose: bool = True,
) -> Dict:
    """
    Construct 'perfect agent output' from GT files and run the eval pipeline.
    All 6 metrics should score ≈ 1.0.
    
    This catches:
    - Format mismatches between GT and eval
    - Missing/misaligned files
    - Bugs in the simulation or eval pipeline
    """
    import tempfile, shutil
    
    gt_dir = Path(gt_dir)
    task_dir = Path(task_dir)
    
    # Create temporary "perfect agent output" directory
    tmpdir = gt_dir / "_perfect_agent_check"
    tmpdir.mkdir(exist_ok=True)
    
    try:
        meta = pd.read_csv(gt_dir / "full_metadata.csv", index_col=0)
        expr = pd.read_csv(gt_dir / "counts_cells_x_genes.csv", index_col=0)
        gene_names = list(expr.columns)

        sim_cfg = {}
        sim_config_path = gt_dir / "simulation_config.json"
        if sim_config_path.exists():
            with open(sim_config_path) as f:
                sim_cfg = json.load(f)

        holdout_bins = sim_cfg.get("holdout_bins", preset.holdout_bins)
        train_bins = sim_cfg.get("train_bins", [b for b in sorted(meta['time_bin'].unique()) if b not in holdout_bins])
        prediction_target_bins = sim_cfg.get("prediction_target_bins", holdout_bins)
        prediction_eval_source = sim_cfg.get("prediction_eval_source", "log1p_counts_ground_truth")

        train_adata = None
        train_path = task_dir / "train.h5ad"
        if train_path.exists():
            import anndata
            train_adata = anndata.read_h5ad(str(train_path))

        if train_adata is not None:
            train_ids = [cid for cid in train_adata.obs_names.astype(str).tolist() if cid in meta.index]
            train_mask = meta.index.isin(train_ids)
        else:
            train_mask = meta['time_bin'].isin(train_bins)
        holdout_mask = meta['time_bin'].isin(prediction_target_bins)

        # 1. velocity_field.csv — GT velocity for training cells
        vel_gt = pd.read_csv(gt_dir / "velocity_ground_truth.csv", index_col=0)
        if train_adata is not None:
            vel_gt.loc[train_ids].to_csv(tmpdir / "velocity_field.csv", index=False)
        else:
            vel_gt[train_mask].to_csv(tmpdir / "velocity_field.csv", index=False)

        # 2. growth_rates.csv — GT growth for training cells
        if train_adata is not None:
            growth_df = pd.DataFrame({'growth_rate': meta.loc[train_ids, 'growth_rate'].values})
        else:
            growth_df = pd.DataFrame({'growth_rate': meta.loc[train_mask, 'growth_rate'].values})
        growth_df.to_csv(tmpdir / "growth_rates.csv", index=False)

        # 3. holdout_prediction.csv — GT holdout expression
        if prediction_eval_source == "train_h5ad_X" and train_adata is not None:
            pred_mask = pd.to_numeric(train_adata.obs["time_bin"], errors="coerce").isin(prediction_target_bins).values
            target_X = train_adata.X[pred_mask]
            if hasattr(target_X, "toarray"):
                target_X = target_X.toarray()
            holdout_expr = pd.DataFrame(np.asarray(target_X), columns=gene_names)
            holdout_expr['time'] = pd.to_numeric(train_adata.obs.loc[pred_mask, 'time'], errors='coerce').values
        else:
            holdout_expr = pd.DataFrame(
                np.log1p(expr.loc[holdout_mask, gene_names].values),
                index=expr.index[holdout_mask],
                columns=gene_names,
            )
            holdout_expr['time'] = meta.loc[holdout_mask, 'time'].values
        holdout_expr['weight'] = 1.0
        holdout_expr.to_csv(tmpdir / "holdout_prediction.csv", index=False)

        # 4. per_cell_fate.json — GT fate probabilities
        if (gt_dir / "percell_fate_ground_truth.json").exists():
            shutil.copy(gt_dir / "percell_fate_ground_truth.json", tmpdir / "per_cell_fate.json")
        else:
            # Fallback: one-hot from fate_true for t=0 cells
            t0_mask = meta['time_bin'] == 0
            fate_dict = {}
            fates = sorted([f for f in meta['fate_true'].unique() if f != 'A'])
            for cid in meta.index[t0_mask]:
                fate_dict[cid] = {f: 0.0 for f in fates}  # placeholder
            with open(tmpdir / "per_cell_fate.json", 'w') as f:
                json.dump(fate_dict, f)

        # 5. perturbation_results.json — GT perturbation
        if (gt_dir / "perturbation_ground_truth.json").exists():
            shutil.copy(gt_dir / "perturbation_ground_truth.json", tmpdir / "perturbation_results.json")

        # 6. driver_genes.json — GT GRN + growth drivers
        with open(gt_dir / "grn_ground_truth.json") as f:
            grn_gt = json.load(f)
        
        # Convert GT edges to scored format (max confidence)
        gt_edges = []
        for e in grn_gt['edges']:
            score = 1.0 if e.get('direction', 'activate') == 'activate' else -1.0
            gt_edges.append({'source': e['from'], 'target': e['to'], 'score': score})
        
        gt_drivers = {g: 1.0 for g in grn_gt.get('growth_drivers', [])}
        for g in gene_names:
            if g not in gt_drivers:
                gt_drivers[g] = 0.0
        
        with open(tmpdir / "driver_genes.json", 'w') as f:
            json.dump({'grn_edges': gt_edges, 'growth_drivers': gt_drivers}, f, indent=2)

        # Run evaluation — need project root for absolute imports in eval module
        project_root = str(Path(__file__).resolve().parent.parent.parent.parent)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        eval_dir = str(Path(__file__).resolve().parent.parent / "eval")
        if eval_dir not in sys.path:
            sys.path.insert(0, eval_dir)
        from dynbench_eval_v2 import evaluate_all

        results = evaluate_all(str(tmpdir), str(gt_dir), verbose=False)
        
        total = results['total_score']
        per_metric = {k: results['per_metric'][k]['score'] for k in results['per_metric']}
        
        # Check thresholds
        issues = []
        for k, s in per_metric.items():
            if s < 0.95:
                issues.append(f"{k}={s:.3f} (expected ≈1.0)")
        
        passed = total > 0.95
        
        if verbose:
            status = "✅ PASSED" if passed else "❌ FAILED"
            print(f"   GT Check: {status} (total={total:.3f})")
            for k, s in per_metric.items():
                marker = "✅" if s >= 0.95 else "⚠️" if s > 0.5 else "❌"
                print(f"     {marker} {k}: {s:.3f}")
            if issues:
                print(f"   ⚠️  Issues: {'; '.join(issues)}")
        
        return {
            'passed': passed,
            'total_score': round(total, 4),
            'per_metric': {k: round(v, 4) for k, v in per_metric.items()},
            'issues': issues,
        }
    
    except Exception as e:
        logger.warning(f"GT check failed: {e}")
        if verbose:
            print(f"   ❌ GT check error: {e}")
        return {'passed': False, 'error': str(e)}
    
    finally:
        # Clean up temp dir
        if tmpdir.exists():
            shutil.rmtree(tmpdir, ignore_errors=True)


def generate_v2_scenario(
    config_path: str | Path,
    output_base: Path = None,
    build_standalone: bool = False,
    verbose: bool = True,
) -> Dict:
    """
    Generate a v2 scenario from a config file.
    - Derives internal seed deterministically from config params (no external seed)
    - Scenario name comes from config 'name' field (no seed suffix)
    - Computes MD5 manifest of generated ground truth files
    """
    config_path = Path(config_path)
    with open(config_path) as f:
        cfg = json.load(f)

    scenario_name = cfg.get('name')
    if not scenario_name:
        raise ValueError(f"Config {config_path} missing 'name' field")

    # Derive deterministic internal seed
    internal_seed = deterministic_seed_from_config(cfg)

    # Build preset from config
    preset = DifficultyPreset(
        name=scenario_name,
        n_fates=cfg['n_fates'],
        n_reporters_per_tf=cfg.get('n_reporters_per_tf', 1),
        n_confounders=cfg.get('n_confounders', 2),
        n_noise_genes=cfg.get('n_noise_genes', 0),
        noise_sigma=cfg.get('noise_sigma', 0.12),
        n_time_bins=cfg.get('n_time_bins', 8),
        holdout_bins=cfg.get('holdout_bins', [2, 3]),
        n_init_cells=cfg.get('n_init_cells', 2000),
        growth_type=cfg.get('growth_type', 'multi'),
        t_end=cfg.get('t_end', 12.0),
    )

    topology = cfg.get('topology', TOPOLOGY_PROCEDURAL)
    topology_spec = cfg.get('topology_spec')

    report = generate_scenario(
        difficulty=scenario_name,
        seed=internal_seed,
        output_base=output_base,
        verbose=verbose,
        preset_override=preset,
        scenario_name_override=scenario_name,
        topology=topology,
        topology_spec=topology_spec,
    )

    # Compute and store MD5 manifest
    if output_base:
        gt_dir = output_base / "ground_truth" / scenario_name
    else:
        gt_dir = Path(__file__).resolve().parent.parent / "ground_truth" / scenario_name

    from .scenario_registry import store_md5_manifest
    md5_path = store_md5_manifest(gt_dir)
    report['md5_manifest'] = str(md5_path)
    if verbose:
        print(f"   MD5 manifest stored: {md5_path}")

    report['config_file'] = str(config_path)
    report['internal_seed'] = internal_seed
    return report


# ══════════════════════════════════════════════════════════════════════════════
#  Batch Generation
# ══════════════════════════════════════════════════════════════════════════════

def batch_generate(
    difficulties: List[str],
    seeds: List[int],
    output_base: Path = None,
    verbose: bool = True,
) -> List[Dict]:
    """Generate multiple scenarios. Results are saved to disk."""
    results = []
    total = len(difficulties) * len(seeds)
    i = 0

    for diff in difficulties:
        for seed in seeds:
            i += 1
            print(f"\n{'━' * 60}")
            print(f"  Batch [{i}/{total}]: {diff}/seed{seed}")
            print(f"{'━' * 60}")
            try:
                report = generate_scenario(diff, seed, output_base, verbose=verbose)
                results.append(report)
            except Exception as e:
                logger.error(f"Failed: {diff}/seed{seed}: {e}", exc_info=True)
                results.append({'difficulty': diff, 'seed': seed, 'error': str(e)})

    # Summary table
    print(f"\n{'=' * 70}")
    print(f"  Batch Summary: {len(results)} scenarios generated")
    print(f"{'=' * 70}")
    print(f"{'Scenario':<25} {'Genes':>5} {'Fates':>5} {'Cells':>6} {'AH':>6} {'Driver':>8} {'Conf':>8}")
    print("-" * 70)
    for r in results:
        if 'error' in r:
            print(f"  {r['difficulty']}/seed{r['seed']}: ❌ {r['error'][:50]}")
        else:
            v = r['validation']
            ah = "✅" if v['anti_heuristic_passed'] else "❌"
            print(f"  {r['scenario']:<23} {r['n_genes']:>5} {r['n_fates']:>5} {r['n_cells']:>6} {ah:>6} {v['max_driver_time_corr']:>8.4f} {v['max_confounder_time_corr']:>8.4f}")
    print("=" * 70)

    # Save batch summary
    if output_base:
        with open(output_base / "batch_summary.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)

    return results


def batch_generate_v2(
    config_dir: str | Path = None,
    output_base: Path = None,
    verbose: bool = True,
    scenario_names: List[str] | None = None,
) -> List[Dict]:
    """
    Generate all v2 scenarios from config files.
    Uses deterministic seeds derived from config params (no external seed).
    Computes MD5 manifests for each generated scenario.
    """
    if config_dir is None:
        config_dir = Path(__file__).resolve().parent / "test_configs" / "v2"
    else:
        config_dir = Path(config_dir)

    config_files = sorted(config_dir.glob("*.json"))
    if not config_files:
        raise FileNotFoundError(f"No JSON configs found in {config_dir}")

    # Filter by names if specified
    if scenario_names:
        config_files = [f for f in config_files
                        if f.stem in scenario_names]
        if not config_files:
            raise FileNotFoundError(
                f"No matching configs for: {scenario_names}")

    results = []
    total = len(config_files)
    for i, config_file in enumerate(config_files, 1):
        print(f"\n{'━' * 60}")
        print(f"  V2 Config [{i}/{total}]: {config_file.name}")
        print(f"{'━' * 60}")
        try:
            report = generate_v2_scenario(
                config_path=config_file,
                output_base=output_base,
                verbose=verbose,
            )
            results.append(report)
        except Exception as e:
            logger.error(f"Failed: {config_file.name}: {e}", exc_info=True)
            results.append({
                'scenario': config_file.stem,
                'config_file': str(config_file),
                'error': str(e),
            })

    # Summary
    print(f"\n{'=' * 70}")
    print(f"  V2 Batch: {len(results)} scenarios generated")
    print(f"{'=' * 70}")
    print(f"{'Scenario':<35} {'Genes':>5} {'Fates':>5} {'Cells':>6} {'AH':>6}")
    print("-" * 70)
    for r in results:
        if 'error' in r:
            print(f"  {r['scenario']}: ❌ {r['error'][:50]}")
        else:
            v = r.get('validation', {})
            ah = "✅" if v.get('anti_heuristic_passed') else "❌"
            print(f"  {r['scenario']:<33} {r.get('n_genes', '?'):>5} "
                  f"{r.get('n_fates', '?'):>5} {r.get('n_cells', '?'):>6} "
                  f"{ah:>6}")
    print("=" * 70)

    if output_base:
        with open(output_base / "v2_batch_summary.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  Config-Driven Mode: Agent designs JSON → factory executes
# ══════════════════════════════════════════════════════════════════════════════

def parse_scenario_config(cfg: dict) -> Tuple[DifficultyPreset, int, str, str, dict | None]:
    """
    Parse one scenario entry from a JSON config into (preset, seed, name).

    JSON schema per scenario:
    {
        "name": "bifurcation_3gene",       # unique scenario name
        "scenario_name": "TestScenario",   # optional exact output directory
        "topology": "procedural",          # procedural or toggle_switch_3gene
        "seed": 42,                         # random seed
        "n_fates": 4,                       # 2, 4, 8, 16
        "n_reporters_per_tf": 1,            # reporters per TF (1 or 2)
        "n_confounders": 2,                 # confounding late-marker genes
        "n_noise_genes": 0,                 # uncorrelated noise genes
        "growth_type": "multi",             # 'single', 'multi', 'interaction'
        "noise_sigma": 0.12,                # SDE noise
        "n_time_bins": 8,                   # total snapshot count
        "holdout_bins": [2, 3],             # held-out bins for evaluation
        "n_init_cells": 2000,               # starting cell count
        "t_end": 12.0                       # simulation end time
    }
    """
    topology = cfg.get('topology', TOPOLOGY_PROCEDURAL)
    topology_spec = cfg.get('topology_spec')
    if topology == TOPOLOGY_TOGGLE_SWITCH_3GENE:
        cfg = {
            'n_fates': 2,
            'n_reporters_per_tf': 1,
            'n_confounders': 0,
            'n_noise_genes': 0,
            'growth_type': 'single',
            'noise_sigma': cfg.get('noise_sigma', 0.12),
            'n_time_bins': cfg.get('n_time_bins', 8),
            'holdout_bins': cfg.get('holdout_bins', [2, 3]),
            'n_init_cells': cfg.get('n_init_cells', 2000),
            't_end': cfg.get('t_end', 12.0),
            **cfg,
        }
    elif topology == TOPOLOGY_LLM_SPEC and isinstance(topology_spec, dict):
        cfg = {
            'n_fates': topology_spec.get('n_fates', cfg.get('n_fates', 4)),
            'n_reporters_per_tf': topology_spec.get('n_reporters_per_tf', cfg.get('n_reporters_per_tf', 1)),
            'n_confounders': topology_spec.get('n_confounders', cfg.get('n_confounders', 2)),
            'n_noise_genes': topology_spec.get('n_noise_genes', cfg.get('n_noise_genes', 0)),
            'growth_type': topology_spec.get('growth_type', cfg.get('growth_type', 'multi')),
            'noise_sigma': topology_spec.get('noise_sigma', cfg.get('noise_sigma', 0.12)),
            'n_time_bins': topology_spec.get('n_time_bins', cfg.get('n_time_bins', 8)),
            'holdout_bins': topology_spec.get('holdout_bins', cfg.get('holdout_bins', [2, 3])),
            'n_init_cells': topology_spec.get('n_init_cells', cfg.get('n_init_cells', 2000)),
            't_end': topology_spec.get('t_end', cfg.get('t_end', 12.0)),
            **cfg,
        }
    preset = DifficultyPreset(
        name=cfg.get('name', 'custom'),
        n_fates=cfg['n_fates'],
        n_reporters_per_tf=cfg.get('n_reporters_per_tf', 1),
        n_confounders=cfg.get('n_confounders', 2),
        n_noise_genes=cfg.get('n_noise_genes', 0),
        noise_sigma=cfg.get('noise_sigma', 0.12),
        n_time_bins=cfg.get('n_time_bins', 8),
        holdout_bins=cfg.get('holdout_bins', [2, 3]),
        n_init_cells=cfg.get('n_init_cells', 2000),
        growth_type=cfg.get('growth_type', 'multi'),
        t_end=cfg.get('t_end', 12.0),
    )
    seed = cfg.get('seed', 42)
    name = cfg.get('name', f'custom_seed{seed}')
    return preset, seed, name, topology, topology_spec


def batch_from_configs(
    config_path: str,
    output_base: Path = None,
    verbose: bool = True,
) -> List[Dict]:
    """
    Read a JSON config file and generate all scenarios defined in it.

    The JSON file should contain a list of scenario configs.
    Each config specifies structural parameters independently.
    """
    with open(config_path) as f:
        configs = json.load(f)

    if not isinstance(configs, list):
        configs = [configs]  # single config → wrap

    results = []
    total = len(configs)

    for i, cfg in enumerate(configs, 1):
        preset, seed, name, topology, topology_spec = parse_scenario_config(cfg)
        scenario_name = cfg.get('scenario_name') or f"S_{name}_seed{seed}"

        print(f"\n{'━' * 60}")
        print(f"  Config [{i}/{total}]: {scenario_name}")
        print(f"  {preset.n_genes_total} genes, {preset.n_fates} fates, "
              f"{preset.n_confounders} confounders, growth={preset.growth_type}")
        print(f"{'━' * 60}")

        try:
            report = generate_scenario(
                difficulty=name,
                seed=seed,
                output_base=output_base,
                verbose=verbose,
                preset_override=preset,
                scenario_name_override=scenario_name,
                topology=topology,
                topology_spec=topology_spec,
            )
            results.append(report)
        except Exception as e:
            logger.error(f"Failed: {scenario_name}: {e}", exc_info=True)
            results.append({'scenario': scenario_name, 'error': str(e)})

    # Summary
    print(f"\n{'=' * 70}")
    print(f"  Config Batch: {len(results)} scenarios generated")
    print(f"{'=' * 70}")
    print(f"{'Scenario':<30} {'Genes':>5} {'Fates':>5} {'Cells':>6} {'AH':>6} {'Driver':>8} {'Conf':>8}")
    print("-" * 70)
    for r in results:
        if 'error' in r:
            print(f"  {r['scenario']}: ❌ {r['error'][:50]}")
        else:
            v = r['validation']
            ah = "✅" if v['anti_heuristic_passed'] else "❌"
            print(f"  {r['scenario']:<28} {r['n_genes']:>5} {r['n_fates']:>5} "
                  f"{r['n_cells']:>6} {ah:>6} {v['max_driver_time_corr']:>8.4f} "
                  f"{v['max_confounder_time_corr']:>8.4f}")
    print("=" * 70)

    if output_base:
        with open(output_base / "batch_summary.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(
        description="DynBench Scenario Factory",
        epilog="Config mode: python scenario_factory.py --config scenarios.json\n"
               "V2 mode: python scenario_factory.py --v2"
    )
    parser.add_argument("--config", default=None,
                        help="Path to JSON config file (config-driven mode)")
    parser.add_argument("--difficulty", default="medium",
                        help="Comma-separated presets: easy,medium,hard,extreme (preset mode)")
    parser.add_argument("--seeds", default="42",
                        help="Comma-separated random seeds (preset mode)")
    parser.add_argument("--output-dir", default=None,
                        help="Output base dir (default: benchmark/dynbench/)")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--v2", action="store_true",
                        help="V2 mode: generate from v2/ config directory (no seed suffix)")
    parser.add_argument("--v2-scenarios", default=None,
                        help="Comma-separated v2 scenario names to generate (with --v2)")

    args = parser.parse_args()
    output_base = Path(args.output_dir) if args.output_dir else None

    if args.v2:
        # V2 mode: deterministic configs, no external seed
        scenario_names = (
            [s.strip() for s in args.v2_scenarios.split(',')]
            if args.v2_scenarios else None
        )
        batch_generate_v2(
            output_base=output_base,
            verbose=not args.quiet,
            scenario_names=scenario_names,
        )
    elif args.config:
        # Config-driven mode: read JSON, generate diverse scenarios
        batch_from_configs(args.config, output_base, verbose=not args.quiet)
    else:
        # Preset mode: same structure, different seeds
        difficulties = [d.strip() for d in args.difficulty.split(',')]
        seeds = [int(s) for s in args.seeds.split(',')]

        if len(difficulties) == 1 and len(seeds) == 1:
            generate_scenario(difficulties[0], seeds[0], output_base, verbose=not args.quiet)
        else:
            batch_generate(difficulties, seeds, output_base, verbose=not args.quiet)


if __name__ == "__main__":
    main()
