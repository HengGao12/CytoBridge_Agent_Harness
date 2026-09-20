"""
BoolODE-style GRN-driven stochastic simulator with growth.

Mathematical framework (following BoolODE / BEELINE):
  dx_i/dt = a_i * f_i(activators, repressors) - d_i * x_i + sigma * dW_i

  f_i uses Hill functions:
    hill_pos(x, K, n) = x^n / (K^n + x^n)     (activation)
    hill_neg(x, K, n) = K^n / (K^n + x^n)      (repression)

Growth wrapper (following TIGON):
  growth_rate(x) = base_rate + amp * sigmoid(beta * (g(x) - threshold))
  Birth/death via Poisson process at each time step.

References:
  - Pratapa et al., Nature Methods 2020 (BoolODE / BEELINE)
  - Chen et al., PNAS 2024 (TIGON)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable


# ─── Hill functions ──────────────────────────────────────────────────────────

def hill_pos(x: np.ndarray, K: float = 0.5, n: float = 2.0) -> np.ndarray:
    """Activation Hill function: x^n / (K^n + x^n)"""
    xn = np.power(np.maximum(x, 0.0), n)
    return xn / (np.power(K, n) + xn + 1e-12)


def hill_neg(x: np.ndarray, K: float = 0.5, n: float = 2.0) -> np.ndarray:
    """Repression Hill function: K^n / (K^n + x^n)"""
    xn = np.power(np.maximum(x, 0.0), n)
    return np.power(K, n) / (np.power(K, n) + xn + 1e-12)


# ─── GRN Edge ────────────────────────────────────────────────────────────────

@dataclass
class GRNEdge:
    """A single edge in the gene regulatory network."""
    source: int      # index of regulator gene
    target: int      # index of target gene
    effect: float    # positive = activation, negative = repression
    K: float = 0.5   # Hill constant
    n: float = 2.0   # Hill coefficient


# ─── Growth Function ─────────────────────────────────────────────────────────

@dataclass
class GrowthConfig:
    """Configuration for cell growth/death (TIGON-style)."""
    enabled: bool = True
    # Which gene(s) drive growth — int or list of ints
    growth_gene_idx: any = 2               # Gene_3 in S1; can be list e.g. [4, 5]
    base_rate: float = 0.0             # base growth rate
    amplitude: float = 1.0             # growth rate amplitude
    threshold: float = 0.5             # sigmoid threshold
    beta: float = 5.0                  # sigmoid steepness
    # Asymmetry: different rates for different fates
    # If None, growth is symmetric
    asymmetry: Optional[Dict[str, float]] = None


def compute_growth_rate(x: np.ndarray, config: GrowthConfig) -> np.ndarray:
    """
    Compute per-cell growth rate based on gene expression.
    
    Args:
        x: (n_cells, n_genes) expression matrix
        config: growth configuration
        
    Returns:
        (n_cells,) growth rates
    """
    if not config.enabled:
        return np.zeros(x.shape[0])
    
    # Support single gene or mean of multiple genes
    idx = config.growth_gene_idx
    if isinstance(idx, (list, tuple)):
        gene_val = np.mean(x[:, idx], axis=1)
    else:
        gene_val = x[:, idx]
    # Sigmoid growth function
    sigmoid = 1.0 / (1.0 + np.exp(-config.beta * (gene_val - config.threshold)))
    growth = config.base_rate + config.amplitude * sigmoid
    return growth


# ─── Scenario Configuration ──────────────────────────────────────────────────

@dataclass
class ScenarioConfig:
    """Full scenario configuration."""
    name: str
    n_genes: int
    n_cells: int
    grn_edges: List[GRNEdge]
    
    # Per-gene parameters
    production_rates: np.ndarray    # a_i: max production rate
    degradation_rates: np.ndarray   # d_i: degradation rate
    
    # Simulation parameters
    t_start: float = 0.0
    t_end: float = 10.0
    dt: float = 0.01
    sigma: float = 0.1              # SDE noise level
    
    # Snapshot time points
    snapshot_times: List[float] = field(default_factory=lambda: [0.0, 2.0, 4.0, 6.0, 8.0, 10.0])
    
    # Initial conditions
    initial_mean: Optional[np.ndarray] = None
    initial_std: float = 0.1
    
    # Growth
    growth: GrowthConfig = field(default_factory=GrowthConfig)
    
    # Random seed
    seed: int = 42


# ─── ODE Right-hand side ─────────────────────────────────────────────────────

def compute_drift(x: np.ndarray, grn_edges: List[GRNEdge],
                  production_rates: np.ndarray,
                  degradation_rates: np.ndarray) -> np.ndarray:
    """
    Compute the deterministic drift dx/dt for each cell.
    
    This IS the ground truth velocity.
    
    Args:
        x: (n_cells, n_genes) current gene expression
        grn_edges: list of GRN edges
        production_rates: (n_genes,) max production rates
        degradation_rates: (n_genes,) degradation rates
        
    Returns:
        (n_cells, n_genes) drift vectors (= GT velocity)
    """
    n_cells, n_genes = x.shape
    drift = np.zeros_like(x)
    
    # For each gene, compute the regulatory input
    for gene_i in range(n_genes):
        # Find all edges targeting this gene
        activators = [e for e in grn_edges if e.target == gene_i and e.effect > 0]
        repressors = [e for e in grn_edges if e.target == gene_i and e.effect < 0]
        
        # Compute regulatory function f_i
        if not activators and not repressors:
            # No regulation → constitutive expression
            f_i = np.ones(n_cells) * 0.5
        else:
            # f_i = product(hill_pos for activators) * product(hill_neg for repressors)
            f_i = np.ones(n_cells)
            for edge in activators:
                f_i *= hill_pos(x[:, edge.source], K=edge.K, n=edge.n)
            for edge in repressors:
                f_i *= hill_neg(x[:, edge.source], K=edge.K, n=edge.n)
        
        # dx_i/dt = a_i * f_i - d_i * x_i
        drift[:, gene_i] = production_rates[gene_i] * f_i - degradation_rates[gene_i] * x[:, gene_i]
    
    return drift


# ─── SDE Simulator ───────────────────────────────────────────────────────────

@dataclass
class SimulationResult:
    """Container for simulation results with all ground truth."""
    # Expression data at snapshot times
    snapshots: Dict[float, np.ndarray]             # {time: (n_cells, n_genes)}
    
    # Ground truth fields (evaluated at snapshot cells)
    velocity_gt: Dict[float, np.ndarray]           # {time: (n_cells, n_genes)} = dx/dt
    growth_gt: Dict[float, np.ndarray]             # {time: (n_cells,)} = growth rate
    
    # Full trajectories (for fate computation)
    trajectories: np.ndarray                        # (n_cells, n_steps, n_genes)
    trajectory_times: np.ndarray                    # (n_steps,)
    
    # Metadata
    cell_fates: np.ndarray                          # (n_cells,) fate labels at final time
    cell_pseudotimes: np.ndarray                    # (n_cells,) pseudotime values
    
    # GRN ground truth
    grn_edges: List[GRNEdge]
    
    # Growth weights (for unbalanced transport)
    cell_weights: Dict[float, np.ndarray]           # {time: (n_cells,)} = effective mass


def simulate(config: ScenarioConfig, verbose: bool = True, fate_fn=None) -> SimulationResult:
    """
    Run the full BoolODE-style simulation.
    
    1. Initialize n_cells at t=0
    2. Integrate SDE: dx = drift(x)*dt + sigma*dW
    3. Apply growth (birth/death) at each step
    4. Record snapshots + GT fields at specified times
    
    Returns:
        SimulationResult with all data + ground truth
    """
    rng = np.random.RandomState(config.seed)
    
    n_cells = config.n_cells
    n_genes = config.n_genes
    n_steps = int((config.t_end - config.t_start) / config.dt)
    dt = config.dt
    
    # Initialize: small random perturbation around initial_mean
    if config.initial_mean is not None:
        x = config.initial_mean[None, :] + rng.randn(n_cells, n_genes) * config.initial_std
    else:
        # Start near the center of the state space
        x = np.ones((n_cells, n_genes)) * 0.5 + rng.randn(n_cells, n_genes) * config.initial_std
    x = np.maximum(x, 0.0)  # gene expression >= 0
    
    # Storage
    snapshots = {}
    velocity_gt = {}
    growth_gt = {}
    cell_weights = {}
    
    # Track cumulative growth for weight computation
    cumulative_log_growth = np.zeros(n_cells)
    
    # Trajectory storage (subsample for memory)
    traj_subsample = max(1, n_steps // 200)  # store ~200 time points
    trajectories = []
    trajectory_times = []
    
    # Sorted snapshot times for detection
    snap_times_sorted = sorted(config.snapshot_times)
    snap_idx = 0
    
    current_time = config.t_start
    
    for step in range(n_steps):
        # 1. Compute drift (= GT velocity)
        drift = compute_drift(x, config.grn_edges,
                             config.production_rates,
                             config.degradation_rates)
        
        # 2. Compute growth rate (= GT growth)
        growth = compute_growth_rate(x, config.growth)
        
        # 3. Check if we need to record a snapshot
        if snap_idx < len(snap_times_sorted):
            next_snap = snap_times_sorted[snap_idx]
            if current_time >= next_snap - dt/2:
                t_key = round(next_snap, 4)
                # Recompute drift/growth for current population (may have changed size)
                snap_drift = compute_drift(x, config.grn_edges,
                                          config.production_rates,
                                          config.degradation_rates)
                snap_growth = compute_growth_rate(x, config.growth)
                snapshots[t_key] = x.copy()
                velocity_gt[t_key] = snap_drift.copy()
                growth_gt[t_key] = snap_growth.copy()
                cell_weights[t_key] = np.exp(cumulative_log_growth.copy())
                snap_idx += 1
                if verbose:
                    print(f"  Snapshot at t={t_key:.2f}: {x.shape[0]} cells, "
                          f"mean expr = [{', '.join(f'{x[:, i].mean():.3f}' for i in range(min(n_genes, 5)))}]")
        
        # 4. Store trajectory (subsampled)
        if step % traj_subsample == 0:
            trajectories.append(x.copy())
            trajectory_times.append(current_time)
        
        # 5. SDE step: Euler-Maruyama
        noise = rng.randn(*x.shape) * config.sigma * np.sqrt(dt)
        x = x + drift * dt + noise
        x = np.maximum(x, 0.0)  # gene expression >= 0
        
        # 6. Growth: actual birth/death events (Poisson process)
        if config.growth.enabled:
            # Birth probability ~ max(growth, 0) * dt
            # Death probability ~ max(-growth, 0) * dt
            birth_prob = np.clip(growth, 0, None) * dt
            death_prob = np.clip(-growth, 0, None) * dt
            
            # Cap probabilities
            birth_prob = np.minimum(birth_prob, 0.1)
            death_prob = np.minimum(death_prob, 0.1)
            
            # Birth events: duplicate cell with small noise
            birth_mask = rng.rand(len(x)) < birth_prob
            if birth_mask.any():
                daughters = x[birth_mask].copy()
                # Add noise to daughters (asymmetric division)
                daughters += rng.randn(*daughters.shape) * config.sigma * 0.3
                daughters = np.maximum(daughters, 0.0)
                
                # Daughter drift/growth
                daughter_drift = compute_drift(daughters, config.grn_edges,
                                              config.production_rates,
                                              config.degradation_rates)
                daughter_growth_vals = compute_growth_rate(daughters, config.growth)
                
                x = np.vstack([x, daughters])
                drift = np.vstack([drift, daughter_drift])
                growth = np.concatenate([growth, daughter_growth_vals])
                cumulative_log_growth = np.concatenate([
                    cumulative_log_growth,
                    cumulative_log_growth[birth_mask].copy()
                ])
            
            # Death events: remove cells
            death_mask = rng.rand(len(x)) < np.concatenate([death_prob, np.zeros(len(x) - len(death_prob))]) if len(x) > len(death_prob) else rng.rand(len(x)) < death_prob
            survive_mask = ~death_mask
            if not survive_mask.all() and survive_mask.sum() > 10:
                x = x[survive_mask]
                cumulative_log_growth = cumulative_log_growth[survive_mask]
        
        # Also track weights for eval compatibility
        cumulative_log_growth += growth[:len(x)] * dt if len(growth) >= len(x) else np.zeros(len(x)) * dt
        
        current_time += dt
    
    # Record final snapshot if needed
    if snap_idx < len(snap_times_sorted):
        t_key = round(snap_times_sorted[snap_idx], 4)
        drift = compute_drift(x, config.grn_edges,
                             config.production_rates,
                             config.degradation_rates)
        growth = compute_growth_rate(x, config.growth)
        snapshots[t_key] = x.copy()
        velocity_gt[t_key] = drift.copy()
        growth_gt[t_key] = growth.copy()
        cell_weights[t_key] = np.exp(cumulative_log_growth.copy())
    
    # Compute cell fates at final time
    x_final = x.copy()
    cell_fates = assign_fates(x_final, config, fate_fn=fate_fn)
    
    # Trajectories: variable cell count, store as list (not stacked array)
    # We only need snapshots for eval, not full trajectories
    trajectories_arr = np.zeros((1, x.shape[0], n_genes))  # placeholder
    trajectory_times_arr = np.array([current_time])
    
    # Pseudotime for current population
    cell_pseudotimes = np.ones(x.shape[0]) * current_time
    
    if verbose:
        print(f"\nSimulation complete: {x.shape[0]} cells (started {n_cells}), {n_genes} genes, "
              f"{len(snapshots)} snapshots")
        fate_counts = {f: (cell_fates == f).sum() for f in np.unique(cell_fates)}
        print(f"Fate distribution: {fate_counts}")
    
    return SimulationResult(
        snapshots=snapshots,
        velocity_gt=velocity_gt,
        growth_gt=growth_gt,
        trajectories=trajectories_arr,
        trajectory_times=trajectory_times_arr,
        cell_fates=cell_fates,
        cell_pseudotimes=cell_pseudotimes,
        grn_edges=config.grn_edges,
        cell_weights=cell_weights,
    )


def assign_fates(x: np.ndarray, config: ScenarioConfig, fate_fn=None) -> np.ndarray:
    """
    Assign fate labels based on final gene expression.
    If fate_fn is provided, use it; otherwise default to toggle switch logic.
    """
    if fate_fn is not None:
        return fate_fn(x)
    # Default: compare the two toggle switch genes
    if config.n_genes >= 2:
        fate_labels = np.where(x[:, 0] > x[:, 1], "Fate_A", "Fate_B")
    else:
        fate_labels = np.array(["Fate_A"] * x.shape[0])
    return fate_labels


# ─── Perturbation GT ─────────────────────────────────────────────────────────

def simulate_perturbation(config: ScenarioConfig,
                          gene_idx: int,
                          perturbation_strength: float = -2.0,
                          n_runs: int = 20,
                          verbose: bool = False,
                          fate_fn=None,
                          initial_cells: Optional[np.ndarray] = None,
                          reference_cells: Optional[np.ndarray] = None,
                          control_z_score: float = 0.0,
                          perturbed_z_score: float = -5.0) -> Dict:
    """
    Generate perturbation GT.

    Benchmark protocol:
    1. Start from an explicit root-cell population (`initial_cells`)
    2. Apply gene-space z-score intervention in the current state coordinates
    3. Compute control / perturbed fate probabilities from the same root cells
    4. Aggregate per-cell fate probabilities into population proportions

    Args:
        config: scenario configuration
        gene_idx: which gene to perturb
        perturbation_strength: deprecated; benchmark perturbations use z-score fields.
        n_runs: number of stochastic re-runs for fate probability
        initial_cells: Optional root-cell population. When provided, this is
            treated as the benchmark-canonical branch.
        reference_cells: Optional reference population for mean/std estimation.
            Defaults to `initial_cells`.
        control_z_score: Benchmark control protocol in z-score units.
        perturbed_z_score: Benchmark perturbed protocol in z-score units.
        
    Returns:
        dict with control and perturbed fate proportions
    """
    if initial_cells is None:
        raise ValueError("simulate_perturbation requires explicit initial_cells for DynBench perturbation ground truth")

    initial_cells = np.asarray(initial_cells, dtype=np.float64)
    if initial_cells.ndim != 2 or initial_cells.shape[1] != config.n_genes:
        raise ValueError(
            f"initial_cells must have shape (n_cells, {config.n_genes}), "
            f"got {initial_cells.shape}"
        )

    if reference_cells is None:
        reference_cells = initial_cells
    reference_cells = np.asarray(reference_cells, dtype=np.float64)
    if reference_cells.ndim != 2 or reference_cells.shape[1] != config.n_genes:
        raise ValueError(
            f"reference_cells must have shape (n_cells, {config.n_genes}), "
            f"got {reference_cells.shape}"
        )

    ref_mu = reference_cells.mean(axis=0)
    ref_sigma = reference_cells.std(axis=0)

    def _apply_zscore_intervention(cells: np.ndarray, z_score: float) -> np.ndarray:
        target_value = max(0.0, float(ref_mu[gene_idx] + z_score * ref_sigma[gene_idx]))
        perturbed_cells = np.array(cells, copy=True)
        perturbed_cells[:, gene_idx] = target_value
        return perturbed_cells

    control_cells = _apply_zscore_intervention(initial_cells, control_z_score)
    perturbed_cells = _apply_zscore_intervention(initial_cells, perturbed_z_score)

    ctrl_probs, ctrl_fates = compute_percell_fate_probability(
        config,
        control_cells,
        n_runs=n_runs,
        fate_fn=fate_fn,
    )
    pert_probs, pert_fates = compute_percell_fate_probability(
        config,
        perturbed_cells,
        n_runs=n_runs,
        fate_fn=fate_fn,
    )

    ctrl_props = {
        fate: float(ctrl_probs[:, i].mean())
        for i, fate in enumerate(ctrl_fates)
    }
    pert_props = {
        fate: float(pert_probs[:, i].mean())
        for i, fate in enumerate(pert_fates)
    }

    all_fates = sorted(set(list(ctrl_props.keys()) + list(pert_props.keys())))
    delta = {f: pert_props.get(f, 0.0) - ctrl_props.get(f, 0.0) for f in all_fates}

    if verbose:
        print(
            f"Gene {gene_idx} perturbation (control_z={control_z_score}, "
            f"perturbed_z={perturbed_z_score}):"
        )
        print(f"  Control: {ctrl_props}")
        print(f"  Perturbed: {pert_props}")
        print(f"  Delta: {delta}")
    
    return {
        "gene_idx": gene_idx,
        "perturbation_strength": perturbed_z_score,
        "control_z_score": control_z_score,
        "perturbed_z_score": perturbed_z_score,
        "intervention_space": "gene_space",
        "control": ctrl_props,
        "perturbed": pert_props,
        "control_proportions": ctrl_props,
        "perturbed_proportions": pert_props,
        "delta": delta,
    }


# ─── Per-cell Fate Probability GT ────────────────────────────────────────────

def compute_percell_fate_probability(config: ScenarioConfig,
                                     initial_cells: np.ndarray,
                                     n_runs: int = 50,
                                     fate_fn=None) -> np.ndarray:
    """
    Compute per-cell fate probability by running multiple stochastic simulations.
    
    Args:
        config: scenario configuration
        initial_cells: (n_cells, n_genes) initial expression for each cell
        n_runs: number of stochastic runs per cell
        fate_fn: optional custom fate assignment function (x -> labels)
        
    Returns:
        (n_cells, n_fates) fate probability matrix
    """
    n_cells = initial_cells.shape[0]
    fate_labels_all = []
    
    for run in range(n_runs):
        rng = np.random.RandomState(config.seed + run * 1000)
        x = initial_cells.copy()
        n_steps = int((config.t_end - config.t_start) / config.dt)
        
        for step in range(n_steps):
            drift = compute_drift(x, config.grn_edges,
                                 config.production_rates,
                                 config.degradation_rates)
            noise = rng.randn(*x.shape) * config.sigma * np.sqrt(config.dt)
            x = x + drift * config.dt + noise
            x = np.maximum(x, 0.0)
        
        fates = assign_fates(x, config, fate_fn=fate_fn)
        fate_labels_all.append(fates)
    
    # Compute probabilities
    fate_labels_all = np.array(fate_labels_all)  # (n_runs, n_cells)
    unique_fates = sorted(np.unique(fate_labels_all))
    
    fate_probs = np.zeros((n_cells, len(unique_fates)))
    for i, fate in enumerate(unique_fates):
        fate_probs[:, i] = (fate_labels_all == fate).sum(axis=0) / n_runs
    
    return fate_probs, unique_fates
