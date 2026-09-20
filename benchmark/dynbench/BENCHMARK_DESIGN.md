# DynBench: Design Principles & Factory Guide

> The first scalable, adversarial benchmark for evaluating AI agents on single-cell dynamical modeling.

## 1. Why DynBench

Existing benchmarks for single-cell analysis agents are either:
- **Static** — testing clustering, annotation, or differential expression (no dynamics)
- **Hand-crafted** — each scenario is a one-off, not reproducible or scalable
- **Leaky** — example values in task descriptions inadvertently reveal ground truth

DynBench solves all three:
- **Dynamical**: 6 metrics covering velocity, growth, fate, perturbation, and GRN
- **Scalable**: `scenario_factory.py` batch-generates scenarios from difficulty presets
- **Anti-heuristic**: Confounding genes with higher time-correlation than real growth drivers

## 2. Architecture Overview

```
scenario_factory.py
    │
    ├── DifficultyPreset        # easy / medium / hard / extreme
    │     ├── n_fates            # 2 / 4 / 8 / 16
    │     ├── n_confounders      # 1 / 2 / 4 / 6
    │     ├── noise_sigma        # 0.10 / 0.12 / 0.15 / 0.18
    │     └── holdout_bins       # [2,3] / [2,3] / [3,4,5] / [4,5,6]
    │
    ├── generate_grn()           # → GRNBlueprint (edges + roles)
    │     ├── Layer 1: Toggle switches (TFs)
    │     ├── Layer 2: Reporters
    │     ├── Layer 3: Confounders (anti-heuristic)
    │     └── Layer 4: Noise genes
    │
    ├── build_config()           # → ScenarioConfig (for boolode_sim)
    ├── simulate()               # → SimulationResult (BoolODE SDE)
    ├── validate_anti_heuristic()
    │
    └── Output:
          ├── ground_truth/S_{diff}_seed{N}/    (7 GT files)
          ├── task_packages/S_{diff}_seed{N}/   (train.h5ad + classifier + TASK.md)
          └── batch_summary.json
```

## 3. The Four-Layer GRN Design

Every scenario's gene regulatory network follows a strict layered architecture. This ensures each difficulty level has the right properties.

### Layer 1: Toggle Switches (Fate Decisions)

Mutually inhibiting TF pairs create bistable fate decisions.

```
n_fates = 2^(n_toggles)
```

| Example | Toggles | Fates | Pattern |
|---------|---------|-------|---------|
| easy    | 1       | 2     | G1 ⊣ G2 → B vs C |
| medium  | 2       | 4     | G1⊣G2, G3⊣G4 → B/C/D/E |
| hard    | 3       | 8     | + G5⊣G6 → B through I |

Each toggle pair has:
- Mutual inhibition (Hill repression, n≈3)
- Self-activation (stabilizes bistability)
- Slight parameter randomness (K ± 0.05) for variety across seeds

### Layer 2: Reporters (Signal Readouts)

One reporter per TF, activated by its parent. These are the **candidate growth drivers**.

```
Reporter_i ← TF_j  (Hill activation, K≈0.3)
```

**Why reporters, not TFs?** TFs have high variance (they ARE the fate markers). If growth depended on TFs directly, agents could trivially identify drivers by variance analysis. Reporters add indirection.

### Layer 3: Confounders (Anti-Heuristic)

**The most important layer.** Late markers that integrate reporter signals from multiple branches.

```
Confounder_k ← Reporter_i + Reporter_j  (cross-branch, low K≈0.2)
```

**Design properties:**
- Low degradation rate (0.12 vs 0.25 for reporters) → slow accumulation
- Activated by multiple reporters → rises monotonically with time
- **Higher time-correlation than real growth drivers** (validated automatically)

**Why this breaks heuristics:**

| Agent Heuristic | What it does | Why it fails |
|----------------|-------------|--------------|
| `argmax(corr(gene, time))` | Picks gene most correlated with time | Confounders have higher time-corr |
| `argmax(variance)` | Picks highest-variance gene | TFs have highest variance, not drivers |
| `PCA loading → driver` | Top PCA genes are drivers | PC1/PC2 capture TF toggles, not growth |
| Co-expression clustering | Growth genes cluster together | Confounders co-express with drivers |

### Layer 4: Noise Genes (Hard+ only)

Uncorrelated random walk genes (no GRN edges). These test whether agents can distinguish signal from noise in the GRN inference task.

## 4. Growth Function Design

```python
growth_rate = base_rate + amplitude × sigmoid(β × (mean(drivers) - threshold))
```

| Parameter | Value | Why |
|-----------|-------|-----|
| `base_rate` | -0.02 | Slight death without driver expression |
| `amplitude` | 0.06 | Net proliferation when drivers are high |
| `threshold` | 0.3 | Sigmoid center matched to reporter activation |
| `β` | 5.0 | Sharp transition |

**Driver selection:**
- `single` (easy): One reporter from toggle 1
- `multi` (medium+): Two reporters from both sides of toggle 1 → mean
- `interaction` (hard+): Same as multi, but could be extended to multiplicative

**Key insight**: Using `mean(Reporter_A, Reporter_B)` where A and B come from opposing sides of the same toggle means growth is approximately constant across fates (since one reporter is high and the other low regardless of fate). This prevents agents from using fate-specific growth patterns as a shortcut.

## 5. Anti-Heuristic Validation

After every scenario generation, the factory automatically validates:

```python
assert max_confounder_time_corr > max_driver_time_corr
```

Typical results from batch generation:

| Difficulty | Driver max corr | Confounder max corr | Gap |
|-----------|----------------|--------------------|----|
| easy | 0.479 | 0.944 | 0.465 |
| medium | 0.432 | 0.933 | 0.501 |
| hard | 0.398 | 0.903 | 0.505 |

If validation fails (rare, can happen with certain seeds), the scenario is flagged and should be re-generated.

## 6. Difficulty Scaling

The factory supports 4 difficulty presets, each independently tunable:

| Dimension | Easy | Medium | Hard | Extreme |
|-----------|------|--------|------|---------|
| Genes | 5 | 10 | 18 | 38 |
| Fates | 2 | 4 | 8 | 16 |
| Toggles | 1 | 2 | 3 | 4 |
| Confounders | 1 | 2 | 4 | 6 |
| Noise genes | 0 | 0 | 2 | 8 |
| σ (noise) | 0.10 | 0.12 | 0.15 | 0.18 |
| Time bins | 6 | 8 | 10 | 12 |
| Holdout bins | 2 | 2 | 3 | 3 |
| Init cells | 1500 | 2000 | 2500 | 3000 |
| GRN edges | ~8 | ~16 | ~32 | ~64 |
| AUPRC baseline | ~0.32 | ~0.16 | ~0.10 | ~0.05 |

**What makes harder scenarios harder:**
1. More genes → more candidate edges → lower AUPRC baseline → harder GRN inference
2. More fates → more complex trajectory → harder distribution prediction
3. More confounders → more distractors → harder growth driver ID
4. More holdout bins → larger temporal gap → harder extrapolation
5. Higher noise → noisier velocity → harder velocity estimation

## 7. Batch Usage

### Generate a benchmark suite
```bash
# 12 scenarios: 4 difficulties × 3 seeds
python scenario_factory.py --difficulty easy,medium,hard,extreme --seeds 42,43,44

# Quick single scenario
python scenario_factory.py --difficulty medium --seeds 100
```

### Output structure
```
benchmark/dynbench/
├── ground_truth/
│   ├── S_easy_seed42/
│   ├── S_easy_seed43/
│   ├── S_medium_seed42/
│   ├── S_hard_seed42/
│   └── ...
├── task_packages/
│   ├── S_easy_seed42/     (train.h5ad + classifier + TASK.md)
│   └── ...
└── batch_summary.json     (all validation reports)
```

### Evaluate an agent
```python
from benchmark.dynbench.eval.dynbench_eval_v2 import evaluate_all
results = evaluate_all('results/S_medium_seed42/agent_output/', 'ground_truth/S_medium_seed42/')
```

## 8. Extending the Factory

### Adding a new GRN motif

To add a new motif (e.g., feedforward loop, oscillator), modify `generate_grn()`:

```python
# Example: add feedforward loop (TF → Reporter → Confounder, AND TF → Confounder)
if preset.name == 'feedforward':
    # Direct edge from TF to confounder (bypasses reporter)
    edges.append(GRNEdge(source=tf_idx, target=c_idx, effect=1.0, K=0.4, n=2.0))
```

### Adding a new growth function type

Extend the `growth_type` field in `DifficultyPreset`:

```python
# In build_config():
if preset.growth_type == 'nonmonotonic':
    # Growth peaks at intermediate expression, then drops
    # Requires custom compute_growth_rate in boolode_sim.py
    ...
```

### Adding a new difficulty preset

```python
PRESETS['nightmare'] = DifficultyPreset(
    name='nightmare', n_fates=32,
    n_reporters_per_tf=2, n_confounders=10, n_noise_genes=20,
    noise_sigma=0.25, n_time_bins=16, holdout_bins=[5,6,7,8],
    n_init_cells=5000, growth_type='interaction', t_end=24.0,
)
```

## 9. Lessons Learned

1. **Confounders must have lower degradation** (0.12 vs 0.25). This creates slow accumulation → late activation → high time-correlation. Without this, the anti-heuristic fails.

2. **Growth drivers should be reporters, not TFs.** TFs have high variance (they determine fate). Reporters add indirection, making identification harder.

3. **`mean(Gene_A, Gene_B)` across toggle branches** makes growth ~fate-independent. This prevents agents from identifying drivers via fate-specific analysis.

4. **TASK.md leakage is subtle.** Even putting `"Gene_3": 0.85` in an example can reveal that Gene_3 has high expression. Use random values and abstract names (Gene_X/Y/Z).

5. **Oracle classifier must use ALL time points** (not just training bins). This ensures consistent fate assignment between agent predictions and ground truth evaluation.

6. **3-gene scenarios are too easy for GRN evaluation.** With 9 possible edges and 6 true edges, AUPRC baseline is 0.667. Need ≥10 genes for meaningful discrimination.

7. **Holdout bins at intermediate time points** (not endpoints) are the right choice. Agents must interpolate, not extrapolate — this tests whether they learned the dynamics, not just the boundary conditions.

## 10. Phase 2: Controlled-Variable Scenarios + Metric Revisions

> See the internal benchmark redesign proposal for the full rationale.

### 10.1 Metric Revisions (Phase 1, zero re-runs)

**M3 piecewise quadratic normalization** replaces the old clipped formula and avoids a cliff at ratio = 1:
```
Old: w1_score = clip(1.0 - ratio, 0, 1)              # cliff at ratio = 1
New: w1_score = 1.0 - ratio           (ratio <= 1)    # unchanged in the useful range
     w1_score = -0.5 * (ratio - 1)^2  (ratio > 1)     # smooth quadratic decay
```
The piecewise quadratic form is preferred over a sigmoid because a sigmoid can
over-compress the useful score range. The new formula preserves the old behavior
for ratio <= 1 and only smooths the failure side.

**M6 direction accuracy** — adds regulatory direction to GRN scoring:
```
Old: score = AUPRC
New: score = 0.5 × AUPRC + 0.5 × direction_accuracy
```
Rationale: pure AUPRC underestimates agents that correctly predict edge direction.

### 10.2 Controlled-Variable Scenarios (Phase 2)

Core principle: **one variable at a time**. All use `scenario_seed=42`.

**Group A — Gene gradient** (fixed 8 fates, varying gene complexity):

| Scene | Genes (approx) | Confounders | Noise | Purpose |
|-------|----------------|-------------|-------|---------|
| A1 | ~12 | 0 | 0 | Low-dim M1/M3 baseline |
| A2 | ~18 | 3 | 3 | Gene count effect |
| A3 | ~25 | 4 | 3 | Match existing asymmetric |
| A4 | ~35 | 8 | 9 | High-dim M1/M3 ceiling |

**Group B — Fate gradient** (varying fate count):

| Scene | Fates | Topology | Purpose |
|-------|-------|----------|---------|
| B1 | 2 | compact_toggle | M4 baseline |
| B2 | 4 | branching_tree | Medium classification |
| B3 | 8 | asymmetric_tree | Match existing asymmetric |
| B4 | 16 | branching_tree | M4 ceiling |

**Group C — Topology contrast** (fixed 4 fates, different topologies):

| Scene | Topology | Purpose |
|-------|----------|---------|
| C1 | branching_tree (symmetric) | Baseline topology |
| C2 | asymmetric_tree | Unbalanced branching |
| C3 | branching_with_feedback | Feedback effects |
| C4 | competitive_multistable | Multi-stability |

### 10.3 Seed Design

- **scenario_seed=42** fixed for Phase 2 (scenarios generated once)
- **agent_seed** ∈ {42, 137, 256} (controls agent stochasticity)
- Phase 3 adds scenario_seed ∈ {137, 256} for dual-factor ANOVA

### 10.4 Generation

```bash
# Dry-run: see what will be generated
python benchmark/dynbench/generate_phase2_scenarios.py

# Generate all 12 scenarios
python benchmark/dynbench/generate_phase2_scenarios.py --execute

# Generate only one group
python benchmark/dynbench/generate_phase2_scenarios.py --group A --execute
```
