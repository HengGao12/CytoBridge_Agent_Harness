# DynBench v2 - Multi-timepoint Prediction Task

## Objective
You are given multi-timepoint scRNA-seq data from a system with 5 measured genes and 3 cell types.
Your task is to build or use an analysis approach from `train.h5ad`, run downstream analyses in the measured gene state space, and write exactly six deliverable files to the requested output directory:
`holdout_prediction.csv`, `velocity_field.csv`, `growth_rates.csv`, `per_cell_fate.json`, `perturbation_results.json`, and `driver_genes.json`.
The deliverables should describe held-out intermediate cell states, per-cell dynamics, growth/mass behavior, root-cell fate behavior, perturbation responses, and candidate regulatory/growth drivers for this run.

## Data

### Input Files
- **`train.h5ad`**: Single-cell gene expression data at 4 observed time points
  - `adata.X`: Gene expression matrix (normalized, log1p)
  - `adata.obs['time']`: Continuous time value for each cell
  - `adata.obs['time_bin']`: Integer time bin index (0, 1, 3, 5 are observed)
  - `adata.obs['fate_true']`: Canonical cell type labels (A, B, C)
  - `adata.obs['cell_type']`: Alias of `fate_true` kept for backward compatibility
  - `adata.var_names`: Gene_1 through Gene_5
  - Time mapping: bin 0→t=0.0, bin 1→t=2.0, bin 3→t=6.0, bin 5→t=10.0
  - **Note**: Time bins 2 (t=4.0) and 4 (t=8.0) are held out

- **`fate_classifier.pkl`**: Pre-trained fate classifier (sklearn MLPClassifier)
  - Input features: `np.hstack([expression_matrix, time.reshape(-1, 1)])` — shape (n_cells, 6)
  - Feature order: [Gene_1, Gene_2, Gene_3, Gene_4, Gene_5, time]
  - Classes: A, B, C
  - **You MUST use this classifier** for all fate assignment tasks
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

Predict what cells look like at the held-out time points (t=4.0 and t=8.0) that were NOT in training data.
Read `prediction_targets.json` in the same task package for the canonical target-bin and target-time metadata.
Write this CSV with `index=False`; implicit pandas index columns are invalid.

| Column | Description |
|--------|-------------|
| Gene_1 ... Gene_5 | Predicted expression |
| time | Time value (4.0 or 8.0) for each predicted cell |
| weight | Optional rollout/sample weight. Include this column if your prediction is a weighted particle rollout. |

- Generate a realistic number of cells per holdout time point
- Expression values should be in the same scale as training data
- Preserve weighted rollouts: if your simulation produces particle weights, keep them in a `weight` column instead of silently equalizing all samples.

---

### Analysis 2: Velocity field — trajectory analysis
**Output file: `velocity_field.csv`**

Predict the velocity of gene expression for each training cell.
Rows must match training cells in `train.h5ad` (same order, same count).
Write this CSV with `index=False`; implicit pandas index columns are invalid.

| Column | Description |
|--------|-------------|
| velocity_Gene_1 ... velocity_Gene_5 | velocity for each gene |

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
{{
  "<cell_id_1>": {{"B": 0.38, "C": 0.16}},
  "<cell_id_2>": {{"B": 0.11, "C": 0.12}}
}}
```

- Include exactly the cells from time_bin=0
- Cell IDs must match the cell names in `train.h5ad`
- Report probabilities for fates B, C (not A, since A is the progenitor state)

---

### Analysis 5: In silico perturbation analysis — gene knockdown
**Output file: `perturbation_results.json`**

Run perturbation analysis: for each gene (Gene_1 through Gene_5), apply the benchmark perturbation protocol in progenitor cells:
- matched control branch with `z_score = 0.0`
- perturbed branch with `z_score = -5.0`

Then report the fate shift (delta) compared to the matched control.

```json
[
  {{
    "gene_name": "Gene_X",
    "control": {{"B": 0.38, "C": 0.16}},
    "perturbed": {{"B": 0.11, "C": 0.12}},
    "delta": {{"B": 0.02, "C": 0.03}}
  }},
  {{
    "gene_name": "Gene_Y",
    "control": {{"B": 0.38, "C": 0.16}},
    "perturbed": {{"B": 0.11, "C": 0.12}},
    "delta": {{"B": 0.14, "C": 0.05}}
  }}
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
{{
  "grn_edges": [
    {{"source": "Gene_X", "target": "Gene_Y", "score": 0.63}},
    {{"source": "Gene_Z", "target": "Gene_Y", "score": -0.41}},
    {{"source": "Gene_X", "target": "Gene_W", "score": 0.29}}
  ],
  "growth_drivers": {{"Gene_X": 0.37, "Gene_Y": 0.12, "Gene_Z": 0.58, "Gene_W": 0.44}}
}}
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
