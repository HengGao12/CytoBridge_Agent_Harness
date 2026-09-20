---
title: "Downstream compute and plot API reference"
summary: "Source-backed reference for CytoBridge downstream compute and plot APIs, including input/output semantics and array shapes."
read_when:
  - "Calling downstream package APIs"
  - "Checking downstream input shape, output shape, payload fields, side effects, or artifacts"
  - "Choosing which downstream source file to inspect"
---
# Downstream Compute And Plot API Reference

This page documents the public downstream APIs an agent should call after a
model is trained. It is intentionally source-backed. If a parameter, shape,
payload field, or side effect is unclear, inspect the source path listed for
that API before writing code or scientific claims.

## 0. Source Escalation

Read docs first. Read source when the exact axis order, shape contract,
projection behavior, or side effect matters for a report, paper, or external
export.

| Need | Source path |
| --- | --- |
| Compute API exports | `CytoBridge-main/CytoBridge/tl/downstream/__init__.py` |
| Shared contract constants and validation | `CytoBridge-main/CytoBridge/tl/downstream/contracts.py` |
| Velocity, velocity graph, velocity drivers, velocity Jacobian drivers | `CytoBridge-main/CytoBridge/tl/downstream/velocity.py` |
| ODE/SDE trajectory bundles | `CytoBridge-main/CytoBridge/tl/downstream/trajectory.py` |
| Trajectory dataset export used by downstream toolkit | `CytoBridge-main/CytoBridge/tl/perturbation.py` |
| Growth summary, growth drivers, growth Jacobian drivers | `CytoBridge-main/CytoBridge/tl/downstream/growth.py` |
| GRN and velocity-Jacobian matrix estimation | `CytoBridge-main/CytoBridge/tl/downstream/grn.py` |
| Plot API exports | `CytoBridge-main/CytoBridge/pl/downstream/__init__.py` |
| Velocity stream plotting | `CytoBridge-main/CytoBridge/pl/downstream/velocity_plot.py` |
| Growth and driver plotting | `CytoBridge-main/CytoBridge/pl/downstream/growth_plot.py` |
| GRN plotting | `CytoBridge-main/CytoBridge/pl/downstream/grn_plot.py` |
| ODE/SDE trajectory plotting | `CytoBridge-main/CytoBridge/pl/downstream/trajectory_plot.py` |
| Agent high-level wrappers and subprocess guards | `cytobridge_agent/tools/downstream_analysis_toolkit.py` |

Do not infer current behavior from old reports, notebooks, or benchmark
adapters. The package source is the authority for API semantics.

## 1. Canonical Data Contract

Most package-native downstream bundle APIs expect the trained-data schema
below. `ensure_contract(...)` raises when required fields are missing.

| Field | Location | Meaning | Shape |
| --- | --- | --- | --- |
| `time_point_processed` | `adata.obs` | Processed scalar time used during training and rollout. | `(n_obs,)` |
| `X_latent` | `adata.obsm` | Model state vector for each observed cell. | `(n_obs, latent_dim)` |
| `velocity_latent` | `adata.obsm` | Model velocity diagnostic in latent state space. | `(n_obs, latent_dim)` |
| `growth_rate` | `adata.obs` or `adata.obsm` | Growth field `d/dt log w`, not absolute population mass. | reshapeable to `(n_obs,)` |
| `PCs` | `adata.varm` | PCA loadings used for latent-to-gene projection. | `(n_vars, >= latent_dim)` |

Gene-space velocity, gene-space drivers, and gene-space GRN summaries require a
compatible `adata.varm["PCs"]`. If PCA loadings are missing or incompatible, use
latent-space outputs and explicitly carry the warning into the manifest.

## 2. Common Bundle Return Payload

Most bundle APIs return a dictionary with a shared structure:

```python
{
    "space_used": "latent" | "gene",
    "projection_backend": "none" | "pca",
    "artifacts": {"artifact_name": "/path/to/file"},
    "warnings": ["..."],
    # API-specific fields...
}
```

`space_used` and `projection_backend` are part of the evidence boundary. Do not
claim gene-level mechanism from latent-only outputs, and do not suppress
warnings when writing paper/report text.

## 3. Compute APIs: `CytoBridge.tl.downstream`

Import examples:

```python
from CytoBridge import tl
from CytoBridge.tl import downstream as ds
```

### 3.1 `compute_velocity_bundle`

Source:

- `CytoBridge-main/CytoBridge/tl/downstream/velocity.py`

Signature:

```python
compute_velocity_bundle(adata, model=None, device="cuda", output_dir=None)
```

Inputs:

- `adata.obsm["X_latent"]`: observed cell states, shape `(n_obs, latent_dim)`.
- `adata.obs["time_point_processed"]`: scalar processed time, shape `(n_obs,)`.
- `model`: optional trained model. If `None`, loaded from `adata`.
- `device`: `"cuda"` falls back to CPU if CUDA is unavailable.
- `output_dir`: optional artifact directory.

Internal tensor contract:

- `all_data`: `(n_obs, latent_dim)`.
- `all_times`: reshaped to `(n_obs, 1)`.
- `net_input = concat([all_data, all_times], dim=1)`: `(n_obs, latent_dim + 1)`.
- `model.velocity_net(net_input)`: expected `(n_obs, latent_dim)`.

Side effects:

- writes `adata.obsm["velocity_latent"]`: `(n_obs, latent_dim)`;
- if `adata.varm["PCs"]` is compatible, writes `adata.layers["velocity"]`:
  `(n_obs, n_vars)` from `velocity_latent @ PCs[:, :latent_dim].T`.

Artifacts:

- `velocity_latent.npy`: `(n_obs, latent_dim)`;
- `velocity_gene.npy`: `(n_obs, n_vars)`, only when PCA projection succeeds;
- `velocity_summary.json`.

Interpretation:

- latent velocity supports local flow diagnostics and latent driver analysis;
- gene velocity is PCA-projected and should be described as projected
  gene-space velocity, not directly measured RNA velocity.

### 3.2 `build_velocity_graph_bundle`

Source:

- `CytoBridge-main/CytoBridge/tl/downstream/velocity.py`

Signature:

```python
build_velocity_graph_bundle(
    adata,
    output_dir=None,
    n_pcs=50,
    n_neighbors=30,
    n_jobs=None,
    reuse_neighbors=True,
    vkey="velocity",
    preferred_basis=None,
    strict_preferred=False,
)
```

Inputs:

- requires `X_latent`, `time_point_processed`, and `velocity_latent`;
- `velocity_latent`: `(n_obs, latent_dim)`;
- `preferred_basis`: optional visualization basis such as `umap`, `pca`,
  `latent`, or `fast`.

Computation:

- builds a lightweight AnnData object for scVelo graph construction;
- reuses compatible neighbors when possible, otherwise recomputes neighbors;
- runs `scv.tl.velocity_graph` and `scv.tl.velocity_embedding`.

Side effects:

- writes graph objects to `adata.uns[f"{vkey}_graph"]`,
  `adata.uns[f"{vkey}_graph_neg"]`, and `adata.uns[f"{vkey}_params"]`;
- writes matching basis coordinates to `adata.obsm[f"X_{basis}"]`;
- writes embedded velocity vectors to `adata.obsm[f"{vkey}_{basis}"]`.

Artifacts:

- `velocity_graph_summary.json`.

Payload fields:

- `basis`: basis selected for stream plotting;
- `representation`: representation used internally.

Interpretation:

- prepares stream and local flow diagnostics;
- does not simulate continuous model trajectories.

### 3.3 `summarize_velocity_drivers_bundle`

Source:

- `CytoBridge-main/CytoBridge/tl/downstream/velocity.py`

Signature:

```python
summarize_velocity_drivers_bundle(
    adata,
    output_dir=None,
    analysis_space="gene",
    top_n=20,
)
```

Inputs:

- requires `X_latent`, `time_point_processed`, and `velocity_latent`;
- `velocity_latent`: `(n_obs, latent_dim)`;
- `analysis_space`: `"gene"`, `"latent"`, or `"auto"`.

Computation:

- latent mode ranks latent dimensions by mean absolute velocity;
- gene mode streams PCA projection `velocity_latent @ PCs.T` and ranks genes by
  mean absolute projected velocity.

Output rows:

```python
{
    "rank": int,
    "feature_index": int,
    "feature_name": str,
    "mean_velocity": float,
    "mean_abs_velocity": float,
}
```

Artifacts:

- `velocity_driver_scores.csv`.

Shapes:

- latent score vector: `(latent_dim,)`;
- gene score vector: `(n_vars,)` when PCA projection succeeds.

Interpretation:

- velocity drivers rank features associated with large model-predicted local
  velocity magnitude;
- they do not prove causal regulation without perturbation or additional
  validation.

### 3.4 `summarize_velocity_jacobian_drivers_bundle`

Source:

- `CytoBridge-main/CytoBridge/tl/downstream/velocity.py`

Signature:

```python
summarize_velocity_jacobian_drivers_bundle(
    adata,
    model=None,
    output_dir=None,
    analysis_space="gene",
    top_n=20,
    n_cells=512,
    random_state=0,
    device="cuda",
)
```

Inputs:

- requires `X_latent` and `time_point_processed`;
- samples up to `n_cells` cells across time;
- optional `model`; if `None`, loaded from `adata`.

Internal tensor contract:

- sampled latent states `z`: `(n_sampled, latent_dim)`;
- sampled times `tt`: `(n_sampled, 1)`;
- net input: `(n_sampled, latent_dim + 1)`;
- velocity output: `(n_sampled, latent_dim)`;
- mean latent Jacobian: `(latent_dim, latent_dim)`.

Gene projection:

- if `PCs` is available, computes `W @ jacobian_latent`;
- projected Jacobian matrix shape: `(n_vars, latent_dim)`;
- gene sensitivity vector shape: `(n_vars,)`.

Artifacts:

- `velocity_jacobian_driver_scores.csv`;
- `velocity_jacobian_mean.npy`: `(latent_dim, latent_dim)`;
- `velocity_jacobian_sampled_cells.csv`;
- `velocity_jacobian_driver_summary.json`.

Output rows include:

- `rank`;
- `feature_index`;
- `feature_name`;
- `jacobian_sensitivity`;
- `mean_abs_velocity`.

Interpretation:

- Jacobian drivers rank features by local sensitivity of the learned velocity
  field;
- report sampled-cell count and projection mode.

### 3.5 `generate_ode_trajectory_bundle`

Source:

- `CytoBridge-main/CytoBridge/tl/downstream/trajectory.py`

Signature:

```python
generate_ode_trajectory_bundle(
    model,
    adata,
    output_dir,
    n_trajectories=50,
    n_bins=100,
    device="cuda",
    split_true=False,
)
```

Inputs:

- trained `model`;
- `adata.obsm["X_latent"]`: `(n_obs, latent_dim)`;
- `adata.obs["time_point_processed"]`: `(n_obs,)`;
- `n_trajectories`: number of initial states sampled deterministically from
  the earliest processed time;
- `n_bins`: number of dense output intervals between first and final processed
  time; exact observed times are always inserted into the grid.

Computation:

- uses the same `CytoBridge.tl.analysis.simulate_trajectory` kernel used by
  `TrainingPipeline.evaluate`;
- sets `sigma=0.0`, so this is deterministic ODE-style rollout;
- records a dense t0-to-final trajectory and observed-time slice indices;
- ignores `split_true` because evaluation-compatible downstream trajectories
  use the standard rollout kernel.

Artifacts:

- `ode_point.npy`;
- `ode_traj.npy`.
- `ode_point_lnw.npy`;
- `ode_traj_lnw.npy`;
- `trajectory_weights.npy`;
- `trajectory_time_points.npy`;
- `observed_time_indices.npy`;
- `evaluation_trajectory.npz`.

Payload fields:

- `point_shape`: shape of `ode_point.npy`;
- `traj_shape`: shape of `ode_traj.npy`.
- `weight_shape`: shape of `trajectory_weights.npy`;
- `trajectory_time_points`: dense generated time grid;
- `observed_time_points`: observed processed times included in the grid;
- `observed_time_indices`: indices into the dense grid.

Shape note:

- `ode_traj.npy`: `(n_time_grid, n_trajectories_used, latent_dim)`;
- `ode_point.npy`: `(n_observed_times, n_trajectories_used, latent_dim)`;
- `trajectory_weights.npy`: `(n_time_grid, n_trajectories_used)`;
- `evaluation_trajectory.npz` uses the same schema as saved
  final-regression/campaign evaluation trajectories.

Interpretation:

- this is model-native continuous dynamics evidence;
- use it for trajectory/fate analysis only after defining source cells, time
  span, and fate/readout contract.

### 3.6 `generate_sde_trajectory_bundle`

Source:

- `CytoBridge-main/CytoBridge/tl/downstream/trajectory.py`

Signature:

```python
generate_sde_trajectory_bundle(
    adata,
    output_dir,
    n_time_steps=100,
    sample_traj_num=1000,
    init_time=0,
    device="cuda",
    split_true=False,
)
```

Inputs:

- `adata.obsm["X_latent"]`: `(n_obs, latent_dim)`;
- `adata.obs["time_point_processed"]`: `(n_obs,)`;
- rollout settings: `n_time_steps`, `sample_traj_num`, and `init_time`.
  Diffusion `sigma` is resolved from the fitted model config and is not a
  caller-controlled downstream parameter.

Computation:

- loads the trained model from `adata`;
- uses the same evaluation-compatible `simulate_trajectory` kernel as
  `TrainingPipeline.evaluate`;
- reads `sigma` from `adata.uns["all_model"]["resolved_config_yaml"]`;
- saves both standard evaluation-compatible artifacts and legacy SDE artifact
  names for plotting compatibility.

Artifacts:

- `sde_trajec.npy`;
- `sde_point.npy`;
- `sde_weight.npy`.
- `evaluation_trajectory.npz`;
- `trajectory_time_points.npy`;
- `observed_time_indices.npy`.

Payload fields:

- `traj_shape`;
- `point_shape`;
- `weight_shape`.

Shape note:

- `sde_trajec.npy`: `(n_time_grid, n_particles, latent_dim)`;
- `sde_point.npy`: `(n_observed_times, n_particles, latent_dim)`;
- `sde_weight.npy`: `(n_time_grid, n_particles)`;
- inspect returned shape fields and source before any custom reshaping.

Interpretation:

- SDE trajectories support stochastic trajectory exploration;
- do not mix SDE and ODE evidence without naming which one supports each
  claim.

### 3.7 `summarize_growth_bundle`

Source:

- `CytoBridge-main/CytoBridge/tl/downstream/growth.py`

Signature:

```python
summarize_growth_bundle(adata, key="growth_rate", output_dir=None)
```

Inputs:

- requires `X_latent` and `time_point_processed`;
- reads growth vector from `adata.obsm[key]` first, then `adata.obs[key]`;
- growth vector is reshaped to `(n_obs,)`.

Computation:

- computes empirical cell counts by processed time;
- `mass_cv` is coefficient of variation of empirical counts;
- if growth values are present, computes mean, std, min, max, positive count,
  negative count, and finite count.

Artifacts:

- `growth_summary.json`.

Interpretation:

- empirical cell counts describe observed sampling/population structure;
- model growth rate describes `d/dt log w`;
- balanced models do not provide mechanistic mass recovery unless the algorithm
  explicitly models growth or mass.

### 3.8 `summarize_growth_drivers_bundle`

Source:

- `CytoBridge-main/CytoBridge/tl/downstream/growth.py`

Signature:

```python
summarize_growth_drivers_bundle(
    adata,
    output_dir=None,
    analysis_space="auto",
    top_n=20,
    max_cells=20000,
    batch_size=2048,
    random_state=0,
    device="cuda",
)
```

Inputs:

- requires `X_latent` and `time_point_processed`;
- loads model from `adata`;
- requires `model.growth_net`;
- samples at most `max_cells` cells across time.
- `analysis_space`: `"auto"`, `"gene"`, or `"latent"`; `"auto"` preserves the
  historical behavior by using gene projection when PCA loadings are available
  and latent drivers otherwise.

Internal tensor contract:

- sampled latent states: `(n_sampled, latent_dim)`;
- sampled times: `(n_sampled, 1)`;
- net input per batch: `(batch, latent_dim + 1)`;
- growth output per batch: `(batch,)`;
- gradient of growth wrt latent state: `(batch, latent_dim)`.

Gene projection:

- if `PCs` is compatible, projects latent gradients into gene space;
- projected gradient batch shape: `(batch, n_vars)`;
- gene driver score vector: `(n_vars,)`.

Artifacts:

- `growth_driver_scores.csv`;
- `growth_driver_sampled_cells.csv`.

Output rows:

```python
{
    "rank": int,
    "feature_index": int,
    "feature_name": str,
    "driver_score": float,
}
```

Interpretation:

- growth drivers rank features whose latent directions most affect the learned
  growth field;
- report whether the output is latent-space or PCA-projected gene-space.

### 3.9 `summarize_growth_jacobian_drivers_bundle`

Source:

- `CytoBridge-main/CytoBridge/tl/downstream/growth.py`

Signature:

```python
summarize_growth_jacobian_drivers_bundle(...)
```

Behavior:

- wraps `summarize_growth_drivers_bundle`;
- adds Jacobian-oriented names:
  `jacobian_sensitivity`, `auxiliary_metric`, and
  `auxiliary_metric_name="mean_abs_growth_gradient"`.

Artifacts:

- `growth_jacobian_driver_scores.csv`;
- `growth_jacobian_driver_summary.json`.

Interpretation:

- use this name when the paper/report frames growth drivers as local
  sensitivity analysis.

### 3.10 `analyze_grn_bundle`

Source:

- `CytoBridge-main/CytoBridge/tl/downstream/grn.py`

Signature:

```python
analyze_grn_bundle(
    adata,
    output_dir=None,
    max_genes=10,
    max_time_points=5,
    genes=None,
    device="cuda",
)
```

Inputs:

- requires `X_latent` and `time_point_processed`;
- loads model from `adata`;
- requires `model.velocity_net`;
- optional `genes` list restricts selected gene features when PCA loadings are
  available.

Feature selection:

- latent mode without `PCs`: selects up to `max_genes` latent dimensions;
- gene mode with `PCs`: selects requested genes, highly variable genes, or the
  first `max_genes` genes.

Internal tensor contract:

- latent states: `(n_obs, latent_dim)`;
- for each selected time, samples up to 128 cells;
- latent Jacobian: `(latent_dim, latent_dim)`;
- gene-mode matrix: `W_selected @ jacobian_latent @ W_selected.T`, shape
  `(n_selected_features, n_selected_features)`;
- latent-mode matrix: `(n_selected_features, n_selected_features)`.

Artifacts:

- `grn_data.json`.

Payload fields:

- `features`: selected feature names;
- `grn`: mapping from time keys to matrices and time metadata.

Interpretation:

- this is a local velocity-Jacobian regulatory proxy, not direct experimental
  TF binding evidence;
- report selected features, sampled times, and whether the analysis is latent
  or PCA-projected gene space.

### 3.11 `estimate_velocity_jacobian_bundle`

Source:

- `CytoBridge-main/CytoBridge/tl/downstream/grn.py`

Signature:

```python
estimate_velocity_jacobian_bundle(adata, model=None, n_cells=128, device="cuda")
```

Inputs:

- requires `X_latent` and `time_point_processed`;
- samples `min(n_cells, n_obs)` cells.

Output:

- `jacobian`: latent mean Jacobian as nested list, shape
  `(latent_dim, latent_dim)`;
- row L2 norms of the Jacobian, length `(latent_dim,)`;
- sampled-cell metadata.

Interpretation:

- use for model-level sensitivity diagnostics;
- not a substitute for gene-level GRN unless projection and selected features
  are explicitly defined.

## 4. Plot APIs: `CytoBridge.pl.downstream`

Import examples:

```python
from CytoBridge import pl
from CytoBridge.pl import downstream as dsp
```

### 4.1 `plot_velocity_stream_bundle`

Source:

- `CytoBridge-main/CytoBridge/pl/downstream/velocity_plot.py`

Signature:

```python
plot_velocity_stream_bundle(
    adata,
    model=None,
    output_dir=".",
    dim_reduction="none",
    device="cuda",
    color_key=None,
    vkey="velocity",
)
```

Inputs:

- requires `adata.uns[f"{vkey}_graph"]`;
- requires a matched coordinate/velocity basis:
  `adata.obsm[f"X_{basis}"]` and `adata.obsm[f"{vkey}_{basis}"]`;
- normally call `build_velocity_graph_bundle(...)` first.

Artifacts:

- `velocity_stream_{basis}.png`.

Payload fields:

- `basis`;
- `plot_mode: "stream"`;
- `fallback_used: False`;
- `missing_requirements: []`.

Failure behavior:

- this public API is strict and raises when graph or basis requirements are
  missing. It does not silently generate a fallback scientific plot.

Interpretation:

- stream plots are embedding-space visual diagnostics;
- use trajectory bundles for continuous path claims.

### 4.2 `plot_growth_bundle`

Source:

- `CytoBridge-main/CytoBridge/pl/downstream/growth_plot.py`

Signature:

```python
plot_growth_bundle(adata, output_dir, key="growth_rate")
```

Inputs:

- reads `adata.obsm[key]` or `adata.obs[key]`;
- growth vector is reshapeable to `(n_obs,)`.

Artifacts:

- `growth_distribution.png`.

Return:

- artifact path on success;
- warning payload if growth key is absent.

### 4.3 `plot_driver_scores_bundle`

Source:

- `CytoBridge-main/CytoBridge/pl/downstream/growth_plot.py`

Signature:

```python
plot_driver_scores_bundle(
    rows,
    output_dir,
    file_name,
    value_key,
    title,
    top_n=20,
)
```

Inputs:

- `rows`: sequence of dictionaries, usually from a driver bundle;
- each row should contain `feature_name` or `feature`;
- each row should contain numeric `value_key`.

Artifacts:

- `<file_name>` under `output_dir`.

Interpretation:

- plotting does not validate whether the driver metric itself is meaningful;
- validate the upstream compute bundle and projection mode first.

### 4.4 `plot_grn_bundle`

Source:

- `CytoBridge-main/CytoBridge/pl/downstream/grn_plot.py`

Signature:

```python
plot_grn_bundle(matrix, output_dir, file_name="grn_heatmap.png")
```

Inputs:

- `matrix`: 2D array-like object, shape `(n_features, n_features)`.

Artifacts:

- `grn_heatmap.png` or the requested `file_name`.

Return:

- warning payload if the matrix is not valid 2D.

### 4.5 `plot_grn_timeseries_bundle`

Source:

- `CytoBridge-main/CytoBridge/pl/downstream/grn_plot.py`

Signature:

```python
plot_grn_timeseries_bundle(grn_payload, output_dir, prefix="grn")
```

Inputs:

- `grn_payload`: the `grn` mapping returned by `analyze_grn_bundle(...)`;
- each item should contain a 2D `matrix`.

Artifacts:

- one heatmap per time key.

Interpretation:

- use with the `features` and time metadata from `analyze_grn_bundle(...)`;
- do not detach heatmaps from the feature-selection contract.

### 4.6 `plot_ode_trajectories_bundle`

Source:

- `CytoBridge-main/CytoBridge/pl/downstream/trajectory_plot.py`

Signature:

```python
plot_ode_trajectories_bundle(
    adata,
    output_dir,
    color_key=None,
    model=None,
    device="cuda",
    n_trajectories=50,
    n_bins=None,
)
```

Behavior:

- loads the trained model from `adata` when `model=None`;
- calls package-native `plot_ode_trajectories(...)`;
- generates deterministic/evaluation-kernel trajectory arrays and a trajectory
  ensemble figure;
- writes `ODE_Trajectories_Plot.png` plus trajectory arrays such as
  `ode_traj.npy`, `ode_point.npy`, `trajectory_time_points.npy`, and
  `evaluation_trajectory.npz` when produced by the underlying API.

Interpretation:

- use this as the preferred trajectory visualization helper for deterministic
  or ODE-style rollouts;
- the figure should show generated path ensembles/time-sliced generated states,
  not only a centroid or mean trajectory.

### 4.7 `plot_sde_trajectories_bundle`

Source:

- `CytoBridge-main/CytoBridge/pl/downstream/trajectory_plot.py`

Signature:

```python
plot_sde_trajectories_bundle(adata, output_dir, color_key=None)
```

Behavior:

- calls legacy `CytoBridge.pl.plot.plot_sde_trajectories`.

Artifacts:

- `SDE_Trajectories_Plot.png` if the legacy helper writes it.

Interpretation:

- use as a trajectory visualization helper;
- verify upstream SDE trajectory generation and shape fields before making
  claims.

## 5. Agent Toolkit Boundary

Source:

- `cytobridge_agent/tools/downstream_analysis_toolkit.py`

Use the toolkit when the main agent needs guarded execution, subprocess
isolation, or workflow-state integration. Use package APIs directly inside
saved downstream scripts when reproducibility and inspectable scientific logic
are more important than UI convenience.

Scripts should record:

- package API name and source path;
- input `adata` path and model path;
- required fields checked;
- returned payload including shape fields and warnings;
- every generated table, figure, and array path.
