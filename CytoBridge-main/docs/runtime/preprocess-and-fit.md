---
title: "Preprocess and fit workflow"
summary: "Preprocessing, fit entrypoints, and training readiness checks."
read_when:
  - "Preparing data for fitting"
  - "Checking fit-time prerequisites"
---
# Preprocess And Fit APIs

Agent-facing reference for preprocessing and fitting entrypoints.

## 5. Preprocess and Fit

### 5.1 `cb.pp.preprocess(...)`

File:
- `CytoBridge/pp/preprocess.py`

Signature in code:
```python
preprocess(
  adata,
  time_key,
  n_top_genes=2000,
  dim_reduction="pca",
  n_pcs=50,
  time_mapping=None,
  normalization=True,
  log1p=True,
  select_hvg=True,
)
```

Inputs:
- `adata`: `AnnData`.
- `time_key`: column name in `adata.obs`.
- `dim_reduction`: `"pca" | "umap" | "none"`.

Side effects on `adata`:
- Adds numeric time:
  - `adata.obs["time_point_processed"]`
- Adds latent representation:
  - `adata.obsm["X_latent"]`
- For PCA/UMAP, may set:
  - `adata.obsm["X_pca"]`, `adata.obsm["X_umap"]`, and related `uns`.

Actual return (important):
- Returns one `AnnData` object (`n_pc_adata` style reduced view).
- It does **not** return a tuple in current implementation.


### 5.2 `cb.tl.fit(...)`

File:
- `CytoBridge/tl/fit.py`

Signature in code:
```python
fit(
  adata,
  config,                # dict or config path/string
  batch_size=None,
  device="cuda",
  stage="final",
  progress_callback=None,
  training_data_builder=None,
  flow_matching_backend_builder=None,
  flow_matching_loss_hook=None,
  evaluation_metrics_hook=None,
)
```

Required input keys:
- `adata.obs["time_point_processed"]`
- `adata.obsm["X_latent"]`

Side effects on `adata`:
- Always writes:
  - `adata.obsm["velocity_latent"]`
- Conditionally writes:
  - `adata.obsm["growth_rate"]` if model has growth component
    - this stores the predicted growth rate field `g(t, x) = d/dt log w`
    - it does not store absolute particle mass
  - `adata.obsm["score_latent"]` if model has score component
- Stores model payload:
  - `adata.uns["all_model"]`
  - `adata.uns["evaluation_metrics"]`
    - builtin keys:
      - `w1_scores`
      - `tmv_scores`
    - optional additive keys:
      - `custom_metrics`

Disk artifacts:
- Writes to config `ckpt_dir`:
  - `adata.h5ad`
  - `config.yaml`

Return:
- `AnnData` (same object with updates).
- For custom algorithm entrypoints and developer hook contracts, see:
  - `docs/runtime/custom-algorithms/README.md`
  - `docs/runtime/flow-matching/README.md`
