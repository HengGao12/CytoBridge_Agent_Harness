---
title: "Runtime compatibility and failure modes"
summary: "Compatibility APIs, projection contracts, and known runtime failure modes."
read_when:
  - "Diagnosing runtime compatibility issues"
  - "Understanding projection or failure-mode behavior"
---
# Compatibility, Projection, And Failure Modes

Compatibility APIs, projection contract, and known runtime failure modes.

## 8. Compatibility Layer APIs

These wrappers keep old import paths stable.

- Compute wrappers in `CytoBridge/tl/analysis.py`:
  - `compute_velocity_bundle`
  - `build_velocity_graph_bundle`
  - `summarize_velocity_drivers_bundle`
  - `summarize_growth_drivers_bundle`
  - `analyze_grn_bundle`
  - `generate_ode_trajectory_bundle`
  - `generate_sde_trajectory_bundle`
- Plot wrappers in `CytoBridge/pl/plot.py`:
  - `plot_velocity_stream_bundle`
  - `plot_ode_trajectories_bundle`
  - `plot_sde_trajectories_bundle`

Contract:
- Same input/output schema as the underlying `tl.downstream` / `pl.downstream` implementations.

## 9. Projection Contract (Important)

- Package-level guaranteed projection: latent -> gene via PCA loadings only.
- If `adata.varm["PCs"]` is unavailable/incompatible:
  - APIs return latent-space results.
  - `projection_backend="none"` and warning message are expected.
- Non-PCA projection (e.g., decoder-based) is not implemented in package APIs.

## 10. Known Runtime Failure Modes

- Missing strict keys:
  - `time_point_processed`, `X_latent`, `velocity_latent`, etc.
- Missing model components:
  - no `growth_net` or `velocity_net`.
- `scvelo` not installed:
  - velocity graph / stream APIs return warning payload.
- Shape mismatch in PCA loadings:
  - fallback to latent output.
