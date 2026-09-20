---
title: "Downstream recipes and artifact contracts"
summary: "Recommended analysis recipes, script layout, manifests, and report-ready artifact contracts."
read_when:
  - "Writing downstream scripts"
  - "Generating report or paper downstream figures"
  - "Combining trajectory, growth, perturbation, and driver analyses"
---
# Downstream Recipes And Artifact Contracts

Use these recipes as starting points. Adapt paths, labels, and feature spaces to
the active workflow context.

## 1. Artifact Layout

Important downstream analyses should write:

```text
<output_dir>/scripts/downstream/<analysis_name>.py
<output_dir>/downstream/<analysis_name>/manifest.json
<output_dir>/downstream/<analysis_name>/tables/
<output_dir>/downstream/<analysis_name>/figures/
<output_dir>/downstream/<analysis_name>/arrays/
```

Scripts should be self-contained: explicit input data path, model path, output
directory, labels, time keys, package APIs, and parameters.

When a recipe depends on exact array axis order, use the payload shape fields
and inspect the source path listed in `api-reference.md`. Do not infer axis
semantics from an old artifact name.

Every substantial downstream script should include a short source provenance
block near the top:

```python
"""
Downstream API provenance:
- compute_velocity_bundle: CytoBridge-main/CytoBridge/tl/downstream/velocity.py
- plot_velocity_stream_bundle: CytoBridge-main/CytoBridge/pl/downstream/velocity_plot.py

Shape contract checked:
- adata.obsm["X_latent"]: (n_obs, latent_dim)
- adata.obs["time_point_processed"]: (n_obs,)
- returned payload shape fields are written to manifest.json
"""
```

If the script consumes `.npy` arrays from trajectory APIs, read the returned
`point_shape`, `traj_shape`, or `weight_shape` from the manifest before slicing.
If the shape is not self-explanatory, inspect the source path listed in
`api-reference.md` and write the resolved axis semantics into the manifest.

## 2. Manifest Contract

Recommended manifest shape:

```json
{
  "analysis_name": "...",
  "input_data": "...",
  "model_path": "...",
  "evaluation_trajectory": null,
  "time_key": "time_point_processed",
  "label_keys": [],
  "feature_space": "latent|gene|projected_gene|mixed",
  "apis": [
    {
      "name": "compute_velocity_bundle",
      "source_path": "CytoBridge-main/CytoBridge/tl/downstream/velocity.py",
      "required_fields": {
        "adata.obsm.X_latent": "(n_obs, latent_dim)",
        "adata.obs.time_point_processed": "(n_obs,)"
      },
      "returned_shapes": {
        "velocity_latent": "(n_obs, latent_dim)"
      }
    }
  ],
  "parameters": {},
  "artifacts": {},
  "warnings": [],
  "claims_supported": []
}
```

A final paper or report should cite manifest-backed artifacts, not transient
notebook variables or unverified narrative.

## 3. Model Semantics Preflight

Before choosing a recipe, identify the model family and components using
`model-semantics.md`. The same figure can mean different things for a balanced
velocity-only model, an unbalanced velocity-growth model, and a stochastic
score-based model.

## 4. Velocity And Stream Recipe

1. Load active `adata` and model.
2. Run `compute_velocity_bundle(...)`.
3. If a stream plot is needed, run `build_velocity_graph_bundle(...)`.
4. Run `plot_velocity_stream_bundle(...)`.
5. Save velocity summary, graph summary, figure paths, and warnings.

Interpretation:

- stream plots show local flow diagnostics;
- they do not by themselves prove simulated continuous paths.

## 5. Trajectory And Fate Recipe

1. Resolve source population, time span, model path, and feature space.
2. Reuse a locked evaluation trajectory if it matches the active model/data.
3. Otherwise generate a trajectory dataset or ODE/SDE trajectory bundle.
4. Apply fate or lineage readout only if a label/classifier/lineage contract
   exists.
5. Save trajectory arrays, endpoint table, fate table, figures, and manifest.

For branching systems, report fate distributions and uncertainty, not only the
top fate.

Trajectory generation should use the evaluation-compatible rollout kernel
(`TrainingPipeline.evaluate` semantics) unless the active custom algorithm
explicitly requires a custom simulation hook. `method` and `sigma` are resolved
from the fitted model components and `resolved_config_yaml`; do not pass them
as downstream knobs. You may set `n_steps` only to change output sampling
density. Save or reuse `evaluation_trajectory.npz` when available.

## 6. Growth And Mass Recipe

1. Run `summarize_growth_bundle(...)`.
2. If driver interpretation matters, run `summarize_growth_drivers_bundle(...)`
   or a Jacobian driver API.
3. Plot growth distribution and driver scores.
4. Separate empirical cell-count changes from model growth-rate evidence.

Report whether the model is balanced or unbalanced. Balanced models do not
provide mechanistic mass recovery unless a custom mechanism explicitly adds it.

## 7. Driver Gene And GRN Recipe

1. Define the target behavior: velocity, growth, fate, perturbation, or
   transition.
2. Run the relevant driver/Jacobian API.
3. Verify gene-space projection before gene-level claims.
4. Save ranked tables, heatmaps/bar plots, orientation notes, and projection
   warnings.

Driver genes without a defined target behavior are weak. First define the
dynamic behavior, then ask which genes explain it.

## 8. Perturbation Recipe

1. Define source population and matched control.
2. Define perturbation in gene or latent space.
3. Roll out control and perturbed dynamics with the same settings.
4. Compare endpoint, fate, growth, trajectory displacement, or expression
   readouts.
5. Save perturbation design, matched artifacts, effect tables, figures, and
   manifest.

If the perturbation cannot be applied in gene space, label latent perturbation
as exploratory.

## 9. Prediction Recipe

1. Define source time, target time, and validation target.
2. Generate model-native prediction at the target time.
3. Compare only against observations intended for validation or held-out use.
4. Save prediction arrays/tables, distances, plots, and manifest.

Do not use the validation target itself as the prediction.

## 10. Integrated Biological Story Recipe

For a paper/report:

1. Start with observed and rollout state structure.
2. Add continuous trajectory/fate evidence.
3. Add growth/mass if the biology involves population size.
4. Add perturbation if the question involves causal/counterfactual effects.
5. Add drivers/GRN after the target behavior is defined.
6. Write claims from generated artifacts and label limitations explicitly.

Every final claim should map to one of:

- a table row or summary statistic;
- a figure path;
- a trajectory/evaluation artifact;
- a known dataset or original-paper fact;
- a clearly labeled hypothesis or limitation.

Before writing the final narrative, open the manifest and verify that each
claim's API source path, input shape, output shape, warning state, and artifact
path are present. Missing provenance is a reason to regenerate or document the
analysis, not a reason to fill in numbers from memory.
