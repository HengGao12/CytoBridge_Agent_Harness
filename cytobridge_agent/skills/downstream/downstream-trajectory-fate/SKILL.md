---
name: downstream-trajectory-fate
description: Continuous trajectory, fate, and flow analysis for CytoBridge neural dynamics models.
---

# Downstream Trajectory And Fate

Use this skill for continuous-time trajectory, fate, rollout, and flow-structure
analysis after model training.

## Core Rule

Continuous paths must come from the trained model's rollout/evaluation API or a
locked trajectory artifact from the same model and data. Local velocity fields
and stream plots are diagnostics, not substitutes for rollout-derived
trajectory evidence.

## Analysis Logic

CytoBridge is useful here because the trained model is generative: it can roll
source cells forward through continuous latent states. Fate analysis is a second
layer on top of those generated paths. First generate trajectories from the
model; then classify or annotate each generated state with a valid cell-type,
fate, lineage, or endpoint classifier. Fate probabilities, branch timing, and
transition matrices should be computed from the rollout-derived sequence of
states and weights.

Do not infer fate dynamics from static endpoint labels alone. Do not draw
connected paths from local velocity interpolation and call them model
trajectories.
For per-cell fate predictions, preserve source-cell provenance throughout the
analysis. The key for a root-cell fate table must be the real AnnData
`obs_names` value, not a synthesized position such as `cell_0` unless that is
the actual `obs_name`. When a trajectory API samples or filters source cells,
the only safe mapping is the returned trajectory provenance, especially
`traj["init_obs_names"]` and `traj["init_indices"]`. Do not write per-cell
results by zipping predicted rows with a separately reconstructed root-cell list
unless you have verified the order is identical.

This skill often combines with state-structure, lineage-transition, driver-gene,
or perturbation analysis. Use the trajectory artifact as the shared evidence
object, then add readouts that answer the biological question.

## Read First

- `CytoBridge-main/docs/runtime/downstream/model-semantics.md`
- `CytoBridge-main/docs/runtime/downstream/semantics-and-evidence.md`
- `CytoBridge-main/docs/runtime/downstream/recipes-and-artifacts.md`
- source when details matter:
  `CytoBridge-main/CytoBridge/tl/perturbation.py` and
  `CytoBridge-main/CytoBridge/tl/downstream/trajectory.py`

Before generating trajectories, identify the active model family/components:
velocity-only, velocity-growth, score/stochastic, interaction-aware, or custom.
Use deterministic ODE-style rollout for velocity-only or velocity-growth mean
dynamics. Use SDE-style rollout only when the model has a stochastic/score
component and the claim depends on stochastic path uncertainty.
Do not add ad-hoc diffusion noise to a deterministic velocity-growth model.
If the resolved training config says `sigma: 0.0`, downstream fate and endpoint
rollouts must remain deterministic unless a separate stochastic model component
is present and explicitly used by the package API.

## Recommended Outputs

- trajectory arrays or trajectory dataset
- fate probability or endpoint table when labels/classifier are defined
- source-cell provenance table with `init_indices`, `init_obs_names`,
  `init_times`, `sampling_seed`, and `sampling_policy`
- trajectory figures from package plotting helpers when available
- flow/stream diagnostics as companion panels, not trajectory substitutes
- transition/fate figures
- manifest with model path, source population, integration times, resolved
  rollout method source, resolved sigma source, and warnings

## API Priority

1. Existing locked final-regression/evaluation trajectory artifact.
2. `generate_trajectory_dataset(...)` from the downstream toolkit, which uses
   the evaluation-compatible rollout kernel.
   This API returns and saves `init_indices`, `init_obs_names`, `init_times`,
   `sampling_seed`, and `sampling_policy`; downstream fate tables should join
   against those fields instead of assuming output row order equals original
   cell IDs. For per-cell JSON or CSV outputs, build keys from
   `traj["init_obs_names"]` directly:
   `for obs_name, pred in zip(traj["init_obs_names"], predictions): ...`.
3. Package `tl.downstream.generate_ode_trajectory_bundle(...)` or
   `generate_sde_trajectory_bundle(...)`.
4. Package plotting helpers before custom trajectory plotting:
   `CytoBridge.pl.downstream.plot_ode_trajectories_bundle(...)` for
   deterministic/evaluation-kernel rollouts, and
   `CytoBridge.pl.downstream.plot_sde_trajectories_bundle(...)` when an SDE or
   SDE-compatible trajectory artifact exists. See
   `CytoBridge-main/docs/runtime/downstream/api-reference.md` and
   `CytoBridge-main/CytoBridge/pl/downstream/trajectory_plot.py` for the
   current contract.
5. Velocity bundles and stream plots for diagnostics.

## Trajectory Figure Rule

Do not reduce generated trajectories to only the mean path, centroid path, or
timepoint-average dynamics. Mean dynamics can be a small companion summary, but
the main trajectory visualization should show the model-generated path ensemble
or time-sliced positions: sampled trajectories, endpoint clouds, fate-colored
particles, uncertainty bands, or KDE contours on a shared embedding. Otherwise
branching, heterogeneity, and uncertainty disappear. If a manuscript figure has
only a single averaged trajectory line as the main dynamics panel, treat it as
unfinished unless the claim is explicitly about the mean.

When the package plotting helper fits the artifact, use it first:

```python
from CytoBridge.pl import downstream as dsp

plot_payload = dsp.plot_ode_trajectories_bundle(
    adata=adata,
    output_dir=str(out / "figures" / "trajectory"),
    color_key=label_key,
)

# If the model/artifact is genuinely stochastic or SDE-compatible:
plot_payload = dsp.plot_sde_trajectories_bundle(
    adata=adata,
    output_dir=str(out / "figures" / "trajectory"),
    color_key=label_key,
)
```

If the active artifact is ODE/evaluation-only and no package plot helper matches
it directly, write a custom manuscript figure from the saved trajectory arrays:
show multiple sampled trajectories or time slices, not only an averaged line.
Then read `workflow/scientific-visualization/SKILL.md` before final rendering.

When using `generate_trajectory_dataset(...)`, do not pass `method` or `sigma`.
Those are training/model semantics and are read from the fitted model's
resolved config and model components. You may set `n_steps` only to adjust
output sampling density; the metadata records whether it came from resolved
config or caller input.

If you find yourself writing a manual Euler/SDE loop with a line like
`x = x + v * dt + sigma * noise`, stop and switch back to the package-native
trajectory API or a locked evaluation trajectory. Manual integration is only
acceptable after the package-native route fails with a concrete error, and then
the rollout semantics (`sigma`, SDE/ODE, score usage, growth weights) must still
be read from the resolved config and model components rather than invented.

## Minimal Template

```python
from pathlib import Path
import scanpy as sc
from CytoBridge.utils import load_model_from_adata
from CytoBridge.tl.perturbation import generate_trajectory_dataset

adata = sc.read_h5ad(input_h5ad)
model = load_model_from_adata(adata)
out = Path(output_dir) / "downstream" / "trajectory_fate"
out.mkdir(parents=True, exist_ok=True)

traj = generate_trajectory_dataset(
    adata=adata,
    model=model,
    n_cells=500,
    n_steps=100,  # output sampling density only
    time_key="time_point_processed",
    cell_type_key=label_key,
    output_space="both",
    save_dir=str(out),
    device=device,
)
print(traj["metadata"])
```

## Hard Stops

- no trained model or matching trajectory artifact
- no source population definition for per-cell/fate claims
- no fate label/classifier contract for fate probabilities
- missing source-cell provenance for per-cell fate tables
- output trajectories cannot be regenerated from saved script
