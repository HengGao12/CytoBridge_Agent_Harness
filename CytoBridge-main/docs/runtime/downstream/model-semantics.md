---
title: "Downstream model semantics by builtin family"
summary: "How to interpret trained CytoBridge builtin models during downstream analysis, including components, mass semantics, and rollout choice."
read_when:
  - "Choosing ODE, SDE, growth, or perturbation downstream inference"
  - "Interpreting what a trained builtin model learned"
  - "Writing biological claims from model rollouts"
---
# Downstream Model Semantics By Builtin Family

Downstream analysis should start from the meaning of the selected model. A
trained CytoBridge model is not just an embedding: it is a neural continuous
dynamics model whose components determine which downstream claims are valid.

For deeper theory, read `docs/theory/README.md`. For implementation source
paths, read `docs/runtime/flow-matching/semantics-and-callgraph.md` and
`docs/runtime/flow-matching/builtin-implementation-guide.md`.

## 1. Shared Model Objects

Most trained models expose some subset of:

- `velocity_net`: deterministic drift `v(t, x)` in latent state space;
- `growth_net`: growth/death field `g(t, x) = d/dt log w`;
- score component: stochastic correction/score used by stochastic bridge
  methods;
- interaction component: mean-field or interaction force used by
  interaction-aware simulation methods.

The standard evaluation/downstream trajectory object is a dense generated
trajectory from the earliest observed time to the final time:

```text
points_by_time[t]   : (n_particles_t, latent_dim)
weights_by_time[t]  : (n_particles_t,)
time_points         : dense generated time grid
observed_time_indices: indices matching observed snapshot times
```

The package uses the same rollout kernel for evaluation-compatible downstream
trajectories and campaign/final-regression metrics. Prefer this generated
trajectory or a locked final-regression `EvaluationTrajectory` for downstream
trajectory, fate, prediction, perturbation, or mass claims.

## 2. Builtin Semantics Quick Table

| Builtin | Components | Balanced? | Recommended rollout | Mass/growth interpretation | Good downstream uses |
| --- | --- | --- | --- | --- | --- |
| `balanced_ot_cfm` | velocity | balanced | ODE / deterministic rollout | no learned growth; TMV diagnostic only | state trajectory, endpoint distribution, fate geometry |
| `dynamical_ot` | velocity | balanced | ODE / deterministic rollout | no learned growth; TMV diagnostic only | neural ODE trajectory and interpolation |
| `sf2m` | velocity + score | balanced stochastic | SDE when stochastic bridge behavior matters; ODE only as drift diagnostic | no learned growth; TMV diagnostic only | stochastic fate uncertainty, bridge-like path variability |
| `vgfm` | velocity + growth | unbalanced | ODE / deterministic weighted rollout | growth weights are mechanistic positive-measure evidence | mass dynamics, proliferation/depletion, weighted fate shifts |
| `wfrfm` | velocity + growth | unbalanced | ODE / deterministic weighted rollout | WFR/OET mass semantics; delta is dataset-scale sensitive and defaults to auto | WFR-style transport-growth interpretation |
| `unbalanced_ot` | velocity + growth | unbalanced | ODE / deterministic weighted rollout | growth weights approximate positive-measure endpoint recovery | mass-aware state dynamics |
| `crufm` | velocity + growth + score | unbalanced stochastic | SDE for stochastic dynamics; ODE only for deterministic drift diagnostics | growth weights plus stochastic correction | stochastic unbalanced trajectories and fate uncertainty |
| `ruot` | velocity + growth + score | unbalanced stochastic | SDE for stochastic dynamics; ODE only for deterministic drift diagnostics | growth weights plus stochastic unbalanced dynamics | stochastic mass-aware transition analysis |
| `cyto_simulation` | velocity + growth + score + interaction | unbalanced stochastic interaction | SDE/interaction-aware rollout | growth and interaction both affect generated trajectories | interaction-aware population dynamics |

If the active model is custom, read its `PROPOSAL.md`, `IMPLEMENTATION_MAP.md`,
resolved config, and final-regression metadata. Then map it onto the nearest
row above: velocity-only, velocity-growth, stochastic, interaction-aware, or a
custom simulation hook.

## 3. ODE Versus SDE In Downstream Analysis

Use ODE-style deterministic rollout when:

- the model is velocity-only or velocity-growth;
- the biological question is mean trajectory, endpoint structure, or weighted
  mass evolution;
- the selected method is `balanced_ot_cfm`, `dynamical_ot`, `vgfm`, `wfrfm`, or
  `unbalanced_ot`.

In the current package kernel, ODE-style rollout means zero diffusion
(`sigma=0`) through the same evaluation path. If a model contains a score
component, that component can still enter the drift according to the package
simulation source. Inspect `CytoBridge-main/CytoBridge/tl/analysis.py` before
making a score-specific claim.

Use SDE-style stochastic rollout when:

- the model has a score/stochastic component and the claim depends on path
  uncertainty, branch probabilities, or stochastic bridge behavior;
- the selected method is `sf2m`, `crufm`, `ruot`, or `cyto_simulation`;
- the report explicitly separates stochastic variability from deterministic
  drift.

Do not call an SDE rollout just because it makes a plot look smoother. Use it
only when the model component and biological question justify stochastic
inference.

## 4. Balanced Versus Unbalanced Claims

Balanced methods conserve probability mass. Valid claims:

- shape of the generated endpoint distribution;
- continuous state trajectory;
- branch geometry or fate probabilities under a fixed particle population.

Invalid or weak claims for balanced-only methods:

- mechanistic proliferation/death;
- total cell-count recovery;
- TMV as evidence of mass modeling.

Unbalanced methods carry particle weights. Valid claims, if artifacts support
them:

- total predicted mass change over time;
- growth/depletion along trajectories;
- weighted fate shifts;
- growth/gene drivers of `d/dt log w`.

Always report whether weights are uniform, learned growth weights, empirical
counts, or externally supplied weights.

## 5. Downstream Inference Rule

Before choosing a downstream API, write down:

```text
selected model:
resolved config:
components: velocity / growth / score / interaction
balanced or unbalanced:
rollout mode: ODE / SDE / custom simulation hook
valid mass interpretation:
valid fate/readout contract:
```

If the model's meaning is unclear, inspect:

1. `docs/theory/README.md`;
2. `docs/runtime/flow-matching/semantics-and-callgraph.md`;
3. `docs/runtime/flow-matching/builtin-implementation-guide.md`;
4. the active run's resolved config and final-regression artifact;
5. package source under `CytoBridge-main/CytoBridge/tl/`.

Do not proceed by copying an old downstream script from another model family
without checking this semantic contract.

## 6. Resolved Config Rule

Downstream trajectory generation should use the fitted model's resolved
training config by default. The active fitted AnnData stores this under:

```python
adata.uns["all_model"]["resolved_config_yaml"]
```

The downstream trajectory dataset API reads that config automatically.
`method` and `sigma` are not agent-facing controls: they are derived from the
fitted model components and resolved config. `n_steps` is the only common
rollout argument an agent may change, and only to adjust output sampling
density.

```python
from CytoBridge.tl.perturbation import generate_trajectory_dataset

result = generate_trajectory_dataset(
    adata=adata,
    model=model,
    n_steps=None,  # use evaluation.trajectory_step when present
)
```

The returned metadata records where each rollout value came from:

- `rollout_config_source`;
- `rollout_method_source`;
- `rollout_sigma_source`;
- `rollout_n_steps_source`;
- `rollout_integration_dt_source`.

This rule prevents downstream reports from accidentally using a trajectory
parameter that differs from campaign/final-regression evaluation.
