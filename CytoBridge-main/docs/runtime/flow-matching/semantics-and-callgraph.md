---
title: "Flow matching semantics and callgraph"
summary: "Default flow-matching runtime semantics and source escalation order."
read_when:
  - "Understanding flow-matching call flow"
  - "Checking default backend semantics"
---
# Flow-Matching Semantics And Call Graph

Default backend semantics, mental model, call graph, and AnnData ownership boundary.

## 2. Default Builtin FM Semantics

Before changing anything, keep the builtin behavior straight.

For the high-level mathematical contract of each builtin family, read
`docs/theory/README.md` first. This file is the lower-level
runtime contract for implementing or extending those semantics.

### 2.1 Default backend composition

The default flow-matching backend is:

- `UnbalancedOTCouplingStrategy`
- `RegularizedUnbalancedConditionalPath`
- `UOTMassStrategy`

Named builtin configs may override this composition explicitly:

- `balanced_ot_cfm`: balanced OT coupling, deterministic linear path, no mass
- `sf2m`: balanced entropic OT coupling, stochastic bridge path, no mass
- `vgfm`: UOT coupling, deterministic linear path, UOT mass targets
- `wfrfm`: WFR-OET semi-coupling, traveling-Gaussian WFR path, WFR terminal mass targets
- `crufm`: UOT coupling, stochastic bridge path, UOT mass targets

### 2.2 Default UOT semantics

For each adjacent interval, the builtin backend:

- builds a latent-space pairwise cost matrix
- uses unit source and target reference masses
  - `a_i = 1`
  - `b_j = 1`
- auto-selects `reg` and `reg_m` unless explicitly overridden
- solves the adjacent UOT plan with the package cost-based Sinkhorn helper
- if `reg` and `reg_m` are configured explicitly, auto-selection is skipped
- if auto-selection is needed, the default path stays on CPU for legacy
  numerical consistency
- `auto_reg_device: cuda` can run candidate Sinkhorn solves on GPU with float64,
  but benchmark it on the target block size before making it the default

Concrete implementation detail:

- the helper eventually calls:
  - `pot.unbalanced.sinkhorn_unbalanced(a, b, M, reg, [reg_m, np.inf])`
- this means the current runtime convention is not fully symmetric unbalanced OT
- it is a source-relaxed / target-constrained semi-relaxed OT convention under
  the package's unit-mass empirical setup

Interpret the regularization parameters precisely:

- `reg`
  - entropy regularization on the transport plan
  - larger `reg` makes the plan smoother / more diffuse
- `reg_m`
  - mass-relaxation strength on the relaxed marginal
  - in the current helper, that relaxed marginal is the source side because the
    target side is passed as `np.inf`
  - larger `reg_m` keeps the source marginal closer to the reference masses
  - smaller `reg_m` permits more source-side mass creation/destruction
- `alpha_regm`
  - package-level multiplier applied after auto-tuning
  - the effective source-side mass penalty is `alpha_regm * auto_reg_m`

This means the builtin package default is not automatically identical to every
paper's OT/UOT implementation. If an external method uses a different solver,
different regularization, a different relaxed side, or a different marginal
convention, that difference must be implemented explicitly.

### 2.3 Default pair sampling semantics

Builtin pair sampling does **not** automatically construct semi-couplings such
as `gamma0` or `gamma1`.

Instead, `sample_from_ot_plan(...)` samples directly from the stored transport
plan:

- sample source row `i` with probability proportional to the row sum
- sample target column `j` conditionally within that row

So the sampled pair distribution is the normalized transport plan itself.

If an external method samples from a derived semi-coupling rather than from the
stored plan, you must implement that behavior explicitly.

### 2.4 Default mass semantics

Builtin `UOTMassStrategy` defines only the terminal mass target:

- for a sampled source particle normalized to local mass `1`, what should its
  mass be at local `t=1`?

Under the builtin unit-mass convention:

- sampled source row sums are already in the correct unit-mass convention
- so the sampled row sum is already the terminal mass ratio

Builtin `RegularizedUnbalancedConditionalPath` then defines the local mass
dynamics:

- local `gt_local = log(terminal_mass)` under `w(0)=1`
- local sample weights come from the path-level mass interpolation and are used
  as training loss weights
- the backend divides by physical interval length before regressing the growth
  head

So the learned growth field remains:

- `g(t, x) = d/dt log w`

and integrating that rate across the interval recovers the intended terminal
mass target.

Important distinction:

- the growth-rate target is a rate target
- the training loss weight is a separate optimization-time weight
- that training loss weight may depend on the sampling convention
- it should not be interpreted as the current path mass `w(t)`

### 2.5 Default loss and evaluation semantics

In the builtin trainer:

- velocity loss is weighted MSE against `ut`
- growth loss is weighted MSE against `gt`
- score loss is active only when the score head is trained

Builtin evaluation then:

- simulates forward from the earliest observed time
- computes builtin `W1`
- computes builtin `TMV`
- optionally appends `custom_metrics`

Custom algorithms may add metrics, but may not redefine builtin `W1` or `TMV`.

### 2.6 Default extension behavior

If a custom workspace returns a `TrainingAlgorithmSpec` without a given hook,
the package keeps the default for that responsibility:

- no `training_data_builder`: standard AnnData split by
  `time_point_processed` and `X_latent`
- no `flow_matching_backend_builder`: config-selected FM backend
- no `model_builder`: package `DynamicalModel`
- no `stage_runner`: package stage loop
- no `flow_matching_loss_hook`: builtin FM losses only
- no `simulation_hook`: package model rollout
- no `evaluation_metrics_hook`: builtin metrics plus registered campaign
  metric plumbing only

This matters for review: absence of a hook is an intentional default, not a
hidden implementation. If a proposal step relies on default behavior, record
that explicitly in `IMPLEMENTATION_MAP.md`.

## 3. Recommended Mental Model

For most custom algorithms, think in this order:

1. choose the coupling rule between adjacent time points
2. keep the default straight-line conditional path unless you have a clear reason to change it
3. keep the default mass strategy unless you have a clear reason to change it
4. only override lower-level state assembly if you truly need global cross-time coordination

The small/medium-data common case is:

- implement `build_pairwise_cost(...)`
- keep default `build_state(...)`
- keep default `sample_pairs(...)`

The large-data common case is different:

- first estimate the largest adjacent pair size;
- if full `n_t x n_{t+1}` cost/mask/plan arrays are not acceptable, implement
  a custom `CouplingStrategy` with true chunked/sparse/streaming state;
- do not rely on the `use_mini_batch` flag alone as the scalability argument.

Do not start from `build_state(...)` unless you need it for semantics or
memory. Do use it when the cost-based helper would force dense state that the
approved algorithm cannot afford on real benchmark data.

## 4. Current Training Call Graph

The flow-matching training path is:

```python
cb.tl.fit(...)
  -> TrainingPipeline.train(...)
    -> TrainingPipeline.run_flow_matching_stage(...)
      -> FlowMatchingBackend.prepare(...)
      -> FlowMatchingBackend.sample_batch(...)
      -> TrainingPipeline.train_flow_matching_epoch(...)
```

The package assumes:

- `adata.obs["time_point_processed"]`
- `adata.obsm["X_latent"]`

These are fixed runtime contracts:

- `time_point_processed` is the canonical processed time axis
- `X_latent` is the canonical transcriptomic representation used by the default training path
- custom algorithms may read additional modalities from `obs`, `obsm`, `layers`, or `uns`
- custom algorithms should not redefine the main time key or replace transcriptomic training with another primary key

## 4A. AnnData Management Boundary

This is the intended split:

- `training_data_builder(...)` handles raw AnnData
- `flow_matching_backend_builder(...)` handles backend composition
- `flow_matching_loss_hook(...)` handles additive loss terms or explicit
  replacement of one builtin velocity/growth/score component loss
- `evaluation_metrics_hook(...)` handles additive evaluation metrics

Practical rule:

- if you need `obs`, `obsm`, `layers`, or row alignment work, do it in
  `training_data_builder(...)`
- if you already have `TrainingDataBundle`, do not go back and treat the backend
  builder like a general-purpose adata hook

The backend builder receives:

- `build_context.training_data`

not a free-form raw adata editing surface.
