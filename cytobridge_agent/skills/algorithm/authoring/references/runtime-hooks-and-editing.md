# Path rule

- this reference file path is relative to the skill directory `~/.cellcompass/skills/algorithm/authoring/`
- so `references/...` means a sibling under that same skill folder
- package docs and source paths in this file are workspace-relative

# Runtime Hooks And Editing

## Read Order In Detail

### Step 1: proposal

Read:

- `~/.cellcompass/training_algorithms/<algorithm_id>/PROPOSAL.md`

You should be able to state clearly:

- what scientific/modeling problem the algorithm is solving
- whether the method should be balanced or unbalanced
- whether the method should be deterministic or stochastic
- what minimal change is actually required
- what recoverability argument the proposal gives
- what pseudocode steps the proposal says the implementation must realize
- what evaluation plan the proposal gives, including which additive custom
  metrics are needed beyond builtin `W1/TMV`

If those are unclear, patch the proposal first. Do not code yet.

### Step 2: local template

Read:

- `~/.cellcompass/training_algorithms/<algorithm_id>/algorithm.py`
- `~/.cellcompass/training_algorithms/<algorithm_id>/config.yaml`
- `~/.cellcompass/training_algorithms/<algorithm_id>/manifest.yaml`
- `~/.cellcompass/training_algorithms/<algorithm_id>/IMPLEMENTATION_MAP.md`

Locate these symbols first:

- `model_builder`
- `stage_runner`
- `simulation_hook`
- `build_pairwise_cost`
- `build_flow_matching_backend`
- `training_data_builder`
- `flow_matching_loss_hook`
- `build_training_algorithm`

### Step 3: package extension contract

Read:

- `CytoBridge-main/docs/runtime/custom-algorithms/README.md`
- `CytoBridge-main/docs/runtime/flow-matching/README.md`

Answer these questions while reading:

- which hook handles raw `adata`?
- which hook builds the model?
- which hook owns the full stage loop when builtin FM/neural_ode is not enough?
- which hook replaces the default evaluation-time simulator?
- which hook assembles the backend?
- which hook adds auxiliary differentiable loss?
- when is `build_pairwise_cost(...)` enough?
- when is `build_state(...)` actually justified?
- which proposal pseudocode steps map to which actual runtime hooks?

### Step 4: runtime source only if needed

Read:

- `CytoBridge-main/CytoBridge/tl/training_algorithm.py`
- `CytoBridge-main/CytoBridge/tl/fit.py`
- `CytoBridge-main/CytoBridge/tl/flow_matching_backends.py`
- `CytoBridge-main/CytoBridge/tl/trainer.py`

Useful search targets:

- `compute_ot_coupling`
- `build_state(`
- `prepare(`
- `sample_batch(`
- `training_data_builder`
- `FlowMatchingBuildContext`
- `TrainingDataBundle`

## The Four Runtime Axes

Do not mix these.

1. `model.components`
- decides which network heads exist at all

2. `train_strategy`
- decides which of those heads are regressed in the current stage

3. `MassStrategy`
- defines the terminal mass target for each sampled source particle over one
  local interval

4. `conditional path`
- defines path geometry, stochasticity, local interpolation, and local mass
  dynamics

Rules:

- do not treat the conditional path as the place where `v/g/s` are turned on or off
- do not treat `train_strategy` as if it redefines path geometry
- do not treat `MassStrategy` as if it chooses which model heads exist
- a backend builder may choose different defaults using stage flags, but that is
  builder logic, not path semantics

## Hook Selection Map

Normal custom algorithm template symbols:

1. optional `model_builder(...)`
2. optional `stage_runner(...)`
3. optional `inference_context_builder(...)`
4. optional `simulation_hook(...)`
5. `CustomCoupling.build_pairwise_cost(...)`
6. `build_flow_matching_backend(build_context)`
7. optional `training_data_builder(...)`
8. optional `flow_matching_loss_hook(...)`
9. optional `evaluation_metrics_hook(...)`
10. `build_training_algorithm(context)`

Responsibility split:

- `model_builder(...)`
  - replace the builtin `DynamicalModel`
  - required when the algorithm uses a different model family
  - expose extra trainable heads through `model.cytobridge_component_modules`,
    stage `trainable_modules`, or `cytobridge_trainable_parameters(...)`
    instead of hiding them under unrelated builtin heads to get optimizer
    coverage
- `stage_runner(...)`
  - replace builtin stage execution
  - use this for custom FM training semantics, custom `neural_ode`,
    higher-order dynamics, or trajectory-level objectives
- `simulation_hook(...)`
  - replace the builtin evaluation-time predictor/simulator
  - builtin `W1/TMV` definitions remain unchanged
  - return the full trajectory for `sim_context.trajectory_time_points`
- `inference_context_builder(...)`
  - prepare a flexible evaluation-time payload for custom simulators
  - payload fields are algorithm-specific
  - provenance should show t=0/exogenous/model-state sources
- `training_data_builder(...)`
  - raw adata hook
  - read `obs`, `obsm`, `layers`
  - attach aligned extra modalities
- `build_pairwise_cost(...)`
  - define adjacent-time transport cost when the full pairwise cost matrix is
    acceptable for the target scale
- custom `CouplingStrategy.build_state(...)` / `sample_pairs(...)`
  - define adjacent-time transport when semantics or memory require custom
    state, streaming, sparse candidates, landmarks, or non-dense sampling
- `build_flow_matching_backend(build_context)`
  - assemble coupling, path, mass, backend
- `flow_matching_loss_hook(...)`
  - add auxiliary differentiable loss or explicitly replace one builtin
    component loss
  - `loss_context.net_input` is the exact tensor passed to builtin heads in
    `[x, t]` order; use `loss_context.x_input` and `loss_context.t_input`
    instead of manually slicing when possible
  - return `FlowMatchingLossResult(replace_velocity_loss=...)`,
    `replace_growth_loss=...`, or `replace_score_loss=...` when replacing a
    component; do not emulate replacement with
    `custom_loss - guessed_builtin_loss`
- `evaluation_metrics_hook(...)`
  - append custom evaluation metrics only
  - never replace builtin `W1` or `TMV`

## Edit Workflow

### 1. Pick the smallest valid hook

Decision order:

1. If only adjacent-pair geometry changes and full pairwise costs fit memory,
   use `build_pairwise_cost(...)`.
2. If adjacent-pair coupling needs true streaming/sparse/custom state, use
   custom `CouplingStrategy.build_state(...)` / `sample_pairs(...)`.
3. If raw AnnData or aligned side modalities are needed, add
   `training_data_builder(...)`.
4. If path/mass/backend composition changes, edit
   `build_flow_matching_backend(build_context)`.
5. If the model family changes, add `model_builder(...)`.
   Extra heads can be trained through:

   ```python
   self.birth_head = torch.nn.Sequential(...)
   self.death_head = torch.nn.Sequential(...)
   self.cytobridge_component_modules = {
       "growth": ["birth_head", "death_head"],
   }
   ```

   Then `train_strategy: g` trains those heads. For one-off config, use:

   ```yaml
   trainable_modules:
     - birth_head
     - death_head
   ```

   Advanced models may implement
   `cytobridge_trainable_parameters(stage_params=..., train_flags=...)`.

   Complete pattern:

   ```python
   def model_builder(model_context: ModelBuildContext) -> torch.nn.Module:
       model = DynamicalModel(
           model_context.latent_dim,
           model_context.resolved_config["model"],
       )
       model.add_module(
           "custom_path_net",
           torch.nn.Sequential(
               torch.nn.Linear(model_context.latent_dim + 1, 32),
               torch.nn.Tanh(),
               torch.nn.Linear(32, model_context.latent_dim),
           ),
       )
       model.cytobridge_component_modules = {
           "velocity": ["custom_path_net"],
       }
       return model

   def build_flow_matching_backend(build_context: FlowMatchingBuildContext):
       path_net = getattr(build_context.model, "custom_path_net", None)
       if path_net is None:
           raise RuntimeError("model_builder must attach custom_path_net")
       # Pass this exact module into the custom path/backend.
       ...
   ```

   Use the component key that matches the stage which should train the module:
   `velocity` for `train_strategy: v`, `growth` for `g`, `score` for `s`,
   `interaction` for `i`, or `always` only when every stage should train it.
6. If only an auxiliary differentiable term is needed, add
   `flow_matching_loss_hook(...)`.
7. If evaluation-time prediction changes, add
   `inference_context_builder(...)` and `simulation_hook(...)`.
8. If only claim/application metrics are needed, add
   `evaluation_metrics_hook(...)`.
9. Use `stage_runner(...)` only when the builtin stage loop cannot express the
   approved algorithm. If the stage only needs custom precomputed state,
   custom batches, or a custom loss, call
   `CytoBridge.tl.custom_stage_loop.run_custom_stage_loop(...)` from the
   runner so the package still owns optimizer setup, checkpointing, device
   placement, timeout checks, and standardized logs.
   If a custom runner is only for an extra stage such as path pretraining,
   return `None` for ordinary `flow_matching` or `neural_ode` stages so the
   package default training loop still runs. An empty `StageRunnerResult` is
   not a trained flow-matching stage.

If the change can be expressed in a smaller hook, do not broaden the edit.

### Inference integrity boundary

Custom `simulation_hook(...)` must roll out dynamics from allowed inference
inputs: t=0 cells, t=0 aligned modalities, known exogenous conditions,
constants/priors, time grid, config, and trained model state.

It must not construct predictions from future observed cells, future counts,
future total mass, future modalities, or future distribution statistics.
Do not post-hoc rescale weights or move particles to improve TMV/W1/claim
metrics.

The runtime now treats the prediction artifact as a dense `EvaluationTrajectory`
from t0 to the final requested time. A custom simulator should use
`sim_context.trajectory_time_points` and return matching
`SimulationResult.predicted_points_by_time`, `predicted_weights_by_time`, and
`trajectory_time_points`. Returning only observed future-time predictions is not
trusted campaign evidence. Additive metrics should use
`EvaluationMetricsContext.evaluation_trajectory` or its trajectory arrays rather
than running a separate inference path.

Two rollout mass representations are supported:

- weighted-particle rollout: particle count may stay fixed while
  `predicted_weights_by_time` changes
- explicit birth/death/splitting rollout: particle count may change between
  trajectory slices and weights may all be one

For explicit unit-weight splitting, the first trajectory slice should also use
unit weights. Builtin TMV compares predicted total mass relative to the first
predicted slice, so `n_pred(t) / n_pred(t0)` is compared to the observed
cell-count ratio.

When custom inference is needed, prefer adding `inference_context_builder(...)`
that returns `InferenceContext(payload, visibility, provenance, notes)`.
The payload is deliberately free-form; the important part is clear provenance.

### 2. Patch the template, do not re-invent the runtime

When modifying `algorithm.py`:

- patch the generated template
- preserve the overall structure unless there is a concrete reason to change it
- keep `build_training_algorithm(context)` intact
- prefer local patches for function-level fixes; do not use a full-file rewrite
  unless the content you pass is the complete, parseable `algorithm.py`
- after any Python edit, run a syntax/import smoke check before preview,
  campaign, or trusted training

Do not invent extra helper methods and assume the runtime will call them unless
you verified the call path from source.

### 3. Change coupling first when the algorithm is about transport

Small/medium-data default:

- implement `CostBasedPairwiseOTCouplingStrategy.build_pairwise_cost(...)`

Real-data pairwise-cost default:

- implement `ChunkedTransportCouplingStrategy.build_pairwise_cost_block(...)`
- build only the bounded block cost on the provided `device`
- let the package split chunks, solve each block in `build_state(...)`, cache
  subplans, and sample from cached transport mass in `sample_pairs(...)`
- if the approved method defines sparse candidate support directly, use
  `CouplingPlanStore.from_edges(...)` / `TransportPlanBuilder.edges_to_store(...)`
  instead of writing a private edge sampler
- do not build a full `n_source x n_target` cost solely to call
  `TransportPlanBuilder.solve_cost_to_store(...)`; use
  `ChunkedTransportCouplingStrategy.build_pairwise_cost_block(...)` for large
  real-data adjacent pairs
- let `CouplingPlanStore` own dense/chunked coupling state; do not create dummy
  `state.plans` entries or manually maintain `sampling_info_plans`
- treat preview/training mini-batch and memory preflight messages as advisory
  unless they report an actual contract error such as `backend.prepare(...)`
  failure, empty `sample_pairs(...)`, invalid mass, NaN/inf, timeout, or OOM

Only override `build_state(...)` if you truly need:

- global cross-time coupling logic
- shared global regularization across time pairs
- custom global cached state
- true large-data streaming/sparse/landmark coupling that cannot be expressed by
  `ChunkedTransportCouplingStrategy`

### 4. Add raw-adata handling only when necessary

If you need auxiliary priors, covariates, extra modalities, or other aligned side
channels:

- add them with `training_data_builder(...)`
- consume them through `build_context.training_data.extra_modalities_by_time`

Do not hard-code a separate `.h5ad` reload path.

### 5. Keep path changes explicit

If you change the path, be able to explain:

- why the builtin path is insufficient
- what replacement path you use
- why it still supports recovery of the observed target distribution
- how the velocity or score target is computed from that replacement path

Path design levels:

1. Linear deterministic path. This is the default flow-matching choice and the
   right starting point unless the proposal's claim needs non-Euclidean,
   stochastic, or birth-death geometry.
2. Analytic or nearly analytic path. Examples include WFR-FM traveling-Gaussian
   paths, Brownian/Schrodinger bridge-style paths, or another closed-form local
   bridge whose endpoint and velocity targets are well defined.
3. Learned path parameterization. Examples include Riemannian/geodesic
   corrections. Use this only when the proposal needs it and provide ablations
   against the simpler path. Longer path pretraining is not automatically better;
   verify downstream W1/TMV and the proposal claim metric, not only path energy.

For multi-timepoint data, a shared custom path network must know which interval
it is modeling. Do not collapse all adjacent biological gaps into the same
local-time input. Path methods receive local path time `s in [0, 1]`; keep this
local parameter for endpoint constraints and derivatives. Add interval/global
context as conditioning features, following the WLF multi-constraint pattern:
local coefficients preserve endpoints, while `t_global`, interval start/end,
interval length, or `time_idx` tell the shared network which biological gap it
is solving. A WLF-style safe pattern is:

```text
x_s = beta0 x0 + beta1 x1
      + (1 - beta0^2 - beta1^2) eta(x0, x1, t_global, s, t0, t1, delta_t)
```

The envelope preserves endpoints; the global/interval features distinguish
different time gaps while allowing shared parameters.

The package backend notifies custom paths before sampling each adjacent gap via
`set_interval_context(t0=..., t1=..., delta_t=..., time_idx=..., device=..., dtype=...)`
when that method exists. A simpler `set_interval(delta_t)` fallback is also
supported. Do not feed raw absolute time into an endpoint envelope without
normalizing it to the interval; that breaks the endpoint semantics.

If the path is nonlinear in time, changing only `compute_mu_t(...)` or
`sample_xt(...)` is not enough. The default deterministic velocity target in
`RegularizedUnbalancedConditionalPath.compute_conditional_flow(...)` reduces to
`x1 - x0` when `sigma=0`, which is correct only for a linear path. For learned
geodesic, bridge, spline, or other nonlinear paths, implement the corresponding
`compute_conditional_flow(...)` from the analytic derivative or a controlled
autograd/finite-difference derivative, and record this in
`IMPLEMENTATION_MAP.md`. Otherwise the model is trained on straight-line
velocity targets while sampling nonlinear `x_t`, which is not faithful to a
proposal that claims non-straight path dynamics.

The derivative is vector-valued. Do not compute `grad(x_t.sum(), s)` and reuse
the resulting `(batch, 1)` derivative for every latent dimension. That is the
derivative of a scalar coordinate sum, not `dx_t/ds`. The returned `u_t` must
have shape `(batch, latent_dim)` and should pass a smoke check where different
latent dimensions have different derivative values.
Do not implement a learned path as two unrelated outputs `(f, df_dt)` unless
the architecture or training objective enforces that the second output is the
actual time derivative of the first. An unconstrained derivative head is just
another learned vector field and does not make `compute_mu_t(...)` and
`compute_conditional_flow(...)` mutually consistent.

### 6. Keep loss-hook usage narrow

Use `flow_matching_loss_hook(...)` for additive supervision on top of the
builtin FM objective, or for an explicit replacement of one builtin component
loss through `FlowMatchingLossResult(replace_velocity_loss=...)`,
`replace_growth_loss=...`, or `replace_score_loss=...`.

Good uses:

- regularization
- consistency loss
- prior-alignment penalty
- replacing the velocity/growth/score component with the same proposal's
  mathematically required loss while leaving the rest of the builtin trainer
  intact

Bad uses:

- replacing the default flow-matching objective implicitly by returning
  `extra_loss = custom_loss - guessed_builtin_loss`
- hiding the main algorithm inside a large custom loss without explanation

Builtin `velocity_net`, `growth_net`, and `score_net` all take one concatenated
input tensor in `[x, t]` order. In a loss hook, call
`model.velocity_net(loss_context.net_input)` or build
`torch.cat([loss_context.x_input, loss_context.t_input], dim=1)`. Do not split
`net_input` as `[t, x]`.

### 7. Keep evaluation-hook usage narrow

Use `evaluation_metrics_hook(...)` only for additive validation metrics.

Good uses:

- prior-conformity metrics
- biological monotonicity checks
- application-specific diagnostics that test the proposal's stated objective

Bad uses:

- redefining builtin `W1`
- redefining builtin `TMV`
- changing metrics after seeing results just to make the run look better

## Scalability And Mini-Batch Rules

Custom coupling must support a real scalability strategy.

Default rule:

- if a time-pair coupling matrix is around `1000 x 1000` or smaller, full-block
  cost/solver state is normally acceptable; tiny/small simulations do not need
  extra mini-batch machinery unless profiling shows a problem
- if a time-pair coupling matrix is much larger than `1000 x 1000`, support
  mini-batch, chunking, sparse candidates, landmarks, coresets, or another
  bounded-memory estimator
- use `chunk_size ~= 1000` as the default boundary: small gaps become one full
  block, while larger gaps must actually construct costs/plans per chunk rather
  than building full dense state first

Preferred implementation pattern:

- use `ChunkedTransportCouplingStrategy` when only pairwise geometry changes
  and full adjacent-pair state is too large
- if a custom state is unavoidable, return
  `CouplingState(plan_stores=[CouplingPlanStore.from_chunked(...)])` and sample
  through `state.plan_store(time_idx).sample_pairs(...)`
- for UOT-style couplings, set `reg` and `reg_m` explicitly in config when you
  have stable values; this skips auto-regularization search. If omitted, the
  package auto-tunes them in `build_state(...)`. The default auto-reg path is
  CPU for legacy numerical consistency; `auto_reg_device: cuda` is available for
  experiments but should be benchmarked before trusting it.
- if your custom cost changes the scale or distribution of the nearest builtin
  cost, do not let preview spend its budget blindly auto-searching
  regularization. First inspect cost statistics on one representative block,
  decide whether `normalize_cost` / cost rescaling is needed, and prefer fixed
  `reg` / `reg_m` once the scale is understood. A timeout during preview
  inspection often means cost-scale / auto-reg search or solver stability, not
  an epoch-training bottleneck.
- do not assume that returning `PairwiseCost` plus `use_mini_batch=True` is
  enough; inspect whether full cost, mask, kernel, or plan matrices are
  materialized before chunking
- if the built-in cost-based path would still require full adjacent-pair dense
  state, move to the chunked-cost API before writing a custom sampler

How a proposal-consistent chunked pattern should work:

- for each adjacent time gap, source and target cells are randomly split into
  matching chunks
- balanced OT solves a local normalized transport problem inside each chunk
- UOT solves a local relaxed-mass problem inside each chunk using the sliced
  source/target mass vectors
- pair sampling uses transport mass, so rows with larger assigned mass are
  sampled more often and target cells are sampled conditionally within the row
- flow matching then trains on sampled endpoint pairs, not on future observed
  summary statistics

Do not downsample by default. Observed cell counts across time points can encode
proliferation, death, sampling efficiency, or total-mass change, and more valid
cells usually give better coupling and training evidence. If you must downsample
a dataset for debugging, keep timepoint proportions by using the same sampling
ratio for each time point. Do not sample a fixed count from each time point when
the algorithm may depend on unbalanced mass or growth, because that destroys the
observed cell-count/mass signal. Formal biological campaign evidence should use
the full prepared real dataset whenever computationally feasible; smaller panels
are smoke/debug evidence unless larger or full-data validation supports the
claim.

## Anti-Hallucination Checks Before You Patch

Verify these from docs/source before editing:

- which function actually constructs coupling at train time?
- which function actually receives raw `adata`?
- whether your intended helper function is ever called by the runtime
- whether the template already gives you the hook you need

## Config Discipline

`config.yaml` is seeded from builtin `vgfm`.

Rules:

- keep the seeded config unless the algorithm truly requires another stage mode
- `flow_matching` remains the default template mode
- use `neural_ode` only when the proposal and implementation actually target that regime
- treat generated hyperparameters as the baseline default
- do not immediately edit `epochs`, `lr`, or `batch_size`
- builtin flow-matching methods use 3000 epochs; keep the seeded 3000-epoch flow-matching default unless convergence evidence justifies a shorter schedule
- if you change `epochs`, `lr`, or `batch_size`, state the reason explicitly and tie it to convergence or numerical stability
- do not use fine-tuning from an already trained run, cached fitted state, or target-specific precomputed predictions as campaign evidence
- prefer changing model/coupling/path/mass/stage logic before scalar knobs

## Handoff

Once you finish editing the algorithm files, do not train immediately.

Before any `run_training(...)`, read:

- `~/.cellcompass/skills/algorithm/review-and-training/SKILL.md`
