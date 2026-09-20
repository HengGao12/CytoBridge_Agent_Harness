---
title: "Flow matching extension points"
summary: "Detailed hook contracts for coupling, path, mass, losses, and metrics."
read_when:
  - "Choosing a flow-matching extension hook"
  - "Implementing custom backend behavior"
---
# Flow-Matching Extension Points

Core types, solver override rules, pair sampling, paths, mass strategy,
metrics, and mini-batch requirements.

## 0. API Contracts At A Glance

| Object / hook | Owns | Must not own |
| --- | --- | --- |
| `TrainingDataBundle` | Time-split `X_latent`, row indices, aligned extra modalities. | Solver logic or metric definitions. |
| `FlowMatchingBackend` | Coupling, conditional path, terminal mass strategy. | Model architecture or epoch loop. |
| `CouplingStrategy.build_state(...)` | Transport state and sampler metadata. | Future-time evaluation correction. |
| `CouplingStrategy.sample_pairs(...)` | Sample endpoint pairs from the transport state. | Invent a different coupling from the one in `build_state`. |
| `ConditionalPath` | Local interpolation, velocity/score targets, local mass dynamics. | Terminal mass ratio semantics. |
| `MassStrategy` | Terminal mass target for sampled source particles. | Path-level interpolation or model-head selection. |
| `flow_matching_loss_hook(...)` | Additive differentiable loss, or explicit replacement of one builtin component loss. | Hidden replacement of builtin FM loss through `custom_loss - guessed_builtin_loss`. |
| `evaluation_metrics_hook(...)` | Additive metrics from sealed rollout artifacts. | Replacement of builtin `W1/TMV` or shortcut prediction. |

When a hook is omitted, the package keeps its documented default behavior from
`README.md`. Do not assume a helper method will be called unless it is part of
one of these contracts.

## 0A. Extra Trainable Network Heads

The standard stage optimizer follows `train_strategy` for standard heads:
`v -> velocity_net`, `g -> growth_net`, `s -> score_net`, and
`i -> interaction_net`.

Custom heads can be trained without pretending to be one of those heads:

- set `model.cytobridge_component_modules = {"growth": ["birth_head"]}` inside
  the custom model, so `train_strategy: g` trains both `growth_net` and
  `birth_head`
- or add stage config `trainable_modules: ["birth_head", "death_head"]`
- or implement `cytobridge_trainable_parameters(stage_params=..., train_flags=...)`
  for advanced parameter selection

This is the preferred interface for source/sink heads, birth/death heads, and
other proposal-defined networks. Do not register unrelated heads under
`growth_net` or `interaction_net` just to make the optimizer include them.

Builtin `velocity_net`, `growth_net`, and `score_net` take one concatenated
tensor in `[x, t]` order. The trainer passes the same tensor to
`flow_matching_loss_hook(...)` as `loss_context.net_input`, and also exposes
`loss_context.x_input` and `loss_context.t_input` to avoid manual slicing.

For a trainable conditional-path module, use the same rule:

- create the module in `model_builder(...)`
- attach it to the returned model with a stable name, for example
  `model.add_module("geodesic_path_net", path_net)`
- include that module in `cytobridge_component_modules(...)` or
  `cytobridge_trainable_parameters(...)` so the stage optimizer owns it
- in `flow_matching_backend_builder(...)`, retrieve it from
  `build_context.model`, for example
  `path_net = getattr(build_context.model, "geodesic_path_net")`
- build the custom `ConditionalFlowMatcher` with that exact module instance,
  not with a placeholder model reference

Do not create a separate trainable module inside `flow_matching_backend_builder`
unless it is intentionally frozen. Backend-created modules are not guaranteed to
be part of the model optimizer, and can also stay on the wrong device if the
model is moved after construction. The backend may own non-trainable cached
state, coupling samplers, and cost objects; trainable neural modules should live
on the model.

Minimal pattern:

```python
def model_builder(model_context: ModelBuildContext) -> torch.nn.Module:
    model = DynamicalModel(
        model_context.latent_dim,
        model_context.resolved_config["model"],
    )
    model.add_module("custom_path_net", torch.nn.Sequential(...))
    model.cytobridge_component_modules = {"velocity": ["custom_path_net"]}
    return model

def build_flow_matching_backend(build_context: FlowMatchingBuildContext):
    path_net = getattr(build_context.model, "custom_path_net", None)
    if path_net is None:
        raise RuntimeError("model_builder must attach custom_path_net")
    return FlowMatchingBackend(path=CustomPath(path_net), coupling=..., mass=...)
```

Choose the `cytobridge_component_modules` key by the stage that should train
the module: `velocity` for `train_strategy: v`, `growth` for `g`, `score` for
`s`, `interaction` for `i`, or `always` only when every stage should train it.

For model-dependent paths, prefer passing the registered submodule directly:

```python
path_net = getattr(build_context.model, "geodesic_path_net", None)
if path_net is None:
    raise ValueError("geodesic_path_net is missing from the model")
path = MyConditionalPath(path_net=path_net, sigma=0.0)
```

Avoid storing `self._model` inside the path and later dereferencing
`self._model.some_head`. Preview, campaign isolation, and checkpoint reloads can
materialize backend objects at different times; a path that silently accepts
`model=None` often fails only during the first sampled batch. Fail fast in the
backend builder if the expected model-owned module is absent.

## 5. Core Types

Source:

- `CytoBridge-main/CytoBridge/tl/flow_matching_backends.py`

Key public types:

- `CouplingState`
- `CouplingResult`
- `PairwiseCost`
- `TrainingDataBundle`
- `FlowMatchingBuildContext`
  - includes `model` when the runtime has already built a model, so backend
    builders can reuse trainable modules created by `model_builder`
- `PairwiseOTCouplingStrategy`
- `CostBasedPairwiseOTCouplingStrategy`
- `ChunkedTransportCouplingStrategy`
- `UnbalancedOTCouplingStrategy`
- `BalancedOTCouplingStrategy`
- `RegularizedUnbalancedConditionalPath`
- `UOTMassStrategy`
- `NullMassStrategy`
- `FlowMatchingBackend`
- `EvaluationMetricsContext`

### `TrainingDataBundle` fields

Backend builders receive time-split training data through
`FlowMatchingBuildContext.training_data`. Use the documented fields directly;
do not guess aliases.

Public fields:

- `adata`: original AnnData object.
- `time_points`: sorted training time values.
- `latent_by_time`: list of `torch.Tensor`, aligned to `time_points`; this is
  the canonical per-time `X_latent` tensor list.
- `obs_indices_by_time`: row indices into `adata` aligned to `latent_by_time`.
- `extra_modalities_by_time`: optional row-aligned modality lists keyed by
  modality name.
- `metadata`: optional runtime metadata.

There is no `X_by_time` field. If a custom algorithm needs per-time latent
states, use `training_data.latent_by_time`.

## 5A. Solver Override Rules

If your method differs from the builtin runtime only in pairwise geometry and
the full adjacent-pair cost matrix is acceptable for the target scale:

- subclass `CostBasedPairwiseOTCouplingStrategy`
- override `build_pairwise_cost(...)`
- keep the package solver

Use this full-cost path for small gaps. As a practical default, a gap with about
1000 source cells and 1000 target cells yields a `1000 x 1000` cost/coupling
matrix and can be treated as one block (`chunk_size ~= 1000`) unless actual
memory/runtime evidence says otherwise. Tiny 2D simulations do not need extra
mini-batch machinery; much larger real-data gaps do.

Important: `CostBasedPairwiseOTCouplingStrategy` is a cost-matrix API. It
requires a full `PairwiseCost.cost_matrix` for each adjacent time pair. Its
`use_mini_batch` option chunks solver/sampling work, but it is not a proof of
true streaming memory behavior.

If your method differs only in pairwise geometry but the full adjacent-pair cost
matrix is too large:

- subclass `ChunkedTransportCouplingStrategy`
- override `build_pairwise_cost_block(...)`
- build only the bounded source/target block cost on the provided `device`
- let the package split chunks, solve each block once in `build_state(...)`,
  cache subplans, and sample pairs from cached state during training

This is the preferred large-data path before writing custom
`build_state(...)` / `sample_pairs(...)` code. Do not solve OT/UOT/WFR inside
the epoch loop just because the full cost matrix does not fit.

## 5B. Custom Path State And Flow Targets

The path object owns both the sampled intermediate state and the velocity/flow
target. These two pieces must stay mathematically consistent.

For deterministic linear paths, the default
`RegularizedUnbalancedConditionalPath.compute_conditional_flow(...)` reduces to
`x1 - x0`. That is correct for VGFM-style straight-line interpolation. It is
not automatically correct for a nonlinear learned path.

If a custom algorithm overrides `compute_mu_t(...)`, `sample_xt(...)`, or adds a
time-nonlinear path correction, it must also provide the corresponding
`compute_conditional_flow(...)` target, for example from an analytic derivative,
autograd through the path parameterization, or a documented finite-difference
approximation. Otherwise training samples the custom `x_t` locations but
regresses the builtin straight-line velocity target, which is a different
algorithm from a proposal that claims geodesic, bridge, spline, or other
nonlinear path dynamics.

Record the path-state and path-flow contract in `IMPLEMENTATION_MAP.md`:

- formula for `x_t`;
- formula or approximation for `u_t`;
- whether `u_t` is exactly the path derivative or an approved surrogate;
- why the surrogate, if any, preserves the proposal semantics.

For nonlinear vector paths, `u_t` must be the per-coordinate derivative of the
path, with the same shape as `x0` and `x1`. A common wrong implementation is:

```python
path_term = x_t.sum()
u_t = torch.autograd.grad(path_term, s)[0]  # shape (batch, 1)
```

This differentiates the sum of all coordinates and then usually broadcasts one
scalar derivative to every latent dimension. It is not the vector field
`dx_t/ds`. Prefer an analytic derivative when the path formula is known. If
using autograd, use a Jacobian/JVP or finite-difference scheme that returns a
full `(batch, latent_dim)` target, and add a smoke test that two latent
dimensions can have different derivative values.

Likewise, do not make a neural path network output `(f, df_dt)` and assume the
second head is the derivative of the first. That is only valid when the
architecture or an explicit consistency loss enforces `df_dt = d f / d t`.
Without that constraint, the sampled state and velocity target describe
different paths. Compute `df/dt` from the same `f(t, ...)` formula instead.

If your method needs the same solver family but different reference masses:

- use `CostBasedPairwiseOTCouplingStrategy` for full-matrix scale or
  `ChunkedTransportCouplingStrategy` for real-scale data
- return `PairwiseCost(..., source_mass=..., target_mass=...)`

If your method needs the same Sinkhorn helper but fixed or custom
regularization rather than auto-tuned `reg/reg_m`:

- override `compute_ot_coupling(...)`
- call `_solve_cost(pairwise_cost, reg=..., reg_m=...)`

If your method only changes the cost geometry, first ask whether it also
changes the cost scale. Auto-tuned `reg/reg_m` can dominate preview time on
large custom-cost blocks because it runs extra solver probes before the actual
coupling solve. For real-data preview, inspect one representative cost block
and prefer one of these scale-controlled paths:

- set `PairwiseCost(normalize_cost=True)` or otherwise rescale the cost when the
  numeric range is far from the nearest builtin;
- pass fixed `reg` / `reg_m` through the coupling config when ordinary UOT
  semantics are unchanged;
- set `regularization_block_policy: first_block_per_time` only after verifying
  the first block is representative.

The diagnostic should compare against the nearest builtin on the same block:
cost quantiles, finite ratio, GPU/CPU device, solve time, selected
regularization, and sampled row/column mass. Passing a tiny simulation is not
enough evidence that a high-dimensional custom cost is scaled correctly. If the
real benchmark W1 explodes while TMV and runtime look acceptable, inspect this
block-level cost/solver evidence before changing learning rate, epochs, or
network width.

For quadratic metric costs `(x-y)^T G(z) (x-y)`, implement the quadratic form
directly when `G` is defined by landmarks, kernels, graph features, basis
expansions, or low-rank factors. Avoid constructing a full
`[n_pairs, expansion, latent_dim, latent_dim]` tensor just to contract it with
`x-y`; this turns many real-data previews into setup-time failures. The usual
pattern is:

```text
diff = x - y
metric_feature = feature(z, basis)
cost = ||diff||^2 + sum_b weight_b(z) * <feature_b(z), diff>^2
```

or the analogous factorized expression for the proposed metric. This preserves
the same cost while reducing the hot cost block from quadratic to linear in
latent dimension. Document the algebraic equivalence in the implementation map.

If your method needs a different solver convention entirely, such as:

- symmetric unbalanced OT
- `mm_unbalanced`
- different KL placement
- different semi-relaxed side
- paper-specific normalization

then do not rely on the package cost-based helper. In that case:

- override `compute_ot_coupling(...)` directly
- or implement a custom coupling strategy from scratch
- and document explicitly which marginal is relaxed, what `reg` means, what
  `reg_m` means, and how pairs are sampled from the resulting plan

If your method must scale to large real datasets and the pairwise object is
large, do not use `build_pairwise_cost(...)` as the main implementation unless
you can prove the actual arrays fit memory. Use a custom `CouplingStrategy`
that constructs costs per chunk, keeps sparse candidate sets, uses landmarks,
or stores only sampler-compatible summaries.

## 6. Cost-Based Entry Point: `CostBasedPairwiseOTCouplingStrategy`

This is the simplest developer hook when you only need to change the transport
cost and the full adjacent-pair cost matrix is acceptable.

```python
class CostBasedPairwiseOTCouplingStrategy(PairwiseOTCouplingStrategy):
    def build_pairwise_cost(
        self,
        x0: np.ndarray,
        x1: np.ndarray,
        *,
        time_idx: int,
        t0: float,
        t1: float,
        device: torch.device,
    ) -> PairwiseCost:
        ...
```

What it means:

- `x0`: latent cells at time `t0`, shape `(n0, latent_dim)`
- `x1`: latent cells at time `t1`, shape `(n1, latent_dim)`
- `time_idx`: index of this adjacent interval
- `t0`, `t1`: real training times
- `device`: optional compute placement for OT work

You return:

```python
PairwiseCost(
    cost_matrix=np.ndarray,           # shape (n0, n1)
    source_mass=None | np.ndarray,    # optional source marginals
    target_mass=None | np.ndarray,    # optional target marginals
    normalize_cost=None | bool,       # optional local normalization override
    metadata=dict(),                  # optional diagnostics
)
```

The package then:

- solves OT/UOT for each pair using the built-in full or chunked solver
- assembles all adjacent-pair results into `CouplingState`
- reuses the default pair sampler
- feeds sampled pairs into the path + mass logic

The package does **not** call `build_pairwise_cost(...)` lazily per mini-batch.
The cost matrix is built before solving. For a time pair with `n0` and `n1`
cells, memory is at least proportional to `n0 * n1` for the cost itself, and
may include plan-sized arrays as well.

If you already have a fully custom plan, you may still drop down to
`PairwiseOTCouplingStrategy.compute_ot_coupling(...)`, but that is now the lower-level path.

## 7. `PairwiseCost` Contract

`cost_matrix`

- type: `np.ndarray`
- shape: `(n_source, n_target)`
- must be finite
- usually should be nonnegative
- may include prior penalties / rewards as long as the final matrix is numerically stable

`source_mass`, `target_mass`

- optional
- if omitted, the package uses uniform marginals
- if provided, they must align with source/target rows

Mini-batch behavior:

- you do not need to build `sampling_info` yourself in the small/medium
  cost-based case
- `CostBasedPairwiseOTCouplingStrategy` produces sampler-compatible chunk
  metadata when chunked solving is used
- this does not remove the requirement to build the full `cost_matrix`
- this does not by itself prove the implementation is safe on large adjacent
  time pairs

## 8. When To Override `build_state(...)`

Override `build_state(...)` only if pairwise independent coupling is not enough.

Typical reasons:

- you need global regularization chosen jointly across all time pairs
- you need nonlocal constraints that depend on more than one adjacent pair
- you need a coupling object that is not expressible as "cost matrix + solver"
- you need to cache custom global structures not naturally expressible as a
  sequence of independent `CouplingResult`s

If your coupling logic can be expressed per adjacent pair, prefer the standard
cost hooks:

- `CostBasedPairwiseOTCouplingStrategy.build_pairwise_cost(...)` when the full
  adjacent-pair object fits memory;
- `ChunkedTransportCouplingStrategy.build_pairwise_cost_block(...)` when the
  pairwise object is too large but the cost still decomposes by bounded blocks.
- `TransportPlanBuilder.solve_cost_to_store(...)` when custom setup code already
  has a `PairwiseCost` and only needs a standard OT/UOT solver plus sampler.
- `CouplingPlanStore.from_edges(...)` when the algorithm defines sparse support
  edges and transport weights directly.

Do not override `build_state(...)` just to get mini-batches or avoid a dense
matrix; the chunked-cost strategy already gives bounded precompute plus
`CouplingPlanStore` sampling.

Do not use `TransportPlanBuilder.solve_cost_to_store(...)` as a way to hide
full-cost allocation. If constructing the input `PairwiseCost` would require a
large full `n_source x n_target` tensor, the correct hook is
`ChunkedTransportCouplingStrategy.build_pairwise_cost_block(...)`.

`TransportPlanBuilder` supports `solver_mode="uot"`, `"balanced"`, and
`"wfr"` for already-constructed pairwise costs. The `"wfr"` mode assumes the
cost matrix is already the WFR-OET pairwise cost and returns a plan store with
pair-specific terminal mass.

If pairwise-independent chunked solving is still not expressive enough,
overriding `build_state(...)` is the escape hatch. In that case store:

- one `CouplingPlanStore` per adjacent time gap in
  `CouplingState(plan_stores=[...])`;
- either a dense plan or bounded chunk/sparse state inside the plan store;
- enough diagnostics in `state.metadata` for review and reproducibility.

Do not create dummy `(1, 1)` plans or manually coordinate `plans` with
`sampling_info_plans`. Those names are legacy read-only compatibility views.
New custom code should use `state.plan_store(time_idx)` or subclass
`ChunkedTransportCouplingStrategy`.

## 8A. Custom Stage Loop Helper

Use `run_custom_stage_loop(...)` when an approved algorithm needs custom
training state, custom batches, or custom objective terms that do not fit
`flow_matching_loss_hook(...)`, but it does not need to redefine the optimizer
loop itself.

```python
from CytoBridge.tl.custom_stage_loop import CustomStageLossResult, run_custom_stage_loop

class Objective:
    def build_state(self, context):
        return precompute_training_state(context.training_data, context.stage_params)

    def sample_batch(self, context, state, epoch):
        return sample_custom_batch(state, context.batch_size, context.device)

    def compute_loss(self, context, state, batch, epoch):
        loss = compute_custom_loss(context.model, batch)
        return CustomStageLossResult(loss=loss, logs={"custom_loss": float(loss.detach().cpu())})

def stage_runner(context):
    if context.stage_params.get("mode") != "custom":
        return None
    return run_custom_stage_loop(context, Objective())
```

The helper provides optimizer setup, scheduler stepping, gradient clipping,
train/eval device placement, timeout checks, checkpoint writing, progress
callbacks, loss logs, and standard `StageRunnerResult` formatting. It uses the
same trainable-module conventions as package stages: `train_strategy`,
`trainable_modules`, `train_all_parameters`, `model.cytobridge_component_modules`,
or `model.cytobridge_trainable_parameters(...)`.

Prefer this helper over a hand-written `for epoch` loop unless the method's
optimizer, checkpointing, or scheduler semantics are themselves part of the
approved algorithm.

Sparse support graph pattern:

```python
from CytoBridge.tl.flow_matching_backends import CouplingPlanStore

store = CouplingPlanStore.from_edges(
    source_n=len(x0),
    target_n=len(x1),
    edge_src=edge_source_indices,
    edge_tgt=edge_target_indices,
    edge_weight=edge_transport_mass,
    terminal_mass_rows=optional_source_terminal_mass,
    metadata={"support": "custom_sparse_edges"},
)
```

Use this when the proposal already defines a sparse set of admissible
source/target pairs and their transport weights. The edge store samples
according to edge mass and provides source terminal mass to mass strategies;
callers still use the same `sample_pairs(...)` interface.

## 9. Default Pair Sampling

If you implement `build_pairwise_cost(...)` correctly and use the cost-based
state layout, you usually do not need to override `sample_pairs(...)`.

The default sampler consumes a single authoritative object:

- `state.plan_store(time_idx)`

The store exposes `sample_pairs(...)` and `terminal_mass_for_batch(...)` for
dense, chunked, and WFR terminal-mass cases. Coupling code should make that
store semantically correct; callers should not reason about storage details.

## 10. Conditional Path

Default path:

- `RegularizedUnbalancedConditionalPath`

Interpretation:

- sample aligned `(x0, x1)` pairs from the coupling
- define an intermediate state rule `x_t`
- regress the implied velocity / score / growth targets

Default expectation for custom algorithms:

- keep the standard straight-line flow-matching path unless there is a real
  theoretical reason to change it

If you override path behavior, document:

- why straight-line interpolation is insufficient
- what new path you use
- why it still supports recovery of the observed target distribution

Path design levels:

1. Linear deterministic path. This is the package's simplest and most robust
   FM path. Prefer it unless the approved proposal needs stochasticity,
   WFR/birth-death geometry, or manifold/geodesic geometry.
2. Analytic or nearly analytic bridge path. Examples include WFR-FM
   traveling-Gaussian paths, Brownian or Schrodinger bridge paths, or another
   closed-form local bridge. Use this when the paper/objective gives endpoint,
   velocity, score, or mass targets directly.
3. Learned path parameterization. Examples include MFM/Riemannian-style learned
   corrections to interpolation. Use this only when the proposal needs it.
   Validate it with ablations against simpler paths; lower path energy or more
   pretraining steps do not by themselves prove better downstream dynamics.

If the path depends on a trainable model-owned submodule, the trusted training
path should fail fast when that submodule is absent. A silent straight-line
fallback is acceptable only as an explicit diagnostic ablation controlled by
config; it should not be the default behavior for campaign evidence.

For multi-timepoint data, distinguish local path time from global biological
time. Path methods receive local time `s in [0, 1]` because endpoint
constraints and conditional-path derivatives are defined over one adjacent
interval. If a shared path network is used across adjacent intervals, it must
also receive enough interval/global context so different biological gaps are not
collapsed into the same local problem. This follows the WLF multi-constraint
pattern: use local coefficients such as
`beta0=(t_{k+1}-t_global)/(t_{k+1}-t_k)` and
`beta1=(t_global-t_k)/(t_{k+1}-t_k)` to preserve endpoints, while passing
`t_global`, `t_k`, `t_{k+1}`, `delta_t`, or `time_idx` as conditioning features.
A common endpoint-preserving template is:

```text
x_s = beta0 x0 + beta1 x1
      + (1 - beta0^2 - beta1^2) eta(x0, x1, t_global, s, t_k, t_{k+1}, delta_t)
```

where the envelope is zero at both endpoints. Other envelopes are acceptable if
they preserve endpoint constraints and the proposal explains the induced
velocity target.

The backend calls optional path interval hooks before each adjacent-gap sample:

```python
path.set_interval_context(t0=..., t1=..., delta_t=..., time_idx=..., device=..., dtype=...)
```

For legacy/simple paths, `path.set_interval(delta_t)` is also supported. Do not
replace the local endpoint parameter with raw absolute time unless the envelope
is re-normalized; otherwise `x_s` will not hit `x0` and `x1` at the interval
endpoints.

## 11. Mass Strategy

Typical choices:

- `UOTMassStrategy`
  - use when terminal per-particle mass targets should come from unbalanced
    coupling row sums
- `NullMassStrategy`
  - use when you want terminal mass to remain `1` over the local interval

Keep mass strategy and `config.yaml` consistent:

- if you use balanced / no-growth assumptions, do not leave growth learning
  enabled accidentally

Current mathematical split:

- `MassStrategy` answers only one question:
  - if a sampled source particle has unit mass at local `t=0`, what mass should
    it have at local `t=1`?
- the conditional path then converts that terminal mass target into:
  - a local growth supervision target `gt`
  - per-sample training weights

Default package semantics are mathematically coherent because:

- `UOTMassStrategy` defines `terminal_mass` from the sampled source row sum of
  the adjacent UOT plan
- the default flow-matching UOT solver uses unit reference marginals
  - `a_i = 1`
  - `b_j = 1`
- so the row sum already lives in the correct unit-mass convention for a local
  source particle normalized to `1`
- the default conditional path assumes log-mass evolves linearly over local
  time, so the local target is:
  - `gt_local = log(terminal_mass)`
- the backend divides by the physical interval length before regressing the
  growth head, so the learned `g(t, x)` remains a per-unit-global-time rate
- integrating that rate across the interval recovers the intended terminal mass
  target

This is why the default split is:

- `MassStrategy`: define the terminal mass ratio
- conditional path: define the local mass dynamics that reach it

Do not move path-level mass dynamics back into `MassStrategy`. That blurs two
different pieces of mathematics:

- what terminal mass target is intended
- how the path reaches that target over time

Default recoverability intuition:

- the coupling defines the adjacent-time joint plan
- pair sampling follows that plan
- `velocity` learns where mass travels
- `growth` learns how much mass reaches the end of the interval
- under exact fit, endpoint aggregation recovers the intended late marginal
  interval by interval

## 12. Additive Evaluation Metrics

Builtin evaluation metrics are fixed:

- `W1`
- `TMV`

Custom algorithms may append metrics, but may not modify builtin metric
definitions.

Use:

- `evaluation_metrics_hook(eval_context: EvaluationMetricsContext)`

The hook receives builtin metrics that have already been computed and may
return only additive entries, for example:

- prior-conformity checks
- temporal monotonicity checks
- custom biological consistency metrics

Rules:

- do not overwrite `w1_scores`
- do not overwrite `tmv_scores`
- do not redefine builtin evaluation semantics in custom workspaces
- return only additive metrics that are directly tied to the proposal's stated
  scientific objective
- prefer reusing the provided inference artifacts instead of re-running
  trajectory simulation

Reusable hook inputs include:

- `eval_context.simulated_points_by_time`
- `eval_context.simulated_weights_by_time`
- `eval_context.timepoint_results`
- `eval_context.metric_params`

The runtime stores these extra metrics under:

- `adata.uns["evaluation_metrics"]["custom_metrics"]`

## 12.5 Inference Context Integrity

Custom `simulation_hook(...)` should not receive or depend on future observed
truth when constructing predictions. When custom inference needs extra inputs,
add `inference_context_builder(...)` and return a flexible `InferenceContext`:

- `payload`: algorithm-specific inputs
- `visibility`: source class for each payload entry
- `provenance`: how the payload was built
- `notes`: optional explanation

Allowed prediction inputs include t=0 cells, t=0 aligned modalities, known
exogenous conditions, constants/priors, time grid, config, and trained model
state. Future observed cells/counts/total mass/modalities/statistics are for
sealed metric computation only, after prediction artifacts have been generated.

## 13. Mini-Batch Requirement

Custom couplings must support large-data execution.

Practical rule:

- if a time-pair plan/cost matrix is around `1000 x 1000` or smaller, meaning
  roughly 1000 source cells by 1000 target cells for that adjacent gap,
  full-block cost is normally acceptable
- if a time-pair plan is much larger than `1000 x 1000`, support a real
  scalability strategy

Recommended default:

- `chunk_size ~= 1000`

Reference implementations:

- `UnbalancedOTCouplingStrategy`
- `BalancedOTCouplingStrategy`

Scalability strategies may include:

- chunked OT/UOT where costs are constructed per chunk rather than as one full
  adjacent-pair matrix;
- sparse candidate generation followed by local OT/UOT;
- landmark or coreset transport with documented reconstruction/sampling;
- stochastic endpoint-pair estimators that preserve the approved proposal
  semantics;
- GPU-vectorized kernels when the actual peak memory is bounded and measured.

What is not enough:

- setting `use_mini_batch=True` while still constructing full dense
  `n_source x n_target` cost/mask/plan arrays for large real data;
- training only on a toy benchmark and claiming scalability without an
  implementation path for larger adjacent time pairs;
- filtering sampled pairs after an unrelated dense coupling has already been
  solved.

If your algorithm does not support a real scalability strategy, `run_training`
may emit scalability warnings and large datasets may fail at runtime. Static
mini-batch markers are advisory only; actual runtime behavior and reviewer
evidence are the source of truth.
