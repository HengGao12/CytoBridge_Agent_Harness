---
title: "Flow matching developer API map"
summary: "Agent-facing map for flow-matching defaults, contracts, and extension hooks."
read_when:
  - "Implementing or reviewing flow-matching hooks"
  - "Checking what the runtime does when a hook is not overridden"
---
# Flow-Matching Developer API Map

Use this directory when implementing or reviewing custom flow-matching
algorithms. The goal is to choose the smallest correct hook, know exactly what
the package will do by default, and avoid reading source unless the documented
contract is insufficient.

Read in this order:

1. this file
2. `semantics-and-callgraph.md`
3. `builtin-implementation-guide.md`
4. `extension-points.md`
5. `examples-and-hooks.md`
6. `agent-checklist-and-failures.md`

If the algorithm is still at proposal/theory stage, read
`docs/theory/README.md` first. Runtime docs are for implementing an approved
algorithm, not for replacing the theory review.

## Fixed Data Contract

The default runtime expects:

- `adata.obs["time_point_processed"]`: canonical processed time axis.
- `adata.obsm["X_latent"]`: canonical transcriptomic state space.

Custom algorithms may use extra modalities from `obs`, `obsm`, `layers`, or
`uns`, but those modalities augment the dynamics model. They should not silently
replace the transcriptomic state or redefine the time axis.

Use `training_data_builder(...)` only when raw AnnData fields need to be
aligned and packed into a `TrainingDataBundle`. Once a bundle exists,
flow-matching backend builders should consume `build_context.training_data`
rather than reloading `.h5ad` files.

## What Happens If You Do Not Override A Hook

| Hook / field | If omitted, package behavior |
| --- | --- |
| `training_data_builder` | Load `time_point_processed`, split `X_latent` by time, and build the canonical `TrainingDataBundle`. |
| `flow_matching_backend_builder` | Use the backend selected by config. Default unnamed FM behavior is UOT coupling + regularized unbalanced path + UOT mass. |
| `model_builder` | Build the package `DynamicalModel` from `config["model"]`. |
| `stage_runner` | Use the standard package training loop for `flow_matching` or other configured stage modes. If a custom `stage_runner` is registered, stages for which it returns `None` are delegated back to this standard loop. |
| `flow_matching_loss_hook` | Train only builtin velocity/growth/score losses active for the stage. Hooks may add `extra_loss` or explicitly replace one active component loss. |
| `inference_context_builder` | Evaluation simulation receives only the standard t0 data/time-grid/model context. |
| `simulation_hook` | Use package rollout from t0 to final requested time. |
| `evaluation_metrics_hook` | Compute only builtin metrics and previously configured additive metrics. |
| `evaluation_metrics_params` | Empty dict. |

Named builtin FM backend configs:

- `balanced_ot_cfm`: balanced OT coupling, deterministic linear path, no mass.
- `sf2m`: balanced entropic OT coupling, stochastic bridge path, no mass.
- `vgfm`: UOT coupling, deterministic linear path, UOT mass targets.
- `wfrfm`: WFR-OET semi-coupling, traveling-Gaussian WFR path, WFR terminal mass targets. Its default `delta: auto` estimates the WFR length scale from adjacent-time latent distances; WFR-FM builtin baselines should normally keep auto delta and only use a fixed value for a deliberate controlled comparison.
- `crufm`: UOT coupling, stochastic bridge path, UOT mass targets.

Before implementing a custom flow-matching algorithm, read
`builtin-implementation-guide.md` and inspect the closest builtin YAML and
source symbols. Custom code should normally reuse or subclass the closest
builtin path. If it instead rebuilds the training loop or solves coupling inside
each epoch, document why that is required by the approved proposal.

## Choose The Smallest Correct Hook

| Need | Hook |
| --- | --- |
| Add aligned auxiliary covariate/modality inputs | `training_data_builder(...)` |
| Change only adjacent-pair cost on small/medium data | `CostBasedPairwiseOTCouplingStrategy.build_pairwise_cost(...)` |
| Change adjacent-pair cost on real data | `ChunkedTransportCouplingStrategy.build_pairwise_cost_block(...)` |
| Convert a custom cost into a standard sampler | `TransportPlanBuilder.solve_cost_to_store(...)` |
| Use a sparse support graph with edge weights | `CouplingPlanStore.from_edges(...)` or `TransportPlanBuilder.edges_to_store(...)` |
| Change transport solver convention or marginal relaxation | subclass the closest builtin coupling and override the solver hook |
| Need true large-data global/sparse coupling not expressible as independent chunks | custom `CouplingStrategy.build_state(...)` with `CouplingPlanStore` |
| Change conditional path geometry/noise/mass interpolation | custom path object in `flow_matching_backend_builder(...)` |
| Change terminal mass target semantics | custom `MassStrategy` |
| Add differentiable auxiliary supervision | `flow_matching_loss_hook(...)` with `extra_loss` |
| Replace one builtin component loss while keeping the standard trainer | `flow_matching_loss_hook(...)` with `replace_velocity_loss`, `replace_growth_loss`, or `replace_score_loss` |
| Replace the model architecture | `model_builder(...)` |
| Replace the whole stage loop for specific stages | `stage_runner(...)`; return `None` for ordinary `flow_matching`/`neural_ode` stages that should use package training |
| Need custom batches/losses but not a hand-written optimizer loop | `stage_runner(...)` plus `run_custom_stage_loop(...)` |
| Replace evaluation-time rollout | `inference_context_builder(...)` plus `simulation_hook(...)` |
| Add claim/application metrics | `evaluation_metrics_hook(...)` |

Builtin velocity/growth/score heads all receive one concatenated input tensor in
`[x, t]` order. In `flow_matching_loss_hook(...)`, use
`loss_context.net_input` directly or the split-safe fields
`loss_context.x_input` and `loss_context.t_input`; do not parse it as `[t, x]`.

Do not broaden the hook just because implementation is difficult. The default
large-data path is the package chunked-cost/plan-store API, not a hand-written
sampler. If the approved proposal requires semantics that do not fit the
chunked API, implement the larger hook faithfully or revise the proposal.
If the proposal defines a sparse candidate graph, do not rebuild sampling logic:
store `(source_index, target_index, edge_weight)` with
`CouplingPlanStore.from_edges(...)` and let the store sample pairs.

For algorithms that truly need custom training batches or nonstandard losses,
prefer `CytoBridge.tl.custom_stage_loop.run_custom_stage_loop(...)` before
writing a full `stage_runner`. The objective supplies `build_state`,
`sample_batch`, and `compute_loss`; the package still owns optimizer setup,
checkpointing, device placement, timeout checks, progress logs, and trainable
parameter selection. Use a fully hand-written `stage_runner` only when the
optimizer loop itself is part of the approved method.

`TransportPlanBuilder.solve_cost_to_store(...)` is not a scalability shortcut:
it consumes a cost object that already exists. On real data, do not first build
a full `n_source x n_target` cost just to pass it into the builder. If the full
adjacent-pair cost may be large, use
`ChunkedTransportCouplingStrategy.build_pairwise_cost_block(...)` so the package
constructs and solves bounded blocks.

For custom conditional paths, read `extension-points.md` section 10 before
coding. The important design choice is not only "which path", but also whether
the path is linear, analytic/bridge-based, or learned; and for multi-timepoint
data, whether the path receives enough global/interval time context to
distinguish adjacent biological gaps.

Runtime convention: conditional-path methods receive local interval time
`s in [0, 1]`, while the trained dynamics model receives global biological
time. A shared learned path network that is reused across multiple adjacent
gaps should use the optional interval hook
`set_interval_context(t0=..., t1=..., delta_t=..., time_idx=...)` or the simpler
`set_interval(delta_t)` fallback, then pass `t_global`/interval features into
its path network as needed. This preserves endpoint constraints while avoiding
the mistake of treating every gap as the same local problem.

## Mini-Batch And Scalability: Precise Meaning

Do not treat `use_mini_batch=True` as proof of scalability.

Current cost-based helper behavior:

- `CostBasedPairwiseOTCouplingStrategy.build_pairwise_cost(...)` must return a
  full `PairwiseCost.cost_matrix` with shape `(n_source, n_target)`.
- The helper may chunk solver calls and expose sampler-compatible chunk
  metadata.
- It can still materialize full cost, mask, kernel, or plan-sized arrays before
  or during solving.

Therefore:

- Cost-based hooks are convenient for small/medium data and for custom geometry
  that is known to fit memory.
- An adjacent time gap with about 1000 source cells and 1000 target cells yields
  a `1000 x 1000` cost/coupling matrix, which is an acceptable full-block solve
  on normal development hardware. For tiny/small simulations, including the 2D
  simulation benchmark, do not add mini-batch complexity unless runtime
  evidence says it is needed. Real-data benchmarks with much larger adjacent
  pairs, such as Weinreb-scale gaps, must use bounded chunking, sparse support,
  or another real scalability strategy.
- For real biological datasets with large adjacent time pairs, the
  implementation must document the actual memory path.
- If a method would require dense `n_t x n_{t+1}` state on large data, implement
  a true streaming/chunked/sparse/landmark/subsampled coupling or revise the
  proposal/data contract.

This is a reviewer and engineering-quality requirement, not a static runtime
heuristic block. Runtime preflight warnings are advisory and must not reject a
valid mini-batch/chunked implementation merely because it uses custom setup
code. Hard failures should come from actual contract errors, runtime
OOM/timeout/NaN, or implementation review evidence.

## Solver Device Discipline

For POT-based coupling solves, device placement is part of the performance
contract:

- UOT/VGFM-style coupling can run `ot.unbalanced.sinkhorn_unbalanced(...)` on
  GPU by passing torch CUDA tensors for the source mass, target mass, and cost
  block.
- WFR-FM strict OET coupling can run `ot.unbalanced.mm_unbalanced(...)` on GPU
  the same way; using CUDA here does not change the WFR-OET objective.
- Avoid converting solver inputs to NumPy just before the POT call unless the
  implementation intentionally chooses CPU. It is acceptable to convert the
  bounded output plan/subplan back to NumPy for package sampler storage.
- If coupling precompute shows CPU saturation and near-zero GPU utilization,
  inspect the solver hot path for early `.cpu().numpy()` / `np.asarray(...)`
  conversions.

Changing solver family, entropy regularization, KL placement, or marginal
constraints is not a device-placement optimization; it is an algorithmic change
that must match the approved proposal.

## Evaluation Integrity

The target is a learned dynamics model. Trusted metrics are computed from one
model-generated trajectory:

- start from t0 particles;
- roll forward to the requested final time;
- slice that trajectory at observed/heldout times;
- compute builtin `W1`, builtin `TMV`, and additive custom metrics from those
  trajectory artifacts.

Custom metrics may read:

- `EvaluationMetricsContext.evaluation_trajectory`
- `simulated_points_by_time`
- `simulated_weights_by_time`
- `timepoint_results`
- builtin metrics already computed by the package

Custom metrics must not run a separate shortcut predictor or use future
observed truth to construct predictions.

## Source Escalation Order

Only inspect source if these docs are insufficient:

1. `CytoBridge-main/CytoBridge/tl/training_algorithm.py`
2. `CytoBridge-main/CytoBridge/tl/fit.py`
3. `CytoBridge-main/CytoBridge/tl/flow_matching_backends.py`
4. `CytoBridge-main/CytoBridge/tl/trainer.py`
