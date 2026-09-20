---
name: authoring
description: Author a custom CytoBridge training algorithm after proposal approval. This is an index skill: use it to choose the right authoring reference before editing code.
---

# Algorithm Authoring

Use this skill only after:

- the proposal exists
- the proposal is approved
- the algorithm workspace has been initialized

This file is the authoring index, not the full manual.

Path rule:

- paths like `references/...` are resolved relative to this skill directory
- for this skill, that directory is `~/.cellcompass/skills/algorithm/authoring/`
- package docs such as `CytoBridge-main/docs/...` are workspace-relative

## Goal

Write a custom algorithm that:

- matches the approved proposal
- fits the existing CytoBridge runtime contracts
- reuses package code where possible
- changes only the necessary layer of the stack

The approved proposal is an implementation contract. Do not implement a
"simpler first version" that changes the algorithm semantics. If implementation
work reveals that a proposal term, coupling, growth/mass mechanism, stochastic
component, conditioning variable, inference path, mathematical claim, or
runtime assumption is wrong, inconsistent, or infeasible under the intended
evidence protocol, patch `PROPOSAL.md` with `apply_workspace_patch(...)` and
let the new proposal version be reviewed again before using training evidence.

Proposal patching rule: first call `get_algorithm_proposal_status(algorithm_id)`
and use its `editable_proposal_path`. Re-read that root `PROPOSAL.md`
immediately before patching. Do not patch `proposal_markdown_registry_path`;
that is an immutable proposal-version archive used for review/audit.

Patch tool syntax rule: `apply_workspace_patch(...)` is not `git apply`.
Patch headers are resolved as real workspace paths. Use absolute paths returned
by file/status tools, or root-relative paths that the tool explicitly accepts.
Do not use git-style `--- a/algorithm.py` / `+++ b/algorithm.py`; those paths
can be interpreted literally or relative to the wrong root. For unified diffs,
write headers like:

```diff
--- /lustre/home/.../.cellcompass/training_algorithms/<algorithm_id>/algorithm.py
+++ /lustre/home/.../.cellcompass/training_algorithms/<algorithm_id>/algorithm.py
@@ ...
```

or use Codex patch format with `*** Update File: /absolute/path`. If a patch
fails, re-read the file and retry with the exact absolute path instead of
rewriting the whole file.

Code editing safety rule:

- Use `apply_workspace_patch(...)` for ordinary edits to `algorithm.py`,
  `claim_metric.py`, `config.yaml`, `IMPLEMENTATION_MAP.md`, and proposal files.
- Use `replace_workspace_file(...)` only when you intentionally provide the
  complete final content of the target file from the first line to the last
  line. Never use it to replace a single function, method, YAML stanza, or table
  section.
- After editing Python code, immediately run a syntax/import smoke check through
  the available tooling before training. If the file fails to parse, fix that
  first; do not start preview or campaign training.

Acceptable approximations are limited to semantics-preserving engineering
approximations such as mini-batch/stochastic estimators, vectorization, caching,
streaming, and numerically equivalent reparameterizations. They must still solve
the same mathematical problem stated by the proposal.

Do not make the implementation look faster by changing the evidence protocol:

- builtin flow-matching methods use 3000 epochs; keep the seeded 3000-epoch
  flow-matching default unless convergence evidence justifies a shorter schedule
- do not lower epochs merely to meet the wall-clock budget
- do not depend on a previously trained checkpoint, cached fitted state, or
  target-specific precomputed predictions unless the approved proposal defines
  that warm-start mechanism and it generalizes to new datasets
- solve runtime pressure, when it exists, with mini-batch/chunking,
  GPU/vectorized code, caching, streaming, or parallelization that preserves the
  proposal semantics
- design for large biological datasets even when the current trial is a tiny
  simulation; obvious dense all-pairs OT/UOT cost, plan, clone-mask, affinity, or
  kernel state is not acceptable campaign evidence for real-data algorithms

Implementation acceleration should preserve semantics:

- choose the simplest correct scale path: if an adjacent time gap has about
  1000 source cells and 1000 target cells, the resulting `1000 x 1000`
  cost/coupling matrix is usually small enough to solve as one full block, so
  tiny/small simulations usually do not need extra mini-batch complexity
- for larger real-data gaps, chunk/mini-batch before constructing solver state
- keep OT/UOT/WFR solver tensors on GPU when possible
- use streaming/sparse/landmark/coreset samplers when full pairwise state is too
  large
- cache invariant costs or metadata; parallelize independent per-gap work
- if the math changes, patch/review the proposal instead of hiding it as optimization

Default target:

- do not rewrite the training system
- do change the scientific algorithm while preserving the standard runtime

## Files You Own

Work only under:

- `~/.cellcompass/training_algorithms/<algorithm_id>/`

Before editing, make that algorithm the active authoring context:

- `activate_algorithm_workspace(algorithm_id="<algorithm_id>")`

If the write tool says the target algorithm does not match the active context,
do not bypass the restriction. Activate the correct algorithm first.

After intentional edits, the snapshot rule depends on the path:

- campaign tuning: do not snapshot manually before each trial; `run_campaign_trial(...)` does that automatically
- manual review/training: if the active context reports `dirty_since_snapshot=true`, call `snapshot_active_algorithm_workspace(...)` before review or `run_training(...)`

Use snapshots as clean-state bindings, not as the main tuning archive. Campaign
git archives are the tuning history.

Primary files:

- `PROPOSAL.md`
- `risk.md`
- `manifest.yaml`
- `algorithm.py`
- `config.yaml`
- `README.md`
- `IMPLEMENTATION_MAP.md`

`risk.md` has two reviewer-maintained sources: proposal-review risks and the
latest implementation-review risks. Treat it as an advisory diagnostic
watchlist for theory/design concerns, concrete code/config risks, scalability
risks, and optimization/debugging checks. It does not replace the approved
proposal as the implementation contract.

`IMPLEMENTATION_MAP.md` is not optional bookkeeping. After implementation,
fill it as the self-check that proves every proposal pseudocode step maps to
real code/config lines. If a row needs an approximation, explain why the
approximation preserves the same mathematical problem. If it does not, revise
the proposal before using the result as trusted evidence.

Do not mark proposal-required components as `deferred`, `Stage 2 refinement`,
or "implemented later" and then use the result as trusted Stage 1 evidence. A
deferred proposal component means the current workspace is a debug prototype,
not an implementation of the approved proposal. Either implement the component
with a semantics-preserving engineering acceleration, or patch/review the
proposal before preview, implementation review, campaign evidence, or stage
gate claims.

For flow-matching algorithms, add a short implementation-diagnosis section to
`IMPLEMENTATION_MAP.md` before trusted campaign evidence:

- required anchor baseline / nearest builtin anchor, such as `vgfm`, `wfrfm`,
  `balanced_ot_cfm`, `sf2m`, or `crufm`. This is mandatory: campaign baseline
  refresh will always include this builtin anchor even if other baseline choices
  are customized.
- exact package classes/functions reused, subclassed, or replaced
- whether a natural degeneration/ablation setting exists and what it is
- how to check that the implementation's runtime path is as efficient as the
  builtin pattern instead of solving OT/UOT/WFR repeatedly in the epoch hot path
- component opening order for diagnosis: coupling/cost, conditional path, mass,
  loss, inference, custom metric

If the proposal defines any custom claim metric, add a `Claim Metric Contract`
section to `IMPLEMENTATION_MAP.md` before preview or campaign evidence. For each
metric, explicitly map:

- proposal claim object;
- original motivating gap or failure mode the metric is meant to validate;
- prediction source from the generated trajectory or frozen post-hoc scorer;
- truth/observed source;
- grouping key;
- evaluation timepoints;
- label source and whether labels are evaluation-only;
- evidence that the source has the biological truth semantics being claimed;
- leakage controls;
- concrete code lines implementing each part.

Do not mark the implementation map complete if the truth source is only a vague
"observed distribution" or if a proposal-defined endpoint, held-out,
descendant, simulated, growth, perturbation, or fate target is silently replaced
by a different proxy.

Do not implement a weakened claim metric as a shortcut around a failing
algorithm. If the approved proposal was motivated by a specific real-data gap,
the implementation must keep the metric tied to that gap. A metric that only
measures a generic side effect, such as broad geometry concordance, smoothness,
spread, activity, or nonzero growth, is not a valid replacement for a
lineage/fate, growth, perturbation, condition, or modality claim unless the
approved proposal explicitly justifies it as necessary for the original claim
and defines controls that would fail when the original mechanism is absent.
Otherwise patch/review the proposal or design a controlled Stage 2 dataset that
directly observes the original claim.

Do not infer truth semantics from column names alone. The fact that an AnnData
object has `cell_type`, `label`, `lineage`, `barcode`, time-bin columns, or
nontrivial category counts only proves that those fields exist. It does not
prove that an initial-time label is terminal fate truth, that a lineage string
defines descendant endpoint truth, that current-state labels are held-out truth,
or that a prefix/clone relation is the intended ground-truth mapping. If the
claim metric needs terminal fate, descendant fate distributions,
lineage-truth endpoints, fate-concordant transitions, growth truth, or
perturbation truth, cite the approved proposal, benchmark card, dataset
registration, preprocessing contract, or dataset documentation that grants that
semantics. If no such contract exists, revise the metric or construct a
controlled simulation instead of treating the metric as valid.

This section is for debugging and review. It does not allow a simplified first
implementation: the code should still implement the approved proposal semantics.

## Required Read Order

Before any code edits, read these in order.

1. proposal
- `~/.cellcompass/training_algorithms/<algorithm_id>/PROPOSAL.md`
- `~/.cellcompass/training_algorithms/<algorithm_id>/risk.md`

2. local template
- `~/.cellcompass/training_algorithms/<algorithm_id>/algorithm.py`
- `~/.cellcompass/training_algorithms/<algorithm_id>/config.yaml`
- `~/.cellcompass/training_algorithms/<algorithm_id>/manifest.yaml`
- `~/.cellcompass/training_algorithms/<algorithm_id>/IMPLEMENTATION_MAP.md`

3. package docs
- `CytoBridge-main/docs/INDEX.md`
- `CytoBridge-main/docs/runtime/custom-algorithms/README.md`
- `CytoBridge-main/docs/runtime/flow-matching/README.md`
- `CytoBridge-main/docs/runtime/flow-matching/builtin-implementation-guide.md`

4. the relevant reference file(s) from this skill
- `references/runtime-hooks-and-editing.md`
- `references/default-fm-semantics.md`

Do not start from source by default.

After the docs pass and before the first code edit, do a closest-builtin source
pass. Pick the nearest builtin anchor, write it as `Required anchor baseline` in
`IMPLEMENTATION_MAP.md`, read its YAML and the source symbols
listed in `builtin-implementation-guide.md`, and make sure you understand each
module's responsibility: coupling/state construction, conditional path, mass
strategy, model heads, training loop, inference rollout, and metrics. Record
the builtin source pass in `IMPLEMENTATION_MAP.md`. If you cannot explain why a
custom module differs from the nearest builtin, reuse or subclass the builtin
module instead.

Before the first preview or trusted trial, do a runtime hot-path audit. For each
custom coupling/path/mass component, write down which work happens once in
`build_state` or equivalent setup and which work happens inside the epoch/batch
path such as `sample_pairs`, `loss`, or rollout. Expensive OT/UOT/WFR solves,
pairwise cost construction, graph building, landmark selection, nearest-neighbor
indices, and side-information joins should usually be precomputed, cached, or
streamed outside the per-epoch hot path unless the approved algorithm explicitly
requires online recomputation. If only the pairwise geometry changes, keep the
nearest builtin solver/sampler lifecycle and replace the cost module rather than
rewriting the whole training data flow. For real-scale pairwise geometry
changes, prefer `ChunkedTransportCouplingStrategy`: implement one bounded
`build_pairwise_cost_block(...)` on the target device and let the package split
chunks, solve OT/UOT once in `build_state(...)`, cache subplans, and sample pairs
during the epoch loop.

If a custom cost changes high-dimensional geometry, run one real-data
cost-scale diagnostic before trusting the campaign result. On a representative
adjacent-time block, compare custom-cost and nearest-builtin-cost quantiles,
finite ratio, device placement, solver time, and selected `reg`/`reg_m`. A
method that fits a tiny simulation but gives catastrophic W1 on a real
benchmark is often failing because the custom cost scale or regularization
overwhelms the builtin solver semantics, not because the proposal needs a new
optimizer. Prefer a small explicit geometry coefficient, fixed regularization,
or documented cost normalization once the scale is understood.
For metric costs of the form `(x-y)^T G(z) (x-y)`, do not materialize
`G(z)` for every source-target pair when `G` has a landmark, kernel, graph, or
feature expansion. Derive the quadratic form directly and compute it in
`O(pairs * expansion * latent_dim)` rather than `O(pairs * expansion *
latent_dim^2)`. In `IMPLEMENTATION_MAP.md`, record the complexity of the cost
block implementation and why the chosen factorization is mathematically
equivalent to the proposal cost. If a preview timeout happens after coupling
preflight passes but before metrics return, separate setup/cost time,
one-epoch training time, and rollout/evaluation time before changing algorithm
semantics.

`build_state` is still part of the runtime budget. Moving work out of the epoch
loop is necessary but not sufficient: the setup pass must finish on the intended
real benchmark scale.

Do not implement or submit a simplified builtin-like algorithm as the first
version. The first implementation should already contain every required
approved-proposal component: custom coupling/cost, conditional path, mass
mechanism, loss terms, inference rule, and metric hook when they are part of the
proposal. If a component is expensive, solve that with equivalent engineering:
chunking, streaming, sparse candidates, GPU/vectorization, caching, or a
proposal-approved estimator. Do not omit the component.

Degeneration/ablation settings are diagnostic controls inside a complete
implementation, not permission to leave components out. After the complete
implementation exists, expose neutral settings where natural, for example
`metric_lambda=0`, `path_correction_scale=0`, or an identity cost, to verify:

- preview chain inspection completes on the real-data benchmark panel;
- the coupling sampler returns valid pairs and mass targets;
- the epoch loop starts from scratch without warm-start shortcuts;
- the natural builtin-degenerate setting gives builtin-like runtime behavior.

Use those controls only for diagnosis. Restore the approved-proposal components
for trusted preview, implementation review, campaign trials, and report claims.
Run degeneration or component-ablation controls with manual `run_training(...)`
or `preview_training_run(...)`, not `run_campaign_trial(...)`; otherwise a
builtin-like diagnostic can be promoted as if it were the new algorithm. If the
diagnostic control is the desired final method, revise and review the proposal
before using campaign selection.
If setup times out before the first epoch, diagnose setup scale first; do not
spend trials tuning learning rate, epoch count, or downstream metrics.

If the proposal adds a trainable conditional path, geometry correction,
birth/death head, or any nonstandard neural module, put the module
on the model in `model_builder(...)` and make optimizer ownership explicit.
Then have the backend retrieve that same module from `build_context.model`.
Pass the retrieved submodule directly into the path object, and fail fast if it
is absent. Do not make the path store a whole model reference or tolerate
`model=None`; preview/campaign isolation may otherwise fail only when the first
batch is sampled.
Do not create a separate trainable module only inside
`flow_matching_backend_builder(...)`: it may not be optimized and can end up on
the wrong device. Backend builders should own coupling state, cached costs,
samplers, and non-trainable geometry objects; trainable neural modules should be
model-owned.

Before implementation review, write the trainable-module ownership evidence in
`IMPLEMENTATION_MAP.md`: module name, where it is attached to the model, which
trainer selection mechanism includes it (`cytobridge_trainable_parameters(...)`,
`model.cytobridge_component_modules`, `component_trainable_modules`,
`trainable_modules`, `extra_trainable_modules`, or ordinary model submodule),
and what smoke check or training log proves it receives gradients. If a
proposal-required module is intentionally frozen, state the proposal-approved
reason; otherwise a backend-local `nn.Module` is a bug, not an approximation.

For custom paths, verify the velocity target, not only the sampled position.
If you override `compute_mu_t(...)`, `sample_xt(...)`, or introduce a
time-nonlinear path, also override `compute_conditional_flow(...)` or implement
the analytic equivalent used by the backend. The builtin deterministic target
`x1 - x0` is only valid for linear paths. A proposal that claims geodesic,
bridge, spline, or other nonlinear path dynamics is not implemented faithfully
if training still uses straight-line velocity targets.
The velocity target is vector-valued. If you use autograd through a learned
path `x_s in R^d`, do not compute `grad(x_s.sum(), s)` and broadcast the
resulting `(batch, 1)` derivative across dimensions. That is the derivative of
the scalar coordinate sum, not `dx_s/ds`. Use an analytic derivative, a
finite-difference derivative on the full vector, or a controlled Jacobian/JVP
implementation that returns `(batch, latent_dim)`. Add a smoke check that
`u_t.shape == x0.shape` and that different latent dimensions can have different
derivative values.
Do not add an unconstrained second network head named like `df_dt` and treat it
as the time derivative of a learned path correction. A derivative head is only
valid if the architecture or loss explicitly enforces consistency with the
position correction. Otherwise compute the derivative from the same path
formula by analytic differentiation, finite differences, or JVP/autograd.
For multi-timepoint learned paths, keep the endpoint parameter as local
interval time `s in [0,1]`, but condition the path network on interval/global
time context. Use `set_interval_context(...)` or `set_interval(delta_t)` in the
path object rather than collapsing all gaps into the same local-time problem.

## Builtin Implementation Examples

When implementing a custom flow-matching algorithm, do not invent the package
integration pattern from scratch. After reading the approved proposal and
runtime docs, inspect the closest builtin implementation paths:

- source map: `CytoBridge-main/docs/runtime/flow-matching/builtin-implementation-guide.md`
- configs: `CytoBridge-main/CytoBridge/configs/{balanced_ot_cfm,sf2m,vgfm,wfrfm,crufm}.yaml`
- backend composition and coupling code: `CytoBridge-main/CytoBridge/tl/flow_matching_backends.py`
- conditional paths: `CytoBridge-main/CytoBridge/tl/flow_matching.py`
- loss/training application: `CytoBridge-main/CytoBridge/tl/trainer.py`
- hook examples: `CytoBridge-main/docs/runtime/flow-matching/examples-and-hooks.md`
- extension guide: `CytoBridge-main/docs/runtime/flow-matching/extension-points.md`

For chunked OT/UOT/WFR specifically, read these source symbols before writing
custom coupling code:

- `BalancedOTCouplingStrategy`
- `UnbalancedOTCouplingStrategy`
- `WFROETCouplingStrategy`
- `ChunkedTransportCouplingStrategy`
- `_split_transport_chunks(...)`
- `_solve_uot_from_pairwise_cost(...)`
- `_solve_balanced_ot_from_pairwise_cost(...)`
- `_build_balanced_coupling(...)`, `_build_sf2m_coupling(...)`,
  `_build_unbalanced_coupling(...)`, `_build_wfr_coupling(...)`

Prefer reusing or subclassing these package strategies. If the custom method
only changes pairwise geometry and the full adjacent-pair cost fits, subclass
`CostBasedPairwiseOTCouplingStrategy`. If the full adjacent-pair cost does not
fit, subclass `ChunkedTransportCouplingStrategy` instead of writing a custom
sampler from scratch. Keep the builtin solver/sampler path when it preserves the
approved semantics. If the custom method needs different marginal relaxation, mass
semantics, stochastic path, or runtime inference behavior, make that difference
explicit in `IMPLEMENTATION_MAP.md` and patch/review the proposal if it changes
the approved algorithm.

Scale rule: do not over-engineer small gaps. If each adjacent time-pair has
about 1000 source cells and 1000 target cells, the resulting `1000 x 1000`
cost/coupling matrix is normally acceptable as one full block and corresponds
to `chunk_size ~= 1000`. If a real benchmark has substantially larger adjacent
pairs, for example Weinreb-scale gaps, use `ChunkedTransportCouplingStrategy`
or a sparse/streaming store before claiming scalability.

Coupling state has one authoritative interface: `CouplingPlanStore` inside
`CouplingState(plan_stores=[...])`. New code should not create dummy
`state.plans`, write `sampling_info_plans`, or make callers understand two
parallel state representations. Runtime preview/training should accept any
coupling whose `backend.prepare(...)` succeeds and whose `sample_pairs(...)`
returns valid batches; static mini-batch markers are advisory only. If the
builtin chunked strategy is close, reuse it and override only the cost/solver
component that is scientifically different. If the proposal defines a sparse
support graph rather than a full block cost, store the graph with
`CouplingPlanStore.from_edges(...)` or
`TransportPlanBuilder.edges_to_store(...)`; do not write a private sampler
unless the package store cannot represent the approved semantics.
Do not construct a full `n_source x n_target` cost just to pass it to
`TransportPlanBuilder.solve_cost_to_store(...)`; that builder standardizes
solver-to-sampler glue, not memory scaling. For real benchmark-scale adjacent
pairs, implement `ChunkedTransportCouplingStrategy.build_pairwise_cost_block(...)`
so the package builds and solves bounded blocks.
Do not recover scalability-sensitive state by stitching
`CouplingPlanStore.sub_plans` back into one global `source_n x target_n`
matrix. Chunked stores are intentionally non-dense; custom logic that needs
row transitions, chained samples, terminal mass, or support statistics should
iterate the store's bounded chunks, use `sample_pairs(...)`, or convert a
bounded/sparse support into `from_edges(...)`. A simulation preview can pass
with full dense reconstruction and still be invalid for Weinreb-scale data.

For custom networks, use the package's explicit trainable-head interface:

- `model.cytobridge_component_modules = {"growth": ["birth_head", ...]}`
- stage `trainable_modules: ["birth_head", ...]`
- or advanced `cytobridge_trainable_parameters(stage_params=..., train_flags=...)`

Do not attach unrelated source/sink/birth/branch heads inside `growth_net` or
`interaction_net` solely to make the optimizer train them.

## Reference Index

Read `references/runtime-hooks-and-editing.md` when you need:

- the concrete pre-read workflow
- the source escalation order
- the four runtime axes
- hook selection rules
- edit workflow and config discipline
- scalability / mini-batch expectations

Read `references/default-fm-semantics.md` when you need:

- the exact builtin FM stack
- current UOT solver semantics
- what `reg`, `reg_m`, and `alpha_regm` mean
- builtin pair sampling semantics
- builtin mass / `gt` / `loss_weights` semantics
- builtin loss / evaluation semantics
- why `v` and `g` are coupled scientifically

If you are reproducing an external paper, read both reference files before you code.

## Runtime Contracts

Treat these as fixed:

Scalability is also a fixed contract. A custom algorithm should remain credible
on large real benchmarks unless the approved proposal explicitly limits its data
contract. A `use_mini_batch` or `chunk_size` flag is not enough by itself; the
actual hot path must avoid full adjacent time-pair dense state when
`n_t * n_{t+1}` is large. Conversely, if a selected benchmark gap has only
about 1000 source cells and 1000 target cells, its `1000 x 1000` matrix is
acceptable as full cost and the agent should not spend trial budget inventing
unnecessary mini-batch machinery.

- `adata.obs["time_point_processed"]`
- `adata.obsm["X_latent"]`

Extra modalities may augment the algorithm, but should not replace those contracts.

## The Four Runtime Axes

Do not mix these:

1. `model.components`
- which heads exist

2. `train_strategy`
- which existing heads are regressed in the current stage

3. `MassStrategy`
- terminal mass target per sampled source particle over one local interval

4. `conditional path`
- path geometry, stochasticity, local interpolation, local mass dynamics

If you are changing coupling/path/mass semantics, read `references/default-fm-semantics.md` before editing.

## Hook Selection Cheat Sheet

Use the smallest valid hook.

1. `model_builder(...)`
- when the model should not be the builtin `DynamicalModel`

2. `stage_runner(...)`
- when a training stage needs custom forward/loss/backward logic
- if the stage needs custom batches/losses but not custom optimizer semantics,
  prefer `CytoBridge.tl.custom_stage_loop.run_custom_stage_loop(...)` inside
  the runner instead of hand-writing the epoch loop
- if a custom runner still loops over epochs/steps, poll `context.should_stop()`
  when available and stop cleanly so package-level time budgets still work
- this is the main override for nonstandard FM, custom `neural_ode`, higher-order dynamics, or trajectory-level objectives
- if the runner only handles an extra custom stage, return `None` for standard
  `flow_matching`/`neural_ode` stages; otherwise you may accidentally skip the
  package training loop and produce a 0-epoch run

3. `inference_context_builder(...)`
- when custom evaluation-time prediction needs algorithm-specific inference inputs
- return a flexible `InferenceContext(payload, visibility, provenance, notes)`
- payload fields are not fixed; provenance should show t=0/exogenous/model-state sources

4. `simulation_hook(...)`
- when evaluation-time prediction should not use the builtin simulator
- use `sim_context.inference_context` when an inference context builder is present
- prediction must roll forward from t=0/exogenous/model-state inputs, not future truth
- return the full trajectory requested by `sim_context.trajectory_time_points`, not only the observed future time points; metrics are computed from trajectory slices

5. `build_pairwise_cost(...)`
- default FM choice when only adjacent-time pairwise transport geometry changes
  and the full pairwise cost matrix is acceptable for the target scale

6. `training_data_builder(...)`
- when raw `adata` or aligned side modalities must be prepared

7. `build_flow_matching_backend(build_context)`
- when coupling/path/mass/backend composition changes

8. `flow_matching_loss_hook(...)`
- additive differentiable loss, or explicit replacement of one builtin
  component loss via `FlowMatchingLossResult(replace_velocity_loss=...)`,
  `replace_growth_loss=...`, or `replace_score_loss=...`
- builtin `velocity_net`, `growth_net`, and `score_net` inputs are concatenated
  `[x, t]`; use `loss_context.net_input` directly or
  `loss_context.x_input` / `loss_context.t_input`, not `[t, x]`

9. `evaluation_metrics_hook(...)`
- additive metrics only; never replace builtin `W1/TMV`
- put expected numeric ranges for custom metrics in
  `evaluation_metrics_params.expected_ranges` when known, for example
  `{"custom_auroc": {"min": 0, "max": 1}}`
- `preview_training_run(...)` will run a 1-epoch smoke test and warn if custom
  metrics are NaN/inf or outside declared/inferred ranges; ignore exact
  1-epoch magnitudes

If hook choice is still unclear after this summary, read `references/runtime-hooks-and-editing.md`.

## Inference Integrity Rule

Custom inference may use t=0 cells, t=0 aligned modalities, known exogenous
conditions, constants/priors, time grid, config, and trained model state.
It must not read future observed cells, future cell counts, future total mass,
future modalities, or future distribution statistics to construct predictions.

Trusted metrics are computed from one sealed `EvaluationTrajectory`: a dense
t0-to-final rollout from the trained dynamics model. Builtin W1/TMV and any
claim metric must read that trajectory. Do not implement a metric-specific
shortcut predictor or a custom metric that bypasses the trajectory artifact.

If the approved proposal models unbalanced mass, predicted cell-count / total
mass changes must be produced during rollout by the approved mass mechanism:
for example a learned growth head integrated as `d log w / dt`, WFR/UOT/SB
birth-death dynamics, proliferation/death rules, condition-driven growth priors,
or another proposal-approved biological/dynamical mechanism. Do not add
post-hoc weight rescaling, target-count lookup, time-only count-ratio clocks, or
renormalization layers whose only purpose is to make TMV pass. Do not ignore the
learned growth output at inference and replace it with a separate mass
correction unless the proposal explicitly approved that mechanism and reviewer
accepted its biological meaning.

If the proposal uses explicit birth/death or splitting, the simulator may return
a variable number of particles per trajectory slice with unit weights. The first
slice should also use unit weights; package TMV compares predicted total mass
relative to the first predicted slice against observed relative cell-count
change. This is a supported alternative to fixed-count weighted particles.

If you change `simulation_hook(...)` or `inference_context_builder(...)`, expect
an automatic read-only inference review before trusted campaign/manual training.
Changing only an additive claim metric definition/evaluator does not by itself
trigger that reviewer; the metric is judged later by campaign evidence and
baseline comparability.

## Source Escalation Order

Only inspect source after the docs and references are insufficient.

1. `CytoBridge-main/CytoBridge/tl/training_algorithm.py`
2. `CytoBridge-main/CytoBridge/tl/fit.py`
3. `CytoBridge-main/CytoBridge/tl/flow_matching_backends.py`
4. `CytoBridge-main/CytoBridge/tl/trainer.py`
5. `CytoBridge-main/CytoBridge/utils/utils.py`

## Minimal Workflow

1. re-read the approved proposal
2. identify the smallest hook that can realize it
3. keep proposal pseudocode traceable to implementation
4. patch `algorithm.py` / `config.yaml`
5. do not train yet
6. hand off to:
- `~/.cellcompass/skills/algorithm/review-and-training/SKILL.md`

## Hard Rules

- do not invent helper methods unless the runtime actually calls them
- do not silently change balanced/unbalanced semantics
- do not silently change deterministic/stochastic semantics
- do not redefine builtin `W1` or `TMV`
- do not confuse growth-rate targets with training loss weights
- do not judge success from an obviously undertrained run
