---
title: "Flow matching checklist and failures"
summary: "Agent checklist and common failure modes for flow-matching extensions."
read_when:
  - "Reviewing flow-matching implementation before training"
  - "Debugging flow-matching extension failures"
---
# Flow-Matching Agent Checklist And Failure Modes

Use this page immediately before preview/training and when a flow-matching
implementation behaves unexpectedly.

## Implementation Checklist

1. State the closest builtin family: `balanced_ot_cfm`, `sf2m`, `vgfm`,
   `wfrfm`, `crufm`, `dynamical_ot`, `ruot`, or other.
2. Read the closest builtin YAML and source symbols listed in
   `builtin-implementation-guide.md`; record that source pass in
   `IMPLEMENTATION_MAP.md`.
3. State the changed component: coupling, path, mass, loss, model,
   simulation, metric, or full stage loop.
4. Verify the fixed data contract:
   - `adata.obs["time_point_processed"]`
   - `adata.obsm["X_latent"]`
5. If extra modalities are used, route them through `training_data_builder`
   into `TrainingDataBundle.extra_modalities_by_time`.
6. If the method changes only small/medium pairwise geometry, use
   `build_pairwise_cost(...)`.
7. If the method must scale to large adjacent time pairs, verify that the
   actual hot path avoids full dense cost/mask/plan state.
8. If the method is unbalanced, verify mass strategy, growth head, and TMV
   gate semantics are aligned.
9. If the method is balanced-only, mark TMV as diagnostic, not a rejection gate.
10. If custom inference is used, return a t0-to-final trajectory and keep
   builtin metrics based on trajectory slices.
11. Fill `IMPLEMENTATION_MAP.md` with exact proposal-step-to-code mappings
    before trusted training.

## Scalability Checklist

For every adjacent time pair the reviewer should ask:

- What is the expected `n_source * n_target` on realistic data?
- Does the implementation construct dense full-pair cost, mask, kernel, or
  plan arrays?
- Does `use_mini_batch=True` only chunk solver/sampling after dense arrays
  already exist?
- Is there a true streaming, sparse, landmark, coreset, chunked-cost, or
  stochastic estimator path?
- Are sampler metadata and terminal mass targets consistent with the approved
  coupling semantics?

Toy-data success is not enough evidence for biological-scale deployment.
However, static suspicion should remain a warning unless the implementation
review or runtime evidence proves an actual contract/scalability failure.

## Common Failure Modes

- Treating `use_mini_batch=True` as proof of memory scalability.
- Returning `PairwiseCost` while constructing dense auxiliary masks or
  `n_source x n_target` cost matrices for large real data.
- Filtering forbidden transitions after an unrelated dense coupling was already
  solved.
- Reloading adata from a hard-coded path instead of using runtime contexts.
- Replacing `X_latent` with another modality while claiming package contract
  compatibility.
- Using balanced mass assumptions while leaving growth learning enabled.
- Changing path and mass strategy inconsistently.
- Hiding the main algorithm inside `flow_matching_loss_hook(...)` instead of
  implementing the intended coupling/path/mass logic.
- Replacing a builtin loss by returning `extra_loss = custom_loss -
  guessed_builtin_loss`; use the explicit `replace_velocity_loss`,
  `replace_growth_loss`, or `replace_score_loss` fields instead.
- Splitting `loss_context.net_input` as `[t, x]`; builtin heads use `[x, t]`.
- Computing claim metrics from a shortcut predictor rather than the sealed
  model-generated trajectory.
- Redefining builtin `W1` or `TMV` inside a custom metric hook.

## When To Revise The Proposal

Revise the proposal rather than silently simplifying implementation when:

- the approved algorithm requires a solver convention not provided by the
  package helper;
- large-data scalability requires an estimator or approximation not justified
  in the proposal;
- the needed inference inputs include information not allowed at t0;
- the implementation cannot preserve the proposed recoverability argument.

## Related Documents

- `CytoBridge-main/docs/runtime/flow-matching/README.md`
- `CytoBridge-main/docs/runtime/flow-matching/builtin-implementation-guide.md`
- `CytoBridge-main/docs/runtime/flow-matching/semantics-and-callgraph.md`
- `CytoBridge-main/docs/runtime/flow-matching/extension-points.md`
- `CytoBridge-main/docs/runtime/custom-algorithms/README.md`
- `CytoBridge-main/docs/runtime/data-and-evaluation-contracts.md`
