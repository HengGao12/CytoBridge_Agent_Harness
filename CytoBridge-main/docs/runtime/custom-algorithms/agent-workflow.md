---
title: "Custom algorithm agent workflow"
summary: "Agent-facing sequence for implementing approved custom algorithms."
read_when:
  - "Following the custom algorithm authoring workflow"
---
# Custom Algorithm Agent Workflow

How the agent consumes custom workspaces, data builders, loss hooks, and anti-patterns.

## 9. How the Agent Consumes This

The agent-side training stack will:

1. read `manifest.yaml`
2. import `algorithm.py`
3. call `build_training_algorithm(context)`
4. resolve `base_config`
5. apply `config_overrides`
6. call `cb.tl.fit(...)` with public hooks when present
7. snapshot your algorithm directory into the run bundle

That means:

- your editable algorithm stays in `~/.cellcompass/training_algorithms/...`
- the exact version used for a run is copied into `algorithm_snapshot/`

## 10. Training Data Builder and Loss Hook

Most custom algorithms should stay inside the standard runtime and extend it
with narrow hooks.

Use `training_data_builder(...)` when:

- the algorithm needs extra aligned modalities beyond `obsm["X_latent"]`
- the backend needs covariates, lineage priors, morphology features, or other
  side channels arranged per time bucket

Use `flow_matching_loss_hook(...)` when:

- the algorithm needs an auxiliary differentiable loss
- the default FM objective should remain intact, with extra supervision added on top

Keep the shared workflow contract stable:

- do not replace `obs["time_point_processed"]`
- do not silently swap transcriptomic training away from `obsm["X_latent"]`
- use extra modalities as additional conditioning/context

Preferred escalation order:

1. `build_pairwise_cost(...)` only when full pairwise costs fit memory
2. `ChunkedTransportCouplingStrategy.build_pairwise_cost_block(...)` when the
   cost is still adjacent-pair/block-decomposable on real data
3. custom `CouplingStrategy.build_state(...)` with `CouplingPlanStore` only
   when streaming/sparse/global state is not expressible by the chunked-cost API
4. `training_data_builder(...)`
5. `flow_matching_backend_builder(build_context)`
6. `flow_matching_loss_hook(...)`
7. `stage_runner(...)` with `run_custom_stage_loop(...)` when custom
   batches/losses are required but optimizer/checkpoint/device/timeout
   semantics should remain package-owned

## 11. What Not To Do

Do not:

- edit package builtin YAMLs for one-off experiments
- write custom algorithm code into `CytoBridge-main/CytoBridge/configs/`
- return both backend and backend factory at the same time
- mutate `context.resolved_base_config` in place and assume the caller will reuse it

If you need to override coupling/path/mass details, use the extension types
described in `docs/runtime/flow-matching/README.md`.
