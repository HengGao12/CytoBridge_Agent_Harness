---
title: "Flow matching examples and hooks"
summary: "Implementation examples and hook usage patterns for custom flow matching."
read_when:
  - "Looking for flow-matching hook examples"
---
# Flow-Matching Examples And Hooks

Minimal custom coupling examples, backend factory context, training data builder, and loss hook.

## 14. Minimal Example: Small-Data Pairwise Custom Coupling

This example is intentionally small-data. It changes only the pairwise cost and
keeps the package solver, path, mass, and sampler. It still materializes a full
`(n0, n1)` cost matrix, so it is not a large-data scalability pattern.

```python
from __future__ import annotations

import numpy as np
import torch

from CytoBridge.tl.flow_matching_backends import (
    CostBasedPairwiseOTCouplingStrategy,
    FlowMatchingBackend,
    PairwiseCost,
    RegularizedUnbalancedConditionalPath,
    UOTMassStrategy,
)


class CustomCoupling(CostBasedPairwiseOTCouplingStrategy):
    def __init__(self, *, chunk_size: int = 1000, alpha_regm: float = 1.0):
        super().__init__(
            solver_mode="uot",
            chunk_size=chunk_size,
            use_mini_batch=True,
            alpha_regm=alpha_regm,
            reg_strategy="per_time",
        )

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
        del t0, t1, device
        x0_sq = np.sum(x0 * x0, axis=1, keepdims=True)
        x1_sq = np.sum(x1 * x1, axis=1, keepdims=True).T
        cost_matrix = np.maximum(x0_sq + x1_sq - 2.0 * (x0 @ x1.T), 0.0)
        return PairwiseCost(cost_matrix=cost_matrix.astype(np.float32), metadata={"time_idx": time_idx})


backend = FlowMatchingBackend(
    coupling=CustomCoupling(chunk_size=1000),
    path=RegularizedUnbalancedConditionalPath(sigma=0.1),
    mass=UOTMassStrategy(),
)
```

Interpretation of that default:

- `CustomCoupling` changes pair selection only
- `UOTMassStrategy` still defines the terminal mass target from UOT row sums
- `RegularizedUnbalancedConditionalPath` still defines the local geometry,
  noise schedule, and local log-mass dynamics
- full-cost helpers are for small/medium data; for real biological adjacent
  pairs, use `ChunkedTransportCouplingStrategy.build_pairwise_cost_block(...)`
  so the package builds bounded cost blocks, solves each block once, and samples
  from `CouplingPlanStore`

## 15. Advanced Example: Large-Data Coupling Skeleton

Most large-data pairwise-cost changes should subclass
`ChunkedTransportCouplingStrategy` rather than writing a sampler. Override
`build_state(...)` and usually `sample_pairs(...)` directly only when you need
cross-time coordination or global/sparse state that the chunked-cost API cannot
express.

That is the advanced path. Use it when the standard chunked-cost API would still
require dense adjacent-pair state or would change the approved coupling
semantics.

Use it for:

- global regularization sharing across all pairs
- nonlocal constraints
- reusable global caches
- streaming or sparse candidate transport
- landmark / coreset summaries

Minimal shape:

```python
import numpy as np
import torch

from CytoBridge.tl.flow_matching_backends import CouplingPlanStore, CouplingState, CouplingStrategy


class StreamingCoupling(CouplingStrategy):
    supports_minibatch = True

    def build_state(self, X, t_train, device):
        plan_stores = []
        metadata = {"kind": "streaming_custom_coupling", "chunks": []}
        for time_idx in range(len(X) - 1):
            # Build only bounded-size chunk/sparse metadata here.
            # Do not create full n_source x n_target cost or mask arrays.
            info_i = build_sampler_metadata_for_gap(X[time_idx], X[time_idx + 1], time_idx=time_idx)
            plan_stores.append(
                CouplingPlanStore.from_chunked(
                    source_n=X[time_idx].shape[0],
                    target_n=X[time_idx + 1].shape[0],
                    sub_plans=info_i["sub_plans"],
                    source_groups=info_i["source_groups"],
                    target_groups=info_i["target_groups"],
                    terminal_mass_rows=info_i.get("terminal_mass_rows"),
                    terminal_mass_subplans=info_i.get("terminal_mass_subplans"),
                    metadata=info_i.get("diagnostics", {}),
                )
            )
            metadata["chunks"].append(info_i.get("diagnostics", {}))
        return CouplingState(plan_stores=plan_stores, metadata=metadata)

    def sample_pairs(self, state, X, time_idx, batch_size, device):
        return state.plan_store(time_idx).sample_pairs(
            X=X,
            time_idx=time_idx,
            batch_size=batch_size,
            device=device,
        )
```

The exact metadata is algorithm-specific. The invariant is that `build_state`
creates a `CouplingPlanStore` with the same approved coupling semantics without
full dense adjacent-pair state, and `sample_pairs` samples from that store.
For ordinary pairwise-cost algorithms, prefer subclassing
`ChunkedTransportCouplingStrategy` instead of writing this skeleton by hand.

If the approved method defines a sparse support graph rather than rectangular
chunks, use the edge-list store instead of hand-writing a sampler:

```python
from CytoBridge.tl.flow_matching_backends import CouplingPlanStore

store = CouplingPlanStore.from_edges(
    source_n=x0.shape[0],
    target_n=x1.shape[0],
    edge_src=edge_src,
    edge_tgt=edge_tgt,
    edge_weight=edge_weight,
    terminal_mass_rows=row_terminal_mass,
    metadata={"support": "custom_sparse_graph"},
)
```

`edge_weight` is transport mass on each candidate edge. Sampling, index lookup,
and terminal-mass lookup are then handled by `CouplingPlanStore`.

## 16. Backend Factory Context

If your backend construction needs access to aligned covariates, extra
modalities, or other runtime metadata, use `FlowMatchingBuildContext`.

```python
from CytoBridge.tl.training_algorithm import FlowMatchingBuildContext


def build_backend(build_context: FlowMatchingBuildContext):
    stage_params = build_context.stage_params
    training_data = build_context.training_data
    device = build_context.device
    ...
```

Use this when:

- coupling depends on cell metadata aligned through `training_data_builder(...)`
- coupling depends on extra modalities carried in `training_data.extra_modalities_by_time`
- you need runtime config/model metadata during backend construction

Do not hard-code `h5ad` reload paths inside the algorithm when the standard
runtime can pass aligned data through `TrainingDataBundle`.

Do not use backend construction as the first place to do AnnData munging.
If the algorithm needs raw adata-derived aligned features, prepare them in
`training_data_builder(...)` first.

## 17. Training Data Builder and Loss Hook

Custom algorithms can extend the standard runtime without replacing `fit(...)`.

Use:

- `training_data_builder(...)` to attach extra aligned modalities
- `flow_matching_loss_hook(...)` to add an auxiliary differentiable loss or
  explicitly replace one builtin component loss

Keep these package-level contracts stable:

- `adata.obs["time_point_processed"]` remains the canonical time field
- transcriptomic training should still treat `adata.obsm["X_latent"]` as the primary representation and backbone state space
- extra modalities should augment `X_latent`, not silently replace it

Interpret the new hooks as:

- a way to enrich training inputs while keeping the same fit loop
- a way to inject auxiliary supervision or replace a named velocity/growth/score
  component without replacing the whole trainer
- not a license to redefine the package's primary transcriptomic representation

Good uses:

- attach auxiliary priors, extra modalities, or covariates while keeping `X_latent` as the main transcriptomic state
- add an extra regularizer or consistency loss on top of the default FM objective
- replace `velocity_loss`, `growth_loss`, or `score_loss` with a proposal-defined
  scalar using `FlowMatchingLossResult(replace_velocity_loss=...)` and the
  standard `[x, t]` model input convention

Bad uses:

- replacing transcriptomic training with another primary matrix while still claiming package compatibility
- inventing a different canonical time field while reusing the same workflow
- replacing the whole epoch loop instead of expressing the change through the public hooks

Default preference order:

1. only change `build_pairwise_cost(...)`
2. if needed, add `training_data_builder(...)`
3. if needed, use `FlowMatchingBuildContext`
4. only then add `flow_matching_loss_hook(...)`

Inside a loss hook, builtin heads take `loss_context.net_input` in `[x, t]`
order. Prefer `loss_context.x_input` and `loss_context.t_input` over manual
slicing.
