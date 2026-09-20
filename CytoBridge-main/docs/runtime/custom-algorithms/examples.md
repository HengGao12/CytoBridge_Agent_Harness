---
title: "Custom algorithm examples"
summary: "Examples and templates for custom algorithm implementation patterns."
read_when:
  - "Looking for custom algorithm examples or templates"
---
# Custom Algorithm Examples

Minimal builtin override and flow-matching backend examples.

## 7. Minimal Builtin-Override Example

This example keeps the workspace config as base and only changes config values:

```python
from CytoBridge.tl.training_algorithm import (
    TrainingAlgorithmContext,
    TrainingAlgorithmSpec,
)


def build_training_algorithm(context: TrainingAlgorithmContext) -> TrainingAlgorithmSpec:
    return TrainingAlgorithmSpec(
        algorithm_id=context.algorithm_id,
        base_config="./config.yaml",
        config_overrides={
            "training.defaults.lr": 5e-4,
            "training.plan.0.epochs": 150,
        },
        notes="Lower learning rate and longer first stage.",
    )
```

## 8. Minimal Flow-Matching Backend Example

Preferred pattern: define a custom cost-based coupling strategy and keep the
package defaults for state assembly, pair sampling, path, and mass unless you
have a clear reason to change them.

Interpret the defaults correctly:

- `UOTMassStrategy()` does not directly define the local growth rate
- it defines the terminal mass target for each sampled source particle over one
  adjacent interval
- `RegularizedUnbalancedConditionalPath(...)` then defines how that target is
  traversed over local time and what two outputs follow from that path:
  - the growth-rate target `gt`
  - the training loss weight
- the default growth mathematics therefore remain:
  - terminal mass target from UOT
  - local log-mass dynamics from the path
  - learned `g(t, x)` interpreted as a rate `d/dt log w`
- the training loss weight is a separate optimization-time reweighting term and
  should not be interpreted as the current path mass `w(t)`
- under the builtin default UOT solver, source/target reference marginals are
  `a_i = 1` and `b_j = 1`, so sampled row sums already represent the correct
  terminal mass ratio for a source particle normalized to unit local mass

```python
from __future__ import annotations

import numpy as np
import torch

from CytoBridge.tl.flow_matching_backends import (
    CostBasedPairwiseOTCouplingStrategy,
    FlowMatchingBackend,
    FlowMatchingBuildContext,
    PairwiseCost,
    RegularizedUnbalancedConditionalPath,
    UOTMassStrategy,
)
from CytoBridge.tl.training_algorithm import (
    TrainingAlgorithmContext,
    TrainingAlgorithmSpec,
)


class CustomCoupling(CostBasedPairwiseOTCouplingStrategy):
    """Small/medium-data example; this API still builds a full cost matrix."""

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
        return PairwiseCost(
            cost_matrix=cost_matrix.astype(np.float32),
            metadata={"time_idx": time_idx},
        )


def build_flow_matching_backend(build_context: FlowMatchingBuildContext) -> FlowMatchingBackend:
    stage_params = build_context.stage_params
    training_data = build_context.training_data
    del training_data
    backend = FlowMatchingBackend(
        coupling=CustomCoupling(chunk_size=int(stage_params.get("chunk_size", 1000))),
        path=RegularizedUnbalancedConditionalPath(sigma=float(stage_params.get("sigma", 0.1))),
        mass=UOTMassStrategy(),
    )
    return backend


def build_training_algorithm(context: TrainingAlgorithmContext) -> TrainingAlgorithmSpec:
    return TrainingAlgorithmSpec(
        algorithm_id=context.algorithm_id,
        base_config="./config.yaml",
        flow_matching_backend_builder=build_flow_matching_backend,
        notes="Custom pairwise coupling with default path/mass behavior.",
    )
```

Only override `build_state(...)` directly when your coupling requires global
cross-time coordination or state that the standard chunked-cost API cannot
express. For small/medium pairwise costs, implement `build_pairwise_cost(...)`.
For real-data adjacent-pair costs, prefer
`ChunkedTransportCouplingStrategy.build_pairwise_cost_block(...)` so the package
owns chunk splitting, solver calls, cached plan stores, and sampling.

Important interpretation:

- if you need raw `adata.obs` / `adata.obsm` access, use `training_data_builder(...)`
- do not change `MassStrategy` just to alter local growth interpolation; that
  belongs in the conditional path
- if you are already inside `build_flow_matching_backend(...)`, prefer consuming
  `build_context.training_data`
- do not treat the backend builder as the place to do general AnnData management
- before approving a custom algorithm proposal, state explicitly why the chosen
  coupling/path/mass split can recover the observed late marginal in the
  exact-fit limit
