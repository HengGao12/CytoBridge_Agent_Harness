---
name: pot-optimal-transport
description: Use when reasoning about or implementing Optimal Transport with the Python Optimal Transport (POT) library. Helps choose between balanced, unbalanced, entropic, GW, and FGW solvers and avoid mixing up marginal relaxation, plan regularization, plan-vs-cost outputs, and cross-domain OT vs structural matching.
---

# POT Optimal Transport

Use this skill when a task depends on the **Python Optimal Transport (POT)** library.

## Use it for
- choosing the right POT solver for a written OT objective
- checking whether a solver is balanced or unbalanced
- checking whether a function returns a **plan** or a **cost**
- distinguishing ordinary OT from **GW / FGW**
- verifying whether regularization acts on the **plan**, the **marginals**, or both

## Minimal decision workflow
Before selecting a solver, answer these questions:

1. Are the marginals enforced exactly (**balanced**) or penalized (**unbalanced**)?
2. Is the objective exact or regularized?
3. Is there a cross-domain cost matrix `M`, or are we matching within-domain structures `C1, C2`?
4. Do we need the transport **plan** or only a scalar **cost**?
5. Does the formula include a penalty on the plan itself, on the marginals, or both?

## Quick solver map
- **Exact balanced OT**: `ot.emd`, `ot.emd2`
- **Entropic balanced OT**: `ot.sinkhorn`, `ot.sinkhorn2`
- **Stable entropic balanced OT**: `sinkhorn_log`, `sinkhorn_stabilized`, `sinkhorn_epsilon_scaling`
- **Unbalanced OT via generalized Sinkhorn**: `ot.unbalanced.sinkhorn_unbalanced`, `...2`
- **Unbalanced OT with explicit control of plan regularization**: `ot.unbalanced.mm_unbalanced`, `...2`
- **Flexible unbalanced OT via SciPy optimizer**: `ot.unbalanced.lbfgsb_unbalanced`
- **Structure-only matching**: `ot.gromov.gromov_wasserstein`, `entropic_gromov_wasserstein`
- **Feature + structure matching**: `ot.gromov.fused_gromov_wasserstein`, `entropic_fused_gromov_wasserstein`

## Rules of interpretation
- `emd` = exact balanced OT
- `sinkhorn` = entropic balanced OT
- `sinkhorn_unbalanced` = unbalanced OT with generalized Sinkhorn-style regularized formulation
- `mm_unbalanced` = unbalanced OT with explicit divergence choices and explicit `reg` control
- `GW` = relational / structural matching, not ordinary OT on a shared `M`
- `FGW` = direct feature cost plus structural matching
- `foo(...)` usually returns a **plan**
- `foo2(...)` usually returns a **cost**

## When to read more
If exact solver semantics matter, read:
- `references/solver-semantics.md`

Read that reference especially when:
- matching a paper equation to a POT solver
- choosing between `sinkhorn_unbalanced` and `mm_unbalanced`
- deciding between OT, GW, and FGW
- checking what a regularization parameter actually means

## Verification rule
Do not rely on memory when solver semantics matter. Inspect the live environment:

```python
import inspect, ot
from ot import unbalanced, gromov

print(ot.__version__)
print(inspect.signature(unbalanced.mm_unbalanced))
print(inspect.getdoc(unbalanced.mm_unbalanced))
```

Use signatures and docstrings from the installed version before making theorem-level claims about a solver.
