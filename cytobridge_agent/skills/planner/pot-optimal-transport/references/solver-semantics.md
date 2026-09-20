# POT Solver Semantics Reference

Use this reference when a task needs a more exact mapping from a mathematical OT objective to a **Python Optimal Transport (POT)** solver.

## Contents
1. Balanced OT
2. Unbalanced OT
3. GW and FGW
4. Plan vs cost outputs
5. Common mistakes
6. Live verification checklist

---

## 1. Balanced OT

### `ot.emd(a, b, M)`
Solves the classic balanced Kantorovich problem:

\[
\min_\gamma \langle \gamma, M \rangle
\quad \text{s.t.}\quad
\gamma \mathbf{1}=a,\; \gamma^T\mathbf{1}=b,\; \gamma\ge 0.
\]

Meaning:
- exact balanced OT
- no entropic smoothing
- returns the **transport plan**

Companion:
- `ot.emd2(...)` returns the **cost**

### `ot.sinkhorn(a, b, M, reg, ...)`
Solves entropic-regularized balanced OT:

\[
\min_\gamma \langle \gamma, M \rangle + \mathrm{reg}\,\Omega(\gamma)
\quad \text{s.t.}\quad
\gamma \mathbf{1}=a,\; \gamma^T\mathbf{1}=b,\; \gamma\ge 0.
\]

Meaning:
- still balanced
- adds regularization on the **plan**
- typically faster/smoother than exact LP methods

Companion:
- `ot.sinkhorn2(...)` returns the **cost**

### Stabilized entropic balanced variants
- `ot.bregman.sinkhorn_log(...)`
- `ot.bregman.sinkhorn_stabilized(...)`
- `ot.bregman.sinkhorn_epsilon_scaling(...)`

These usually keep the same entropic balanced OT semantics as `ot.sinkhorn`, but change the numerical scheme.

---

## 2. Unbalanced OT

Unbalanced OT relaxes exact marginal matching. Instead of forcing row and column sums to equal the input marginals exactly, the optimizer penalizes mismatch.

### `ot.unbalanced.sinkhorn_unbalanced(a, b, M, reg, reg_m, ...)`
POT documents this as solving an unbalanced entropic/KL-regularized OT problem of the form:

\[
\min_\gamma
\langle \gamma, M \rangle
+ \mathrm{reg}\,\mathrm{KL}(\gamma, c)
+ \mathrm{reg}_{m1}\,\mathrm{KL}(\gamma\mathbf{1}, a)
+ \mathrm{reg}_{m2}\,\mathrm{KL}(\gamma^T\mathbf{1}, b)
\]

with `\gamma \ge 0`.

Interpretation:
- `reg_m` controls marginal relaxation
- `reg` controls regularization/divergence involving the **plan**
- use it when a generalized Sinkhorn-style unbalanced formulation is appropriate

Companion:
- `ot.unbalanced.sinkhorn_unbalanced2(...)` returns the **cost**

### `ot.unbalanced.mm_unbalanced(a, b, M, reg_m, reg=0, div='kl', ...)`
POT documents this as solving a more general unbalanced objective:

\[
\min_\gamma
\langle \gamma, M \rangle
+ \mathrm{reg}_{m1}\,\mathrm{div}(\gamma\mathbf{1}, a)
+ \mathrm{reg}_{m2}\,\mathrm{div}(\gamma^T\mathbf{1}, b)
+ \mathrm{reg}\,\mathrm{div}(\gamma, c)
\]

with `\gamma \ge 0`.

Interpretation:
- unbalanced OT with explicit divergence choice
- `div` is typically KL or half-squared `l2`
- `reg=0` is allowed, which is useful when the intended formula has no extra plan regularization term

Companion:
- `ot.unbalanced.mm_unbalanced2(...)` returns the **cost**

### `ot.unbalanced.lbfgsb_unbalanced(...)`
Flexible unbalanced OT optimized with L-BFGS-B.

Use when:
- divergence flexibility matters
- the problem is not a natural fit for generalized Sinkhorn updates
- runtime is less important than matching the desired objective class

---

## 3. GW and FGW

## GW: Gromov-Wasserstein
Use GW when the problem is not “match source point `i` to target point `j` through a shared cross-domain cost `M`”, but instead “match the **internal geometry** of one space to the internal geometry of another”.

Inputs are usually:
- `C1`: within-source distance/cost matrix
- `C2`: within-target distance/cost matrix
- `p`, `q`: source and target weights

### `ot.gromov.gromov_wasserstein(C1, C2, p, q, ...)`
Optimizes a quadratic structural mismatch objective under balanced marginal constraints.

Meaning:
- structure-to-structure matching
- no ordinary cross-domain `M` is required
- use when source and target do not naturally share the same coordinate system or direct feature cost

### `ot.gromov.entropic_gromov_wasserstein(...)`
Entropic GW variant, typically easier numerically but not identical to plain GW.

## FGW: Fused Gromov-Wasserstein
Use FGW when you need both:
- direct feature matching through `M`
- structural matching through `C1, C2`

### `ot.gromov.fused_gromov_wasserstein(M, C1, C2, p, q, alpha, ...)`
Conceptually mixes feature cost and structural mismatch:

\[
\min_T (1-\alpha)\langle T, M \rangle
+ \alpha \cdot \text{structural mismatch}(C1,C2,T)
\]

Interpretation:
- `alpha=0`: feature-only side dominates
- `alpha=1`: pure structure side dominates
- intermediate `alpha`: trade-off

### `ot.gromov.entropic_fused_gromov_wasserstein(...)`
Entropic FGW variant.

---

## 4. Plan vs cost outputs

In POT, a common naming pattern is:
- `foo(...)` -> returns a **plan**
- `foo2(...)` -> returns a **cost**

Examples:
- `ot.emd` -> plan
- `ot.emd2` -> cost
- `ot.sinkhorn` -> plan
- `ot.sinkhorn2` -> cost
- `ot.unbalanced.sinkhorn_unbalanced` -> plan
- `ot.unbalanced.sinkhorn_unbalanced2` -> cost
- `ot.unbalanced.mm_unbalanced` -> plan
- `ot.unbalanced.mm_unbalanced2` -> cost

Always verify in the installed version before relying on naming alone.

---

## 5. Common mistakes

### Mistake A: treating all “Sinkhorn” functions as the same mathematical object
Balanced Sinkhorn, unbalanced Sinkhorn, and entropic GW/FGW are different solver families with different objectives.

### Mistake B: mixing up marginal relaxation and plan regularization
If a written objective distinguishes these, map POT arguments carefully:
- `reg_m` usually governs marginal relaxation
- `reg` usually governs a plan-side regularization/divergence term

### Mistake C: using GW when ordinary OT is the right object
If the task already has a meaningful cross-domain cost matrix `M`, ordinary OT may be the correct first choice.

### Mistake D: using ordinary OT when the real task is structural matching
If the scientific object is relational geometry between spaces, GW/FGW may be the right family.

### Mistake E: forgetting mass compatibility in balanced OT
Balanced OT expects compatible total mass. If masses differ and should remain different, use an unbalanced formulation.

---

## 6. Live verification checklist

When semantics matter, inspect the installed POT library directly:

```python
import inspect, ot
from ot import unbalanced, gromov

print(ot.__version__)
print(inspect.signature(ot.sinkhorn))
print(inspect.signature(unbalanced.sinkhorn_unbalanced))
print(inspect.signature(unbalanced.mm_unbalanced))
print(inspect.signature(gromov.gromov_wasserstein))
print(inspect.getdoc(unbalanced.mm_unbalanced))
```

Use this checklist:
- confirm the installed POT version
- read the actual function signature
- read the leading docstring equation
- check whether the function returns a plan or a cost
- only then claim equivalence to a mathematical objective
