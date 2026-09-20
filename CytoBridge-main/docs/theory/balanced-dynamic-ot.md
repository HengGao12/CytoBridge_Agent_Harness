---
title: "Balanced dynamic OT algorithms"
summary: "Theory for dynamical OT and balanced OT-CFM algorithms."
read_when:
  - "Understanding balanced OT / OT-CFM theory"
  - "Auditing balanced-only proposal claims"
---
# Builtin Algorithm Theory: Balanced Dynamic OT

Balanced methods fit probability distributions and intentionally do not model total mass change.

## `dynamical_ot`

### Mathematical Problem

`dynamical_ot` is the balanced neural-ODE approximation to dynamic optimal
transport. The ideal continuous problem is the Benamou-Brenier formulation:

```text
min_{rho_t, v_t}
    int_0^1 int 0.5 * ||v_t(x)||^2 rho_t(x) dx dt
subject to
    partial_t rho_t + div(rho_t v_t) = 0,
    rho_0 = mu_k,
    rho_1 = mu_{k+1}.
```

The constraint is a mass-conserving continuity equation. Total mass is fixed.

### Package Approximation

The package parameterizes the flow map by a neural ODE:

```text
dX_t / dt = v_theta(t, X_t).
```

Training simulates particles forward and penalizes mismatch between simulated
endpoint distributions and observed snapshots, with energy-style regularization
encouraging short transport paths.

This is not a closed-form OT solver. It is a neural approximation to the
balanced dynamic OT problem.

### Distribution Fit Argument

If the model class can represent the optimal velocity field and optimization
reaches zero endpoint distribution mismatch, then the ODE pushforward satisfies:

```text
(Phi_{t_k -> t_{k+1}})_# mu_k = mu_{k+1}.
```

The learned curve `rho_t = (Phi_{t_k -> t})_# mu_k` satisfies the continuity
equation induced by `v_theta`. Therefore the predicted endpoint distribution
matches the observed target distribution.

### Mass Fit Argument

There is no mass-fit argument. The dynamic OT constraint conserves total mass:

```text
d/dt int rho_t(x) dx = 0.
```

The package model has no growth head in this config. Any apparent cell-count
change is outside the modeled object. `TMV` is therefore diagnostic only.

### References

- `cytobridge_agent/rag/literature_db/literature/TrajectoryNet A Dynamic Optimal Transport Network for Modeling Cellular Dynamics.pdf`

## `balanced_ot_cfm`

### Mathematical Problem

`balanced_ot_cfm` is the simulation-free flow-matching route to balanced dynamic
OT. It starts from the static OT problem:

```text
pi* = argmin_{pi in Pi(mu_k, mu_{k+1})}
    int ||y - x||^2 d pi(x,y),
```

where `Pi(mu_k, mu_{k+1})` is the set of balanced couplings with source marginal
`mu_k` and target marginal `mu_{k+1}`.

The induced deterministic displacement interpolation is:

```text
X_tau = (1 - tau) x + tau y,
u_tau = y - x,
tau in [0, 1].
```

In the zero-noise limit, this is the dynamic OT geodesic associated with the
quadratic static OT plan.

### Equivalence To The Benamou-Brenier Dynamic OT Form

The OT-CFM paper makes a stronger statement than "OT-CFM uses OT-like pairs".
It relies on the equivalence between:

1. the static 2-Wasserstein problem

```text
W_2(mu_k, mu_{k+1})^2
= min_{pi in Pi(mu_k, mu_{k+1})}
    int ||y - x||^2 d pi(x,y),
```

and

2. the Benamou-Brenier dynamic problem

```text
W_2(mu_k, mu_{k+1})^2
= min_{rho_t, u_t}
    int_0^1 int ||u_t(z)||^2 rho_t(z) dz dt
subject to
    partial_t rho_t + div(rho_t u_t) = 0,
    rho_0 = mu_k,
    rho_1 = mu_{k+1}.
```

Given an optimal static coupling `pi*`, define:

```text
T_t(x,y) = (1 - t) x + t y,
rho_t = (T_t)_# pi*.
```

At the path level, each coupled pair moves along the straight line:

```text
d/dt T_t(x,y) = y - x.
```

The Eulerian velocity field corresponding to the mixture of these straight
paths is the conditional mean:

```text
u_t(z) = E[y - x | T_t(x,y) = z].
```

This pair `(rho_t, u_t)` satisfies the continuity equation because it is the
pushforward of moving particles under deterministic straight-line dynamics.
Its action is bounded by the static OT cost:

```text
int_0^1 int ||u_t(z)||^2 rho_t(z) dz dt
<= int_0^1 E_{(x,y)~pi*} ||y - x||^2 dt
= int ||y - x||^2 d pi*(x,y)
= W_2(mu_k, mu_{k+1})^2.
```

The inequality is Jensen's inequality applied to the conditional expectation
`E[y-x | T_t=z]`. The Benamou-Brenier formula gives the reverse lower bound:
no admissible dynamic path can have action below `W_2^2`. Therefore equality
holds, and the displacement interpolation induced by `pi*` is a dynamic OT
minimizer.

When the optimal coupling is induced by a Monge map, the conditional variance is
zero almost everywhere along the interpolation, so the velocity is simply:

```text
u_t((1-t)x + tT(x)) = T(x) - x.
```

For general couplings, the conditional-mean Eulerian field is still the dynamic
minimizer in the Benamou-Brenier sense.

The OT-CFM paper implements a smoothed version:

```text
p_t^sigma(z | x,y) = N(z | (1-t)x + ty, sigma^2 I),
q(z) = pi*(x,y).
```

Its Proposition 4.2 states that, as `sigma^2 -> 0`, the marginal path and
vector field recover the dynamic OT minimizer between the source and target
distributions. The package `balanced_ot_cfm` uses the deterministic `sigma=0`
version of this construction.

Minibatch OT is an approximation: if the minibatch support equals the full
empirical support, the construction recovers the full empirical OT-CFM dynamic
OT path; smaller minibatches approximate that path.

### Package Approximation

Runtime semantics:

- Coupling: `BalancedOTCouplingStrategy`.
- Path: `LinearDeterministicConditionalPath`.
- Mass: `NullMassStrategy`.
- Config: `CytoBridge/configs/balanced_ot_cfm.yaml`.

The training objective is the CFM regression:

```text
min_theta E_{(x,y)~pi*, tau~U[0,1]}
    ||v_theta(tau, X_tau) - u_tau||^2.
```

### Distribution Fit Argument

Because `pi*` has marginals `mu_k` and `mu_{k+1}`, the path marginal satisfies:

```text
rho_0 = mu_k,
rho_1 = mu_{k+1}.
```

By Lemma A, the exact CFM population minimizer recovers the marginal vector
field of this path. Integrating that field recovers the target marginal at
`tau = 1`.

So `balanced_ot_cfm` has a direct distribution-fit argument: it learns the
velocity field for the OT displacement interpolation.

### Mass Fit Argument

There is no mass-fit argument. The static OT constraint is balanced:

```text
pi in Pi(mu_k, mu_{k+1}),
sum mass(mu_k) = sum mass(mu_{k+1}) = 1.
```

`NullMassStrategy` fixes terminal mass to `1`. The method models probability
transport, not cell proliferation/death. `TMV` is diagnostic only.

### References

- `cytobridge_agent/rag/literature_db/literature/Conditional Flow Matching Simulation-Free Dynamic Optimal Transport.pdf`
