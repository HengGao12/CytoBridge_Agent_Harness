---
title: "Unbalanced deterministic algorithms"
summary: "Theory for unbalanced OT and VGFM-style deterministic mass modeling."
read_when:
  - "Understanding deterministic unbalanced methods"
  - "Auditing mass recovery claims"
---
# Builtin Algorithm Theory: Deterministic Unbalanced Methods

Deterministic velocity-growth methods model positive measures and use TMV as a hard gate.

## `unbalanced_ot`

### Mathematical Problem

`unbalanced_ot` is the deterministic neural-ODE route to unbalanced dynamic OT.
The ideal positive-measure problem is a WFR-style transport-growth problem:

```text
min_{mu_t, v_t, g_t}
    int_0^1 int [0.5 * ||v_t(x)||^2 + beta * g_t(x)^2] dmu_t(x) dt
subject to
    partial_t mu_t + div(mu_t v_t) = g_t mu_t,
    mu_0 = mu_k,
    mu_1 = mu_{k+1}.
```

Here `g_t` is the growth/death rate. The total mass evolves as:

```text
d/dt mu_t(Omega) = int g_t(x) dmu_t(x).
```

So this is not balanced OT. It is a positive-measure dynamics problem where
transport and growth jointly explain the next snapshot.

### Package Approximation

The package parameterizes:

```text
dX_t / dt = v_theta(t, X_t),
d log w_t / dt = g_phi(t, X_t).
```

The predicted measure is:

```text
hat_mu_t = sum_i w_i(t) delta_{X_i(t)}.
```

Training simulates particles forward and compares the predicted weighted
measure against observed snapshots while regularizing motion and growth.

### Distribution Fit Argument

If the learned fields fit the unbalanced dynamic problem exactly, then the
weighted pushforward measure at the target time satisfies:

```text
hat_mu_{k+1} = mu_{k+1}.
```

Because `W1` is computed on the predicted weighted measure, exact weighted
measure recovery implies distribution recovery.

### Mass Fit Argument

The model explicitly evolves weights:

```text
w_i(t_{k+1}) = w_i(t_k) *
    exp(int_{t_k}^{t_{k+1}} g_phi(t, X_i(t)) dt).
```

Therefore the total predicted mass is:

```text
hat_M_{k+1} = sum_i w_i(t_{k+1}).
```

If the continuity-growth equation is fit with endpoint
`hat_mu_{k+1} = mu_{k+1}`, then:

```text
hat_M_{k+1} = mu_{k+1}(Omega).
```

So mass recovery follows from exact positive-measure endpoint recovery. `TMV`
is a hard gate.

### References

- `cytobridge_agent/rag/literature_db/literature/Reconstructing growth and dynamic trajectories from single-cell transcriptomics data.pdf`

## `vgfm`

### Mathematical Problem

`vgfm` is the deterministic unbalanced flow-matching default. It does not claim
to solve one established global dynamic OT functional end-to-end. Instead, it
solves a package-defined interval problem:

1. solve adjacent static/semi-relaxed UOT to define the target positive measure;
2. learn a deterministic velocity-growth path that realizes that endpoint
   measure without training-time ODE simulation.

For each adjacent interval, the package UOT problem is:

```text
min_{pi >= 0}
    <C, pi> + eps * Ent(pi) + lambda * KL(pi 1 | a)
subject to
    pi^T 1 = b,
    a_i = 1,
    b_j = 1.
```

This is source-relaxed and target-constrained under the current package
convention.

### Package Approximation

Runtime semantics:

- Coupling: `UnbalancedOTCouplingStrategy`.
- Path: `LinearDeterministicConditionalPath`.
- Mass: `UOTMassStrategy`.
- Config: `CytoBridge/configs/vgfm.yaml`.

For sampled endpoint pair `(x_i, y_j)`:

```text
X_tau = (1 - tau) x_i + tau y_j,
u_tau = y_j - x_i,
w_i(1) = r_i = sum_j pi(i,j),
g_tau = log(r_i) / (t_{k+1} - t_k).
```

The objective is:

```text
min_{theta,phi}
E[ ||v_theta(tau, X_tau) - u_tau||^2
   + kappa ||g_phi(tau, X_tau) - g_tau||^2 ].
```

### Distribution Fit Argument

The endpoint plan has target column marginal:

```text
c_j = sum_i pi(i,j) = b_j.
```

The velocity regression learns the conditional transport geometry induced by
the UOT endpoint plan. At the measure level, this is the same logic as Lemma A,
except the endpoint plan is a positive-measure UOT plan rather than a balanced
probability coupling.

If the velocity field is fit exactly, the endpoint locations follow the UOT
transport geometry. If the growth field is also fit exactly, the transported
mass attached to those endpoints is the UOT mass.

Endpoint mass arriving at target `j` is:

```text
sum_i r_i * P(j | i)
= sum_i r_i * pi(i,j) / r_i
= sum_i pi(i,j)
= c_j
= b_j.
```

Thus the predicted weighted endpoint measure has the same target marginal as
the UOT plan. Because the UOT target marginal is the observed target empirical
measure, distribution fit follows.

### Mass Fit Argument

The total predicted terminal mass is:

```text
sum_i r_i = sum_i sum_j pi(i,j) = sum_j b_j.
```

Under the unit target-mass convention, `sum_j b_j = n_{k+1}`. Relative to the
local source convention `a_i = 1`, the source total is `n_k`. Therefore the
terminal total mass encodes the observed cell-count / mass change
`n_{k+1} / n_k`.

The learned growth head represents this through:

```text
d log w / dt = g_phi(t,x).
```

Exact growth regression implies:

```text
w_i(t_{k+1}) = r_i.
```

Therefore the weighted-particle inference measure recovers both endpoint
distribution and total mass. `TMV` is a hard gate.

### Boundaries

- There is no stochastic score head.
- The current UOT convention is source-relaxed and target-constrained.
- The method is a simulation-free regression surrogate for UOT endpoint
  dynamics, not a claim of solving a named global PDE functional.

### References

- `cytobridge_agent/rag/literature_db/literature/Joint Velocity-Growth Flow Matching for Single-Cell Dynamics Modeling.pdf`
- `cytobridge_agent/rag/literature_db/literature/Reconstructing growth and dynamic trajectories from single-cell transcriptomics data.pdf`

## `wfrfm`

### Mathematical Problem

`wfrfm` is the simulation-free flow-matching route to a named dynamic
unbalanced OT problem: Wasserstein-Fisher-Rao dynamic OT. For adjacent
snapshots, the ideal problem is:

```text
WFR_delta^2(mu_0, mu_1)
= inf_{rho_t, u_t, g_t}
  int_0^1 int (||u_t(x)||^2 + delta^2 g_t(x)^2) rho_t(x) dx dt
subject to
  partial_t rho_t + div(rho_t u_t) = rho_t g_t,
  rho_0 = mu_0, rho_1 = mu_1.
```

This is a positive-measure dynamics problem, not a balanced transport problem.
The growth rate `g_t` is part of the state equation and part of the action.
The hyperparameter `delta` controls the transport-vs-growth tradeoff.
In package configs, `wfrfm` defaults to `delta: auto`: the backend samples
cross-timepoint latent distances between adjacent snapshots, takes the
configured quantile (default `0.9`), and sets
`delta = distance_quantile / (2 * delta_target_angle)` with
`delta_target_angle=1.0` by default. This keeps typical adjacent-time WFR
angles away from the cone cutoff, avoiding accidental kill/create degeneracy on
large-scale latent embeddings. Builtin WFR-FM benchmark configs should normally
use this auto rule; a fixed `delta` is still accepted by the code, but should be
reserved for deliberate controlled comparisons rather than copied across
datasets.

The WFR-FM paper uses two equivalent views:

1. Dynamic WFR action above.
2. Static WFR semi-coupling:

```text
min_{gamma0, gamma1}
  int WFR-DD_delta^2(gamma0(x,y) delta_x,
                    gamma1(x,y) delta_y) dx dy
subject to
  int gamma0(x,y) dy = mu_0(x),
  int gamma1(x,y) dx = mu_1(y).
```

The semi-coupling is obtained from the equivalent OET problem:

```text
min_{gamma >= 0}
  int -2 log cos(||x-y|| / (2 delta)) gamma(x,y) dx dy
  + KL(int gamma(x,y) dy || mu_0)
  + KL(int gamma(x,y) dx || mu_1).
```

This display omits the common positive factor `2 delta^2`, which does not
change the optimizer.

If `gamma` solves OET, the WFR semi-coupling is:

```text
gamma0(x,y) = gamma(x,y) * mu_0(x) / int gamma(x,z) dz,
gamma1(x,y) = gamma(x,y) * mu_1(y) / int gamma(z,y) dz.
```

For the full derivation chain from the dynamic WFR Benamou-Brenier form to
Dirac geodesics, semi-couplings, and then OET, read
`theory/wfrfm-derivation-example.md`. The short version is that the dynamic WFR
path is decomposed into optimal Dirac-to-Dirac WFR rays; a semi-coupling chooses
the best superposition of those rays; and the OET dual/KKT system gives a
single relaxed coupling `gamma` that can be reweighted back into the optimal
semi-couplings above.

This is the key distinction from `vgfm`. `vgfm` uses the package UOT endpoint
semantics as a deterministic FM surrogate. `wfrfm` starts from a named WFR
dynamic problem and uses the WFR-OET equivalence to construct the FM training
coupling.

### Package Approximation

Runtime semantics:

- Coupling: `WFROETCouplingStrategy`.
- Path: `CytoBridge/tl/flow_matching.py::WFRTravelingGaussianPath`.
- Mass: `WFRTerminalMassStrategy`.
- Config: `CytoBridge/configs/wfrfm.yaml`.

For a sampled pair `(x_0, x_1)` from the source-side semi-coupling `gamma0`,
the package computes the pair-specific terminal mass:

```text
m_1(x_0,x_1) = gamma1(x_0,x_1) / gamma0(x_0,x_1).
```

With local initial mass `m_0 = 1`, the WFR traveling-Dirac formulas define:

```text
m(t) = A t^2 - 2 B t + 1,
u(t) m(t) = omega,
g(t) = (2 A t - 2 B) / m(t).
```

`WFRTravelingGaussianPath` uses the corresponding integrated trajectory mean
and optional Gaussian width `sigma`. The builtin config uses `sigma=0.0` as the
deterministic limit.

For biological-scale data, `wfrfm` uses mini-batch WFR-OET by default. It solves
OET subproblems on matched source/target chunks and samples from the stored
subplans. This is an approximation to the full OET coupling and should be
reported as such in algorithm comparisons.

### Distribution Fit Argument

The WFR-FM argument has two layers.

First, WFR-OET is equivalent to the static WFR semi-coupling problem. Therefore
an exact OET solve gives a semi-coupling `(gamma0, gamma1)` whose induced
WFR traveling-Dirac paths solve the dynamic WFR action between the two endpoint
measures.

Second, conditional flow matching regresses the marginal velocity and growth
fields induced by those conditional WFR paths. In the exact-fit limit, the
learned `v_theta(t,x)` and `g_phi(t,x)` match the WFR conditional vector field
and growth rate almost everywhere under the training path distribution. Running
the learned dynamics from the source measure therefore recovers the WFR path.

As `sigma -> 0`, the boundary marginals of the traveling Gaussian converge to
`mu_0` and `mu_1`. Thus exact WFR-FM recovers the target distribution at the
endpoint, and W1 on the sealed rollout trajectory should vanish in the
idealized limit.

### Mass Fit Argument

Mass recovery comes from the semi-coupling constraint, not from a post-hoc
rescale. For every sampled pair, `WFRTerminalMassStrategy` supervises the local
terminal mass ratio:

```text
m_1(x_0,x_1) = gamma1(x_0,x_1) / gamma0(x_0,x_1).
```

If the growth head exactly fits:

```text
d log w_t / dt = g_phi(t, X_t),
```

then each source-side unit of semi-coupling mass evolves to the corresponding
target-side semi-coupling mass. Summing over all source-target pairs gives:

```text
int gamma1(x,y) dx = mu_1(y),
```

so both endpoint weighted distribution and total target mass are recovered.
Because `wfrfm` explicitly models unbalanced mass, `TMV` is a hard gate.

### Boundaries

- WFR-FM is a strong example of deriving a concrete dynamic unbalanced OT
  problem into a simulation-free FM objective.
- It should not be treated as the only valid UOT-FM design. It complements
  `vgfm`, which is the simpler deterministic package default.
- Classical WFR uses a quadratic growth penalty. The Var-RUOT paper argues
  that this penalty couples the learned velocity and growth fields in a way
  that may contradict biological priors: for some priors, upstream stem-like
  cells should have larger growth that decreases along differentiation, while
  standard WFR optimality can force the opposite directional relation depending
  on the growth penalty curvature.
- Therefore `wfrfm` is mathematically principled for WFR dynamic OT, but its
  growth-velocity relationship should be diagnosed on biological data instead
  of assumed biologically correct.

### References

- `cytobridge_agent/rag/literature_db/literature/WFR-FM Simulation-Free Dynamic Unbalanced Optimal Transport.pdf`
- `cytobridge_agent/rag/literature_db/literature/Variational Regularized Unbalanced Optimal Transport Single Network, Least Action.pdf`
