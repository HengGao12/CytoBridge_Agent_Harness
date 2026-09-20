---
title: "WFR-FM derivation example"
summary: "Worked example showing how a named dynamic WFR objective becomes a concrete simulation-free flow-matching implementation."
read_when:
  - "Writing a proposal with a mathematical problem and derivation"
  - "Understanding how WFR-FM maps theory to package code"
---
# WFR-FM Derivation Example

This page is a self-contained worked example for proposal authors. It shows how
to start from a high-level mathematical objective and derive the concrete
training algorithm, including the intermediate equivalent problems that make the
implementation possible. Use it as a style guide when a new algorithm claims to
solve a named dynamic problem.

Primary reference:

- `cytobridge_agent/rag/literature_db/literature/WFR-FM Simulation-Free Dynamic Unbalanced Optimal Transport.pdf`

Supporting theory references:

- `cytobridge_agent/rag/literature_db/literature/Unbalanced Optimal Transport Dynamic and Kantorovich Formulation.pdf`
- `cytobridge_agent/rag/literature_db/literature/Optimal Entropy-Transport problems and a new Hellinger–Kantorovich distance between positive measures.pdf`

Caveat reference:

- `cytobridge_agent/rag/literature_db/literature/Variational Regularized Unbalanced Optimal Transport Single Network, Least Action.pdf`

The references above are provenance. The derivation below is written so an
agent can understand the design without opening those papers.

## Step 1: State The Dynamic Problem

The target problem is dynamic unbalanced optimal transport under the
Wasserstein-Fisher-Rao metric. For two adjacent observed positive measures
`mu_0` and `mu_1`, solve:

```text
min_{rho_t, u_t, g_t}
  int_0^1 int (||u_t(x)||^2 + delta^2 g_t(x)^2) rho_t(x) dx dt
subject to
  partial_t rho_t + div(rho_t u_t) = rho_t g_t,
  rho_0 = mu_0,
  rho_1 = mu_1.
```

Interpretation:

- `u_t(x)` moves mass in latent space.
- `g_t(x)` creates or removes mass through `d log w_t / dt = g_t(x)`.
- `delta` controls the transport-vs-growth tradeoff.

This problem is meaningful at the positive-measure level. It is not just a
formula for a neural network loss.

The derivation must answer four questions:

```text
1. What endpoint problem is being solved?
2. How is the dynamic problem converted into a finite coupling object?
3. How does that coupling object produce supervised velocity/growth targets?
4. Why does exact training recover both the target distribution and target mass?
```

WFR-FM answers them by deriving a simulation-free flow-matching loss from a
known equivalent static form of dynamic WFR.

## Step 2: Reduce Dynamic WFR To WFR-OET

This step is the important bridge. OET is not an ad-hoc surrogate for the
dynamic WFR problem. It is the computational form of the same endpoint problem,
obtained through:

```text
dynamic Benamou-Brenier WFR
  -> Dirac-to-Dirac WFR endpoint cost
  -> static semi-coupling Kantorovich problem
  -> logarithmic entropy-transport / OET problem
```

### 2.1 First reduce the dynamic problem to atomic rays

For any admissible dynamic solution `(rho_t,u_t,g_t)`, the continuity equation
with source,

```text
partial_t rho_t + div(rho_t u_t) = rho_t g_t,
```

can be understood as moving many infinitesimal source masses along
source-target rays while each ray is allowed to change its mass. The elementary
endpoint problem is therefore:

```text
source atom: m_0 delta_x
target atom: m_1 delta_y
```

with the same WFR action restricted to a one-particle path. WFR theory gives the
closed-form value for this atomic dynamic problem:

```text
C_delta(m_0,x; m_1,y)
  = WFR-DD_delta^2(m_0 delta_x, m_1 delta_y)
  = 2 delta^2 (m_0 + m_1
      - 2 sqrt(m_0 m_1) cos_+(||x-y|| / (2 delta))).
```

Here `cos_+(r) = cos(min(r, pi/2))`. If the distance is beyond the cut locus,
the cosine term is zero, which means it is cheaper to kill mass at `x` and
create mass at `y` than to transport it.

This atomic formula is the first nontrivial reduction: the continuous dynamic
action is now represented by the cost of a single source-target WFR geodesic.
WFR-FM later uses the associated traveling-Dirac formulas to build conditional
paths.

For a reader who wants the intuition without external references, this formula
comes from interpreting WFR as transport on the cone over the data space. The
cone radius is the square root of mass. Moving from `m_0 delta_x` to
`m_1 delta_y` is then a geodesic from `(x,sqrt(m_0))` to `(y,sqrt(m_1))` on that
cone. The cone law of cosines gives:

```text
cone_distance^2
  = m_0 + m_1 - 2 sqrt(m_0 m_1) cos_+(||x-y||/(2 delta)),
```

and the WFR convention used here multiplies this by `2 delta^2`.

### 2.2 Then superpose atomic rays with semi-couplings

For a full measure, one ray per source-target pair is not enough because
unbalanced transport can send one amount of mass from `x` and receive a
different amount at `y`. The static variable is therefore a pair of
semi-couplings:

```text
gamma0(x,y): source-side mass assigned from x to the ray ending at y
gamma1(x,y): target-side mass received at y from the ray starting at x
```

with endpoint constraints:

```text
int gamma0(x,y) dy = mu_0(x),
int gamma1(x,y) dx = mu_1(y).
```

For each pair `(x,y)`, the ray starts with mass `gamma0(x,y)` and ends with mass
`gamma1(x,y)`, so the cost of that ray is:

```text
C_delta(gamma0(x,y), x; gamma1(x,y), y).
```

Integrating these atomic ray costs gives the static WFR Kantorovich problem:

```text
WFR_delta^2(mu_0,mu_1)
  = min_{(gamma0,gamma1)}
      int C_delta(gamma0(x,y), x; gamma1(x,y), y) dx dy.
```

This equality is the dynamic-to-static theorem, not an approximation. Chizat et
al.'s dynamic/Kantorovich formulation proves the general statement by:

```text
1. Approximating semi-couplings by atomic rays and concatenating their dynamic
   geodesics, giving a dynamic competitor from a static plan.
2. Integrating characteristics of smooth dynamic fields, giving a semi-coupling
   competitor from a dynamic path.
3. Extending the equality from smooth densities to general positive measures by
   lower-semicontinuity and regularization.
```

That is the missing bridge from the Benamou-Brenier WFR action to endpoint
couplings: an optimal WFR dynamic path can be represented as a superposition of
optimal Dirac-to-Dirac WFR rays, and the best superposition is found by the
semi-coupling problem.

In discrete empirical data, this means:

```text
mu_0 = sum_i w_i^0 delta_{x_i},
mu_1 = sum_j w_j^1 delta_{y_j}.
```

The semi-couplings become two nonnegative matrices:

```text
Gamma0[i,j]: source mass from x_i assigned to endpoint y_j
Gamma1[i,j]: target mass at y_j assigned back to source x_i
```

with:

```text
sum_j Gamma0[i,j] = w_i^0,
sum_i Gamma1[i,j] = w_j^1.
```

Each nonzero entry `(i,j)` defines one conditional WFR ray from
`Gamma0[i,j] delta_{x_i}` to `Gamma1[i,j] delta_{y_j}`.

### 2.3 Finally convert the semi-coupling problem to OET

The semi-coupling form is conceptually clean but inconvenient numerically
because it has two endpoint masses, `gamma0` and `gamma1`, on the same pair
space. The logarithmic entropy-transport formulation introduces one relaxed
coupling `gamma >= 0` whose marginals are allowed to deviate from `mu_0` and
`mu_1`, with KL penalties:

```text
bar_gamma_0(x) = int gamma(x,y) dy
bar_gamma_1(y) = int gamma(x,y) dx

min_{gamma >= 0}
  int c_delta(x,y) gamma(x,y) dx dy
  + KL(bar_gamma_0 || mu_0)
  + KL(bar_gamma_1 || mu_1),
```

where:

```text
c_delta(x,y) = -2 log cos_+(||x-y|| / (2 delta)).
```

Here `KL(a || b)` is the generalized positive-measure KL:

```text
KL(a || b) = int [a log(a / b) - a + b].
```

It is zero when the relaxed marginal equals the observed endpoint measure, and
it penalizes creating a relaxed coupling whose row/column masses are
inconsistent with `mu_0` or `mu_1`.

The equivalence can be checked from the optimality systems. Start from the
semi-coupling objective normalized by `2 delta^2`:

```text
J(a,b,x,y) = a + b - 2 sqrt(a b) cos_+(||x-y||/(2 delta)).
```

Introduce Lagrange multipliers `phi(x), psi(y)` for the two semi-coupling
constraints. The first-order conditions for the optimal semi-couplings imply:

```text
partial_a J(gamma0,gamma1,x,y) = phi(x),
partial_b J(gamma0,gamma1,x,y) = psi(y),
```

with the dual feasibility condition:

```text
(1 - phi(x))(1 - psi(y)) >= cos_+(||x-y||/(2 delta))^2.
```

With the change of variables:

```text
u(x) = -log(1 - phi(x)),
v(y) = -log(1 - psi(y)),
```

this becomes the OET dual constraint:

```text
u(x) + v(y) <= -log cos_+(||x-y||/(2 delta))^2
              = c_delta(x,y).
```

Now write the OET problem with auxiliary marginals:

```text
bar_gamma_0(x) = int gamma(x,y) dy,
bar_gamma_1(y) = int gamma(x,y) dx.
```

The Lagrangian for OET separates into three inner minimizations: one over
`gamma`, one over `bar_gamma_0`, and one over `bar_gamma_1`. The KL terms give
the primal-dual relations:

```text
bar_gamma_0(x) / mu_0(x) = exp(-u(x)),
bar_gamma_1(y) / mu_1(y) = exp(-v(y)).
```

On the support of the optimal OET coupling, complementary slackness gives:

```text
u(x) + v(y) = c_delta(x,y).
```

These OET KKT conditions are exactly the dual conditions above for the
semi-coupling problem after the change of variables. Therefore the two
optimization problems have the same optimal value and the same endpoint
coupling information.

The corresponding primal conversion is the one used by WFR-FM. If `gamma`
solves OET, define:

```text
gamma0(x,y) = gamma(x,y) * mu_0(x) / bar_gamma_0(x),
gamma1(x,y) = gamma(x,y) * mu_1(y) / bar_gamma_1(y).
```

Because `bar_gamma_0` and `bar_gamma_1` are the marginals of `gamma`, these
definitions satisfy the semi-coupling constraints automatically:

```text
int gamma0(x,y) dy = mu_0(x),
int gamma1(x,y) dx = mu_1(y).
```

Using the KKT relations above, WFR-FM Appendix A.1 verifies that these
`gamma0,gamma1` also satisfy the semi-coupling stationarity equations. Since
the semi-coupling problem is convex in the endpoint masses, the converted
semi-couplings are optimal. This is the concrete algebra behind WFR-FM
Theorem 3.1.

In discrete data, the same conversion is:

```text
row_sum_i = sum_j Gamma[i,j],
col_sum_j = sum_i Gamma[i,j],

Gamma0[i,j] = Gamma[i,j] * w_i^0 / row_sum_i,
Gamma1[i,j] = Gamma[i,j] * w_j^1 / col_sum_j.
```

This is why WFR-FM can solve one OET coupling `Gamma`, then obtain the two
semi-coupling matrices needed for WFR geometry. The KL terms in OET do not
replace the endpoint constraints; they are the unconstrained way to solve for
the same endpoint mass split. After reweighting, the semi-coupling endpoint
constraints hold exactly in the empirical problem up to numerical epsilon.

So the practical algorithm is:

```text
solve OET gamma
  -> reweight gamma into semi-couplings gamma0,gamma1
  -> compute terminal mass ratio m_1 = gamma1 / gamma0 per ray
  -> train velocity and growth against closed-form WFR geodesic targets
```

Implementation mapping:

- `WFROETCouplingStrategy._compute_wfr_cost_matrix(...)` builds `c_delta`.
- `WFROETCouplingStrategy._solve_gamma(...)` solves the OET problem.
- `WFROETCouplingStrategy._convert_to_semicouplings(...)` applies the
  reweighting formulas and constructs `gamma0`, `gamma1`, and terminal mass
  ratios.

Source anchors:

- WFR-FM paper: Eq. 3.3 is the dynamic WFR problem; Eq. 3.4 gives the
  Dirac-to-Dirac WFR cost; Eq. 3.7 gives the semi-coupling form; Eq. 3.8 and
  Theorem 3.1 give the OET form and conversion back to semi-couplings.
- `Unbalanced Optimal Transport: Dynamic and Kantorovich Formulations`:
  Theorem 4.3 proves the equivalence between dynamic unbalanced OT and
  semi-coupling Kantorovich formulations.
- `Optimal Entropy-Transport problems and a new Hellinger-Kantorovich distance`:
  the logarithmic entropy-transport/HK construction explains why the
  `-log cos^2` cost plus KL marginal relaxation is the entropy-transport form
  of the WFR/Hellinger-Kantorovich distance.

## Step 3: Turn Semi-Coupling Into Conditional Paths

Sample `(x_0, x_1)` from `gamma0`. The source-side mass is normalized to:

```text
m_0 = 1.
```

The pair-specific terminal mass is:

```text
m_1(x_0,x_1) = gamma1(x_0,x_1) / gamma0(x_0,x_1).
```

For a Dirac-to-Dirac WFR geodesic, the mass path and velocity satisfy:

```text
m(t) = A t^2 - 2 B t + 1,
u(t) m(t) = omega,
g(t) = d log m(t) / dt = (2 A t - 2 B) / m(t).
```

The constants `A`, `B`, and `omega` are closed-form functions of
`x_0`, `x_1`, `m_1`, and `delta`. With the package convention `m_0=1`, let:

```text
r = ||x_1 - x_0||,
l = (x_1 - x_0) / r when r > 0, otherwise 0,
tau = tan(min(r / (2 delta), pi/2 - eps_angle)),
s = sqrt(m_1 / (1 + tau^2)).
```

`eps_angle` is only a numerical margin that keeps the tangent finite near the
cut locus.

Then:

```text
A = 1 + m_1 - 2 s,
B = 1 - s,
omega = 2 delta tau s l.
```

These constants satisfy the endpoint mass constraints:

```text
m(0) = 1,
m(1) = A - 2B + 1 = m_1.
```

The trajectory follows:

```text
eta_t = x_0 + int_0^t omega / m(s) ds
```

and optionally a Gaussian width:

```text
x_t = eta_t + sigma * epsilon.
```

The integral has a closed form used by the package:

```text
int_0^t 1 / m(s) ds
  = [atan((A t - B) / sqrt(A - B^2)) - atan(-B / sqrt(A - B^2))]
      / sqrt(A - B^2).
```

The conditional targets are therefore:

```text
u_target(t) = d eta_t / dt = omega / m(t),
g_target(t) = d log m(t) / dt = (2 A t - 2 B) / m(t).
```

This is the key simulation-free step. The algorithm does not integrate an ODE
during training to discover `u_target` or `g_target`; it obtains both from the
closed-form WFR geodesic ray induced by the OET/semi-coupling solution.

Implementation mapping:

- `CytoBridge/tl/flow_matching.py::WFRTravelingGaussianPath._pair_parameters(...)`
  computes `A`, `B`, and
  `omega`.
- `CytoBridge/tl/flow_matching.py::WFRTravelingGaussianPath._mass_and_integral(...)`
  computes `m(t)` and the integral of `1 / m(t)`.
- `CytoBridge/tl/flow_matching.py::WFRTravelingGaussianPath.sample_xt_wfr(...)`
  computes `x_t`.
- `CytoBridge/tl/flow_matching.py::WFRTravelingGaussianPath.compute_conditional_flow_wfr(...)`
  computes
  `u(t) = omega / m(t)`.
- `CytoBridge/tl/flow_matching.py::WFRTravelingGaussianPath.compute_conditional_mass_wfr(...)`
  computes
  `g(t)` and uses `m(t)` as the loss weight.
- `CytoBridge/tl/flow_matching_backends.py::WFROETCouplingStrategy` subclasses
  `ChunkedTransportCouplingStrategy` and only specializes the WFR cost and
  semi-coupling conversion; the chunking/cache/sample lifecycle is shared with
  balanced and unbalanced OT flow-matching builtins.

## Step 4: Write The Simulation-Free Training Objective

Draw:

```text
(x_0, x_1) ~ gamma0,
t ~ Uniform(0,1),
epsilon ~ N(0,I).
```

Construct:

```text
x_t = eta_t + sigma epsilon,
u_target(t) = omega / m(t),
g_target(t) = (2 A t - 2 B) / m(t).
```

Train networks:

```text
min_{theta,phi}
  E[ m(t) ||v_theta(t,x_t) - u_target(t)||^2
   + m(t) ||g_phi(t,x_t) - g_target(t)||^2 ].
```

The key engineering point is that no training-time ODE simulation is needed.
The OET solve creates endpoint supervision, and the WFR geodesic formulas create
closed-form conditional velocity and growth targets.

The weight `m(t)` is not cosmetic. At time `t`, a conditional ray contributes
mass proportional to `m(t)`, so the marginal vector field and marginal growth
rate are mass-weighted conditional averages:

```text
v*(t,x) = E[ u_target(t,z) m(t,z) | x_t = x ]
          / E[ m(t,z) | x_t = x ],

g*(t,x) = E[ g_target(t,z) m(t,z) | x_t = x ]
          / E[ m(t,z) | x_t = x ].
```

Weighted squared-error regression has these conditional expectations as its
population minimizer. Therefore, in the exact-fit limit:

```text
v_theta(t,x) = v*(t,x),
g_phi(t,x) = g*(t,x).
```

The induced marginal measure then satisfies:

```text
partial_t rho_t + div(rho_t v_theta) = rho_t g_phi,
```

which is the same WFR dynamic equation as the conditional path mixture.

## Step 5: Explain Distribution And Mass Recovery

This section is what a proposal should emulate when it claims that an
unbalanced flow-matching algorithm is theoretically sound.

Distribution recovery:

```text
1. OET gamma is reweighted into optimal WFR semi-couplings gamma0,gamma1.
2. Each nonzero semi-coupling entry defines a WFR geodesic ray.
3. The mixture of those rays is an optimal dynamic WFR path from mu_0 to mu_1.
4. The trained networks recover the marginal velocity/growth fields of that
   ray mixture in the population exact-fit limit.
5. Sealed rollout from the t=0 measure with those fields follows the same
   continuity equation and reaches mu_1.
```

Mass recovery:

```text
1. Each sampled ray starts with unit local mass and has terminal mass
   m_1 = gamma1 / gamma0.
2. The growth target satisfies d log m(t) / dt = g_target(t).
3. Exact growth-head training recovers this log-mass derivative along the
   marginal path.
4. Integrating the learned growth along the learned trajectory maps each
   source-side semi-coupling mass gamma0(x,y) to gamma1(x,y).
5. Summing over source locations gives int gamma1(x,y) dx = mu_1(y).
```

Thus the model is evaluated through the learned continuous dynamics, not by
post-hoc rescaling predicted particles to match target mass. In the idealized
exact OET, infinite-data, exact-regression limit, WFR-FM fits both the endpoint
distribution and total mass. In package practice, finite neural training,
mini-batch OET, finite sampling, and numerical stabilization introduce
approximation error; these errors are what W1 and TMV measure.

Implementation mapping:

- `WFRFlowMatchingBackend.sample_batch(...)` produces `x_t`, `u_target`,
  `g_target`, and `loss_weights`.
- `CytoBridge/configs/wfrfm.yaml` selects:
  - `flow_matching.backend: wfrfm`
  - `model.components: ['velocity', 'growth']`
  - `sigma: 0.0` by default
  - mini-batch WFR-OET enabled by default.

## Step 6: Extend To Multiple Time Points

For snapshots `mu_t0, ..., mu_tK`, WFR-FM solves adjacent WFR problems and
concatenates the interval solutions. The paper's multi-time proposition states
that this adjacent concatenation solves the corresponding multi-time WFR
problem under the stated setup.

Implementation mapping:

- The backend builds one coupling state per adjacent gap.
- The same `v_theta(t,x)` and `g_phi(t,x)` are shared across gaps.
- The final evaluator rolls out a continuous dynamics trajectory from the
initial time point and computes W1/TMV from trajectory slices.

## Step 7: Boundary And Biological Caveat

WFR-FM is a good example of rigorous theory-to-implementation design because it
starts from a named dynamic action and derives the coupling, path, and loss.

It is not automatically the biologically best growth model. Classical WFR uses
a quadratic growth penalty. The Var-RUOT paper points out that the growth
penalty curvature imposes a relationship between velocity and growth. In some
cell differentiation settings, biological priors may expect upstream/stem-like
states to have larger growth that decreases along the trajectory. Standard WFR
can encode a different directional relation depending on the penalty. Therefore
use `wfrfm` as a principled dynamic-WFR baseline, but diagnose whether its
growth field matches the biological question.
