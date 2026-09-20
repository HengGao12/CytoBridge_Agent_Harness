---
title: "Stochastic and interaction algorithms"
summary: "Theory for SF2M, RUOT, CRUFM, and interaction-aware simulation methods."
read_when:
  - "Understanding stochastic bridge or interaction-aware methods"
---
# Builtin Algorithm Theory: Stochastic And Interaction Methods

Stochastic bridge, RUOT, CRUFM, and interaction-aware unbalanced dynamics.

## `sf2m`

### Mathematical Problem

`sf2m` is the balanced Schrodinger bridge / score-and-flow-matching builtin.
The ideal dynamic problem is the entropic bridge:

```text
min_P KL(P || R_sigma)
subject to
    P_0 = mu_k,
    P_1 = mu_{k+1},
```

where `R_sigma` is Brownian reference path measure. Equivalently, in stochastic
control form:

```text
dX_t = u_t(X_t) dt + sigma dB_t,

min_{rho_t, u_t}
    int_0^1 int 0.5 * ||u_t(x)||^2 rho_t(x) dx dt
subject to
    partial_t rho_t + div(rho_t u_t)
        = (sigma^2 / 2) Delta rho_t,
    rho_0 = mu_k,
    rho_1 = mu_{k+1}.
```

This is balanced. Both endpoint marginals are probability measures.

### Package Approximation

Runtime semantics:

- Coupling: `BalancedOTCouplingStrategy`.
- Path: `SchrodingerBridgeConditionalPath`.
- Mass: `NullMassStrategy`.
- Config: `CytoBridge/configs/sf2m.yaml`.

The theory-aligned endpoint coupling is entropic OT:

```text
pi_epsilon = argmin_{pi in Pi(mu_k, mu_{k+1})}
    int ||y - x||^2 d pi(x,y) + epsilon * KL(pi | mu_k otimes mu_{k+1}),

epsilon = 2 * sigma^2.
```

The package default sets `sigma = 0.05`, hence `epsilon = 0.005`.

For each sampled pair `(x,y)`, SF2M uses a Brownian-bridge conditional path and
trains both:

- a flow/velocity target;
- a score target for the stochastic bridge.

### Distribution Fit Argument

The entropic OT plan has source and target marginals `mu_k` and `mu_{k+1}`.
The mixture of Brownian bridges over `(x,y) ~ pi_epsilon` therefore has endpoint
marginals:

```text
rho_0 = mu_k,
rho_1 = mu_{k+1}.
```

SF2M's score and flow matching objectives identify the drift/score fields of
that bridge mixture in the population optimum. If those objectives are fit
exactly, the learned SDE has the same bridge marginals, including the target
endpoint marginal.

So distribution recovery follows from Schrodinger bridge endpoint constraints
and exact score/flow matching.

### Mass Fit Argument

There is no mass-fit argument. The Schrodinger bridge described above is a
probability bridge:

```text
int rho_t(x) dx = 1
```

for all `t`. `NullMassStrategy` is used. The method models stochastic balanced
transport, not proliferation/death. `TMV` is diagnostic only.

### References

- `cytobridge_agent/rag/literature_db/literature/Simulation-free Schrödinger bridges via score and flow matching.pdf`

## `ruot`

### Mathematical Problem

`ruot` targets regularized unbalanced stochastic dynamics. A useful dynamic
form is a positive-measure Fokker-Planck equation with growth:

```text
dX_t = b_t(X_t) dt + sigma dB_t,

partial_t mu_t
    = -div(b_t mu_t) + (sigma^2 / 2) Delta mu_t + g_t mu_t.
```

The corresponding RUOT objective can be understood as an unbalanced
Schrodinger-bridge / Fisher-regularized transport problem:

```text
min_{mu_t, b_t, g_t}
    int [ kinetic energy
          + stochastic / Fisher regularization
          + growth penalty
          + coupling terms between density and growth ] dmu_t dt
subject to
    positive-measure Fokker-Planck-growth dynamics,
    mu_0 = mu_k,
    mu_1 = mu_{k+1}.
```

The exact paper form is more detailed, but the important modeling object is
clear: stochastic transport plus explicit growth/death, with non-conserved
total mass.

### Package Approximation

The package uses a hybrid route:

1. pretrain `velocity + growth` by neural ODE;
2. train the `score` component with flow matching;
3. finetune `velocity + growth + score`.

This is not a fully simulation-free algorithm. It is a practical deep
approximation to RUOT-style stochastic unbalanced dynamics.

### Distribution Fit Argument

If the learned `v`, `g`, and `score` fields fit the positive-measure
Fokker-Planck-growth dynamics exactly, then the simulated weighted stochastic
process satisfies:

```text
hat_mu_{k+1} = mu_{k+1}.
```

The score term supplies stochastic/diffusive correction; velocity supplies
transport; growth supplies mass change. Exact endpoint positive-measure
recovery implies `W1` distribution recovery.

### Mass Fit Argument

Mass evolves through the growth term:

```text
d/dt hat_mu_t(Omega) = int g_t(x) d hat_mu_t(x).
```

If the final positive measure equals `mu_{k+1}`, then:

```text
hat_mu_{k+1}(Omega) = mu_{k+1}(Omega).
```

Thus total cell-count / mass change is part of the modeled object. `TMV` is a
hard gate.

### References

- `cytobridge_agent/rag/literature_db/literature/Learning stochastic dynamics from snapshots through regularized unbalanced optimal transport.pdf`

## `crufm`

### Mathematical Problem

`crufm` is a CytoBridge-specific stochastic unbalanced flow-matching surrogate.
There is no single external paper whose objective exactly equals the package
implementation.

The package-defined interval problem is:

1. use UOT to define an unbalanced endpoint positive-measure plan;
2. use Brownian-bridge-style stochastic conditional paths to learn velocity and
   score;
3. use UOT row sums to learn growth;
4. recover the UOT target positive measure at the endpoint.

In shorthand:

```text
crufm = vgfm endpoint mass semantics + SF2M-style stochastic bridge regression.
```

### Package Approximation

Runtime semantics:

- Coupling: `UnbalancedOTCouplingStrategy`.
- Path: `RegularizedUnbalancedConditionalPath`.
- Mass: `UOTMassStrategy`.
- Config: `CytoBridge/configs/crufm.yaml`.

The interval UOT plan is the same positive-measure object as in `vgfm`. The
nonzero `sigma` path adds stochastic bridge supervision and score learning.

### Distribution Fit Argument

The endpoint plan still has UOT target marginal:

```text
c_j = sum_i pi(i,j) = b_j.
```

The stochastic path changes the intermediate bridge dynamics, but it does not
change the endpoint marginal encoded by the UOT plan. Exact velocity/score
matching recovers the bridge-marginal dynamics induced by this endpoint plan.

With exact growth fitting, endpoint mass aggregation is the same as Lemma B:

```text
sum_i r_i * P(j | i) = c_j = b_j.
```

Therefore the predicted weighted endpoint measure can match the observed target
distribution.

### Mass Fit Argument

The mass argument is identical to `vgfm` because `crufm` uses the same UOT row
sum terminal mass target:

```text
w_i(1) = r_i = sum_j pi(i,j).
```

The score head changes stochastic bridge geometry, not the target mass
contract. If growth is fit exactly:

```text
sum_i w_i(1) = sum_i r_i = sum_j b_j.
```

So weighted particles recover total target mass. `TMV` is a hard gate.

### Boundaries

- This is a package composition, not a named theorem from one paper.
- Its mass correctness depends on the UOT plan and `UOTMassStrategy`, not on the
  stochastic score head.
- If a future CRUFM variant changes the UOT marginal convention, the mass proof
  must be rewritten.

### References

- `cytobridge_agent/rag/literature_db/literature/Joint Velocity-Growth Flow Matching for Single-Cell Dynamics Modeling.pdf`
- `cytobridge_agent/rag/literature_db/literature/Simulation-free Schrödinger bridges via score and flow matching.pdf`

## `cyto_simulation`

### Mathematical Problem

`cyto_simulation` is the interaction-aware unbalanced stochastic family. It is
closest to an unbalanced mean-field Schrodinger bridge. A useful dynamic form is:

```text
dX_t = [v_t(X_t) + I_t(X_t, mu_t)] dt + sigma dB_t,

partial_t mu_t
    = -div((v_t + I_t[mu_t]) mu_t)
      + (sigma^2 / 2) Delta mu_t
      + g_t mu_t.
```

Here:

- `v_t` is cell-state transport;
- `I_t[mu_t]` is an interaction / mean-field term;
- `g_t` is growth/death;
- the score component represents stochastic bridge correction.

The target object is a positive-measure stochastic dynamics with interaction,
not merely a more expressive neural network.

### Package Approximation

The package uses a hybrid training route with separate pretraining and
finetuning stages for velocity/growth, interaction, score, and the joint model.

This is an approximation to interaction-aware unbalanced stochastic dynamics.
It should only be selected when the data and scientific question justify an
interaction term.

### Distribution Fit Argument

If the interaction field, velocity field, score field, and growth field are all
fit exactly, the predicted positive measure solves the intended
interaction-aware Fokker-Planck-growth equation with endpoint:

```text
hat_mu_{k+1} = mu_{k+1}.
```

The interaction term changes where mass moves, but endpoint distribution
recovery is still measured through the final weighted particle measure.

### Mass Fit Argument

Mass is modeled through the same growth-rate principle:

```text
d log w_t / dt = g_t(X_t).
```

The total predicted mass obeys:

```text
d/dt hat_mu_t(Omega) = int g_t(x) d hat_mu_t(x).
```

Exact positive-measure endpoint recovery implies the predicted total mass
matches the observed target total mass. `TMV` is a hard gate.

### Boundaries

- Interaction is easy to misuse. It is a scientific assumption, not a free
  performance knob.
- If the dataset does not contain evidence supporting interaction semantics,
  prefer a simpler family.

### References

- `cytobridge_agent/rag/literature_db/literature/Modeling Cell Dynamics and Interactions with Unbalanced Mean Field Schrödinger Bridge.pdf`
