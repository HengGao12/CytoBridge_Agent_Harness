---
title: "Core exact-fit lemmas"
summary: "Reusable balanced and unbalanced exact-fit arguments."
read_when:
  - "Reusing exact-fit proof lemmas"
  - "Auditing proposal recoverability arguments"
---
# Builtin Algorithm Theory: Core Lemmas

Reusable exact-fit arguments for balanced endpoint coupling and package UOT mass recovery.

## Two Core Lemmas Used Repeatedly

### Lemma A: balanced endpoint-coupling recoverability

Let `pi(i,j)` be a balanced coupling between probability measures:

```text
sum_j pi(i,j) = a_i,
sum_i pi(i,j) = b_j,
sum_i a_i = sum_j b_j = 1.
```

If a conditional path samples `(x_i, y_j)` from `pi` and defines a path
`X_t = psi_t(x_i, y_j)`, then the path marginal is:

```text
rho_t = (psi_t)_# pi.
```

At the endpoints:

```text
rho_0 = mu_k,
rho_1 = mu_{k+1}.
```

Conditional flow matching trains a vector field `v_t` to match the conditional
path derivative. The population minimizer is the marginal vector field:

```text
v_t(x) = E[d/dt X_t | X_t = x].
```

That vector field satisfies the continuity equation:

```text
partial_t rho_t + div(rho_t v_t) = 0.
```

Therefore, in the exact regression limit, integrating `v_t` recovers the same
path marginals and hence the target endpoint distribution.

This proves distribution recoverability for balanced flow-matching methods.
It does not prove mass recoverability, because all measures are probabilities
with conserved total mass.

### Lemma B: unbalanced endpoint-mass recoverability under package UOT semantics

For each adjacent interval, the package UOT helper constructs a transport plan
`pi(i,j)` between source and target cells.

The current package convention is:

```text
source reference mass: a_i = 1
target reference mass: b_j = 1
```

and the helper is source-relaxed / target-constrained:

```text
min_{pi >= 0}
    <C, pi> + eps * Ent(pi) + lambda * KL(pi 1 | a)
subject to
    pi^T 1 = b.
```

The actual implementation uses the POT unbalanced Sinkhorn helper with a
finite source-side mass penalty and an infinite target-side mass penalty.

Define:

```text
r_i = sum_j pi(i,j)     source row sum
c_j = sum_i pi(i,j)     target column sum
```

Because the target side is constrained:

```text
c_j = b_j = 1
```

under the default target empirical mass convention. More generally, `c_j`
equals the target marginal chosen by the solver.

The package pair sampler samples:

```text
P(i,j) = pi(i,j) / sum_{i',j'} pi(i',j').
```

Equivalently:

```text
P(i) = r_i / M,
P(j | i) = pi(i,j) / r_i,
M = sum_i r_i = sum_j c_j.
```

The package `UOTMassStrategy` assigns a sampled source particle terminal mass:

```text
w_i(1) = r_i
```

under local normalization `w_i(0) = 1`. The path converts this into the growth
target:

```text
g_t(x) = d/dt log w_t(x).
```

For the deterministic endpoint interpretation, source cell `i` carries terminal
mass `r_i` and distributes that mass to targets according to `P(j | i)`. The
mass arriving at target cell `j` is:

```text
sum_i r_i * P(j | i)
= sum_i r_i * pi(i,j) / r_i
= sum_i pi(i,j)
= c_j.
```

Because `c_j` is the target marginal, endpoint aggregation recovers the target
positive measure.

Important scale convention:

- no extra `1 / a_i` factor is needed because the current UOT source reference
  mass is `a_i = 1`;
- if a future solver uses normalized source masses `a_i = 1 / n_k`, then the
  terminal mass ratio should be `r_i / a_i`;
- this document describes the current package runtime, not every possible UOT
  convention.

This proves the package-level unbalanced exact-fit story:

- velocity/path fitting recovers endpoint geometry;
- growth fitting recovers source terminal mass;
- endpoint aggregation recovers the UOT target marginal;
- total predicted mass matches the total target mass represented by `b`.
