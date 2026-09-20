---
title: "Builtin theory overview and contract"
summary: "Definitions, evaluation contract, and builtin algorithm summary table."
read_when:
  - "Comparing builtin algorithm families"
  - "Checking theory/evaluation definitions"
---
# Builtin Algorithm Theory: Overview And Contract

Canonical definitions, evaluation contract, and builtin family table.

# Builtin Algorithm Principles

This is the canonical package-level theory reference for CytoBridge builtin
algorithms.

Use it before:

- choosing builtin baselines;
- proposing a custom algorithm relative to builtin capabilities;
- deciding whether a method is balanced or unbalanced;
- deciding whether `TMV` is a hard gate or only diagnostic;
- reviewing whether an algorithm has a real exact-fit argument.

The standard expected here is intentionally high. A builtin description should
answer:

- what mathematical problem the method is approximating or solving;
- whether that problem is a balanced probability problem or an unbalanced
  positive-measure problem;
- why distribution fit is plausible in the exact-fit limit;
- if mass is modeled, why weighted particles recover total cell-count / mass
  change;
- if mass is not modeled, why TMV must be diagnostic only.

Skills should point here instead of duplicating builtin-theory sections.

## Notation And Evaluation Contract

For adjacent observed time points `t_k < t_{k+1}`, let:

- source cells be `x_i^k`, `i = 1, ..., n_k`;
- target cells be `y_j^{k+1}`, `j = 1, ..., n_{k+1}`;
- source empirical measure be `mu_k`;
- target empirical measure be `mu_{k+1}`;
- learned velocity be `v_t(x)`;
- learned growth rate be `g_t(x) = d/dt log w_t(x)`;
- learned score be `s_t(x)`, when a stochastic head is present;
- predicted positive measure be `hat_mu_t`.

Balanced methods model probability measures. They can fit the shape of
`mu_{k+1}` but intentionally conserve total mass.

Unbalanced methods model positive measures. They can fit both the shape and the
total mass of `mu_{k+1}`.

CytoBridge evaluation uses forward-simulated particles:

- `W1` compares the observed later snapshot with the predicted weighted
  particle measure in latent space.
- `TMV` compares total predicted relative mass with the observed relative
  cell-count / total-mass change.
- `TMV` is a hard gate only for methods that explicitly model unbalanced mass.
- Balanced-only methods still report `TMV`, but `TMV` is diagnostic only.

## Builtin Summary Table

| builtin | problem family | components | training style | mass model | TMV gate |
|---|---|---|---|---|---|
| `dynamical_ot` | balanced dynamic OT | `velocity` | neural ODE | no | diagnostic |
| `balanced_ot_cfm` | balanced OT-CFM dynamic OT | `velocity` | flow matching | no | diagnostic |
| `unbalanced_ot` | deterministic unbalanced dynamic OT / WFR-style growth transport | `velocity`, `growth` | neural ODE | yes | hard |
| `vgfm` | deterministic unbalanced velocity-growth flow matching | `velocity`, `growth` | flow matching | yes | hard |
| `wfrfm` | simulation-free WFR dynamic unbalanced OT | `velocity`, `growth` | flow matching | yes | hard |
| `sf2m` | balanced Schrodinger bridge / SB-CFM | `velocity`, `score` | flow matching | no | diagnostic |
| `ruot` | regularized unbalanced stochastic OT | `velocity`, `growth`, `score` | hybrid ODE/FM/ODE | yes | hard |
| `crufm` | CytoBridge stochastic unbalanced FM surrogate | `velocity`, `growth`, `score` | flow matching | yes | hard |
| `cyto_simulation` | unbalanced mean-field Schrodinger bridge with interaction | `velocity`, `growth`, `score`, `interaction` | hybrid ODE/FM/ODE | yes | hard |
