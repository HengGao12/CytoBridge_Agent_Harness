# Runtime And Scalability Checks

Use this reference after proposal-to-code consistency is clear.

## Data Contract

Before training, confirm the active data provides:

- `adata.obs["time_point_processed"]`
- `adata.obsm["X_latent"]`

Extra modalities may augment `X_latent`, but must not replace it. If the
algorithm needs raw `adata` manipulation, that should be expressed in
`training_data_builder(...)`. Backend builders should consume prepared
`build_context.training_data`, not redo ad hoc data management.

If preprocessing is unclear, read:

- `~/.cellcompass/skills/workflow/preprocessing-execution/SKILL.md`

## Scalability And Efficiency

Check the actual hot path, not just flags:

- are there Python loops over cells or full pairwise matrices?
- can heavy operations stay on GPU?
- is CPU OT/UOT transfer minimized and localized?
- does the code avoid full `n_t x n_{t+1}` cost/mask/plan/affinity/kernel/logit
  state on real time gaps?
- for each adjacent pair, what is `n_t * n_{t+1}` and how many dense float32
  matrices are live?
- if pairwise costs are modified, are they streamed/chunked/sparse, or are
  several full dense matrices built before chunking?

Default expectation:

- no naive dense full-plan logic for large time pairs
- no cell-wise Python loop in the hot path unless bounded by small sampled
  batches
- Weinreb-scale data is a scalability smoke test for real-data algorithms
- toy-panel success is not enough if the proposal targets biological data

Valid bounded-memory strategies include mini-batch, chunking, sparse candidates,
landmarks, coresets, GPU-vectorized bounded kernels, streaming state, or another
proposal-consistent estimator. A `use_mini_batch` flag alone is not evidence if
full adjacent-pair state is still constructed first.

### OT Coupling Solver Acceleration

For POT-based OT/UOT/WFR coupling solves, keep the solver tensors on the target
device whenever possible:

- `vgfm` / package UOT uses `ot.unbalanced.sinkhorn_unbalanced(...)` with
  CUDA tensors when `device="cuda"`, so the Sinkhorn solve can run on GPU.
- `wfrfm` uses `ot.unbalanced.mm_unbalanced(...)`; this POT solver also accepts
  CUDA tensors, so strict WFR-OET coupling can run on GPU without changing the
  mathematical objective.
- Do not convert `a`, `b`, or the cost matrix to NumPy immediately before a POT
  solve unless CPU execution is intentional. A stray `np.asarray(...)` or
  `.cpu().numpy()` in the solver hot path silently moves coupling work back to
  CPU.
- After solving, it is fine to copy the bounded plan/subplan back to CPU if the
  package sampler stores plans as NumPy arrays. The important part is that the
  expensive iterative solver receives GPU tensors.

This is an implementation acceleration trick, not a change in proposal
semantics. If switching solver family, regularization, marginal relaxation, or
approximation objective, patch `PROPOSAL.md` for proposal review or document the approximation in
`IMPLEMENTATION_MAP.md`.

For the detailed pattern, read:

- `../../campaign-tuning/references/scalable-tuning-and-failures.md`

For concrete package code examples, inspect
`CytoBridge-main/CytoBridge/tl/flow_matching_backends.py`:

- `BalancedOTCouplingStrategy` for balanced OT-CFM/SF2M-style chunked pairing
- `UnbalancedOTCouplingStrategy` for VGFM/CRUFM-style UOT pairing
- `WFROETCouplingStrategy` for WFR-FM chunked semi-coupling
- `_split_transport_chunks(...)`
- `_solve_uot_from_pairwise_cost(...)`
- `_solve_balanced_ot_from_pairwise_cost(...)`

Also inspect the matching builtin YAMLs under
`CytoBridge-main/CytoBridge/configs/` before approving custom config fields.

## Mass Objective Review

If the algorithm changes `MassStrategy`, `gt`, growth supervision, or terminal
mass baselines, verify the math before training.

Package default meaning:

- the network predicts `g = d/dt log w`
- `w(t)` is evolving particle mass
- `MassStrategy` defines the interval terminal mass target
- the conditional path converts that terminal target into a local growth target
  `gt` plus a separate optimization loss weight
- `gt` supervises log-mass dynamics; it is not a generic normalization factor
- training loss weight is an optimization weight, not current path mass

Default UOT semantics are consistent when left unchanged:

- builtin UOT row sums provide terminal mass ratio
- unit source/target reference marginals put row sums in the correct unit-mass
  convention
- local log-mass interpolation gives `gt_local = log(terminal_mass)` under
  `w(0)=1`
- the backend converts this into a rate for `d/dt log w`

Ask:

- what exact quantity does `gt` represent?
- what terminal baseline is assumed?
- does it depend on sample counts?
- after integrating `g`, what total mass should inference produce?
- is that definition compatible with downstream TMV?

High-risk patterns:

- baselines proportional to `1/n`
- baselines using `n_source`, `n_target`, local bucket sizes, or adjacent count
  ratios without derivation
- treating sampling density as desired biological mass
- post-hoc weight rescaling or normalization after rollout to pass TMV
- time-only count-ratio clocks whose only evidence is the observed target count
- ignoring the trained growth output at inference and replacing it with a
  separate mass correction

A time-only or global mass component is acceptable only if the approved proposal
defines its biological meaning, allowed training evidence, inference-time data
contract, and how it avoids evaluation target leakage. Otherwise treat it as a
shortcut, not a growth model.

If mass semantics cannot be justified, revert to builtin mass strategy or revise
the proposal before trusted training.

## Config Sanity

Check:

- each `training.plan[*].mode` matches the algorithm
- custom heads added by `model_builder(...)` are actually trainable through
  `model.cytobridge_component_modules`, stage `trainable_modules`, or
  `cytobridge_trainable_parameters(...)`
- custom training loops preserve those gradients until `optimizer.step()`.
  When using multiple losses, separate `backward()` calls, or manual stage
  runners, verify that `zero_grad()`, `detach()`, `no_grad()`, or a separate
  optimizer step does not silently erase the gradients for proposal-required
  trainable heads before they can be updated.
- balanced algorithms do not leave incompatible growth settings enabled
- nonstandard `neural_ode` / custom stages are justified by proposal/code
- `epochs`, `lr`, and `batch_size` changes have convergence or numerical
  rationale
- custom metric expected ranges are declared when known

Epoch and warm-start fairness:

- start custom flow-matching from 3000 epochs by default
- do not reduce epochs only to satisfy wall-clock budget
- do not use prior trained checkpoints, cached fitted state, or target-specific
  predictions unless the approved proposal defines a generalizable warm-start
