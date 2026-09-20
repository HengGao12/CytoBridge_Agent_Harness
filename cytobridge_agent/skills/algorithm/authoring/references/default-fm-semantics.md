# Path rule

- this reference file path is relative to the skill directory `~/.cellcompass/skills/algorithm/authoring/`
- so `references/...` means a sibling under that same skill folder
- package docs and source paths in this file are workspace-relative

# Default Flow-Matching Semantics

This file is an authoring index. It should not duplicate the full builtin
theory. For mathematical intent and exact-fit arguments, read:

- `CytoBridge-main/docs/theory/README.md`
- `CytoBridge-main/docs/runtime/flow-matching/README.md`

Use this file only when deciding how to implement or override package runtime
hooks.

## Builtin Families To Calibrate Against

- `balanced_ot_cfm`: balanced OT coupling, deterministic linear path, no mass.
- `sf2m`: entropic balanced OT coupling, stochastic bridge path, no mass.
- `vgfm`: UOT coupling, deterministic velocity-growth path, UOT mass targets.
- `wfrfm`: WFR-OET semi-coupling, traveling-Gaussian WFR path, WFR terminal mass targets; default configs use `delta: auto` so the WFR length scale follows adjacent-time latent distances.
- `crufm`: UOT coupling, stochastic velocity-growth-score path, UOT mass targets.

Use `vgfm` for deterministic unbalanced velocity-growth flow matching in
proposals, configs, docs, and benchmark cards when the intended semantics are
the package's standard UOT endpoint surrogate. Use `wfrfm` when the intended
semantics are explicitly WFR dynamic unbalanced OT. Do not copy a fixed WFR
`delta` across datasets by habit: use the default auto value. A fixed `delta`
is only appropriate for an explicit controlled comparison or ablation.

## Runtime Composition

Flow-matching builtins are assembled from:

- `CouplingStrategy`: which source-target endpoint pairs are sampled.
- `ConditionalPath`: how `x_t`, velocity targets, optional score targets, and
  optional local mass targets are produced.
- `MassStrategy`: how terminal particle mass is assigned for unbalanced methods.

Concrete source paths:

- backend assembly: `CytoBridge-main/CytoBridge/tl/flow_matching_backends.py`
- path formulas: `CytoBridge-main/CytoBridge/tl/flow_matching.py`
- loss application: `CytoBridge-main/CytoBridge/tl/trainer.py`
- config examples: `CytoBridge-main/CytoBridge/configs/`

Concrete builtin examples to inspect:

- `CytoBridge-main/CytoBridge/configs/balanced_ot_cfm.yaml`: balanced
  simulation-free OT-CFM config using `flow_matching.backend: balanced_ot_cfm`.
- `CytoBridge-main/CytoBridge/configs/sf2m.yaml`: stochastic balanced
  bridge/SF2M config using `flow_matching.backend: sf2m`.
- `CytoBridge-main/CytoBridge/configs/vgfm.yaml`: deterministic UOT
  velocity-growth FM config using `flow_matching.backend: vgfm`.
- `CytoBridge-main/CytoBridge/configs/wfrfm.yaml`: WFR-OET FM config using
  `flow_matching.backend: wfrfm`.
- `CytoBridge-main/CytoBridge/configs/crufm.yaml`: stochastic UOT
  velocity-growth-score FM config.

Source symbols for package coupling patterns:

- `BalancedOTCouplingStrategy`: balanced OT endpoint pairing; read the chunked
  branch before implementing balanced simulation-free methods.
- `UnbalancedOTCouplingStrategy`: package UOT endpoint pairing and UOT mass
  target source used by `vgfm`/`crufm`.
- `WFROETCouplingStrategy`: WFR-OET semi-coupling and terminal mass target
  source used by `wfrfm`.
- `_split_transport_chunks(...)`: shared adjacent-time chunk partition helper.
- `_solve_uot_from_pairwise_cost(...)`: chunked UOT solver path that keeps POT
  solver tensors on the target device and records sampler-compatible
  `sub_plans`.
- `_solve_balanced_ot_from_pairwise_cost(...)`: chunked balanced OT solver path.
- `_build_balanced_coupling(...)`, `_build_sf2m_coupling(...)`,
  `_build_unbalanced_coupling(...)`, `_build_wfr_coupling(...)`: config-to-code
  builders that show which YAML fields actually control each builtin.

## UOT Solver Semantics

For UOT-based builtins, the package helper currently uses unit empirical source
and target reference masses and solves a source-relaxed / target-constrained
problem:

- source marginal penalty: finite `reg_m`
- target marginal penalty: `inf`
- pair sampling: normalized stored transport plan
- growth target: `g(t, x) = d/dt log w`

Important consequence:

- changing pairwise cost changes geometry, not the solver convention;
- changing source/target masses changes the UOT measure contract;
- changing the marginal-relaxation side or KL placement changes the method's
  semantics and usually requires proposal revision.

## Override Rules

If the method differs only in pairwise geometry:

- subclass `CostBasedPairwiseOTCouplingStrategy` and override
  `build_pairwise_cost(...)` only when the full adjacent-pair cost matrix fits
  the target scale. An adjacent gap with about 1000 source cells and 1000 target
  cells yields a `1000 x 1000` matrix, which is normally small enough to run as
  one full block;
- subclass `ChunkedTransportCouplingStrategy` and override
  `build_pairwise_cost_block(...)` when the pairwise geometry must scale to real
  data without materializing a full cost matrix;
- keep the package solver/sampler lifecycle unless the proposal requires a
  genuinely different coupling convention.

Important: returning `PairwiseCost` can still mean a full dense cost matrix was
materialized. For real-scale algorithms, inspect the memory path. A
`use_mini_batch` flag is not a proof of scalability if the full cost, mask,
kernel, or plan exists before chunking.

`ChunkedTransportCouplingStrategy` is the preferred API for the common
large-data case. It splits each adjacent time gap, calls your
`build_pairwise_cost_block(...)` for bounded source/target blocks, solves OT/UOT
once during `build_state(...)`, caches subplans, and samples pairs from that
cached state during training.

Builtin mini-batch semantics:

- mini-batch/chunked OT is an estimator-level approximation to the same
  transport-derived flow-matching target
- each time gap is chunked independently
- balanced chunks normalize local source/target masses
- UOT chunks keep the sliced local source/target mass vectors and preserve the
  same marginal-relaxation convention
- sampled training pairs still come from transport mass, not from arbitrary
  nearest-neighbor pairing or future-count correction
- do not downsample by default; cell count by time point may be a biological
  mass/growth signal and more valid cells usually improve coupling evidence
- if debugging forces a smaller dataset, ratio-based subsampling preserves
  timepoint count proportions; fixed per-timepoint subsampling does not
- formal biological campaign evidence should use the full prepared real dataset
  whenever computationally feasible; smaller panels are smoke/debug evidence
  unless larger or full-data validation supports the same claim

Solver device discipline:

- package UOT/VGFM-style coupling sends `ot.unbalanced.sinkhorn_unbalanced(...)`
  CUDA tensors when training on GPU;
- package WFR-FM sends `ot.unbalanced.mm_unbalanced(...)` CUDA tensors when
  training on GPU;
- when implementing a custom POT-based coupling, avoid converting cost blocks or
  mass vectors to NumPy immediately before the solver call unless CPU execution
  is deliberate;
- copying bounded solver results back to NumPy for sampler storage is fine, but
  the expensive iterative solve should stay on the target device when possible.

If the method needs the same solver family but different reference masses:

- use `CostBasedPairwiseOTCouplingStrategy` only if the full pairwise object is
  acceptable for the target scale, otherwise use
  `ChunkedTransportCouplingStrategy`;
- return `PairwiseCost(..., source_mass=..., target_mass=...)`.

If the method needs fixed/custom regularization:

- set `reg` and `reg_m` in the flow-matching coupling config when ordinary
  UOT semantics are unchanged;
- if a custom cost changes the numeric scale relative to the nearest builtin,
  inspect representative cost statistics and set `normalize_cost`, `reg`, and
  `reg_m` deliberately instead of relying on auto-regularization in preview;
- only override solver code if the algorithm needs a different UOT convention.

If the method needs a different solver convention, such as symmetric UOT,
different KL placement, different semi-relaxed side, or paper-specific
normalization:

- implement `compute_ot_coupling(...)` directly or write a custom
  `CouplingStrategy`;
- document which marginal is relaxed, how pairs are sampled, and how terminal
  mass is produced;
- patch `PROPOSAL.md` and wait for proposal review if this changes the approved theory contract.

## Loss And Evaluation Semantics

Current trainer behavior:

- velocity loss is weighted MSE against `ut`;
- growth loss is weighted MSE against `gt`;
- score loss is active only when the score head is trained;
- additional custom metrics must be additive and must not replace builtin
  `W1` / `TMV`.
- extra custom heads are trainable through
  `model.cytobridge_component_modules`, stage `trainable_modules`, or
  `cytobridge_trainable_parameters(...)`; do not register unrelated heads under
  `growth_net` or `interaction_net` merely as an optimizer workaround.

Current evaluation behavior:

- simulate from the earliest observed time point to future observed time points;
- compare predicted weighted particles against observed future snapshots;
- `W1` evaluates distribution fit;
- `TMV` evaluates predicted total mass relative to the first predicted slice
  against observed relative cell-count change.
- the predicted trajectory may be fixed-count weighted particles or
  variable-count explicit birth/death/splitting particles. For explicit
  splitting with unit weights, use unit weights at t0 too; TMV then compares
  `n_pred(t) / n_pred(t0)` to `n_obs(t) / n_obs(t0)`.

TMV is a hard gate only when the algorithm intentionally models unbalanced
mass. For balanced-only methods, TMV remains diagnostic and must not be used as
the reason to reject a theoretically balanced algorithm.

For unbalanced methods, TMV must be earned by the rollout mass dynamics. The
predicted weights should come from the trained growth field or another
proposal-approved WFR/UOT/SB/proliferation-death mechanism. Do not add a
post-hoc target-count correction, time-only count-ratio clock, or renormalizing
layer solely to make TMV pass. If a global/time-only growth component is part of
the intended method, the proposal must justify its biological meaning, training
evidence, inference data contract, and lack of target leakage before it is used
as trusted evidence.

## Authoring Checklist

Before training a custom algorithm, state in `IMPLEMENTATION_MAP.md`:

- closest builtin: `balanced_ot_cfm`, `sf2m`, `vgfm`, `wfrfm`, `crufm`, or another family;
- changed component: coupling, path, mass, loss, simulation, or metric;
- whether the method is balanced-only or unbalanced;
- whether TMV is a hard gate or diagnostic;
- exact hook or class implementing each proposal pseudocode step.
