# CytoBridge Package Capability Map

Use this reference before proposing a new idea or selecting a builtin model
family.

This file is for boundary calibration. It is intentionally short. For the full
package-level theory of builtin algorithms, read:

- `CytoBridge-main/docs/theory/README.md`
- `CytoBridge-main/docs/runtime/flow-matching/README.md`

At idea/proposal stage, do not start by reading low-level source code unless
the question has already become implementation-specific.

## Package Scope

CytoBridge is a deep-learning package for learning continuous generative
dynamics from multi-timepoint single-cell snapshot data. It is not a catch-all
container for arbitrary single-cell bioinformatics problems.

The canonical data contract is still:

- `adata.obs["time_point_processed"]`
- `adata.obsm["X_latent"]`

## Minimal Mental Model

Current builtins combine four dynamical terms:

- `velocity`: deterministic drift / transport.
- `growth`: proliferation, death, or total-mass change.
- `score`: stochasticity / diffusion-style correction.
- `interaction`: interaction or external-field-style coupling.

They also span multiple training paradigms:

- simulation-based neural ODE;
- hybrid neural ODE plus flow matching;
- simulation-free flow matching.

So a proposal should not claim novelty merely because it uses flow matching.
The package already contains both balanced and unbalanced flow-matching
families.

## Builtin Families

### `dynamical_ot`

- Components: `velocity`.
- Training style: neural ODE.
- Theory boundary: balanced deterministic dynamic OT / CNF-style transport.
- Use when mass is approximately conserved and stochasticity is not the target.
- TMV is diagnostic only.

### `balanced_ot_cfm`

- Components: `velocity`.
- Training style: simulation-free flow matching.
- Theory boundary: balanced OT-CFM dynamic OT.
- Uses balanced OT endpoint pairs and deterministic linear conditional paths.
- Use as the balanced simulation-free FM baseline.
- TMV is diagnostic only.

### `unbalanced_ot`

- Components: `velocity + growth`.
- Training style: neural ODE.
- Theory boundary: deterministic unbalanced dynamic transport.
- Use when proliferation/death matters but stochasticity is not required.
- TMV is a hard gate.

### `vgfm`

- Components: `velocity + growth`.
- Training style: simulation-free flow matching.
- Theory boundary: deterministic unbalanced velocity-growth flow matching.
- This is the default deterministic unbalanced FM family.
- TMV is a hard gate.

### `wfrfm`

- Components: `velocity + growth`.
- Training style: simulation-free flow matching.
- Theory boundary: WFR dynamic unbalanced OT solved through WFR-OET
  semi-coupling and a traveling-Gaussian WFR conditional path.
- Default scale handling: `delta: auto` estimates the WFR length scale from
  adjacent-time latent distances. For WFR-FM builtins, prefer auto delta even
  on datasets that previously used fixed constants; use fixed `delta` only for
  an explicit controlled comparison.
- This is the named-WFR alternative to the simpler `vgfm` default.
- Use it when a proposal specifically wants a concrete WFR action and
  semi-coupling derivation; diagnose whether WFR's velocity-growth relation is
  biologically appropriate for the target data.
- TMV is a hard gate.

### `sf2m`

- Components: `velocity + score`.
- Training style: simulation-free flow matching.
- Theory boundary: balanced stochastic bridge via score and flow matching.
- Uses entropic balanced OT tied to the bridge diffusion scale.
- Use when stochasticity matters but total mass is conserved.
- TMV is diagnostic only.

### `ruot`

- Components: `velocity + growth + score`.
- Training style: hybrid neural ODE -> FM -> neural ODE.
- Theory boundary: regularized unbalanced stochastic dynamics.
- Use when mass change and stochasticity both matter and the hybrid route is
  acceptable.
- TMV is a hard gate.

### `crufm`

- Components: `velocity + growth + score`.
- Training style: simulation-free flow matching.
- Theory boundary: CytoBridge-specific stochastic unbalanced FM composition.
- Treat it as `vgfm` plus SF2M-style stochastic score learning.
- Use when stochastic unbalanced dynamics are needed without ODE-stage training.
- TMV is a hard gate.

### `cyto_simulation`

- Components: `velocity + growth + score + interaction`.
- Training style: hybrid neural ODE plus flow matching.
- Theory boundary: interaction-aware unbalanced stochastic dynamics.
- Use only when interaction is a core scientific mechanism.
- TMV is a hard gate.

## What The Package Already Covers

The builtin boundary already includes:

- deterministic balanced transport;
- simulation-free balanced OT-CFM;
- deterministic unbalanced transport;
- deterministic unbalanced FM via `vgfm`;
- WFR dynamic unbalanced FM via `wfrfm`;
- stochastic balanced bridges via `sf2m`;
- stochastic unbalanced dynamics via `ruot` and `crufm`;
- interaction-aware unbalanced stochastic dynamics via `cyto_simulation`.

Strong new ideas should therefore be phrased relative to a sharper gap, such
as:

- identifiability under sparse snapshots;
- geometry-aware or manifold-aware transport;
- lineage-aware coupling;
- partial observability;
- multimodal constraints;
- interaction semantics grounded in data rather than just adding a head;
- principled routing or evaluation across builtin families.

## What Is Not Yet First-Class Builtin Scope

Do not overclaim package coverage. These are not mature builtin defaults:

- fully native time-series spatial transcriptomics workflow end to end;
- geometry-aware / manifold-aware FM as the default;
- lineage-aware coupling as the default;
- partial-observation-specific inference;
- general multimodal dynamics;
- perturbation-centered dynamics.

Some of these can be implemented through custom algorithms, but proposals must
state the additional data assumptions and evaluation evidence.

## Proposal Implication

Before creating a new research idea, answer:

1. Which builtin is closest?
2. What exact semantic component changes: coupling, path, mass, loss,
   simulation, or metric?
3. Is the method balanced-only or unbalanced?
4. If balanced-only, why is distribution fit enough and why is TMV only
   diagnostic?
5. If unbalanced, why do weighted particles recover both distribution and
   cell-count / total-mass changes?

Use `CytoBridge-main/docs/theory/README.md` as the canonical
reference for those answers.
