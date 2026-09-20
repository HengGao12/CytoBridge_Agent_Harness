---
title: "Downstream analysis developer API map"
summary: "Agent-facing map for CytoBridge downstream analysis, model-native evidence, package APIs, and artifact contracts."
read_when:
  - "Planning downstream biological analysis after training"
  - "Choosing downstream compute or plot APIs"
  - "Checking whether a downstream claim is supported by model-native evidence"
---
# Downstream Analysis Developer API Map

Use this directory when implementing, reviewing, or debugging downstream
analysis after a CytoBridge model has been trained. The goal is to choose the
smallest correct package API, keep biological claims artifact-backed, and avoid
turning evaluator file contracts into general scientific rules.

Read in this order:

1. this file
2. `semantics-and-evidence.md`
3. `model-semantics.md`
4. `api-reference.md`
5. `recipes-and-artifacts.md`
6. `checklist-and-failures.md`

If the task is still model training or custom algorithm development, read
`docs/runtime/flow-matching/README.md` or
`docs/runtime/custom-algorithms/README.md` instead. Downstream docs assume a
trained model or a locked evaluation trajectory already exists.

Downstream analysis is code, not just prose. Important analyses should be
implemented as saved scripts, and those scripts should name the package API
source file they depend on. If an agent is unsure how an API uses an input,
what shape it returns, or whether a result is latent-space or projected
gene-space, it should inspect the source file listed in `api-reference.md`
before proceeding.

## Fixed Downstream Data Contract

Package-native downstream bundle APIs expect the trained-data schema:

- `adata.obs["time_point_processed"]`: canonical processed time axis.
- `adata.obsm["X_latent"]`: canonical state used by the trained dynamics model.
- optional `adata.obsm["velocity_latent"]`: local velocity diagnostic.
- optional `adata.obs["growth_rate"]` or `adata.obsm["growth_rate"]`: growth
  field `g(t, x) = d/dt log w`.
- optional `adata.varm["PCs"]`: latent-to-gene projection for gene-space
  velocities, drivers, or GRN summaries.

Do not silently replace `X_latent`, reinterpret the time axis, or make
gene-level claims from latent-only outputs without a documented projection path.

## Evidence Boundary

CytoBridge is a neural continuous dynamics package. Downstream claims about
paths, intermediate times, fate over time, perturbations, or mass dynamics
should come from trained-model rollout, final-regression evaluation
trajectories, or package-native trajectory/perturbation APIs.

Local stream plots, saved per-cell velocity arrays, static embedding
neighborhoods, and expression differences are diagnostics. They can support
visualization and hypothesis generation, but they do not replace model-native
trajectory evidence.

## Choose The Smallest Correct Analysis Path

| Need | Start with |
| --- | --- |
| Model-native trajectory or fate over time | `recipes-and-artifacts.md`, trajectory recipe |
| Which model components exist and whether to use ODE or SDE | `model-semantics.md` |
| Velocity field or stream diagnostics | `api-reference.md`, velocity APIs |
| Growth or mass dynamics | `api-reference.md`, growth APIs |
| Driver genes or GRN/Jacobian | `api-reference.md`, driver/GRN APIs |
| Perturbation or counterfactual dynamics | `recipes-and-artifacts.md`, perturbation recipe |
| Prediction/interpolation/extrapolation | `recipes-and-artifacts.md`, prediction recipe |
| Report/paper downstream section | `recipes-and-artifacts.md`, integrated story recipe |
| Debugging weak or contradictory downstream results | `checklist-and-failures.md` |

## External Evaluation Boundary

External tasks may require fixed filenames or schemas. Those file contracts
belong to the task runner or export tool. The downstream package docs define
scientific computations and artifact provenance, not benchmark submission
schemas.

The correct order is:

1. train or load the model;
2. run model-native downstream computation;
3. save generic artifacts and manifests;
4. convert those artifacts to the external file contract with the task-specific
   export layer.

Do not encode external evaluator filenames in general downstream skills or
general biological analysis scripts.

## Source Escalation

Use docs before source. If docs are insufficient, inspect source in this order:

1. `CytoBridge-main/CytoBridge/tl/downstream/__init__.py`
2. `CytoBridge-main/CytoBridge/tl/downstream/*.py`
3. `CytoBridge-main/CytoBridge/pl/downstream/__init__.py`
4. `CytoBridge-main/CytoBridge/pl/downstream/*.py`
5. `cytobridge_agent/tools/downstream_analysis_toolkit.py` for agent-tool
   wrappers and guarded subprocess behavior

The API reference lists the source path for every downstream bundle. If a
figure, table, or claim depends on an exact shape or side effect, inspect that
source path before trusting the result. Record the checked source path, input
shape, output shape, and warnings in the downstream script or manifest.
