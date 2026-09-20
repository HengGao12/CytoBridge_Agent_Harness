---
name: downstream-analysis
description: Unified workflow-stage entry skill for CytoBridge downstream biological analysis, visualization, and model-native evidence generation after training.
---

# Downstream Analysis

Use this as the single entry point for all post-training downstream work:
biological interpretation, continuous trajectories, fate, growth/mass,
driver genes, perturbation, prediction, figures, tables, and report-ready
evidence.

Do not start from an external file contract, a static embedding, or an old
figure. Start from what the trained CytoBridge model is.

For any downstream task involving temporal prediction, trajectories, fate,
growth or mass, perturbation response, or driver genes, a trained or loaded
CytoBridge dynamics model is required whenever the package supports the data.
Standalone sklearn/Ridge/RandomForest or hand-written PyTorch predictors are
not substitutes for the CytoBridge model. They may only post-process
model-native rollouts, trajectories, latent states, weights, or neural
components. If no model can be trained or loaded, stop and report the package
or data-contract blocker instead of silently completing the task with a
separate non-CytoBridge model.

## Core Mental Model

A trained CytoBridge model is a neural generative continuous dynamics model.
It can generate states along a continuous time axis from source cells. Depending
on the fitted components, it may also carry growth/mass weights, stochastic
score dynamics, or interaction forces.

This means downstream analysis should usually be built from:

- model-native rollout or locked evaluation trajectories;
- differentiable neural components such as velocity, growth, score, and
  interaction networks;
- valid projections from latent space back to gene space when gene-level claims
  are needed;
- classifiers or label contracts layered on generated trajectories when fate or
  lineage readouts are needed.

The central downstream question is: what new biological or dynamical evidence
can be obtained because the model is a learned continuous generative neural
system, rather than only a static embedding or a one-shot endpoint predictor?
Use that question to design analyses.

## One Workflow

1. Confirm active data artifact, trained model path, resolved config, output
   directory, time key, label keys, and existing final-regression/evaluation
   trajectory artifacts.
2. Identify model components: velocity, growth, score, interaction, or custom
   simulation hook. Decide whether the model is balanced, unbalanced,
   stochastic, or interaction-aware.
3. Define the biological/dynamical question before choosing tools. If this
   downstream work is intended to support a biological-application paper, first
   read `~/.cellcompass/skills/workflow/biological-story-building/SKILL.md`.
   Use it to decide the dataset story and evidence gap, then read the narrow
   downstream skills needed for that story.
4. Read one or more narrow downstream skills that match the question before
   writing analysis code. This is mandatory, not optional. Start with the
   smallest relevant skill, then combine skills when the biological question
   requires a multi-part analysis.
   If the task requires training or retraining a CytoBridge model before
   downstream analysis, first read
   `~/.cellcompass/skills/workflow/training-orchestration/SKILL.md` and follow
   its training-budget rules. A shortened-epoch training run is debug evidence,
   not a substitute for a real trained model unless dataset/model-specific
   convergence evidence justifies that schedule.
5. Run package-native APIs or a saved script under
   `<output_dir>/scripts/downstream/*.py`.
   - The saved script may orchestrate package calls and post-process model
     outputs, but it must not replace the primary dynamics model with an
     unrelated standalone predictor unless a concrete package/API blocker has
     been recorded.
   - For multi-file downstream deliverables, first create or load a
     CytoBridge training artifact and resolved config, then write scripts that
     consume that model. A "script-backed pipeline" is not acceptable if the
     script itself trains or implements a separate non-CytoBridge dynamics
     model instead of calling CytoBridge training/model APIs.
6. Save tables, figures, arrays, and a manifest under the current output
   directory.
7. For manuscript-quality downstream figures, read
   `~/.cellcompass/skills/workflow/scientific-visualization/SKILL.md` before
   final rendering. Use it to improve visual grammar, layout, and export
   quality, not to alter analysis outputs.
8. Interpret only what the artifacts support; preserve warnings and limits.
9. Commit downstream outputs back into workflow state.

## Analysis Navigation

This is guidance, not a rigid decision tree. You may combine skills or design a
new analysis workflow when the biology calls for it, but the workflow must
still follow the generative-model logic: define the dynamical question, produce
model-native evidence, attach valid biological readouts, then write claims from
the resulting artifacts.

- Continuous paths, fate timing, branch timing, or endpoint flow usually start
  from `downstream-trajectory-fate`; generate or reuse model-native
  trajectories before plotting. For trajectory figures, use the package
  plotting helpers indexed there (`plot_ode_trajectories_bundle` or
  `plot_sde_trajectories_bundle`) before custom plotting, and do not use a
  single averaged/centroid path as the main dynamics panel.
- Stable states, terminal basins, multimodality, or one-route-versus-many-route
  questions usually start from `downstream-state-structure`.
- Future, intermediate, held-out, or extrapolated prediction usually starts
  from `downstream-prediction`.
- Proliferation, depletion, mass, growth, or TMV-like behavior usually starts
  from `downstream-growth-mass`.
- Gene mechanisms, regulators, or drivers usually start from
  `downstream-driver-genes`, after defining the model-derived target.
- KO/OE, condition scans, or counterfactuals usually start from
  `downstream-perturbation`.
- Stream plots start from `downstream-scvelo-stream`, but remain local flow
  diagnostics unless paired with rollout evidence.
- Source-target, lineage, clone, or fate transition matrices usually start from
  `downstream-lineage-transition`.

For deeper analysis, combine paths. Examples: generate continuous trajectories,
classify fate over time, then run driver-gene analysis on the branch-shift
target; or compute growth/mass rollouts, identify expansion phases, then
analyze growth Jacobian drivers. The template code in narrow skills is a
starting point, not a ceiling.

For paper-oriented biological downstream work, do not run modules as a checklist.
Use `biological-story-building` to choose one dataset-level story worth
explaining deeply, then run the downstream analyses that make that story
convincing.

## Downstream Skill Index

Read the narrowest matching skill before implementation. Do not proceed from
this entry skill directly to package source or hand-written model reconstruction
unless the matching narrow skill has already been read and is insufficient for
the active question.

- state structure, endpoints, attractors, multimodality:
  `cytobridge_agent/skills/downstream/downstream-state-structure/SKILL.md`
- continuous trajectories, fate over time, rollout-derived flow:
  `cytobridge_agent/skills/downstream/downstream-trajectory-fate/SKILL.md`
- held-out, future, or intermediate state prediction:
  `cytobridge_agent/skills/downstream/downstream-prediction/SKILL.md`
- explicit scVelo-style stream plots:
  `cytobridge_agent/skills/downstream/downstream-scvelo-stream/SKILL.md`
- growth, mass, proliferation, or depletion:
  `cytobridge_agent/skills/downstream/downstream-growth-mass/SKILL.md`
- velocity/growth drivers, gene-level dynamics, GRN/Jacobian analysis:
  `cytobridge_agent/skills/downstream/downstream-driver-genes/SKILL.md`
- source-target lineage or transition summaries:
  `cytobridge_agent/skills/downstream/downstream-lineage-transition/SKILL.md`
- in silico KO/OE, perturbation, or condition scans:
  `cytobridge_agent/skills/downstream/downstream-perturbation/SKILL.md`

### Mandatory Narrow-Skill Routing

If the task contains any of the terms below, first read the listed narrow skill
or skills and follow their package-native templates before writing custom code:

- `trajectory`, `rollout`, `continuous path`, `fate`, `branch`, `endpoint`:
  read `downstream-trajectory-fate/SKILL.md`; trajectory plots must show
  generated path ensembles or time-sliced generated states unless the claim is
  explicitly about a mean path.
- `held-out`, `future`, `intermediate`, `prediction`, `extrapolate`:
  read `downstream-prediction/SKILL.md`.
- `perturbation`, `knockdown`, `knockout`, `overexpression`, `KO`, `OE`,
  `counterfactual`: read `downstream-perturbation/SKILL.md`.
- `driver`, `GRN`, `regulator`, `Jacobian`, `gene mechanism`: read
  `downstream-driver-genes/SKILL.md`.
- `growth`, `mass`, `proliferation`, `depletion`, `TMV`: read
  `downstream-growth-mass/SKILL.md`.
- `source-target`, `lineage`, `transition matrix`, `clone`: read
  `downstream-lineage-transition/SKILL.md`.

For any task with multiple requested downstream artifacts, map each artifact to
the matching scientific analysis type and read the corresponding narrow skill
before coding. For example, trajectory/fate artifacts should use
`downstream-trajectory-fate`, prediction artifacts should use
`downstream-prediction`, perturbation artifacts should use
`downstream-perturbation`, and driver/GRN artifacts should use
`downstream-driver-genes`.

## Reference Index

Use these references when API semantics, shapes, or artifacts matter:

- docs map: `CytoBridge-main/docs/INDEX.md`
- downstream API reference:
  `CytoBridge-main/docs/runtime/downstream/api-reference.md`
- model semantics by builtin family:
  `CytoBridge-main/docs/runtime/downstream/model-semantics.md`
- evidence boundaries and trajectory semantics:
  `CytoBridge-main/docs/runtime/downstream/semantics-and-evidence.md`
- recipes, manifests, and artifacts:
  `CytoBridge-main/docs/runtime/downstream/recipes-and-artifacts.md`
- checklist and failure modes:
  `CytoBridge-main/docs/runtime/downstream/checklist-and-failures.md`

If the docs are not specific enough, inspect the package source paths named in
the API reference before writing code or making claims.

The narrow skills include minimal templates. Treat variables such as
`input_h5ad`, `output_dir`, `label_key`, `device`, `source_time`, and
`target_time` as values from the active workflow context; do not invent them
without checking the current data and model artifacts.

Templates are not mandatory recipes. If the active biological question needs a
new analysis, write a new script that combines package APIs or inspects model
components directly. The constraints are evidence integrity and reproducibility,
not template conformity.

## Analysis Logic By Question

- Trajectory and fate: generate continuous trajectories from the trained model,
  then attach fate labels through a classifier or label contract. Fate changes
  are read from rollout-derived state evolution, not from static metadata alone.
  For per-cell fate tables, keep source-cell provenance from trajectory
  generation and key results by the trajectory artifact's returned
  `init_obs_names`; never assume output row order alone identifies the source
  cell, and never zip predictions with a separately reconstructed root list
  unless order identity has been verified.
- Growth and mass: use growth outputs or rollout weights from growth-capable
  models. Balanced models can support empirical-count diagnostics but not
  mechanistic growth claims.
- Driver genes: start from a model-derived target such as velocity, growth,
  fate shift, perturbation effect, or transition. Gene-level interpretation
  requires measured gene space or documented latent-to-gene projection.
- Prediction: roll the model from source time to held-out or target time and
  compare to the correct target distribution without leaking target snapshots.
- Perturbation: apply a defined intervention in gene or latent space, generate
  matched control and perturbed rollouts, and compare downstream readouts.
- Visualization: figures are evidence presentation, not a separate analysis
  layer. Plot only after the source artifact and its axis semantics are known.
  For paper-ready figures, route through
  `~/.cellcompass/skills/workflow/scientific-visualization/SKILL.md`, preserve
  the upstream script and manifest, and export both PNG and editable vector PDF.

## No-Shortcut Rules

- Do not satisfy a downstream file contract by inventing a new measured-gene
  dynamics script before training or loading a CytoBridge model. The required
  order is: prepare the AnnData contract, train/load the CytoBridge dynamics
  model, record the model artifact/resolved config, then export downstream
  files from model-native rollouts, components, or documented post-processing.
- Do not manually instantiate `DynamicalModel`, reconstruct model state dicts,
  or hand-write ODE/SDE integration as the first implementation path. First use
  the narrow-skill template and package-native helpers such as
  `load_model_from_adata(...)`, trajectory bundle generation, or
  `generate_trajectory_dataset(...)`. Only inspect or write lower-level code
  after the relevant narrow skill and package-native API path fail with a
  concrete error.
- Do not claim continuous paths from nearest-neighbor interpolation or local
  velocity plots. Continuous-time claims require model rollout or locked
  evaluation trajectory artifacts.
- Do not pass training-decided rollout semantics such as `method` or `sigma`
  from downstream tools. They must come from the fitted model components and
  resolved config. `n_steps` may be adjusted only as output sampling density.
- Do not synthesize source-cell IDs for per-cell predictions. Use provenance
  fields such as `init_indices`, `init_obs_names`, `init_times`,
  `sampling_seed`, and `sampling_policy` from the trajectory artifact.
- Do not override trained-model semantics to make a figure easier. If the
  resolved config is missing, repair provenance or report the limitation before
  making a biological claim.
- Do not change labels, filtering, feature space, projection, or model outputs
  only to match a plotting template. Scientific-visualization is for
  presentation quality after the evidence artifact is fixed.
- Do not claim gene-level biology from latent-only outputs unless a valid
  projection or gene-space computation is documented.
- Do not silently mix old and new artifacts.
- Do not write the biological story before the tables and figures exist.
- Do not hide warnings about missing labels, missing PCA loadings, weak
  metadata, failed projection, or fallback classifiers.

## Provenance Minimum

Every important downstream result must have a rerunnable script or manifest
recording:

- input data path and trained model or evaluation trajectory path;
- resolved config or final-regression source;
- API/function names and key parameters;
- feature space, time axis, labels, weights/mass semantics;
- output paths for tables, figures, arrays, and warnings;
- for manuscript figures, the plotting script path plus PNG/PDF figure paths;
- claims supported by each artifact.

## Completion

Call:

`commit_workflow_state(phase="downstream_analysis", ...)`

Include:

- `downstream_results`
- `downstream_summary`
- `downstream_figures`
- artifact paths for figures, tables, arrays, scripts, and manifests

Preferred `downstream_figures` item:

```python
{
    "path": "<figure path>",
    "caption": "<short caption>",
    "analysis": "<what this figure demonstrates>",
}
```
