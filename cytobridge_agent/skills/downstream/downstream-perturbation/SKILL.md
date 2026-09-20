---
name: downstream-perturbation
description: In silico perturbation and counterfactual downstream analysis for CytoBridge dynamics models.
---

# Downstream Perturbation

Use this skill when the question asks about gene KO/OE, condition scans,
intervention effects, or counterfactual dynamics.

## Analysis Logic

Perturbation analysis is a counterfactual rollout problem. Apply a defined
intervention to the initial state or gene expression, project it into the model
state space when needed, then generate matched control and perturbed
trajectories from the same source-cell distribution. Effects should be measured
on rollout-derived readouts: fate probabilities, trajectory displacement,
growth/mass change, driver response, or gene-space dynamics.

Direct gene perturbations require a valid gene-to-latent transformation. If the
intervention is only latent-space, call it exploratory and do not overstate it
as a concrete gene KO/OE.

Perturbation analysis should usually reuse the same readouts as the main
biological question: fate probabilities, growth weights, transition matrices,
driver responses, or predicted endpoint distributions.
Define the sign convention before reporting effects. Unless the active
scientific question states otherwise, report perturbation effects as
`perturbed - matched_control`. The same convention should be used for fate,
growth, expression, trajectory, and driver-response readouts.

## Read First

- `cytobridge_agent/skills/workflow/downstream-analysis/SKILL.md`
- `CytoBridge-main/docs/runtime/downstream/semantics-and-evidence.md`
- `CytoBridge-main/docs/runtime/downstream/api-reference.md`
- source when details matter:
  `CytoBridge-main/CytoBridge/tl/perturbation.py`

## Core Rules

- Perturbation must be applied in a defined feature space.
- Compare matched control and perturbed rollouts from the same source cells.
- Do not fit, calibrate, or choose a perturbation operator from the same
  observed perturbation endpoint that is later used as the validation target.
  This is endpoint leakage, not predictive perturbation validation.
- If observed perturbation endpoints are used to estimate any latent shift or
  response operator, validation must use an independent holdout: a held-out
  replicate, held-out condition, held-out dose, held-out time point, or a
  pre-registered split that was not used to define the operator. If no such
  independent evidence exists, label the result descriptive or exploratory.
- Prefer a defined intervention on the initial/source state, gene expression,
  condition covariate, or model state followed by matched-control and perturbed
  rollouts through the trained model. Endpoint-derived latent translations may
  be shown only as descriptive endpoint alignment, not as a claimed model
  prediction.
- Report deltas as `perturbed - matched_control` and record this convention in
  the manifest or table metadata.
- Report fate, growth, trajectory, or expression effects only when the relevant
  readout exists and is documented.
- If direct gene-space perturbation is unavailable, label latent perturbation as
  exploratory.
- Reuse the same model-native rollout semantics as unperturbed downstream
  trajectories. Do not pass or invent `sigma`, `method`, SDE/ODE mode, score
  usage, or growth-weight semantics in perturbation code. These must come from
  the trained model, locked evaluation trajectory, or resolved config.
- For velocity-only or velocity-growth models with `sigma: 0.0`, perturbation
  rollouts should be deterministic matched-control rollouts. Adding stochastic
  noise to "explore uncertainty" changes the model semantics and can invalidate
  fate/perturbation readouts.
- Do not hand-write a low-level Euler/SDE perturbation loop as the first path.
  Use `generate_trajectory_dataset(...)`, locked evaluation trajectories, or the
  package perturbation APIs first; only drop to low-level code after a concrete
  API failure, and record why.
- Do not hardcode external task file schemas in this skill.

## Recommended Outputs

- perturbation design table
- matched control and perturbed rollout artifacts
- effect summary table
- figures
- manifest with feature space, perturbation strength, target genes, and warnings

## Minimal Template

```python
from pathlib import Path
import json
import scanpy as sc
from CytoBridge.utils import load_model_from_adata
from CytoBridge.tl.perturbation import generate_trajectory_dataset

adata = sc.read_h5ad(input_h5ad)
model = load_model_from_adata(adata)
out = Path(output_dir) / "downstream" / "perturbation"
ctrl_dir = out / "control"
pert_dir = out / "perturbed"
ctrl_dir.mkdir(parents=True, exist_ok=True)
pert_dir.mkdir(parents=True, exist_ok=True)
warnings = []

control = generate_trajectory_dataset(
    adata=adata,
    model=model,
    n_cells=500,
    n_steps=100,
    time_key="time_point_processed",
    cell_type_key=label_key,
    output_space="both",
    save_dir=str(ctrl_dir),
    device=device,
)
perturbed = generate_trajectory_dataset(
    adata=adata,
    model=model,
    n_cells=500,
    n_steps=100,
    time_key="time_point_processed",
    cell_type_key=label_key,
    output_space="both",
    perturbed_latent=perturbed_latent,
    save_dir=str(pert_dir),
    device=device,
)
(out / "manifest.json").write_text(json.dumps({
    "analysis": "perturbation",
    "feature_space": perturbation_feature_space,
    "control_metadata": control.get("metadata", {}),
    "perturbed_metadata": perturbed.get("metadata", {}),
    "warnings": warnings,
}, indent=2), encoding="utf-8")
```
