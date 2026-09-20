---
name: downstream-prediction
description: Model-native prediction, interpolation, extrapolation, and held-out state analysis for CytoBridge models.
---

# Downstream Prediction

Use this skill when the question asks whether a trained model can predict
intermediate, future, held-out, or counterfactual state distributions.

## Analysis Logic

Prediction uses the generative model as a time-conditional simulator. Choose a
source time and target time, roll source cells through the trained model, then
compare the generated distribution to a held-out or explicitly designated
target observation. Interpolation claims use intermediate generated times;
extrapolation claims require extra caution and should be separated from
within-range validation.

Do not use target snapshots to construct the prediction. Do not compare raw and
processed time axes without documenting the mapping.

Prediction analysis can be combined with state-structure, fate, or driver
analysis when the question is not only "is the endpoint close?" but "which
continuous dynamical features make the prediction biologically plausible?"

## Read First

- `cytobridge_agent/skills/workflow/downstream-analysis/SKILL.md`
- `CytoBridge-main/docs/runtime/downstream/api-reference.md`
- `CytoBridge-main/docs/runtime/downstream/semantics-and-evidence.md`

## Operating Rules

- Use model-native rollout or locked evaluation trajectories.
- Record target times, source times, integration time axis, and feature space.
- Compare predictions to observed data only when the target observation is held
  out from training or explicitly intended for validation.
- Do not hardcode external evaluator filenames in this skill.
- If an external task requires fixed files, produce a generic prediction
  artifact and let the task-specific export tool format it.

## Required Outputs

- prediction table, array, or `h5ad` artifact
- prediction manifest with model path, time mapping, feature space, and warnings
- optional distance metrics and plots

## Minimal Template

```python
from pathlib import Path
import json
import scanpy as sc
from CytoBridge.utils import load_model_from_adata
from CytoBridge.tl.perturbation import generate_trajectory_dataset

adata = sc.read_h5ad(input_h5ad)
model = load_model_from_adata(adata)
out = Path(output_dir) / "downstream" / "prediction"
out.mkdir(parents=True, exist_ok=True)
warnings = []

pred = generate_trajectory_dataset(
    adata=adata,
    model=model,
    n_cells=1000,
    n_steps=100,
    init_time=source_time,
    end_time=target_time,
    time_key="time_point_processed",
    cell_type_key=label_key,
    output_space="both",
    save_dir=str(out),
    device=device,
)
(out / "manifest.json").write_text(json.dumps({
    "analysis": "prediction",
    "source_time": source_time,
    "target_time": target_time,
    "metadata": pred.get("metadata", {}),
    "warnings": warnings,
}, indent=2), encoding="utf-8")
```

## Failure Modes

- using target snapshots as predictions
- mixing raw and processed time axes
- silently discarding rollout weights
- claiming gene-level prediction from latent-only output
