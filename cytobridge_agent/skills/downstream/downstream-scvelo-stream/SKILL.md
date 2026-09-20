---
name: downstream-scvelo-stream
description: scVelo-style stream plotting from CytoBridge velocity diagnostics.
---

# Downstream scVelo Stream

Use this skill only for explicit stream/flow visualization requests.

## Core Rule

Stream plots are local flow diagnostics. They are not continuous trajectory
evidence by themselves.

## Analysis Logic

Use stream plots to visualize the local velocity field learned by the neural
model on a 2D basis. This answers "what is the local direction of motion around
observed cells?" It does not answer "what path does this cell follow over
continuous time?" unless paired with a model rollout trajectory. If the report
needs continuous paths or fate changes, use the trajectory/fate skill first and
use stream plots as supporting diagnostics.

Use this skill as a visualization companion, not as the core evidence path,
unless the user explicitly asks only for local flow-field plots.

## Read First

- `CytoBridge-main/docs/runtime/downstream/api-reference.md`
- source when details matter:
  `CytoBridge-main/CytoBridge/tl/downstream/velocity.py` and
  `CytoBridge-main/CytoBridge/pl/downstream/velocity_plot.py`

## Recommended APIs

- `tl.downstream.compute_velocity_bundle(...)`
- `tl.downstream.build_velocity_graph_bundle(...)`
- `pl.downstream.plot_velocity_stream_bundle(...)`

## Minimal Template

```python
from pathlib import Path
import json
import scanpy as sc
from CytoBridge.tl.downstream import compute_velocity_bundle, build_velocity_graph_bundle
from CytoBridge.pl.downstream import plot_velocity_stream_bundle

adata = sc.read_h5ad(input_h5ad)
out = Path(output_dir) / "downstream" / "velocity_stream"
out.mkdir(parents=True, exist_ok=True)

vel = compute_velocity_bundle(adata=adata, output_dir=str(out))
graph = build_velocity_graph_bundle(adata=adata, output_dir=str(out), dim_reduction="umap")
plot = plot_velocity_stream_bundle(
    adata=adata,
    output_dir=str(out),
    dim_reduction="umap",
    color_key=label_key,
)
(out / "manifest.json").write_text(json.dumps({
    "analysis": "velocity_stream",
    "evidence_type": "local flow diagnostic",
    "artifacts": {
        "velocity": vel.get("artifacts", {}),
        "graph": graph.get("artifacts", {}),
        "plot": plot.get("artifacts", {}),
    },
    "warnings": (
        vel.get("warnings", [])
        + graph.get("warnings", [])
        + plot.get("warnings", [])
    ),
}, indent=2, default=str), encoding="utf-8")
```

## Required Outputs

- stream figure
- velocity computation/graph summary
- manifest with basis, velocity key, model path, and warnings
