---
name: downstream-driver-genes
description: Driver gene, growth-driver, velocity-driver, and GRN/Jacobian downstream analysis for CytoBridge models.
---

# Downstream Driver Genes

Use this skill when the question asks which genes, latent dimensions, or
regulatory interactions explain velocity, growth, fate, perturbation, or
transition behavior.

## Analysis Logic

Driver analysis should start from a model-derived dynamical target, not from a
generic differential-expression list. Valid targets include velocity magnitude
or direction, growth rate, rollout fate shift, perturbation response, transition
probability, or a neural-network Jacobian. Because CytoBridge is differentiable,
driver analysis can inspect model components and Jacobians; because the model is
trained in latent space, gene-level claims require a documented projection back
to measured genes, usually through PCA loadings or a package gene-space bundle.

If projection is unavailable or unstable, report latent drivers as latent
factors and avoid gene-level mechanism claims.

Driver analysis is usually a second-stage analysis. First define a dynamical
phenotype from another downstream path, such as branch shift, growth phase,
transition probability, perturbation response, or prediction error; then compute
drivers for that phenotype.

## Read First

- `cytobridge_agent/skills/workflow/downstream-analysis/SKILL.md`
- `CytoBridge-main/docs/runtime/downstream/api-reference.md`
- source when details matter:
  `CytoBridge-main/CytoBridge/tl/downstream/velocity.py`,
  `CytoBridge-main/CytoBridge/tl/downstream/growth.py`, and
  `CytoBridge-main/CytoBridge/tl/downstream/grn.py`

## Core Rules

- Define the biological target first: velocity, growth, fate, perturbation, or
  transition.
- Gene-level claims require measured gene space or documented PCA/projection.
- Latent drivers are valid only as latent factors unless mapped to genes.
- GRN/Jacobian orientation must be documented.
- `summarize_velocity_drivers_bundle(...)` and
  `summarize_growth_drivers_bundle(...)` both support
  `analysis_space="auto" | "gene" | "latent"`. Use `auto` by default and check
  the API reference or `inspect.signature(...)` before writing export scripts.

## Recommended APIs

- `tl.downstream.summarize_velocity_drivers_bundle(...)`
- `tl.downstream.summarize_growth_drivers_bundle(...)`
- `tl.downstream.summarize_velocity_jacobian_drivers_bundle(...)`
- `tl.downstream.summarize_growth_jacobian_drivers_bundle(...)`
- `tl.downstream.analyze_grn_bundle(...)`

## Minimal Template

Velocity drivers:

```python
from pathlib import Path
import json
import scanpy as sc
from CytoBridge.tl.downstream import summarize_velocity_drivers_bundle

adata = sc.read_h5ad(input_h5ad)
out = Path(output_dir) / "downstream" / "driver_genes"
out.mkdir(parents=True, exist_ok=True)

res = summarize_velocity_drivers_bundle(
    adata=adata,
    output_dir=str(out),
    analysis_space="auto",
    top_n=30,
)
(out / "manifest.json").write_text(json.dumps({
    "analysis": "driver_genes",
    "target": "velocity",
    "api": "summarize_velocity_drivers_bundle",
    "source_path": "CytoBridge-main/CytoBridge/tl/downstream/velocity.py",
    "space_used": res.get("space_used"),
    "projection_backend": res.get("projection_backend"),
    "artifacts": res.get("artifacts", {}),
    "warnings": res.get("warnings", []),
}, indent=2), encoding="utf-8")
```

Growth drivers:

```python
from pathlib import Path
import json
import scanpy as sc
from CytoBridge.tl.downstream import summarize_growth_drivers_bundle

adata = sc.read_h5ad(input_h5ad)
out = Path(output_dir) / "downstream" / "growth_drivers"
out.mkdir(parents=True, exist_ok=True)

res = summarize_growth_drivers_bundle(
    adata=adata,
    output_dir=str(out),
    analysis_space="auto",
    top_n=30,
    device="cuda",
)
(out / "manifest.json").write_text(json.dumps({
    "analysis": "growth_drivers",
    "target": "growth",
    "api": "summarize_growth_drivers_bundle",
    "source_path": "CytoBridge-main/CytoBridge/tl/downstream/growth.py",
    "space_used": res.get("space_used"),
    "projection_backend": res.get("projection_backend"),
    "artifacts": res.get("artifacts", {}),
    "warnings": res.get("warnings", []),
}, indent=2), encoding="utf-8")
```

## Required Outputs

- ranked driver table
- optional GRN/Jacobian matrix
- figure(s)
- manifest with target, feature space, projection, model path, and warnings
