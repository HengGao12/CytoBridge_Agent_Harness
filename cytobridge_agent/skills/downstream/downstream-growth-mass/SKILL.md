---
name: downstream-growth-mass
description: Growth-rate and population-mass downstream analysis for CytoBridge dynamics.
---

# Downstream Growth And Mass

Use this skill when the biological question involves proliferation, depletion,
unbalanced mass, cell-count changes, growth rates, or TMV-like behavior.

## Analysis Logic

Growth and mass analysis is only model-native when the trained model includes a
growth or unbalanced-mass component. The growth network represents local
`d/dt log w`; generated trajectory weights then summarize how effective mass
changes along the rollout. Analyze both local growth fields and integrated
trajectory weights when possible.

Balanced velocity-only models do not learn mechanistic growth. For those
models, empirical cell counts by time can be reported as data context or TMV
diagnostics, but not as model-predicted proliferation/death.

Growth analysis often becomes stronger when combined with trajectory/fate or
driver-gene analysis: first locate where mass changes along generated dynamics,
then ask which fates or model components explain that change.

## Read First

- `cytobridge_agent/skills/workflow/downstream-analysis/SKILL.md`
- `CytoBridge-main/docs/runtime/downstream/api-reference.md`
- `CytoBridge-main/docs/runtime/downstream/model-semantics.md`
- source when details matter:
  `CytoBridge-main/CytoBridge/tl/downstream/growth.py`

## Core Rules

- Distinguish growth rate `d/dt log w` from absolute population mass.
- Cell counts by time are empirical observations, not automatically model
  growth.
- Model growth claims require a trained growth/mass mechanism or explicit
  growth output.
- Balanced models may still support empirical-count diagnostics, but not
  mechanistic mass recovery claims.

## Recommended APIs

- `tl.downstream.summarize_growth_bundle(...)`
- `tl.downstream.summarize_growth_drivers_bundle(...)`
- `pl.downstream.plot_growth_bundle(...)`

## Minimal Template

```python
from pathlib import Path
import json
import scanpy as sc
from CytoBridge.tl.downstream import summarize_growth_bundle

adata = sc.read_h5ad(input_h5ad)
out = Path(output_dir) / "downstream" / "growth_mass"
out.mkdir(parents=True, exist_ok=True)

res = summarize_growth_bundle(adata=adata, key="growth_rate", output_dir=str(out))
(out / "manifest.json").write_text(json.dumps({
    "analysis": "growth_mass",
    "api": "summarize_growth_bundle",
    "source_path": "CytoBridge-main/CytoBridge/tl/downstream/growth.py",
    "stats": res.get("stats", {}),
    "warnings": res.get("warnings", []),
}, indent=2), encoding="utf-8")
```

## Required Outputs

- growth/mass summary table
- figure(s)
- manifest describing growth source and interpretation limits
