---
name: downstream-lineage-transition
description: Source-target lineage and transition downstream analysis from model rollouts and lineage metadata.
---

# Downstream Lineage Transition

Use this skill when the question asks which source populations transition into
which target states, lineages, or fates.

## Analysis Logic

Lineage/transition analysis combines generated model dynamics with an external
readout contract. First define source cells and valid target states from
metadata, lineage labels, clone labels, or a classifier. Then roll source cells
through the trained model and summarize where the generated states go over
time. If true lineage evidence exists, compare rollout-derived transitions to
that evidence; otherwise label the result as model-predicted transition
structure, not lineage truth.

For branching systems, keep the full distribution over targets. A top-1 fate is
not enough evidence for lineage-concordant dynamics.

This skill is commonly paired with trajectory/fate analysis. The transition
matrix should be derived from rollout states plus a valid lineage/fate readout,
then optionally linked to drivers or perturbations.

## Read First

- `cytobridge_agent/skills/workflow/downstream-analysis/SKILL.md`
- `CytoBridge-main/docs/runtime/downstream/semantics-and-evidence.md`
- `CytoBridge-main/docs/runtime/downstream/api-reference.md`

## Core Rules

- Define source and target populations from metadata, classifiers, or explicit
  user constraints.
- Rollout-derived transitions and observed lineage evidence are different
  evidence types; keep both visible.
- Do not infer lineage truth from generic cell-type labels unless the dataset
  contract says they encode lineage/fate.
- For branching systems, report distributions rather than only top-1 fate.

## Recommended Outputs

- source-target transition matrix
- per-source fate distribution
- lineage/fate concordance diagnostics when ground truth exists
- figure(s)
- manifest with source/target definitions and warnings

## Minimal Template

```python
from pathlib import Path
import json
import numpy as np
import pandas as pd

out = Path(output_dir) / "downstream" / "lineage_transition"
out.mkdir(parents=True, exist_ok=True)
warnings = []

# `fate_prob` should be computed from generated trajectory states using a
# frozen classifier or explicit target-label contract. Shape:
# (n_source_cells, n_target_fates).
fate_prob = np.asarray(fate_prob)
target_fates = list(target_fates)
source_groups = np.asarray(source_groups)

rows = []
for group in sorted(set(source_groups)):
    mask = source_groups == group
    rows.append(pd.Series(fate_prob[mask].mean(axis=0), index=target_fates, name=group))
transition = pd.DataFrame(rows)
transition.to_csv(out / "rollout_transition_matrix.csv")

(out / "manifest.json").write_text(json.dumps({
    "analysis": "lineage_transition",
    "source_groups": sorted(map(str, set(source_groups))),
    "target_fates": target_fates,
    "fate_prob_shape": list(fate_prob.shape),
    "evidence": "rollout-derived transition structure",
    "warnings": warnings,
}, indent=2), encoding="utf-8")
```
