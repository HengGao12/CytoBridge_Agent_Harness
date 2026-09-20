---
name: downstream-state-structure
description: Stable-state, endpoint, attractor, multimodality, and branching-structure downstream analysis.
---

# Downstream State Structure

Use this skill before writing a global dynamical narrative when the system may
be branched, multimodal, cyclic, or attractor-like.

## Analysis Logic

State-structure analysis checks whether the generated dynamics support a simple
story or require branching, multimodality, cyclic behavior, or multiple
attractors. Use observed state organization together with model-generated
rollouts. This step protects later fate, growth, and driver claims from being
collapsed into an invalid single mean trajectory.

If endpoint labels are weak or missing, discover terminal structure first and
then attach labels carefully.

State-structure analysis is often the first layer of a deeper workflow. Use it
to decide whether later fate, transition, growth, or driver analyses should be
single-path, branch-specific, cyclic, or multimodal.

## Read First

- `cytobridge_agent/skills/workflow/downstream-analysis/SKILL.md`
- `CytoBridge-main/docs/runtime/downstream/api-reference.md`
- `CytoBridge-main/docs/runtime/downstream/semantics-and-evidence.md`

## Core Rules

- Check structure before claiming a single mean trajectory.
- Use observed state organization and model rollout organization together.
- Separate terminal-state discovery from known label annotation.
- Report uncertainty when endpoints are weakly separated or label metadata is
  incomplete.

## Recommended Outputs

- state/cluster/endpoint summary
- transition or basin table
- structure figure(s)
- manifest documenting basis, labels, and limitations

## Minimal Template

```python
from pathlib import Path
import json
import numpy as np
import pandas as pd

out = Path(output_dir) / "downstream" / "state_structure"
out.mkdir(parents=True, exist_ok=True)
warnings = []

# Prefer rollout endpoints from a locked evaluation or trajectory artifact.
# endpoint_states shape: (n_trajectories, latent_dim) or (n_trajectories, n_genes).
endpoint_states = np.asarray(endpoint_states)
summary = {
    "analysis": "state_structure",
    "endpoint_shape": list(endpoint_states.shape),
    "feature_space": feature_space,
    "label_key": label_key,
    "warnings": warnings,
}
pd.DataFrame(endpoint_states).to_csv(out / "endpoint_states.csv", index=False)
(out / "manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
```
