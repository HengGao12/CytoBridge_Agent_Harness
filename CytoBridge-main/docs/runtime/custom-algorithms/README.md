---
title: "Custom algorithm workflow map"
summary: "Entry map for custom algorithm workspace and runtime integration documents."
read_when:
  - "Implementing or reviewing a custom algorithm workspace"
---
# Custom Algorithm Workflow Map

Use this directory for custom training algorithm workspace and runtime integration details.

# Custom Training Workflow

This document defines the supported workflow for letting an agent or developer
plug custom training logic into `CytoBridge-main` without modifying package
source files or builtin YAML configs.

## 1. Goal

Use this workflow when:

- builtin configs such as `balanced_ot_cfm`, `sf2m`, `vgfm`, `wfrfm`, `ruot`, or `crufm`
  are close, but not enough
- you want to replace the builtin `DynamicalModel` with a custom model
- you want custom stage execution for `flow_matching` or `neural_ode`
- you want custom evaluation-time prediction while keeping builtin `W1/TMV` definitions
- you want to override flow-matching coupling / path / mass logic
- you want to append algorithm-specific evaluation metrics while keeping builtin
  `W1` and `TMV` unchanged
- you want the training algorithm to remain editable Python
- you want every training run to be reproducible from a frozen snapshot

Do **not** use this workflow just to tune epochs, regularization, batch size,
or similar scalar settings. For that, keep using builtin configs plus
config overrides.

## 1A. Custom Model Training Contract

The builtin trainer always knows how to train these standard component modules:

- `velocity_net` when `train_strategy` contains `v`
- `growth_net` when `train_strategy` contains `g`
- `score_net` when `train_strategy` contains `s`
- `interaction_net` when `train_strategy` contains `i`

If a custom `model_builder(...)` adds extra trainable heads, do not hide them
inside unrelated modules just to get optimizer coverage. Use one of the explicit
contracts:

```python
class MyModel(torch.nn.Module):
    def __init__(self):
        ...
        self.velocity_net = ...
        self.growth_net = ...
        self.birth_head = ...
        self.death_head = ...
        self.cytobridge_component_modules = {
            "growth": ["birth_head", "death_head"],
        }
```

or set a stage config field:

```yaml
trainable_modules:
  - birth_head
  - death_head
```

Advanced models can implement:

```python
def cytobridge_trainable_parameters(self, *, stage_params, train_flags):
    return {"modules": [self.birth_head, self.death_head]}
```

Use `train_all_parameters: true` only for simple models where every parameter is
intended to train in that stage. This interface prevents the old workaround of
registering source/sink/birth heads under `growth_net` or `interaction_net`
only so `train_strategy` can see them.

## 2. Filesystem Layout

Editable algorithms live outside the package, under:

```text
~/.cellcompass/training_algorithms/
  <algorithm_id>/
    manifest.yaml
    algorithm.py
    README.md
```

Per-run immutable snapshots live under:

```text
<output_dir>/training_runs/<run_id>/
  run_manifest.json
  resolved_config.yaml
  logs/
    training.log
    planner_context.json
  algorithm_snapshot/
  artifacts/
    trained_model.h5ad
    metrics.json
    checkpoints/
```
