---
title: "Custom algorithm workspace contract"
summary: "Filesystem layout, snapshot, manifest, and reproducibility contract for custom algorithms."
read_when:
  - "Editing custom algorithm workspace files"
  - "Checking snapshot or manifest rules"
---
# Custom Algorithm Workspace Contract

Manifest, entrypoint, context, AnnData ownership, and TrainingAlgorithmSpec contract.

## 3. `manifest.yaml`

Required structure:

```yaml
algorithm_id: custom_ruot_v1
api_version: 1
entrypoint: algorithm.py
entry_function: build_training_algorithm
description: >
  Custom velocity-growth flow-matching backend with modified coupling.
requirements: >
  Requires biologically meaningful time_point_processed, usable X_latent, and
  any algorithm-specific preprocessing assumptions should be listed here.
base_config: ./config.yaml
tags:
  - flow-matching
  - unbalanced
  - custom
author: agent
```

Rules:

- `algorithm_id` must match the directory name
- `api_version` must be `1`
- `entrypoint` must be `algorithm.py`
- `entry_function` must be `build_training_algorithm`
- `requirements` should explicitly state data assumptions and preprocessing prerequisites
- for planner-generated custom workspaces, keep `base_config: ./config.yaml`
- the workspace `config.yaml` is seeded from builtin `vgfm`
- advanced/manual setups may still use another builtin config name, config path, or dict

## 4. `algorithm.py` Entry Contract

Your algorithm file must export:

```python
def build_training_algorithm(context: TrainingAlgorithmContext) -> TrainingAlgorithmSpec:
    ...
```

Import from the package:

```python
from CytoBridge.tl.training_algorithm import (
    EvaluationMetricsContext,
    TrainingAlgorithmContext,
    TrainingAlgorithmSpec,
)
```

## 5. `TrainingAlgorithmContext`

Fields:

```python
TrainingAlgorithmContext(
    algorithm_id: str,
    input_adata_path: str,
    output_dir: str,
    stage: str,                  # "pilot" | "final"
    base_config_name: str | None,
    resolved_base_config: dict,
    metadata: dict[str, Any],
)
```

Purpose:

- `algorithm_id`: stable logical identifier
- `input_adata_path`: source AnnData path being trained on
- `output_dir`: root output dir for this user task
- `stage`: whether this invocation is pilot or final
- `base_config_name`: builtin config name when known
- `resolved_base_config`: already-loaded base config dict before your override
- `metadata`: extra caller-side context, for example planner metadata

`stage` is only a lightweight run-context tag:

- `pilot` and `final` share the same training code path
- the runtime does **not** automatically downsample pilot data
- if you want a smaller pilot run, prepare a smaller AnnData explicitly before training

Package-level runtime contracts remain fixed:

- `adata.obs["time_point_processed"]` is the canonical processed time axis
- `adata.obsm["X_latent"]` is the canonical transcriptomic training representation
- extra modalities may be read and used, but should augment rather than replace `X_latent`

## 5A. AnnData Ownership Model

Keep the data flow mentally separated into three layers:

1. dataset selection
2. raw AnnData handling
3. training-time bundle consumption

### Dataset selection

`run_training(...)` chooses the dataset in this order:

1. explicit `adata_path`
2. current workflow dataset path (usually `preprocessed_path`)
3. fallback `input_path`

This decides **which** `.h5ad` file is loaded for the run.

### Raw AnnData handling

Raw AnnData belongs to:

- `TrainingDataBuilderContext.adata`

This is the right place to:

- read `obs`
- read `obsm`
- read `layers`
- attach extra aligned modalities
- construct a `TrainingDataBundle`

This is **not** the place to redefine the canonical contracts:

- `obs["time_point_processed"]`
- `obsm["X_latent"]`

### Training-time bundle consumption

Once training starts, the main runtime consumes:

- `TrainingDataBundle`

At that point:

- backend builders receive `build_context.training_data`
- they do **not** receive raw adata as their primary input

This separation is intentional:

- raw data assembly happens in `training_data_builder(...)`
- model construction happens in `model_builder(...)` when needed
- backend construction happens in `flow_matching_backend_builder(...)`
- custom stage execution happens in `stage_runner(...)` when builtin stage
  logic is insufficient; if only custom batches/losses are needed, prefer
  `run_custom_stage_loop(...)` so optimizer/checkpoint/device/timeout behavior
  remains package-owned
- training loss augmentation happens in `flow_matching_loss_hook(...)`
- evaluation-time prediction changes happen in `simulation_hook(...)`

## 6. `TrainingAlgorithmSpec`

Your entrypoint must return:

```python
TrainingAlgorithmSpec(
    algorithm_id: str,
    base_config: str | dict,
    config_overrides: dict[str, Any] = {},
    training_data_builder: Callable[[TrainingDataBuilderContext], TrainingDataBundle] | None = None,
    flow_matching_backend_builder: Callable[[FlowMatchingBuildContext], FlowMatchingBackend] | None = None,
    flow_matching_loss_hook: Callable[[FlowMatchingLossContext], FlowMatchingLossResult | torch.Tensor | None] | None = None,
    evaluation_metrics_hook: Callable[[EvaluationMetricsContext], dict[str, Any] | None] | None = None,
    model_builder: Callable[[ModelBuildContext], torch.nn.Module] | None = None,
    serialize_model: Callable[[ModelSerializeContext], dict[str, Any]] | None = None,
    deserialize_model: Callable[[ModelLoadContext], torch.nn.Module] | None = None,
    stage_runner: Callable[[StageRunnerContext], StageRunnerResult | dict[str, Any] | None] | None = None,
    inference_context_builder: Callable[[InferenceContextBuilderContext], InferenceContext | dict[str, Any] | None] | None = None,
    simulation_hook: Callable[[SimulationContext], SimulationResult | dict[str, Any]] | None = None,
    evaluation_metrics_params: dict[str, Any] = {},
    notes: str | None = None,
)
```

Rules:

- `algorithm_id` must equal the manifest `algorithm_id`
- `model_builder` replaces the builtin `DynamicalModel` when the algorithm needs another model family
- `serialize_model` / `deserialize_model` are only needed when config + state_dict are insufficient for reload
- `stage_runner` is the stage override for custom FM, custom `neural_ode`,
  higher-order, or nonstandard optimization semantics. Prefer
  `CytoBridge.tl.custom_stage_loop.run_custom_stage_loop(...)` inside the
  runner when the approved method needs custom batches/losses but not custom
  optimizer/checkpoint semantics.
- `inference_context_builder` prepares a flexible inference payload for custom
  evaluation-time prediction
- `simulation_hook` replaces the builtin evaluation-time predictor, but builtin `W1/TMV` definitions remain fixed
- `training_data_builder` may add aligned extra modalities, but may not replace
  `time_point_processed` or `X_latent`
- `flow_matching_backend_builder` is the standard backend composition hook
- `flow_matching_loss_hook` may add `extra_loss` or explicitly replace one
  active builtin component loss with `replace_velocity_loss`,
  `replace_growth_loss`, or `replace_score_loss`
- `evaluation_metrics_hook` is additive only; it may append custom metrics but
  may not replace builtin `W1` or `TMV`
- `evaluation_metrics_params` is the explicit place to pass custom metric
  thresholds, priors, or other evaluation-only parameters

Recommended policy:

- decide custom metrics at proposal time
- keep builtin `W1` and `TMV` fixed
- if custom inference is required, build a small `InferenceContext` first and
  document payload provenance
- use `evaluation_metrics_hook(...)` only for additive metrics that validate the
  proposal's stated scientific objective
- for flow-matching coupling/path/mass defaults, read
  `docs/runtime/flow-matching/README.md` before editing `algorithm.py`
- `use_mini_batch` or `chunk_size` fields are not enough to prove real-data
  scalability; document the actual dense/streaming/sparse memory path in
  `IMPLEMENTATION_MAP.md`

Inference context shape:

```python
def inference_context_builder(ctx: InferenceContextBuilderContext) -> InferenceContext:
    return InferenceContext(
        payload={"seed": ctx.initial_data, "time_points": ctx.time_points},
        visibility={"seed": "t0_inference", "time_points": "exogenous_inference"},
        provenance={"time_filter": "t0 only", "future_rows_used": False},
        notes="Prediction inputs are prepared from t=0 cells and known time grid.",
    )
```

The payload keys are not fixed. The runtime and reviewer care about source
provenance, not the exact field names.

Minimal hook shape:

```python
def evaluation_metrics_hook(eval_context: EvaluationMetricsContext) -> dict[str, Any] | None:
    return {
        "lineage_prior_conformity": 0.91,
    }
```

Important:

- the hook receives the same forward-simulated trajectory artifacts that builtin
  `W1/TMV` already use
- reuse `eval_context.simulated_points_by_time`,
  `eval_context.simulated_weights_by_time`, or
  `eval_context.timepoint_results`
- use `eval_context.metric_params` for extra evaluation-only parameters
- do not re-run inference unless there is a very strong reason

The runtime will store this under:

```python
adata.uns["evaluation_metrics"]["custom_metrics"]
```
- `config_overrides` uses dot-notation keys understood by the agent training stack
- `base_config` is the config that will be resolved before overrides are applied
