---
title: "Data and evaluation contracts"
summary: "Canonical data fields, evaluation metrics, and downstream artifact contracts."
read_when:
  - "Checking input data requirements"
  - "Understanding W1/TMV and evaluation semantics"
---
# Data And Evaluation Contracts

Package map, AnnData contract, return payload convention, and training/evaluation semantics.

## 1. Package Map

- Compute APIs:
  - `CytoBridge/pp/preprocess.py`
  - `CytoBridge/tl/fit.py`
  - `CytoBridge/tl/downstream/*.py`
- Plot APIs:
  - `CytoBridge/pl/downstream/*.py`
- Compatibility wrappers:
  - `CytoBridge/tl/analysis.py`
  - `CytoBridge/pl/plot.py`

## 2. Strict AnnData Contract

Downstream code assumes fixed keys.

- Required baseline:
  - `adata.obs["time_point_processed"]` (numeric time)
  - `adata.obsm["X_latent"]` (`n_obs x latent_dim`)
- Optional but commonly required by specific APIs:
  - `adata.obsm["velocity_latent"]`
  - `adata.obsm["growth_rate"]` or `adata.obs["growth_rate"]`
    - interpreted as a growth rate field `g(t, x) = d/dt log w`
    - not as an absolute mass / cell-count value
  - `adata.varm["PCs"]` (PCA loadings for latent->gene projection)

Contract constants are defined in:
- `CytoBridge/tl/downstream/contracts.py`

## 3. Return Payload Convention

Most `tl.downstream` and `pl.downstream` APIs return:

```python
{
  "space_used": "latent" | "gene",
  "projection_backend": "none" | "pca",
  "artifacts": Dict[str, str],  # artifact name -> file path
  "warnings": List[str],
  # optional API-specific fields...
}
```

Notes:
- `space_used="gene"` currently only appears when PCA projection is available.
- `projection_backend="pca"` means latent output was projected with `adata.varm["PCs"]`.

## 4. Training Recoverability And Evaluation Semantics

Before changing coupling, path, or mass behavior, keep the default mathematical
story straight.

### 4.1 What the default coupling means

For each adjacent interval `(t_k, t_{k+1})`, the default flow-matching UOT
solver constructs a plan `pi(i, j)` between source cells `x_i^k` and target
cells `x_j^{k+1}`.

Under the builtin default:

- source reference marginal: `a_i = 1`
- target reference marginal: `b_j = 1`

So:

- row sum `r_i = sum_j pi(i, j)` is already the correct terminal mass ratio for
  a source particle whose local starting mass is normalized to `1`
- column sum `c_j = sum_i pi(i, j)` recovers the intended late target marginal

Do not reinterpret this as a normalized `1 / n` convention unless you have
actually changed the OT marginal definition.

### 4.2 What pair sampling means

`sample_from_ot_plan(...)` samples:

- source index `i` proportional to row sum `r_i`
- target index `j` conditionally proportional to `pi(i, j) / r_i`

So the sampled pair distribution is the normalized transport plan itself.

### 4.3 What velocity and growth each contribute

In the default split:

- `velocity` learns where sampled source particles move
- `MassStrategy` defines what terminal mass each source particle should have by
  the end of the interval
- the conditional path defines the local mass dynamics used to reach that
  terminal mass target
- `growth_rate` means `g(t, x) = d/dt log w`

The conditional path defines two different outputs:

- a growth-rate regression target
- a training loss weight

These are not the same quantity.

- the growth target is a rate target and should not change just because the
  particle started with a different absolute mass before local normalization
- the training loss weight is an optimization-time reweighting term
- that training loss weight may depend on the sampling convention and transport
  semantics
- it should not be interpreted as the current path mass `w(t)`

So `velocity` and `growth` are not scientifically independent in evaluation.
They jointly determine the predicted future measure.

### 4.4 Why the default exact-fit story is coherent

Under exact fit:

- the learned path/velocity recovers the coupling-implied endpoint geometry
- the learned growth dynamics recover the intended terminal per-particle mass
- endpoint aggregation therefore recovers the late marginal interval by
  interval

This is the intended recoverability argument behind the current builtin
flow-matching semantics.

### 4.5 What current evaluation actually consumes

Current evaluation starts from the earliest observed time point, simulates
forward, and compares predicted future measures against observed future
measures.

- `W1` compares observed future cells against predicted future cells in latent
  space
- observed side uses uniform weights
- predicted side uses normalized predicted particle weights
- `TMV` compares total predicted relative mass against observed relative cell
  count ratio

So:

- bad growth semantics can worsen `W1`, not only `TMV`
- good `TMV` does not guarantee correct weighted distribution shape
- changing `v` alone can still change weighted distribution fit
- do not confuse predicted particle weights at evaluation time with the
  path-defined training loss weights used during FM regression
- custom algorithms may append extra metrics under `custom_metrics`, but may
  not replace builtin `W1` or `TMV`
- additive custom metrics should preferably reuse the same forward-simulated
  trajectory artifacts already produced for builtin evaluation, rather than
  re-running inference
- evaluation-only thresholds or prior parameters should be passed through the
  custom algorithm's `evaluation_metrics_params`

## 5. Unified Claim Metric Evaluator Contract

Campaign Stage 2 may use an algorithm-specific claim metric, but its evaluator
must consume the same package-level evaluation context as builtin `W1/TMV`.
This keeps claim metrics comparable across the candidate algorithm and builtin
baselines.

Declare the metric in `claim_metric_spec`:

```json
{
  "name": "claim_metric",
  "direction": "greater",
  "evaluator_path": "metrics/claim_metric.py",
  "evaluator_function": "evaluate_claim_metric",
  "version": "v1"
}
```

The evaluator file must define:

```python
def evaluate_claim_metric(context):
    ...
    return {"claim_metric": 0.73}
```

The `context` is `EvaluationMetricsContext`. It includes:

- `adata`: the trained AnnData object being evaluated
- `training_data`: the resolved training data bundle
- `model`: the trained model
- `config`: resolved training config
- `time_points`: ordered evaluation time points
- `builtin_metrics`: builtin W1/TMV results already computed so far
- `metric_params`: evaluator-only params
- `simulated_points_by_time`: model-simulated particles by time point
- `simulated_weights_by_time`: model-simulated particle weights by time point
- `timepoint_results`: per-timepoint builtin evaluation records
- `device` and `metadata`

Rules:

- Use only forward-simulated trajectory outputs and allowed t0/context inputs.
- Do not read future observed labels to repair predictions or post-hoc correct
  particle mass/position.
- Do not replace builtin `W1/TMV`; return only additive custom metrics.
- If a benchmark baseline already has `trained_model_path`, the campaign can
  load that saved baseline model and compute the new claim metric posthoc
  without retraining.
- If the evaluator or simulation generator changes, bump its version and rerun
  or posthoc-refresh the affected baseline metrics.
