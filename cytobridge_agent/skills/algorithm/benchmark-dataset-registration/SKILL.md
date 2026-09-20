---
name: benchmark-dataset-registration
description: Register prepared datasets into the algorithm benchmark catalog without hidden preprocessing or metric pollution.
---

# Benchmark Dataset Registration

Use this skill only when the user asks to add, repair, or inspect an algorithm
benchmark dataset.

Do not create or register a new dataset as the first response to weak campaign
metrics. First inspect existing benchmark datasets and decide whether they can
test the algorithm's stated assumptions.

For agent-designed Stage 2 claim-validation simulations, prefer
`generate_and_register_stage2_simulation_dataset(...)` for DynBench/BoolODE
configs. If the data was produced by custom code, use
`register_stage2_simulation_dataset(...)` over plain
`register_algorithm_benchmark_dataset(...)`. These tools record the generated
data, the simulation generator fingerprint, and the frozen
`simulation_version` that Stage 2 baselines must match.

Agent-designed simulations are allowed, but they are exceptional. Use them only
when the current benchmark catalog cannot fairly test the proposal claim. If an
existing dataset can test the claim, reuse the existing dataset and its
builtin/reference baselines.

## Hard Boundary

`register_algorithm_benchmark_dataset(...)` does not preprocess.

It only:

- validates a prepared `.h5ad`
- copies it into `~/.cellcompass/algorithm_benchmarks/datasets/<dataset_id>/`
- optionally copies the exact preprocessing script passed as
  `preprocessing_script_path` into
  `datasets/<dataset_id>/scripts/preprocessing/`
- writes `dataset.json`, `README.md`, `leaderboard.json`, and catalog entries

If the file lacks required fields, the tool returns `invalid_contract` and no
dataset card is created.

## Required Data Contract

The source `.h5ad` must already contain:

- `adata.obs["time_point_processed"]`
- `adata.obsm["X_latent"]`

The time axis must be biologically meaningful. Do not turn arbitrary batch,
sample, or condition labels into time just to satisfy the contract.

## Before Registration

Read:

- `~/.cellcompass/skills/workflow/preprocessing-execution/SKILL.md`

Then inspect the candidate `.h5ad`:

- shape
- `obs` columns
- `obsm` keys
- time-point counts
- whether `X_latent` is appropriate for the intended benchmark

If preprocessing is needed, create a new prepared `.h5ad` first. Do not expect
the registration tool to infer fields from `samples`, `X`, PCA, or UMAP.

The prepared dataset must be reproducible. Save the final preprocessing routine
as a script before registration, then pass that file through
`preprocessing_script_path`. The registration tool does not run the script; it
copies it into the benchmark dataset directory and records the original path,
copied path, and SHA256 in `dataset.json`, the dataset README, and the registry
summary. If the dataset was produced by paper-intake materialization code, pass
the materialization/preprocessing script path that actually generated the final
registered `.h5ad`.

Before registering a Stage 2 simulation, also record:

- which existing benchmark datasets were considered
- why they cannot test the claim or algorithm assumption
- what specific proposal mechanism the simulation isolates
- what observable claim metric will be computed from rollout artifacts
- why that claim metric is meaningful for the stated problem, and which
  claim-absent, shuffled, collapsed, or ablated control should make it fail
- which builtin/reference baselines must be rerun on the same
  `simulation_version`
- if using BoolODE/DynBench, the scenario config path or inline config passed
  to `generate_and_register_stage2_simulation_dataset(...)`

## Registration Flow

1. Validate the prepared `.h5ad` has the required fields.
2. Call `register_algorithm_benchmark_dataset(...)` with the prepared path and,
   for newly prepared real datasets, `preprocessing_script_path`.
3. Read the returned `dataset.json` path or call
   `get_algorithm_benchmark_dataset(dataset_id=...)`.
4. Confirm `contract_status == "ready"`.
5. Confirm `paths.preprocessing_script_path` is populated for newly prepared
   real datasets and that the copied script is the exact routine used to produce
   the registered `.h5ad`.
6. If builtin/reference baselines are available, store them with
   `record_algorithm_benchmark_baseline(...)`.
   This archives the trained model, metrics, resolved config, and logs under the
   dataset directory. Use `list_algorithm_benchmark_baselines(...)` to inspect
   those local paths later.

## Metric Policy

The global benchmark catalog stores only comparable global metrics:

- W1
- TMV
- runtime
- memory

Agent-proposed claim metrics are campaign-local. Do not write custom claim
metrics into the benchmark leaderboard, because different algorithms may define
different claim metrics.

## Config Policy

Dataset-specific configs are allowed, but keep the scope clear:

- builtin/reference configs may live under `builtin_configs/`
- custom algorithm per-dataset configs may live under `algorithm_configs/`
- campaign archives must capture the exact config used for a trial

Do not hide algorithm-specific performance assumptions inside the shared
benchmark data card.

## If Registration Fails

If the tool returns `invalid_contract`:

- stop registration
- read the missing fields in `missing_fields`
- read or re-read `preprocessing-execution`
- create a prepared `.h5ad`
- retry registration with the prepared path

Do not bypass this by editing `dataset.json` manually.
