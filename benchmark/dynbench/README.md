# DynBench v2: Dynamical Modeling Benchmark

> Benchmarking AI agents on dynamical modeling tasks using BoolODE-style synthetic ground truth.

## Overview

DynBench v2 evaluates an agent's ability to reconstruct **dynamical processes** from snapshot single-cell data. It tests 6 dynamics-aware metrics covering velocity, growth, cell fate, perturbation response, and gene regulatory network inference.

### Key Design Principles

1. **Independent Ground Truth**: BoolODE-style Hill-function ODEs + SDE noise + Poisson growth
2. **Gold Standard Classifier**: Oracle MLP trained on ALL time points for consistent fate classification
3. **Anti-leakage**: TASK.md uses placeholder gene names (Gene_X/Y/Z) and random values in all examples
4. **Anti-heuristic**: Growth drivers are hidden among confounding genes with similar time correlations

## Scenarios

| Scenario | Genes | Fates | Growth Drivers | Confounding |
|----------|-------|-------|----------------|-------------|
| **S1**: Toggle Switch | 3 | 2 (B/C) | Gene_3 | — (too few genes) |
| **S3**: Complex Tree | 10 | 4 (B/C/D/E) | Gene_5 + Gene_6 | Gene_9, Gene_10 |

### S1: Toggle Switch (3 genes, 2 fates)
- Gene_1 ↔ Gene_2: mutual inhibition → bifurcation
- Gene_3: downstream reporter, drives growth
- 6 time bins, holdout bins 2,3
- 6 GRN edges out of 9 possible → baseline AUPRC = 0.667

### S3: Complex Tree (10 genes, 4 fates)
- Two independent toggle switches: Gene_1↔Gene_2, Gene_3↔Gene_4
- 4 reporters: Gene_5 (←G1), Gene_6 (←G2), Gene_7 (←G3), Gene_8 (←G4)
- 2 late markers (confounding): Gene_9 (←G5,G6), Gene_10 (←G7,G8)
- **Growth**: driven by mean(Gene_5, Gene_6) — NOT Gene_9/Gene_10
- **Anti-heuristic**: Gene_9 time-corr=0.290 > Gene_5 (0.282) > Gene_6 (0.272). Simple argmax(corr) fails
- 16 GRN edges out of 100 possible → baseline AUPRC = 0.160
- 8 time bins, holdout bins 2,3

## 6 Evaluation Metrics (equal weight: 1/6 each)

| # | Metric | Score definition | What it measures |
|---|--------|-----------------|-----------------|
| M1 | **Velocity** | Per-cluster cosine similarity (10D) | Velocity field direction accuracy |
| M2 | **Growth drivers** | Growth-driver AUPRC | Growth/proliferation driver identification |
| M3 | **Distribution** | Joint-distribution W1 score (normalized) | Holdout distribution prediction quality |
| M4 | **Fate** | Top-1 terminal-state agreement | Progenitor fate assignment accuracy |
| M5 | **Perturbation** | 50% effect-detection F1 + 50% sign accuracy | Gene knockdown fate shift prediction |
| M6 | **GRN** | Edge AUPRC | Gene regulatory network inference |

### Metric Details

- **Validation**: Output validation is strict and per-metric. The evaluator does not repair malformed outputs. If a required artifact is missing or schema-invalid, only the dependent metric receives score 0 and records the validation error; independent metrics continue to score normally.
- **M1**: Groups cells by (time_bin, fate), computes cosine similarity of mean velocity vectors in gene space. Reports per-group breakdown.
- **M2**: AUPRC of growth driver gene identification. `growth_rates.csv` is still validated as a required output, but per-cell growth-rate values are model-specific internal quantities and are not used in the official score.
- **M3**: Wasserstein-1 / EMD on the joint predicted-vs-GT holdout distribution, normalized by self/random baselines. Composition JSD is reported as a diagnostic. By design, GT holdout cells are treated as a **uniform empirical distribution** unless the task metadata explicitly overrides this; agent outputs may optionally include a `weight` column when they represent weighted particles rather than expanded samples.
- **M4**: Top-1 terminal-state agreement for each common cell. Legacy Pearson-over-probabilities is still reported as a diagnostic but is no longer the official score.
- **M5**: Treat each `(gene, fate)` pair as a 3-state perturbation target: no effect vs positive effect vs negative effect using a small delta threshold. The official score is `0.5 * detection_F1 + 0.5 * sign_accuracy_on_GT_active`.
- **M6**: Average Precision (AUPRC) over all directed gene pairs. AUROC and direction accuracy (activate/inhibit) reported as reference.

## Results

### Historical Reference Runs (`gpt-5.4`, `skills-on`, `cpu`, `seed=42`)

These records are retained as historical smoke-test references, not as the
current leaderboard. Re-run the batch runner after evaluator or task-contract
changes before using numbers in reports.

#### S3 (10 genes, 4 fates) — Canonical Holdout Benchmark

Result file:
- `benchmark/dynbench/results/S3/agent_skills-on_seed42/eval_results.json`

| Scenario | M1 | M2 | M3 | M4 | M5 | M6 | **Total** | Runtime |
|----------|-----|-----|-----|-----|-----|-----|-----------|---------|
| **S3** | 0.906 | 0.787 | 0.708 | 0.688 | 0.808 | 0.602 | **0.750** | 1028s |

Diagnostics:
- `M2 driver_auprc = 0.5833` under the previous reference run. Current M2 uses driver AUPRC only.
- `M3 w1_emd = 0.2973`, `w1_self = 0.1723`, `w1_self_std = 0.0024`, `w1_random = 0.6000`, `w1_self_method = bootstrap_same_size`
- `M6 edge_auprc = 0.6016`, `direction_accuracy = 1.0`

#### S3_no_holdout_sanity — Pipeline Sanity Split

Result file:
- `benchmark/dynbench/results/S3_no_holdout_sanity/agent_skills-on_seed42/eval_results.json`

| Scenario | M1 | M2 | M3 | M4 | M5 | M6 | **Total** | Runtime |
|----------|-----|-----|-----|-----|-----|-----|-----------|---------|
| **S3_no_holdout_sanity** | 0.948 | 0.592 | 0.906 | 0.740 | 0.814 | 0.597 | **0.766** | 1046s |

Diagnostics:
- `M2 driver_auprc = 0.1964` under the previous reference run. Current M2 uses driver AUPRC only.
- `M3 w1_emd = 0.4125`, `w1_self = 0.3183`, `w1_self_std = 0.0042`, `w1_random = 1.3223`, `w1_self_method = bootstrap_same_size`
- `M6 edge_auprc = 0.5968`, `direction_accuracy = 1.0`

#### How to interpret the two S3 variants

- `S3` is the **real benchmark**. It hides bins `2,3` and measures actual holdout prediction quality.
- `S3_no_holdout_sanity` is a **pipeline sanity check**. It is useful for checking contract alignment, strict export behavior, and evaluator plumbing.
- The two scores should **not** be compared as if they were the same task difficulty.
- Expected pattern:
  - `S3_no_holdout_sanity` usually has higher `M1` and much higher `M3`
  - `M5/M6` should be close across the two splits if strict downstream export is behaving correctly
  - `M2` can still differ substantially because the learned growth-driver ranking itself can change between splits

### S1 (3 genes, 2 fates)

| Agent | M1 | M2 | M3 | M4 | M5 | M6 | **Total** |
|-------|-----|-----|-----|-----|-----|-----|-----------|
| CytoBridge seed43 | 0.965 | 0.997 | 0.873 | 0.942 | 0.939 | 0.333 | **0.842** |
| Codex | 0.962 | 0.665 | 0.960 | 0.940 | 0.939 | 0.333 | **0.800** |

> Note: S1 M6 baseline=0.667 (6/9 edges), limited discriminative power with 3 genes. S3 is the primary benchmark.

## Directory Structure

```
benchmark/dynbench/
├── README.md                        ← this file
├── BENCHMARK_DESIGN.md              ← design principles & factory guide
├── simulators/
│   ├── boolode_sim.py               # Core BoolODE SDE simulator
│   ├── scenario_factory.py          # ⭐ Batch scenario generator
│   ├── S1_toggle_switch.py          # S1 hand-crafted scenario
│   └── S3_complex_tree.py           # S3 hand-crafted scenario
├── eval/
│   ├── fate_classifier.py           # Unified MLP fate classifier
│   └── dynbench_eval_v2.py          # 6-metric evaluation pipeline
├── run_batch_agents.py              # Default batch runner for agents/repeats
├── run_dynbench.py                  # Single-scenario debug runner
├── ground_truth/                    # Generated ground truth
│   ├── S1_v2/, S3/                  # Hand-crafted scenarios
│   └── S_{easy,medium,hard}_seed*/  # Factory-generated scenarios
├── task_packages/                   # Agent input packages
│   ├── S1_v2/, S3/
│   └── S_{easy,medium,hard}_seed*/
└── results/                         # Agent outputs & eval results
```

## How to Run

### Batch Generate (Factory) — Recommended
```bash
conda activate CytoCompass

# Generate 9 scenarios: 3 difficulties × 3 seeds
cd benchmark/dynbench/simulators
python scenario_factory.py --difficulty easy,medium,hard --seeds 42,43,44

# Single scenario
python scenario_factory.py --difficulty medium --seeds 100
```

Each scenario auto-generates: ground truth (7 files), task package (train.h5ad + classifier + TASK.md), and validation report. See [BENCHMARK_DESIGN.md](BENCHMARK_DESIGN.md) for design principles.

### Hand-Crafted Scenarios
```bash
python benchmark/dynbench/simulators/S1_toggle_switch.py   # 3 genes, 2 fates
python benchmark/dynbench/simulators/S3_complex_tree.py    # 10 genes, 4 fates
```

### Evaluate Agent Output
```python
from benchmark.dynbench.eval.dynbench_eval_v2 import evaluate_all
results = evaluate_all('path/to/agent/output/', 'benchmark/dynbench/ground_truth/S3/')
```

### Run Agents in Batch — Default

Use `run_batch_agents.py` for normal DynBench experiments. It runs multiple
scenarios, agents, and repeats; uses GPU by default; verifies public output
schemas; scores completed runs; writes machine-readable summaries; and supports
resume from existing valid outputs.

```bash
cd /lustre/home/2501111653/CytoBridge-agent

# Recommended production-style batch run.
# CytoBridge tries Xiaomi first; if all Xiaomi attempts fail with provider/quota
# errors, it falls back to openai-codex / gpt-5.5 / medium and records that in
# batch_summary.jsonl/csv.
CYTOBRIDGE_XIAOMI_API_KEYS='key1,key2,key3' \
CYTOBRIDGE_XIAOMI_BASE_URLS='https://token-plan-sgp.xiaomimimo.com/v1,https://token-plan-cn.xiaomimimo.com/v1' \
/lustre/home/2501111653/miniconda3/envs/agent/bin/python benchmark/dynbench/run_batch_agents.py \
  --agents cytobridge,codex \
  --repeats 3 \
  --parallel 2 \
  --device cuda \
  --timeout 3600 \
  --run-root /lustre/home/2501111653/CytoBridge-agent/cytobridge_output/dynbench_batch_YYYYMMDD

# Limit to a subset while debugging the batch harness.
/lustre/home/2501111653/miniconda3/envs/agent/bin/python benchmark/dynbench/run_batch_agents.py \
  --scenarios S_balanced_easy_01_seed42,S_balanced_medium_01_seed142 \
  --agents cytobridge,codex \
  --repeats 3 \
  --parallel 2 \
  --device cuda
```

Outputs:
- `batch_manifest.json`: run configuration and scenario list
- `batch_summary.jsonl`: append-only per-run status and metric rows
- `batch_summary.csv`: spreadsheet-friendly summary
- `synthetic/<scenario>/agent_<agent>_skills-on_<run_label>/eval_results.json`: per-run official score
- `batch_runner_logs/*.log`: subprocess logs with secrets redacted

### Run One Scenario for Debugging

Use `run_dynbench.py` only for focused debugging of one scenario or one runner.
Do not use `cytobridge_agent.cli run` directly for DynBench.

The CytoBridge runner enables its public-data scientific harness by default.
Pass `--harness-revisions 0` to disable both the audit/revision loop and final
scientific calibration for a paired ablation. Implementation details and the
measured replay are documented in [`HARNESS_ENGINEERING.md`](../../HARNESS_ENGINEERING.md).

```bash
cd /lustre/home/2501111653/CytoBridge-agent

# Single CytoBridge run on GPU.
/lustre/home/2501111653/miniconda3/envs/agent/bin/python benchmark/dynbench/run_dynbench.py \
  --scenario S_balanced_easy_01_seed42 \
  --agent-type cytobridge \
  --mode skills-on \
  --device cuda \
  --seed 42

# Single Codex run.
/lustre/home/2501111653/miniconda3/envs/agent/bin/python benchmark/dynbench/run_dynbench.py \
  --scenario S_balanced_easy_01_seed42 \
  --agent-type codex \
  --mode skills-on \
  --device cuda \
  --seed 42
```

**Why not use `cytobridge_agent.cli run` directly?** The DynBench runners do
benchmark-specific orchestration that the general CLI does not:
1. Stage a clean per-run workspace and pass `TASK.md` as the user task.
2. Provide the correct output directory and public task resources.
3. Enforce benchmark read restrictions so hidden ground truth is not available to the agent.
4. Run strict public output verification before scoring.
5. Run `dynbench_eval_v2` after the agent finishes and save `eval_results.json`.
6. Record logs, summaries, provider/model metadata, and resume state.

Without these, runs are not comparable and can silently write files to the
wrong place, miss public resources, or bypass the strict output contract.

### Recommended usage pattern

When modifying task contracts, downstream skills, evaluator code, or benchmark-generation logic, use the following order:

1. Generate or select a balanced set of task packages across difficulty levels.
2. Run `run_batch_agents.py` with `--repeats 3`, `--device cuda`, and enough
   `--parallel` workers to use the GPU without overwhelming provider quotas.
3. Compare `batch_summary.csv` by scenario, agent, repeat, and provider.
4. If one run is anomalous, use the per-run output directory and
   `batch_runner_logs/*.log` for postmortem debugging, then reproduce with
   `run_dynbench.py` only if single-run isolation is needed.

### Debugging notes

- Batch runner logs are written to:
  - `<run-root>/batch_runner_logs/*.log`
- Per-run runner stdout/stderr and agent logs are written under:
  - `<run-root>/synthetic/<scenario>/agent_<agent>_skills-on_<run_label>/`
- If a run executes `execute_python`, the executed source is now persisted under:
  - `<run-root>/synthetic/<scenario>/agent_<agent>_skills-on_<run_label>/.runtime/execute_python_history/`
- For postmortem debugging, inspect `runner_stdout.log` first, then the `.runtime` directory, then the final manifests:
  - `holdout_prediction_manifest.json`
  - `fate_perturbation_manifest.json`
  - `driver_genes_manifest.json`
