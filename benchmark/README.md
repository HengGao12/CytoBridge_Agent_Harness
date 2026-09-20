# CytoBridge Benchmark Suite

This directory contains benchmark tooling for evaluating CytoBridge and other
agents on single-cell dynamical modeling tasks.

## Main Components

- `benchmark/dynbench/`: BoolODE-style synthetic dynamics benchmark with known
  ground truth for velocity, growth, fate, perturbation, and GRN metrics.
- `benchmark/agent_runners/`: adapters for running Codex, CytoBridge, and
  BioMini-style agents through a shared benchmark interface.
- `benchmark/dynbench/run_dynbench.py`: canonical DynBench runner for one
  scenario and one agent.
- `benchmark/run_all_benchmarks.py`: legacy batch orchestration for registered
  DynBench scenarios.
- `benchmark/run_all.py`: legacy real-data closed-loop runner based on YAML
  task cards such as `benchmark/configs/weinreb_2020.yaml`.

## DynBench Quick Start

Run a fixed v2 scenario once:

```bash
python benchmark/dynbench/run_dynbench.py \
  --scenario branching_tree_medium_seed42 \
  --agent-type cytobridge \
  --mode skills-on \
  --device cpu
```

CytoBridge runs enable the public-data scientific harness by default. It adds
one audit/revision round plus auditable growth-driver and perturbation
calibration. Use `--harness-revisions 0` for an ablation run. See
[`HARNESS_ENGINEERING.md`](../HARNESS_ENGINEERING.md) for the design, measured
score change, artifacts, and validation protocol.

Run the same fixed v2 scenario with downstream skills masked:

```bash
python benchmark/dynbench/run_dynbench.py \
  --scenario branching_tree_medium_seed42 \
  --agent-type cytobridge \
  --mode skills-off \
  --device cpu
```

V2 scenario data is generated from fixed JSON configs under
`benchmark/dynbench/simulators/test_configs/v2/`. These configs define the
benchmark data; repeated agent runs should use `--repeat`, not different seeds.

The repository includes the `S_balanced_easy_01_seed42` ground-truth and task
package as a small smoke-test fixture used by `baseline.sh` and `harness.sh`.
Other generated ground-truth and task-package directories are intentionally
excluded from Git to keep clone and push sizes manageable. Regenerate the
balanced scenario suite when needed:

```bash
python benchmark/dynbench/simulators/scenario_factory.py \
  --config benchmark/dynbench/generated_20_balanced_configs.json
```

Generate or inspect a legacy/custom synthetic scenario:

```bash
python benchmark/dynbench/simulators/scenario_factory.py \
  --config benchmark/dynbench/simulators/test_configs/compact_toggle_high_noise.json
```

Legacy preset/custom scenario generation still accepts numeric seeds, but this
is not the default comparison protocol for v2 DynBench.

## Legacy Runners

The older closed-loop real-data benchmark is still available through:

```bash
python benchmark/run_all.py --config benchmark/configs/weinreb_2020.yaml
python benchmark/run_all.py --config benchmark/configs/weinreb_2020.yaml --execute --device cpu
```

This path prepares folds, runs an agent, evaluates W1/W2 in PCA space, and
generates a report. It is separate from DynBench's ground-truth synthetic
evaluation.

Older batch wrappers such as `benchmark/batch_benchmark_runner.py` and
`benchmark/scripts/full_benchmark_loop.py` are historical orchestration paths.
Use `benchmark/dynbench/run_dynbench.py` for current DynBench runs unless you
are deliberately reproducing old results.

## Directory Layout

```text
benchmark/
├── agent_runners/        # Pluggable agent adapters
├── configs/              # Real-data YAML task cards
├── dynbench/             # BoolODE-style synthetic dynamics benchmark
├── scripts/              # Benchmark utility scripts
├── run_all.py            # Legacy real-data benchmark entry point
└── run_all_benchmarks.py # DynBench batch runner
```

Runtime outputs are written under `benchmark/results/` and are ignored by git.
