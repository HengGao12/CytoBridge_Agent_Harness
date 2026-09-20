# Benchmark Automation Architecture

This document describes the imported benchmark automation layer.

## Directory Layout

- `benchmark/run_all_benchmarks.py`: unified batch runner for registered DynBench entries.
- `benchmark/generate_summary_report.py`: converts a batch result directory into `summary.md`.
- `benchmark/dynbench/run_dynbench.py`: canonical DynBench single-run executor and evaluator.
- `benchmark/dynbench/task_packages/<scenario>/`: public DynBench task packages.
- `benchmark/dynbench/ground_truth/<scenario>/`: generated DynBench ground truth.
- `benchmark/results/<batch_id>/`: automation output. Each run has an `eval_results.json`.

The automation layer does not replace single-benchmark runners. It calls them,
records the command/output, copies native evaluation results, and writes a
normalized `eval_results.json` for each attempted run.

## Benchmark Registry

The registry lives in `benchmark/run_all_benchmarks.py` as `BENCHMARKS`.
Current entries are:

- `dynbench-s3`: canonical DynBench S3 holdout scenario.
- `dynbench-s3-sanity`: S3 no-holdout pipeline sanity scenario.
- `dynbench-test-toggle`: compact toggle smoke-test scenario.

Add a new DynBench scenario by adding an entry such as:

```python
"dynbench-my-scenario": {
    "type": "dynbench",
    "scenario": "My_Scenario",
    "mode": "skills-on",
}
```

## Adding A New DynBench Scenario

Preferred path:

1. Choose or create a scenario config for `benchmark/dynbench/simulators/scenario_factory.py`.
2. Build the scenario with `scenario_factory.py`.
3. Verify the generated `ground_truth/<scenario>/simulation_config.json`.
4. Verify `TASK.md` preserves weighted rollout and GRN-orientation requirements.
5. Add the scenario to `BENCHMARKS`.

Example:

```bash
python benchmark/dynbench/simulators/scenario_factory.py \
  --config benchmark/dynbench/simulators/test_toggle_medium_config.json

python benchmark/run_all_benchmarks.py \
  --agent-type codex \
  --benchmarks dynbench-test-toggle \
  --seeds 42
```

## Adding A New Agent Type

Agent selection is centralized in `benchmark/agent_config.py`.

1. Add aliases to `AGENT_TYPE_ALIASES`.
2. Add a branch in `resolve_agent_selection`.
3. Choose an existing runner or add a new one under `benchmark/agent_runners/`.
4. Add the new type to CLI `choices` in `benchmark/run_all_benchmarks.py` and
   `benchmark/dynbench/run_dynbench.py`.
5. Run `python benchmark/run_all_benchmarks.py --agent-type <agent> --check-auth-only`.

## Running The Suite

Single benchmark:

```bash
python benchmark/run_all_benchmarks.py \
  --agent-type codex \
  --benchmarks dynbench-s3-sanity \
  --seeds 42
```

All registered DynBench entries:

```bash
python benchmark/run_all_benchmarks.py \
  --agent-type codex \
  --benchmarks all \
  --seeds 42
```

Batch output contains:

- `command.txt`: exact command.
- `runner_stdout.log`: stdout/stderr captured from the single-run runner.
- `native_eval_results.json`: evaluator output from the canonical runner, if available.
- `eval_results.json`: normalized automation record with status, runtime, scores, and error.
- `summary.csv`, `summary.md`, `run_manifest.json`: batch-level summaries.

## Troubleshooting

- Authentication fails: run `cytobridge-agent auth codex-login`, then
  `python benchmark/run_all_benchmarks.py --agent-type codex --check-auth-only`.
- Native `eval_results.json` missing: inspect `runner_stdout.log` and any
  `native_error.json` in the run directory.
- BioMini fails early: inspect `benchmark/agent_runners/biomni_runner.py`
  preflight output and the configured BioMini repository.
- Long silent phases: inspect native runner logs such as `agent_log.txt` and
  `runner_stdout.log`.
- No-holdout sanity entries are pipeline checks, not formal holdout scores.
