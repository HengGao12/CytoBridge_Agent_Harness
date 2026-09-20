# CytoBridge Benchmark Batch Plan

This file records the intended three-agent, three-seed DynBench stress-test
matrix. Prefer topology-spec JSON files over natural-language scenario
requests when reproducibility matters.

## Execution Pattern

```bash
python benchmark/batch_benchmark_runner.py \
  --topology-spec-file benchmark/dynbench/simulators/test_configs/<config>.json \
  --difficulties medium \
  --seeds 42,137,256 \
  --agents codex,cytobridge,biomini \
  --output-dir benchmark/results/synthetic_batches/<batch_name>
```

Rules:

- Existing generated scenario assets may be reused.
- A stale `results.json`, `summary.md`, or `report.html` is not evidence that a
  new benchmark run has completed.
- Completion evidence must come from the current batch output directory and the
  current native runner logs.

## Scenario Matrix

| # | Scenario | Topology | Fates | Approx. genes | Noise | Confounders | Holdout | Main stress |
|---|----------|----------|-------|---------------|-------|-------------|---------|-------------|
| 1 | asymmetric_8fates | asymmetric_tree | 8 | 27 | 0.18 | 4 | hard | Asymmetric fate proportions and anti-confounding |
| 2 | competitive_4fates | competitive_multistable | 4 | 21 | 0.22 | 3 | very hard | Long-gap interpolation and competitive topology |
| 3 | compact_toggle | compact_toggle | 2 | 3 | 0.25 | 0 | hard | Small-sample, high-noise robustness |
| 4 | branching_16fates | branching_tree | 16 | 35 | 0.15 | 5 | hard | Large GRN and deep fate hierarchy |
| 5 | feedback_6fates | branching_with_feedback | 6 | 16 | 0.14 | 2 | medium | Feedback loops and asymmetric branches |
| 6 | confbomb_4fates | branching_tree | 4 | 18 | 0.12 | 8 | medium | Growth-driver and GRN anti-heuristic stress |
| 7 | sparse_4fates | branching_tree | 4 | 8 | 0.08 | 0 | medium | Easy sparse-GRN sanity baseline |
| 8 | competitive_8fates | competitive_multistable | 8 | 46 | 0.20 | 6 | very hard | Maximum-complexity stress test |

Recommended order:

1. Sparse baseline and compact-toggle stress test.
2. Confounder-heavy stress test.
3. Feedback/asymmetric medium-complexity scenarios.
4. Competitive long-gap and 16-fate deep-tree scenarios.
5. Extreme competitive scenario.
