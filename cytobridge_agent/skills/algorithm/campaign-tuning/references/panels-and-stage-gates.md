# Campaign Panels And Stage Gates

Use this reference when choosing datasets, refreshing baselines, or deciding
whether a campaign stage can advance.

## Frozen Panel Contract

Stage panel identity is stable by default, but can be intentionally switched
inside a stage when the current panel is clearly wrong or too narrow.

- Freeze the panel with `set_campaign_stage_panel(...)`, or let the first
  `run_campaign_trial(...)` payload with `dataset_config_overrides.datasets`
  freeze it.
- After freezing, ordinary trials should not change dataset ids.
- Later trials may pass config-only overrides for those same dataset ids.
- If you must change dataset ids inside the same stage, use
  `switch_campaign_stage_panel(...)`. This preserves `trial_count`,
  `promote_count`, and `reject_count`, but clears the current stage active best,
  external baseline metrics, and gate evidence because old metrics are not
  comparable to the new data panel.
- After switching, run `run_campaign_trial(...)` on the new panel and then
  `refresh_campaign_stage_baselines(...)`; the stage gate will then compare
  against baselines selected for the new `target_dataset_ids`.
- `run_campaign_trial(...)` snapshots the current workspace if no trial is
  open. `start_campaign_trial(...)` is only for restoring a specific base before
  edits; it does not change the frozen panel and must not be used as a restart
  button.
- If an interrupted run leaves an open `status=running` trial with no live
  process, recover with `abort_current_campaign_trial(...)` before the next
  campaign trial. This is an operational cleanup, not a reject decision.
- If another dataset is needed only for debugging, use manual
  `run_training(...)` / `preview_training_run(...)`. It is not campaign gate
  evidence unless the dataset is part of the frozen panel for a stage/campaign.

Use benchmark catalog tools instead of ad hoc paths:

- `list_algorithm_benchmarks(...)`
- `get_algorithm_benchmark_dataset(...)`
- `list_algorithm_benchmark_baselines(...)`
- `make_benchmark_dataset_config(...)`

Dataset-specific configs are allowed. The frozen panel locks dataset ids, not
per-dataset hyperparameters.

For persistent workspace config tuning, edit the durable `config.yaml` with
`apply_workspace_patch(...)`. Use `patch_algorithm_config(...)` only for simple
scalar leaf-path edits when its type/list guards are helpful.
`run_campaign_trial(...)` reads the live workspace `config.yaml` when it
launches training; it does not train from the pre-trial snapshot except as
rollback provenance. On promote, if all dataset runs produced one identical
`resolved_config.yaml`, the campaign controller syncs that resolved config back
to workspace `config.yaml` and snapshots it as the active best. If different
datasets have distinct resolved configs, the controller keeps them as
dataset-specific evidence and does not collapse one into the global workspace
config.

`make_benchmark_dataset_config(...)` overrides are per-trial/per-dataset
payloads. For benchmark-managed dataset defaults, store them under
`.cellcompass/algorithm_benchmarks/datasets/<dataset_id>/trial_configs/` as
`default.yaml`/`.json` or `<stage>.yaml`/`.json` with a `config_overrides`
object. `make_benchmark_dataset_config(...)` loads that dataset config first,
then applies explicit `per_dataset_config_overrides`.

```python
make_benchmark_dataset_config(
    dataset_ids=["simulation_gene_2d", "weinreb_rawrebuild_full_k10mindiff1"],
    common_config_overrides={},
    per_dataset_config_overrides={
        "simulation_gene_2d": {},
        "weinreb_rawrebuild_full_k10mindiff1": {
            # real-data/scalability-specific batch size, chunking, or UOT params
        },
    },
)
```

## Recommended Stage Panels

Stage 1 should test feasibility and scalability.

- Prefer including `simulation_gene_2d` for fast smoke evidence.
- Include `weinreb_rawrebuild_full_k10mindiff1` when the proposal is intended
  for real biological data.
- Passing only tiny 2D simulations is not enough evidence of real-data
  feasibility unless the proposal is explicitly toy/simulation-only.

Stage 2 should test the claimed capability.

- Prefer existing benchmark datasets if they satisfy the claim assumptions.
- Create a Stage 2 simulation only if existing data cannot fairly test the
  claim.
- A custom simulation must freeze generator/source into `simulation_version` and
  rerun builtin/reference baselines on that exact version.
- The Stage 2 claim metric must be meaningful for the approved claim. It should
  distinguish the proposed mechanism from generic rollout activity, variance,
  nonzero growth, or another proxy that could improve when the claimed structure
  is absent.
- If no existing dataset contains the labels, side information, perturbation,
  spatial structure, interaction graph, lineage signal, or other ground truth
  needed for the claim, build a controlled simulation with claim-present and
  claim-absent/shuffled controls.

Stage 3 is real-data tuning/generalization.

- Prefer `weinreb_rawrebuild_full_k10mindiff1` as the real-data W1 tuning target.
- Carry the Stage 2 dataset forward as a claim-regression guardrail when
  applicable.
- If Weinreb is omitted, document why the algorithm is not intended for that
  real-data scale.

## Baseline Policy

Do not manually choose a weak baseline. Use
`refresh_campaign_stage_baselines(...)`.

- Stage 1 uses W1 and compares to the second-weakest relevant builtin/reference
  W1 baseline, falling back to the only available comparable baseline.
- Stage 2 uses the strongest comparable builtin/reference claim metric baseline
  and requires 10 percent improvement.
- When a scientifically required control/ablation exists, such as shuffled
  labels or a no-signal/no-lag variant, run it on the same frozen panel with
  `run_campaign_control_baseline(...)` using an explicit `control_code_source`
  (`config_overrides` or `workspace_patch`). The tool measures and registers the
  control without promoting/rejecting a trial or changing active best. Active
  registered controls raise, not replace, the builtin/reference gate bar when
  they are the strongest comparable baseline for a metric. If the control was
  mis-coded, use `update_campaign_control_baseline(...)` to deactivate or
  replace it with revision history.
- Stage 2 W1 is a scale-aware guardrail against the selected baseline: low-scale
  simulation may allow up to 1.5x W1, while higher-W1 real-data panels are
  tightened automatically.
- Stage 3 has an optional SOTA/Pareto early-completion gate, but it is not an
  algorithm failure gate. It should actively lower W1 while preserving the
  validated claim metric before final regression. The optional gate expects W1
  and the validated claim metric to be non-worse than the strongest comparable
  external baseline. Claim metric regression is capped at 5 percent against the
  fixed Stage 2 active best that passed claim validation; this floor does not
  move downward with later Stage 3 active-best regressions.
- Final regression is confirmation-only. It inherits the latest active best,
  confirms the fixed Stage 2 claim-metric floor still holds, locks/reports that
  frozen result, and does not impose an external baseline pass/fail gate. Do not
  run new final-regression tuning/config trials.

Stage gates compare against external baseline evidence from the same frozen
panel. Self-improvement alone is not a passed stage gate.

## Claim Metric Baselines

For custom Stage 2 claim metrics, `claim_metric_spec.name` and `direction` are
only metadata. Provide an executable evaluator:

- `claim_metric_spec.evaluator_path`
- `claim_metric_spec.evaluator_function` if not using the default

The evaluator should read `EvaluationMetricsContext.evaluation_trajectory` or
its observed-time slices. It must not repair predictions, rescale mass, move
particles, or use future truth during prediction.

If a campaign was already started without `claim_metric_spec.evaluator_path` and
Stage 2 returns `claim_metric_evaluator_required`, repair the active campaign
with `update_campaign_claim_metric_spec(...)`. Do not edit `campaign.json`
directly, do not pass `evaluator_path` through `dataset_config_overrides`, and
do not start a duplicate campaign just to repair the evaluator.

The evaluator should also be a valid observable for the claim. A high score
must mean that the proposed mechanism succeeded under the declared data
contract, not merely that the rollout produced a large auxiliary signal. When
that cannot be established on existing data, run the claim metric on a frozen
controlled simulation with relevant builtin/reference and ablation controls.

If builtin/reference baselines already have saved `evaluation_trajectory_path`,
refresh claim metrics from those trajectories without retraining. If only a
saved model exists, the system can regenerate the standard trajectory. Train
only genuinely missing baselines.

If a benchmark dataset requires a special builtin baseline config, store it in
`.cellcompass/algorithm_benchmarks/datasets/<dataset_id>/builtin_configs/` as
`builtin_<algorithm>.yaml`/`.json` or `<algorithm>.yaml`/`.json` with a
`config_overrides` object. `refresh_campaign_stage_baselines(...)` applies this
dataset-scoped config whenever it must rerun that builtin. Builtin defaults are
the first choice, but dataset-scoped configs are appropriate for justified
data-sensitive overrides. WFR-FM should normally use its default auto-delta;
fixed `delta` values require explicit evidence or a controlled diagnostic
comparison.

Use `compute_campaign_claim_metric_for_baselines(...)` when the candidate has a
claim metric but baselines only have W1/TMV.

`baseline_metric_adapters` are allowed only for I/O standardization, baseline
metadata/params, or explicit `supported=false` declarations with a concrete
reason. Do not provide replacement numeric claim metrics through adapters, and
do not change the metric definition per baseline.

For Stage 2, the canonical baseline claim metric storage is
`metrics.custom_metrics[claim_metric_name]` plus a top-level
`metrics.claim_metric_evaluator` identity. Candidate and baseline metrics must
carry the same evaluator identity. Direct fields such as
`metrics[claim_metric_name]`, manually typed numbers, and numeric declarative
adapter values are not comparable Stage 2 evidence unless they were produced by
the current campaign evaluator and carry matching provenance.

## Stage Budgets

- Stage 1 feasibility: up to 10 trials.
- Stage 2 claim validation: up to 50 trials.
- Stage 3 tuning/generalization: up to 30 trials. It may finish early if the
  optional SOTA/Pareto gate passes; if budget is exhausted without that optional
  gate, final regression is still allowed.
- Final regression: one confirmation/lock slot inherited from the latest active
  best; no new tuning/config trials.

If a Stage 1 or Stage 2 gate passes before budget is exhausted, the agent may
advance or keep tuning within budget. If Stage 3 cannot pass the optional
SOTA/Pareto gate within budget, stop tuning and proceed to final regression
without treating the algorithm as failed.

## Lifecycle Completion Rule

Do not treat the custom algorithm lifecycle as finished until all of these are
true:

- Stage 1 feasibility gate has passed on its frozen panel.
- Stage 2 claim-validation gate has passed on its frozen panel.
- Stage 3 tuning/generalization has either completed its intended tuning plan
  or exhausted policy/budget and intentionally advanced.
- Final regression has run and produced the final result / locked release.

Final regression is not a pass/fail gate, but it is the only formal completion
artifact. Use the final-regression locked-release metrics for user-facing
summaries and downstream analysis; preview runs, manual `run_training(...)`, and
intermediate promoted trials are debugging or campaign-selection evidence only.

`trial promoted` only means the trial improved the algorithm's own stage
`active_best`. It does not mean the stage gate passed, and it does not mean the
lifecycle is complete.

## Stored State

`check_campaign_stage_gate(...)` persists gate evidence in:

`.cellcompass/algorithm_campaigns/<algorithm_id>/<campaign_id>/campaign.json`

Inspect `get_algorithm_campaign_status(...)` before recomputing baselines.

Proposal revision reset:

- material proposal revision creates a new `proposal_id` and resets the active
  campaign to Stage 1
- Stage 1 W1/TMV evidence may remain useful if only the claim metric changed
- Stage 2 evidence is tied to proposal semantics, frozen panel,
  claim-metric/evaluator version, and active-best snapshot
