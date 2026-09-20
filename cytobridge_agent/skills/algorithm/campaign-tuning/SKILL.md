---
name: campaign-tuning
description: Autoresearch-style campaign loop for custom algorithm tuning with frozen benchmark panels, automatic promote/reject, and stage gates.
---

# Campaign Tuning

Use this skill when an approved custom algorithm already has a workspace and
you need iterative tuning evidence.

This is the entrypoint, not the full manual. Read only the reference section
needed for the immediate next action.

## Core Rule

The agent edits code/config and requests trials. The campaign controller
archives, runs, evaluates, and decides `promote` or `reject`.

Do not pass manual trial decisions.

Campaign `promote` is an optimization-control decision against the current
stage `active_best`. It is not, by itself, scientific validation of the claimed
mechanism. If a promoted trial is still far from the same-panel baseline, hides
a real-dataset failure, or has a high custom claim metric that contradicts
mechanism-level diagnostics, stop the trial loop and use `../tuning-playbook`
before editing or launching another trusted trial.

A Stage 2 claim metric is credible only when it measures the algorithm's stated
claim. It should answer "did this method solve the problem it was designed for?"
beyond W1/TMV. Do not let a generic proxy such as activity, variance, spread,
nonzero growth, or an internal diagnostic become the formal claim metric unless
the proposal explains why that proxy is tied to the claimed mechanism and why it
would fail under a random, collapsed, shuffled, or claim-absent control. For
biological-application claims, do not pass Stage 2 on a synthetic/toy
simulation; obtain a real biological dataset that observes the claim or defer
the claim. Use controlled Stage 2 simulations only for explicitly theory-first
or non-biological claims, or as supplementary controls that do not replace the
real-data biological gate.

Do not downgrade Stage 2 to pass a failing direction. If the algorithm was
proposed to fix a real-data gap, the Stage 2 primary metric must still test that
gap. A weaker proxy, such as geometry concordance in place of true
lineage/fate-concordant transitions, broad distribution smoothness in place of
condition response, or nonzero growth in place of mass recovery, is not valid
unless a reviewed proposal proves it is necessary for the original claim and
the frozen controls would fail without the claimed mechanism.

Do not downgrade the algorithm itself to rescue a failing campaign. Tuning may
change hyperparameters, regularization, architecture details, or a proposal-
faithful implementation bug. It must not turn the candidate into a baseline
config variant, diagnostic-only workflow, metric adapter, post-hoc correction,
or low-novelty proxy method just to pass a gate. If the candidate cannot solve
the original gap, stop trusted campaign promotion and send the direction back to
proposal review for a stronger algorithm or rejection.

During a custom algorithm lifecycle, tune the candidate algorithm, not builtin
baselines. New campaigns default to `strict_all_builtin`: builtin baselines are
fixed comparators, and stage gates audit the full default builtin comparator
set. An explicit `baseline_algorithms` list on
`refresh_campaign_stage_baselines(...)` is only a repair scope for
running/refreshing those named missing or stale baselines; required builtins
outside that scope remain explicit blockers until they are repaired too.
Permissive mode is an operator/server compatibility setting, not an
agent-tunable shortcut; use it only when the user explicitly wants the older
flexible mode. Refresh builtin baselines only when baseline evidence is
missing/stale, a config path is wrong, or a builtin run has
concrete abnormal evidence such as extreme W1/TMV, mass collapse, timeout, NaN,
or a large unexplained gap to other baselines. For agent-proposed OT/UOT/WFR
algorithms with anomalously bad metrics, diagnose the candidate's coupling
quality first: cost scale, solver mass/convergence, row/column coverage, plan
sparsity, chunking, and setup-versus-epoch placement. Key builtin knobs, when
builtin diagnosis is explicitly justified, include VGFM/CRUFM `reg` and
`reg_m`, regularization strategy, and WFR-FM `delta` or auto-delta settings.

If a proposal needs a scientific control comparator, such as shuffled labels,
claim-absent side information, a no-lag/no-signal ablation, or another
mechanism-specific negative control, run that control on the same frozen stage
panel with `run_campaign_control_baseline(...)`. Declare the source as either a
sparse `control_code_source.config_overrides` ablation or a temporary
`control_code_source.workspace_patch`. The tool measures metrics and registers
the control without promoting/rejecting a campaign trial or changing active
best. Registered controls are active audited extra gate comparators: they do not
replace builtin/reference baselines, but Stage 2/3 gate checks include active
controls when selecting the strongest comparable baseline for each metric. If a
control was mis-coded, use `update_campaign_control_baseline(...)` to deactivate
or replace it with revision history. Do not hand-edit `campaign.json` or type
unmeasured control numbers.

If the candidate is a refinement, extension, or deeper variant of a previous
agent-completed algorithm, run that completed method as a same-panel comparator
with `run_campaign_locked_algorithm_baseline(...)`. The reference must be a
final-regression locked release. The tool evaluates the locked reference on the
current frozen stage panel and registers it as audited gate evidence. Do not
reuse old paper/final-regression numbers from a different panel. If the current
`.h5ad` contract does not match the locked reference algorithm, write an
explicit `reference_dataset_adapter` Python script that reads `--input-adata`
and writes `--output-adata`; do not use config overrides or guesses to change
the locked reference semantics. If the current claim-metric evaluator expects a
different context shape for the locked reference, pass
`reference_claim_metric_adapter` with an explicit
`adapt_baseline_metric_context(context)` wrapper. This wrapper may standardize
context fields before the current evaluator runs; it must not replace the
evaluator, inject manual metric values, or retune the locked reference.

## Minimal Trial Loop

1. `get_current_workflow_context()`
2. `get_algorithm_campaign_status(...)` or `start_algorithm_campaign(...)`
3. edit the active algorithm workspace/config. For config edits, change the
   real `config.yaml` with `apply_workspace_patch(...)`; for simple scalar
   hyperparameters, `patch_algorithm_config(...)` is a safe shortcut.
4. `preview_training_run(...)` if code, inference, or metrics changed
5. choose/freeze the stage panel with `make_benchmark_dataset_config(...)` or a real-data dataset payload. For biological-application campaigns, trusted Stage 1/2/3/final evidence should use the full prepared real dataset whenever computationally feasible; smaller panels are smoke/debug evidence unless larger or full-data validation supports the same claim.
6. `run_campaign_trial(...)`
7. inspect `decision`, `metrics_summary.per_dataset`, `stage_internal_status`,
   `stage_gate_status`, and `next_required_action`
8. if anomaly triggers fire, diagnose before edits/trials: suspicious custom
   claim metric, large baseline gap, aggregate improvement hiding real-data
   failure, or unaudited side information on the active benchmark
9. refresh missing baselines with `refresh_campaign_stage_baselines(...)` or
   `compute_campaign_claim_metric_for_baselines(...)`
10. repeat, or call `check_campaign_stage_gate(...)`

Do not manually call `snapshot_active_algorithm_workspace(...)` before campaign
trials. `run_campaign_trial(...)` creates the trial snapshot and archive commit.

## Acceleration Cheatsheet

If a trial is slow, OOM-prone, or CPU-bound, first try semantics-preserving
engineering acceleration:

- chunk OT/UOT/WFR per adjacent time gap instead of building full
  `n_t x n_{t+1}` cost/plan/mask state
- use true mini-batch, streaming, sparse candidates, landmarks, or coresets when
  the proposal permits the estimator
- keep POT solver inputs on GPU when possible; `sinkhorn_unbalanced(...)` and
  `mm_unbalanced(...)` can consume CUDA tensors
- cache invariant costs and parallelize independent per-gap work
- do not treat `use_mini_batch=True` as proof if dense state is still built first

If Stage 1 hits a training timeout or extreme per-epoch slowdown twice, stop
campaign trials before changing more hyperparameters. Compare the same dataset
against the nearest builtin run using run metadata, not progress-log fractions:
completed epochs, elapsed seconds, per-epoch time, timeout flag, build/setup
time, W1/TMV, and per-dataset breakdown. Then inspect whether the custom method
is doing work inside the epoch path that the builtin precomputes in setup.
Common offenders are repeated OT/UOT/WFR solves, repeated pairwise cost
matrices, repeated side-information joins, and noisy per-batch growth targets.
Fix the computational data flow first; do not lower epochs or drop benchmark
datasets just to satisfy the time budget.

## Non-Negotiable Rules

- Normal tuning does not require `start_campaign_trial(...)`. Edit the current
  workspace/config and call `run_campaign_trial(...)`; if no trial is open, the
  current workspace is snapshotted as the candidate.
- `start_campaign_trial(...)` is only for intentionally restoring a working
  base before edits, such as resuming a rejected branch. It is not a restart
  button. A campaign has at most one open trial; repeated calls return the same
  open trial.
- If a pre-training review or launch guard blocks a trial, fix the issue and
  rerun that same trial instead of trying to start a replacement trial.
- If `run_campaign_trial(...)` reports an open `status=running` trial but no
  training/evaluation process is actually alive, treat it as stale interrupted
  state. Call `abort_current_campaign_trial(...)` with a concrete reason, keep
  the safe default `restore_active_best=true`, then rerun the trial from the
  current intended workspace/config.
- Persistent config tuning must modify the real workspace `config.yaml`, so
  campaign promote/reject archives the actual configuration. Use
  `apply_workspace_patch(...)` for ordinary config edits. Use
  `patch_algorithm_config(...)` only as a shortcut for simple scalar
  hyperparameters such as `training.plan[0].lr`; it blocks accidental whole-list
  replacements like `training.plan=[...]` unless explicitly allowed.
- Campaign/tool config overrides should be sparse. If you tune one
  hyperparameter, pass only that leaf path/value, for example
  `{"config_overrides": {"training.plan[0].lr": 0.002}}` or the corresponding
  `per_dataset_config_overrides` entry. Do not copy the whole `training` or
  `model` block into `dataset_config_overrides`; broad copied defaults make
  trial evidence stale/noisy and can trip fairness guards. The runtime prunes
  redundant copied defaults, but sparse leaf overrides are the expected form.
- Each stage normally freezes its dataset ids once. If the chosen panel is
  clearly wrong or missing an important scalability/real-data check, use
  `switch_campaign_stage_panel(...)` instead of ad hoc paths. This preserves the
  stage trial count/budget but invalidates the stage active best, gate evidence,
  and external baseline metrics; rerun `run_campaign_trial(...)` and
  `refresh_campaign_stage_baselines(...)` for the new panel.
- If you need to inspect another dataset for debugging, use manual
  `run_training(...)` or `preview_training_run(...)`. Do not treat that run as
  campaign promote/reject or stage-gate evidence.
- If you need to run a nearest-builtin degeneration, disabled-component
  ablation, or other diagnostic control, also use manual `run_training(...)` or
  `preview_training_run(...)`, not `run_campaign_trial(...)`. A campaign trial
  is a formal candidate trial; proposal-required components must be enabled
  unless the proposal has already been revised and reviewed to define the
  simpler method.
- Stage 1 and Stage 3 should include
  `weinreb_rawrebuild_full_k10mindiff1` for algorithms intended for real
  biological data, so feasibility/tuning also tests scalability.
- If the algorithm is motivated by a real biological problem and a suitable
  registered real benchmark exists, prefer including real data in each stage
  panel or as a guardrail so failures appear early. Simulation-only panels are
  appropriate for theory-first algorithms, controlled mechanism tests, Stage 2
  claim isolation, or cases where no real benchmark matches the assumptions.
- Campaign metrics must be computed from the trained dynamics rollout /
  `EvaluationTrajectory`, not from future-data lookup, post-hoc correction, or
  metric-specific shortcut prediction.
- Optional `holdout_time_evaluation` on `run_campaign_trial(...)` is a
  default-off auxiliary diagnostic for trajectory generalization. It first runs
  the normal non-holdout campaign trial, then runs split training that omits
  selected time point(s) and computes held-out W1 against the real omitted
  cells. Use it only when this evidence is scientifically relevant; attach it
  to custom claim metrics only if candidate and baselines use the same
  hold-out protocol. This diagnostic does not replace the normal Stage 2
  `claim_metric_spec.evaluator_path` / provenance requirement.
- To make held-out timepoint W1 the formal Stage 2 metric, configure the
  campaign claim metric as `evaluator_path: builtin:holdout_time_w1`,
  `direction: lower`, and put the split protocol under
  `claim_metric_spec.holdout_time_evaluation`. Do not rely on a per-trial
  diagnostic flag for gate evidence.
- Stage 2 claim metrics must be semantically meaningful for the approved
  proposal. If the available dataset cannot support the claim, register a
  controlled Stage 2 simulation that isolates the mechanism and reruns
  comparable baselines on the frozen simulation version. For BoolODE-style
  controlled simulations, read `../boolode-stage2-simulation/SKILL.md`.
- A proposal revision that changes the primary claim metric must be treated as
  a claim-change event, not a tuning trick. Before using new Stage 2 evidence,
  verify that the revised metric still answers the original motivating gap; if
  it does not, redesign the algorithm or the controlled benchmark instead of
  continuing the campaign.
- When real data is used for a claim metric, verify that metadata semantics are
  contracted, not guessed. A column named `cell_type`, `label`, `lineage`,
  `barcode`, or a time-bin field does not by itself prove terminal fate truth,
  descendant fate truth, lineage endpoint truth, growth truth, or perturbation
  truth. If the metric depends on such semantics, point to the approved
  proposal, implementation map, benchmark card, dataset registration,
  preprocessing contract, or dataset documentation before trusting the gate.
- If stop-hook/lifecycle pressure says to continue but metrics are
  contradictory, continue by diagnosing. Reading metrics, auditing a claim
  metric, inspecting rollout geometry, writing a diagnosis artifact, or checking
  side-information coverage all count as concrete progress.
- For mixed panels, inspect `metrics_summary.per_dataset`. A simulation win must
  not hide failure on the real or biologically central benchmark.
- When a real benchmark contains side information used by the algorithm's
  claim, audit coverage, cardinality, imbalance, missingness, adjacent-time
  comparability, interval reliability, and representation compatibility before
  trusting campaign evidence.
- For unbalanced-mass algorithms, campaign TMV must come from the approved
  growth/dynamics mechanism in that rollout. Do not tune by adding target-count
  clocks, post-hoc rescaling, or renormalization that has no biological or
  theoretical meaning.
- If runtime is too high, improve implementation efficiency with
  semantics-preserving mini-batch/chunking/streaming/GPU/vectorization/caching,
  not by weakening the evidence protocol.
- Builtin flow-matching methods use 3000 epochs by default. Start custom
  flow-matching configs from the seeded 3000-epoch default unless convergence
  evidence justifies a shorter schedule.
- Do not fine-tune from an already trained run, cached fitted state, or
  target-specific predictions to make a trial look fast.
- Do not claim the algorithm lifecycle is complete after Stage 1 or Stage 2
  alone. Completion means: Stage 1 feasibility gate passed, Stage 2 claim
  validation gate passed, Stage 3 tuning/generalization was completed or
  intentionally exhausted by policy, and final regression produced a final
  result / locked release. Only the final-regression locked release is the
  formal final algorithm result; use that result, not preview/manual
  `run_training(...)` or intermediate campaign trials, for user-facing
  summaries and downstream analysis.

## What To Read Next

- Stage panels, baselines, claim metrics, budgets, and stage semantics:
  `references/panels-and-stage-gates.md`
- Scalability patterns, tuning discipline, simulations, and failure handling:
  `references/scalable-tuning-and-failures.md`
- If Stage 2 needs a controlled BoolODE-style synthetic dataset:
  `../boolode-stage2-simulation/SKILL.md`
- If implementation review blocks the trial:
  `../review-and-training/SKILL.md`
- If progress stalls after several trials:
  `../tuning-playbook/SKILL.md`

## Quick Interpretation

- `trial promoted`: this trial improved over the campaign's own stage
  `active_best`.
- `stage gate passed`: the stage `active_best` also passed its external
  builtin/reference baseline gate.
- Stage 3 is optional as a failure gate, not optional as effort. After Stage 2
  passes, the algorithm is validated; Stage 3 should actively lower W1 toward
  SOTA while preserving the validated claim metric and Stage 2 guardrails.
  Its optional early-completion gate expects W1 and the validated claim metric
  to be non-worse than the strongest comparable external baseline, with claim
  metric regression capped at 5 percent. If budget is exhausted without this
  optional gate, proceed to final regression rather than treating the algorithm
  as failed.
- Final regression is not a pass/fail gate. It freezes semantics, runs the
  final benchmark/regression panel, locks the release, and changes the
  algorithm lifecycle status to complete. If final regression has not locked a
  release yet, treat the algorithm as still developing even if earlier stages
  look promising.
- If component diagnostics show that the current algorithm direction cannot
  satisfy the user goal, call `mark_algorithm_failed(...)` with concrete
  evidence. A `failed` lifecycle status is not completion; after marking it,
  revise/repair the algorithm, revise the proposal if the proposal assumption
  is wrong, or design a replacement algorithm.

Before claiming a stage is complete, call `get_algorithm_campaign_status(...)`
and read the explicit user-facing status fields.
