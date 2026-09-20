---
name: tuning-playbook
description: Diagnosis and tuning playbook for custom scientific algorithms after campaign trials exist. Use when hyperparameter tuning stalls, active best remains far from builtin/reference baselines, or the agent needs to inspect risk.md, trained active-best models, baseline gaps, and diagnostics before deciding whether to keep tuning or patch/review the proposal.
---

# Tuning Playbook

Use this skill when campaign results exist but the next improvement is unclear.
This is a diagnosis workflow, not a generic training API manual.

## Core Operating Model

Improve the approved algorithm with the fewest justified experiments while
preserving proposal semantics, active-best lineage, campaign `promote`/`reject`
discipline, reproducible evidence, and clear stopping decisions.

Campaign tuning has three layers:

1. local hyperparameter/config search around the current stage `active_best`;
2. risk-driven diagnosis using `risk.md`, the trained active-best model,
   builtin/reference baseline gaps, rollout artifacts, and targeted metrics;
3. proposal revision only when diagnostics show a real semantic/modeling failure.

Do not jump to redesign just because W1 or the claim metric plateaus. Also do
not keep burning trials on blind hyperparameter search after repeated
non-progress. A plateau is a diagnosis trigger.

## Required Context

Before tuning, inspect:

- `get_current_workflow_context()`
- `get_algorithm_campaign_status(campaign_id=...)`
- `list_campaign_trials(campaign_id=..., decision="")`
- `~/.cellcompass/training_algorithms/<algorithm_id>/PROPOSAL.md`
- `~/.cellcompass/training_algorithms/<algorithm_id>/risk.md`
- `~/.cellcompass/training_algorithms/<algorithm_id>/IMPLEMENTATION_MAP.md`

Treat `risk.md` as a severity-ordered diagnosis queue, not proof that the
proposal is wrong.

## Builtin Anchor Rule

Treat the nearest working builtin algorithm as the engineering anchor. Many
builtin CytoBridge algorithms already have valid W1/TMV behavior, stable
runtime placement, chunking, solver lifecycle, and benchmark-specific configs.
If the custom method is much worse, times out, or fails completely, first find
which component differs from the builtin before assuming the proposal is
impossible.

During a custom algorithm lifecycle, tune the candidate algorithm, not the
builtin baselines. Builtin baselines are fixed comparators by default. Refresh
or tune them only when baseline evidence is missing/stale or concrete evidence
shows abnormal baseline training quality; then inspect the resolved config and
change only justified data-sensitive knobs.

Compare against the builtin on the same dataset panel:

- builtin config, epochs, batch/chunk size, solver regularization, and
  dataset-specific overrides;
- reused package classes/functions for coupling, conditional path, loss,
  backend, rollout, and metric evaluation;
- runtime placement: expensive OT/UOT/WFR solves and invariant cost/state should
  not move into the epoch or batch hot path unless explicitly required;
- degeneration behavior: if a natural setting makes the custom method behave
  like the builtin, test it with manual `run_training(...)` or preview as a
  diagnostic control.

Never run nearest-builtin degeneration or component-neutralized ablations as
`run_campaign_trial(...)`. They are debugging evidence, not formal candidates.
If a degenerate control works better, use it to identify the broken component;
do not let it become campaign active best unless the proposal is formally
revised and reviewed to make that simpler method the actual algorithm.

## Stage Progress Rules

- Stage 1 primary metric is W1; TMV is hard only if the algorithm models
  unbalanced mass.
- Stage 2 primary metric is the frozen claim metric. The gate uses the strongest
  comparable builtin/reference baseline and requires 10 percent improvement;
  W1 is a scale-aware guardrail.
- Stage 3 primary metric returns to W1. It is optional as a failure gate, but not
  optional as effort: actively tune W1 toward SOTA while preserving the validated
  claim metric. Claim-metric regression is capped at 5 percent against the fixed
  Stage 2 active best that passed claim validation, not against a drifting Stage
  3 active best. If the optional SOTA/Pareto gate is not met before budget is
  exhausted, run final regression rather than failing the algorithm only if this
  fixed floor still holds.
- Final regression is not a pass/fail gate or tuning stage. It inherits the
  latest active best, confirms the fixed Stage 2 claim floor still holds, locks
  the release, and marks the algorithm complete. Do not run new final-regression
  tuning/config trials.

## Local Search Rules

Start with low-risk config changes only when a concrete hypothesis exists:
learning rate, scheduler, batch/chunk size, regularization, solver tolerance,
proposal-approved noise/coupling/mass parameters, non-semantic caching or
runtime acceleration, and per-dataset final/benchmark config.

Rules:

- change one hypothesis at a time when possible;
- express one-parameter config changes as sparse leaf overrides, for example
  `training.plan[0].lr`, rather than copying the full `training` or `model`
  block into a tool call;
- if a knob helps, explore a small neighborhood;
- if 2-3 nearby settings do not move the primary metric, stop that direction;
- use `preview_training_run(...)` before spending campaign trials when code,
  inference, or metrics changed;
- use `run_campaign_trial(...)` for formal candidates and never hand-label
  campaign decisions;
- use manual `run_training(...)` or preview for nearest-builtin degeneration,
  component-neutralized ablations, or debugging controls; do not let these runs
  enter campaign promote/reject or active best unless the proposal is revised and
  reviewed.

Do not lower epochs only to pass a wall-clock budget, and do not use warm-start,
cached fitted state, or target-specific predictions as campaign evidence.

For builtin-related tuning, remember the default is "do not tune". When
evidence justifies an exception, prioritize data-sensitive knobs with clear
metric or diagnostic support. Important examples are VGFM/CRUFM UOT `reg`,
`reg_m`, and regularization strategy; WFR-FM `delta` or auto-delta settings;
chunk size; batch size; learning rate; and schedule length.

## Diagnosis Triggers

Stop blind tuning and create a diagnosis artifact before another trusted
campaign trial when any of these occur:

- recent trials mostly reject or active best remains far from the relevant
  builtin/reference baseline;
- primary metric is flat after a small local search;
- the same failure repeats across several nearby trials;
- runtime, timeout, memory, or setup cost dominates;
- failures cluster in one time point, dataset, subgroup, or real-data benchmark;
- W1/TMV/claim metric trade off in a stable pattern;
- a custom claim metric is high while mechanism diagnostics are random, zero,
  missing, or contradictory;
- aggregate improvement hides failure on a biologically central or real-data
  benchmark;
- side information used by the claim has not been audited for coverage,
  cardinality, imbalance, missingness, cross-time comparability, and reliability.

A stop-hook or lifecycle reminder to "continue" does not mean "launch another
trial". Reading metrics, auditing rollout geometry, writing a diagnosis
artifact, checking side-information coverage, or inspecting baseline gaps are
valid progress.

## Diagnosis Minimum

A useful diagnosis should compare same-panel active best against relevant
builtin/reference baselines and identify the first failing boundary:

- data/panel contract;
- OT/UOT/WFR coupling or cost scale;
- conditional path / supervision targets;
- growth or mass targets;
- optimization/training;
- inference rollout and mass dynamics;
- metric/evaluator contract;
- proposal/theory assumptions.

If an agent-proposed algorithm uses OT, UOT, or WFR coupling and W1/TMV/claim
metrics are anomalously bad, treat coupling quality as the first failure
boundary. Inspect finite costs, scale, solver convergence, plan total mass,
row/column coverage, plan sparsity or collapse, chunk/mini-batch equivalence,
and whether terminal mass targets implied by the coupling are biologically
plausible before spending more scalar optimizer trials or revising the proposal.

If a high-severity `risk.md` item is active, do not spend more campaign trials on
unrelated scalar hyperparameters until the risk is fixed, ruled out, or the
proposal is revised.

## When To Read The Detail Reference

Read `references/tuning-diagnosis-details.md` when:

- creating the required tuning diagnosis table;
- debugging a severe plateau, timeout, memory issue, or builtin gap;
- auditing custom claim metrics or side-information-dependent claims;
- deciding whether to continue local search, run an ablation, resume a rejected
  trial, stop a stage, proceed to final regression, or revise the proposal;
- writing a handoff tuning summary.

The reference contains the full diagnosis table, component-level checklist,
risk-driven diagnosis workflow, subagent guidance, error categories, stop rules,
and summary template.

## Common Hard Boundaries

- Bad W1/TMV/claim metrics are evidence about the model, not an invitation to
  manipulate metric paths.
- Do not add target-count repair, post-hoc rescaling, metric-specific shortcuts,
  or future-time information to force metrics to pass.
- Do not make semantic code changes without patching/reviewing the proposal.
- Balanced-only algorithms should not be blocked by TMV.
- A promoted trial is self-improvement, not external stage-gate success.
- Final user-facing results must come from final-regression locked release, not
  preview, manual `run_training(...)`, or intermediate campaign trials.
