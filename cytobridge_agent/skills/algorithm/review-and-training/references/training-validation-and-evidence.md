# Training, Validation, And Evidence Hygiene

Use this reference after pretraining checks pass.

## Preview

Run `preview_training_run(...)` before spending real trials when code,
inference, or metrics changed.

Preview has two phases:

- full-data chain inspection on the requested benchmark/config, including
  backend construction, coupling/path/mass setup, `build_state`, `sample_pairs`,
  and `sample_batch`;
- if inspection succeeds, a 1-epoch training/inference/evaluation smoke run.

Use it to catch:

- broken hooks
- NaN/inf
- impossible metric ranges
- inference code that cannot produce a standard trajectory
- obvious runtime/memory failures

Do not interpret 1-epoch metric magnitude as algorithm performance. If preview
times out before epoch logs appear, treat it as setup/build-state scalability or
contract failure, not as evidence that one training epoch is slow.

Preview `ok=false`, training timeout, inference timeout, OOM, or runtime
exception means there is no trusted W1/TMV/claim evidence yet. Treat it as a
blocking implementation/runtime issue and fix the failing component before
starting or continuing campaign trials. A partial W1 printed before an
inference timeout is diagnostic only, not campaign evidence.

If custom inference changed, especially `simulation_hook(...)` or
`inference_context_builder(...)`, the runtime may request a read-only
`inference_evaluator`. Claim-metric-only edits should not trigger this reviewer
unless they mutate prediction artifacts or replace the rollout path.

## Execution Paths

Manual one-off validation:

- `run_training(training_algorithm_id="<algorithm_id>")`
- records a run only
- does not update campaign active best
- does not auto-promote/reject
- does not restore rejected workspace edits
- requires a clean target snapshot for trusted custom evidence
- if the algorithm lifecycle status is `developing`, this run is debug/manual
  evidence only; final user-facing or downstream results must come from a
  campaign `final_regression` locked release

Campaign tuning:

- `start_algorithm_campaign(...)`
- edit active workspace/config
- choose/freeze a prepared benchmark panel
- `run_campaign_trial(...)`
- `check_campaign_stage_gate(advance=false|true)`

Campaign trials automatically snapshot the current workspace if no trial is
open, archive code/config, run the backend, aggregate metrics, decide
`promote`/`reject`, and restore active best after rejected trials. Do not call
`snapshot_active_algorithm_workspace(...)` before each campaign trial.

Use `start_campaign_trial(...)` or `resume_rejected_trial(...)` only when you
intentionally want to restore a specific base before editing. The common loop is
edit current workspace/config, then call `run_campaign_trial(...)`.

If a previous interruption leaves `current_trial_id` pointing at a
`status=running` trial and no live training/evaluation process remains, do not
start another trial by hand. Use `abort_current_campaign_trial(...)` with a
specific reason. The tool clears `current_trial_id`, records the trial as
`aborted`, and by default restores the stage active-best workspace snapshot.

If the current stage panel is the wrong evidence target, switch it with
`switch_campaign_stage_panel(...)`. This keeps the stage budget count intact but
invalidates old active-best/gate/baseline evidence for that stage, so rerun a
trial and refresh baselines on the new panel before checking the gate.

Prepared benchmark data live under:

`.cellcompass/algorithm_benchmarks/datasets/<dataset_id>/`

Do not spend campaign budget on preprocessing unless the task is to add or
repair a benchmark dataset.

## Evaluation Semantics

Trusted metrics are generated from dynamics rollout.

- runtime first generates a t0-to-final `EvaluationTrajectory`
- W1/TMV/claim metrics read this trajectory or observed-time slices
- separate metric-specific prediction paths are not trusted evidence
- W1 compares observed future cells to the simulated trajectory slice in latent
  space
- predicted particle weights are normalized for W1
- TMV measures total predicted mass mismatch against observed relative mass
  `n_k / n_0`

Reject or revise if:

- builtin W1/TMV definitions are edited without explicit package-level metric
  redesign
- `simulation_hook(...)` uses future observed cells/counts/mass or future
  distribution statistics to build predictions
- prediction artifacts are post-hoc corrected to satisfy W1/TMV/claim metrics
- a hook returns only metric-specific predictions instead of the full requested
  trajectory
- total mass is corrected after rollout only to satisfy TMV
- unbalanced mass is claimed but rollout mass comes from target-count lookup,
  time-only count-ratio repair, or post-hoc renormalization rather than the
  approved growth/dynamics mechanism

Balanced-only methods may report poor TMV as diagnostic evidence; TMV should be
a hard failure only for algorithms/runs that claim to model unbalanced mass.

## Baseline Comparison

Compare against relevant builtin/reference algorithms, not one number.

Useful anchors:

- `crufm`
- `vgfm` for deterministic unbalanced flow matching
- `wfrfm` for WFR dynamic unbalanced OT flow matching
- `balanced_ot_cfm` for balanced simulation-free flow matching
- `sf2m` for stochastic bridge claims

Interpret `v` and `g` jointly:

- `v` changes predicted locations
- `g` changes predicted masses
- validation consumes the resulting predicted measure
- growth changes can affect both TMV and W1
- worse W1 may come from distorted weights, not only positions

Use `list_algorithm_benchmark_baselines(...)` and campaign status before making
claims about baseline gaps.

## Evidence Hygiene

After meaningful custom results:

- inspect `risk.md`
- update linked research idea progress when relevant
- separate `trial promoted` from `stage gate passed`
- mark obsolete runs if a later semantic finding invalidates them
- use `rollback_algorithm_workspace(...)` for bad semantic edits

Proposal revision reset:

- material proposal revision creates a new `proposal_id`
- campaign evidence belongs to the proposal semantics that produced it
- after Stage 2 passes, avoid proposal/core semantic edits unless there is a
  concrete scientific error

Campaign decisions:

- `promote` updates stage active best
- `reject` archives the trial and restores active best
- direct `run_training(..., decision=...)` is legacy/manual mode and should not
  be used for campaign tuning

If review finds a problem, do not train anyway. Fix code, patch/review the proposal, or
run a manual debug job explicitly labeled as non-campaign evidence.
