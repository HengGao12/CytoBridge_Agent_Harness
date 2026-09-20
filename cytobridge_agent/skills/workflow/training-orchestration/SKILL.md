---
name: training-orchestration
description: Run a reproducible CytoBridge training workflow, compare runs against baselines, and produce a defensible training decision.
---

# Training Orchestration

Use this skill when training a CytoBridge model, evaluating candidate configurations, or developing a custom training algorithm.

## Goal
Run a reproducible CytoBridge training workflow that produces a clear run record, stable artifacts, and a defensible training choice.

## Read first when unsure
- `CytoBridge-main/docs/runtime/custom-algorithms/README.md`
- `CytoBridge-main/docs/runtime/flow-matching/README.md`
- `~/.cellcompass/training_algorithms/<algorithm_id>/manifest.yaml`
- `~/.cellcompass/training_algorithms/<algorithm_id>/algorithm.py`
- `~/.cellcompass/training_algorithms/<algorithm_id>/config.yaml`

## Operating rules
- Prefer builtin configs when algorithmic changes are not required.
- Create a custom algorithm workspace only when the task truly requires new model, stage-runner, simulation, coupling, path, mass, or backend logic.
- Prefer small file patches over full rewrites.
- Keep algorithm development under `~/.cellcompass/training_algorithms/`; do not modify package configs or package source directly.
- For custom algorithms, active context controls editing, while `training_algorithm_id` controls the training target. Training B while active context is A is allowed and should not switch the active editing target.
- Training artifacts should be reproducible and traceable to a specific run directory.
- Do not change `epochs` by default. Builtin flow-matching configs use 3000
  epochs; custom flow-matching configs should start from that seeded default.
  Do not lower epochs merely to finish faster, fit a wall-clock budget, make a
  downstream deliverable quick, or run a benchmark case cheaply. A shorter
  schedule is acceptable only when you have concrete convergence evidence for
  this dataset/model family, such as stable loss/metric curves from prior full
  runs or an already validated early-stopping schedule. If you need fast
  debugging, use `preview_training_run(...)` or train on an explicitly reduced
  dataset with `adata_path`; do not treat that debug run as final evidence.
- Do not use an already trained checkpoint, cached fitted state, or target-specific precomputed predictions to make a trial look fast. Training evidence should reflect scratch training from the configured initialization on the target dataset unless the approved proposal explicitly defines a transferable warm-start mechanism.

## Recommended workflow
1. Confirm training prerequisites
   - usable `preprocessed_path`
   - valid `time_key`
   - valid `label_key` when needed
   - chosen theory/model family
2. Decide whether builtin or custom training is needed
   - builtin for config/parameter choice
   - custom only for algorithmic change
3. If custom:
   - first read the algorithm lifecycle entrypoint:
     - `~/.cellcompass/skills/algorithm/algorithm-orchestrator/SKILL.md`
   - before proposal approval, complete the theory/interface gate and write the
     proposal pseudocode
   - define the evaluation plan at proposal time:
     - builtin `W1/TMV` stay fixed
     - any new metric must be additive and scientifically justified
   - for actual implementation work, read:
     - `~/.cellcompass/skills/algorithm/authoring/SKILL.md`
   - create/update conceptual proposal first (`create_algorithm_proposal`)
   - ensure proposal status is approved (or auto-approved) before workspace edits/training
   - if proposal is pending user review, wait for explicit user UI/API decision
   - initialize the algorithm workspace
   - read the manifest and algorithm entrypoint
   - read and verify local `config.yaml` (generated from selected builtin baseline)
   - ensure backend assumptions are consistent with `config.yaml` components/train_strategy
   - if the method changes the model family itself, use `model_builder(...)`
   - if the method changes stage semantics, use `stage_runner(...)`; if only
     custom batches/losses are needed, prefer package
     `run_custom_stage_loop(...)` inside the runner over a hand-written
     optimizer loop
   - if evaluation-time prediction differs from the builtin simulator, use
     `inference_context_builder(...)` plus `simulation_hook(...)`; the inference
     context payload is flexible, but provenance should show t=0/exogenous/model
     state sources
   - patch files incrementally
   - validate that the algorithm returns a valid `TrainingAlgorithmSpec`
   - after code/config edits are done and before any training run, read:
     - `~/.cellcompass/skills/algorithm/review-and-training/SKILL.md`
   - complete that self-review, including the pseudocode-to-code line mapping,
     before manual `run_training(...)` or campaign `run_campaign_trial(...)`
   - treat the custom algorithm directory as a registry-backed experiment track:
     - inspect history first:
       - `list_experiment_history(...)`
     - if semantic changes are required:
       - patch `PROPOSAL.md` with `apply_workspace_patch(...)` to create a reviewed proposal revision
     - if you later invalidate old evidence:
       - `mark_result_obsolete(...)`
     - if you need to restore a known-good workspace:
       - `rollback_algorithm_workspace(...)`
4. Choose the correct training runner
   - for one-off/manual validation, use the unified `run_training(...)` tool
   - for iterative custom-algorithm tuning, use campaign tools and let
     `run_campaign_trial(...)` call the training backend
   - before launching a real run, first call:
     - `preview_training_run(...)`
   - use the preview to inspect:
     - merged config summary
     - selected model/backend/coupling/path/mass preview
     - whether a custom `stage_runner(...)` or `simulation_hook(...)` is active
     - data shape and time-point summary
     - concrete preflight errors and non-blocking scalability warnings from
       full-data chain inspection
     - the optional 1-epoch inference/evaluation smoke test after inspection
     - metric sanity warnings, especially custom metrics outside declared range
   - preview metrics are not performance evidence; use them only to catch broken
     inference, NaN/inf, invalid shapes, or impossible metric ranges
   - if preview times out before epoch logs appear, debug setup/build-state first:
     coupling cost, solver calls, plan store, sampler metadata, and path setup
   - if a custom metric should live in a known range, declare it in
     `evaluation_metrics_params.expected_ranges`, for example
     `{"rare_branch_auroc": {"min": 0, "max": 1}}`
   - do not start epochs if preview shows a concrete contract error; scalability
     preflight messages are warnings, so use them to improve the implementation
     but do not treat them as automatic blockers
   - use `candidate_name` for builtin configs/families
   - use `training_algorithm_id` for custom algorithm workspaces
   - before custom training, the tool checks the target algorithm's approved proposal, clean snapshot, and dirty state
   - optionally pass `adata_path` to explicitly train on a specific `.h5ad` for that run
   - `run_training(...)` is a manual/debug run. Prefer the full valid dataset; observed cell counts across time points may encode proliferation/death or total-mass change, and more valid cells usually improve evidence. If you truly need a smaller smoke/debug dataset, prepare a reduced adata explicitly in Python first using the same sampling ratio within each `obs["time_point_processed"]` value, then pass it with `adata_path`
   - optional `holdout_time_evaluation` is default-off. Enable it only when trajectory generalization or dynamics reasonableness is part of the claim, or when held-out timepoint W1 is intentionally used as a comparable claim metric. The normal full-data run completes first; auxiliary split run(s) then omit selected time point(s), simulate the held-out distribution, and write W1 diagnostics under the parent run's `holdout_time_evaluation/` directory. Supported modes are `single`, `sequential`, and `simultaneous`; do not enable it by habit if a better mechanism-specific claim metric is available. If this becomes a formal Stage 2 claim metric, it still needs a campaign `claim_metric_spec.evaluator_path` and matching baseline provenance; the diagnostic flag alone is not enough.
   - for formal Stage 2 held-out timepoint W1, set campaign `claim_metric_spec.evaluator_path` to `builtin:holdout_time_w1`, `direction` to `lower`, and put the split protocol in `claim_metric_spec.holdout_time_evaluation`. The campaign controller will run the same auxiliary split protocol for candidate and baselines.
   - for custom algorithms, persistent parameter/component changes should be made in local workspace `config.yaml`; reserve `config_overrides` for temporary ad-hoc runs
   - use `apply_workspace_patch(...)` for normal `config.yaml` edits; use `patch_algorithm_config(...)` only as a shortcut for simple scalar leaf paths such as `training.plan[0].lr`
   - `patch_algorithm_config(...)` writes `config.yaml` by default and returns a resolved diff, so campaign promote/reject archives the actual tuned config; use `dry_run=true` only when you explicitly want preview-only
   - default flow-matching runs have coupling preflight validation: invalid/near-zero coupling mass now fails early with explicit diagnostics (fix coupling/path/regularization instead of retrying blindly)
   - custom `stage_runner(...)` runs are previewed through `preview_only=True`; if the runner explicitly declines preview support, treat that as a warning and review the stage runner before full training
   - only fall back to direct Python/backend calls when you have a concrete
     reason not to use the standard workflow tool, such as a documented missing
     tool capability for this model family or analysis target. Convenience,
     speed, benchmark pressure, or uncertainty about the API is not enough.
   - if you must use direct package training calls, you own the same quality
     contract as `run_training(...)`: save the resolved config, metrics, model
     artifact, and training log; compute or inspect W1/TMV/quality diagnostics;
     write an explicit accepted/rejected verdict; and do not proceed to
     downstream deliverables from a rejected or unassessed model.
5. Capture:
   - run id
   - resolved config
   - trained model path
   - metrics
   - structured run verdict
   - after every non-preview `run_training(...)`, open the saved run artifacts
     before downstream work:
     - `resolved_config.yaml`
     - `artifacts/metrics.json`
     - `logs/training.log` when metrics look suspicious or the tool output was
       truncated
   - do not rely on the chat/tool snippet alone; long metric arrays can be
     truncated. The authoritative training-quality evidence is in the saved
     config, metrics, and log files.
6. Evaluate training quality before accepting the run
   - this is mandatory for both builtin algorithms and custom algorithms
   - do not treat “training finished” as “training succeeded”
   - do not move to downstream analysis or final file delivery until the chosen
     run is usable as a dynamics model. A schema-valid export from a collapsed,
     unstable, or clearly underfit model is not a successful CytoBridge result.
   - inspect builtin `W1` and `TMV` for every run from
     `artifacts/metrics.json`
   - inspect `run_verdict.status` in `artifacts/metrics.json`. If it is
     `rejected`, the run is not usable for downstream analysis, benchmark file
     delivery, or reporting even if the tool call says artifacts were saved and
     even if a public/schema verifier passes.
   - inspect `completed_epochs`, `training_timeout`,
     `training_timeout_message`, configured `training.plan[*].epochs`, and the
     loss trace in `logs/training.log` when available
   - explicitly write a short training-quality verdict before downstream:
     `accepted_for_downstream`, `debug_only_undertrained`, or
     `failed_needs_retrain`
   - acceptable quality gate:
     - `W1` should stay within the data scale range in latent space
     - `TMV < 0.3` only when the algorithm/run models unbalanced mass or
       cell-count / total-mass change
     - if the algorithm is balanced-only, TMV is still recorded but is not a
       hard rejection gate
     - final evidence should not use a shortened epoch schedule unless saved
       metrics/logs show dataset/model-specific convergence evidence; a run
       with reduced epochs, unstable loss spikes, large W1, or collapsed rollout
       composition is debug evidence only
     - for prediction or downstream-delivery tasks, inspect whether generated
       rollout distributions and fate/readout summaries are biologically and
       numerically plausible. Obvious mode collapse, extreme fate concentration,
       exploding/oscillating loss after the best checkpoint, or W1/TMV far worse
       than nearby runs means the model is not yet acceptable even if the public
       file verifier passes
   - if the quality gate fails:
     - first check whether the run is undertrained or numerically unstable
     - if the run used fewer epochs than the resolved builtin/default schedule
       and there is no convergence evidence, retrain with the default schedule
       or a justified validated schedule before downstream claims
     - first try hyperparameter adjustment before changing algorithm logic:
       learning rate, regularization, coupling/mass settings, batch size, and
       schedule are normal levers when saved metrics show poor W1/TMV, unstable
       loss, or collapsed rollout composition
     - if several justified settings for one model family remain poor, switch to
       another suitable CytoBridge model family instead of exporting from a bad
       run. For example, compare VGFM, CRUFM/CUR-FM, WFR-FM, balanced versus
       unbalanced variants, or SDE/score-enabled variants according to the data
       and theory-selection skill
     - during a custom algorithm lifecycle, tune the candidate algorithm, not
       builtin baselines
     - builtin baselines are fixed comparators by default. In
       `strict_all_builtin` mode, gates audit the full default builtin
       comparator set. Explicit shorter builtin baseline lists are repair
       scopes for named missing/stale baselines, not gate comparator selection.
       Permissive mode is an operator/server compatibility setting; use it only
       when the user explicitly asks for the older flexible baseline mode
     - refresh builtin baselines only when baseline evidence is missing/stale,
       a config path is wrong, or a builtin run has concrete abnormal evidence
     - when builtin tuning is justified, inspect the resolved config and change
       only justified data-sensitive knobs before treating the builtin as
       unusable
     - typical first adjustments:
       - learning rate
       - regularization strength
       - coupling-related hyperparameters
       - batch size
       - training schedule, but not epoch reduction unless convergence evidence
         already shows the shorter schedule is scientifically comparable
     - for agent-proposed OT/UOT/WFR methods with abnormal W1/TMV or mass
       behavior, diagnose coupling quality first: cost scale, solver mass/convergence,
       row/column coverage, plan sparsity or collapse, chunking, and whether
       coupling construction is in setup rather than the epoch hot path
     - important builtin knobs, when builtin tuning is justified, include
       VGFM/CRUFM `reg` and `reg_m`, regularization strategy, and WFR-FM
       `delta` or auto-delta settings
     - only move to algorithm modification when the evidence suggests the issue
       is not mainly due to hyperparameters
7. Record the run decision explicitly
   - for iterative algorithm tuning, prefer campaign tools:
     - `start_algorithm_campaign(...)`
     - `start_campaign_trial(...)`
     - edit the active algorithm workspace/config
     - `run_campaign_trial(...)`
     - `check_campaign_stage_gate(advance=false)` to inspect readiness without moving stages
     - `check_campaign_stage_gate(advance=true)` to intentionally advance or lock
   - campaign trials are automatically `promote` or `reject`; do not hand-label them
   - do not manually snapshot before each campaign trial; `run_campaign_trial(...)` creates the trial snapshot and archive commit
   - `promote` updates the stage active best
   - `reject` archives the trial and restores the stage active best
   - strict stage budgets: Stage 1 has 10 trials, Stage 2 has 50, Stage 3 has 30, final regression has one confirmation/lock slot inherited from the latest active best
   - proposal revisions reset the active campaign to Stage 1; do not carry later-stage evidence across changed proposal semantics
   - use `refresh_campaign_stage_baselines(...)` rather than choosing a baseline manually; strict campaigns use all runnable default builtin baselines on the frozen panel as fixed comparators. A shorter explicit builtin list is allowed only when the operator/server campaign policy is set to `permissive`.
   - if a shuffled/claim-absent/no-signal ablation or other scientific control is needed, run it on the same frozen stage panel with `run_campaign_control_baseline(...)` using explicit `control_code_source` (`config_overrides` or `workspace_patch`). It measures metrics and registers an active audited extra comparator for gate selection without changing active best; it does not replace builtin/reference baselines. If the control was mis-coded, call `update_campaign_control_baseline(...)` to deactivate or replace it with revision history.
   - if the candidate extends a previous agent-completed algorithm, use `run_campaign_locked_algorithm_baseline(...)` to rerun that final-regression locked reference on the same frozen stage panel and register it as audited comparator evidence. Do not cite old paper/final-regression numbers from another panel. If the `.h5ad` contract differs, provide an explicit `reference_dataset_adapter` Python script that materializes an adapted h5ad; do not use config overrides or guesses to alter locked reference semantics. If the current claim-metric evaluator needs a different context shape for the locked reference, provide `reference_claim_metric_adapter` with `adapt_baseline_metric_context(context)`; it must wrap the current evaluator, not replace the metric or inject manual values.
   - Stage 2 gates require the claim metric to beat the strongest comparable baseline by 10 percent; W1 is a secondary guardrail with a scale-aware tolerance: low-scale simulation benchmarks may allow up to 1.5x the selected baseline, but higher-W1 real-data baselines are tightened automatically (for example around 1.2x at high latent W1 scale)
   - Stage 2 passing means the algorithm is validated/workable; after that, avoid proposal or core semantic changes unless there is a concrete scientific error
   - Stage 2 datasets are automatically inherited by Stage 3 as claim-regression guardrails
   - Stage 3 is optional as a failure gate, but not optional as effort: W1 is the universal distribution-fit metric for every algorithm and lower is better; tune actively toward the optional SOTA/Pareto early-completion gate while preserving or improving the validated claim metric. The optional gate expects W1 and the claim metric to be non-worse than the strongest comparable external baseline. Claim-metric regression is capped at 5 percent against the fixed Stage 2 active best that passed claim validation; this floor does not move downward with later Stage 3 active-best regressions. If budget is exhausted without this optional gate, proceed to final regression rather than treating the algorithm as failed only if that fixed Stage 2 claim floor still holds.
   - final regression is not a pass/fail gate or tuning stage; it inherits the latest active best, confirms the fixed Stage 2 claim-metric floor still holds, locks/reports that frozen result, and marks the algorithm lifecycle complete. Do not run new final-regression tuning/config trials. Only this locked-release result should be used for user-facing reporting or downstream analysis
   - the locked final-regression release must use the approved-proposal mechanism, not a diagnostic degeneration/ablation setting. Degenerate controls are useful for debugging and baseline comparison, but they are not a valid final result unless the proposal has been honestly revised to that simpler method
   - claim/custom metric names must match what the evaluator actually computes. Do not report a proxy under the name of a different biological or statistical quantity; either compute the named metric faithfully or rename the metric and update the proposal/campaign claim
   - lifecycle status is explicit: active development is `developing`; a locked final-regression release is `complete`; if evidence shows the current direction cannot satisfy the user goal, use `mark_algorithm_failed(...)` and then repair/revise/design a replacement rather than stopping
   - after a Stage 1 or Stage 2 gate passes, you may keep tuning until that stage budget is exhausted, or advance when the result is good enough
   - every `cb.tl.fit` training run has a wall-clock budget based on adjacent time gaps: about 1 minute per 5000 bottleneck cells `min(n_t, n_{t+1})`, without 5000-cell ceiling inflation, plus small sublinear coupling/setup overhead and safety rounding; timeout runs still evaluate but are not promotable
   - direct `run_training(..., decision=...)` is legacy/manual mode and should not be used for campaign tuning
   - direct `run_training(...)` does not update campaign active best, does not auto-judge `promote/reject`, and does not restore rejected edits. For a `developing` custom algorithm, it is debug evidence only, not lifecycle completion
   - for manual custom-algorithm training, ensure the target algorithm has a clean active snapshot; if `dirty_since_snapshot=true`, snapshot first
8. Commit the training result into workflow state

## Common failure modes
- creating a custom algorithm when a builtin config would suffice
- editing package source instead of the algorithm workspace
- using custom backend logic that conflicts with the workspace `config.yaml` components/train_strategy
- failing to record the final trained model path or run directory
- not preserving the relationship between training run, config, and metrics
- jumping into full training without a clear theory selection decision
- running a custom algorithm without first checking registry state
- accepting a run without either a campaign auto-decision or an explicit legacy manual decision
- continuing to cite old invalid runs because they were never marked obsolete

## Training Data Preparation Contract (important)
Current package fit path (`CytoBridge-main/CytoBridge/tl/fit.py`) builds training tensors as:
1. read `time_key = "time_point_processed"`
2. `time_points = sorted(unique(time_point_processed))`
3. for each time point: `subset = adata[adata.obs[time_key] == t]`
4. training `X` per timepoint comes from `subset.obsm["X_latent"]`

Implications for pilot data prep:
- Pilot data should still preserve biologically meaningful `time_point_processed`.
- Do not downsample by default. For biological-application campaigns, trusted Stage 1, Stage 2, Stage 3, final regression, and downstream claims should use the full prepared real dataset whenever computationally feasible.
- If you subsample for an unavoidable smoke/debug run, do it per time point with the same sampling ratio to preserve temporal cell-count proportions, record original counts and the reason, and keep the run labeled as debug/smoke evidence.
- Do not rely on fixed absolute sample counts as a universal rule; fixed per-time counts can erase proliferation/death or total-mass evidence and should not support strong mass/growth claims without larger/full-data validation.
- Keep the full prepared dataset as the authoritative artifact for real training, campaign evidence, final regression, and downstream claims unless the user or data constraints explicitly require otherwise.

Order consistency notes:
- Within each time-point subset, row order follows the original `adata` row order for those selected cells.
- Model output projection (`all_data = adata.obsm["X_latent"]`, `all_times = adata.obs[time_key]`) uses the current `adata` row order directly.
- Therefore `velocity_latent` / `growth_rate` rows align with the final `adata` rows one-to-one.

## Metric Semantics (W1 / TMV / RME)
Current implementation is in `CytoBridge-main/CytoBridge/tl/trainer.py:evaluate`.

Per target time point `t_i` (`i >= 1`):
- the runtime first creates one dense t0-to-final model rollout
  (`EvaluationTrajectory`, default output step 0.1)
- predicted trajectory points and masses are selected from that rollout:
  - positions: `x_hat(t_i)`
  - masses: `w_hat(t_i)`
- target observed samples are `X(t_i)` from `adata.obsm["X_latent"]` subset at that time.
- custom claim metrics must read this same trajectory artifact, not rerun a
  separate metric-specific predictor.

### W1
- computed with OT `ot.emd2(...)` on Euclidean cost matrix between `X(t_i)` and `x_hat(t_i)`.
- target weights are uniform over observed cells.
- predicted weights are normalized `w_hat(t_i) / sum(w_hat(t_i))`.
- lower is better; `0` is perfect distributional alignment in latent space.

### TMV (can be read as absolute Relative Mass Error, RME)
- code formula:
  - `TMV_i = abs(sum(w_hat(t_i)) - N_i / N_0)`
- where `N_i` is observed cell count at `t_i`, `N_0` is count at initial time.
- this is effectively absolute mass mismatch on a relative scale.
- lower is better; `0` means predicted total relative mass matches observed relative mass.

### Practical interpretation
- Absolute values alone are dataset-dependent and hard to judge globally.
- Always compare model metrics against a simple baseline and report relative gain.
- A run is not acceptable just because metrics exist.
- For routine acceptance:
  - `TMV` should be below `0.3` only when the algorithm/run models unbalanced
    mass or cell-count / total-mass change
  - `W1` should be interpretable as being within the latent data scale range,
    not obviously much larger than the geometry of the data itself
- If these conditions fail, investigate training quality before trusting any
  downstream analysis.

### Component semantics
- Do not interpret separate component flags (`velocity`, `growth`, `score`) as
  proving that these parts are scientifically independent.
- In the current runtime, `velocity` and `growth` jointly determine the
  predicted future measure:
  - `velocity` determines where simulated particles move
  - `growth` determines what weights those particles carry
- Because `W1` is computed against normalized predicted weights, growth affects
  distribution-fit quality, not only mass metrics.
- Therefore a model can worsen `W1` through bad growth semantics even when the
  latent trajectories themselves look reasonable.

## Baseline Protocol (recommended)
Use a baseline that requires no training and is easy to explain.

### Baseline A (primary): adjacent-time persistence
- for each target time `t_i` (`i >= 1`), use previous observed distribution `X(t_{i-1})` as baseline prediction.
- use constant baseline total mass (`sum(w_hat)=1`) unless you explicitly model a different prior mass hypothesis.
- compute:
  - `W1_baseline_i = OT(X(t_i), X(t_{i-1}))` with uniform weights.
  - `RME_baseline_i = abs(1 - N_i / N_0)`.
- this baseline is usually more realistic for temporal continuity and is the default reference.

### Baseline B (practical model baseline): CRUFM
- run builtin `crufm` as a practical algorithm baseline because it is usually fast and stable.
- do not require every new method to strictly beat CRUFM on all metrics.
- if a new method is comparable to CRUFM while clearly improving the target scientific objective, that is acceptable.
- still enforce quality floor: metrics must not be substantially worse without a clear justification.

## Reporting Rule for Metrics
When reporting training quality, include both absolute metrics and baseline-relative improvement:
- `mean_W1_model`, `mean_RME_model` (or `mean_TMV_model`)
- `mean_W1_baseline`, `mean_RME_baseline`
- `mean_W1_crufm`, `mean_RME_crufm` when CRUFM run is available
- relative gains:
  - `W1_gain = 1 - mean_W1_model / mean_W1_baseline`
  - `RME_gain = 1 - mean_RME_model / mean_RME_baseline`

If denominator is `0`, mark gain as `NA` and explain why.

## Mandatory Post-Training Quality Gate
After every builtin or custom training run:

1. open the run directory's `artifacts/metrics.json` and `resolved_config.yaml`
2. inspect `run_verdict.status`, `W1`, `TMV`, `completed_epochs`,
   `training_timeout`, configured `epochs`, and whether the final schedule was
   shortened relative to the package/default config
3. if the tool output was truncated, ignore the truncated snippet and use the
   saved files
4. decide whether the run is acceptable
5. if unacceptable, do not continue as if training is finished

If `run_verdict.status == "rejected"`, immediately label the run
`failed_needs_retrain` and retry/tune before any downstream deliverable. Do not
reinterpret a rejected verdict as success just because model artifacts,
deliverable files, or verifier logs exist.

Write the decision in workflow state or notes before downstream analysis. Use
these labels:

- `accepted_for_downstream`: metrics are plausible for the data scale and the
  training schedule is justified.
- `debug_only_undertrained`: the run completed but used a shortened schedule
  without convergence evidence, has unstable loss spikes, timed out, or has
  clearly poor W1/TMV. Do not use it for downstream deliverables except to
  diagnose the next training attempt.
- `failed_needs_retrain`: the run has NaN/inf, broken coupling/mass behavior,
  missing model artifacts, or other contract failure.

Acceptance rule:
- `TMV < 0.3` only for algorithms/runs that model unbalanced mass; balanced-only
  methods should not be rejected only because TMV is poor
- `W1` is within the data scale range in latent space
- a final run that reduces builtin/default flow-matching epochs must cite saved
  convergence evidence. "Faster", "benchmark time", or "downstream deadline" is
  not enough.

Escalation order when the rule fails:
1. check for obvious training failure
   - NaN/inf
   - exploding or unstable loss
   - obviously broken coupling / mass behavior
2. try hyperparameter adjustment first
   - prefer learning rate, regularization, coupling settings, and batch/chunk
     size before touching epoch count
   - do not interpret "training schedule" as permission to reduce epochs for
     speed; epoch reductions need dataset/model-specific convergence evidence
3. only then consider algorithm changes if the failure is not plausibly caused by hyperparameters

Do not jump straight to algorithm redesign when a training-quality problem may
still be explained by training configuration.

## When to read more source/docs
- `CytoBridge-main/docs/runtime/custom-algorithms/README.md`
- `CytoBridge-main/docs/runtime/flow-matching/README.md`
- package fit/training source when backend integration details are unclear

## Completion
Call:

`commit_workflow_state(phase="training", ...)`

Use these preferred update shapes:

```python
updates = {
    "final_config": {
        "name": "<config_or_algorithm_name>",
        "path": "<model_artifact_path_or_legacy_trained_model_path>",
        "run_id": "<run_id>",
        "run_dir": "<run_dir>",
        "resolved_config_path": "<resolved_config_path>",
    },
    "final_metrics": {...},
    "training_runs": [...],
}
artifacts = {
    "model_artifact_path": "<model_artifact_path>",
    "trained_model_path": "<legacy_trained_model_h5ad_path_if_present>",
    "metrics_path": "<metrics_path>",
    "resolved_config_path": "<resolved_config_path>",
}
```

Training runs normally save a compact `model_artifact.json` plus `model_state.pt`
instead of copying the full AnnData into every trial. Treat `model_artifact_path`
as the preferred trained-model reference. A full `trained_model.h5ad` is legacy
opt-in and should not be required for ordinary campaign trials.

Do not commit:
- `final_config="crufm"`
- `final_config="ruot"`
- any other bare string as `final_config`

`final_config` must be a mapping because later stages read `final_config["path"]` to find the trained model.
