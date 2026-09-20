---
name: review-and-training
description: Pre-training review entrypoint for custom algorithms. Bind the correct proposal/workspace, check implementation/scalability/inference contracts, then choose manual training or campaign trial.
---

# Algorithm Review And Training

Use this skill after an approved proposal has been implemented or modified and
before trusting custom training evidence.

This is the short entrypoint. For detailed checks, open only the reference file
that matches the immediate blocker.

## Minimal Sequence

1. `get_current_workflow_context()`
2. confirm active algorithm, `proposal_id`, snapshot/dirty state, and campaign
   state
3. re-read local files: `PROPOSAL.md`, `manifest.yaml`, `algorithm.py`,
   `config.yaml`, `IMPLEMENTATION_MAP.md`, and `risk.md`
4. fill/self-check `IMPLEMENTATION_MAP.md`
5. check proposal-to-code consistency, runtime contract, scalability, config,
   and inference/evaluation integrity
6. for any custom claim metric or side-information-dependent claim, complete
   the metric and side-information integrity checks below
7. run `preview_training_run(...)` if code, inference, or metrics changed
8. choose the execution path:
   - campaign tuning: `run_campaign_trial(...)`
   - one-off debug/manual evidence: `run_training(...)`

Do not skip the review just because the code runs.

## Scalability Quick Check

Before trusted training, check the actual hot path:

- no full dense `n_t x n_{t+1}` cost/plan/mask/kernel state for real-scale gaps
- mini-batch/chunking must construct bounded blocks, not chunk after full dense
  materialization
- no reconstruction of chunked `CouplingPlanStore.sub_plans` into one global
  dense plan for real-scale gaps; chunked stores should be sampled or iterated
  as bounded stores
- OT/UOT/WFR solver tensors should stay on GPU when CUDA is available
- valid accelerations include streaming, sparse candidates, landmarks, coresets,
  caching invariant costs, vectorization, and parallel per-gap work

Use the package builtins as implementation references before approving custom
coupling code. Read
`CytoBridge-main/docs/runtime/flow-matching/builtin-implementation-guide.md`,
then inspect the closest builtin YAML and source symbols it lists. Compare the
custom code against the closest builtin's module responsibilities, not only its
final metrics.

During a custom algorithm lifecycle, tune the candidate algorithm, not builtin
baselines. Builtins are fixed comparators by default; refresh or tune them only
when baseline evidence is missing/stale or a builtin run has concrete abnormal
evidence. If an agent-proposed OT/UOT/WFR method has abnormally poor W1/TMV,
mass collapse, NaNs, or a large unexplained baseline gap, first inspect coupling
quality and resolved config: cost scale, solver mass/convergence, row/column
coverage, plan sparsity or collapse, chunking, and setup-versus-epoch placement.
Important builtin knobs, when builtin tuning is justified, include VGFM/CRUFM
`reg` and `reg_m`, regularization strategy, and WFR-FM `delta` or auto-delta
settings.

## Hard Rules

- Review must bind to the current exact `proposal_id` and workspace snapshot.
- The implementation must match the approved proposal. If semantics changed,
  patch `PROPOSAL.md` with `apply_workspace_patch(...)` and wait for the new
  proposal version to be reviewed before using new training evidence.
- `IMPLEMENTATION_MAP.md` is mandatory. Missing/TODO/pending/deferred/unmapped
  rows block trusted training when they correspond to proposal-required
  components.
- Campaign path does not need manual snapshot; `run_campaign_trial(...)` creates
  the trial snapshot and archive commit.
- Manual `run_training(...)` does need a clean target context. If dirty after
  intended edits, call `snapshot_active_algorithm_workspace(...)` first.
- Persistent config changes should modify the real workspace `config.yaml`
  rather than temporary `config_overrides`, so campaign promote/reject archives
  the actual config. Prefer `apply_workspace_patch(...)` for normal config
  edits. `patch_algorithm_config(...)` is optional for simple scalar leaf paths
  such as `training.plan[0].lr` and protects against accidental whole-list
  replacement.
- Temporary `config_overrides` passed to `run_campaign_trial(...)` affect that
  trial only. They are not frozen into the stage data panel. If the trial
  promotes, the promoted resolved config is synced into workspace
  `config.yaml`; if it rejects, the workspace is restored to the active best.
- A custom algorithm meant for real biological data must be scalable beyond toy
  data. Obvious full dense all-pairs hot paths are blocking implementation
  issues.
- Metrics must be computed from the trained dynamics rollout /
  `EvaluationTrajectory`, not future snapshot lookup, post-hoc correction, or
  metric-specific shortcut prediction.
- A custom claim metric must not be trusted merely because it is in range or
  high. If the claim metric is high while mechanistic diagnostics for the
  claimed object are random, zero, missing, or contradictory, block trusted
  training evidence and run a diagnosis before further tuning.
- When a method uses side information (lineage/barcode, labels, spatial fields,
  perturbations, conditions, batches, auxiliary modalities), review the actual
  target dataset structure before trusted training: coverage, cardinality,
  imbalance, missingness, cross-time comparability, and interval-specific
  reliability. The implementation's representation of that side information
  must remain valid at the observed scale.
- For algorithms that claim unbalanced mass, predicted mass/cell-count changes
  must come from the approved growth/dynamics mechanism during rollout. A
  target-count lookup, post-hoc weight rescale, time-only count-ratio clock, or
  replacement of learned growth by a separate correction is a blocking issue
  unless the approved proposal explicitly defines and biologically justifies
  that mechanism.
- If an automatic implementation or inference evaluator is requested, wait for
  it and fix/revise before retrying.
- Nearest-builtin degenerations, disabled-component ablations, and other
  diagnostic controls are not trusted campaign candidates. Run them only with
  manual `run_training(...)` or `preview_training_run(...)`, and do not let their
  metrics update active best or stage-gate evidence. If such a control is the
  best-performing configuration, first revise and review the proposal before
  entering it into campaign selection.

## Custom Claim Metric Validity Gate

Before trusting a run whose stage gate or paper claim depends on a custom claim
metric, verify the metric against the standard rollout output:

- `IMPLEMENTATION_MAP.md` must contain a `Claim Metric Contract` that maps the
  proposal claim object to prediction source, truth/observed source, grouping
  key, timepoints, label source, evidence for biological truth semantics,
  leakage controls, and code lines;
- the contract must also name the original motivating gap or real-data failure
  mode, and explain why passing the metric demonstrates that this gap was
  solved rather than avoided;
- do not treat `cell_type`, `label`, `lineage`, `barcode`, time bins, or
  category counts as proof of the metric's truth semantics. These only prove
  fields exist. If the metric depends on terminal fate, descendant fate,
  lineage-truth endpoints, fate-concordant transitions, growth truth, or
  perturbation truth, verify that this semantics is explicitly documented in
  the approved proposal, implementation map, benchmark card, dataset
  registration, preprocessing contract, or dataset documentation;
- if an automatic `inference_evaluator` is required for the custom metric, wait
  for it and treat a metric semantic mismatch as blocking trusted campaign
  evidence;
- if the metric has been downgraded from the original claim to an easier proxy,
  block trusted evidence until the proposal reviewer approves that change and
  the new metric still directly tests the original problem. Do not use a proxy
  that can improve while lineage/fate transitions, mass/growth recovery,
  perturbation response, condition-specific dynamics, or the proposal's named
  biological target remains unsolved;
- random and collapsed predictors should not receive a strong score;
- a relevant builtin or expression-only baseline score should be known or
  explicitly reported missing before using the custom score comparatively;
- the metric should decompose into mechanistic submetrics tied to the claim;
- the top-level score must agree in direction with at least one mechanistic
  diagnostic, such as retrieval, separation AUC, diversity/coverage, fate
  distribution concordance, mass/growth agreement, or another proposal-defined
  observable;
- for high-cardinality categorical objects, test whether pairwise same/different
  statistics become degenerate because most random pairs are different;
- if missing labels, unshared categories, censored observations, or partial
  modalities exist, state whether absence is treated as unknown or negative.

If any check fails, do not repair the metric path post hoc. Write a diagnosis,
then either revise the metric/evaluator under the approved proposal semantics or
patch/review the proposal if the claimed evidence target changed.

## Side-Information Data Audit Gate

For side-information-dependent algorithms, audit the target dataset before
launching trusted evidence on a new real benchmark:

- valid-cell coverage and missingness by time point;
- category/unit cardinality, median group size, singleton/rare-group count, and
  top-k concentration;
- adjacent-time overlap/comparability or another justification that the side
  information can supervise temporal dynamics;
- interval-specific reliability; do not assume one global side-information
  weight is valid across all gaps;
- representation compatibility, such as embedding/hash dimension versus category
  count, sparse-candidate support versus dense all-pairs assumptions, or
  modality normalization scale;
- whether side information is available at inference time or only training-time
  supervision, and how the learned dynamics carries the signal into rollout.

If the audit reveals severe mismatch, switch to diagnosis or proposal revision
before campaign tuning. A successful small simulation does not waive this gate
for real biological data.

The audit must be performed before the first trusted real-data trial whose claim
depends on that side information, and again whenever the benchmark panel changes
to a dataset with different metadata structure. If the audit reveals low
adjacent-time overlap, strong category imbalance, high missingness, or a scale
mismatch between side-information cost and latent-expression cost, the next step
is not blind hyperparameter tuning. Choose one of: add an approved
coverage-aware fallback, redesign the claim metric, add a controlled
claim-validation benchmark, or patch/review the proposal.

Record metric contradictions explicitly. A run whose top-level claim metric
passes while required mechanistic diagnostics are chance-level or zero may remain
an engineering artifact, but it must not be used as scientific validation, stage
claim evidence, or paper evidence until a diagnosis artifact resolves the
contradiction.

## What To Read Next

- Binding, files, proposal consistency, implementation map:
  `references/pretraining-checklist.md`
- Scalability, runtime contract, mass semantics, config checks:
  `references/runtime-and-scalability-checks.md`
- Preview, manual training, campaign execution, validation, evidence hygiene:
  `references/training-validation-and-evidence.md`
- Campaign panel/stage rules:
  `../campaign-tuning/references/panels-and-stage-gates.md`
- Tuning diagnosis after plateau:
  `../tuning-playbook/SKILL.md`

## Execution Choice

Use campaign tools for iterative tuning and automatic promote/reject:

- `start_algorithm_campaign(...)`
- `run_campaign_trial(...)`
- `check_campaign_stage_gate(...)`

Use `start_campaign_trial(...)` / `resume_rejected_trial(...)` only when you
want to restore a specific base before editing. Normal tuning can edit the
current workspace/config and call `run_campaign_trial(...)` directly.

Use direct `run_training(...)` only for one-off validation, compatibility
checks, debugging on a dataset outside the frozen campaign panel, or
nearest-builtin degeneration / component-ablation diagnostics. Direct manual
runs do not update campaign active best and do not count as stage-gate evidence.
If the custom algorithm lifecycle status is `developing`, a direct manual run is
not a completed algorithm result; use only a campaign `final_regression` locked
release for user-facing reporting or downstream analysis.

## Failure Discipline

If review finds any of these, do not train as trusted evidence:

- proposal/implementation mismatch
- missing or inaccurate implementation map
- non-scalable hot path for the intended data scale
- broken runtime data contract
- config/algorithm inconsistency
- inference or metric code that bypasses dynamics rollout
- mass/growth semantics that cannot be justified

Fix the code, patch/review the proposal, or use manual debug runs explicitly
marked as non-campaign evidence.
