# Tuning Diagnosis Details

This file preserves the full detailed tuning playbook. Read it when a campaign stalls, active best remains far from builtin/reference baselines, custom claim metrics contradict mechanism diagnostics, runtime/scalability anomalies appear, or a concrete diagnosis artifact is needed before more campaign trials.

# Tuning Playbook

Use this skill when a campaign has results but the next improvement is unclear.
It is a diagnosis workflow, not a generic package API manual.

## Goal

Improve an approved algorithm with the fewest justified experiments while preserving:

- approved proposal semantics
- active best lineage
- campaign `promote` / `reject` discipline
- reproducible evidence
- clear stopping decisions

## Operating Model

Campaign tuning has three layers:

1. local hyperparameter/config search around the current stage `active_best`
2. risk-driven diagnosis using `risk.md`, the trained `active_best` model, builtin/reference baseline gaps, and targeted metrics
3. proposal revision only when diagnostics show a real semantic/modeling failure

Do not jump to algorithm redesign just because W1 or the claim metric plateaus.
Do not keep burning trials on blind hyperparameter search after repeated
non-progress.
When metrics remain very poor after reasonable tuning, return to first
principles: inspect whether the approved mathematical problem, biological
mechanism, data assumptions, and inference rule are actually capable of solving
the benchmark. Do not try to rescue bad evidence with target-count repair,
post-hoc rescaling, metric-specific shortcuts, or other corrections whose only
purpose is to force W1/TMV/claim metrics to pass.
Reading `risk.md` is not a diagnosis by itself. After reading a risk, write or
run the smallest code-based check that can show whether that risk is actually
active in the current trained model, rollout, data panel, or logs.

Treat the nearest builtin algorithm as the engineering anchor. Custom methods
are usually built on a tuned builtin runtime: the builtin config, epochs, solver
placement, chunking strategy, and sampler lifecycle already have acceptable
W1/TMV and scalability on the registered benchmarks. If the custom method is
far slower than the nearest builtin, times out, or has much worse W1/TMV, do
not first assume the proposal is impossible. Check which changed component
relative to the builtin broke the behavior. If a component can naturally
degenerate back to the builtin, run that ablation as a diagnostic control with
manual `run_training(...)` or `preview_training_run(...)`, not
`run_campaign_trial(...)`. Degenerate controls can explain failures and compare
runtime/metrics against the builtin, but they must not enter campaign
promote/reject or become active best unless the proposal is formally revised and
reviewed to make that simplified method the actual algorithm. If no natural
degeneration exists, compare the component's direct statistics against the
builtin and the highest-severity `risk.md` items.

For flow-matching methods that model unbalanced mass, large TMV failures usually
come from the transport-derived supervision signal before they come from scalar
learning-rate tuning. Prioritize coupling diagnostics first: cost scale, UOT/WFR
regularization, row/column mass, solver residuals, chunk equivalence, terminal
mass targets, and whether the sampled endpoint/mass pairs match the intended
mass semantics. Only after coupling evidence looks sane should you spend many
trials on path, model capacity, or generic optimizer sweeps.

If preview inspection times out before epoch logs appear, treat it as a
coupling setup problem first. For a custom cost, compare one representative
cost block to the nearest builtin: min/median/max, finite ratio, device, solve
time, and whether auto-regularization is repeatedly searching unstable
`reg/reg_m`. Prefer cost normalization or fixed regularization once the scale is
understood; do not respond only by lowering epochs or shrinking model capacity.

## Required Context

Before tuning, inspect:

- `get_current_workflow_context()`
- `get_algorithm_campaign_status(campaign_id=...)`
- `list_campaign_trials(campaign_id=..., decision="")`
- `~/.cellcompass/training_algorithms/<algorithm_id>/PROPOSAL.md`
- `~/.cellcompass/training_algorithms/<algorithm_id>/risk.md`
- `~/.cellcompass/training_algorithms/<algorithm_id>/IMPLEMENTATION_MAP.md`

`risk.md` is written by the proposal reviewer and should be ordered by severity,
highest first. Treat it as a diagnosis queue, not as proof that the proposal is
wrong.

## Required Tuning Diagnosis Table

When progress stalls, create or update a compact tuning diagnosis table before
launching more trusted campaign trials. This table can live in
`diagnostics/tuning_diagnosis.md`, `IMPLEMENTATION_MAP.md`, or a campaign note,
but it should be an explicit artifact the next turn can read.

Part A: nearest-builtin degeneration / ablation comparison.

| Row | Question | Evidence to fill |
| --- | --- | --- |
| Nearest builtin | Which builtin is the engineering and semantic anchor? Examples: `vgfm`, `wfrfm`, `balanced_ot_cfm`, `sf2m`, `crufm`. |
| Shared runtime path | Which package classes/functions are reused or subclassed? Name the exact coupling/path/backend/loss symbols. |
| Degeneration setting | What config/code setting makes the custom method behave like the nearest builtin, if such a setting is natural? This is a debugging control to run with manual `run_training(...)` or preview, not campaign evidence. |
| Degeneration efficiency | Does runtime/memory resemble the builtin pattern? Check no repeated full OT/UOT/WFR solve in the epoch/batch hot path. |
| Degeneration metrics | Under the degeneration setting, are W1/TMV comparable to the builtin on the same dataset panel? |
| Builtin delta | Which exact component differs from the nearest builtin: coupling/cost, solver regularization, conditional path, mass target, loss, inference, or metric? |
| Component diagnostics | If the complete implementation fails, which already-implemented proposal components will be neutralized or ablated for diagnosis: coupling/cost, path, mass, loss, inference, custom metric? |
| First failing component | Which component first breaks W1/TMV/runtime/claim metric relative to the previous row? |

Part B: reviewer-risk validation.

| Risk | Severity | Hypothesis | Minimal diagnostic | Evidence path | Verdict |
| --- | --- | --- | --- | --- | --- |
| `<risk.md item>` | High/Medium/Low | Concrete failure pattern | One script/plot/statistic | `diagnostics/...` | active / inactive / unresolved |

This table does not permit a simplified implementation. The implementation under
review must already realize the approved proposal. Degeneration controls and
component ablations are debugging switches inside or around the complete
implementation, not a staged plan to implement proposal components later. If a
degeneration setting is unnatural or would distort the method, state
`not applicable` and compare against the closest builtin through component-level
diagnostics instead.

Do not run campaign trials for degeneration controls or component-neutralized
ablations. Campaign trials are for the current formal candidate whose
proposal-required components are enabled. If a degenerate diagnostic produces
better W1/TMV, record it as evidence about the broken component, then either fix
the full method or revise/review the proposal before allowing that simpler
configuration into campaign selection.

## Step 1: Define Progress

Before changing anything, decide what counts as progress in the current stage.

Use the campaign policy:

- Stage 1: primary metric is W1, with TMV as a hard gate only if the algorithm models unbalanced mass
- Stage 2: primary metric is the frozen claim metric; the gate uses the strongest comparable builtin/reference baseline and requires 10 percent improvement, while W1 is a scale-aware guardrail. Low-scale simulation can allow up to 1.5x baseline W1, but higher-W1 real-data panels are tightened automatically.
- Stage 3: primary metric returns to W1; the inherited Stage 2 data panel carries the Stage 2 claim metric guardrail. Its optional early-completion gate expects W1 and the validated claim metric to be non-worse than the strongest comparable external baseline. Claim-metric regression is capped at 5 percent against the fixed Stage 2 active best that passed claim validation, not against a drifting Stage 3 active best.

Non-progress means one or more of:

- no `promote` after a small local neighborhood search
- primary metric changes are below `min_delta`
- active best remains far from the relevant builtin/reference baseline required by the stage gate
- secondary metric repeatedly violates tolerance
- the same failure pattern repeats across 3 or more nearby trials
- runtime/timeout/memory failures dominate
- preview or evaluation metrics show impossible values
- qualitative diagnostics disagree with aggregate metrics

For runtime failures, use actual run metadata rather than progress-log
timestamps. Record completed epochs, elapsed seconds, budget seconds, per-epoch
time, build/setup time if available, timeout flag, memory/OOM status, and the
nearest builtin's same-dataset timing. If the custom method is much slower than
the builtin, inspect the epoch hot path before tuning scalar hyperparameters.
The usual boundary is `build_state` or setup versus `sample_pairs`, loss, and
rollout: expensive solvers and invariant cost/state construction should not be
repeated every epoch unless that online solve is central to the approved method.
If a custom pairwise cost or solver is the bottleneck, check whether the nearest
builtin keeps equivalent cost/solver tensors on GPU and caches bounded coupling
metadata before reducing `chunk_size`. A smaller chunk can diagnose coupling
fragmentation, but it is not a valid final scalability conclusion until a
GPU/vectorized, builtin-lifecycle version has been considered.

## Step 2: Hyperparameter-First Local Search

Start with low-risk changes around the current `active_best`.

Low-risk knobs:

- learning rate and scheduler
- batch size / chunk size
- epochs or early-stop patience only when convergence evidence supports the change
- regularization strength
- solver tolerance / iteration caps
- stochastic noise scale when it is already part of the approved proposal
- coupling or mass penalty parameters already present in the proposal
- per-dataset config values for final or benchmark-specific tuning
- non-semantic runtime speedups and caching

Rules:

- change one hypothesis at a time when possible
- if a knob helps, explore a small neighborhood around it
- if 2-3 nearby settings worsen or do not move the primary metric, stop that direction
- keep dataset-specific config separate and archived through campaign tools
- use `preview_training_run(...)` before spending real trials when code, inference, or metrics changed
- use `run_campaign_trial(...)`; do not hand-label decisions

Do not count warm-start-only speedups as evidence that scratch training improved.

Epoch and warm-start guardrails:

- builtin flow-matching methods use 3000 epochs; use that as the starting point
  for custom flow-matching configs
- do not lower epochs only to pass the time budget
- a shorter schedule is acceptable only after evidence shows comparable
  convergence and metrics from scratch training
- do not use fine-tuning from an already trained model, cached fitted state, or
  target-specific precomputed predictions as campaign evidence
- when runtime blocks progress, prioritize semantics-preserving acceleration:
  mini-batch/chunking, GPU/vectorization, caching repeated coupling/cost
  structures, and parallelizing independent per-gap work

## Step 3: Detect Plateau And Switch To Diagnosis

Switch from hyperparameter search to diagnosis when:

- recent trials are mostly `reject`
- the best primary metric is flat despite reasonable local tuning
- the active best is still far from the relevant builtin/reference baseline
- failures cluster in one time point, dataset, or subgroup
- TMV/claim metric/W1 trade off against each other in a stable pattern
- timeouts or memory failures persist after simple budget-aware adjustments
- a custom primary metric is more than about 20% worse than the relevant
  same-panel builtin/reference baseline, unless the stage policy explicitly
  defines a looser exploratory tolerance
- a custom claim metric is high while mechanistic diagnostics for the claimed
  biological object are random, zero, missing, or contradictory
- an aggregate promoted trial hides a per-dataset failure on a biologically
  central or real-data benchmark
- a new real benchmark introduces side information used by the algorithm
  (lineage, barcode, spatial coordinates, perturbation, condition, batch,
  multimodal features, etc.) whose coverage, cardinality, imbalance, and
  cross-time comparability have not been audited
- a custom claim metric has not been sanity-checked against random,
  collapsed, and relevant builtin/expression-only predictors

At this point, stop guessing. Use `risk.md`. A plateau is a diagnosis trigger,
not permission to keep sweeping random hyperparameters. Before the next
substantive change, inspect the active-best run and make visual evidence for the
failure mode.

If the active-best metrics are far worse than builtin/reference baselines, first
ask whether the algorithm's first principles are wrong:

- what exactly changed relative to the nearest builtin that already works on
  the same panel?
- if the custom method can be degenerated to the builtin, does that setting
  recover builtin-like runtime and W1/TMV?
- if degeneration cannot be expressed cleanly, do direct component diagnostics
  show that coupling, path, mass target, loss, or inference is the first failing
  boundary?
- does the proposed dynamic objective actually imply the desired distribution or
  mass evolution?
- does the growth/mass mechanism have biological or mathematical meaning, or is
  it only a correction layer?
- are the data assumptions satisfied by the current benchmark panel?
- does the inference rule use only legal t0/context information and truly roll
  the dynamics forward?
- would a stronger implementation of the same proposal plausibly fix the gap, or
  is proposal revision the honest next step?

Do this analysis before adding metric repairs. A bad W1/TMV/claim score is
evidence about the model, not an invitation to manipulate the metric path.

### Mandatory Failure-Analysis Gate Before Further Tuning

When any anomaly trigger above fires, stop blind tuning. This is a hard gate:
before editing `config.yaml`, `algorithm.py`, `claim_metric.py`, or
`PROPOSAL.md`, and before launching another trusted campaign trial, create a
read-only diagnosis artifact. A stop-hook or lifecycle reminder to "continue"
does not mean "continue training"; concrete progress may be diagnosis.

The diagnosis artifact must include, as applicable:

1. same-panel builtin/reference baseline comparison, not only self-best deltas;
2. per-timepoint and per-dataset W1/TMV/claim decomposition;
3. rollout endpoint geometry checks, such as centroid shift, dispersion,
   nearest-neighbor distances, mode coverage, and particle-weight concentration;
4. side-information audit for modalities used by the claim: coverage,
   cardinality, imbalance/top-k concentration, missingness, adjacent-time
   overlap or comparability, and interval-specific reliability;
5. implementation-to-data assumption audit: embedding/hash dimension versus
   category count, dense/sparse hot paths, fallback behavior, loss/cost scale,
   and whether training-time side information actually affects inference-time
   rollout dynamics;
6. custom metric integrity audit: random predictor score, collapsed predictor
   score, expression-only/builtin baseline score, decomposition into mechanistic
   submetrics, and checks for high-cardinality label artifacts;
7. a decision: local config/optimization tuning, implementation bug fix,
   metric revision, or proposal-semantics revision.

Forbidden until this artifact exists:

- patching `PROPOSAL.md` to justify an unverified rescue idea;
- editing `algorithm.py` or `claim_metric.py` to chase the metric;
- patching `config.yaml` except for diagnostics-only logging or inspection;
- launching another campaign trial whose only purpose is to see whether a
  poorly understood anomaly disappears.
- launching a campaign trial for a nearest-builtin degeneration or
  component-neutralized ablation. Use manual `run_training(...)` or preview for
  those diagnostics.

Treat campaign `promote` as an optimization controller decision, not scientific
validation. If a promoted trial fails the diagnosis gate, keep it as an
engineering active best if needed, but do not cite it as evidence for the
biological or methodological claim until the contradiction is resolved.

### Component-Level Diagnosis Without Retraining

Start by isolating one component at a time using existing active-best artifacts,
logs, metrics, checkpoints, and saved rollout trajectories. A useful diagnosis
does not need to spend a new training trial first.

Work through the likely failure boundary:

- data and panel contract: time ordering, representation, side-information
  coverage, missingness, and whether the selected benchmark matches the claim
- OT/UOT/WFR coupling: cost scale, mass penalty, solver residuals, plan sparsity,
  row/column mass, chunking/minibatch equivalence, and whether the coupling
  itself encodes the intended biology or math
- for unbalanced flow matching with high TMV, start here before optimizer
  sweeps: inspect whether the coupling row/column mass and terminal mass targets
  already imply the wrong total mass trajectory
- conditional path / supervision targets: endpoint sampling, path variance,
  growth target construction, time coordinate, and whether the targets implied by
  the coupling are learnable
- optimization/training: loss curves, component losses, best epoch, timeout,
  gradient/NaN behavior, capacity, and whether the trained checkpoint actually
  learned the intended targets
- inference rollout and mass dynamics: rollout geometry, trajectory continuity,
  weight/mass evolution, particle count behavior, and whether metrics are
  computed from the trained dynamics trajectory
- metric/evaluator: per-dataset/per-gap decomposition, random/collapsed/builtin
  sanity checks, and whether a high custom metric contradicts mechanism-level
  evidence
- proposal/theory assumptions: whether the stated mathematical or biological
  problem actually implies the implementation and observed failure pattern

If the evidence points to a proposal/theory problem, stop local tuning. Search
for the exact failed assumption with `search_theory(...)`,
`search_literature(...)`, and direct source reading, then patch `PROPOSAL.md` and
rerun proposal review. Do not continue deeper down a path whose objective,
coupling, path, growth mechanism, or data assumption is likely wrong.

## Step 4: Risk-Driven Diagnosis

Read `risk.md` from top to bottom. For each risk in severity order:

1. State the concrete failure hypothesis.
2. Pick the smallest diagnostic that can support or falsify it.
3. Compare at least:
   - current stage `active_best`
   - the relevant builtin/reference baseline metrics for the same benchmark panel
   - if available, baseline model artifacts or baseline diagnostic outputs
4. Load the `active_best` trained model/run artifacts and inspect the actual behavior.
5. Decide whether the risk is active, inactive, or unresolved.

Keep the diagnosis table synchronized with these checks. If a high-severity risk
is active, do not spend more campaign trials on unrelated scalar
hyperparameters until the risk is fixed, ruled out, or the proposal is revised.

For each high-priority risk, produce executable evidence, not only prose:

- write a small diagnostic script under the algorithm workspace, for example
  `diagnostics/check_<risk_name>.py`, or use an existing saved-run analysis
  script if one already answers the question
- create the diagnostics directory or files with
  `create_algorithm_workspace_artifact(...)`; use relative paths such as
  `diagnostics/check_mass_drift.py`, not absolute paths
- load the active-best run artifacts, saved rollout trajectory, model checkpoint,
  metrics JSON, logs, or benchmark dataset needed for the check
- compute a targeted statistic or visualization tied to the risk hypothesis
- save outputs under the algorithm workspace or run directory, for example
  `diagnostics/<risk_name>_summary.json`, `.csv`, `.png`, or `.svg`
- report the path and the verdict: `active`, `inactive`, or `unresolved`

Good diagnostic code is narrow and disposable. It should answer one failure
hypothesis with runtime evidence. Do not rewrite the training/evaluation logic
or use future-time information to make metrics look better.

For real plateaus, do not rely only on scalar leaderboard rows. Create or inspect
visualizations that expose where the dynamics fail, then tie each visual symptom
back to a risk item, implementation-map component, or config hypothesis.

Rejected trials are optional evidence. Use them only when a specific failed
direction is scientifically informative or when comparing against a rejected
trial can isolate the effect of one change. Do not make loading rejected trials
the default diagnosis path.

Useful diagnostics:

- per-timepoint W1 and TMV
- total weight / mass trajectories
- particle weight histograms and extreme-weight counts
- rollout drift by time interval
- loss curves split by component, if available
- coupling statistics and transport sparsity
- claim metric decomposition
- subgroup/lineage/condition-specific errors
- observed-versus-predicted overlays at each observed time point
- rollout trajectory plots from t0 to final time, including intermediate slices
- side-by-side active-best versus builtin/reference baseline panels
- residual plots for the specific claim metric or biological structure
- mass / weight curves over continuous rollout time for unbalanced algorithms
- runtime, memory, timeout, and last-epoch metadata
- preview warnings and impossible metric ranges
- implementation-map rows related to the failing component

If a risk is active, choose the smallest fix that addresses that risk. Prefer
config/optimization fixes first. If the fix changes solver, path, mass,
coupling, loss, inference, or scientific semantics, patch `PROPOSAL.md` with
`apply_workspace_patch(...)` and wait for proposal review instead of silently
patching `algorithm.py`.

## Step 5: Use Subagents Only When Useful

If diagnosis stalls after checking the top risks, spawn a bounded `general`
subagent for an independent read.

The brief should include:

- algorithm id and campaign id
- current stage and budget remaining
- active best trial id and metrics
- relevant builtin/reference baseline metrics and artifacts, if available
- optional failed trial ids only if they isolate a specific hypothesis
- the relevant `risk.md` section
- paths to `PROPOSAL.md`, `IMPLEMENTATION_MAP.md`, and run directories
- the exact question to answer

Good subagent tasks:

- "Which risk.md item best explains why active best is still far from the baseline?"
- "Compare active best diagnostics with the baseline metrics and identify the failing component."
- "Find whether this is optimization instability or proposal/code semantic mismatch."

Do not spawn a subagent just to continue blind search.

## Step 6: Decide The Next Move

After diagnosis, choose one:

- continue local config search because a knob is moving the metric
- run a targeted ablation because one risk hypothesis is plausible
- resume a rejected trial because it has a promising partial fix
- stop the stage because the gate passed and further tuning is not worth budget
- proceed to final regression after Stage 2/3 criteria are satisfied
- patch/review the proposal because the approved semantics are wrong or incomplete

Stage-specific rule:

- Stage 1 should be cheap and forgiving; do not over-optimize it
- Stage 2 proves the algorithm works; after it passes, avoid semantic edits
- Stage 3 is optional as a failure gate, but not optional as effort. W1 is universal and lower is better; tune actively toward SOTA while preserving the validated claim metric. If the optional SOTA/Pareto gate is not met before budget is exhausted, stop tuning and run final regression rather than treating the algorithm as failed.
- final regression is confirmation-only: inherit the latest active best, confirm the fixed Stage 2 claim floor, lock/report the release, and do not run new tuning/config trials

## Error Categories

Optimization problem:

- signs: unstable loss, seed/batch sensitivity, checkpoint dependence
- first moves: learning rate, scheduler, clipping, batch size, regularization

Capacity problem:

- signs: stable underfit, larger models consistently help
- first moves: width/depth, latent dimension, richer model component
- do not assume this by default

Objective mismatch:

- signs: training loss improves but campaign metric does not
- first moves: selection metric, loss weighting, evaluation fidelity, aligned auxiliary objective

Structural/modeling mismatch:

- signs: repeated plateau, concentrated geometric/time-region errors, complexity does not help
- first moves: risk.md diagnosis, implementation-map audit, proposal revision if semantics must change

Data/preprocessing mismatch:

- signs: contradictory supervision, subset-specific failures, sensitivity to preprocessing
- first moves: confirm data contract, time ordering, representation, benchmark card, balance assumptions

Evaluation/inference mismatch:

- signs: preview passes but real metrics impossible, claim metric behaves oddly, t0-only inference review blocks
- first moves: inspect inference context provenance, metric range checks, model rollout artifacts

## Stop Rules

Stop a direction after:

- 2-3 nearby trials fail to improve
- improvements are proxy-only and do not move the stage primary metric
- runtime or memory cost grows without metric gain
- a fix requires changing approved semantics
- diagnosis shows the suspected risk is inactive

Stop the stage after:

- the gate passes and remaining budget is better saved for the next stage
- Stage 3 reaches budget without improving active best
- final regression completes, locks the release, and the algorithm should be
  marked complete

## Common Mistakes

- treating every plateau as under-capacity
- ignoring `risk.md` until all budget is gone
- reading only global mean metrics
- reading `risk.md` and then guessing without writing/running a diagnostic check
- tuning around a broken claim metric or inference path
- changing too many knobs in one trial
- confusing active best self-improvement with external stage gate success
- continuing bad directions because of sunk cost
- making semantic code changes without patching/reviewing the proposal
- using future-time information or post-hoc correction to improve metrics
- treating a fundamentally weak algorithm as a metric-calibration problem
- adding mass/weight corrections to pass TMV instead of revisiting the growth or
  birth-death mechanism
- forgetting that balanced-only algorithms should not be blocked by TMV

## Minimal Tuning Summary

When handing off or ending a tuning session, write:

- current stage, budget used, and budget remaining
- active best trial id and metrics
- top `risk.md` items checked and verdict for each
- diagnostic code/output paths supporting those verdicts
- hyperparameter/config changes that helped
- changes that failed or were misleading
- likely plateau cause
- next concrete action
