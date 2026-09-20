---
name: algorithm-orchestrator
description: Top-level operating manual for CytoBridge custom algorithm design, workspace authoring, benchmark selection, and campaign tuning.
---

# Algorithm Orchestrator

Use this skill first for any custom algorithm design, implementation, or tuning
task. It is the routing layer. Read only the next stage skill you need.

This entry is intentionally short. If a decision depends on exact lifecycle,
proposal, workspace, benchmark, or campaign policy details, read
`references/orchestrator-detailed-rules.md` after this entry.

## One-Screen Workflow

1. Define or confirm the scientific problem.
2. Write or revise the algorithm proposal and let the evaluator approve it.
3. Initialize or activate the algorithm workspace.
4. Author the smallest complete implementation matching every required approved
   proposal component.
5. Fill `IMPLEMENTATION_MAP.md`, then run review and preview.
6. Tune with campaign tools on a frozen benchmark panel.
7. Use stage gates for promotion to the next lifecycle stage.
8. Run final regression to lock the completed algorithm version.
9. Run downstream biological/model analysis from the locked final-regression
   release before writing a report or paper. For biological-application work,
   first read `~/.cellcompass/skills/workflow/biological-story-building/SKILL.md`
   to decide the dataset story and evidence gap. Then read
   `~/.cellcompass/skills/workflow/downstream-analysis/SKILL.md`, read the
   narrow downstream skill(s) that match the claim, and save script-backed
   downstream manifests, tables, figures, and warnings.
10. Write the final report or paper from final-regression and downstream
    artifacts, not transient previews, unpromoted trials, or unsupported prose.

Do not jump from an idea or fresh code directly to `run_training(...)` or
`run_campaign_trial(...)`. Do not report completion from a promoted trial alone:
only final-regression locked release is the completed algorithm result.
Implementation alone is not delivery. A custom algorithm is usable only after
the campaign passes Stage 1, Stage 2, Stage 3, and final regression locks a
release so the registry lifecycle status is `complete`. If this lifecycle has
not happened, call the result `developing`, `blocked`, or `failed` with evidence,
not a finished algorithm.

Avoid duplicate algorithm ideas. Before writing or revising a proposal, inspect
the relevant builtin methods, current algorithm catalog/history, and existing
proposal assets. Do not rename a builtin or repeat a prior CytoBridge-designed
algorithm. Reusing an idea is acceptable only when the proposal states a clear
deeper contribution, new data regime, stronger claim metric, stronger theory, or
other material improvement that the campaign can distinguish empirically.

Lifecycle status is operational: approved proposal or active campaign means
`developing`; final regression locked release means `complete`;
`mark_algorithm_failed(...)` means this direction cannot satisfy the user goal.
A failed algorithm is not a completed answer; repair, revise, or replace it.

## Golden Path Discipline

When unsure, follow this exact path:

1. `get_current_workflow_context()`
2. read this orchestrator
3. read only the next stage skill
4. use the dedicated tool for that stage
5. inspect the returned state before choosing the next action

Default next-stage mapping:

- no approved proposal: read `proposal-theory`
- proposal approved but no workspace: initialize workspace
- workspace exists and code/config must change: read `authoring`
- implementation changed: fill `IMPLEMENTATION_MAP.md`, then read
  `review-and-training`
- normal tuning: read `campaign-tuning`
- plateau or surprising failures: read `tuning-playbook`
- benchmark data missing or invalid: read `benchmark-dataset-registration`
- final regression locked but downstream evidence is missing: read
  `workflow/downstream-analysis`
- downstream evidence exists and a manuscript/report is needed: read
  `workflow/paper-authoring` or `workflow/report-authoring`

Do not read all stage skills up front.

## Evidence-First Exception

If a run has produced surprising or contradictory evidence, do not interpret
"continue" as "launch another trial". First make the smallest diagnosis artifact
that can localize the failure.

Isolate whether the issue is in the OT/UOT/WFR coupling, conditional path,
training convergence, inference rollout, mass dynamics, evaluator/metric, data
contract, or proposal theory. If the failed component is a proposal assumption,
do targeted theory/literature reading for that exact issue and revise the
proposal instead of pushing more trials.

For side-information algorithms, audit the data before claiming success:
coverage, missingness, category cardinality, imbalance, adjacent-time
comparability, and fallback behavior must be explicit.

## Core Mental Model

Keep these objects separate:

- proposal: theory contract and pseudocode, versioned by exact `proposal_id`
- active workspace: the algorithm you are currently editing
- training target: the algorithm explicitly passed to training
- campaign: trial archive, automatic `promote`/`reject`, stage state
- benchmark catalog: prepared datasets and builtin/reference W1/TMV baselines

When any binding is unclear, call `get_current_workflow_context()` first.

## Non-Negotiable Rules

- The proposal must stand on its own before code: problem, claimed capability,
  recoverability, inductive rollout, mass scope, pseudocode, evaluation plan,
  and literature/package grounding for genuinely new methods.
- Claim metrics must be meaningful observables of the stated problem. If real
  benchmarks cannot observe the claimed structure, design a controlled Stage 2
  simulation with frozen generator/source, comparable baselines, and
  claim-absent/shuffled controls. For BoolODE-style synthetic mechanisms, read
  `boolode-stage2-simulation`.
- Custom algorithms must be dynamics models that roll a valid t0 population
  forward; do not rely on training-cell ids, memorized OT rows, target-specific
  lookup/correction, future snapshots, or direct metric prediction.
- Semantic changes to solver, coupling, path, mass, loss, or evaluation require
  proposal revision and review before new evidence is trusted.
- Use campaign tools for normal tuning. Do not hand-label decisions and do not
  casually restart trials.
- Trial `promote`/`reject` compares against the algorithm's own active best;
  stage gates compare against external builtin/reference baselines.
- TMV is hard only for algorithms whose approved proposal models unbalanced
  mass.
- For real biological algorithms, include real biological benchmarks as early as
  feasible; simulation success alone is not full evidence for a real biological
  claim.

## Stage Skills And Docs

Read the narrowest next-stage skill:

- Package docs map: `CytoBridge-main/docs/INDEX.md`
- Builtin theory map: `CytoBridge-main/docs/theory/README.md`
- Flow-matching API semantics: `CytoBridge-main/docs/runtime/flow-matching/README.md`
- Builtin flow-matching source map:
  `CytoBridge-main/docs/runtime/flow-matching/builtin-implementation-guide.md`
- Custom workspace contract: `CytoBridge-main/docs/runtime/custom-algorithms/README.md`
- Proposal theory: `~/.cellcompass/skills/algorithm/proposal-theory/SKILL.md`
- Authoring code: `~/.cellcompass/skills/algorithm/authoring/SKILL.md`
- Review/training: `~/.cellcompass/skills/algorithm/review-and-training/SKILL.md`
- Campaign tuning: `~/.cellcompass/skills/algorithm/campaign-tuning/SKILL.md`
- Plateau diagnosis: `~/.cellcompass/skills/algorithm/tuning-playbook/SKILL.md`
- Benchmark registration: `~/.cellcompass/skills/algorithm/benchmark-dataset-registration/SKILL.md`
- BoolODE Stage 2 simulations:
  `~/.cellcompass/skills/algorithm/boolode-stage2-simulation/SKILL.md`
- Dataset preprocessing: `~/.cellcompass/skills/workflow/preprocessing-execution/SKILL.md`
- Downstream biological/model analysis after locked release:
  `~/.cellcompass/skills/workflow/downstream-analysis/SKILL.md`
- Paper authoring from final-regression and downstream artifacts:
  `~/.cellcompass/skills/workflow/paper-authoring/SKILL.md`

## Locked-Release Downstream Handoff

Final regression is not the end of a biological algorithm lifecycle. It locks
the version that downstream analysis must interpret. For biological-application
algorithms, do not jump straight from `final_regression` to paper/report prose.
It says the algorithm has passed the metric-level lifecycle, not that the
biological story is complete. Do not weaken the quantitative bar: a high-quality
empirical paper should have both competitive benchmark/claim metrics and a
strong biological interpretation. It should show what the new algorithm reveals
about the system, mechanism, perturbation, fate, growth, or gene program, and
why that evidence is scientifically meaningful, not only that a benchmark
number improved. First run downstream analysis from the locked model, resolved
config, and evaluation trajectory:

- read `~/.cellcompass/skills/workflow/biological-story-building/SKILL.md` to
  decide the dataset story, algorithm-enabled insight, and downstream evidence
  gap;
- read `~/.cellcompass/skills/workflow/downstream-analysis/SKILL.md`;
- choose narrow downstream skills by the claim, such as trajectory/fate,
  growth/mass, perturbation, lineage transition, prediction, state structure,
  or driver genes;
- for gene-mechanism, regulator, pathway, or biology-mechanism claims, read
  `~/.cellcompass/skills/downstream/downstream-driver-genes/SKILL.md` and
  produce gene-space or documented projection artifacts when measured genes or
  PCA loadings are available;
- if gene-space projection is unavailable or unstable, record a downstream
  manifest warning such as `gene_space_unavailable` and avoid gene-mechanism
  claims in the paper;
- save reusable code under `<output_dir>/scripts/downstream/*.py` and outputs
  under the algorithm output directory, with a manifest that names the locked
  model/evaluation trajectory, feature space, projection backend, warnings,
  tables, figures, and supported claims.

Paper/report authoring should consume these downstream artifacts. It should not
retrofit a biological story from metrics alone or treat metric-level completion
as biological validation.

## When To Read The Detailed Reference

Read `references/orchestrator-detailed-rules.md` when:

- writing or revising a proposal with new scientific semantics;
- unsure whether a code/config change requires proposal revision;
- active workspace, training target, registry, or campaign binding is confusing;
- choosing benchmark panels, simulations, or claim metric evidence;
- interpreting promote/reject versus stage-gate versus final-regression status;
- planning paper/report claims from campaign artifacts.

## Next-Step Selector

- No stable research problem: read `research-question-framing`.
- Concrete algorithm idea but no approved proposal: read `proposal-theory`.
- Approved proposal but no workspace: initialize workspace.
- Workspace exists and code must change: read `authoring`.
- Code/config changed: read `review-and-training`, then preview/review.
- Tuning a custom algorithm: read `campaign-tuning`.
- Campaign has plateaued or results are surprising: read `tuning-playbook`.
- Adding a benchmark dataset: preprocess first, then read
  `benchmark-dataset-registration`.
- Creating a controlled Stage 2 synthetic mechanism test: read
  `boolode-stage2-simulation`, then use the closed-loop generation/registration
  tool or register the frozen custom dataset.
