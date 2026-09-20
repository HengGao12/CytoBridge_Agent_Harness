---
name: algorithm-orchestrator
description: Top-level operating manual for CytoBridge custom algorithm design, workspace authoring, benchmark selection, and campaign tuning.
---

# Algorithm Orchestrator

Use this skill first for any custom algorithm design, implementation, or tuning
task. It is the routing layer. Read only the next stage skill you need.

## One-Screen Workflow

1. Define or confirm the scientific problem.
2. Write/revise the algorithm proposal and let the evaluator approve it.
3. Initialize the algorithm workspace.
4. Author the smallest complete implementation that matches every required
   approved-proposal component.
5. Run preview and review before training.
6. Tune with campaign tools on the chosen frozen stage panel.
7. Use stage gates for promotion to the next lifecycle stage.
8. Final regression locks the algorithm version.
9. Run downstream biological/model analysis from the locked release before
   reporting or paper writing.

Do not jump from an idea or fresh code directly to `run_training(...)` or
`run_campaign_trial(...)`.

Do not report the lifecycle as complete until the campaign has explicit
evidence for Stage 1 feasibility, Stage 2 claim validation, Stage 3
tuning/generalization completion or policy exhaustion, and final regression /
locked release. A promoted trial is not the same as a passed stage gate. Only
the final-regression locked release is the completed algorithm result; use that
result for downstream analysis. User-facing summaries, reports, and papers
should then be grounded in both the locked release and its downstream artifacts.

Lifecycle status is operational: approved proposal / active campaign means the
algorithm is still `developing`; final regression locked release means
`complete`; and `mark_algorithm_failed(...)` records that the current direction
cannot satisfy the user goal. A `failed` algorithm is not a completed answer;
repair it, revise the proposal, or design a replacement.

## Golden Path Discipline

When unsure, follow this exact operating path instead of inventing a new one:

1. `get_current_workflow_context()`
2. read this orchestrator
3. read only the next stage skill
4. use the dedicated tool for that stage
5. inspect the returned state before choosing the next action

Evidence-first exception: if a run has already produced surprising or
contradictory evidence, do not interpret "continue" as "launch another trial".
The next valid action can be a diagnosis artifact. This applies when aggregate
metrics disagree with mechanism-level diagnostics, side-information claims fail
on real data, or stage/paper claims depend on metadata whose coverage and
cross-time comparability have not been audited.

Diagnosis can be component-level and reuse existing results: isolate whether the
failure is in the OT/UOT/WFR coupling, conditional path, training convergence,
inference rollout/mass dynamics, metric/evaluator, data contract, or the
proposal's theory. If the failed component is a proposal assumption, do targeted
theory/literature reading for that exact issue and revise the proposal instead
of pushing more trials.

Default next-stage mapping:

- no approved proposal: `proposal-theory`
- proposal approved but no workspace: initialize workspace
- workspace exists and code/config must change: `authoring`
- implementation changed: fill `IMPLEMENTATION_MAP.md`, then
  `review-and-training`
- tuning: `campaign-tuning`
- plateau or surprising failures: `tuning-playbook`
- benchmark data missing or invalid: `benchmark-dataset-registration`
- final regression locked but downstream evidence is missing:
  `workflow/downstream-analysis`
- downstream evidence exists and a manuscript/report is needed:
  `workflow/paper-authoring` or `workflow/report-authoring`

Do not use manual `run_training(...)` as the normal tuning loop. Do not hand
write benchmark paths when `make_benchmark_dataset_config(...)` can build the
payload. Do not choose weak baselines by hand when
`refresh_campaign_stage_baselines(...)` can select the comparable baseline.

For algorithms that depend on side information such as lineage/barcode, spatial
coordinates, perturbation, condition, batch, labels, or paired modalities, make
the data audit a first-class artifact before claiming success: coverage,
missingness, category cardinality, imbalance, adjacent-time comparability, and
fallback behavior must be checked before more tuning or scientific claims.

## Core Mental Model

Keep these concepts separate:

- proposal: theory contract and pseudocode, versioned by exact `proposal_id`
- active workspace: the algorithm you are currently editing
- training target: the algorithm explicitly passed to training
- campaign: trial archive, automatic `promote` / `reject`, stage state
- benchmark catalog: prepared datasets and builtin/reference W1/TMV baselines

When any binding is unclear, call `get_current_workflow_context()` first.

## Stage Skills

Read the narrowest skill for the next action:

- Package docs map: `CytoBridge-main/docs/INDEX.md`
- Builtin algorithm theory map: `CytoBridge-main/docs/theory/README.md`
- Flow-matching API semantics: `CytoBridge-main/docs/runtime/flow-matching/README.md`
- Builtin flow-matching implementation source map:
  `CytoBridge-main/docs/runtime/flow-matching/builtin-implementation-guide.md`
- Custom algorithm workspace contract: `CytoBridge-main/docs/runtime/custom-algorithms/README.md`
- Proposal theory: `~/.cellcompass/skills/algorithm/proposal-theory/SKILL.md`
- Authoring code: `~/.cellcompass/skills/algorithm/authoring/SKILL.md`
- Review/training: `~/.cellcompass/skills/algorithm/review-and-training/SKILL.md`
- Campaign tuning: `~/.cellcompass/skills/algorithm/campaign-tuning/SKILL.md`
- Tuning diagnosis after plateau: `~/.cellcompass/skills/algorithm/tuning-playbook/SKILL.md`
- Benchmark registration: `~/.cellcompass/skills/algorithm/benchmark-dataset-registration/SKILL.md`
- Dataset preprocessing: `~/.cellcompass/skills/workflow/preprocessing-execution/SKILL.md`
- Biological story building before paper downstream:
  `~/.cellcompass/skills/workflow/biological-story-building/SKILL.md`
- Downstream analysis after locked release:
  `~/.cellcompass/skills/workflow/downstream-analysis/SKILL.md`
- Paper authoring from final-regression and downstream evidence:
  `~/.cellcompass/skills/workflow/paper-authoring/SKILL.md`

Do not read all of them up front. Read the next one when the workflow reaches
that stage.

## Locked-Release Downstream Handoff

Final regression locks the algorithm version; it does not by itself create the
biological interpretation needed for a paper. For biological-application
algorithms, run downstream analysis from the locked model, resolved config, and
evaluation trajectory before writing report or manuscript prose. Treat final
regression as metric-level completion only. Biological validation requires
downstream evidence showing what the algorithm reveals biologically and why the
result matters beyond a better benchmark number. This does not relax empirical
metrics: a high-quality biological algorithm paper needs both competitive
benchmark/claim metrics and strong downstream biological interpretation.

Required handoff:

- read `~/.cellcompass/skills/workflow/biological-story-building/SKILL.md` to
  decide the dataset story, algorithm-enabled insight, and downstream evidence
  gap;
- read `~/.cellcompass/skills/workflow/downstream-analysis/SKILL.md`;
- choose the narrow downstream skills that match the claim and dataset, such as
  trajectory/fate, growth/mass, prediction, perturbation, lineage transition,
  state structure, or driver genes;
- for gene, regulator, pathway, or biological mechanism claims, read
  `~/.cellcompass/skills/downstream/downstream-driver-genes/SKILL.md` and
  produce model-derived gene-space or documented projection artifacts when the
  registered data preserves measured genes, PCA loadings, or another valid
  gene-space bridge;
- if gene-space projection is unavailable or unstable, record a downstream
  manifest warning such as `gene_space_unavailable` and avoid gene-mechanism
  claims in the paper;
- save reusable downstream code under `<output_dir>/scripts/downstream/*.py`;
- write a downstream manifest that names the locked model/evaluation trajectory,
  feature space, projection backend, warnings, tables, figures, and supported
  claims.

Paper and report writing consume the downstream manifest and tables. Do not use
final metrics alone as a substitute for downstream biological analysis, and do
not claim a biological contribution until downstream artifacts support it.

## Proposal Rule

Before code exists, the proposal must stand on its own:

- for a genuinely new algorithm, first do proposal research grounding: read the
  local survey synthesis/notes, read builtin algorithm docs, inspect relevant
  builtin configs/implementation paths enough to understand package semantics,
  then use `search_theory(...)` for mathematical foundations and
  `search_literature(...)` for algorithm papers / baseline methods. RAG snippets
  are not enough by themselves. The required pattern is search -> read original
  source -> reason from the source: read the recommended theory-book page ranges
  and the original PDF or important paper sections for the methods that shape
  the proposal, and use `web_search(...)` / `web_fetch(...)` when local sources
  are stale or insufficient. Web snippets also require source reading before
  citation.
- calibrate against the builtin family map in
  `CytoBridge-main/docs/theory/README.md`
- include a `Literature and Package Grounding` section that states which builtin
  algorithms and prior papers were checked, what gap remains, and why the method
  is not duplicating an existing CytoBridge method. For genuinely new
  algorithms, cite at least 10 directly relevant papers or literature notes with
  one-line relevance notes; do not pad the bibliography with unrelated papers.
  If the proposal uses nontrivial OT/SB/WFR/UOT/continuity-equation theory, also
  cite the theory-book page or chapter/section/page ranges found through
  `search_theory(...)`
- include an `Abstract`, `Problem Statement`, optional `Problem Mathematical
  Form`, `Claimed Capability`, and `Expected Evaluation Outcome`
- make clear that the proposal is a dynamics algorithm: it must train a model
  that rolls the t0 population through a continuous trajectory. If it claims to
  solve a theoretical problem, formulate the dynamic problem when possible, such
  as dynamic OT, Schrödinger bridge, WFR/UOT, mean-field dynamics, or a clearly
  defined new dynamic objective
- include an `Inductive Generalization Argument`: the proposed runtime dynamics
  must apply to new valid t=0 cells/particles under the stated data contract,
  not depend on training-cell ids, row ids, memorized OT rows, barcode-specific
  hardcoding, future observed snapshots, or target-specific lookup/correction
- use `Expected Evaluation Outcome` to state the intended working scenario, the
  expected result pattern beyond builtin W1/TMV, and why that evidence validates
  the claim; this should guide later benchmark selection or simulation design
- define any claim metric as a meaningful observable of the algorithm's stated
  problem. If existing real benchmarks cannot observe the claimed structure,
  design a controlled Stage 2 simulation with frozen generator/source,
  comparable baselines, and claim-absent/shuffled controls
- define the mathematical objects
- declare `mass_modeling_scope` as `balanced_only` or
  `models_unbalanced_mass`; this controls whether TMV is a hard gate
- explain exact-fit recoverability of observed time-indexed marginals
- if unbalanced mass is enabled, explain weighted-particle recovery of
  observed cell-count / total-mass changes
- include implementation-oriented pseudocode and an evaluation plan
- keep package hook selection out of proposal approval unless explicitly needed

The proposal evaluator approves/rejects based on theory correctness,
sufficiency, and whether the proposed method can solve its stated claimed
problem. Implementation risks returned by the evaluator are advisory unless
later evidence shows they are real failures.

## Workspace Rule

Active context controls writes:

- call `activate_algorithm_workspace(...)` before editing another algorithm
- read-only tools such as `list_experiment_history(...)` do not change active
  context
- if a write to algorithm B is rejected while active context is A, activate B
  instead of bypassing the guard
- proposal approval automatically activates that algorithm for authoring

Training target controls training:

- `run_training(training_algorithm_id="B", ...)` trains B even if active
  context is A
- campaign trials use the campaign algorithm and create their own snapshots

## Registry Rule

Custom algorithms are registry-backed:

- proposal revisions create immutable `proposal_id`s
- workspace snapshots create immutable `snapshot_id`s
- training runs create immutable `run_id` records
- rejected or invalidated evidence should be marked obsolete

Before moving between proposal, authoring, review, and training, inspect:

- `get_current_workflow_context()`
- `list_experiment_history(...)`
- `check_workflow_gate(...)`

## Semantic Change Rule

If you change solver, coupling, path, mass, loss, or evaluation semantics, do
not silently patch `algorithm.py` and continue.

Use this discipline:

- if the scientific semantics changed, patch `PROPOSAL.md` with `apply_workspace_patch(...)`; the runtime will create a new `proposal_id` and trigger proposal review
- before patching, call `get_algorithm_proposal_status(algorithm_id)`, re-read `editable_proposal_path`, and target that root `PROPOSAL.md`; never patch `proposal_markdown_registry_path`
- let the new approved proposal become active before review/training
- if the change invalidates old evidence, use `mark_result_obsolete(...)`
- if a branch is wrong, use `rollback_algorithm_workspace(...)` or campaign
  rejected-trial restore/resume tools

Config-only tuning can proceed without proposal revision, but every run still
belongs in the registry or campaign history.

## Benchmark Rule

Benchmark datasets must already contain:

- `adata.obs["time_point_processed"]`
- `adata.obsm["X_latent"]`

`register_algorithm_benchmark_dataset(...)` validates and catalogs only. It
does not infer missing fields. If either field is missing, read
`preprocessing-execution`, create a prepared `.h5ad`, and register that file.

Global benchmark cards store comparable builtin/reference metrics only:

- W1
- TMV
- runtime
- memory

Agent-proposed claim metrics stay in campaign registries.

Prefer existing benchmark datasets. Before proposing a new Stage 2 simulation,
inspect the catalog with `list_algorithm_benchmarks(...)` and
`get_algorithm_benchmark_dataset(...)`. Create a new simulation only when the
current catalog cannot test the proposal's stated claim or algorithm
assumption. If an existing prepared dataset can test the claim, reuse it.

For algorithms motivated by real biological questions, use real biological
benchmarks as early as feasible, ideally in every stage panel or as a guardrail
when a registered dataset matches the assumptions. Simulations are still useful
for theory-first methods, controlled mechanism validation, or missing real-data
coverage, but simulation success alone should not be treated as full evidence
for a real biological claim.

Do not create a simulation only to make the algorithm look good, avoid a hard
baseline, or avoid writing an observable claim metric. A valid custom simulation
must isolate a real claim, freeze a generator/source into `simulation_version`,
and rerun builtin/reference baselines on that exact version.

The Stage 2 claim metric must be semantically tied to the claim. It should not
be a generic proxy that can improve without solving the problem, such as raw
activity, variance, spread, or nonzero auxiliary output, unless the proposal
shows why that proxy is sufficient and how a claim-absent control falsifies it.

## Campaign Rule

Use campaign tools for normal tuning. Do not hand-label decisions and do not
restart trials casually.

Golden path:

1. `start_algorithm_campaign(...)`
2. edit active workspace/config
3. `make_benchmark_dataset_config(...)`
4. `run_campaign_trial(...)`
5. inspect automatic `promote` / `reject`
6. `check_campaign_stage_gate(...)`

Keep only the high-level rules in mind here:

- trial `promote` / `reject` compares against the algorithm's own stage active
  best
- stage gates compare against external builtin/reference baselines
- each stage freezes its dataset panel once selected; vary config, not datasets
- Stage 1 and Stage 3 should normally include Weinreb for real biological
  algorithms so scalability is tested
- TMV is hard only for algorithms that model unbalanced mass
- material proposal revisions reset the campaign to Stage 1

For exact budgets, panel rules, claim metric baseline requirements, timeout
semantics, and failure handling, read `campaign-tuning/SKILL.md` and only then
the referenced file you need.

## Next-Step Selector

- No stable research problem: read `research-question-framing`.
- Concrete algorithm idea but no approved proposal: read `proposal-theory`.
- Approved proposal but no workspace: initialize the workspace.
- Workspace exists and code must change: read `authoring`.
- Code/config changed: read `review-and-training`, then preview/review.
- Tuning a custom algorithm: read `campaign-tuning`.
- Campaign has plateaued or results are surprising: read `tuning-playbook`.
- Adding a benchmark dataset: preprocess first, then read
  `benchmark-dataset-registration`.
