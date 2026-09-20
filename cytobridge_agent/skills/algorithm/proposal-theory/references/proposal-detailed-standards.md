# Proposal Detailed Standards

This file preserves the detailed proposal-theory standards. Read it when drafting or revising a custom algorithm proposal, especially for genuinely new algorithms, mathematical objectives, branching claims, literature grounding, derivation quality, output templates, or failure conditions.

## Branching And Microscopic Realizations

Do not turn single-particle branching into a default requirement.

For many CytoBridge algorithms, the target object is the evolving weighted
empirical distribution. In the infinite-particle limit, a non-branching
characteristic flow, a stochastic endpoint sampler, and an explicit branching
particle story can be different microscopic realizations of the same marginal
distribution evolution. If the proposal's scientific claim is distribution
transport, mass/growth modeling, smooth dynamics, or another non-branching
objective, it is enough to explain why the induced weighted distribution over
time can match the target marginals.

Branching should be treated as a key theoretical requirement only when it is
part of the claim, for example lineage bifurcation, multimodal fate uncertainty
for one initial cell, branch coverage, clone-level fate splitting, or another
problem where the microscopic branching interpretation itself is the scientific
target. In those cases, the proposal must state the branching data contract,
what can be observed at inference time, and how the learned dynamics represents
branch probabilities or branch-specific trajectories without future-target
lookup.

## Research Grounding Before New Algorithms

If this is a genuinely new algorithm proposal, not a minor config change or a
small variant already covered by the builtin family, do not start by inventing a
method from scratch.

Use the same research-onboarding discipline as the idea workflow, but with
algorithm design as the target:

1. read the local survey overview first:
   - `cytobridge_agent/skills/planner/research-question-framing/references/agent_algo_survey_20260417/synthesis.md`
   - `cytobridge_agent/skills/planner/research-question-framing/references/agent_algo_survey_20260417/reading-order.md`
   - a small number of relevant files under `.../notes/`
2. read the package algorithm map:
   - `CytoBridge-main/docs/INDEX.md`
   - `CytoBridge-main/docs/theory/README.md`
   - relevant builtin algorithm pages under `CytoBridge-main/docs/theory/`
3. inspect the current builtin implementation only enough to understand what
   CytoBridge already does:
   - relevant `CytoBridge-main/CytoBridge/configs/*.yaml`
   - relevant builtin training/model code paths referenced by the docs
   - do not turn this into hook-selection or integration work
4. check existing assets before proposing another method:
   - `list_research_ideas(...)`
   - `get_algorithm_proposal_status()`
   - `list_experiment_history(...)`
   - `list_algorithm_benchmarks(...)`
5. use `search_theory(...)` for mathematical theory questions when the proposal
   depends on OT, dynamic OT, WFR/UOT, Schrodinger bridge, continuity-equation,
   variational-flow, or proof-level arguments. Do not stop after the search
   result. Follow the returned page ranges with
   `read_file(..., pdf_mode="text", pages="...")` and read the relevant
   original book pages before citing or using the theory.
6. use `search_literature(...)` / RAG for algorithm papers, baseline methods,
   implementation precedents, and targeted gaps after the notes, builtin docs,
   implementation scan, existing-asset check, and theory-book lookup are not enough
7. RAG snippets are discovery evidence, not sufficient grounding by themselves.
   The required pattern is: search -> identify the relevant source/page/section
   -> read the original source with `read_file(...)` -> only then cite or build
   proposal logic from it. For papers that materially shape the proposal, read
   the original PDF or the important sections with `read_file(...)` / the PDF
   skill before citing them as support.
8. use `web_search(...)` / `web_fetch(...)` when local notes, local literature,
   local theory books, and package docs are stale, incomplete, or too narrow.
   Use web results to find additional relevant papers, then fetch/read enough of
   the source to understand the objective, assumptions, and limitation being
   cited. Do not cite web-search snippets without reading the source.

For a genuinely new algorithm, the proposal must cite at least 10 related
papers or literature notes that are directly relevant to the proposed method.
Do not pad this list. Each citation should explain why it matters to the
proposal, for example inherited objective, contrastive limitation, baseline
method, mathematical tool, or evaluation precedent.

For any proposal with a nontrivial mathematical objective or exact-fit
argument, the grounding must include both sides:

- theory-book grounding from `search_theory(...)`, with book/page or
  chapter/section/page references after reading the recommended page range
- algorithm-literature grounding from `search_literature(...)`, with directly
  relevant papers or notes and full-paper / important-section reading for
  methods that shape the proposal
- optional web grounding from `web_search(...)` / `web_fetch(...)` only when
  local sources are insufficient, again followed by source reading rather than
  snippet-only citation

The resulting proposal should include a `Literature and Package Grounding`
section that states:

- which builtin CytoBridge algorithms were considered
- which relevant papers or local notes were read
- at least 10 directly relevant literature citations for genuinely new
  algorithms, each with a one-line relevance note
- what concrete gap remains after those methods
- why the proposed method is not merely a renamed builtin or already-covered
  variant
- which package semantic axis changes, such as coupling, path, mass strategy,
  stochasticity, geometry, or evaluation target

If this grounding cannot be written, the proposal is not ready. First do more
research and define evidence that can actually test the claim.

## Evaluator Decision Semantics

When the runtime proposal evaluator reviews a proposal:

- `approve` / `revise` / `reject` should be based on theoretical correctness and sufficiency within the proposal's declared modeling scope
- the evaluator should check whether the method can solve its own stated problem / claimed capability
- the evaluator should still report concrete implementation / optimization / numerical risk points
- those implementation risks are advisory and do not by themselves overturn a theoretically valid approval

Planner behavior after approval:

- if the proposal is approved, continue into authoring under the current plan
- keep the evaluator's implementation risks as a watchlist for implementation and training
- only revisit the proposal if later evidence shows that those risks correspond to real algorithmic or numerical failure

Risk artifact:

- the reviewer writes the current proposal-version risk assessment to `risk.md`
- `risk.md` is not a second approval gate; it is a diagnostic watchlist for the
  implementation and campaign phases
- each useful risk entry should state the risk, how it may manifest in
  CytoBridge training/campaign/evaluation, and what diagnostics can reveal it
- after a proposal revision, the latest `risk.md` belongs only to the latest
  proposal version; older risk reports are preserved in the proposal registry

## Proposal Self-Containment Requirement

The proposal itself must be logically self-contained.

This means:

- a reader should be able to understand the proposed algorithm from the
  proposal alone
- the core mathematical objects must be defined explicitly
- notation must be introduced instead of assumed
- the algorithm statement must not rely on hidden context from later code,
  runtime hooks, or oral explanation

Do not write proposals that only become understandable after reading:

- `algorithm.py`
- package source
- later review notes
- implementation comments

The proposal should stand on its own as a mathematically readable algorithm
specification.

## Ambition Standard

Do not aim only for the smallest safe increment.

A strong CytoBridge proposal should actively try to solve a real biological
problem, close a real theoretical gap, or ideally do both. The proposer should
use all available resources: local survey notes, builtin algorithm docs,
package implementation semantics, theory-book RAG, literature RAG, full PDF
reading, web search/fetch when local sources are insufficient, and concrete
benchmark context.

The agent should not be satisfied with a method that only renames a builtin
algorithm or adds a minor component without a clear reason. It should think
hard about:

- what biological question or failure mode would actually matter
- what mathematical object would make that problem precise
- which part of existing OT/SB/UOT/WFR/FM theory is insufficient
- whether the same target admits a simulation-free or direct variational route,
  for example a static equivalence, lifted-state construction, endpoint bridge
  identity, conditional-path identity, or closed-form supervision target
- whether the proposal can combine biological motivation and theoretical
  innovation rather than choosing the easiest side
- what evidence would make the contribution convincing

It is acceptable for a proposal to be primarily biological or primarily
theoretical, but the best proposal usually has both: a meaningful biological
motivation and a coherent mathematical route. If one side is intentionally not
claimed, state that explicitly and explain why.

Prefer elegant, one-piece constructions. A strong proposal should read as:

mathematical problem -> computable representation -> supervision targets ->
trained dynamics -> rollout/evidence.

Do not default to a patchwork of endpoint couplings, generic flow-matching
losses, growth heads, post-hoc mass fixes, and metric-specific corrections
unless each component is derived from the same stated problem. If the intended
target can be solved by a simulation-free route under comparable assumptions,
use that route or explain precisely why it is not appropriate. Non
simulation-free algorithms are allowed, but they need a real reason: for example
the target dynamics has no known static reduction, requires online interaction,
or the simulation step is itself the scientifically meaningful object.

Quality gate:

- biology-first algorithms must have real biological meaning in their
  components, conditioning variables, mass/growth mechanism, inference rule, and
  expected evidence. Do not pass an arbitrary correction layer just because it
  improves a metric.
- theory-first algorithms must solve or approximate a meaningful mathematical
  dynamics problem, prove a nontrivial equivalence/estimator, or introduce a
  clean modeling axis. Do not pass stitched surrogate losses or renamed builtin
  variants as new theory.
- data-driven algorithms must keep the metric tied to the real data gap that
  motivated the proposal. A revision that turns a hard lineage/fate, growth,
  perturbation, condition, or modality gap into a weaker generic proxy is a
  regression, even if the weaker metric is mathematically well-defined.
- if the core idea is low-value or patchwork even after likely revisions, reject
  it and give concrete better routes rather than trying to approve the narrowest
  salvageable version.

## No Automatic Algorithm Downgrade

Do not let proposal revision become an automatic ambition-lowering mechanism.
When a custom algorithm lifecycle begins from a user/data-driven gap, every
approved revision must still answer that gap with a meaningful algorithmic
contribution.

Invalid algorithm downgrades include:

- replacing a new dynamics model with a diagnostic-only analysis and still
  treating the lifecycle as an algorithm success;
- turning the method into a baseline hyperparameter/config variant after the
  custom method fails;
- keeping only a metric adapter, report wrapper, visualization, or post-hoc
  correction and calling it the algorithm;
- removing the biological/theoretical mechanism that made the proposal new and
  retaining only generic flow matching, endpoint coupling, or smoothing losses;
- approving the narrowest passable surrogate after Stage 1/2 failures even
  though it no longer solves the motivating real-data gap.

If the strong claim cannot be made true, the correct reviewer outcome is not to
auto-approve a weaker algorithm. Return `revise` when there is a credible route
to a stronger method that still solves the original gap; return `reject` when
the remaining idea is only a low-value surrogate, builtin variant, or diagnostic
workflow.

## No Claim Downgrade

Do not approve or draft proposal revisions that make the algorithm easier to
validate by changing the claim into a different, weaker problem.

The proposal must preserve a clear chain:

original motivating gap -> claimed capability -> primary claim metric -> Stage 2
evidence.

If any link changes, the proposal must explicitly say what changed and why the
new claim still solves the original user/data-driven problem. If it cannot do
that, the correct action is to revise the algorithm, design a controlled
simulation that directly observes the original claim, or mark the direction as
failed and propose a new algorithm. Do not salvage the lifecycle by making a
proxy the primary claim.

Invalid downgrades include:

- replacing lineage/fate-concordant transition recovery with a generic
  correlation between lineage labels and generated geometry;
- replacing descendant or held-out fate-distribution recovery with an endpoint
  distribution statistic that ignores the source lineage group;
- replacing mass or clone expansion recovery with a nonzero-growth or
  smooth-weight diagnostic;
- replacing perturbation response or condition-specific dynamics with generic
  W1/TMV improvement;
- making a builtin-adjacent config variant look successful by dropping the part
  of the claim that distinguished it from the builtin.

A proxy may be used as a primary claim metric only when the proposal gives a
strong necessity or equivalence argument and specifies controls under which the
proxy fails whenever the original biological/dynamical claim is absent.

## Novelty And Non-Duplication Standard

A new proposal must show that it is not just a recombination or rephrasing of
work the package or proposal registry already contains.

Before approving a new algorithm, check the nearby builtin family and existing
proposal/history assets. The proposal should explicitly state one of these
relationships:

- **new biological mechanism:** it uses biologically meaningful state,
  lineage, perturbation, regulatory, clone, time, mass/growth, spatial, or
  condition information in a way that changes what the algorithm can learn;
- **new theoretical route:** it introduces a distinct objective, estimator,
  equivalence, relaxation, dynamics, bridge/path identity, or recoverability
  argument rather than only adding another loss term;
- **deeper extension:** it extends a prior/builtin method along a named axis
  and explains why that axis was missing or insufficient in the earlier method;
- **new evidence contract:** it makes a claim that existing methods could not
  test and defines the simulation or real-data benchmark needed to test it.

Reject or revise proposals that only:

- concatenate standard flow matching, OT/UOT/WFR couplings, growth heads,
  smoothing penalties, and metric-specific correction layers without one
  unifying derivation;
- rename an existing builtin or previous proposal while changing only minor
  hyperparameters or implementation hooks;
- add a component because it is fashionable or easy to implement, rather than
  because the stated biological/theoretical problem requires it;
- claim novelty through a custom metric that can improve without solving the
  proposed biological or mathematical problem.

It is acceptable to develop an existing idea more deeply. In that case, the
proposal must name the earlier method, state the shared core, identify the new
axis, and define evidence that would separate the deeper method from the
original. The bar is not "never reuse ideas"; the bar is "do not spend a full
custom-algorithm lifecycle on a disguised duplicate or patchwork."

## Mathematical Derivation To Algorithm Design

For a proposal that claims a new mathematical objective or algorithmic
innovation, do not stop at a shallow mathematical restatement of the
implementation.

The proposal should include a `Mathematical Derivation to Algorithm Design`
section, or equivalent content, that is self-contained enough for a technically
strong reviewer to follow without opening the source code or reading the
original papers first. It should walk from the stated mathematical problem to
the concrete algorithm design without hidden jumps:

1. Restate the high-level problem in operational variables.
   - Define measures, particles, weights, controls, growth rates, scores,
     couplings, paths, constraints, target marginals, and time indexing.
   - If the problem is dynamic OT, Schrodinger bridge, WFR/UOT,
     continuity-equation control, mean-field dynamics, or a new dynamic
     problem, write the state equation and objective/constraint system.
   - Do not merely rewrite the proposed loss in mathematical notation.
2. Derive the first computable representation.
   - Show the exact equivalence, relaxation, estimator, coupling construction,
     bridge identity, path identity, score/flow matching identity, or
     finite-sample approximation that turns the high-level problem into
     something trainable.
   - State why this transformation is legitimate: exact theorem, limiting
     argument, variational bound, consistency argument, or explicit proposal
     assumption.
   - Before introducing simulation-heavy optimization or heuristic rollouts,
     check whether the problem has a simulation-free representation such as a
     static endpoint problem, lifted balanced problem, conditional bridge/path
     identity, or direct regression target.
3. Derive the supervision targets or optimization objective.
   - Define the conditional path, target velocity, target growth, score target,
     coupling weights, mass weights, or other labels used by training.
   - Explain how each target is computed from allowed training data.
   - If an estimator is biased or approximate, state the bias source and why it
     is acceptable at proposal level.
4. Derive the inference rule.
   - State the trained objects returned by the algorithm.
   - State how a new valid t=0 cell/particle is rolled through continuous time.
   - State which information is allowed at inference and which training-only
     objects are not available.
5. Derive recoverability.
   - Explain why exact optimization of the proposed design recovers the claimed
     distribution evolution.
   - If `mass_modeling_scope = models_unbalanced_mass`, separately explain why
     the weighted-particle measure recovers total mass/cell-count changes.
   - If the method intentionally does not model unbalanced mass, say so and do
     not claim TMV recovery.
6. Name every proposal-stage assumption or approximation.
   - Keep assumptions minimal and explicit.
   - Distinguish conceptual approximations in the proposed algorithm from later
     engineering accelerations such as mini-batching.

This section should answer:

- why this algorithm is the right way to solve the stated mathematical problem
- what is exact, what is approximate, and what limit is being invoked
- which approximation is intrinsic to the proposal and which approximation is
  only an implementation/runtime acceleration to be justified later in
  `IMPLEMENTATION_MAP.md`

Keep proposal-stage assumptions and approximations minimal. It is acceptable to
introduce reasonable assumptions, but every assumption must be named, justified,
and connected to the loss. Do not add convenience approximations only because
they make coding easier; if the exact proposal is too hard, narrow the proposal
explicitly rather than silently changing the mathematical problem.

Use `CytoBridge-main/docs/theory/wfrfm-derivation-example.md` as the quality
bar for this section when the proposal claims to solve a formal dynamic problem:
it starts from a named dynamic objective, derives the static coupling object,
derives the training targets, and then proves distribution and mass recovery.
Do not copy WFR-FM; copy the level of explicitness.

If the proposal is primarily a biological-motivation method, a reproduction,
a config/builtin variant, or otherwise not claiming a new mathematical algorithm,
the section may say `Not applicable`, but it must explain why no derivation is
needed and still state the modeling target and evaluation contract. If a useful
derivation can be written anyway, write it; it strengthens the proposal.

## Novelty And Contributions

For a genuinely new algorithm, the proposal must include `Novelty and
Contributions` or equivalent content.

Write it like a top-conference contribution statement:

- what concrete gap remains after builtin CytoBridge methods and prior papers
- what is new mathematically, statistically, biologically, or computationally
- which part is a contribution versus a borrowed component
- why the proposal is not merely a renamed builtin, a standard flow-matching
  variant, or a known prior method
- what evidence would validate the contribution

Do not claim novelty from generic phrases such as "uses flow matching",
"uses OT", "adds a neural network", or "handles single-cell dynamics" when the
package or literature already covers that axis.

If the task is not an algorithm-innovation task, state that explicitly and
explain the actual contribution type, for example reproduction fidelity,
biological hypothesis test, new benchmark, or implementation cleanup.

## No Skipped Derivation Steps

The proposal must not skip the critical reasoning steps that connect:

- the mathematical definition
- the supervision targets
- the induced dynamics
- the recoverability claim

It does not need to be a journal-proof with every algebraic detail, but it must
show the full logical chain without hand-waving phrases like:

- “similarly”
- “it is obvious that”
- “the rest follows”
- “details omitted”

When a quantity is central to the method, the proposal must spell out:

- what it is
- how it is defined
- why that definition is the correct one for the method
- how it enters training or inference
- what data it is allowed to depend on
- what exact-fit or limiting statement it supports

If that chain cannot be written clearly in the proposal, the proposal is not
ready.

## Proposal-Stage Scope

At proposal time:

- do **not** spend time on CytoBridge hook selection
- do **not** spend time on package integration details
- for a genuinely new algorithm, do inspect package docs, relevant builtin
  configs, and relevant builtin implementation paths enough to understand what
  the package already solves
- stay focused on:
  - literature and package grounding
  - mathematical semantics
  - abstract, problem statement, and claimed capability
  - recoverability
  - inductive generalization to new valid t=0 cells/particles
  - machine-readable algorithm attributes such as `mass_modeling_scope`
  - algorithm semantics table
  - implementation-oriented paper-style pseudocode
  - evaluation plan
  - expected evaluation outcome for later benchmark or simulation selection

Detailed hook selection, file layout, and integration mechanics belong to
authoring, not to theory approval. Package docs and selected builtin source are
allowed at proposal time when they prevent duplicate or under-researched
algorithm proposals.

## Required Question

For the proposed method, answer this explicitly:

- what problem is this algorithm trying to solve, and what capability does it
  claim beyond existing/default methods?
- if the training objective is fit exactly, why should the learned dynamics be
  able to recover the observed dataset distributions at the target time points?

If `mass_modeling_scope = models_unbalanced_mass`, also answer
this explicitly:

- why should the final weighted-particle measure recover the observed
  cell-count / total-mass changes at the target time points?
- what biological or dynamical quantity produces that mass change during
  rollout?
- is the mass mechanism a learned growth field, WFR/UOT/SB mass term,
  proliferation/death process, condition-driven prior, or another interpretable
  mechanism?
- why is it not a target-count correction, post-hoc rescaling, or metric-specific
  TMV repair?

This is not optional.

Also answer this explicitly:

- can a technically strong reader understand the algorithm from the proposal
  alone, without needing package code as a prerequisite?
- what is the inductive rule that maps a new valid t=0 cell/particle and allowed
  context to a rollout, growth/weight update, or endpoint distribution?
- which proposal quantities are training-only supervision, and which are
  inference-time inputs that must be available for new cells?
- why does the proposal not rely on cell-id, row-id, barcode-specific, or
  target-specific memorization that cannot generalize to new cells?

If the answer is no, the proposal is incomplete.

## Builtin Theory Reference

Package-level builtin theory is centralized in:

- `CytoBridge-main/docs/theory/README.md`

Use that document as the shared reference for:

- what each builtin solves
- why balanced builtins can fit distributions but not total mass
- why unbalanced builtins must justify both distribution and mass fit
- when TMV is a hard gate versus a diagnostic metric
- why `vgfm` is the deterministic unbalanced flow-matching default
- why `wfrfm` is the named-WFR dynamic unbalanced OT flow-matching alternative
- how `wfrfm` derives a concrete implementation from a WFR objective
  (`CytoBridge-main/docs/theory/wfrfm-derivation-example.md`)

Do not copy long builtin derivations into proposals. A proposal should instead
state which builtin it is closest to, what semantic axis it changes, and how the
recoverability argument changes.

## What Every New Proposal Must State

CytoBridge proposals are dynamics-model proposals. The proposal must make clear
how the trained model rolls the t0 population forward through a continuous
trajectory, and how every evaluation claim is computed from that trajectory.
Do not design a proposal whose evidence depends on direct metric prediction,
future-snapshot lookup, or post-hoc particle/weight repair.

Every custom algorithm proposal should contain a short section answering:

1. `Abstract`: what is the core idea in one compact paragraph?
2. `Literature and Package Grounding`: what local survey notes, builtin
   algorithms, package docs/source paths, and key papers were checked, and what
   gap remains after them? For genuinely new algorithms, include at least 10
   directly relevant citations with a short note for each; these must come from
   notes/RAG plus full-paper or important-section reading, not from snippets
   alone. If the proposal uses OT/SB/WFR/UOT/continuity-equation or other
   formal dynamics theory, also cite the relevant theory-book page or
   chapter/section/page ranges found through `search_theory(...)`.
3. `Problem Statement`: what problem, gap, or failure mode is being solved?
4. `Problem Mathematical Form`: if available, what objective/dynamic problem is
   being solved? If no clean global form exists, say so and define the problem
   in precise prose.
5. `Mathematical Derivation to Algorithm Design`: for new mathematical
   algorithm proposals, how does the stated problem become the concrete
   algorithm design, including estimators, coupling/path/dynamics choices,
   losses/objectives where relevant, and inference rule? If not applicable,
   why not?
6. `Novelty and Contributions`: for genuinely new algorithms, what is new
   relative to builtin methods and directly relevant literature? If not an
   algorithm-innovation proposal, what is the actual contribution type?
7. `Claimed Capability`: what capability should improve, and what evidence will
   validate it?
8. `Expected Evaluation Outcome`: where the algorithm should work, what result
   pattern beyond builtin W1/TMV should appear, and why that evidence validates
   the claimed capability. The expected evidence must be measured from the
   trained dynamics rollout / evaluation trajectory. Write this so a later agent
   can choose benchmark datasets or design simulations without rereading the
   whole theory. If the proposal is motivated by a real biological problem,
   name what real benchmark evidence should be sought early when available;
   if simulation-only evidence is acceptable, explain whether the reason is
   theory isolation, controlled mechanism validation, or lack of a suitable real
   dataset.
9. `Inductive Generalization Argument`: what learned rule applies to a new
   valid t=0 cell/particle, what context or modalities it may consume at
   inference time, what is training-only supervision, and why inference does not
   depend on memorized training-cell identities or target-specific lookup.
10. `mass_modeling_scope`: exactly one of `balanced_only` or
   `models_unbalanced_mass`
11. what joint coupling or target measure is being fit?
12. what is the intended target marginal at each adjacent interval?
13. what role does the path play in reaching that marginal?
14. what role does growth or mass dynamics play in reaching that marginal?
15. in the exact-fit limit, why should the induced predicted measure recover the
   observed late distribution?
16. if unbalanced mass is enabled, why should the final weighted particles also
   recover the observed late total mass / cell-count change?
   The answer must tie mass to the learned dynamics or a biologically meaningful
   proposal-approved mechanism, not to a post-hoc clock or target-count repair.
17. which runtime axis changes:
   - `model.components`
   - `train_strategy`
   - `MassStrategy`
   - `conditional path`
16. a pseudocode sketch for the actual implementation
   - with global `Inputs` / `Outputs`
   - and for every step `P1/P2/...`:
     - `Inputs`
     - `Outputs`
     - `Invariants`
     - `Forbidden deviations`
17. an algorithm semantics table
   - mapping the key mathematical objects to runtime layers
   - and stating the semantic definition of each object in the method
18. an evaluation plan:
   - builtin `W1` and `TMV` remain fixed and are not redefined in a custom
     workspace
   - any new metric is additive only
   - all metrics read the generated t0-to-final trajectory / trained model
     artifact rather than using a separate metric-specific predictor
   - the proposal explains why that metric validates the stated scientific goal
   - the claim metric is semantically aligned with the stated problem and would
     not stay high under a random, collapsed, shuffled, or claim-absent control
   - if existing real datasets cannot observe the claim, the plan defines a
     controlled Stage 2 simulation whose frozen generator and baselines can test
     it
19. an expected evaluation outcome:
   - intended working scenario, such as the data regime, biological prior, or
     simulation mechanism where the method should help
   - expected result pattern beyond builtin W1/TMV
   - rationale for why that pattern supports the claimed capability

`mass_modeling_scope` is an algorithm attribute, not prose decoration. It is
used by campaign and training logic to decide whether TMV is a hard gate.

## If The Proposal Changes The Defaults

If the proposal changes any of these:

- coupling semantics
- target marginal semantics
- path geometry
- mass strategy
- growth dynamics

then the proposal must explain how the recoverability argument changes.

Examples:

- if the target marginal is no longer uniform empirical late mass, state the new
  target marginal explicitly
- if the path is no longer the builtin path, explain why the new path still
  reaches the intended endpoint distribution
- if mass semantics change, explain how endpoint mass aggregation still matches
  the intended late marginal
- if a time-only or global growth component is proposed, explain the biological
  meaning, allowed training evidence, inference-time data contract, and why it
  is not leaking evaluation target counts

## Failure Conditions

The proposal is not theory-ready if it says things like:

- “we change the coupling but do not specify the target marginal”
- “we add growth but do not explain what total mass should be recovered”
- “we use a new path but do not explain why endpoint recovery still holds”
- “we optimize a heuristic loss and assume the late distribution will work out”
- “we fit TMV by rescaling weights after rollout”
- “we add a time-only count-ratio clock without biological meaning or without
  proving it is learned from allowed training evidence”
- “the learned growth head is ignored at inference and total mass is supplied by
  a separate correction”
- “inference uses a stored row, barcode, or cell id from the training data
  instead of an inductive rule for new valid t=0 cells”
- “the method claims to generalize, but the proposal never states what inputs
  are available for new cells or which signals are training-only supervision”

## Required Proposal Output

Before proposal approval, the proposal should contain sections with titles
similar to:

- To see the current full section-by-section template, call
  `get_algorithm_proposal_template()` before drafting a new proposal or
  patching `PROPOSAL.md` for a revision.
- To revise an existing proposal, first call
  `get_algorithm_proposal_status(algorithm_id)`, re-read
  `editable_proposal_path`, and make the patch target that root `PROPOSAL.md`.
  Do not patch `proposal_markdown_registry_path`; it is an immutable archive
  copy for the current proposal version.
- `Abstract`
- `Literature and Package Grounding`
- `Problem Statement`
- `Problem Mathematical Form`
- `Mathematical Derivation to Algorithm Design`
- `Novelty and Contributions`
- `Claimed Capability`
- `Expected Evaluation Outcome`
- `Inductive Generalization Argument`
- `Mass Modeling Scope`
- `Recoverability Argument`
- `Why The Method Can Recover Observed Marginals`
- `Exact-Fit Distribution Semantics`
- `Implementation Pseudocode`
- `Evaluation Plan`

The proposal should not spend time on concrete hook/interface selection. That
belongs to authoring after the workspace exists.

For a genuinely new algorithm, the proposal should not be package-agnostic in
the sense of ignoring the builtin family. It should understand the builtin
algorithms and prior literature well enough to name the actual gap. The proposal
still should not choose concrete hook files or implementation details before the
workspace exists.

## Required Pseudocode Standard

The pseudocode must be implementation-oriented but still not real code.

Do not write it as loose prose bullets. Write it in a paper-style algorithm
format that a reviewer could read as a real algorithm specification.

Required form:

1. start with `Inputs:` and `Outputs:`
2. define the main state variables, operators, or targets using symbols when
   the method depends on exact formulas
3. state the dynamics problem being trained: CytoBridge algorithms must produce
   a learned dynamics model that rolls t0 particles through a continuous
   trajectory. If the method claims to solve a mathematical problem, formulate
   the dynamic problem explicitly when possible, for example dynamic OT,
   Schrödinger bridge, WFR/UOT, mean-field dynamics, or a new well-defined
   dynamic objective. If no clean global objective exists, say so and define the
   local dynamics, target marginals, and validation contract precisely.
4. present the main procedure as labeled steps `P1`, `P2`, ...
5. make each `P#` step one concrete runtime action that can later map to code
6. include the key equations inside the relevant step instead of keeping all
   mathematics in a separate narrative paragraph
7. end with the returned trained objects, inferred dynamics, or produced
   metrics/artifacts

Recommended skeleton:

```text
Inputs:
- adjacent snapshots {(X_k, t_k)}_{k=0}^{K}
- model heads v_theta, g_phi
- solver parameters (delta, reg, reg_m, sigma, ...)

Outputs:
- trained fields v_theta, g_phi
- optional custom metrics M_custom

Definitions:
- define the coupling/cost objective
- define the conditional path quantities used by the loss

Procedure:
P1. For each adjacent interval, build the pairwise cost matrix C_k.
P2. Solve the transport problem to obtain gamma_k.
P3. Convert gamma_k into the conditional supervision objects used for
    sampling and mass targets.
P4. Sample (k, x_0, x_1, tau, eps) and construct x_tau, u_tau, g_tau,
    loss_weight_tau.
P5. Evaluate v_theta(x_tau, t) and g_phi(x_tau, t).
P6. Minimize the stated objective
    L = E[ loss_weight_tau (||v_theta-u_tau||^2 + kappa ||g_phi-g_tau||^2) ].
P7. During inference, integrate the learned fields to produce future states and
    masses.
```

Good example shape:

- includes the actual training/inference flow
- includes the exact objective or update equation
- names the conditional targets and sampled variables explicitly
- uses stable step ids `P1`, `P2`, ... for later implementation mapping

Bad example shape:

- “Use a better coupling.”
- “Improve growth.”
- “Tune the algorithm until it works.”
- long prose paragraphs with no explicit algorithm steps or equations
