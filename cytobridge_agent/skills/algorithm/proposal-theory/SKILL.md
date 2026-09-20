---
name: proposal-theory
description: Theory gate for custom CytoBridge algorithms. Use before proposal approval and before any coding to justify that, in the exact-fit limit, the proposed method can in principle recover the observed time-indexed distributions, and when unbalanced mass is explicitly modeled, the observed cell-count / total-mass changes as well.
---

# Algorithm Proposal Theory Gate

Use this skill before proposal approval, workspace initialization, or any
algorithm implementation work.

This is the theory-first gate. Keep the proposal focused on mathematical
semantics, recoverability, claimed capability, and evaluation evidence. Concrete
CytoBridge hook selection belongs to authoring after the workspace exists.

## Good Algorithm Standard

A proposal should describe a real algorithmic contribution, not a bag of
compatible components. Approve directions that have at least one defensible
source of value:

- a biologically meaningful mechanism or conditioning signal that addresses a
  real temporal single-cell question;
- a new or cleaner theoretical route, estimator, variational identity,
  dynamics formulation, or recoverability argument;
- a deeper, clearly justified extension of a builtin or prior proposal that
  solves a limitation the original method does not solve.

Do not approve a method that is mainly a stitched combination of existing
losses, heads, couplings, correction layers, or metrics unless those pieces are
derived from one coherent problem statement. Before proposing or approving a
new algorithm, check builtin methods and existing proposal/history assets; avoid
repeating an already proposed algorithm under new names. Building deeper on an
existing idea is allowed, but the new proposal must state what is genuinely new,
why the extra depth matters, and how evidence will distinguish it from the
earlier method.

The proposal must also state the lifecycle evidence needed before the algorithm
can be called usable: implementation review, preview, campaign Stage 1/2/3, and
final regression locked release. Proposal approval only authorizes
implementation; it is not evidence that the algorithm works.

Do not automatically downgrade a custom-algorithm direction to make it easier
to pass. If the user/data-driven motivation was to solve a real biological,
dynamical, or theoretical gap, a revision must still propose a meaningful and
worthwhile algorithm for that gap. Do not silently turn a failed direction into
a diagnostic-only analysis, a baseline hyperparameter/config variant, a
metric-only wrapper, a generic post-hoc correction, or a low-novelty builtin
adjacent method. If the original direction is not working, either strengthen the
algorithm while preserving the gap, or reject the direction and propose better
routes.

## Hard Gate

A proposal is not ready unless it explains all of the following:

- what problem or capability the algorithm claims to solve;
- why exact optimization of the proposed training objective can recover the
  observed time-indexed marginals;
- what learned inductive rule rolls a new valid t=0 cell/particle through time;
- which quantities are training-only supervision and which are inference-time
  inputs for new cells;
- why inference does not use training-cell ids, row ids, memorized OT rows,
  barcode-specific lookup, future observed snapshots, future distribution
  statistics, or target-specific correction;
- what evidence, beyond generic W1/TMV when relevant, would validate the claimed
  capability.

If `mass_modeling_scope = models_unbalanced_mass`, the proposal must also
explain why the final weighted-particle inference measure can recover observed
cell-count / total-mass changes. Mass recovery must come from a meaningful
dynamics or biological mechanism, not target-count lookup, post-hoc weight
rescaling, time-only count-ratio clocks, renormalization layers, or any metric
repair whose only purpose is to make TMV small after rollout.

## Claim Metric Rule

Claim metrics must be meaningful, not merely easy to optimize. A valid claim
metric directly measures the problem the algorithm claims to solve, and a better
value must imply that the proposed mechanism worked rather than only producing
more activity, variance, smoothness, spread, nonzero growth, or another generic
side effect.

Do not downgrade the claim to make a failing algorithm pass. If the proposal is
motivated by a real dataset gap or an earlier failure analysis, the primary
claim metric must still measure that gap. It is not acceptable to replace the
hard target with a weaker proxy that can improve while the original problem
remains unsolved, such as swapping lineage/fate-concordant transitions for
generic lineage-distance geometry, swapping mass/growth recovery for a
nonzero-growth diagnostic, or swapping perturbation response accuracy for broad
state smoothness. A proxy can be the primary metric only when the proposal
states a defensible equivalence or necessity argument, and includes controls
showing that random, collapsed, shuffled, claim-absent, and nearest-builtin
solutions would not pass.

For custom claim metrics, state:

- why the metric is a faithful observable for the claimed object;
- which original user/data-driven gap it answers, and why passing the metric
  would demonstrate that the gap was solved rather than avoided;
- why it would fail under random, collapsed, shuffled, or claim-absent controls;
- which low-level mechanism diagnostic would falsify the top-level score;
- whether required labels, lineage/barcodes, spatial coordinates, perturbations,
  conditions, batches, paired modalities, or other side information are complete,
  noisy, censored, missing, or only partially observed.

Do not define the truth source by column names alone. `cell_type`, `label`,
`lineage`, `barcode`, time bins, or category counts show what metadata exists,
not what biological truth it encodes. If the claim metric uses terminal fate,
descendant fate distribution, lineage-truth endpoints, fate-concordant
transitions, growth truth, perturbation truth, or another biological target,
state the dataset/proposal/documentation contract that makes that interpretation
valid. If no real dataset contract supports the claim, use a controlled Stage 2
simulation or choose a different claim metric.

If existing real benchmarks do not contain the metadata or ground truth needed
to evaluate the claim, design a controlled Stage 2 simulation that exposes the
claimed mechanism. The simulation should freeze the generator/source, rerun
builtin/reference baselines on the same version, and include a claim-absent,
shuffled, collapsed, or ablated control.

## Branching Rule

Do not require single-particle branching unless branching is part of the claim.
For ordinary distribution transport, mass/growth modeling, smooth dynamics, or
other marginal-distribution objectives, it is enough to explain why the induced
weighted distribution over time can match the target marginals. If the claim is
lineage bifurcation, multimodal fate uncertainty for one initial cell, clone fate
splitting, or branch probabilities, then the proposal must define the branching
data contract and how branch probabilities or branch-specific trajectories are
inferred without future-target lookup.

## Research Grounding Rule

For a genuinely new algorithm, do not invent from scratch. Before approval, read
relevant survey notes, builtin CytoBridge docs, package algorithm maps, theory
sources, and directly relevant papers. RAG or web snippets are discovery only:
identify the source, read the relevant original page/section/PDF, then cite or
reason from it.

A genuinely new proposal should include a `Literature and Package Grounding`
section with the builtin methods considered, directly relevant papers/notes
read, the concrete remaining gap, and why the method is not a renamed builtin or
already-covered variant. For nontrivial OT/SB/WFR/UOT/continuity-equation
objectives, include theory-book/page or chapter references after reading the
returned page ranges.

## Proposal Content Checklist

Every proposal should contain, or clearly answer, these items:

- `Abstract`
- `Literature and Package Grounding`
- `Problem Statement`
- `Problem Mathematical Form` when available
- `Mathematical Derivation to Algorithm Design` for new mathematical methods
- `Novelty and Contributions` for genuinely new algorithms
- `Claimed Capability`
- `Expected Evaluation Outcome`
- `Inductive Generalization Argument`
- `mass_modeling_scope`: exactly `balanced_only` or `models_unbalanced_mass`
- recoverability of observed marginals, and mass recovery if unbalanced
- implementation-oriented paper-style pseudocode with `Inputs`, `Outputs`, and
  labeled steps `P1`, `P2`, ...
- algorithm semantics table mapping mathematical objects to runtime layers
- evaluation plan where builtin W1/TMV remain fixed and custom metrics are
  additive, rollout-derived, and claim-aligned

Before drafting a new proposal or major revision, call
`get_algorithm_proposal_template()`. To revise an existing proposal, call
`get_algorithm_proposal_status(algorithm_id)`, re-read `editable_proposal_path`,
and patch the root `PROPOSAL.md`, not the immutable registry markdown copy.

## When To Read The Detail Reference

Read `references/proposal-detailed-standards.md` when any of these are true:

- the algorithm is genuinely new rather than a minor config or builtin variant;
- the method claims a new mathematical objective or algorithmic contribution;
- the proposal uses OT, dynamic OT, WFR/UOT, Schrodinger bridge,
  continuity-equation, variational-flow, mean-field, stochastic, branching, or
  side-information-dependent semantics;
- the claim metric or benchmark evidence is custom or non-obvious;
- the proposal was rejected for missing derivation, grounding, novelty,
  recoverability, pseudocode, or self-containment.

## Evaluator Semantics

The evaluator decides `approve` / `revise` / `reject` based on theoretical
correctness, sufficiency, and whether the proposed method can solve its stated
claimed problem under the declared modeling scope. Implementation risks are
advisory unless later evidence shows they are real failures.

`risk.md` is a diagnostic watchlist for implementation and campaign phases, not
a second approval gate. After a proposal revision, the latest `risk.md` belongs
to the latest proposal version; older risk reports remain in the proposal
registry.

Do not approve or code an algorithm that lacks the recoverability argument,
inductive runtime rule, meaningful claim/evidence contract, or required
`mass_modeling_scope`.
