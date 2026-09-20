---
name: research-question-framing
description: Use when the task is to formulate a strong scientific problem, motivation, gap statement, or paper positioning before proposing any concrete algorithm. Especially relevant for CytoBridge research planning on multi-timepoint snapshot single-cell omics, OT/UOT, Schrödinger bridge, flow matching, spatial-temporal dynamics, or interaction modeling. Covers two separate tracks: biology-journal framing and ML top-conference framing, with default scope narrowed to non-perturbation multi-timepoint snapshot data.
---

# Research Question Framing

Use this skill before proposing a method when the real need is:

- identify a good problem to solve
- write a strong motivation
- turn a vague idea into a sharp research question
- separate a biology-journal question from an ML-topconf question
- decide whether a direction is scientifically meaningful before discussing implementation

This skill is for **problem formulation and taste**, not for algorithm design, theorem writing, or code planning.

If the user already has a concrete algorithm and wants to justify recoverability, switch to `proposal-theory` after using this skill.

## Default Scope For CytoBridge Ideas

Unless the user explicitly reopens the scope, keep ideas inside:

- multi-timepoint snapshot single-cell omics
- baseline or naturally varying systems
- optional multimodal measurements
- no perturbation-centered problem formulation
- deep-learning continuous generative dynamics inside the CytoBridge package frame

This means:

- do not frame the main question around perturbation response
- do not assume intervention labels, CRISPR readouts, or Perturb-seq structure
- prefer problems that can already be attacked with current time-resolved snapshot data
- prefer problems whose natural solution class is a learned continuous dynamical or generative model
- do not drift into traditional bioinformatics tasks whose natural endpoint is a static graph, ranking, clustering heuristic, or rule-based pipeline rather than a continuous dynamics model

## Package-Frame Constraint

CytoBridge is not a generic single-cell methods umbrella.

For this skill, a valid `idea` should stay inside the package frame:

- use deep learning
- use multi-timepoint snapshot single-cell data
- aim to learn a continuous dynamical or generative model
- treat transport / growth / stochasticity / interaction / hidden state / geometry as candidate dynamical objects

So:

- do not propose purely traditional pseudotime, graph abstraction, clustering, or marker-ranking problems as final idea assets
- do not treat "better annotation", "better branch calling", or "better DE testing" as sufficient idea objects by themselves
- those can appear as auxiliary evidence or evaluation layers, but not as the core scientific object

## Core Rule

Do not start from "what model should we build?"

Start from:

1. what exact question is still unanswered
2. why existing work cannot answer it cleanly
3. why the answer would matter to the target audience
4. what kind of evidence would count as success

One idea should correspond to **one concrete unresolved question under one specific data regime**.

If the problem statement could headline several different papers, it is still too broad.

## First Decision: Pick the Track

Always classify the request into one of these tracks first:

- `biology-journal`
  - target is a strong biology / methods journal where the algorithm is an enabler of a biological claim
- `ml-topconf`
  - target is NeurIPS / ICML / ICLR style novelty, where the main object is a reusable theoretical or algorithmic gap

If the user is vague, produce both tracks separately. Do not blend them.

Read:

- [biology-journal.md](references/biology-journal.md) for biology-journal framing
- [ml-topconf.md](references/ml-topconf.md) for ML-topconf framing
- [reading-order.md](references/agent_algo_survey_20260417/reading-order.md) when field grounding is needed before proposing or judging an idea

## Literature Onboarding Order

When the agent is not already fluent in this literature, do **not** jump straight to RAG or full PDFs.

Use the packaged survey copy under [agent_algo_survey_20260417](references/agent_algo_survey_20260417/) in this order:

1. Read [synthesis.md](references/agent_algo_survey_20260417/synthesis.md) to recover the field map, major trends, and the default CytoBridge scope.
2. Read [timeline.md](references/agent_algo_survey_20260417/timeline.md) if chronology matters or if the question depends on what came first.
3. Read a small number of specific notes under [notes/](references/agent_algo_survey_20260417/notes/) before expanding outward.
4. Read `~/.cellcompass/skills/workflow/theory-selection/references/package-capability-map.md` or the repo copy under `cytobridge_agent/skills/workflow/theory-selection/references/` to understand what CytoBridge already implements and where the current package boundary still ends.
5. Check existing persistent research assets before inventing a fresh idea:
   - `list_research_ideas(...)`
   - `get_research_idea_status(...)`
   - `get_algorithm_proposal_status()` for the current proposal catalog
   - `list_experiment_history(...)` if a closely related algorithm already exists
6. Use `search_literature` / RAG only after the notes, package boundary review, and existing-asset check are no longer enough for the current question.
7. Read original PDFs only for the few papers whose exact method, theorem, or limitation now matters.

Do not read the entire copied note set by default.
Use the survey copy as a fast field familiarization layer before targeted retrieval.

At this stage, do **not** inspect concrete implementation source code by default.
The goal is to judge the scientific question, not to let current code structure overconstrain the problem statement.

## Optional Second Stage: Bounded Idea Directions

After the problem is sharp, this skill may also propose a **small number of viable algorithm direction families**.

This is allowed only to keep the framing grounded.

The goal is:

- show that the question is not empty talk
- sketch 1-5 plausible solution directions
- keep them abstract enough to avoid premature overfitting to one model
- filter out directions that are clearly impossible under the available data regime
- rank multiple directions by feasibility, from most practical to most speculative

Do not output full method proposals here.

Do output:

- what kind of algorithm family could address the question
- what ambiguity it would try to break
- what extra signal or assumption it would need
- the main implementation or identifiability risk
- why it is feasible now rather than just interesting in principle

Keep these direction families inside the CytoBridge package frame:

- deep dynamical models
- continuous generative models
- OT / UOT / SB / FM style transport or bridge formulations
- structured latent-state or interaction-aware continuous dynamics

Do **not** use this section to suggest classical bioinformatics endpoints as the main solution class.

## Mandatory Workflow

### 1. Define the novelty unit

The question must have a primary novelty unit. Choose one, not five:

- unresolved biological mechanism
- missing measurement regime
- confounded causal interpretation
- identifiability failure
- missing theoretical object
- missing scalable solver for a known object
- missing evaluation/routing principle

If the novelty unit is unclear, the problem is not ready.

### 1.5 Bound the scope before wording the pitch

State the narrow scope explicitly:

- what exact time-resolved snapshot regime
- what exact ambiguity
- what exact inferential output
- what this idea will **not** try to solve

Avoid problem statements that implicitly cover:

- all trajectory inference
- all fate prediction
- all multi-omics integration
- all perturbation reasoning

### 2. State the scientific object precisely

Name what is actually being inferred or explained:

- local direction
- global coupling
- continuous dynamics
- fate distribution
- growth/death field
- interaction field
- hidden state
- uncertainty / refusal boundary

Avoid vague phrases like:

- "better trajectory inference"
- "more interpretable model"
- "more robust prediction"
- "study multi-timepoint single-cell dynamics"
- "build a general foundation model for development"

### 3. Make the gap falsifiable

A good question should imply a concrete failure mode of current work:

- current methods confound transport with growth
- current methods need full observability
- current methods break under context shift
- current methods cannot separate shared and condition-specific dynamics
- current methods use space only as annotation, not as a driver

If no concrete failure mode can be named, the gap is probably cosmetic.

### 4. Keep algorithm out of the motivation

At this stage, describe:

- the problem
- the ambiguity
- the scientific stakes
- the type of advance needed

Do **not** anchor the motivation on:

- OT
- FM
- SB
- GNN
- transformer
- diffusion

Those belong to solution space, not problem formulation.

### 5. Force a win condition

Every candidate direction should answer:

- what would be newly knowable if this worked?
- what kind of result would change the field's behavior?
- why is this worth a paper at the chosen venue level?

### 5.5 State prior work before calling it an idea

Every mature idea must include a `prior_work` view that answers:

- what prior methods already achieved
- what ambiguity they still leave unresolved
- what remaining headroom or improvement space is actually left

This section is not a literature dump.

It should be a compact working diagnosis of:

- the current frontier
- the current package boundary
- the specific failure boundary
- why your proposed question is still open

That diagnosis should explicitly answer:

- what the broader literature already did
- what CytoBridge already supports with current builtin families
- why the proposed idea is not just a rerun of an existing family

### 6. If ideating, keep directions bounded and feasible

Every proposed idea direction must answer:

- what exact object would it model?
- why is that object compatible with the available data regime?
- what minimum assumptions or side information would it need?
- what makes it more than a buzzword recombination?

If those cannot be answered in 1-3 sentences, do not propose the direction.

If there are multiple directions, order them by feasibility.

Preferred format:

1. `Direction family | Why feasible now | Needed signal / assumption | Main risk`
2. `Direction family | Why feasible now | Needed signal / assumption | Main risk`

If one direction is clearly dominant, give only one instead of padding the list.

If the only plausible direction is a traditional non-generative bioinformatics algorithm, that is evidence the question is outside the current CytoBridge idea frame and should not be persisted as a package-facing idea.

## Output Contract

When using this skill, prefer the following structure:

### If the result is ready to become a persistent idea asset

Make sure the output can map directly into:

- `title`
- `primary_track`
- `alternate_track`
- `problem_definition`
- `scientific_object`
- `current_method_failure_mode`
- `prior_work`
- `why_this_matters`
- `why_now`
- `falsifiable_success_criteria`
- `non_goals`
- `evidence_basis`
- `feasible_direction_families`
- `feasibility_constraints`

If one of these fields is still missing, the problem is not yet ready for
`create_research_idea(...)`.

For `prior_work`, prefer 3 compact parts:

1. `What prior work already established`
2. `What it still cannot cleanly answer`
3. `What improvement space remains`

### If the user wants candidate directions

For each direction, give:

1. `Problem`
2. `Why It Is Still Open`
3. `Why This Matters`
4. `What Success Would Mean`
5. `Why This Fits biology-journal` or `Why This Fits ml-topconf`
6. `Main Risk`

If the user also wants possible algorithm directions, add:

7. `Plausible Idea Directions`
   - 1-5 flat bullets only
   - if more than one, rank them by feasibility
   - each bullet should use:
     - `Direction family | Why feasible now | Needed signal / assumption | Main risk`

### If the user wants to sharpen one direction

Return:

1. one-sentence problem statement
2. the exact ambiguity/gap
3. the audience-level motivation
4. a falsifiable success claim
5. what not to claim

## Hard Filters

Reject or down-rank directions that are mainly:

- an algorithm in search of a problem
- a benchmark win without a new scientific object
- a generic integration/annotation task with no dynamic question
- a descriptive atlas question with no ambiguity that algorithm resolves
- a domain-specific application with no reusable methodological tension
- a topconf pitch that depends entirely on one biology story
- broad enough that the likely next step is still "pick an actual question"
- unable to say what prior work already achieved and what space is still open

Reject or down-rank idea directions that are mainly:

- impossible without unavailable supervision or side information
- "just combine OT + FM + SB + GNN" without a specific ambiguity target
- too broad to imply any concrete object of inference
- dependent on oracle knowledge that real datasets do not provide
- merely renaming a standard model family without a new role in the question
- centered on perturbation data or intervention labels unless the user explicitly asks to expand scope
- not ranked by feasibility when multiple directions are offered

## Taste Rules

- Prefer questions that reduce wrong explanations, not just improve fit.
- Prefer questions that become possible because of a new data regime or a newly formalized ambiguity.
- Prefer one sharp claim over a broad "virtual cell" narrative.
- Prefer questions where failure modes are intellectually interesting.
- Prefer framing that survives even if the eventual model family changes.
- Prefer a few feasible idea families over many decorative ones.
- Prefer directions that explicitly say what extra signal makes them possible.
- Prefer questions that one algorithm cycle could plausibly attack without redefining the scientific object.

## Handoff

After the problem is sharp:

- use `idea-lifecycle` to store and review the idea as a persistent problem asset
- use `proposal-theory` for recoverability logic
- use `literature-citation` if the framing needs tightly cited support
- use `algorithm-orchestrator` only after the question and motivation are stable

If bounded idea directions were produced:

- keep them at the level of algorithm families and enabling assumptions
- do not commit to one until theory review starts
