---
name: idea-lifecycle
description: Use when the task is to create, revise, review, activate, track, or close a persistent research idea. This skill manages problem assets, not concrete algorithms.
---

# Idea Lifecycle

Use this skill when the task is about a **research idea as a durable problem asset**:

- create a new scientific question asset
- revise an existing idea without losing history
- review whether an idea is sharp enough to enter the active portfolio
- link algorithms to an idea
- update idea progress after proposal / training / analysis attempts
- decide whether an idea is still unresolved, partially resolved, resolved, or invalidated

This skill is **not** for writing a concrete algorithm.  
It sits between `research-question-framing` and `proposal-theory`.

## Core Boundary

An `idea` answers:

- what exact scientific problem is being pursued
- why it matters
- why current methods fail
- what success would mean
- what broad direction families look plausible

The idea must also be:

- narrow enough that one paper could plausibly own it
- concrete enough that one algorithm cycle could attack it
- explicit about what it will not try to solve
- grounded in the current CytoBridge scope: multi-timepoint snapshot single-cell omics, optional multimodal data, no perturbation-first framing by default
- grounded in the current CytoBridge package frame: deep-learning continuous generative dynamics rather than generic traditional bioinformatics
- explicit about prior work: what has already been done, what still fails, and what headroom remains

An `algorithm` answers:

- what exact method will solve the problem
- what mathematical object is optimized
- what implementation hooks and training plan are required

Do not collapse the two.

## Required Object Model

Every idea must stay explicit about:

- `primary_track`
  - `biology-journal` or `ml-topconf`
- `review_status`
  - draft / pending review / approved / revise / rejected / superseded
- `portfolio_status`
  - active / parked / retired
- `resolution_status`
  - unresolved / partially_resolved / resolved / invalidated
- `execution_status`
  - idle / agent_working / awaiting_review / blocked

These axes are different. Do not compress them into one vague status.

## Required Creation Flow

### Step 1. Sharpen the question first

Before creating an idea, load:

- `~/.cellcompass/skills/planner/research-question-framing/SKILL.md`
- `~/.cellcompass/skills/planner/research-question-framing/references/agent_algo_survey_20260417/reading-order.md` if the problem still needs literature grounding
- `~/.cellcompass/skills/workflow/theory-selection/references/package-capability-map.md` to understand what the package already does and where its builtin boundary still ends

Before inventing a new idea, also inspect existing research assets when available:

- `list_research_ideas(...)`
- `get_research_idea_status(...)`
- `get_algorithm_proposal_status()`
- `list_experiment_history(...)` when a likely-overlapping algorithm already exists

The idea is not ready if it still lacks:

- a precise scientific object
- a concrete current-method failure mode
- a prior-work section that states the current frontier and remaining room
- a package-boundary check showing why this is not just a builtin-family rerun
- a package-frame check showing that the problem really belongs to learned continuous dynamics rather than a static traditional bioinformatics task
- falsifiable success criteria
- a bounded scope
- at least one realistic solution direction
- feasibility-ranked directions when more than one direction is listed

At this stage, do **not** inspect low-level implementation source code by default.
Source inspection belongs later, after the problem asset is already stable enough for algorithm authoring.

### Step 2. Create the persistent idea

Use:

- `create_research_idea(...)`

Required fields are the problem asset itself:

- problem definition
- scientific object
- failure mode
- prior work / research basis
- why this matters / why now
- falsifiable success criteria
- non-goals
- evidence basis
- 1-5 feasible direction families, ranked by feasibility if there is more than one
- feasibility constraints

Do not create an idea whose direction families are just model names.
Do not create an idea whose main question is still a broad theme rather than a concrete ambiguity on a specific snapshot regime.
Do not create an idea whose novelty is asserted without first stating what prior work already covered and what exact improvement space remains.
Do not create an idea whose claimed gap disappears once you compare it against the current package capability map.
Do not create an idea whose natural solution is a traditional static bioinformatics algorithm rather than a deep continuous generative dynamics model.

### Step 3. Review before promoting

Use:

- `review_research_idea(...)`

Approval means:

- the problem is sharp enough to guide downstream method work
- the track is coherent
- the evidence basis is non-empty and traceable
- the prior-work section makes clear what previous methods already achieved and where the remaining headroom lies
- the prior-work section also makes clear what CytoBridge already implements and why this idea still exceeds that boundary
- the core problem still belongs to CytoBridge's deep-learning continuous-dynamics frame rather than generic trajectory plotting, clustering, ranking, or annotation
- the direction families are plausible under the available data regime
- the problem is narrow enough to be owned by one paper rather than a whole subfield
- the direction families are concrete enough to indicate a plausible next step

When the reviewer is uncertain about taste or scope, first calibrate using:

- `~/.cellcompass/skills/planner/research-question-framing/references/agent_algo_survey_20260417/reading-order.md`
- then the copied survey notes before falling back to broader RAG or PDF reading

Approval does **not** mean the problem is already solved.

## Revision Rules

When the scientific question changes materially, do not overwrite history in place.

Use:

- `revise_research_idea(...)`

Revise when any of these changed:

- the scientific object
- the target ambiguity
- the failure mode being claimed
- the success criteria
- the target audience / track

Do not revise just because the candidate algorithm changed.  
That is algorithm-layer drift, not necessarily idea-layer drift.

## Progress Tracking Rules

After proposal review, training, or downstream analysis, update the idea with:

- `update_research_idea_progress(...)`

Use it to record:

- what was attempted
- what evidence was produced
- whether the attempt changed belief about the idea
- what next step follows

If the attempt changes your understanding of the frontier itself, follow the progress update with:

- `revise_research_idea(...)`

Use that revision to maintain:

- `prior_work`
- the claimed failure boundary
- the remaining improvement space

Only this tool should move `resolution_status`.

## Linking Rules

If an algorithm is meant to solve an existing idea, explicitly bind it:

- `link_algorithm_to_idea(...)`

Weak association is allowed:

- ideas may exist without algorithms
- algorithms may exist without ideas
- one idea may accumulate multiple algorithm attempts over time

But once a real linkage exists, record it.  
Do not rely on memory.

## Resolution Discipline

Use these meanings strictly:

- `unresolved`
  - the question is still open
- `partially_resolved`
  - some subclaims are supported, but the full question is not yet answered
- `resolved`
  - the idea's own success criteria have been met with sufficient evidence
- `invalidated`
  - the question as framed is wrong, unanswerable under the regime, or not worth pursuing

Do not mark an idea `resolved` just because:

- one run improved metrics
- one proposal was approved
- one reviewer liked the framing

## Track-Specific Taste

### For `biology-journal`

The idea should make a biologically important ambiguity answerable:

- fate versus growth
- transport versus selection
- spatial context versus intrinsic state
- shared versus condition-specific trajectory rewiring

### For `ml-topconf`

The idea should expose a reusable formal gap:

- identifiability
- partial observability
- geometry mismatch
- unbalanced stochastic interaction modeling
- scalable inference under realistic snapshot regimes
- multimodal or multi-view constraints that narrow an otherwise ill-posed object

## Handoff

Once an idea is approved and stable:

- use `proposal-theory` to formulate a method-level solution
- use `algorithm-orchestrator` only after the problem asset is stable enough that implementation would not redefine the question

If the problem itself is still unstable:

- go back to `research-question-framing`
