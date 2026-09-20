# ML Top-Conference Framing

Use this file when the target is NeurIPS, ICML, or ICLR.

The center of gravity is:

- a reusable methodological gap
- a crisp theoretical object
- a general algorithmic or statistical tension

The biology can motivate the problem, but it cannot be the only reason the problem matters.

Default CytoBridge scope here is:

- multi-timepoint snapshot single-cell omics
- optional multimodal measurements
- no perturbation-centered framing unless the user explicitly expands scope

## What a strong ML-topconf question looks like

Good topconf questions usually have this form:

- what exact object is unidentifiable or badly specified under current assumptions?
- what important setting is missing from the current formal toolbox?
- what classes of methods systematically confound two mechanisms?
- what practically important problem lacks a stable, scalable, or simulation-free solver?

## Strong novelty units

Best candidates usually fall into one of these:

1. Identifiability of snapshot dynamics.
2. Partial observability / latent-state trajectory inference.
3. Unified unbalanced + stochastic + interaction-aware dynamics.
4. Geometry-aware flow or bridge learning.
5. Multi-view or multimodal constraints for otherwise ill-posed snapshot dynamics.
6. Model selection and routing: when should one use balanced, unbalanced, latent, or interaction-aware models?
7. Uncertainty and refusal for extrapolative dynamics.
8. Scalable solvers for RUOT / SB / FM / mean-field problems.

## Framing principles

### 1. State the object exactly

Name what is being learned or recovered:

- path measure
- time-indexed marginals
- coupling family
- growth field
- latent state
- interaction operator
- geometry-aware vector field
- uncertainty region

If the object is vague, the problem is not topconf-ready.

The object should also stay narrow enough that one paper can define and test it cleanly under the intended snapshot regime.

### 2. State the assumption gap

A strong ML framing says what current methods assume that is too strong:

- full observability
- mass conservation
- Euclidean geometry
- single-condition stationarity
- no interaction
- no measurement mismatch

### 3. Make the gap reusable beyond one biology story

The question should still make sense for:

- scientific time-series
- population dynamics
- controlled stochastic systems
- spatiotemporal generative modeling

If the framing collapses outside one biology dataset, it is probably too narrow for topconf positioning.

### 4. Separate "new setting" from "new backbone"

Topconf-friendly novelty is often:

- a new formal setting
- a sharper impossibility/possibility boundary
- a cleaner objective
- a solver with a new tradeoff

It is usually weaker when framed as:

- "use model X on problem Y"
- "combine A and B"
- "add a module for C"

It is also weaker when the formal problem silently depends on intervention labels that the baseline target regime does not have.

### 5. Force benchmarkability

A strong ML-topconf question should imply:

- what kind of task family it defines
- what failure of current methods it predicts
- what evaluation would falsify the claim

If the success criterion is only qualitative, the problem is under-formed.

## A good ML-topconf question should survive these checks

1. Can the gap be stated without naming the proposed model class?
2. Is the gap general enough to matter outside one biology use case?
3. Does the question expose a structural limitation rather than a small weakness?
4. Would solving it change the way people formulate future methods?

## Common failure modes

- The pitch is just a biology story with ML decoration.
- The contribution is really engineering, but framed as theory.
- The setting is too broad and not falsifiable.
- The gap is only "current methods are inaccurate."
- The question depends on stacking buzzword modules.

## If proposing idea directions for this track

Only propose a few direction families, not a detailed architecture.

The most acceptable direction families here are those that correspond to a clean methodological move, for example:

- reformulating the inference object under weaker assumptions
- introducing a new formal setting such as partial observability or interaction-aware unbalanced dynamics
- defining a cleaner objective that separates confounded mechanisms
- building a scalable or simulation-free solver for an already meaningful object
- adding geometry, uncertainty, or routing in a way that changes the formal problem class
- exploiting multi-view or multimodal snapshot information to narrow an otherwise unidentifiable object

For each direction, say:

- what exact formal object it would operate on
- why it is feasible now under sparse multi-timepoint snapshot data
- what assumption it relaxes or replaces
- why the move is reusable beyond one biology setting
- what makes it plausibly solvable rather than purely aspirational

If there are multiple directions, rank them by feasibility.

Preferred line format:

- `Direction family | Why feasible now | Needed signal / assumption | Main risk`

### Feasibility filter for ML-topconf directions

Do not propose directions that require:

- solving an obviously ill-posed problem with no extra assumptions
- proving theorems for an object that has not been cleanly defined
- combining many model families without a single central methodological move
- biological supervision that makes the method non-general
- perturbation or intervention labels when the default scope is non-perturbation snapshot dynamics

Down-rank directions whose only novelty is complexity or scale.

## Output style for this track

When proposing directions, emphasize:

- exact formal object
- which assumption breaks
- why the gap is structural
- why it matters beyond single-cell
- what kind of theoretical or algorithmic advance would close it
- what would make the claim strong enough for a top conference
- if ideating, which solution families are plausible and what minimum assumptions make them tractable
