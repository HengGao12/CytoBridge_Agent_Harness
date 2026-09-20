# Biology-Journal Framing

Use this file when the target is a biology journal or a strong methods journal whose editorial logic is:

- the paper must answer an important biological question
- the algorithm is valuable because it makes that answer possible

Default CytoBridge scope here is:

- multi-timepoint snapshot single-cell omics
- optional multimodal measurements
- no perturbation-centered idea unless the user explicitly expands scope

## What a strong biology-journal question looks like

The center of gravity is not the model. It is the biological ambiguity.

Good biology-journal questions usually have this form:

- what biological process is being misread because current analyses confound two mechanisms?
- what new assay or data regime now makes an old biological question answerable?
- what previously descriptive phenomenon can now be tested mechanistically?

## Strong novelty units

Best candidates usually fall into one of these:

1. Separate true fate conversion from growth/death selection.
2. Explain how trajectories are rewritten across baseline contexts, samples, or natural state differences without invoking intervention labels.
3. Explain how space and cell-cell interaction drive fate, not just correlate with it.
4. Recover hidden pre-commitment states from multi-omic or multi-view time snapshots.
5. Identify rare transition windows or unstable intermediate states that matter biologically.
6. Make new measurement technologies interpretable:
   - spatiotemporal transcriptomics
   - lineage tracing
   - multimodal time-course single-cell data
   - imaging plus endpoint omics

## Framing principles

### 1. Lead with the biology, not the architecture

Bad:

- "Current OT methods cannot model ..."

Better:

- "It is still unclear whether the apparent branch expansion reflects true fate conversion or selective growth."
- "Between adjacent timepoints, it is still unclear whether the emerging branch needs incoming transport at all, or whether branch-local growth already explains the occupancy change."

### 2. The question should change biological interpretation

A publishable biology-journal question usually changes one of:

- what the key transition actually is
- which cells are true precursors
- which signals drive a branch decision
- whether a shared trajectory exists across contexts
- whether an apparent fate shift is really transport, growth, or spatial selection

### 3. Tie the question to a real data regime

The framing gets stronger if the question only becomes answerable because of:

- multiple time points
- spatial context
- lineage information
- multimodal measurements
- better baseline coverage of naturally evolving systems

### 4. Force a concrete biological output

Good outputs:

- branch probability shift
- growth-versus-transport decomposition
- driver signal or interaction
- commitment timing
- conserved versus context-specific dynamic structure

Weak outputs:

- prettier embedding
- smoother pseudotime
- better cluster alignment

### 4.5 Keep the question narrow enough to be operational

A good biology-journal idea should usually name:

- one regime
  - for example: adjacent timepoints, one branching window, one spatial microenvironment, one conserved-versus-rewritten contrast
- one ambiguity
  - for example: transport versus growth, intrinsic state versus niche effect, hidden pre-commitment versus visible late marker
- one primary output
  - for example: branch-level mass budget, transition-window score, context-specific rewiring map

If the question still sounds like a whole subfield, it is not ready.

### 5. Avoid "AI for biology" vagueness

Do not pitch:

- virtual cells
- foundation models
- interpretable dynamics

unless the framing immediately says what new biological claim becomes testable.

## A good biology-journal question should survive these checks

1. If the algorithm family changed, would the question still be important?
2. Would an experimentalist care about the answer even without ML novelty?
3. Does the question resolve a real ambiguity rather than decorate a known story?
4. Would a positive result change how people interpret a biological system?

## Common failure modes

- The "problem" is really just better prediction.
- The story depends entirely on one benchmark dataset.
- The question is only computationally hard, not biologically important.
- The framing is too generic to imply any biological surprise.
- The method is the hero and the biology is an afterthought.

## If proposing idea directions for this track

Only propose a few direction families, not a full method.

The most acceptable direction families here are those that visibly help answer a biological ambiguity, for example:

- separating transport from growth/death
- separating shared backbone dynamics from context-specific rewiring without interventions
- injecting lineage or spatial constraints to reduce ambiguity
- modeling hidden pre-commitment states with multi-view information
- turning cell-cell interaction into a driver rather than a post hoc annotation
- coupling time snapshots with an experimentally grounded side signal

For each direction, say:

- what biological ambiguity it attacks
- why it is feasible now in the intended data regime
- what side information makes it viable
- why it could plausibly change biological interpretation
- what its main failure risk is

If there are multiple directions, rank them by feasibility.

Preferred line format:

- `Direction family | Why feasible now | Needed signal / assumption | Main risk`

### Feasibility filter for biology-journal directions

Do not propose directions that require:

- side information the target assay regime does not have
- unrealistically dense time resolution if the intended story is sparse snapshots
- unobservable causal variables with no proxy signal at all
- a training regime so unconstrained that no biological claim would be defensible
- perturbation labels or intervention structure when the scope is baseline multi-timepoint snapshots

Down-rank directions that would only improve fit but would not change biological interpretation.

## Output style for this track

When proposing directions, emphasize:

- unresolved biological mechanism
- why current analyses misinterpret it
- what new data regime creates the opportunity
- what claim would count as a biological advance
- what overclaim to avoid
- if ideating, which algorithm families are plausible and what biological side signal they would need
