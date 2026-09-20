# Hero Panel Audit

## Purpose

This note focuses on the part the earlier template library was weakest at:

- how top-tier papers organize the **top half** of a main figure
- what makes one panel feel like a true **hero panel**
- which panels are especially memorable or more distinctive than routine field defaults

This audit was informed by directly reviewing the top part of main figure pages from the local
PDF corpus and pairing that with repository-level plotting/code review.

It should now be read together with:

- `main_result_figure_20paper_audit.md`

The newer audit broadens the evidence from a smaller top-half sample to a larger page-level set.

## Main Observation

The strongest main figures do **not** usually begin with one lonely pretty embedding.

They begin with a **hero zone** that does one of these:

1. establishes the whole object and the modeling setup at once
2. shows a geometry panel plus an immediately adjacent quantitative payoff
3. compresses a workflow and the decisive result into one integrated top row

That is why many published main figures feel “full” before the reader even reaches the lower rows.

## Hero Panel Archetypes

### 1. Workflow + decisive example

Common in:

- `Learning single-cell perturbation responses using neural optimal transport`
- `Mapping cells through time and space with moscot`
- `Optimal-Transport Analysis of Single-Cell Gene Expression...`

Characteristics:

- cartoon/workflow across the top or left
- one or two real data examples directly attached
- enough numeric evidence nearby that the panel does not feel like a concept slide

When to reuse:

- when the method framing is essential to understanding the biological result
- when the paper needs one “how this works + why it matters” entry point

### 2. Shared-basis compare row

Common in:

- `Generalizing RNA velocity...`
- `CellRank 2`
- trajectory/comparison papers more broadly

Characteristics:

- several children on the same basis
- same crop/aspect/point treatment
- one child is usually the focal method or focal state
- often paired with a small right-side or lower quantitative block

When to reuse:

- method comparison
- clone truth vs model vs baseline
- measured vs predicted

### 3. Geometry hero + quantitative companion

Common in:

- `Tangram`
- `Cell2location`
- many spatial/transcriptomic mapping papers

Characteristics:

- one large geometry/spatial panel earns the attention
- companion block gives the claim teeth immediately
- geometry alone is not trusted to do all the work

When to reuse:

- one basis is visually strong and biologically meaningful
- but the page would feel too thin if it stopped there

### 4. Structured dashboard hero

Common in:

- `Benchmarking atlas-level data integration in single-cell genomics`
- `Pertpy`
- some multiview or benchmark-heavy papers

Characteristics:

- many smaller units, but one very disciplined grid
- not scrapbook-like
- repeated glyphs, repeated scales, repeated typography

When to reuse:

- benchmark or toolkit papers where no single manifold can carry the whole story

## Which Hero Panels Felt Most Impressive

### Most distinctive

#### `Learning single-cell perturbation responses using neural optimal transport`

Why it stands out:

- does not rely on one grammar
- combines shared-basis embeddings, perturbation displays, and intervention summaries
- top-half layout already feels like a story, not a panel inventory

What to learn:

- build a hero zone out of **coupled panel families**, not one isolated plot

#### `Mapping cells through time and space with moscot`

Why it stands out:

- workflow framing and data display are integrated
- transport/coupling is shown in ways that feel process-oriented, not just matrix-oriented

What to learn:

- when the method object is itself visual, use it to structure the hero zone

#### `Generalizing RNA velocity to transient cell states through dynamical modeling`

Why it stands out:

- embedding panels are not thin because they are tied to kinetic/support panels very early
- geometry and dynamics are coupled tightly

What to learn:

- a hero panel can be driven by manifold structure, but only if the manifold is immediately tied to the process claim

### Most useful, even if less flashy

#### `CellRank 2`

Why it matters:

- very disciplined about shared-basis compare and multiview support
- a good model for method papers that still need biology credibility

#### `Tangram` / `Cell2location`

Why they matter:

- strong examples of spatial compare grammar
- good at making predicted-vs-observed panels feel controlled and publication-safe

## One More Lesson From The Larger Audit

Some published papers are scientifically excellent but still have weak or overtextual hero zones.

So the practical rule is:

- copy the **strong hero archetype**
- not just any panel from a famous paper

## What Makes a Hero Panel Fail

A top panel often fails when:

- it is just one embedding with no payoff nearby
- it repeats a lower-row grammar too early
- it needs too much caption explanation
- it uses many small panels without one organizing dominant structure

## Main-Figure Layout Rules To Reuse

### Rule 1

The top half of a main figure should define a **hero zone**, not just `A`, `B`, `C` as separate chores.

### Rule 2

The hero zone usually needs **one dominant visual object** plus **one immediate evidence companion**.

### Rule 3

If a page begins with a manifold/spatial hero, the next adjacent panel should usually be:

- compare
- benchmark
- distribution
- or compact mechanism support

not another manifold of the same kind.

### Rule 4

If the main figure is benchmark-heavy, make the top half a disciplined dashboard rather than pretending one small chart is a hero.

### Rule 5

The most memorable hero panels usually mix **at least two display grammars**:

- geometry + statistics
- workflow + real data
- compare + quantitative summary
- spatial map + zoom or companion bar/distribution

## Practical Instruction For The Skill

When planning a main figure, ask before drawing:

1. What is the hero zone?
2. What is the dominant object in that zone?
3. What companion panel makes the claim feel earned immediately?
4. Is the top half more like:
   - workflow + example
   - shared-basis compare
   - geometry + quantitative companion
   - structured dashboard

If none of these are true, the top half is probably still underdesigned.
