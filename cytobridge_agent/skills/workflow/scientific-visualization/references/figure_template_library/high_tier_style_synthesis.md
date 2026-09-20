# High-Tier Style Synthesis

## Purpose

This note distills what high-tier computational biology papers consistently do that makes
similar results feel more substantial and more publication-grade than an ordinary lab plot.

This is not about copying one journal house style.
It is about extracting reusable display tactics from many good papers and codebases.

## The Core Difference

The gap is usually **not** one palette or one font.
The gap is that stronger papers do three things at once:

1. choose the right plotting grammar for the claim
2. pair geometry with quantitative support early
3. control visual hierarchy aggressively

Poorer figures often fail at step 1 and then try to compensate with styling.

## Repeated High-Tier Tactics

### A. Do not let a geometry panel stand alone

Embedding, latent, and spatial maps usually look thin when shown alone.
High-tier figures often pair them with:

- a benchmark block
- a distribution panel
- a method compare row
- a compact quantitative inset/companion panel

This is one reason the same UMAP-like result can feel much richer in a published paper.

### B. Use a quiet background and a deliberate foreground

Many strong figures do **not** saturate every point equally.
Typical pattern:

- background: grey or low-alpha context
- foreground: one or two highlighted groups, trajectories, predictions, or states

This gives the panel both structure and focus.

### C. Richness comes from grammar variation across rows

A professional figure page rarely repeats the same manifold grammar three times.
Instead:

- row 1: global context / compare
- row 2: hidden structure / mechanism
- row 3: perturbation / validation / support

The page feels rich because each row answers a different question using a different encoding.

### D. Companion quantitative blocks are compact and geometric

Better papers do not scatter tiny charts randomly.
They group related summaries into one compact geometry:

- multi-metric benchmark block
- aligned violins
- aligned time tracks
- heatmap + side bar

This makes the figure feel intentional instead of assembled.

### E. Statistical plots show spread, not only means

High-tier statistical panels often prefer:

- box + jitter
- boxen
- violin + points
- ribbon/band around trend lines

Overly flat bars usually make the page feel cheap unless the effect is extremely simple.

### F. Spatial panels are disciplined about comparability

Strong spatial papers usually standardize:

- crop
- point/spot size
- color scale
- aspect
- frame removal

Across a whole compare block.
This is one of the clearest gaps between polished papers and ad hoc project figures.

### G. Heatmaps become good when the matrix is not alone

Good heatmaps frequently add:

- cluster strips
- block boundaries
- companion bars
- annotation tracks
- selective labels only

The high-tier version is usually a *matrix + context* object, not just a colored table.

### H. Perturbation panels are semantically simple

The strongest perturbation figures usually use:

- dose-response boxes/bars
- volcano, when many hits exist
- before/after embedding compare

They do not rely on obscure encodings that require caption-heavy explanation.

## Concrete Display Rules To Reuse

### For embeddings / manifolds

- equal aspect unless the basis itself is anisotropic by design
- no frame unless axes carry scientific meaning
- fixed crop across compare children
- background context low-alpha, foreground deliberate

### For distributions

- show spread
- keep sibling panel widths equal
- do not use three different x-ranges unless needed
- if panel is narrow, avoid legends and verbose titles

### For time panels

- one shared time axis whenever the signals belong to the same process
- use median + IQR or mean + CI consistently
- reference line at zero only when semantically useful

### For bars / perturbation

- symmetric positive/negative layout is strong when direction matters
- anchor control clearly
- keep the sign semantics obvious without long text

### For heatmaps

- cap label density early
- add annotation instead of adding more text
- reserve heatmaps for genuine matrices, not disguised tables of few values

## Why Earlier Project Figures Still Felt Thin

The main failure mode was not only style.
It was that panels were often:

- too isolated
- too uniform in grammar
- too dependent on caption interpretation
- or too table-like in mechanism sections

The high-tier audit suggests the fix is:

- better panel pairing
- more deliberate row structure
- stronger foreground/background separation
- more use of compact quantitative companions

## Practical Rule

When a figure still feels underpowered after light styling, do **not** immediately add more color
or typography.

Instead ask:

1. Does this panel need a companion quantitative block?
2. Is this the same grammar as the previous row?
3. Is the background too loud?
4. Is the figure showing spread, uncertainty, or only central tendency?
5. Is the page too homogeneous?
