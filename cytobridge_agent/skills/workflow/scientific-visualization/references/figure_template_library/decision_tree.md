# Main Figure Decision Tree

This is the strict first-pass decision tree for choosing what kind of panel to draw.

Do **not** start from:

- panel count
- favorite plot type
- what looked fancy in another paper

Start from the biological object and the claim.

## Step 0. Is the claim clear?

Before entering the tree, ask: **can I state the panel's claim in one sentence?**

- If yes, proceed to Step 1.
- If no, you are in **iterative claim discovery** mode:
  1. Generate a quick exploratory draft to help the user see the data.
  2. After the user reacts, refine the claim and re-enter at Step 1.
  3. If the user questions a scientific conclusion, **pause figure work** and investigate the
     upstream analysis code before redrawing. Do not cycle through encodings hoping the problem
     is visual—it may be analytical.
  4. Keep a record of what was tried and why it was rejected.

This is normal for methods papers where the "story" co-evolves with the visualization.

## Precondition

Before choosing a panel family, check one implementation constraint:

- the final reusable template should be renderable from a general plotting stack
- do not make the template depend on `scdiffeq`, `moslin`, `moscot`, or another paper-specific package
- use those repositories to learn the grammar, layer order, palette logic, annotation strategy, and
  beautification rules
- then reimplement the plotting pattern with general-purpose plotting tools

## Step 1. What is the biological object?

Choose one:

- full system
- founder cloud / progenitor pool
- local branch / ROI
- terminal states
- generated trajectories / model outputs
- perturbation screen
- regulatory network / module dynamics
- molecular support summary

If you cannot name the object in one line, do not draw yet.

## Step 2. What is the most natural basis?

Choose one:

- observed manifold
  - `UMAP`, `SPRING`, `PHATE`, `PCA`, etc.
- learned latent basis
  - latent `UMAP`, latent `t-SNE`, latent PCA
- pseudo-time / trajectory time
  - along a generated or inferred trajectory
- spatial coordinates
- schematic / diagrammatic
  - node-arrow layouts, circuit diagrams (no coordinate basis from data)
- no coordinate basis
  - benchmark summaries, perturbation summaries, distributions

### Rule

Use the observed basis first when the claim is about **data geometry**.

Use the latent basis only when the claim is about **hidden structure learned by the model**.

Use pseudo-time when the claim is about **how quantities change along a trajectory**.

Use schematic layout when the claim is about **relationships between entities** (modules,
genes, pathways) rather than spatial or manifold positions.

Do not force a coordinate plot when the object is already a summary statistic.

## Step 3. Full view or local view?

### Use a full view when:

- this is the first time the dataset appears
- the reader needs orientation
- the local region has not yet been established

### Use a local view when:

- a previous panel already established the full basis
- the current claim is specifically about one branch or ROI

### Use full + ROI when:

- the local result matters, but its location is not obvious

## Step 4. Is this still a geometry question?

If the panel is answering:

- where something sits
- how methods differ in the same space
- what local region is being zoomed
- how generated trajectories diverge over time on a shared embedding

then keep using manifold / latent / spatial coordinates.

If the panel is answering:

- how scores differ
- how modules differ
- how gene expression changes along a trajectory
- how regulatory influence changes over time
- how perturbation changes outcomes

then stop drawing on a coordinate basis and switch encodings.

## Step 5. Did the previous panel already use the same encoding for the same object?

If yes, do **not** just recolor it.

Change the grammar.

### Good grammar switches

- manifold -> violin / ridge
- manifold -> aligned dynamics tracks
- manifold -> dose-response or perturbation summary
- manifold -> along-trajectory heatmap
- compare row -> distribution summary
- latent map -> module distribution
- trajectory overlay -> gene dynamics heatmap
- gene dynamics heatmap -> regulatory circuit diagram

## Step 6. Choose a panel family

### Family A. Global context

Use when the point is:

- data overview
- cell-type layout
- time progression

Typical form:

- full manifold or spatial plot

### Family B. Method comparison

Use when the point is:

- compare methods on the same biological object

Typical form:

- row of small multiples

### Family C. ROI / local branch

Use when the point is:

- local branch difference
- local coupling difference

Typical form:

- zoomed manifold panel
- ROI callout from global view

### Family D. Latent geometry

Use when the point is:

- learned hidden structure

Typical form:

- latent `t-SNE`
- latent `UMAP`
- axis projection support

### Family E. Distribution/state support

Use when the point is:

- score differences across cohorts

Typical form:

- violin
- box/boxen
- ridge
- raincloud

### Family F. Dynamics

Use when the point is:

- how multiple quantities evolve over time

Typical form:

- aligned tracks
- line + band
- stacked ribbons
- along-trajectory gene dynamics heatmap (z-scored expression × pseudo-time)

### Family G. Perturbation

Use when the point is:

- directional intervention effect

Typical form:

- dose-response bars or boxplots
- volcano, only if many hits plus significance

### Family H. Validation

Use when the point is:

- external support
- enrichment
- hyperparameter dashboard

Usually supplement, not headline main figure.

### Family I. Trajectory divergence (NEW)

Use when the point is:

- how generated trajectories separate into different fates over time
- the generative model's ability to produce continuous, interpretable cell dynamics

Typical form:

- time-sliced UMAP panels (Early/Mid/Late) with KDE contours for each fate
- shared embedding background + colored trajectory subsets

### Family J. Regulatory network / module dynamics (NEW)

Use when the point is:

- how gene modules regulate each other
- how regulatory influence changes across conditions or time
- directed module-to-module interactions (e.g., Jacobian perturbation results)

Typical form:

- circuit diagram: nodes = modules, arrows = directed influence
- small multiples across conditions (e.g., BM Early / BM Late / AL Early / AL Late)

### Family K. Fate score gallery (NEW)

Use when the point is:

- how the model assigns fate probabilities to different cell states
- per-cell-type fate prediction strength

Typical form:

- small multiples of the same UMAP, each showing one cell type with a scalar color overlay

## Step 6b. No family matches?

If none of the families above fit the panel's message, **do not force it**.

Enter **custom visualization mode** (see SKILL.md). Build from matplotlib primitives
(`fig.add_axes`, `ax.scatter`, `ax.annotate`, `matplotlib.patches`) while still following
the global style rules. After approval, consider extracting the pattern as a new template.

## Step 7. Main figure or supplement?

### Main figure if:

- directly supports the one-sentence main claim
- has external support if it is mechanistic
- visually distinct from neighboring panels

### Supplement if:

- mostly validation
- mostly parameter exploration
- mostly robustness
- mostly enrichment support
- detailed but not central

## Step 8. Final rejection checks

Reject a panel if any of these are true:

- it only becomes understandable after a long spoken explanation
- it repeats the previous panel with a different colormap
- it reads like a spreadsheet rather than a figure
- it is too sparse for the space it occupies
- it is not script-traceable to raw inputs
- the scientific conclusion it illustrates has not been verified against the upstream analysis
