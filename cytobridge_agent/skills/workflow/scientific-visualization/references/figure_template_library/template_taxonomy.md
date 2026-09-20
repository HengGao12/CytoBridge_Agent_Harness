# Template Taxonomy

This taxonomy defines the first-pass reusable figure templates derived from `scDiffEq`, `moslin`,
`moscot`, `cytobridge-downstream`, `ipop_aging`, and **generative trajectory models** (e.g.,
Latent Flow Matching, scDiffEq trajectory outputs).

For beautification rules, read:

- `style_playbook.md`

## Tier 1: Core templates

These are strong enough to reuse across many datasets.

### T1. Stylish box + jitter

Inspired by:

- `scDiffEq` benchmark panels

Use for:

- method benchmarks
- replicate-level summary metrics
- compact support comparisons

Why core:

- generic
- strong signal density
- main-figure compatible
- strong source precedent for layered box styling

### T2. Manifold overlay panel

Inspired by:

- `scDiffEq` simulated trajectory overlays
- `moslin` coupling overlays

Use for:

- global context
- overlay trajectories or couplings
- compare observed background with selected foreground
- quiet-background / strong-foreground hierarchy

### T3. Velocity stream panel

Inspired by:

- `scDiffEq` velocity-stream panels

Use for:

- vector field / flow field display on a fixed manifold
- quantile-clipped scalar overlays when needed

### T4. Temporal ribbons

Inspired by:

- `scDiffEq` temporal composition plots

Use for:

- stacked category changes across time

### T5. Volcano panel

Inspired by:

- `scDiffEq` perturbation-screen volcanoes

Use for:

- many perturbation hits with effect size and significance
- sparse-label, symmetric-sign styling

### T5b. Perturbation dose-response

Inspired by:

- `scDiffEq` ensemble perturbation panels

Use for:

- designed perturbation magnitudes
- module or TF ensemble scans
- endpoint fraction or composition changes across signed perturbation strengths

Why core:

- more appropriate than a volcano when there are only a handful of designed directions
- high manuscript precedent in perturbation figures

### T6. Sweep heatmap + grouped benchmark bars

Inspired by:

- `moslin` grid-search visualizations

Use for:

- hyperparameter searches
- per-timepoint benchmark sweeps
- explicit best-cell highlight rule

### T7. Transition alluvial

Inspired by:

- `moslin` zebrafish alluvial plots

Use for:

- state-to-state transition summaries
- lineage or coupling mass redistribution

### T8. Transition heatmap

Inspired by:

- `moscot` cell-transition plots

Use for:

- aggregate transition matrices
- annotated state-to-state transition summaries
- row/column category comparison

### T9. Spatial small multiples

Inspired by:

- `cytobridge-downstream` spatial snapshot and comparison panels

Use for:

- spatial slices across time
- observed vs generated spatial panels
- compact spatial supplement rows or columns

### T10. Bubble summary

Inspired by:

- `cytobridge-downstream` growth-interaction bubble panels

Use for:

- size-coded group summaries
- one metric vs another metric with color = time or condition

### T11. Heatmap + curve companion

Inspired by:

- `cytobridge-downstream` gene-pattern summary panels

Use for:

- left-right panels where a matrix view needs a matched dynamic summary

### T12. Communication / attention heatmap

Inspired by:

- `cytobridge-downstream` attention and asymmetry heatmaps

Use for:

- sender-receiver interaction matrices
- positive-only attention summaries
- signed asymmetry matrices with optional significance markers

### T13. Embedding + generated-trajectory overlay

Inspired by:

- `scDiffEq` simulated-trajectory overlays

Use for:

- fixed embedding basis plus generated rollouts
- learned trajectory display on top of a manifold or latent basis
- quiet background with multiple semi-transparent generated paths

### T13b. Perturbation embedding compare

Inspired by:

- `scDiffEq` perturbation projected onto a shared UMAP basis

Use for:

- side-by-side `control` vs `perturbed` panels on the same embedding
- perturbation outcomes that should stay anchored to a fixed manifold
- optional contour emphasis for the shifted fate region

### T13c. Embedding time-overlay

Inspired by:

- `scDiffEq` time-colored simulated rollouts on a fixed UMAP basis

Use for:

- simulated cells or rollout states projected onto one embedding
- continuous time encoded by color
- optional observed-lineage overlay and start-state marker

### T14. Rich scatter/line/facet panel

Inspired by:

- `ipop_aging` longitudinal and cross-omic trend summaries

Use for:

- per-feature or per-subject trend lines
- faceted longitudinal summaries
- baseline-centered trajectories with strong manual palette control

### T15. Annotated block heatmap

Inspired by:

- `ipop_aging` ComplexHeatmap panels

Use for:

- grouped matrices with row/column block annotations
- heatmaps where separators and side strips matter as much as the color field

### T16. Bipartite pathway network panel

Inspired by:

- `ipop_aging` pathway/module network panels

Use for:

- multi-class bipartite or multipartite manuscript network summaries
- node-shape plus fill plus size encoding

### T17. Labeled network summary

Inspired by:

- `ipop_aging` cluster-summary graphs

Use for:

- compact labeled correlation or association networks
- muted edges plus emphasized nodes plus readable text backing

### T18. Trajectory divergence panel (NEW)

Inspired by:

- CREST main figure Panel D (lineage tracing Latent Flow Matching project)

Use for:

- time-sliced visualization of how generated trajectories separate into different fates
- demonstrating a generative model's ability to produce continuous, interpretable dynamics
- Early/Mid/Late panels on a shared embedding with KDE contours per fate

Why core:

- essential for any generative trajectory/dynamics model paper
- immediately demonstrates the model's central claim (trajectory generation)
- strong visual impact with quiet background + colored fate divergence

### T19. Along-trajectory gene dynamics heatmap (NEW)

Inspired by:

- CREST main figure Panel E (lineage tracing Latent Flow Matching project)

Use for:

- showing how gene expression changes along a generated trajectory
- z-scored expression × pseudo-time heatmaps for a selected cell subset
- revealing temporal gene activation/deactivation patterns

Why core:

- connects trajectory generation to downstream biological interpretation
- common in trajectory inference and generative model papers
- complements trajectory divergence panels by switching from geometry to molecular dynamics

### T20. Regulatory circuit diagram (NEW)

Inspired by:

- CREST main figure Panel F (Jacobian perturbation analysis)

Use for:

- module-to-module influence networks derived from Jacobian or perturbation analysis
- directed regulatory diagrams with activation/suppression arrows
- small multiples comparing regulatory states across conditions or time

Why core:

- fills a gap between static pathway networks (T16, T17) and dynamic gene-level analysis
- critical for papers that perform in-silico perturbation or Jacobian analysis on learned models
- node-arrow format is intuitive and space-efficient

### T21. Fate score gallery (NEW)

Inspired by:

- CREST main figure Panel B (lineage tracing Latent Flow Matching project)

Use for:

- small multiples of the same UMAP, each highlighting one cell type with scalar fate overlay
- comparing model-predicted fate probabilities across cell states
- gallery-style presentation of per-type prediction strength

Why core:

- common pattern in fate prediction and optimal transport papers
- reuses the quiet-background / strong-foreground hierarchy
- generalizes beyond a single scalar overlay to per-type galleries

## Tier 2: Secondary templates

These are useful, but usually support panels or supplement templates.

### S1. Fate/probability clustermap

Inspired by:

- `scDiffEq` fate-bias clustermaps

### S2. Metric/loss curves

Inspired by:

- `scDiffEq` fit-loss helpers

### S3. Validation scatter dashboard

Inspired by:

- `moslin` hyperparameter validation plots

### S4. Simulation tree/cost panels

Inspired by:

- `moslin` TedSim analysis

### S5. Sankey transport summary

Inspired by:

- `moscot` sankey plots

## Tier 3: Scaffold patterns, not plotting-library patterns

These should be standardized as **analysis scaffolds**, not as reusable drawing functions.

### P1. CellRank comparison notebooks

Reason:

- logic is mostly in state-definition and analysis setup, not the plotting code

### P2. Dataset-specific latent / founder selection notebooks

Reason:

- reusable idea
- low code reuse

### P3. Tutorial-native scverse plotting books

Inspired by:

- `single-cell-best-practices`

Reason:

- strong source of native `scanpy/scvelo/squidpy` plot usage patterns
- weak source of bespoke reusable plotting helper code

### P4. Native CellRank and Squidpy wrappers

Reason:

- the ecosystem-native plotting function is already canonical
- a thin wrapper is better than over-reimplementing a stable plot class

## Selection Heuristic

When building a new main figure, prefer:

1. one or two Tier 1 anchor templates
2. one Tier 1 or Tier 2 support template of a different grammar
3. avoid repeating the same template class in neighboring panels unless the biological object changes

For **methods papers with generative models**, a strong combination is:

1. T2 or T21 (manifold overview / fate gallery) as the hero panel
2. T18 (trajectory divergence) to demonstrate the model's generative capability
3. T19 (gene dynamics) or T20 (regulatory circuit) for mechanistic interpretation
4. T1 or Family E (box/violin) for quantitative validation

### When no template fits

If none of the templates match the panel's message, enter **custom visualization mode**
(see SKILL.md). Build from matplotlib primitives while following global style rules.
After the panel is approved and stable, consider extracting it as a new template (T22+).
