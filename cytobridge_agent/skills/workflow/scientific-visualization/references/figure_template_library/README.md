# Figure Template Library

## Goal

Build a reusable, dataset-agnostic figure-template library by auditing good published figure code and figure grammars, then extracting what is worth standardizing into:

- a **decision tree** for choosing plot types
- an **inventory** of proven figure grammars
- a **starter template set** that can be adapted to new datasets

This first pass is based on two published codebases already available locally:

- `scDiffEq`
- `moslin`
- `moscot`
- `cytobridge-downstream`
- `single-cell-best-practices`
- `ipop_aging`

It now also includes a broader audit of published high-tier papers across Nature Methods,
Nature Biotechnology, Nature, Nature Genetics, and related top venues.

The point is **not** to copy their figures. The point is to extract the parts that are:

- logically reusable
- visually proven in manuscripts
- generic enough to survive across datasets

## What This Folder Contains

- `inventory_scdiffeq.md`
  - audit of what `scDiffEq` actually plots, where the code lives, and what is reusable
- `inventory_moslin.md`
  - same audit for `moslin`
- `inventory_moscot.md`
  - same audit for `moscot`
- `inventory_cytobridge_downstream.md`
  - same audit for `cytobridge-downstream`
- `inventory_single_cell_best_practices.md`
  - audit of what `single-cell-best-practices` contributes as a plotting reference
- `inventory_ipop_aging.md`
  - audit of what `ipop_aging` contributes as a plotting reference
- `inventory_squidpy.md`
  - source-backed audit for spatial neighborhood/domain plot classes
- `inventory_cellrank.md`
  - source-backed audit for lineage/fate-trend plot classes
- `inventory_pertpy_milo.md`
  - source-backed audit for DA/compositional plot classes
- `inventory_high_tier_papers.md`
  - cross-paper audit of >=20 published high-tier computational biology papers and the figure families worth reusing
- `high_tier_style_synthesis.md`
  - cross-paper summary of what repeatedly makes similar results look richer and more publication-grade
- `repo_code_style_notes.md`
  - concrete repository-level rendering habits and the template implications they suggest
- `hero_panel_audit.md`
  - main-figure top-half layout patterns, hero-panel archetypes, and memorable published examples
- `result_figure_layout_audit.md`
  - how strong published result figures arrange zones, vary grammar, and keep dense pages coherent
- `main_result_figure_20paper_audit.md`
  - detailed page-level audit of 20 published result-figure sets, including which skeletons are actually worth copying
- `layout_templates.md`
  - concrete reusable page skeletons such as asymmetric hero pages, compare bands, matrix-led pages, and spatial walls
- `page_layout_skeletons.py`
  - executable matplotlib page skeletons for high-frequency main-result layouts
- `decision_tree.md`
  - strict panel-selection logic: what to draw on a manifold, what not to, when to switch encodings
- `template_taxonomy.md`
  - grouped view of reusable figure archetypes
- `style_playbook.md`
  - per-template beautification logic distilled from the source papers
- `palette_bank.md`
  - shared palette classes and when each one should be used
- `axis_policy.md`
  - hard decision rules for when axes/spines/ticks should be kept or removed
- `panel_finish_checklist.md`
  - last-pass QA checklist for typography, spacing, legends, clipping, and export safety
- `style_defaults.py`
  - conservative shared matplotlib defaults and finish helpers for common panel classes
- `palette_bank.py`
  - shared categorical, perturbation, scalar, matrix, and benchmark palettes
- `style_presets.py`
  - stronger page-level presets for typography, colorbar placement, panel letters, and final page finish
- `style_variants.md`
  - where similar plot classes use different style strategies across repos
- `template_inputs_and_usage.md`
  - required inputs, optional inputs, and applicability rules for each template
- `native_scverse_playbook.md`
  - when to keep `scanpy/scvelo/cellrank` native plotting instead of reimplementing
- `template_gap_backlog.md`
  - missing template classes and the best published sources to mine next
- `templates/`
  - generic starter scripts and one R template

## Important Distinction

The starter templates in `templates/` are generic abstractions, not one-to-one copies of the
original manuscript plotting code.

The paper-specific beautification logic is documented explicitly in:

- `style_playbook.md`

So the library now contains both:

- generic code skeletons
- source-informed styling rules
- source-informed palette classes
- harder defaults for typography / axes / finishing
- a cross-paper style synthesis grounded in top-tier published figures

## Dependency Rule

The template library should **not** depend on source-repo runtime packages such as:

- `scdiffeq`
- `moslin`
- `moscot`

Those repositories are treated as **style and grammar references**, not as plotting backends.

Preferred template dependencies are:

- `matplotlib`
- `seaborn`
- `numpy`
- `pandas`

Accepted optional scientific-plotting dependencies are:

- `anndata`
- `scanpy`
- `scvelo`
- `squidpy`
- `cellrank`
- `pertpy`
- `networkx`
- `statsmodels`

These are acceptable when the template truly needs ecosystem-native plotting or preprocessing
behavior and the dependency is broadly used beyond one project.

Optional exceptions must be marked explicitly. At the moment, the only such exception is:

- `templates/template_alluvial_coupling.R`
  - depends on `ggplot2` and `ggalluvial`
  - should be treated as an optional reference template, not the default cross-project backend

In other words:

- inspect published code to learn the plotting grammar and beautification logic
- re-express that grammar using general-purpose plotting libraries whenever possible
- do not build templates that require the original algorithm package or its object model

## Core Principle

Do **not** standardize around a fixed number of panels.

Standardize around:

1. biological object
2. coordinate basis
3. full-view vs ROI decision
4. whether the next panel is still geometric, or already statistical / mechanistic / perturbational
5. whether the encoding is repeating the previous panel

## What Counts as a Good Template

A plotting template is worth keeping only if it meets all of these:

- can be described without referring to one dataset
- can take column names / embedding names / grouping names as arguments
- is generated from script, not from manual collage
- still reads clearly with different labels and different scales
- does not require a source-paper algorithm package just to render the figure

## What Does **Not** Count

These should not become core shared templates:

- notebook-only one-off figures whose logic is mostly manual filtering
- figures that are basically tables with colored cells
- panels that only work after a long verbal explanation
- dataset-specific artistic layouts that do not generalize

## First-Pass Recommended Templates

The strongest extraction targets from the current audit are:

1. stylish box + jitter benchmark panel
2. manifold overlay panel
3. velocity-stream panel
4. temporal ribbons / stacked composition panel
5. perturbation volcano panel
6. perturbation dose-response panel
7. sweep benchmark heatmap + grouped bar panel
8. alluvial transition panel
9. spatial small-multiples panel
10. bubble summary panel
11. heatmap + curve companion panel
12. communication / attention heatmap panel
13. embedding + generated-trajectory overlay panel
14. perturbation embedding compare panel
15. embedding time-overlay panel
16. rich scatter/line/facet panel
17. annotated block heatmap panel
18. bipartite pathway network panel
19. labeled network summary panel

## When Local Templates Still Feel Thin

If a panel still looks ordinary after following the normal decision tree and style docs, read:

- `inventory_high_tier_papers.md`
- `high_tier_style_synthesis.md`

before inventing more local heuristics.

## Next Expansion Targets

After this first pass, the most useful additions would be:

- spatial overview template
- latent-regime overlay template
- aligned dynamics tracks template
- compact distribution triptych template
- native CellRank fate/trend wrappers
- native Squidpy interaction/co-occurrence wrappers
