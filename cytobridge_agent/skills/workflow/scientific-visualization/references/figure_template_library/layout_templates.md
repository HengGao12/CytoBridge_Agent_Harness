# Result Figure Layout Templates

## Purpose

This note gives **page skeletons** for manuscript result figures.

It is not a rigid panel-count rulebook.
It is a library of layouts that repeatedly appeared in the audited published papers.

Use this after deciding:

- the object of the page
- the coordinate basis
- whether the page is geometry-led, matrix-led, perturbation-led, or mixed
- whether the page really needs one secondary model-result zone

Read with:

- `main_result_figure_20paper_audit.md`
- `hero_panel_audit.md`
- `result_figure_layout_audit.md`

## Core Rule

Pick a layout by:

1. page skeleton
2. zone logic
3. relative weight of the hero object
4. only then raw panel count

The wrong question is:

- “Should this be 6 or 8 panels?”

The right question is:

- “Is this page an asymmetric hero page, a compare-band page, a matrix page, a spatial wall, or a multi-scale interleave page?”

## What Published Result Pages Actually Do

Across the audited papers:

- one dominant object often takes roughly `40-55%` of the page
- the immediate companion block often takes `20-30%`
- the lower support/closure band often takes `20-35%`

This is why strict equal-size grids often feel weaker than good published pages.

## Template A: Asymmetric Hero Page

Best for:

- one clear page anchor
- one immediate evidence companion
- one lower support band

Typical structure:

```text
|      A hero      | B |
|      A hero      | C |
| D | E | F |
```

Use when:

- one manifold/spatial/matrix object clearly deserves the most space
- the page should feel decisive before the reader reaches the bottom band

Good fits:

- atlas/manifold hero
- perturbation embedding hero
- matrix hero with compact right-side proof

Avoid when:

- no panel is obviously more important than the others
- the right block needs equal weight

Published analogue:

- common in CellRank-like, CellOT-like, atlas-like result pages

## Template B: Shared-Basis Compare Band + Lower Proof Band

Best for:

- trajectory/velocity/transport pages
- compare rows where children truly share crop and aspect

Typical structure:

```text
| A | B | C | D |
|     E wide / mechanism / dynamics     |
```

or

```text
| A | B | C |
| D | E | F |
```

Use when:

- the top band defines the object through same-basis comparison
- the page becomes stronger after a clear grammar shift

Good fits:

- clone truth vs model vs baseline
- measured vs predicted
- multiple kernels/methods on one UMAP or one spatial crop

Avoid when:

- the top band does not actually share one basis
- the lower zone is just another recolored manifold

Published analogue:

- common in scVelo-like and trajectory-comparison pages

## Template C: Matrix-Led Dashboard Page

Best for:

- benchmark pages
- toolkit pages
- mechanism pages where the matrix is the object

Typical structure:

```text
|     A matrix hero     | B |
|     C matrix/support  | D |
```

Use when:

- a score matrix, interaction matrix, program matrix, or ranking table is the main result object
- companion bars/scatters/strips give the matrix context

Good fits:

- benchmark score summaries
- ligand-target/program heatmaps
- interaction matrices with compact summary blocks

Avoid when:

- the matrix is only support and not the main object

Published analogue:

- atlas integration benchmark, Pertpy, NicheNet-like pages

## Template D: Spatial Wall + Anchor

Best for:

- spatial mapping/deconvolution pages
- slice alignment pages
- several same-basis maps that are all genuinely needed

Typical structure:

```text
| A | B | C |
| D | E | F |
```

Where one child, often `F`, is the non-spatial anchor:

- distribution summary
- error/score panel
- interaction or abundance matrix

Use when:

- multiple same-basis spatial children are the page
- one statistical anchor prevents the page from becoming wallpaper

Avoid when:

- all spatial children are weak and repetitive
- no anchor panel is present

Published analogue:

- Slide-seqV2-like and spatial mapping papers

## Template E: Multi-Scale Interleave Page

Best for:

- multiview methods
- transport papers
- dynamics papers with several evidence scales

Typical structure:

```text
| top mixed hero band          |
| mid support band             |
| lower closure / validation   |
```

Use when:

- the method output naturally lives on several scales
- the page needs embeddings/maps, curves, matrices, and compact summaries together

Good fits:

- multiview fate or velocity pages
- transport/coupling pages
- integration pages with modality-level support

Avoid when:

- the page is already hard to parse
- a cleaner two-zone design would be easier

Published analogue:

- GLUE-like, moscot-like, multi-omic velocity-like pages

## Template F: Compact Four-Panel Portrait

Best for:

- narrow stories
- supplement pages
- one anchor plus three distinct companions

Typical structure:

```text
| A | B |
| C | D |
```

Use when:

- the page is small in scope
- no same-basis compare wall is required

Avoid when:

- the anchor really needs more than a quarter-page slot

## Template G: Wide Support Portrait

Best for:

- a compare row plus one broad lower proof object
- pages that need a wide heatmap/dotplot/dynamics closure

Typical structure:

```text
| A | B | C |
|    D wide   | E |
|      F wide      |
```

Use when:

- the lower zone contains one support object that genuinely needs width

Avoid when:

- the lower zone fragments into many weak mini-panels

## Template H: Structured Dashboard

Best for:

- toolkit pages
- supplement dashboards
- repeated multiview/parameter pages

Typical structure:

```text
| A | B | C | D |
| E | F | G | H |
```

or

```text
| A | B | C |
| D | E | F |
| G | H | I |
```

Use when:

- repeated structure is the point
- scales, glyphs, titles, and legends can be disciplined tightly

Avoid when:

- the page actually has only 2-3 jobs
- one child clearly deserves much more space

## Portrait vs Landscape

### Prefer portrait when:

- the final manuscript page is portrait-first
- the page has a top hero zone and lower support rows
- the main object is not excessively wide

### Prefer landscape when:

- the compare row itself is the page
- spatial/trajectory crops become too cramped in portrait
- the page is more about horizontal alignment than vertical narrative

Important:

- most audited published result pages are still effectively portrait/two-column pages
- do not default to landscape just because the page feels busy

## Panel Count: Use It Late, Not Early

### 4 panels

Use when:

- the story is tight
- each panel has distinct grammar
- one or two panels are clear anchors

### 6 panels

Use when:

- there are 2-3 zones
- one band is compare and one band is support
- moderate richness is enough

### 8 panels

Use when:

- repeated structure is disciplined
- the page is benchmark/toolkit/support heavy

### 9 panels

Use when:

- it is explicitly a grid/dashboard page
- every child belongs to one repeated logic

## Quick Selection Heuristic

Start here:

- if one object is obviously dominant: `Template A`
- if 3-5 same-basis children define the page: `Template B`
- if the matrix is the main object: `Template C`
- if the page is mostly spatial maps: `Template D`
- if the page inherently mixes several scales: `Template E`
- if the story is narrow and clean: `Template F`
- if one lower support object needs width: `Template G`
- if the page is benchmark/dashboard heavy: `Template H`

## Practical Rules

1. Give more space to the page anchor.
2. If all panels are equal size, repeated structure must genuinely be the point.
3. If the page feels thin, first change the skeleton, not the palette.
4. If the page feels busy, remove one weak support panel before shrinking everything.
5. A secondary model-result block can stay if it is method-relevant, coherent, and not algorithm-irrelevant filler.
