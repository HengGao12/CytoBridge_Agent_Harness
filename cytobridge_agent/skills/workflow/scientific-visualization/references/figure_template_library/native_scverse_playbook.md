# Native scverse Plotting Playbook

This note records when a figure should use native plotting functions from broadly adopted
single-cell/spatial packages instead of being reimplemented from scratch.

This playbook is informed mainly by:

- `single-cell-best-practices`
- official `scanpy` docs
- official `scvelo` docs
- official `cellrank` docs
- official `squidpy` docs

## Principle

If the biological object is already well served by a stable native grammar, use it.

Do **not** reimplement a plot in raw matplotlib just to prove generality.

Reimplement only when:

- the native plot is too rigid
- the style clashes badly with the manuscript
- multiple native outputs must be merged into one custom panel
- the plot class needs arguments or layout behavior that are not exposed cleanly

## Good Native Plot Classes

## 1. Scanpy embedding panels

Typical calls:

- `sc.pl.umap`
- `sc.pl.embedding`
- `sc.pl.pca`
- `sc.pl.tsne`

Use natively when:

- the goal is a standard embedding view
- you need quick categorical or gene overlays
- `frameon=False` and `legend_loc="on data"` already solve the panel

## 2. Scanpy marker/program summaries

Typical calls:

- `sc.pl.dotplot`
- `sc.pl.matrixplot`
- `sc.pl.rank_genes_groups_dotplot`
- `sc.pl.rank_genes_groups_matrixplot`
- `sc.pl.stacked_violin`

Use natively when:

- marker summaries are the object
- the standard scanpy legend grammar is acceptable
- the panel does not need unusual layout integration

## 3. scvelo velocity views

Typical calls:

- `scv.pl.velocity_embedding_stream`
- `scv.pl.scatter`

Use natively when:

- the vector field is already in a standard basis such as UMAP
- the native smoothing behavior is acceptable

Prefer custom wrappers when:

- you need a nonstandard basis
- you need to synchronize colors/layout with other panels
- you need to suppress default clutter

## 4. CellRank lineage plots

Typical calls:

- `cellrank.pl.aggregate_fate_probabilities`
- `cellrank.pl.gene_trends`

Use natively when:

- fate probabilities or lineage-weighted trends are the actual object
- the goal is a method-standard lineage summary

Prefer custom plots when:

- the trends need to be merged tightly into a custom multi-panel manuscript layout
- you need stronger layout control than the native function offers

## 5. Squidpy spatial neighborhood plots

Typical calls:

- `sq.pl.nhood_enrichment`
- `sq.pl.interaction_matrix`
- `sq.pl.co_occurrence`

Use natively when:

- the plot class is a standard spatial neighborhood summary
- the manuscript does not require heavy recomposition

Prefer custom plots when:

- the native panel must be merged tightly with non-spatial panels
- color or annotation control needs to be stronger than the default

## 6. Squidpy spatial overview / image overlays

Typical calls:

- `sq.pl.spatial_scatter`

Use natively when:

- the object is a standard spatial coordinate overview
- histology image plus spots should be shown without custom compositing

Prefer custom plots when:

- multiple spatial views must be aligned to non-native neighbors
- the panel needs unusual ROI or multi-layer compositing

## Native Style Rules

Even when using native scverse plots:

- set white background
- prefer `frameon=False` for embeddings unless axes matter
- keep legends small
- avoid large default titles
- export through a wrapper script, not manually from the notebook

## Decision Rule

Use native scverse plotting when:

1. the plot class is already canonical in the ecosystem
2. the native output is close to manuscript quality
3. the needed customization is light

Use a custom template when:

1. the panel needs nonstandard composition
2. the plot must be tightly aligned with neighboring panels
3. the native function would require too much manual post-processing
