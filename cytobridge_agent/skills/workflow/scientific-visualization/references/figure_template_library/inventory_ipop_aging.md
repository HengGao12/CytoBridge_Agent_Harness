# iPOP Aging Figure Inventory

This note audits the plotting value of:

- `https://github.com/jaspershen-lab/ipop_aging`

## High-Level Verdict

This repository is **not** a clean plotting package. It is a large R analysis repository with
figure code embedded in many analysis scripts.

Still, it contains clear, source-backed plotting grammars that are worth extracting because they
are visibly styled rather than being raw default plots.

The strongest signals are:

- centralized color palettes and base themes in `1-code/100-tools.R`
- `ComplexHeatmap` panels with structured block annotations
- `ggraph` pathway/network panels with manual node/edge styling
- well-composed `ggplot2` scatter/line/facet summaries with quiet backgrounds

## Evidence of Real Beautification

Representative code lives in:

- `1-code/100-tools.R`
- `1-code/11_combined_omics/cross_sectional_loess/heatmap.R`
- `1-code/11_combined_omics/cross_sectional_loess/DE-SWAN_molecule_pathway_enrichment/figure5_b_pathway_network/combine.R`
- `1-code/11_combined_omics/cross_sectional_loess/summary_male_female/summary_cluster.R`
- `1-code/11_combined_omics/invidual_level/2_scatter_plot.R`

Examples of explicit style choices:

- `theme_base <- theme_bw() + theme(...)`
- `ggsci::pal_lancet()`, `pal_aaas()`, `RColorBrewer`
- transparent panel and plot backgrounds for compositing
- `ComplexHeatmap` block annotations and linked side summaries
- `ggraph` node shapes, fills, edge colors, label treatments
- `shadowtext::geom_shadowtext` for readable graph labels

So this repo should be treated as:

- a source of **styled R analysis-figure grammars**
- not a source of reusable package-native plotting APIs

## Figure Types Worth Extracting

### 1. Rich scatter/line/facet summaries

**Where**

- `1-code/11_combined_omics/invidual_level/2_scatter_plot.R`
- `1-code/100-tools.R`

**Why it matters**

These are ordinary plot classes done carefully:

- subject or feature trajectories as thin lines
- facet-wise organization
- quiet theme
- baseline `geom_hline`
- strong manual palette

**Template direction**

Create a generic `rich_scatter_facet` template that:

- supports thin per-feature lines
- optional smoother
- optional facet columns
- quiet white background and light/no grid

### 2. ComplexHeatmap-style annotated block heatmaps

**Where**

- `1-code/100-tools.R`
- `1-code/11_combined_omics/cross_sectional_loess/heatmap.R`

**Why it matters**

This is the strongest distinctive grammar in the repo:

- matrix heatmap
- row/column group blocks
- split boundaries
- side annotations
- companion summaries such as box/point/text annotations

**Template direction**

Do not try to clone every ComplexHeatmap feature.
Keep a generic Python template for:

- matrix + row groups + column groups
- top and side color strips
- explicit group separators

### 3. Bipartite pathway / module network panel

**Where**

- `1-code/11_combined_omics/cross_sectional_loess/DE-SWAN_molecule_pathway_enrichment/figure5_b_pathway_network/combine.R`

**Why it matters**

This panel has a clear manuscript grammar:

- manual or semi-manual layout
- edge color channels
- node fill, node size, node shape
- label angle and clean theme

**Template direction**

Create a generic bipartite/pathway network template with:

- explicit node table
- explicit edge table
- node-class shapes
- edge-class colors

### 4. Labeled summary network

**Where**

- `1-code/11_combined_omics/cross_sectional_loess/summary_male_female/summary_cluster.R`

**Why it matters**

This is a more compact graph-summary grammar than the pathway network:

- weighted grey edges
- class-colored nodes
- large text labels with white backing
- graph theme with minimal clutter

**Template direction**

Create a generic network-summary template that emphasizes:

- readable labels
- muted edges
- class-based node emphasis

## What Not To Copy

- the enormous data-specific analysis tree
- hardcoded file-system paths
- dataset-specific omics naming and palette choices
- ComplexHeatmap extras that depend on custom grob logic unless they really generalize

## Template-Library Impact

This repo mainly strengthens:

1. rich scatter/line/facet summaries
2. annotated block heatmaps
3. pathway/network panels
4. label-heavy network summaries
