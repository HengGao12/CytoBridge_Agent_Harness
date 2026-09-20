# Squidpy Figure Inventory

This note records what `Squidpy` contributes to the template library.

## Public Sources

- Squidpy API docs:
  - `pl.nhood_enrichment`
  - `pl.interaction_matrix`
  - `pl.co_occurrence`
  - `pl.ligrec`
- Squidpy neighborhood-enrichment tutorial
- Squidpy paper

## Why It Matters

`Squidpy` is the strongest public source for missing **spatial neighborhood/domain** plot classes.

## Strongest Template Targets

## 1. Neighborhood enrichment heatmap

Use for:

- cluster-neighborhood enrichment
- spatial neighborhood support panels

Decision:

- usually prefer the native `squidpy.pl.nhood_enrichment` grammar or a light wrapper around it

## 2. Interaction matrix

Use for:

- aggregate cluster-cluster spatial interaction summaries
- a spatial analogue of communication/attention heatmaps

Decision:

- should become either a native-wrapper template or a custom heatmap template with Squidpy-like defaults

## 3. Co-occurrence curves

Use for:

- how pairwise co-occurrence changes with spatial distance

Decision:

- strong candidate for a new spatial-distance trend template

## 4. Ligand-receptor result plots

Use for:

- spatial communication support figures

Decision:

- likely supplement template rather than main-figure default

## Net Value

Squidpy is the best public source for filling:

- spatial neighborhood enrichment
- interaction matrices in spatial analysis
- co-occurrence over distance
- spatial communication support panels
