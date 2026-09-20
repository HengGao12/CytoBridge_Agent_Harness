# Palette Bank

This file records the shared palette classes used across the figure-template library.

The goal is not to lock the whole project into one aesthetic forever. The goal is to avoid ad hoc
color picking and to preserve source-backed color logic that repeatedly works in published papers.

Implementation lives in:

- `palette_bank.py`

## Why A Palette Bank Exists

Without a shared palette layer, the library still drifts into:

- random category colors across related figures
- incompatible compare rows
- perturbation panels that switch red/blue meanings
- sequential overlays that are either too aggressive or too pale
- heatmaps that look spreadsheet-like

The palette bank solves this by standardizing **palette classes**, not one universal palette.

## Palette Classes

## 1. Quiet Context Palette

Use for:

- manifold backgrounds
- spatial background points
- latent background cells
- quiet compare context

Logic:

- one light grey for the large context cloud
- one darker grey for outlines or secondary context
- black-brown ink for text/foreground

Source logic:

- repeated across `moslin`, `moscot`, `cytobridge`, and many top-tier result pages where the
  background geometry should remain visible but not dominate

Implementation:

- `quiet_context_palette()`

## 2. Atlas Categorical Palette

Use for:

- cell-state atlases
- lineage categories
- compact state legends

Logic:

- muted but distinct hues
- avoids neon saturation
- stays readable when 6-10 classes must coexist

Implementation:

- `atlas_categorical_palette(n)`

## 3. Compare / Method Palette

Use for:

- shared-basis method compare rows
- method-centric boxplots or bars

Logic:

- `Ours` gets the strongest accent
- `Clone` / `Truth` stays dark neutral
- baselines stay distinct but slightly quieter

Implementation:

- `compare_method_palette(labels=None)`

## 4. Perturbation Diverging Palette

Use for:

- signed perturbation bars
- dose-response
- signed delta summaries
- centered effect heatmaps when the sign matters

Logic:

- positive pole = warm red family
- negative pole = cool blue family
- control = quiet grey
- signed dose steps can deepen gradually away from zero

This follows the most stable visual convention across `scDiffEq`-style perturbation plots and
later manuscript-friendly signed summaries.

Implementation:

- `perturbation_diverging_palette()`

## 5. Sequential Scalar Palettes

Use for:

- future-probability overlays
- competence overlays
- scalar spatial fields
- positive-only matrix summaries

Available families:

- `warm_signal`
- `cool_signal`
- `green_signal`
- `grey_signal`

Implementation:

- `sequential_scalar_palette(name)`

## 6. Diverging Matrix Palettes

Use for:

- centered heatmaps
- signed module-balance matrices
- asymmetry maps

Available families:

- `soft_balance`
- `cool_warm_quiet`
- `teal_ochre`

Implementation:

- `diverging_matrix_palette(name)`

## 7. Attention / Communication Matrix Palette

Use for:

- source-target magnitude matrices
- asymmetry matrices

Logic:

- positive-only magnitude uses sequential cool tones
- signed asymmetry uses a centered diverging scheme

Implementation:

- `attention_matrix_palette(mode='magnitude'|'asymmetry')`

## 8. Benchmark Palette

Use for:

- benchmark bars
- compact benchmark dashboards

Logic:

- `ours` = strongest accent
- truth = black
- baselines = compact multi-hue accents
- neutral = grey

Implementation:

- `benchmark_palette()`

## 9. Opposing Program / Biology Pole Palette

Use for:

- switch-vs-buffer modules
- mono-vs-neut
- any two opposing biological programs

Implementation:

- `gene_program_pole_palette()`

## Source Logic, Not Literal Copying

These palettes are informed by:

- `scDiffEq`
- `moslin`
- `moscot`
- `cytobridge-downstream`
- `ipop_aging`
- the 20-paper high-tier result-page audit

They are not exact reproductions of a single source paper palette. The aim is:

- preserve the visual strategy
- keep manuscript-level restraint
- remain generic across datasets

## Usage Rules

- Keep semantic polarity stable across a project.
  - If warm = program A in one perturbation panel, do not reverse it in the next panel.
- Use quiet greys for context whenever the geometry is not itself the headline object.
- Prefer shared palette classes across sibling panels.
- Do not use the categorical atlas palette for signed perturbation results.
- Do not use a diverging palette for strictly positive scalar overlays unless zero is a meaningful center.

## What Still Requires Judgment

The bank does not eliminate judgment. You still need to decide:

- whether a panel should be categorical or scalar
- whether a matrix is magnitude-like or asymmetry-like
- whether one result needs a project-specific palette override

The bank is there to prevent random choices, not to remove all flexibility.
