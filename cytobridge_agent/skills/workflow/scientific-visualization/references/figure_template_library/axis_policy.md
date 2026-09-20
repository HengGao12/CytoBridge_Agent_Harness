# Axis Policy

This file turns axis/spine/tick handling into a harder decision rule.

The goal is to stop figures from drifting into inconsistent styles where some panels keep
meaningless axes and others remove useful ones.

## Rule 1: Ask Whether Coordinates Themselves Carry Meaning

If the reader does **not** need numeric coordinates to interpret the panel, remove them.

If the reader **does** need numeric coordinates, keep them but simplify the frame.

## Panel Classes

### A. Manifold / Latent / Spatial Overview

Default policy:

- remove all ticks
- remove all spines
- keep equal aspect
- no axis labels

Use this for:

- UMAP / t-SNE / SPRING / PHATE
- latent embeddings
- spatial scatter where geometry is the object but numeric coordinates are not
- embedding + trajectory overlays

Exception:

- keep axes only if the coordinate system itself is scientifically meaningful, such as:
  - true physical coordinates
  - image-pixel coordinates that will be cross-referenced
  - explicitly metric projections where distances are interpreted quantitatively

### B. Local ROI / Zoom Insets

Default policy:

- usually remove ticks
- remove spines unless the inset border itself is serving as a zoom box
- preserve equal aspect

If the ROI is linked to a full-view panel, the connection geometry should carry the meaning,
not numeric ticks.

### C. Distribution Panels

Default policy:

- keep the measurement axis
- keep category labels
- remove top/right spines
- keep only the grid that helps comparison, usually `y` for vertical distributions and `x`
  for horizontal distributions

Use for:

- violin
- box/boxen
- raincloud
- jitter + box

### D. Benchmark / Summary Bars

Default policy:

- keep the value axis
- keep category labels
- remove top/right spines
- use a faint grid only on the value axis

Do not keep a full four-sided frame unless the matrix/grid structure itself is the object.

### E. Time / Dynamics Panels

Default policy:

- keep both axes
- remove top/right spines
- faint grid on the comparison axis
- keep zero or threshold reference lines when they carry meaning

Use for:

- line + band
- aligned tracks
- ribbons
- time heat strips

### F. Perturbation / Intervention Panels

Default policy:

- keep both axes
- remove top/right spines
- always show the control baseline explicitly
- keep zero/reference lines when sign matters

Use for:

- dose-response
- local delta bars
- volcano

### G. Heatmaps / Matrices

Default policy:

- keep row/column labels
- no outer box emphasis unless it helps grouping
- keep ticks only as label carriers
- remove redundant axis labels if the matrix title already explains the mapping

Use separators and annotations geometrically, not with extra prose.

### H. Networks / Alluvial / Sankey

Default policy:

- remove all numeric axes
- remove spines
- no ticks

These panels are about relationships, not coordinate scales.

## Spine Defaults

Unless a panel class explicitly needs a full frame:

- remove `top`
- remove `right`

For manifold/spatial/network classes:

- remove all four spines

## Tick Defaults

- Do not keep ticks after removing their meaning.
- If a tick label exists only to fill space, remove it.
- If labels are categorical and long, prefer rotation or horizontal plots before shrinking the
  font too far.

## Reference Lines

Reference lines are encouraged only when they mean something:

- `x = 0`
- `y = 0`
- significance threshold
- decision threshold
- crossing time

Do not add decorative guide lines with no analytical meaning.

## Final Rule

If you are unsure whether to keep axes:

1. ask whether the reader needs numeric coordinates
2. ask whether removing them improves focus
3. if yes, remove them

Most manuscript figures are cleaner with fewer axes than first drafts.
