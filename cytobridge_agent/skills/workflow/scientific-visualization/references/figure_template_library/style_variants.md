# Style Variants Across Repositories

This file records cases where the **same figure class** appears in both codebases or in multiple
forms, but with **different styling strategies**.

The point is not to choose one forever. The point is to know what the legitimate options are.

Shared palette families are centralized in:

- `palette_bank.py`
- `palette_bank.md`

## 1. Benchmark Summary: Box + Jitter vs Grouped Bar

### scDiffEq strategy

- Uses layered boxplots with jittered foreground points.
- Emphasizes replicate distribution and within-method spread.
- Best when multiple seeds / replicates are central.

### moslin strategy

- Uses grouped bars after selecting best runs.
- Emphasizes best-performing configurations across timepoints.
- Best when the comparison unit is already aggregated.

### Decision

- Use **box + jitter** if replicate dispersion matters.
- Use **grouped bar** if only one filtered summary per method/timepoint matters.

## 2. Manifold Context: Quiet Background vs Colored Background

### scDiffEq strategy

- Often uses a colored manifold background with a controlled palette and overlay layer.
- In velocity panels, cell-state colors remain visible even beneath streams.

### moslin strategy

- Often uses a neutral manifold context and highlights subset cells or coupling endpoints with outlines.
- More likely to quiet the whole embedding and focus attention on the coupling object.

### Decision

- Use **colored background manifold** when the cell-type geography itself is part of the claim.
- Use **quiet/grey background manifold** when the real message is the overlay.
- Prefer to implement this distinction via the palette bank instead of custom local colors.

## 3. Embedding Support: Scalar Overlay vs Explicit ROI Overlay

### scDiffEq strategy

- Often overlays continuous scalar values or simulated points directly over the manifold.

### moslin strategy

- Often uses selected source/target subsets, outlines, and explicit coupling lines.

### Decision

- Use **continuous scalar overlay** when the message is field structure.
- Use **ROI/coupling overlay** when the message is transition specificity.

## 4. Perturbation Summary: Volcano vs Dose-Response / Delta Summary

### scDiffEq strategy

- Uses volcano plots for large perturbation screens with significance.
- Strong when there are many hits.

### Our later manuscript practice

- Often needs small directional perturbation summaries, not large genome-wide screens.
- In that setting, simple dose-response or signed delta bars are clearer than volcano.

### Decision

- Use **volcano** when there are many perturbations plus significance.
- Use **dose-response or delta bars** when there are a few designed directions or modules.
- In both cases, keep the signed warm/cool polarity stable across the project.

## 4b. Native Wrapper vs Custom Reimplementation

### single-cell-best-practices / official scverse strategy

- Keep canonical plot classes native when the API already exposes the right grammar.
- Spend effort on whitening background, legend suppression, and layout composition instead of
  rewriting the whole plot.

### custom-template strategy

- Reimplement when the panel must align tightly with non-native neighbors or when the native API
  is too rigid.

### Decision

- Use a **native wrapper** for canonical ecosystem plots such as CellRank aggregate fate or
  Squidpy interaction matrices.
- Use a **custom matplotlib template** when manuscript layout control matters more than ecosystem
  fidelity.

## 5. Transition Story: Alluvial vs Heatmap

### moslin strategy

- Uses alluvial plots when the story is movement of composition across time.
- Uses heatmaps when the story is the transition matrix itself.

### Decision

- Use **alluvial** when flow is the object.
- Use **heatmap** when matrix structure or local contrast is the object.

## 6. Temporal Structure: Ribbons vs Aligned Tracks

### scDiffEq strategy

- Uses stacked ribbons for changing composition.

### Our later manuscript practice

- Uses aligned multi-track line panels when several dynamic quantities must share one time axis.

### Decision

- Use **ribbons** for category composition.
- Use **aligned tracks** for multiple metrics over the same time axis.

## 7. Spatial Panels: Pure Categorical Snapshots vs Scalar Spatial Overlays

### cytobridge-downstream categorical strategy

- Uses cell-type-colored spatial snapshots or panel columns.
- Strong emphasis on axis-free, equal-aspect, legend-outside layouts.

### cytobridge-downstream scalar strategy

- Uses quiet grey context points below a colored scalar layer.
- Strong emphasis on robust clipping, shared colorbar, and panel-to-panel geometric consistency.

### Decision

- Use **categorical snapshots** when tissue composition is the object.
- Use **scalar overlays** when one score or communication field is the object.
- The palette bank exists so these two modes do not drift into conflicting color logic.

## 8. Dynamic Matrix Support: Standalone Heatmap vs Heatmap + Companion Curves

### generic matrix-only strategy

- Best when the matrix itself is the message.

### cytobridge-downstream companion strategy

- Pairs a heatmap with grouped curve panels to show both matrix structure and temporal signatures.

### Decision

- Use **matrix only** when patterns are obvious and timing detail is secondary.
- Use **heatmap + companion curves** when the reader must see both grouped matrix structure and the
  corresponding temporal trajectories.

## 9. Matrix Communication Summary: Native Scanpy/Squidpy Matrixplot vs Custom Attention Heatmap

### single-cell-best-practices strategy

- Frequently relies on native scverse matrix-style plots such as dotplot and matrixplot.
- Good when the ecosystem default already matches the biological object.

### cytobridge-downstream strategy

- Uses custom seaborn heatmaps with explicit palette choice, asymmetry mode, and optional
  significance dots.

### Decision

- Use **native scverse matrix plots** when the goal is a standard marker/program summary.
- Use **custom attention heatmaps** when the object is a directional source-target or asymmetry matrix.

## 10. Generated Paths on Embeddings: Streamlines vs Explicit Trajectory Overlays

### scvelo/native stream strategy

- Best when the object is a local vector field or smooth directional tendency.

### scDiffEq-inspired explicit-overlay strategy

- Best when the object is a set of generated trajectories or rollout paths on a fixed embedding.

### Decision

- Use **streamlines** for vector fields.
- Use **trajectory overlays** for explicit generated paths or decoded rollouts.

## 11. Ordinary Trend Plots: Plain Defaults vs Styled Faceted Summaries

### generic quick-look strategy

- Use default line/scatter plots with minimal filtering.

### ipop_aging strategy

- Use quiet themed backgrounds, manual palettes, baseline lines, and faceting to turn ordinary
  trends into manuscript-grade summaries.

### Decision

- Use **plain defaults** for quick exploratory checks.
- Use **styled faceted summaries** when the ordinary plot itself is supporting a main claim.

## 12. Matrix Structure: Plain Heatmap vs Annotated Block Heatmap

### basic heatmap strategy

- Best when the matrix itself is already enough.

### ipop_aging ComplexHeatmap-inspired strategy

- Add row/column strips, group separators, and one or two companion annotations.

### Decision

- Use a **plain heatmap** when the matrix pattern is obvious.
- Use an **annotated block heatmap** when grouped structure is part of the message.

## 13. Network Summary: Generic Graph vs Styled Pathway / Cluster Graph

### generic graph strategy

- Focus on topology, with minimal aesthetics.

### ipop_aging strategy

- Use muted edges, class-aware node styling, manual layout, and readable labels with backing.

### Decision

- Use a **generic graph** for exploratory topology.
- Use a **styled summary network** when the panel must be publication-facing.
