# Template Style Playbook

This file records how each template class should be beautified.

The goal is not to hardcode one paper's exact parameters. The goal is to preserve the
visual logic that those papers already solved well.

Color-family defaults now live in:

- `palette_bank.md`
- `palette_bank.py`

## Global Rules

- White background by default.
- Use one stable typography ladder unless there is a strong reason not to:
  - panel letter: about `11-12 pt`, bold
  - panel title: about `9-10 pt`, bold
  - axis label: about `8-9 pt`
  - ticks / legend: about `7-8 pt`
- Remove top/right spines unless the chart truly needs a full frame.
- Prefer one strong accent palette plus quiet greys.
- Prefer palette-bank classes over ad hoc local color picks.
- Use transparency to create hierarchy.
- Label only what helps the reader decode structure.
- Do not add sentence-like descriptive text on the canvas.
- Keep legends minimal.
- Preserve equal aspect for manifold / latent / spatial panels.
- Use `axis_policy.md` rather than guessing whether a panel should keep axes.
- Use `panel_finish_checklist.md` before approval; many failures happen at this stage.

When a panel still feels thin after these rules, read:

- `inventory_high_tier_papers.md`
- `high_tier_style_synthesis.md`

The usual problem is then not a missing font tweak. It is that the panel needs a stronger
grammar, a quieter background, or a better companion quantitative block.

## Cross-Paper Richness Rules

- A geometry panel often needs a quantitative companion block in the same row.
- Rich pages vary plotting grammar across rows instead of repeating the same manifold style.
- Quiet context plus deliberate foreground is more effective than globally stronger saturation.
- Spread and uncertainty usually make a panel look more credible than a single summary bar.
- Heatmaps become more publication-grade when they carry structural annotation, not more prose.

## Stylish Box + Jitter

- Use a two-layer box: translucent fill below, crisp outline above.
- Jitter points should sit above the box, not behind it.
- Grid only on `y`, and keep it faint.
- Rotate method labels only if needed.

## Manifold Overlay

- Background cells must be visibly quieter than overlays.
- Use grey background and colored overlay by default.
- Prefer `quiet_context_palette()` for the manifold body and a sequential scalar palette or
  atlas categorical palette for the foreground.
- Start markers should be few and obvious.
- Quantile clipping is usually better than raw min/max for scalar overlays.
- No visible ticks or frame lines.

## Velocity Stream

- The manifold should still be readable beneath the stream.
- The stream layer should not dominate the cell-state context.
- If using scalar coloring, quantile clipping is usually better than raw min/max.
- Legends should be small and pushed away from the panel body.
- Keep state-color backgrounds and streamline colors semantically separate.

## Temporal Ribbons

- Categories should stack cleanly with no white seams.
- If one category is the focus, it can be more saturated.
- Avoid too many categories in one panel.

## Volcano

- Insignificant points must be visually quiet.
- Significant positive and negative groups should be symmetric in styling.
- Prefer the signed warm/cool logic from `perturbation_diverging_palette()`.
- Only label a small curated subset.
- Always use reference lines for `x = 0` and significance threshold.

## Perturbation Dose-Response

- Keep `control` visually distinct from perturbed doses.
- Signed doses should read monotonically left-to-right or center-out from zero.
- Positive and negative directions should use one coherent two-pole color system.
- Prefer `perturbation_diverging_palette()` instead of manually choosing reds and blues each time.
- Show spread when replicates exist; box/violin plus sparse points is better than a naked mean line.
- Do not overload the panel with many perturbation families at once.

## Sweep Heatmap + Grouped Bars

- Heatmaps should highlight the best cell with geometry, not extra prose.
- Use one shared value scale across comparable facets.
- Missing values should be visible but not aggressive.
- Bars should prioritize alignment and readability over decoration.

## Transition Heatmap

- Matrix values should remain legible without turning the panel into a spreadsheet.
- If annotated, keep the font small and let the color carry most of the message.
- Row and column labels should read cleanly; rotate only when necessary.
- Use this when matrix structure is the object, not when flow is the object.
- Use a sequential palette for positive-only transport and a diverging matrix palette when zero is meaningful.

## Spatial Small Multiples

- Keep all panels on the same basis and aspect ratio.
- Remove axes and frame lines completely unless coordinates themselves are the object.
- Use one shared legend outside the panel body.
- If the spatial convention requires rotation or inversion, apply it consistently across the whole row/column.
- Use rasterization for very large point clouds.

## Bubble Summary

- Bubble size must encode a real group quantity, not decorative emphasis.
- Add a size legend explicitly; color alone is not enough.
- White or very light outlines usually help bubble separation.
- Keep the panel body clean: one scatter layer, one size legend, one scalar colorbar.

## Heatmap + Curve Companion

- The heatmap and the curve grid should tell the same grouping story.
- Mark group boundaries geometrically with separators before adding text.
- Keep the right-side curve panels on a shared y-scale when comparison matters.
- Use mean/median + band, not a naked line.

## Communication / Attention Heatmap

- Use a positive-only sequential palette for one-way magnitude summaries.
- Use a diverging palette centered at zero for asymmetry.
- Prefer `attention_matrix_palette(mode=...)`.
- Keep cell borders light and white so the matrix remains readable.
- If significance needs to be shown, use sparse point markers rather than dense numeric labels.
- This class should stay square and compact; if it becomes too wide, split it into multiple panels.

## Embedding + Generated-Trajectory Overlay

- The embedding basis must remain visible; generated paths should be semi-transparent.
- Use many light paths rather than a few very thick ones.
- Keep the background quiet unless the background category geography is itself important.
- Remove axes and frame lines.
- Use equal aspect and export with white facecolor.

## Perturbation Embedding Compare

- Both panels must share the exact same basis, limits, and aspect.
- The background manifold should be quieter than both overlays.
- Control and perturbed foregrounds should use the same palette logic so the shift, not the style, is what changes.
- Keep titles minimal, usually just `Unperturbed` and `Perturbed`.
- If a contour is used, it should be thin, black, and secondary to the point cloud.
- Do not add extra prose to explain the shift; the side-by-side comparison should carry the message.

## Embedding Time-Overlay

- The simulated overlay should be sorted by time before plotting so later states do not randomly obscure earlier structure.
- Use one continuous colormap for simulated time, not many discrete colors.
- If an observed lineage is overlaid, it should have a black-under-color treatment so it stays readable.
- Start-state markers should be few and obvious.
- The background should stay quieter than the simulation layer.

## Rich Scatter/Line/Facet

- Thin individual lines are usually better than thick spaghetti lines.
- Add one simple baseline reference line if the scale is centered.
- Prefer quiet themes with no panel grid or only a faint major `y` grid.
- Faceting should reduce clutter, not create dozens of unreadable mini-panels.
- Use a strong manual palette only when group identity matters; otherwise keep the lines mostly quiet.

## Annotated Block Heatmap

- The heatmap should remain readable before annotations are added.
- Row/column group strips should be thin and geometric, not oversized.
- Group separators are usually more effective than extra prose labels.
- Reserve side annotations for one or two genuinely useful companions.
- Use a diverging palette only when the matrix is centered meaningfully.

## Bipartite Pathway Network

- Manual or semi-manual layout is often worth it for manuscript clarity.
- Keep edges lighter than nodes.
- Use node shape for ontology/type and fill for omic/class when both matter.
- Labels should be sparse and placed with intent; over-labeling destroys the panel.
- Prefer white or transparent backgrounds with no axes.

## Labeled Network Summary

- Edge weight should be visible but muted.
- Labels need backing or halos when they overlap the network body.
- Node size should encode one real summary variable, not decoration.
- Use one quiet edge color plus one compact node palette.

## Alluvial / Transition Flow

- Strata and flows should share one state color system.
- Do not over-annotate every small flow.
- If zooming, suppress irrelevant flows instead of letting the plot become spaghetti.
- Time labels must read cleanly left-to-right.

## Sankey / Transport Summary

- Similar to alluvial, but usually better when the input is already aggregated into transitions.
- Keep captions minimal.
- Colors should be consistent across all stages.

## Validation Dashboards

- Usually supplement, not main-figure headline.
- Different subplots can use different grammars, but they should share one palette family.
- Keep subplot spacing generous.

## Trajectory Divergence Panel

- Background cells must be uniformly quiet (grey) so the trajectory overlay dominates.
- Use one warm color (e.g., red) for one fate and one cool color (e.g., blue) for the other.
- KDE contours should be thin (1-2 levels) and use the same hue as the scatter points.
- Time-sliced panels (Early / Mid / Late) must share the exact same UMAP limits and aspect.
- Remove axes and frame lines; add only a small title per sub-panel.
- If both E11 (progenitor) and E15 (terminal) cells are on the same UMAP, do not color
  the E15 background by terminal fate—keep it grey to avoid confusion with the trajectories.

## Along-Trajectory Gene Dynamics Heatmap

- x-axis = pseudo-time (0 → 1), y-axis = genes, color = z-scored expression.
- Use a diverging colormap centered at 0 (e.g., `RdBu_r` or `coolwarm`).
- Gene names should be in *italic* on the y-axis, with adequate font size (≥ 7pt).
- Add thin vertical lines or ticks to mark trajectory steps if they are few (< 25).
- If showing only a subset of cells (e.g., AL-committed), state this in the panel title.
- x-axis label should say "Trajectory pseudo-time" or similar, not "Steps".

## Regulatory Circuit Diagram

- Nodes: circles with module name inside, colored by module identity.
- Self-reinforcement value should be displayed inside the node.
- Arrows: straight lines with arrowheads. Green = activation (positive influence),
  Red = suppression (negative influence). Arrow thickness ∝ |influence|.
- Set a minimum thickness threshold below which arrows are not drawn (reduces clutter).
- Use `ax.annotate` with `arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0")`
  for straight arrows. Curved arrows can look decorative but often reduce clarity.
- Small multiples (e.g., 2×2 for BM/AL × Early/Late) should share node positions exactly.
- Add a legend explaining arrow color semantics and thickness scale.
- No axes, no frame, no ticks. White background.
- Numeric influence values can be placed near the arrow midpoint in small font (5-6pt).

## Fate Score Gallery

- All sub-panels share the same UMAP background and limits.
- Background cells: light grey, small point size.
- Foreground cells (selected cell type): colored by a scalar (e.g., fate probability),
  using a sequential colormap.
- Each sub-panel gets a concise title (cell type name) and optionally a summary statistic
  (e.g., mean δ score).
- One shared colorbar outside the row, not per-panel colorbars.

## Domain-Specific Color Semantics

The palette bank provides generic color classes. However, many biological domains have
established color conventions. **Domain colors take priority over palette bank classes.**

Common conventions in lineage tracing / cell fate:

- **BM** (Branchiomotor): warm red (`#d62728` or similar)
- **AL** (Autonomic-like): cool blue (`#1f77b4` or similar)
- **FP** (Floor plate): green (`#2ca02c` or similar)
- **Progenitors**: grey or light grey
- **Activation/positive regulation**: green arrows/edges
- **Suppression/negative regulation**: red arrows/edges

When working on a new dataset, ask the user or check existing figures for domain conventions
before falling back to the palette bank.

Define domain colors as named constants at the top of each script:

```python
BM_C  = "#d62728"
AL_C  = "#1f77b4"
FP_C  = "#2ca02c"
BG_C  = "#d0d0d0"
INK   = "#222222"
```

## PDF and Rasterization Policy

- **Default: vector everything.** All `ax.scatter`, `ax.plot`, `ax.annotate`, `patches.Circle`
  etc. should remain as vector elements in PDF output.
- **Do not** use `rasterized=True` on `ax.scatter` unless:
  1. The point count exceeds ~50,000, AND
  2. The PDF will NOT be edited in Illustrator/Inkscape downstream.
- For typical single-cell panels (< 20,000 cells), keep scatter as vector. The file size
  increase is acceptable for editability.
- `ax.imshow` (heatmaps, images) is inherently raster. This is fine—heatmaps do not need
  to be vector-editable at the cell level.
- After saving a PDF, always zoom to 400% on a scatter-heavy panel to verify points are
  crisp. If they look pixelated, check for stray `rasterized=True`.
- Set `dpi=300` or higher in `savefig` for both PNG and PDF.

## What Not To Do

- Do not make sparse tables look like heatmaps and call that a figure.
- Do not use tinted page backgrounds by default.
- Do not invent a new red/blue convention in every perturbation panel.
- Do not add prose labels to rescue a weak encoding.
- Do not keep the same manifold grammar for two adjacent panels if the message has changed.
- Do not use `rasterized=True` on scatter plots without checking the point count first.
- Do not use palette bank colors when the domain has established conventions.
- Do not force a panel into a template when the message requires a custom visualization.
