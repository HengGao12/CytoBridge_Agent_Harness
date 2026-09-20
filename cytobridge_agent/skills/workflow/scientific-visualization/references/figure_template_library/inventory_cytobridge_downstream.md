# cytobridge-downstream Figure Inventory

This note audits the plotting code in:

- `/lustre/home/2501111653/cytobridge_downstream_fetch`

The goal is the same as the rest of this library:

- do **not** depend on the original project runtime
- extract reusable plotting grammars
- extract beautification logic that survives across datasets

## Access Note

This repository appears to be git-accessible in the current environment, but not fully exposed via
anonymous GitHub REST/archive endpoints. The working tree used for this audit was materialized via
direct `git fetch + checkout`.

Additional branches checked:

- `admouse-notebook-update`
- `zebrafish-api-release`

Those branches did not introduce a clearly new top-level plotting family beyond what is already
captured from `main`; they mainly extend the same spatial, velocity, and helper pipelines.

## Main Plotting Entrypoints

The most relevant plotting code lives in:

- `downstream_helpers/admouse_figures.py`
- `downstream_helpers/arista_legacy.py`
- `vendor/evaluation/arista_code/arista_helpers.py`
- `vendor/legacy_arista_stack/evaluation/arista_code/arista_focus_anchor_pipeline_shared.py`
- `notebooks/mosta/mosta_wnt3a_fzd7_lrp6_total_hotspot_triptych.ipynb`
- `downstream_helpers/mosta_baseline_video.py`

## What Is Worth Extracting

## 1. Spatial small-multiples panels

Source examples:

- `downstream_helpers/admouse_figures.py::_plot_spatial_panels`
- `vendor/legacy_arista_stack/evaluation/arista_code/arista_focus_anchor_pipeline_shared.py::save_timepoint_snapshots`

Core grammar:

- one column or row of spatial panels
- each panel uses the same coordinate basis
- points colored by cell type or condition
- equal aspect
- axes and spines removed
- one shared legend outside the panel body

Why reusable:

- fits spatial transcriptomics
- fits slice interpolation displays
- fits temporal sample snapshots

What to keep from the style:

- white background
- point-only rendering, no heavy outlines
- external legend
- quiet panel body
- optional coordinate rotation for visual convention

## 2. Spatial scalar triptych / strip with shared color scale

Source examples:

- `downstream_helpers/admouse_figures.py::plot_temporal_spatial_trajectory`
- `notebooks/mosta/mosta_wnt3a_fzd7_lrp6_total_hotspot_triptych.ipynb`

Core grammar:

- 3 or more aligned spatial panels
- same metric rendered across time or condition
- quiet grey background points plus colored foreground
- shared colorbar
- per-panel titles only

Why reusable:

- ideal for spatial score evolution
- ideal for ligand-receptor hotspots
- ideal for model-vs-observed spatial scalar comparison

What to keep from the style:

- equal aspect
- axis-off
- robust clipping instead of raw min/max
- dual scatter: low-alpha grey context below, colored metric layer above

## 3. Heatmap + curve companion panel

Source example:

- `downstream_helpers/admouse_figures.py::plot_combined_heatmap_uniform_curves`

Core grammar:

- left: gene-by-time heatmap
- right: small multiples of pattern trajectories
- pattern boundaries explicitly marked

Why reusable:

- dynamic module or gene-pattern stories
- supplement figures that need to show both matrix structure and cluster-level dynamics

What to keep from the style:

- explicit pattern separators
- restrained annotation boxes
- shared y-range across the right-side trajectory panels
- line + band rather than line only

## 4. Bubble summary scatter

Source example:

- `downstream_helpers/arista_legacy.py::run_arista_legacy_growth_interaction_celltype_bubble`

Core grammar:

- x = one group summary
- y = another group summary
- point size = group size
- point color = time or batch
- small external size legend + scalar colorbar

Why reusable:

- growth vs interaction
- score vs uncertainty
- abundance vs effect size

What to keep from the style:

- white edge on bubbles
- clipped size range
- legend for bubble size, not just color

## 5. Communication / attention heatmaps

Source example:

- `vendor/evaluation/arista_code/arista_helpers.py`

Core grammar:

- square sender-receiver heatmap
- separate asymmetry heatmap
- optional significance markers
- optional distance-stratified facet heatmaps

Why reusable:

- cell-cell communication
- source-target attention
- transport asymmetry

What to keep from the style:

- `PuBu`-style main matrix for positive-only summaries
- diverging palette for asymmetry
- light white cell borders
- significance dots instead of dense text labels

## 6. Focus-anchor spatial mosaic

Source example:

- `vendor/legacy_arista_stack/evaluation/arista_code/arista_focus_anchor_pipeline_shared.py`

Core grammar:

- many compact spatial snapshots
- one separate legend sheet
- optional tiled mosaic export

Why reusable:

- supplement workload figures
- timepoint grids
- method comparison over spatial slices

What to keep from the style:

- compact cell size
- rasterize large clouds
- no axis furniture
- separate legend artifact when on-canvas legend would dominate

## What Should Not Become Core Templates

## 1. Pipeline-specific 3D Sankey / focus-anchor plotly figures

Reason:

- too tied to this project stack
- too tied to one biological story
- more useful as inspiration than as a shared base template

## 2. Direct CytoBridge package wrappers

Reason:

- violates the template-library dependency rule
- not portable to projects without that runtime

## Recommended New Generic Templates

Based on this repo, the most useful additions are:

1. spatial small-multiples template
2. spatial scalar triptych template
3. heatmap + curve companion template
4. bubble summary template

## Net Effect on the Library

Compared with `scDiffEq`, `moslin`, and `moscot`, this repo adds the strongest support for:

- spatial panel layout
- spatial small multiples
- grey-context + colored-foreground spatial scalar overlays
- compact bubble summaries
- matrix-plus-curves supplement panels
