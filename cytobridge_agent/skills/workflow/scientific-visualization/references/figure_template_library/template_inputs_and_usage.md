# Template Inputs and Usage

This file records, for each template class:

- what inputs are required
- what optional inputs are helpful
- when the template is appropriate
- when it should not be used

## Global Rule

Template code should assume only a **general plotting stack**:

- `matplotlib`
- `seaborn`
- `numpy`
- `pandas`

Allowed optional scientific dependencies:

- `anndata`
- `scanpy`
- `scvelo`
- `squidpy`
- `cellrank`
- `pertpy`
- `networkx`
- `statsmodels`

The template should **not** require the runtime API or object model of the paper-specific source
repository. For example, a transition heatmap template may be inspired by `moscot`, but it should
accept a plain dataframe or matrix rather than a `moscot` problem object.

If a template cannot be expressed cleanly without a special plotting package, mark it as:

- optional
- reference-only
- not the default template for cross-dataset reuse

## 1. Stylish Box + Jitter

**Required inputs**

- tidy dataframe
- one categorical column
- one numeric column

**Optional inputs**

- display order
- palette
- custom axis limits

**Use when**

- comparing methods, cohorts, or conditions
- replicate spread matters

**Do not use when**

- there is only one value per group
- the result is fundamentally temporal or spatial

## 2. Manifold Overlay

**Required inputs**

- 2D coordinates
- background dataframe

**Optional inputs**

- overlay dataframe
- scalar value for overlay coloring
- start/seed markers

**Use when**

- showing full context plus a selected foreground
- overlaying trajectories, simulated cells, or selected subsets

**Do not use when**

- the real question is already a distribution or summary statistic

## 2b. Perturbation Embedding Compare

**Required inputs**

- one fixed 2D basis
- background dataframe on that basis
- control overlay dataframe
- perturbed overlay dataframe

**Optional inputs**

- categorical background grouping and palette
- categorical overlay grouping and palette
- a contour subset for control
- a contour subset for perturbed

**Use when**

- the reader should compare `control` and `perturbed` in the same geometric frame
- a perturbation result is more interpretable on a shared embedding than as a summary statistic
- the shift in occupied region is the object

**Do not use when**

- the perturbation result is better summarized by a dose-response or volcano
- the comparison really needs more than two conditions

## 2c. Embedding Time-Overlay

**Required inputs**

- one fixed 2D basis
- background dataframe
- simulated dataframe on the same basis
- one continuous time column for the simulated dataframe

**Optional inputs**

- categorical background grouping and palette
- observed lineage dataframe
- lineage time grouping and palette
- start-state dataframe

**Use when**

- a generated rollout should remain anchored to one embedding
- continuous temporal progression is the visual object
- an observed lineage should be compared against the generated path on the same basis

**Do not use when**

- only a few discrete trajectories matter more than the point cloud
- the result is easier to read as a track, ribbon, or dose-response panel

## 3. Velocity Stream

**Required inputs**

- 2D coordinates
- 2D vectors (`u`, `v`)

**Optional inputs**

- scalar magnitude
- custom palette

**Use when**

- the result is a flow field or local directional tendency

**Do not use when**

- you only have a few discrete trajectories
- stream smoothing would distort the real message

## 4. Temporal Ribbons

**Required inputs**

- time column
- category column
- numeric value column

**Optional inputs**

- palette
- category order

**Use when**

- category composition changes over time

**Do not use when**

- the real object is not compositional
- there are too many categories for one panel

## 5. Volcano

**Required inputs**

- effect-size column
- p-value column

**Optional inputs**

- label column
- highlighted genes or features

**Use when**

- many perturbation hits or differential effects must be screened together

**Do not use when**

- there are only a handful of designed perturbations
- significance is not meaningful or not available

## 5b. Perturbation Dose-Response

**Required inputs**

- perturbation label column
- signed magnitude column
- numeric outcome column

**Optional inputs**

- direction/color grouping
- replicate column
- control row or explicit baseline

**Use when**

- the perturbations are a small designed family
- the reader should see monotonic signed response across doses
- endpoint fraction or composition change is the object

**Do not use when**

- there are hundreds or thousands of hits
- significance ranking is the main object

## 6. Sweep Heatmap

**Required inputs**

- one row hyperparameter
- one column hyperparameter
- one numeric metric

**Optional inputs**

- facet variable
- best-cell rule

**Use when**

- exploring a 2-parameter sweep
- identifying robust or optimal settings

**Do not use when**

- only one parameter is varying
- readers do not need to see the whole search surface

## 7. Grouped Benchmark Bar

**Required inputs**

- categorical x column
- numeric y column
- hue grouping

**Optional inputs**

- custom palette
- y limits

**Use when**

- comparing already-aggregated summaries across methods or conditions

**Do not use when**

- replicate spread is important
- the panel would become too sparse

## 8. Alluvial / Transition Flow

**Required inputs**

- long-form transition table
- time axis
- stratum identity
- alluvium identity
- numeric fraction or mass

**Optional inputs**

- state palette
- zoom/focus subset

**Use when**

- composition flow across time is the biological object

**Do not use when**

- the matrix itself is the object
- the flow graph is too dense to read even after thresholding

**Dependency note**

- the current reference implementation is an optional R template using `ggplot2` + `ggalluvial`
- this should not be treated as the default Python-side shared backend

## 9. Transition Heatmap

**Required inputs**

- transition matrix as a dataframe or 2D table

**Optional inputs**

- row/column annotations
- annotation format
- custom colormap

**Use when**

- aggregate state-to-state transition structure is the object

**Do not use when**

- the biological message is flow continuity rather than matrix contrast

## 10. Sankey / Transport Summary

**Required inputs**

- stage-to-stage aggregated transitions
- stage captions or ordering

**Optional inputs**

- color dictionary
- interpolation of colors across stages

**Use when**

- transport mass across stages is the object and the reader should follow it left-to-right

**Do not use when**

- the system is too dense and a matrix would be easier to read

## 11. Native Scanpy Embedding Wrapper

**Required inputs**

- `AnnData`
- embedding basis name
- one categorical or numeric color key

**Optional inputs**

- `legend_loc`
- `frameon`
- explicit palette

**Use when**

- a standard embedding panel already looks close to manuscript quality

**Do not use when**

- the panel requires heavy recomposition with custom overlays

## 11b. Native Scanpy Dotplot Wrapper

**Required inputs**

- `AnnData`
- `var_names`
- `groupby`

**Optional inputs**

- standard scale / dendrogram / swap_axes settings

**Use when**

- marker summaries are the object and native scanpy grammar is already appropriate

**Do not use when**

- the panel must be tightly fused into a more custom manuscript layout

## 11c. Native Scanpy Matrixplot Wrapper

**Required inputs**

- `AnnData`
- `var_names`
- `groupby`

**Optional inputs**

- standard scale
- colormap

**Use when**

- the object is a grouped marker/program matrix

**Do not use when**

- a custom heatmap with nonstandard annotations is required

## 12. Native scvelo Stream Wrapper

**Required inputs**

- `AnnData`
- velocity graph already computed or computable
- embedding basis name

**Optional inputs**

- color key
- density / smoothness settings

**Use when**

- the flow field is the object and native scvelo already captures it well

**Do not use when**

- the basis is nonstandard and must be custom-projected

## 13. Native Squidpy Neighborhood Wrapper

**Required inputs**

- `AnnData`
- neighborhood statistics already computed or computable

**Optional inputs**

- cluster key
- color limits

**Use when**

- the object is standard neighborhood enrichment

**Do not use when**

- the panel must be merged into a tighter custom spatial composition

## 13b. Native Squidpy Spatial Scatter Wrapper

**Required inputs**

- `AnnData` with spatial coordinates

**Optional inputs**

- color key
- histology image toggle
- crop / library id arguments

**Use when**

- a standard spatial overview or image-plus-spots panel is needed

**Do not use when**

- the manuscript needs more customized spatial compositing than the native function exposes

## 14. Native CellRank Aggregate Fate Wrapper

**Required inputs**

- `AnnData` with CellRank results
- cluster key

**Optional inputs**

- lineage subset
- mode

**Use when**

- aggregate fate probabilities are already the object

**Do not use when**

- the figure needs stronger custom layout control than CellRank exposes

## 15. Native CellRank Gene Trends Wrapper

**Required inputs**

- fitted CellRank model or compatible object
- list of genes

**Optional inputs**

- lineage subset
- shared axis or confidence-band settings

**Use when**

- lineage-weighted trends are already a method-standard object

**Do not use when**

- the trend panel must be merged tightly with non-native manuscript subpanels

## 16. Native Squidpy Interaction Matrix Wrapper

**Required inputs**

- `AnnData`
- cluster key

**Optional inputs**

- normalization
- palette or colormap

**Use when**

- the object is a standard interaction/count matrix over neighborhoods

**Do not use when**

- sender-receiver asymmetry or significance overlays require a custom matrix grammar

## 17. Native Squidpy Co-Occurrence Wrapper

**Required inputs**

- `AnnData`
- cluster key

**Optional inputs**

- distance limits
- color scale

**Use when**

- the object is spatial co-occurrence over distance

**Do not use when**

- a simpler single-distance summary would be clearer

## 18. Embedding + Generated-Trajectory Overlay

**Required inputs**

- 2D embedding dataframe
- sequence of 2D trajectories

**Optional inputs**

- background scalar or category coloring
- seed markers
- foreground grouping

**Use when**

- generated rollouts or decoded trajectories must be shown on a fixed basis
- the biological message depends on seeing both context and simulated paths

**Do not use when**

- only a few endpoints matter and a simpler ROI summary would be clearer

## 19. Rich Scatter/Line/Facet

**Required inputs**

- tidy dataframe
- `x`
- `y`
- grouping key for individual lines

**Optional inputs**

- color grouping
- facet variable
- baseline value
- smoother toggle

**Use when**

- many ordinary trajectories must still look manuscript-grade
- feature-wise or subject-wise trends are the object

**Do not use when**

- there are too many facets or too many overlapping groups for one panel

## 20. Annotated Block Heatmap

**Required inputs**

- matrix dataframe

**Optional inputs**

- row groups
- column groups
- group palette
- centered diverging scale

**Use when**

- grouped matrix structure is the message
- row/column blocks need to be shown explicitly

**Do not use when**

- a plain heatmap already communicates the result

## 21. Bipartite Pathway Network

**Required inputs**

- node table with explicit positions or a layout-ready graph
- edge table

**Optional inputs**

- node fill group
- node size
- node shape class
- edge class palette

**Use when**

- pathway/module/gene relationships must be summarized in one stylized graph

**Do not use when**

- the graph is too dense to read even after thresholding

## 22. Labeled Network Summary

**Required inputs**

- node table with positions
- edge table

**Optional inputs**

- edge weight
- node class palette
- text labels

**Use when**

- a compact association/correlation network needs readable labels

**Do not use when**

- labels are so numerous that the panel becomes text-dominated

## 23. Trajectory Divergence (NEW)

**Required inputs**

- 2D background coordinates (all cells)
- trajectory coordinates over T time steps for M cells
- fate label per trajectory cell

**Optional inputs**

- fate colors
- background color/size
- KDE contour levels
- time step selection and titles

**Use when**

- the model generates continuous trajectories and fate divergence is the message
- the panel should show Early/Mid/Late snapshots of trajectory separation

**Do not use when**

- there are no generated trajectories (use a static embedding overlay instead)
- the divergence is better shown as a scalar summary (e.g., divergence metric over time)

## 24. Along-Trajectory Gene Dynamics Heatmap (NEW)

**Required inputs**

- expression matrix (n_steps × n_genes), averaged across cells per time step
- gene names

**Optional inputs**

- colormap, z-score limits
- title, x-axis label
- gene grouping annotations

**Use when**

- the message is about how specific genes change along a generated trajectory
- the panel should connect trajectory generation to molecular-level interpretation

**Do not use when**

- the real question is about inter-gene relationships (use a network or circuit diagram)
- expression changes are better shown as individual trend lines (use T14 Rich Scatter)

## 25. Regulatory Circuit Diagram (NEW)

**Required inputs**

- module names
- module positions (x, y)
- influence matrix (source, target) → float

**Optional inputs**

- module colors
- activation/suppression colors
- influence threshold, max line width
- whether to show numeric values on arrows

**Use when**

- the message is about directed module-to-module regulatory influence
- the influence data comes from Jacobian perturbation or similar causal analysis
- small multiples across conditions/time are needed

**Do not use when**

- the graph is undirected (use T17 Labeled Network Summary)
- there are too many modules (> 6-7) for a clean node-arrow layout
- the data is a correlation matrix (use T12 Attention Heatmap)

## 26. Fate Score Gallery (NEW)

**Required inputs**

- 2D background coordinates (e.g., UMAP)
- per-cell-type dataframes with scalar fate scores
- cell type names

**Optional inputs**

- shared colorbar range
- background color/size
- summary statistic per panel (e.g., mean delta)

**Use when**

- the model predicts fate probabilities and you want to show per-cell-type patterns
- the gallery format is needed to compare across multiple cell types

**Do not use when**

- only one cell type matters (use a single manifold overlay)
- the comparison is between methods rather than cell types (use T2 side-by-side)

## 11. Validation Dashboard

**Required inputs**

- at least two validation metrics or one benchmark metric plus one biology-support metric

**Optional inputs**

- shared palette family
- facet variable

**Use when**

- supplement-level validation and workload display are the goal

**Do not use when**

- a single clean plot would answer the question better

## 12. Spatial Small Multiples

**Required inputs**

- one dataframe per panel
- one x coordinate column
- one y coordinate column
- one categorical label column
- one panel title column or per-panel title value

**Optional inputs**

- coordinate rotation or y-axis inversion rule
- explicit color dictionary
- point-size and alpha settings

**Use when**

- comparing spatial slices across time
- comparing observed vs generated spatial samples
- building a clean supplement row/column of spatial snapshots

**Do not use when**

- the real object is a scalar field rather than a categorical layout

## 13. Bubble Summary

**Required inputs**

- x metric
- y metric
- bubble-size metric
- bubble-color metric

**Optional inputs**

- custom size clipping range
- explicit colorbar label
- explicit size legend title

**Use when**

- one point naturally summarizes one group, one timepoint, or one cohort
- a third quantitative attribute should be encoded as size

**Do not use when**

- raw replicate distribution is the object
- there are too many points for size encoding to remain readable

## 14. Heatmap + Curve Companion

**Required inputs**

- 2D heatmap matrix
- x labels
- grouped row slices or pattern boundaries
- one x vector for curves
- one mean curve per group

**Optional inputs**

- one uncertainty band per group
- custom color order
- explicit group titles

**Use when**

- a matrix pattern and its dynamic signatures must be shown together
- the grouped right-side summaries are necessary to decode the heatmap

**Do not use when**

- either the heatmap alone or the curves alone already answer the question

## 15. Communication / Attention Heatmap

**Required inputs**

- one 2D matrix
- row and column labels

**Optional inputs**

- significance mask
- positive-only vs asymmetry mode
- compact title and custom colorbar label

**Use when**

- the object is a sender-receiver, source-target, or interaction matrix
- asymmetry itself is biologically meaningful

**Do not use when**

- a native dotplot or matrixplot already answers the question more directly
- the matrix is too dense to read without aggressive aggregation
