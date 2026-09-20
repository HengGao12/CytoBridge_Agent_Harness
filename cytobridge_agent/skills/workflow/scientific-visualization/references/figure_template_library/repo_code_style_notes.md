# Repository Code Style Notes

## Purpose

This note records concrete style and rendering habits seen in audited reproducibility or
package repositories that are especially worth carrying into the generic template library.

This is a complement to:

- `inventory_high_tier_papers.md`
- `high_tier_style_synthesis.md`

Those documents answer *what the good papers do*.
This one answers *where in code the habits are actually implemented*.

## CellRank Reproducibility

Paths:

- `/lustre/home/2501111653/figure_code_audit/cellrank_reproducibility/notebooks/fig_2_pancreas_main/ML_2021-09-21_fig_2_and_3_pancreas_main.ipynb`
- `/lustre/home/2501111653/figure_code_audit/cellrank_reproducibility/notebooks/suppl_fig_robustness/analysis_notebooks/utils/utils.py`

Reusable habits:

- consistent `scvelo` figure preset with high export DPI
- fate/embedding plots using quiet context plus explicit accent categories
- benchmark and robustness panels built as coherent multi-axes blocks
- legends used for exploration, then simplified or detached for export

Library implication:

- trajectory paper preset
- shared-basis embedding compare helper
- benchmark-grid helper with one external legend

## CellRank 2 Reproducibility

Paths:

- `/lustre/home/2501111653/figure_code_audit/cellrank2_reproducibility/src/cr2/analysis/_plotting.py`
- `/lustre/home/2501111653/figure_code_audit/cellrank2_reproducibility/scripts/pseudotime_kernel/hematopoiesis/cr1_vs_cr2.py`

Reusable habits:

- `mplscience` + `whitegrid` for benchmark/statistical summaries
- detached legends and clean export variants
- strong use of fixed categorical palettes and quiet method-comparison accents
- aligned time/fate summaries rather than many disconnected line charts

Library implication:

- stronger benchmark/stat summary preset
- aligned dynamics-track helper
- explore/export figure mode separation

## Tangram

Paths:

- `/lustre/home/2501111653/figure_code_audit/Tangram/tangram/plot_utils.py`
- `/lustre/home/2501111653/figure_code_audit/Tangram/tutorial_tangram_with_squidpy.ipynb`

Reusable habits:

- percentile clipping for scalar maps
- measured-vs-predicted spatial pairs on the same basis
- small horizontal companion colorbars
- aspect-locked spatial panels with frame removed

Library implication:

- robust percentile clipping utility
- paired spatial compare template
- companion colorbar helper

## Cell2location

Paths:

- `/lustre/home/2501111653/figure_code_audit/cell2location/cell2location/run_colocation.py`
- `/lustre/home/2501111653/figure_code_audit/cell2location/docs/notebooks/tutorial_utils.py`

Reusable habits:

- dense spatial grids packed tightly but consistently
- quantile-synchronized color limits across related spatial subpanels
- multi-factor spatial map grammar
- optional black-background facets that make low-signal spatial overlays clearer

Library implication:

- faceted spatial abundance grid template
- synchronized spatial color-limit helper
- optional dark facet preset for spatial overlays

## PASTE

Paths:

- `/lustre/home/2501111653/figure_code_audit/paste_reproducibility/notebooks/visualize-dlpfc.ipynb`
- `/lustre/home/2501111653/figure_code_audit/paste_reproducibility/notebooks/pairwise-simulation.ipynb`

Reusable habits:

- fixed categorical palettes across slices/conditions
- spatial compare panels that maintain geometry and crop
- summary plots paired with alignment visuals

Library implication:

- spatial alignment compare template
- fixed categorical palette registry for slice/domain panels

## GEARS

Paths:

- `/lustre/home/2501111653/figure_code_audit/GEARS_misc/gears/gears.py`
- `/lustre/home/2501111653/figure_code_audit/GEARS_misc/paper/CPA_reproduce/plotting.py`

Reusable habits:

- perturbation panels with truth distribution vs prediction marker vs baseline
- richer scatter/contour overlays with low-clutter labeling
- `despine()` + tick-focused minimalist stat styling

Library implication:

- perturbation truth-vs-prediction composite template
- rich scatter + contour overlay template

## CellCharter

Paths:

- `/lustre/home/2501111653/figure_code_audit/cellcharter_analyses/visium_human_dlpfc.ipynb`
- `/lustre/home/2501111653/figure_code_audit/cellcharter_analyses/codex_mouse_spleen.ipynb`

Reusable habits:

- benchmark boxplot + overplotted points
- frozen categorical palettes
- neighborhood/domain heatmaps with bounded diverging scales

Library implication:

- benchmark boxplot+strip template
- domain/neighborhood heatmap template

## DestVI

Paths:

- `/lustre/home/2501111653/figure_code_audit/DestVI-reproducibility/embryo/utils.py`
- `/lustre/home/2501111653/figure_code_audit/DestVI-reproducibility/lymph_node/deconvolution/DestVI-LN.ipynb`

Reusable habits:

- deconvolution summary bars paired with multiple spatial maps
- inset colorbars rather than large detached colorbars
- `legend_loc="on data"` style Scanpy embeddings when the state labels are sparse

Library implication:

- deconvolution result scaffold
- inset colorbar helper

## TISSUE

Paths:

- `/lustre/home/2501111653/figure_code_audit/tissue-figures-and-analyses/notebooks/01_prediction_model_evaluation_and_TISSUE.ipynb`
- `/lustre/home/2501111653/figure_code_audit/tissue-figures-and-analyses/notebooks/03_TISSUE_multiple_imputation_for_differential_gene_expression.ipynb`

Reusable habits:

- wide statistical layouts for method comparison
- figure-level legends outside dense panel bodies
- explicit uncertainty/error density views

Library implication:

- wide supplement benchmark layout template
- uncertainty density panel template
