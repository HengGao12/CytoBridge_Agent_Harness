# High-Tier Paper Audit

## Goal

Build a source-backed inventory of published high-tier computational biology papers whose
figures and publicly available code are worth mining for reusable display grammar and figure
polish.

This inventory prioritizes:

- Nature Methods
- Nature Biotechnology
- Nature
- Nature Genetics
- Cell

The point is not to imitate one paper. The point is to learn what consistently makes similar
results look richer, clearer, and more publication-ready.

## Paper Inventory

| # | Paper | Venue | Year | Evidence | Figure/Code Availability | What To Mine |
|---:|---|---|---:|---|---|---|
| 1 | Reversed graph embedding resolves complex single-cell trajectories | Nature Methods | 2017 | local PDF | paper only | branch topology and schematic + manifold pairing |
| 2 | RNA velocity of single cells | Nature | 2018 | local PDF | indirect via scvelo ecosystem | embedding flow overlays and direction fields |
| 3 | A comparison of single-cell trajectory inference methods | Nature Biotechnology | 2019 | local PDF | paper only | benchmark small multiples and ranking blocks |
| 4 | NicheNet: modeling intercellular communication by linking ligands to target genes | Nature Methods | 2020 | local PDF | package code, figure code not audited | ligand-target heatmap + network coupling |
| 5 | Generalizing RNA velocity to transient cell states through dynamical modeling | Nature Biotechnology | 2020 | local PDF | scvelo package/tutorial code | velocity stream, kinetics plots, phase portraits |
| 6 | Deep learning and alignment of spatially resolved single-cell transcriptomes with Tangram | Nature Methods | 2021 | local PDF; `/figure_code_audit/Tangram/README.md` | yes | measured/predicted spatial compare grammar |
| 7 | Highly sensitive spatial transcriptomics at near-cellular resolution with Slide-seqV2 | Nature Biotechnology | 2021 | local PDF | paper only | dense spatial scalar maps |
| 8 | Joint probabilistic modeling of single-cell multi-omic data with totalVI | Nature Methods | 2021 | remote docs evidence | package/tutorial code | multimodal embeddings, uncertainty-aware summaries |
| 9 | CellRank for directed single-cell fate mapping | Nature Methods | 2022 | `/figure_code_audit/cellrank_reproducibility/README.rst` | yes | fate maps, benchmark blocks, uncertainty panels |
| 10 | Benchmarking atlas-level data integration in single-cell genomics | Nature Methods | 2022 | local PDF | paper only | benchmark matrix + summary companion blocks |
| 11 | PASTE: alignment and integration of spatial transcriptomics data | Nature Methods | 2022 | `/figure_code_audit/paste_reproducibility/README.md` | yes | slice overlays, alignment compare, spatial panel packing |
| 12 | Cell2location maps fine-grained cell types in spatial transcriptomics | Nature Biotechnology | 2022 | `/figure_code_audit/cell2location/README.md` | yes | spatial abundance grids and colocation summaries |
| 13 | Multi-omics single-cell data integration and regulatory inference with graph-linked embedding | Nature Biotechnology | 2022 | local PDF | public package exists, figure code not yet cloned | multimodal benchmark and embedding compare grammar |
| 14 | DestVI identifies continuums of cell types in spatial transcriptomics data | Nature Biotechnology | 2022 | remote article evidence; `/figure_code_audit/DestVI-reproducibility` | yes | deconvolution maps and simulation summary grammar |
| 15 | Mapping single-cell data to reference atlases by transfer learning (scArches) | Nature Biotechnology | 2022 | remote article evidence | package/repo available | query-reference compare panels |
| 16 | MultiVI: deep generative model for the integration of multimodal data | Nature Methods | 2023 | remote article evidence | package/tutorial code | multimodal integration rows and benchmark summaries |
| 17 | Learning single-cell perturbation responses using neural optimal transport | Nature Methods | 2023 | local PDF; local `scdiffeq-analyses` code | yes | perturbation dose-response, embedding compare, simulated overlays |
| 18 | Multi-omic single-cell velocity models epigenome-transcriptome interactions and improves cell fate prediction | Nature Biotechnology | 2023 | local PDF | code not yet audited | multiview dynamics and branch-support panels |
| 19 | CellCharter | Nature Genetics | 2024 | `/figure_code_audit/cellcharter_analyses/README.md` | yes | spatial domain maps, neighborhood summaries |
| 20 | CellRank 2: unified fate mapping in multiview single-cell data | Nature Methods | 2024 | local PDF; `/figure_code_audit/cellrank2_reproducibility/README.md` | yes | multiview fate, benchmark, pseudotime/fate panel grammar |
| 21 | Mapping cells through time and space with moscot | Nature | 2025 | local PDF | package/repro code available | transition heatmaps, coupling, sankey/alluvial |
| 22 | Gene trajectory inference for single-cell data by optimal transport metrics | Nature Biotechnology | 2025 | local PDF | code not yet audited | OT-based trajectory compare and validation layouts |
| 23 | Pertpy: an end-to-end framework for perturbation analysis | Nature Methods | 2026 | local PDF | package/repro code available | perturbation dashboards and benchmark summaries |

## Repositories Audited Directly

These repos were cloned or already available locally and actually inspected for plotting code:

- `/lustre/home/2501111653/figure_code_audit/cellrank_reproducibility`
- `/lustre/home/2501111653/figure_code_audit/cellrank2_reproducibility`
- `/lustre/home/2501111653/figure_code_audit/Tangram`
- `/lustre/home/2501111653/figure_code_audit/paste_reproducibility`
- `/lustre/home/2501111653/figure_code_audit/cell2location`
- `/lustre/home/2501111653/figure_code_audit/DestVI-reproducibility`
- `/lustre/home/2501111653/figure_code_audit/GEARS_misc`
- `/lustre/home/2501111653/figure_code_audit/tissue-figures-and-analyses`
- `/lustre/home/2501111653/figure_code_audit/cellcharter_analyses`
- local `scdiffeq-analyses`

## Repeated Figure Families Seen Across High-Tier Papers

### 1. Shared-basis manifold compare

- same embedding crop
- same aspect
- same point sizing
- only the overlaid assignment/prediction changes

### 2. Geometry + quantitative companion

- embedding or spatial map on one side
- distribution / bar / heatmap / benchmark block on the other

### 3. Time/dynamics tracks

- aligned time axis
- multiple signals stacked vertically
- uncertainty shown as band, ribbon, or repeated samples

### 4. Perturbation response panels

- dose-response box/bar panels
- embedding compare before/after perturbation
- compact volcano only when many hits are screened

### 5. Spatial multi-panel compare

- measured vs predicted
- method A vs method B
- same color scale
- small multiples with minimal frame furniture

### 6. Matrix with companion annotation

- matrix/heatmap is not alone
- block annotations, side bars, cluster strips, or companion curves/bars carry context

## What This Audit Changed

This audit supports two upgrades to the template library:

1. stronger page-level composition rules
2. a stronger distinction between:
   - thin starter templates
   - richer, source-backed manuscript grammars
