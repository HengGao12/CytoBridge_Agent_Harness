# moslin Figure Audit

## Bottom Line

`moslin` is **analysis-first**, not **figure-library-first**.

The manuscript mapping is documented in dataset READMEs:

- `/lustre/home/2501111653/moslin_clean/moslin/analysis/packer_c_elegans/README.rst`
- `/lustre/home/2501111653/moslin_clean/moslin/analysis/hu_zebrafish_linnaeus/README.rst`
- `/lustre/home/2501111653/moslin_clean/moslin/analysis/simulations/pan_tedsim/README.rst`

Reusable plotting utilities are concentrated in:

- `/lustre/home/2501111653/moslin_clean/moslin/src/moslin_utils/pl/_plotting.py`

Most other plotting is notebook- or script-level.

## Figure Types Worth Extracting

### 1. Grid-search benchmark heatmaps

**Where**

- `/lustre/home/2501111653/moslin_clean/moslin/src/moslin_utils/pl/_plotting.py`
  via `gridsearch_heatmap`
- called from:
  `/lustre/home/2501111653/moslin_clean/moslin/analysis/packer_c_elegans/ML_2024-06-04_visualize_gridsearch_results.ipynb`

**Why it matters**

This is already fairly reusable:

- facet by timepoint
- pivot on two hyperparameters
- heatmap a performance metric
- outline the best cell

**Reuse potential**

High.

**Template direction**

Keep as a generic sweep-heatmap panel with:

- configurable metric
- configurable facet variable
- configurable best-cell rule

---

### 2. Grid-search grouped bars

**Where**

- `/lustre/home/2501111653/moslin_clean/moslin/src/moslin_utils/pl/_plotting.py`
  via `gridsearch_bar`

**Why it matters**

This is the cleanest compact benchmark grammar on the moslin side:

- grouped bar by timepoint
- one bar per method
- best run already filtered upstream

**Reuse potential**

High.

**Template direction**

Keep as a standard benchmark-summary template.

---

### 3. Coupling / embedding overlays

**Where**

- `/lustre/home/2501111653/moslin_clean/moslin/src/moslin_utils/pl/_plotting.py`
  via:
  - `plot2D_samples_mat`
  - `plot_coupling`
  - `plot_ancestor_descendant_error`

**Why it matters**

These are strong for:

- local coupling inspection
- predicted transport edges
- error differences in an embedding

**Reuse potential**

Medium-high.

**Template direction**

Good candidate for:

- ROI coupling panel
- delta/error-on-embedding panel

---

### 4. Alluvial flow plots

**Where**

- `/lustre/home/2501111653/moslin_clean/moslin/analysis/hu_zebrafish_linnaeus/Zebrafish_coupling_analysis.R`

**Why it matters**

This is the most distinctive manuscript-grade `moslin` figure grammar:

- time on x-axis
- stratum heights encode composition
- flows encode transition mass
- easy to read as lineage/transition story

**Reuse potential**

High for grammar, low for current code.

**Template direction**

Worth extracting as a generic R template:

- transition table in
- alluvial plot out

---

### 5. Validation scatter dashboards

**Where**

- `/lustre/home/2501111653/moslin_clean/moslin/analysis/hu_zebrafish_linnaeus/Hyperparameters_with_persistency.py`

**Why it matters**

This script mixes:

- heatmaps
- scatter validation plots
- clustermaps
- bars

The useful part is not the exact code, but the grammar:

- benchmark metric vs biology-support metric

**Reuse potential**

Medium-high.

**Template direction**

Keep as a dashboard-style supplement template, not a main-figure default.

---

### 6. TedSim tree / cost visuals

**Where**

- `/lustre/home/2501111653/moslin_clean/moslin/analysis/simulations/pan_tedsim/utils_analysis.py`

**Why it matters**

These cover:

- state tree visualization
- lineage tree visualization
- barcode or cost heatmaps

**Reuse potential**

Medium.

**Template direction**

Use as simulation-support templates only.

---

### 7. CellRank aggregate fate probability panels

**Where**

- multiple notebooks under:
  `/lustre/home/2501111653/moslin_clean/moslin/analysis/packer_c_elegans/ML_2024-03-12_cellrank2_*`
- and:
  `/lustre/home/2501111653/moslin_clean/moslin/analysis/packer_c_elegans/ML_2024-06-29_cellrank2_moslin.ipynb`

**Why it matters**

The reusable insight is the grammar:

- aggregate fate probabilities
- compare the same biology across methods

But the actual code is mostly library calls plus dataset-specific state definitions.

**Reuse potential**

Medium as scaffold, low as shared plotting code.

**Template direction**

Standardize as pipeline scaffold, not core plotting function.

## Best Extraction Targets

Recommended order:

1. gridsearch heatmap
2. gridsearch grouped bar
3. alluvial transition panel
4. coupling overlay panel
5. validation scatter dashboard

## What Not To Copy

- notebook-local CellRank state definitions
- monolithic downstream scripts that combine preprocessing and plotting
- raw scanpy embedding calls without a reusable wrapper

