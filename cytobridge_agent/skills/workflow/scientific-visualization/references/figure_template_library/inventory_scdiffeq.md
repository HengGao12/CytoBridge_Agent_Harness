# scDiffEq Figure Audit

## Bottom Line

`scDiffEq` plotting is mostly **manuscript-notebook-centric**, but a few figure grammars are strong and reusable.

There is a thin reusable plotting layer in:

- `/lustre/home/2501111653/scdiffeq-analyses/scdiffeq_analyses/_plotting/__init__.py`
- `/lustre/home/2501111653/scdiffeq-analyses/scdiffeq_analyses/_plotting/_box_plot/_styled_box_plot.py`
- `/lustre/home/2501111653/scdiffeq-analyses/scdiffeq_analyses/_plotting/_fit_loss.py`
- `/lustre/home/2501111653/scdiffeq-analyses/scdiffeq_analyses/computational_complexity/_simulation_visualization.py`

Most manuscript figure generation lives under:

- `/lustre/home/2501111653/scdiffeq-analyses/manuscript/figure_*/notebooks`

## Figure Types Worth Extracting

### 1. Benchmark boxplots with jittered points

**Where**

- `/lustre/home/2501111653/scdiffeq-analyses/manuscript/figure_2/notebooks/Figure2D.ipynb`
- `/lustre/home/2501111653/scdiffeq-analyses/manuscript/figure_3/notebooks/Figure3CD.ipynb`
- `/lustre/home/2501111653/scdiffeq-analyses/manuscript/figure_4/notebooks/Figure4D.ipynb`
- `/lustre/home/2501111653/scdiffeq-analyses/manuscript/figure_s7/notebooks/FigureS7.ipynb`
- packaged version:
  `/lustre/home/2501111653/scdiffeq-analyses/scdiffeq_analyses/_plotting/_box_plot/_styled_box_plot.py`

**Why it matters**

This is the cleanest already-functionized grammar in the repo:

- translucent background box
- outlined foreground box
- jittered foreground points
- compact benchmark-ready summary

**Reuse potential**

High.

**Template direction**

Use this as the canonical benchmark-comparison template.

---

### 2. Velocity stream on manifold

**Where**

- `/lustre/home/2501111653/scdiffeq-analyses/manuscript/figure_2/notebooks/Figure2B.ipynb`
- `/lustre/home/2501111653/scdiffeq-analyses/manuscript/figure_3/notebooks/Figure3HI.ipynb`
- `/lustre/home/2501111653/scdiffeq-analyses/manuscript/figure_3/notebooks/Figure3M.ipynb`
- `/lustre/home/2501111653/scdiffeq-analyses/manuscript/figure_s9/notebooks/FigureS9.ipynb`
- `/lustre/home/2501111653/scdiffeq-analyses/manuscript/figure_s12/notebooks/FigureS12EFGHI.ipynb`

**Why it matters**

This is one of the most recognizable `scDiffEq` visual grammars:

- manifold background
- typed cell-state colors
- vector field or stream overlay
- optional scalar overlay for diffusion / drift magnitude

**Reuse potential**

High for grammar, medium for full pipeline.

**Template direction**

Create a generic wrapper for:

- background embedding
- streamlines or arrows
- optional scalar coloring
- optional start/seed markers

---

### 3. UMAP/manifold with simulated overlay

**Where**

- `/lustre/home/2501111653/scdiffeq-analyses/manuscript/figure_4/notebooks/Figure4ABC.ipynb`
- `/lustre/home/2501111653/scdiffeq-analyses/manuscript/figure_4/notebooks/Figure4HI.ipynb`
- `/lustre/home/2501111653/scdiffeq-analyses/manuscript/figure_4/notebooks/Figure4EF.ipynb`

**Why it matters**

This is the grammar behind:

- full manifold as context
- simulated cells or trajectories overlaid
- highlighted lineage / clone / selected examples

**Reuse potential**

Medium-high.

**Template direction**

A generic `manifold_overlay_panel(...)` is worth keeping, but it should not hardcode:

- `X_umap`
- clone keys
- timepoint labels

---

### 3b. Perturbation projected onto a fixed embedding basis

**Where**

- `/lustre/home/2501111653/scdiffeq-analyses/manuscript/figure_3/notebooks/Figure3AB.ipynb`
- related embedding-projection overlays:
  - `/lustre/home/2501111653/scdiffeq-analyses/manuscript/figure_4/notebooks/Figure4ABC.ipynb`
  - `/lustre/home/2501111653/scdiffeq-analyses/manuscript/figure_4/notebooks/Figure4HI.ipynb`

**Why it matters**

This is the cleanest `scDiffEq` grammar for perturbation results that still stays spatially
anchored:

- fit or reuse one embedding basis
- project control and perturbed simulations into that same basis
- show the full manifold quietly in the background
- compare `unperturbed` and `perturbed` side-by-side
- optionally add a contour around the fate-focused subset

This is different from a generic simulated-trajectory overlay:

- it is explicitly comparative
- it assumes one fixed embedding basis shared by both conditions
- it uses contour or hull-like emphasis to show the perturbation-shifted region

**Reuse potential**

High.

**Template direction**

Keep a separate template for:

1. fixed-basis side-by-side comparison
2. quiet typed manifold background
3. control vs perturbed overlay
4. optional contour highlight

---

### 3c. Time-colored simulation overlay on a fixed embedding

**Where**

- `/lustre/home/2501111653/scdiffeq-analyses/manuscript/figure_4/notebooks/Figure4ABC.ipynb`
- related examples:
  - `/lustre/home/2501111653/scdiffeq-analyses/manuscript/figure_4/notebooks/Figure4HI.ipynb`
  - `/lustre/home/2501111653/scdiffeq-analyses/manuscript/figure_4/notebooks/Figure4EF.ipynb`

**Why it matters**

This is a different grammar from both plain trajectory overlays and plain perturbation compare:

- simulated cells are projected into one fixed basis
- simulation time is encoded continuously by color
- an observed lineage can be overlaid as a second layer
- the start state is explicitly marked

This is one of the cleanest ways to make a generated rollout feel geometric rather than abstract.

**Reuse potential**

High.

**Template direction**

Keep a separate template for:

1. fixed basis background
2. time-colored simulated overlay
3. optional observed-lineage companion overlay
4. optional start-state marker

---

### 4. Perturbation volcano plots

**Where**

- `/lustre/home/2501111653/scdiffeq-analyses/manuscript/figure_3/notebooks/Figure3KL.ipynb`
- `/lustre/home/2501111653/scdiffeq-analyses/manuscript/figure_s10/notebooks/FigureS10.ipynb`
- `/lustre/home/2501111653/scdiffeq-analyses/manuscript/figure_s11/notebooks/FigureS11.ipynb`

**Why it matters**

This is one of the few perturbation grammars in the repo that is both:

- manuscript-ready
- generic enough to survive across datasets

The workflow is:

- aggregate effects across models
- combine p-values
- assign four groups:
  - `sig.pos`
  - `sig.neg`
  - `insig.pos`
  - `insig.neg`
- color and label the most important hits

**Reuse potential**

High for plotting grammar.

**Template direction**

Separate:

1. meta-analysis preparation
2. volcano rendering

---

### 4b. Perturbation dose-response panels

**Where**

- `/lustre/home/2501111653/scdiffeq-analyses/manuscript/figure_3/notebooks/Figure3J.ipynb`

**Why it matters**

This captures the other major perturbation grammar in the paper:

- a small designed perturbation family
- signed perturbation magnitude on the x-axis
- endpoint fraction or composition on the y-axis
- replicate-aware summary rather than a single point estimate

**Reuse potential**

High for module- or TF-level perturbation panels when the perturbation family is small.

**Template direction**

Keep this separate from the volcano template:

1. volcano for many hits with significance
2. dose-response for a few designed perturbation directions

---

### 5. Fate-bias clustermaps / heatmaps

**Where**

- `/lustre/home/2501111653/scdiffeq-analyses/manuscript/figure_2/results/fate_prediction/TIGON/LARRY.full_dataset.plot_fate_bias_matrices.TIGON.ipynb`
- `/lustre/home/2501111653/scdiffeq-analyses/manuscript/figure_s4/notebooks/FigureS4.ipynb`

**Why it matters**

These plots are useful for:

- predicted vs observed fate-distribution matrices
- grouped or clustered probability patterns
- supplement-style detailed support

**Reuse potential**

Medium-high.

**Template direction**

Keep this as a supplement/support template, not a main-figure default.

---

### 6. Temporal composition ribbons

**Where**

- `/lustre/home/2501111653/scdiffeq-analyses/manuscript/figure_3/notebooks/Figure3AB.ipynb`

**Why it matters**

Useful when the actual message is:

- changing category composition over time

rather than pointwise manifold geometry.

**Reuse potential**

Medium.

**Template direction**

A small helper is enough.

---

### 7. Metric and loss curves

**Where**

- `/lustre/home/2501111653/scdiffeq-analyses/scdiffeq_analyses/_plotting/_fit_loss.py`
- `/lustre/home/2501111653/scdiffeq-analyses/manuscript/figure_s12/notebooks/FigureS12D.ipynb`

**Why it matters**

These are strong supplement / diagnostics templates:

- train-vs-validation curves
- smoothed metrics
- best-epoch markers

**Reuse potential**

Medium-high.

**Template direction**

Keep as a diagnostics module, not main-figure core.

## Best Extraction Targets

Recommended order:

1. stylish box + jitter
2. velocity stream panel
3. volcano panel
4. manifold overlay panel
5. temporal ribbons
6. clustermap support panel

## What Not To Copy

- dataset-specific clone or lineage filtering logic inside notebooks
- hardcoded path conventions
- figure assembly done outside scriptable plotting code
