# moscot Figure Audit

## Bottom Line

`moscot` is different from `scDiffEq` and `moslin` in one important way:

- it contains a **real plotting API**
- not just manuscript notebooks

The main plotting entry points are:

- `/lustre/home/2501111653/moscot/src/moscot/plotting/_plotting.py`
- `/lustre/home/2501111653/moscot/src/moscot/plotting/_utils.py`

Public plotting functions are documented in:

- `/lustre/home/2501111653/moscot/docs/user/plotting.rst`

Core public plot types:

- `cell_transition`
- `sankey`
- `push`
- `pull`

## Figure Types Worth Extracting

### 1. Annotated transition heatmap

**Where**

- `/lustre/home/2501111653/moscot/src/moscot/plotting/_plotting.py`
  via `cell_transition`
- rendered by:
  `/lustre/home/2501111653/moscot/src/moscot/plotting/_utils.py`
  via `_heatmap`

**Why it matters**

This is a high-quality reusable grammar for:

- transition matrices
- row/column category annotation bars
- annotated values
- consistent colorbar behavior

**Reuse potential**

High.

**Template direction**

This is the best `moscot` extraction target for the library.

---

### 2. Sankey transition diagram

**Where**

- `/lustre/home/2501111653/moscot/src/moscot/plotting/_plotting.py`
  via `sankey`
- implemented in:
  `/lustre/home/2501111653/moscot/src/moscot/plotting/_utils.py`
  via `_sankey`

**Why it matters**

Unlike the `moslin` zebrafish alluvial, this is already a clean plotting API.

It is useful for:

- category mass transfer across consecutive stages
- transport summaries over multiple timepoints

**Reuse potential**

High for grammar.

**Template direction**

Good candidate for a second flow-style template alongside alluvial.

---

### 3. Push / pull embedding scatter

**Where**

- `/lustre/home/2501111653/moscot/src/moscot/plotting/_plotting.py`
  via `push` and `pull`
- implemented in:
  `/lustre/home/2501111653/moscot/src/moscot/plotting/_utils.py`
  via `_plot_scatter`

**Why it matters**

This is a more formalized version of the “highlight selected batches or subsets on an embedding” grammar.

Useful for:

- transport mass projected onto embedding
- comparing selected timepoints or subsets
- support panels around transport destinations

**Reuse potential**

Medium-high.

**Template direction**

This overlaps strongly with the generic manifold overlay template, but offers more explicit input conventions.

---

### 4. Backend optimization curves

**Where**

- `/lustre/home/2501111653/moscot/src/moscot/backends/ott/output.py`
  via `plot_costs` and `plot_errors`

**Why it matters**

These are useful diagnostics and supplement templates:

- optimization cost over iterations
- optimization error over iterations

**Reuse potential**

Medium.

**Template direction**

Useful as diagnostics; not a main-figure anchor.

## What moscot Adds to the Template Library

Compared with the other two repos:

- `scDiffEq` contributes manuscript-oriented figure grammars
- `moslin` contributes benchmark and transition-story grammars
- `moscot` contributes a cleaner **API-style plotting interface**

So `moscot` is especially valuable for:

- standardized inputs
- explicit plotting contracts
- generic transition heatmap and sankey patterns

## Best Extraction Targets

Recommended order:

1. annotated transition heatmap
2. sankey diagram
3. push/pull scatter grammar
4. optimization curves

