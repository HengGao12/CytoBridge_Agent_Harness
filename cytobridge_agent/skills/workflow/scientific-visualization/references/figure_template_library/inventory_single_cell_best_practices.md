# single-cell-best-practices Figure Inventory

This note audits the plotting value of:

- `/lustre/home/2501111653/single_cell_best_practices_fetch`

## High-Level Verdict

This repository is **not** primarily a handcrafted plotting-code repository.

Its value is different from `scDiffEq`, `moslin`, `moscot`, or `cytobridge-downstream`:

- it is a broad tutorial/book-style repository
- most plotting happens directly in notebooks
- many figures are produced through native `scanpy`, `scvelo`, `squidpy`, or `seaborn` calls
- there are few reusable custom plotting helpers

So it should be treated as:

- a source of **native scverse plotting conventions**
- a source of **when-to-use-which-plot** examples
- a weaker source of bespoke matplotlib template extraction

## Main Value for the Template Library

## 1. Native scanpy/scvelo usage patterns

Repeated patterns across the book include:

- `sc.pl.umap(..., frameon=False)`
- `sc.pl.umap(..., legend_loc="on data")`
- `sc.pl.dotplot(...)`
- `sc.pl.matrixplot(...)`
- `sc.pl.rank_genes_groups_dotplot(...)`
- `sc.pl.violin(..., multi_panel=True)`
- `scv.pl.velocity_embedding_stream(...)`

These are not custom templates, but they are highly reusable **native grammars**.

## 2. Consistent white-background defaults

The repository repeatedly uses or recommends:

- `sc.settings.set_figure_params(dpi=..., facecolor="white")`
- white-background seaborn settings
- frame-free embeddings for annotation-focused views

This supports our current template-library rule that:

- white background should be the default
- manifold frames are often unnecessary
- on-data labels are acceptable when the embedding itself is the object

## 3. Useful notebook-level grammar examples

Some notebook sections are still worth remembering as grammar references:

- QC histograms and jointplots
- dotplot / matrixplot for marker summaries
- scanpy embedding panels with controlled `ncols` and `wspace`
- scvelo stream plots on a UMAP basis
- spatial notebook sections that rely on native `squidpy` / `scanpy` plot types

## What It Does *Not* Add Strongly

This repo does **not** currently add much in the way of:

- custom layout engines
- manuscript-specific plotting helper modules
- reusable bespoke aesthetics beyond standard scverse practice

So it should not be over-weighted as a source of custom template code.

## What To Extract From It

The right way to use this repository in the template library is:

1. document it as a source of **native scverse plotting defaults**
2. treat it as support for allowing `scanpy/scvelo/squidpy` native plots in the library
3. avoid pretending that it provides many new custom matplotlib figure classes

## Template-Library Impact

This repository mainly strengthens:

- the dependency whitelist for high-impact scientific plotting packages
- the decision rule for when native scverse plots are already good enough
- the use of scanpy/scvelo-native grammars without unnecessary reimplementation
