# Template Gap Backlog

This file tracks template classes that are still missing or underdeveloped, together with the best
published/open sources to mine next.

## Priority A

## 1. Spatial neighborhood / domain templates

Missing:

- spatial domain comparison panel

Best public sources:

- `single-cell-best-practices` spatial chapters
- `squidpy` docs and tutorials
- published Squidpy examples
- see also `inventory_squidpy.md`

## 2. Native scverse wrapper templates

Missing:

- grouped marker/feature wrappers beyond dotplot/matrixplot
- stronger layout-compose wrappers for native scverse outputs

Best public sources:

- `single-cell-best-practices`
- official `scanpy` docs
- official `scvelo` docs
- see also `native_scverse_playbook.md`

## 3. Lineage / fate trend templates

Missing:

- custom-composed lineage dashboards beyond thin native wrappers

Best public sources:

- `CellRank` docs and manuscript notebooks
- published CellRank paper figures
- see also `inventory_cellrank.md`

## 4. Executable page-layout skeletons

Status:

- first pass implemented via `page_layout_skeletons.py`
- still missing demos and more specialized variants

Still missing:

- reusable matplotlib skeletons for:
  - asymmetric hero page
  - compare band + lower proof band
  - matrix-led dashboard page
  - spatial wall + one anchor
  - multi-scale interleave page

Why this matters:

- current layout guidance is strong at the document level, but still too manual in code
- high-quality main-figure production will be more reliable once these page skeletons exist as code, not only prose

Best public sources:

- `main_result_figure_20paper_audit.md`
- `layout_templates.md`
- audited manuscript repos with repeated figure helpers

## 5. Stronger style preset layer

Status:

- first pass implemented via `style_presets.py`
- first-pass shared palette layer now implemented via `palette_bank.py`
- still needs more field-tested page-class presets and real-data demos

Still missing:

- a more opinionated, reusable typography / spacing / legend / colorbar preset layer
- page-class-aware finishers that do more than basic cleanup

Why this matters:

- current style rules prevent obvious mistakes
- they do not yet guarantee the stronger “paper figure” finish by default

Best public sources:

- `repo_code_style_notes.md`
- `high_tier_style_synthesis.md`
- audited published page contact sheets

## Priority B

## 6. Differential abundance / compositional templates

Missing:

- neighborhood DA MA/volcano panel
- compositional compare dashboard
- abundance-with-confidence summary panel

Best public sources:

- `single-cell-best-practices` compositional chapter
- `milo` tutorials
- `pertpy` tutorials
- see also `inventory_pertpy_milo.md`

## 7. Enrichment / validation support templates

Missing:

- compact enrichment lollipop
- pathway/program support dotplot
- supplement-ready validation dashboard

Best public sources:

- `decoupler` tutorials
- `gseapy` examples
- published supplement figures in single-cell methods papers

## Priority C

## 8. Spatial communication overlays

Missing:

- ligand-receptor hotspot overlay with shared scale
- sender/receiver paired spatial panels

Best public sources:

- `cytobridge-downstream`
- published spatial communication papers using Squidpy/CellChat-like displays

## 9. Supplement assembly templates

Missing:

- clean multi-panel supplement dashboard assembly
- appendix-style contact sheet builders with consistent spacing

Best public sources:

- published supplement figures in methods papers
- manuscript assembly scripts from mature single-cell tool repositories
