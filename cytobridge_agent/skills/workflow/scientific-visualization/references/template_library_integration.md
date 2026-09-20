# Template Library Integration

Use this note when `scientific-visualization` should route through the shared figure template
library.

## Library Path

- `/home/ubuntu/CytoBridge-agent/cytobridge_agent/skills/workflow/scientific-visualization/references/figure_template_library`

This path is bundled with the installed skill. Treat it as a plotting grammar,
layout, and style reference. Do not treat it as a data-processing or analysis
library.

## Read Order

### 1. Choose the plot class

Read:

- `decision_tree.md`
- `template_taxonomy.md`
- `template_inputs_and_usage.md`

These answer:

- what basis to use
- whether to show full view or ROI
- whether to stay on a manifold or switch to a summary grammar
- which template class fits the object

### 2. Decide how to style it

Read:

- `style_playbook.md`
- `palette_bank.md`
- `style_variants.md`
- `high_tier_style_synthesis.md`
- `repo_code_style_notes.md`
- `hero_panel_audit.md`
- `result_figure_layout_audit.md`
- `main_result_figure_20paper_audit.md`
- `layout_templates.md`
- `style_presets.py`
- `palette_bank.py`
- `page_layout_skeletons.py`

These answer:

- how the panel should be beautified
- which source repo solved the same plot class differently
- what default visual hierarchy to preserve
- what top-tier papers repeatedly do when ordinary cleanup still feels too thin
- what published result pages actually look like at whole-page scale via `references/high_tier_figure_contacts/`

### 3. Decide whether native plotting is enough

Read:

- `native_scverse_playbook.md`

Use this when a panel may be better drawn with:

- `scanpy`
- `scvelo`
- `squidpy`
- `cellrank`
- `pertpy`

## Template-Selection Heuristic

Use templates first when the panel is:

- method compare
- overlay-heavy
- perturbation-oriented
- layout-sensitive
- matrix/network rich

Use native wrappers first when the panel is:

- a standard embedding
- a standard stream plot
- a standard dotplot or matrixplot
- a standard neighborhood enrichment or interaction matrix

## Important Constraints

- The template library is pure plotting or thin wrapper logic.
- It should not be treated as an analysis library.
- Source repos are style references, not plotting backends.
- Private project packages should not become template dependencies.
- Broad ecosystem packages such as `scanpy`, `scvelo`, `squidpy`, `cellrank`, and `pertpy`
  are acceptable optional dependencies.
- For CytoBridge figures, first inspect the downstream-analysis script,
  manifest, and source artifact. Do not change model outputs, labels,
  filtering, dimensionality, or feature space to satisfy a template.
- Final manuscript figures should be script-generated and exported as both PNG
  and editable vector PDF, with the script and source artifact path recorded.

## Most Important Docs in the Library

- `README.md`
- `inventory_high_tier_papers.md`
- `high_tier_style_synthesis.md`
- `repo_code_style_notes.md`
- `hero_panel_audit.md`
- `result_figure_layout_audit.md`
- `main_result_figure_20paper_audit.md`
- `layout_templates.md`
- `decision_tree.md`
- `template_taxonomy.md`
- `style_playbook.md`
- `style_presets.py`
- `style_variants.md`
- `template_inputs_and_usage.md`
- `native_scverse_playbook.md`

## Examples of High-Value Template Classes

- stylish box + jitter
- manifold overlay
- velocity stream
- embedding + generated trajectory overlay
- perturbation embedding compare
- embedding time-overlay
- perturbation dose-response
- transition heatmap
- spatial small multiples
- annotated block heatmap
- bipartite pathway network

## Practical Rule

Do not force a template if the panel needs a small local adaptation.
The correct workflow is:

1. choose the nearest template
2. inherit its grammar and style logic
3. adapt the details to the data

The library is a starting point with source-backed defaults, not a rigid cage.
