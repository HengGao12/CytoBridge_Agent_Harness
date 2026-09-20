---
name: scientific-visualization
description: "Create publication figures and manuscript panels with a standardized operating manual. Use for scientific plots, multi-panel figure construction, figure refactoring, journal-style cleanup, and when a figure should be chosen or styled via the shared figure template library, native scverse plotting conventions, or custom matplotlib primitives."
---

# Scientific Visualization

Use this skill for manuscript figures, supplement figures, figure refactoring, and panel-level
visual QA. This skill operates as an **SOP + template selection manual + custom visualization
guide**, not just a bag of matplotlib tips.

## Default Operating Mode

When a user asks to make or improve a figure:

1. **Understand the upstream analysis first.** Before drawing anything, read the analysis scripts
   that produced the data. The figure design should be informed by how the result was computed,
   not just the shape of the output table. Misunderstanding the analysis leads to panels that
   look right but say the wrong thing.
2. Define the biological object and the one-sentence claim. If the claim is unclear or evolving,
   that is normal—see "Iterative Claim Discovery" below.
3. Decide the coordinate basis:
   - observed manifold
   - latent manifold
   - spatial coordinates
   - no coordinate basis
4. Decide whether the panel should show:
   - full view
   - ROI/zoom
   - side-by-side compare
   - distribution/statistical summary
   - dynamics
   - perturbation/intervention
   - support/validation
5. Check the template library for a matching template. **If no template matches within 2 minutes
   of browsing, stop searching and go custom**—build the panel from matplotlib primitives
   directly. See "Custom Visualization Mode" below.
6. Apply the relevant style rules before polishing labels.
7. Export a script-generated figure only. No manual collage for final figures.

## Iterative Claim Discovery

The decision tree assumes the claim is known upfront. In practice, especially for methods papers,
the claim often co-evolves with the visualization:

- The user may not know what the figure should show until they see a draft.
- A draft may reveal that the underlying analysis does not support the intended narrative.
- The "right" panel type may only become clear after 2-3 failed attempts with other encodings.

This is normal. When it happens:

1. Do not force a premature claim. Generate a quick exploratory panel first.
2. After the user reacts, refine the claim and the encoding together.
3. If the user questions a scientific conclusion shown in the panel, **pause figure work and
   investigate the upstream analysis code** before redrawing. Do not redraw with a different
   encoding hoping the problem goes away.
4. Keep a running record of what was tried and why it was rejected, so you do not revisit
   dead ends.

## Hard Rules

- White background by default.
- Default matplotlib output is not manuscript-ready. A final figure must pass a
  deliberate visual-design pass: clear focal point, quiet context, restrained
  palette, readable typography, aligned panels, and no spreadsheet-like panels.
- Default manuscript page assumption: vertical A4 unless the user says otherwise.
- Do not add descriptive prose onto the canvas unless explicitly needed.
- Do not keep using the same manifold grammar once the question has changed.
- Prefer clean, aligned, non-thin figures over fake complexity.
- When a result still feels thin, first change the grammar or add the right companion panel before adding decorative styling.
- If a fancy encoding loses clarity, fall back to a cleaner one.
- Final figures must come from scripts and remain editable.
- Analysis logic and plotting logic should stay separate whenever possible.
- Before delivery or approval, run one explicit **layout-only** review pass that ignores biological interpretation and checks panel letters, alignment, gutters, colorbar placement, edge clipping, and page hygiene.
- Do not package legacy outputs unchanged just because they already exist. If an old script/figure violates the current style rules, patch the script and rerun it.
- Do not promote a stitched contact sheet or pasted montage as a final figure. Final approved figures must be regenerated from source scripts and source analysis outputs.
- Do not accept "technically correct but ugly" figures for main-paper use. If a
  panel looks like raw debug output, an equal-grid metric dump, tiny default
  text, rainbow colors, or a mean-line placeholder, redesign the grammar before
  polishing.

## Environment Rule

Default project plotting environment for CytoBridge/CellCompass use:

- Use the current CytoBridge/CellCompass Python environment or the active task
  environment. Do not create a new plotting environment unless the user asks.
- Save reusable figure scripts under the active output directory, preferably
  `<output_dir>/scripts/figures/` or `<output_dir>/scripts/downstream/`.
- Save final manuscript-ready figures under the active output directory,
  preferably `<output_dir>/outputs/figures/` or `<output_dir>/paper/figures/`.

Optional plotting packages such as `scanpy`, `scvelo`, `squidpy`, `cellrank`,
`pertpy`, `seaborn`, or `matplotlib` may be used when already available. If an
optional dependency is missing, prefer a clean matplotlib implementation over
changing the analysis or silently dropping the figure.

## CytoBridge Integration

For CytoBridge algorithm lifecycle work, use this skill after downstream
analysis has produced a model-derived table, trajectory, projection, or
summary artifact, and before final paper/report figure selection.

- Read the upstream downstream-analysis script, manifest, and source artifact
  first. A figure should present evidence, not create new evidence.
- For generated trajectories, use package trajectory plotting helpers before
  custom plotting: `CytoBridge.pl.downstream.plot_ode_trajectories_bundle(...)`
  for deterministic/evaluation-kernel rollouts and
  `CytoBridge.pl.downstream.plot_sde_trajectories_bundle(...)` for stochastic or
  SDE-compatible rollouts. If a custom trajectory figure is needed, show path
  ensembles, time-sliced generated states, endpoint clouds, fate-colored
  particles, uncertainty ribbons, or KDE contours. Do not make the main dynamics
  panel a single mean/centroid trajectory unless the paper's claim is explicitly
  about the mean; a mean path belongs only as a companion summary.
- Do not alter model outputs, labels, filtering, dimensionality, or biological
  interpretation only to make a figure look better.
- Keep analysis and plotting separate when possible. If a plotting script must
  compute small display-only summaries, document them clearly.
- Export both a review PNG and an editable vector PDF for manuscript figures.
- Record figure provenance: source data path, script path, figure path,
  intended claim, and any warnings about projection, labels, or feature space.
- Use the template library as style grammar and layout precedent, not as an
  analysis backend.

## Custom Visualization Mode

Enter custom mode when:

- No template in the taxonomy matches the panel's message within 2 minutes of browsing.
- The panel type is novel to the project (e.g., trajectory divergence with KDE contours,
  Jacobian-derived regulatory circuit diagrams, along-trajectory gene dynamics heatmaps).
- The user explicitly says "don't use templates" or "make something custom."

Custom mode rules:

- Build from `matplotlib` primitives (`fig.add_axes`, `ax.scatter`, `ax.annotate`,
  `matplotlib.patches`, `matplotlib.collections`).
- Still follow the global style rules (typography hierarchy, white background, quiet context).
- Still follow `axis_policy.md` and `panel_finish_checklist.md`.
- Document the custom panel's structure in comments so it can be maintained later.
- After the panel is approved, consider whether it generalizes enough to become a new template.
  If yes, extract it into the template library.

Common custom panel families for methods papers:

- **Trajectory divergence**: UMAP background + time-sliced trajectory positions + KDE contours.
  Key choices: which cells form the background, how to define fate commitment, contour levels.
- **Along-trajectory gene dynamics**: Heatmap where x = pseudo-time along generated trajectory,
  y = genes, color = z-scored expression estimated via kNN. Key choices: which cells to include,
  gene selection, normalization.
- **Regulatory circuit diagram**: Node-arrow diagram where nodes = gene modules, arrows = directed
  influence (e.g., from Jacobian perturbation), arrow color = activation/suppression, arrow
  thickness ∝ magnitude. Key choices: module definitions, layout positions, influence threshold.
- **Fate score gallery**: Small multiples of the same UMAP background, each highlighting a
  different cell type with a scalar overlay (e.g., fate probability). Key choices: cell type
  selection, shared vs per-panel color scale.

## Template Library

Use the template library when a matching template exists, but **do not force-fit** a panel into
a template that does not match its message. Templates are a starting point, not a constraint.
Do **not** copy a source paper literally; reuse its plotting grammar and style logic.

Primary library path:

- `/home/ubuntu/CytoBridge-agent/cytobridge_agent/skills/workflow/scientific-visualization/references/figure_template_library`

Read this reference when using the library:

- `references/template_library_integration.md`

The library contains:

- decision tree for choosing plot classes
- taxonomy of template classes (including generative-model-specific templates)
- high-tier paper inventory
- high-tier style synthesis
- repository code-style notes
- hero-panel audit
- result-figure layout audit
- style playbook
- axis policy
- panel finish checklist
- shared style defaults helper
- stronger page-level style presets
- executable page-layout skeletons
- style variants across source repos
- per-template input/usage rules
- native scverse plotting playbook
- source-backed inventories
- reusable Python templates
- approved panel workflow guide

## What to Read, and When

Read only what the current task needs.

### Always relevant for manuscript figure work

- `references/iterative_figure_workflow.md`
- `references/journal_figure_taste.md`
- `references/template_library_integration.md`

### Read when choosing a plot class

From the template library:

- `decision_tree.md`
- `template_taxonomy.md`
- `template_inputs_and_usage.md`

### Read when styling or cleaning a panel

From the template library:

- `aesthetic_principles.md` (visual design principles, "fancy figure" techniques, Nature Methods checklist)
- `style_playbook.md`
- `high_tier_style_synthesis.md`
- `palette_bank.md`
- `axis_policy.md`
- `panel_finish_checklist.md`
- `style_variants.md`
- `style_defaults.py`
- `palette_bank.py`
- `style_presets.py`
- `page_layout_skeletons.py`

Also read `aesthetic_principles.md` when:

- a figure is correct but feels "ordinary" or "too academic"
- you want specific techniques to make panels look more polished (KDE contours, text halos,
  quantile clipping, clean axis removal, etc.)
- you need the "Nature Methods checklist" before declaring a figure ready
- the user asks for something "fancy" or "impressive"

Also read `inventory_high_tier_papers.md` when:

- a figure still feels ordinary after normal cleanup
- a row feels too thin even though the code is tidy
- you need a stronger published precedent for how to enrich or pair a panel

Read `main_result_figure_20paper_audit.md` when:

- planning a main result figure from scratch
- a page feels ordinary and you need real published page precedents
- you want to know what strong papers actually do with hero size, companions, and zone shifts
- you need to decide whether the page should be asymmetric, compare-band, matrix-led, spatial-wall, or multi-scale

Do not assume every published page is a good page.
Use the audit to copy the strongest skeletons, not the average published page.

Use `references/high_tier_figure_contacts/` when:

- you want a quick visual reminder of audited published page skeletons
- you want to compare candidate page weight against real published result pages
- the page feels too sparse or too evenly gridded and you need a visual calibration

Read `repo_code_style_notes.md` when:

- you want to know which repo actually implemented a useful rendering habit
- you need a concrete code precedent for a panel family, not just a high-level style rule

Read `hero_panel_audit.md` when:

- planning the top half of a main figure
- deciding whether a figure has a real hero zone
- the page feels clean but still not memorable

Read `result_figure_layout_audit.md` when:

- designing a result figure after the intro/workflow page
- deciding how many panel zones a result page really needs
- trying to understand why published result pages feel dense but still coherent

Read `layout_templates.md` when:

- choosing an actual page skeleton rather than only a raw panel count
- deciding between portrait and landscape
- needing a concrete starting layout rather than only abstract zone logic

### Read when standard ecosystem plotting may be enough

From the template library:

- `native_scverse_playbook.md`

### Read when setting up approved panel archiving

From the template library:

- `approved_panel_workflow.md`

## Figure-Type Decision Rules

Use this compact decision tree first:

### If the question is geometric

Stay on a coordinate basis:

- global manifold/spatial overview
- ROI zoom
- method compare on one shared basis
- latent geometry panel
- embedding + trajectory overlay
- trajectory divergence (time-sliced positions + KDE contours on a shared embedding)
- perturbation projected onto one shared embedding

### If the question is statistical or mechanistic

Stop drawing manifolds and switch encoding:

- violin / box / raincloud for state distributions
- ribbons / aligned tracks for dynamics
- bars / dose-response / volcano for perturbations
- heatmap / block heatmap / interaction matrix when the matrix itself is the object
- along-trajectory gene dynamics heatmap when the object is expression change over pseudo-time
- network / bipartite network when relations are the object
- regulatory circuit diagram when module-to-module influence is the object

### If the current panel repeats the previous one

Change the plotting grammar, not just the colors.

### If no existing grammar fits

Enter custom visualization mode. See the "Custom Visualization Mode" section above.

## Typography and Axis Policy

Do not improvise typography or axes panel by panel.

- Use one stable hierarchy unless the figure truly needs a special treatment:
  - panel letter about `11-12 pt`, bold
  - panel title about `9-10 pt`, bold
  - axis label about `8-9 pt`
  - tick / legend about `7-8 pt`
- Use `axis_policy.md` to decide whether axes, spines, and ticks stay or go.
- Use `style_defaults.py` when you want a stable base instead of ad hoc rcParams.
- For colors, check domain-specific conventions first (see "Analysis-Visualization Synergy"
  below), then fall back to `palette_bank.py` for generic classes:
  - quiet context
  - categorical atlas
  - signed perturbation
  - sequential scalar overlay
  - diverging matrix
  - benchmark compare
- Before approving a panel, run `panel_finish_checklist.md` mentally:
  - typography
  - axes/frame
  - layout
  - legend/labels
  - visual weight
  - export safety

## Preferred Workflow by Figure Stage

### 1. Figure planning

Use the decision tree to map each planned panel to:

- object
- basis
- view scope
- panel type
- candidate template (or "custom" if no template matches)

Also decide the figure zones:

- compare/geometry zone
- support/mechanism zone
- optional validation zone
- optional secondary model-result zone

### Corrections From The 20-Paper Audit

Earlier versions of this skill were too rigid in two ways.

They overemphasized:

- forcing every zone to serve one literal story sentence
- choosing layouts by raw panel count too early

The corrected rule is:

- first choose the **page skeleton**
- then assign zones
- then decide whether a secondary model-result block is justified
- only then decide raw panel count

Also:

- a strong result page is often denser and more asymmetric than a clean but ordinary project figure
- page quality depends more on hero weight, companion proof, and grammar transition than on cosmetic cleanup alone

### 2. Candidate generation

For a nontrivial panel, generate 2-5 encodings if needed.
Do not over-polish a weak grammar.

### 3. Main-figure assembly

Prioritize:

- reading order
- outer-edge alignment
- clean grouping
- non-thin visual weight
- one clear hero row before lower support rows

Do not think only in panel letters here.
Use `hero_panel_audit.md` to decide whether the top half should behave like:

- workflow + example
- shared-basis compare
- geometry + quantitative companion
- structured dashboard

For result-heavy pages after the entry figure, also use `result_figure_layout_audit.md`:

- prioritize zone structure over raw panel count
- force grammar change after the compare zone
- keep the main proof close to the main visual claim

Then use `main_result_figure_20paper_audit.md` and `layout_templates.md` together:

- identify which published page skeleton the figure most resembles
- decide whether the page is hero-led, compare-band, matrix-led, spatial-wall, or multi-scale
- choose panel count only after the skeleton is clear
- let page asymmetry come from the skeleton rather than forcing equal-sized chores panels

Do not interpret this too rigidly.
A strong algorithm paper may also include a secondary result block that is not the single main
biological story, as long as it is:

- clearly relevant to the method
- logically self-consistent with the rest of the page
- not dominated by results unrelated to the algorithm

In other words:

- prefer coherence over narrowness
- the whole page does **not** need to behave like one literal claim sentence
- but every zone should still justify its space

### 4. Approval and archiving

Use the **per-panel approved directory workflow** described in `approved_panel_workflow.md`.

For any approved panel, preserve:

- final PNG and **editable vector PDF** (see PDF export rules below)
- final self-contained plotting script
- key intermediate tables if used by the plot
- provenance from active analysis outputs
- a `README.md` in the approved root describing each panel: content, data sources, analysis
  principle, and script locations

#### PDF Export Rules

- Default: all elements should be vector (editable in Illustrator/Inkscape).
- **Do not** use `rasterized=True` on `ax.scatter` unless the point count exceeds ~50,000 AND
  the PDF will not be edited downstream. For typical single-cell panels (< 20,000 cells),
  keep scatter points as vector.
- `ax.imshow` (heatmaps) is inherently raster; this is fine.
- Always set `dpi=300` or higher for the PDF `savefig` call.
- After saving, spot-check the PDF by zooming to 400% on a scatter-heavy panel. If background
  points appear pixelated, find and remove `rasterized=True`.
- When both PNG and PDF are saved, the PNG can use higher DPI (e.g., 400) for quick preview,
  while the PDF prioritizes editability.

### 5. Pre-delivery QA

Before sending a figure candidate to the user as a likely final or near-final delivery:

- run one review pass for message/content
- run one separate **layout-only** review pass
- run one explicit "ugly-figure" pass: check for default matplotlib colors,
  cramped labels, tiny fonts, unaligned panels, overlong legends, unsupported
  rainbow palettes, uninformative mean-only dynamics, and equal-size grids that
  should instead have a hero panel plus compact companions.

The layout-only pass should ignore scientific claims and focus only on:

- panel-letter placement
- alignment and outer edges
- whitespace and gutters
- axis/title/legend collisions
- colorbar placement
- page-edge clipping

Do not skip this step just because the figure already "looks fine".

## Native vs Template vs Custom

Use **native scverse wrappers** when:

- a standard Scanpy/scVelo/CellRank/Squidpy panel already matches the message
- custom reimplementation would add no value

Use **library templates** when:

- the panel requires stronger manuscript styling
- overlays, comparisons, or layout control exceed what the native API gives cleanly

Use **custom matplotlib** when:

- the panel type does not exist in the template library or native ecosystem
- the visualization is specific to the method being presented (e.g., generative model outputs,
  Jacobian-derived networks, ODE-integrated trajectory visualizations)
- the user explicitly requests freedom from predefined templates
- multiple iteration attempts with templates have not produced a satisfactory result

## Analysis-Visualization Synergy

This skill is primarily for plotting and figure structure, not for running upstream analyses.
However, good figures require understanding the analysis that produced the data:

- **Before drawing a mechanistic panel** (e.g., gene regulatory networks, perturbation effects,
  trajectory dynamics), read the analysis script that generated the input data. Understand what
  the numbers mean, how they were computed, and what biological conclusion they support.
- **If the user questions a conclusion shown in the figure**, do not just try a different visual
  encoding. First verify whether the conclusion is actually supported by the data. Read the
  original analysis code, check the intermediate outputs, and report findings before redrawing.
- **Domain-specific color semantics** often come from the biology, not from a generic palette
  bank. For example, in lineage tracing: BM = warm red, AL = cool blue, FP = green. These
  conventions should be defined once as named constants at the top of the script and reused
  across all panels in the figure. The palette bank is useful for generic classes (quiet
  context, diverging matrix), but domain colors take priority when they exist.

This skill is not a substitute for the upstream scientific analysis, benchmark computation,
perturbation inference, or fate estimation. Those should come from the project's analysis code.
The figure layer should consume prepared results and render them clearly—but the figure designer
must understand the results first.

## Scripts Bundled With This Skill

Use the local helpers for styling/export:

- `scripts/style_presets.py`
- `scripts/figure_export.py`

These help with:

- publication defaults
- journal sizing
- consistent export
