# Result Figure Layout Audit

## Scope

This note focuses on **result figures**, not entry-point workflow/cartoon figures.

It is now superseded and strengthened by:

- `main_result_figure_20paper_audit.md`

The present note is still useful as a compact summary, but the newer audit is the better source
for concrete page skeletons and paper-by-paper patterns.

The question here is:

- how do strong papers actually lay out result panels?
- how many panels do they use?
- how do they stop rich result pages from turning messy?
- which result-panel grammars feel especially memorable or distinctive?

## Main Result

The strongest result figures are usually **not** built as a flat list of many panels.

They are built as **2-3 grouped zones**:

1. a top result zone
2. a mechanistic or trend-support zone
3. optionally a benchmark/validation zone

That is a more useful standard than “how many panels should a main figure have”.

## Repeated Layout Patterns

### Pattern A: compare row + support row

Seen often in trajectory, fate, and method papers.

Structure:

- top: one row of 2-4 shared-basis panels
- bottom: distributions, trends, or mechanism support

Why it works:

- the reader understands the object quickly
- the second zone changes grammar and adds credibility

This is the most reusable layout for our own work.

### Pattern B: large left hero + compact right block

Seen in spatial and integration papers.

Structure:

- one large geometry/spatial result on the left
- one compact quantitative or comparative block on the right
- lower row then expands mechanism or validation

Why it works:

- one panel earns attention
- the right block keeps the page from feeling underpowered

### Pattern C: disciplined dashboard

Seen in benchmark-heavy papers such as integration or toolkit papers.

Structure:

- many panels, but repeated glyphs and repeated scales
- strong grid discipline
- typography and legend policy extremely controlled

Why it works:

- richness comes from repeated structure, not from one dominant manifold

### Pattern D: spatial compare wall with one statistical anchor

Seen in spatial mapping/deconvolution papers.

Structure:

- many same-basis spatial children
- one aligned score/distribution/heatmap anchor nearby

Why it works:

- the page is visibly about one basis
- one non-spatial panel keeps it from becoming wallpaper

## How Many Panels?

Across the audited result figures, the raw panel count is often around `6-12`, but the more
important number is:

- **usually 2 or 3 visual zones**

Good result pages tolerate many panel letters because the zones are legible.
Bad result pages fail even with 5-6 panels because the grouping is weak.

The 20-paper audit also made one additional point clearer:

- even published papers can have weak result-page composition

So these rules should be used selectively, by copying the strongest skeletons rather than the
average published page.

## What Makes a Result Figure Feel Impressive

### 1. Shared-basis discipline

The impressive pages often reuse:

- identical crop
- identical aspect
- identical point/spot scale
- identical color semantics

across a whole compare zone.

This gives the page authority.

### 2. Immediate grammar change after the compare zone

The strongest pages do not spend the whole figure on more of the same manifold.

They pivot quickly into:

- trends
- distributions
- heatmaps
- perturbation summaries
- or targeted zooms

That shift is a major source of “richness”.

### 3. One dominant object, one supporting proof

Impressive result figures usually feel like:

- “here is the object”
- “here is the proof it matters”

in very short distance.

### 4. High density without text overload

Good pages are often information-dense but text-light.

They get there by:

- repeated panel structure
- careful whitespace
- selective labels
- one external legend

not by adding more explanatory sentences to the canvas.

## Most Impressive Result-Figure Styles In This Audit

### `Learning single-cell perturbation responses using neural optimal transport`

Why it stands out:

- result figures mix compare, intervention, and mechanism grammars cleanly
- embeddings are coupled to perturbation outputs, not left isolated

What to learn:

- build result pages out of **grammar transitions**, not repeated embeddings

### `Generalizing RNA velocity to transient cell states through dynamical modeling`

Why it stands out:

- result panels move naturally from geometry to kinetics to gene-level support

What to learn:

- dynamic papers should not stop at the velocity field; they should quickly add process-level companions

### `CellRank 2`

Why it stands out:

- extremely disciplined shared-basis compare
- support panels feel subordinate but necessary, not decorative

What to learn:

- strong result figures often look calm because the compare row is rigidly controlled

### `Mapping cells through time and space with moscot`

Why it stands out:

- transport/coupling results are turned into visually diverse but still coherent result zones

What to learn:

- if the method object is nontrivial, use multiple coordinated grammars instead of forcing everything into one matrix

### `Multi-omic single-cell velocity...`

Why it stands out:

- result pages feel rich because modality, dynamics, and gene trends are interleaved

What to learn:

- multiview papers benefit from pairing different scales of evidence in one page

## Code-Level Habits Behind Good Result Layouts

From the audited repos, strong result pages are usually backed by code that enforces:

- shared basis/crop across compare children
- fixed palettes across a panel family
- detached external legends
- figure-level export settings
- helper functions for repeated panel furniture

This is why “just using the same plot type” is not enough.
The code itself often encodes the page discipline.

## Practical Rules For Our Skill

When planning a result figure:

1. Decide the zones first, not the panel count.
2. Decide which zone is the compare/geometry zone.
3. Force the next zone to change grammar.
4. Ensure at least one nearby quantitative or mechanistic proof exists for the main visual claim.
5. If a result page still feels ordinary, it usually needs:
   - stronger shared-basis discipline
   - a better companion block
   - or a clearer zone transition

## What To Avoid

- result figures that are only repeated manifolds
- mechanism rows that look like tiny spreadsheets
- too many unrelated small panels with no zoning
- pages where the quantitative proof is too far from the main visual claim
