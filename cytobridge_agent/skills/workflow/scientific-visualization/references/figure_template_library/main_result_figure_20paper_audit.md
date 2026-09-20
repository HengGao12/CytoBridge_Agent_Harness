# Main Result Figure 20-Paper Audit

## Purpose

This note audits **main result figures/pages**, not only Figure 1 workflow pages and not only
individual plotting helpers.

The goal is to answer, with source-backed examples:

- how strong published computational biology papers actually lay out their result pages
- how many visual zones they use
- what kind of hero panel they choose
- how they mix geometry, statistics, dynamics, perturbation, and mechanism in one page
- which page types look rich for good reasons, and which published pages are still visually weak

This note should be read together with:

- `hero_panel_audit.md`
- `result_figure_layout_audit.md`
- `layout_templates.md`
- `repo_code_style_notes.md`

## Audit Basis

### What was inspected

- local PDFs from the literature store:
  `/lustre/home/2501111653/CytoBridge-agent/cytobridge_agent/rag/literature_db/literature`
- main-article pages, usually the page range where the paper's main figures live
- not only code repos, because page-level composition often lives in the manuscript layout itself

### Audit artifacts

Rendered contact sheets were created under:

- `/lustre/home/2501111653/tmp/pdfs/high_tier_page_audit`
- `/lustre/home/2501111653/tmp/pdfs/high_tier_page_audit_extra`
- `/lustre/home/2501111653/tmp/pdfs/high_tier_page_audit_extra2`

These were used to inspect whole result pages rather than isolated panels.

## Audited Set

### Core high-tier methods/tool papers

1. `Reversed graph embedding resolves complex single-cell trajectories`
2. `RNA velocity of single cells`
3. `A comparison of single-cell trajectory inference methods`
4. `NicheNet: modeling intercellular communication by linking ligands to target genes`
5. `Generalizing RNA velocity to transient cell states through dynamical modeling`
6. `Highly sensitive spatial transcriptomics at near-cellular resolution with Slide-seqV2`
7. `Benchmarking atlas-level data integration in single-cell genomics`
8. `Multi-omics single-cell data integration and regulatory inference with graph-linked embedding`
9. `Learning single-cell perturbation responses using neural optimal transport`
10. `Multi-omic single-cell velocity models epigenome-transcriptome interactions and improves cell fate prediction`
11. `CellRank 2: unified fate mapping in multiview single-cell data`
12. `Mapping cells through time and space with moscot`
13. `Gene trajectory inference for single-cell data by optimal transport metrics`
14. `Pertpy: an end-to-end framework for perturbation analysis`

### Additional published pages used to widen the page-layout sample

15. `PAGA graph abstraction reconciles clustering with trajectory inference through a topology preserving map of single cells`
16. `Reconstructing growth and dynamic trajectories from single-cell transcriptomics data`
17. `Robust integration and annotation of single-cell and spatial omics data using interpretable gene programs`
18. `Improved single-cell ATAC-seq reveals chromatin dynamics of in vitro corticogenesis`
19. `Mapping lineage-traced cells across time points with moslin`
20. `Constructing the spatiotemporal atlas of single-cell lineage trajectories in stereotypic biological structures`

## Important Reality Check

Not every published paper has strong page design.

This audit is useful partly because it shows both:

- page types worth copying
- page types that are published but still visually weak or overtextual

So the lesson is not "published means automatically beautiful".
The lesson is to identify the repeated **strong grammars** inside the good pages.

## Paper-Level Notes

| Paper | Dominant result-page skeleton | Most memorable / useful hero object | What the page does well | What not to imitate blindly | Closest template family |
|---|---|---|---|---|---|
| Reversed graph embedding... | text-heavy brief format with one compact figure object | compact trajectory topology block | one page can still work if the figure object is conceptually sharp | too much text around the object; little whitespace control | compact four-panel portrait |
| RNA velocity of single cells | geometry page followed by process support | velocity field plus phase portraits | moves from manifold geometry to process evidence quickly | raw article page is not spacious; needs modern cleanup if reused | geometry hero + quantitative companion |
| A comparison of single-cell trajectory inference methods | benchmark dashboard | matrix/table-like benchmark block | strong repeated structure; one page can carry many methods | can become visually cold if not paired with a more biological anchor | matrix-led benchmark page |
| NicheNet | heatmap-led mechanism page | ligand-target heatmap block | matrix is contextualized by supporting summaries | page can feel cramped if copied literally | matrix hero + companion support |
| Generalizing RNA velocity... | shared-basis geometry pages with dynamic companions | stream/manifold compare tied to kinetics/phase panels | excellent grammar transition from geometry to process | too many same-basis panels would feel repetitive without the companions | compare row + dynamic support row |
| Slide-seqV2 | spatial wall pages | dense spatial maps | strong same-basis discipline; spatial abundance is the object | multiple spatial maps alone can become wallpaper | spatial wall + statistical anchor |
| Benchmarking atlas-level integration... | disciplined benchmark dashboard | one large score matrix / ranking block | repeated glyphs, repeated scales, rigid grid | not every paper should imitate this if the matrix is not the object | matrix-led benchmark page |
| GLUE | top method/result block then multiscale support | multimodal embedding plus downstream regulatory blocks | interleaves geometry, curves, heatmaps, and modality-specific support | page would fail if all panels were treated equally; it depends on hierarchy | multi-scale interleave page |
| CellOT | compare/intervention page | perturbation embedding plus response summaries | impressive because the page changes grammar early and often | some pages are text-dense in the original typeset version | perturbation compare + intervention support |
| Multi-omic single-cell velocity... | multiview dynamics page | modality-aware dynamic manifold block | different evidence scales are interleaved well | can get busy if legends and color semantics drift | multi-scale interleave page |
| CellRank 2 | calm shared-basis compare with subordinate support | multiview fate compare | very disciplined same-basis children; lower support feels necessary, not decorative | requires strong crop and palette discipline to work | shared-basis compare page |
| moscot | workflow/result blend then coupling/transport pages | coupling/transition plus map companions | succeeds by using more than one grammar for transport | if reduced to only a matrix, the page would flatten | workflow + decisive example; transport dashboard |
| Gene trajectory inference by OT metrics | trajectory method page with compare/support mix | trajectory compare + validation summaries | page richness comes from mixing embeddings, ribbons, and validation blocks | can become crowded if every support block gets equal weight | compare row + support row |
| Pertpy | toolkit dashboard | large mixed-method dashboard block | strong because one page can show toolkit breadth without scrapbook chaos | requires very strict grid/legend policy | structured dashboard hero |
| PAGA | small geometry-first method pages | graph/topology panel | compact pages with a clear object can still be memorable | many pages are more functional than luxurious | compact geometry page |
| Reconstructing growth and dynamic trajectories... | alternating geometry/trend pages | trajectory manifold plus trend companion | good example of letting dynamics and summary panels alternate | if every page stayed on embeddings, it would be weaker | dynamics cascade page |
| Robust integration and annotation using interpretable gene programs | integrated pipeline/result page | program matrix + spatial/UMAP support | combines matrix, embedding, and tissue panels cleanly | page depends on a strong central program object; without it the page would scatter | matrix hero + geometry support |
| Improved single-cell ATAC-seq corticogenesis | multiscale developmental page | lineage / chromatin-state result block | ties developmental progression to companion summaries | some pages remain conventional; useful but not especially stylish | top hero + lower support band |
| Mapping lineage-traced cells across time points with moslin | text-heavy page with occasional compact result block | lineage mapping support panel | useful as a reminder that strong science can still have weak page design | not a page to imitate closely for manuscript aesthetics | negative control / do not copy layout literally |
| Constructing the spatiotemporal atlas... | richly packed atlas pages | wide atlas/trajectory page | strong because atlas panels are paired with local summaries, not left alone | can become dense if labels and mini-panels multiply too far | wide atlas page + support row |

## What Strong Papers Actually Do

## 1. Most result pages are built as 2-3 zones, not as a flat grid

Across the audited papers, the raw panel count varies widely, but the page logic is usually:

- one hero or compare zone
- one support/mechanism zone
- optionally one validation or benchmark zone

That is more stable than counting letters.

## 2. Hero panels are not always embeddings

The most effective hero object depends on the page:

- manifold / spatial atlas
- shared-basis compare strip
- benchmark matrix
- perturbation compare page
- transition/coupling matrix with map companions

So a hero panel is not "the prettiest UMAP".
It is the object that most directly carries the page's main visual claim.

## 3. The richest pages change grammar quickly

The most impressive result pages usually do this:

- start with geometry / compare / spatial object
- switch immediately into distributions, heatmaps, kinetics, perturbation, or validation

They do **not** spend a whole page repeating the same basis with only recoloring.

## 4. Many good pages are asymmetric

A strong result page often allocates:

- about `40-55%` of the page to one dominant object
- about `20-30%` to an immediate companion block
- about `20-35%` to lower support or closure

This is why strict equal-sized grids often feel weaker than the published page.

## 5. Matrix-led pages are legitimate heroes

Benchmark/toolkit papers often make the matrix or score table the main object.
That works when:

- repeated scales are strict
- the grid is calm
- companion plots are nearby
- the matrix itself answers the main question

This is why benchmark pages should not be forced into manifold-first layouts.

## 6. Published pages are often portrait and still dense

The strongest journal result pages are usually still portrait/two-column pages.
They feel rich not because they are wide, but because:

- the hero zone is clear
- the page is asymmetrically zoned
- legends and colorbars are disciplined
- support blocks are packed tightly without becoming scrapbook-like

Landscape can still be useful, but most audited papers do not rely on landscape to look strong.

## Repeated Skeletons Worth Reusing

### Skeleton A: Asymmetric hero page

Use when:

- one geometry/spatial/manifold object is clearly the page anchor
- one companion proof must sit nearby
- a lower support band closes the claim

Rough layout:

```text
|      hero      | companion 1 |
|      hero      | companion 2 |
| support 1 | support 2 | support 3 |
```

Seen strongly in:

- CellRank 2
- CellOT
- atlas-style papers

### Skeleton B: Shared-basis compare band + lower proof band

Use when:

- several same-basis panels must be compared directly
- the page needs one clear grammar shift afterward

Rough layout:

```text
| A | B | C | D |
|     wide quantitative / mechanism block     |
```

Seen strongly in:

- scVelo dynamical
- trajectory comparison pages
- OT/trajectory papers

### Skeleton C: Matrix-led dashboard page

Use when:

- benchmark score matrices, program matrices, or toolkit dashboards are the main object
- repeated scales matter more than one large manifold

Rough layout:

```text
| matrix hero | summary block |
| supporting matrix/curves/bars |
```

Seen strongly in:

- atlas integration benchmark
- Pertpy
- NicheNet-like mechanism pages

### Skeleton D: Spatial wall + one anchor

Use when:

- many same-basis spatial maps are unavoidable
- one statistical or summary panel must keep the page from becoming wallpaper

Rough layout:

```text
| map | map | map |
| map | map | anchor |
```

Seen strongly in:

- Slide-seqV2
- spatial mapping papers

### Skeleton E: Multi-scale interleave page

Use when:

- the method output lives at more than one scale
- page needs embeddings/maps plus curves plus heatmaps plus small summary blocks

Rough layout:

```text
| top mixed hero band |
| mid support band    |
| lower closure band  |
```

Seen strongly in:

- GLUE
- multi-omic velocity
- moscot

This is harder to execute well, but very effective when the method itself is multi-view.

## Which Pages Felt Most Impressive

### Most impressive page families

1. `Learning single-cell perturbation responses using neural optimal transport`
   - because the page couples perturbation embeddings, dose-response summaries, and intervention logic
2. `Generalizing RNA velocity to transient cell states through dynamical modeling`
   - because geometry is immediately tied to dynamics/kinetics
3. `CellRank 2`
   - because same-basis compare is extremely disciplined
4. `Mapping cells through time and space with moscot`
   - because coupling/transport is shown with multiple coordinated grammars
5. `Multi-omics single-cell data integration and regulatory inference with graph-linked embedding`
   - because modality-level support is interleaved instead of dumped into one appendix-like row

### Most useful negative lesson

Some papers in the set are scientifically strong but visually overtextual or weakly zoned.

That matters because it means:

- do not copy a page just because the paper is famous
- imitate the best **page grammars**, not the average published page

## How This Should Change Our Own Figure Planning

## What Earlier Internal Guidance Underestimated

This audit corrected several internal biases:

1. **Panel count was overweighted**
   - strong pages are chosen by skeleton first, not by whether they have 4, 6, 8, or 9 children
2. **Story tightness was overweighted**
   - algorithm papers often benefit from one coherent secondary model-result zone
3. **Cleanliness was overweighted relative to page hierarchy**
   - a page can be clean but still weak if it lacks a real hero object and companion proof
4. **Equal grids were treated too neutrally**
   - many strong published pages are intentionally asymmetric
5. **Published papers were treated too uniformly**
   - some published result pages are worth copying; others are mainly cautionary examples

### For main figures

- do not start by counting panel letters
- first choose one page skeleton
- then choose the hero object
- then choose the companion proof
- then decide whether a secondary model-result zone is justified

### For supplement figures

- a page can be denser and more dashboard-like
- but it still needs one organizing grammar
- if a supplement page mixes too many unrelated mini-panels, it degrades fast

## Mapping Page Skeletons To Our Template Library

| Published page skeleton | Use these templates/wrappers first |
|---|---|
| shared-basis compare band | `template_manifold_overlay.py`, `template_spatial_small_multiples.py`, `template_native_scanpy_embedding.py` |
| geometry hero + quantitative companion | `template_embedding_trajectory_overlay.py`, `template_stylish_box_jitter.py`, `template_bubble_summary.py` |
| perturbation result page | `template_perturbation_embedding_compare.py`, `template_perturbation_dose_response.py`, `template_volcano_meta.py` |
| matrix-led benchmark/mechanism page | `template_transition_heatmap.py`, `template_annotated_block_heatmap.py`, `template_attention_heatmap.py`, `template_gridsearch_heatmap_bar.py` |
| dynamics closure page | `template_temporal_ribbons.py`, `template_embedding_time_overlay.py`, `template_native_cellrank_gene_trends.py` |
| relation/network closure | `template_bipartite_pathway_network.py`, `template_labeled_network_summary.py` |

## Practical Rule

When a new figure feels ordinary, check the page-level failure mode first:

- no clear hero object
- no nearby companion proof
- repeated manifold grammar
- matrix with no context
- too many equal-sized chores panels

That is usually a better diagnosis than "maybe the palette is wrong".
