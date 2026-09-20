---
name: research-report-authoring
description: Write evidence-backed HTML research reports for CytoBridge workflows. Use for report.html, RAG-backed explanations, final workflow summaries, artifact-backed reviews, figures, metrics, formulas, and cited research reports.
---

# Research Report Authoring

Use this skill when the deliverable is a readable HTML report, usually
`<output_dir>/report.html`.

## Goal

Write a report that is useful for review: claims are tied to concrete evidence,
figures are embedded when available, formulas render correctly, and missing
evidence is stated as a limitation.

## Read First

- `references/report-html-contract.md`
- `references/evidence-ledger.md`
- current output directory figure inventory
- existing `report.html` or `report.md`, if present
- RAG logs under `<output_dir>/rag_log/`, if relevant
- workflow state and artifact index, if available
- for biological-application algorithm lifecycle reports, downstream manifests,
  saved downstream scripts, tables, figures, feature-space/projection metadata,
  warnings, and supported-claim records produced after final regression
- `~/.cellcompass/skills/workflow/scientific-visualization/SKILL.md` when
  selecting or refactoring report figures that may later become manuscript
  figures
- cited PDFs or retrieved source snippets needed for the user's question

## Operating Rules

- Write the report as `<output_dir>/report.html` unless the user explicitly asks
  for another path.
- For biological-application algorithm lifecycle reports, do not write the
  biological interpretation from final-regression metrics alone. First verify
  downstream analysis exists from the locked model/evaluation trajectory. If it
  is missing, run or request downstream analysis before final reporting, or
  state clearly that biological downstream interpretation is incomplete.
- Treat final-regression success as metric-level validation only. The report
  should explain the biological meaning demonstrated by downstream analysis,
  not merely state that the new algorithm has better benchmark metrics. Do not
  ignore quantitative performance either: empirical reports should present both
  competitive metrics and strong biological interpretation.
- For high-value figures, especially figures likely to be reused in a paper,
  route through scientific-visualization before final rendering. Preserve the
  upstream analysis artifact and script; improve layout, typography, palette,
  and export quality without changing the underlying evidence.
- Gene, regulator, pathway, marker, or biological-mechanism claims require
  model-derived gene-space/projection downstream artifacts, or an explicit
  warning such as `gene_space_unavailable` plus downgraded claims.
- Write direct HTML. Do not rely on runtime postprocessing to repair formulas,
  tables, or citations.
- Include MathJax in the HTML head when the report contains math.
- Use TeX delimiters directly: `\( ... \)` for inline math and `\[ ... \]` for
  display math.
- Do not use `<pre>` or code blocks for equations unless showing literal code.
- Use real HTML `<table>` elements for report tables.
- Ground important claims in saved artifacts, RAG chunk ids, paper titles, PDF
  paths, logs, metrics, or figure paths.
- If evidence is weak, absent, or only conceptual, say so in the report.
- Keep the report readable; do not write a paper-style novelty claim unless the
  evidence supports it.

## Workflow

1. Inspect the available artifacts, RAG logs, downstream manifests/tables,
   figures, metrics, and existing report draft.
2. Build an internal evidence ledger before writing: claim, evidence, source,
   artifact path, confidence.
3. If a selected figure is central to the report or likely to become a
   manuscript figure, read scientific-visualization and regenerate a
   script-backed PNG/PDF pair when needed.
4. Draft the report around evidence, not around generic narrative.
5. Add formulas only where they clarify the scientific point, and write them in
   MathJax-compatible TeX.
6. Add an artifact index and limitations section.
7. Save the report and commit workflow state with `report_path` and a short
   `final_summary` when the workflow tool is available.

## Completion Checklist

- `report.html` exists.
- Formulas are directly renderable by MathJax.
- Tables are real HTML tables.
- Figures use valid relative or absolute paths.
- Claims cite concrete evidence or are explicitly marked as interpretation.
- Limitations and missing analyses are present.
