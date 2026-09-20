# Evidence Ledger

Build `paper/evidence_ledger.md` before drafting a paper. Use it to convert
workflow evidence into paper claims. Do not copy run logs directly into the
main manuscript.

## Minimum Fields

| Field | Meaning |
| --- | --- |
| paper_claim | The claim as it would appear in the manuscript |
| evidence | Artifact, metric, figure, source paper, or run supporting it |
| source_path | Local path, URL, run id, RAG chunk id, or citation key from `citation_ledger.md` |
| confidence | supported, partial, conceptual, missing |
| manuscript_location | abstract, intro, method, results, discussion, supplement, omit |
| figure_decision | optional for figure evidence: main, supplement, reject, or not_applicable |

## Rules

- Only supported claims may appear in the abstract as achieved results.
- Partial evidence may appear in Results or Discussion with measured language.
- Conceptual evidence belongs in Motivation, Method, or Future Work.
- Missing evidence belongs in Discussion, supplement, or a "Needed before
  submission" note.
- If a result is from a small proof-of-concept study, label it in the ledger.
  The main text may simply call it an evaluation, example, or proof-of-concept
  unless the exact pilot status matters.
- If a comparison lacks baselines, do not claim state-of-the-art performance.
- For algorithm lifecycle baseline comparisons, record
  `query_campaign_baseline_metrics` output or the exact baseline `metrics.json`
  path for every builtin/reference baseline value. Draft tables, prior prose,
  chat summaries, and memory are not valid numeric evidence.
- Keep long local paths and full run ids in the ledger or supplement, not in the
  main paper.
- For each candidate figure, create or update a ledger entry with source path,
  supported claim, confidence, manuscript location, and figure decision.
- For each Results paragraph, identify which ledger claim it supports before
  writing the paragraph.
- Evidence table details, run ids, local paths, failed runs, and configuration
  snapshots belong in this ledger or `supplement.tex`, not in `main.tex`.
- If a claim cannot be supported by this ledger, remove it from the main paper
  or mark it as future work.
- Literature evidence must reference a key in `citation_ledger.md`; do not use
  free-text references that have no provenance record.
- Claims supported only by `needs_verification` citation ledger entries cannot
  appear as achieved main-paper claims.

## Evidence To Paper Mapping

- Supported method/theory claims can appear in Abstract, Introduction, Method,
  and Theory.
- Supported quantitative claims can appear in Abstract and Results.
- Partial evidence can support proof-of-concept language but not superiority
  claims.
- Missing experiments should not interrupt the main claim narrative. Put them
  in Discussion, supplement, or a submission-readiness note.
- Failed or weaker runs should be included only if they explain an experimental
  choice or belong in a supplement ablation table.
