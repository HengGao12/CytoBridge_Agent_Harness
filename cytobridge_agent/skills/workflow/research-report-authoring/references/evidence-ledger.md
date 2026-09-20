# Evidence Ledger

Build this ledger internally before writing a report or paper. It does not need
to be saved unless it is useful for review.

## Minimum Fields

| Field | Meaning |
| --- | --- |
| claim | The sentence-level claim to make |
| evidence | Artifact, metric, figure, RAG chunk, paper, or log supporting it |
| source_path | Local path, URL, run id, or chunk id |
| confidence | supported, partial, conceptual, missing |
| use_in_output | main text, table, figure caption, limitations, omit |

## Rules

- A claim with `missing` evidence cannot appear as a result.
- A claim with `conceptual` evidence can appear as interpretation only.
- A claim with `partial` evidence must include a caveat.
- Do not merge unrelated evidence into one stronger-sounding claim.
- Keep negative and failed results if they matter for interpretation.

## Common Evidence Types

- Metrics: `metrics.json`, run summaries, W1/TMV values, custom metric tables.
- Figures: saved PNG/SVG/PDF/HTML plots and their generation scripts.
- RAG: `outputs/rag_log/*.json`, chunk ids, retrieved passages.
- Literature: paper title, DOI/arXiv id, local PDF path, page or section.
- Code/config: training config, algorithm proposal, manifest, commit hash.
