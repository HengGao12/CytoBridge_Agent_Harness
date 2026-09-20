# Citation Provenance Contract

Write `paper/citation_ledger.md` before writing Related Work, Introduction,
Method, Theory, or `references.bib`.

## Purpose

The citation ledger prevents hallucinated references. Do not create citations,
BibTeX entries, author lists, venues, DOIs, arXiv ids, or years from memory.
Every cited source must have provenance.

## Required Fields

| Field | Meaning |
| --- | --- |
| key | BibTeX key used in `main.tex` |
| claim_supported | The paper claim or background point this source supports |
| source_type | local_pdf, extracted_text, rag_log, trusted_bib, doi, arxiv, proceedings, publisher_page, user_provided, other |
| source_path_or_url | Local path, URL, DOI, arXiv URL/id, RAG log path, or trusted `.bib` path |
| verified_metadata | Title, authors, year, venue, DOI/arXiv id as verified from provenance |
| used_in_sections | Main-paper sections that cite this key |
| status | verified, user_provided, needs_verification, remove |

## Author Rules

- Build `citation_ledger.md` before drafting prose that cites literature.
- Use only keys with status `verified` or `user_provided` in `main.tex`.
- Do not use `needs_verification` citations for Abstract, Introduction,
  Related Work positioning, novelty boundaries, Method claims, or Theory claims.
- Remove `remove` entries from both `main.tex` and `references.bib`.
- Generate `references.bib` only from cited ledger entries.
- Do not add uncited references to pad the bibliography.
- Do not cite a source unless the source actually supports the surrounding
  sentence.
- When a source is user-provided but not independently checked, mark it
  `user_provided` and avoid using it as the only support for a central novelty
  claim.

## Acceptable Provenance

- Local PDF or extracted paper text that includes enough metadata.
- RAG log or source chunk with paper metadata.
- Trusted existing BibTeX entry from a local `.bib` file.
- DOI landing page, arXiv page, official proceedings page, or publisher page.
- User-provided full citation.

## BibTeX Rules

- Every citation key in `main.tex` must appear in both `references.bib` and
  `citation_ledger.md`.
- Every key in `references.bib` must be cited in `main.tex`, unless the user
  explicitly requested a reading-list appendix.
- BibTeX metadata must match the ledger provenance.
- If metadata is incomplete, mark status `needs_verification` and do not cite
  it in the main paper.

## Audit Requirements

`checks/bib_hygiene.md` must include:

- all citation keys used in `main.tex`
- all keys present in `references.bib`
- all keys present in `citation_ledger.md`
- status for every key
- mismatches, missing provenance, uncited entries, and unresolved compile
  warnings

If any mismatch remains, the paper is not complete.
