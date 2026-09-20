# Paper LaTeX Contract

Use this contract for `paper/main.tex`, `paper/macros.tex`,
`paper/sections/*.tex`, `paper/references.bib`, and `paper/supplement.tex`.

## Files

- `paper/narrative_report.md`: required source-to-story distillation.
- `paper/paper_plan.md`: required plan used to write the manuscript.
- `paper/evidence_ledger.md`: required claim-evidence ledger.
- `paper/citation_ledger.md`: required provenance ledger for every cited
  source.
- `paper/algorithm_detail_ledger.md`: required for method, model, training, or
  inference papers.
- `paper/source_obligation_matrix.md`: required for method, model, training, or
  inference papers; maps source-derived central obligations to required paper
  treatment.
- `paper/theory_obligation_ledger.md`: required when the manuscript contains
  formal or semi-formal theory claims, endpoint or mass claims, recovery claims,
  theorem/proposition language, or proof claims.
- `paper/macros.tex`: required reusable commands, theorem environments, and
  notation macros used by main paper and supplement.
- `paper/sections/*.tex`: required main-paper section files for full
  manuscripts.
- `paper/main.tex`: master file for the main paper only.
- `paper/references.bib`: BibTeX entries for every cited source.
- `paper/supplement.tex`: required when there are proofs, detailed limitations,
  configs, additional figures, extended tables, run ids, implementation maps,
  or reproducibility details.
- `paper/checks/*.md`: required audit outputs before completion, including
  `checks/source_obligation_audit.md` for method, model, training, or inference
  papers and `checks/proof_completeness_audit.md` when proof claims appear.
- `paper/figures/`: optional copies or relative links to selected figures.

Do not substitute `scripts/main.tex`, `paper_main.tex`, or any other fallback
name for `paper/main.tex` unless the user explicitly requests it. If the write
tool reports success, verify with an independent directory listing or file read
before mentioning the path in the final answer.

## Compile Contract

- Prefer `latexmk -pdf main.tex` from the `paper/` directory when available.
  If TeX Live tools such as `latexmk`, `pdflatex`, or `bibtex` are missing,
  try `tectonic main.tex` before declaring that no LaTeX toolchain is
  available. Use the same fallback for `supplement.tex` when it exists.
- A draft compile is acceptable when `main.pdf` exists and the main text is
  readable. Check whether `main.bbl` exists, but treat bibliography-only
  failures as advisory unless the user requests a production-ready compiled
  bibliography.
- Inspect `main.log` for unresolved citations/references, especially:
  `undefined`, `Citation`, `Reference`, and `No file main.bbl`.
- Fix compile errors, missing figures, and unresolved cross-references that
  affect reading the paper. Record unresolved citation or missing bibliography
  issues in `checks/bib_hygiene.md` and continue to content review.
- Do not report a paper as complete until the `paper_reviewer` subagent
  approves. Reviewer approval prioritizes content, theory, evidence, figures,
  style, and structure; bibliography bookkeeping is advisory unless it affects a
  central claim.
- A self-authored fallback check is not a `paper_reviewer` approval. If the
  reviewer failed because of runtime/provider/checkpoint infrastructure before a
  structured verdict, report a compiled draft pending independent reviewer
  infrastructure and retry later; do not mark the paper lifecycle complete.
- If `supplement.tex` exists and the TeX toolchain is available, compile it
  separately with `latexmk -pdf supplement.tex` or, when only Tectonic is
  available, `tectonic supplement.tex`.
- `main.pdf` should not contain appendix-lettered sections by default. If
  `main.tex` contains `\appendix`, `\input{appendix}`, or
  `\input{supplement}`, this must be because the user explicitly requested a
  combined PDF.

## LaTeX Defaults

- Use a generic article-compatible preamble unless the user provides a venue
  template.
- Include common packages: `amsmath`, `amssymb`, `graphicx`, `booktabs`,
  `hyperref`, `natbib`, `xcolor`.
- Prefer `\paragraph{}` only for compact substructure.
- Use `\citep{key}` and `\citet{key}` consistently with natbib.
- Put reusable commands and theorem environments in `macros.tex`, then load it
  from both `main.tex` and `supplement.tex` when supplement exists.

## Required Structure

Default to a flexible ML research-paper structure. Do not force a report-style
sequence.

```tex
\begin{abstract}
...
\end{abstract}

\input{sections/introduction}
\input{sections/related_work}
\input{sections/method}
\input{sections/theory} % or analysis, when needed
\input{sections/experiments}
\input{sections/results}
\input{sections/discussion} % or conclusion
\bibliographystyle{plainnat}
\bibliography{references}
```

Do not require standalone `Limitations` or `Reproducibility Statement` sections
in the main body unless the target venue requires them. Detailed limitations,
configuration snapshots, run ids, evidence tables, missing experiments, and
reproducibility notes should usually live in `supplement.tex` or
`evidence_ledger.md`.

By default, `main.tex` must not include:

```tex
\appendix
\input{appendix}
\input{supplement}
\include{appendix}
\include{supplement}
```

## Claim Discipline

- Do not call the work novel unless the related work evidence supports the
  novelty boundary.
- Do not claim improved performance unless there is a real baseline comparison.
- Do not hide missing baselines, failed runs, or small-scale evaluations, but
  avoid repeating these caveats after every claim.
- Use positive but bounded framing: state what was achieved in the main text,
  then collect limitations in Discussion or supplement.
- If evidence is only a proof-of-concept, frame the paper as method/theory plus
  proof-of-concept evidence, not as a benchmark-winning empirical paper.
- Avoid report-defensive phrasing such as "this is not state of the art" in the
  abstract or contribution list. Put comparison boundaries in Results or
  Discussion.

## Main-Text Hygiene

- Do not put long run ids, local absolute paths, file-system details, campaign
  bookkeeping, trial ids, or tool names in the main body.
- Do not use agent/runtime engineering words in the manuscript body unless the
  user explicitly asked for a technical report or the term is quoted from a
  source: `rollout`, `artifact`, `runtime`, `contract`, `preflight`,
  `hard gate`, `gate`, `campaign`, `trial`, `trial id`, `trial_`, `fallback`,
  `agent`, `workspace`, `implementation map`, `local path`,
  `sealed trajectory`, `tool call`.
- Replace implementation-context words with paper-context words where possible:
  `generated trajectory`, `trajectory simulation`, `trained model`,
  `evaluation protocol`, `evaluation criterion`, `saved figure`, `training
  configuration`, `experiment`, `run`, or `implementation`.
- Before reviewer submission, run a literal banned-term search over `main.tex`
  and `sections/*.tex`. Any unresolved hit is a failed style audit and must be
  rewritten.
- Avoid overusing quotation marks, colons, and dash-heavy sentence structures.
- Do not write repeated "we do X, however..." sentences that undercut the
  contribution. Move caveats to the right section.

## Math And Theory

- Define every symbol before first use: measures, random variables, time
  variables, sampling laws, losses, weights, and model fields.
- If the proposal, algorithm ledger, implementation map, or source notes contain
  a `Problem Mathematical Form`, dynamic objective, Fokker--Planck or
  continuity equation, endpoint recovery argument, mass recovery argument, or
  exact-fit logic, translate it into formal paper mathematics. Do not compress
  it into a narrative paragraph.
- A method theory section should normally contain:
  - mathematical objects and notation;
  - explicit assumptions;
  - proposition/theorem statements for claimed endpoint, mass, stochastic, or
    recovery properties;
  - proofs or derivations with algebraic steps;
  - a final paragraph naming what is not proved.
- For theory-heavy papers, include a notation paragraph or table before the
  first dense mathematical derivation.
- Theorem, lemma, proposition, corollary, and assumption statements must be
  complete and checkable.
- Default to complete proofs for formal claims. A main-body proof sketch is
  acceptable only as a summary of a complete proof that already appears in
  `supplement.tex`.
- Formal claims must be traceable to `theory_obligation_ledger.md`. If a
  theorem, proposition, guarantee, endpoint/mass claim, or proof phrase has no
  ledger item, do not include it as a formal claim.
- Do not write phrases such as `a full proof is given in the supplement` unless
  the supplement contains a complete proof with assumptions, definitions, proof
  steps, and conclusion.
- If the proof is incomplete, do not state the result as a theorem or
  proposition. Use `derivation`, `intuition`, `design rationale`, `proof
  obligation`, or `conjecture` as appropriate, and make the missing step
  explicit.
- Audit supplement restatements against main-body statements. Hypotheses,
  domains, variable names, case splits, and quantifiers must match unless an
  explicit notation bridge is written.

## Method And Algorithm Detail

For method, model, training, or inference papers, `sections/method.tex` must be
written from `algorithm_detail_ledger.md`, `source_obligation_matrix.md`, and
`paper_plan.md`, not from a short report summary or old draft prose.

The method section must cover:

- problem setup, including data, empirical measures or samples, state space,
  time index, context variables, and prediction target;
- assumptions that control what the method can claim;
- supervision or target construction, including couplings, labels, bridge
  points, velocity targets, score targets, growth targets, or other generated
  training signals when present;
- model parameterization, including all learned fields, decompositions,
  projections, graph or regulatory operators, residual components, and what
  each component represents;
- objective, including every essential loss term and the role of regularizers or
  diagnostics;
- training procedure, including sampling or batch construction at conceptual
  pseudocode level when source material provides it;
- inference procedure, including integration or simulation rule, weight update,
  generated outputs, and what information is unavailable at inference time;
- evaluation protocol, including the generated object being scored and any
  no-repair or no-leakage rule.

For every central item in `source_obligation_matrix.md`, the method, theory,
experiments, results, or supplement must contain the required treatment. A
central source objective, constraint, loss, target, sampling law, inference
rule, or evaluation rule cannot be replaced by a vague verbal summary.

If the source material contains pseudocode or step labels, translate them into a
compact algorithm box, numbered procedure, or paragraph sequence. Do not copy
run-log language or local file names into the main paper.

For transport, flow-matching, stochastic-dynamics, growth, or regulatory
methods, define the endpoint measures, coupling, particle weights, dynamics,
target fields, learned fields, and mass semantics before using them in formulas.
A single equation for the final neural field is not enough.

`sections/theory.tex` must derive or clearly state the status of any endpoint,
mass, stochastic, or recovery property that the method relies on. If the
derivation is not complete, the paper must call it a derivation target or proof
obligation instead of presenting it as established theory.

For method papers with a proposal-stage exact-fit argument, `sections/theory.tex`
must not be shorter than the proposal's mathematical core in substance. It must
carry over the actual mathematical objects, assumptions, and proof chain, while
removing internal workflow language and avoiding unsupported finite-sample
claims.

## Figures And Tables

- Use `\usepackage{graphicx}` when the paper contains figures. Prefer copying
  or symlinking selected figures into `paper/figures/` and referencing them with
  relative paths.
- Every figure must point to a real file path during drafting or be explicitly
  marked as planned/missing before submission.
- If candidate figures exist, the main paper should include at least one
  paper-relevant method, trajectory, qualitative result, or quantitative summary
  figure unless `paper_plan.md` gives a concrete rejection reason for every
  candidate.
- Do not put all figures in the supplement by default. Main-paper figures should
  support the central paper claims; dense diagnostics, failed runs, config
  screenshots, and secondary ablations belong in `supplement.tex`.
- Every figure caption must state the paper claim or reader takeaway supported
  by the figure, not only the file name or run label.
- Every table value must trace to an artifact or evidence ledger entry.
- In algorithm comparison tables, every builtin/reference baseline metric must
  trace to `query_campaign_baseline_metrics` output or an exact baseline
  `metrics.json` path. Do not use draft manuscript tables as metric evidence.
- Use `booktabs` tables.
- Put extended details in `supplement.tex` when main text becomes crowded.
- Results should be organized by claim and reader takeaway, not by chronological
  run history.

## References

- Include BibTeX entries for citation keys used in `main.tex` where provenance
  is available.
- Every citation key used for a central claim should appear in
  `paper/citation_ledger.md`.
- Every BibTeX entry should have provenance in `paper/citation_ledger.md`.
- Do not cite keys marked `remove`; avoid using `needs_verification` keys for
  central claims.
- Do not invent BibTeX metadata from memory. If metadata is missing, record the
  gap rather than fabricating it.
- Prefer original papers for methods and theory.
- Use surveys only for orientation or broader framing.
- Keep `references.bib` limited to cited entries.
