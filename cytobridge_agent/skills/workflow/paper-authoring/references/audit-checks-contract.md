# Audit Checks Contract

Write audit files under `paper/checks/` before reporting completion.

## Required Files

- `checks/reverse_outline.md`
- `checks/style_audit.md`
- `checks/source_obligation_audit.md` for method, model, training, or
  inference papers
- `checks/theory_consistency.md`
- `checks/proof_completeness_audit.md` when `theory_obligation_ledger.md`
  exists or when the paper contains theorem, proposition, guarantee, recovery,
  endpoint, mass, stochastic-dynamics, or proof language
- `checks/algorithm_coverage.md` for method, model, training, or inference
  papers
- `checks/bib_hygiene.md`

## Reverse Outline

For `checks/reverse_outline.md`, list every main-paper paragraph by section and
write its first sentence or intended role. The outline should show a coherent
paper argument without reading the whole manuscript.

Required verdicts:

- Does the Introduction lead from problem to contribution?
- Are Results organized by claim rather than run order?
- If candidate figures existed, does the main paper include at least one central
  figure or document explicit rejection reasons in `paper_plan.md`?
- Does Discussion collect boundaries without making the whole paper defensive?
- Are supplement-only details absent from the main body?

## Style Audit

For `checks/style_audit.md`, inspect `main.tex` and section files for
report-style or agent-style language.

Required checks:

- No long run ids or local absolute paths in the main paper.
- No main-body use of banned internal vocabulary: `rollout`, `artifact`,
  `runtime`, `contract`, `preflight`, `hard gate`, `gate`, `campaign`,
  `trial`, `trial id`, `trial_`, `fallback`, `agent`, `workspace`,
  `implementation map`, `local path`, `sealed trajectory`, or `tool call`,
  unless the user explicitly requested a technical report or the term is quoted
  from a source.
- Rewrite internal vocabulary into manuscript language before reviewer
  submission: generated trajectory, trajectory simulation, trained model,
  saved result, implementation, inference procedure, evaluation protocol,
  evaluation criterion, validation threshold, experiment, or run.
- Avoid repeated defensive phrasing such as "not state of the art" in the
  abstract, contribution list, or every Results paragraph.
- Avoid excessive quotation marks, colon-heavy prose, and dash-heavy sentences.
- The style audit must include the literal search command or term list used,
  the number of hits by file, the replacement made for each hit, and a final
  `verdict: pass` or `verdict: fail`.
- Any unresolved banned-term hit in `main.tex` or `sections/*.tex` is a style
  audit failure and blocks reviewer approval.

## Theory Consistency

For `checks/theory_consistency.md`, audit mathematical content.

Required checks:

- Every displayed-equation symbol is defined before first use.
- Shared notation appears in `macros.tex` when reused.
- Theorem, proposition, lemma, corollary, and assumption statements are complete.
- Each formal claim has a proof status: complete, partial, missing, or downgraded.
- If the source proposal or algorithm ledger contains a mathematical form,
  dynamic objective, endpoint recovery argument, mass recovery argument,
  Fokker--Planck equation, or exact-fit logic, the audit must map each source
  item to a location in `sections/theory.tex` or `supplement.tex`.
- The audit must not claim that assumptions, propositions, or proofs exist
  unless it lists their exact labels or first lines and file locations.
- Any proof sketch in the main body matches a complete proof in `supplement.tex`.
- Any phrase claiming a full proof appears in the supplement is verified against
  an actual complete proof.
- Hypotheses, domains, variables, and quantifiers match between main paper and
  supplement.
- If a proof is partial or missing, the main paper does not present the claim as
  a proved theorem/proposition.

If the paper is not theory-heavy, write a short file saying no formal theory
audit was required and list the equations that were checked.

Do not write a one-paragraph theory pass. Use a table with columns:
`source_theory_item`, `paper_location`, `proof_status`, `blocking_issue`.
If a central source theory item is missing from the paper, the verdict is
`fail` and the reviewer must return `revise`.

## Proof Completeness Audit

For `checks/proof_completeness_audit.md`, compare
`theory_obligation_ledger.md`, `sections/*.tex`, `main.tex`, and
`supplement.tex`.

This audit is stricter than the theory consistency audit. It asks whether a
claim is actually proved, derived, or downgraded, not merely whether notation is
consistent.

Required checks:

- `theory_obligation_ledger.md` exists whenever the manuscript contains formal
  or semi-formal theory claims.
- Every central ledger item has a main-paper location and, when a full proof is
  claimed, a supplement proof location.
- Each formal theorem/proposition/lemma/corollary has assumptions, statement,
  proof steps, and conclusion.
- Every phrase such as `we prove`, `guarantee`, `exactly recovers`,
  `full proof`, `satisfies the dynamics`, or `recovers the endpoint` is mapped
  to a ledger item.
- A main-body proof sketch is accepted only when the supplement contains a full
  proof for the same labeled statement.
- Claims with partial or missing proofs are explicitly downgraded in the paper
  to derivation, design rationale, proof obligation, or conjecture.
- Source objectives, dynamic equations, endpoint/mass arguments, and exact-fit
  claims are not replaced by intuition-only prose.

Use a table with columns: `theory_id`, `claim_or_object`,
`required_result_type`, `main_location`, `supplement_location`, `proof_status`,
and `blocking_issue`.

If a central proof claim is missing, incomplete, or contradicted by the paper's
wording, the verdict is `fail` and the reviewer must return `revise`.

## Source Obligation Audit

For `checks/source_obligation_audit.md`, compare
`source_obligation_matrix.md`, `paper_plan.md`, `sections/*.tex`,
`main.tex`, and `supplement.tex`.

This is the highest-level method-paper audit. It checks whether central
source-derived obligations actually appear in the manuscript. It is stricter
than `checks/algorithm_coverage.md`: algorithm coverage can pass while source
obligation audit fails.

Required checks:

- `source_obligation_matrix.md` exists for method, model, training, or inference
  papers.
- Every central obligation has an actual manuscript or supplement location.
- Central `mathematical_construction` obligations use the required formula,
  objective, constraint, loss, target, sampling law, update rule, or inference
  equation when the source contains one.
- Central `algorithm_reconstruction` obligations describe data-to-target,
  training, inference, and generated output steps at conceptual reproduction
  level.
- Central `claim_mechanism_link` obligations connect each contribution claim to
  a concrete modeling choice rather than only to a result.
- Central `theory_or_proof` obligations list assumptions, statement, proof
  status, and location. Missing proofs must be downgraded in the paper.
- Central `evaluation_semantics` obligations identify prediction source, truth
  source, baseline source, filtering, grouping, weights, leakage boundary, and
  whether the metric path is shared.
- Central `internal_consistency` obligations check that Method, Theory,
  Experiments, Results, and supplement do not give conflicting semantics.
- Central `paper_framing` obligations check that a new method is introduced as a
  general problem and solver before specializing to one dataset or application.

Use a table with columns: `obligation_id`, `category`, `required_form`,
`source_evidence`, `paper_location`, `status`, and `blocking_issue`.

If a central obligation is missing, only verbally summarized when a formula or
algorithm step is required, or contradicted elsewhere in the manuscript, the
verdict is `fail` and the reviewer must return `revise`.

## Algorithm Coverage

For `checks/algorithm_coverage.md`, compare `algorithm_detail_ledger.md`,
`paper_plan.md`, `sections/method.tex`, `sections/theory.tex`,
`sections/experiments.tex`, `sections/results.tex`, and `supplement.tex`.

Required checks:

- `algorithm_detail_ledger.md` exists for method, model, training, or inference
  papers.
- `source_obligation_matrix.md` and `checks/source_obligation_audit.md` exist
  for method, model, training, or inference papers.
- `paper_plan.md` contains a method detail budget.
- `paper_plan.md` contains an obligation-to-section map.
- The method section defines the problem setup, assumptions, notation, and
  prediction target before using formulas.
- The method section explains supervision or target construction rather than
  only presenting final model equations.
- The method section explains model parameterization and the role of each major
  component.
- The objective lists essential loss terms and distinguishes algorithm-defining
  losses from diagnostics or optional regularizers.
- The training procedure is recoverable at conceptual pseudocode level.
- The inference procedure specifies how the reported outputs are generated.
- The experiments or results section defines the evaluated object and prevents
  target leakage or post-hoc repair when relevant.
- The theory section or supplement covers endpoint, mass, stochastic, recovery,
  or consistency claims that appear in the main paper.

Use a table with columns: `required_detail`, `source_in_ledger`,
`manuscript_location`, `status`, and `blocking_issue`. Missing central details
are blocking even when style, citations, and compile are otherwise acceptable.

Do not write a one-paragraph "pass" file. The audit must name the actual method
details checked. If `algorithm_detail_ledger.md` is missing, the audit verdict
is `fail` and the manuscript must not be sent for approval.

## Figure Hygiene

For `checks/style_audit.md` or a dedicated figure subsection, audit figure use.

Required checks:

- `paper_plan.md` contains a candidate figure inventory when figure files exist.
- Every selected figure path exists and is referenced with a relative path from
  `main.tex` or `supplement.tex`.
- At least one central figure appears in the main paper when candidate figures
  support method, trajectory, qualitative result, or quantitative summary
  claims.
- Supplement-only figures are dense diagnostics, secondary ablations, failed
  runs, or implementation details rather than the main evidence for the paper.
- Captions state the paper claim or reader takeaway supported by the figure.

## Bib Hygiene

For `checks/bib_hygiene.md`, audit citations, bibliography, and citation
provenance. This audit is advisory by default; it becomes blocking only when a
citation problem leaves a central paper claim unsupported or introduces an
apparently fabricated critical reference.

Required checks:

- Every `\citep`, `\citet`, and `\cite` key in the main paper has a matching
  `references.bib` entry.
- Every cited key has a matching `citation_ledger.md` entry.
- Every BibTeX entry has a matching `citation_ledger.md` entry.
- Every cited key has status `verified` or `user_provided`.
- No cited key has status `needs_verification` or `remove`.
- `references.bib` contains only cited entries unless the user requested a
  reading list.
- BibTeX metadata matches the cited provenance: title, authors, year, venue,
  DOI, and arXiv id when available.
- Compile logs do not contain unresolved citations, unresolved references, or
  `No file main.bbl`.
- `main.bbl` exists after compilation when a TeX toolchain is available.

## Failure Handling

If a content, theory, figure, style, structure, or compile audit fails, fix the
manuscript and rerun the relevant audit before reporting completion. If a
bibliography audit fails only on metadata or ledger bookkeeping, record it as an
advisory issue and continue. If any failure means evidence is missing for a
central claim, move the unsupported claim out of the main paper and record it in
the ledger or supplement.
