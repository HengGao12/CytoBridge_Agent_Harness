# Paper Planning Contract

Write `paper/paper_plan.md` after `paper/editorial_brief.md` and
`paper/narrative_report.md`, and before writing LaTeX.

## Purpose

The paper plan turns reports, metrics, figures, theory notes, and source papers
into a manuscript story. It prevents the final paper from reading like a run log
or workflow report.

## Required Contents

- One-sentence contribution: the shortest accurate statement of what the paper
  contributes.
- Public title and method name: for new algorithm papers, choose a short,
  readable method name and expanded phrase that communicate the central idea.
  Also record the internal algorithm id separately. The public name must not be
  a snake_case identifier, registry id, or keyword pile.
- Paper type: method, theory, empirical, systems, benchmark, or
  method-plus-proof-of-concept.
- Target style: default to ML conference unless the user specifies a venue.
- Audience: who the paper is written for and what they should understand first.
- Reader-friendly motivation spine: the sequence of ideas that takes a
  non-project reader from the motivating biological or mathematical problem to
  the proposed method. Include the concrete gap, the design principle, and the
  first mathematical object introduced to make the idea precise.
- Paper insight: the specific conceptual, biological, or mathematical insight
  the reader should remember. If the only insight is "our metric is better",
  the plan is not paper-ready.
- Claims-evidence matrix: 3 to 5 main claims, evidence source, confidence,
  planned section, and whether the claim is main-paper, supplement-only, or
  omitted.
- Citation plan: citation keys to use, their source in `citation_ledger.md`,
  and which claims each source supports.
- Section plan: section names, purpose of each section, and key claims.
- Coherence map: for each main section, state which part of the motivation
  spine it advances. A section with no role in the spine should be moved to the
  supplement or removed.
- Section-file plan: exact `sections/*.tex` files to create and the section
  each file owns.
- Method detail budget for method papers: map each central entry in
  `algorithm_detail_ledger.md` to the destination section or supplement. This
  budget must include problem setup, assumptions, variables, supervision or
  target construction, model parameterization, objective, training procedure,
  inference procedure, evaluation protocol, and theory obligations.
- Obligation-to-section map for method papers: map every central item in
  `source_obligation_matrix.md` to its planned manuscript or supplement
  location. Include the obligation id, category, required form, destination,
  and whether the planned treatment is formula, objective, constraint,
  algorithm step, proof, derivation, evaluation rule, framing, or caveat.
- Candidate figure inventory: every candidate image or PDF figure found in
  `outputs/figures/`, existing reports, or user-provided figure folders, with
  source file, evidence claim, quality status, and decision: `main`,
  `supplement`, or `reject`.
- Figure and table plan: each selected figure/table, source file, intended
  takeaway, target section, and whether it appears in the main paper or
  supplement.
- Supplement boundary: what must stay out of the main body, including long run
  ids, local paths, configs, evidence tables, failed runs, detailed
  reproducibility notes, implementation maps, and full proofs.
- Macro and notation plan: shared notation, reusable commands, and whether a
  notation paragraph or table is required.
- Proof plan: each formal theorem/proposition/lemma, exact claim, required
  assumptions, proof location, and whether the proof is complete, partial, or
  missing.
- Theory/proof obligation map when `theory_obligation_ledger.md` exists: map
  every central theory item to a main section or supplement location. Include
  the theory id, source claim, required result type, assumptions, proof status,
  planned statement label, and whether the treatment is proof, derivation,
  design rationale, conjecture, or proof obligation.
- Audit plan: expected reverse-outline, style, theory, and bibliography checks.
- Missing-before-submission notes: experiments, baselines, citations, or theory
  checks needed before a real submission.

## Planning Rules

- If an existing `report.html`, `report.md`, or experiment report exists,
  extract paper claims from it. Do not reuse its section order by default.
- Use `narrative_report.md` as the source of the paper story. If it is missing,
  write it before planning the paper.
- Use `editorial_brief.md` as the source of the public method name,
  one-sentence contribution, central insight, closest comparisons, and
  figure-first plan. If it is missing, write it before planning the paper.
- The plan may narrow claims based on the brief, but it must not convert a
  completed algorithm lifecycle into "no paper". The expected output remains a
  paper draft with honest scope.
- For method, model, training, or inference papers, use
  `algorithm_detail_ledger.md` as the source of algorithm details. If it is
  missing, write it before planning the method, theory, experiments, or results
  sections.
- For method, model, training, or inference papers, use
  `source_obligation_matrix.md` as the source of central paper obligations. If
  it is missing, write it before planning the method, theory, experiments, or
  results sections.
- When source material contains mathematical objectives, dynamic equations,
  endpoint or mass claims, recovery arguments, exact-fit claims, or formal
  proof language, use `theory_obligation_ledger.md` as the source of proof
  obligations. If it is missing, write it before planning `sections/theory.tex`
  or proof-related supplement text.
- Use `citation_ledger.md` as the only source of citation keys. If it is
  missing, write it before planning cited Related Work or novelty framing.
- Before saying no figures are used, inspect available candidate files under
  `outputs/figures/` and any figure paths mentioned by reports, ledgers, or the
  user. If candidates exist, explain for each why it is main, supplement, or
  rejected.
- Do not reject a real figure only because it came from a pilot run. If it
  supports a bounded proof-of-concept claim, use it as a proof-of-concept figure
  or place it in supplement.
- By default, plan at least one main-paper figure when a real method,
  trajectory, qualitative result, or quantitative summary figure exists. A
  figure-free main paper requires an explicit reason in `paper_plan.md`.
- Results should be organized by claim and reader takeaway, not by chronological
  experiment history.
- If evidence is mostly theoretical plus one small evaluation, plan the paper as
  method/theory plus proof-of-concept evidence.
- Do not plan formal theorem/proposition claims unless the plan includes a
  complete proof path. If only the idea is available, plan it as a derivation,
  intuition, or proof obligation instead.
- Do not plan to write `a full proof is given in the supplement` unless
  `theory_obligation_ledger.md` identifies the exact statement, assumptions,
  and supplement proof location.
- Do not plan a method section that only states the final model equations. It
  must explain how the training targets are made, what losses are optimized,
  how inference is run, and what generated object is evaluated.
- Do not plan a method or theory section that omits central obligations from
  `source_obligation_matrix.md`. A source objective, constraint, loss, sampling
  law, inference rule, proof obligation, or evaluation rule must be assigned to
  a concrete main-paper or supplement location.
- If the source material contains algorithm semantics, pseudocode, endpoint
  recovery arguments, mass-change derivations, or no-leakage evaluation rules,
  the plan must assign them to `sections/method.tex`, `sections/theory.tex`,
  `sections/experiments.tex`, or `supplement.tex`.
- If evidence lacks baselines, do not plan a performance-superiority narrative.
- The Introduction should make the what, why, and why it matters clear before
  detailed methods.
- The title, abstract, and first two Introduction paragraphs must make a
  positive argument. Do not plan a defensive opening built around caveats,
  internal lifecycle events, or repeated "not X but Y" contrasts.
- The first half of the paper should not require prior knowledge of CytoBridge,
  internal algorithms, or previous run history. Define the scientific object,
  data setting, and prediction target before naming project-specific machinery.
- Method and theory should read as a derivation from the problem: define the
  failure mode, introduce the minimal mathematical object needed to represent
  it, then introduce losses, networks, or algorithms.
- Related Work should synthesize categories and position the method. Do not
  write a paper-by-paper list.
- The plan should decide which caveats belong in Discussion or supplement so
  the main paper does not sound defensive.
- Do not plan to include `\input{appendix}` or `\input{supplement}` in
  `main.tex` unless the user explicitly requested an arXiv full paper or a
  combined technical report PDF.
- Do not plan claims that depend on citations with status `needs_verification`
  or `remove`.
- Plan a visible contribution paragraph or concise contribution bullets for the
  Introduction based on `editorial_brief.md`. The main text should not mention
  the brief itself.
- For full manuscripts, default to section files:
  `sections/introduction.tex`, `sections/related_work.tex`,
  `sections/method.tex`, optional `sections/theory.tex`,
  `sections/experiments.tex`, `sections/results.tex`, and
  `sections/discussion.tex`.

## Incomplete Plan Criteria

A paper plan is incomplete and must be rewritten before LaTeX drafting when any
of these apply:

- it lacks a claims-evidence matrix;
- it lacks a public method name for a new algorithm paper, or the proposed name
  is only an internal id or keyword pile;
- it lacks `editorial_brief.md` or contradicts the public method name,
  contribution, insight, or closest-comparison framing in that brief;
- it lacks a reader-friendly motivation spine or coherence map;
- it lacks a clear paper insight beyond metric improvement;
- it lacks a method detail budget for a method paper;
- it lacks an obligation-to-section map for a method paper;
- it does not mention `algorithm_detail_ledger.md` even though the paper
  describes an algorithm;
- it does not mention `source_obligation_matrix.md` even though the paper
  describes a method, model, training procedure, or inference workflow;
- it has no figure inventory despite available candidate figure files;
- it has no proof or derivation plan despite theory or endpoint-recovery
  claims;
- it lacks a theory/proof obligation map despite a non-empty
  `theory_obligation_ledger.md` or formal proof language in the manuscript;
- it is only a short bullet list that cannot guide section-by-section writing.
