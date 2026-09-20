# Paper Reviewer Contract

Use this contract after the manuscript, supplement, ledgers, audits, and compile
outputs exist.

## Required Gate

Before reporting a paper as complete, spawn a read-only reviewer:

- `subagent_type="paper_reviewer"`
- `relevant_paths` must include the paper directory and the key manuscript files
- the reviewer must return `proposed_state_updates.paper_review.decision`
- only `decision=approve` allows completion

If the decision is `revise`, revise the paper and run the reviewer again. If the
decision is `reject`, return to the narrative report or paper plan instead of
patching prose locally.

## Reviewer Scope

The reviewer checks:

- content and claim grounding
- reader-friendly motivation, self-contained setup, and narrative coherence
- source-to-paper obligations independently inferred from proposal,
  implementation map, configs, code, reports, metrics, and figures
- algorithm coverage against proposal, implementation, config, and
  `algorithm_detail_ledger.md`
- proof obligations against source mathematics and `theory_obligation_ledger.md`
- theory depth, notation, assumptions, and proof consistency
- evidence support for central claims
- baseline/result table numeric provenance against `query_campaign_baseline_metrics`
  output or exact baseline `metrics_path` artifacts
- citation sanity as an advisory check unless it affects central claims
- research-paper style and language
- figure inventory, figure placement, and caption quality
- file structure and main/supplement boundary
- LaTeX compile health, with bibliography issues treated as advisory unless they block reading or support a central claim
- audit-file substance

The reviewer is not a coauthor. It must not edit files or generate replacement
sections.

## Hard Blocking Rules

The reviewer must return `revise`, not `approve`, when any of these conditions
hold:

- the manuscript is a method, model, training, or inference paper and
  `algorithm_detail_ledger.md` is missing;
- the manuscript is a method, model, training, or inference paper and
  `source_obligation_matrix.md` is missing;
- the manuscript is a method, model, training, or inference paper and
  `checks/source_obligation_audit.md` is missing or superficial;
- the manuscript is a method, model, training, or inference paper and
  `paper_plan.md` lacks a method detail budget;
- the manuscript is a method, model, training, or inference paper and
  `paper_plan.md` lacks an obligation-to-section map;
- the manuscript is a method, model, training, or inference paper and
  `checks/algorithm_coverage.md` is missing or superficial;
- the manuscript contains formal or semi-formal theory claims and
  `theory_obligation_ledger.md` is missing;
- the manuscript contains theorem, proposition, guarantee, recovery, endpoint,
  mass, stochastic-dynamics, exact-fit, or proof language and
  `checks/proof_completeness_audit.md` is missing or superficial;
- `paper_plan.md` is only a short bullet list and cannot guide
  section-by-section writing;
- `paper_plan.md` lacks a reader-friendly motivation spine or coherence map,
  or the written paper no longer follows that spine;
- the manuscript reads like a workflow report rather than a publishable paper:
  it is organized around approvals, gates, stages, run chronology, metric
  definitions, or internal bookkeeping instead of motivation, insight, method,
  evidence, and implication;
- a new algorithm paper lacks a publication-quality method name, or uses only
  an internal id, snake_case name, or keyword pile in the title/abstract;
- the Abstract/Introduction do not make the problem, consequence, gap, method
  idea, and evidence level understandable to a non-project reader;
- the Abstract/Introduction are mostly defensive caveats, "not X but Y"
  contrasts, or component lists rather than a positive argument for why the
  work matters;
- `sections/method.tex` only states high-level equations or architecture
  without target construction, objective, training procedure, inference
  procedure, and evaluation protocol;
- a new method paper never defines a general problem abstraction and instead
  presents only a dataset-specific recipe;
- `sections/method.tex` introduces model machinery without explaining how it
  follows from the motivating failure mode or scientific question;
- source materials contain an objective, constraint, loss, target, sampling law,
  update rule, inference rule, or evaluation rule, but the paper omits the
  corresponding formula or procedural step;
- a contribution claim lacks a mechanism link explaining which modeling choice
  supports it;
- `sections/theory.tex` claims endpoint, mass, stochastic, or recovery behavior
  without assumptions and derivation, or references a complete supplement proof
  that is not present;
- Method and Theory, or main paper and supplement, give conflicting semantics
  for the same construction;
- the proposal, algorithm ledger, or implementation map contains a mathematical
  form, dynamic objective, endpoint recovery argument, mass recovery argument,
  Fokker--Planck equation, or exact-fit logic, but the paper reduces it to a
  narrative paragraph without formal objects, assumptions, proposition/theorem
  statements, and proof or derivation steps;
- a baseline/result table reports builtin/reference baseline metrics without
  `query_campaign_baseline_metrics` evidence or exact baseline `metrics_path`
  provenance;
- evaluation metrics are reported without clear prediction source, truth source,
  baseline source, filtering/grouping rule, weight convention, and information
  boundary;
- a manuscript baseline number disagrees with the campaign baseline metrics
  query output, a recorded `metrics.json`, or the evidence ledger provenance;
- a SOTA or baseline-comparison claim appears to be copied from a draft table,
  preview note, memory, or transient chat summary rather than measured campaign
  baseline evidence;
- `checks/theory_consistency.md` claims assumptions, propositions, or complete
  proofs exist but does not name their exact labels or file locations, or those
  labels cannot be found in the manuscript files;
- `checks/proof_completeness_audit.md` does not map each central theory ledger
  item to a proof, derivation, or explicit downgrade, or approves a claimed full
  proof that is not present in `supplement.tex`;
- `main.tex` or `sections/*.tex` contains unresolved internal workflow
  vocabulary such as `rollout`, `artifact`, `runtime`, `contract`, `preflight`,
  `hard gate`, `gate`, `campaign`, `trial`, `trial_`, `fallback`, `agent`,
  `workspace`, `implementation map`, `local path`, `sealed trajectory`,
  `tool call`, `approved`, `final locked`, `locked release`,
  `final regression`, `Stage 1`, `Stage 2`, or `Stage 3`, unless the user
  explicitly requested a technical report or the term is quoted from a source.
- equations, symbols, or notation are not self-consistent across Method,
  Theory, Experiments, Results, and supplement;
- metric definitions, validation thresholds, or defensive limitations dominate
  the main text and obscure the paper's insight.

These blockers are independent of compile success and bibliography hygiene.

## Spawn Brief Requirements

The parent brief should ask the reviewer to inspect:

- `editorial_brief.md`
- `narrative_report.md`
- `paper_plan.md`
- `evidence_ledger.md`
- `query_campaign_baseline_metrics` output, or exact baseline `metrics.json`
  paths named in `evidence_ledger.md`, when the paper compares baselines
- `citation_ledger.md`
- `algorithm_detail_ledger.md` for method papers
- `source_obligation_matrix.md` for method papers
- `theory_obligation_ledger.md` for papers with formal or semi-formal theory
  claims
- `macros.tex`
- `sections/*.tex`
- `main.tex`
- `references.bib`
- `supplement.tex`
- `checks/*.md`
- `main.log`, `main.bbl`, and compile artifacts when present

Success criteria should require the reviewer to first infer its own central
source obligations and, when theory claims exist, its own proof obligations.
Compare those against `source_obligation_matrix.md` and
`theory_obligation_ledger.md`, then return an approve/revise/reject verdict and
concrete blocking issues. The reviewer must not approve only because
author-written audits say "covered" or "complete".

## Completion Rule

Do not report the paper as complete unless:

- reviewer status is completed
- `paper_review.decision` is `approve`
- `blocking_issues` is empty
- `editorial_brief.md` exists and the title, abstract, introduction, and
  method name are consistent with its public contribution framing
- citation issues are advisory, or any blocking citation issue is tied to an unsupported central claim rather than BibTeX bookkeeping
- all baseline table values and SOTA claims trace to actual measured campaign
  baseline metrics rather than manuscript-internal tables
- for method papers, `source_obligation_matrix.md` and
  `checks/source_obligation_audit.md` exist and cover all central obligations
  inferred by the reviewer
- for theory-bearing papers, `theory_obligation_ledger.md` and
  `checks/proof_completeness_audit.md` exist and cover all central proof
  obligations inferred by the reviewer
- method papers explain the problem setup, target construction,
  parameterization, objective, training procedure, inference procedure, and
  evaluation object without requiring the reader to inspect local run logs
- new method papers define the general problem and solver before specializing
  to a dataset or application, unless the user explicitly asked for a dataset
  note rather than a method paper
- every contribution claim has a visible mechanism link and every reported
  evaluation has clear prediction/truth/baseline/filtering semantics
- the Introduction contains a clear contribution paragraph or concise
  contribution bullets adapted from the editorial brief without mentioning
  internal workflow language
- the paper is self-contained enough that a reader unfamiliar with CytoBridge,
  WFR-FM, and the project history can follow the motivation and central method
  idea
- the main manuscript has passed a banned-term style audit for internal
  workflow vocabulary

Keep the reviewer result in the final summary so the user can see why the paper
passed or what remains.
