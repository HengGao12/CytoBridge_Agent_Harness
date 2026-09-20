---
name: paper-authoring
description: Write ML conference-style LaTeX paper drafts from CytoBridge research artifacts. Use for paper, manuscript, top conference, NeurIPS, ICLR, ICML, LaTeX, publication-style method or experiment writeups.
---

# Paper Authoring

Use this skill when the user wants a paper or manuscript rather than a workflow
report.

## Goal

Write a credible ML conference-style LaTeX draft from available evidence. The
paper should read like a research submission, not an experiment report. It must
not invent experiments, results, citations, or novelty.

Good paper drafts must be self-contained, motivated, and internally coherent.
A reader who does not know CytoBridge, WFR-FM, or the project history should be
able to understand the biological or mathematical problem, why existing
approaches leave a gap, what the method changes, how the construction follows
from that motivation, and which evidence supports each claim. Do not hide the
main idea behind abstract terms or ledger bookkeeping. Package the contribution
positively, but keep every claim bounded by the available evidence.

The paper is not a compliance packet. Ledgers, audits, and final-regression
records are scaffolding for the writing process, not the manuscript's voice.
The final paper must read like a publishable research article: motivated,
insightful, confident, naturally written, and organized around a small number of
reader-facing contributions. If the available evidence supports only a run
summary, write a report instead of disguising it as a paper.

Default to an auditable writing workflow:

1. extract a paper narrative from evidence
2. plan claims, sections, figures, and supplement boundaries
3. write section files and assemble the LaTeX manuscript
4. run explicit paper-quality checks
5. pass a read-only paper reviewer gate before reporting completion

## Read First

- `references/editorial-brief-contract.md`
- `references/narrative-report-contract.md`
- `references/algorithm-detail-contract.md`
- `references/source-to-paper-obligation-contract.md`
- `references/theory-proof-obligation-contract.md`
- `references/paper-planning-contract.md`
- `references/paper-latex-contract.md`
- `references/citation-provenance-contract.md`
- `references/evidence-ledger.md`
- `references/writing-style-contract.md`
- `references/audit-checks-contract.md`
- `references/paper-reviewer-contract.md`
- existing `report.html` or `report.md`, if present
- experiment metrics, run manifests, configs, figures, and logs
- downstream manifests, downstream summaries, saved downstream scripts, tables,
  figures, and warnings produced after the locked final-regression release
- `~/.cellcompass/skills/workflow/biological-story-building/SKILL.md` before
  writing biological-application Results from downstream evidence
- `~/.cellcompass/skills/workflow/scientific-visualization/SKILL.md` when
  selecting, refactoring, or rendering manuscript figures
- RAG logs and paper/PDF sources relevant to related work
- proposal or algorithm documents, if the paper describes a new method

## Operating Rules

- Output under `<output_dir>/paper/` by default for a single standalone paper.
- For CytoBridge algorithm lifecycle papers, use an algorithm-scoped paper
  directory instead: `<output_dir>/paper_<algorithm_id>/`. Do not write a new
  algorithm paper into a shared `<output_dir>/paper/` directory when that
  directory already contains artifacts for another algorithm. Treat such a
  directory as legacy evidence only; create or use the algorithm-scoped
  directory, and verify that `narrative_report.md`, `evidence_ledger.md`,
  `main.tex`, and any compiled PDFs name the current algorithm before reporting
  completion.
- "Rewrite", "revise", "improve", "polish", "make it a paper", and
  "regenerate paper" requests still use this full workflow unless the user
  explicitly asks for a narrow local edit such as "only fix this paragraph" or
  "only patch this LaTeX error". Do not jump directly into editing
  `main.tex` or `sections/*.tex` when required planning, algorithm-detail,
  evidence, or audit files are missing or stale.
- Any rewrite or revision that changes `main.tex`, `sections/*.tex`,
  `supplement.tex`, `paper_plan.md`, `source_obligation_matrix.md`, or
  `theory_obligation_ledger.md`, or `checks/*.md` invalidates all previous
  paper reviewer approvals. Immediately treat `checks/paper_reviewer_gate.md`
  as stale, rerun `paper_reviewer` after the edit and required checks, and report
  completion only after a fresh reviewer result for the current file set returns
  `approve`.
- Write the required chain:
  `editorial_brief.md -> narrative_report.md -> evidence_ledger.md ->
  citation_ledger.md -> algorithm_detail_ledger.md when method details are relevant ->
  source_obligation_matrix.md when method details are relevant ->
  theory_obligation_ledger.md when theory or proof claims are relevant ->
  paper_plan.md -> sections/*.tex -> macros.tex -> main.tex ->
  references.bib -> supplement.tex -> checks/*.md -> paper_reviewer`.
- For new algorithm papers, create a publication-quality method name before
  drafting the manuscript. The name should be short, pronounceable, and tied to
  the central idea, not an internal snake_case identifier or a keyword pile such
  as `<metadata>_<loss>_<bridge>`. Record the internal algorithm id and the
  public method name in `paper_plan.md`, use the public name in title/abstract,
  and reserve the internal id for supplement or provenance files.
- A completed algorithm lifecycle should still produce a paper draft. Do not
  use the editorial brief to decide not to write the paper. Use it to make the
  best honest paper from the available evidence: narrow the claim, sharpen the
  motivation, and organize the figures and contribution.
- For method papers, do not start LaTeX section writing until
  `algorithm_detail_ledger.md` exists and records the mathematical problem,
  supervision construction, model parameterization, objective, training loop,
  inference procedure, evaluation protocol, and theory obligations, and until
  `source_obligation_matrix.md` exists and records the central obligations the
  paper must satisfy.
- For method papers with mathematical objectives, dynamic equations, endpoint
  or mass claims, recovery claims, exact-fit logic, or theorem/proposition
  language, do not start LaTeX section writing until
  `theory_obligation_ledger.md` exists and records assumptions, formal objects,
  proof status, and main/supplement locations for each central mathematical
  claim.
- Treat an existing paper as incomplete when it lacks
  `algorithm_detail_ledger.md`, `source_obligation_matrix.md`,
  `checks/source_obligation_audit.md`, `checks/algorithm_coverage.md`, or a
  method-detail budget and obligation-to-section map in `paper_plan.md`.
  For theory-bearing papers, also treat it as incomplete when it lacks
  `theory_obligation_ledger.md`, `checks/proof_completeness_audit.md`, or a
  theory/proof obligation map in `paper_plan.md`. Rebuild those files first,
  then rewrite the sections from them.
- Build `citation_ledger.md` before writing cited prose or `references.bib`.
  No provenance means no citation.
- Do not create BibTeX entries, authors, years, venues, DOIs, or arXiv ids from
  model memory. Every citation must come from local PDFs/text, RAG logs, trusted
  BibTeX, DOI/arXiv/proceedings/publisher pages, or user-provided full
  citations.
- For algorithm lifecycle papers with builtin/reference baselines, call
  `query_campaign_baseline_metrics` after baseline refresh or final regression
  and before writing Results tables. Baseline numbers must come from that
  output or the returned `metrics_path`; do not copy baseline values from draft
  paper tables, memory, preview notes, or transient chat summaries.
- For biological-application algorithm lifecycle papers, do not start manuscript
  drafting from final-regression metrics alone. First verify that downstream
  analysis has run from the locked final-regression model/evaluation trajectory
  and produced a manifest, reusable script, tables, figures, warnings, and
  supported-claims list. If those artifacts are missing, stop paper authoring,
  read `~/.cellcompass/skills/workflow/biological-story-building/SKILL.md`,
  then read `~/.cellcompass/skills/workflow/downstream-analysis/SKILL.md` before
  drafting Results or biological interpretation.
- Treat final-regression success as metric-level validation only. A high-quality
  biological empirical paper must be strong on both axes: competitive
  quantitative evidence, including benchmark and claim metrics, and biological
  meaning demonstrated through downstream analysis. Explain what the algorithm
  reveals about the system, mechanism, cell states, fate, perturbation response,
  growth/mass, spatial organization, or gene program, why that finding matters,
  and which model-derived artifacts support it. Do not frame a
  biological-application paper as complete merely because the algorithm improves
  W1, TMV, or the Stage 2 claim metric, and do not substitute a biological story
  for weak empirical evidence.
- A first autonomous biological paper may focus on a single real dataset. Do not
  compensate for shallow interpretation by adding more datasets. Instead, use
  the biological-story-building skill to connect the algorithm's strongest
  metric advantage to one well-supported dataset story, then show the
  trajectory/state/gene/module evidence that makes that story meaningful.
- If the paper makes gene, regulator, pathway, marker, or biological-mechanism
  claims, the evidence ledger must point to model-derived gene-space or
  documented projection artifacts from downstream analysis, such as driver-gene
  tables, pathway/enrichment tables, Jacobian/GRN summaries, or an explicit
  `gene_space_unavailable` warning that downgrades the claim. Do not turn
  latent trajectory metrics, cell-type summaries, or dataset marker lists into
  gene-mechanism claims without this downstream evidence.
- Main-paper biological figures should be refactored through
  `~/.cellcompass/skills/workflow/scientific-visualization/SKILL.md` unless the
  existing figure already satisfies manuscript quality. Figure polishing must
  not change labels, filtering, projection, feature space, model output, or the
  scientific claim. Keep the source downstream artifact fixed, rerun a
  script-generated figure, and preserve PNG plus editable vector PDF outputs.
- Use only citation keys marked `verified` or `user_provided` in `main.tex`.
  Keys marked `needs_verification` or `remove` must not support main-paper
  claims.
- Keep `main.tex` as the master file for the main paper only. By default it
  must not `\input{appendix}`, `\input{supplement}`, or include appendix
  sections. A combined arXiv or technical-report PDF is allowed only when the
  user explicitly asks for one.
- Put proofs, detailed limitations, full run ids, local paths, configuration
  snapshots, evidence tables, failed runs, implementation maps, and
  reproducibility details in `supplement.tex` or support files, not in the main
  paper PDF.
- Use `sections/*.tex` by default for full manuscripts:
  `introduction.tex`, `related_work.tex`, `method.tex`, `theory.tex` when
  needed, `experiments.tex`, `results.tex`, and `discussion.tex`.
- `sections/method.tex` must be detailed enough for a reader to reconstruct the
  algorithm conceptually from the paper. High-level formulas alone are not
  sufficient when proposal or implementation documents contain endpoint
  couplings, training targets, loss terms, inference rules, or evaluation
  constraints.
- `sections/method.tex` and `sections/theory.tex` must be written from
  `source_obligation_matrix.md` for method papers. Do not patch old prose or
  draft TeX when the central source obligations have not been mapped to paper
  locations.
- If the selected paper directory is blocked by workspace policy, stop and
  report the policy error. Do not silently write paper files under `scripts/`
  or another fallback directory. For algorithm-scoped directories, the selected
  directory is `<output_dir>/paper_<algorithm_id>/`, not the legacy shared
  `<output_dir>/paper/`.
- After each file write, verify the exact path with `list_workspace_tree` or
  `read_workspace_file`. Do not tell the user a file exists until verification
  succeeds. When an existing paper tree is present, read a small header from
  `narrative_report.md` or `evidence_ledger.md`; if it describes a different
  algorithm, do not reuse it as the current paper draft.
- When a TeX toolchain is available, run `latexmk -pdf` from the `paper/`
  directory. If `latexmk`, `pdflatex`, or `bibtex` are missing but `tectonic`
  exists, run `tectonic main.tex` before reporting a missing TeX toolchain. A
  successful draft compile requires `main.pdf`. Check `main.bbl` and citation
  warnings when present, but treat bibliography-only failures as advisory unless
  the user explicitly requests a production-ready compiled bibliography.
- If the log contains unresolved cross-references, missing figures, or compile
  errors that affect reading the paper, fix and rerun. If the remaining issue is
  only undefined citations, missing BibTeX metadata, or `No file main.bbl`,
  record it in `checks/bib_hygiene.md` and continue with content review.
- If `supplement.tex` is present and the TeX toolchain is available, compile it
  separately as `supplement.pdf`; use `tectonic supplement.tex` when Tectonic is
  the only available compiler. Do not require supplement compilation to call the
  main paper compiled.
- After compile and required paper checks, spawn a read-only `paper_reviewer` subagent.
  The paper is not complete unless the reviewer returns
  `proposed_state_updates.paper_review.decision=approve`. Reviewer approval
  should prioritize content, theory, evidence, style, figure use, and structure;
  citation/BibTeX hygiene is advisory unless it affects a central claim.
- If the `paper_reviewer` fails because of runtime/provider/checkpoint
  infrastructure before returning a structured verdict, this is not a paper
  approval and not a paper rejection. Do not use a self-authored fallback check as
  an approval substitute. Mark the manuscript as a compiled draft pending
  independent reviewer infrastructure, retry through the runtime when available,
  and do not report the paper lifecycle as complete until a fresh
  `paper_reviewer` approval exists.
- After every reviewer pass, update `checks/paper_reviewer_gate.md` with the
  latest reviewer decision, reviewed file set, required revisions if any, and
  final approval status. Do not leave this file saying that a reviewer check is
  pending after a later reviewer approval has been received.
- Default style is ML conference: NeurIPS/ICLR/ICML-like structure.
- First extract paper claims from any existing report, metrics, figures, logs,
  RAG output, or proposal. Do not copy a report structure into the paper.
- Build a reader-first narrative spine before drafting: motivation, concrete
  failure of prior/builtin approaches, method idea, mathematical construction,
  experiment design, result, limitation. Each section should advance this
  spine; if a section only dumps details, rewrite the plan.
- Write the manuscript as a positive argument, not a defensive explanation of
  workflow constraints. Avoid repeated "not X but Y" constructions, repeated
  caveat-first sentences, and paragraphs that mainly explain what the method is
  not. State the contribution directly, then bound it once with evidence.
- Do not let metric definitions dominate the paper. Define metrics only after
  the reader understands the scientific or mathematical question they measure,
  and connect each metric to the claim it tests. A Results section that is
  mostly metric definitions or gate bookkeeping must be rewritten.
- Use positive but bounded paper writing:
  - lead with what the method contributes
  - explain why the contribution is needed before naming the model machinery
  - introduce mathematical objects from the motivating problem, not as isolated
    notation
  - keep caveats concentrated in Discussion or supplement
  - do not attach a defensive disclaimer to every result sentence
- Separate paper claims from available evidence: supported claims go in the
  main narrative, partial evidence is framed in Results or Discussion, and
  missing evidence goes to supplement notes or future work.
- Use BibTeX keys for citations and include corresponding entries in
  `references.bib`.
- Prefer concise, technical writing over promotional language.
- Main-paper prose must pass the banned-term style audit before reviewer
  submission. Do not write internal workflow vocabulary in `main.tex` or
  `sections/*.tex`: `rollout`, `artifact`, `runtime`, `contract`, `preflight`,
  `hard gate`, `gate`, `campaign`, `trial`, `trial id`, `trial_`, `fallback`,
  `agent`, `workspace`, `implementation map`, `local path`,
  `sealed trajectory`, `tool call`, `approved`, `final locked`,
  `locked release`, `final regression`, `Stage 1`, `Stage 2`, or `Stage 3`.
  Rewrite these as manuscript language:
  generated trajectory, trajectory simulation, trained model, saved result,
  implementation, inference procedure, evaluation protocol, evaluation
  criterion, validation threshold, experiment, or run.
- For downstream CytoBridge analyses, describe continuous paths only when they
  come from the trained model's trajectory/inference interface or a verified
  locked/evaluation trajectory. Do not describe nearest-neighbor interpolation
  of saved per-cell velocities as the model's continuous trajectory. Use
  "local flow-field visualization" for velocity stream figures.

## Default Paper Shape

- Abstract
- Introduction
- Related Work
- Method
- Theory or Analysis, if the work is theory-heavy
- Experiments
- Results
- Discussion or Conclusion
- References

Do not require standalone `Limitations` or `Reproducibility Statement` sections
in the main body by default. Put detailed limitations, configuration snapshots,
run ids, evidence tables, missing experiments, and reproducibility details in
`supplement.tex` or `evidence_ledger.md` unless the venue explicitly requires a
main-body statement.

## Workflow

1. Inventory evidence: metrics, figures, downstream manifests/tables/scripts,
   source papers, RAG chunks, reports, proposal docs, configs, and logs.
2. Write `paper/editorial_brief.md` using the editorial brief contract. This is
   the one-page paper argument: public method name, contribution, motivation,
   closest comparisons, main insight, evidence package, and figure-first plan.
   It is not a decision to skip the paper.
3. Write `paper/narrative_report.md` using the narrative report contract. This
   is a bridge from evidence to paper story, not a workflow report.
4. Write `paper/evidence_ledger.md` using the evidence ledger contract.
5. For algorithm comparison papers, call `query_campaign_baseline_metrics` for
   the relevant campaign and stage(s). Add the tool output path or exact
   returned `metrics_path` entries to `evidence_ledger.md` before drafting any
   baseline table or SOTA claim.
6. For biological-application algorithm papers, add a downstream evidence
   section to `evidence_ledger.md` before drafting Results. It must list the
   downstream manifest, scripts, tables, figures, feature space, projection
   backend, warnings, and which biological claims each artifact supports. If
   gene-level downstream evidence is missing, explicitly mark gene-mechanism
   claims unsupported or downgraded.
7. Write `paper/citation_ledger.md` using the citation provenance contract.
   Verify candidate literature before it appears in the main text.
8. If the paper describes a method, write `paper/algorithm_detail_ledger.md`
   using the algorithm detail contract. Extract details from proposal files,
   implementation maps, configs, relevant code, metrics, and reports. Do not
   rely on memory for algorithm steps.
9. If the paper describes a method, write `paper/source_obligation_matrix.md`
   using the source-to-paper obligation contract. Extract central obligations
   from source material before planning or drafting section files.
10. If the paper makes formal or semi-formal theory claims, write
   `paper/theory_obligation_ledger.md` using the theory and proof obligation
   contract. Include every objective, dynamic equation, endpoint or mass claim,
   recovery argument, exact-fit statement, theorem/proposition candidate, and
   proof obligation from source material.
11. Inventory candidate figures from `outputs/figures/`, existing reports, and
   any user-provided figure folders before deciding that no figure is available.
12. For selected main-paper or high-value supplement figures, read
   `~/.cellcompass/skills/workflow/scientific-visualization/SKILL.md`, inspect
   the upstream figure script or downstream source artifact, and regenerate the
   figure when it needs manuscript-quality layout, typography, palette, or
   vector export. Do not use this step to change the underlying analysis.
13. Write `paper/paper_plan.md` using the paper planning contract. Include the
   one-sentence contribution, paper type, claim-evidence matrix, section plan,
   figure inventory and figure/table plan, section-file plan, and supplement
   boundary. For method papers, include a method detail budget and an
   obligation-to-section map derived from `source_obligation_matrix.md`.
   For theory-bearing papers, include a theory/proof obligation map derived
   from `theory_obligation_ledger.md`.
14. Draft `paper/macros.tex` for shared commands and notation.
15. Draft section files under `paper/sections/` from `editorial_brief.md`,
   `paper_plan.md`, `algorithm_detail_ledger.md`, and
   `source_obligation_matrix.md`, not from raw run logs or old report prose.
   For theory-bearing papers, draft `sections/theory.tex` and proof-related
   supplement text from `theory_obligation_ledger.md`, not from intuition alone.
16. Assemble `paper/main.tex` from `macros.tex` and `sections/*.tex`. The main
   file must end at references unless the user explicitly requested a combined
   supplement PDF.
17. For theory-heavy work, define all notation before equations, add theorem or
   proposition statements only when the full proof can be written and audited.
   Put complete proofs in `supplement.tex` by default, not just proof sketches.
   If the proposal or algorithm ledger contains a mathematical form, dynamic
   objective, endpoint/mass recovery argument, Fokker--Planck equation, or
   exact-fit derivation, carry those mathematical objects and proof steps into
   `sections/theory.tex` and `supplement.tex`; do not replace them with a
   narrative summary.
18. Draft `paper/references.bib` only from cited entries in
   `citation_ledger.md`.
19. Add `paper/supplement.tex` for proofs, detailed limitations,
   configuration, run ids, reproducibility notes, extra figures, and evidence
   tables.
20. Write the audit files in `paper/checks/`: reverse outline, style audit,
   source obligation audit, theory consistency, algorithm coverage, figure
   hygiene, proof completeness when theory claims exist, and BibTeX/citation
   hygiene with provenance checks.
   The style audit must include a literal banned-term search over `main.tex`
   and `sections/*.tex`, list hits by file, record replacements, and fail if
   any unresolved main-body hit remains.
21. Verify the written files by listing or reading `<output_dir>/paper/`.
22. Run a LaTeX compile check when the toolchain is available; otherwise report
   that compile was not run.
23. Inspect the compile result for `main.pdf`, missing figures, unresolved
   cross-references, and citation warnings. Keep bibliography-only problems in
   `checks/bib_hygiene.md` unless production-ready bibliography was requested.
24. Spawn `paper_reviewer` with the paper directory and required files as
   relevant paths. If the reviewer returns `revise`, fix the required revisions
   and run the reviewer again. If it returns `reject`, return to
   `narrative_report.md` or `paper_plan.md`.
25. Before reporting completion, reread `checks/paper_reviewer_gate.md` and
   confirm it records the final reviewer decision instead of an intermediate or
   pending state.

## Rewrite Mode

Use this mode when the user asks to rewrite, regenerate, improve, or make an
existing paper more publication-like.

1. Audit the existing `paper/` directory before editing TeX.
2. If the paper is a method, model, training, or inference paper and lacks
   `algorithm_detail_ledger.md`, write that ledger first.
3. If the paper is a method, model, training, or inference paper and lacks
   `source_obligation_matrix.md`, write that matrix before touching
   `sections/*.tex`.
4. If the paper contains mathematical objectives, dynamics, endpoint or mass
   claims, recovery claims, theorem/proposition language, or proof claims and
   lacks `theory_obligation_ledger.md`, write that ledger before touching
   `sections/theory.tex` or `supplement.tex`.
5. If `paper_plan.md` is only a short bullet list, lacks a claims-evidence
   matrix, lacks a method detail budget, or lacks an obligation-to-section map,
   rewrite the plan before touching `sections/*.tex`. For theory-bearing
   papers, also require a theory/proof obligation map.
6. If `checks/source_obligation_audit.md`, `checks/algorithm_coverage.md`, or
   `checks/proof_completeness_audit.md` when theory claims exist is missing or
   stale, write it after drafting and before reviewer submission.
7. Rewrite `sections/method.tex`, `sections/theory.tex`,
   `sections/experiments.tex`, and `sections/results.tex` from the plan and
   ledgers. Do not patch only `main.tex` unless the user explicitly requested a
   narrow LaTeX fix.
8. If any manuscript, plan, ledger, or audit file is changed during rewrite,
   invalidate previous reviewer approval before reporting progress. Replace or
   update `checks/paper_reviewer_gate.md` with a stale marker such as:
   `Status: stale_after_rewrite; reason: paper files changed after the previous
   reviewer decision; required next step: rerun paper_reviewer`.
9. Rerun compile and reviewer after the full chain is consistent. If the
   reviewer returns `revise` or `reject`, continue the revision loop instead of
   reporting the paper as complete.

An existing `main.pdf` or previous reviewer approval does not waive these
requirements after the skill has changed or after the user requests a rewrite.

## Completion Checklist

- `paper/editorial_brief.md` exists and states the public method name,
  one-sentence contribution, closest comparisons, main insight, evidence
  package, and figure-first plan.
- `paper/narrative_report.md` exists.
- `paper/paper_plan.md` exists.
- `paper/evidence_ledger.md` exists.
- `paper/citation_ledger.md` exists.
- `paper/algorithm_detail_ledger.md` exists for method, model, training, or
  inference papers.
- `paper/source_obligation_matrix.md` exists for method, model, training, or
  inference papers.
- `paper/theory_obligation_ledger.md` exists when the paper contains formal or
  semi-formal theory claims, endpoint or mass claims, recovery claims,
  theorem/proposition language, or proof claims.
- `paper/macros.tex` exists.
- `paper/sections/*.tex` exists for every planned main section.
- `paper/main.tex` exists.
- `paper/references.bib` exists.
- `paper/supplement.tex` exists unless there is truly no proof, evidence table,
  run id, limitation detail, or reproducibility detail.
- `paper/checks/reverse_outline.md` exists.
- `paper/checks/style_audit.md` exists.
- `paper/checks/theory_consistency.md` exists.
- `paper/checks/proof_completeness_audit.md` exists when
  `theory_obligation_ledger.md` exists or when the paper contains theorem,
  proposition, guarantee, recovery, endpoint, mass, or proof language.
- `paper/checks/source_obligation_audit.md` exists for method, model, training,
  or inference papers.
- `paper/checks/algorithm_coverage.md` exists for method, model, training, or
  inference papers.
- For algorithm comparison papers, `evidence_ledger.md` includes
  `query_campaign_baseline_metrics` evidence or exact baseline `metrics_path`
  provenance for every baseline value in the manuscript.
- `paper/checks/bib_hygiene.md` exists.
- If compiled, `paper/main.pdf` exists; `paper/main.bbl` is checked and any
  bibliography-only issue is recorded in `checks/bib_hygiene.md`.
- The `paper_reviewer` subagent returned `paper_review.decision=approve`.
- Self-authored paper checks, fallback notes, or manually edited
  `paper_reviewer_gate.md` updates are not reviewer approval. A reviewer
  infrastructure failure must leave the paper as pending reviewer
  infrastructure, not complete.
- For a new algorithm paper, the title/abstract use a public method name that is
  short, readable, and conceptually motivated; the internal algorithm id appears
  only as provenance outside the main argument.
- The Introduction contains a concise contribution paragraph or contribution
  bullets adapted from `editorial_brief.md`; it must read as manuscript prose,
  not as a note about the brief.
- `paper/checks/paper_reviewer_gate.md` records that final approval and does
  not contain stale `pending` language from an earlier reviewer round.
- The reviewer approval is fresh for the current file set. If `main.tex`,
  `sections/*.tex`, `supplement.tex`, `paper_plan.md`,
  `source_obligation_matrix.md`, `theory_obligation_ledger.md`, or
  `checks/*.md` changed after the reviewer decision, that approval is stale and
  completion is blocked until
  `paper_reviewer` approves the revised files.
- A reviewer approval is valid only if it considered the current file set. For
  method papers, approval is invalid when `algorithm_detail_ledger.md`,
  `source_obligation_matrix.md`, `checks/source_obligation_audit.md`,
  `checks/algorithm_coverage.md`, a method-detail budget, or an
  obligation-to-section map is missing.
- `paper/main.tex` does not input appendix or supplement files unless the user
  explicitly requested a combined PDF.
- Citation commands have BibTeX entries where possible; missing or imperfect
  bibliography entries are advisory unless they affect a central claim.
- Citation and bibliography issues are recorded in `checks/bib_hygiene.md`;
  they block completion only when they leave a central paper claim unsupported or
  introduce an apparently fabricated critical reference.
- Results are organized by paper claim, not by run-log order.
- For method papers, `sections/method.tex` explains the problem setup,
  supervision or target construction, model parameterization, training
  objective, training procedure, inference procedure, and evaluation protocol.
- For method papers, `paper_plan.md` contains a method detail budget and
  obligation-to-section map, and `checks/source_obligation_audit.md` confirms
  central obligations from source material appear in the manuscript or
  supplement.
- `paper_plan.md` includes a candidate figure inventory. If candidate image
  files exist, each is marked `main`, `supplement`, or `reject` with a reason.
- The main paper includes at least one method, trajectory, or result figure when
  a suitable real figure exists. Extra diagnostic figures belong in
  `supplement.tex`.
- Selected manuscript figures have source artifacts and script provenance. If a
  figure was refactored with scientific-visualization, both PNG and editable
  vector PDF outputs exist or the exception is recorded.
- Tables and figures correspond to real evidence or are clearly marked as
  planned/missing.
- Main text does not contain long run ids, local absolute paths, internal saved
  result paths, experiment bookkeeping, internal workflow language, or tool
  language unless the user explicitly asks for a report-style paper.
- `checks/style_audit.md` records a banned-term search over `main.tex` and
  `sections/*.tex`; the verdict is `pass`, with zero unresolved hits for
  `rollout`, `artifact`, `runtime`, `contract`, `preflight`, `hard gate`,
  `gate`, `campaign`, `trial`, `trial_`, `fallback`, `agent`, `workspace`,
  `implementation map`, `local path`, `sealed trajectory`, `tool call`,
  `approved`, `final locked`, `locked release`, `final regression`, `Stage 1`,
  `Stage 2`, and `Stage 3`.
- Detailed limitations, missing baselines, small sample sizes, failed runs, and
  reproducibility details are present in Discussion, supplement, or
  `evidence_ledger.md`.
- Mathematical notation is defined before use; theorem statements and
  supplement restatements agree.
- When source materials contain formal or semi-formal theory, `sections/theory.tex`
  contains explicit mathematical objects, assumptions, propositions/theorems,
  and proof or derivation steps rather than only explanatory prose.
- When formal or semi-formal theory appears, `theory_obligation_ledger.md` maps
  each source theory item to assumptions, objects, proof status, and exact main
  or supplement location.
- `checks/proof_completeness_audit.md` compares every item in
  `theory_obligation_ledger.md` against `sections/*.tex`, `main.tex`, and
  `supplement.tex`, with a `pass` verdict only when all central proof claims are
  fully proved, correctly derived, or explicitly downgraded.
- Any sentence claiming a full proof exists in `supplement.tex` is true: the
  supplement contains a complete proof, not only intuition, citation, or a proof
  sketch.
- If a complete proof is not available, the claim is downgraded to a derivation,
  conjecture, design rationale, or proof obligation rather than stated as a
  theorem/proposition.
