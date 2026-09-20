# Theory And Proof Obligation Contract

Write `paper/theory_obligation_ledger.md` whenever a manuscript makes formal or
semi-formal claims about objectives, dynamics, endpoint behavior, mass behavior,
stochastic processes, identifiability, consistency, recovery, optimality, or
exact-fit properties.

## Purpose

This ledger prevents mathematical content from collapsing into polished prose.
It separates three different things:

- what the source material mathematically claims;
- what the paper can actually prove or derive;
- what must be downgraded to design rationale, conjecture, or future proof
  obligation.

The ledger is not optional for method papers whose proposal, implementation
map, algorithm ledger, or source notes contain equations, objectives,
constraints, SDE/Fokker--Planck or continuity equations, endpoint or mass
arguments, or claimed recovery behavior.

## Required Sources

Inspect all available sources that define the mathematical content:

- proposal sections with mathematical forms, derivations, objectives, or
  theory notes;
- `algorithm_detail_ledger.md` and `source_obligation_matrix.md`;
- implementation maps, configs, and source code when they define losses,
  sampling laws, inference rules, or constraints;
- reports, metric definitions, and evaluator code when evaluation semantics
  affect a mathematical claim.

If a source is missing, record the gap explicitly. Do not supply a proof from
memory unless the necessary assumptions and derivation are written down and
auditable.

## Required Ledger Fields

Use a table with these columns:

- `id`: stable short identifier such as `T1`, `T2`.
- `source_evidence`: exact source file, section, line range, config key, code
  symbol, or equation.
- `claim_or_object`: the mathematical claim, objective, dynamic equation,
  recovery property, or object being formalized.
- `centrality`: `central`, `supporting`, or `supplement_only`.
- `assumptions_needed`: domains, regularity, sampling, mass, endpoint,
  stochastic, independence, or finite-sample assumptions required for the claim.
- `objects_and_notation`: measures, variables, time indices, weights, losses,
  fields, couplings, kernels, or maps that must be defined before the claim.
- `required_result_type`: `definition`, `derivation`, `proposition`,
  `theorem`, `lemma`, `proof_obligation`, `conjecture`, or `design_rationale`.
- `proof_status`: `complete`, `partial`, `missing`, `downgraded`, or
  `not_applicable`.
- `main_location`: exact planned or actual location in `sections/theory.tex`,
  `sections/method.tex`, or another main section.
- `supplement_location`: exact planned or actual location in `supplement.tex`
  for full proof or details.
- `blocking_issue`: empty only when the item is fully proved, correctly
  derived, or safely downgraded in the manuscript.

## Rules

- A formal theorem, proposition, lemma, or corollary may appear in the paper
  only when the corresponding ledger item has a complete proof path.
- A main-body proof sketch is allowed only if a complete proof for the same
  labeled statement exists in `supplement.tex`.
- Phrases such as `we prove`, `guarantee`, `exactly recovers`, `full proof is
  given in the supplement`, or `satisfies the dynamics` are formal claims. They
  require matching assumptions, statement, and proof or derivation.
- If the proof is incomplete, the paper must downgrade the language to
  `derivation`, `design rationale`, `proof obligation`, or `conjecture`, and it
  must name the missing step.
- A mathematical construction is not covered by a paragraph that only describes
  intuition. When the source contains an objective, constraint, sampling law,
  dynamic equation, or inference rule, the paper must include the formula or a
  numbered procedural derivation.
- `checks/proof_completeness_audit.md` must compare this ledger against
  `sections/*.tex`, `main.tex`, and `supplement.tex` before reviewer approval.

## Relationship To Other Files

- `algorithm_detail_ledger.md` extracts method and theory details.
- `source_obligation_matrix.md` decides which source-derived items are central
  paper obligations.
- `theory_obligation_ledger.md` specializes those mathematical obligations into
  assumptions, statements, and proof status.
- `paper_plan.md` must include a "Theory/proof obligation map" derived from
  this ledger whenever the ledger exists.
- `checks/theory_consistency.md` checks notation and consistency.
- `checks/proof_completeness_audit.md` checks whether proof claims are actually
  proved, derived, or downgraded.
- The `paper_reviewer` must independently infer proof obligations and compare
  them against this ledger before approving.
