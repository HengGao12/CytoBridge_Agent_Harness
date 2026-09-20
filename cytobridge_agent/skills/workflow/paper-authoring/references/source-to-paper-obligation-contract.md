# Source-To-Paper Obligation Contract

Write `paper/source_obligation_matrix.md` for every method, model, training, or
inference paper before writing `paper/paper_plan.md` or `paper/sections/*.tex`.

## Purpose

The obligation matrix prevents a paper from passing review because author-written
audits say "covered" while central source obligations are still missing from the
manuscript. It extracts what the paper must explain from source materials, then
forces the paper and reviewer to account for each obligation.

This contract is general. Do not add task-specific checklists for clone, fate,
CITE, GRN, OT, diffusion, or any other domain. Instead, map each source-derived
obligation into the categories below.

## Required Sources

Inspect all available sources that define the work:

- proposal, algorithm, method, design, or theory documents;
- `IMPLEMENTATION_MAP.md`, configs, manifests, reports, metrics, and figures;
- relevant source code when source documents are incomplete;
- evaluation scripts, metric definitions, benchmark outputs, and baseline
  records when the paper makes comparison claims.

If a source is unavailable, record the missing source explicitly. Do not fill the
gap from memory.

## Obligation Categories

Each central source obligation must be assigned one category:

- `problem_abstraction`: inputs, outputs, observed objects, hidden variables,
  supervision, prediction target, and the general problem statement.
- `mathematical_construction`: objectives, constraints, divergences, losses,
  regularizers, targets, sampling laws, update rules, and inference equations.
- `algorithm_reconstruction`: the procedural path from data to training targets,
  model fitting, inference, and generated outputs.
- `claim_mechanism_link`: which modeling choice supports each contribution or
  performance claim.
- `theory_or_proof`: assumptions, statements, derivations, proofs, proof
  obligations, and what is not proved.
- `evaluation_semantics`: scored object, truth source, baseline source,
  filtering, grouping, weights, leakage boundary, and shared metric code path.
- `internal_consistency`: places where Method, Theory, Experiments, Results, or
  supplement must use the same semantics.
- `paper_framing`: whether a new method is introduced as a general problem and
  solver before specializing to one dataset or application.

## Required Matrix Fields

Use a table with these columns:

- `id`: stable short identifier such as `O1`, `O2`.
- `category`: one of the categories above.
- `source_evidence`: exact source file, section, line range, config key,
  metric path, or code symbol.
- `centrality`: `central`, `supporting`, or `supplement_only`.
- `why_it_matters`: what claim or reader understanding would fail if omitted.
- `expected_paper_location`: planned main or supplement location.
- `required_form`: `formula`, `objective`, `constraint`, `algorithm_step`,
  `proof`, `derivation`, `definition`, `evaluation_rule`, `framing`, or
  `caveat`.
- `actual_paper_location`: fill after drafting; use exact section or equation.
- `status`: `covered`, `partial`, `missing`, `downgraded`, or `not_applicable`.
- `blocking_issue`: empty only when a central obligation is genuinely covered
  or safely downgraded.

## Rules

- A central `mathematical_construction` obligation cannot be satisfied by a
  verbal paragraph when the source contains an objective, constraint, target,
  loss, sampling law, or inference rule. It needs the corresponding formula or
  step sequence in the paper or supplement.
- A central `algorithm_reconstruction` obligation is incomplete unless a reader
  can conceptually reproduce data preparation, target construction, training,
  inference, and output generation without local run logs.
- A central `evaluation_semantics` obligation is incomplete unless the paper
  states what is predicted, what is treated as truth, which baselines are
  compared, and whether the same metric code/filtering is used.
- A central `internal_consistency` obligation is blocking when two sections use
  incompatible semantics for the same construction.
- A new method paper must include a `paper_framing` obligation. If the paper
  only presents a dataset-specific recipe and never defines the general problem
  it solves, the obligation is missing.
- If a proof is not complete, downgrade the claim to a derivation, intuition,
  design rationale, or proof obligation. Do not present it as a theorem or
  proposition.

## Relationship To Other Files

- `algorithm_detail_ledger.md` extracts algorithm details.
- `source_obligation_matrix.md` decides which extracted details are mandatory
  for the paper and how they must appear.
- `theory_obligation_ledger.md` is required after this matrix when any central
  `theory_or_proof` or `mathematical_construction` obligation contains a formal
  claim, objective, dynamic equation, endpoint/mass argument, recovery argument,
  exact-fit statement, or theorem/proposition candidate.
- `paper_plan.md` must include an "Obligation-to-section map" derived from this
  matrix.
- `checks/source_obligation_audit.md` verifies the final manuscript against this
  matrix.
- The `paper_reviewer` must independently infer source obligations and compare
  them against the author's matrix before approving.
