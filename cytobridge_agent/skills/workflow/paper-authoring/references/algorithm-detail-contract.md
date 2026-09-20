# Algorithm Detail Contract

Write `paper/algorithm_detail_ledger.md` before `paper/paper_plan.md` whenever
the manuscript describes a method, algorithm, model, training procedure, or
inference workflow.

## Purpose

This ledger prevents the paper from collapsing into a high-level method summary.
It extracts the actual algorithm from proposals, implementation maps, configs,
source code, reports, and metrics before any LaTeX section is written.

## Required Sources

Inspect all available sources that can define the method:

- proposal or algorithm documents, including files such as `PROPOSAL.md`,
  `IMPLEMENTATION_MAP.md`, theory notes, design notes, or method specs.
- training configs, run manifests, evaluator configs, and metric summaries.
- relevant implementation files when the proposal is incomplete.
- figures and tables that explain the method or evaluation.
- previous reports only as evidence sources, not as section structure.

If a source is unavailable, record it explicitly as missing instead of filling
the gap from memory.

## Required Contents

`algorithm_detail_ledger.md` must include:

- Method name and one-sentence algorithmic contribution.
- Motivation-to-method chain: the concrete failure mode or scientific question,
  why existing/builtin methods do not represent it, and the minimal new
  mathematical object or modeling choice introduced to address it.
- Problem setup: input data, empirical measures, time index, state space,
  selected views, context variables, and prediction target.
- Assumptions: equal or unequal mass, stochastic or deterministic dynamics,
  known or learned endpoints, intervention or observational setting, and any
  constraints needed for the method to make sense.
- Core variables and notation: all measures, couplings, weights, time variables,
  particle states, neural fields, loss weights, and evaluation quantities.
- Supervision construction: how endpoint pairs, couplings, bridge points,
  velocities, scores, growth targets, or labels are produced.
- Model parameterization: the exact learned fields, decompositions, residuals,
  projections, graph or regulatory operators, and what each component is
  allowed to represent.
- Objective: every training loss term, regularizer, diagnostic term, and which
  terms define the algorithm versus only monitoring it.
- Training procedure: sampling loop, batch construction, optimizer-relevant
  choices when available, curriculum or stages, and stopping or selection rule.
- Inference procedure: initial condition, integration/simulation rule, weight
  update, generated trajectory, output objects, and what is not allowed at
  inference time.
- Evaluation contract: metrics, baselines or comparisons, exact generated
  object being scored, and leakage or target-repair constraints.
- Theory obligations: endpoint recovery, mass conservation or mass change,
  stochastic dynamics, consistency, identifiability, or causal interpretation
  claims that require derivation or proof.
- Paper mapping: which details must appear in `sections/method.tex`,
  `sections/theory.tex`, `sections/experiments.tex`, `sections/results.tex`,
  and which details belong only in `supplement.tex`.
- Missing details: any algorithm component that cannot be recovered from the
  available evidence.

## Method Coverage Requirements

`sections/method.tex` must cover the central algorithm in paper language. For a
new method paper, the method section is incomplete unless it explains:

1. the mathematical problem being solved;
2. the motivation-to-method chain from failure mode to modeling choice;
3. how supervision or training targets are constructed;
4. the model parameterization and each major component;
5. the training objective and which terms are essential;
6. the training loop or data sampling procedure at a level sufficient to
   reproduce the algorithm conceptually;
7. the inference procedure used to generate the reported outputs;
8. the evaluation object and any no-leakage or no-repair rule.

For algorithms based on transport, flow matching, stochastic dynamics, or
growth fields, explicitly define couplings, row or column masses, particle
weights, bridge interpolation or simulation paths, velocity targets, score
targets, growth targets, and endpoint measures when those objects exist in the
source material.

## Theory Coverage Requirements

`sections/theory.tex` must not be only a slogan when the method has a stated
endpoint, mass, stochastic, or recovery property.

- If the proposal contains sections such as `Problem Mathematical Form`,
  `Mathematical Derivation to Algorithm Design`, `Theoretical Core`, or
  `Distribution Recovery Argument`, extract those items into a dedicated
  theory-obligation subsection in `algorithm_detail_ledger.md`.
- The paper plan must allocate each theory obligation to `sections/theory.tex`
  or `supplement.tex`.
- State the assumptions and mathematical objects before the claim.
- Introduce each mathematical object because the motivation requires it. Avoid
  dumping notation before the reader knows what problem the notation solves.
- Derive the endpoint or mass property algebraically when it follows from the
  construction.
- Use theorem/proposition only when a complete proof exists.
- If a complete proof is not available, write a derivation or proof obligation
  and name the missing step.
- Do not claim that the supplement contains a full proof unless the proof has
  been written and checked.

## Planning Rule

After this ledger is written, write `source_obligation_matrix.md` before
`paper_plan.md`. The obligation matrix decides which source-derived method,
math, theory, inference, and evaluation details are central paper obligations.
If any extracted theory obligation contains a formal claim, objective, dynamic
equation, endpoint or mass recovery argument, exact-fit logic, or theorem-level
language, write `theory_obligation_ledger.md` before `paper_plan.md`.

`paper_plan.md` must include a "Method detail budget" section that maps every
required method and theory detail from this ledger to a destination section, and
an "Obligation-to-section map" derived from `source_obligation_matrix.md`. A
paper plan without these sections is incomplete.

## Audit Rule

`checks/source_obligation_audit.md`, `checks/theory_consistency.md`,
`checks/proof_completeness_audit.md` when theory claims exist, and
`checks/algorithm_coverage.md` must compare `algorithm_detail_ledger.md`,
`source_obligation_matrix.md`, and `theory_obligation_ledger.md` when present
against the written sections. Missing central source obligations are blocking
issues, even if the prose is polished.
