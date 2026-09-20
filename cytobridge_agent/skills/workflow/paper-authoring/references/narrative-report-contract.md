# Narrative Report Contract

Write `paper/narrative_report.md` before `paper_plan.md`.

## Purpose

The narrative report converts available materials into a paper story. It is not
the final manuscript and should not preserve workflow order, run-log order, or
agent execution details.

## Required Contents

- Working title and one-paragraph paper thesis.
- Reader model: the intended reader, what they know already, and what must be
  explained without assuming project-specific background.
- Motivation ladder: biological or mathematical problem, why it matters, why
  existing/builtin approaches are insufficient, and the smallest new idea that
  resolves the gap.
- Evidence inventory grouped by scientific role: motivation, method, theory,
  experiment, result, related work, and missing-before-submission.
- Candidate contributions, each written as a paper claim.
- Claim boundaries: what the paper can say positively and what must stay out of
  the main paper.
- Main narrative arc: problem, gap, method idea, mathematical construction,
  evaluation, and implication.
- Coherence check: how the Introduction promise, Method construction,
  Experiments, Results, and Discussion answer the same central question.
- Material to exclude from the main paper: long run ids, local paths, detailed
  configuration, failed runs, workflow/tool notes, and bookkeeping.

## Rules

- Use this file to distill any existing `report.html`, `report.md`, figures,
  logs, RAG output, proposal notes, or experiment summaries.
- Do not copy section order from an experiment report unless it already matches
  a research-paper argument.
- Do not write final prose here. Write concise planning prose that can be used
  by `paper_plan.md`.
- Do not use unexplained project shorthand. If the thesis depends on a term
  such as WFR, flow matching, bridge, growth, branch, or fate probability, state
  the object in plain language before using the acronym.
- If the evidence is mostly theory plus a small evaluation, state that the
  likely paper type is method/theory plus proof-of-concept evidence.
- If the evidence does not support broad superiority, do not create a
  benchmark-winning narrative.
