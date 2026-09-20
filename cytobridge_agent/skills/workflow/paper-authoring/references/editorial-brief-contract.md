# Editorial Brief Contract

Write `paper/editorial_brief.md` after evidence inventory and before
`narrative_report.md`.

## Purpose

The editorial brief is the paper's high-level argument in one page. It is for
turning a completed algorithm lifecycle into a readable research contribution,
not for deciding whether to write a paper. When an algorithm completes the
required lifecycle, the agent should still produce a paper; the brief makes the
paper sharper.

## Required Contents

- Public method name: short, pronounceable, and tied to the central idea.
- Internal algorithm id: recorded separately for provenance.
- One-sentence contribution: what the paper contributes in reader-facing terms.
- Target reader: who should care and what they need to understand first.
- Motivation: the biological or mathematical problem and why it matters.
- Gap: the nearest method family or builtin behavior that leaves the problem
  unresolved.
- Core idea: the new modeling object, assumption, objective, or evaluation
  target that makes the method different.
- Main insight: the conceptual, biological, or mathematical takeaway that should
  remain after reading the paper.
- Closest comparisons: the two or three nearest methods/baselines and the
  concrete difference from each.
- Evidence package: the main quantitative result, the main downstream
  biological or mechanistic result when applicable, and the ablation/control
  result when available.
- Figure-first plan: candidate main figures and what each figure should prove.
- Claim boundaries: what the paper can say confidently, what must be bounded,
  and what belongs in the supplement.
- Article-facing contribution summary: a concise paragraph or contribution
  bullets that can be adapted into the Introduction without mentioning this
  internal brief.

## Rules

- Do not write the brief as a run summary.
- Do not mention approvals, stages, gates, final locked state, local paths,
  internal ids, or workflow chronology except for the separate provenance id.
- Do not let metric names substitute for motivation. Explain why the measured
  quantity matters for the paper's claim.
- If evidence is modest, write a narrower paper argument rather than a
  defensive one. The output is still a paper draft, but its claims should be
  bounded honestly.
- The brief should make it possible for a reader to understand the paper's
  contribution before reading any formulas.
