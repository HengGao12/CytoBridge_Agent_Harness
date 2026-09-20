---
name: report-authoring
description: Route final writing requests to the right workflow writing skill. Use for report.html, final summaries, research reports, papers, manuscripts, top-conference drafts, or LaTeX writing.
---

# Report Authoring Router

Use this skill as the compatibility entrypoint for final writing. Do not treat
this file as the full writing manual. Pick the most specific writing skill below,
read it, and follow it.

## Route

- For biological-application algorithm lifecycle outputs, first verify that the
  final-regression locked release has been interpreted through
  `~/.cellcompass/skills/workflow/downstream-analysis/SKILL.md`. If downstream
  manifests, script-backed tables/figures, warnings, and supported-claim
  records are missing, run downstream analysis before routing to report or
  paper writing. Final metrics alone are not enough biological interpretation.
  They establish metric-level completion only, not biological validation.
  A high-quality biological report must preserve both bars: competitive
  quantitative metrics and a clear explanation of what the algorithm reveals
  biologically, with downstream artifacts supporting that interpretation.
- If the user asks for `report.html`, final workflow summary, research report,
  RAG-backed explanation, readable HTML deliverable, artifact-backed review, or
  figure-backed interpretation:
  - read `~/.cellcompass/skills/workflow/research-report-authoring/SKILL.md`
- If the user asks for paper, manuscript, top conference, NeurIPS, ICLR, ICML,
  LaTeX, submission draft, method paper, or publication-style writing:
  - read `~/.cellcompass/skills/workflow/paper-authoring/SKILL.md`
- If both are requested, write the report first, then use it as one evidence
  source for the paper draft.
- For central report or paper figures, the selected downstream/report/paper
  writing skill should route through
  `~/.cellcompass/skills/workflow/scientific-visualization/SKILL.md` before
  final rendering, while preserving the underlying analysis evidence.
- If the target is ambiguous, default to `research-report-authoring` unless the
  user explicitly mentions paper/manuscript/LaTeX/conference.

## Compatibility

Older prompts may refer to this skill directly. In that case, route immediately
using the rules above. Do not copy legacy report-writing rules into the answer.
