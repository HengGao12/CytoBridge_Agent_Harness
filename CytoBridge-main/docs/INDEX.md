---
title: "CytoBridge documentation map"
summary: "Agent-facing entrypoint for choosing the narrowest CytoBridge package document."
read_when:
  - "Deciding which CytoBridge package document to read next"
---
# CytoBridge Documentation Map

This is the agent-facing entrypoint. Do not read every document up front. Pick the narrowest file for the current task.

## Agent-Friendly Docs Design

This docs layout is designed for agent retrieval, not human cover-to-cover
reading:

- compact discovery index first;
- category-level routing before long references;
- progressive disclosure: read the current task page, then follow deeper links only when needed;
- short task pages with source-escalation order only when docs are insufficient.

## Start Here

| Need | Read |
| --- | --- |
| I need to decide what document to read | `docs/runtime/retrieval-map.md` |
| I need builtin algorithm theory or exact-fit arguments | `docs/theory/README.md` |
| I need package data, evaluation, preprocessing, or fit contracts | `docs/runtime/data-and-evaluation-contracts.md` |
| I need custom algorithm workspace rules | `docs/runtime/custom-algorithms/README.md` |
| I need flow-matching runtime extension details | `docs/runtime/flow-matching/README.md` |
| I need to implement a custom flow-matching algorithm from the closest builtin source path | `docs/runtime/flow-matching/builtin-implementation-guide.md` |
| I need downstream biological analysis after training | `docs/runtime/downstream/README.md` |
| I need downstream model semantics, ODE/SDE choice, or balanced/unbalanced interpretation | `docs/runtime/downstream/model-semantics.md` |
| I need downstream compute or plotting APIs | `docs/runtime/downstream/api-reference.md` |
| I need runtime compatibility or known failure modes | `docs/runtime/compatibility-projection-failures.md` |

## Skill Routing

Do not use a separate skill index file. The runtime already exposes skill names
and descriptions in the prompt. If the current stage is unclear, call
`list_skills()` and then read only the selected `SKILL.md`.

For custom algorithm work, start from
`cytobridge_agent/skills/algorithm/algorithm-orchestrator/SKILL.md`.
