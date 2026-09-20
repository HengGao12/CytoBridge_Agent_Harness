---
title: "Runtime retrieval map"
summary: "Routing map for runtime docs and source escalation order."
read_when:
  - "Choosing runtime docs or source files to inspect"
---
# Runtime Retrieval Map

Start here when deciding which package document to read next.

# CytoBridge Local API Reference (Agent-Facing)

This document is the canonical local reference for agents that call
`CytoBridge-main/CytoBridge/*` APIs in this workspace.

Focus:
- Exact input contract.
- Exact return payload shape.
- Side effects on `adata`.
- Flow-matching developer extension points.

## Developer Docs

- Custom training workflow:
  - `docs/runtime/custom-algorithms/README.md`
  - Use this when an agent or developer needs to save an editable custom
    training algorithm under `~/.cellcompass/training_algorithms/` and run it
    through the standardized training run bundle system.

- Flow-matching extension API:
  - `docs/runtime/flow-matching/README.md`
  - Use this when you want to override coupling construction, stochastic path
    construction, mass/growth targets, or the full flow-matching backend used
    by `cb.tl.fit(...)`. This is also where default hook behavior,
    `TrainingDataBundle` contracts, and mini-batch/scalability boundaries are
    documented.

- Downstream analysis API:
  - `docs/runtime/downstream/README.md`
  - Use this after a trained model exists and you need model-native trajectory,
    fate, growth, perturbation, driver, prediction, or report-facing biological
    evidence. This is also where downstream artifact provenance and
    continuous-dynamics evidence boundaries are documented.

- Builtin algorithm principles:
  - `docs/theory/README.md`
  - Use this when you need to understand what each builtin algorithm solves,
    why its distribution / mass fitting claim is mathematically plausible, and
    whether TMV is a hard gate or only diagnostic.

## Documentation Roadmap

Use the docs in this order instead of jumping straight into source files.

### If you are still at proposal stage

Do not use package docs or source as required pre-read.

At proposal time, stay at:

- mathematical semantics
- recoverability
- implementation-oriented paper-style pseudocode
- evaluation plan

Use `docs/theory/README.md` only as a capability and baseline
map. Do not require proposal authors to inspect runtime source code before the
theory gate.

Package docs and source become relevant at authoring time, after the proposal
is already approved in theory.

### If you are implementing an approved proposal

Read:

1. this file, especially:
   - `0. Custom Algorithm Workflow`
   - `4. Training Recoverability And Evaluation Semantics`
2. `docs/theory/README.md`
3. `docs/runtime/custom-algorithms/README.md`
4. `docs/runtime/flow-matching/README.md`

That should be enough for most custom algorithms.

### If you are doing downstream biological analysis

Read:

1. `docs/runtime/downstream/README.md`
2. `docs/runtime/downstream/semantics-and-evidence.md`
3. `docs/runtime/downstream/model-semantics.md`
4. `docs/runtime/downstream/api-reference.md`
5. `docs/runtime/downstream/recipes-and-artifacts.md`
6. `docs/runtime/downstream/checklist-and-failures.md`

Do not jump to evaluator file contracts or benchmark runners until the
model-native downstream computation and generic artifacts are defined.

### If the docs are still not enough

Escalate to source inspection in this order:

1. `CytoBridge-main/CytoBridge/tl/training_algorithm.py`
   - custom algorithm extension points
2. `CytoBridge-main/CytoBridge/tl/fit.py`
   - how those extension points are wired into `cb.tl.fit(...)`
3. `CytoBridge-main/CytoBridge/tl/flow_matching_backends.py`
   - coupling, pair sampling, path, and mass semantics
4. `CytoBridge-main/CytoBridge/tl/trainer.py`
   - training loss, evaluation, and runtime orchestration

Only inspect more source once the previous layer is insufficient.

## 10. Recommended Retrieval Pattern for Agents

Use this order only when API/interface details are unclear:

1. `find_files(path="CytoBridge-main/docs", pattern="*.md")`
2. `grep_files(path="CytoBridge-main/docs", pattern="<symbol>", include="*.md")`
3. `read_file(file_path="<matched doc path>")`
4. If still unclear, inspect source in:
   - `CytoBridge-main/CytoBridge/tl/downstream/*.py`
   - `CytoBridge-main/CytoBridge/pl/downstream/*.py`
