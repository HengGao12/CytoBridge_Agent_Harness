---
name: theory-selection
description: Select the single best CytoBridge model family for the current temporal dataset, with a concrete rationale tied to data properties and user constraints.
---

# Theory Selection

Use this skill when selecting a CytoBridge model family or configuration strategy for a temporal snapshot dataset.

## Goal
Choose the single best modeling family for the current biological system and user request. This stage is not generic brainstorming. It should produce a defensible decision grounded in both dataset properties and CytoBridge theory.

This skill also serves a second purpose during idea work:

- before proposing a new idea, use it to understand what the package already covers
- use that boundary to avoid proposing fake novelty or asking the package to solve a question it already directly supports
- do not drop into source-code inspection at this stage

## Read first when unsure
- `references/package-capability-map.md`
- `CytoBridge-main/docs/INDEX.md`
- `CytoBridge-main/docs/theory/README.md`
- `CytoBridge-main/docs/runtime/flow-matching/README.md`
- `CytoBridge-main/readme.md`
- `cytobridge_agent/skills/planner/research-question-framing/references/agent_algo_survey_20260417/reading-order.md`
- `search_literature` results
- relevant config files under `CytoBridge-main/CytoBridge/configs/`

## Use This Skill For Boundary Calibration At Idea Stage

If the task is still at the research-idea stage, use this skill narrowly:

- read `references/package-capability-map.md`
- read package docs and builtin configs
- understand what current builtins already implement
- understand what is still outside the first-class builtin boundary

At this stage:

- do **not** inspect low-level source code by default
- do **not** confuse package capability review with implementation authoring
- do **not** call a direction novel if it is already a straightforward application of an existing builtin family

## Runtime assumptions
- The active dataset is already loaded as the current runtime `adata`.
- Reuse the current in-memory object whenever possible.
- Do not repeatedly reload `.h5ad` unless the active dataset/path has explicitly changed or rollback is required.
- Treat committed workflow state and explicit user/system constraints as authoritative when they are already available.

## Mandatory outcome
- Choose exactly one configuration family.
- Do not return a vague shortlist as the final answer.
- The selected candidate must be representable as:
  - `name`
  - `overrides`
  - `rationale`
- User override wins. If the user explicitly requests a family, honor it and document that it is a user-imposed override.

## Model families to reason about
You should reason explicitly about the current builtin CytoBridge families:

- `dynamical_ot`
  - `velocity` only
  - pure `neural_ode`
  - deterministic transport baseline
  - best when mass is approximately conserved and dynamics are relatively simple

- `balanced_ot_cfm`
  - `velocity` only
  - simulation-free `flow_matching`
  - balanced OT-CFM dynamic OT baseline
  - TMV is diagnostic only

- `unbalanced_ot`
  - `velocity + growth`
  - pure `neural_ode`
  - use when proliferation/death or mass imbalance matters
  - deterministic relative to stochastic families

- `vgfm`
  - `velocity + growth`
  - simulation-free `flow_matching`
  - deterministic unbalanced FM default
  - use when mass imbalance matters but stochasticity is not the target

- `wfrfm`
  - `velocity + growth`
  - simulation-free `flow_matching`
  - explicitly solves WFR dynamic unbalanced OT through WFR-OET semi-coupling
  - use as the named-WFR alternative to the simpler `vgfm` default
  - diagnose whether WFR's velocity-growth relation fits the biological prior

- `sf2m`
  - `velocity + score`
  - simulation-free `flow_matching`
  - balanced stochastic bridge baseline
  - use when stochasticity matters but total mass is intentionally conserved

- `ruot`
  - `velocity + growth + score`
  - hybrid `neural_ode -> flow_matching -> neural_ode`
  - regularized unbalanced OT style stochastic dynamics
  - stronger stochastic family, but not fully simulation-free end to end

- `crufm`
  - `velocity + growth + score`
  - `simulation-free flow_matching`
  - faster stochastic alternative, often practically preferred in practice

- `cyto_simulation`
  - `velocity + growth + score + interaction`
  - hybrid interaction-aware route
  - use only when the biological question truly needs interaction as a dynamical driver

Read `references/package-capability-map.md` for the exact training-style and boundary summary of each family.

## Current Package Boundary You Must Keep Straight

The current package is **not** just a Neural ODE toolkit anymore.

It already spans:

- pure ODE families
- hybrid ODE plus flow-matching families
- balanced and unbalanced simulation-free FM families
- an interaction-aware builtin family

So:

- do not present "move from Neural ODE to FM" as a sufficient novelty claim by itself
- do not propose a new idea without first checking whether `balanced_ot_cfm`,
  `sf2m`, `vgfm`, `wfrfm`, `crufm`, or `ruot` already cover the object
- do not assume time-series spatial transcriptomics, lineage-aware coupling, geometry-aware FM, or general multimodal dynamics are already first-class builtins

## Operating rules
- Inspect the actual dataset before deciding.
- Use `search_literature` when the modeling choice is unclear or when the user asks for a theory-grounded justification.
- Use `list_path` and `read_file` only when path context or metadata needs verification.
- Prefer one strong candidate rather than a weak ranked list.
- If the user explicitly requests a model family, that overrides your recommendation; document the override clearly.
- Do not spend excessive turns on theory selection. Be decisive once the evidence is sufficient.
- Do not invent interaction-aware or stochastic requirements that the data/task does not support.

## Recommended workflow
1. If this decision is part of idea work, calibrate package boundary first
   - read `references/package-capability-map.md`
   - read the relevant builtin configs
   - avoid source-code inspection unless the task has become implementation authoring
2. Inspect the current dataset
   - cell count by time
   - number of time points
   - label structure
   - whether there is obvious growth or contraction over time
3. Estimate whether mass is approximately conserved
   - if cell counts vary strongly across time points, treat unbalanced families as preferred
4. Evaluate whether stochastic dynamics are likely important
   - heterogeneous transitions
   - branching uncertainty
   - noisy developmental systems
5. Evaluate whether interaction-aware modeling is actually needed
   - only choose `cyto_simulation` when the data/task supports it
6. If literature context is needed, first use the packaged survey copy in `research-question-framing/references/agent_algo_survey_20260417/`:
   - `synthesis.md`
   - then `timeline.md`
   - then a few relevant `notes/*.md`
7. Use `search_literature` only after those notes stop being sufficient
8. Choose exactly one family and explain why it fits better than the obvious alternatives

## Data checks that should usually happen
- compute or estimate cell count variation over time
- check whether mass variance is high enough to prefer unbalanced models
- check whether the system suggests simple deterministic transport or noisy/branching dynamics
- check whether the task actually requires interaction-aware modeling
- verify whether spatial or interaction evidence is real rather than assumed

## Ambiguity handling
- Ask the user for clarification when:
  - evidence is contradictory
  - the user request conflicts with the data properties
  - a modeling choice would be dominated by a constraint not stated clearly
- Ask directly instead of guessing when the wrong family choice would distort the downstream biological claim.

## What a good decision looks like
A good theory-selection output should include:
- chosen model family
- concise rationale tied to data properties
- any critical config overrides
- explicit mention of why balanced vs unbalanced and deterministic vs stochastic were chosen

## Common failure modes
- Choosing a stochastic family just because it is more expressive
- Choosing interaction-aware models without evidence that interaction matters
- Ignoring strong mass imbalance across time
- Returning many vague candidates instead of one decision
- Failing to justify why the chosen family fits the biological system
- Returning a string like `unbalanced_ot.default` instead of a structured candidate object

## When to read more source/docs
Read these when the choice is uncertain:
- `references/package-capability-map.md`
- `CytoBridge-main/docs/INDEX.md`
- `CytoBridge-main/docs/theory/README.md`
- `CytoBridge-main/docs/runtime/flow-matching/README.md`
- `CytoBridge-main/CytoBridge/configs/*.yaml`

Only inspect source after these documents are insufficient and the task has
already crossed from theory selection into authoring or implementation.

## Completion
Call:

`commit_workflow_state(phase="theory_selection", ...)`

Include:
- `plan_decision`
- `candidates`
- a summary of why the chosen family fits the dataset
- any key overrides or user-imposed constraints
