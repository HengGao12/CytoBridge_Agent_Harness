# Pretraining Checklist

Use this reference before trusted custom `run_training(...)` or
`run_campaign_trial(...)`.

## Binding Check

Review is invalid unless it is bound to:

- active algorithm id
- exact approved `proposal_id`
- active workspace snapshot or explicit campaign trial snapshot

Inspect:

- `get_current_workflow_context()`
- `get_algorithm_campaign_status(...)` for campaign runs
- `list_experiment_history(...)` only as read-only history

Context split:

- active context is the algorithm being edited/reviewed
- `run_training(training_algorithm_id="...")` is the explicit training target
- training another algorithm does not switch active editing context
- campaign trial snapshots are created by `run_campaign_trial(...)`
- manual training needs a clean target snapshot if the workspace is dirty

If proposal/workspace bindings are stale or unclear, stop and fix the binding
before training.

## Required Files To Re-Read

Do not rely on memory. Re-read:

- `~/.cellcompass/training_algorithms/<algorithm_id>/PROPOSAL.md`
- `~/.cellcompass/training_algorithms/<algorithm_id>/manifest.yaml`
- `~/.cellcompass/training_algorithms/<algorithm_id>/algorithm.py`
- `~/.cellcompass/training_algorithms/<algorithm_id>/config.yaml`
- `~/.cellcompass/training_algorithms/<algorithm_id>/IMPLEMENTATION_MAP.md`
- `~/.cellcompass/training_algorithms/<algorithm_id>/risk.md`

If the proposal links `primary_idea_id`, also re-read the linked research idea.

## Proposal-To-Code Consistency

Check that the code still implements:

- the same scientific problem
- balanced/unbalanced choice
- deterministic/stochastic choice
- proposal-required coupling, constraints, losses, mass/growth mechanism,
  conditioning variables, stochastic components, and inference path
- any allowed approximation only as semantics-preserving mini-batch,
  stochastic-estimator, vectorization, caching, streaming, or equivalent
  reparameterization

Do not silently implement a simpler method. If implementation reveals that the
approved proposal is wrong, internally inconsistent, or infeasible under the
intended evidence protocol, patch `PROPOSAL.md` with `apply_workspace_patch(...)`
and wait for the new proposal review before using training evidence.

If older evidence is invalidated by a semantic mismatch:

- record the decision
- mark old results obsolete
- update the linked idea progress when relevant

## Implementation Map Gate

`IMPLEMENTATION_MAP.md` is mandatory before training.

Required columns:

- `Step ID`
- `Pseudocode Step`
- `Implementation File`
- `Status`
- `Semantic Check`
- `Deviation Status`
- `Notes`

Optional column:

- `Line Number(s)`

Completion rules:

- one row per approved pseudocode step
- implementation locations should be concrete enough to audit, but exact
  1-based line numbers are optional
- line references may be written as `algorithm.py:123-150` or as
  `Implementation File = algorithm.py` plus `Line Number(s) = L123-L150` when
  useful
- if a row delegates to package default behavior, cite the package
  class/function and the local line where the hook/config chooses that default
- state whether each step is fully implemented, partial, or delegated to package
  default behavior
- explain which proposal invariant is preserved
- classify deviation as `exact`, `acceptable approximation`, or `semantic drift`
- if any row is unmapped, still TODO/pending, or marked `semantic drift`, do not
  train as trusted evidence
- if any proposal-required row is marked `deferred`, `Stage 2 refinement`, or
  "implemented later", do not train as trusted evidence; that is a prototype,
  not the approved algorithm
- if an approximation is used, explain why it preserves the same mathematical
  problem rather than merely making implementation easier

The automatic `implementation_evaluator` checks this file against the approved
proposal and cited code. It can reject a structurally filled but inaccurate map.
