---
title: "Builtin flow-matching implementation guide"
summary: "Source map for reading the closest builtin before implementing a custom flow-matching algorithm."
read_when:
  - "Implementing a custom flow-matching algorithm"
  - "Choosing which builtin source path to reuse or subclass"
  - "Checking whether a custom implementation accidentally re-solves coupling in the epoch loop"
---
# Builtin Flow-Matching Implementation Guide

Use this page before editing a custom `algorithm.py`. It is not a theory
document. It tells an agent which builtin source path to read so the custom
algorithm is implemented by modifying the right package layer rather than by
rebuilding the runtime from scratch.

## Required Builtin Source Pass

Before writing code for a custom flow-matching algorithm:

1. Pick the closest builtin anchor.
2. Read that builtin YAML under `CytoBridge-main/CytoBridge/configs/`.
3. Read the corresponding builder and strategy classes in
   `CytoBridge-main/CytoBridge/tl/flow_matching_backends.py`.
4. Read the default call graph in `semantics-and-callgraph.md` enough to know
   when coupling is precomputed and when minibatches are sampled.
5. Record the source pass in `IMPLEMENTATION_MAP.md`: closest builtin, files
   read, exact symbols reused/subclassed/replaced, and why each changed module
   is necessary.

Do not skip this pass because the proposal is clear. Many implementation bugs
come from putting proposal semantics in the wrong layer, for example solving
OT/UOT/WFR inside every epoch instead of precomputing coupling state once before
flow-matching training.

## Builtin Anchors

| Builtin | Use as anchor when the custom method changes | YAML | Main source symbols to read |
| --- | --- | --- | --- |
| `balanced_ot_cfm` | balanced deterministic OT-CFM coupling or Euclidean pair cost | `configs/balanced_ot_cfm.yaml` | `_build_balanced_coupling`, `BalancedOTCouplingStrategy`, `_solve_balanced_ot_from_pairwise_cost`, `LinearDeterministicConditionalPath`, `NullMassStrategy` |
| `sf2m` | balanced OT plus stochastic bridge / score matching | `configs/sf2m.yaml` | `_build_sf2m_coupling`, `BalancedOTCouplingStrategy`, `SchrodingerBridgeConditionalPath`, score-head training path in `trainer.py` |
| `vgfm` | unbalanced deterministic velocity-growth flow matching | `configs/vgfm.yaml` | `_build_unbalanced_coupling`, `UnbalancedOTCouplingStrategy`, `_solve_uot_from_pairwise_cost`, `RegularizedUnbalancedConditionalPath`, `UOTMassStrategy` |
| `wfrfm` | WFR/OET-style unbalanced dynamic problem | `configs/wfrfm.yaml` | `_build_wfr_coupling`, `WFROETCouplingStrategy`, `WFROETMassStrategy`, `WFRGeodesicConditionalPath` / WFR path classes, `_split_transport_chunks` |
| `crufm` | UOT velocity-growth plus stochastic bridge / score matching | `configs/crufm.yaml` | `_build_unbalanced_coupling`, `UnbalancedOTCouplingStrategy`, `RegularizedUnbalancedConditionalPath`, score-head training path in `trainer.py` |

If no builtin is a close semantic match, still choose the nearest engineering
anchor and state what must change. A custom method should normally change one
or two of these modules first:

- coupling/cost;
- conditional path;
- mass strategy;
- model heads;
- additive loss;
- inference rollout;
- additive metrics.

Replacing the whole stage runner is a last resort.

## Module Responsibilities

### Coupling Strategy

Coupling chooses adjacent-time endpoint pairs and, for unbalanced methods,
terminal mass targets. Builtin flow-matching computes coupling state before the
epoch loop. The epoch loop should sample from that state.

Read:

- `CouplingStrategy.build_state(...)`
- `PairwiseOTCouplingStrategy.compute_ot_coupling(...)`
- `CouplingStrategy.sample_pairs(...)`
- `_sample_pair_batch_from_state(...)`

Implementation rule:

- If the proposal changes only pairwise geometry, prefer subclassing
  `CostBasedPairwiseOTCouplingStrategy` and overriding
  `build_pairwise_cost(...)` when the adjacent time-pair cost is small enough
  to fit comfortably. About 1000 source cells by 1000 target cells yields a
  `1000 x 1000` cost/coupling matrix, which is normally acceptable as one full
  block and does not require extra mini-batch machinery.
- If the proposal changes pairwise geometry but the full adjacent-pair cost is
  too large, prefer subclassing `ChunkedTransportCouplingStrategy` and
  overriding `build_pairwise_cost_block(...)`. The package then handles chunk
  splitting, per-block OT/UOT solving, cached subplans, and pair sampling.
- Write custom `build_state(...)` / `sample_pairs(...)` only when the package
  chunked-cost API cannot express the algorithm's state.
- Do not solve OT/UOT/WFR inside `flow_matching_loss_hook(...)` or inside each
  training epoch unless the approved proposal explicitly defines an online
  coupling algorithm.

### Solver And Chunking

Builtin chunking is not only a flag. The relevant source functions are:

- `_split_transport_chunks(...)`
- `_solve_uot_from_pairwise_cost(...)`
- `_solve_balanced_ot_from_pairwise_cost(...)`
- `ChunkedTransportCouplingStrategy`
- `WFROETCouplingStrategy._solve_gamma(...)`

Read these before implementing custom chunked OT/UOT/WFR. The expected pattern
is:

- split each adjacent time gap into bounded source/target chunks;
- build cost only for each chunk when possible;
- solve each chunk on GPU when POT supports torch CUDA tensors;
- store sampler-compatible subplans or sparse metadata;
- sample minibatches from stored state during training.

A `chunk_size` parameter is not sufficient if the implementation first builds a
full dense cost, mask, kernel, or plan for a large adjacent pair.

For the common case “same OT/UOT solver, new pairwise geometry”, do not re-write
the sampler. Subclass `ChunkedTransportCouplingStrategy`, compute one block
cost on the target device, and let the package cache the transport subplans in
`build_state(...)`.

### Conditional Path

The conditional path converts sampled endpoints and terminal mass into training
targets such as `ut`, `gt`, score targets, path weights, and noisy path points.

Read:

- `LinearDeterministicConditionalPath`
- `RegularizedUnbalancedConditionalPath`
- `SchrodingerBridgeConditionalPath`
- WFR path classes used by `wfrfm`

Implementation rule:

- straight-line path is the simplest default;
- analytic bridge/WFR paths are preferred when the proposal derives them;
- learned/neural paths need enough time or interval context to distinguish
  different adjacent biological gaps;
- path training must not become a shortcut predictor for metrics.

### Mass Strategy

Mass strategy defines terminal mass semantics, not a post-hoc correction.

Read:

- `NullMassStrategy`
- `UOTMassStrategy`
- WFR mass strategy used by `wfrfm`

Implementation rule:

- if the algorithm claims unbalanced modeling, mass should come from growth
  dynamics or another biologically/theoretically meaningful mechanism;
- do not rescale predicted weights after rollout only to satisfy TMV;
- if the algorithm is balanced-only, state that TMV is diagnostic rather than a
  gate.

### Model And Trainable Heads

Use the package trainable-head contract instead of hiding custom heads inside
unrelated modules.

Preferred patterns:

- `model.cytobridge_component_modules = {"growth": ["birth_head", ...]}`
- stage `trainable_modules: ["birth_head", ...]`
- advanced `cytobridge_trainable_parameters(stage_params=..., train_flags=...)`

If a new head is not receiving gradients, fix the trainable-head declaration
instead of moving the logic into an unrelated builtin head.

## What To Put In `IMPLEMENTATION_MAP.md`

Before preview or campaign evidence, add:

```markdown
## Baseline Anchor

- Required anchor baseline: vgfm
- Why this is the nearest builtin:
- Components reused or intentionally changed:

## Builtin Source Pass

- Closest builtin anchor:
- YAML read:
- Source symbols read:
- Reused unchanged:
- Subclassed/overridden:
- Replaced:
- Reason each replacement is necessary:

## Runtime Hot Path Check

- Coupling precomputed before epoch loop: yes/no
- OT/UOT/WFR solved inside epoch loop: yes/no
- Largest dense object expected on real benchmark:
- Chunk/sparse/streaming state used:
- GPU solver path used:
- Natural degeneration/ablation against builtin:
```

This is not paperwork. If the table reveals that the implementation cannot
reuse the builtin path without changing algorithm semantics, revise the proposal
or implement the missing module directly.

`Required anchor baseline` is also a campaign contract. When
`refresh_campaign_stage_baselines(...)` runs builtin baselines, this anchor is
included even if the agent supplies a shorter explicit baseline list. Other
baselines can still use the default panel or explicit user choices.
