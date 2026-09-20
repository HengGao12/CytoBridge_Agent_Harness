---
title: "Custom algorithm flow"
summary: "High-level custom algorithm lifecycle and runtime integration flow."
read_when:
  - "Understanding custom algorithm lifecycle flow"
  - "Connecting proposal, workspace, and training stages"
---
# Custom Algorithm Flow

High-level lifecycle for proposal, workspace initialization, implementation, review, and training.

## 0. Custom Algorithm Workflow

For custom algorithm work, the intended order is:

1. proposal theory gate
2. proposal with pseudocode and evaluation plan
3. approval
4. workspace initialization
5. implementation
6. review with pseudocode-to-code line mapping
7. training and evaluation

Do not jump directly from “new algorithm idea” to editing `algorithm.py`.

Proposal-stage minimum requirements:

- state the recoverability argument
- state which runtime axes change:
  - `model.components`
  - `train_strategy`
  - `MassStrategy`
  - `conditional path`
- provide implementation-oriented paper-style pseudocode:
  - include `Inputs` and `Outputs`
  - use labeled steps `P1`, `P2`, ...
  - include the key objective / path / update equations inside the algorithm
    description, not only in surrounding prose
- provide an evaluation plan:
  - builtin `W1` and `TMV` stay fixed
  - any new metric must be additive and tied to the scientific objective

Review-stage minimum requirements:

- re-read proposal, config, and implementation
- choose and verify the actual runtime hooks during authoring/review
- fill the pseudocode-to-code mapping with concrete files, hooks, classes, or
  functions; exact line numbers are useful but optional
- only then run training
