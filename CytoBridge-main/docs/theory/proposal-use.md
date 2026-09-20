---
title: "Using builtin theory in proposals"
summary: "Guidance for citing builtin algorithm theory in custom proposals."
read_when:
  - "Writing or reviewing algorithm proposal theory sections"
---
# Builtin Algorithm Theory: Proposal Usage

How proposal authors and evaluators should use builtin theory references.

## How To Use This In Algorithm Proposals

Every new algorithm proposal must explicitly say which builtin it is closest to
and what it changes:

- coupling;
- path;
- mass strategy;
- loss;
- simulation / inference;
- evaluation metric.

If the method is balanced-only:

- write the dynamic problem it solves;
- prove distribution fit;
- state that mass / cell-count change is not modeled;
- state that `TMV` is diagnostic only.

If the method is unbalanced:

- write the positive-measure dynamic problem or endpoint UOT problem;
- prove distribution fit;
- prove weighted-particle mass fit;
- explain how terminal mass or growth targets recover observed total-mass
  changes.

If the method has no named global energy functional:

- say that explicitly;
- define the package-level interval problem instead;
- still prove endpoint distribution and, if applicable, mass recovery.

## Source Pointers

- Configs: `CytoBridge/configs/*.yaml`
- Flow matching runtime: `CytoBridge/tl/flow_matching_backends.py`
- Base path formulas: `CytoBridge/tl/flow_matching.py`
- Training losses and evaluation: `CytoBridge/tl/trainer.py`
