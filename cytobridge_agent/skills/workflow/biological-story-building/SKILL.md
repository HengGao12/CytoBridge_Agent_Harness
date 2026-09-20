---
name: biological-story-building
description: Build a manuscript-oriented biological story from a completed CytoBridge algorithm lifecycle before choosing downstream analyses and figures.
---

# Biological Story Building

Use this skill after an algorithm has metric-level lifecycle evidence and before
writing a biological-application paper or choosing downstream analyses for that
paper.

This skill does **not** teach package APIs and does not add a new gate. Its job
is to decide what downstream evidence is needed so one real dataset can support
a coherent biological story.

## Core Principle

Final-regression or benchmark success is metric-level validation, not a paper
story. A strong first-pass autonomous paper may focus on one real dataset, but
it must explain what the method reveals in that dataset that would otherwise be
hard to see.

The question is not "which downstream modules can I run?" The question is:

> What biological or dynamical phenomenon does this algorithm make visible on
> this dataset, and what model-derived evidence is needed to make that claim
> convincing?

## Story Before Tools

Before reading narrow downstream skills, write a short working story in notes or
the paper editorial brief:

- **Dataset object:** tissue/system, time axis, labels, perturbation/condition,
  lineage, spatial, or morphology structure available in this dataset.
- **Algorithm advantage:** the specific capability that passed metric-level
  validation, such as trajectory timing, fate separation, mass/growth, spatial
  transport, perturbation response, stochastic uncertainty, or gene-space
  dynamics.
- **Biological story candidate:** one concrete phenomenon to explain deeply in
  this dataset, not a survey of every possible result.
- **Evidence gap:** what downstream evidence is still needed to turn metrics
  into biology.
- **Figure logic:** which 3-5 figure panels would let a reader follow the
  story without seeing internal workflow artifacts.

Do not start manuscript prose from a metric table alone.

## Choosing Downstream Analyses

Select downstream skills because they serve the story:

- trajectory, branch timing, or fate commitment -> read
  `downstream-trajectory-fate`;
- terminal basins, stable states, or route diversity -> read
  `downstream-state-structure`;
- growth, depletion, or mass redistribution -> read `downstream-growth-mass`;
- gene programs, regulators, modules, or pathways -> read
  `downstream-driver-genes`;
- condition response, KO/OE, or counterfactual effect -> read
  `downstream-perturbation`;
- source-target or clone/fate transition evidence -> read
  `downstream-lineage-transition`;
- local flow-field visualization -> read `downstream-scvelo-stream`, but pair
  it with rollout-derived evidence before making trajectory claims.

Run enough downstream analysis to answer the chosen story, not a checklist of
all possible modules.

## What Counts As Deep Enough

For a one-dataset biological paper, aim for at least three connected evidence
layers when the data supports them:

- model-derived dynamics: generated trajectories, fate timing, growth/mass,
  perturbation response, or endpoint structure;
- biological annotation: cell states, time points, regions, conditions,
  lineages, or marker-defined populations;
- mechanistic readout: genes, modules, pathways, regulators, spatial
  neighborhoods, or perturbation-sensitive programs.

If gene-space projection or labels are missing, say so and narrow the claim
rather than inventing mechanism.

## Figure Expectations

Every main figure panel should answer a reader-facing question:

- What is the biological system and transition?
- Where does the method improve the dynamic interpretation?
- What trajectory, fate, growth, perturbation, or state pattern does it reveal?
- Which gene/module/pathway/state evidence makes the result biologically
  meaningful?
- Which baseline or ablation shows this is not a generic plotting artifact?

After downstream artifacts exist, read
`workflow/scientific-visualization/SKILL.md` before final rendering. Use it to
make figures manuscript-grade; do not use it to change analysis outputs.

## Writing Handoff

After downstream analysis, hand the paper authoring skill:

- the one-sentence story;
- the supported claim boundaries;
- downstream manifest paths, scripts, tables, figures, and warnings;
- figure candidates and the question each panel answers;
- missing evidence that should stay in Discussion or supplement.

The final manuscript should read as a positive scientific argument, not as a
workflow report.
