# Journal Figure Taste

Use this note when the user wants figures with a stronger "big-paper" feel rather than just
technically correct plots.

## Core Principle

Big-paper figures usually feel strong because they are **hierarchical and restrained**, not
because they are visually busy.

The main figure should answer one question cleanly.

If there is tension between:
- looking fancy
- and looking clean, orderly, and obvious

choose clean, orderly, and obvious.

## Figure-Level Rules

### 1. One figure, one sentence

Before drawing, write the figure message as a single sentence.

If a figure needs two unrelated verbs, it is probably two figures.

### 2. Build around a hero panel

Each main figure should have one panel that the eye lands on first:
- a summary benchmark
- a representative embedding
- a decisive mechanism panel

Other panels should support that panel, not compete with it.

### 3. Use a stable panel grammar

For method/biology papers, a strong default pattern is:
- small setup schematic
- quantitative comparison block
- representative examples
- one mechanism or interpretation block

Do not distribute visual weight evenly across eight unrelated panels.

### 4. Main figures are selective

Main figures should show:
- the most representative benchmark
- one or two canonical examples
- the minimum support needed to trust the claim

Move exhaustive sweeps, secondary examples, and robustness details to supplement.

### 5. Let the method win visually without cheating

Use:
- gray or desaturated colors for baselines
- one consistent accent color for the main method
- one secondary accent for a key comparator if needed

This keeps comparisons fair while preserving visual hierarchy.

### 6. Favor white space and blocks

High-density figures should still read as separate blocks:
- benchmark block
- embedding block
- heatmap block
- perturbation block

Use alignment and white space to separate blocks. Avoid continuous clutter.

White space is not the same as emptiness.

If a panel occupies meaningful space but contains only a few marks:
- enrich the representation
- combine it with a closely related readout
- or shrink/demote it

Do not let a thin panel sit in a large slot just because the grid expects it.

Likewise, do not let a needlessly ornate panel stay in a prime slot if a cleaner plot would
say the same thing more directly.

### 6c. Default page posture: vertical A4, white, editorial

For manuscript main figures, default to:
- white background
- vertical A4-like page ratio
- an editorial page posture rather than a dashboard posture

This usually makes it easier to:
- create a strong top-to-bottom story flow
- establish one mid-page hero block
- keep support panels subordinate without making the figure feel sparse

Landscape should be a deliberate exception, not the default.

### 6b. Keep prose out of the canvas

Main figures should not default to caption-like sentences printed on the figure.

Prefer:
- short block titles
- route names
- module names
- concise axis labels

Avoid:
- explanatory subtitle sentences
- interpretation text floating above panels
- using prose to compensate for weak layout
- repeatedly adding descriptive sentences because the panel is still weak; that usually means the visual encoding should change instead

### 7. Reuse the same visual language across the paper

Keep these consistent across all figures:
- same method colors
- same baseline styling
- same embedding style
- same heatmap style
- same typography scale
- same panel-label placement

This makes the paper feel cohesive and expensive.

### 7b. Final manuscript figures must be provenance-clean

For paper-ready figures:
- do not assemble the final deliverable from old rendered PNG panels
- do not let a good-looking preview become the final source of truth

Allowed:
- contact sheets or collage previews for discussion

Required for approved figures:
- a script-generated final panel
- traceable inputs from raw tables, embeddings, ckpts, or saved analysis outputs
- editable intermediate components when the figure has multiple layers

If the figure cannot be reproduced from traceable inputs, it is not final, no matter how good it looks.

### 8. Use the most concrete biological object available

When a story is about:
- route separation, show routes
- founder fate bias, show founder fate domains
- commit timing, show commit-window state
- a published case study, show the actual case if it is visually strong

Avoid replacing the real object with an overly abstract proxy if the proxy is harder to read.

### 9. Prefer compact rich panels over sparse generic panels

For mechanism-heavy blocks, prefer:
- route-specific small multiples
- mini-heatmaps
- dot matrices
- compact gene/module tiles

If a panel still looks too generic or "ordinary":
- replace a plain heatmap with a more editorial encoding such as:
  - bracketed module tiles
  - aligned dot cards
  - ribbon summaries
  - stacked rectangular glyph panels
  - small-multiple tiles with one shared visual grammar

The goal is not decoration. The goal is to make the panel feel deliberate, selective, and
hierarchical.

If a richer encoding repeatedly fails:
- revert to a simpler grammar
- keep it crisp
- and make it look expensive through alignment, spacing, and typography instead of novelty

Be cautious with:
- tiny three-bar summaries
- decorative pies
- panels that require caption prose to explain what is being compared
- over-abstract mechanism encodings that the user immediately reads as confusing or contrived

### 10. Simplicity beats strained cleverness

Acceptable high-end main-figure mechanism panels include:
- violin summaries
- box / boxen summaries
- grouped bars when directionality is the only real message
- compact dose-response plots
- small multiples with one repeated grammar

Do not force:
- phase-map abstractions
- pseudo-volcanoes
- schematic-looking summary panels

unless they are clearly more interpretable than the simple alternative.

### 11. Alignment is part of taste

Figures feel expensive when:
- panel letters line up
- rows have clear outer anchors
- peer panels share comparable widths and heights
- lower-row subplots respect the column geometry set by upper rows

Do not assume `subplot` defaults are good enough for manuscript assembly.
If alignment matters, position axes explicitly.

## What to Avoid

- Making every panel colorful
- Giving baselines the same visual weight as the main method
- Putting every dataset and every ablation into the main figure
- Using captions to rescue unclear layouts
- Mixing three different figure styles across the paper
- Forcing a panel to stay in the main figure after repeated failed redesigns

## Practical Checklist

Before finalizing a main figure, check:

- Can I describe the figure in one sentence?
- Is there an obvious first panel to look at?
- Is the main method visually easiest to track?
- Are support panels clearly subordinate?
- Could at least one panel be moved to supplement without hurting the claim?
- Does this figure look like it belongs to the same paper as the others?

## Best Fit Use Cases

This reference is most useful when:
- planning Nature/Science/Cell/Methods-style main figures
- converting many analyses into a concise primary figure set
- deciding what to demote to supplement
- improving "taste" after the raw plots already exist
