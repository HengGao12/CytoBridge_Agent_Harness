---
title: "Downstream checklist and common failures"
summary: "Pre-report checklist and failure-mode guide for downstream CytoBridge analysis."
read_when:
  - "Reviewing downstream outputs before report or paper writing"
  - "Debugging weak, contradictory, or suspicious downstream results"
---
# Downstream Checklist And Common Failures

Use this page before writing report/paper claims and when downstream artifacts
look inconsistent.

## Pre-Analysis Checklist

1. Active model path is known.
2. Active `adata` path is known.
3. `time_point_processed` exists.
4. `X_latent` exists.
5. Output directory is fixed.
6. Script path under `<output_dir>/scripts/downstream/` is planned.
7. Feature space is explicit: latent, gene, projected gene, or mixed.
8. Label/readout contract is explicit for fate, lineage, or perturbation.
9. External evaluator file schemas, if any, are handled outside generic
   downstream analysis.
10. If an API input/output shape is unclear, inspect the source path listed in
    `api-reference.md` before running or interpreting the analysis.
11. Each script records the package source path for every downstream API it
    calls.
12. Each manifest records required input fields, expected input shapes, returned
    shape fields, warnings, and generated artifacts.

## Pre-Report Checklist

1. Every important analysis has a saved script.
2. Every script writes a manifest.
3. Every figure/table path exists.
4. Continuous-time claims use model-native trajectory/evaluation artifacts.
5. Fate/lineage claims cite a valid readout contract.
6. Growth/mass claims distinguish growth rate from empirical counts.
7. Gene-level claims cite a projection or measured gene-space computation.
8. Limitations and warnings from manifests are reflected in the narrative.
9. Any array slicing or axis interpretation is backed by returned shape fields
   or source inspection, not by artifact filename guesses.

## Common Failure Modes

- Writing downstream narrative before generating artifacts.
- Reading every downstream skill and mixing incompatible rules.
- Treating a stream plot as a simulated trajectory.
- Treating saved per-cell velocity vectors as continuous paths.
- Claiming fate truth from generic cell-type labels without a lineage/fate
  contract.
- Claiming gene-level drivers from latent-only outputs.
- Mixing raw and processed time axes.
- Using validation/held-out snapshots as predictions.
- Treating empirical cell counts as model growth.
- Running perturbation as static expression contrast instead of matched rollout.
- Overwriting old downstream artifacts without recording the refresh.
- Letting external evaluator file contracts leak into general biological
  analysis.
- Guessing array axis order instead of using payload shape fields or checking
  the source.
- Omitting source paths and shape contracts from saved downstream scripts, which
  makes later biological interpretation non-auditable.
- Treating a projected gene-space result as measured gene-space evidence
  without naming the PCA projection.

## Debugging Order

When downstream results look wrong:

1. Check data contract: time, latent, labels, projection.
2. Check model binding: final regression versus old campaign/diagnostic run.
3. Check feature-space and time-axis mapping.
4. Regenerate a small trajectory artifact with explicit parameters.
5. Compare observed structure and rollout structure.
6. Inspect warnings in manifests and package return payloads.
7. Inspect the relevant `CytoBridge-main/CytoBridge/tl/downstream/*.py` or
   `CytoBridge-main/CytoBridge/pl/downstream/*.py` source file if the shape,
   source population, projection, or side effect is still unclear.
8. Only then revise the biological interpretation.

Do not patch the narrative to fit bad artifacts. Fix or label the analysis.
