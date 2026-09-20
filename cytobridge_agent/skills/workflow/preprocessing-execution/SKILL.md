---
name: preprocessing-execution
description: Prepare a temporal single-cell dataset for CytoBridge by establishing a biologically valid time axis, a defensible X_latent, and a minimal safe preprocessing path.
---

# Preprocessing Execution

Use this skill when the task is to inspect, clean, normalize, or minimally prepare a single-cell temporal snapshot dataset before theory selection or training.

## Goal
Produce a usable AnnData object for CytoBridge. The result must preserve biological meaning, avoid unnecessary reprocessing, and satisfy the minimum training contract.

## Read first when unsure
- `CytoBridge-main/docs/INDEX.md`
- package/source files relevant to the current preprocessing need

## Runtime assumptions
- The active dataset is managed through the shared runtime AnnData manager.
- Reuse the current in-memory `adata` whenever possible.
- Do not repeatedly reload `.h5ad` unless:
  - the active dataset/path has explicitly changed
  - rollback is required after a failed overwrite or suspected corruption
  - `adata` is currently missing and you need to bootstrap from raw files
- Treat committed workflow state and explicit user/system path or key constraints as authoritative when they are already available.

## Core contract
CytoBridge expects the runtime AnnData to satisfy these conditions before training:

1. `adata.obs["time_point_processed"]` exists
   - integer sequence starting from 0
   - represents biologically meaningful temporal order
   - do not invent a fake trajectory from arbitrary batch labels

2. `adata.obsm["X_latent"]` exists
   - `X_latent` should be a representation that can reasonably reconstruct or map back to the original gene-feature space, because downstream interpretation ultimately lives at the gene level
   - if you choose a compressed representation, prefer one with an explicit or checkable reconstruction path and document that choice
   - when feasible, check reconstruction quality rather than assuming the embedding is acceptable
   - if the original data matrix is already low-dimensional (roughly <100 features) and still biologically meaningful in the original feature space, it is acceptable to use `adata.X` directly as `X_latent`
   - do not choose an embedding only because it is convenient if it destroys the link back to gene-space interpretation

3. gene names are usable
   - `adata.var_names` should be meaningful feature names when possible
   - if only IDs are available, note that limitation in the summary

4. if you run PCA, keep:
   - `adata.obsm["X_pca"]`
   - `adata.varm["PCs"]`

5. preprocessing provenance is script-backed
   - every new dataset preprocessing path must be reproducible from a saved Python script, not only from transient `execute_python` snippets
   - save the stable script under `<output_dir>/scripts/preprocessing/*.py` before marking preprocessing complete
   - rerun the saved script with `run_saved_python_script(...)` or explicitly validate that it reproduces the active `.h5ad`
   - register the script path and key inputs/outputs in workflow state so later stages can inspect exactly how the dataset was prepared

## Required tool behavior
- Prefer code-first execution with:
  - `inspect_adata_state`
  - `execute_python`
  - `run_saved_python_script`
  - `list_path`
  - `read_file`
  - `persist_runtime_adata`
- Use `list_path` and `read_file` when you need to inspect raw files, README notes, metadata tables, or directory structure before loading/building `adata`.
- Modify `adata` in place with `execute_python`.
- Use `execute_python` for exploratory preprocessing logic only. Before completing preprocessing for any new dataset, convert the final routine into a self-contained script under `<output_dir>/scripts/preprocessing/*.py` and rerun it with `run_saved_python_script(...)` or validate that it exactly reproduces the persisted output. Saved scripts must not depend on transient exploratory variables from `execute_python`.
- Do not write custom completion tags or ad hoc phase markers.
- When preprocessing is complete, commit state through:
  - `commit_workflow_state(phase="preprocessing", ...)`

## Operating rules
- Be conservative. If the dataset is already prepared, do not redo normalization/HVG/PCA just because you can.
- Do not apply `normalize_total`, `log1p`, HVG selection, or PCA mechanically. Decide based on what the matrix currently represents and what preprocessing would preserve biological meaning.
- Use biological and measurement knowledge to judge whether preprocessing is needed at all. Raw counts, already-normalized expression, logged expression, CLR-like transforms, and low-dimensional assay outputs should not be treated as interchangeable.
- Do not downsample by default. Observed cell counts across time points can be a real biological signal of proliferation, death, sampling efficiency, or total-mass change; using more valid cells is usually better for training and downstream evidence.
- Distinguish biological subsetting from statistical downsampling. A user-specified arm, tissue, condition, perturbation, or time course is a binding biological scope; random cell reduction for speed is an engineering approximation and must be justified separately.
- If a smaller smoke/debug dataset is unavoidable, preserve the original time-point cell-count proportions by applying the same sampling ratio within each time point. Do not sample a fixed absolute number from each time point unless the analysis explicitly does not depend on cell-count or mass information and you record that limitation.
- Keep the full prepared real dataset as the authoritative artifact for biological campaign evidence whenever computationally feasible; smaller smoke/debug outputs should be labeled as such and should not replace full-data Stage 1/2/3/final evidence without larger/full-data validation.
- If instructions are ambiguous but the data clearly reveals the correct `time_key` or `label_key`, proceed and document the choice.
- If instructions are ambiguous and the data does not support a confident choice, ask the user directly rather than guessing.
- Do not fabricate a time axis or a label column to satisfy the contract.

## Recommended workflow
1. If `adata` is missing, bootstrap it from raw files first
   - inspect the directory
   - identify the raw format
   - build a valid `AnnData`
2. Inspect the current object
   - shape
   - `obs` columns
   - `obsm` keys
   - `layers`
   - `uns`
   - whether `X_latent`, PCA, and UMAP already exist
3. Determine the biological time axis
   - find the most plausible time/stage column
   - verify ordering is meaningful
   - map it to `time_point_processed`
4. Determine the label column
   - choose the most interpretable cell identity column available
   - if there is no trustworthy label column, note that clearly rather than fabricating one
5. Decide whether preprocessing is needed
   - inspect what `adata.X` and relevant layers already represent before changing anything
   - if the matrix is raw counts and standard preprocessing is appropriate, run only the necessary steps
   - if the matrix is already normalized/logged/embedded or otherwise prepared, avoid destructive modifications
   - if you are unsure whether a transformation is appropriate, prefer a minimal reversible path and document the uncertainty
   - avoid random downsampling; if it is unavoidable for a smoke/debug run, use per-time-point ratio sampling and keep the full prepared dataset as the authoritative artifact
6. Ensure `X_latent`
   - prefer a representation that preserves a usable link back to the original gene space
   - if the original feature dimension is already small enough and biologically meaningful, consider using `adata.X` directly
   - if PCA or another embedding is used, explain why it is appropriate and, when feasible, check reconstruction quality
7. Check for save-time hazards before persisting
   - conflicting index names in `adata.obs` or `adata.var`
   - NaN/Inf values
   - empty cells or zero-feature objects
8. Persist the runtime AnnData if a new stable checkpoint should be saved
9. Save and validate the preprocessing script under `<output_dir>/scripts/preprocessing/`
10. Commit preprocessing outputs and script provenance into workflow state

## Typical checks
- confirm the active data path and current `adata` shape
- inspect `obs` for candidate time and label columns
- inspect what `adata.X` and major layers actually mean:
  - raw counts
  - normalized counts
  - logged expression
  - already-embedded features
- verify whether the object already has:
  - `time_point_processed`
  - `X_latent`
  - PCA
  - UMAP
- if you plan to derive `X_latent` from a compressed representation, check whether reconstruction back to gene-space is acceptable for downstream use
- check whether gene names are readable
- check for obviously broken values:
  - empty cells
  - NaN/Inf
  - zero-feature objects

## Troubleshooting
- If `adata.obs` or `adata.var` has an index name that conflicts with a column name, clear the conflicting index name before saving.
- Filter empty cells before normalization when raw data contains zero-count observations.
- Clean NaN/Inf values before saving or running PCA.
- If gene symbols are missing, inspect alternative columns in `adata.var` and document any limitation if replacement is impossible.
- If preprocessing semantics are unclear, first determine whether the matrix is already normalized or transformed before applying `normalize_total` or `log1p`.
- If an embedding gives poor reconstruction or breaks gene-level interpretability, do not use it as `X_latent` just to satisfy the contract.

## Ambiguity handling
- Ask the user for clarification when:
  - the dataset is empty or malformed
  - multiple time columns are plausible and the choice changes biological meaning
  - the label choice is materially ambiguous
  - the raw files are not self-describing enough to load confidently
- Prefer explicit clarification over silent guessing when a wrong preprocessing choice would invalidate training.

## Common failure modes
- Treating batch as time
- Reprocessing an already-trained or already-embedded object
- Blindly applying `normalize_total` / `log1p` to a matrix whose semantics were not checked
- Forgetting to create `time_point_processed`
- Forgetting `X_latent`
- Choosing an `X_latent` representation that cannot reasonably support downstream gene-level interpretation
- Overwriting a good AnnData with an incompatible save path
- Guessing the label column when several similar columns exist and the choice matters biologically

## When to read more source/docs
Read these when the interface or contract is uncertain:
- `CytoBridge-main/docs/INDEX.md`
- `CytoBridge-main/CytoBridge/tl/fit.py`
- any package code that consumes `time_point_processed` or `X_latent`

## Completion
After producing a usable dataset, call:

`commit_workflow_state(phase="preprocessing", ...)`

Include at least:
- `preprocessed_path`
- `preprocessing_script_path`
- `preprocessing_script_sha256` when feasible
- `preprocessing_input_paths`
- `preprocessing_output_paths`
- `time_key`
- `label_key`
- what `adata.X` represented before preprocessing
- why the chosen `X_latent` is appropriate
- whether reconstruction quality was checked and what the result was
- any key preprocessing choices
- a concise summary of what changed and what was reused

## Benchmark Registration Handoff

If the preprocessing output is meant to become an algorithm benchmark case,
read:

- `~/.cellcompass/skills/algorithm/benchmark-dataset-registration/SKILL.md`

Then register only the prepared `.h5ad` that already contains:

- `adata.obs["time_point_processed"]`
- `adata.obsm["X_latent"]`

Do not rely on `register_algorithm_benchmark_dataset(...)` to infer these
fields. That tool validates and catalogs only.
For newly prepared benchmark datasets, pass the exact script that produced the
registered `.h5ad` as `preprocessing_script_path` when calling
`register_algorithm_benchmark_dataset(...)`. The registration tool copies that
script into the benchmark dataset directory and records its SHA256, so later
downstream gene-level analysis can audit filtering, normalization, feature
selection, and latent-space choices.
