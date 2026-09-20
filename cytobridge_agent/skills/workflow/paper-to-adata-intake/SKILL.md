---
name: paper-to-adata-intake
description: Use when the user provides a scRNA-seq paper PDF or article URL and asks what analyzable datasets/resources exist, where the data can be downloaded, which accessions are usable, or how to download/materialize the data as AnnData for CytoBridge.
---
# Paper To AnnData Intake

Use this skill when the user wants to identify analyzable datasets from a scRNA-seq paper, find the real download locations behind accessions, extract repository resources, or materialize data from a paper PDF, article page, or manuscript-derived text file rather than an existing `.h5ad`.

Do not trigger this skill merely because the current session input is a PDF. For normal paper reading or discussion, use the standard PDF/file-reading tools first. Trigger this skill when the user asks which data can be analyzed, where accessions can be downloaded, whether paper data are usable, or when they explicitly ask to download/materialize the data.

## Goal
Materialize a usable AnnData checkpoint from manuscript evidence with minimal runtime changes, then hand the workflow back to normal CytoBridge preprocessing and downstream analysis.

Paper intake is part of preprocessing provenance. Any materialization or follow-up preprocessing used for a new dataset must remain reproducible from saved scripts and registered workflow state, not only from runtime memory or LLM notes.

## Read first when unsure
- `cytobridge_agent/paper_intake/pipeline.py`
- `cytobridge_agent/paper_intake/manuscript.py`
- `cytobridge_agent/paper_intake/providers.py`
- `cytobridge_agent/paper_intake/converters.py`
- `skills/workflow/preprocessing-execution/SKILL.md`
- `references/bioinformatics-database-routing.md` whenever an accession, repository name, database page, or failed provider lookup is involved. Read it before declaring that no processed or directly analyzable data are available.

## Operating rules
- Keep the existing single-agent runtime. Do not invent a parallel pipeline.
- Use `execute_python` for the actual paper intake steps.
- Pass `runtime_llm` into paper intake helpers when it is available so the workflow can use the current session model for multi-round manuscript understanding, page selection, and paper-specific materialization.
- Treat manuscript parsing and dataset resolution as a preprocessing substage.
- Treat the module as an LLM-led evidence loop. The built-in heuristics are only weak recall and ranking aids; the LLM should make the final page-reading, asset-selection, fallback, and materialization decisions from the evidence trace.
- For inspect-only requests such as "what data are available" or "which resources can be analyzed", run the inspection path, review candidates and traces, and report classified resources without materializing unless the user asks to download or continue the workflow.
- Apply the database-routing reference to every accession or repository link discovered from the manuscript and fetched pages. Do not stop at a generic landing page, BioStudies/ENA metadata page, or failed dynamic page fetch.
- If a page fetch/parser fails on a repository site, treat it as a route failure, not evidence that data are absent. Try provider-specific downloads/API/FTP routes and the paper-intake helpers before concluding.
- For ArrayExpress single-cell accessions (`E-*`, especially `E-MTAB-*`), check Single Cell Expression Atlas/GXA download routes before falling back to raw ENA/SRA files.
- When the user names a subset of the paper data, convert it into `target_dataset_prompt` and pass it through every paper-intake call. This prompt is binding scope, not a weak ranking hint. Example: `target_dataset_prompt="Materialize only the in vitro culture arm (days 2/4/6); exclude in vivo transplantation, cytokine perturbation, and full-study archives unless no subset-specific file exists."`
- Leave `target_dataset_prompt` empty when the user did not specify a subset; then use the default best-dataset behavior.
- Do not manually discard an asset only because the heuristic score is lower if the LLM selected it from manuscript/page evidence.
- Recoverable access errors such as 404s, timeouts, missing files, and failed downloads are expected observations. Let the module feed them back into the next LLM round before declaring failure.
- Query-style bulk download endpoints such as GEO `download/?acc=...&format=file` are assets, not web pages to parse. Let paper intake download them only during materialization.
- `llm_discovery_trace.json` is updated during discovery, so inspect it after an interrupt to see the last URL/action in progress.
- Large downloads use resumable streaming. When available, the downloader may use `aria2c` with low parallelism for static NCBI/EBI file hosts, or `curl`/`wget` for robust single-connection transfer; set `CYTOBRIDGE_PAPER_INTAKE_DOWNLOADER=httpx` to force the pure-Python path.
- Original/raw downloaded assets should be stored through the shared raw data pool when it is configured. Set `CYTOBRIDGE_RAW_DATA_POOL=/data/cytobridge/raw_data_pool`; new paper-intake downloads will go under that pool while `<output_dir>/paper_intake/downloads` remains a symlink or pointer for run-local provenance. Keep converted `paper_input.h5ad`, scripts, and reports in the run directory.
- Prefer direct analysis assets in this order:
  1. `h5ad`
  2. 10x `filtered_feature_bc_matrix.h5`
  3. 10x MEX bundle (`matrix.mtx`, `barcodes.tsv`, `features.tsv`)
  4. `loom`
  5. dense count tables
  6. raw FASTQ/SRA only if no processed data is available
- If only Seurat files are provided and there is no equivalent analyzable asset, stop and report the blocker explicitly. Do not claim a pure-Python Seurat conversion that is not actually available.

## Required workflow
1. Inspect the paper source
   - If `input_path` is a PDF or URL, run manuscript inspection first.
   - In the upgraded flow, inspection includes:
     - PDF/full-text plus program-selected highlights -> LLM extraction of likely data links
     - fetched HTML pages plus program-selected highlights -> LLM selection of candidate downloadable assets
     - a binding target prompt, when provided, that limits discovery and selection to the requested subset
   - This writes:
     - `<output_dir>/paper_intake/manuscript_summary.md`
     - `<output_dir>/paper_intake/manuscript_manifest.json`
     - `<output_dir>/paper_intake/resource_candidates.json`
     - `<output_dir>/paper_intake/llm_pdf_analysis.json`
     - `<output_dir>/paper_intake/llm_discovery_trace.json`
     - `<output_dir>/paper_intake/llm_web_analysis.json`
2. Read the manuscript summary and candidate assets
   - Use the LLM trace to decide which discovered datasets are original scRNA-seq data and whether a `dataset_hint` is needed.
   - If there are multiple original datasets, pick the one that best matches the user request and state that choice explicitly.
   - If a page or candidate URL failed, check the trace before retrying or reporting the blocker. Failed URLs are not automatically fatal.
   - If the user only asked which data exist or whether the paper contains directly analyzable data, stop after inspection and database-route review. Do not infer "no directly analyzable data" from repository landing pages alone.
3. Materialize AnnData
   - Run the conversion pipeline and assign the returned object to `adata`.
   - In the upgraded flow, the module will:
     - download the LLM-selected primary asset and, when relevant, supporting metadata sidecars
     - generate a base Python materialization template
     - let the LLM adapt it into a paper-specific script
     - execute that script and validate the result
     - retry with script/asset adjustments when metadata or time columns are still missing
     - feed download and script errors back to the LLM so it can switch to fallback assets or repair the script
   - The module writes `<output_dir>/paper_intake/paper_input.h5ad`.
   - The paper-specific scripts and retry summary are written under `<output_dir>/paper_intake/`.
4. Promote the converted AnnData into the active runtime dataset
   - Call `load_or_switch_adata(path=..., force_reload=True)` on `paper_input.h5ad`.
5. Continue with normal preprocessing
   - Read `preprocessing-execution` skill and continue from there.
   - Ensure the final dataset preparation path is represented by a saved script. If paper intake generated a paper-specific materialization script under `<output_dir>/paper_intake/`, either record that exact script in workflow state or wrap/promote the final end-to-end preprocessing routine under `<output_dir>/scripts/preprocessing/*.py`.
6. Commit the workflow state
   - Record the paper source, selected dataset asset, converted path, preprocessing/materialization script path, and key input/output paths.

## Inspection snippet
Use this first:

```python
from cytobridge_agent.paper_intake import inspect_paper_source
import json

paper = inspect_paper_source(
    input_source=input_path,
    output_dir=output_dir,
    fetch_online_if_needed=True,
    target_dataset_prompt="",  # set when user requested a specific paper subset
    llm=runtime_llm,
    use_llm=True,
)

print(json.dumps({
    "summary_path": paper["summary_path"],
    "asset_report_path": paper["asset_report_path"],
    "inspection_bundle_path": paper["inspection_bundle_path"],
    "candidate_asset_count": len(paper.get("candidate_assets", [])),
    "accessions": paper.get("accessions", {}),
    "target_dataset_prompt": paper.get("target_dataset_prompt", ""),
}, ensure_ascii=False, indent=2))
```

Then read:
- `<output_dir>/paper_intake/manuscript_summary.md`
- `<output_dir>/paper_intake/resource_candidates.json`
- `<output_dir>/paper_intake/llm_discovery_trace.json`
- `references/bioinformatics-database-routing.md` if any accession or repository page was found

## Conversion snippet
Use this after deciding the target dataset or after confirming auto-selection is safe:

```python
from cytobridge_agent.paper_intake import prepare_anndata_from_paper
import json

paper_result = prepare_anndata_from_paper(
    input_path=input_path,
    output_dir=output_dir,
    dataset_hint=None,  # replace when multiple datasets are present
    target_dataset_prompt=None,  # pass the user's subset request here when present
    fetch_online_if_needed=True,
    prefer_processed=True,
    allow_raw=True,
    llm=runtime_llm,
    use_llm=True,
    require_downstream_ready=True,
)

adata = paper_result["adata"]

print(json.dumps({
    "adata_path": paper_result["adata_path"],
    "selected_asset": paper_result["selected_asset"],
    "report_path": paper_result["report_path"],
    "inspection_bundle_path": paper_result["inspection_bundle_path"],
    "llm_discovery_trace_path": paper_result.get("llm_discovery_trace_path", ""),
}, ensure_ascii=False, indent=2))
```

After that, call:

`load_or_switch_adata(path="<output_dir>/paper_intake/paper_input.h5ad", force_reload=True)`

## Raw data handling
- Raw sequencing support is optional and environment-dependent.
- Current automated raw path assumes a 10x-style pipeline and requires:
  - `cellranger` on `PATH` or `CYTOBRIDGE_CELLRANGER_BIN`
  - a reference specified by `CYTOBRIDGE_CELLRANGER_REFERENCE`
  - `fasterq-dump` on `PATH` or `CYTOBRIDGE_FASTERQ_DUMP_BIN` when the source is `.sra`
- If these are missing, report the exact missing dependency and prefer processed data if any exists.

## Completion
Before leaving this skill, make sure all of the following are true:
- runtime `adata` is loaded from `paper_input.h5ad`
- the selected asset is documented
- the LLM discovery and materialization traces are documented, especially when failed URLs or fallback assets influenced the result
- the paper-derived artifacts are on disk
- the paper materialization script or final preprocessing script is registered in workflow state
- the next step is the normal preprocessing workflow, not an ad hoc branch
