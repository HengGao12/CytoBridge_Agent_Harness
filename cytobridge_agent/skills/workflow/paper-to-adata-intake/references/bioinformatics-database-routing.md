# Bioinformatics Database Routing

Use this reference when a paper-intake task includes repository names, accession numbers, repository pages, or a failed download/provider lookup. The goal is to enumerate plausible download listings from the accession and database context before concluding that no directly analyzable data exist.

## Minimal Routing Algorithm

1. Extract exact accessions, repository names, and data availability snippets from the paper and candidate pages.
2. Identify the provider from both the accession pattern and nearby text. Repository words in the paper override ambiguous accession guesses.
3. Generate provider-specific routes: landing page, download page, API/file listing, and public FTP path when available.
4. Inspect exact file listings before classifying availability. Prefer filenames and anchor text over generic page summaries.
5. Classify each resource as processed-ready, matrix-plus-metadata, raw-reprocessable, metadata-only, or blocked/controlled-access.
6. Report routes tried and the strongest evidence. A failed route is not repository-level absence.

## Common Provider Routes

| Provider | Accession cues | Routes to try | Notes |
| --- | --- | --- | --- |
| ArrayExpress / BioStudies / Expression Atlas | `E-[A-Z0-9]+-\d+`, often `E-MTAB-\d+`; paper text says ArrayExpress | Single-cell GXA: `https://www.ebi.ac.uk/gxa/sc/experiments/{ACC}/downloads`; experiment page: `https://www.ebi.ac.uk/gxa/sc/experiments/{ACC}`; bulk GXA fallback: `https://www.ebi.ac.uk/gxa/experiments/{ACC}/downloads`; BioStudies: `https://www.ebi.ac.uk/biostudies/arrayexpress/studies/{ACC}`; legacy files: `https://www.ebi.ac.uk/arrayexpress/experiments/{ACC}/files`; FTP: `https://ftp.ebi.ac.uk/pub/databases/microarray/data/atlas/sc_experiments/{ACC}/`, `https://ftp.ebi.ac.uk/pub/databases/microarray/data/atlas/experiments/{ACC}/`, `https://ftp.ebi.ac.uk/pub/databases/arrayexpress/data/experiment/{FAMILY}/{ACC}/` where `{FAMILY}` is the middle token such as `MTAB` | For scRNA-seq ArrayExpress papers, never stop at BioStudies or ENA metadata. Single Cell Expression Atlas often exposes processed matrices and metadata behind the `/downloads` page. If a guessed `download?fileType=...` route returns 400, inspect `/downloads` for exact links. |
| GEO | `GSE\d+`, `GSM\d+`; paper text says GEO | Landing: `https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={ACC}`; series FTP supplementary files under `https://ftp.ncbi.nlm.nih.gov/geo/series/GSEnnn/{GSE}/suppl/`; sample FTP under `https://ftp.ncbi.nlm.nih.gov/geo/samples/GSMnnn/{GSM}/suppl/`; GEO download bulk endpoint and SOFT/MINiML links from the landing page | Series matrix files are often sample annotations or bulk summaries, not scRNA count matrices. Prefer supplementary `h5`, `h5ad`, `loom`, `mtx`, `barcodes`, `features/genes`, and metadata files. |
| SRA / ENA / BioProject | `SRP`, `ERP`, `DRP`, `SRR`, `ERR`, `DRR`, `PRJNA`, `PRJEB`, `PRJDB`; paper text says SRA, ENA, BioProject | ENA browser and filereport API for run files; SRA Run Selector; BioProject links to child runs and companion GEO/ArrayExpress records | Usually raw sequencing. Treat FASTQ/BAM as raw-reprocessable, not processed-ready. Search linked GEO, ArrayExpress, CELLxGENE, HCA, Zenodo, or lab URLs for processed matrices before falling back to raw. |
| CELLxGENE / CZ CELLxGENE | Collection or dataset UUID, `cellxgene` URLs | Collection/dataset page; public API asset listing; direct `.h5ad` asset URLs when exposed | Usually best source when present because `.h5ad` is directly analyzable. Preserve collection and dataset IDs in provenance. |
| Human Cell Atlas | HCA project UUIDs, `data.humancellatlas.org`, Azul/explore links | HCA data portal project page; Azul file API; matrix service outputs; linked contributor repositories | Prefer processed matrix, loom, h5ad, or bundle metadata. Some projects expose only raw sequence files plus metadata. |
| Single Cell Portal | `SCP\d+`, Broad Single Cell Portal URLs | Study page and download tabs/API where accessible | Expression matrices and metadata may require login or terms acceptance. Report authentication as a blocker, not missing data. |
| Zenodo / Figshare / Dryad / OSF | DOI/record URL, repository name in data availability | Record landing page plus file-list API where available | Inspect all file names and sizes. Large zip/tar archives can contain matrices even when the landing page only shows a dataset title. |
| Synapse | `syn\d+`, Synapse URLs | Synapse entity page and file children listing | Often requires authentication or controlled access. Report access requirements explicitly. |
| NGDC / GSA / OMIX / CNSA / EGA / dbGaP | `CRA`, `PRJCA`, `OMIX`, `CNP`, `EGAS`, `phs` | Repository landing page and public file tables | Often raw or controlled-access. Do not claim absence if credentials, approval, or regional mirrors are required. |

## ArrayExpress And GXA Details

ArrayExpress accessions can be mirrored through several EBI services. For single-cell papers, the useful processed files are frequently in Expression Atlas/GXA rather than the BioStudies landing page.

When the paper says "scRNA-Seq data have been deposited in ArrayExpress under accession number E-MTAB-9715", derive at least:

- `https://www.ebi.ac.uk/gxa/sc/experiments/E-MTAB-9715/downloads`
- `https://www.ebi.ac.uk/gxa/sc/experiments/E-MTAB-9715`
- `https://www.ebi.ac.uk/biostudies/arrayexpress/studies/E-MTAB-9715`
- `https://ftp.ebi.ac.uk/pub/databases/microarray/data/atlas/sc_experiments/E-MTAB-9715/`

On GXA download pages, look for anchors or filenames containing:

- `normalised`, `raw-counts`, `quantification`, `matrix`, `mtx`, `h5ad`, `loom`, `zip`
- `experiment-design`, `sdrf`, `metadata`, `cluster`, `marker`

`experiment_design.tsv` or `sdrf.txt` alone is metadata, not an expression matrix. A `normalised-files.zip`, raw count archive, MEX directory, loom, h5, or h5ad is evidence of directly analyzable data when paired with feature/cell metadata.

## Evidence Rules

- Landing page success is not enough; inspect download listings or API file lists.
- Metadata-only files do not prove that expression data are unavailable.
- Raw FASTQ/BAM availability does not rule out processed matrices hosted by companion portals.
- HTTP 400, 403, JavaScript-heavy pages, or HTML parser failures are route-specific failures. Try another official route before concluding.
- In the final answer, separate processed-ready data, data requiring conversion, raw-reprocessable data, metadata-only resources, and blocked/controlled-access resources.
