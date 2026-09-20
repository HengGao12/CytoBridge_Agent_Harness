from __future__ import annotations

import csv
import io
import json
import os
import re
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import parse_qs, urljoin, urlparse

import httpx

try:
    from bs4 import BeautifulSoup
except Exception:  # pragma: no cover - optional fallback
    BeautifulSoup = None

from .manuscript import DATA_HOST_HINTS, USER_AGENT, extract_accessions, fetch_limited_text_response
from .models import DataAsset, ManuscriptInspection


DIRECT_EXTENSIONS: List[Tuple[Tuple[str, ...], str, str]] = [
    ((".h5ad", ".h5ad.gz"), "anndata", "h5ad"),
    ((".h5seurat",), "seurat", "h5seurat"),
    ((".loom",), "processed", "loom"),
    ((".mtx", ".mtx.gz"), "processed", "10x_mtx"),
    ((".h5",), "processed", "10x_h5"),
    ((".csv", ".csv.gz", ".tsv", ".tsv.gz", ".txt", ".txt.gz"), "processed", "table"),
    ((".zip", ".tar", ".tar.gz", ".tgz"), "archive", "archive"),
    ((".fastq", ".fastq.gz", ".fq", ".fq.gz"), "raw", "fastq"),
    ((".sra",), "raw", "sra"),
    ((".bam", ".cram"), "raw", "aligned_reads"),
]

DATA_HINTS = (
    "matrix",
    "counts",
    "filtered",
    "feature_bc_matrix",
    "h5ad",
    "loom",
    "download",
    "filetype",
    "normalised",
    "normalized",
    "quantification",
)
METADATA_HINTS = (
    "metadata",
    "annotation",
    "celltype",
    "readme",
    "summary",
    "filelist",
    "manifest",
    "checksum",
    "experiment-design",
    "experiment_design",
)
EXPRESSION_TABLE_HINTS = (
    "count",
    "counts",
    "raw-counts",
    "raw_counts",
    "normalised",
    "normalized",
    "quantification",
    "expression",
    "matrix",
)
MATRIX_HINTS = ("mtx", "matrix-market", "feature_bc_matrix", "matrix.mtx")
ARCHIVE_HINTS = ("archive", "zip", "tar", "tgz", "bundle")
RAW_READ_HINTS = ("fastq", "fq.gz", "bam", "cram", "sra", "submitted reads")
DYNAMIC_DOWNLOAD_QUERY_KEYS = ("filetype", "file_type", "downloadtype", "download_type", "type")
PAGE_CRAWL_HINTS = (
    "download",
    "dataset",
    "supp",
    "files",
    "matrix",
    "atlas",
    "arrayexpress",
    "experiment",
    "record",
)
ARRAYEXPRESS_ACCESSION_RE = re.compile(r"\bE-[A-Z0-9]{3,12}-\d+\b", re.IGNORECASE)
PROCESSED_ARCHIVE_HINTS = (
    "matrix",
    "count",
    "counts",
    "filtered",
    "feature_bc_matrix",
    "cellranger",
    "10x",
    "loom",
    "h5ad",
    "barcodes",
    "features",
    "genes",
)
RAW_ARCHIVE_HINTS = ("raw", "fastq", "fq", "bam", "cram", "fragments", "sra", "reads")
GEO_FILELIST_ENTRY_RE = re.compile(
    r"(?:^|\s)(?:Archive\s+)?(?P<name>[^\s]+\.(?:h5ad|h5seurat|loom|mtx(?:\.gz)?|h5|csv(?:\.gz)?|tsv(?:\.gz)?|txt(?:\.gz)?|tar(?:\.gz)?|tgz|zip|fastq(?:\.gz)?|fq(?:\.gz)?|sra|bam|cram))"
    r"\s+\d{1,2}/\d{1,2}/\d{4}\s+\d{1,2}:\d{2}:\d{2}\s+\d+\s+[A-Za-z]+\s+File",
    flags=re.IGNORECASE,
)
HTTP_TIMEOUT = httpx.Timeout(10.0, read=45.0, write=45.0, pool=45.0)
TraceCallback = Optional[Callable[[str], None]]


def _int_from_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return max(0, int(raw.strip()))
    except Exception:
        return default


DEFAULT_GEO_SAMPLE_EXPANSIONS = _int_from_env("CYTOBRIDGE_GEO_SAMPLE_EXPANSIONS", 0)


def _dedupe(values: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for value in values:
        item = (value or "").strip()
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _safe_filename_token(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text or "")).strip("._")
    return cleaned or "asset"


def _query_first(query: Dict[str, List[str]], *keys: str) -> str:
    lowered = {str(key).lower(): values for key, values in query.items()}
    for key in keys:
        values = lowered.get(key.lower()) or []
        for value in values:
            text = str(value or "").strip()
            if text:
                return text
    return ""


def _query_evidence(query: Dict[str, List[str]]) -> str:
    parts: List[str] = []
    for key, values in query.items():
        for value in values:
            parts.append(f"{key}={value}")
    return " ".join(parts)


def _arrayexpress_accession_from_text(text: str) -> str:
    match = ARRAYEXPRESS_ACCESSION_RE.search(text or "")
    return match.group(0).upper() if match else ""


def _dynamic_download_extension(evidence_lower: str) -> str:
    if "h5ad" in evidence_lower or "anndata" in evidence_lower:
        return ".h5ad"
    if "loom" in evidence_lower:
        return ".loom"
    if any(token in evidence_lower for token in MATRIX_HINTS):
        return ".mtx.gz"
    if "h5" in evidence_lower and "h5ad" not in evidence_lower:
        return ".h5"
    if any(token in evidence_lower for token in ARCHIVE_HINTS):
        return ".zip"
    return ".tsv"


def _safe_filename_from_url(url: str) -> str:
    parsed = urlparse(url)
    name = Path(parsed.path).name
    query = parse_qs(parsed.query)
    explicit_name = _query_first(query, "file", "filename", "file_name", "fileName", "name")
    if explicit_name:
        return str(explicit_name)
    if name and "." in name:
        return name
    dynamic_type = _query_first(query, *DYNAMIC_DOWNLOAD_QUERY_KEYS)
    if dynamic_type:
        dynamic_name = Path(dynamic_type).name
        if "." in dynamic_name:
            return dynamic_name
        accession = _arrayexpress_accession_from_text(url)
        stem = _safe_filename_token(f"{accession}_{dynamic_type}" if accession else dynamic_type)
        return f"{stem}{_dynamic_download_extension(dynamic_type.lower())}"
    for value in query.get("download") or []:
        text = str(value).strip()
        if text and text.lower() not in {"1", "true", "yes"}:
            return text
    acc_values = query.get("acc") or []
    format_values = query.get("format") or []
    if "geo/download" in parsed.path.lower() and acc_values and any(value.lower() == "file" for value in format_values):
        return f"{str(acc_values[0]).upper()}_geo_download.tar"
    if name:
        return name
    return "downloaded_asset"


def _trace(callback: TraceCallback, message: str) -> None:
    if callback is None:
        return
    try:
        callback(message)
    except Exception:
        return


def _asset_text(asset: DataAsset) -> str:
    return " ".join(
        [
            asset.filename,
            asset.url,
            asset.description,
            json.dumps(asset.metadata or {}, ensure_ascii=False),
        ]
    ).lower()


def _is_geo_filelist_name(filename: str) -> bool:
    lowered = (filename or "").strip().lower()
    return lowered in {"filelist.txt", "filelist.tsv", "filelist.csv"} or lowered.startswith("filelist.")


def _parse_geo_filelist_entries(text: str) -> List[str]:
    names = [match.group("name").strip() for match in GEO_FILELIST_ENTRY_RE.finditer(text or "")]
    out: List[str] = []
    seen: Set[str] = set()
    for name in names:
        lowered = name.lower()
        if not name or lowered in seen or _is_geo_filelist_name(lowered):
            continue
        seen.add(lowered)
        out.append(name)
    return out


def _expand_geo_filelist_asset(asset: DataAsset, trace: TraceCallback = None) -> List[DataAsset]:
    if not _is_geo_filelist_name(asset.filename):
        return []
    try:
        payload = _fetch_html(asset.url, trace=trace)
    except Exception as exc:
        _trace(trace, f"[providers] GEO filelist fetch failed url={asset.url} error={exc}")
        return []

    names = _parse_geo_filelist_entries(str(payload.get("text") or ""))
    if not names:
        _trace(trace, f"[providers] GEO filelist produced no expandable entries url={asset.url}")
        return []

    expanded: List[DataAsset] = []
    for name in names:
        candidate = classify_reference(urljoin(asset.url, name), source=asset.url, description=f"GEO filelist entry {name}")
        candidate.score += 6.0
        expanded.append(candidate)
    _trace(trace, f"[providers] GEO filelist expanded url={asset.url} entries={len(expanded)}")
    return expanded


def asset_looks_raw_candidate(asset: DataAsset) -> bool:
    if asset.asset_kind == "raw":
        return True
    if asset.asset_kind != "archive":
        return False
    text = _asset_text(asset)
    return any(token in text for token in RAW_ARCHIVE_HINTS)


def asset_looks_processed_candidate(asset: DataAsset) -> bool:
    if asset.asset_kind in {"anndata", "processed"}:
        return True
    if asset.file_format in {"h5ad", "10x_h5", "10x_mtx", "loom"}:
        return True
    if asset.asset_kind != "archive":
        return False
    text = _asset_text(asset)
    if any(token in text for token in RAW_ARCHIVE_HINTS):
        return False
    return any(token in text for token in PROCESSED_ARCHIVE_HINTS)


def classify_reference(url: str, source: str = "", description: str = "") -> DataAsset:
    lowered = url.lower()
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    filename = _safe_filename_from_url(url)
    name_lower = filename.lower()
    path_lower = parsed.path.lower()
    query_lower = _query_evidence(query).lower()
    evidence_lower = " ".join([lowered, name_lower, str(description or "").lower(), query_lower])
    asset_kind = "page"
    file_format = "html"
    score = 0.0

    for suffixes, kind, fmt in DIRECT_EXTENSIONS:
        if any(lowered.endswith(suffix) for suffix in suffixes):
            asset_kind = kind
            file_format = fmt
            break

    if asset_kind == "page":
        if any(marker in lowered for marker in ("format=h5ad", "filetype=h5ad", "download=h5ad")):
            asset_kind, file_format = "anndata", "h5ad"
        elif any(marker in evidence_lower for marker in ("feature_bc_matrix", "filtered_feature_bc_matrix")):
            asset_kind, file_format = "processed", "10x_h5"

    provider = "generic"
    netloc = parsed.netloc.lower()
    if "geo" in netloc or "ncbi.nlm.nih.gov" in netloc or "ftp.ncbi.nlm.nih.gov" in netloc:
        provider = "geo"
    elif "zenodo.org" in netloc:
        provider = "zenodo"
    elif "cellxgene.cziscience.com" in netloc:
        provider = "cellxgene"
    elif "humancellatlas.org" in netloc:
        provider = "hca"
    elif "ebi.ac.uk" in netloc:
        if "/gxa/" in path_lower or "microarray/data/atlas" in path_lower or "expression atlas" in evidence_lower:
            provider = "expression_atlas"
        elif "arrayexpress" in netloc or "/arrayexpress/" in path_lower or "/biostudies/" in path_lower or "arrayexpress" in evidence_lower:
            provider = "arrayexpress"
        else:
            provider = "ena"

    if asset_kind == "page" and provider == "geo" and "/geo/download" in parsed.path.lower():
        format_values = [str(value).lower() for value in query.get("format", [])]
        if "file" in format_values:
            if filename.lower().endswith((".h5ad", ".h5ad.gz")):
                asset_kind = "anndata"
                file_format = "h5ad"
            else:
                asset_kind = "archive"
                file_format = "archive"
                acc_values = query.get("acc") or []
                if acc_values:
                    filename = f"{str(acc_values[0]).upper()}_geo_download.tar"
            score += 25.0

    dynamic_type = _query_first(query, *DYNAMIC_DOWNLOAD_QUERY_KEYS)
    is_downloads_page = (
        provider == "expression_atlas"
        and not dynamic_type
        and path_lower.rstrip("/").endswith("/downloads")
    )
    dynamic_download = (
        provider in {"expression_atlas", "arrayexpress"}
        and not is_downloads_page
        and (
            bool(dynamic_type)
            or path_lower.rstrip("/").endswith("/download")
            or "/download/" in path_lower
        )
    )
    if asset_kind == "page" and dynamic_download:
        if "h5ad" in evidence_lower or "anndata" in evidence_lower:
            asset_kind, file_format = "anndata", "h5ad"
        elif "loom" in evidence_lower:
            asset_kind, file_format = "processed", "loom"
        elif any(token in evidence_lower for token in RAW_READ_HINTS):
            asset_kind, file_format = "raw", "fastq"
        elif any(token in evidence_lower for token in ARCHIVE_HINTS):
            asset_kind, file_format = "archive", "archive"
        elif any(token in evidence_lower for token in MATRIX_HINTS):
            asset_kind, file_format = "processed", "10x_mtx"
        elif any(token in evidence_lower for token in EXPRESSION_TABLE_HINTS + METADATA_HINTS):
            asset_kind, file_format = "processed", "table"

    if asset_kind == "page" and any(token in evidence_lower for token in ("download", "supp", "matrix", "dataset")):
        score += 5.0
    if any(token in evidence_lower for token in ("filtered", "matrix", "counts", "normalised", "normalized")):
        score += 8.0
    raw_read_like = any(token in name_lower for token in ("fastq", "fq", "bam", "cram", "sra")) or (
        "raw" in name_lower and not any(token in name_lower for token in ("raw-count", "raw_count", "raw.count"))
    )
    if raw_read_like:
        score -= 12.0
    if any(token in evidence_lower for token in METADATA_HINTS):
        score -= 15.0
    if provider == "expression_atlas" and asset_kind != "page":
        score += 14.0
    elif provider == "arrayexpress" and asset_kind != "page":
        score += 8.0
    if any(token in evidence_lower for token in ("raw-counts", "raw_counts", "normalised", "normalized", "quantification")):
        score += 12.0
    if asset_kind == "anndata":
        score += 80.0
    elif file_format == "10x_h5":
        score += 70.0
    elif file_format == "10x_mtx":
        score += 60.0
    elif file_format == "loom":
        score += 50.0
    elif asset_kind == "processed":
        score += 35.0
    elif asset_kind == "archive":
        score += 25.0
    elif asset_kind == "seurat":
        score += 10.0
    elif asset_kind == "raw":
        score -= 20.0

    return DataAsset(
        url=url,
        provider=provider,
        asset_kind=asset_kind,
        file_format=file_format,
        filename=filename,
        score=score,
        source=source,
        description=description,
    )


def _fetch_html(url: str, timeout: object = HTTP_TIMEOUT, trace: TraceCallback = None) -> Dict[str, object]:
    _trace(trace, f"[providers] fetch_html url={url}")
    body, final_url, headers, truncated = fetch_limited_text_response(url, timeout=timeout)
    links: List[str] = []
    anchors: List[Tuple[str, str]] = []
    if BeautifulSoup is not None:
        soup = BeautifulSoup(body, "html.parser")
        for anchor in soup.find_all("a", href=True):
            href = urljoin(final_url, anchor.get("href", ""))
            text = " ".join(anchor.stripped_strings)
            links.append(href)
            anchors.append((href, text))
        text = "\n".join(chunk.strip() for chunk in soup.stripped_strings if chunk.strip())
    else:
        for href in re.findall(r"""href=["']([^"']+)["']""", body, flags=re.IGNORECASE):
            resolved = urljoin(final_url, href.rstrip(".,);"))
            links.append(resolved)
            anchors.append((resolved, ""))
        text = re.sub(r"<[^>]+>", " ", body)
        text = re.sub(r"\s+", " ", text).strip()
    _trace(
        trace,
        f"[providers] fetched url={final_url} content_type={headers.get('content-type', '')} "
        f"truncated={truncated} discovered_links={len(links)}",
    )
    return {
        "url": final_url,
        "text": text,
        "links": anchors,
        "raw_links": _dedupe(links),
        "truncated": truncated,
        "content_type": headers.get("content-type", ""),
    }


def _crawl_directory_listing(url: str, trace: TraceCallback = None) -> List[DataAsset]:
    try:
        page = _fetch_html(url, trace=trace)
    except Exception as exc:
        _trace(trace, f"[providers] directory listing failed url={url} error={exc}")
        return []
    assets: List[DataAsset] = []
    for href, text in page.get("links") or []:
        href_str = str(href)
        if href_str.endswith("/"):
            continue
        asset = classify_reference(href_str, source=str(page.get("url") or url), description=text)
        if asset.asset_kind != "page":
            asset.score += 4.0
            assets.append(asset)
    return assets


def _geo_prefix(accession: str, root: str) -> str:
    digits = accession[3:]
    return f"{root}{digits[:-3]}nnn" if len(digits) > 3 else f"{root}nnn"


def resolve_geo_accession(
    accession: str,
    *,
    include_nested_samples: bool = False,
    max_nested_samples: int = 0,
    visited: Optional[Set[str]] = None,
    trace: TraceCallback = None,
) -> List[DataAsset]:
    acc = accession.upper()
    active = visited if visited is not None else set()
    if acc in active:
        _trace(trace, f"[providers] GEO {acc}: skipping already-visited accession")
        return []
    active.add(acc)

    assets: List[DataAsset] = []
    page_url = f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={acc}"
    nested_samples: List[str] = []
    try:
        _trace(trace, f"[providers] GEO {acc}: resolving landing page")
        page = _fetch_html(page_url, trace=trace)
        discovered = extract_accessions(str(page.get("text") or ""))
        nested_samples = [nested for nested in discovered.get("geo_sample", []) if nested != acc]
        for href, text in page.get("links") or []:
            asset = classify_reference(str(href), source=page_url, description=text)
            if asset.asset_kind != "page" or any(key in str(text).lower() for key in DATA_HINTS):
                asset.score += 6.0
                assets.append(asset)
    except Exception as exc:
        _trace(trace, f"[providers] GEO {acc}: landing page failed error={exc}")

    if acc.startswith("GSE"):
        base_dir = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{_geo_prefix(acc, 'GSE')}/{acc}"
        assets.extend(_crawl_directory_listing(f"{base_dir}/suppl/", trace=trace))
        assets.extend(_crawl_directory_listing(f"{base_dir}/matrix/", trace=trace))
        expanded_assets: List[DataAsset] = []
        for asset in list(assets):
            if _is_geo_filelist_name(asset.filename):
                expanded_assets.extend(_expand_geo_filelist_asset(asset, trace=trace))
        assets.extend(expanded_assets)
        if include_nested_samples and max_nested_samples > 0 and not _has_high_confidence_assets(assets):
            limit = min(max_nested_samples, len(nested_samples))
            if limit:
                _trace(trace, f"[providers] GEO {acc}: expanding {limit} nested GSM pages")
            for nested in nested_samples[:limit]:
                assets.extend(
                    resolve_geo_accession(
                        nested,
                        include_nested_samples=False,
                        max_nested_samples=0,
                        visited=active,
                        trace=trace,
                    )
                )
                if _has_high_confidence_assets(assets):
                    break
    elif acc.startswith("GSM"):
        base_dir = f"https://ftp.ncbi.nlm.nih.gov/geo/samples/{_geo_prefix(acc, 'GSM')}/{acc}"
        assets.extend(_crawl_directory_listing(f"{base_dir}/suppl/", trace=trace))

    for asset in assets:
        asset.metadata["geo_accession"] = acc
        if asset.asset_kind == "anndata":
            asset.score += 8.0
        lowered = asset.filename.lower()
        if _is_geo_filelist_name(lowered):
            asset.score -= 28.0
        if "series_matrix" in lowered:
            asset.score -= 24.0
    return assets


def _extract_zenodo_record_id(identifier: str) -> str:
    text = (identifier or "").strip()
    match = re.search(r"zenodo[./](\d+)", text, re.IGNORECASE)
    if match:
        return match.group(1)
    match = re.search(r"/records?/(\d+)", text, re.IGNORECASE)
    return match.group(1) if match else ""


def resolve_zenodo_identifier(identifier: str, trace: TraceCallback = None) -> List[DataAsset]:
    record_id = _extract_zenodo_record_id(identifier)
    if not record_id:
        return []
    api_url = f"https://zenodo.org/api/records/{record_id}"
    headers = {"User-Agent": USER_AGENT}
    _trace(trace, f"[providers] Zenodo {record_id}: resolving record metadata")
    with httpx.Client(follow_redirects=True, timeout=HTTP_TIMEOUT, headers=headers) as client:
        response = client.get(api_url)
        response.raise_for_status()
        payload = response.json()
    _trace(trace, f"[providers] Zenodo {record_id}: resolved {len(payload.get('files', []))} files")
    assets: List[DataAsset] = []
    for item in payload.get("files", []):
        links = item.get("links") or {}
        url = links.get("self") or item.get("self") or ""
        if not url:
            continue
        asset = classify_reference(str(url), source=api_url, description=str(item.get("key") or ""))
        asset.score += 10.0
        asset.metadata["zenodo_record"] = record_id
        assets.append(asset)
    return assets


def resolve_ena_accession(accession: str, trace: TraceCallback = None) -> List[DataAsset]:
    acc = accession.upper()
    api_url = (
        "https://www.ebi.ac.uk/ena/portal/api/filereport"
        f"?accession={acc}&result=read_run&fields=run_accession,study_accession,fastq_ftp,submitted_ftp,sra_ftp,library_layout"
    )
    headers = {"User-Agent": USER_AGENT}
    _trace(trace, f"[providers] ENA {acc}: resolving filereport")
    with httpx.Client(follow_redirects=True, timeout=HTTP_TIMEOUT, headers=headers) as client:
        response = client.get(api_url)
        response.raise_for_status()
        text = response.text
    reader = csv.DictReader(io.StringIO(text), delimiter="\t")
    assets: List[DataAsset] = []
    for row in reader:
        urls: List[str] = []
        for key in ("fastq_ftp", "submitted_ftp", "sra_ftp"):
            raw_value = str(row.get(key) or "").strip()
            if not raw_value:
                continue
            for item in raw_value.split(";"):
                value = item.strip()
                if not value:
                    continue
                if not value.startswith(("http://", "https://")):
                    value = f"https://{value.lstrip('/')}"
                urls.append(value)
        urls = _dedupe(urls)
        if not urls:
            continue
        primary = urls[0]
        asset = classify_reference(primary, source=api_url, description=f"ENA raw files for {row.get('run_accession') or acc}")
        asset.asset_kind = "raw"
        asset.file_format = "fastq" if any(".fastq" in url.lower() or ".fq" in url.lower() for url in urls) else asset.file_format
        asset.score -= 8.0
        asset.metadata.update(
            {
                "ena_accession": acc,
                "run_accession": row.get("run_accession") or "",
                "study_accession": row.get("study_accession") or "",
                "library_layout": row.get("library_layout") or "",
                "raw_urls": urls,
            }
        )
        assets.append(asset)
    return assets


def _arrayexpress_ftp_family(accession: str) -> str:
    parts = accession.upper().split("-")
    if len(parts) >= 3 and parts[1]:
        return parts[1]
    return accession.upper().rsplit("-", 1)[0]


def _arrayexpress_seed_urls(accession: str) -> List[str]:
    acc = accession.upper()
    family = _arrayexpress_ftp_family(acc)
    return _dedupe(
        [
            f"https://www.ebi.ac.uk/gxa/sc/experiments/{acc}/downloads",
            f"https://www.ebi.ac.uk/gxa/sc/experiments/{acc}",
            f"https://www.ebi.ac.uk/gxa/experiments/{acc}/downloads",
            f"https://www.ebi.ac.uk/gxa/experiments/{acc}",
            f"https://www.ebi.ac.uk/biostudies/arrayexpress/studies/{acc}",
            f"https://www.ebi.ac.uk/arrayexpress/experiments/{acc}/files",
            f"https://ftp.ebi.ac.uk/pub/databases/microarray/data/atlas/sc_experiments/{acc}/",
            f"https://ftp.ebi.ac.uk/pub/databases/microarray/data/atlas/experiments/{acc}/",
            f"https://ftp.ebi.ac.uk/pub/databases/arrayexpress/data/experiment/{family}/{acc}/",
        ]
    )


def resolve_arrayexpress_accession(
    accession: str,
    *,
    visited: Optional[Set[str]] = None,
    trace: TraceCallback = None,
) -> List[DataAsset]:
    acc = accession.upper()
    if not ARRAYEXPRESS_ACCESSION_RE.fullmatch(acc):
        return []

    active = visited if visited is not None else set()
    if acc in active:
        _trace(trace, f"[providers] ArrayExpress {acc}: skipping already-visited accession")
        return []
    active.add(acc)

    assets: List[DataAsset] = []
    for seed_url in _arrayexpress_seed_urls(acc):
        _trace(trace, f"[providers] ArrayExpress {acc}: crawling seed {seed_url}")
        discovered = crawl_download_page(
            seed_url,
            max_depth=1,
            trace=trace,
            visited_accessions=active,
            allow_nested_arrayexpress=False,
            scope_arrayexpress_accession=acc,
        )
        if not discovered and seed_url.endswith("/"):
            discovered = _crawl_directory_listing(seed_url, trace=trace)
        for asset in discovered:
            asset_accessions = {
                match.group(0).upper()
                for match in ARRAYEXPRESS_ACCESSION_RE.finditer(
                    " ".join([str(asset.url or ""), str(asset.filename or ""), str(asset.description or "")])
                )
            }
            if asset_accessions and acc not in asset_accessions:
                _trace(
                    trace,
                    f"[providers] ArrayExpress {acc}: skipping out-of-scope asset accessions={sorted(asset_accessions)} url={asset.url}",
                )
                continue
            asset_accession = str(asset.metadata.get("arrayexpress_accession") or "").strip().upper()
            if asset_accession and asset_accession != acc:
                _trace(
                    trace,
                    f"[providers] ArrayExpress {acc}: skipping nested asset from {asset_accession} url={asset.url}",
                )
                continue
            asset.metadata["arrayexpress_accession"] = acc
            if asset.provider in {"expression_atlas", "arrayexpress"}:
                asset.score += 6.0
            if asset.asset_kind == "anndata":
                asset.score += 8.0
            assets.append(asset)
    return assets


def _discover_nested_accessions(text: str) -> Dict[str, List[str]]:
    payload = extract_accessions(text)
    return {key: values for key, values in payload.items() if values}


def crawl_download_page(
    url: str,
    depth: int = 0,
    max_depth: int = 1,
    trace: TraceCallback = None,
    visited_accessions: Optional[Set[str]] = None,
    allow_nested_arrayexpress: bool = True,
    scope_arrayexpress_accession: str = "",
) -> List[DataAsset]:
    assets: List[DataAsset] = []
    try:
        page = _fetch_html(url, trace=trace)
    except Exception as exc:
        _trace(trace, f"[providers] crawl failed url={url} error={exc}")
        return assets

    for href, text in page.get("links") or []:
        href_str = str(href)
        asset = classify_reference(href_str, source=str(page.get("url") or url), description=text)
        if asset.asset_kind == "page":
            lowered = href_str.lower()
            should_crawl = depth < max_depth and any(token in lowered for token in PAGE_CRAWL_HINTS)
            current_url = str(page.get("url") or url)
            current_parsed = urlparse(current_url)
            href_parsed = urlparse(href_str)
            scope_acc = str(scope_arrayexpress_accession or "").strip().upper()
            if scope_acc and href_parsed.netloc.lower() == "ftp.ebi.ac.uk" and "microarray/data/atlas" in href_parsed.path.lower():
                should_crawl = should_crawl and scope_acc in href_parsed.path.upper()
            if scope_acc:
                href_accessions = {match.group(0).upper() for match in ARRAYEXPRESS_ACCESSION_RE.finditer(href_str)}
                should_crawl = should_crawl and (not href_accessions or scope_acc in href_accessions)
            if current_parsed.netloc.lower() == "ftp.ebi.ac.uk" and href_parsed.netloc.lower() == "ftp.ebi.ac.uk":
                base_path = current_parsed.path
                if not base_path.endswith("/"):
                    base_path = base_path.rsplit("/", 1)[0] + "/"
                # Directory listings include parent links; following them broadens one accession
                # into unrelated EBI experiments and pollutes the candidate set.
                should_crawl = should_crawl and href_parsed.path.startswith(base_path) and href_parsed.path != base_path
            if should_crawl:
                assets.extend(
                    crawl_download_page(
                        href_str,
                        depth=depth + 1,
                        max_depth=max_depth,
                        trace=trace,
                        visited_accessions=visited_accessions,
                        allow_nested_arrayexpress=allow_nested_arrayexpress,
                        scope_arrayexpress_accession=scope_arrayexpress_accession,
                    )
                )
            continue
        if any(token in asset.filename.lower() for token in METADATA_HINTS):
            asset.score -= 12.0
        assets.append(asset)

    nested = _discover_nested_accessions(str(page.get("text") or ""))
    for accession in nested.get("geo_series", [])[:6]:
        assets.extend(resolve_geo_accession(accession, trace=trace))
    for accession in nested.get("geo_sample", [])[:6]:
        assets.extend(resolve_geo_accession(accession, trace=trace))
    for accession in nested.get("sra_study", [])[:4]:
        try:
            assets.extend(resolve_ena_accession(accession, trace=trace))
        except Exception:
            continue
    for accession in nested.get("bioproject", [])[:4]:
        try:
            assets.extend(resolve_ena_accession(accession, trace=trace))
        except Exception:
            continue
    active = visited_accessions if visited_accessions is not None else set()
    if allow_nested_arrayexpress:
        for accession in nested.get("arrayexpress", [])[:4]:
            try:
                assets.extend(resolve_arrayexpress_accession(accession, visited=active, trace=trace))
            except Exception:
                continue
    for doi in nested.get("zenodo_doi", [])[:4]:
        try:
            assets.extend(resolve_zenodo_identifier(doi, trace=trace))
        except Exception:
            continue
    return assets


def _asset_identity(asset: DataAsset) -> Tuple[str, str, str]:
    return (asset.url, asset.asset_kind, asset.file_format)


def _has_high_confidence_assets(assets: List[DataAsset]) -> bool:
    return any(asset_looks_processed_candidate(asset) for asset in assets)


def _seed_urls_for_fallback(inspection: ManuscriptInspection) -> List[str]:
    seeds: List[str] = []
    for url in inspection.discovered_urls:
        lowered = url.lower()
        if not lowered.startswith(("http://", "https://")):
            continue
        if lowered.endswith("/geo/query/acc"):
            continue
        if "acc=" not in lowered and not any(host in lowered for host in DATA_HOST_HINTS):
            continue
        seeds.append(url)
    return _dedupe(seeds)[:8]


def resolve_manuscript_assets(inspection: ManuscriptInspection, trace: TraceCallback = None) -> List[DataAsset]:
    assets: List[DataAsset] = []

    for accession in inspection.accessions.get("geo_series", []):
        assets.extend(resolve_geo_accession(accession, trace=trace))
    for accession in inspection.accessions.get("geo_sample", []):
        assets.extend(resolve_geo_accession(accession, trace=trace))
    for accession in inspection.accessions.get("sra_study", []):
        try:
            assets.extend(resolve_ena_accession(accession, trace=trace))
        except Exception:
            continue
    for accession in inspection.accessions.get("bioproject", []):
        try:
            assets.extend(resolve_ena_accession(accession, trace=trace))
        except Exception:
            continue

    for accession in inspection.accessions.get("arrayexpress", []):
        try:
            assets.extend(resolve_arrayexpress_accession(accession, trace=trace))
        except Exception:
            continue

    for doi in inspection.accessions.get("zenodo_doi", []):
        try:
            assets.extend(resolve_zenodo_identifier(doi, trace=trace))
        except Exception:
            continue

    if not _has_high_confidence_assets(assets):
        for url in _seed_urls_for_fallback(inspection):
            asset = classify_reference(url, source="manuscript")
            if asset.asset_kind == "page":
                lowered = url.lower()
                if any(host in lowered for host in DATA_HOST_HINTS) or any(token in lowered for token in PAGE_CRAWL_HINTS + ("geo/query",)):
                    assets.extend(crawl_download_page(url, trace=trace))
                continue
            assets.append(asset)

    if not _has_high_confidence_assets(assets) and DEFAULT_GEO_SAMPLE_EXPANSIONS > 0:
        for accession in inspection.accessions.get("geo_series", []):
            try:
                assets.extend(
                    resolve_geo_accession(
                        accession,
                        include_nested_samples=True,
                        max_nested_samples=DEFAULT_GEO_SAMPLE_EXPANSIONS,
                        trace=trace,
                    )
                )
            except Exception:
                continue
            if _has_high_confidence_assets(assets):
                break

    deduped: Dict[Tuple[str, str, str], DataAsset] = {}
    for asset in assets:
        key = _asset_identity(asset)
        previous = deduped.get(key)
        if previous is None or asset.score > previous.score:
            deduped[key] = asset

    return sorted((asset for asset in deduped.values() if asset.url), key=lambda item: item.score, reverse=True)


def write_asset_report(assets: List[DataAsset], output_dir: str) -> str:
    target = Path(output_dir).expanduser().resolve() / "paper_intake" / "resource_candidates.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps([asset.to_dict() for asset in assets], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return str(target)
