from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple
from urllib.parse import quote, urljoin

import httpx
from pypdf import PdfReader

from .models import ManuscriptInspection

try:
    from bs4 import BeautifulSoup
except Exception:  # pragma: no cover - optional fallback
    BeautifulSoup = None

try:
    import pdfplumber
except Exception:  # pragma: no cover - optional fallback
    pdfplumber = None


USER_AGENT = "CytoBridgeAgent/1.0 (+https://github.com/your-org/cytobridge-agent)"
DEFAULT_HTML_FETCH_MAX_BYTES = 2_000_000
URL_RE = re.compile(r"https?://[^\s<>{}\"')]+", re.IGNORECASE)
DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+\b", re.IGNORECASE)
UUID_RE = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b",
    re.IGNORECASE,
)

ACCESSION_PATTERNS = {
    "geo_series": re.compile(r"\bGSE\d{3,}\b", re.IGNORECASE),
    "geo_sample": re.compile(r"\bGSM\d{3,}\b", re.IGNORECASE),
    "sra_run": re.compile(r"\b[SED]RR\d{3,}\b", re.IGNORECASE),
    "sra_experiment": re.compile(r"\b[SED]RX\d{3,}\b", re.IGNORECASE),
    "sra_study": re.compile(r"\b[SED]RP\d{3,}\b", re.IGNORECASE),
    "bioproject": re.compile(r"\bPRJ(?:NA|EB|DB)\d+\b", re.IGNORECASE),
    "arrayexpress": re.compile(r"\bE-[A-Z0-9]{3,12}-\d+\b", re.IGNORECASE),
    "zenodo_doi": re.compile(r"\b10\.5281/zenodo\.\d+\b", re.IGNORECASE),
}

SECTION_PATTERNS = {
    "abstract": re.compile(r"\babstract\b", re.IGNORECASE),
    "methods": re.compile(r"\b(?:materials?\s+and\s+methods?|methods?)\b", re.IGNORECASE),
    "results": re.compile(r"\bresults?\b", re.IGNORECASE),
    "data_availability": re.compile(r"\b(?:data availability|availability of data|data access)\b", re.IGNORECASE),
    "supplementary": re.compile(r"\b(?:supplementary|supplemental)\b", re.IGNORECASE),
}

DATA_KEYWORDS = (
    "data availability",
    "gse",
    "gsm",
    "sra",
    "arrayexpress",
    "expression atlas",
    "single cell expression atlas",
    "bioproject",
    "zenodo",
    "e-mtab",
    "ebi",
    "cellxgene",
    "humancellatlas",
    "single cell portal",
    "download",
    "count matrix",
    "fastq",
    "10x genomics",
    "raw data",
    "processed data",
)

DATA_HOST_HINTS = (
    "ncbi.nlm.nih.gov",
    "ftp.ncbi.nlm.nih.gov",
    "zenodo.org",
    "cellxgene.cziscience.com",
    "data.humancellatlas.org",
    "explore.data.humancellatlas.org",
    "ebi.ac.uk",
    "figshare.com",
    "dropbox.com",
    "drive.google.com",
)


def _int_from_env(name: str, default: int, minimum: int = 64_000, maximum: int = 20_000_000) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw.strip())
    except Exception:
        return default
    return max(minimum, min(maximum, value))


HTML_FETCH_MAX_BYTES = _int_from_env("CYTOBRIDGE_PAPER_INTAKE_MAX_PAGE_BYTES", DEFAULT_HTML_FETCH_MAX_BYTES)
TEXTUAL_CONTENT_TYPES = (
    "application/html",
    "application/json",
    "application/ld+json",
    "application/xhtml+xml",
    "application/xml",
    "text/",
)
BINARY_MAGIC_PREFIXES = (
    b"\x1f\x8b",  # gzip
    b"PK\x03\x04",  # zip
    b"\x89HDF",  # hdf5
    b"\x89PNG",
    b"%PDF",
)


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


def _read_pdf_with_pypdf(pdf_path: Path) -> Dict[str, object]:
    reader = PdfReader(str(pdf_path))
    pages: List[str] = []
    links: List[str] = []
    title = ""
    try:
        title = str((reader.metadata or {}).get("/Title") or "").strip()
    except Exception:
        title = ""
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")
        try:
            annots = page.get("/Annots") or []
            for annot_ref in annots:
                annot = annot_ref.get_object()
                action = annot.get("/A") or {}
                uri = action.get("/URI")
                if uri:
                    links.append(str(uri))
        except Exception:
            continue
    return {
        "text": "\n\n".join(pages),
        "links": _dedupe(links),
        "page_count": len(reader.pages),
        "title": title,
    }


def _read_pdf_with_pdfplumber(pdf_path: Path) -> str:
    if pdfplumber is None:
        return ""
    parts: List[str] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            try:
                parts.append(page.extract_text() or "")
            except Exception:
                continue
    return "\n\n".join(parts)


def _extract_urls(text: str) -> List[str]:
    return _dedupe(m.group(0).rstrip(".,);") for m in URL_RE.finditer(text or ""))


def _extract_doi(text: str) -> str:
    match = DOI_RE.search(text or "")
    return match.group(0).rstrip(".,);") if match else ""


def extract_accessions(text: str) -> Dict[str, List[str]]:
    payload: Dict[str, List[str]] = {}
    for key, pattern in ACCESSION_PATTERNS.items():
        payload[key] = _dedupe(match.group(0).upper() for match in pattern.finditer(text or ""))
    payload["uuid"] = _dedupe(match.group(0) for match in UUID_RE.finditer(text or ""))
    payload["doi"] = _dedupe(match.group(0).rstrip(".,);") for match in DOI_RE.finditer(text or ""))
    return payload


def _section_flags(text: str) -> Dict[str, bool]:
    lowered = text or ""
    return {name: bool(pattern.search(lowered)) for name, pattern in SECTION_PATTERNS.items()}


def _completeness_score(text: str, page_count: int, flags: Dict[str, bool]) -> float:
    score = 0.0
    score += min(page_count, 8) / 8.0 * 0.35
    score += sum(1.0 for value in flags.values() if value) / max(len(flags), 1) * 0.45
    score += min(len(text.split()), 6000) / 6000.0 * 0.20
    return round(min(score, 1.0), 3)


def _candidate_article_urls(explicit_urls: Sequence[str], doi: str) -> List[str]:
    urls: List[str] = []
    doi_text = (doi or "").strip().lower()
    doi_variants = {doi_text}
    if doi_text:
        doi_variants.add(quote(doi_text, safe="").lower())
        doi_variants.add(quote(doi_text, safe="/").lower())
    for url in explicit_urls:
        lowered = url.lower()
        if lowered.endswith(".pdf"):
            continue
        if any(host in lowered for host in DATA_HOST_HINTS):
            continue
        if doi_text:
            if any(variant and variant in lowered for variant in doi_variants):
                urls.append(url)
            continue
        if "doi.org/" in lowered or "/article" in lowered or "/full" in lowered or "/abs/" in lowered:
            urls.append(url)
    if doi:
        urls.insert(0, f"https://doi.org/{doi}")
    return _dedupe(urls)


def _looks_textual_content_type(content_type: str) -> bool:
    lowered = (content_type or "").split(";", 1)[0].strip().lower()
    if not lowered:
        return True
    return any(lowered == item.rstrip("/") or lowered.startswith(item) for item in TEXTUAL_CONTENT_TYPES)


def _looks_binary_sample(sample: bytes) -> bool:
    if not sample:
        return False
    if any(sample.startswith(prefix) for prefix in BINARY_MAGIC_PREFIXES):
        return True
    head = sample[:2048]
    if b"\x00" in head:
        return True
    controls = sum(1 for byte in head if byte < 32 and byte not in {9, 10, 13})
    return controls / max(len(head), 1) > 0.05


def fetch_limited_text_response(
    url: str,
    timeout: object = 60.0,
    max_bytes: int = HTML_FETCH_MAX_BYTES,
) -> Tuple[str, str, Dict[str, str], bool]:
    headers = {"User-Agent": USER_AGENT}
    with httpx.Client(follow_redirects=True, timeout=timeout, headers=headers) as client:
        with client.stream("GET", url) as response:
            response.raise_for_status()
            content_type = response.headers.get("content-type", "")
            if content_type and not _looks_textual_content_type(content_type):
                raise ValueError(
                    "Refusing to parse non-text response as a web page "
                    f"(url={url}, content-type={content_type})"
                )

            chunks: List[bytes] = []
            total = 0
            truncated = False
            for chunk in response.iter_bytes():
                if not chunk:
                    continue
                remaining = max_bytes - total
                if remaining <= 0:
                    truncated = True
                    break
                if len(chunk) > remaining:
                    chunks.append(chunk[:remaining])
                    total += remaining
                    truncated = True
                    break
                chunks.append(chunk)
                total += len(chunk)

            body_bytes = b"".join(chunks)
            if _looks_binary_sample(body_bytes):
                raise ValueError(
                    "Refusing to parse binary-looking response as a web page "
                    f"(url={url}, content-type={content_type or 'unknown'})"
                )
            encoding = response.encoding or "utf-8"
            return body_bytes.decode(encoding, errors="replace"), str(response.url), dict(response.headers), truncated


def fetch_html_document(url: str, timeout: object = 60.0) -> Dict[str, object]:
    body, final_url, response_headers, truncated = fetch_limited_text_response(url, timeout=timeout)
    if BeautifulSoup is not None:
        soup = BeautifulSoup(body, "html.parser")
        links = []
        for anchor in soup.find_all("a", href=True):
            href = anchor.get("href", "")
            if not href:
                continue
            links.append(urljoin(final_url, href))
        title = ""
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
        text = "\n".join(chunk.strip() for chunk in soup.stripped_strings if chunk.strip())
    else:
        links = [
            urljoin(final_url, href.rstrip(".,);"))
            for href in re.findall(r"""href=["']([^"']+)["']""", body, flags=re.IGNORECASE)
        ]
        title_match = re.search(r"<title[^>]*>(.*?)</title>", body, flags=re.IGNORECASE | re.DOTALL)
        title = re.sub(r"\s+", " ", title_match.group(1)).strip() if title_match else ""
        text = re.sub(r"<[^>]+>", " ", body)
        text = re.sub(r"\s+", " ", text).strip()
    return {
        "url": final_url,
        "title": title,
        "content_type": response_headers.get("content-type", ""),
        "text": text,
        "links": _dedupe(links),
        "truncated": truncated,
    }


def _relevant_sentences(*texts: str, limit: int = 24) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for text in texts:
        if not text:
            continue
        normalized = re.sub(r"\s+", " ", text)
        for sentence in re.split(r"(?<=[.!?])\s+", normalized):
            candidate = sentence.strip()
            if len(candidate) < 30:
                continue
            lowered = candidate.lower()
            if not any(keyword in lowered for keyword in DATA_KEYWORDS):
                continue
            if candidate in seen:
                continue
            seen.add(candidate)
            out.append(candidate)
            if len(out) >= limit:
                return out
    return out


def _write_text(path: Path, text: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return str(path)


def inspect_manuscript(
    input_source: str,
    output_dir: str,
    fetch_online_if_needed: bool = True,
    article_url: Optional[str] = None,
) -> ManuscriptInspection:
    out_dir = Path(output_dir).expanduser().resolve() / "paper_intake"
    out_dir.mkdir(parents=True, exist_ok=True)

    primary_text = ""
    discovered_urls: List[str] = []
    title = ""
    page_count = 0
    input_type = "url" if str(input_source).startswith(("http://", "https://")) else "file"
    notes: List[str] = []

    if input_type == "url":
        html_doc = fetch_html_document(str(input_source))
        primary_text = str(html_doc.get("text") or "")
        discovered_urls.extend(html_doc.get("links") or [])
        discovered_urls.append(str(html_doc.get("url") or input_source))
        title = str(html_doc.get("title") or "")
    else:
        source_path = Path(input_source).expanduser().resolve()
        if not source_path.exists():
            raise FileNotFoundError(f"Input source does not exist: {source_path}")
        suffix = source_path.suffix.lower()
        if suffix == ".pdf":
            pdf_info = _read_pdf_with_pypdf(source_path)
            primary_text = str(pdf_info["text"])
            if len(primary_text.split()) < 500:
                fallback_text = _read_pdf_with_pdfplumber(source_path)
                if len(fallback_text) > len(primary_text):
                    primary_text = fallback_text
                    notes.append("Used pdfplumber fallback because PyPDF text extraction was sparse.")
            discovered_urls.extend(pdf_info["links"])  # type: ignore[arg-type]
            title = str(pdf_info.get("title") or "")
            page_count = int(pdf_info.get("page_count") or 0)
        else:
            primary_text = source_path.read_text(encoding="utf-8")
            title = source_path.stem
    discovered_urls.extend(_extract_urls(primary_text))

    doi = _extract_doi(primary_text)
    primary_accessions = extract_accessions(primary_text)
    flags = _section_flags(primary_text)
    completeness = _completeness_score(primary_text, page_count or 1, flags)
    article_candidates = _candidate_article_urls(discovered_urls, doi)
    if article_url:
        article_candidates = _dedupe([article_url] + article_candidates)

    article_text = ""
    should_fetch_article = False
    if fetch_online_if_needed and article_candidates:
        should_fetch_article = (
            bool(article_url)
            or completeness < 0.72
            or not flags.get("data_availability")
            or not primary_accessions.get("zenodo_doi")
        )

    if should_fetch_article:
        for candidate in article_candidates[:3]:
            try:
                html_doc = fetch_html_document(candidate)
            except Exception as exc:
                notes.append(f"Failed to fetch article candidate {candidate}: {exc}")
                continue
            fetched_text = str(html_doc.get("text") or "")
            if len(fetched_text.split()) < 600:
                continue
            article_text = fetched_text
            discovered_urls.extend(html_doc.get("links") or [])
            discovered_urls.append(str(html_doc.get("url") or candidate))
            if not title:
                title = str(html_doc.get("title") or "")
            notes.append(f"Fetched online article content from {html_doc.get('url')}.")
            break

    merged_text = primary_text
    if article_text:
        merged_text += "\n\n" + article_text

    accessions = extract_accessions(merged_text)
    relevant = _relevant_sentences(primary_text, article_text)

    primary_text_path = _write_text(out_dir / "manuscript_text.txt", primary_text)
    article_text_path = ""
    if article_text:
        article_text_path = _write_text(out_dir / "article_text.txt", article_text)

    inspection = ManuscriptInspection(
        input_source=str(input_source),
        input_type=input_type,
        title=title,
        doi=doi,
        page_count=page_count,
        completeness_score=completeness,
        section_flags=flags,
        discovered_urls=_dedupe(discovered_urls),
        candidate_article_urls=article_candidates,
        accessions=accessions,
        relevant_sentences=relevant,
        primary_text_path=primary_text_path,
        article_text_path=article_text_path,
        notes=notes,
    )

    summary_lines = [
        "# Paper Intake Summary",
        "",
        f"- Input source: `{inspection.input_source}`",
        f"- Input type: `{inspection.input_type}`",
        f"- Title: {inspection.title or '(unknown)'}",
        f"- DOI: {inspection.doi or '(not detected)'}",
        f"- PDF pages: {inspection.page_count}",
        f"- Completeness score: {inspection.completeness_score}",
        f"- Sections detected: {json.dumps(inspection.section_flags, ensure_ascii=False)}",
        "",
        "## Accessions",
        "",
    ]
    for key, values in inspection.accessions.items():
        if values:
            summary_lines.append(f"- `{key}`: {', '.join(values[:20])}")
    summary_lines.extend(["", "## Relevant Sentences", ""])
    for sentence in inspection.relevant_sentences:
        summary_lines.append(f"- {sentence}")
    summary_lines.extend(["", "## Candidate Article URLs", ""])
    for url in inspection.candidate_article_urls:
        summary_lines.append(f"- {url}")
    if inspection.notes:
        summary_lines.extend(["", "## Notes", ""])
        for note in inspection.notes:
            summary_lines.append(f"- {note}")

    inspection.summary_path = _write_text(out_dir / "manuscript_summary.md", "\n".join(summary_lines))
    inspection.manifest_path = str(out_dir / "manuscript_manifest.json")
    Path(inspection.manifest_path).write_text(
        json.dumps(inspection.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return inspection
