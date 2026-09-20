from __future__ import annotations

import gzip
import json
import os
import re
import shutil
import subprocess
import tarfile
import time
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

import anndata as ad
import httpx
import numpy as np
import pandas as pd
from scipy import io as scipy_io
from scipy import sparse

from .manuscript import USER_AGENT
from .models import DataAsset
from .providers import classify_reference, resolve_geo_accession


class PaperIntakeError(RuntimeError):
    """Raised when the paper-to-AnnData pipeline cannot continue automatically."""


def _int_from_env(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(minimum, min(maximum, value))


def _bool_from_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    if raw in {"1", "true", "yes", "y", "on"}:
        return True
    if raw in {"0", "false", "no", "n", "off"}:
        return False
    return default


DOWNLOAD_TIMEOUT = httpx.Timeout(15.0, read=60.0, write=60.0, pool=60.0)
DOWNLOAD_PROGRESS_INTERVAL_SECONDS = 15.0
DOWNLOAD_PROGRESS_INTERVAL_BYTES = 64 * 1024 * 1024
DOWNLOAD_CHUNK_SIZE = _int_from_env(
    "CYTOBRIDGE_PAPER_INTAKE_DOWNLOAD_CHUNK_BYTES",
    1024 * 1024,
    64 * 1024,
    16 * 1024 * 1024,
)
DOWNLOAD_RESUME_ENABLED = _bool_from_env("CYTOBRIDGE_PAPER_INTAKE_DOWNLOAD_RESUME", True)
DOWNLOAD_EXTERNAL_MODE = os.environ.get("CYTOBRIDGE_PAPER_INTAKE_DOWNLOADER", "auto").strip().lower()
DOWNLOAD_CONNECTIONS = _int_from_env("CYTOBRIDGE_PAPER_INTAKE_DOWNLOAD_CONNECTIONS", 4, 1, 8)
DOWNLOAD_EXTERNAL_MIN_BYTES = _int_from_env(
    "CYTOBRIDGE_PAPER_INTAKE_EXTERNAL_MIN_BYTES",
    64 * 1024 * 1024,
    1024 * 1024,
    1024 * 1024 * 1024,
)
DOWNLOAD_PARALLEL_SAFE_HOSTS = (
    "ftp.ncbi.nlm.nih.gov",
    "ftp.ebi.ac.uk",
    "ftp.sra.ebi.ac.uk",
)


def _import_scanpy():
    import scanpy as sc

    return sc


def _safe_name(text: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in (text or "asset"))
    cleaned = cleaned.strip("._")
    return cleaned or "asset"


def _download_mode() -> str:
    mode = DOWNLOAD_EXTERNAL_MODE or "auto"
    if mode not in {"auto", "httpx", "curl", "wget", "aria2c", "off"}:
        return "auto"
    return "httpx" if mode == "off" else mode


def _format_rate(bytes_per_second: float) -> str:
    if bytes_per_second >= 1024 * 1024:
        return f"{bytes_per_second / (1024 * 1024):.2f} MiB/s"
    if bytes_per_second >= 1024:
        return f"{bytes_per_second / 1024:.1f} KiB/s"
    return f"{bytes_per_second:.0f} B/s"


def _host_is_parallel_safe(url: str) -> bool:
    host = urlparse(url).netloc.lower()
    return any(host == item or host.endswith(f".{item}") for item in DOWNLOAD_PARALLEL_SAFE_HOSTS)


def _remote_file_info(url: str, progress_callback: Optional[Callable[[str], None]] = None) -> Dict[str, object]:
    headers = {"User-Agent": USER_AGENT}
    try:
        with httpx.Client(follow_redirects=True, timeout=DOWNLOAD_TIMEOUT, headers=headers) as client:
            response = client.head(url)
            response.raise_for_status()
    except Exception as exc:
        if progress_callback is not None:
            progress_callback(f"[download] remote size probe unavailable url={url} error={exc}")
        return {"total_bytes": 0, "accept_ranges": False}

    total_bytes = 0
    try:
        total_bytes = int(response.headers.get("content-length") or 0)
    except ValueError:
        total_bytes = 0
    accept_ranges = "bytes" in str(response.headers.get("accept-ranges") or "").lower()
    return {
        "total_bytes": total_bytes,
        "accept_ranges": accept_ranges,
        "final_url": str(response.url),
    }


def _target_complete(target: Path, total_bytes: int) -> bool:
    return target.exists() and total_bytes > 0 and target.stat().st_size >= total_bytes


def _external_download_command(
    url: str,
    target: Path,
    *,
    total_bytes: int,
) -> Optional[List[str]]:
    mode = _download_mode()
    if mode == "httpx":
        return None
    if mode == "auto" and (total_bytes <= 0 or total_bytes < DOWNLOAD_EXTERNAL_MIN_BYTES):
        return None

    parallel_safe = _host_is_parallel_safe(url)
    connections = DOWNLOAD_CONNECTIONS if parallel_safe else 1
    aria2 = shutil.which("aria2c")
    if aria2 and mode in {"auto", "aria2c"} and (mode == "aria2c" or connections > 1):
        return [
            aria2,
            "--continue=true",
            "--allow-overwrite=true",
            "--auto-file-renaming=false",
            "--file-allocation=none",
            "--max-tries=5",
            "--retry-wait=3",
            "--summary-interval=0",
            "--console-log-level=warn",
            f"--user-agent={USER_AGENT}",
            f"--max-connection-per-server={connections}",
            f"--split={connections}",
            "--min-split-size=1M",
            "--dir",
            str(target.parent),
            "--out",
            target.name,
            url,
        ]

    curl = shutil.which("curl")
    if curl and mode in {"auto", "curl"}:
        return [
            curl,
            "--location",
            "--fail",
            "--silent",
            "--show-error",
            "--retry",
            "5",
            "--retry-delay",
            "3",
            "--connect-timeout",
            "15",
            "--continue-at",
            "-",
            "--user-agent",
            USER_AGENT,
            "--output",
            str(target),
            url,
        ]

    wget = shutil.which("wget")
    if wget and mode in {"auto", "wget"}:
        return [
            wget,
            "--quiet",
            "--continue",
            "--tries=5",
            "--waitretry=3",
            "--user-agent",
            USER_AGENT,
            "--output-document",
            str(target),
            url,
        ]

    return None


def _run_external_download(
    command: List[str],
    target: Path,
    *,
    total_bytes: int,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tool = Path(command[0]).name
    starting_bytes = target.stat().st_size if target.exists() else 0
    if progress_callback is not None:
        size_text = str(total_bytes) if total_bytes > 0 else "unknown"
        progress_callback(
            f"[download] external start tool={tool} target={target} existing_bytes={starting_bytes} bytes={size_text}"
        )

    process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    start = time.monotonic()
    last_report = start
    while True:
        return_code = process.poll()
        now = time.monotonic()
        if progress_callback is not None and now - last_report >= DOWNLOAD_PROGRESS_INTERVAL_SECONDS:
            current_bytes = target.stat().st_size if target.exists() else 0
            rate = (current_bytes - starting_bytes) / max(now - start, 0.001)
            if total_bytes > 0:
                progress_callback(
                    f"[download] external progress tool={tool} target={target.name} "
                    f"bytes={current_bytes}/{total_bytes} rate={_format_rate(rate)}"
                )
            else:
                progress_callback(
                    f"[download] external progress tool={tool} target={target.name} "
                    f"bytes={current_bytes} rate={_format_rate(rate)}"
                )
            last_report = now
        if return_code is not None:
            break
        time.sleep(1.0)

    stderr = ""
    try:
        _, stderr = process.communicate(timeout=5)
    except Exception:
        stderr = ""
    if return_code != 0:
        raise PaperIntakeError(f"External downloader {tool} failed with code {return_code}: {stderr.strip()[:500]}")
    if not target.exists():
        raise PaperIntakeError(f"External downloader {tool} finished but target file was not created: {target}")
    if total_bytes > 0 and target.stat().st_size < total_bytes:
        raise PaperIntakeError(
            f"External downloader {tool} produced a short file: {target.stat().st_size}/{total_bytes} bytes"
        )
    if progress_callback is not None:
        elapsed = max(time.monotonic() - start, 0.001)
        written = target.stat().st_size - starting_bytes
        progress_callback(
            f"[download] external completed tool={tool} target={target} "
            f"bytes={target.stat().st_size} rate={_format_rate(written / elapsed)}"
        )


def _try_external_download(
    url: str,
    target: Path,
    *,
    total_bytes: int,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> bool:
    command = _external_download_command(url, target, total_bytes=total_bytes)
    if not command:
        return False
    try:
        _run_external_download(command, target, total_bytes=total_bytes, progress_callback=progress_callback)
        return True
    except Exception as exc:
        if progress_callback is not None:
            progress_callback(f"[download] external downloader unavailable/failed; falling back to httpx: {exc}")
        return False


def _parse_content_range_total(value: str) -> int:
    match = re.search(r"/(\d+)\s*$", value or "")
    if not match:
        return 0
    try:
        return int(match.group(1))
    except ValueError:
        return 0


def _download_with_httpx(
    url: str,
    target: Path,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    info = _remote_file_info(url, progress_callback=progress_callback)
    probed_total = int(info.get("total_bytes") or 0)
    if _target_complete(target, probed_total):
        if progress_callback is not None:
            progress_callback(f"[download] reuse completed target={target} bytes={target.stat().st_size}")
        return target
    if _try_external_download(url, target, total_bytes=probed_total, progress_callback=progress_callback):
        return target

    headers = {"User-Agent": USER_AGENT}
    existing_bytes = target.stat().st_size if target.exists() else 0
    if DOWNLOAD_RESUME_ENABLED and existing_bytes > 0:
        headers["Range"] = f"bytes={existing_bytes}-"
    with httpx.Client(follow_redirects=True, timeout=DOWNLOAD_TIMEOUT, headers=headers) as client:
        with client.stream("GET", url) as response:
            if response.status_code == 416 and existing_bytes > 0:
                if progress_callback is not None:
                    progress_callback(f"[download] server reports range complete target={target} bytes={existing_bytes}")
                return target
            response.raise_for_status()
            append_mode = response.status_code == 206 and existing_bytes > 0
            if not append_mode:
                existing_bytes = 0
            total_bytes = _parse_content_range_total(str(response.headers.get("content-range") or "")) or probed_total
            if total_bytes <= 0:
                try:
                    content_length = int(response.headers.get("content-length") or 0)
                except ValueError:
                    content_length = 0
                total_bytes = content_length + existing_bytes if content_length > 0 else 0
            if progress_callback is not None:
                size_text = str(total_bytes) if total_bytes > 0 else "unknown"
                resume_text = f" resume_from={existing_bytes}" if existing_bytes else ""
                progress_callback(f"[download] start url={url} target={target} bytes={size_text}{resume_text}")
            mode = "ab" if append_mode else "wb"
            with target.open(mode, buffering=max(DOWNLOAD_CHUNK_SIZE, 1024 * 1024)) as handle:
                downloaded = existing_bytes
                session_start_bytes = existing_bytes
                next_report = downloaded + DOWNLOAD_PROGRESS_INTERVAL_BYTES
                start_time = time.monotonic()
                last_report = time.monotonic()
                for chunk in response.iter_bytes(chunk_size=DOWNLOAD_CHUNK_SIZE):
                    if not chunk:
                        continue
                    handle.write(chunk)
                    downloaded += len(chunk)
                    now = time.monotonic()
                    should_report = (
                        downloaded >= next_report
                        or now - last_report >= DOWNLOAD_PROGRESS_INTERVAL_SECONDS
                    )
                    if progress_callback is not None and should_report:
                        rate = (downloaded - session_start_bytes) / max(now - start_time, 0.001)
                        if total_bytes > 0:
                            progress_callback(
                                f"[download] progress target={target.name} bytes={downloaded}/{total_bytes} "
                                f"rate={_format_rate(rate)}"
                            )
                        else:
                            progress_callback(
                                f"[download] progress target={target.name} bytes={downloaded} rate={_format_rate(rate)}"
                            )
                        next_report = downloaded + DOWNLOAD_PROGRESS_INTERVAL_BYTES
                        last_report = now
            if progress_callback is not None:
                elapsed = max(time.monotonic() - start_time, 0.001)
                progress_callback(
                    f"[download] completed target={target} bytes={downloaded} "
                    f"rate={_format_rate((downloaded - session_start_bytes) / elapsed)}"
                )
    return target


DOWNLOAD_BINARY_MAGIC_PREFIXES = (
    b"\x1f\x8b",
    b"PK\x03\x04",
    b"\x89HDF",
    b"BAM\x01",
)
DOWNLOAD_ERROR_TEXT_HINTS = (
    "not found",
    "access denied",
    "forbidden",
    "unauthorized",
    "temporarily unavailable",
    "service unavailable",
    "placeholder",
    "no data",
    "error",
)


def _read_download_sample(path: Path, max_bytes: int = 8192) -> bytes:
    with path.open("rb") as handle:
        return handle.read(max_bytes)


def _looks_like_html_or_error_payload(sample: bytes) -> bool:
    if not sample:
        return True
    if any(sample.startswith(prefix) for prefix in DOWNLOAD_BINARY_MAGIC_PREFIXES):
        return False
    text = sample.decode("utf-8", errors="ignore").lstrip()
    lowered = re.sub(r"\s+", " ", text[:4000].lower())
    if lowered.startswith(("<!doctype html", "<html", "<?xml")):
        return True
    if any(token in lowered[:1000] for token in ("<html", "<head", "<body", "</html>")):
        return True
    if lowered.startswith("{") and any(token in lowered[:1000] for token in DOWNLOAD_ERROR_TEXT_HINTS):
        return True
    if len(sample) < 2048 and any(token in lowered for token in DOWNLOAD_ERROR_TEXT_HINTS):
        return True
    return False


def _validate_downloaded_asset(asset: DataAsset, path: Path, source_url: str = "") -> Path:
    if path.is_dir() or asset.asset_kind == "page":
        return path
    if not path.exists():
        raise PaperIntakeError(f"Downloaded asset was not created: {path}")
    size = path.stat().st_size
    if size <= 0:
        raise PaperIntakeError(f"Downloaded asset is empty: {path}")
    sample = _read_download_sample(path)
    if _looks_like_html_or_error_payload(sample):
        raise PaperIntakeError(
            "Downloaded asset looks like an HTML/error placeholder rather than data: "
            f"url={source_url or asset.url}, path={path}, bytes={size}"
        )
    return path


def _download_and_validate(
    asset: DataAsset,
    url: str,
    target: Path,
    *,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> Path:
    downloaded = _download_with_httpx(url, target, progress_callback=progress_callback)
    return _validate_downloaded_asset(asset, downloaded, source_url=url)


def _canonical_mex_name(filename: str) -> Optional[str]:
    lower = (filename or "").lower()
    if lower.endswith((".mtx", ".mtx.gz")) and any(
        token in lower for token in ("matrix.mtx", "counts.mtx", "normed_count", "normed_counts", "expression")
    ):
        return "matrix.mtx.gz" if lower.endswith(".gz") else "matrix.mtx"
    if "features.tsv" in lower:
        return "features.tsv.gz" if lower.endswith(".gz") else "features.tsv"
    if "feature_names" in lower:
        return "features.tsv.gz" if lower.endswith(".gz") else "features.tsv"
    if "genes.tsv" in lower:
        return "genes.tsv.gz" if lower.endswith(".gz") else "genes.tsv"
    if "gene_names" in lower:
        return "genes.tsv.gz" if lower.endswith(".gz") else "genes.tsv"
    if "barcodes.tsv" in lower:
        return "barcodes.tsv.gz" if lower.endswith(".gz") else "barcodes.tsv"
    if "cell_barcodes" in lower:
        return "barcodes.tsv.gz" if lower.endswith(".gz") else "barcodes.tsv"
    return None


def _looks_like_metadata_name(name: str) -> bool:
    lowered = (name or "").lower()
    return any(
        token in lowered
        for token in (
            "metadata",
            "meta",
            "annotation",
            "annot",
            "celltype",
            "cell_type",
            "cluster",
            "sample",
            "donor",
            "barcode_annotation",
            "library_name",
            "library_names",
            "readme",
        )
    )


def _looks_like_expression_mex(filename: str) -> bool:
    lowered = (filename or "").lower()
    return lowered.endswith((".mtx", ".mtx.gz")) and any(
        token in lowered
        for token in (
            "matrix.mtx",
            "count",
            "counts",
            "expression",
            "expr",
            "umi",
        )
    )


def _looks_like_clone_or_lineage_matrix(filename: str) -> bool:
    lowered = (filename or "").lower()
    return any(
        token in lowered
        for token in (
            "clone_matrix",
            "clone",
            "lineage",
            "barcode_matrix",
        )
    )


def _score_mex_matrix_candidate(asset: DataAsset, preferred_filename: str = "") -> float:
    filename = asset.filename or Path(urlparse(asset.url).path).name
    lowered = filename.lower()
    score = float(asset.score)
    if preferred_filename and filename == preferred_filename:
        score += 40.0
    if _looks_like_expression_mex(filename):
        score += 20.0
    if any(token in lowered for token in ("normed_count", "normed_counts", "counts", "expression")):
        score += 8.0
    if _looks_like_clone_or_lineage_matrix(filename):
        score -= 28.0
    return score


def _is_metadata_sidecar_asset(asset: DataAsset) -> bool:
    filename = asset.filename or Path(urlparse(asset.url).path).name
    lowered = filename.lower()
    if not lowered.endswith((".csv", ".csv.gz", ".tsv", ".tsv.gz", ".txt", ".txt.gz")):
        return False
    return _looks_like_metadata_name(lowered)


def _download_geo_mex_bundle(
    asset: DataAsset,
    download_dir: Path,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> Optional[Path]:
    accession = str(asset.metadata.get("geo_accession") or "").strip().upper()
    if not accession:
        return None

    siblings = resolve_geo_accession(accession)
    matrix_candidates: List[DataAsset] = []
    selected: Dict[str, DataAsset] = {}
    metadata_sidecars: Dict[str, DataAsset] = {}
    preferred_filename = asset.filename or Path(urlparse(asset.url).path).name
    for sibling in siblings:
        sibling_name = sibling.filename or Path(urlparse(sibling.url).path).name
        if str(sibling_name).lower().endswith((".mtx", ".mtx.gz")):
            matrix_candidates.append(sibling)
        canonical = _canonical_mex_name(sibling.filename)
        if canonical and not canonical.startswith("matrix.mtx"):
            previous = selected.get(canonical)
            if previous is None or sibling.score > previous.score:
                selected[canonical] = sibling
            continue
        if _is_metadata_sidecar_asset(sibling):
            filename = sibling.filename or Path(urlparse(sibling.url).path).name or "metadata.tsv"
            target_name = _safe_name(filename)
            previous = metadata_sidecars.get(target_name)
            if previous is None or sibling.score > previous.score:
                metadata_sidecars[target_name] = sibling

    matrix_asset = None
    if matrix_candidates:
        matrix_asset = max(
            matrix_candidates,
            key=lambda candidate: _score_mex_matrix_candidate(candidate, preferred_filename=preferred_filename),
        )

    has_matrix = matrix_asset is not None
    has_barcodes = any(name.startswith("barcodes.tsv") for name in selected)
    has_features = any(name.startswith("features.tsv") or name.startswith("genes.tsv") for name in selected)
    if not (has_matrix and has_barcodes and has_features):
        return None

    staging = download_dir / _safe_name(accession or asset.filename or "geo_mex_bundle")
    staging.mkdir(parents=True, exist_ok=True)
    matrix_path: Optional[Path] = None
    if matrix_asset is not None:
        original_matrix_name = _safe_name(
            matrix_asset.filename or Path(urlparse(matrix_asset.url).path).name or "matrix.mtx.gz"
        )
        matrix_path = staging / original_matrix_name
        _download_and_validate(matrix_asset, matrix_asset.url, matrix_path, progress_callback=progress_callback)
        canonical_matrix_name = _canonical_mex_name(matrix_asset.filename)
        if canonical_matrix_name:
            canonical_matrix_path = staging / canonical_matrix_name
            if canonical_matrix_path != matrix_path and not canonical_matrix_path.exists():
                try:
                    os.link(matrix_path, canonical_matrix_path)
                except Exception:
                    try:
                        os.symlink(matrix_path, canonical_matrix_path)
                    except Exception:
                        shutil.copy2(matrix_path, canonical_matrix_path)
    for canonical_name, sibling in selected.items():
        target = staging / canonical_name
        _download_and_validate(sibling, sibling.url, target, progress_callback=progress_callback)
    for sidecar_name, sibling in metadata_sidecars.items():
        target = staging / sidecar_name
        _download_and_validate(sibling, sibling.url, target, progress_callback=progress_callback)
    return matrix_path


def download_asset(
    asset: DataAsset,
    download_dir: str,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> Path:
    out_dir = Path(download_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if asset.provider == "geo" and asset.file_format == "10x_mtx":
        bundle_matrix = _download_geo_mex_bundle(asset, out_dir, progress_callback=progress_callback)
        if bundle_matrix is not None:
            return bundle_matrix

    urls = list(asset.metadata.get("raw_urls") or [])
    if urls:
        staging = out_dir / _safe_name(asset.metadata.get("run_accession") or asset.filename or "raw_bundle")
        staging.mkdir(parents=True, exist_ok=True)
        for url in urls:
            filename = Path(urlparse(url).path).name or _safe_name(asset.filename)
            _download_and_validate(asset, url, staging / filename, progress_callback=progress_callback)
        return staging

    filename = asset.filename or Path(urlparse(asset.url).path).name or "downloaded_asset"
    filename = _safe_name(filename)
    return _download_and_validate(asset, asset.url, out_dir / filename, progress_callback=progress_callback)


def _extract_archive(path: Path, extract_dir: Path) -> Path:
    extract_dir.mkdir(parents=True, exist_ok=True)
    lower = path.name.lower()
    if lower.endswith(".zip"):
        with zipfile.ZipFile(path) as archive:
            archive.extractall(extract_dir)
    elif lower.endswith((".tar", ".tar.gz", ".tgz")):
        with tarfile.open(path) as archive:
            archive.extractall(extract_dir)
    else:
        raise PaperIntakeError(f"Unsupported archive format: {path}")
    return extract_dir


def _looks_like_metadata(path: Path) -> bool:
    return _looks_like_metadata_name(path.name)


def _classify_local(path: Path) -> DataAsset:
    return classify_reference(path.as_uri(), source="local")


def _local_candidates(root: Path) -> List[Path]:
    if root.is_file():
        return [root]
    return sorted(path for path in root.rglob("*") if path.is_file())


def _score_local_candidate(path: Path) -> float:
    asset = _classify_local(path)
    score = asset.score
    if _looks_like_metadata(path):
        score -= 20.0
    return score


def _pick_best_local(root: Path) -> Optional[Path]:
    files = _local_candidates(root)
    if not files:
        return None
    return sorted(files, key=_score_local_candidate, reverse=True)[0]


def _read_table(path: Path) -> pd.DataFrame:
    compression = "gzip" if path.suffix.lower() == ".gz" or path.name.lower().endswith((".csv.gz", ".tsv.gz", ".txt.gz")) else None
    suffix = path.name.lower()
    sep = "\t" if suffix.endswith((".tsv", ".tsv.gz", ".txt", ".txt.gz")) else ","
    df = pd.read_csv(path, sep=sep, compression=compression)
    if df.shape[1] > 1:
        first = df.columns[0]
        if not pd.api.types.is_numeric_dtype(df[first]):
            df = df.set_index(first)
    return df


def read_sidecar_table(path: str | Path) -> pd.DataFrame:
    """Public wrapper for reading candidate metadata/count tables."""
    return _read_table(Path(path).expanduser().resolve())


def describe_local_inputs(root: str | Path, include_siblings: bool = True) -> List[Dict[str, object]]:
    """Summarize downloaded local files for LLM-guided materialization."""
    root_path = Path(root).expanduser().resolve()
    search_root = root_path
    if include_siblings and root_path.is_file():
        search_root = root_path.parent
    items: List[Dict[str, object]] = []
    for candidate in _local_candidates(search_root):
        try:
            stat = candidate.stat()
        except Exception:
            stat = None
        asset = _classify_local(candidate)
        try:
            relative = str(candidate.relative_to(search_root))
        except Exception:
            relative = candidate.name
        items.append(
            {
                "path": str(candidate),
                "relative_path": relative,
                "name": candidate.name,
                "suffix": candidate.suffix.lower(),
                "size_bytes": int(stat.st_size) if stat is not None else None,
                "asset_kind": asset.asset_kind,
                "file_format": asset.file_format,
                "metadata_like": _looks_like_metadata(candidate),
            }
        )
    return items


def _normalize_obs_key(value: object, mode: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if mode == "identity":
        return text
    if mode == "drop_trailing_dash_numeric":
        return re.sub(r"[-_]\d+$", "", text)
    if mode == "drop_trailing_dot_numeric":
        return re.sub(r"\.\d+$", "", text)
    return text


def _candidate_barcode_columns(frame: pd.DataFrame, barcode_candidates: Optional[List[str]] = None) -> List[str]:
    explicit = [str(item).strip() for item in (barcode_candidates or []) if str(item).strip()]
    discovered = [
        str(column)
        for column in frame.columns
        if any(token in str(column).lower() for token in ("barcode", "cell", "cell_id", "cellid"))
    ]
    ordered: List[str] = []
    seen: set[str] = set()
    for column in explicit + discovered:
        if column in frame.columns and column not in seen:
            seen.add(column)
            ordered.append(column)
    return ordered


def try_merge_obs_table(
    adata: ad.AnnData,
    table_path: str | Path,
    barcode_candidates: Optional[List[str]] = None,
    min_overlap_ratio: float = 0.02,
    min_overlap_count: int = 10,
) -> Dict[str, object]:
    """Merge a sidecar table into `adata.obs` when cell identifiers can be aligned."""
    path = Path(table_path).expanduser().resolve()
    report: Dict[str, object] = {
        "table_path": str(path),
        "merged_columns": [],
        "index_source": "",
        "normalization": "",
        "overlap_count": 0,
        "overlap_ratio": 0.0,
    }
    frame = read_sidecar_table(path)
    if frame.empty:
        report["reason"] = "empty_table"
        return report

    candidates: List[tuple[str, pd.Series, pd.DataFrame]] = [
        ("__index__", pd.Series(frame.index.astype(str), index=frame.index), frame.copy())
    ]
    for column in _candidate_barcode_columns(frame, barcode_candidates=barcode_candidates):
        base = frame.copy()
        candidates.append((column, base[column].astype(str), base.drop(columns=[column], errors="ignore")))

    obs_index = pd.Index(adata.obs_names.astype(str), name="obs_name")
    minimum = max(min_overlap_count, int(min_overlap_ratio * max(adata.n_obs, 1)))
    best: Optional[Dict[str, object]] = None

    for source_name, source_values, value_frame in candidates:
        for normalization in ("identity", "drop_trailing_dash_numeric", "drop_trailing_dot_numeric"):
            normalized_source = source_values.map(lambda item: _normalize_obs_key(item, normalization))
            normalized_obs = obs_index.map(lambda item: _normalize_obs_key(item, normalization))
            if not normalized_source.any():
                continue
            candidate_frame = value_frame.copy()
            candidate_frame.index = normalized_source
            candidate_frame = candidate_frame[~candidate_frame.index.duplicated(keep="first")]
            overlap = pd.Index(normalized_obs).intersection(candidate_frame.index)
            overlap_count = int(len(overlap))
            if overlap_count < minimum:
                continue
            overlap_ratio = overlap_count / max(adata.n_obs, 1)
            score = (overlap_count, overlap_ratio)
            if best is None or score > (best["overlap_count"], best["overlap_ratio"]):
                merged = candidate_frame.reindex(normalized_obs)
                merged.index = adata.obs_names
                best = {
                    "merged": merged,
                    "index_source": source_name,
                    "normalization": normalization,
                    "overlap_count": overlap_count,
                    "overlap_ratio": overlap_ratio,
                }

    if best is None:
        report["reason"] = "insufficient_overlap"
        return report

    merged = best.pop("merged")
    merged_columns: List[str] = []
    for column in merged.columns:
        column_name = str(column)
        adata.obs[column_name] = merged[column].values
        merged_columns.append(column_name)

    report.update(best)
    report["merged_columns"] = merged_columns
    report["time_candidates"] = [
        column
        for column in merged_columns
        if any(token in column.lower() for token in ("time", "day", "stage", "age", "week"))
    ]
    return report


def _barcode_score(labels: Iterable[object]) -> float:
    values = [str(item) for item in list(labels)[:50] if str(item)]
    if not values:
        return 0.0
    good = 0
    for value in values:
        if len(value) >= 8 and any(ch.isdigit() for ch in value) and not value.startswith("ENSG"):
            good += 1
    return good / len(values)


def _gene_score(labels: Iterable[object]) -> float:
    values = [str(item) for item in list(labels)[:50] if str(item)]
    if not values:
        return 0.0
    good = 0
    for value in values:
        if len(value) <= 20 and not value.startswith(("cell_", "barcode_")):
            good += 1
    return good / len(values)


def _dense_table_to_adata(path: Path) -> ad.AnnData:
    df = _read_table(path)
    numeric = df.apply(pd.to_numeric, errors="coerce")
    if numeric.isna().all().all():
        raise PaperIntakeError(f"Could not parse numeric matrix from {path}")
    numeric = numeric.fillna(0.0)

    rows_barcode = _barcode_score(numeric.index)
    cols_barcode = _barcode_score(numeric.columns)
    rows_gene = _gene_score(numeric.index)
    cols_gene = _gene_score(numeric.columns)

    matrix = numeric
    if cols_barcode >= rows_barcode and rows_gene >= cols_gene:
        matrix = numeric.T
    elif rows_barcode < cols_barcode and numeric.shape[0] > numeric.shape[1]:
        matrix = numeric.T

    X = sparse.csr_matrix(matrix.to_numpy(dtype=np.float32))
    adata = ad.AnnData(
        X=X,
        obs=pd.DataFrame(index=pd.Index(matrix.index.astype(str), name="cell_id")),
        var=pd.DataFrame(index=pd.Index(matrix.columns.astype(str), name="gene_id")),
    )
    return adata


def _load_mtx_bundle(matrix_path: Path) -> ad.AnnData:
    bundle_dir = matrix_path.parent
    try:
        sc = _import_scanpy()
        return sc.read_10x_mtx(str(bundle_dir), var_names="gene_symbols", cache=False)
    except Exception:
        pass

    feature_file = None
    for name in (
        "features.tsv.gz",
        "features.tsv",
        "genes.tsv.gz",
        "genes.tsv",
        "gene_names.txt.gz",
        "gene_names.txt",
        "feature_names.txt.gz",
        "feature_names.txt",
    ):
        candidate = bundle_dir / name
        if candidate.exists():
            feature_file = candidate
            break
    barcode_file = None
    for name in (
        "barcodes.tsv.gz",
        "barcodes.tsv",
        "cell_barcodes.txt.gz",
        "cell_barcodes.txt",
    ):
        candidate = bundle_dir / name
        if candidate.exists():
            barcode_file = candidate
            break
    if feature_file is None or barcode_file is None:
        raise PaperIntakeError(f"Missing features/barcodes files for MEX bundle under {bundle_dir}")

    if matrix_path.suffix.lower() == ".gz":
        with gzip.open(matrix_path, "rb") as handle:
            matrix = scipy_io.mmread(handle).tocsr()
    else:
        matrix = scipy_io.mmread(str(matrix_path)).tocsr()

    features = pd.read_csv(feature_file, sep="\t", header=None, compression="infer")
    barcodes = pd.read_csv(barcode_file, sep="\t", header=None, compression="infer")
    var_names = features.iloc[:, 1] if features.shape[1] > 1 else features.iloc[:, 0]
    obs_names = barcodes.iloc[:, 0]

    matrix = matrix.tocsr()
    expected_gene_count = int(len(var_names))
    expected_cell_count = int(len(obs_names))
    if matrix.shape == (expected_gene_count, expected_cell_count):
        X = matrix.T.tocsr()
    elif matrix.shape == (expected_cell_count, expected_gene_count):
        X = matrix.tocsr()
    else:
        raise PaperIntakeError(
            "MEX bundle dimensions do not match the discovered feature/barcode files under "
            f"{bundle_dir}: matrix_shape={matrix.shape}, cells={expected_cell_count}, genes={expected_gene_count}"
        )

    adata = ad.AnnData(
        X=X,
        obs=pd.DataFrame(index=pd.Index(obs_names.astype(str), name="cell_id")),
        var=pd.DataFrame(index=pd.Index(var_names.astype(str), name="gene_id")),
    )
    return adata


def _attach_sidecar_metadata(adata: ad.AnnData, search_root: Path, exclude: Path) -> Optional[str]:
    for candidate in _local_candidates(search_root):
        if candidate == exclude or candidate.suffix.lower() not in {".csv", ".tsv", ".txt", ".gz"}:
            continue
        if not _looks_like_metadata(candidate):
            continue
        try:
            report = try_merge_obs_table(adata, candidate)
        except Exception:
            continue
        if not report.get("merged_columns"):
            continue
        return str(candidate)
    return None


def _load_single_path(path: Path) -> Tuple[ad.AnnData, Dict[str, object]]:
    lower = path.name.lower()
    details: Dict[str, object] = {"selected_path": str(path)}
    if lower.endswith(".h5ad"):
        return ad.read_h5ad(str(path)), details
    if lower.endswith(".loom"):
        sc = _import_scanpy()
        return sc.read_loom(str(path)), details
    if lower.endswith(".h5") and "seurat" not in lower:
        sc = _import_scanpy()
        return sc.read_10x_h5(str(path)), details
    if lower.endswith((".mtx", ".mtx.gz")):
        return _load_mtx_bundle(path), details
    if lower.endswith((".csv", ".csv.gz", ".tsv", ".tsv.gz", ".txt", ".txt.gz")):
        return _dense_table_to_adata(path), details
    raise PaperIntakeError(f"Unsupported processed data format: {path.name}")


def _require_binary(name: str, env_key: Optional[str] = None) -> str:
    if env_key and os.environ.get(env_key):
        candidate = os.environ[env_key]
        if shutil.which(candidate):
            return shutil.which(candidate) or candidate
    found = shutil.which(name)
    if found:
        return found
    raise PaperIntakeError(f"Required executable not found: {name}")


def _run_raw_pipeline(asset: DataAsset, staged_path: Path, work_dir: Path, raw_reference_path: Optional[str]) -> Tuple[ad.AnnData, Dict[str, object]]:
    reference = raw_reference_path or os.environ.get("CYTOBRIDGE_CELLRANGER_REFERENCE")
    if not reference:
        raise PaperIntakeError(
            "Raw sequencing data was detected but no Cell Ranger reference was configured. "
            "Set CYTOBRIDGE_CELLRANGER_REFERENCE or pass raw_reference_path."
        )
    cellranger = _require_binary("cellranger", env_key="CYTOBRIDGE_CELLRANGER_BIN")

    fastq_dir = staged_path
    if staged_path.is_file() and staged_path.suffix.lower() == ".sra":
        fasterq_dump = _require_binary("fasterq-dump", env_key="CYTOBRIDGE_FASTERQ_DUMP_BIN")
        fastq_dir = work_dir / "fastq"
        fastq_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [fasterq_dump, str(staged_path), "-O", str(fastq_dir)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    elif staged_path.is_file():
        fastq_dir = staged_path.parent

    run_id = _safe_name(asset.metadata.get("run_accession") or asset.filename or "paper_raw")
    out_root = work_dir / "cellranger"
    out_root.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            cellranger,
            "count",
            "--id",
            run_id,
            "--fastqs",
            str(fastq_dir),
            "--transcriptome",
            str(reference),
        ],
        cwd=str(out_root),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    outs = out_root / run_id / "outs"
    matrix_h5 = outs / "filtered_feature_bc_matrix.h5"
    if matrix_h5.exists():
        sc = _import_scanpy()
        adata = sc.read_10x_h5(str(matrix_h5))
        return adata, {"selected_path": str(matrix_h5), "raw_pipeline": "cellranger"}

    matrix_dir = outs / "filtered_feature_bc_matrix"
    if matrix_dir.exists():
        sc = _import_scanpy()
        adata = sc.read_10x_mtx(str(matrix_dir), var_names="gene_symbols", cache=False)
        return adata, {"selected_path": str(matrix_dir), "raw_pipeline": "cellranger"}

    raise PaperIntakeError(f"Cell Ranger output not found under {outs}")


def _materialize_staged_path(
    asset: DataAsset,
    staged_path: Path,
    work_root: Path,
    raw_reference_path: Optional[str] = None,
) -> Tuple[ad.AnnData, Dict[str, object]]:
    details: Dict[str, object] = {"downloaded_path": str(staged_path)}

    if asset.asset_kind == "raw":
        adata, raw_details = _run_raw_pipeline(asset, staged_path, work_root / "raw_pipeline", raw_reference_path)
        details.update(raw_details)
        return adata, details

    candidate_path = staged_path
    if staged_path.is_file() and staged_path.name.lower().endswith((".zip", ".tar", ".tar.gz", ".tgz")):
        extract_dir = work_root / "extracted" / _safe_name(staged_path.stem)
        candidate_path = _extract_archive(staged_path, extract_dir)
        details["archive_extract_dir"] = str(candidate_path)

    if candidate_path.is_file() and candidate_path.name.lower().endswith(".h5ad.gz"):
        decompressed_dir = work_root / "decompressed"
        decompressed_dir.mkdir(parents=True, exist_ok=True)
        decompressed_path = decompressed_dir / candidate_path.name[:-3]
        if not decompressed_path.exists() or decompressed_path.stat().st_size == 0:
            with gzip.open(candidate_path, "rb") as src, decompressed_path.open("wb") as dst:
                shutil.copyfileobj(src, dst)
        candidate_path = decompressed_path
        details["decompressed_path"] = str(candidate_path)

    if candidate_path.is_dir():
        picked = _pick_best_local(candidate_path)
        if picked is None:
            raise PaperIntakeError(f"No usable data files were found after unpacking {staged_path}")
        candidate_path = picked
        details["selected_path"] = str(candidate_path)

    adata, load_details = _load_single_path(candidate_path)
    details.update(load_details)
    search_root = candidate_path.parent if candidate_path.is_file() else candidate_path
    metadata_path = _attach_sidecar_metadata(adata, search_root, candidate_path)
    if metadata_path:
        details["metadata_path"] = metadata_path
    return adata, details


def convert_asset_to_anndata(
    asset: DataAsset,
    download_root: str,
    raw_reference_path: Optional[str] = None,
) -> Tuple[ad.AnnData, Dict[str, object]]:
    download_dir = Path(download_root).expanduser().resolve()
    staged = download_asset(asset, str(download_dir))
    return _materialize_staged_path(asset, staged, download_dir, raw_reference_path=raw_reference_path)


def convert_local_path_to_anndata(
    asset: DataAsset,
    local_path: str,
    work_root: str,
    raw_reference_path: Optional[str] = None,
) -> Tuple[ad.AnnData, Dict[str, object]]:
    staged = Path(local_path).expanduser().resolve()
    if not staged.exists():
        raise PaperIntakeError(f"Local staged path does not exist: {staged}")
    materialize_root = Path(work_root).expanduser().resolve()
    materialize_root.mkdir(parents=True, exist_ok=True)
    adata, details = _materialize_staged_path(asset, staged, materialize_root, raw_reference_path=raw_reference_path)
    details["local_source_path"] = str(staged)
    return adata, details


def standardize_adata(
    adata: ad.AnnData,
    asset: DataAsset,
    manuscript_summary: Dict[str, object],
) -> ad.AnnData:
    adata.obs_names = adata.obs_names.astype(str)
    adata.var_names = adata.var_names.astype(str)
    adata.obs_names_make_unique()
    adata.var_names_make_unique()
    if not sparse.issparse(adata.X):
        try:
            adata.X = sparse.csr_matrix(np.asarray(adata.X, dtype=np.float32))
        except Exception:
            pass

    obs_columns = list(map(str, adata.obs.columns))
    time_like_columns = [
        column
        for column in obs_columns
        if any(token in column.lower() for token in ("time", "day", "stage", "age", "pseudotime"))
    ]
    label_like_columns = [
        column
        for column in obs_columns
        if any(token in column.lower() for token in ("cell", "type", "cluster", "label", "annot"))
    ]
    ready_for_preprocessing = bool(obs_columns) and bool(time_like_columns)
    ready_for_runtime_contract = "time_point_processed" in obs_columns and "X_latent" in adata.obsm
    validation_warnings: List[str] = []
    if not obs_columns:
        validation_warnings.append(
            "No observation metadata columns were found. This AnnData contains expression values only."
        )
    if not time_like_columns:
        validation_warnings.append(
            "No time-like observation column was detected. Add or merge a time/stage column before CytoBridge preprocessing."
        )

    adata.uns.setdefault("paper_intake", {})
    adata.uns["paper_intake"].update(
        {
            "selected_asset": asset.to_dict(),
            "manuscript": _compact_manuscript_summary_for_uns(manuscript_summary),
            "validation": {
                "n_obs": int(adata.n_obs),
                "n_vars": int(adata.n_vars),
                "obs_column_count": len(obs_columns),
                "obs_columns_sample": obs_columns[:20],
                "obsm_keys": list(map(str, adata.obsm.keys())),
                "has_latent": "X_latent" in adata.obsm,
                "time_like_obs_columns": time_like_columns[:10],
                "label_like_obs_columns": label_like_columns[:10],
                "ready_for_preprocessing": ready_for_preprocessing,
                "ready_for_runtime_contract": ready_for_runtime_contract,
                "selected_asset_provider": asset.provider,
                "selected_asset_format": asset.file_format,
                "warnings": validation_warnings,
            },
        }
    )
    return adata


def _string_list(values: object, limit: int = 0) -> List[str]:
    if values is None:
        return []
    if isinstance(values, (str, bytes)):
        items = [values]
    elif isinstance(values, Sequence):
        items = list(values)
    else:
        items = [values]
    out: List[str] = []
    for item in items:
        text = str(item or "").strip()
        if not text:
            continue
        out.append(text)
        if limit > 0 and len(out) >= limit:
            break
    return out


def _compact_fetched_pages_summary(pages: object) -> Dict[str, object]:
    if not isinstance(pages, Sequence) or isinstance(pages, (str, bytes)):
        return {"fetched_pages_count": 0}
    page_list = list(pages)
    compact: Dict[str, object] = {
        "fetched_pages_count": len(page_list),
        "fetched_page_urls": [],
        "fetched_page_titles": [],
        "fetched_page_content_types": [],
    }
    excerpts: List[str] = []
    for page in page_list[:5]:
        if not isinstance(page, Mapping):
            continue
        url = str(page.get("url") or "").strip()
        title = str(page.get("title") or "").strip()
        content_type = str(page.get("content_type") or "").strip()
        excerpt = str(page.get("text_excerpt") or "").strip()
        if url:
            cast_urls = compact.setdefault("fetched_page_urls", [])
            if isinstance(cast_urls, list):
                cast_urls.append(url)
        if title:
            cast_titles = compact.setdefault("fetched_page_titles", [])
            if isinstance(cast_titles, list):
                cast_titles.append(title)
        if content_type:
            cast_types = compact.setdefault("fetched_page_content_types", [])
            if isinstance(cast_types, list):
                cast_types.append(content_type)
        if excerpt:
            excerpts.append(excerpt[:400])
    if excerpts:
        compact["fetched_page_excerpts"] = excerpts
    return compact


def _compact_manuscript_summary_for_uns(manuscript_summary: Dict[str, object]) -> Dict[str, object]:
    summary = dict(manuscript_summary or {})
    accessions_raw = summary.get("accessions")
    accessions: Dict[str, List[str]] = {}
    if isinstance(accessions_raw, Mapping):
        for key, value in accessions_raw.items():
            values = _string_list(value, limit=20)
            if values:
                accessions[str(key)] = values

    section_flags_raw = summary.get("section_flags")
    section_flags: Dict[str, bool] = {}
    if isinstance(section_flags_raw, Mapping):
        section_flags = {str(key): bool(value) for key, value in section_flags_raw.items()}

    compact: Dict[str, object] = {
        "input_source": str(summary.get("input_source") or ""),
        "input_type": str(summary.get("input_type") or ""),
        "title": str(summary.get("title") or ""),
        "doi": str(summary.get("doi") or ""),
        "page_count": int(summary.get("page_count") or 0),
        "completeness_score": float(summary.get("completeness_score") or 0.0),
        "section_flags": section_flags,
        "accessions": accessions,
        "candidate_article_urls": _string_list(summary.get("candidate_article_urls"), limit=20),
        "discovered_urls_sample": _string_list(summary.get("discovered_urls"), limit=40),
        "relevant_sentences_sample": _string_list(summary.get("relevant_sentences"), limit=20),
        "notes": _string_list(summary.get("notes"), limit=20),
        "selected_asset_urls": _string_list(summary.get("selected_asset_urls"), limit=10),
        "supporting_asset_urls": _string_list(summary.get("supporting_asset_urls"), limit=20),
        "fallback_asset_urls": _string_list(summary.get("fallback_asset_urls"), limit=20),
        "resolved_dataset_hint": str(summary.get("resolved_dataset_hint") or ""),
        "target_dataset_prompt": str(summary.get("target_dataset_prompt") or ""),
        "primary_text_path": str(summary.get("primary_text_path") or ""),
        "article_text_path": str(summary.get("article_text_path") or ""),
        "summary_path": str(summary.get("summary_path") or ""),
        "manifest_path": str(summary.get("manifest_path") or ""),
        "inspection_bundle_path": str(summary.get("inspection_bundle_path") or ""),
        "asset_report_path": str(summary.get("asset_report_path") or ""),
        "llm_enabled": bool(summary.get("llm_enabled")),
        "llm_pdf_analysis_path": str(summary.get("llm_pdf_analysis_path") or ""),
        "llm_web_analysis_path": str(summary.get("llm_web_analysis_path") or ""),
        "llm_discovery_trace_path": str(summary.get("llm_discovery_trace_path") or ""),
    }
    compact.update(_compact_fetched_pages_summary(summary.get("fetched_pages")))
    return compact


def write_conversion_report(payload: Dict[str, object], output_dir: str) -> str:
    target = Path(output_dir).expanduser().resolve() / "paper_intake" / "conversion_report.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(target)
