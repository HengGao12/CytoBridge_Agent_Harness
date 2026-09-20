#!/usr/bin/env python3
"""Build a non-destructive CytoBridge raw data pool.

The pool stores symlinks and a manifest for raw/downloaded source material.
It intentionally does not move or copy large files, so existing campaign
paths remain valid while future agents get one canonical place to look.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator


RAW_DIR_NAMES = {"raw", "downloads", "materialize"}
RAW_DIR_PREFIXES = ("downloads_",)
DATA_EXTENSIONS = {
    ".csv",
    ".gz",
    ".h5",
    ".h5ad",
    ".loom",
    ".mtx",
    ".rds",
    ".tar",
    ".tgz",
    ".tsv",
    ".zip",
}
SKIP_DIR_NAMES = {
    ".git",
    "__pycache__",
    "raw_data_pool",
    "cellcompass",
    "conversations",
    "checkpoints",
}


@dataclass(frozen=True)
class Candidate:
    kind: str
    path: Path


def _safe_slug(path: Path, root: Path) -> str:
    try:
        rel = path.relative_to(root)
    except ValueError:
        rel = path
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", str(rel).strip("/"))
    digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:10]
    return f"{stem[:120]}__{digest}"


def _dir_size_and_count(path: Path) -> tuple[int, int]:
    total = 0
    count = 0
    if path.is_file():
        return path.stat().st_size, 1
    for dirpath, dirnames, filenames in os.walk(path):
        dirnames[:] = [name for name in dirnames if name not in SKIP_DIR_NAMES]
        for filename in filenames:
            file_path = Path(dirpath) / filename
            try:
                stat = file_path.stat()
            except OSError:
                continue
            total += stat.st_size
            count += 1
    return total, count


def _has_data_file(path: Path) -> bool:
    try:
        children = list(path.iterdir())
    except OSError:
        return False
    for child in children:
        if child.is_file() and any(str(child.name).lower().endswith(ext) for ext in DATA_EXTENSIONS):
            return True
    return False


def discover_candidates(scan_root: Path) -> Iterator[Candidate]:
    scan_root = scan_root.resolve()
    datasets_root = scan_root / "datasets"
    if datasets_root.is_dir():
        for child in sorted(datasets_root.iterdir()):
            if child.is_dir() and _has_data_file(child):
                yield Candidate("dataset_root", child)

    for dirpath, dirnames, filenames in os.walk(scan_root):
        current = Path(dirpath)
        rel_parts = set(current.relative_to(scan_root).parts) if current != scan_root else set()
        if rel_parts & SKIP_DIR_NAMES:
            dirnames[:] = []
            continue
        dirnames[:] = [
            name
            for name in dirnames
            if name not in SKIP_DIR_NAMES and not (current / name).resolve().is_relative_to(scan_root / "raw_data_pool")
        ]
        name = current.name
        if name in RAW_DIR_NAMES or any(name.startswith(prefix) for prefix in RAW_DIR_PREFIXES):
            kind = "raw_dir" if name == "raw" else "download_dir" if name.startswith("downloads") else "materialize_dir"
            yield Candidate(kind, current)


def _deduplicate(candidates: Iterable[Candidate]) -> list[Candidate]:
    seen: set[Path] = set()
    result: list[Candidate] = []
    for candidate in candidates:
        path = candidate.path.resolve()
        if path in seen:
            continue
        seen.add(path)
        result.append(Candidate(candidate.kind, path))
    return result


def build_pool(scan_root: Path, pool_root: Path) -> Path:
    scan_root = scan_root.resolve()
    pool_root = pool_root.resolve()
    sources_root = pool_root / "sources"
    incoming_root = pool_root / "incoming"
    staging_root = pool_root / "staging"
    for path in (sources_root, incoming_root, staging_root):
        path.mkdir(parents=True, exist_ok=True)

    manifest_path = pool_root / "manifest.tsv"
    candidates = _deduplicate(discover_candidates(scan_root))
    rows = [
        [
            "slug",
            "kind",
            "source_path",
            "pool_link",
            "size_bytes",
            "file_count",
            "modified_utc",
        ]
    ]
    for candidate in sorted(candidates, key=lambda item: (item.kind, str(item.path))):
        slug = _safe_slug(candidate.path, scan_root)
        link_dir = sources_root / candidate.kind
        link_dir.mkdir(parents=True, exist_ok=True)
        link_path = link_dir / slug
        if link_path.is_symlink():
            if link_path.resolve() != candidate.path:
                link_path.unlink()
                link_path.symlink_to(candidate.path, target_is_directory=candidate.path.is_dir())
        elif not link_path.exists():
            link_path.symlink_to(candidate.path, target_is_directory=candidate.path.is_dir())
        size_bytes, file_count = _dir_size_and_count(candidate.path)
        try:
            modified_utc = datetime.fromtimestamp(candidate.path.stat().st_mtime, timezone.utc).isoformat()
        except OSError:
            modified_utc = ""
        rows.append(
            [
                slug,
                candidate.kind,
                str(candidate.path),
                str(link_path),
                str(size_bytes),
                str(file_count),
                modified_utc,
            ]
        )

    readme = pool_root / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# CytoBridge Raw Data Pool",
                "",
                "This directory is the canonical pool for original downloaded/raw data.",
                "",
                "- `sources/` contains symlinks to already-downloaded raw/download/materialization inputs.",
                "- `incoming/` is where new paper-intake downloads should be written.",
                "- `staging/` is reserved for resumable or temporary raw download work.",
                "- `manifest.tsv` records every indexed source path, link, size, and modified time.",
                "",
                "Do not edit completed campaign outputs through these links. If a new algorithm needs",
                "a prior locked result as an input or baseline, register it through the campaign tools",
                "and keep any compatibility wrapper separate from the locked algorithm code.",
                "",
                f"Last refreshed: {datetime.now(timezone.utc).isoformat()}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    manifest_path.write_text("\n".join("\t".join(row) for row in rows) + "\n", encoding="utf-8")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-root", default="/data/cytobridge")
    parser.add_argument("--pool-root", default="/data/cytobridge/raw_data_pool")
    args = parser.parse_args()
    manifest = build_pool(Path(args.scan_root), Path(args.pool_root))
    print(manifest)


if __name__ == "__main__":
    main()
