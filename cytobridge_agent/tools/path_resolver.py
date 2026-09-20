"""Resolve dataset paths from free-form user messages."""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional


_PATH_TOKEN_RE = re.compile(r"(?:'([^']+)'|\"([^\"]+)\"|([^\s]+))")


def _clean_token(token: str) -> str:
    return token.strip().strip(",.;:()[]{}")


def _looks_like_path(token: str) -> bool:
    if not token:
        return False
    if token.startswith(("/", "./", "../", "~", "\\")):
        return True
    if "/" in token or "\\" in token:
        return True
    if re.search(r"\.[A-Za-z0-9]{1,8}$", token):
        return True
    return False


def extract_candidate_paths(text: str) -> List[str]:
    """Extract path-like tokens from arbitrary text."""
    candidates: List[str] = []
    seen = set()
    for m in _PATH_TOKEN_RE.finditer(text or ""):
        token = m.group(1) or m.group(2) or m.group(3) or ""
        token = _clean_token(token)
        if not _looks_like_path(token):
            continue
        if token in seen:
            continue
        seen.add(token)
        candidates.append(token)
    return candidates


def canonicalize_path(path_str: str, cwd: Optional[str] = None) -> str:
    """Convert a path string to an absolute canonical path."""
    base = Path(cwd).expanduser().resolve() if cwd else Path.cwd()
    p = Path(path_str).expanduser()
    if not p.is_absolute():
        p = (base / p).resolve()
    else:
        p = p.resolve()
    return str(p)


def pick_path_from_message(text: str, cwd: Optional[str] = None) -> Optional[str]:
    """Pick the most likely dataset path in user text; return None if none exist."""
    existing: List[Path] = []
    for candidate in extract_candidate_paths(text):
        try:
            if candidate.strip() in {"/", "\\"}:
                continue
            canonical = canonicalize_path(candidate, cwd=cwd)
            path = Path(canonical)
            if path == Path(path.anchor):
                continue
            if path.exists():
                existing.append(path)
        except Exception:
            continue
    if not existing:
        return None

    data_suffixes = {".h5ad", ".h5mu", ".loom", ".zarr", ".csv", ".tsv", ".mtx"}
    for path in existing:
        if path.is_file() and path.suffix.lower() in data_suffixes:
            return str(path)
    for path in existing:
        if path.is_file():
            return str(path)
    for path in existing:
        if path.is_dir():
            return str(path)
    return None
