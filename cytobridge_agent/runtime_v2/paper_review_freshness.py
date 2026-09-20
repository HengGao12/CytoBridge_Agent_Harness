from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


_PAPER_ROOT_FILES = {
    "main.tex",
    "supplement.tex",
    "paper_plan.md",
    "source_obligation_matrix.md",
    "theory_obligation_ledger.md",
    "algorithm_detail_ledger.md",
    "evidence_ledger.md",
    "citation_ledger.md",
    "narrative_report.md",
}

_DERIVED_REVIEW_FILES = {
    "paper_reviewer_gate.md",
}


def _is_tracked_file(path: Path) -> bool:
    if path.name in _DERIVED_REVIEW_FILES:
        return False
    if path.name in _PAPER_ROOT_FILES:
        return True
    parent = path.parent.name
    if parent == "sections" and path.suffix == ".tex":
        return True
    if parent == "checks" and path.suffix == ".md":
        return True
    return False


def _paper_files_from_dir(path: Path) -> List[Path]:
    if path.name == "sections":
        return sorted(child for child in path.glob("*.tex") if child.is_file())
    if path.name == "checks":
        return sorted(child for child in path.glob("*.md") if child.is_file())

    files: List[Path] = []
    for name in sorted(_PAPER_ROOT_FILES):
        candidate = path / name
        if candidate.is_file():
            files.append(candidate)
    sections_dir = path / "sections"
    if sections_dir.is_dir():
        files.extend(sorted(child for child in sections_dir.glob("*.tex") if child.is_file()))
    checks_dir = path / "checks"
    if checks_dir.is_dir():
        files.extend(
            sorted(
                child
                for child in checks_dir.glob("*.md")
                if child.is_file() and child.name not in _DERIVED_REVIEW_FILES
            )
        )
    return files


def collect_paper_review_files(paths: Iterable[Any]) -> List[Path]:
    found: Dict[str, Path] = {}
    for raw_path in paths or []:
        text = str(raw_path or "").strip()
        if not text:
            continue
        path = Path(text).expanduser()
        try:
            path = path.resolve()
        except Exception:
            pass
        if path.is_dir():
            for child in _paper_files_from_dir(path):
                try:
                    resolved = child.resolve()
                except Exception:
                    resolved = child
                found[str(resolved)] = resolved
        elif path.is_file() and _is_tracked_file(path):
            found[str(path)] = path
    return [found[key] for key in sorted(found)]


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_paper_review_manifest(paths: Iterable[Any]) -> Dict[str, Any]:
    source_paths = [str(item) for item in (paths or []) if str(item or "").strip()]
    files = []
    for path in collect_paper_review_files(source_paths):
        try:
            stat = path.stat()
            file_hash = _hash_file(path)
        except Exception:
            continue
        files.append(
            {
                "path": str(path),
                "sha256": file_hash,
                "size": int(stat.st_size),
                "mtime_ns": int(getattr(stat, "st_mtime_ns", 0)),
            }
        )

    overall = hashlib.sha256()
    for item in files:
        overall.update(str(item["path"]).encode("utf-8", errors="surrogateescape"))
        overall.update(b"\0")
        overall.update(str(item["sha256"]).encode("ascii"))
        overall.update(b"\0")

    return {
        "version": 1,
        "algorithm": "sha256",
        "source_paths": source_paths,
        "digest": overall.hexdigest(),
        "file_count": len(files),
        "files": files,
    }


def _coerce_review_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def paper_review_is_approved(review: Any) -> bool:
    if not isinstance(review, dict):
        return False
    decision = str(review.get("decision") or "").strip().lower()
    if decision != "approve":
        return False
    return not _coerce_review_list(review.get("blocking_issues"))


def paper_review_manifest_status(manifest: Any) -> Dict[str, Any]:
    if not isinstance(manifest, dict):
        return {"fresh": False, "reason": "review has no file-hash manifest"}
    source_paths = manifest.get("source_paths")
    if not isinstance(source_paths, list) or not source_paths:
        source_paths = [
            str(item.get("path") or "")
            for item in (manifest.get("files") or [])
            if isinstance(item, dict) and str(item.get("path") or "").strip()
        ]
    if not source_paths:
        return {"fresh": False, "reason": "review hash manifest has no source paths"}

    current = build_paper_review_manifest(source_paths)
    expected_digest = str(manifest.get("digest") or "").strip()
    current_digest = str(current.get("digest") or "").strip()
    if not expected_digest:
        return {"fresh": False, "reason": "review hash manifest has no digest", "current": current}
    if int(current.get("file_count") or 0) <= 0:
        return {"fresh": False, "reason": "no tracked paper files found for review freshness check", "current": current}
    if current_digest != expected_digest:
        return {
            "fresh": False,
            "reason": "tracked paper file hashes changed after reviewer approval",
            "expected_digest": expected_digest,
            "current_digest": current_digest,
            "current": current,
        }
    return {"fresh": True, "reason": "tracked paper file hashes match reviewer approval", "current": current}


def _looks_like_paper_root(path: Path) -> bool:
    if any((path / name).is_file() for name in _PAPER_ROOT_FILES):
        return True
    if (path / "sections").is_dir() or (path / "checks").is_dir():
        return True
    return False


def _paper_root_from_path(raw_path: Any) -> Optional[Path]:
    text = str(raw_path or "").strip()
    if not text:
        return None
    path = Path(text).expanduser()
    try:
        path = path.resolve()
    except Exception:
        pass

    if path.is_file():
        if path.name in _PAPER_ROOT_FILES:
            return path.parent
        if path.parent.name in {"sections", "checks"}:
            return path.parent.parent
        return None

    if path.is_dir():
        if path.name in {"sections", "checks"}:
            return path.parent
        if _looks_like_paper_root(path):
            return path
        for candidate in (path / "paper", path / "outputs" / "paper"):
            if candidate.is_dir() and _looks_like_paper_root(candidate):
                return candidate
    return None


def _candidate_paper_roots(review: Dict[str, Any], relevant_paths: Iterable[Any] | None) -> List[Path]:
    roots: Dict[str, Path] = {}
    manifest = review.get("reviewed_file_hashes")
    source_paths: List[Any] = list(relevant_paths or [])
    if isinstance(manifest, dict):
        source_paths.extend(manifest.get("source_paths") or [])
        for item in manifest.get("files") or []:
            if isinstance(item, dict):
                source_paths.append(item.get("path"))

    for item in source_paths:
        root = _paper_root_from_path(item)
        if root is None:
            continue
        try:
            resolved = root.resolve()
        except Exception:
            resolved = root
        roots[str(resolved)] = resolved
    return [roots[key] for key in sorted(roots)]


def _format_review_field(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return "\n".join(f"- {str(item).strip()}" for item in value if str(item).strip())
    return str(value).strip()


def _render_paper_reviewer_gate(
    *,
    review: Dict[str, Any],
    manifest_status: Dict[str, Any],
    synced_at: str,
) -> str:
    manifest = review.get("reviewed_file_hashes") if isinstance(review, dict) else {}
    file_count = int(manifest.get("file_count") or 0) if isinstance(manifest, dict) else 0
    digest = str(manifest.get("digest") or "").strip() if isinstance(manifest, dict) else ""
    feedback = _format_review_field(review.get("reviewer_feedback"))
    summary = _format_review_field(review.get("summary"))
    notes = _format_review_field(review.get("notes"))

    lines = [
        "# Paper Reviewer Gate",
        "",
        "Latest reviewer decision: approve.",
        "Status: approved.",
        "",
        "This file is synchronized automatically from the completed `paper_reviewer` workflow state.",
        "It is a derived gate record, not a separate reviewer pass.",
        "",
        "## Synchronized Approval",
        "",
        f"- synced_at: {synced_at}",
        "- source: completed `paper_reviewer` subagent result",
        f"- file_hash_manifest_fresh: {bool(manifest_status.get('fresh'))}",
        f"- reviewed_file_count: {file_count}",
        f"- review_digest: {digest or '(missing)'}",
        f"- freshness: {manifest_status.get('reason') or ''}",
        "- blocking_issues: none",
    ]
    if feedback:
        lines.extend(["", "## Reviewer Feedback", "", feedback])
    if summary:
        lines.extend(["", "## Reviewer Summary", "", summary])
    if notes:
        lines.extend(["", "## Notes", "", notes])
    lines.append("")
    return "\n".join(lines)


def sync_paper_reviewer_gate_from_review(
    review: Any,
    relevant_paths: Iterable[Any] | None = None,
    *,
    synced_at: str | None = None,
) -> Dict[str, Any]:
    """Write the durable paper gate record from an approved fresh paper review.

    The gate file is derived from workflow state and is intentionally excluded
    from the reviewer freshness manifest to avoid a self-invalidating hash loop.
    """
    if not isinstance(review, dict):
        return {"status": "skipped", "reason": "paper_review is not a mapping"}
    if not paper_review_is_approved(review):
        return {"status": "skipped", "reason": "paper_review is not an approve decision with zero blockers"}

    manifest_status = paper_review_manifest_status(review.get("reviewed_file_hashes"))
    if not bool(manifest_status.get("fresh")):
        return {
            "status": "skipped",
            "reason": str(manifest_status.get("reason") or "review hash manifest is not fresh"),
            "manifest_status": manifest_status,
        }

    roots = _candidate_paper_roots(review, relevant_paths)
    if not roots:
        return {"status": "skipped", "reason": "no paper root path found for reviewer gate sync"}

    sync_time = synced_at or datetime.utcnow().isoformat(timespec="seconds") + "Z"
    content = _render_paper_reviewer_gate(
        review=review,
        manifest_status=manifest_status,
        synced_at=sync_time,
    )
    written_paths: List[str] = []
    errors: List[str] = []
    for root in roots:
        try:
            checks_dir = root / "checks"
            checks_dir.mkdir(parents=True, exist_ok=True)
            gate_path = checks_dir / "paper_reviewer_gate.md"
            gate_path.write_text(content, encoding="utf-8")
            written_paths.append(str(gate_path))
        except Exception as exc:
            errors.append(f"{root}: {exc}")

    if not written_paths:
        return {
            "status": "failed",
            "reason": "could not write any paper_reviewer_gate.md file",
            "errors": errors,
        }
    return {
        "status": "synced",
        "paths": written_paths,
        "errors": errors,
        "synced_at": sync_time,
        "manifest_digest": str((review.get("reviewed_file_hashes") or {}).get("digest") or ""),
    }
