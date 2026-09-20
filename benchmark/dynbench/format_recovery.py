from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Iterable


def _is_nonempty_file(path: Path, started_at: float | None = None) -> bool:
    if not path.exists() or not path.is_file():
        return False
    try:
        if path.stat().st_size <= 0:
            return False
        if started_at is not None and path.stat().st_mtime < started_at:
            return False
    except FileNotFoundError:
        return False
    return True


def _dedupe_roots(roots: Iterable[str | Path]) -> list[Path]:
    unique: list[Path] = []
    seen: set[Path] = set()
    for raw in roots:
        root = Path(raw).expanduser().resolve()
        if root in seen or not root.exists():
            continue
        unique.append(root)
        seen.add(root)
    return unique


def _find_candidates(root: Path, names: list[str]) -> list[Path]:
    matches: list[Path] = []
    for name in names:
        direct = root / name
        if direct.exists() and direct.is_file():
            matches.append(direct)
        matches.extend(path for path in root.rglob(name) if path.is_file())
    return matches


def recover_outputs_format_only(
    *,
    output_dir: str | Path,
    expected_output_files: list[str],
    search_roots: Iterable[str | Path],
    started_at: float | None = None,
) -> dict[str, object]:
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    roots = _dedupe_roots(search_roots)

    already_present: list[str] = []
    recovered_files: list[dict[str, str]] = []
    missing_files: list[str] = []

    for canonical_name in expected_output_files:
        target = output_dir / canonical_name
        if _is_nonempty_file(target, started_at=started_at):
            already_present.append(canonical_name)
            continue

        candidate_names = [canonical_name]
        candidates: list[Path] = []
        for root in roots:
            candidates.extend(_find_candidates(root, candidate_names))
        candidates = [path for path in candidates if _is_nonempty_file(path, started_at=started_at)]
        if not candidates:
            missing_files.append(canonical_name)
            continue

        chosen = sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)[0]
        if chosen.resolve() != target.resolve():
            shutil.copy2(chosen, target)
        method = "format_copy"
        recovered_files.append(
            {
                "name": canonical_name,
                "source": str(chosen),
                "target": str(target),
                "method": method,
            }
        )

    if missing_files:
        format_status = "incomplete"
        recovery_status = "failed" if recovered_files else "not_applied"
        artifact_origin = "mixed" if already_present or recovered_files else "none"
        recovery_method = "none"
    elif recovered_files:
        format_status = "recovered_complete"
        recovery_status = "succeeded"
        artifact_origin = "mixed" if already_present else "recovered"
        recovery_method = "format_copy"
    else:
        format_status = "native_complete"
        recovery_status = "not_applied"
        artifact_origin = "native"
        recovery_method = "none"

    manifest = {
        "output_dir": str(output_dir),
        "search_roots": [str(root) for root in roots],
        "already_present": already_present,
        "recovered_files": recovered_files,
        "missing_files": missing_files,
        "format_status": format_status,
        "recovery_status": recovery_status,
        "recovery_method": recovery_method,
        "artifact_origin": artifact_origin,
    }
    manifest_path = output_dir / "format_recovery_manifest.json"
    manifest["recovery_manifest"] = str(manifest_path) if recovered_files else None
    if recovered_files:
        manifest_path.write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )
    return manifest
