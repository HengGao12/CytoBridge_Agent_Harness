from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Iterable, Optional


def get_cellcompass_root() -> Path:
    return (Path.home() / ".cellcompass").resolve()


def get_cellcompass_skills_root() -> Path:
    return (get_cellcompass_root() / "skills").resolve()


def get_builtin_skills_root() -> Path:
    return (Path(__file__).resolve().parents[1] / "skills").resolve()


def get_cellcompass_skills_manifest_path() -> Path:
    return (get_cellcompass_skills_root() / ".builtin_manifest.json").resolve()


def _iter_copyable_paths(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        rel_parts = path.relative_to(root).parts
        if "__pycache__" in rel_parts or ".builtin_manifest.json" in rel_parts:
            continue
        yield path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_manifest(path: Path, manifest: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _skill_sync_mode() -> str:
    """Return the runtime skill sync mode.

    During active CytoBridge development the package skill tree is canonical.
    The user mirror is overwritten by default so stale runtime skills cannot
    shadow repository updates. Set CYTOBRIDGE_SKILL_SYNC_MODE=preserve or
    CYTOBRIDGE_PRESERVE_USER_SKILLS=1 to recover the old local-edit-preserving
    behavior for deployments with user-maintained skill trees.
    """

    if os.environ.get("CYTOBRIDGE_PRESERVE_USER_SKILLS", "").strip().lower() in {"1", "true", "yes", "on"}:
        return "preserve"
    raw = os.environ.get("CYTOBRIDGE_SKILL_SYNC_MODE", "").strip().lower()
    if raw in {"preserve", "local", "user"}:
        return "preserve"
    return "overwrite"


def _remove_empty_dirs(root: Path) -> None:
    if not root.exists():
        return
    for path in sorted((p for p in root.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
        try:
            path.rmdir()
        except OSError:
            pass


def ensure_cellcompass_skills_migrated(
    *,
    builtin_root: Optional[Path] = None,
    user_root: Optional[Path] = None,
) -> Path:
    import logging
    src_root = Path(builtin_root or get_builtin_skills_root()).expanduser().resolve()
    dst_root = Path(user_root or get_cellcompass_skills_root()).expanduser().resolve()
    manifest_path = dst_root / ".builtin_manifest.json"
    previous_manifest = _load_manifest(manifest_path)
    manifest: dict[str, str] = {}
    sync_mode = _skill_sync_mode()
    dst_root.mkdir(parents=True, exist_ok=True)
    if not src_root.exists():
        return dst_root

    for src_path in _iter_copyable_paths(src_root):
        try:
            rel = src_path.relative_to(src_root)
            dst_path = dst_root / rel
            if src_path.is_dir():
                dst_path.mkdir(parents=True, exist_ok=True)
                continue
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            rel_key = rel.as_posix()
            src_hash = _sha256(src_path)
            manifest[rel_key] = src_hash
            if not dst_path.exists():
                shutil.copy2(src_path, dst_path)
                continue
            dst_hash = _sha256(dst_path)
            if dst_hash == src_hash:
                continue

            if sync_mode == "overwrite":
                shutil.copy2(src_path, dst_path)
                continue

            previous_hash = previous_manifest.get(rel_key)
            if previous_hash and dst_hash == previous_hash:
                shutil.copy2(src_path, dst_path)
                continue
            # Preserve local edits when the mirrored file has diverged from the last synced builtin copy.
            if previous_hash:
                manifest[rel_key] = previous_hash
            else:
                manifest.pop(rel_key, None)
        except Exception as exc:
            logging.getLogger("cytobridge_agent").warning(
                f"Skipping migration for {src_path} due to OS/path limits: {exc}"
            )
    if sync_mode == "overwrite":
        for dst_path in dst_root.rglob("*"):
            if dst_path.is_dir():
                continue
            try:
                rel_key = dst_path.relative_to(dst_root).as_posix()
                if rel_key == ".builtin_manifest.json":
                    continue
                if rel_key not in manifest and (dst_path.is_file() or dst_path.is_symlink()):
                    dst_path.unlink()
            except Exception as exc:
                logging.getLogger("cytobridge_agent").warning(
                    f"Skipping removal for stale skill file {dst_path} due to OS/path limits: {exc}"
                )
        _remove_empty_dirs(dst_root)
    _save_manifest(manifest_path, manifest)
    return dst_root
