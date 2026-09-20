from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


INTERMEDIATE_FILE_MAP: dict[str, str] = {
    "holdout_prediction": "holdout_prediction.csv",
    "velocity_field": "velocity_field.csv",
    "growth_rates": "growth_rates.csv",
    "per_cell_fate": "per_cell_fate.json",
    "perturbation_results": "perturbation_results.json",
    "driver_genes": "driver_genes.json",
}

TABULAR_INTERMEDIATES = {"holdout_prediction", "velocity_field", "growth_rates"}
JSON_INTERMEDIATES = {"per_cell_fate", "perturbation_results", "driver_genes"}


def _dedupe_paths(paths: Iterable[str | Path]) -> list[Path]:
    resolved: list[Path] = []
    seen: set[Path] = set()
    for raw in paths:
        path = Path(raw).expanduser().resolve()
        if not path.exists() or path in seen:
            continue
        seen.add(path)
        resolved.append(path)
    return resolved


def _nonempty(path: Path, started_at: float | None = None) -> bool:
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


def _load_output_payload(path: Path, key: str) -> Any:
    if key in TABULAR_INTERMEDIATES:
        return pd.read_csv(path).to_dict(orient="records")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_intermediate_file(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _find_candidates(root: Path, relpaths: list[str]) -> list[Path]:
    candidates: list[Path] = []
    for rel in relpaths:
        direct = root / rel
        if direct.exists() and direct.is_file():
            candidates.append(direct)
        name = Path(rel).name
        candidates.extend(path for path in root.rglob(name) if path.is_file())
    return candidates


def materialize_intermediates_from_files(
    *,
    output_dir: str | Path,
    search_roots: Iterable[str | Path],
    started_at: float | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir).resolve()
    roots = _dedupe_paths(search_roots)
    intermediates_dir = output_dir / "intermediates"
    intermediates_dir.mkdir(parents=True, exist_ok=True)

    available_keys: list[str] = []
    missing_keys: list[str] = []
    sources: dict[str, str] = {}

    for key, canonical_name in INTERMEDIATE_FILE_MAP.items():
        intermediate_path = intermediates_dir / f"{key}.json"
        if _nonempty(intermediate_path, started_at=started_at):
            available_keys.append(key)
            sources[key] = str(intermediate_path)
            continue

        candidate_relpaths = [
            f"intermediates/{key}.json",
            canonical_name,
        ]
        candidates: list[Path] = []
        for root in roots:
            candidates.extend(_find_candidates(root, candidate_relpaths))
        candidates = [path for path in candidates if _nonempty(path, started_at=started_at)]
        if not candidates:
            missing_keys.append(key)
            continue

        chosen = sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)[0]
        if chosen.suffix.lower() == ".json" and chosen.name == f"{key}.json":
            payload = json.loads(chosen.read_text(encoding="utf-8"))
        else:
            payload = _load_output_payload(chosen, key)
        _write_intermediate_file(intermediate_path, payload)
        available_keys.append(key)
        sources[key] = str(chosen)

    manifest = {
        "contract": "dynbench_intermediates_v1",
        "output_dir": str(output_dir),
        "intermediates_dir": str(intermediates_dir),
        "available_keys": available_keys,
        "missing_keys": missing_keys,
        "sources": sources,
    }
    manifest_path = intermediates_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    return manifest
