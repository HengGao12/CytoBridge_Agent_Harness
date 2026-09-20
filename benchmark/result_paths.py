from __future__ import annotations

from datetime import datetime
from pathlib import Path


BENCHMARK_ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = BENCHMARK_ROOT / "results"
REPO_ROOT = BENCHMARK_ROOT.parent
LEGACY_BENCHMARK_OUTPUT_ROOT = REPO_ROOT / "cytobridge_output" / "figures"


def timestamp_minute() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M")


def remap_legacy_benchmark_output_root(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    try:
        relative = resolved.relative_to(LEGACY_BENCHMARK_OUTPUT_ROOT)
    except ValueError:
        return resolved
    return (RESULTS_ROOT / relative).resolve()


def resolve_run_root(run_root: str | Path | None = None, *, label: str = "run") -> Path:
    if run_root is None:
        return RESULTS_ROOT / f"{label}_{timestamp_minute()}"
    return remap_legacy_benchmark_output_root(run_root)


def resolve_dynbench_paths(
    *,
    scenario: str,
    run_name: str,
    run_root: str | Path | None = None,
) -> tuple[Path, Path, Path]:
    root = resolve_run_root(run_root)
    scenario_dir = root / "synthetic" / scenario
    output_dir = scenario_dir / run_name
    return root, scenario_dir, output_dir


def resolve_realdynbench_paths(
    *,
    dataset_id: str,
    fold_id: str,
    run_name: str,
    run_root: str | Path | None = None,
) -> tuple[Path, Path, Path]:
    root = resolve_run_root(run_root, label="realdynbench")
    fold_dir = root / "real" / dataset_id / fold_id
    output_dir = fold_dir / run_name
    return root, fold_dir, output_dir
