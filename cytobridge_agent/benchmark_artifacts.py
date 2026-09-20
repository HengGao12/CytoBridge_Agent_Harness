"""Utilities for distributing CellCompass algorithm benchmark artifacts.

Large benchmark datasets, saved trajectories, and baseline checkpoints should
not live in the git history.  This module keeps a small manifest in the package
and installs the binary bundle from a local file or release asset.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import tarfile
import tempfile
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import unquote, urlparse

try:
    import zstandard as zstd
except Exception:  # pragma: no cover - exercised only in missing dependency envs.
    zstd = None  # type: ignore[assignment]


MANIFEST_RESOURCE = "resources/algorithm_benchmarks/manifest.json"
DEFAULT_INSTALL_ROOT = Path.home() / ".cellcompass"
DEFAULT_BENCHMARK_ROOT = DEFAULT_INSTALL_ROOT / "algorithm_benchmarks"
DEFAULT_RELEASE_TAG = "algorithm-benchmarks-2026-05-03"
DEFAULT_ASSET_NAME = "cellcompass-algorithm-benchmarks-2026-05-03.tar.zst"
DEFAULT_ASSET_URL = (
    "https://github.com/JackkWangzh/CytoBridge-agent/releases/download/"
    f"{DEFAULT_RELEASE_TAG}/{DEFAULT_ASSET_NAME}"
)


@dataclass
class InstallResult:
    ok: bool
    output_root: str
    benchmark_root: str
    source: str
    files_written: int
    files_skipped: int
    bytes_written: int
    datasets: list[str]
    warnings: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "output_root": self.output_root,
            "benchmark_root": self.benchmark_root,
            "source": self.source,
            "files_written": self.files_written,
            "files_skipped": self.files_skipped,
            "bytes_written": self.bytes_written,
            "datasets": self.datasets,
            "warnings": self.warnings,
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _expand_path(path: str | os.PathLike[str] | None) -> Path | None:
    if path is None:
        return None
    return Path(path).expanduser().resolve()


def bundled_manifest_path() -> Path:
    """Return a pathlib view of the bundled manifest."""
    return Path(str(resources.files("cytobridge_agent").joinpath(MANIFEST_RESOURCE)))


def load_manifest(path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Load a benchmark artifact manifest.

    When *path* is omitted, the package-bundled manifest is used.
    """
    if path:
        return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    with resources.files("cytobridge_agent").joinpath(MANIFEST_RESOURCE).open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(handle)


def sha256_file(path: str | os.PathLike[str], chunk_size: int = 1024 * 1024 * 8) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _default_asset(manifest: dict[str, Any]) -> dict[str, Any]:
    assets = manifest.get("assets") or []
    if not assets:
        raise ValueError("Benchmark manifest does not define any assets.")
    return assets[0]


def _download_asset(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    parsed = urlparse(url)
    headers = {"User-Agent": "cytobridge-agent"}
    if parsed.hostname == "api.github.com" and "/releases/assets/" in parsed.path:
        headers["Accept"] = "application/octet-stream"
    token = _github_token_for_url(url)
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response, destination.open("wb") as out:
        shutil.copyfileobj(response, out)


def _github_token_for_url(url: str) -> str | None:
    parsed = urlparse(url)
    host = parsed.hostname or ""
    if not host.endswith("github.com"):
        return None
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        return token
    credentials_path = Path.home() / ".git-credentials"
    if not credentials_path.exists():
        return None
    try:
        for line in credentials_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            credential = urlparse(line.strip())
            if credential.hostname == "github.com" and credential.password:
                return unquote(credential.password)
    except OSError:
        return None
    return None


def _download_manifest_asset(asset: dict[str, Any], destination: Path) -> None:
    """Download one manifest asset, including multipart release assets."""
    parts = asset.get("parts") or []
    if not parts:
        url = asset.get("url")
        if not url:
            raise ValueError(f"Benchmark asset {asset.get('name') or '<unnamed>'} has no URL.")
        _download_asset(url, destination)
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="cytobridge_benchmark_parts_") as tmp:
        tmp_root = Path(tmp)
        with destination.open("wb") as out:
            for index, part in enumerate(parts, start=1):
                part_url = part.get("url")
                part_name = part.get("name") or f"part-{index:02d}"
                if not part_url:
                    raise ValueError(f"Benchmark asset part {part_name} has no URL.")
                part_path = tmp_root / part_name
                _download_asset(part_url, part_path)
                expected = part.get("sha256")
                if expected:
                    actual = sha256_file(part_path)
                    if actual != expected:
                        raise ValueError(
                            "Benchmark asset part checksum mismatch: "
                            f"{part_name} expected {expected}, got {actual}"
                        )
                with part_path.open("rb") as handle:
                    shutil.copyfileobj(handle, out)


def _open_archive_stream(path: Path):
    suffixes = "".join(path.suffixes)
    if suffixes.endswith(".tar.zst"):
        if zstd is None:
            raise RuntimeError(
                "Installing .tar.zst benchmark bundles requires the zstandard Python package."
            )
        raw = path.open("rb")
        reader = zstd.ZstdDecompressor().stream_reader(raw)
        return raw, reader, "r|"
    if suffixes.endswith(".tar.gz") or suffixes.endswith(".tgz"):
        raw = path.open("rb")
        return raw, raw, "r|gz"
    if suffixes.endswith(".tar"):
        raw = path.open("rb")
        return raw, raw, "r|"
    raise ValueError(f"Unsupported benchmark archive format: {path.name}")


def _is_allowed_member(
    member_name: str,
    dataset_filter: set[str] | None,
    include_baselines: bool,
) -> bool:
    parts = Path(member_name).parts
    if not parts or parts[0] != "algorithm_benchmarks":
        return False
    if len(parts) >= 2 and parts[1] == "builtin_runs":
        return include_baselines
    if dataset_filter and len(parts) >= 3 and parts[1] == "datasets":
        return parts[2] in dataset_filter
    return True


def _safe_member_destination(output_root: Path, member_name: str) -> Path:
    pure = Path(member_name)
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError(f"Unsafe archive member path: {member_name}")
    destination = (output_root / pure).resolve()
    output_root_resolved = output_root.resolve()
    if destination != output_root_resolved and output_root_resolved not in destination.parents:
        raise ValueError(f"Archive member escapes output root: {member_name}")
    return destination


def _extract_archive(
    archive_path: Path,
    output_root: Path,
    *,
    force: bool = False,
    dataset_filter: set[str] | None = None,
    include_baselines: bool = True,
) -> tuple[int, int, int, list[str], list[str]]:
    output_root.mkdir(parents=True, exist_ok=True)
    files_written = 0
    files_skipped = 0
    bytes_written = 0
    datasets: set[str] = set()
    warnings: list[str] = []

    raw_handle, stream_handle, mode = _open_archive_stream(archive_path)
    try:
        with raw_handle, stream_handle:
            with tarfile.open(fileobj=stream_handle, mode=mode) as tar:
                for member in tar:
                    name = member.name
                    if not _is_allowed_member(name, dataset_filter, include_baselines):
                        continue
                    if member.issym() or member.islnk():
                        warnings.append(f"Skipped link member: {name}")
                        continue
                    destination = _safe_member_destination(output_root, name)
                    parts = Path(name).parts
                    if len(parts) >= 3 and parts[1] == "datasets":
                        datasets.add(parts[2])
                    if member.isdir():
                        destination.mkdir(parents=True, exist_ok=True)
                        continue
                    if not member.isfile():
                        warnings.append(f"Skipped unsupported member type: {name}")
                        continue
                    if destination.exists() and not force:
                        files_skipped += 1
                        continue
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    source = tar.extractfile(member)
                    if source is None:
                        warnings.append(f"Skipped unreadable file member: {name}")
                        continue
                    with source, destination.open("wb") as out:
                        copied = shutil.copyfileobj(source, out)
                    if member.mode:
                        try:
                            destination.chmod(member.mode & 0o777)
                        except OSError:
                            pass
                    files_written += 1
                    size = member.size if member.size is not None else 0
                    bytes_written += int(size)
    finally:
        # ``with raw_handle, stream_handle`` closes both in the normal path.
        pass
    return files_written, files_skipped, bytes_written, sorted(datasets), warnings


def install_algorithm_benchmarks(
    *,
    manifest_path: str | os.PathLike[str] | None = None,
    output_root: str | os.PathLike[str] | None = None,
    asset_path: str | os.PathLike[str] | None = None,
    url: str | None = None,
    force: bool = False,
    datasets: Iterable[str] | None = None,
    include_baselines: bool = True,
    verify_sha256: bool = True,
) -> InstallResult:
    """Install benchmark assets into ``~/.cellcompass`` or a custom root."""
    manifest = load_manifest(manifest_path)
    asset = _default_asset(manifest)
    resolved_output_root = _expand_path(output_root) or DEFAULT_INSTALL_ROOT
    dataset_filter = {item for item in (datasets or []) if item}

    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    warnings: list[str] = []
    if asset_path:
        archive_path = _expand_path(asset_path)
        if archive_path is None or not archive_path.exists():
            raise FileNotFoundError(f"Benchmark asset not found: {asset_path}")
        source = str(archive_path)
    else:
        source_url = url or asset.get("url")
        if not source_url and not asset.get("parts"):
            raise ValueError("No benchmark asset path or URL is available.")
        temp_dir = tempfile.TemporaryDirectory(prefix="cytobridge_benchmarks_")
        archive_path = Path(temp_dir.name) / (asset.get("name") or Path(source_url or "benchmark.tar.zst").name)
        if url:
            _download_asset(url, archive_path)
            source = url
        else:
            _download_manifest_asset(asset, archive_path)
            source = asset.get("url") or "multipart manifest asset"

    try:
        expected_sha = asset.get("sha256")
        if verify_sha256 and expected_sha:
            actual_sha = sha256_file(archive_path)
            if actual_sha != expected_sha:
                raise ValueError(
                    "Benchmark asset checksum mismatch: "
                    f"expected {expected_sha}, got {actual_sha}"
                )

        written, skipped, bytes_written, installed_datasets, extract_warnings = _extract_archive(
            archive_path,
            resolved_output_root,
            force=force,
            dataset_filter=dataset_filter or None,
            include_baselines=include_baselines,
        )
        warnings.extend(extract_warnings)
        return InstallResult(
            ok=True,
            output_root=str(resolved_output_root),
            benchmark_root=str(resolved_output_root / "algorithm_benchmarks"),
            source=source,
            files_written=written,
            files_skipped=skipped,
            bytes_written=bytes_written,
            datasets=installed_datasets,
            warnings=warnings,
        )
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


def _iter_bundle_paths(source_root: Path, include_builtin_runs: bool) -> list[Path]:
    allowed_roots = ["registry.json", "datasets", "simulation_generators"]
    if include_builtin_runs:
        allowed_roots.append("builtin_runs")
    paths: list[Path] = []
    for name in allowed_roots:
        candidate = source_root / name
        if not candidate.exists():
            continue
        if candidate.is_file():
            paths.append(candidate)
        else:
            paths.extend(path for path in candidate.rglob("*") if path.is_file())
    return sorted(paths, key=lambda path: path.relative_to(source_root).as_posix())


def _add_paths_to_tar(tar: tarfile.TarFile, source_root: Path, paths: list[Path]) -> None:
    root_arcname = source_root.name
    root_info = tarfile.TarInfo(root_arcname)
    root_info.type = tarfile.DIRTYPE
    root_info.mode = 0o755
    root_info.mtime = 0
    tar.addfile(root_info)

    dirs_added = {root_arcname}
    for path in paths:
        rel = path.relative_to(source_root)
        parent_parts = rel.parent.parts
        running = root_arcname
        for part in parent_parts:
            running = f"{running}/{part}"
            if running in dirs_added:
                continue
            info = tarfile.TarInfo(running)
            info.type = tarfile.DIRTYPE
            info.mode = 0o755
            info.mtime = 0
            tar.addfile(info)
            dirs_added.add(running)
        tar.add(path, arcname=f"{root_arcname}/{rel.as_posix()}", recursive=False)


def build_algorithm_benchmark_bundle(
    *,
    source_root: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    include_builtin_runs: bool = False,
    compression_level: int = 19,
) -> dict[str, Any]:
    """Build a benchmark artifact archive and return its asset metadata."""
    resolved_source = _expand_path(source_root)
    resolved_output = _expand_path(output_path)
    if resolved_source is None or not resolved_source.exists():
        raise FileNotFoundError(f"Benchmark root does not exist: {source_root}")
    if resolved_output is None:
        raise ValueError("output_path is required")
    resolved_output.parent.mkdir(parents=True, exist_ok=True)

    paths = _iter_bundle_paths(resolved_source, include_builtin_runs)
    suffixes = "".join(resolved_output.suffixes)
    if suffixes.endswith(".tar.zst"):
        if zstd is None:
            raise RuntimeError("Building .tar.zst bundles requires the zstandard package.")
        compressor = zstd.ZstdCompressor(level=compression_level, threads=-1)
        with resolved_output.open("wb") as raw:
            with compressor.stream_writer(raw) as zfh:
                with tarfile.open(fileobj=zfh, mode="w|") as tar:
                    _add_paths_to_tar(tar, resolved_source, paths)
    elif suffixes.endswith(".tar.gz") or suffixes.endswith(".tgz"):
        with tarfile.open(resolved_output, "w:gz") as tar:
            _add_paths_to_tar(tar, resolved_source, paths)
    elif suffixes.endswith(".tar"):
        with tarfile.open(resolved_output, "w") as tar:
            _add_paths_to_tar(tar, resolved_source, paths)
    else:
        raise ValueError(f"Unsupported output archive format: {resolved_output.name}")

    return {
        "name": resolved_output.name,
        "path": str(resolved_output),
        "sha256": sha256_file(resolved_output),
        "size_bytes": resolved_output.stat().st_size,
        "file_count": len(paths),
        "archive_root": resolved_source.name,
        "include_builtin_runs": include_builtin_runs,
        "created_at": _utc_now(),
    }


def _read_json_if_present(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def collect_benchmark_manifest(
    *,
    source_root: str | os.PathLike[str],
    asset_metadata: dict[str, Any] | None = None,
    asset_url: str | None = None,
    version: str = "2026-05-03",
) -> dict[str, Any]:
    """Collect lightweight manifest metadata from a benchmark root."""
    resolved_source = _expand_path(source_root)
    if resolved_source is None or not resolved_source.exists():
        raise FileNotFoundError(f"Benchmark root does not exist: {source_root}")

    datasets: list[dict[str, Any]] = []
    for dataset_dir in sorted((resolved_source / "datasets").iterdir()):
        if not dataset_dir.is_dir():
            continue
        dataset_json = _read_json_if_present(dataset_dir / "dataset.json")
        leaderboard = _read_json_if_present(dataset_dir / "leaderboard.json")
        baseline_dir = dataset_dir / "builtin_baselines"
        artifact_dir = dataset_dir / "baseline_artifacts"
        datasets.append(
            {
                "dataset_id": dataset_json.get("dataset_id", dataset_dir.name),
                "title": dataset_json.get("title", dataset_dir.name),
                "contract_status": dataset_json.get("contract_status"),
                "tags": dataset_json.get("tags", []),
                "stage_relevance": dataset_json.get("stage_relevance", []),
                "n_obs": (dataset_json.get("data_profile") or {}).get("n_obs")
                or dataset_json.get("n_obs"),
                "n_vars": (dataset_json.get("data_profile") or {}).get("n_vars")
                or dataset_json.get("n_vars"),
                "time_points": (dataset_json.get("data_profile") or {}).get("time_points")
                or dataset_json.get("time_points", []),
                "builtin_baselines": sorted(path.stem for path in baseline_dir.glob("*.json"))
                if baseline_dir.exists()
                else [],
                "baseline_artifacts": sorted(path.name for path in artifact_dir.iterdir() if path.is_dir())
                if artifact_dir.exists()
                else [],
                "leaderboard_entries": len(leaderboard.get("entries") or []),
                "data_h5ad_size_bytes": (dataset_dir / "data.h5ad").stat().st_size
                if (dataset_dir / "data.h5ad").exists()
                else None,
            }
        )

    asset = {
        "name": DEFAULT_ASSET_NAME,
        "url": asset_url or DEFAULT_ASSET_URL,
        "sha256": "",
        "size_bytes": None,
        "archive_root": "algorithm_benchmarks",
        "contains": ["registry.json", "datasets", "simulation_generators"],
        "excluded": ["builtin_runs"],
    }
    if asset_metadata:
        asset.update({key: value for key, value in asset_metadata.items() if key != "path"})
        if asset_url:
            asset["url"] = asset_url

    return {
        "schema_version": 1,
        "name": "cellcompass-algorithm-benchmarks",
        "version": version,
        "created_at": _utc_now(),
        "recommended_install_root": "~/.cellcompass",
        "release_tag": DEFAULT_RELEASE_TAG,
        "assets": [asset],
        "datasets": datasets,
        "notes": [
            "The git repository stores only this manifest and installer code.",
            "Large h5ad files, saved trajectories, and baseline checkpoints are distributed as a release asset.",
            "Install into ~/.cellcompass so runtime tools find ~/.cellcompass/algorithm_benchmarks.",
        ],
    }


def _print_json(data: Any) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=False))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m cytobridge_agent.benchmark_artifacts",
        description="Manage CellCompass algorithm benchmark artifact bundles.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest_parser = subparsers.add_parser("manifest", help="Print the bundled manifest")
    manifest_parser.add_argument("--manifest", type=str, default=None, help="Optional manifest path")

    install_parser = subparsers.add_parser("install", help="Install benchmark artifacts")
    install_parser.add_argument("--manifest", type=str, default=None, help="Optional manifest path")
    install_parser.add_argument("--asset", type=str, default=None, help="Local archive path")
    install_parser.add_argument("--url", type=str, default=None, help="Release asset URL")
    install_parser.add_argument(
        "--output-root",
        type=str,
        default=str(DEFAULT_INSTALL_ROOT),
        help="Install root, default ~/.cellcompass",
    )
    install_parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    install_parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Install only one dataset id. Repeat for multiple datasets.",
    )
    install_parser.add_argument(
        "--no-baselines",
        action="store_true",
        help="Skip top-level builtin_runs entries if present in the archive.",
    )
    install_parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip sha256 verification.",
    )

    build_parser = subparsers.add_parser("build-bundle", help="Build a local benchmark bundle")
    build_parser.add_argument(
        "--source-root",
        type=str,
        default=str(DEFAULT_BENCHMARK_ROOT),
        help="Benchmark root to bundle",
    )
    build_parser.add_argument("--output", type=str, required=True, help="Archive output path")
    build_parser.add_argument(
        "--include-builtin-runs",
        action="store_true",
        help="Include top-level builtin_runs history. Defaults to false.",
    )

    collect_parser = subparsers.add_parser(
        "collect-manifest",
        help="Collect manifest metadata from a benchmark root",
    )
    collect_parser.add_argument(
        "--source-root",
        type=str,
        default=str(DEFAULT_BENCHMARK_ROOT),
        help="Benchmark root to inspect",
    )
    collect_parser.add_argument("--asset", type=str, default=None, help="Optional archive path")
    collect_parser.add_argument("--url", type=str, default=DEFAULT_ASSET_URL, help="Asset URL")
    collect_parser.add_argument("--version", type=str, default="2026-05-03")
    collect_parser.add_argument("--output", type=str, default=None, help="Write manifest here")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.command == "manifest":
        _print_json(load_manifest(args.manifest))
        return 0

    if args.command == "install":
        result = install_algorithm_benchmarks(
            manifest_path=args.manifest,
            output_root=args.output_root,
            asset_path=args.asset,
            url=args.url,
            force=args.force,
            datasets=args.dataset,
            include_baselines=not args.no_baselines,
            verify_sha256=not args.no_verify,
        )
        _print_json(result.as_dict())
        return 0

    if args.command == "build-bundle":
        metadata = build_algorithm_benchmark_bundle(
            source_root=args.source_root,
            output_path=args.output,
            include_builtin_runs=args.include_builtin_runs,
        )
        _print_json(metadata)
        return 0

    if args.command == "collect-manifest":
        asset_metadata = None
        if args.asset:
            asset_path = Path(args.asset).expanduser().resolve()
            asset_metadata = {
                "name": asset_path.name,
                "sha256": sha256_file(asset_path),
                "size_bytes": asset_path.stat().st_size,
                "archive_root": "algorithm_benchmarks",
                "contains": ["registry.json", "datasets", "simulation_generators"],
                "excluded": ["builtin_runs"],
            }
        manifest = collect_benchmark_manifest(
            source_root=args.source_root,
            asset_metadata=asset_metadata,
            asset_url=args.url,
            version=args.version,
        )
        if args.output:
            Path(args.output).expanduser().parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).expanduser().write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        else:
            _print_json(manifest)
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
