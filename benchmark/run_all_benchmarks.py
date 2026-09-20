#!/usr/bin/env python
"""Unified benchmark runner for registered DynBench entries.

This script is intentionally a thin orchestration layer. The canonical
per-benchmark runners still own task execution and evaluation; this layer
handles batch selection, timing, result collation, and summary tables.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_ROOT = ROOT / "benchmark"
sys.path.insert(0, str(ROOT))

from benchmark.result_paths import (
    remap_legacy_benchmark_output_root,
    resolve_dynbench_paths,
    resolve_realdynbench_paths,
)
from benchmark.timeout_policy import effective_timeout_for_agent


def allocate_fresh_output_dir(requested: Path) -> Path:
    requested = requested.resolve()
    if not requested.exists():
        return requested
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = requested.parent / f"{requested.name}_{timestamp}"
    suffix = 2
    while candidate.exists():
        candidate = requested.parent / f"{requested.name}_{timestamp}_{suffix}"
        suffix += 1
    return candidate


BENCHMARKS: dict[str, dict[str, Any]] = {
    "dynbench-s3": {
        "type": "dynbench",
        "scenario": "S3",
        "mode": "skills-on",
    },
    "dynbench-s3-sanity": {
        "type": "dynbench",
        "scenario": "S3_no_holdout_sanity",
        "mode": "skills-on",
    },
    "dynbench-test-toggle": {
        "type": "dynbench",
        "scenario": "Test_Toggle_Medium",
        "mode": "skills-on",
    },
}


@dataclass(frozen=True)
class RunSpec:
    key: str
    seed: int
    benchmark: dict[str, Any]


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_benchmarks(value: str) -> list[str]:
    if value == "all":
        return list(BENCHMARKS)
    requested = _parse_csv(value)
    unknown = [item for item in requested if item not in BENCHMARKS]
    if unknown:
        known = ", ".join(sorted(BENCHMARKS))
        raise SystemExit(f"Unknown benchmark(s): {', '.join(unknown)}\nKnown benchmarks: {known}")
    return requested


def parse_seeds(value: str) -> list[int]:
    try:
        return [int(item) for item in _parse_csv(value)]
    except ValueError as exc:
        raise SystemExit(f"--seeds must be a comma-separated integer list: {value}") from exc


def check_auth(agent_type: str) -> None:
    from benchmark.agent_config import load_saved_config, resolve_agent_selection

    selection = resolve_agent_selection(agent_type=agent_type, saved_config=load_saved_config())
    if selection.runner_id == "codex":
        print(f"Auth OK: agent_type={selection.agent_type} runner=codex_cli")
        return
    profile_id = selection.runner_kwargs.get("llm_profile_id")
    auth_mode = selection.runner_kwargs.get("llm_auth_mode")
    if auth_mode != "codex_oauth":
        raise SystemExit(f"{agent_type} resolved to unsupported auth mode for CI: {auth_mode}")
    if not profile_id:
        raise SystemExit(
            f"{agent_type} did not resolve to a Codex OAuth profile. "
            "Run: cytobridge-agent auth codex-login"
        )
    print(f"Auth OK: agent_type={selection.agent_type} auth_mode={auth_mode} profile={profile_id}")


def build_command(
    spec: RunSpec,
    *,
    agent_type: str,
    device: str,
    timeout_sec: float | None,
    extra_args: list[str],
    native_run_root: Path | None,
) -> list[str]:
    bench = spec.benchmark
    if bench["type"] == "dynbench":
        cmd = [
            sys.executable,
            "benchmark/dynbench/run_dynbench.py",
            "--scenario",
            bench["scenario"],
            "--mode",
            bench.get("mode", "skills-on"),
            "--device",
            device,
            "--seed",
            str(spec.seed),
            "--agent-type",
            agent_type,
        ]
        if timeout_sec is not None:
            cmd.extend(["--timeout", str(timeout_sec)])
    elif bench["type"] == "realdynbench":
        cmd = [
            sys.executable,
            "benchmark/realdynbench/run_realdynbench.py",
            "--config",
            bench["config"],
            "--fold",
            bench["fold"],
            "--mode",
            bench.get("mode", "skills-on"),
            "--device",
            device,
            "--seed",
            str(spec.seed),
            "--agent-type",
            agent_type,
        ]
    else:
        raise ValueError(f"Unsupported benchmark type: {bench['type']}")
    if native_run_root is not None:
        cmd.extend(["--run-root", str(native_run_root)])
    cmd.extend(extra_args)
    return cmd


def expected_native_result_path(spec: RunSpec, agent_type: str, native_run_root: Path | None) -> Path:
    bench = spec.benchmark
    run_name = f"agent_{agent_type}_{bench.get('mode', 'skills-on')}_seed{spec.seed}"
    if bench["type"] == "dynbench":
        _, _, output_dir = resolve_dynbench_paths(
            scenario=bench["scenario"],
            run_name=run_name,
            run_root=native_run_root,
        )
        return output_dir / "eval_results.json"

    dataset_id = _dataset_id_from_config(ROOT / bench["config"])
    _, _, output_dir = resolve_realdynbench_paths(
        dataset_id=dataset_id,
        fold_id=bench["fold"],
        run_name=run_name,
        run_root=native_run_root,
    )
    return output_dir / "eval_results.json"


def _dataset_id_from_config(path: Path) -> str:
    import yaml

    with path.open() as handle:
        card = yaml.safe_load(handle)
    return str(card["dataset_id"])


def extract_scores(payload: dict[str, Any]) -> dict[str, float | None]:
    results = payload.get("results", {}) if isinstance(payload, dict) else {}
    if "student_vs_teacher" in results:
        svt = results.get("student_vs_teacher", {})
        per_metric = svt.get("per_metric", {})
        scores = {
            name: _as_float(data.get("score") if isinstance(data, dict) else None)
            for name, data in per_metric.items()
        }
        scores["total_score"] = _as_float(svt.get("total_score"))
        return scores
    per_metric = results.get("per_metric", {})
    scores = {
        name: _as_float(data.get("score") if isinstance(data, dict) else None)
        for name, data in per_metric.items()
    }
    if "total_score" in results:
        scores["total_score"] = _as_float(results.get("total_score"))
    return scores


def _as_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def run_one(
    spec: RunSpec,
    *,
    agent_type: str,
    device: str,
    output_dir: Path,
    timeout_sec: float | None,
    extra_args: list[str],
    native_run_root: Path | None,
) -> dict[str, Any]:
    effective_timeout = effective_timeout_for_agent(agent_type, timeout_sec)
    run_id = f"{spec.key}_seed{spec.seed}"
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    cmd = build_command(
        spec,
        agent_type=agent_type,
        device=device,
        timeout_sec=timeout_sec,
        extra_args=extra_args,
        native_run_root=native_run_root,
    )
    native_result = expected_native_result_path(spec, agent_type, native_run_root)
    started_at = datetime.now().isoformat(timespec="seconds")
    t0 = time.time()

    (run_dir / "command.txt").write_text(" ".join(cmd) + "\n")
    with (run_dir / "runner_stdout.log").open("w", buffering=1) as stdout:
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            stdout=stdout,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=effective_timeout,
            env=_run_env(),
        )

    runtime_sec = time.time() - t0
    finished_at = datetime.now().isoformat(timespec="seconds")
    status = "succeeded" if proc.returncode == 0 and native_result.exists() else "failed"
    error = None
    original_payload: dict[str, Any] | None = None

    if native_result.exists():
        original_payload = json.loads(native_result.read_text())
        shutil.copy2(native_result, run_dir / "native_eval_results.json")
    else:
        error = f"native eval_results.json not found: {native_result}"
        native_error = native_result.with_name("error.json")
        if native_error.exists():
            shutil.copy2(native_error, run_dir / "native_error.json")
            try:
                error = json.loads(native_error.read_text()).get("error") or error
            except json.JSONDecodeError:
                pass

    record = {
        "status": status,
        "benchmark": spec.key,
        "benchmark_type": spec.benchmark["type"],
        "agent_type": agent_type,
        "seed": spec.seed,
        "runtime_sec": runtime_sec,
        "started_at": started_at,
        "finished_at": finished_at,
        "returncode": proc.returncode,
        "command": cmd,
        "run_dir": str(run_dir),
        "native_eval_results": str(native_result),
        "scores": extract_scores(original_payload or {}),
        "error": error,
    }
    (run_dir / "eval_results.json").write_text(json.dumps(record, indent=2, default=str))
    return record


def _run_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("NUMBA_CACHE_DIR", "/tmp/cytobridge_numba_cache")
    env.setdefault("MPLCONFIGDIR", "/tmp/cytobridge_mplconfig")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    return env


def write_summary(records: list[dict[str, Any]], output_dir: Path) -> None:
    fieldnames = [
        "benchmark",
        "benchmark_type",
        "agent_type",
        "seed",
        "status",
        "runtime_sec",
        "total_score",
        "M1_velocity",
        "M2_growth",
        "M3_distribution",
        "M4_fate",
        "M5_perturbation",
        "M6_grn",
        "error",
    ]
    with (output_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            scores = record.get("scores", {})
            row = {name: record.get(name) for name in fieldnames}
            row["runtime_sec"] = f"{record.get('runtime_sec', 0.0):.1f}"
            row["total_score"] = scores.get("total_score")
            for metric in fieldnames[7:13]:
                row[metric] = scores.get(metric)
            writer.writerow(row)

    lines = [
        "# Benchmark Summary",
        "",
        "| Benchmark | Agent | Seed | Status | Runtime sec | Total | Error |",
        "|---|---:|---:|---|---:|---:|---|",
    ]
    for record in records:
        scores = record.get("scores", {})
        total = scores.get("total_score")
        total_s = "" if total is None else f"{total:.3f}"
        error = (record.get("error") or "").replace("\n", " ")[:180]
        lines.append(
            f"| {record['benchmark']} | {record['agent_type']} | {record['seed']} | "
            f"{record['status']} | {record['runtime_sec']:.1f} | {total_s} | {error} |"
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n")
    (output_dir / "run_manifest.json").write_text(json.dumps(records, indent=2, default=str))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all CytoBridge benchmarks through one entry point.")
    parser.add_argument("--agent-type", default="codex", choices=["codex", "biomini", "biomni", "cytobridge"])
    parser.add_argument("--benchmarks", default="all", help="'all' or comma-separated benchmark keys")
    parser.add_argument("--seeds", "--seed", dest="seeds", default="42", help="Comma-separated seeds (default: 42). Use 42,137,256 for 3-seed tests.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--parallel", type=int, default=1, help="Number of concurrent runs. Default: serial.")
    parser.add_argument("--timeout", type=float, default=None, help="Per-run timeout in seconds.")
    parser.add_argument("--continue-on-failure", action="store_true", default=True)
    parser.add_argument("--check-auth-only", action="store_true")
    parser.add_argument("extra_args", nargs=argparse.REMAINDER, help="Arguments after -- are passed to benchmark runners.")
    args = parser.parse_args()

    if args.check_auth_only:
        check_auth(args.agent_type)
        return

    from benchmark.agent_config import normalize_agent_type

    check_auth(args.agent_type)
    resolved_agent_type = normalize_agent_type(args.agent_type)
    benchmark_keys = parse_benchmarks(args.benchmarks)
    seeds = parse_seeds(args.seeds)
    specs = [RunSpec(key=key, seed=seed, benchmark=BENCHMARKS[key]) for key in benchmark_keys for seed in seeds]

    raw_requested_output_dir = Path(args.output_dir) if args.output_dir else BENCHMARK_ROOT / "results" / f"batch_{datetime.now():%Y%m%d_%H%M%S}_{resolved_agent_type}"
    requested_output_dir = remap_legacy_benchmark_output_root(raw_requested_output_dir)
    output_dir = allocate_fresh_output_dir(requested_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    native_run_root = output_dir / "native_runs"
    native_run_root.mkdir(parents=True, exist_ok=True)

    extra_args = list(args.extra_args)
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]

    print(f"Running {len(specs)} benchmark run(s)")
    print(f"Agent: {resolved_agent_type}")
    if requested_output_dir != raw_requested_output_dir.resolve():
        print(
            "Legacy benchmark output path under cytobridge_output/figures was remapped to benchmark/results: "
            f"{requested_output_dir}"
        )
    if output_dir != requested_output_dir.resolve():
        print(f"Requested output dir already existed; using fresh batch dir: {output_dir}")
    print(f"Output: {output_dir}")

    records: list[dict[str, Any]] = []
    if args.parallel <= 1:
        for spec in specs:
            print(f"[run] {spec.key} seed={spec.seed}")
            try:
                record = run_one(
                    spec,
                    agent_type=resolved_agent_type,
                    device=args.device,
                    output_dir=output_dir,
                    timeout_sec=args.timeout,
                    extra_args=extra_args,
                    native_run_root=native_run_root,
                )
            except subprocess.TimeoutExpired as exc:
                record = {
                    "status": "failed",
                    "benchmark": spec.key,
                    "benchmark_type": spec.benchmark["type"],
                    "agent_type": resolved_agent_type,
                    "seed": spec.seed,
                    "runtime_sec": float(args.timeout or 0.0),
                    "scores": {},
                    "error": f"timeout after {exc.timeout}s",
                }
                run_dir = output_dir / f"{spec.key}_seed{spec.seed}"
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "eval_results.json").write_text(json.dumps(record, indent=2, default=str))
            records.append(record)
            print(f"[{record['status']}] {spec.key} seed={spec.seed} runtime={record.get('runtime_sec', 0.0):.1f}s")
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as pool:
            future_to_spec = {
                pool.submit(
                    run_one,
                    spec,
                    agent_type=resolved_agent_type,
                    device=args.device,
                        output_dir=output_dir,
                        timeout_sec=args.timeout,
                        extra_args=extra_args,
                        native_run_root=native_run_root,
                    ): spec
                for spec in specs
            }
            for future in concurrent.futures.as_completed(future_to_spec):
                spec = future_to_spec[future]
                try:
                    record = future.result()
                except Exception as exc:
                    record = {
                        "status": "failed",
                        "benchmark": spec.key,
                        "benchmark_type": spec.benchmark["type"],
                        "agent_type": resolved_agent_type,
                        "seed": spec.seed,
                        "runtime_sec": 0.0,
                        "scores": {},
                        "error": str(exc),
                    }
                    run_dir = output_dir / f"{spec.key}_seed{spec.seed}"
                    run_dir.mkdir(parents=True, exist_ok=True)
                    (run_dir / "eval_results.json").write_text(json.dumps(record, indent=2, default=str))
                records.append(record)
                print(f"[{record['status']}] {spec.key} seed={spec.seed} runtime={record.get('runtime_sec', 0.0):.1f}s")

    records.sort(key=lambda item: (item["benchmark"], item["seed"]))
    write_summary(records, output_dir)
    failed = [record for record in records if record["status"] != "succeeded"]
    print(f"Summary: {output_dir / 'summary.md'}")
    if failed:
        print(f"Failed runs: {len(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
