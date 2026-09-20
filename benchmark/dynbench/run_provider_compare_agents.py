#!/usr/bin/env python3
"""Run DynBench provider comparison with the same CytoBridge agent.

This launcher compares LLM providers inside the CytoBridge agent runtime.  It
does not run the Codex CLI agent.  Every job calls run_dynbench.py with
--agent-type cytobridge and only changes the CytoBridge LLM provider/model.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path("/lustre/home/2501111653/miniconda3/envs/agent/bin/python")
RUN_DYNBENCH = ROOT / "benchmark" / "dynbench" / "run_dynbench.py"


DEFAULT_SCENARIOS = [
    # Highest difficulty.
    "DB100_adversarial_001_seed6000",
    "DB100_adversarial_002_seed6001",
    "DB100_adversarial_003_seed6002",
    "DB100_adversarial_004_seed6003",
    "DB100_adversarial_005_seed6004",
    "DB100_adversarial_006_seed6005",
    "DB100_adversarial_007_seed6006",
    "DB100_adversarial_008_seed6007",
    "DB100_adversarial_009_seed6008",
    "DB100_adversarial_010_seed6009",
    # Hard cases.
    "DB100_hard_001_seed5000",
    "DB100_hard_004_seed5003",
    "DB100_hard_007_seed5006",
    "DB100_hard_010_seed5009",
    "DB100_hard_013_seed5012",
    "DB100_hard_019_seed5018",
    "DB100_hard_025_seed5024",
    # Medium anchors.
    "DB100_medium_008_seed4007",
    "DB100_medium_020_seed4019",
    "DB100_medium_035_seed4034",
]


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    provider: str
    model: str
    thinking: str
    auth_mode: str
    base_url: str | None = None


PROVIDERS = {
    "codex55_high": ProviderSpec(
        name="codex55_high",
        provider="openai-codex",
        model="gpt-5.5",
        thinking="high",
        auth_mode="codex_oauth",
    ),
    "xiaomi_high": ProviderSpec(
        name="xiaomi_high",
        provider="xiaomi",
        model="mimo-v2.5-pro",
        thinking="high",
        auth_mode="api_key",
    ),
}


@dataclass(frozen=True)
class Job:
    scenario: str
    provider_name: str
    repeat: int
    seed: int


def _safe_symlink_assets(run_root: Path, assets_root: Path | None) -> None:
    if not assets_root:
        return
    link = run_root / "dynbench_assets"
    if link.exists() or link.is_symlink():
        return
    try:
        link.symlink_to(assets_root.resolve(), target_is_directory=True)
    except OSError:
        (run_root / "dynbench_assets_path.txt").write_text(str(assets_root.resolve()), encoding="utf-8")


def output_dir_for(run_root: Path, scenario: str, provider_name: str, repeat: int) -> Path:
    run_label = f"cytobridge_{provider_name}_r{repeat:02d}"
    return run_root / "synthetic" / scenario / f"agent_cytobridge_skills-on_{run_label}"


def load_scores(eval_path: Path) -> dict[str, Any]:
    payload = json.loads(eval_path.read_text(encoding="utf-8"))
    results = payload.get("results") or {}
    per_metric = results.get("per_metric") or {}
    row: dict[str, Any] = {
        "total_score": results.get("total_score"),
        "runtime_sec": payload.get("runtime_sec"),
    }
    for key in ["M1_velocity", "M2_growth", "M3_distribution", "M4_fate", "M5_perturbation", "M6_grn"]:
        metric = per_metric.get(key) or {}
        row[key] = metric.get("score")
    return row


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(summary_jsonl: Path, summary_csv: Path) -> None:
    rows = []
    if summary_jsonl.exists():
        for line in summary_jsonl.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def completed_keys(summary_jsonl: Path) -> set[tuple[str, str, int]]:
    keys: set[tuple[str, str, int]] = set()
    if not summary_jsonl.exists():
        return keys
    for line in summary_jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("status") == "ok":
            keys.add((str(row.get("scenario")), str(row.get("provider_name")), int(row.get("repeat"))))
    return keys


def difficulty_rank(scenario: str) -> int:
    name = str(scenario or "").lower()
    if "easy" in name:
        return 0
    if "medium" in name:
        return 1
    if "hard" in name:
        return 2
    if "adversarial" in name:
        return 3
    return 2


def provider_scenario_order(provider_name: str, scenarios: list[str]) -> list[str]:
    """Use the same easy-to-hard scenario order for every provider."""
    return sorted(
        list(scenarios),
        key=lambda item: (difficulty_rank(item), item),
    )


def _first_env_csv(*names: str) -> str | None:
    """Return the first non-empty env value, splitting comma-separated pools."""
    for name in names:
        raw = os.environ.get(name)
        if not raw:
            continue
        for item in str(raw).split(","):
            value = item.strip()
            if value:
                return value
    return None


def provider_runtime_overrides(spec: ProviderSpec) -> tuple[str | None, str | None]:
    """Resolve provider-specific benchmark overrides without editing user config."""
    if spec.provider == "xiaomi":
        api_key = _first_env_csv("CYTOBRIDGE_XIAOMI_API_KEY", "CYTOBRIDGE_XIAOMI_API_KEYS")
        base_url = _first_env_csv("CYTOBRIDGE_XIAOMI_BASE_URL", "CYTOBRIDGE_XIAOMI_BASE_URLS")
        return api_key, base_url
    return None, None


def run_job(job: Job, args: argparse.Namespace) -> dict[str, Any]:
    spec = PROVIDERS[job.provider_name]
    override_api_key, override_base_url = provider_runtime_overrides(spec)
    out_dir = output_dir_for(args.run_root, job.scenario, job.provider_name, job.repeat)
    eval_path = out_dir / "eval_results.json"
    if eval_path.exists():
        row = {
            "status": "ok",
            "scenario": job.scenario,
            "provider_name": job.provider_name,
            "repeat": job.repeat,
            "seed": job.seed,
            "provider_used": spec.provider,
            "model_used": spec.model,
            "thinking_used": spec.thinking,
            "output_dir": str(out_dir),
            "resumed_from_existing_outputs": True,
        }
        row.update(load_scores(eval_path))
        return row

    run_label = f"cytobridge_{job.provider_name}_r{job.repeat:02d}"
    log_path = args.run_root / "logs" / job.provider_name / f"{job.scenario}__r{job.repeat:02d}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(PYTHON),
        str(RUN_DYNBENCH),
        "--scenario",
        job.scenario,
        "--agent-type",
        "cytobridge",
        "--mode",
        "skills-on",
        "--device",
        args.device,
        "--seed",
        str(job.seed),
        "--run-label",
        run_label,
        "--run-root",
        str(args.run_root),
        "--timeout",
        str(args.timeout),
        "--llm-provider",
        spec.provider,
        "--llm-model",
        spec.model,
        "--llm-thinking-level",
        spec.thinking,
        "--llm-auth-mode",
        spec.auth_mode,
    ]
    if spec.base_url:
        cmd.extend(["--llm-base-url", spec.base_url])
    if override_base_url:
        cmd.extend(["--llm-base-url", override_base_url])

    env = os.environ.copy()
    # Prevent accidental Codex fallback in Xiaomi rows; provider failures should
    # be visible as Xiaomi failures in this comparison.
    env.setdefault("CYTOBRIDGE_DISABLE_PROVIDER_FALLBACK", "1")
    if override_api_key:
        env["CYTOBRIDGE_OPENAI_API_KEY"] = override_api_key
    started = time.time()
    with log_path.open("w", encoding="utf-8") as log_handle:
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=float(args.timeout) + 180,
        )
    elapsed = time.time() - started
    row: dict[str, Any] = {
        "scenario": job.scenario,
        "provider_name": job.provider_name,
        "repeat": job.repeat,
        "seed": job.seed,
        "provider_used": spec.provider,
        "model_used": spec.model,
        "thinking_used": spec.thinking,
        "output_dir": str(out_dir),
        "log_path": str(log_path),
        "returncode": proc.returncode,
        "wall_sec": elapsed,
    }
    if eval_path.exists():
        row["status"] = "ok"
        row.update(load_scores(eval_path))
    else:
        row["status"] = "failed"
        error_path = out_dir / "error.json"
        log_tail = ""
        if log_path.exists():
            log_tail = log_path.read_text(encoding="utf-8", errors="replace")[-2000:]
        row["error"] = f"returncode={proc.returncode}; eval_results_exists=false; log_tail={log_tail}"
        if error_path.exists():
            row["error_json"] = error_path.read_text(encoding="utf-8", errors="replace")[:2000]
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare LLM providers inside the same CytoBridge DynBench agent.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--assets-root", type=Path, default=None)
    parser.add_argument("--scenarios", default=",".join(DEFAULT_SCENARIOS))
    parser.add_argument("--providers", default="codex55_high,xiaomi_high")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--parallel-per-provider", type=int, default=2)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--timeout", type=float, default=5400)
    parser.add_argument("--base-seed", type=int, default=26000)
    args = parser.parse_args()

    args.run_root = args.run_root.resolve()
    args.run_root.mkdir(parents=True, exist_ok=True)
    if args.assets_root:
        args.assets_root = args.assets_root.resolve()
        _safe_symlink_assets(args.run_root, args.assets_root)

    scenarios = [item.strip() for item in args.scenarios.split(",") if item.strip()]
    provider_names = [item.strip() for item in args.providers.split(",") if item.strip()]
    unknown = [name for name in provider_names if name not in PROVIDERS]
    if unknown:
        raise SystemExit(f"Unknown providers: {unknown}. Available: {sorted(PROVIDERS)}")

    manifest = {
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "comparison": "same CytoBridge agent; provider/model changed only",
        "run_root": str(args.run_root),
        "assets_root": str(args.assets_root) if args.assets_root else None,
        "scenarios": scenarios,
        "providers": {name: PROVIDERS[name].__dict__ for name in provider_names},
        "repeats": args.repeats,
        "parallel_per_provider": args.parallel_per_provider,
        "device": args.device,
        "timeout": args.timeout,
        "base_seed": args.base_seed,
    }
    (args.run_root / "provider_compare_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    summary_jsonl = args.run_root / "provider_compare_summary.jsonl"
    summary_csv = args.run_root / "provider_compare_summary.csv"
    done = completed_keys(summary_jsonl)
    jobs_by_provider: dict[str, list[Job]] = {name: [] for name in provider_names}
    for sidx, scenario in enumerate(scenarios):
        for provider_name in provider_names:
            for repeat in range(1, args.repeats + 1):
                if (scenario, provider_name, repeat) in done:
                    continue
                seed = args.base_seed + sidx * 100 + repeat
                jobs_by_provider[provider_name].append(Job(scenario, provider_name, repeat, seed))

    total = sum(len(jobs) for jobs in jobs_by_provider.values())
    print(f"Run root: {args.run_root}", flush=True)
    print(f"Scenarios: {len(scenarios)}; providers: {provider_names}; remaining jobs: {total}", flush=True)
    for provider_name in provider_names:
        ordered = provider_scenario_order(provider_name, scenarios)
        print(
            f"Scenario order for {provider_name}: {ordered[0] if ordered else 'none'} -> "
            f"{ordered[-1] if ordered else 'none'}",
            flush=True,
        )

    futures: dict[concurrent.futures.Future[dict[str, Any]], Job] = {}
    executors: list[concurrent.futures.ThreadPoolExecutor] = []
    try:
        for provider_name, jobs in jobs_by_provider.items():
            executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=args.parallel_per_provider,
                thread_name_prefix=f"dynbench_{provider_name}",
            )
            executors.append(executor)
            provider_order = provider_scenario_order(provider_name, scenarios)
            order_index = {scenario: idx for idx, scenario in enumerate(provider_order)}
            ordered_jobs = sorted(
                jobs,
                key=lambda job: (order_index.get(job.scenario, 10_000), job.repeat),
            )
            for job in ordered_jobs:
                futures[executor.submit(run_job, job, args)] = job

        completed = 0
        for future in concurrent.futures.as_completed(futures):
            job = futures[future]
            try:
                row = future.result()
            except subprocess.TimeoutExpired as exc:
                row = {
                    "status": "failed",
                    "scenario": job.scenario,
                    "provider_name": job.provider_name,
                    "repeat": job.repeat,
                    "seed": job.seed,
                    "error": f"timeout: {exc}",
                }
            except Exception as exc:  # noqa: BLE001 - benchmark runner must record failures.
                row = {
                    "status": "failed",
                    "scenario": job.scenario,
                    "provider_name": job.provider_name,
                    "repeat": job.repeat,
                    "seed": job.seed,
                    "error": repr(exc),
                }
            completed += 1
            row["completed_index"] = completed
            row["completed_total"] = total
            append_jsonl(summary_jsonl, row)
            write_csv(summary_jsonl, summary_csv)
            score = row.get("total_score")
            score_text = f"{score:.3f}" if isinstance(score, (int, float)) else "NA"
            print(
                f"[{completed}/{total}] {row.get('status')} "
                f"{job.provider_name} {job.scenario} r{job.repeat} total={score_text}",
                flush=True,
            )
    finally:
        for executor in executors:
            executor.shutdown(wait=False, cancel_futures=True)

    print(f"Summary JSONL: {summary_jsonl}", flush=True)
    print(f"Summary CSV:   {summary_csv}", flush=True)


if __name__ == "__main__":
    main()
