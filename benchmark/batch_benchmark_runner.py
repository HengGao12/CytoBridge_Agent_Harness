#!/usr/bin/env python
"""Batch DynBench scenario generation and three-agent comparison runner.

This script is the executable backend for the batch-benchmark-runner skill. It
keeps the workflow deliberately serial because Codex OAuth profile locking is a
shared resource across the three agent shells.

Default semantics:
- scenario data/task packages are reused when they already exist and validate
- missing synthetic scenarios are generated on demand
- agent benchmark runs are always executed fresh for the current batch
- matching historical batch results are reference-only and must not be treated as completion of the current request
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_ROOT = ROOT / "benchmark"
DYNBENCH_ROOT = BENCHMARK_ROOT / "dynbench"
sys.path.insert(0, str(ROOT))

from benchmark.result_paths import remap_legacy_benchmark_output_root, resolve_dynbench_paths
from benchmark.timeout_policy import effective_timeout_for_agent
from benchmark.agent_runners.auth_utils import clear_benchmark_auth_lock
from benchmark.dynbench.llm_topology_designer import design_topology_spec, save_topology_spec


PRESET_DEFAULTS: dict[str, dict[str, Any]] = {
    "easy": {
        "n_fates": 2,
        "n_reporters_per_tf": 1,
        "n_confounders": 1,
        "n_noise_genes": 0,
        "noise_sigma": 0.10,
        "n_time_bins": 6,
        "holdout_bins": [2, 3],
        "n_init_cells": 1500,
        "growth_type": "single",
        "t_end": 10.0,
    },
    "medium": {
        "n_fates": 4,
        "n_reporters_per_tf": 1,
        "n_confounders": 2,
        "n_noise_genes": 0,
        "noise_sigma": 0.12,
        "n_time_bins": 8,
        "holdout_bins": [2, 3],
        "n_init_cells": 2000,
        "growth_type": "multi",
        "t_end": 12.0,
    },
    "hard": {
        "n_fates": 8,
        "n_reporters_per_tf": 1,
        "n_confounders": 4,
        "n_noise_genes": 2,
        "noise_sigma": 0.15,
        "n_time_bins": 10,
        "holdout_bins": [3, 4, 5],
        "n_init_cells": 2500,
        "growth_type": "interaction",
        "t_end": 14.0,
    },
}

DEFAULT_DIFFICULTY = "medium"

AGENT_LABELS = {
    "codex": "Codex",
    "cytobridge": "CytoBridge",
    "biomini": "BioMini",
    "biomni": "BioMini",
}

METRIC_KEYS = [
    "M1_velocity",
    "M2_growth",
    "M3_distribution",
    "M4_fate",
    "M5_perturbation",
    "M6_grn",
]


@dataclass(frozen=True)
class RunTarget:
    scenario: str
    agent: str
    seed: int | str


def canonical_agent_id(agent: str) -> str:
    raw = str(agent).strip().lower()
    if raw == "biomni":
        return "biomini"
    return raw


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


def path_href(target: Path, anchor_dir: Path) -> str:
    return os.path.relpath(target.resolve(), start=anchor_dir.resolve()).replace(os.sep, "/")


def parse_csv(value: str | None, *, default: list[str]) -> list[str]:
    if value is None:
        return default
    items: list[str] = []
    for chunk in value.replace(";", ",").split(","):
        item = chunk.strip()
        if item:
            items.append(item)
    return items or default


def parse_int_csv(value: str | None, *, default: list[int]) -> list[int]:
    return [int(item) for item in parse_csv(value, default=[str(x) for x in default])]


def normalize_topology(value: str) -> str:
    raw = value.strip().lower().replace("-", "_")
    if raw in {"toggle", "toggle_switch", "toggle_switch_3gene"}:
        return "toggle_switch_3gene"
    if raw in {"complex_tree", "procedural", "tree"}:
        return "procedural"
    raise SystemExit(f"Unsupported topology: {value}")


def scenario_name_for(topology: str, difficulty: str, seed: int) -> str:
    label = "complex_tree" if topology == "procedural" else topology
    return f"{label}_{difficulty}_seed{seed}"


def scenario_name_for_spec(topology_spec: dict[str, Any], difficulty: str, seed: int) -> str:
    family = str(topology_spec.get("topology_family") or "llm_spec").strip().lower()
    return f"{family}_{difficulty}_seed{seed}"

def scenario_assets_ready(scenario: str, topology_spec: dict = None) -> bool:
    """Check if scenario assets (task package + ground truth) already exist AND match config."""
    task_dir = DYNBENCH_ROOT / "task_packages" / scenario
    gt_dir = DYNBENCH_ROOT / "ground_truth" / scenario
    required_task_files = [
        task_dir / "train.h5ad",
        task_dir / "TASK.md",
        task_dir / "prediction_targets.json",
    ]
    required_gt_files = [
        gt_dir / "simulation_config.json",
        gt_dir / "grn_ground_truth.json",
    ]
    if not all(path.exists() for path in [*required_task_files, *required_gt_files]):
        return False
    
    # If topology_spec provided, verify the existing assets match the requested config
    if topology_spec is not None:
        sim_config_path = gt_dir / "simulation_config.json"
        if sim_config_path.exists():
            try:
                import json
                existing_config = json.loads(sim_config_path.read_text())
                # Check key fields that affect the scenario
                key_fields = ["n_confounders", "n_noise_genes", "n_fates", "n_time_bins",
                              "n_reporters_per_tf", "growth_type", "holdout_bins"]
                for field in key_fields:
                    spec_val = topology_spec.get(field)
                    existing_val = existing_config.get(field)
                    if spec_val is not None and existing_val is not None and spec_val != existing_val:
                        return False
            except Exception:
                pass  # If we can't read config, assume it's ready (will regenerate on force)
    
    return True


def extract_seed_from_scenario_name(scenario_name: str) -> int:
    """Extract numeric seed from scenario name like 'branching_tree_medium_seed137'."""
    import re
    match = re.search(r'seed(\d+)$', scenario_name)
    if match:
        return int(match.group(1))
    return 42


def ensure_v2_scenarios(
    *,
    output_dir: Path,
    scenario_names: list[str] | None = None,
    force_regenerate: bool = False,
) -> tuple[list[str], dict[str, str]]:
    """
    Generate v2 scenarios from deterministic config files.
    No external seed is used; scenario names have no seed suffix.
    Only run_seeds (for agent execution) are separate.
    """
    from benchmark.dynbench.simulators.scenario_factory import generate_v2_scenario
    from benchmark.dynbench.simulators.scenario_registry import get_all_v2_names, CONFIG_DIR

    if scenario_names is None:
        scenario_names = get_all_v2_names()

    scenarios: list[str] = []
    scenario_sources: dict[str, str] = {}

    for name in scenario_names:
        config_path = CONFIG_DIR / f"{name}.json"
        if not config_path.exists():
            print(f"WARNING: Config not found for {name}, skipping", flush=True)
            continue

        scenarios.append(name)

        # Load config to check if existing assets match
        try:
            _spec_payload = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            _spec_payload = None
        if not force_regenerate and scenario_assets_ready(name, _spec_payload):
            scenario_sources[name] = "existing"
            print(f"Reusing existing v2 scenario assets: {name}", flush=True)
            continue

        # Generate from deterministic config (no seed in name)
        print(f"Generating v2 scenario: {name}", flush=True)
        generate_v2_scenario(
            config_path=config_path,
            output_base=output_dir / "dynbench_assets",
            verbose=True,
        )
        scenario_sources[name] = "generated"
        print(f"Generated v2 scenario assets: {name}", flush=True)

    return scenarios, scenario_sources


def clear_auth_lock() -> None:
    clear_benchmark_auth_lock()


def scenario_config(topology: str, difficulty: str, seed: int) -> dict[str, Any]:
    if difficulty not in PRESET_DEFAULTS:
        raise SystemExit(f"Unsupported difficulty: {difficulty}")
    config = dict(PRESET_DEFAULTS[difficulty])
    config.update(
        {
            "name": f"{topology}_{difficulty}",
            "scenario_name": scenario_name_for(topology, difficulty, seed),
            "topology": topology,
            "seed": seed,
        }
    )
    if topology == "toggle_switch_3gene":
        # Keep the compact 3-gene topology while still allowing difficulty to
        # control noise, time-bin count, and cell count.
        config.update(
            {
                "n_fates": 2,
                "n_reporters_per_tf": 1,
                "n_confounders": 0,
                "n_noise_genes": 0,
                "growth_type": "single",
            }
        )
    return config


def scenario_config_from_spec(topology_spec: dict[str, Any], difficulty: str, seed: int) -> dict[str, Any]:
    config = dict(PRESET_DEFAULTS[difficulty])
    config.update(
        {
            "name": f"{topology_spec['topology_family']}_{difficulty}",
            "scenario_name": scenario_name_for_spec(topology_spec, difficulty, seed),
            "topology": "llm_spec",
            "topology_spec": topology_spec,
            "seed": seed,
            "n_fates": topology_spec.get("n_fates", config["n_fates"]),
            "n_reporters_per_tf": topology_spec.get("n_reporters_per_tf", config["n_reporters_per_tf"]),
            "n_confounders": topology_spec.get("n_confounders", config["n_confounders"]),
            "n_noise_genes": topology_spec.get("n_noise_genes", config["n_noise_genes"]),
            "noise_sigma": topology_spec.get("noise_sigma", config["noise_sigma"]),
            "n_time_bins": topology_spec.get("n_time_bins", config["n_time_bins"]),
            "holdout_bins": topology_spec.get("holdout_bins", config["holdout_bins"]),
            "n_init_cells": topology_spec.get("n_init_cells", config["n_init_cells"]),
            "growth_type": topology_spec.get("growth_type", config["growth_type"]),
            "t_end": topology_spec.get("t_end", config["t_end"]),
        }
    )
    return config


def ensure_scenarios(
    *,
    topology: str,
    difficulties: list[str],
    seeds: list[int],
    output_dir: Path,
    topology_request: str | None = None,
    topology_spec_file: str | None = None,
    force_regenerate: bool = False,
) -> tuple[list[str], dict[str, str]]:
    config_dir = output_dir / "scenario_configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    topology_spec_dir = output_dir / "topology_specs"
    topology_spec_dir.mkdir(parents=True, exist_ok=True)

    scenarios: list[str] = []
    scenario_sources: dict[str, str] = {}
    for difficulty in difficulties:
        for seed in seeds:
            if topology_spec_file:
                spec_payload = json.loads(Path(topology_spec_file).read_text(encoding="utf-8"))
                cfg = scenario_config_from_spec(spec_payload, difficulty, seed)
            elif topology_request:
                spec_payload = design_topology_spec(
                    request_text=topology_request,
                    difficulty=difficulty,
                    seed=seed,
                )
                spec_path = topology_spec_dir / f"{difficulty}_seed{seed}.json"
                save_topology_spec(spec_payload, spec_path)
                cfg = scenario_config_from_spec(spec_payload, difficulty, seed)
                cfg["topology_spec_path"] = str(spec_path)
            else:
                cfg = scenario_config(topology, difficulty, seed)
            scenario_name = str(cfg["scenario_name"])
            scenarios.append(scenario_name)
            config_path = config_dir / f"{scenario_name}.json"
            config_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
            if not force_regenerate and scenario_assets_ready(scenario_name, spec_payload if topology_spec_file else None):
                scenario_sources[scenario_name] = "existing"
                print(f"Reusing existing scenario assets: {scenario_name}", flush=True)
                continue
            cmd = [
                sys.executable,
                "benchmark/dynbench/simulators/scenario_factory.py",
                "--config",
                str(config_path),
                "--quiet",
            ]
            subprocess.run(cmd, cwd=ROOT, check=True)
            scenario_sources[scenario_name] = "generated"
            print(f"Generated scenario assets: {scenario_name}", flush=True)
    return scenarios, scenario_sources


def expected_eval_path(target: RunTarget, *, native_run_root: Path) -> Path:
    agent_id = canonical_agent_id(target.agent)
    # Support both numeric seeds (seed42) and string labels (run1)
    if isinstance(target.seed, int):
        seed_label = f"seed{target.seed}"
    else:
        seed_label = str(target.seed)
    _, _, output_dir = resolve_dynbench_paths(
        scenario=target.scenario,
        run_name=f"agent_{agent_id}_skills-on_{seed_label}",
        run_root=native_run_root,
    )
    return output_dir / "eval_results.json"


def expected_error_path(target: RunTarget, *, native_run_root: Path) -> Path:
    return expected_eval_path(target, native_run_root=native_run_root).with_name("error.json")


def load_run_result_metadata(target: RunTarget, *, native_run_root: Path) -> dict[str, Any]:
    eval_path = expected_eval_path(target, native_run_root=native_run_root)
    error_path = expected_error_path(target, native_run_root=native_run_root)
    payload: dict[str, Any] = {}
    if eval_path.exists():
        try:
            payload = json.loads(eval_path.read_text(encoding="utf-8"))
            return payload.get("run_result") or {}
        except Exception:
            return {}
    if error_path.exists():
        try:
            payload = json.loads(error_path.read_text(encoding="utf-8"))
            return payload.get("run_result") or {}
        except Exception:
            return {}
    return {}


def run_benchmark(
    target: RunTarget,
    *,
    output_dir: Path,
    timeout_sec: float | None,
    native_run_root: Path,
) -> dict[str, Any]:
    effective_timeout = effective_timeout_for_agent(target.agent, timeout_sec)
    # Support both numeric seeds (seed42) and string labels (run1)
    seed_label = f"seed{target.seed}" if isinstance(target.seed, int) else str(target.seed)
    run_dir = output_dir / target.scenario / f"{target.agent}_{seed_label}"
    run_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = run_dir / "runner_stdout.log"
    cmd = [
        sys.executable,
        "benchmark/dynbench/run_dynbench.py",
        "--scenario",
        target.scenario,
        "--agent-type",
        target.agent,
        "--mode",
        "skills-on",
        "--device",
        "cpu",
        "--run-root",
        str(native_run_root),
    ]
    # Only pass --seed for numeric seeds (codex/cytobridge), skip for repeat mode
    if isinstance(target.seed, int):
        cmd.insert(-2, "--seed")
        cmd.insert(-2, str(target.seed))
    else:
        # Repeat mode: pass --run-label instead of --seed
        cmd.insert(-2, "--run-label")
        cmd.insert(-2, str(target.seed))
    started = datetime.now().isoformat(timespec="seconds")
    t0 = time.time()
    print(f"Running: {target.scenario} / {target.agent} / {seed_label}", flush=True)
    clear_auth_lock()
    with stdout_path.open("w", encoding="utf-8") as handle:
        last_proc = subprocess.run(
            cmd,
            cwd=ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=effective_timeout,
        )
    clear_auth_lock()

    runtime = time.time() - t0
    eval_path = expected_eval_path(target, native_run_root=native_run_root)
    error_path = expected_error_path(target, native_run_root=native_run_root)
    run_result = load_run_result_metadata(target, native_run_root=native_run_root)
    status = "passed" if eval_path.exists() else "failed"
    format_status = str(run_result.get("format_status") or "")
    if eval_path.exists() and format_status == "native_complete":
        strict_status = "native_complete"
    elif eval_path.exists() and format_status == "recovered_complete":
        strict_status = "format_recovered"
    else:
        strict_status = "failed"
    error_message = None
    if status == "failed":
        if error_path.exists():
            try:
                error_message = json.loads(error_path.read_text(encoding="utf-8")).get("error")
            except Exception:
                error_message = error_path.read_text(encoding="utf-8")[:1000]
        else:
            error_message = f"returncode={getattr(last_proc, 'returncode', 'unknown')}; eval_results.json not found"
    return {
        "scenario": target.scenario,
        "agent": target.agent,
        "seed": target.seed,
        "status": status,
        "strict_status": strict_status,
        "started_at": started,
        "runtime_sec": runtime,
        "command": cmd,
        "log": str(stdout_path),
        "eval_results": str(eval_path) if eval_path.exists() else None,
        "error": error_message,
        "native_status": run_result.get("native_status"),
        "format_status": run_result.get("format_status"),
        "recovery_method": run_result.get("recovery_method"),
        "recovery_status": run_result.get("recovery_status"),
        "artifact_origin": run_result.get("artifact_origin"),
        "recovery_manifest": run_result.get("recovery_manifest"),
        "native_error": run_result.get("native_error_message"),
        "scores": extract_scores(eval_path) if eval_path.exists() else {},
        "cost_summary": _extract_cost_summary(eval_path) if eval_path.exists() else None,
    }


def extract_scores(path: Path) -> dict[str, float | None]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    results = payload.get("results") or {}
    per_metric = results.get("per_metric") or {}
    scores = {"total_score": as_float(results.get("total_score"))}
    for key in METRIC_KEYS:
        metric_payload = per_metric.get(key) or {}
        scores[key] = as_float(metric_payload.get("score"))
    return scores


def _extract_cost_summary(path: Path) -> dict[str, Any] | None:
    """Extract cost_summary from eval_results.json auxiliary section."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload.get("auxiliary", {}).get("cost_summary")
    except Exception:
        return None


def as_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def fmt(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.4f}"
    return ""


def write_reports(records: list[dict[str, Any]], output_dir: Path, summary_file: Path | None) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "results.json"
    json_path.write_text(json.dumps(records, indent=2, default=str), encoding="utf-8")
    manifest_path = output_dir / "run_manifest.json"
    manifest = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}

    lines = [
        "# Synthetic Benchmark Batch Summary",
        "",
        f"Generated at: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "Run policy: synthetic scenario assets may be reused, but agent benchmark runs in this batch are executed fresh.",
        "",
        "| Scenario | Agent | Seed | Status | M1 | M2 | M3 | M4 | M5 | M6 | Total | Runtime(s) | Cost($) |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    scenario_sources = manifest.get("scenario_sources") if isinstance(manifest, dict) else None
    if isinstance(scenario_sources, dict) and scenario_sources:
        lines.extend(["", "## Scenario Asset Source", ""])
        for scenario, source in sorted(scenario_sources.items()):
            lines.append(f"- `{scenario}`: `{source}`")
        lines.append("")
    for record in records:
        scores = record.get("scores") or {}
        cost = record.get("cost_summary") or {}
        cost_val = cost.get("total_cost_estimate", 0.0) or 0.0
        lines.append(
            "| {scenario} | {agent} | {seed} | {status} | {m1} | {m2} | {m3} | {m4} | {m5} | {m6} | {total} | {runtime} | {cost} |".format(
                scenario=record["scenario"],
                agent=AGENT_LABELS.get(record["agent"], record["agent"]),
                seed=record["seed"],
                status=f"{record['status']} / strict={record.get('strict_status', '')}",
                m1=fmt(scores.get("M1_velocity")),
                m2=fmt(scores.get("M2_growth")),
                m3=fmt(scores.get("M3_distribution")),
                m4=fmt(scores.get("M4_fate")),
                m5=fmt(scores.get("M5_perturbation")),
                m6=fmt(scores.get("M6_grn")),
                total=fmt(scores.get("total_score")),
                runtime=fmt(record.get("runtime_sec")),
                cost=fmt(cost_val),
            )
        )

    failures = [r for r in records if r.get("status") != "passed"]
    if failures:
        lines.extend(["", "## Failures", ""])
        for record in failures:
            lines.append(
                f"- `{record['scenario']}` / `{record['agent']}` / seed `{record['seed']}`: {record.get('error') or 'unknown error'}"
            )

    recovery_notes = [
        r for r in records
        if r.get("recovery_status") not in {None, "", "not_applied"} or r.get("strict_status") == "failed"
    ]
    if recovery_notes:
        lines.extend(["", "## Strict And Recovery Notes", ""])
        for record in recovery_notes:
            details: list[str] = []
            if record.get("strict_status"):
                details.append(f"strict={record['strict_status']}")
            if record.get("native_status"):
                details.append(f"native={record['native_status']}")
            if record.get("format_status"):
                details.append(f"format={record['format_status']}")
            if record.get("recovery_method"):
                details.append(f"recovery_method={record['recovery_method']}")
            if record.get("recovery_status") not in {None, "", "not_applied"}:
                details.append(f"recovery_status={record['recovery_status']}")
            if record.get("artifact_origin"):
                details.append(f"artifact_origin={record['artifact_origin']}")
            if record.get("recovery_manifest"):
                details.append(f"recovery_manifest={record['recovery_manifest']}")
            if record.get("native_error"):
                details.append(f"native_error={record['native_error']}")
            lines.append(
                f"- `{record['scenario']}` / `{record['agent']}` / seed `{record['seed']}`: "
                + "; ".join(details)
            )

    markdown = "\n".join(lines) + "\n"
    report_path = output_dir / "summary.md"
    report_path.write_text(markdown, encoding="utf-8")
    if summary_file is not None:
        summary_file.parent.mkdir(parents=True, exist_ok=True)
        summary_file.write_text(markdown, encoding="utf-8")
    return report_path


def write_batch_manifest(
    *,
    output_dir: Path,
    requested_output_dir: Path | None,
    topology: str,
    topology_request: str | None,
    topology_spec_file: str | None,
    difficulties: list[str],
    seeds: list[int],
    agents: list[str],
    scenarios: list[str],
    scenario_sources: dict[str, str],
    run_seeds: list[int] | None = None,
) -> Path:
    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "requested_output_dir": str(requested_output_dir) if requested_output_dir else None,
        "actual_output_dir": str(output_dir),
        "topology": topology,
        "topology_request": topology_request or None,
        "topology_spec_file": topology_spec_file,
        "difficulties": difficulties,
        "seeds": seeds,
        "run_seeds": run_seeds,
        "agents": agents,
        "scenarios": scenarios,
        "scenario_sources": scenario_sources,
        "run_policy": {
            "scenario_assets": "reuse_if_present_else_generate",
            "benchmark_runs": "always_rerun_for_current_batch",
            "historical_results": "reference_only_not_completion_evidence",
        },
    }
    path = output_dir / "run_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def _safe_slug(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value)


def _write_scenario_report(
    *,
    scenario: str,
    scenario_records: list[dict[str, Any]],
    output_dir: Path,
) -> Path:
    import pandas as pd

    figures_dir = output_dir / "comparisons" / f"{_safe_slug(scenario)}_comparison"
    figures_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for record in scenario_records:
        scores = record.get("scores") or {}
        rows.append(
            {
                "scenario": scenario,
                "agent": AGENT_LABELS.get(record["agent"], record["agent"]),
                "seed": record["seed"],
                "final_status": record.get("status"),
                "strict_status": record.get("strict_status"),
                "native_status": record.get("native_status"),
                "format_status": record.get("format_status"),
                "artifact_origin": record.get("artifact_origin"),
                "recovery_method": record.get("recovery_method"),
                "recovery_status": record.get("recovery_status"),
                "recovery_manifest": record.get("recovery_manifest"),
                "runtime_sec": record.get("runtime_sec"),
                "official_error": record.get("error"),
                "native_error": record.get("native_error"),
                "total_score": scores.get("total_score"),
                **{key: scores.get(key) for key in METRIC_KEYS},
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return output_dir / f"report_{_safe_slug(scenario)}.html"

    comparison_csv = figures_dir / "agent_comparison_metrics.csv"
    df.to_csv(comparison_csv, index=False)

    ranking_df = df.copy()
    ranking_df["_status_rank"] = ranking_df["final_status"].map({"passed": 0, "failed": 1}).fillna(2)
    ranking_df = ranking_df.sort_values(
        by=["_status_rank", "total_score"],
        ascending=[True, False],
        na_position="last",
    ).drop(columns=["_status_rank"]).reset_index(drop=True)
    ranking_df.insert(0, "rank", ranking_df.index + 1)
    ranking_csv = figures_dir / "agent_rankings.csv"
    ranking_df.to_csv(ranking_csv, index=False)

    generated_config = output_dir / "scenario_configs" / f"{scenario}.json"
    scenario_summary = {
        "scenario": scenario,
        "generated_config": str(generated_config) if generated_config.exists() else None,
        "comparison_csv": str(comparison_csv),
        "ranking_csv": str(ranking_csv),
        "records": rows,
    }

    plot_paths: list[Path] = []
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        plot_df = df.copy()
        plot_df["runtime_min"] = plot_df["runtime_sec"].astype(float) / 60.0
        axes[0].bar(plot_df["agent"], plot_df["total_score"].fillna(0.0), color="#5B8FF9")
        axes[0].set_title("Total Score")
        axes[0].set_ylim(0, 1.05)
        axes[1].bar(plot_df["agent"], plot_df["runtime_min"].fillna(0.0), color="#F6BD16")
        axes[1].set_title("Runtime (min)")
        total_plot = figures_dir / "total_and_runtime.png"
        fig.tight_layout()
        fig.savefig(total_plot, dpi=160)
        plt.close(fig)
        plot_paths.append(total_plot)

        fig, ax = plt.subplots(figsize=(8, max(2.5, 0.8 * len(plot_df))))
        heat = plot_df.set_index("agent")[METRIC_KEYS].astype(float).fillna(0.0)
        im = ax.imshow(heat.values, aspect="auto", cmap="YlGnBu", vmin=0.0, vmax=1.0)
        ax.set_xticks(range(len(METRIC_KEYS)))
        ax.set_xticklabels(METRIC_KEYS, rotation=30, ha="right")
        ax.set_yticks(range(len(heat.index)))
        ax.set_yticklabels(list(heat.index))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        heatmap_path = figures_dir / "per_metric_heatmap.png"
        fig.tight_layout()
        fig.savefig(heatmap_path, dpi=160)
        plt.close(fig)
        plot_paths.append(heatmap_path)

        fig, ax = plt.subplots(figsize=(11, 4.8))
        metric_positions = list(range(len(METRIC_KEYS)))
        width = 0.8 / max(1, len(plot_df))
        for idx, (_, row) in enumerate(plot_df.iterrows()):
            vals = [float(row.get(key) or 0.0) for key in METRIC_KEYS]
            xs = [x - 0.4 + width / 2 + idx * width for x in metric_positions]
            ax.bar(xs, vals, width=width, label=row["agent"])
        ax.set_xticks(metric_positions)
        ax.set_xticklabels(METRIC_KEYS, rotation=20, ha="right")
        ax.set_ylim(0, 1.05)
        ax.legend()
        grouped_path = figures_dir / "per_metric_grouped_bars.png"
        fig.tight_layout()
        fig.savefig(grouped_path, dpi=160)
        plt.close(fig)
        plot_paths.append(grouped_path)
    except Exception as exc:
        scenario_summary["plot_error"] = str(exc)

    scenario_summary["plots"] = [str(path) for path in plot_paths]
    summary_json_path = figures_dir / "comparison_summary.json"
    summary_json_path.write_text(json.dumps(scenario_summary, indent=2, default=str), encoding="utf-8")

    report_path = output_dir / f"report_{_safe_slug(scenario)}.html"
    html_lines = [
        "<!DOCTYPE html>",
        "<html lang='zh'>",
        "<head>",
        "  <meta charset='utf-8'>",
        f"  <title>{scenario} benchmark report</title>",
        "  <style>",
        "    body { font-family: Arial, Helvetica, sans-serif; margin: 32px; color: #222; line-height: 1.55; }",
        "    h1, h2, h3 { color: #123a63; }",
        "    code, pre { background: #f5f7fa; padding: 2px 4px; border-radius: 4px; }",
        "    table { border-collapse: collapse; width: 100%; margin: 16px 0 24px 0; }",
        "    th, td { border: 1px solid #cfd8e3; padding: 8px 10px; text-align: left; font-size: 14px; }",
        "    th { background: #eef4fb; }",
        "    img { max-width: 100%; border: 1px solid #d9e1ea; margin: 10px 0 18px 0; }",
        "  </style>",
        "</head>",
        "<body>",
        f"  <h1>{scenario} Benchmark Report</h1>",
        f"  <p>Generated at: {datetime.now().isoformat(timespec='seconds')}</p>",
        "  <h2>Run Summary</h2>",
        "  <table>",
        "    <thead><tr><th>Agent</th><th>Seed</th><th>Final</th><th>Strict</th><th>Native</th><th>Recovery</th><th>Origin</th><th>M1</th><th>M2</th><th>M3</th><th>M4</th><th>M5</th><th>M6</th><th>Total</th></tr></thead>",
        "    <tbody>",
    ]
    for _, row in ranking_df.iterrows():
        html_lines.append(
            "      <tr>"
            f"<td>{row['agent']}</td>"
            f"<td>{row['seed']}</td>"
            f"<td>{row['final_status']}</td>"
            f"<td>{row['strict_status']}</td>"
            f"<td>{row['native_status']}</td>"
            f"<td>{row['recovery_status']}</td>"
            f"<td>{row['artifact_origin']}</td>"
            f"<td>{fmt(row.get('M1_velocity'))}</td>"
            f"<td>{fmt(row.get('M2_growth'))}</td>"
            f"<td>{fmt(row.get('M3_distribution'))}</td>"
            f"<td>{fmt(row.get('M4_fate'))}</td>"
            f"<td>{fmt(row.get('M5_perturbation'))}</td>"
            f"<td>{fmt(row.get('M6_grn'))}</td>"
            f"<td>{fmt(row.get('total_score'))}</td>"
            "</tr>"
        )
    html_lines.extend(["    </tbody>", "  </table>"])
    if plot_paths:
        html_lines.extend(["  <h2>Figures</h2>"])
        for plot in plot_paths:
            rel = path_href(plot, report_path.parent)
            html_lines.append(f"  <img src='{rel}' alt='{plot.name}'>")
    html_lines.extend(
        [
            "  <h2>Artifacts</h2>",
            "  <ul>",
            f"    <li>Comparison CSV: <code>{comparison_csv}</code></li>",
            f"    <li>Ranking CSV: <code>{ranking_csv}</code></li>",
            f"    <li>Comparison summary JSON: <code>{summary_json_path}</code></li>",
            "  </ul>",
            "</body>",
            "</html>",
        ]
    )
    report_path.write_text("\n".join(html_lines) + "\n", encoding="utf-8")
    return report_path


def write_visual_reports(records: list[dict[str, Any]], output_dir: Path) -> list[Path]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["scenario"])].append(record)
    report_paths: list[Path] = []
    for scenario, scenario_records in sorted(grouped.items()):
        report_paths.append(
            _write_scenario_report(
                scenario=scenario,
                scenario_records=scenario_records,
                output_dir=output_dir,
            )
        )
    return report_paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", default="toggle_switch_3gene")
    parser.add_argument("--topology-request", default="", help="Natural-language topology request; triggers LLM-driven topology spec mode.")
    parser.add_argument("--topology-request-file", default="", help="Path to a text file containing a natural-language topology request.")
    parser.add_argument("--topology-spec-file", default="", help="Path to a precomputed topology_spec.json; skips LLM design.")
    parser.add_argument("--difficulties", default=DEFAULT_DIFFICULTY)
    parser.add_argument("--seeds", default="42", help="Legacy scenario-generation seeds for non-v2 synthetic scenarios. V2 scenarios ignore this for data generation.")
    parser.add_argument("--run-seeds", default=None, help="Legacy/debug numeric run seeds for non-v2 agent execution. For v2 benchmark repeats, use --repeat instead.")
    parser.add_argument("--repeat", type=int, default=None, help="Run each scenario N times with no agent seed control (LLM natural variation). Recommended for v2 benchmarks.")
    parser.add_argument("--agents", default="codex,cytobridge,biomini")
    parser.add_argument("--scenarios", default=None, help="Comma-separated existing scenarios; skips name generation.")
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument(
        "--force-regenerate",
        action="store_true",
        help="Regenerate scenario assets even if task package and ground truth already exist.",
    )
    parser.add_argument("--timeout-sec", type=float, default=None)
    parser.add_argument(
        "--fail-on-agent-error",
        action="store_true",
        help="Return non-zero when any agent run fails. Default behavior is to return 0 after writing reports.",
    )
    parser.add_argument(
        "--output-dir",
        default=f"benchmark/results/synthetic_batches/batch_synthetic_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    parser.add_argument(
        "--summary-file",
        default="",
        help=(
            "Optional extra fixed-path Markdown summary. By default the runner only writes "
            "summary.md inside the current batch output directory."
        ),
    )
    parser.add_argument(
        "--v2-scenarios",
        default=None,
        help=(
            "Comma-separated v2 scenario names to generate and run. "
            "Uses deterministic config files (no seed in scenario name). "
            "Example: --v2-scenarios baseline_25g_8f,gene_gradient_12g_8f"
        ),
    )
    args = parser.parse_args()

    topology_request = args.topology_request.strip()
    if args.topology_request_file:
        topology_request = Path(args.topology_request_file).read_text(encoding="utf-8").strip()
    topology_spec_file = args.topology_spec_file.strip() or None
    topology = normalize_topology(args.topology)
    difficulties = parse_csv(args.difficulties, default=[DEFAULT_DIFFICULTY])
    seeds = parse_int_csv(args.seeds, default=[42])
    run_seeds = parse_int_csv(args.run_seeds, default=[]) if args.run_seeds else None
    # When --run-seeds is set, scenario generation uses only the first seed
    generation_seeds = [seeds[0]] if run_seeds else seeds
    agents = [agent.lower().strip() for agent in parse_csv(args.agents, default=["codex", "cytobridge", "biomini"])]
    raw_requested_output_dir = (ROOT / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    requested_output_dir = remap_legacy_benchmark_output_root(raw_requested_output_dir)
    output_dir = allocate_fresh_output_dir(requested_output_dir)
    native_run_root = output_dir / "native_runs"
    native_run_root.mkdir(parents=True, exist_ok=True)
    summary_file = None
    if args.summary_file:
        summary_file = (ROOT / args.summary_file).resolve() if not Path(args.summary_file).is_absolute() else Path(args.summary_file)

    scenario_sources: dict[str, str]
    v2_scenario_list = parse_csv(args.v2_scenarios, default=[]) if args.v2_scenarios else None
    v2_mode = bool(v2_scenario_list)
    if v2_mode and args.run_seeds:
        raise SystemExit("V2 benchmarks use fixed deterministic data; use --repeat N for multiple agent runs, not --run-seeds.")
    if v2_scenario_list:
        # V2 mode: deterministic configs, no seed in scenario names
        scenarios, scenario_sources = ensure_v2_scenarios(
            output_dir=output_dir,
            scenario_names=v2_scenario_list or None,
            force_regenerate=args.force_regenerate,
        )
    elif args.scenarios:
        scenarios = parse_csv(args.scenarios, default=[])
        scenario_sources = {}
        missing = [scenario for scenario in scenarios if not scenario_assets_ready(scenario)]
        if missing and args.skip_generate:
            raise SystemExit(
                "Requested --scenarios include missing synthetic assets while --skip-generate is set: "
                + ", ".join(missing)
            )
        if missing:
            raise SystemExit(
                "Requested --scenarios must already exist as DynBench synthetic assets: "
                + ", ".join(missing)
            )
        scenario_sources = {scenario: "existing" for scenario in scenarios}
    elif args.skip_generate:
        if topology_request or topology_spec_file:
            raise SystemExit("--skip-generate is not supported with --topology-request or --topology-spec-file.")
        scenarios = [scenario_name_for(topology, difficulty, seed) for difficulty in difficulties for seed in generation_seeds]
        missing = [scenario for scenario in scenarios if not scenario_assets_ready(scenario)]
        if missing:
            raise SystemExit(
                "--skip-generate was set but required synthetic assets are missing: "
                + ", ".join(missing)
            )
        scenario_sources = {scenario: "existing" for scenario in scenarios}
    else:
        scenarios, scenario_sources = ensure_scenarios(
            topology=topology,
            difficulties=difficulties,
            seeds=generation_seeds,
            output_dir=output_dir,
            topology_request=topology_request or None,
            topology_spec_file=topology_spec_file,
            force_regenerate=args.force_regenerate,
        )

    manifest_path = write_batch_manifest(
        output_dir=output_dir,
        requested_output_dir=requested_output_dir,
        topology=topology,
        topology_request=topology_request or None,
        topology_spec_file=topology_spec_file,
        difficulties=difficulties,
        seeds=seeds,
        agents=agents,
        scenarios=scenarios,
        scenario_sources=scenario_sources,
        run_seeds=run_seeds,
    )
    if requested_output_dir != raw_requested_output_dir:
        print(
            "Legacy benchmark output path under cytobridge_output/figures was remapped to benchmark/results: "
            f"{requested_output_dir}",
            flush=True,
        )
    if output_dir != requested_output_dir:
        print(f"Requested output dir already existed; using fresh batch dir: {output_dir}", flush=True)
    records: list[dict[str, Any]] = []
    # --repeat takes precedence; v2 defaults to an unseeded run label; legacy
    # paths keep numeric seeds for scenario-generation compatibility.
    for scenario in scenarios:
        if args.repeat:
            effective_seeds = [f"run{i}" for i in range(1, args.repeat + 1)]
        elif run_seeds:
            effective_seeds = run_seeds
        elif v2_mode:
            effective_seeds = ["run1"]
        elif args.scenarios:
            effective_seeds = [extract_seed_from_scenario_name(scenario)]
        else:
            effective_seeds = seeds
        for seed in effective_seeds:
            for agent in agents:
                try:
                    record = run_benchmark(
                        RunTarget(scenario=scenario, agent=agent, seed=seed),
                        output_dir=output_dir,
                        timeout_sec=args.timeout_sec,
                        native_run_root=native_run_root,
                    )
                except subprocess.TimeoutExpired as exc:
                    clear_auth_lock()
                    record = {
                        "scenario": scenario,
                        "agent": agent,
                        "seed": seed,
                        "status": "failed",
                        "strict_status": "failed",
                        "runtime_sec": args.timeout_sec,
                        "error": f"timeout after {exc.timeout}s",
                        "native_status": "native_error",
                        "format_status": "error",
                        "recovery_method": "none",
                        "recovery_status": "not_applied",
                        "artifact_origin": "none",
                        "recovery_manifest": None,
                        "native_error": f"timeout after {exc.timeout}s",
                        "scores": {},
                    }
                records.append(record)
                print(
                    f"  {record['status']}: {agent} total={fmt((record.get('scores') or {}).get('total_score'))}",
                    flush=True,
                )

    report_path = write_reports(records, output_dir, summary_file)
    html_reports = write_visual_reports(records, output_dir)
    print(f"\nBatch summary: {report_path}", flush=True)
    print(f"Batch manifest: {manifest_path}", flush=True)
    for html_report in html_reports:
        print(f"HTML report: {html_report}", flush=True)
    if summary_file is not None:
        print(f"Top-level summary: {summary_file}", flush=True)
    any_failed = any(record.get("status") != "passed" for record in records)
    if args.fail_on_agent_error and any_failed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
