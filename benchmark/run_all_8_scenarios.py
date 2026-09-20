#!/usr/bin/env python3
"""Sequentially run all 8 benchmark scenarios with 3 agents each.

Between scenarios, pauses to allow auth switching if configured.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYTHON = Path("/home/os/miniconda3/envs/cytobridge/bin/python")
BATCH_RUNNER = ROOT / "benchmark" / "batch_benchmark_runner.py"
TEST_CONFIGS = ROOT / "benchmark" / "dynbench" / "simulators" / "test_configs"
RESULTS_ROOT = ROOT / "benchmark" / "results" / "three_agent_full_rerun_20260502_155815"
STATE_PATH = RESULTS_ROOT / "state.json"

CST = timezone(timedelta(hours=8))
SEEDS = "42"
RUN_SEEDS = "42,137,256"
AGENTS = "codex,cytobridge,biomini"
DIFFICULTIES = "medium"

# Suggested order from test_prompts_batch.md
SCENARIOS = [
    {
        "index": 0,
        "config": "branching_tree_4fates_sparse.json",
        "batch_name": "batch_sparse_4fates_3seeds",
        "description": "Scenario 7: sparse GRN baseline",
    },
    {
        "index": 1,
        "config": "compact_toggle_high_noise.json",
        "batch_name": "batch_compact_toggle_stress_3seeds",
        "description": "Scenario 3: compact high-noise toggle stress test",
    },
    {
        "index": 2,
        "config": "branching_tree_4fates_confbomb.json",
        "batch_name": "batch_confbomb_4fates_3seeds",
        "description": "Scenario 6: confounder-heavy stress test",
    },
    {
        "index": 3,
        "config": "feedback_6fates_asymmetric.json",
        "batch_name": "batch_feedback_6fates_asym_3seeds",
        "description": "Scenario 5: feedback plus asymmetry",
    },
    {
        "index": 4,
        "config": "asymmetric_tree_8fates_dense.json",
        "batch_name": "batch_asymmetric_8fates_3seeds",
        "description": "Scenario 1: asymmetric tree",
    },
    {
        "index": 5,
        "config": "competitive_4fates_longgap.json",
        "batch_name": "batch_competitive_4fates_longgap_3seeds",
        "description": "Scenario 2: competitive multistability with long gap",
    },
    {
        "index": 6,
        "config": "branching_tree_16fates_deep.json",
        "batch_name": "batch_branching_16fates_deep_3seeds",
        "description": "Scenario 4: deep 16-fate tree",
    },
    {
        "index": 7,
        "config": "competitive_8fates_extreme.json",
        "batch_name": "batch_competitive_8fates_extreme_3seeds",
        "description": "Scenario 8: extreme difficulty",
    },
]


def now_str():
    return datetime.now(tz=CST).isoformat(timespec="seconds")


def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    state = {
        "created_at": now_str(),
        "results_root": str(RESULTS_ROOT),
        "current_index": -1,
        "status": "idle",
        "started_at": None,
        "completed_at": None,
        "last_check": now_str(),
        "history": [],
        "scenarios": [
            {
                "index": s["index"],
                "config": s["config"],
                "batch_name": s["batch_name"],
                "description": s["description"],
                "status": "pending",
                "started_at": None,
                "finished_at": None,
                "exit_code": None,
                "output_dir": None,
            }
            for s in SCENARIOS
        ],
    }
    save_state(state)
    return state


def save_state(state: dict):
    state["last_check"] = now_str()
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2))
    write_status_md(state)


def write_status_md(state: dict):
    lines = [
        f"Updated at: {state['last_check']}",
        f"Results root: {state['results_root']}",
        f"Status: {state['status']}",
        f"agents: {AGENTS}",
        f"seeds: {SEEDS}",
        "",
    ]
    for s in state["scenarios"]:
        marker = "✅" if s["status"] == "completed" else ("🔄" if s["status"] == "running" else "⏳")
        lines.append(f"  {marker} [{s['index']}] {s['description']} | {s['status']}")
        if s.get("output_dir"):
            lines.append(f"      output: {s['output_dir']}")
        if s.get("exit_code") is not None:
            lines.append(f"      exit_code: {s['exit_code']}")
    (RESULTS_ROOT / "LATEST_STATUS.md").write_text("\n".join(lines) + "\n")


def run_scenario(scenario: dict) -> int:
    config_path = TEST_CONFIGS / scenario["config"]
    output_dir = RESULTS_ROOT / scenario["batch_name"]

    cmd = [
        str(PYTHON),
        str(BATCH_RUNNER),
        "--topology-spec-file",
        str(config_path),
        "--difficulties",
        DIFFICULTIES,
        "--seeds",
        SEEDS,
        "--run-seeds",
        RUN_SEEDS,
        "--agents",
        AGENTS,
        "--output-dir",
        str(output_dir),
    ]

    log_path = RESULTS_ROOT / f"runner_{scenario['batch_name']}.log"
    print(f"\n{'='*60}")
    print(f"[{now_str()}] Starting: {scenario['description']}")
    print(f"  config: {scenario['config']}")
    print(f"  output: {output_dir}")
    print(f"  log: {log_path}")
    print(f"{'='*60}\n")

    with log_path.open("a") as log:
        log.write(f"\n===== START {now_str()} =====\n")
        log.write(f"COMMAND: {' '.join(cmd)}\n")
        log.flush()
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            env=os.environ.copy(),
        )
        log.write(f"===== END {now_str()} exit={proc.returncode} =====\n")
        return proc.returncode


def switch_auth():
    """Run codex-login to switch to personal Plus account."""
    print(f"\n{'='*60}")
    print(f"[{now_str()}] Switching auth: running codex-login...")
    print(f"{'='*60}\n")

    login_cmd = [
        str(PYTHON),
        "-m", "cytobridge_agent.cli",
        "auth", "codex-login",
    ]

    log_path = RESULTS_ROOT / "auth_switch.log"
    with log_path.open("a") as log:
        log.write(f"\n===== AUTH SWITCH START {now_str()} =====\n")
        log.flush()
        proc = subprocess.run(
            login_cmd,
            cwd=str(ROOT),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            env=os.environ.copy(),
            timeout=300,
        )
        log.write(f"===== AUTH SWITCH END {now_str()} exit={proc.returncode} =====\n")

    # Verify which profile is now active
    verify_cmd = [
        str(PYTHON),
        "-m", "cytobridge_agent.cli",
        "auth", "profiles", "status",
    ]
    result = subprocess.run(
        verify_cmd, cwd=str(ROOT), capture_output=True, text=True, env=os.environ.copy()
    )
    try:
        status = json.loads(result.stdout)
        last_good = status.get("last_good_profile_id", "unknown")
        print(f"[{now_str()}] Auth switch done. last_good_profile_id: {last_good}")
    except Exception:
        print(f"[{now_str()}] Auth switch completed (exit={proc.returncode}), could not parse status")

    return proc.returncode


def main() -> int:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    state = load_state()

    # Find next scenario to run
    next_idx = None
    for i, s in enumerate(state["scenarios"]):
        if s["status"] in ("pending", "retry"):
            next_idx = i
            break

    if next_idx is None:
        state["status"] = "completed"
        state["completed_at"] = now_str()
        save_state(state)
        print("All scenarios completed!")
        return 0

    scenario = state["scenarios"][next_idx]
    scenario["status"] = "running"
    scenario["started_at"] = now_str()
    state["current_index"] = next_idx
    state["status"] = "running"
    state["started_at"] = state.get("started_at") or now_str()
    save_state(state)

    try:
        exit_code = run_scenario(scenario)
        scenario["exit_code"] = exit_code
        scenario["output_dir"] = str(RESULTS_ROOT / scenario["batch_name"])
        scenario["finished_at"] = now_str()
        scenario["status"] = "completed" if exit_code == 0 else "failed"
        state["history"].append({
            "index": scenario["index"],
            "batch_name": scenario["batch_name"],
            "exit_code": exit_code,
            "finished_at": scenario["finished_at"],
        })
    except Exception as exc:
        scenario["status"] = "failed"
        scenario["exit_code"] = -1
        scenario["finished_at"] = now_str()
        scenario["output_dir"] = str(RESULTS_ROOT / scenario["batch_name"])
        state["history"].append({
            "index": scenario["index"],
            "batch_name": scenario["batch_name"],
            "exit_code": -1,
            "error": str(exc),
            "finished_at": scenario["finished_at"],
        })

    save_state(state)

    # Print summary
    completed = sum(1 for s in state["scenarios"] if s["status"] == "completed")
    total = len(state["scenarios"])
    print(f"\n[{now_str()}] Progress: {completed}/{total} scenarios completed")

    # Auth switch checkpoint between scenarios
    if completed < total:
        print(f"\n[{now_str()}] Running auth switch checkpoint before next scenario...")
        switch_auth()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
