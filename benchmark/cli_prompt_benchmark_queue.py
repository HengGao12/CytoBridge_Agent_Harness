#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/os/miniconda3/envs/cytobridge/bin/python")
DEFAULT_RESULTS_ROOT = ROOT / "benchmark" / "results" / "cli_prompt_benchmark"
RESULTS_ROOT = Path(os.environ.get("CYTOBRIDGE_CLI_PROMPT_RESULTS_ROOT", str(DEFAULT_RESULTS_ROOT))).expanduser().resolve()
STATE_PATH = RESULTS_ROOT / "state.json"
EVENT_LOG_PATH = RESULTS_ROOT / "event_log.jsonl"
LOCK_PATH = RESULTS_ROOT / "runner.lock"
LATEST_STATUS_PATH = RESULTS_ROOT / "LATEST_STATUS.md"
QUESTION = (
    "Use the optimal transport algorithm to analyse cell fate predictions in this dataset and train the model."
)
SCENE_ORDER = [7, 3, 6, 5, 1, 2, 4, 8]
SEEDS = [42, 137, 256]
DEFAULT_DEVICE = "cpu"

SCENES: dict[int, dict[str, Any]] = {
    1: {
        "slug": "asymmetric_tree_current_request",
        "title": "Asymmetric tree with multiple confounders",
        "config_template": "benchmark/results/synthetic_batches/asymmetric_tree_8fates_3seeds_current_request/scenario_configs/asymmetric_tree_hard_seed{seed}.json",
        "seed_in_source": True,
    },
    2: {
        "slug": "competitive_4fates_longgap",
        "title": "Competitive multistability with high noise and long time series",
        "config_template": "benchmark/dynbench/simulators/test_configs/competitive_4fates_longgap.json",
        "seed_in_source": False,
    },
    3: {
        "slug": "compact_toggle_stress",
        "title": "Compact high-noise toggle stress test",
        "config_template": "benchmark/dynbench/simulators/test_configs/compact_toggle_high_noise.json",
        "seed_in_source": False,
    },
    4: {
        "slug": "branching_16fates_deep",
        "title": "Deep tree with dense GRN and multiple holdout gaps",
        "config_template": "benchmark/dynbench/simulators/test_configs/branching_tree_16fates_deep.json",
        "seed_in_source": False,
    },
    5: {
        "slug": "feedback_6fates_asym",
        "title": "Feedback with asymmetric branches",
        "config_template": "benchmark/dynbench/simulators/test_configs/feedback_6fates_asymmetric.json",
        "seed_in_source": False,
    },
    6: {
        "slug": "confbomb_4fates",
        "title": "Confounder-heavy anti-heuristic stress test",
        "config_template": "benchmark/dynbench/simulators/test_configs/branching_tree_4fates_confbomb.json",
        "seed_in_source": False,
    },
    7: {
        "slug": "sparse_4fates",
        "title": "Sparse GRN baseline under ideal conditions",
        "config_template": "benchmark/dynbench/simulators/test_configs/branching_tree_4fates_sparse.json",
        "seed_in_source": False,
    },
    8: {
        "slug": "competitive_8fates_extreme",
        "title": "Extreme difficulty stress test",
        "config_template": "benchmark/dynbench/simulators/test_configs/competitive_8fates_extreme.json",
        "seed_in_source": False,
    },
}


@dataclass
class QueueItem:
    queue_id: str
    scene_index: int
    seed: int
    slug: str
    title: str
    source_config: str
    seed_in_source: bool


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def ensure_dirs() -> None:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)


def build_queue() -> list[QueueItem]:
    items: list[QueueItem] = []
    for scene_index in SCENE_ORDER:
        scene = SCENES[scene_index]
        for seed in SEEDS:
            items.append(
                QueueItem(
                    queue_id=f"scene{scene_index:02d}_seed{seed}",
                    scene_index=scene_index,
                    seed=seed,
                    slug=scene["slug"],
                    title=scene["title"],
                    source_config=scene["config_template"].format(seed=seed),
                    seed_in_source=bool(scene["seed_in_source"]),
                )
            )
    return items


def load_state() -> dict[str, Any]:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    queue = build_queue()
    state = {
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "question": QUESTION,
        "device": DEFAULT_DEVICE,
        "scene_order": SCENE_ORDER,
        "seeds": SEEDS,
        "active_pid": None,
        "active_queue_id": None,
        "items": [
            {
                **asdict(item),
                "status": "pending",
                "attempts": 0,
                "started_at": None,
                "finished_at": None,
                "last_error": None,
                "run_dir": None,
                "task_package_dir": None,
                "train_h5ad": None,
                "output_dir": None,
                "report_path": None,
                "scenario_factory_exit": None,
                "cli_exit": None,
                "result_files": [],
            }
            for item in queue
        ],
    }
    save_state(state)
    return state


def save_state(state: dict[str, Any]) -> None:
    state["updated_at"] = now_iso()
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2))
    write_latest_status(state)


def append_event(event: dict[str, Any]) -> None:
    ensure_dirs()
    with EVENT_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": now_iso(), **event}, ensure_ascii=False) + "\n")


def write_latest_status(state: dict[str, Any]) -> None:
    lines = []
    lines.append(f"Updated at: {state.get('updated_at')}")
    lines.append(f"Question: {state.get('question')}")
    lines.append(f"Device: {state.get('device')}")
    lines.append("")
    active = state.get("active_queue_id")
    if active:
        lines.append(f"Active run: {active} (pid={state.get('active_pid')})")
    else:
        lines.append("Active run: none")
    lines.append("")
    lines.append("Queue status:")
    for item in state["items"]:
        lines.append(
            f"- {item['queue_id']} | scene {item['scene_index']} | seed {item['seed']} | {item['status']} | attempts={item['attempts']}"
        )
        if item.get("run_dir"):
            lines.append(f"  run_dir: {item['run_dir']}")
        if item.get("output_dir"):
            lines.append(f"  output_dir: {item['output_dir']}")
        if item.get("report_path"):
            lines.append(f"  report_path: {item['report_path']}")
        if item.get("last_error"):
            lines.append(f"  last_error: {item['last_error']}")
    LATEST_STATUS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


@contextmanager
def file_lock() -> Any:
    ensure_dirs()
    if LOCK_PATH.exists():
        try:
            data = json.loads(LOCK_PATH.read_text())
        except Exception:
            data = {}
        old_pid = data.get("pid")
        if old_pid and pid_alive(old_pid):
            print(f"runner already active pid={old_pid}")
            return
        LOCK_PATH.unlink(missing_ok=True)
    LOCK_PATH.write_text(json.dumps({"pid": os.getpid(), "created_at": now_iso()}))
    try:
        yield True
    finally:
        LOCK_PATH.unlink(missing_ok=True)


def slugify_scene_name(item: dict[str, Any]) -> str:
    return f"{item['queue_id']}__{item['slug']}"


def get_item_run_dir(item: dict[str, Any]) -> Path:
    return RESULTS_ROOT / slugify_scene_name(item)


def prepare_config(item: dict[str, Any], run_dir: Path) -> Path:
    source = ROOT / item["source_config"]
    if not source.exists():
        raise FileNotFoundError(f"missing source config: {source}")
    if item["seed_in_source"]:
        target = run_dir / "inputs" / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        return target

    payload = json.loads(source.read_text())
    payload["seed"] = item["seed"]
    payload.setdefault("name", f"scene{item['scene_index']:02d}_{item['slug']}")
    payload["scenario_name"] = f"scene{item['scene_index']:02d}_{item['slug']}_seed{item['seed']}"
    target = run_dir / "inputs" / f"scene{item['scene_index']:02d}_{item['slug']}_seed{item['seed']}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def write_command(path: Path, cmd: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(" ".join(shlex_quote(x) for x in cmd) + "\n", encoding="utf-8")


def shlex_quote(s: str) -> str:
    import shlex
    return shlex.quote(s)


def run_command(cmd: list[str], log_path: Path, cwd: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n===== START {now_iso()} =====\n")
        log.write("COMMAND: " + " ".join(shlex_quote(x) for x in cmd) + "\n")
        log.flush()
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            env=os.environ.copy(),
        )
        log.write(f"===== END {now_iso()} exit={proc.returncode} =====\n")
        return proc.returncode


def find_train_h5ad(generated_dir: Path) -> tuple[Path, Path]:
    candidates = sorted(generated_dir.glob("task_packages/*/train.h5ad"))
    if not candidates:
        raise FileNotFoundError(f"train.h5ad not found under {generated_dir}")
    train_h5ad = candidates[0]
    return train_h5ad, train_h5ad.parent


def collect_result_files(output_dir: Path) -> list[str]:
    wanted = []
    for pattern in ["report.html", "report.md", "*.json", "*.csv"]:
        for p in sorted(output_dir.glob(pattern)):
            if p.is_file():
                wanted.append(str(p))
    return wanted[:50]


def choose_next_item(state: dict[str, Any]) -> dict[str, Any] | None:
    for item in state["items"]:
        if item["status"] == "running":
            return item
    for item in state["items"]:
        if item["status"] in {"pending", "retry"}:
            return item
    return None


def process_one(state: dict[str, Any]) -> int:
    item = choose_next_item(state)
    if item is None:
        append_event({"event": "queue_complete"})
        print("queue complete")
        return 0

    run_dir = get_item_run_dir(item)
    generated_dir = run_dir / "generated"
    output_dir = run_dir / "cytobridge_output"
    logs_dir = run_dir / "logs"
    run_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    state["active_pid"] = os.getpid()
    state["active_queue_id"] = item["queue_id"]
    item["status"] = "running"
    item["attempts"] += 1
    item["started_at"] = item.get("started_at") or now_iso()
    item["finished_at"] = None
    item["last_error"] = None
    item["run_dir"] = str(run_dir)
    item["output_dir"] = str(output_dir)
    save_state(state)

    append_event({
        "event": "item_started",
        "queue_id": item["queue_id"],
        "scene_index": item["scene_index"],
        "seed": item["seed"],
        "run_dir": str(run_dir),
    })

    try:
        config_path = prepare_config(item, run_dir)
        item["prepared_config"] = str(config_path)
        save_state(state)

        scenario_cmd = [
            str(PYTHON),
            "benchmark/dynbench/simulators/scenario_factory.py",
            "--config",
            str(config_path),
            "--output-dir",
            str(generated_dir),
        ]
        write_command(logs_dir / "scenario_factory.command.txt", scenario_cmd)
        append_event({"event": "scenario_factory_start", "queue_id": item["queue_id"], "command": scenario_cmd})
        scenario_exit = run_command(scenario_cmd, logs_dir / "scenario_factory.log", ROOT)
        item["scenario_factory_exit"] = scenario_exit
        save_state(state)
        if scenario_exit != 0:
            raise RuntimeError(f"scenario_factory exit {scenario_exit}")

        train_h5ad, task_package_dir = find_train_h5ad(generated_dir)
        item["train_h5ad"] = str(train_h5ad)
        item["task_package_dir"] = str(task_package_dir)
        save_state(state)

        cli_cmd = [
            str(PYTHON),
            "-m",
            "cytobridge_agent.cli",
            "run",
            str(train_h5ad),
            "--question",
            QUESTION,
            "--output",
            str(output_dir),
            "--device",
            DEFAULT_DEVICE,
            "--seed",
            str(item["seed"]),
            "--disable-multimodal",
        ]
        write_command(logs_dir / "cytobridge_cli.command.txt", cli_cmd)
        append_event({"event": "cytobridge_cli_start", "queue_id": item["queue_id"], "command": cli_cmd})
        cli_exit = run_command(cli_cmd, logs_dir / "cytobridge_cli.log", ROOT)
        item["cli_exit"] = cli_exit
        item["result_files"] = collect_result_files(output_dir) if output_dir.exists() else []
        report_html = output_dir / "report.html"
        report_md = output_dir / "report.md"
        if report_html.exists():
            item["report_path"] = str(report_html)
        elif report_md.exists():
            item["report_path"] = str(report_md)
        item["finished_at"] = now_iso()
        if cli_exit == 0:
            item["status"] = "completed"
            append_event({
                "event": "item_completed",
                "queue_id": item["queue_id"],
                "output_dir": str(output_dir),
                "report_path": item.get("report_path"),
            })
        else:
            item["status"] = "failed"
            item["last_error"] = f"cytobridge cli exit {cli_exit}"
            append_event({
                "event": "item_failed",
                "queue_id": item["queue_id"],
                "error": item["last_error"],
                "output_dir": str(output_dir),
            })
        save_state(state)
        return 0 if cli_exit == 0 else 1
    except Exception as exc:
        item["status"] = "failed"
        item["finished_at"] = now_iso()
        item["last_error"] = f"{type(exc).__name__}: {exc}"
        append_event({"event": "item_exception", "queue_id": item["queue_id"], "error": item["last_error"]})
        save_state(state)
        return 1
    finally:
        state["active_pid"] = None
        state["active_queue_id"] = None
        save_state(state)


def main() -> int:
    ensure_dirs()
    if not PYTHON.exists():
        print(f"missing python: {PYTHON}", file=sys.stderr)
        return 2
    with file_lock() as acquired:
        if not acquired:
            return 0
        state = load_state()
        old_pid = state.get("active_pid")
        if old_pid and old_pid != os.getpid() and pid_alive(old_pid):
            append_event({"event": "skip_active_pid", "pid": old_pid})
            print(f"active pid still alive: {old_pid}")
            return 0
        return process_one(state)


if __name__ == "__main__":
    raise SystemExit(main())
