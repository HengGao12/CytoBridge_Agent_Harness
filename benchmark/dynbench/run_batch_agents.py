#!/usr/bin/env python
"""Run DynBench scenarios across multiple agents and repeats.

This is an orchestration wrapper around ``benchmark/dynbench/run_dynbench.py``.
It does not inspect hidden ground truth directly; scoring is delegated to the
canonical per-run DynBench runner.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import re
import select
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "benchmark" / "dynbench" / "run_dynbench.py"
PYTHON = Path("/lustre/home/2501111653/miniconda3/envs/agent/bin/python")
if not PYTHON.exists():
    PYTHON = Path(sys.executable)


EXPECTED_OUTPUT_FILES = [
    "velocity_field.csv",
    "growth_rates.csv",
    "holdout_prediction.csv",
    "per_cell_fate.json",
    "perturbation_results.json",
    "driver_genes.json",
]


PROVIDER_FAILURE_PATTERNS = (
    "quota",
    "insufficient_quota",
    "rate limit",
    "rate_limit",
    "provider error",
    "connection error",
    "api key",
    "unauthorized",
    "authentication",
    "reasoning_content",
)

PROVIDER_FAILURE_REGEXES = (
    re.compile(r"http/\d(?:\.\d)?\"?\s+429\b", re.IGNORECASE),
    re.compile(r"\bstatus(?:_code)?\s*[=:]\s*429\b", re.IGNORECASE),
    re.compile(r"\b429\b[^\n]{0,120}(?:too many requests|rate limit|quota)", re.IGNORECASE),
    re.compile(r"(?:too many requests|rate limit|quota)[^\n]{0,120}\b429\b", re.IGNORECASE),
)


_ACTIVE_PROCESS_GROUPS: set[int] = set()
_ACTIVE_PROCESS_LOCK = threading.Lock()


@dataclass(frozen=True)
class Job:
    scenario: str
    agent: str
    repeat: int
    seed: int


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def discover_scenarios(task_packages: Path) -> list[str]:
    scenarios = []
    for path in sorted(task_packages.iterdir()):
        if path.is_dir() and (path / "train.h5ad").exists() and (path / "TASK.md").exists():
            scenarios.append(path.name)
    return scenarios


def load_scores(eval_path: Path) -> dict[str, Any]:
    payload = json.loads(eval_path.read_text(encoding="utf-8"))
    results = payload.get("results") or {}
    per_metric = results.get("per_metric") or {}
    row: dict[str, Any] = {
        "total_score": results.get("total_score"),
        "runtime_sec": payload.get("runtime_sec"),
    }
    for name, metric in per_metric.items():
        if isinstance(metric, dict):
            row[name] = metric.get("score")
    return row


def write_csv(summary_jsonl: Path, summary_csv: Path) -> None:
    rows = []
    if summary_jsonl.exists():
        for line in summary_jsonl.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        return
    fields = [
        "status",
        "scenario",
        "agent",
        "repeat",
        "seed",
        "provider_used",
        "model_used",
        "thinking_used",
        "total_score",
        "M1_velocity",
        "M2_growth",
        "M3_distribution",
        "M4_fate",
        "M5_perturbation",
        "M6_grn",
        "runtime_sec",
        "output_dir",
        "error",
    ]
    extras = sorted({k for row in rows for k in row if k not in fields})
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields + extras)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def output_dir_for(root: Path, scenario: str, agent: str, run_label: str) -> Path:
    return root / "synthetic" / scenario / f"agent_{agent}_skills-on_{run_label}"


def batch_log_for(root: Path, scenario: str, agent: str, run_label: str) -> Path:
    # Keep batch-runner logs outside per-run output_dir because run_dynbench.py
    # resets output_dir at process start.
    return root / "batch_runner_logs" / f"{scenario}__{agent}__{run_label}.log"


def task_dir_for(scenario: str, task_packages: Path | None = None) -> Path:
    root = task_packages or ROOT / "benchmark" / "dynbench" / "task_packages"
    return root / scenario


def ground_truth_dir_for(scenario: str, ground_truth: Path | None = None) -> Path:
    root = ground_truth or ROOT / "benchmark" / "dynbench" / "ground_truth"
    return root / scenario


def expected_outputs_present(output_dir: Path) -> bool:
    return all((output_dir / name).exists() and (output_dir / name).stat().st_size > 0 for name in EXPECTED_OUTPUT_FILES)


def public_verifier_passes(output_dir: Path, scenario: str, *, task_packages: Path | None = None) -> bool:
    if not expected_outputs_present(output_dir):
        return False
    train_h5ad = task_dir_for(scenario, task_packages) / "train.h5ad"
    if not train_h5ad.exists():
        return False
    try:
        from benchmark.dynbench.verify_dynbench_outputs import verify_outputs

        report = verify_outputs(output_dir, train_h5ad)
        return bool(report.get("ok"))
    except Exception:
        return False


def ensure_eval_results(
    output_dir: Path,
    scenario: str,
    *,
    task_packages: Path | None = None,
    ground_truth: Path | None = None,
    runtime_sec: float | None = None,
) -> bool:
    eval_path = output_dir / "eval_results.json"
    if eval_path.exists():
        return True
    if not public_verifier_passes(output_dir, scenario, task_packages=task_packages):
        return False
    gt_dir = ground_truth_dir_for(scenario, ground_truth)
    if not gt_dir.exists():
        return False
    try:
        from benchmark.dynbench.eval.dynbench_eval_v2 import evaluate_all

        results = evaluate_all(str(output_dir), str(gt_dir))
        eval_path.write_text(
            json.dumps(
                {
                    "scenario": scenario,
                    "runtime_sec": runtime_sec,
                    "results": results,
                    "batch_scored_after_agent_completion": True,
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        return True
    except Exception as exc:
        (output_dir / "batch_eval_error.json").write_text(
            json.dumps({"error": repr(exc)}, indent=2),
            encoding="utf-8",
        )
        return False


def _redact_command(cmd: list[str]) -> list[str]:
    redacted = list(cmd)
    secret_flags = {"--llm-api-key"}
    for idx, token in enumerate(redacted):
        if token in secret_flags and idx + 1 < len(redacted):
            redacted[idx + 1] = "<redacted>"
    return redacted


def run_command(
    cmd: list[str],
    log_path: Path,
    *,
    env_overrides: dict[str, str] | None = None,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(_redact_command(cmd)) + "\n\n")
        log.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        with _ACTIVE_PROCESS_LOCK:
            _ACTIVE_PROCESS_GROUPS.add(proc.pid)
        chunks: list[str] = []
        timed_out = False
        try:
            deadline = time.time() + float(timeout) if timeout is not None else None
            while True:
                if proc.stdout is not None:
                    ready, _, _ = select.select([proc.stdout], [], [], 0.0)
                    line = proc.stdout.readline() if ready else ""
                    if line:
                        chunks.append(line)
                        log.write(line)
                        log.flush()
                        continue
                if proc.poll() is not None:
                    if proc.stdout is not None:
                        rest = proc.stdout.read()
                        if rest:
                            chunks.append(rest)
                            log.write(rest)
                            log.flush()
                    break
                if deadline is not None and time.time() >= deadline:
                    raise subprocess.TimeoutExpired(cmd, timeout)
                time.sleep(1.0)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                stdout, _ = proc.communicate(timeout=15)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                stdout, _ = proc.communicate()
            if stdout:
                chunks.append(stdout)
                log.write(stdout)
        finally:
            with _ACTIVE_PROCESS_LOCK:
                _ACTIVE_PROCESS_GROUPS.discard(proc.pid)
        if timed_out:
            log.write(f"\n\n[batch] subprocess timed out after {timeout}s and was terminated\n")
        log.write(f"\n\n[batch] returncode={proc.returncode} elapsed={time.time() - started:.1f}s\n")
    returncode = proc.returncode if proc.returncode is not None else 124
    result = subprocess.CompletedProcess(cmd, returncode, "".join(chunks))
    setattr(result, "elapsed_sec", time.time() - started)
    return result


def terminate_active_process_groups(sig: int = signal.SIGTERM) -> None:
    with _ACTIVE_PROCESS_LOCK:
        pids = list(_ACTIVE_PROCESS_GROUPS)
    for pid in pids:
        try:
            os.killpg(pid, sig)
        except ProcessLookupError:
            pass


def install_shutdown_handlers() -> None:
    def _handler(signum: int, _frame: object) -> None:
        terminate_active_process_groups(signal.SIGTERM)
        time.sleep(2.0)
        terminate_active_process_groups(signal.SIGKILL)
        raise SystemExit(128 + signum)

    for sig in (signal.SIGINT, signal.SIGTERM, getattr(signal, "SIGHUP", None)):
        if sig is not None:
            signal.signal(sig, _handler)


def looks_like_provider_failure(text: str) -> bool:
    lower = text.lower()
    if any(pattern in lower for pattern in PROVIDER_FAILURE_PATTERNS):
        return True
    return any(pattern.search(text) for pattern in PROVIDER_FAILURE_REGEXES)


def provider_failure_evidence(text: str, *, max_chars: int = 2000) -> str:
    """Return a compact provider-failure excerpt suitable for summary rows."""
    lines = [line for line in text.splitlines() if looks_like_provider_failure(line)]
    if not lines:
        return ""
    excerpt = "\n".join(lines[-12:])
    return excerpt[-max_chars:]


def fallback_provider_evidence_from_logs(row: dict[str, Any]) -> str:
    """Recover provider-failure evidence for legacy fallback rows.

    Older runner versions only stored the final output-validation error in the
    summary row. The original Xiaomi run logs still live next to the fallback
    output directory, so clean resume logic can recover whether fallback was
    actually justified without accepting false-positive fallback rows.
    """
    try:
        output_dir = Path(str(row.get("output_dir") or ""))
    except TypeError:
        return ""
    if not output_dir.exists():
        return ""
    scenario_dir = output_dir.parent
    repeat = row.get("repeat")
    if not isinstance(repeat, int):
        return ""
    pattern = f"agent_cytobridge_skills-on_cytobridge_xiaomi*_r{repeat:02d}"
    snippets: list[str] = []
    for run_dir in sorted(scenario_dir.glob(pattern)):
        for log_name in ("agent_log.txt", "runner_stdout.log"):
            log_path = run_dir / log_name
            if not log_path.exists():
                continue
            text = log_path.read_text(encoding="utf-8", errors="replace")[-12000:]
            evidence = provider_failure_evidence(text)
            if evidence:
                snippets.append(f"{run_dir.name}/{log_name}:\n{evidence}")
    return "\n".join(snippets)[-2000:]


def fallback_row_has_provider_failure(row: dict[str, Any]) -> bool:
    if row.get("provider_used") != "openai-codex-fallback":
        return True
    reason = "\n".join(
        str(row.get(key) or "")
        for key in ("xiaomi_provider_failure_before_fallback", "xiaomi_error_before_fallback")
    )
    if reason and looks_like_provider_failure(reason):
        return True
    recovered = fallback_provider_evidence_from_logs(row)
    return bool(recovered and looks_like_provider_failure(recovered))


def xiaomi_attempts(args: argparse.Namespace) -> list[dict[str, str | None]]:
    keys = [k.strip() for k in (args.xiaomi_api_keys or os.environ.get("CYTOBRIDGE_XIAOMI_API_KEYS", "")).split(",") if k.strip()]
    base_urls = [
        u.strip()
        for u in (args.xiaomi_base_urls or os.environ.get("CYTOBRIDGE_XIAOMI_BASE_URLS", "")).split(",")
        if u.strip()
    ]
    if not base_urls:
        base_urls = ["https://token-plan-cn.xiaomimimo.com/v1"]
    if not keys:
        return [{"api_key": None, "base_url": None}]
    return [{"api_key": key, "base_url": base_url} for key in keys for base_url in base_urls]


def build_base_cmd(job: Job, args: argparse.Namespace, *, provider_label: str, run_label: str) -> tuple[list[str], Path]:
    output_root = args.run_root
    cmd = [
        str(PYTHON),
        str(RUNNER),
        "--scenario",
        job.scenario,
        "--agent-type",
        job.agent,
        "--mode",
        "skills-on",
        "--device",
        args.device,
        "--seed",
        str(job.seed),
        "--run-label",
        run_label,
        "--run-root",
        str(output_root),
        "--timeout",
        str(args.timeout),
    ]
    out_dir = output_dir_for(output_root, job.scenario, job.agent, run_label)
    return cmd, out_dir


def run_codex(job: Job, args: argparse.Namespace) -> dict[str, Any]:
    run_label = f"codex_r{job.repeat:02d}"
    cmd, out_dir = build_base_cmd(job, args, provider_label="codex", run_label=run_label)
    cmd.extend(["--llm-model", "gpt-5.5"])
    log_path = batch_log_for(args.run_root, job.scenario, job.agent, run_label)
    proc = run_command(
        cmd,
        log_path,
        timeout=float(args.timeout) + 120,
    )
    return finalize_run(job, args, out_dir, proc, provider="codex", model="gpt-5.5", thinking="medium")


def run_cytobridge(job: Job, args: argparse.Namespace) -> dict[str, Any]:
    last_error = ""
    last_provider_failure = ""
    last_out_dir = args.run_root / "synthetic" / job.scenario / "agent_cytobridge_skills-on_cytobridge_xiaomi_failed"
    for idx, attempt in enumerate(xiaomi_attempts(args), 1):
        run_label = f"cytobridge_xiaomi{idx}_r{job.repeat:02d}"
        cmd, out_dir = build_base_cmd(job, args, provider_label="xiaomi", run_label=run_label)
        last_out_dir = out_dir
        cmd.extend([
            "--llm-provider",
            "xiaomi",
            "--llm-model",
            "mimo-v2.5-pro",
            "--llm-thinking-level",
            "high",
        ])
        env_overrides: dict[str, str] = {}
        if attempt["api_key"]:
            cmd.extend(["--llm-auth-mode", "api_key"])
            env_overrides["CYTOBRIDGE_OPENAI_API_KEY"] = str(attempt["api_key"])
        if attempt["base_url"]:
            cmd.extend(["--llm-base-url", str(attempt["base_url"])])
        log_path = batch_log_for(args.run_root, job.scenario, job.agent, run_label)
        proc = run_command(
            cmd,
            log_path,
            env_overrides=env_overrides or None,
            timeout=float(args.timeout) + 120,
        )
        result = finalize_run(
            job,
            args,
            out_dir,
            proc,
            provider="xiaomi",
            model="mimo-v2.5-pro",
            thinking="high",
            return_failures=True,
        )
        if result["status"] == "ok":
            return result
        last_error = str(result.get("error") or "")
        log_text = ""
        if log_path.exists():
            log_text = log_path.read_text(encoding="utf-8", errors="replace")[-8000:]
        failure_text = last_error + "\n" + log_text
        last_provider_failure = provider_failure_evidence(failure_text)
        if not looks_like_provider_failure(failure_text):
            return result

    if args.no_codex_fallback:
        return {
            "status": "failed",
            "scenario": job.scenario,
            "agent": job.agent,
            "repeat": job.repeat,
            "seed": job.seed,
            "provider_used": "xiaomi",
            "model_used": "mimo-v2.5-pro",
            "thinking_used": "high",
            "output_dir": str(last_out_dir),
            "error": last_error or "All Xiaomi attempts failed; Codex fallback disabled.",
            "xiaomi_provider_failure_before_fallback": last_provider_failure,
        }

    run_label = f"cytobridge_codex_fallback_r{job.repeat:02d}"
    cmd, out_dir = build_base_cmd(job, args, provider_label="openai-codex", run_label=run_label)
    cmd.extend([
        "--llm-provider",
        "openai-codex",
        "--llm-model",
        "gpt-5.5",
        "--llm-thinking-level",
        "medium",
        "--llm-auth-mode",
        "codex_oauth",
    ])
    log_path = batch_log_for(args.run_root, job.scenario, job.agent, run_label)
    proc = run_command(
        cmd,
        log_path,
        timeout=float(args.timeout) + 120,
    )
    result = finalize_run(
        job,
        args,
        out_dir,
        proc,
        provider="openai-codex-fallback",
        model="gpt-5.5",
        thinking="medium",
        return_failures=True,
    )
    if last_error:
        result["xiaomi_error_before_fallback"] = last_error
    if last_provider_failure:
        result["xiaomi_provider_failure_before_fallback"] = last_provider_failure
    return result


def finalize_run(
    job: Job,
    args: argparse.Namespace,
    out_dir: Path,
    proc: subprocess.CompletedProcess[str],
    *,
    provider: str,
    model: str,
    thinking: str,
    return_failures: bool = False,
) -> dict[str, Any]:
    eval_path = out_dir / "eval_results.json"
    row: dict[str, Any] = {
        "scenario": job.scenario,
        "agent": job.agent,
        "repeat": job.repeat,
        "seed": job.seed,
        "provider_used": provider,
        "model_used": model,
        "thinking_used": thinking,
        "output_dir": str(out_dir),
    }
    runtime_sec = getattr(proc, "elapsed_sec", None)
    # Score valid artifacts even if the wrapper observed a non-zero return code
    # after the files were materialized, e.g. timeout/SIGTERM racing with final
    # validation. Missing or malformed outputs still fail below.
    if ensure_eval_results(
        out_dir,
        job.scenario,
        task_packages=args.task_packages,
        ground_truth=args.ground_truth,
        runtime_sec=runtime_sec,
    ):
        row.update(load_scores(eval_path))
        row["status"] = "ok"
        return row
    error = f"returncode={proc.returncode}; eval_results_exists={eval_path.exists()}"
    error_path = out_dir / "error.json"
    if error_path.exists():
        try:
            error += "; " + json.dumps(json.loads(error_path.read_text(encoding="utf-8")), ensure_ascii=False)[:1200]
        except Exception:
            error += "; " + error_path.read_text(encoding="utf-8", errors="replace")[:1200]
    row["status"] = "failed"
    row["error"] = error
    if return_failures:
        return row
    return row


def run_job(job: Job, args: argparse.Namespace) -> dict[str, Any]:
    # Resume support: if a previous batch attempt already produced valid
    # scored outputs for this exact job label, reuse them instead of resetting
    # the output directory and rerunning the agent.
    if job.agent == "codex":
        run_label = f"codex_r{job.repeat:02d}"
        out_dir = output_dir_for(args.run_root, job.scenario, job.agent, run_label)
        if ensure_eval_results(out_dir, job.scenario, task_packages=args.task_packages, ground_truth=args.ground_truth):
            row = {
                "scenario": job.scenario,
                "agent": job.agent,
                "repeat": job.repeat,
                "seed": job.seed,
                "provider_used": "codex",
                "model_used": "gpt-5.5",
                "thinking_used": "medium",
                "output_dir": str(out_dir),
                "status": "ok",
                "resumed_from_existing_outputs": True,
            }
            row.update(load_scores(out_dir / "eval_results.json"))
            return row
    if job.agent == "cytobridge":
        for idx, _attempt in enumerate(xiaomi_attempts(args), 1):
            run_label = f"cytobridge_xiaomi{idx}_r{job.repeat:02d}"
            out_dir = output_dir_for(args.run_root, job.scenario, job.agent, run_label)
            if ensure_eval_results(out_dir, job.scenario, task_packages=args.task_packages, ground_truth=args.ground_truth):
                row = {
                    "scenario": job.scenario,
                    "agent": job.agent,
                    "repeat": job.repeat,
                    "seed": job.seed,
                    "provider_used": "xiaomi",
                    "model_used": "mimo-v2.5-pro",
                    "thinking_used": "high",
                    "output_dir": str(out_dir),
                    "status": "ok",
                    "resumed_from_existing_outputs": True,
                }
                row.update(load_scores(out_dir / "eval_results.json"))
                return row
        fallback_label = f"cytobridge_codex_fallback_r{job.repeat:02d}"
        fallback_dir = output_dir_for(args.run_root, job.scenario, job.agent, fallback_label)
        if not args.no_codex_fallback and ensure_eval_results(
            fallback_dir,
            job.scenario,
            task_packages=args.task_packages,
            ground_truth=args.ground_truth,
        ):
            existing_fallback_rows = [
                row
                for row in load_existing_summary(args.run_root / "batch_summary.jsonl")
                if row.get("scenario") == job.scenario
                and row.get("agent") == job.agent
                and row.get("repeat") == job.repeat
                and row.get("provider_used") == "openai-codex-fallback"
                and row.get("status") == "ok"
            ]
            if not any(fallback_row_has_provider_failure(row) for row in existing_fallback_rows):
                return run_cytobridge(job, args)
            row = {
                "scenario": job.scenario,
                "agent": job.agent,
                "repeat": job.repeat,
                "seed": job.seed,
                "provider_used": "openai-codex-fallback",
                "model_used": "gpt-5.5",
                "thinking_used": "medium",
                "output_dir": str(fallback_dir),
                "status": "ok",
                "resumed_from_existing_outputs": True,
            }
            row.update(load_scores(fallback_dir / "eval_results.json"))
            return row
    if job.agent == "codex":
        return run_codex(job, args)
    if job.agent == "cytobridge":
        return run_cytobridge(job, args)
    raise ValueError(f"Unsupported agent: {job.agent}")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def load_existing_summary(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def completed_job_keys(rows: list[dict[str, Any]], *, skip_failed: bool = False) -> set[tuple[str, str, int]]:
    keys: set[tuple[str, str, int]] = set()
    for row in rows:
        status = row.get("status")
        if status == "failed" and skip_failed:
            pass
        elif status == "ok":
            if not fallback_row_has_provider_failure(row):
                continue
        else:
            continue
        scenario = row.get("scenario")
        agent = row.get("agent")
        repeat = row.get("repeat")
        if isinstance(scenario, str) and isinstance(agent, str) and isinstance(repeat, int):
            keys.add((scenario, agent, repeat))
    return keys


def main() -> None:
    install_shutdown_handlers()

    parser = argparse.ArgumentParser(description="Batch-run DynBench across agents/repeats.")
    parser.add_argument("--scenarios", default=None, help="Comma-separated scenario names. Default: discover task_packages.")
    parser.add_argument("--agents", default="cytobridge,codex", help="Comma-separated agents.")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--parallel", type=int, default=2)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("--run-root", type=Path, default=ROOT / "cytobridge_output" / f"dynbench_batch_{_timestamp()}")
    parser.add_argument("--task-packages", type=Path, default=ROOT / "benchmark" / "dynbench" / "task_packages")
    parser.add_argument("--ground-truth", type=Path, default=ROOT / "benchmark" / "dynbench" / "ground_truth")
    parser.add_argument(
        "--assets-root",
        type=Path,
        default=None,
        help="Optional dynbench_assets root containing task_packages/ and ground_truth/. Linked into run-root for run_dynbench.py.",
    )
    parser.add_argument("--base-seed", type=int, default=24000)
    parser.add_argument("--xiaomi-api-keys", default=None, help="Comma-separated Xiaomi keys. Prefer env CYTOBRIDGE_XIAOMI_API_KEYS.")
    parser.add_argument("--xiaomi-base-urls", default=None, help="Comma-separated Xiaomi base URLs. Prefer env CYTOBRIDGE_XIAOMI_BASE_URLS.")
    parser.add_argument("--no-codex-fallback", action="store_true", help="Fail instead of falling back to Codex after Xiaomi provider failures.")
    parser.add_argument(
        "--skip-existing-failed",
        action="store_true",
        help="Resume mode: treat existing failed summary rows as already attempted, so only missing jobs are run.",
    )
    args = parser.parse_args()

    if args.assets_root:
        args.assets_root = args.assets_root.resolve()
        args.task_packages = args.assets_root / "task_packages"
        args.ground_truth = args.assets_root / "ground_truth"
    args.task_packages = args.task_packages.resolve()
    args.ground_truth = args.ground_truth.resolve()
    if not args.task_packages.exists():
        raise SystemExit(f"Task packages not found: {args.task_packages}")
    if not args.ground_truth.exists():
        raise SystemExit(f"Ground truth not found: {args.ground_truth}")

    task_packages = args.task_packages
    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()] if args.scenarios else discover_scenarios(task_packages)
    agents = [a.strip() for a in args.agents.split(",") if a.strip()]
    if not scenarios:
        raise SystemExit(f"No scenarios found under {task_packages}")

    args.run_root.mkdir(parents=True, exist_ok=True)
    if args.assets_root:
        link_path = args.run_root / "dynbench_assets"
        if not link_path.exists():
            try:
                link_path.symlink_to(args.assets_root, target_is_directory=True)
            except OSError:
                # Fall back to a tiny marker file; run_dynbench still receives
                # the explicit scenario list, while batch scoring uses the
                # explicit task/ground-truth paths above.
                (args.run_root / "dynbench_assets_path.txt").write_text(str(args.assets_root), encoding="utf-8")
    summary_jsonl = args.run_root / "batch_summary.jsonl"
    summary_csv = args.run_root / "batch_summary.csv"
    manifest = {
        "started_at": _timestamp(),
        "scenarios": scenarios,
        "agents": agents,
        "repeats": args.repeats,
        "parallel": args.parallel,
        "device": args.device,
        "run_root": str(args.run_root),
        "task_packages": str(args.task_packages),
        "ground_truth": str(args.ground_truth),
        "assets_root": str(args.assets_root) if args.assets_root else None,
        "no_codex_fallback": bool(args.no_codex_fallback),
        "skip_existing_failed": bool(args.skip_existing_failed),
    }
    (args.run_root / "batch_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    existing_rows = load_existing_summary(summary_jsonl)
    completed_keys = completed_job_keys(existing_rows, skip_failed=args.skip_existing_failed)

    jobs: list[Job] = []
    for sidx, scenario in enumerate(scenarios):
        for agent in agents:
            for repeat in range(1, args.repeats + 1):
                if (scenario, agent, repeat) in completed_keys:
                    continue
                jobs.append(Job(scenario=scenario, agent=agent, repeat=repeat, seed=args.base_seed + sidx * 100 + repeat))

    skipped = len(completed_keys)
    total_jobs = len(scenarios) * len(agents) * args.repeats
    print(f"Running {len(jobs)} remaining jobs out of {total_jobs}: {len(scenarios)} scenarios × {len(agents)} agents × {args.repeats} repeats")
    if skipped:
        print(f"Resume: skipping {skipped} previously completed ok jobs; failed or incomplete jobs remain eligible.")
    print(f"Run root: {args.run_root}")
    print(f"Parallel: {args.parallel}")

    completed = max([int(row.get("completed_index", 0) or 0) for row in existing_rows], default=0)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = {pool.submit(run_job, job, args): job for job in jobs}
        for future in concurrent.futures.as_completed(futures):
            job = futures[future]
            try:
                row = future.result()
            except Exception as exc:
                row = {
                    "status": "failed",
                    "scenario": job.scenario,
                    "agent": job.agent,
                    "repeat": job.repeat,
                    "seed": job.seed,
                    "error": repr(exc),
                }
            completed += 1
            row["completed_index"] = completed
            row["completed_total"] = total_jobs
            append_jsonl(summary_jsonl, row)
            write_csv(summary_jsonl, summary_csv)
            score = row.get("total_score")
            score_text = f"{score:.3f}" if isinstance(score, (int, float)) else "NA"
            print(
                f"[{completed}/{total_jobs}] {row.get('status')} "
                f"{job.scenario} {job.agent} r{job.repeat} "
                f"provider={row.get('provider_used')} total={score_text}",
                flush=True,
            )

    print(f"Summary JSONL: {summary_jsonl}")
    print(f"Summary CSV:   {summary_csv}")


if __name__ == "__main__":
    main()
