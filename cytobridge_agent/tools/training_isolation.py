"""Subprocess isolation for high-risk custom training execution.

The main planner process must not be the process that allocates untrusted custom
algorithm tensors on large biological datasets.  This module runs the actual
training/evaluation call in a child process and returns only JSON-safe metrics
plus artifact paths to the parent.
"""
from __future__ import annotations

import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

try:
    import resource
except Exception:  # pragma: no cover - non-Unix fallback
    resource = None  # type: ignore[assignment]

from .training_run_manager import TrainingRunBundle


DEFAULT_TRAINING_SUBPROCESS_TIMEOUT_SEC = 60 * 60
DEFAULT_PREVIEW_SUBPROCESS_TIMEOUT_SEC = 5 * 60
DEFAULT_PREVIEW_SUBPROCESS_MAX_TIMEOUT_SEC = 15 * 60
TRAINING_TIME_BUDGET_CELL_BLOCK = 5000
TRAINING_TIME_BUDGET_SECONDS_PER_CELL_BLOCK = 90
TRAINING_TIME_BUDGET_BASE_SECONDS = 60
TRAINING_TIME_BUDGET_GAP_OVERHEAD_SECONDS = 64
TRAINING_TIME_BUDGET_SAFETY_MULTIPLIER = 1.33
TRAINING_TIME_BUDGET_ROUNDING_SECONDS = 30
TRAINING_TIME_BUDGET_MIN_SECONDS = 120
TRAINING_SUBPROCESS_TIMEOUT_EXTRA_SEC = 420
TRAINING_SUBPROCESS_TIMEOUT_MULTIPLIER = 2.0
PREVIEW_TIME_BUDGET_SECONDS_PER_CELL_BLOCK = 8
PREVIEW_TIME_BUDGET_GAP_OVERHEAD_SECONDS = 30
PREVIEW_TIME_BUDGET_LARGE_W1_SECONDS = 4
PREVIEW_TIME_BUDGET_SAFETY_MULTIPLIER = 1.20
PREVIEW_W1_AUTO_PAIR_THRESHOLD = 16_000_000
DEFAULT_CPU_MEMORY_FRACTION = 0.80
DEFAULT_CPU_MEMORY_MAX_MB = 64 * 1024
DEFAULT_CPU_MEMORY_MIN_MB = 4096

_ACTIVE_TRAINING_PROCESSES: dict[int, subprocess.Popen[str]] = {}
_ACTIVE_TRAINING_PROCESSES_LOCK = threading.Lock()


def terminate_active_training_subprocesses(reason: str = "stop requested") -> list[dict[str, Any]]:
    """Best-effort termination for isolated training workers.

    Agent thread interruption does not reliably interrupt a thread blocked in
    ``proc.wait()`` before the child exits.  Keep a small in-process registry so
    /api/stop can also terminate active training process groups.
    """
    terminated: list[dict[str, Any]] = []
    with _ACTIVE_TRAINING_PROCESSES_LOCK:
        items = list(_ACTIVE_TRAINING_PROCESSES.items())
    for pid, proc in items:
        if proc.poll() is not None:
            with _ACTIVE_TRAINING_PROCESSES_LOCK:
                _ACTIVE_TRAINING_PROCESSES.pop(pid, None)
            continue
        record: dict[str, Any] = {"pid": pid, "reason": reason}
        try:
            os.killpg(pid, signal.SIGTERM)
            record["signal"] = "SIGTERM"
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(pid, signal.SIGKILL)
                record["signal"] = "SIGKILL"
                proc.wait(timeout=5)
        except ProcessLookupError:
            record["signal"] = "already_exited"
        except Exception as exc:  # noqa: BLE001
            record["error"] = str(exc)
            try:
                proc.kill()
            except Exception:
                pass
        finally:
            record["returncode"] = proc.poll()
            with _ACTIVE_TRAINING_PROCESSES_LOCK:
                _ACTIVE_TRAINING_PROCESSES.pop(pid, None)
            terminated.append(record)
    return terminated


def _truthy_env(name: str, *, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"", "0", "false", "no", "off"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except Exception:
        return default


def _env_float_optional(name: str) -> Optional[float]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except Exception:
        return None


def _timepoint_counts_from_h5ad(adata_path: str) -> list[int]:
    path = Path(str(adata_path or "")).expanduser()
    if not path.is_file():
        return []
    try:
        import anndata as ad  # type: ignore

        data = ad.read_h5ad(path, backed="r")
        try:
            obs = data.obs
            if "time_point_processed" not in obs:
                return []
            counts = obs["time_point_processed"].value_counts(sort=False)
            try:
                counts = counts.sort_index()
            except Exception:
                pass
            return [int(v) for v in list(counts.values)]
        finally:
            try:
                data.file.close()
            except Exception:
                pass
    except Exception:
        return []


def _estimate_training_budget_sec(adata_path: str) -> Optional[int]:
    counts = _timepoint_counts_from_h5ad(adata_path)
    if len(counts) < 2:
        return None
    gap_effective_counts = [
        max(1, min(int(counts[i]), int(counts[i + 1])))
        for i in range(len(counts) - 1)
    ]
    gap_cell_units = [
        float(count) / float(TRAINING_TIME_BUDGET_CELL_BLOCK)
        for count in gap_effective_counts
    ]
    linear_cell_budget_sec = sum(
        unit * float(TRAINING_TIME_BUDGET_SECONDS_PER_CELL_BLOCK)
        for unit in gap_cell_units
    )
    gap_overhead_sec = sum(
        math.sqrt(max(unit, 0.0)) * float(TRAINING_TIME_BUDGET_GAP_OVERHEAD_SECONDS)
        for unit in gap_cell_units
    )
    raw_budget_sec = (
        float(TRAINING_TIME_BUDGET_BASE_SECONDS)
        + linear_cell_budget_sec
        + gap_overhead_sec
    ) * float(TRAINING_TIME_BUDGET_SAFETY_MULTIPLIER)
    rounding = max(1, int(TRAINING_TIME_BUDGET_ROUNDING_SECONDS))
    return int(
        max(
            TRAINING_TIME_BUDGET_MIN_SECONDS,
            math.ceil(raw_budget_sec / rounding) * rounding,
        )
    )


def _estimate_preview_timeout_from_counts(counts: list[int]) -> Optional[int]:
    if len(counts) < 2:
        return None
    gap_effective_counts = [
        max(1, min(int(counts[i]), int(counts[i + 1])))
        for i in range(len(counts) - 1)
    ]
    adjacent_pair_counts = [
        max(1, int(counts[i])) * max(1, int(counts[i + 1]))
        for i in range(len(counts) - 1)
    ]
    cell_units = sum(
        float(count) / float(TRAINING_TIME_BUDGET_CELL_BLOCK)
        for count in gap_effective_counts
    )
    gap_count = max(1, len(gap_effective_counts))
    max_pair_count = max(adjacent_pair_counts) if adjacent_pair_counts else 0
    pair_threshold = _env_float("CYTOBRIDGE_W1_AUTO_PAIR_THRESHOLD", PREVIEW_W1_AUTO_PAIR_THRESHOLD)
    pair_ratio = float(max_pair_count) / float(max(pair_threshold, 1.0))
    large_w1_overhead = (
        math.sqrt(max(pair_ratio, 0.0)) * float(PREVIEW_TIME_BUDGET_LARGE_W1_SECONDS)
        if pair_ratio > 1.0
        else 0.0
    )
    raw_budget_sec = (
        float(DEFAULT_PREVIEW_SUBPROCESS_TIMEOUT_SEC)
        + cell_units * float(PREVIEW_TIME_BUDGET_SECONDS_PER_CELL_BLOCK)
        + gap_count * float(PREVIEW_TIME_BUDGET_GAP_OVERHEAD_SECONDS)
        + large_w1_overhead
    ) * float(PREVIEW_TIME_BUDGET_SAFETY_MULTIPLIER)
    rounding = max(1, int(TRAINING_TIME_BUDGET_ROUNDING_SECONDS))
    return int(
        max(
            DEFAULT_PREVIEW_SUBPROCESS_TIMEOUT_SEC,
            math.ceil(raw_budget_sec / rounding) * rounding,
        )
    )


def _estimate_preview_timeout_sec(adata_path: str) -> Optional[int]:
    counts = _timepoint_counts_from_h5ad(adata_path)
    return _estimate_preview_timeout_from_counts(counts)


def _default_timeout_sec(*, purpose: str, adata_path: str) -> int:
    forced = _env_float_optional("CYTOBRIDGE_TRAINING_SUBPROCESS_TIMEOUT_SEC")
    if forced and forced > 0:
        return int(forced)
    purpose_name = str(purpose or "").strip().lower()
    if purpose_name in {"preview", "preview_inspect"}:
        preview_budget = _estimate_preview_timeout_sec(adata_path)
        if preview_budget is None:
            return int(DEFAULT_PREVIEW_SUBPROCESS_TIMEOUT_SEC)
        max_preview = int(
            _env_float(
                "CYTOBRIDGE_PREVIEW_SUBPROCESS_MAX_TIMEOUT_SEC",
                DEFAULT_PREVIEW_SUBPROCESS_MAX_TIMEOUT_SEC,
            )
        )
        max_preview = max(int(DEFAULT_PREVIEW_SUBPROCESS_TIMEOUT_SEC), max_preview)
        return int(min(max_preview, preview_budget))
    timeout_default = int(DEFAULT_TRAINING_SUBPROCESS_TIMEOUT_SEC)
    training_budget = _estimate_training_budget_sec(adata_path)
    if training_budget is None:
        return timeout_default
    budget_based = max(
        300,
        int(training_budget + TRAINING_SUBPROCESS_TIMEOUT_EXTRA_SEC),
        int(training_budget * TRAINING_SUBPROCESS_TIMEOUT_MULTIPLIER + 120),
    )
    return int(min(timeout_default, budget_based))


def _available_memory_mb() -> Optional[float]:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                return float(line.split()[1]) / 1024.0
    except Exception:
        return None
    return None


def _default_memory_limit_mb(device: str) -> Optional[float]:
    forced = _env_float("CYTOBRIDGE_TRAINING_SUBPROCESS_MEMORY_MB", 0.0)
    if forced > 0:
        return forced
    # RLIMIT_AS commonly breaks CUDA/PyTorch virtual-memory reservations; keep GPU
    # runs process-isolated but do not impose address-space limits unless forced.
    if str(device or "").strip().lower().startswith("cuda"):
        return None
    available = _available_memory_mb()
    if available is None:
        return DEFAULT_CPU_MEMORY_MAX_MB
    limit = max(DEFAULT_CPU_MEMORY_MIN_MB, available * DEFAULT_CPU_MEMORY_FRACTION)
    return min(DEFAULT_CPU_MEMORY_MAX_MB, limit)


def should_isolate_training_target(training_mode: str) -> bool:
    """Return whether actual training should run out-of-process."""
    if not _truthy_env("CYTOBRIDGE_ISOLATE_TRAINING", default=True):
        return False
    if _truthy_env("CYTOBRIDGE_ISOLATE_ALL_TRAINING", default=True):
        return True
    if not _truthy_env("CYTOBRIDGE_ISOLATE_CUSTOM_TRAINING", default=True):
        return False
    return str(training_mode or "").strip().lower() == "custom"


def _bundle_payload(bundle: TrainingRunBundle) -> Dict[str, str]:
    return {
        "run_id": bundle.run_id,
        "run_dir": str(bundle.run_dir),
        "logs_dir": str(bundle.logs_dir),
        "artifacts_dir": str(bundle.artifacts_dir),
        "checkpoints_dir": str(bundle.checkpoints_dir),
        "algorithm_snapshot_dir": str(bundle.algorithm_snapshot_dir),
        "run_manifest_path": str(bundle.run_manifest_path),
        "resolved_config_path": str(bundle.resolved_config_path),
        "training_log_path": str(bundle.training_log_path),
        "planner_context_path": str(bundle.planner_context_path),
        "trained_model_path": str(bundle.trained_model_path),
        "model_artifact_path": str(bundle.model_artifact_path),
        "model_state_path": str(bundle.model_state_path),
        "metrics_path": str(bundle.metrics_path),
    }


def run_training_in_subprocess(
    *,
    adata_path: str,
    stage: str,
    device: str,
    bundle: TrainingRunBundle,
    candidate_name: str = "",
    training_algorithm_id: str = "",
    output_dir: str = "",
    workspace_root: str = "",
    config_overrides: Optional[Dict[str, Any]] = None,
    claim_metric_spec: Optional[Dict[str, Any]] = None,
    max_epochs: Optional[int] = None,
    seed: int = 42,
    purpose: str = "training",
) -> Tuple[Dict[str, Any], Optional[str], Dict[str, Any]]:
    """Run execute_training_target in a supervised child process.

    The child writes a compact model artifact on success and writes its raw JSON
    result to a temp file for the parent. A full trained AnnData copy is only
    written when CYTOBRIDGE_SAVE_TRAINED_H5AD=1.
    """
    timeout_sec = _default_timeout_sec(purpose=purpose, adata_path=adata_path)
    memory_limit_mb = _default_memory_limit_mb(device)
    payload = {
        "adata_path": str(adata_path),
        "stage": str(stage),
        "device": str(device),
        "candidate_name": str(candidate_name or ""),
        "training_algorithm_id": str(training_algorithm_id or ""),
        "output_dir": str(output_dir or ""),
        "workspace_root": str(workspace_root or ""),
        "config_overrides": dict(config_overrides or {}),
        "claim_metric_spec": dict(claim_metric_spec or {}) if isinstance(claim_metric_spec, dict) else {},
        "max_epochs": max_epochs,
        "seed": int(seed),
        "bundle": _bundle_payload(bundle),
        "memory_limit_mb": memory_limit_mb,
        "purpose": str(purpose or "training"),
    }
    with tempfile.TemporaryDirectory(prefix="cytobridge_training_isolation_") as tmp:
        tmpdir = Path(tmp)
        payload_path = tmpdir / "payload.json"
        result_path = tmpdir / "result.json"
        stdout_path = tmpdir / "stdout.log"
        payload_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        env = os.environ.copy()
        env.setdefault("OPENBLAS_NUM_THREADS", "1")
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("MKL_NUM_THREADS", "1")
        cmd = [
            sys.executable,
            "-m",
            "cytobridge_agent.tools.training_isolation_worker",
            str(payload_path),
            str(result_path),
        ]
        start = time.time()
        with stdout_path.open("w", encoding="utf-8") as stdout:
            proc = subprocess.Popen(
                cmd,
                cwd=str(Path(workspace_root or Path.cwd()).expanduser().resolve()),
                env=env,
                stdout=stdout,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            with _ACTIVE_TRAINING_PROCESSES_LOCK:
                _ACTIVE_TRAINING_PROCESSES[proc.pid] = proc
            timed_out = False
            try:
                proc.wait(timeout=max(1, timeout_sec))
            except subprocess.TimeoutExpired:
                timed_out = True
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except Exception:
                    proc.kill()
                proc.wait(timeout=5)
            finally:
                with _ACTIVE_TRAINING_PROCESSES_LOCK:
                    _ACTIVE_TRAINING_PROCESSES.pop(proc.pid, None)
        elapsed = time.time() - start
        stdout_text = ""
        try:
            stdout_text = stdout_path.read_text(encoding="utf-8")[-8000:]
        except Exception:
            pass
        isolation = {
            "isolated": True,
            "returncode": proc.returncode,
            "elapsed_sec": round(elapsed, 3),
            "timeout_sec": timeout_sec,
            "timed_out": timed_out,
            "memory_limit_mb": memory_limit_mb,
            "stdout_tail": stdout_text,
        }
        if timed_out:
            error = f"isolated training subprocess timed out after {timeout_sec} seconds"
            return {
                "error": error,
                "runtime_sec": elapsed,
                "training_isolation": isolation,
            }, error, isolation
        if result_path.exists():
            try:
                result = json.loads(result_path.read_text(encoding="utf-8"))
            except Exception as exc:
                error = f"isolated training subprocess wrote invalid JSON: {exc}"
                return {
                    "error": error,
                    "runtime_sec": elapsed,
                    "training_isolation": isolation,
                }, error, isolation
            metrics = dict(result.get("metrics") or {})
            error = result.get("error")
            metrics["training_isolation"] = isolation
            return metrics, str(error) if error else None, isolation
        error = (
            "isolated training subprocess exited without a result JSON "
            f"(returncode={proc.returncode})"
        )
        if proc.returncode and proc.returncode < 0:
            sig = -int(proc.returncode)
            error += f"; likely killed by signal {sig}"
        return {
            "error": error,
            "runtime_sec": elapsed,
            "training_isolation": isolation,
        }, error, isolation
