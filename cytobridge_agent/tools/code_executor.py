from __future__ import annotations

import io
import os
import runpy
import shutil
import signal
import sys
import time
import traceback
import uuid
import logging
import threading
import tempfile
import multiprocessing as mp
from queue import Empty
from pathlib import Path
from typing import Dict, Any, Set, List, Optional, Callable
from contextlib import redirect_stdout, redirect_stderr

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from scipy import sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .figure_policy import (
    apply_matplotlib_style,
    build_appendix_gallery,
    build_figure_stem,
    load_result_table,
    normalize_preset,
    pick_top_figures,
    save_figure_bundle,
)

logger = logging.getLogger(__name__)

_MANAGED_WRITE_SUFFIXES = {".py", ".md", ".json", ".yaml", ".yml", ".toml"}


def _executor_process_context() -> mp.context.BaseContext:
    requested = os.environ.get("CYTOBRIDGE_CODE_EXECUTOR_START_METHOD", "").strip().lower()
    candidates = [requested] if requested else []
    if os.name == "posix":
        main_path = Path(sys.argv[0] or "")
        can_spawn_main = bool(sys.argv[0] and sys.argv[0] not in {"-c", "-"} and main_path.exists())
        if threading.active_count() > 1 and can_spawn_main:
            # Forking a multithreaded agent process can deadlock before the
            # timeout loop starts. In normal CLI/script entrypoints, forkserver
            # avoids inheriting locked thread state while keeping strict
            # subprocess isolation. Interactive stdin/-c runs cannot be safely
            # re-imported by spawn/forkserver, so they fall through to fork.
            candidates.extend(["forkserver", "spawn", "fork"])
        else:
            candidates.extend(["fork", "forkserver", "spawn"])
    else:
        candidates.append("spawn")
    for name in candidates:
        if not name:
            continue
        try:
            return mp.get_context(name)
        except ValueError:
            continue
    return mp.get_context()


class ExecutionTimedOut(Exception):
    """Raised when execute_python exceeds the configured timeout."""


class ExecutionInterrupted(Exception):
    """Raised when execute_python is interrupted by a stop request."""


def _supports_signal_timeout() -> bool:
    return hasattr(signal, "SIGALRM") and threading.current_thread() is threading.main_thread()


def _terminate_process(proc: mp.Process, *, terminate_timeout: float = 2.0, kill_timeout: float = 2.0) -> None:
    """Best-effort hard stop for isolated execution workers."""
    if proc is None:
        return
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=terminate_timeout)
    if proc.is_alive():
        try:
            proc.kill()
        except AttributeError:  # pragma: no cover - Python < 3.7 fallback
            os.kill(proc.pid, signal.SIGKILL)
        proc.join(timeout=kill_timeout)


def _isolated_timeout_error(kind: str, timeout: int) -> str:
    if kind == "script":
        return f"⏳ Script execution timed out (limit: {timeout}s)."
    return (
        f"⏳ Execution Timed Out (Limit: {timeout}s). "
        "Code was interrupted after exceeding the timeout."
    )


def _isolated_timeout_result(
    *,
    kind: str,
    timeout: int,
    stdout_parts: List[str],
    stderr_parts: List[str],
) -> Dict[str, Any]:
    return {
        "success": False,
        "stdout": "".join(stdout_parts),
        "stderr": "".join(stderr_parts),
        "error": _isolated_timeout_error(kind, timeout),
        "traceback": "",
        "isolated_adata_path": None,
        "interrupted": False,
        "timed_out": True,
    }


class _StreamingCapture(io.TextIOBase):
    def __init__(self, stream_name: str, on_chunk: Optional[Callable[[str, str], None]] = None):
        super().__init__()
        self._stream_name = stream_name
        self._on_chunk = on_chunk
        self._buffer = io.StringIO()

    def write(self, s: str) -> int:
        text = str(s or "")
        if not text:
            return 0
        self._buffer.write(text)
        if self._on_chunk:
            self._on_chunk(self._stream_name, text)
        return len(text)

    def flush(self) -> None:
        return None

    def getvalue(self) -> str:
        return self._buffer.getvalue()


def _adata_fingerprint(adata_obj: Optional[sc.AnnData]) -> Optional[str]:
    if adata_obj is None:
        return None
    try:
        obs_sample = ""
        if getattr(adata_obj, "n_obs", 0) > 0:
            obs_sample = adata_obj.obs.head(min(5, adata_obj.n_obs)).to_json(date_format="iso")
        payload = {
            "n_obs": int(adata_obj.n_obs),
            "n_vars": int(adata_obj.n_vars),
            "obs_keys": list(map(str, adata_obj.obs_keys())),
            "var_columns": list(map(str, getattr(adata_obj.var, "columns", []))),
            "obsm": {str(k): list(getattr(v, "shape", ())) for k, v in adata_obj.obsm.items()},
            "layers": {str(k): list(getattr(v, "shape", ())) for k, v in adata_obj.layers.items()},
            "uns_keys": sorted(map(str, adata_obj.uns.keys())),
            "obs_sample": obs_sample,
        }
        return repr(payload)
    except Exception:
        return None


def _is_h5ad_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, bytes, int, float, bool, np.integer, np.floating, np.bool_))


def _sanitize_uns_value_for_h5ad(value: Any) -> Any:
    """Convert common non-HDF5-serializable `.uns` payloads into stable forms.

    Execution isolation should not fail before user code runs just because a
    prior workflow stored rich Python objects in `adata.uns`.
    """
    if _is_h5ad_scalar(value):
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        if isinstance(value, np.generic):
            return value.item()
        if value is None:
            return ""
        return value
    if isinstance(value, dict):
        return {str(k): _sanitize_uns_value_for_h5ad(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        if all(_is_h5ad_scalar(item) for item in value):
            return [_sanitize_uns_value_for_h5ad(item) for item in value]
        return {
            f"item_{idx:04d}": _sanitize_uns_value_for_h5ad(item)
            for idx, item in enumerate(value)
        }
    if isinstance(value, np.ndarray):
        if value.dtype.kind in {"O", "U"}:
            return _sanitize_uns_value_for_h5ad(value.tolist())
        return value
    if isinstance(value, pd.DataFrame):
        return value.astype(str)
    if isinstance(value, pd.Series):
        return value.astype(str).to_frame(name=str(value.name or "value"))
    return str(value)


def _write_h5ad_for_executor(adata_obj: sc.AnnData, path: str | Path) -> None:
    """Write AnnData for execution transport, sanitizing `.uns` on failure."""
    try:
        adata_obj.write_h5ad(path)
        return
    except Exception as first_error:
        sanitized = adata_obj.copy()
        sanitized.uns = {str(k): _sanitize_uns_value_for_h5ad(v) for k, v in sanitized.uns.items()}
        try:
            sanitized.write_h5ad(path)
        except Exception:
            raise first_error
        logger.warning("Sanitized AnnData .uns for isolated execution H5AD transport after write failure: %s", first_error)


def _managed_write_roots(owner: Optional[str] = None) -> List[Path]:
    owner_key = str(owner or "").strip().lower()
    if owner_key:
        roots = [
            Path.home() / ".cellcompass" / "training_algorithms" / owner_key,
            Path.home() / ".cellcompass" / "algorithm_campaigns" / owner_key,
        ]
    else:
        roots = [
            Path.home() / ".cellcompass" / "training_algorithms",
            Path.home() / ".cellcompass" / "algorithm_campaigns",
        ]
    extra = os.environ.get("CYTOBRIDGE_EXECUTE_PYTHON_PROTECTED_ROOTS", "")
    for item in extra.split(os.pathsep):
        item = item.strip()
        if item:
            roots.append(Path(item).expanduser())
    return [root.resolve() for root in roots if root.exists()]


def _managed_write_guard_deadline(tool_deadline: Optional[float]) -> Optional[float]:
    try:
        budget = float(os.environ.get("CYTOBRIDGE_EXECUTE_PYTHON_MANAGED_GUARD_BUDGET_SEC", "10.0"))
    except ValueError:
        budget = 10.0
    if budget <= 0:
        return time.perf_counter()
    deadline = time.perf_counter() + budget
    if tool_deadline is not None:
        deadline = min(deadline, tool_deadline)
    return deadline


def _snapshot_managed_text_files(
    deadline: Optional[float] = None,
    owner: Optional[str] = None,
) -> Optional[Dict[str, bytes]]:
    snapshot: Dict[str, bytes] = {}
    timed_out = False
    for root in _managed_write_roots(owner=owner):
        if deadline is not None and time.perf_counter() >= deadline:
            timed_out = True
            break
        for path in root.rglob("*"):
            if deadline is not None and time.perf_counter() >= deadline:
                timed_out = True
                break
            if not path.is_file() or path.suffix.lower() not in _MANAGED_WRITE_SUFFIXES:
                continue
            try:
                snapshot[str(path.resolve())] = path.read_bytes()
            except OSError:
                continue
        if timed_out:
            break
    if timed_out:
        scope = f" for owner `{owner}`" if owner else ""
        logger.warning("Managed-file write guard snapshot%s stopped at execute_python guard deadline.", scope)
        return None
    return snapshot


def _managed_owner_key(path_str: str) -> str:
    """Return the algorithm owner for a managed file path when it is inferable."""
    parts = Path(path_str).resolve().parts
    for marker in ("training_algorithms", "algorithm_campaigns"):
        if marker in parts:
            idx = parts.index(marker)
            if idx + 1 < len(parts):
                return parts[idx + 1]
    return path_str


def _rollback_managed_text_writes(
    before: Optional[Dict[str, bytes]],
    deadline: Optional[float] = None,
    owner: Optional[str] = None,
) -> tuple[List[str], bool]:
    if before is None:
        logger.warning("Skipping managed-file write rollback because the pre-execution snapshot was incomplete.")
        return [], False
    after = _snapshot_managed_text_files(deadline=deadline, owner=owner)
    if after is None:
        logger.warning("Skipping managed-file write rollback because the post-execution snapshot was incomplete.")
        return [], False
    changed = sorted(
        path for path, content in after.items()
        if path not in before or before[path] != content
    )
    deleted = sorted(path for path in before if path not in after)
    violations = changed + deleted
    if not violations:
        return [], False

    owners = {_managed_owner_key(path_str) for path_str in violations}
    allow_multi_owner = os.environ.get("CYTOBRIDGE_EXECUTE_PYTHON_ALLOW_MULTI_OWNER_ROLLBACK", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    if len(owners) > 1 and not allow_multi_owner:
        logger.error(
            "Detected managed-file changes spanning multiple algorithm owners during execute_python; "
            "skipping rollback to avoid reverting concurrent sessions. owners=%s files=%s",
            sorted(owners),
            violations[:12],
        )
        return violations, False

    # Roll back text/code/config mutations made through execute_python. Dedicated
    # planner tools own these files so proposal/workspace gates remain enforceable.
    for path_str in changed:
        path = Path(path_str)
        try:
            if path_str in before:
                path.write_bytes(before[path_str])
            else:
                path.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("Failed to roll back execute_python write to %s: %s", path_str, exc)
    for path_str in deleted:
        path = Path(path_str)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(before[path_str])
        except OSError as exc:
            logger.warning("Failed to restore execute_python-deleted file %s: %s", path_str, exc)
    return violations, True


def _build_stop_tracer(stop_check: Optional[Callable[[], bool]]) -> Optional[Callable[..., Any]]:
    if stop_check is None:
        return None

    def _tracer(frame, event, arg):
        if event in {"call", "line"} and stop_check():
            raise ExecutionInterrupted("Execution interrupted by user stop request.")
        return _tracer

    return _tracer


def _run_script_path(script_path: str, init_globals: Dict[str, Any]) -> Dict[str, Any]:
    script_file = str(Path(script_path).resolve())
    script_dir = str(Path(script_file).parent)
    old_argv = list(sys.argv)
    inserted = False
    try:
        if not sys.path or sys.path[0] != script_dir:
            sys.path.insert(0, script_dir)
            inserted = True
        sys.argv = [script_file]
        return runpy.run_path(script_file, run_name="__main__", init_globals=init_globals)
    finally:
        sys.argv = old_argv
        if inserted and sys.path and sys.path[0] == script_dir:
            del sys.path[0]


def _isolated_execute_worker(
    code: str,
    output_dir_str: str,
    input_path: Optional[str],
    figure_quality_preset: str,
    adata_input_path: Optional[str],
    adata_output_path: str,
    result_queue: "mp.Queue[Dict[str, Any]]",
) -> None:
    output_dir = Path(output_dir_str).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir_path = output_dir / "figures"
    figures_dir_path.mkdir(parents=True, exist_ok=True)
    scripts_dir_path = output_dir / "scripts"
    scripts_dir_path.mkdir(parents=True, exist_ok=True)

    adata_obj = None
    effective_adata_input = adata_input_path
    if not effective_adata_input and input_path:
        candidate_input = Path(input_path).expanduser()
        if candidate_input.exists() and candidate_input.suffix.lower() == ".h5ad":
            effective_adata_input = str(candidate_input)
    if effective_adata_input:
        try:
            adata_obj = ad.read_h5ad(effective_adata_input)
        except BaseException as e:  # noqa: BLE001
            result_queue.put(
                {
                    "kind": "result",
                    "data": {
                        "success": False,
                        "stdout": "",
                        "stderr": "",
                        "error": f"Failed to load AnnData in isolated worker: {e}",
                        "traceback": traceback.format_exc(),
                        "adata_written": False,
                    },
                }
            )
            return

    apply_matplotlib_style(figure_quality_preset)

    def save_pubfig(
        fig=None,
        name: str = "figure",
        formats: Optional[List[str]] = None,
        dpi: Optional[int] = None,
        bbox_inches: str = "tight",
    ) -> Dict[str, str]:
        target_fig = fig if fig is not None else plt.gcf()
        return save_figure_bundle(
            target_fig,
            base_name=name,
            output_dir=figures_dir_path,
            preset=figure_quality_preset,
            formats=formats,
            dpi=dpi,
            bbox_inches=bbox_inches,
        )

    def load_result_table_local(path_or_glob: str) -> Any:
        return load_result_table(path_or_glob, base_dir=output_dir)

    def pick_top_figures_local(metrics_json: Any, k: int = 6) -> List[Dict[str, Any]]:
        return pick_top_figures(metrics_json, k=k)

    def build_appendix_gallery_local(
        figures: List[Dict[str, Any]],
        title: str = "All Figures Appendix",
        description: str = "Auto-generated gallery for comprehensive figure coverage.",
    ) -> str:
        return build_appendix_gallery(figures, title=title, description=description)

    def make_figure_name(analysis: str, plot_type: str, variant: str = "default") -> str:
        return build_figure_stem(analysis=analysis, plot_type=plot_type, variant=variant)

    local_scope: Dict[str, Any] = {
        "adata": adata_obj,
        "sc": sc,
        "ad": ad,
        "sp": sp,
        "plt": plt,
        "np": np,
        "pd": pd,
        "os": os,
        "Path": Path,
        "input_path": input_path,
        "output_dir": str(output_dir),
        "figures_dir": str(figures_dir_path),
        "scripts_dir": str(scripts_dir_path),
        "save_pubfig": save_pubfig,
        "load_result_table": load_result_table_local,
        "pick_top_figures": pick_top_figures_local,
        "build_appendix_gallery": build_appendix_gallery_local,
        "make_figure_name": make_figure_name,
    }

    cb_module = None
    for module_name in ("CytoBridge", "cytobridge"):
        try:
            cb_module = __import__(module_name)
            break
        except ImportError:
            continue
    if cb_module is not None:
        local_scope["cb"] = cb_module

    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    success = False
    error_msg = ""
    traceback_text = ""
    adata_written = False

    try:
        stdout_stream = _StreamingCapture(
            "stdout",
            lambda stream_name, chunk: result_queue.put(
                {"kind": "delta", "stream": stream_name, "delta": chunk}
            ),
        )
        stderr_stream = _StreamingCapture(
            "stderr",
            lambda stream_name, chunk: result_queue.put(
                {"kind": "delta", "stream": stream_name, "delta": chunk}
            ),
        )
        with redirect_stdout(stdout_stream), redirect_stderr(stderr_stream):
            exec(code, local_scope)
        stdout_capture.write(stdout_stream.getvalue())
        stderr_capture.write(stderr_stream.getvalue())
        success = True
    except Exception as e:
        error_msg = str(e)
        traceback_text = traceback.format_exc()

    try:
        scope_adata = local_scope.get("adata")
        if success and scope_adata is not None:
            _write_h5ad_for_executor(scope_adata, adata_output_path)
            adata_written = True
    except Exception as e:  # noqa: BLE001
        success = False
        error_msg = f"Failed to serialize adata from isolated execution: {e}"
        traceback_text = traceback.format_exc()
        adata_written = False

    plt.close("all")
    result_queue.put(
        {
            "kind": "result",
            "data": {
                "success": success,
                "stdout": stdout_capture.getvalue(),
                "stderr": stderr_capture.getvalue(),
                "error": error_msg,
                "traceback": traceback_text,
                "adata_written": adata_written,
            },
        }
    )


def _isolated_run_script_worker(
    script_path_str: str,
    output_dir_str: str,
    input_path: Optional[str],
    figure_quality_preset: str,
    adata_input_path: Optional[str],
    adata_output_path: str,
    result_queue: "mp.Queue[Dict[str, Any]]",
) -> None:
    output_dir = Path(output_dir_str).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir_path = output_dir / "figures"
    figures_dir_path.mkdir(parents=True, exist_ok=True)
    scripts_dir_path = output_dir / "scripts"
    scripts_dir_path.mkdir(parents=True, exist_ok=True)

    adata_obj = None
    effective_adata_input = adata_input_path
    if not effective_adata_input and input_path:
        candidate_input = Path(input_path).expanduser()
        if candidate_input.exists() and candidate_input.suffix.lower() == ".h5ad":
            effective_adata_input = str(candidate_input)
    if effective_adata_input:
        try:
            adata_obj = ad.read_h5ad(effective_adata_input)
        except BaseException as e:  # noqa: BLE001
            result_queue.put(
                {
                    "kind": "result",
                    "data": {
                        "success": False,
                        "stdout": "",
                        "stderr": "",
                        "error": f"Failed to load AnnData in isolated worker: {e}",
                        "traceback": traceback.format_exc(),
                        "adata_written": False,
                    },
                }
            )
            return

    apply_matplotlib_style(figure_quality_preset)

    def save_pubfig(
        fig=None,
        name: str = "figure",
        formats: Optional[List[str]] = None,
        dpi: Optional[int] = None,
        bbox_inches: str = "tight",
    ) -> Dict[str, str]:
        target_fig = fig if fig is not None else plt.gcf()
        return save_figure_bundle(
            target_fig,
            base_name=name,
            output_dir=figures_dir_path,
            preset=figure_quality_preset,
            formats=formats,
            dpi=dpi,
            bbox_inches=bbox_inches,
        )

    def load_result_table_local(path_or_glob: str) -> Any:
        return load_result_table(path_or_glob, base_dir=output_dir)

    def pick_top_figures_local(metrics_json: Any, k: int = 6) -> List[Dict[str, Any]]:
        return pick_top_figures(metrics_json, k=k)

    def build_appendix_gallery_local(
        figures: List[Dict[str, Any]],
        title: str = "All Figures Appendix",
        description: str = "Auto-generated gallery for comprehensive figure coverage.",
    ) -> str:
        return build_appendix_gallery(figures, title=title, description=description)

    def make_figure_name(analysis: str, plot_type: str, variant: str = "default") -> str:
        return build_figure_stem(analysis=analysis, plot_type=plot_type, variant=variant)

    local_scope: Dict[str, Any] = {
        "adata": adata_obj,
        "sc": sc,
        "ad": ad,
        "sp": sp,
        "plt": plt,
        "np": np,
        "pd": pd,
        "os": os,
        "Path": Path,
        "input_path": input_path,
        "output_dir": str(output_dir),
        "figures_dir": str(figures_dir_path),
        "scripts_dir": str(scripts_dir_path),
        "save_pubfig": save_pubfig,
        "load_result_table": load_result_table_local,
        "pick_top_figures": pick_top_figures_local,
        "build_appendix_gallery": build_appendix_gallery_local,
        "make_figure_name": make_figure_name,
    }

    cb_module = None
    for module_name in ("CytoBridge", "cytobridge"):
        try:
            cb_module = __import__(module_name)
            break
        except ImportError:
            continue
    if cb_module is not None:
        local_scope["cb"] = cb_module

    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    success = False
    error_msg = ""
    traceback_text = ""
    adata_written = False

    try:
        stdout_stream = _StreamingCapture(
            "stdout",
            lambda stream_name, chunk: result_queue.put(
                {"kind": "delta", "stream": stream_name, "delta": chunk}
            ),
        )
        stderr_stream = _StreamingCapture(
            "stderr",
            lambda stream_name, chunk: result_queue.put(
                {"kind": "delta", "stream": stream_name, "delta": chunk}
            ),
        )
        with redirect_stdout(stdout_stream), redirect_stderr(stderr_stream):
            local_scope = _run_script_path(script_path_str, local_scope)
        stdout_capture.write(stdout_stream.getvalue())
        stderr_capture.write(stderr_stream.getvalue())
        success = True
    except Exception as e:
        error_msg = str(e)
        traceback_text = traceback.format_exc()

    try:
        scope_adata = local_scope.get("adata")
        if success and scope_adata is not None:
            _write_h5ad_for_executor(scope_adata, adata_output_path)
            adata_written = True
    except Exception as e:  # noqa: BLE001
        success = False
        error_msg = f"Failed to serialize adata from isolated script execution: {e}"
        traceback_text = traceback.format_exc()
        adata_written = False

    plt.close("all")
    result_queue.put(
        {
            "kind": "result",
            "data": {
                "success": success,
                "stdout": stdout_capture.getvalue(),
                "stderr": stderr_capture.getvalue(),
                "error": error_msg,
                "traceback": traceback_text,
                "adata_written": adata_written,
            },
        }
    )

class CodeExecutor:
    
    def __init__(
        self,
        adata: Optional[sc.AnnData],
        output_dir: Optional[Path] = None,
        input_path: Optional[str] = None,
        figure_quality_preset: str = "publication",
        managed_write_owner: Optional[str] = None,
    ):
        """
        Initialize
        
        Args:
            adata: AnnData 对象
            output_dir: 输出目录（用于保存图片等）
            input_path: Original input path hint for data bootstrapping.
            managed_write_owner: Optional algorithm id used to scope managed
                source/campaign write-guard snapshots.
        """
        self.adata = adata
        self.input_path = input_path
        self.managed_write_owner = str(managed_write_owner or "").strip().lower() or None
        self.figure_quality_preset = normalize_preset(figure_quality_preset)
        # 确保 output_dir 是绝对路径，方便管理
        self.output_dir = Path(output_dir).resolve() if output_dir else Path(".").resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.execution_history = []
        self.user_scope: Dict[str, Any] = {}
        
        # 记录当前目录下已有的文件 (用于 Diff)
        self.known_files = self._snapshot_files()

    def _build_runtime_scope(self, adata_obj: Optional[sc.AnnData], include_user_scope: bool = True) -> Dict[str, Any]:
        apply_matplotlib_style(self.figure_quality_preset)

        figures_dir_path = self.output_dir / "figures"
        figures_dir_path.mkdir(parents=True, exist_ok=True)
        scripts_dir_path = self.output_dir / "scripts"
        scripts_dir_path.mkdir(parents=True, exist_ok=True)

        def save_pubfig(
            fig=None,
            name: str = "figure",
            formats: Optional[List[str]] = None,
            dpi: Optional[int] = None,
            bbox_inches: str = "tight",
        ) -> Dict[str, str]:
            target_fig = fig if fig is not None else plt.gcf()
            return save_figure_bundle(
                target_fig,
                base_name=name,
                output_dir=figures_dir_path,
                preset=self.figure_quality_preset,
                formats=formats,
                dpi=dpi,
                bbox_inches=bbox_inches,
            )

        def load_result_table_local(path_or_glob: str) -> Any:
            return load_result_table(path_or_glob, base_dir=self.output_dir)

        def pick_top_figures_local(metrics_json: Any, k: int = 6) -> List[Dict[str, Any]]:
            return pick_top_figures(metrics_json, k=k)

        def build_appendix_gallery_local(
            figures: List[Dict[str, Any]],
            title: str = "All Figures Appendix",
            description: str = "Auto-generated gallery for comprehensive figure coverage.",
        ) -> str:
            return build_appendix_gallery(figures, title=title, description=description)

        def make_figure_name(analysis: str, plot_type: str, variant: str = "default") -> str:
            return build_figure_stem(analysis=analysis, plot_type=plot_type, variant=variant)

        base_scope: Dict[str, Any] = {
            "adata": adata_obj,
            "sc": sc,
            "ad": ad,
            "sp": sp,
            "plt": plt,
            "np": np,
            "pd": pd,
            "os": os,
            "Path": Path,
            "input_path": self.input_path,
            "output_dir": str(self.output_dir),
            "figures_dir": str(figures_dir_path),
            "scripts_dir": str(scripts_dir_path),
            "save_pubfig": save_pubfig,
            "load_result_table": load_result_table_local,
            "pick_top_figures": pick_top_figures_local,
            "build_appendix_gallery": build_appendix_gallery_local,
            "make_figure_name": make_figure_name,
        }
        if include_user_scope:
            local_scope = dict(self.user_scope)
            local_scope.update(base_scope)
        else:
            local_scope = dict(base_scope)

        cb_module = None
        for module_name in ("CytoBridge", "cytobridge"):
            try:
                cb_module = __import__(module_name)
                break
            except ImportError:
                continue
        if cb_module is not None:
            local_scope["cb"] = cb_module

        return local_scope

    def _snapshot_files(self) -> Set[str]:
        files: Set[str] = set()
        for f in self.output_dir.rglob("*"):
            if not f.is_file():
                continue
            try:
                files.add(str(f.relative_to(self.output_dir)))
            except Exception:
                files.add(str(f))
        return files

    def _scan_new_files(self) -> List[str]:
        """扫描输出目录，找出新生成的文件"""
        current_files = self._snapshot_files()
        new_files = list(current_files - self.known_files)
        # 更新已知文件列表
        self.known_files = current_files
        return new_files

    def execute(
        self,
        code: str,
        timeout: int = 300,
        timeout_mode: str = "isolated",
        stream_callback: Optional[Callable[[str, str], None]] = None,
        stop_check: Optional[Callable[[], bool]] = None,
    ) -> Dict[str, Any]:
        """
        执行 Python 代码，并返回详细的执行报告（包括生成了什么图）。
        """
        timeout = max(1, int(timeout))
        start_time = time.perf_counter()
        timeout_deadline = start_time + float(timeout)
        had_adata_before = self.adata is not None
        adata_fingerprint_before = _adata_fingerprint(self.adata)

        local_scope = self._build_runtime_scope(self.adata, include_user_scope=True)
        protected_names = set(local_scope.keys())

        # 2. 捕获标准输出和错误
        stdout_capture = _StreamingCapture("stdout", stream_callback)
        stderr_capture = _StreamingCapture("stderr", stream_callback)
        
        # 3. 执行前：同步一下当前文件列表
        # (防止把以前生成的文件误报为这次生成的)
        self.known_files = self._snapshot_files()
        managed_files_before = _snapshot_managed_text_files(
            deadline=_managed_write_guard_deadline(timeout_deadline),
            owner=self.managed_write_owner,
        )
        
        success = False
        error_msg = ""
        traceback_text = ""
        interrupted = False
        timed_out = False
        timeout_enforced = False
        new_adata = None
        converted_path = None
        stdout_text = ""
        stderr_text = ""
        
        timeout_mode = str(timeout_mode or "isolated").strip().lower()
        if timeout_mode not in {"shared", "isolated", "auto"}:
            timeout_mode = "isolated"

        effective_mode = timeout_mode
        if effective_mode == "auto":
            effective_mode = "shared" if _supports_signal_timeout() else "isolated"

        previous_trace = sys.gettrace()

        remaining_timeout = max(1, int(timeout_deadline - time.perf_counter()))
        if effective_mode == "isolated":
            if time.perf_counter() >= timeout_deadline:
                timed_out = True
                error_msg = _isolated_timeout_error("code", timeout)
                timeout_enforced = True
            else:
                isolated_result = self._execute_isolated(
                    code,
                    timeout=remaining_timeout,
                    stream_callback=stream_callback,
                    stop_check=stop_check,
                )
                success = isolated_result["success"]
                error_msg = isolated_result["error"]
                traceback_text = isolated_result["traceback"]
                stdout_text = isolated_result["stdout"]
                stderr_text = isolated_result["stderr"]
                interrupted = bool(isolated_result.get("interrupted"))
                timed_out = bool(isolated_result.get("timed_out"))
                if timed_out:
                    error_msg = _isolated_timeout_error("code", timeout)
                timeout_enforced = True
                if success and isolated_result.get("isolated_adata_path"):
                    try:
                        isolated_adata_path = isolated_result["isolated_adata_path"]
                        converted = self.output_dir / "converted_input.h5ad"
                        shutil.copy2(isolated_adata_path, converted)
                        self.adata = None
                        self.input_path = str(converted)
                        new_adata = None
                        converted_path = str(converted)
                        try:
                            from .adata_manager import AnnDataManager
                            manager = AnnDataManager()
                            manager.bind_path(str(converted))
                        except ImportError:
                            pass
                        except Exception as e:
                            logger.warning(f"Failed to bind isolated updated adata path: {e}")
                        finally:
                            try:
                                os.unlink(isolated_adata_path)
                            except OSError:
                                pass
                    except Exception as e:
                        success = False
                        error_msg = f"Failed to persist isolated adata result: {e}"
                        traceback_text = traceback.format_exc()
            # In isolated mode, user-defined variable scope is intentionally not persisted.
        else:
            if time.perf_counter() >= timeout_deadline:
                timed_out = True
                error_msg = _isolated_timeout_error("code", timeout)
                timeout_enforced = True
            else:
                try:
                    with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                        tracer = _build_stop_tracer(stop_check)
                        if tracer is not None:
                            sys.settrace(tracer)
                        if _supports_signal_timeout():
                            previous_handler = signal.getsignal(signal.SIGALRM)

                            def _alarm_handler(signum, frame):
                                raise ExecutionTimedOut(
                                    f"Execution timed out after {timeout} seconds."
                                )

                            signal.signal(signal.SIGALRM, _alarm_handler)
                            signal.setitimer(signal.ITIMER_REAL, max(0.001, timeout_deadline - time.perf_counter()))
                            timeout_enforced = True
                            try:
                                exec(code, local_scope)
                            finally:
                                signal.setitimer(signal.ITIMER_REAL, 0.0)
                                signal.signal(signal.SIGALRM, previous_handler)
                        else:
                            logger.warning(
                                "execute_python timeout enforcement unavailable in current thread; "
                                "running without hard timeout"
                            )
                            exec(code, local_scope)

                    # 如果没抛出异常，视为成功
                    success = True

                except BaseException as e:
                    # 捕获各种异常 (包括超时)
                    if isinstance(e, ExecutionTimedOut) or type(e).__name__ == 'FunctionTimedOut':
                        error_msg = (
                            f"⏳ Execution Timed Out (Limit: {timeout}s). "
                            "Code was interrupted after exceeding the timeout."
                        )
                        traceback_text = ""
                        timed_out = True
                    elif isinstance(e, ExecutionInterrupted):
                        error_msg = "⏹️ Execution interrupted by user stop request."
                        traceback_text = ""
                        interrupted = True
                    elif isinstance(e, KeyboardInterrupt):
                        error_msg = "⏹️ Execution interrupted by KeyboardInterrupt."
                        traceback_text = ""
                        interrupted = True
                    elif isinstance(e, SystemExit):
                        code_value = getattr(e, "code", "")
                        error_msg = f"SystemExit: {code_value}" if code_value else "SystemExit"
                        traceback_text = traceback.format_exc()
                    else:
                        error_msg = str(e)
                        traceback_text = traceback.format_exc()
                finally:
                    try:
                        sys.settrace(previous_trace)
                    except Exception:
                        sys.settrace(None)

        # 4. 执行后：文件侦测逻辑
        if success and effective_mode != "isolated":
            scope_adata = local_scope.get("adata")
            if scope_adata is not None:
                if scope_adata is not self.adata:
                    logger.info("检测到代码执行后 adata 对象发生了变更（重新赋值），正在更新引用...")
                    new_adata = scope_adata
                self.adata = scope_adata

                # Sync to global AnnDataManager (also covers in-place mutations)
                try:
                    from .adata_manager import AnnDataManager
                    manager = AnnDataManager()
                    logger.info("同步更新 AnnDataManager 中的 adata...")
                    manager.update(scope_adata)
                    # The object may now diverge from the path recorded by the manager.
                    # Mark it dirty so future path-based loads re-read from disk.
                    manager.mark_dirty()
                    if not had_adata_before:
                        converted = self.output_dir / "converted_input.h5ad"
                        _write_h5ad_for_executor(scope_adata, converted)
                        manager.update(scope_adata, str(converted))
                        converted_path = str(converted)
                except ImportError:
                    pass
                except Exception as e:
                    logger.warning(f"Failed to sync/save updated adata: {e}")
            self.user_scope = {
                key: value
                for key, value in local_scope.items()
                if key not in protected_names
                and key != "__builtins__"
                and not key.startswith("__")
            }
        if effective_mode != "isolated":
            stdout_text = stdout_capture.getvalue()
            stderr_text = stderr_capture.getvalue()
        managed_write_violations, managed_writes_rolled_back = _rollback_managed_text_writes(
            managed_files_before,
            deadline=_managed_write_guard_deadline(timeout_deadline),
            owner=self.managed_write_owner,
        )
        if managed_write_violations:
            success = False
            managed_action = (
                "Rolled back attempted writes"
                if managed_writes_rolled_back
                else "Detected possible concurrent managed writes and skipped rollback to avoid reverting other sessions"
            )
            error_msg = (
                "execute_python is read-only for managed CytoBridge algorithm/campaign "
                "source files. Use proposal_patch, workspace file tools, campaign tools, "
                f"or diagnostics/output files instead. {managed_action}: "
                + ", ".join(managed_write_violations[:12])
            )
            if len(managed_write_violations) > 12:
                error_msg += f", ... ({len(managed_write_violations)} total)"
            traceback_text = ""
        adata_fingerprint_after = _adata_fingerprint(self.adata)
        adata_changed = adata_fingerprint_before != adata_fingerprint_after
        duration_ms = int((time.perf_counter() - start_time) * 1000)

        # 不再自动保存悬空图，直接关闭防止下次重叠
        plt.close('all')
        
        # 扫描磁盘，看 Agent 自己显式保存了什么 (plt.savefig)
        all_new_files = self._scan_new_files()
        
        # 5. 构造返回结果
        output_log = stdout_text
        if stderr_text:
            output_log += "\n[STDERR]\n" + stderr_text

        # 构造给 Agent 看的最终文本 (Prompt Friendly)
        final_observation = ""
        if success:
            final_observation = "✅ Code Executed Successfully.\n"
        else:
            final_observation = f"❌ Execution Failed: {error_msg}\n"
            if traceback_text.strip():
                final_observation += f"Traceback:\n{traceback_text[:4000]}\n"
            
        if output_log.strip():
            final_observation += f"Logs:\n{output_log[:2000]}\n" # 截断防爆
        if success and effective_mode == "isolated":
            final_observation += (
                "Note: strict timeout mode used isolated execution. "
                "AnnData changes were synchronized back, but temporary Python variables were not persisted across calls.\n"
            )
            

        # 更新历史
        self.execution_history.append({
            "code": code,
            "success": success,
            "output": final_observation,
            "images": all_new_files
        })
        
        return {
            "success": success,
            "output": final_observation, # 这是直接喂给 LLM 的
            "images": all_new_files,     # 这是给上层程序用的
            "error": error_msg,
            "traceback": traceback_text,
            "new_adata": new_adata,
            "converted_path": converted_path,
            "stdout": stdout_text,
            "stderr": stderr_text,
            "effective_mode": effective_mode,
            "timed_out": timed_out,
            "interrupted": interrupted,
            "timeout_enforced": timeout_enforced,
            "adata_changed": adata_changed,
            "duration_ms": duration_ms,
        }

    def execute_script_file(
        self,
        script_path: str,
        timeout: int = 300,
        timeout_mode: str = "isolated",
        stream_callback: Optional[Callable[[str, str], None]] = None,
        stop_check: Optional[Callable[[], bool]] = None,
    ) -> Dict[str, Any]:
        """
        Execute a saved Python script using script semantics (`__name__ == "__main__"`),
        a fresh script scope, and the same runtime helpers/managed AnnData bindings.
        """
        timeout = max(1, int(timeout))
        start_time = time.perf_counter()
        timeout_deadline = start_time + float(timeout)
        script_file = Path(script_path).expanduser().resolve()
        had_adata_before = self.adata is not None
        adata_fingerprint_before = _adata_fingerprint(self.adata)
        local_scope = self._build_runtime_scope(self.adata, include_user_scope=False)

        stdout_capture = _StreamingCapture("stdout", stream_callback)
        stderr_capture = _StreamingCapture("stderr", stream_callback)

        self.known_files = self._snapshot_files()
        managed_files_before = _snapshot_managed_text_files(
            deadline=_managed_write_guard_deadline(timeout_deadline),
            owner=self.managed_write_owner,
        )

        success = False
        error_msg = ""
        traceback_text = ""
        interrupted = False
        timed_out = False
        timeout_enforced = False
        new_adata = None
        converted_path = None
        stdout_text = ""
        stderr_text = ""

        timeout_mode = str(timeout_mode or "isolated").strip().lower()
        if timeout_mode not in {"shared", "isolated", "auto"}:
            timeout_mode = "isolated"

        effective_mode = timeout_mode
        if effective_mode == "auto":
            effective_mode = "shared" if _supports_signal_timeout() else "isolated"

        previous_trace = sys.gettrace()

        if effective_mode == "isolated":
            if time.perf_counter() >= timeout_deadline:
                timed_out = True
                timeout_enforced = True
                error_msg = _isolated_timeout_error("script", timeout)
            else:
                remaining_timeout = max(1, int(timeout_deadline - time.perf_counter()))
                adata_input_path = None
                if self.adata is not None:
                    tmp_path = tempfile.NamedTemporaryFile(prefix="cb_isolated_adata_", suffix=".h5ad", delete=False)
                    tmp_path.close()
                    _write_h5ad_for_executor(self.adata, tmp_path.name)
                    adata_input_path = tmp_path.name

                ctx = _executor_process_context()
                result_queue: "mp.Queue[Dict[str, Any]]" = ctx.Queue()
                isolated_adata_path = str(Path(tempfile.gettempdir()) / f"cb_isolated_result_{uuid.uuid4().hex}.h5ad")
                proc = ctx.Process(
                    target=_isolated_run_script_worker,
                    args=(
                        str(script_file),
                        str(self.output_dir),
                        self.input_path,
                        self.figure_quality_preset,
                        adata_input_path,
                        isolated_adata_path,
                        result_queue,
                    ),
                    daemon=True,
                )
                proc.start()

                streamed_stdout_parts: List[str] = []
                streamed_stderr_parts: List[str] = []
                result_payload: Optional[Dict[str, Any]] = None
                deadline = time.time() + remaining_timeout
                while True:
                    now = time.time()
                    if now >= deadline:
                        timed_out = True
                        _terminate_process(proc)
                        break
                    if stop_check and stop_check():
                        interrupted = True
                        _terminate_process(proc)
                        break
                    remaining = max(0.0, deadline - now)
                    if remaining <= 0.0:
                        timed_out = True
                        _terminate_process(proc)
                        break
                    try:
                        item = result_queue.get(timeout=min(0.2, remaining))
                    except Empty:
                        if not proc.is_alive() and result_payload is not None:
                            break
                        if not proc.is_alive() and result_payload is None:
                            break
                        continue
                    kind = item.get("kind")
                    if kind == "delta":
                        stream_name = str(item.get("stream") or "")
                        delta = str(item.get("delta") or "")
                        if stream_name == "stdout":
                            streamed_stdout_parts.append(delta)
                        elif stream_name == "stderr":
                            streamed_stderr_parts.append(delta)
                    elif kind == "result":
                        if time.time() >= deadline:
                            timed_out = True
                            _terminate_process(proc)
                            break
                        result_payload = item.get("data") or {}
                        break

                timeout_enforced = True
                stdout_text = "".join(streamed_stdout_parts)
                stderr_text = "".join(streamed_stderr_parts)
                if interrupted:
                    error_msg = "⏹️ Script execution interrupted by user stop request."
                elif timed_out:
                    error_msg = _isolated_timeout_error("script", timeout)
                elif result_payload is None:
                    error_msg = "Script execution terminated without returning a result."
                else:
                    success = bool(result_payload.get("success"))
                    error_msg = str(result_payload.get("error") or "")
                    traceback_text = str(result_payload.get("traceback") or "")
                    stdout_text += str(result_payload.get("stdout") or "")
                    stderr_text += str(result_payload.get("stderr") or "")
                    if success and Path(isolated_adata_path).exists():
                        try:
                            converted = self.output_dir / "converted_input.h5ad"
                            shutil.copy2(isolated_adata_path, converted)
                            self.adata = None
                            self.input_path = str(converted)
                            new_adata = None
                            converted_path = str(converted)
                            try:
                                from .adata_manager import AnnDataManager
                                manager = AnnDataManager()
                                manager.bind_path(str(converted))
                            except Exception as e:
                                logger.warning(f"Failed to bind isolated updated adata path from script execution: {e}")
                        except Exception as e:
                            success = False
                            error_msg = f"Failed to persist isolated adata result from script execution: {e}"
                            traceback_text = traceback.format_exc()
                try:
                    if proc.is_alive():
                        _terminate_process(proc)
                    else:
                        proc.join(timeout=2)
                finally:
                    if adata_input_path:
                        try:
                            os.unlink(adata_input_path)
                        except OSError:
                            pass
                    try:
                        os.unlink(isolated_adata_path)
                    except OSError:
                        pass
        else:
            try:
                with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                    tracer = _build_stop_tracer(stop_check)
                    if tracer is not None:
                        sys.settrace(tracer)
                    if _supports_signal_timeout():
                        previous_handler = signal.getsignal(signal.SIGALRM)

                        def _alarm_handler(signum, frame):
                            raise ExecutionTimedOut(
                                f"Script execution timed out after {timeout} seconds."
                            )

                        signal.signal(signal.SIGALRM, _alarm_handler)
                        signal.setitimer(signal.ITIMER_REAL, max(0.001, timeout_deadline - time.perf_counter()))
                        timeout_enforced = True
                        try:
                            local_scope = _run_script_path(str(script_file), local_scope)
                        finally:
                            signal.setitimer(signal.ITIMER_REAL, 0.0)
                            signal.signal(signal.SIGALRM, previous_handler)
                    else:
                        logger.warning(
                            "run_saved_python_script timeout enforcement unavailable in current thread; "
                            "running without hard timeout"
                        )
                        local_scope = _run_script_path(str(script_file), local_scope)
                success = True
            except BaseException as e:
                if isinstance(e, ExecutionTimedOut) or type(e).__name__ == 'FunctionTimedOut':
                    error_msg = (
                        f"⏳ Script execution timed out (limit: {timeout}s). "
                        "Execution was interrupted after exceeding the timeout."
                    )
                    traceback_text = ""
                    timed_out = True
                elif isinstance(e, ExecutionInterrupted):
                    error_msg = "⏹️ Script execution interrupted by user stop request."
                    traceback_text = ""
                    interrupted = True
                elif isinstance(e, KeyboardInterrupt):
                    error_msg = "⏹️ Script execution interrupted by KeyboardInterrupt."
                    traceback_text = ""
                    interrupted = True
                elif isinstance(e, SystemExit):
                    code_value = getattr(e, "code", "")
                    error_msg = f"SystemExit: {code_value}" if code_value else "SystemExit"
                    traceback_text = traceback.format_exc()
                else:
                    error_msg = str(e)
                    traceback_text = traceback.format_exc()
            finally:
                try:
                    sys.settrace(previous_trace)
                except Exception:
                    sys.settrace(None)

        if effective_mode != "isolated":
            stdout_text = stdout_capture.getvalue()
            stderr_text = stderr_capture.getvalue()
        managed_write_violations, managed_writes_rolled_back = _rollback_managed_text_writes(
            managed_files_before,
            deadline=_managed_write_guard_deadline(timeout_deadline),
            owner=self.managed_write_owner,
        )
        if managed_write_violations:
            success = False
            managed_action = (
                "Rolled back attempted writes"
                if managed_writes_rolled_back
                else "Detected possible concurrent managed writes and skipped rollback to avoid reverting other sessions"
            )
            error_msg = (
                "run_saved_python_script is read-only for managed CytoBridge algorithm/campaign "
                "source files. Use proposal_patch, workspace file tools, campaign tools, "
                f"or diagnostics/output files instead. {managed_action}: "
                + ", ".join(managed_write_violations[:12])
            )
            if len(managed_write_violations) > 12:
                error_msg += f", ... ({len(managed_write_violations)} total)"
            traceback_text = ""

        if success and effective_mode != "isolated":
            scope_adata = local_scope.get("adata")
            if scope_adata is not None:
                if scope_adata is not self.adata:
                    new_adata = scope_adata
                self.adata = scope_adata
                try:
                    from .adata_manager import AnnDataManager
                    manager = AnnDataManager()
                    manager.update(scope_adata)
                    manager.mark_dirty()
                    if not had_adata_before:
                        converted = self.output_dir / "converted_input.h5ad"
                        _write_h5ad_for_executor(scope_adata, converted)
                        manager.update(scope_adata, str(converted))
                        converted_path = str(converted)
                except ImportError:
                    pass
                except Exception as e:
                    logger.warning(f"Failed to sync/save updated adata from script execution: {e}")

        adata_fingerprint_after = _adata_fingerprint(self.adata)
        adata_changed = adata_fingerprint_before != adata_fingerprint_after
        duration_ms = int((time.perf_counter() - start_time) * 1000)

        plt.close("all")
        all_new_files = self._scan_new_files()

        output_log = stdout_text
        if stderr_text:
            output_log += "\n[STDERR]\n" + stderr_text

        final_observation = f"Saved script: {script_file}\n"
        if success:
            final_observation += "✅ Script Executed Successfully.\n"
        else:
            final_observation += f"❌ Script Execution Failed: {error_msg}\n"
            if traceback_text.strip():
                final_observation += f"Traceback:\n{traceback_text[:4000]}\n"

        if output_log.strip():
            final_observation += f"Logs:\n{output_log[:2000]}\n"
        if success and effective_mode == "isolated":
            final_observation += (
                "Note: strict timeout mode used isolated script execution. "
                "AnnData changes were synchronized back, but script globals were not persisted across calls.\n"
            )

        self.execution_history.append({
            "script_path": str(script_file),
            "success": success,
            "output": final_observation,
            "images": all_new_files,
        })

        return {
            "success": success,
            "output": final_observation,
            "images": all_new_files,
            "error": error_msg,
            "traceback": traceback_text,
            "new_adata": new_adata,
            "converted_path": converted_path,
            "stdout": stdout_text,
            "stderr": stderr_text,
            "effective_mode": effective_mode,
            "timed_out": timed_out,
            "interrupted": interrupted,
            "timeout_enforced": timeout_enforced,
            "adata_changed": adata_changed,
            "duration_ms": duration_ms,
        }

    def _execute_isolated(
        self,
        code: str,
        timeout: int,
        stream_callback: Optional[Callable[[str, str], None]] = None,
        stop_check: Optional[Callable[[], bool]] = None,
    ) -> Dict[str, Any]:
        with tempfile.TemporaryDirectory(prefix="cytobridge_exec_") as tmpdir:
            tmpdir_path = Path(tmpdir)
            adata_input_path: Optional[str] = None
            fd, isolated_adata_path = tempfile.mkstemp(prefix="cytobridge_exec_result_", suffix=".h5ad")
            os.close(fd)
            try:
                os.unlink(isolated_adata_path)
            except FileNotFoundError:
                pass

            if self.adata is not None:
                adata_input = tmpdir_path / "isolated_input.h5ad"
                _write_h5ad_for_executor(self.adata, adata_input)
                adata_input_path = str(adata_input)

            ctx = _executor_process_context()
            result_queue: "mp.Queue[Dict[str, Any]]" = ctx.Queue()
            process = ctx.Process(
                target=_isolated_execute_worker,
                args=(
                    code,
                    str(self.output_dir),
                    self.input_path,
                    self.figure_quality_preset,
                    adata_input_path,
                    isolated_adata_path,
                    result_queue,
                ),
            )

            self.known_files = self._snapshot_files()
            process.start()
            deadline = time.monotonic() + float(timeout)
            result: Optional[Dict[str, Any]] = None
            streamed_stdout_parts: List[str] = []
            streamed_stderr_parts: List[str] = []

            while True:
                if time.monotonic() >= deadline:
                    _terminate_process(process, terminate_timeout=5.0, kill_timeout=5.0)
                    return _isolated_timeout_result(
                        kind="code",
                        timeout=timeout,
                        stdout_parts=streamed_stdout_parts,
                        stderr_parts=streamed_stderr_parts,
                    )
                try:
                    item = result_queue.get(timeout=0.1)
                    if item.get("kind") == "delta":
                        stream_name = str(item.get("stream") or "stdout")
                        delta = str(item.get("delta") or "")
                        if stream_name == "stderr":
                            streamed_stderr_parts.append(delta)
                        else:
                            streamed_stdout_parts.append(delta)
                        if stream_callback:
                            stream_callback(stream_name, delta)
                    elif item.get("kind") == "result":
                        if time.monotonic() >= deadline:
                            _terminate_process(process, terminate_timeout=5.0, kill_timeout=5.0)
                            return _isolated_timeout_result(
                                kind="code",
                                timeout=timeout,
                                stdout_parts=streamed_stdout_parts,
                                stderr_parts=streamed_stderr_parts,
                            )
                        result = dict(item.get("data") or {})
                        break
                except Empty:
                    pass

                if stop_check and stop_check():
                    _terminate_process(process, terminate_timeout=5.0, kill_timeout=5.0)
                    return {
                        "success": False,
                        "stdout": "".join(streamed_stdout_parts),
                        "stderr": "".join(streamed_stderr_parts),
                        "error": "⏹️ Execution interrupted by user stop request.",
                        "traceback": "",
                        "isolated_adata_path": None,
                        "interrupted": True,
                        "timed_out": False,
                    }

                if time.monotonic() >= deadline:
                    _terminate_process(process, terminate_timeout=5.0, kill_timeout=5.0)
                    return _isolated_timeout_result(
                        kind="code",
                        timeout=timeout,
                        stdout_parts=streamed_stdout_parts,
                        stderr_parts=streamed_stderr_parts,
                    )

                if not process.is_alive() and result_queue.empty():
                    break

            process.join(1)
            while True:
                try:
                    item = result_queue.get_nowait()
                    if item.get("kind") == "delta":
                        stream_name = str(item.get("stream") or "stdout")
                        delta = str(item.get("delta") or "")
                        if stream_name == "stderr":
                            streamed_stderr_parts.append(delta)
                        else:
                            streamed_stdout_parts.append(delta)
                        if stream_callback:
                            stream_callback(stream_name, delta)
                    elif item.get("kind") == "result" and result is None:
                        result = dict(item.get("data") or {})
                except Empty:
                    break

            if result is None:
                result = {
                    "success": False,
                    "stdout": "",
                    "stderr": "",
                    "error": "Isolated execution terminated without returning a result.",
                    "traceback": "",
                }
            result["isolated_adata_path"] = isolated_adata_path if result.get("adata_written") else None
            result.setdefault("interrupted", False)
            result.setdefault("timed_out", False)
            return result

    def get_history(self) -> list:
        return self.execution_history
    
    def clear_history(self):
        self.execution_history = []
