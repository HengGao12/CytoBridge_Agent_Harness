from __future__ import annotations

import multiprocessing as mp
import os
import pickle
import resource
import signal
import sys
import time
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional


def _configure_native_tool_environment() -> None:
    """Make process-isolated tools safer under macOS/native libraries."""
    if sys.platform == "darwin":
        os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(key, "1")


_configure_native_tool_environment()

@dataclass
class IsolatedToolResult:
    ok: bool
    result: Any = None
    error_type: Optional[str] = None
    error: Optional[str] = None
    traceback: Optional[str] = None
    returncode: Optional[int] = None
    timed_out: bool = False
    signal_name: Optional[str] = None
    stringified_result: bool = False


def _memory_limit_bytes() -> Optional[int]:
    raw = os.environ.get("CYTOBRIDGE_TOOL_MEMORY_LIMIT_MB", "").strip()
    if not raw:
        return None
    try:
        value = int(float(raw))
    except ValueError:
        return None
    if value <= 0:
        return None
    return value * 1024 * 1024


def _apply_child_resource_limits() -> None:
    limit = _memory_limit_bytes()
    if limit is None:
        return
    try:
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    except Exception:
        # Resource limits are a guardrail. Failure to set them should not
        # prevent the tool from running under process isolation.
        pass


def _safe_result_payload(result: Any) -> Dict[str, Any]:
    try:
        pickle.dumps(result)
        return {"ok": True, "result": result, "stringified_result": False}
    except BaseException:
        return {"ok": True, "result": str(result), "stringified_result": True}


def _tool_child_entry(
    conn: Any,
    func: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: Dict[str, Any],
) -> None:
    try:
        _configure_native_tool_environment()
        _apply_child_resource_limits()
        result = func(*args, **kwargs)
        conn.send(_safe_result_payload(result))
    except BaseException as exc:  # noqa: BLE001
        try:
            conn.send(
                {
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(limit=80),
                }
            )
        except BaseException:
            pass
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _process_contexts() -> list[mp.context.BaseContext]:
    requested = os.environ.get("CYTOBRIDGE_TOOL_ISOLATION_START_METHOD", "").strip().lower()
    names: list[str] = []
    if requested:
        names.append(requested)
    if os.name == "posix":
        # Prefer fork as the practical fallback because runtime tools are bound
        # methods that may hold locks/clients and therefore are not reliably
        # pickleable under spawn/forkserver.
        names.extend(["fork", "forkserver", "spawn"])
    else:
        names.append("spawn")

    contexts: list[mp.context.BaseContext] = []
    seen: set[str] = set()
    for name in names:
        if not name or name in seen:
            continue
        seen.add(name)
        try:
            contexts.append(mp.get_context(name))
        except ValueError:
            continue
    if contexts:
        return contexts
    return [mp.get_context()]


def run_tool_in_subprocess(
    func: Callable[..., Any],
    *args: Any,
    isolation_timeout: Optional[float] = None,
    tool_kwargs: Optional[Dict[str, Any]] = None,
    **legacy_kwargs: Any,
) -> IsolatedToolResult:
    """Run one tool function in a child process and return a non-throwing result.

    This protects the runtime loop from native crashes, process-level OOM kills,
    SystemExit, and uncaught BaseException raised by tool code. It intentionally
    returns crash details as data so the agent can diagnose and repair the
    failing tool/action in a later step.
    """

    kwargs: Dict[str, Any] = dict(tool_kwargs or {})
    if legacy_kwargs:
        # Backward compatibility for direct tests/callers that used
        # ``timeout=...`` as the subprocess timeout before tool keyword
        # arguments were passed separately.
        if set(legacy_kwargs) == {"timeout"} and isolation_timeout is None and not kwargs:
            isolation_timeout = legacy_kwargs["timeout"]
        else:
            kwargs.update(legacy_kwargs)

    start_error: Optional[BaseException] = None
    parent_conn = child_conn = proc = None
    for ctx in _process_contexts():
        parent_conn, child_conn = ctx.Pipe(duplex=False)
        proc = ctx.Process(target=_tool_child_entry, args=(child_conn, func, args, kwargs))
        try:
            proc.start()
            break
        except BaseException as exc:  # noqa: BLE001
            start_error = exc
            for conn in (parent_conn, child_conn):
                try:
                    conn.close()
                except Exception:
                    pass
            parent_conn = child_conn = proc = None
            continue
    if proc is None or parent_conn is None or child_conn is None:
        return IsolatedToolResult(
            ok=False,
            error_type=type(start_error).__name__ if start_error else "ProcessStartError",
            error=str(start_error) if start_error else "failed to start tool subprocess",
            traceback=traceback.format_exc(limit=80) if start_error else None,
        )
    try:
        child_conn.close()
    except Exception:
        pass

    payload: Optional[Dict[str, Any]] = None
    deadline = (
        time.monotonic() + float(isolation_timeout)
        if isolation_timeout is not None
        else None
    )
    try:
        # Read from the pipe while the child is alive.  Joining first can
        # deadlock for large structured results such as image data URLs: the
        # child blocks in ``conn.send`` once the OS pipe fills, while the parent
        # waits for the child to exit.
        while True:
            if parent_conn.poll(0.05):
                payload = parent_conn.recv()
                break
            if not proc.is_alive():
                if parent_conn.poll():
                    payload = parent_conn.recv()
                break
            if deadline is not None and time.monotonic() >= deadline:
                proc.terminate()
                proc.join(5)
                if proc.is_alive():
                    try:
                        proc.kill()
                    except Exception:
                        pass
                    proc.join(5)
                return IsolatedToolResult(
                    ok=False,
                    error_type="TimeoutError",
                    error=(
                        f"tool subprocess exceeded timeout {isolation_timeout:.1f}s"
                        if isolation_timeout
                        else "tool subprocess timed out"
                    ),
                    returncode=proc.exitcode,
                    timed_out=True,
                )
    except EOFError:
        payload = None
    finally:
        if payload is not None and proc.is_alive():
            proc.join(5)
            if proc.is_alive():
                proc.terminate()
                proc.join(5)
                if proc.is_alive():
                    try:
                        proc.kill()
                    except Exception:
                        pass
                    proc.join(5)
        else:
            proc.join(0)
        try:
            parent_conn.close()
        except Exception:
            pass

    returncode = proc.exitcode
    if isinstance(payload, dict):
        return IsolatedToolResult(
            ok=bool(payload.get("ok")),
            result=payload.get("result"),
            error_type=payload.get("error_type"),
            error=payload.get("error"),
            traceback=payload.get("traceback"),
            returncode=returncode,
            stringified_result=bool(payload.get("stringified_result")),
        )

    signal_name = None
    if isinstance(returncode, int) and returncode < 0:
        signum = -returncode
        try:
            signal_name = signal.Signals(signum).name
        except ValueError:
            signal_name = f"SIG{signum}"
    return IsolatedToolResult(
        ok=False,
        error_type="ToolProcessCrashed",
        error=(
            f"tool subprocess exited without returning a result"
            f" (returncode={returncode}, signal={signal_name or 'none'})"
        ),
        returncode=returncode,
        signal_name=signal_name,
    )
