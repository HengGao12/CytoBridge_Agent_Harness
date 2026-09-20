from __future__ import annotations

import argparse
import json
import threading
import traceback
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional


RESULT_PREFIX = "__CYTOBRIDGE_TOOL_WORKER_RESULT__="
EVENT_PREFIX = "__CYTOBRIDGE_WORKER_EVENT__="
_IMAGE_CAPTURE_QUEUE: list[str] = []
_IMAGE_CAPTURE_LOCK = threading.Lock()
_SAVEFIG_HOOKED = False


def _capture_path(path_like: Any) -> None:
    try:
        if path_like is None:
            return
        path = Path(path_like).expanduser()
        with _IMAGE_CAPTURE_LOCK:
            _IMAGE_CAPTURE_QUEUE.append(str(path.resolve()))
    except Exception:
        return


def _setup_savefig_hooks() -> None:
    """Install matplotlib savefig hooks inside worker process."""
    global _SAVEFIG_HOOKED
    if _SAVEFIG_HOOKED:
        return
    try:
        import matplotlib.pyplot as plt
        from matplotlib.figure import Figure
    except Exception:
        return

    original_plt_savefig = plt.savefig
    original_fig_savefig = Figure.savefig

    def hooked_plt_savefig(*args, **kwargs):
        original_plt_savefig(*args, **kwargs)
        fname = args[0] if args else kwargs.get("fname")
        _capture_path(fname)

    def hooked_fig_savefig(self, *args, **kwargs):
        original_fig_savefig(self, *args, **kwargs)
        fname = args[0] if args else kwargs.get("fname")
        _capture_path(fname)

    plt.savefig = hooked_plt_savefig
    Figure.savefig = hooked_fig_savefig
    _SAVEFIG_HOOKED = True


def _drain_captured_images() -> list[str]:
    with _IMAGE_CAPTURE_LOCK:
        imgs = list(_IMAGE_CAPTURE_QUEUE)
        _IMAGE_CAPTURE_QUEUE.clear()
    # Stable de-dup preserving order
    seen = set()
    out = []
    for p in imgs:
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


def _json_safe(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(v) for v in obj]
    try:
        import numpy as np
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except Exception:
        pass
    return str(obj)


def _ok_payload(
    text: str,
    analysis_results: Dict[str, Any],
    analysis_history: list,
    captured_images: list[str],
    updated_adata_path: str | None = None,
) -> Dict[str, Any]:
    payload = {
        "ok": True,
        "text": text,
        "analysis_results": _json_safe(analysis_results),
        "analysis_history": _json_safe(analysis_history),
        "captured_images": _json_safe(captured_images),
    }
    if updated_adata_path:
        payload["updated_adata_path"] = updated_adata_path
    return payload


def _err_payload(exc: BaseException) -> Dict[str, Any]:
    return {
        "ok": False,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback": traceback.format_exc(),
    }


def _emit_event(event_type: str, data: Optional[Dict[str, Any]] = None) -> None:
    payload = {
        "type": str(event_type or "status"),
        "data": data or {},
    }
    print(EVENT_PREFIX + json.dumps(payload, ensure_ascii=False), flush=True)


def _build_toolkit(adata_path: str, output_dir: str, device: str = "cpu"):
    from .downstream_analysis_toolkit import DownstreamAnalysisToolkit

    shared_state = {
        "tool_subprocess_guard_enabled": False,
        "tool_subprocess_timeout": 300,
        "tool_subprocess_persistent": False,
    }
    toolkit = DownstreamAnalysisToolkit(
        adata_path=adata_path,
        output_dir=output_dir,
        device=device,
        shared_state=shared_state,
    )
    return toolkit


def _warmup_umap(toolkit, adata_path: str) -> None:
    try:
        has_umap = "X_umap" in toolkit.adata.obsm
        if has_umap:
            _emit_event("status", {"message": "UMAP ready in downstream worker (cached)."})
            return
        _emit_event("status", {"message": "UMAP not found; downstream worker is computing UMAP..."})
        t0 = time.time()
        toolkit._ensure_embedding(preferred="umap")
        cost = time.time() - t0
        try:
            toolkit.adata.write_h5ad(str(adata_path))
        except Exception:
            pass
        _emit_event("status", {"message": f"UMAP computed in downstream worker ({cost:.1f}s)."})
    except Exception as exc:
        _emit_event("status", {"message": f"UMAP warmup failed in downstream worker: {exc}"})


def _run_tool_call(
    toolkit,
    method: str,
    kwargs: Dict[str, Any],
    analysis_results_seed: Optional[Dict[str, Any]] = None,
    output_dir: Optional[str] = None,
    core_method: bool = False,
) -> Dict[str, Any]:
    if isinstance(analysis_results_seed, dict):
        toolkit.analysis_results = analysis_results_seed
    if core_method:
        core_fn = getattr(toolkit._downstream_core, method)
        result = core_fn(adata=toolkit.adata, **kwargs)
        payload = {
            "ok": True,
            "title": str(getattr(result, "title", method)),
            "summary": _json_safe(getattr(result, "summary", [])),
            "artifacts": _json_safe(getattr(result, "artifacts", {})),
            "warnings": _json_safe(getattr(result, "warnings", [])),
            "payload": _json_safe(getattr(result, "payload", {})),
            "captured_images": _json_safe(_drain_captured_images()),
        }
        return payload

    fn = getattr(toolkit, method)
    text = fn(**kwargs)
    # Keep adata in worker memory for persistent mode.
    # Do not force h5ad write on every execute_python call:
    # object-typed columns can make h5ad serialization fail and incorrectly
    # surface as execute_python failure.
    updated_adata_path = None
    if method == "persist_runtime_adata":
        raw_target = kwargs.get("save_path")
        if raw_target:
            target = Path(str(raw_target)).expanduser()
        else:
            base_dir = Path(output_dir or toolkit.output_dir)
            target = base_dir / "downstream_final_adata.h5ad"
        try:
            target = target.resolve()
            if target.exists():
                updated_adata_path = str(target)
        except Exception:
            updated_adata_path = None
    captured_images = _drain_captured_images()
    return _ok_payload(
        str(text),
        toolkit.analysis_results,
        toolkit.analysis_history,
        captured_images=captured_images,
        updated_adata_path=updated_adata_path,
    )


def _server_loop(init_payload: Dict[str, Any]) -> int:
    current_adata_path = str(Path(str(init_payload["adata_path"])).expanduser().resolve())
    current_output_dir = str(Path(str(init_payload["output_dir"])).expanduser().resolve())
    current_device = str(init_payload.get("device", "cpu"))
    toolkit = _build_toolkit(current_adata_path, current_output_dir, current_device)
    _warmup_umap(toolkit, current_adata_path)

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception as exc:
            print(RESULT_PREFIX + json.dumps({"ok": False, "error": f"invalid request json: {exc}"}, ensure_ascii=False), flush=True)
            continue
        if str(req.get("command", "")).lower() == "shutdown":
            print(RESULT_PREFIX + json.dumps({"ok": True, "status": "shutdown"}, ensure_ascii=False), flush=True)
            return 0

        try:
            req_adata_path = str(Path(str(req.get("adata_path", current_adata_path))).expanduser().resolve())
            req_output_dir = str(Path(str(req.get("output_dir", current_output_dir))).expanduser().resolve())
            req_device = str(req.get("device", current_device))
            if req_adata_path != current_adata_path or req_output_dir != current_output_dir or req_device != current_device:
                toolkit = _build_toolkit(req_adata_path, req_output_dir, req_device)
                current_adata_path = req_adata_path
                current_output_dir = req_output_dir
                current_device = req_device
                _warmup_umap(toolkit, current_adata_path)

            method = str(req["method"])
            kwargs = dict(req.get("kwargs") or {})
            seed = req.get("analysis_results_seed")
            core_method = bool(req.get("core_method", False))
            payload = _run_tool_call(
                toolkit=toolkit,
                method=method,
                kwargs=kwargs,
                analysis_results_seed=seed if isinstance(seed, dict) else None,
                output_dir=current_output_dir,
                core_method=core_method,
            )
            updated_path = payload.get("updated_adata_path")
            if isinstance(updated_path, str) and updated_path:
                current_adata_path = str(Path(updated_path).expanduser().resolve())
            print(RESULT_PREFIX + json.dumps(payload, ensure_ascii=False), flush=True)
        except Exception as exc:
            print(RESULT_PREFIX + json.dumps(_err_payload(exc), ensure_ascii=False), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload-file", required=True)
    parser.add_argument("--server", action="store_true")
    args = parser.parse_args()

    try:
        _setup_savefig_hooks()
        payload = json.loads(Path(args.payload_file).read_text(encoding="utf-8"))
        if args.server:
            return _server_loop(payload)

        method = str(payload["method"])
        adata_path = str(payload["adata_path"])
        output_dir = str(payload["output_dir"])
        device = str(payload.get("device", "cpu"))
        kwargs = dict(payload.get("kwargs") or {})
        seed_analysis_results = dict(payload.get("analysis_results_seed") or {})
        core_method = bool(payload.get("core_method", False))

        toolkit = _build_toolkit(adata_path, output_dir, device)
        result_payload = _run_tool_call(
            toolkit=toolkit,
            method=method,
            kwargs=kwargs,
            analysis_results_seed=seed_analysis_results,
            output_dir=output_dir,
            core_method=core_method,
        )
        print(RESULT_PREFIX + json.dumps(result_payload, ensure_ascii=False))
        return 0
    except Exception as exc:
        print(RESULT_PREFIX + json.dumps(_err_payload(exc), ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
