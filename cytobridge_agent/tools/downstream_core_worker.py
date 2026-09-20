from __future__ import annotations

import argparse
import json
import threading
import traceback
import sys
import time
from pathlib import Path
from typing import Any, Dict

import anndata as ad
import scanpy as sc

from .umap_policy import build_umap_policy, compact_policy_signature


RESULT_PREFIX = "__CYTOBRIDGE_WORKER_RESULT__="
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


def _ok_payload(result: Any) -> Dict[str, Any]:
    return {
        "ok": True,
        "title": result.title,
        "summary": result.summary,
        "artifacts": result.artifacts,
        "warnings": result.warnings,
        "payload": result.payload,
        "render_text": result.render_text(),
        "captured_images": _drain_captured_images(),
    }


def _err_payload(exc: BaseException) -> Dict[str, Any]:
    return {
        "ok": False,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback": traceback.format_exc(),
    }


def _emit_event(event_type: str, data: Dict[str, Any] | None = None) -> None:
    payload = {"type": str(event_type or "status"), "data": data or {}}
    print(EVENT_PREFIX + json.dumps(payload, ensure_ascii=False), flush=True)


def _warmup_umap(adata, adata_path: str) -> None:
    try:
        if "X_umap" in adata.obsm:
            _emit_event("status", {"message": "UMAP ready in downstream core worker (cached)."})
            return
        _emit_event("status", {"message": "UMAP not found; downstream core worker is computing UMAP..."})
        t0 = time.time()
        use_rep = None
        if "X_latent" in adata.obsm:
            use_rep = "X_latent"
        elif "X_pca" in adata.obsm:
            use_rep = "X_pca"
        policy = build_umap_policy(
            n_obs=int(adata.n_obs),
            use_rep=use_rep,
            quality_preset="publication",
            viz_goal="publication",
            mode="auto",
            overrides=None,
        )
        neighbors_cfg = policy.get("neighbors", {}) if isinstance(policy, dict) else {}
        umap_cfg = policy.get("umap", {}) if isinstance(policy, dict) else {}
        rep = str(neighbors_cfg.get("use_rep") or use_rep or "X")
        n_neighbors = int(neighbors_cfg.get("n_neighbors", 30))
        metric = str(neighbors_cfg.get("metric", "euclidean"))
        conn = adata.obsp.get("connectivities") if hasattr(adata, "obsp") else None
        if "neighbors" not in adata.uns or conn is None:
            if rep == "X":
                sc.pp.neighbors(adata, n_neighbors=n_neighbors, metric=metric)
            else:
                sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep=rep, metric=metric)
        sc.tl.umap(
            adata,
            min_dist=float(umap_cfg.get("min_dist", 0.3)),
            spread=float(umap_cfg.get("spread", 1.0)),
            random_state=int(umap_cfg.get("random_state", 0)),
        )
        adata.uns["_cytobridge_umap_config"] = compact_policy_signature(policy)
        try:
            adata.write_h5ad(str(adata_path))
        except Exception:
            pass
        cost = time.time() - t0
        _emit_event("status", {"message": f"UMAP computed in downstream core worker ({cost:.1f}s)."})
    except Exception as exc:
        _emit_event("status", {"message": f"UMAP warmup failed in downstream core worker: {exc}"})


def _build_core(adata_path: str, output_dir: str, device: str):
    from .downstream_refactor_core import DownstreamRefactorCore

    adata = ad.read_h5ad(adata_path)
    core = DownstreamRefactorCore(output_dir=Path(output_dir), device=device)
    return adata, core


def _run_core_call(adata, core, method: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    fn = getattr(core, method)
    result = fn(adata=adata, **kwargs)
    return _ok_payload(result)


def _server_loop(init_payload: Dict[str, Any]) -> int:
    current_adata_path = str(Path(str(init_payload["adata_path"])).expanduser().resolve())
    current_output_dir = str(Path(str(init_payload["output_dir"])).expanduser().resolve())
    current_device = str(init_payload.get("device", "cpu"))
    umap_warmup = bool(init_payload.get("umap_warmup", False))
    adata, core = _build_core(current_adata_path, current_output_dir, current_device)
    if umap_warmup:
        _warmup_umap(adata, current_adata_path)
    else:
        _emit_event("status", {"message": "UMAP warmup disabled in downstream core worker (lazy mode)."})

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
            if (
                req_adata_path != current_adata_path
                or req_output_dir != current_output_dir
                or req_device != current_device
            ):
                adata, core = _build_core(req_adata_path, req_output_dir, req_device)
                current_adata_path = req_adata_path
                current_output_dir = req_output_dir
                current_device = req_device
                if umap_warmup:
                    _warmup_umap(adata, current_adata_path)

            method = str(req["method"])
            kwargs = dict(req.get("kwargs") or {})
            payload = _run_core_call(adata=adata, core=core, method=method, kwargs=kwargs)
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

        adata, core = _build_core(adata_path, output_dir, device)
        result_payload = _run_core_call(adata=adata, core=core, method=method, kwargs=kwargs)
        print(RESULT_PREFIX + json.dumps(result_payload, ensure_ascii=False))
        return 0
    except Exception as exc:
        print(RESULT_PREFIX + json.dumps(_err_payload(exc), ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
