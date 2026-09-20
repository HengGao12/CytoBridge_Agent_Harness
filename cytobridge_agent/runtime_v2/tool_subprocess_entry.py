from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional


def _configure_native_tool_environment() -> None:
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


def _inspect_h5ad_contract(payload: Dict[str, Any]) -> str:
    target = Path(str(payload.get("file_path") or "")).expanduser()
    time_key = str(payload.get("time_key") or "")
    label_keys = payload.get("label_keys") or []
    max_categories = payload.get("max_categories") or 20
    if not target.exists():
        return json.dumps({"ok": False, "error": f"path does not exist: {target}"}, ensure_ascii=False, indent=2)
    if target.suffix.lower() != ".h5ad":
        return json.dumps({"ok": False, "error": f"expected .h5ad file, got: {target}"}, ensure_ascii=False, indent=2)
    max_items = max(3, min(int(max_categories or 20), 50))
    import anndata as ad  # type: ignore

    adata = None
    try:
        adata = ad.read_h5ad(str(target), backed="r")
        obs = adata.obs.copy()
        obs_columns = list(map(str, obs.columns))
        requested_time = str(time_key or "").strip()
        common_time_keys = [
            requested_time,
            "time_point_processed",
            "time",
            "day",
            "stage",
            "Time point",
        ]
        selected_time_key = next((key for key in common_time_keys if key and key in obs.columns), "")
        common_label_keys = [
            "lineage",
            "clone",
            "clone_id",
            "barcode",
            "fate",
            "cell_type",
            "cell_type_broad",
            "label",
            "condition",
            "branch",
            "state",
            "state_info",
        ]
        keys: List[str] = []
        for key in list(label_keys or []) + common_label_keys:
            key = str(key or "").strip()
            if key and key in obs.columns and key not in keys:
                keys.append(key)
        if selected_time_key and selected_time_key not in keys:
            keys.insert(0, selected_time_key)

        def _value_counts(series: Any) -> Dict[str, int]:
            counts = series.astype(str).fillna("<NA>").value_counts(dropna=False).head(max_items)
            return {str(index): int(value) for index, value in counts.items()}

        obs_summary: Dict[str, Any] = {}
        for key in keys[:20]:
            series = obs[key]
            summary: Dict[str, Any] = {
                "dtype": str(series.dtype),
                "unique_count": int(series.astype(str).nunique(dropna=False)),
                "top_counts": _value_counts(series),
            }
            if selected_time_key and key != selected_time_key:
                try:
                    grouped = (
                        obs[[selected_time_key, key]]
                        .astype(str)
                        .groupby(selected_time_key)[key]
                        .nunique(dropna=False)
                        .head(max_items)
                    )
                    summary["unique_by_time"] = {str(index): int(value) for index, value in grouped.items()}
                except Exception:
                    pass
            obs_summary[key] = summary

        time_counts = _value_counts(obs[selected_time_key]) if selected_time_key else {}
        result = {
            "ok": True,
            "path": str(target.resolve()),
            "shape": [int(adata.n_obs), int(adata.n_vars)],
            "obs_columns": obs_columns[:120],
            "obs_columns_truncated": len(obs_columns) > 120,
            "selected_time_key": selected_time_key,
            "time_counts": time_counts,
            "summarized_obs_keys": keys[:20],
            "obs_summary": obs_summary,
            "obsm_keys": {str(key): list(map(int, adata.obsm[key].shape)) for key in list(adata.obsm.keys())[:30]},
            "layers": list(map(str, adata.layers.keys()))[:30],
            "uns_keys": list(map(str, adata.uns.keys()))[:50],
            "read_only": True,
        }
        return json.dumps(result, ensure_ascii=False, indent=2)
    finally:
        if adata is not None and getattr(adata, "isbacked", False):
            try:
                adata.file.close()
            except Exception:
                pass


def _search_literature(payload: Dict[str, Any]) -> Dict[str, Any]:
    from cytobridge_agent.rag.rag_main import DocumentRetriever, EnhancedKnowledgeSearcher, RAGManager

    query = str(payload.get("query") or "")
    output_dir = payload.get("output_dir") or None
    queries = [str(item).strip() for item in list(payload.get("queries") or []) if str(item).strip()]
    if not queries:
        queries = [query]
    top_k = max(1, min(int(payload.get("top_k") or 8), 20))
    manager = RAGManager(llm_client=None)
    # RAGManager.initialize() normally requires an LLM for index building and
    # HyDE/rerank. This clean subprocess search only uses already-built local
    # indexes. The parent process supplies any LLM-generated HyDE strings as
    # plain text query variants, and the parent reranks returned candidates.
    manager.searcher = EnhancedKnowledgeSearcher(llm_client=None)
    manager.document_retriever = DocumentRetriever()
    manager.knowledge_base = manager._load_knowledge_base()
    manager._load_paragraph_index()
    manager.initialized = True
    manager._generate_hyde_expansions = lambda _query, _num_expansions=0: list(queries)
    content = manager.search_literature(
        query,
        top_k=top_k,
        use_hyde=True,
        num_hyde_expansions=0,
        use_rerank=False,
        output_dir=output_dir,
    )
    result = manager.last_search_result or {}
    candidate_results = list(result.get("results", []) or [])
    trace = {
        "query": query,
        "hyde_expansions": queries,
        "selected_results": [
            {
                "title": item.get("title"),
                "pdf_filename": item.get("pdf_filename"),
                "relevance_score": item.get("relevance_score"),
                "pdf_path": item.get("pdf_path"),
            }
            for item in candidate_results
        ],
        "saved_records": dict(manager.last_search_record_paths or {}),
        "formatted_context": content,
        "subprocess_mode": "clean_python_no_fork",
        "llm_features_location": "parent_process_hyde_and_rerank",
    }
    return {"content": content, "trace": trace, "candidates": candidate_results}


def _search_theory(payload: Dict[str, Any]) -> Dict[str, Any]:
    from cytobridge_agent.rag.rag_main import RAGManager

    query = str(payload.get("query") or "")
    top_k = int(payload.get("top_k") or 5)
    output_dir = payload.get("output_dir") or None
    manager = RAGManager(llm_client=None)
    content = manager.search_theory(query, top_k=top_k, output_dir=output_dir)
    result = manager.last_theory_search_result or {}
    trace = {
        "query": query,
        "top_k": top_k,
        "selected_results": [
            {
                "title": item.get("title"),
                "book_id": item.get("book_id"),
                "page": item.get("page"),
                "recommended_read_pages": item.get("recommended_read_pages"),
                "relevance_score": item.get("relevance_score"),
                "pdf_path": item.get("pdf_path"),
            }
            for item in result.get("results", [])
        ],
        "saved_records": dict(manager.last_search_record_paths or {}),
        "formatted_context": content,
        "subprocess_mode": "clean_python_no_fork",
    }
    return {"content": content, "trace": trace}


def main() -> int:
    _configure_native_tool_environment()
    tool_name = sys.argv[1] if len(sys.argv) > 1 else ""
    payload = json.loads(sys.stdin.read() or "{}")
    try:
        if tool_name == "inspect_h5ad_contract":
            result = {"ok": True, "content": _inspect_h5ad_contract(payload)}
        elif tool_name == "search_literature":
            result = {"ok": True, **_search_literature(payload)}
        elif tool_name == "search_theory":
            result = {"ok": True, **_search_theory(payload)}
        else:
            result = {"ok": False, "error_type": "UnknownTool", "error": f"unsupported clean subprocess tool: {tool_name}"}
    except BaseException as exc:  # noqa: BLE001
        result = {
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(limit=80),
        }
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
