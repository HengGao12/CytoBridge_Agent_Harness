from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional

import httpx


BOHRIUM_PAPER_SEARCH_ENDPOINT = "https://openapi.dp.tech/openapi/v1/paper/rag/pass/keyword"
DEFAULT_TIMEOUT_SEC = 30.0
DEFAULT_PAGE_SIZE = 10
MAX_PAGE_SIZE = 100
BOHRIUM_ACCESS_KEY_ENV_VARS = ("ACCESS_KEY", "BOHRIUM_ACCESS_KEY")


def _configured_access_key_env_var() -> str:
    for name in BOHRIUM_ACCESS_KEY_ENV_VARS:
        if str(os.environ.get(name) or "").strip():
            return name
    return ""


def get_bohrium_paper_search_status() -> Dict[str, Any]:
    """Return Bohrium search availability without exposing the secret value."""
    configured_env_var = _configured_access_key_env_var()
    available = bool(configured_env_var)
    return {
        "available": available,
        "backend": "bohrium-paper-search",
        "tool_name": "search_bohrium_paper",
        "required_env_vars": list(BOHRIUM_ACCESS_KEY_ENV_VARS),
        "configured_env_var": configured_env_var,
        "missing_reason": "" if available else "ACCESS_KEY or BOHRIUM_ACCESS_KEY is not set.",
        "safe_to_call": available,
        "secret_value_exposed": False,
        "agent_guidance": (
            "Bohrium paper search is configured; use search_bohrium_paper for broad external literature discovery."
            if available
            else "Bohrium paper search is not configured; use search_literature and web_search instead."
        ),
    }


def _normalize_page_size(value: int) -> int:
    try:
        page_size = int(value)
    except Exception:
        page_size = DEFAULT_PAGE_SIZE
    return max(1, min(page_size, MAX_PAGE_SIZE))


def _derive_words(query: str, *, limit: int = 8) -> List[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_+-]{2,}", str(query or ""))
    words: List[str] = []
    seen = set()
    for token in tokens:
        lowered = token.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        words.append(token)
        if len(words) >= limit:
            break
    return words or [str(query or "").strip()]


def _clean_string_list(value: Optional[List[str]]) -> List[str]:
    cleaned: List[str] = []
    seen = set()
    for item in value or []:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        cleaned.append(text)
    return cleaned


def search_bohrium_papers(
    *,
    query: str,
    words: Optional[List[str]] = None,
    page_size: int = DEFAULT_PAGE_SIZE,
    start_time: str = "",
    end_time: str = "",
    search_type: int = 5,
    jcr_zones: Optional[List[str]] = None,
    include_dbs: Optional[List[str]] = None,
    timeout_sec: float = DEFAULT_TIMEOUT_SEC,
) -> Dict[str, Any]:
    """Search Bohrium's paper RAG API with ACCESS_KEY or BOHRIUM_ACCESS_KEY."""
    normalized_query = str(query or "").strip()
    if not normalized_query:
        return {
            "success": False,
            "backend": "bohrium-paper-search",
            "query": "",
            "results": [],
            "summary": "",
            "error": "query must not be empty",
        }

    configured_env_var = _configured_access_key_env_var()
    access_key = str(os.environ.get(configured_env_var) or "").strip() if configured_env_var else ""
    if not access_key:
        return {
            "success": False,
            "backend": "bohrium-paper-search",
            "query": normalized_query,
            "results": [],
            "summary": "",
            "error": "ACCESS_KEY or BOHRIUM_ACCESS_KEY is not set.",
            "availability": get_bohrium_paper_search_status(),
        }

    normalized_words = _clean_string_list(words) or _derive_words(normalized_query)
    payload: Dict[str, Any] = {
        "words": normalized_words,
        "question": normalized_query,
        "type": int(search_type),
        "startTime": str(start_time or ""),
        "endTime": str(end_time or ""),
        "pageSize": _normalize_page_size(page_size),
    }
    cleaned_jcr = _clean_string_list(jcr_zones)
    cleaned_dbs = _clean_string_list(include_dbs)
    if cleaned_jcr:
        payload["jcrZones"] = cleaned_jcr
    if cleaned_dbs:
        payload["includeDbs"] = cleaned_dbs

    try:
        response = httpx.post(
            BOHRIUM_PAPER_SEARCH_ENDPOINT,
            headers={"accessKey": access_key, "Content-Type": "application/json"},
            json=payload,
            timeout=float(timeout_sec or DEFAULT_TIMEOUT_SEC),
        )
        response.raise_for_status()
        data = response.json()
    except Exception as exc:
        return {
            "success": False,
            "backend": "bohrium-paper-search",
            "query": normalized_query,
            "words": normalized_words,
            "results": [],
            "summary": "",
            "error": str(exc),
        }

    if int(data.get("code", 0) or 0) != 0:
        return {
            "success": False,
            "backend": "bohrium-paper-search",
            "query": normalized_query,
            "words": normalized_words,
            "results": [],
            "summary": "",
            "error": str(data.get("message") or "Bohrium paper API returned non-zero code"),
            "raw_code": data.get("code"),
        }

    results: List[Dict[str, Any]] = []
    for item in data.get("data", []) or []:
        if not isinstance(item, dict):
            continue
        results.append(
            {
                "title": item.get("enName") or item.get("zhName") or "",
                "doi": item.get("doi") or "",
                "paper_id": item.get("paperId") or "",
                "abstract": item.get("enAbstract") or item.get("zhAbstract") or "",
                "authors": item.get("authors") or [],
                "journal": item.get("publicationEnName") or "",
                "published_at": item.get("coverDateStart") or "",
                "impact_factor": item.get("impactFactor"),
                "citation_count": item.get("citationNums"),
                "popularity": item.get("popularity"),
                "pieces": item.get("pieces") or [],
                "figures": item.get("figures") or [],
            }
        )

    summary_lines = []
    for idx, item in enumerate(results[:5], start=1):
        title = str(item.get("title") or "").strip()
        journal = str(item.get("journal") or "").strip()
        published_at = str(item.get("published_at") or "").strip()
        doi = str(item.get("doi") or "").strip()
        bits = [title]
        meta = ", ".join(part for part in [journal, published_at, f"DOI: {doi}" if doi else ""] if part)
        if meta:
            bits.append(f"({meta})")
        summary_lines.append(f"{idx}. " + " ".join(bits).strip())

    return {
        "success": True,
        "backend": "bohrium-paper-search",
        "query": normalized_query,
        "words": normalized_words,
        "availability": get_bohrium_paper_search_status(),
        "summary": "\n".join(summary_lines),
        "results": results,
        "notes": [
            "Uses Bohrium OpenAPI paper RAG with ACCESS_KEY from the OpenCLAW skill environment.",
            "Use alongside search_literature for local CytoBridge/theory-grounded evidence.",
        ],
    }
