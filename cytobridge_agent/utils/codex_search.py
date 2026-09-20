from __future__ import annotations

from typing import Any, Dict, List, Optional

from .codex_sidecar_client import codex_sidecar_call


def codex_native_web_search(
    *,
    query: str,
    model: str,
    preferred_profile_id: Optional[str] = None,
    count: int = 5,
    allowed_domains: Optional[List[str]] = None,
    context_size: str = "medium",
    mode: str = "cached",
) -> Dict[str, Any]:
    return codex_sidecar_call(
        "chat.codex.web_search",
        {
            "query": str(query or "").strip(),
            "model": str(model or "").strip(),
            "preferred_profile_id": (preferred_profile_id or "").strip() or None,
            "count": int(count),
            "allowed_domains": list(allowed_domains or []),
            "context_size": str(context_size or "medium").strip().lower() or "medium",
            "mode": str(mode or "cached").strip().lower() or "cached",
        },
    ) or {}
