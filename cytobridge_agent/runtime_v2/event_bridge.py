from __future__ import annotations

from typing import Any, Dict


def enrich_event(payload: Dict[str, Any], phase: str) -> Dict[str, Any]:
    data = dict(payload)
    data.setdefault("phase", phase)
    data.setdefault("agent", "cytobridge")
    return data
