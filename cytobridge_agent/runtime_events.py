"""Shared runtime event envelopes for CLI, TUI, and Web adapters."""
from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return _json_safe(value.tolist())
        if isinstance(value, np.generic):
            return _json_safe(value.item())
    except Exception:
        pass
    if hasattr(value, "model_dump"):
        try:
            return _json_safe(value.model_dump())
        except Exception:
            pass
    if hasattr(value, "dict"):
        try:
            return _json_safe(value.dict())
        except Exception:
            pass
    try:
        import json

        json.dumps(value, allow_nan=False)
        return value
    except Exception:
        return str(value)


def now_iso() -> str:
    return datetime.now().isoformat()


def runtime_event(
    event_type: str,
    data: Optional[Dict[str, Any]] = None,
    *,
    event_id: Optional[str] = None,
    timestamp: Optional[str] = None,
    session_id: Optional[str] = None,
    turn_id: Optional[str] = None,
    process_id: Optional[str] = None,
    source: Optional[str] = None,
) -> Dict[str, Any]:
    event: Dict[str, Any] = {
        "type": str(event_type or "event"),
        "event_id": str(event_id or uuid.uuid4()),
        "timestamp": timestamp or now_iso(),
        "data": _json_safe(dict(data or {})),
    }
    if session_id:
        event["session_id"] = str(session_id)
    if turn_id:
        event["turn_id"] = str(turn_id)
    if process_id:
        event["process_id"] = str(process_id)
    if source:
        event["source"] = str(source)
    return event


def coerce_runtime_event(event_data: Dict[str, Any], *, source: Optional[str] = None) -> Dict[str, Any]:
    event = dict(event_data or {})
    data = event.get("data")
    if not isinstance(data, dict):
        payload = event.get("payload")
        data = payload if isinstance(payload, dict) else {}
    return runtime_event(
        str(event.get("type") or event.get("event_type") or "event"),
        data,
        event_id=event.get("event_id"),
        timestamp=event.get("timestamp") or event.get("ts"),
        session_id=event.get("session_id"),
        turn_id=event.get("turn_id"),
        process_id=event.get("process_id"),
        source=event.get("source") or source,
    )
