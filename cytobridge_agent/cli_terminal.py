"""Terminal-facing helpers for CellCompass CLI sessions."""
from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


def _plain(value: Any) -> str:
    return str(value or "").replace("\n", " ").strip()


def _shorten(value: Any, width: int) -> str:
    text = _plain(value)
    if width <= 0 or len(text) <= width:
        return text
    return text[: max(0, width - 3)] + "..."


def _format_updated_at(value: Any) -> str:
    text = _plain(value)
    if not text:
        return ""
    normalized = text.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(normalized)
        return dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return text[:16]


def summarize_session(item: Dict[str, Any], *, max_title: int = 72) -> Dict[str, str]:
    metadata = dict(item.get("metadata") or {})
    session_id = _plain(item.get("session_id"))
    title = _plain(metadata.get("title") or item.get("preview") or "Untitled")
    preview = _plain(item.get("preview"))
    cwd = _plain(metadata.get("cwd"))
    output_dir = _plain(metadata.get("output_dir"))
    input_path = _plain(metadata.get("input_path"))
    location = cwd or output_dir or (str(Path(input_path).parent) if input_path else "")
    return {
        "session_id": session_id,
        "updated_at": _format_updated_at(metadata.get("updated_at") or metadata.get("created_at")),
        "title": _shorten(title, max_title),
        "preview": _shorten(preview, max_title),
        "location": _shorten(location, 48),
    }


def _session_search_text(item: Dict[str, Any]) -> str:
    row = summarize_session(item, max_title=0)
    metadata = dict(item.get("metadata") or {})
    parts = [
        row.get("session_id"),
        row.get("title"),
        row.get("preview"),
        row.get("location"),
        metadata.get("input_path"),
        metadata.get("output_dir"),
        metadata.get("cwd"),
    ]
    return " ".join(_plain(part).lower() for part in parts if _plain(part))


def filter_sessions(
    sessions: List[Dict[str, Any]],
    *,
    query: Optional[str] = None,
    cwd: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Filter saved-session summaries by text query and/or working directory."""
    result = list(sessions)
    if query:
        needle = _plain(query).lower()
        result = [item for item in result if needle in _session_search_text(item)]
    if cwd:
        cwd_text = _plain(cwd).lower()
        result = [
            item
            for item in result
            if cwd_text in summarize_session(item, max_title=0).get("location", "").lower()
        ]
    return result


def format_session_listing(
    sessions: List[Dict[str, Any]],
    *,
    numbered: bool = True,
    show_location: bool = True,
    max_title: int = 72,
) -> str:
    if not sessions:
        return "No saved sessions found."

    rows = [summarize_session(item, max_title=max_title) for item in sessions]
    index_width = len(str(len(rows))) if numbered else 0
    id_width = max(8, min(16, max(len(row["session_id"]) for row in rows)))
    time_width = max(12, min(16, max(len(row["updated_at"]) for row in rows)))

    lines: List[str] = []
    for idx, row in enumerate(rows, start=1):
        prefix = f"{idx:>{index_width}}. " if numbered else ""
        sid = row["session_id"][:id_width].ljust(id_width)
        updated = row["updated_at"][:time_width].ljust(time_width)
        line = f"{prefix}{sid}  {updated}  {row['title']}"
        if show_location and row["location"]:
            line += f"  [{row['location']}]"
        lines.append(line.rstrip())
    return "\n".join(lines)


def format_session_detail(status: Dict[str, Any], events: Optional[List[Dict[str, Any]]] = None) -> str:
    if not status or status.get("status") == "error":
        return str((status or {}).get("message") or "Session not found.")
    metadata = dict(status.get("metadata") or {})
    lines = [
        f"Session: {status.get('session_id')}",
        f"Title: {_plain(metadata.get('title') or 'Untitled')}",
        f"Updated: {_format_updated_at(metadata.get('updated_at') or metadata.get('created_at'))}",
        f"Input: {_plain(status.get('input_path') or metadata.get('input_path') or '(not set)')}",
        f"Output: {_plain(status.get('output_dir') or metadata.get('output_dir') or '(not set)')}",
        f"Turn: {_plain(status.get('conversation_turn') or '(unknown)')}",
        f"Phase: {_plain(status.get('planner_phase') or '(unknown)')}",
    ]
    preview = _plain(status.get("preview"))
    if preview:
        lines.extend(["", "Preview:", _shorten(preview, 600)])
    diagnostics = status.get("resume_diagnostics") or {}
    warnings = diagnostics.get("warnings") if isinstance(diagnostics, dict) else None
    if warnings:
        lines.extend(["", "Resume warnings:"])
        lines.extend(f"- {_plain(item)}" for item in warnings[:5])
    if events:
        lines.extend(["", "Recent events:"])
        for event in events[-8:]:
            event_type = _plain(event.get("type") or event.get("event_type"))
            timestamp = _format_updated_at(event.get("timestamp") or event.get("ts"))
            data = event.get("data") if isinstance(event.get("data"), dict) else {}
            summary = _plain(data.get("message") or data.get("content") or data.get("response") or "")
            lines.append(f"- {timestamp}  {event_type}  {_shorten(summary, 100)}".rstrip())
    return "\n".join(lines)


def choose_session_interactively(
    sessions: List[Dict[str, Any]],
    *,
    prompt: str = "Resume session",
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> Optional[str]:
    """Prompt for a session id or numbered selection.

    Returns ``None`` when the user cancels. Raises ``RuntimeError`` when there is
    no interactive stdin available, so callers can produce a CLI-friendly usage
    error rather than silently picking the wrong session.
    """
    if not sessions:
        output_fn("No saved sessions found.")
        return None
    if not sys.stdin.isatty() and input_fn is input:
        raise RuntimeError("No session id was provided and stdin is not interactive.")

    output_fn(format_session_listing(sessions, numbered=True))
    while True:
        raw = input_fn(f"{prompt} [1-{len(sessions)}, session id, q]: ").strip()
        if raw.lower() in {"", "q", "quit", "exit"}:
            return None
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(sessions):
                return _plain(sessions[idx - 1].get("session_id")) or None
        for item in sessions:
            sid = _plain(item.get("session_id"))
            if raw == sid or (sid and sid.startswith(raw)):
                return sid
        output_fn("Invalid selection. Enter a list number, a session id/prefix, or q.")


def enable_readline_history(history_path: Optional[str] = None) -> None:
    """Enable basic shell-like input history when GNU readline is available."""
    try:
        import atexit
        import readline
    except Exception:
        return

    path = Path(history_path or (Path.home() / ".cellcompass" / "cli_history"))
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            readline.read_history_file(str(path))
        readline.set_history_length(1000)
        atexit.register(readline.write_history_file, str(path))
    except Exception:
        return
