from __future__ import annotations

import re
from typing import Any


PHASE_COMMENTARY = "commentary"
PHASE_FINAL_ANSWER = "final_answer"
PHASE_NEEDS_INPUT = "needs_input"

_PHASE_TAG_RE = re.compile(
    r"^\s*(?:\[)?\s*phase\s*[:=]\s*([A-Za-z][A-Za-z0-9_\-]*)\s*(?:\])?\s*",
    re.IGNORECASE,
)
_PHASE_XML_RE = re.compile(
    r"^\s*<phase>\s*([A-Za-z][A-Za-z0-9_\-]*)\s*</phase>\s*",
    re.IGNORECASE,
)
_EMBEDDED_FINAL_PHASE_RE = re.compile(
    r"(?im)(?:^|\n)\s*(?:\[)?\s*phase\s*[:=]\s*(?:final|final_answer|completed)\s*(?:\])?\s*(?:\n|$)"
)
_TOOL_TRANSCRIPT_MARKER_RE = re.compile(
    r"(?im)(?:^|\n)\s*(?:"
    r"to\s*=\s*functions\."
    r"|to\s*=\s*multi_tool_use\."
    r"|[\|\s`>\uFFFD]*H\s*=\s*functions\."
    r"|[\|\s`>\uFFFD]*H\s*=\s*multi_tool_use\."
    r")"
)


def _normalize_phase(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if text == "commentary":
        return PHASE_COMMENTARY
    if text == "needs_input":
        return PHASE_NEEDS_INPUT
    if text in {"final", "final_answer", "completed"}:
        return PHASE_FINAL_ANSWER
    return None


def _extract_phase_from_text(content: str | None) -> str | None:
    if not content:
        return None
    match = _PHASE_TAG_RE.match(content) or _PHASE_XML_RE.match(content)
    if not match:
        return None
    return _normalize_phase(match.group(1))


def strip_phase_tag(content: Any) -> str:
    if not isinstance(content, str):
        return str(content)
    text = _PHASE_TAG_RE.sub("", content, count=1)
    text = _PHASE_XML_RE.sub("", text, count=1)
    return text.strip()


def strip_tool_call_transcript(content: Any) -> str:
    """Remove model-emitted pseudo tool-call transcripts from assistant text.

    Some chat models occasionally place a textual tool-call transcript in
    `content` instead of, or in addition to, structured `tool_calls`. The UI
    should never display those snippets as normal assistant prose.
    """
    text = strip_phase_tag(content)
    if not text:
        return ""

    final_phase_matches = list(_EMBEDDED_FINAL_PHASE_RE.finditer(text))
    for match in reversed(final_phase_matches):
        candidate = strip_phase_tag(text[match.end() :])
        if candidate and not _TOOL_TRANSCRIPT_MARKER_RE.match(candidate):
            text = candidate
            break

    marker = _TOOL_TRANSCRIPT_MARKER_RE.search(text)
    if not marker:
        return text.strip()

    prefix = text[: marker.start()].strip()
    return prefix


def resolve_message_phase(message: Any) -> str | None:
    direct = _normalize_phase(getattr(message, "phase", None))
    if direct:
        return direct

    additional_kwargs = getattr(message, "additional_kwargs", None) or {}
    meta_phase = _normalize_phase(additional_kwargs.get("phase"))
    if meta_phase:
        return meta_phase

    response_metadata = getattr(message, "response_metadata", None) or {}
    response_phase = _normalize_phase(response_metadata.get("phase"))
    if response_phase:
        return response_phase

    content = getattr(message, "content", None)
    if isinstance(content, str):
        return _extract_phase_from_text(content)
    return None


def should_continue_turn(message: Any) -> bool:
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        return True
    return resolve_message_phase(message) == PHASE_COMMENTARY
